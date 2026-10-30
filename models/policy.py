"""Latent-conditioned pointer policy for BatchCVRPEnv-compatible PyG states.

Architecture
------------
1. Geometry-static branch: obtains a reusable node embedding from immutable
   coordinates / baseline-distance information supplied at reset time.
2. Stepwise dynamic branch: encodes the six features emitted by BatchCVRPEnv:
   observed demand, expected demand, visited, current-node flag, vehicle load,
   and current traffic distance from the vehicle.
3. Latent-conditioned pointer: forms a vehicle query from the current node,
   graph summary, dynamic summary, and externally supplied latent z, then
   scores each graph node as the next action.

The model deliberately contains no optimiser, training loop, critic, or search
procedure. It retains the original layers and state_dict keys. Geometry graph
summaries are cached; the geometry-only part of node fusion may be reused during
ordinary no_grad inference. Dynamic features, auxiliary state, attention and
legality masks are always recomputed. It returns logits so a training program can use the exact same
masked distribution for sampling and log-probability calculations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch


@dataclass
class StaticGeometry:
    """Immutable per-instance geometry cache, owned by the caller.

    Rebuild after geometry/node-order changes, encoder parameter updates, or
    device/dtype changes. Do not mutate cached tensors in place.
    The optional affine cache is only used during ordinary no_grad forwards.
    """

    node_embeddings: Tensor
    padding_mask: Tensor
    batch: Tensor
    graph_summary: Optional[Tensor] = None
    fusion_static: Optional[Tensor] = None
    fusion_signature: Optional[tuple] = None


class GeometryDynamicPointerPolicy(nn.Module):
    """Geometry-static encoder + dynamic encoder + latent-conditioned pointer.

    Expected dynamic state is a PyG ``Data`` object emitted by ``BatchCVRPEnv``:
    ``data.x`` has shape ``[B*N, 6]`` and ``data.batch`` identifies graph rows.
    A geometry state must have the same node order and batch layout. Its
    ``x`` may be either coordinates ``[B*N, 2]`` or a feature matrix containing
    coordinates in the first two columns. If coordinates are unavailable,
    ``build_geometry_data`` can construct a geometry state from a baseline
    distance matrix using classical MDS.
    """

    def __init__(
        self,
        dynamic_input_dim: int = 6,
        geometry_input_dim: int = 2,
        hidden_dim: int = 128,
        dynamic_dim: int = 64,
        latent_dim: int = 16,
        num_heads: int = 8,
        static_gat_layers: int = 2,
        static_transformer_layers: int = 2,
        dropout: float = 0.0,
        pointer_clip: float = 10.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or dynamic_dim <= 0 or latent_dim <= 0:
            raise ValueError("hidden_dim, dynamic_dim, and latent_dim must be positive.")
        if num_heads < 1 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if static_gat_layers < 0 or static_transformer_layers < 1:
            raise ValueError("Layer counts must be non-negative.")

        self.dynamic_input_dim = dynamic_input_dim
        self.geometry_input_dim = geometry_input_dim
        self.hidden_dim = hidden_dim
        self.dynamic_dim = dynamic_dim
        self.latent_dim = latent_dim
        self.pointer_clip = float(pointer_clip)
        # Plain runtime switch, no new parameters or checkpoint fields.
        self.cache_static_fusion = True

        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.static_gats = nn.ModuleList(
            [
                GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, edge_dim=1, dropout=dropout)
                for _ in range(static_gat_layers)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.static_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=static_transformer_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.static_norm = nn.LayerNorm(hidden_dim)

        self.auxiliary_encoder = nn.Sequential(
            nn.Linear(5, dynamic_dim), nn.GELU(), nn.LayerNorm(dynamic_dim))
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dynamic_dim),
            nn.GELU(),
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim + dynamic_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        query_input_dim = 2 * hidden_dim + dynamic_dim + latent_dim
        self.query_network = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_bias = nn.Sequential(
            nn.Linear(dynamic_dim, dynamic_dim),
            nn.GELU(),
            nn.Linear(dynamic_dim, 1),
        )

    @staticmethod
    def sample_prior(batch_size: int, latent_dim: int, device: torch.device | str,
                     bound: float = 1.0) -> Tensor:
        """Samples one external latent vector per parallel rollout."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return (2.0 * torch.rand(batch_size, latent_dim, device=device) - 1.0) * float(bound)

    @staticmethod
    def _validate_batch(data: Data) -> Tensor:
        if not hasattr(data, "x") or not hasattr(data, "batch"):
            raise ValueError("data must provide PyG attributes x and batch.")
        if data.x.ndim != 2 or data.batch.ndim != 1 or data.x.size(0) != data.batch.numel():
            raise ValueError("data.x and data.batch have incompatible shapes.")
        return data.batch

    def _geometry_edge_attr(self, geometry_data: Data) -> Tensor:
        edge_count = geometry_data.edge_index.size(1)
        if hasattr(geometry_data, "edge_attr") and geometry_data.edge_attr is not None:
            edge_attr = geometry_data.edge_attr
            if edge_attr.ndim == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            if edge_attr.size(0) == edge_count:
                return edge_attr[:, :1].float()
        return torch.ones((edge_count, 1), dtype=geometry_data.x.dtype, device=geometry_data.x.device)

    def encode_geometry(self, geometry_data: Data) -> StaticGeometry:
        """Encode immutable geometry once after ``env.reset()``.

        Do not call this with the live dynamic ``BatchCVRPEnv`` state when its
        edge attributes contain changing traffic costs. Use coordinates or a
        baseline-distance graph instead.

        Rollout: call under no_grad and reuse for this unchanged instance.
        PPO backward: call again with gradients enabled using current weights;
        never use a detached rollout cache to train the geometry encoder.
        """
        batch = self._validate_batch(geometry_data)
        if geometry_data.x.size(-1) < self.geometry_input_dim:
            raise ValueError("geometry_data.x has fewer columns than geometry_input_dim.")
        x = geometry_data.x[:, :self.geometry_input_dim].float()
        x = self.geometry_projection(x)
        edge_attr = self._geometry_edge_attr(geometry_data)
        for layer in self.static_gats:
            x = x + layer(x, geometry_data.edge_index, edge_attr=edge_attr)
            x = torch.nn.functional.gelu(x)

        dense, padding_mask = to_dense_batch(x, batch)
        encoded = self.static_transformer(dense, src_key_padding_mask=~padding_mask)
        encoded = self.static_norm(encoded)
        graph_summary = self._masked_mean(encoded, padding_mask)
        fusion_static = None
        fusion_signature = None
        if self._can_cache_static_fusion(encoded):
            layer = self.node_fusion[0]
            fusion_static = F.linear(encoded, layer.weight[:, :self.hidden_dim], layer.bias)
            fusion_signature = self._fusion_signature(encoded)
        return StaticGeometry(
            node_embeddings=encoded, padding_mask=padding_mask,
            batch=batch.detach().clone(), graph_summary=graph_summary,
            fusion_static=fusion_static, fusion_signature=fusion_signature,
        )

    @staticmethod
    def _autocast_active(device_type: str) -> bool:
        try:
            return torch.is_autocast_enabled(device_type)
        except TypeError:
            if device_type == "cpu":
                checker = getattr(torch, "is_autocast_cpu_enabled", None)
                return bool(checker()) if checker is not None else False
            return torch.is_autocast_enabled()

    def _can_cache_static_fusion(self, static: Tensor) -> bool:
        # Backward always uses the original complete fusion layer.
        return (self.cache_static_fusion and not torch.is_grad_enabled()
                and not torch.is_inference_mode_enabled()
                and not self._autocast_active(static.device.type))

    def _fusion_signature(self, static: Tensor) -> tuple:
        layer = self.node_fusion[0]
        return (id(layer.weight), layer.weight._version,
                id(layer.bias), layer.bias._version if layer.bias is not None else None,
                layer.weight.device, layer.weight.dtype,
                id(static), static._version, static.device, static.dtype)

    def _fuse_nodes(self, static: Tensor, dynamic: Tensor,
                    geometry: StaticGeometry) -> Tensor:
        """Same affine map; optionally reuse its geometry-only term.

        Linear([static,dynamic]) = W_s*static + bias + W_d*dynamic.
        No-gradient float32/float64 calls may reuse the first term. The order
        of floating-point operations differs slightly from the original path.
        """
        cached = geometry.fusion_static
        if (cached is not None and self._can_cache_static_fusion(static)
                and geometry.fusion_signature == self._fusion_signature(static)):
            layer = self.node_fusion[0]
            dynamic_term = F.linear(dynamic, layer.weight[:, self.hidden_dim:], None)
            fused = cached + dynamic_term
            return self.node_fusion[2](self.node_fusion[1](fused))
        return self.node_fusion(torch.cat((static, dynamic), dim=-1))

    def _prepare_latent(self, z: Tensor, batch_size: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        z = torch.as_tensor(z, dtype=dtype, device=device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.ndim != 2 or z.size(1) != self.latent_dim:
            raise ValueError(f"z must have shape [B, {self.latent_dim}] or [{self.latent_dim}].")
        if z.size(0) == 1 and batch_size > 1:
            z = z.expand(batch_size, -1)
        if z.size(0) != batch_size:
            raise ValueError("The z batch dimension must be 1 or equal to the graph batch size.")
        return z

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _dynamic_dense(self, dynamic_data: Data, expected_mask: Tensor) -> Tuple[Tensor, Tensor]:
        batch = self._validate_batch(dynamic_data)
        if dynamic_data.x.size(-1) != self.dynamic_input_dim:
            raise ValueError(
                f"dynamic data must have {self.dynamic_input_dim} features, got {dynamic_data.x.size(-1)}."
            )
        if not hasattr(dynamic_data, "node_aux") or not hasattr(dynamic_data, "context"):
            raise ValueError("Scheme B Actor requires node_aux and context from the revised BatchCVRPEnv")
        env_context = dynamic_data.context[dynamic_data.batch, :2]
        auxiliary = torch.cat((dynamic_data.node_aux, env_context), dim=-1).float()
        dynamic = self.dynamic_encoder(dynamic_data.x.float()) + self.auxiliary_encoder(auxiliary)
        dense, mask = to_dense_batch(dynamic, batch)
        if mask.shape != expected_mask.shape or not torch.equal(mask, expected_mask):
            raise ValueError("Geometry and dynamic state must have identical batch layout and node order.")
        return dense, mask

    def forward(self, dynamic_data: Data, geometry: StaticGeometry, z: Tensor,
                action_mask: Optional[Tensor] = None) -> Tensor:
        """Return masked next-node logits with shape ``[B, max_nodes]``.

        ``action_mask`` is True for legal actions. Its shape must match the
        output dense-node layout. Padding is always made illegal.
        """
        static = geometry.node_embeddings
        if (geometry.batch.shape != dynamic_data.batch.shape
                or not torch.equal(geometry.batch, dynamic_data.batch)):
            raise ValueError("Cached geometry batch layout changed; rebuild geometry after reset.")
        dynamic, mask = self._dynamic_dense(dynamic_data, geometry.padding_mask)
        if static.shape[:2] != dynamic.shape[:2]:
            raise ValueError("Cached geometry and dynamic state do not have matching node dimensions.")
        batch_size, node_count, _ = static.shape
        z = self._prepare_latent(z, batch_size, static.dtype, static.device)

        node_embeddings = self._fuse_nodes(static, dynamic, geometry)
        graph_summary = geometry.graph_summary
        if graph_summary is None:
            # Support manually constructed three-field StaticGeometry objects.
            graph_summary = self._masked_mean(static, mask)
        dynamic_summary = self._masked_mean(dynamic, mask)

        current_flag = dynamic_data.x[:, 3].float()
        current_dense, current_mask = to_dense_batch(current_flag, dynamic_data.batch)
        if not torch.equal(current_mask, mask):
            raise ValueError("Current-node feature layout does not match graph layout.")
        current_weights = current_dense * mask.to(current_dense.dtype)
        fallback = mask.to(current_dense.dtype)
        current_weights = torch.where(
            current_weights.sum(dim=1, keepdim=True) > 0,
            current_weights,
            fallback,
        )
        current_weights = current_weights / current_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        current_embedding = (node_embeddings * current_weights.unsqueeze(-1)).sum(dim=1)

        query_input = torch.cat((graph_summary, dynamic_summary, current_embedding, z), dim=-1)
        query = self.query_network(query_input).unsqueeze(1)
        attended, _ = self.cross_attention(
            query=query,
            key=node_embeddings,
            value=node_embeddings,
            key_padding_mask=~mask,
            need_weights=False,
        )
        context = self.query_norm(query + attended).squeeze(1)
        logits = (self.pointer_query(context).unsqueeze(1) * self.pointer_key(node_embeddings)).sum(-1)
        logits = logits / math.sqrt(self.hidden_dim)
        logits = self.pointer_clip * torch.tanh(logits) + self.node_bias(dynamic).squeeze(-1)

        legal = mask.clone()
        if action_mask is not None:
            action_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
            if action_mask.shape != legal.shape:
                raise ValueError(f"action_mask must have shape {tuple(legal.shape)}.")
            legal &= action_mask
        no_legal = ~legal.any(dim=1)
        if no_legal.any():
            raise ValueError("No legal action; masks must include done-row depot")
        return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)

    def distribution(self, dynamic_data: Data, geometry: StaticGeometry, z: Tensor,
                     action_mask: Optional[Tensor] = None) -> Categorical:
        return Categorical(logits=self.forward(dynamic_data, geometry, z, action_mask))

    def act(self, dynamic_data: Data, geometry: StaticGeometry, z: Tensor,
            action_mask: Optional[Tensor] = None, deterministic: bool = False):
        """Return ``action, log_prob, entropy, logits`` for one rollout step."""
        logits = self.forward(dynamic_data, geometry, z, action_mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), logits


def build_geometry_data(dynamic_data: Data, baseline_distance_matrix, coords: Optional[Tensor] = None,
                        coordinate_dim: int = 2) -> Data:
    """Build a static PyG graph in the exact batch order of ``dynamic_data``.

    Preferred usage passes actual normalized coordinates via ``coords``.
    If coordinates are unavailable, this creates two-dimensional classical-MDS
    coordinates from the supplied symmetric baseline distance matrix. The
    result uses baseline (not traffic) distances as ``edge_attr``.
    """
    if not hasattr(dynamic_data, "edge_index") or not hasattr(dynamic_data, "batch"):
        raise ValueError("dynamic_data must include edge_index and batch.")
    device = dynamic_data.x.device
    batch = dynamic_data.batch
    batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
    if batch_size < 1:
        raise ValueError("dynamic_data contains no graphs.")
    counts = torch.bincount(batch, minlength=batch_size)
    if not torch.all(counts == counts[0]):
        raise ValueError("All batch graphs must have the same number of nodes.")
    n = int(counts[0].item())
    baseline = torch.as_tensor(baseline_distance_matrix, dtype=torch.float32, device=device)
    if baseline.shape != (n, n):
        raise ValueError(f"baseline_distance_matrix must have shape {(n, n)}.")

    if coords is None:
        squared = baseline.square()
        centering = torch.eye(n, device=device) - torch.full((n, n), 1.0 / n, device=device)
        gram = -0.5 * centering @ squared @ centering
        values, vectors = torch.linalg.eigh(gram)
        take = min(coordinate_dim, n)
        values = values[-take:].clamp_min(0.0)
        base_coords = vectors[:, -take:] * values.sqrt().unsqueeze(0)
        if take < coordinate_dim:
            base_coords = torch.nn.functional.pad(base_coords, (0, coordinate_dim - take))
    else:
        base_coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
        if base_coords.ndim != 2 or base_coords.size(0) != n:
            raise ValueError("coords must have shape [num_nodes, coordinate_dim].")
        if base_coords.size(1) < coordinate_dim:
            base_coords = torch.nn.functional.pad(base_coords, (0, coordinate_dim - base_coords.size(1)))
        base_coords = base_coords[:, :coordinate_dim]

    src, dst = dynamic_data.edge_index
    local_src = torch.remainder(src, n)
    local_dst = torch.remainder(dst, n)
    edge_attr = baseline[local_src, local_dst].unsqueeze(-1)
    return Data(
        x=base_coords.repeat(batch_size, 1),
        edge_index=dynamic_data.edge_index,
        edge_attr=edge_attr,
        batch=batch,
    )
