"""隐变量条件指针策略网络(精简版:已删除输入形状/类型检查,网络结构与参数名保持不变)。

结构:
1. 静态几何分支:由坐标和基准距离得到可复用的节点嵌入(reset 后编码一次)。
2. 动态分支:编码 BatchCVRPEnv 输出的 6 维节点特征和辅助特征。
3. 隐变量条件指针:图摘要 + 动态摘要 + 当前节点 + 隐变量 z 形成查询,对节点打分得到 logits。
模型只返回 logits,采样与 log_prob 由训练代码用同一个 masked 分布计算。
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


# 核心模块:静态几何缓存,由调用方持有。几何、节点顺序、编码器参数或设备变化后需重建。
@dataclass
class StaticGeometry:
    node_embeddings: Tensor
    padding_mask: Tensor
    batch: Tensor
    graph_summary: Optional[Tensor] = None
    fusion_static: Optional[Tensor] = None
    fusion_signature: Optional[tuple] = None


class GeometryDynamicPointerPolicy(nn.Module):
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
        self.dynamic_input_dim = dynamic_input_dim
        self.geometry_input_dim = geometry_input_dim
        self.hidden_dim = hidden_dim
        self.dynamic_dim = dynamic_dim
        self.latent_dim = latent_dim
        self.pointer_clip = float(pointer_clip)
        self.cache_static_fusion = True  # 运行时开关,不产生新参数

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

        self.auxiliary_encoder = nn.Sequential(nn.Linear(5, dynamic_dim), nn.GELU(), nn.LayerNorm(dynamic_dim))
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
    def sample_prior(batch_size: int, latent_dim: int, device: torch.device | str, bound: float = 1.0) -> Tensor:
        """每个并行回合采样一个 [-bound, bound] 均匀分布的外部隐变量。"""
        return (2.0 * torch.rand(batch_size, latent_dim, device=device) - 1.0) * float(bound)

    @staticmethod
    def _geometry_edge_attr(geometry_data: Data) -> Tensor:
        return geometry_data.edge_attr[:, :1].float()

    # 核心模块:静态几何编码(坐标投影 -> GAT 残差层 -> Transformer)。
    # rollout 时在 no_grad 下编码一次并复用;PPO 反向时需用当前权重带梯度重新编码。
    def encode_geometry(self, geometry_data: Data) -> StaticGeometry:
        batch = geometry_data.batch
        x = self.geometry_projection(geometry_data.x[:, : self.geometry_input_dim].float())
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
            fusion_static = F.linear(encoded, layer.weight[:, : self.hidden_dim], layer.bias)
            fusion_signature = self._fusion_signature(encoded)
        return StaticGeometry(
            node_embeddings=encoded,
            padding_mask=padding_mask,
            batch=batch.detach().clone(),
            graph_summary=graph_summary,
            fusion_static=fusion_static,
            fusion_signature=fusion_signature,
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
        # 反向传播始终走完整融合层
        return (
            self.cache_static_fusion
            and not torch.is_grad_enabled()
            and not torch.is_inference_mode_enabled()
            and not self._autocast_active(static.device.type)
        )

    def _fusion_signature(self, static: Tensor) -> tuple:
        layer = self.node_fusion[0]
        return (
            id(layer.weight),
            layer.weight._version,
            id(layer.bias),
            layer.bias._version if layer.bias is not None else None,
            layer.weight.device,
            layer.weight.dtype,
            id(static),
            static._version,
            static.device,
            static.dtype,
        )

    # 核心模块:节点融合 Linear([static, dynamic]) = W_s*static + b + W_d*dynamic。
    # 无梯度时可复用几何项 W_s*static + b,浮点运算顺序略有不同。
    def _fuse_nodes(self, static: Tensor, dynamic: Tensor, geometry: StaticGeometry) -> Tensor:
        cached = geometry.fusion_static
        if (
            cached is not None
            and self._can_cache_static_fusion(static)
            and geometry.fusion_signature == self._fusion_signature(static)
        ):
            layer = self.node_fusion[0]
            dynamic_term = F.linear(dynamic, layer.weight[:, self.hidden_dim :], None)
            fused = cached + dynamic_term
            return self.node_fusion[2](self.node_fusion[1](fused))
        return self.node_fusion(torch.cat((static, dynamic), dim=-1))

    @staticmethod
    def _prepare_latent(z: Tensor, batch_size: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        z = torch.as_tensor(z, dtype=dtype, device=device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.size(0) == 1 and batch_size > 1:
            z = z.expand(batch_size, -1)
        return z

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    # 核心模块:动态特征编码。6 维节点特征 + (node_aux 3 维 + 环境 context 前 2 维) 的辅助特征。
    def _dynamic_dense(self, dynamic_data: Data) -> Tuple[Tensor, Tensor]:
        env_context = dynamic_data.context[dynamic_data.batch, :2]
        auxiliary = torch.cat((dynamic_data.node_aux, env_context), dim=-1).float()
        dynamic = self.dynamic_encoder(dynamic_data.x.float()) + self.auxiliary_encoder(auxiliary)
        return to_dense_batch(dynamic, dynamic_data.batch)

    # 核心模块:前向。查询 = MLP([图摘要, 动态摘要, 当前节点嵌入, z]) -> 交叉注意力 -> 指针打分
    # -> tanh 截断 -> 非法节点填最小值。返回 [B, max_nodes] 的 masked logits。
    def forward(
        self, dynamic_data: Data, geometry: StaticGeometry, z: Tensor, action_mask: Optional[Tensor] = None
    ) -> Tensor:
        static = geometry.node_embeddings
        dynamic, mask = self._dynamic_dense(dynamic_data)
        batch_size = static.size(0)
        z = self._prepare_latent(z, batch_size, static.dtype, static.device)

        node_embeddings = self._fuse_nodes(static, dynamic, geometry)
        graph_summary = geometry.graph_summary
        if graph_summary is None:
            graph_summary = self._masked_mean(static, mask)
        dynamic_summary = self._masked_mean(dynamic, mask)

        current_dense, _ = to_dense_batch(dynamic_data.x[:, 3].float(), dynamic_data.batch)
        current_weights = current_dense * mask.to(current_dense.dtype)
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
        logits = self.pointer_clip * torch.tanh(logits + self.node_bias(dynamic).squeeze(-1))

        legal = mask.clone()
        if action_mask is not None:
            legal &= torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
        return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)

    def distribution(
        self, dynamic_data: Data, geometry: StaticGeometry, z: Tensor, action_mask: Optional[Tensor] = None
    ) -> Categorical:
        return Categorical(logits=self.forward(dynamic_data, geometry, z, action_mask))

    def act(
        self,
        dynamic_data: Data,
        geometry: StaticGeometry,
        z: Tensor,
        action_mask: Optional[Tensor] = None,
        deterministic: bool = False,
    ):
        """返回 action, log_prob, entropy, logits。"""
        logits = self.forward(dynamic_data, geometry, z, action_mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), logits


# 核心模块:按动态状态的批次顺序构造静态几何图。优先使用真实归一化坐标;
# 无坐标时用基准距离矩阵做经典 MDS。edge_attr 使用基准距离(不含交通变化)。
def build_geometry_data(
    dynamic_data: Data, baseline_distance_matrix, coords: Optional[Tensor] = None, coordinate_dim: int = 2
) -> Data:
    device = dynamic_data.x.device
    batch = dynamic_data.batch
    batch_size = int(batch.max().item()) + 1
    n = int(torch.bincount(batch, minlength=batch_size)[0].item())
    baseline = torch.as_tensor(baseline_distance_matrix, dtype=torch.float32, device=device)

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
        if base_coords.size(1) < coordinate_dim:
            base_coords = torch.nn.functional.pad(base_coords, (0, coordinate_dim - base_coords.size(1)))
        base_coords = base_coords[:, :coordinate_dim]

    src, dst = dynamic_data.edge_index
    local_src = torch.remainder(src, n)
    local_dst = torch.remainder(dst, n)
    edge_attr = (baseline[local_src, local_dst] / baseline.max().clamp_min(1e-6)).unsqueeze(-1)
    return Data(
        x=base_coords.repeat(batch_size, 1),
        edge_index=dynamic_data.edge_index,
        edge_attr=edge_attr,
        batch=batch,
    )
