"""双流隐变量条件指针策略：静态几何编码 + 动态状态编码。"""
from dataclasses import dataclass
from typing import Optional
import math
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch


@dataclass
class StaticGeometry:
    node_embeddings: Tensor
    padding_mask: Tensor  # True 表示有效节点。
    batch: Tensor
    graph_summary: Optional[Tensor] = None


class GeometryDynamicPointerPolicy(nn.Module):
    def __init__(self, dynamic_input_dim=6, geometry_input_dim=2, hidden_dim=128,
                 dynamic_dim=64, latent_dim=16, num_heads=8, static_gat_layers=2,
                 static_transformer_layers=2, dropout=0.0, pointer_clip=10.0):
        super().__init__()
        self.dynamic_input_dim = dynamic_input_dim
        self.geometry_input_dim = geometry_input_dim
        self.hidden_dim = hidden_dim
        self.dynamic_dim = dynamic_dim
        self.latent_dim = latent_dim
        self.pointer_clip = float(pointer_clip)

        # 静态流：坐标投影 -> 残差 GAT -> Transformer。
        self.geometry_projection = nn.Sequential(
            nn.Linear(geometry_input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.static_gats = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False,
                      edge_dim=1, dropout=dropout)
            for _ in range(static_gat_layers)])
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.static_transformer = nn.TransformerEncoder(
            layer, num_layers=static_transformer_layers, norm=nn.LayerNorm(hidden_dim))
        self.static_norm = nn.LayerNorm(hidden_dim)

        # 动态流：节点特征与辅助上下文分别编码后相加。
        self.auxiliary_encoder = nn.Sequential(
            nn.Linear(5, dynamic_dim), nn.GELU(), nn.LayerNorm(dynamic_dim))
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dynamic_dim), nn.GELU())
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim + dynamic_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

        # 条件查询 -> 交叉注意力 -> 指针打分。
        self.query_network = nn.Sequential(
            nn.Linear(2 * hidden_dim + dynamic_dim + latent_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_bias = nn.Sequential(
            nn.Linear(dynamic_dim, dynamic_dim), nn.GELU(), nn.Linear(dynamic_dim, 1))

    @staticmethod
    def sample_prior(batch_size, latent_dim, device, bound=1.0):
        return (2 * torch.rand(batch_size, latent_dim, device=device) - 1) * float(bound)

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.unsqueeze(-1).to(values.dtype)
        return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def encode_geometry(self, geometry_data):
        """静态图每个 rollout 编码一次；训练更新时重新编码以保留梯度。"""
        x = self.geometry_projection(geometry_data.x[:, :self.geometry_input_dim].float())
        edge_attr = geometry_data.edge_attr[:, :1].float()
        for layer in self.static_gats:
            x = torch.nn.functional.gelu(x + layer(x, geometry_data.edge_index, edge_attr=edge_attr))
        dense, valid = to_dense_batch(x, geometry_data.batch)
        encoded = self.static_norm(self.static_transformer(dense, src_key_padding_mask=~valid))
        return StaticGeometry(encoded, valid, geometry_data.batch.detach().clone(),
                              self._masked_mean(encoded, valid))

    def forward(self, dynamic_data, geometry, z, action_mask=None):
        static = geometry.node_embeddings
        context = dynamic_data.context[dynamic_data.batch, :2]
        auxiliary = torch.cat((dynamic_data.node_aux, context), -1).float()
        dynamic = self.dynamic_encoder(dynamic_data.x.float()) + self.auxiliary_encoder(auxiliary)
        dynamic, valid = to_dense_batch(dynamic, dynamic_data.batch)
        nodes = self.node_fusion(torch.cat((static, dynamic), -1))
        z = torch.as_tensor(z, dtype=static.dtype, device=static.device)
        z = z.unsqueeze(0) if z.ndim == 1 else z
        z = z.expand(static.size(0), -1) if z.size(0) == 1 else z

        graph = geometry.graph_summary
        if graph is None:
            graph = self._masked_mean(static, valid)
        current, _ = to_dense_batch(dynamic_data.x[:, 3].float(), dynamic_data.batch)
        current = current * valid.to(current.dtype)
        current = current / current.sum(1, keepdim=True).clamp_min(1.0)
        current = (nodes * current.unsqueeze(-1)).sum(1)
        query = self.query_network(torch.cat(
            (graph, self._masked_mean(dynamic, valid), current, z), -1)).unsqueeze(1)
        attended, _ = self.cross_attention(
            query, nodes, nodes, key_padding_mask=~valid, need_weights=False)
        query = self.query_norm(query + attended).squeeze(1)
        logits = (self.pointer_query(query).unsqueeze(1) * self.pointer_key(nodes)).sum(-1)
        logits = logits / math.sqrt(self.hidden_dim)
        logits = self.pointer_clip * torch.tanh(logits + self.node_bias(dynamic).squeeze(-1))
        legal = valid.clone()
        if action_mask is not None:
            legal &= torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device)
        return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)

    def distribution(self, dynamic_data, geometry, z, action_mask=None):
        return Categorical(logits=self(dynamic_data, geometry, z, action_mask))

    def act(self, dynamic_data, geometry, z, action_mask=None, deterministic=False):
        logits = self(dynamic_data, geometry, z, action_mask)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), logits


def build_geometry_data(dynamic_data, baseline_distance_matrix, coords=None, coordinate_dim=2):
    """构造共享实例的批量静态图；缺少坐标时使用经典 MDS。"""
    device, batch = dynamic_data.x.device, dynamic_data.batch
    batch_size = int(batch.max().item()) + 1
    n = int(torch.bincount(batch, minlength=batch_size)[0].item())
    baseline = torch.as_tensor(baseline_distance_matrix, dtype=torch.float32, device=device)
    if coords is None:
        centering = torch.eye(n, device=device) - torch.full((n, n), 1.0 / n, device=device)
        gram = -0.5 * centering @ baseline.square() @ centering
        values, vectors = torch.linalg.eigh(gram)
        take = min(coordinate_dim, n)
        coords = vectors[:, -take:] * values[-take:].clamp_min(0).sqrt().unsqueeze(0)
    else:
        coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
    coords = torch.nn.functional.pad(coords, (0, max(0, coordinate_dim - coords.size(1))))
    coords = coords[:, :coordinate_dim]
    src, dst = dynamic_data.edge_index
    edge_attr = (baseline[src % n, dst % n] / baseline.max().clamp_min(1e-6)).unsqueeze(-1)
    return Data(x=coords.repeat(batch_size, 1), edge_index=dynamic_data.edge_index,
                edge_attr=edge_attr, batch=batch)
