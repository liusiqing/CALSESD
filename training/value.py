"""独立双流状态编码器与集成价值头，评价在线控制器条件下的 V(s, z)。"""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch

ONLINE_CONTEXT_DIM = 6
ONLINE_VALUE_SEMANTICS = "online_actor_controller_v1"


class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim=128, dynamic_dim=64, latent_dim=16,
                 n_critics=4, kind="hybrid", prior_scale=0.0, *, latent_fusion=False):
        super().__init__()
        self.kind, self.latent_dim, self.n_critics = kind, latent_dim, n_critics
        self.online_context_dim = ONLINE_CONTEXT_DIM
        self.value_semantics = ONLINE_VALUE_SEMANTICS
        self.state_feature_dim = 4 * hidden_dim + 2 + ONLINE_CONTEXT_DIM + 2
        H = hidden_dim
        # 双流：坐标与动态特征独立编码，再融合并进行空间消息传递。
        self.geometry_encoder = nn.Sequential(nn.Linear(2, H), nn.GELU(), nn.LayerNorm(H))
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(9, H), nn.GELU(), nn.Linear(H, H), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.LayerNorm(H))
        self.spatial = GATv2Conv(H, H, heads=4 if H % 4 == 0 else 1,
                                 concat=False, edge_dim=1, dropout=0.0)
        self.norm = nn.LayerNorm(H)
        # 多个价值头共享状态表示，但分别对 [状态特征, z] 打分。
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(self.state_feature_dim + latent_dim, H), nn.GELU(),
                          nn.Linear(H, H), nn.GELU(), nn.Linear(H, 1))
            for _ in range(n_critics)])
        self.prior_scale = float(prior_scale)
        if self.prior_scale > 0:
            self.priors = nn.ModuleList([
                nn.Sequential(nn.Linear(self.state_feature_dim + latent_dim, H // 2),
                              nn.GELU(), nn.Linear(H // 2, 1))
                for _ in range(n_critics)])
            for parameter in self.priors.parameters():
                parameter.requires_grad_(False)

        self.latent_fusion = bool(latent_fusion)
        if self.latent_fusion:
            width = min(H, 64)
            self.latent_fusion_state = nn.Sequential(nn.LayerNorm(self.state_feature_dim),
                                                     nn.Linear(self.state_feature_dim, width))
            self.latent_fusion_codes = nn.ModuleList(
                [nn.Linear(latent_dim, width, bias=False) for _ in range(n_critics)])
            self.latent_fusion_readouts = nn.ModuleList(
                [nn.Linear(width, 1, bias=False) for _ in range(n_critics)])
            for readout in self.latent_fusion_readouts:
                nn.init.zeros_(readout.weight)

    # 训练入口、验证和 checkpoint 仍调用这两个接口；缓存实现已删除。
    def configure_static_cache(self, *, enabled=True, max_entries=4):
        pass

    def clear_static_cache(self):
        pass

    def encode_static(self, state):
        return self.geometry_encoder(state.pos.float())

    @staticmethod
    def pool(x, mask):
        weights = mask.to(x.dtype).unsqueeze(-1)
        return (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def encode_state(self, state):
        """融合节点后做全体/剩余/当前/仓库池化，并拼接在线上下文。"""
        dynamic = self.dynamic_encoder(torch.cat((state.x, state.node_aux), -1).float())
        node = self.fusion(torch.cat((self.encode_static(state), dynamic), -1))
        node = self.norm(node + torch.nn.functional.gelu(
            self.spatial(node, state.edge_index, state.edge_attr)))
        dense, valid = to_dense_batch(node, state.batch)
        x, _ = to_dense_batch(state.x, state.batch)
        aux, _ = to_dense_batch(state.node_aux, state.batch)
        remaining = (aux[:, :, 0] > 0.5) & (x[:, :, 2] < 0.5) & valid
        current = (x[:, :, 3] > 0.5) & valid
        depot = (aux[:, :, 1] > 0.5) & valid
        if state.online_context.shape != (dense.size(0), self.online_context_dim):
            raise ValueError(f"online_context must have shape [B, {self.online_context_dim}]")
        counts = valid.sum(1).clamp_min(1).to(dense.dtype)
        extra = torch.stack((remaining.sum(1) / counts,
                             torch.log1p((x[:, :, 0] * remaining).sum(1))), -1)
        return torch.cat((self.pool(dense, valid), self.pool(dense, remaining),
                          self.pool(dense, current), self.pool(dense, depot),
                          state.context[:, :2].to(dense), state.online_context.to(dense), extra), -1)

    def evaluate_features(self, features, z):
        """复用已编码状态，返回 [n_critics, batch_size]。"""
        z = z.unsqueeze(0) if z.ndim == 1 else z
        if features.ndim != 2 or features.size(1) != self.state_feature_dim:
            raise ValueError(f"features must have shape [B, {self.state_feature_dim}]")
        if z.ndim != 2 or z.size(1) != self.latent_dim or z.size(0) not in (1, features.size(0)):
            raise ValueError(f"z must have shape [B, {self.latent_dim}] or [1, {self.latent_dim}]")
        z = z.expand(features.size(0), -1) if z.size(0) == 1 else z
        inp = torch.cat((features, z.to(features)), -1)
        values = torch.stack([head(inp).squeeze(-1) for head in self.heads], 0)
        if self.latent_fusion:
            state_gate = self.latent_fusion_state(features).tanh()
            code = z.to(features)
            interaction = torch.stack([
                head(state_gate * projection(code).tanh()).squeeze(-1)
                for projection, head in zip(self.latent_fusion_codes, self.latent_fusion_readouts)
            ], dim=0)
            values = values + interaction
        if self.prior_scale > 0:
            values = values + self.prior_scale * torch.stack(
                [prior(inp).squeeze(-1) for prior in self.priors], 0)
        return values

    def all_values(self, policy, state, geometry, z):
        return self.evaluate_features(self.encode_state(state), z)

    def mean_value(self, state, z):
        return self.evaluate_features(self.encode_state(state), z).mean(0)

    def forward(self, policy, state, geometry, z):
        return self.mean_value(state, z)

    def disagreement(self, policy, state, geometry, z):
        return self.all_values(policy, state, geometry, z).std(0, unbiased=False)

    def population_values(self, features, population):
        """批量评价状态与共享种群的笛卡尔积，返回 [M, B, K]。"""
        B, K = features.size(0), population.size(0)
        features = features[:, None, :].expand(B, K, -1).reshape(B * K, -1)
        z = population[None, :, :].expand(B, K, -1).reshape(B * K, -1)
        return self.evaluate_features(features, z).reshape(self.n_critics, B, K)


__all__ = ["ValueNetwork", "ONLINE_CONTEXT_DIM", "ONLINE_VALUE_SEMANTICS"]
