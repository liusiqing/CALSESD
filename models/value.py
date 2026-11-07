"""在线 Actor-控制器价值网络:集成 Critic + 静态几何缓存(精简版:已删除输入检查与缓存统计)。

状态特征 = 节点池化(全体/剩余客户/当前节点/仓库) + 物理 context[:2] + 在线触发器 online_context[6] + 2 个任务摘要。
价值目标必须包含未来真实发生的在线隐变量切换;网络本身不保证旧的固定 z 目标正确。
本模块不推进环境,也不优化参数。
"""

from collections import OrderedDict
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch

ONLINE_CONTEXT_DIM = 6
ONLINE_VALUE_SEMANTICS = "online_actor_controller_v1"


class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim=128, dynamic_dim=64, latent_dim=16, n_critics=4, kind="hybrid", prior_scale=0.0):
        super().__init__()
        self.kind, self.latent_dim, self.n_critics = kind, latent_dim, n_critics
        H = hidden_dim
        self.online_context_dim = ONLINE_CONTEXT_DIM
        self.value_semantics = ONLINE_VALUE_SEMANTICS
        self.state_feature_dim = 4 * H + 2 + ONLINE_CONTEXT_DIM + 2
        self.geometry_encoder = nn.Sequential(nn.Linear(2, H), nn.GELU(), nn.LayerNorm(H))
        self.dynamic_encoder = nn.Sequential(nn.Linear(9, H), nn.GELU(), nn.Linear(H, H), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.LayerNorm(H))
        self.spatial = GATv2Conv(H, H, heads=4 if H % 4 == 0 else 1, concat=False, edge_dim=1, dropout=0.0)
        self.norm = nn.LayerNorm(H)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.state_feature_dim + latent_dim, H),
                    nn.GELU(),
                    nn.Linear(H, H),
                    nn.GELU(),
                    nn.Linear(H, 1),
                )
                for _ in range(n_critics)
            ]
        )
        self.prior_scale = float(prior_scale)
        if self.prior_scale > 0:  # 冻结的随机先验:数据之外各 head 的差异被放大
            in_dim = self.state_feature_dim + latent_dim
            self.priors = nn.ModuleList(
                [nn.Sequential(nn.Linear(in_dim, H // 2), nn.GELU(), nn.Linear(H // 2, 1)) for _ in range(n_critics)]
            )
            for p in self.priors.parameters():
                p.requires_grad_(False)
        # 普通属性:不进入 state_dict,不与 Actor 共享
        self.cache_static = True
        self.static_cache_size = 4
        self._static_cache = OrderedDict()
        self._static_cache_signature = None
        self._static_cache_next_key = 0

    def clear_static_cache(self):
        self._static_cache.clear()
        self._static_cache_signature = None
        self._static_cache_next_key = 0

    def configure_static_cache(self, *, enabled=True, max_entries=4):
        self.clear_static_cache()
        self.cache_static = enabled
        self.static_cache_size = int(max_entries)

    def _apply(self, fn, *args, **kwargs):
        # .to()/.cuda()/.cpu()/dtype 转换后不能保留旧设备上的缓存
        self.clear_static_cache()
        return super()._apply(fn, *args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.clear_static_cache()
        try:
            return super().load_state_dict(state_dict, *args, **kwargs)
        finally:
            self.clear_static_cache()

    @staticmethod
    def _autocast_active(device_type):
        try:
            return torch.is_autocast_enabled(device_type)
        except TypeError:
            if device_type == "cpu":
                checker = getattr(torch, "is_autocast_cpu_enabled", None)
                return bool(checker()) if checker is not None else False
            return torch.is_autocast_enabled()

    def _geometry_signature(self):
        # 优化器或 no_grad 的原地更新会增加 Tensor._version
        parameters = tuple((id(p), p._version, p.device, p.dtype) for p in self.geometry_encoder.parameters())
        modes = tuple(module.training for module in self.geometry_encoder.modules())
        return parameters, modes

    # 核心模块:坐标编码及有界 LRU 缓存。带梯度、inference_mode 或 autocast 时总是重算,
    # 保证反向传播不会使用被 detach 的 rollout 缓存。
    def encode_static(self, state):
        pos = state.pos.float()
        if (
            not self.cache_static
            or torch.is_grad_enabled()
            or torch.is_inference_mode_enabled()
            or self._autocast_active(pos.device.type)
        ):
            self.clear_static_cache()
            return self.geometry_encoder(pos)
        signature = self._geometry_signature()
        if signature != self._static_cache_signature:
            self.clear_static_cache()
            self._static_cache_signature = signature
        hit = None
        for key, (saved_pos, embedding) in self._static_cache.items():
            if (
                saved_pos.shape == pos.shape
                and saved_pos.device == pos.device
                and saved_pos.dtype == pos.dtype
                and torch.equal(saved_pos, pos)
            ):
                hit = key
                break
        if hit is not None:
            self._static_cache.move_to_end(hit)
            return self._static_cache[hit][1]
        embedding = self.geometry_encoder(pos)
        key = self._static_cache_next_key
        self._static_cache_next_key += 1
        self._static_cache[key] = (pos.detach().clone(), embedding.detach())
        while len(self._static_cache) > self.static_cache_size:
            self._static_cache.popitem(last=False)
        return embedding

    @staticmethod
    def pool(x, mask):
        weights = mask.to(x.dtype).unsqueeze(-1)
        return (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    # 核心模块:状态编码。节点融合 -> 一层 GATv2 残差 -> 四种池化 -> 拼接物理 context、
    # 在线控制器 context 和两个任务摘要,得到 [B, state_feature_dim]。搜索时每个状态只编码一次。
    def encode_state(self, state):
        online = state.online_context
        node = self.fusion(
            torch.cat(
                (self.encode_static(state), self.dynamic_encoder(torch.cat((state.x, state.node_aux), -1).float())), -1
            )
        )
        node = self.norm(node + torch.nn.functional.gelu(self.spatial(node, state.edge_index, state.edge_attr)))
        dense, valid = to_dense_batch(node, state.batch)
        x, _ = to_dense_batch(state.x, state.batch)
        aux, _ = to_dense_batch(state.node_aux, state.batch)
        remaining = (aux[:, :, 0] > 0.5) & (x[:, :, 2] < 0.5) & valid
        current = (x[:, :, 3] > 0.5) & valid
        depot = (aux[:, :, 1] > 0.5) & valid
        context = state.context[:, :2].to(dense)
        online = online.to(dense)
        counts = valid.sum(1).clamp_min(1).to(dense.dtype)
        extra = torch.stack((remaining.sum(1) / counts, torch.log1p((x[:, :, 0] * remaining).sum(1))), -1)
        return torch.cat(
            (
                self.pool(dense, valid),
                self.pool(dense, remaining),
                self.pool(dense, current),
                self.pool(dense, depot),
                context,
                online,
                extra,
            ),
            -1,
        )

    # 核心模块:集成价值头。输入 [状态特征, z],输出 [n_critics, B];可选加冻结随机先验。
    def evaluate_features(self, features, z):
        if z.ndim == 1:
            z = z[None, :]
        if z.size(0) == 1:
            z = z.expand(features.size(0), -1)
        inp = torch.cat((features, z.to(features)), -1)
        out = torch.stack([h(inp).squeeze(-1) for h in self.heads], 0)
        if self.prior_scale > 0:
            out = out + self.prior_scale * torch.stack([p(inp).squeeze(-1) for p in self.priors], 0)
        return out

    def all_values(self, policy, state, geometry, z):
        return self.evaluate_features(self.encode_state(state), z)

    def mean_value(self, state, z):
        return self.evaluate_features(self.encode_state(state), z).mean(0)

    def forward(self, policy, state, geometry, z):
        return self.all_values(policy, state, geometry, z).mean(0)

    def disagreement(self, policy, state, geometry, z):
        return self.all_values(policy, state, geometry, z).std(0, unbiased=False)

    # 核心模块:种群评价。把 B 个状态与 K 个候选 z 做笛卡尔积,返回 [n_critics, B, K]。
    def population_values(self, features, population):
        B, K = features.size(0), population.size(0)
        f = features[:, None, :].expand(B, K, -1).reshape(B * K, -1)
        z = population[None, :, :].expand(B, K, -1).reshape(B * K, -1)
        return self.evaluate_features(f, z).reshape(self.n_critics, B, K)


__all__ = ["ValueNetwork", "ONLINE_CONTEXT_DIM", "ONLINE_VALUE_SEMANTICS"]
