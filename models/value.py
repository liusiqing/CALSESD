"""Online Actor-controller value ensemble with static geometry caching.

Context fusion: physical context[:2] (traffic phase, remaining episode fraction)
online_context[6] + two task summaries.

Targets must include the actual future online-controller switches. The network
alone does not make old fixed-z GAE/returns correct; collector/update adaptation
is still required. No environment transitions or parameter optimization here.
"""
from collections import OrderedDict
from numbers import Integral
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch

ONLINE_CONTEXT_DIM = 6
ONLINE_VALUE_SEMANTICS = 'online_actor_controller_v1'


class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim=128, dynamic_dim=64, latent_dim=16,
                 n_critics=4, kind="hybrid", prior_scale=0.0):
        super().__init__()
        if n_critics < 2 or kind != "hybrid":
            raise ValueError("invalid value configuration")
        self.kind, self.latent_dim, self.n_critics = kind, latent_dim, n_critics
        H = hidden_dim
        self.online_context_dim = ONLINE_CONTEXT_DIM
        self.value_semantics = ONLINE_VALUE_SEMANTICS
        self.state_feature_dim = 4*H+2+ONLINE_CONTEXT_DIM+2
        self.geometry_encoder = nn.Sequential(nn.Linear(2,H), nn.GELU(), nn.LayerNorm(H))
        self.dynamic_encoder = nn.Sequential(nn.Linear(9,H), nn.GELU(), nn.Linear(H,H), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(2*H,H), nn.GELU(), nn.LayerNorm(H))
        self.spatial = GATv2Conv(H,H,heads=4 if H%4==0 else 1,concat=False,edge_dim=1,dropout=0.)
        self.norm = nn.LayerNorm(H)
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(self.state_feature_dim+latent_dim,H),nn.GELU(),nn.Linear(H,H),nn.GELU(),nn.Linear(H,1))
            for _ in range(n_critics)])
        self.prior_scale = float(prior_scale)
        if self.prior_scale < 0:
            raise ValueError('prior_scale must be non-negative')
        if self.prior_scale > 0:   # 冻结的随机先验:数据之外各 head 的差异被放大
            in_dim = self.state_feature_dim + latent_dim
            self.priors = nn.ModuleList([nn.Sequential(
                nn.Linear(in_dim, H // 2), nn.GELU(), nn.Linear(H // 2, 1))
                for _ in range(n_critics)])
            for p in self.priors.parameters():
                p.requires_grad_(False)
        # Ordinary attributes: never serialized in state_dict, never shared with Actor.
        self.cache_static = True
        self.static_cache_size = 4
        self._static_cache = OrderedDict()
        self._static_cache_signature = None
        self._static_cache_next_key = 0
        self._static_cache_hits = 0
        self._static_cache_misses = 0
        self._static_cache_bypasses = 0

    def clear_static_cache(self, *, reset_stats=False):
        """Drop cached tensors; call after nonstandard parameter edits via .data."""
        self._static_cache.clear()
        self._static_cache_signature = None
        self._static_cache_next_key = 0
        if reset_stats:
            self._static_cache_hits = 0
            self._static_cache_misses = 0
            self._static_cache_bypasses = 0

    def configure_static_cache(self, *, enabled=True, max_entries=4):
        """Optional cache controls; no trainer/config changes are required."""
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be bool')
        if isinstance(max_entries, bool) or not isinstance(max_entries, Integral) or max_entries < 1:
            raise ValueError('max_entries must be a positive integer')
        self.clear_static_cache()
        self.cache_static = enabled
        self.static_cache_size = int(max_entries)

    def static_cache_info(self):
        return dict(enabled=self.cache_static, max_entries=self.static_cache_size,
                    entries=len(self._static_cache), hits=self._static_cache_hits,
                    misses=self._static_cache_misses, bypasses=self._static_cache_bypasses)

    def _apply(self, fn, *args, **kwargs):
        # .to(), .cuda(), .cpu(), dtype conversions must not retain old tensors.
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
            if device_type == 'cpu':
                checker = getattr(torch, 'is_autocast_cpu_enabled', None)
                return bool(checker()) if checker is not None else False
            return torch.is_autocast_enabled()

    def _geometry_signature(self):
        # Optimizer/no_grad in-place updates increment Tensor._version.
        parameters = tuple((id(p), p._version, p.device, p.dtype)
                           for p in self.geometry_encoder.parameters())
        modes = tuple(module.training for module in self.geometry_encoder.modules())
        return parameters, modes

    def encode_static(self, state):
        """Encode coordinates only; exact row-wise position matching, bounded LRU.

        Gradient-enabled calls always recompute, including policy.eval() PPO
        updates. Detached rollout geometry is never reused for backward.
        """
        pos = state.pos.float()
        if pos.ndim != 2 or pos.size(1) != 2:
            raise ValueError('Critic expects pos with shape [N,2]')
        if pos.size(0) != state.x.size(0):
            raise ValueError('Critic position/node count mismatch')
        if (not self.cache_static or torch.is_grad_enabled()
                or torch.is_inference_mode_enabled()
                or self._autocast_active(pos.device.type)):
            self._static_cache_bypasses += 1
            self.clear_static_cache()
            return self.geometry_encoder(pos)
        signature = self._geometry_signature()
        if signature != self._static_cache_signature:
            self.clear_static_cache()
            self._static_cache_signature = signature
        hit = None
        for key, (saved_pos, embedding) in self._static_cache.items():
            if (saved_pos.shape == pos.shape and saved_pos.device == pos.device
                    and saved_pos.dtype == pos.dtype and torch.equal(saved_pos, pos)):
                hit = key
                break
        if hit is not None:
            self._static_cache.move_to_end(hit)
            self._static_cache_hits += 1
            return self._static_cache[hit][1]
        self._static_cache_misses += 1
        embedding = self.geometry_encoder(pos)
        key = self._static_cache_next_key
        self._static_cache_next_key += 1
        self._static_cache[key] = (pos.detach().clone(), embedding.detach())
        while len(self._static_cache) > self.static_cache_size:
            self._static_cache.popitem(last=False)
        return embedding


    @staticmethod
    def pool(x,mask):
        weights=mask.to(x.dtype).unsqueeze(-1)
        return (x*weights).sum(1)/weights.sum(1).clamp_min(1.)

    def encode_state(self,state):
        if state.x.ndim != 2 or state.node_aux.ndim != 2:
            raise ValueError('Critic expects node matrices x and node_aux')
        if state.x.size(1) + state.node_aux.size(1) != 9:
            raise ValueError('Critic expects 9 concatenated dynamic node features')
        if state.context.ndim != 2 or state.context.size(1) < 2:
            raise ValueError('Critic expects traffic phase and remaining episode fraction in context[:2]')
        online = getattr(state, 'online_context', None)
        if not torch.is_tensor(online) or online.shape != (state.context.size(0), self.online_context_dim):
            raise ValueError('Critic requires online_context[B,6]; bind the online trigger before producing state')
        if not online.is_floating_point() or not bool(torch.isfinite(online).all()):
            raise ValueError('online_context must be finite floating point')
        if bool(((online < -1e-6) | (online > 1.+1e-6)).any()):
            raise ValueError('online_context must contain normalized controller features in [0,1]')
        node=self.fusion(torch.cat((self.encode_static(state),
            self.dynamic_encoder(torch.cat((state.x,state.node_aux),-1).float())), -1))
        node=self.norm(node+torch.nn.functional.gelu(self.spatial(node,state.edge_index,state.edge_attr)))
        dense,valid=to_dense_batch(node,state.batch)
        x,_=to_dense_batch(state.x,state.batch)
        aux,_=to_dense_batch(state.node_aux,state.batch)
        remaining=(aux[:,:,0]>0.5)&(x[:,:,2]<0.5)&valid
        current=(x[:,:,3]>0.5)&valid
        depot=(aux[:,:,1]>0.5)&valid
        if dense.size(0) != state.context.size(0):
            raise ValueError('Critic graph/context batch size mismatch')
        context=state.context[:, :2].to(dense)
        online=online.to(dense)
        counts=valid.sum(1).clamp_min(1).to(dense.dtype)
        extra=torch.stack((remaining.sum(1)/counts,torch.log1p((x[:,:,0]*remaining).sum(1))), -1)
        return torch.cat((self.pool(dense,valid),self.pool(dense,remaining),
            self.pool(dense,current),self.pool(dense,depot),context,online,extra),-1)

    def evaluate_features(self,features,z):
        if features.ndim != 2 or features.size(1) != self.state_feature_dim:
            raise ValueError('online Critic state feature width mismatch')
        if z.ndim==1:z=z[None,:]
        if z.ndim!=2 or z.size(1)!=self.latent_dim:
            raise ValueError("latent shape mismatch")
        if z.size(0)==1:z=z.expand(features.size(0),-1)
        if z.size(0)!=features.size(0):raise ValueError("latent batch mismatch")
        inp=torch.cat((features,z.to(features)), -1)
        out=torch.stack([h(inp).squeeze(-1) for h in self.heads],0)
        if self.prior_scale > 0:
            out=out+self.prior_scale*torch.stack([p(inp).squeeze(-1) for p in self.priors],0)
        return out
    def all_values(self,policy,state,geometry,z):
        return self.evaluate_features(self.encode_state(state),z)

    def mean_value(self, state, z):
        return self.evaluate_features(self.encode_state(state), z).mean(0)

    def forward(self,policy,state,geometry,z):
        return self.all_values(policy,state,geometry,z).mean(0)

    def disagreement(self,policy,state,geometry,z):
        return self.all_values(policy,state,geometry,z).std(0,unbiased=False)

    def population_values(self,features,population):
        B,K=features.size(0),population.size(0)
        f=features[:,None,:].expand(B,K,-1).reshape(B*K,-1)
        z=population[None,:,:].expand(B,K,-1).reshape(B*K,-1)
        return self.evaluate_features(f,z).reshape(self.n_critics,B,K)


__all__ = ['ValueNetwork', 'ONLINE_CONTEXT_DIM', 'ONLINE_VALUE_SEMANTICS']
