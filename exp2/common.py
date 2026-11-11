"""模型构造与运行设备。"""
import torch
from CALSESD.models import GeometryDynamicPointerPolicy,ValueNetwork

def device_for(value):
    d=torch.device('cuda' if value=='auto' and torch.cuda.is_available() else 'cpu' if value=='auto' else value)
    return torch.device('cuda',torch.cuda.current_device()) if d.type=='cuda' and d.index is None else d

def models(cfg,device):
    actor=GeometryDynamicPointerPolicy(hidden_dim=cfg.hidden_dim,dynamic_dim=cfg.dynamic_dim,
        latent_dim=cfg.latent_dim,num_heads=cfg.num_heads,static_gat_layers=cfg.static_gat_layers,
        static_transformer_layers=cfg.static_transformer_layers,dropout=cfg.dropout).to(device)
    critic=ValueNetwork(hidden_dim=cfg.hidden_dim,latent_dim=cfg.latent_dim,
        n_critics=cfg.n_critics,prior_scale=cfg.critic_prior_scale).to(device)
    actor.cache_static_fusion=cfg.actor_cache_static_fusion
    critic.configure_static_cache(enabled=cfg.critic_cache_static,max_entries=cfg.critic_static_cache_size)
    return actor,critic
