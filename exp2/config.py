"""复用全部 PPO 命令行参数，只增加 exp2 参数。"""
import argparse
from dataclasses import dataclass,asdict,fields,replace
from CALSESD.training.ppo_config import PPOConfig,parse_args

@dataclass
class Exp2Config(PPOConfig):
    exp2_backend: str='rollout'
    exp2_chunk: int=4
    exp2_repeats: int=1

def normalize(cfg):
    # 独立入口自行标记 exp2，原配置映射仅作为参数构造工具。
    return replace(cfg,ablation_family='none',ablation_variant='default',mode='evolution',
        ablation_fixed_sigma=None,ablation_min_gain=None,gradient_steps=0,
        lcb_lambda=0.,reopt_min_gain_ratio=0.,ablation_record_diagnostics=False,
        online_search_budget_seconds=0.,online_batch_budget_seconds=0.,
        vary_instance_seed=True,bootstrap_time_limits=False,
        validation_initial_latent_source='common')

def parse(argv=None):
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--exp2-backend',choices=['rollout','critic'],default='rollout')
    p.add_argument('--exp2-chunk',type=int,default=4)
    p.add_argument('--exp2-repeats',type=int,default=1)
    own,rest=p.parse_known_args(argv)
    cfg=normalize(Exp2Config(**asdict(parse_args(rest)),**vars(own)))
    if cfg.exp2_chunk<1 or cfg.exp2_repeats<1:
        p.error('exp2-chunk and exp2-repeats must be positive')
    if cfg.resume and cfg.init_checkpoint: p.error('choose resume or init-checkpoint')
    return cfg

def from_saved(saved):
    keys={f.name for f in fields(Exp2Config)}
    return Exp2Config(**{k:v for k,v in saved.items() if k in keys})
