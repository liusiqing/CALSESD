"""PPO 配置:回合内周期性 Critic 引导的隐空间搜索。

搜索时间预算为 0 表示不设对应的墙钟上限。exp1 固定使用单步修正协议。
本精简版不包含参数合法性检查,配置错误会在使用处暴露。
"""

from __future__ import annotations
import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Optional


# 核心模块:全部超参数。分组为运行输出、有限任务/环境、PPO、网络、在线搜索触发、
# 隐空间种群搜索、消融、验证、日志与断点。
@dataclass
class PPOConfig:
    # Run/output.
    output_dir: str = "outputs"
    checkpoints_dir: str = "checkpoints"
    seed: int = 42
    device: str = "auto"
    updates: int = 300
    resume: str = ""
    init_checkpoint: str = ""

    # Finite task; the environment applies its real cutoff and failure reward.
    grid_size: int = 12
    num_customers: int = 100
    vehicle_capacity: int = 15
    max_episode_steps: int = 200
    failure_penalty: float = 2.0
    demand_noise_std: float = 1.0
    traffic_noise_std: float = 1.0
    traffic_trend: float = 5.0
    traffic_beta: float = 0.07
    traffic_trend_mode: str = "relative"
    traffic_trend_ratio: float = 0.3
    vary_instance_seed: bool = True
    bootstrap_time_limits: bool = False

    # PPO.
    batch_size: int = 16
    minibatch_size: int = 64
    ppo_epochs: int = 4
    learning_rate: float = 0.0003
    value_learning_rate: float = 0.0003
    target_kl: Optional[float] = 0.03
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    bootstrap_prob: float = 0.8
    reward_scale: float = 0.1  # 固定奖励缩放,Critic 目标落在 O(1~10)
    gae_keep_bootstrap: bool = True  # 切换步用 V(s',z_old) 做 bootstrap

    # Actor / Critic network.
    hidden_dim: int = 128
    dynamic_dim: int = 64
    latent_dim: int = 64
    num_heads: int = 8
    static_gat_layers: int = 2
    static_transformer_layers: int = 2
    dropout: float = 0.0
    n_critics: int = 4
    critic_prior_scale: float = 0.5  # 随机先验网络强度
    latent_bound: float = 1.0
    z_resample_prob: float = 0.5  # 每个 episode 开始时按行重采样 z 的概率
    actor_cache_static_fusion: bool = True
    critic_cache_static: bool = True
    critic_static_cache_size: int = 4

    # Attempts are per row; search wall-clock budgets are shared across rows.
    search_scope: str = "episode"
    mode: str = "evolution"  # keep 关闭搜索
    search_warmup_updates: int = 1
    online_search_interval: int = 30
    online_cooldown_steps: int = 5
    online_max_attempts_per_episode: int = 3
    online_search_budget_seconds: float = 0.0  # 0: no per-call time limit.
    online_batch_budget_seconds: float = 0.0  # 0: no full-batch time limit.
    online_min_remaining_customers: int = 2

    # Shared latent population search.
    reopt_candidates: int = 64
    evolution_generations: int = 3
    gradient_steps: int = 1  # exp1 要求为 1;其 none 变体实际不修正
    gradient_lr: float = 0.05
    latent_gradient_normalize: bool = True  # 步长 = lr * 单位方向,与批大小和 Critic 尺度无关
    uncertainty_kappa: float = 1.0
    reopt_min_gain: float = 0.0
    reopt_min_gain_ratio: float = 1.0  # 接受阈值 = min_gain + ratio * 不确定性尺度
    lcb_lambda: float = 1.0  # 保守得分 = 均值 - lambda * 集成标准差
    elite_fraction: float = 0.25
    immigrant_fraction: float = 0.125
    mutation_sigma_base: float = 0.15
    mutation_sigma_min: float = 0.03
    mutation_sigma_max: float = 0.4
    max_gradient_norm: float = 1.0
    crossover_alpha_min: float = 0.2
    crossover_alpha_max: float = 0.8

    # none/default 使用完整默认搜索;exp1/exp3 是显式消融。
    ablation_family: str = "none"
    ablation_variant: str = "default"
    # 仅用于 exp1/* 以及 exp3 的 fixed、trust_only。
    ablation_fixed_sigma: Optional[float] = None
    ablation_min_gain: Optional[float] = None
    ablation_record_diagnostics: bool = True

    # system: 在线控制器;policy: 不搜索,使用公共带种子先验。
    validation_interval: int = 50
    validation_seeds: int = 16
    validation_paired: bool = True  # 每次验证同时跑 policy 协议,记录搜索收益
    validation_seed: int = 1000000
    validation_protocol: str = "system"
    validation_initial_latent_source: str = "common"  # policy 始终使用 common

    # Logging/checkpoints.
    checkpoint_interval: int = 25
    latest_checkpoint_interval: int = 5
    save_step_csv: bool = False


_OPTIONAL_FLOATS = {"target_kl", "ablation_fixed_sigma", "ablation_min_gain"}
_HIDDEN_FIELDS = {
    "demand_noise_std", "traffic_noise_std", "traffic_trend", "traffic_beta",
    "vary_instance_seed", "bootstrap_time_limits", "value_learning_rate", "gamma",
    "gae_lambda", "clip_coef", "value_coef", "entropy_coef", "max_grad_norm",
    "bootstrap_prob", "hidden_dim", "dynamic_dim", "latent_dim", "num_heads",
    "static_gat_layers", "static_transformer_layers", "dropout", "n_critics",
    "latent_bound", "actor_cache_static_fusion", "critic_cache_static",
    "critic_static_cache_size", "search_scope", "reopt_min_gain", "elite_fraction",
    "immigrant_fraction", "mutation_sigma_base", "mutation_sigma_min",
    "mutation_sigma_max", "max_gradient_norm", "crossover_alpha_min",
    "crossover_alpha_max", "validation_seed", "checkpoint_interval", "save_step_csv",
    "ablation_record_diagnostics", "reward_scale", "gae_keep_bootstrap",
    "z_resample_prob", "reopt_min_gain_ratio", "lcb_lambda", "traffic_trend_mode",
    "traffic_trend_ratio", "critic_prior_scale", "latent_gradient_normalize",
    "latest_checkpoint_interval",
}


def _optional_float(text):
    return None if text.lower() in ("none", "off") else float(text)


# 核心模块:命令行解析。每个数据类字段对应一个 --flag(下划线换成连字符);
# 布尔字段支持 --flag / --no-flag;高级参数默认隐藏,用 --help-all 显示。
def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    show_all = "--help-all" in argv
    if show_all:
        argv = [arg for arg in argv if arg != "--help-all"] + ["--help"]
    parser = argparse.ArgumentParser(
        description="Actor-only PPO with periodic in-episode Critic-guided latent search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--help-all", action="store_true", help="显示高级参数后退出")
    parser.add_argument("--show-config", action="store_true", help="打印完整配置后退出")
    for name, field in PPOConfig.__dataclass_fields__.items():
        value = field.default
        flag = "--" + name.replace("_", "-")
        help_text = name if show_all or name not in _HIDDEN_FIELDS else argparse.SUPPRESS
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value, help=help_text)
        else:
            converter = _optional_float if name in _OPTIONAL_FLOATS else type(value)
            parser.add_argument(flag, type=converter, default=value, help=help_text)
    args = vars(parser.parse_args(argv))
    dump = args.pop("show_config")
    args.pop("help_all", None)
    cfg = PPOConfig(**args)
    if dump:
        print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2, allow_nan=False))
        parser.exit()
    return cfg


__all__ = ["PPOConfig", "parse_args"]

if __name__ == "__main__":
    print(json.dumps(asdict(parse_args()), ensure_ascii=False, indent=2, allow_nan=False))