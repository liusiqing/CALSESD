"""PPO configuration for periodic in-episode Critic-guided latent search.

No update-period search, representative pools, rollout-search providers or
second-stage route-planner settings. Zero time budgets mean no corresponding
wall-clock limit. exp1 uses a fixed one-step refinement protocol.
"""

from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass, asdict
from numbers import Integral, Real
from typing import Optional


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
    mode: str = "evolution"  # keep disables search; none/default is not keep.
    search_warmup_updates: int = 10
    online_search_interval: int = 30
    online_cooldown_steps: int = 5  # Attempt-based; interval is the stronger guard.
    online_max_attempts_per_episode: int = 3
    online_search_budget_seconds: float = 0.0  # 0: no per-call time limit.
    online_batch_budget_seconds: float = 0.0  # 0: no full-batch time limit.
    online_min_remaining_customers: int = 2

    # Shared latent population search.
    reopt_candidates: int = 64
    evolution_generations: int = 3
    gradient_steps: int = 1  # exp1 requires 1; its none variant applies 0 steps.
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

    # none/default keeps the full default search; exp1/exp3 are explicit ablations.
    ablation_family: str = "none"
    ablation_variant: str = "default"
    # Only valid for exp1/* and exp3/fixed or exp3/trust_only.
    ablation_fixed_sigma: Optional[float] = None
    ablation_min_gain: Optional[float] = None
    ablation_record_diagnostics: bool = True

    # system: online controller; policy: no search with a common seeded prior.
    validation_interval: int = 50
    validation_seeds: int = 16
    validation_paired: bool = True  # 每次验证同时跑 policy 协议,记录搜索收益
    validation_seed: int = 1000000
    validation_protocol: str = "system"
    validation_initial_latent_source: str = "common"  # policy always uses common.

    # Logging/checkpoints.
    checkpoint_interval: int = 25
    latest_checkpoint_interval: int = 5
    save_step_csv: bool = False


_OPTIONAL_FLOATS = {"target_kl", "ablation_fixed_sigma", "ablation_min_gain"}
_HIDDEN_FIELDS = {
    "demand_noise_std",
    "traffic_noise_std",
    "traffic_trend",
    "traffic_beta",
    "vary_instance_seed",
    "bootstrap_time_limits",
    "value_learning_rate",
    "gamma",
    "gae_lambda",
    "clip_coef",
    "value_coef",
    "entropy_coef",
    "max_grad_norm",
    "bootstrap_prob",
    "hidden_dim",
    "dynamic_dim",
    "latent_dim",
    "num_heads",
    "static_gat_layers",
    "static_transformer_layers",
    "dropout",
    "n_critics",
    "latent_bound",
    "actor_cache_static_fusion",
    "critic_cache_static",
    "critic_static_cache_size",
    "search_scope",
    "reopt_min_gain",
    "elite_fraction",
    "immigrant_fraction",
    "mutation_sigma_base",
    "mutation_sigma_min",
    "mutation_sigma_max",
    "max_gradient_norm",
    "crossover_alpha_min",
    "crossover_alpha_max",
    "validation_seed",
    "checkpoint_interval",
    "save_step_csv",
    "ablation_record_diagnostics",
    "reward_scale",
    "gae_keep_bootstrap",
    "z_resample_prob",
    "reopt_min_gain_ratio",
    "lcb_lambda",
    "traffic_trend_mode",
    "traffic_trend_ratio",
    "critic_prior_scale",
    "latent_gradient_normalize",
    "latest_checkpoint_interval",
}


def _optional_float(text):
    return None if text.lower() in ("none", "off") else float(text)


def resolved_config(cfg):
    validate_config(cfg)
    return asdict(cfg)


def parse_args(argv=None):
    import sys

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
    try:
        cfg = PPOConfig(**args)
        validate_config(cfg)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if dump:
        print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2, allow_nan=False))
        parser.exit()
    return cfg


def _check_types(cfg):
    for name, field in PPOConfig.__dataclass_fields__.items():
        value, default = getattr(cfg, name), field.default
        if name in _OPTIONAL_FLOATS:
            if value is None:
                continue
            valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
        elif isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            valid = not isinstance(value, bool) and isinstance(value, Integral)
        elif isinstance(default, float):
            valid = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"{name}类型错误或值无效")


def validate_config(cfg):
    _check_types(cfg)
    positive = (
        "updates",
        "grid_size",
        "vehicle_capacity",
        "max_episode_steps",
        "batch_size",
        "minibatch_size",
        "ppo_epochs",
        "hidden_dim",
        "dynamic_dim",
        "latent_dim",
        "num_heads",
        "static_transformer_layers",
        "n_critics",
        "critic_static_cache_size",
        "online_search_interval",
        "online_max_attempts_per_episode",
        "reopt_candidates",
        "evolution_generations",
        "validation_interval",
        "validation_seeds",
        "checkpoint_interval",
        "latest_checkpoint_interval",
    )
    if any(getattr(cfg, name) < 1 for name in positive):
        raise ValueError("训练、网络、搜索计数必须为正整数")
    nonnegative = (
        "seed",
        "num_customers",
        "static_gat_layers",
        "gradient_steps",
        "search_warmup_updates",
        "online_cooldown_steps",
        "online_min_remaining_customers",
        "validation_seed",
    )
    if any(getattr(cfg, name) < 0 for name in nonnegative):
        raise ValueError("种子、计数或预热参数不能为负")
    if cfg.search_scope != "episode" or cfg.mode not in ("keep", "evolution"):
        raise ValueError("仅支持search_scope=episode，mode=keep/evolution")
    if cfg.online_cooldown_steps > cfg.online_search_interval:
        raise ValueError("online_cooldown_steps不能大于online_search_interval")
    if cfg.online_search_budget_seconds < 0 or cfg.online_batch_budget_seconds < 0:
        raise ValueError("搜索软预算不能为负；0表示不设对应时间上限")
    if cfg.seed + 2000 >= 2**63:
        raise ValueError("seed过大，不能构造独立搜索随机流")
    if cfg.grid_size < 2 or cfg.num_customers > cfg.grid_size * cfg.grid_size - 1:
        raise ValueError("grid_size不足以容纳客户数")
    if cfg.n_critics < 2 or cfg.hidden_dim % cfg.num_heads:
        raise ValueError("n_critics>=2且hidden_dim须能被num_heads整除")
    if not (
        0 < cfg.gamma <= 1
        and 0 <= cfg.gae_lambda <= 1
        and 0 < cfg.clip_coef < 1
        and 0 < cfg.bootstrap_prob <= 1
        and 0 <= cfg.traffic_beta <= 1
        and 0 <= cfg.dropout < 1
    ):
        raise ValueError("PPO/网络/交通参数范围错误")
    positive_scales = (
        "learning_rate",
        "value_learning_rate",
        "max_grad_norm",
        "latent_bound",
        "max_gradient_norm",
        "failure_penalty",
        "reward_scale",
    )
    if any(getattr(cfg, name) <= 0 for name in positive_scales):
        raise ValueError("学习率、裁剪量和任务尺度必须为正")
    nonnegative_scales = (
        "value_coef",
        "entropy_coef",
        "demand_noise_std",
        "traffic_noise_std",
        "gradient_lr",
        "uncertainty_kappa",
        "reopt_min_gain",
        "reopt_min_gain_ratio",
        "lcb_lambda",
        "traffic_trend_ratio",
        "critic_prior_scale",
    )
    if not 0.0 <= cfg.z_resample_prob <= 1.0:
        raise ValueError("z_resample_prob必须在[0,1]内")
    if any(getattr(cfg, name) < 0 for name in nonnegative_scales):
        raise ValueError("损失系数或搜索尺度不能为负")
    if cfg.target_kl is not None and cfg.target_kl <= 0:
        raise ValueError("target_kl必须为正或None")
    if not 0 < cfg.elite_fraction <= 1 or not 0 <= cfg.immigrant_fraction < 1:
        raise ValueError("精英/移民比例范围错误")
    if not 0 <= cfg.crossover_alpha_min <= cfg.crossover_alpha_max <= 1:
        raise ValueError("交叉系数范围错误")
    if not 0 < cfg.mutation_sigma_min <= cfg.mutation_sigma_base <= cfg.mutation_sigma_max:
        raise ValueError("变异范围错误")

    if cfg.ablation_family == "exp2":
        raise ValueError("阶段一暂不接入纯rollout搜索")
    if cfg.traffic_trend_mode not in ("additive", "relative"):
        raise ValueError("traffic_trend_mode只支持additive/relative")
    supported = {
        "none": ("default",),
        "exp1": ("none", "value_gradient", "random_matched"),
        "exp3": ("fixed", "trust_only", "mutation_only", "both"),
    }
    if cfg.ablation_family not in supported or cfg.ablation_variant not in supported[cfg.ablation_family]:
        raise ValueError(f"不支持的消融组合：{cfg.ablation_family}/{cfg.ablation_variant}")
    if cfg.ablation_family != "none" and cfg.mode != "evolution":
        raise ValueError("消融对比要求mode=evolution")
    if cfg.ablation_family == "exp1":
        if cfg.gradient_lr <= 0:
            raise ValueError("exp1需要gradient_lr>0")
        if cfg.gradient_steps != 1:
            raise ValueError("exp1固定使用单步修正协议，gradient_steps必须为1；none变体实际不修正")
    if cfg.ablation_family == "exp3" and (cfg.gradient_steps < 1 or cfg.gradient_lr <= 0 or cfg.uncertainty_kappa <= 0):
        raise ValueError("exp3需要有效梯度步和正uncertainty_kappa")
    if cfg.ablation_family == "none" and (cfg.ablation_fixed_sigma is not None or cfg.ablation_min_gain is not None):
        raise ValueError("固定sigma/min_gain覆盖仅用于显式消融")
    if cfg.ablation_fixed_sigma is not None:
        if cfg.ablation_family == "exp3" and cfg.ablation_variant in ("mutation_only", "both"):
            raise ValueError("自适应变异组不接受ablation_fixed_sigma；请使用既有sigma上下界")
        if not cfg.mutation_sigma_min <= cfg.ablation_fixed_sigma <= cfg.mutation_sigma_max:
            raise ValueError("ablation_fixed_sigma须在既有变异上下界内")
    if cfg.ablation_min_gain is not None and cfg.ablation_min_gain < 0:
        raise ValueError("ablation_min_gain不能为负")

    if cfg.validation_protocol not in ("policy", "system"):
        raise ValueError("局内版本只支持validation_protocol=policy/system")
    if cfg.validation_initial_latent_source not in ("common", "provided"):
        raise ValueError("validation_initial_latent_source只支持common/provided")
    if cfg.bootstrap_time_limits or not cfg.vary_instance_seed:
        raise ValueError("当前要求有限任务截止和逐update实例种子")
    if cfg.resume and cfg.init_checkpoint:
        raise ValueError("resume与init_checkpoint不能同时设置")
    if not cfg.output_dir.strip() or not cfg.checkpoints_dir.strip() or not cfg.device.strip():
        raise ValueError("输出目录或device不能为空")
    train = range(cfg.seed, cfg.seed + cfg.updates)
    validation = range(cfg.validation_seed, cfg.validation_seed + cfg.validation_seeds)
    if max(train.start, validation.start) < min(train.stop, validation.stop):
        raise ValueError("训练和验证seed区间必须分离")


__all__ = ["PPOConfig", "parse_args", "validate_config", "resolved_config"]

if __name__ == "__main__":
    print(json.dumps(asdict(parse_args()), ensure_ascii=False, indent=2, allow_nan=False))
