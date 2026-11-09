"""消融实验到搜索配置的映射(回合内周期 Critic 引导搜索)。"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math
from .online_trigger import OnlineTriggerConfig, ONLINE_CONTEXT_FIELDS, ONLINE_CONTEXT_DIM

VARIANTS = {
    "none": ("default",),
    "exp1": ("none", "value_gradient", "random_matched"),
    "exp3": ("fixed", "trust_only", "mutation_only", "both"),
}
TRUST_FLAGS = {  # (自适应梯度信任, 自适应变异)
    "fixed": (False, False),
    "trust_only": (True, False),
    "mutation_only": (False, True),
    "both": (True, True),
}


@dataclass(frozen=True)
class AblationPlan:
    family: str
    variant: str
    search_scope: str
    evaluation_backend: str
    schedule: str
    warmup_updates: int
    search_overrides: dict = field(default_factory=dict)


# 核心模块：将消融名称映射为搜索配置。
# exp1 固定单步精修协议，比较 none/value_gradient/random_matched。
# exp3 保持评分与接受规则不变，仅比较梯度信任和自适应变异。
def resolve_ablation(cfg):
    family = cfg.ablation_family
    variant = cfg.ablation_variant

    # 防止实验名称写错后静默运行成另一个实验。
    if family not in VARIANTS or variant not in VARIANTS[family]:
        raise ValueError(f"Unsupported ablation: {family}/{variant}")

    if family != "none" and cfg.mode != "evolution":
        raise ValueError("exp1/exp3 require mode=evolution")
        
    overrides = dict(refinement_mode="value_gradient",adaptive_gradient_trust=True,adaptive_mutation=True,
                     record_refinement_diagnostics=cfg.ablation_record_diagnostics,
    )

    if family == "exp1":
        # exp1 比较修正方向：不修正 / 价值梯度 / 投影前等长随机方向。
        # value_gradient 和 random_matched 都需要非零参考步长。
        if variant != "none" and cfg.gradient_lr <= 0:
            raise ValueError(
                "exp1 refinement variants require gradient_lr > 0"
            )

        overrides.update(
            refinement_mode=variant,
            gradient_steps=1,
            adaptive_gradient_trust=False,
            adaptive_mutation=False,
        )
        
    elif family == "exp3":
        # 四个组都保留有效的梯度精修，避免开关退化为无效比较。
        if (
            cfg.gradient_steps < 1
            or cfg.gradient_lr <= 0
            or cfg.uncertainty_kappa <= 0
        ):
            raise ValueError(
                "exp3 requires gradient_steps >= 1, "
                "gradient_lr > 0 and uncertainty_kappa > 0"
            )

        trust, mutation = TRUST_FLAGS[variant]
        overrides.update(adaptive_gradient_trust=trust,adaptive_mutation=mutation,)

    if cfg.ablation_fixed_sigma is not None:
        # 自适应变异不使用 sigma_base；拒绝会被静默忽略的覆盖。
        if overrides["adaptive_mutation"]:
            raise ValueError("ablation_fixed_sigma requires fixed mutation")

        sigma = cfg.ablation_fixed_sigma
        if not cfg.mutation_sigma_min <= sigma <= cfg.mutation_sigma_max:
            raise ValueError("ablation_fixed_sigma must lie within mutation bounds")

        overrides["mutation_sigma_base"] = sigma

    if cfg.ablation_min_gain is not None:
        overrides["min_gain"] = cfg.ablation_min_gain

    trigger = OnlineTriggerConfig.from_config(cfg)

    return AblationPlan(family=family,variant=variant,search_scope="episode",evaluation_backend="critic",
        schedule="periodic",warmup_updates=trigger.warmup_updates,search_overrides=overrides,)


# 核心模块:由训练配置与消融覆盖构造 HybridSearchConfig。
def build_search_config(cfg, config_cls=None):
    plan = resolve_ablation(cfg)
    if config_cls is None:
        from CALSESD.models.hybrid_latent_search import HybridSearchConfig
        config_cls = HybridSearchConfig
    args = dict(
        candidates=cfg.reopt_candidates, generations=cfg.evolution_generations, latent_bound=cfg.latent_bound,
        min_gain=cfg.reopt_min_gain, min_gain_ratio=cfg.reopt_min_gain_ratio, lcb_lambda=cfg.lcb_lambda,
        normalize_gradient=cfg.latent_gradient_normalize, gradient_steps=cfg.gradient_steps,
        gradient_lr=cfg.gradient_lr, max_gradient_norm=cfg.max_gradient_norm,
        uncertainty_kappa=cfg.uncertainty_kappa, elite_fraction=cfg.elite_fraction,
        immigrant_fraction=cfg.immigrant_fraction, crossover_alpha_min=cfg.crossover_alpha_min,
        crossover_alpha_max=cfg.crossover_alpha_max, mutation_sigma_base=cfg.mutation_sigma_base,
        mutation_sigma_min=cfg.mutation_sigma_min, mutation_sigma_max=cfg.mutation_sigma_max,
        seed=cfg.seed + 2000, acceptance="critic", evaluation_backend="critic", search_scope="episode",
        budget_seconds=cfg.online_search_budget_seconds,
    )
    args.update(plan.search_overrides)
    return config_cls(**args)


def validation_settings(cfg):
    system = cfg.validation_protocol == "system"
    return dict(
        protocol=cfg.validation_protocol,
        initial_latent_source=cfg.validation_initial_latent_source if system else "common",
        search_scope="episode",
        search_enabled=system and cfg.mode == "evolution",
        execution_seed_start=cfg.validation_seed,
        execution_seed_count=cfg.validation_seeds,
        trigger=asdict(OnlineTriggerConfig.from_config(cfg)),
        training_warmup_applies=False,
        state_source="current execution state; no representative reset states",
    )


def ablation_manifest(cfg, search_config):
    return dict(
        plan=asdict(resolve_ablation(cfg)),
        resolved_search_config=asdict(search_config),
        validation_protocol=cfg.validation_protocol,
        validation_settings=validation_settings(cfg),
        online_trigger=asdict(OnlineTriggerConfig.from_config(cfg)),
        online_context_dim=ONLINE_CONTEXT_DIM,
        online_context_fields=list(ONLINE_CONTEXT_FIELDS),
        integration_stage="episode_online_critic_search",
    )


__all__ = ["AblationPlan", "VARIANTS", "TRUST_FLAGS", "resolve_ablation", "build_search_config",
           "validation_settings", "ablation_manifest"]
