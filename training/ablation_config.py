"""Ablation mapping for periodic in-episode Critic-guided latent search.

Only current-state Critic scoring/acceptance is supported. Search overrides
must be resolved before recording or constructing the search configuration.
A manifest describes the protocol; it does not certify an end-to-end run.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math
from numbers import Integral, Real
from .online_trigger import OnlineTriggerConfig, ONLINE_CONTEXT_FIELDS, ONLINE_CONTEXT_DIM


VARIANTS = {
    "none": ("default",),
    "exp1": ("none", "value_gradient", "random_matched"),
    "exp3": ("fixed", "trust_only", "mutation_only", "both"),
}
TRUST_FLAGS = {
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


def resolve_ablation(cfg):
    if cfg.search_scope != "episode" or cfg.mode not in ("keep", "evolution"):
        raise ValueError("requires episode-scope config and mode=keep/evolution")
    family, variant = cfg.ablation_family, cfg.ablation_variant
    if family == "exp2":
        raise ValueError("Online full-rollout continuation is not implemented in phase one")
    if family not in VARIANTS or variant not in VARIANTS[family]:
        raise ValueError(f"invalid ablation {family}/{variant}")
    trigger = OnlineTriggerConfig.from_config(cfg)
    trigger.validate()
    if family != "none" and cfg.mode != "evolution":
        raise ValueError("ablations require evolution; keep baseline uses none/default")
    if not isinstance(cfg.ablation_record_diagnostics, bool):
        raise ValueError("invalid ablation_record_diagnostics")
    for name in ("ablation_fixed_sigma", "ablation_min_gain"):
        value = getattr(cfg, name)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"invalid {name}")

    overrides = dict(
        refinement_mode="value_gradient",
        adaptive_gradient_trust=True,
        adaptive_mutation=True,
        record_refinement_diagnostics=cfg.ablation_record_diagnostics,
    )
    if family == "none":
        if cfg.ablation_fixed_sigma is not None or cfg.ablation_min_gain is not None:
            raise ValueError("ablation overrides require exp1/exp3")
    elif family == "exp1":
        if cfg.gradient_lr <= 0:
            raise ValueError("exp1 requires positive gradient_lr")
        if cfg.gradient_steps != 1:
            raise ValueError("exp1 requires gradient_steps=1; none applies zero refinement steps")
        overrides.update(
            refinement_mode=variant,
            gradient_steps=1,
            adaptive_gradient_trust=False,
            adaptive_mutation=False,
        )
    else:
        if cfg.gradient_steps < 1 or cfg.gradient_lr <= 0 or cfg.uncertainty_kappa <= 0:
            raise ValueError("exp3 requires active gradient refinement and positive kappa")
        trust, mutation = TRUST_FLAGS[variant]
        overrides.update(adaptive_gradient_trust=trust, adaptive_mutation=mutation)

    if cfg.ablation_fixed_sigma is not None:
        if overrides["adaptive_mutation"]:
            raise ValueError("ablation_fixed_sigma requires adaptive_mutation=False")
        sigma = cfg.ablation_fixed_sigma
        if not cfg.mutation_sigma_min <= sigma <= cfg.mutation_sigma_max:
            raise ValueError("fixed sigma outside unchanged bounds")
        overrides["mutation_sigma_base"] = sigma
    if cfg.ablation_min_gain is not None:
        overrides["min_gain"] = cfg.ablation_min_gain
    return AblationPlan(
        family,
        variant,
        "episode",
        "critic",
        "periodic",
        trigger.warmup_updates,
        overrides,
    )


def build_search_config(cfg, config_cls=None):
    plan = resolve_ablation(cfg)
    if config_cls is None:
        from CALSESD.models.hybrid_latent_search import HybridSearchConfig

        config_cls = HybridSearchConfig
    args = dict(
        candidates=cfg.reopt_candidates,
        generations=cfg.evolution_generations,
        latent_bound=cfg.latent_bound,
        min_gain=cfg.reopt_min_gain,
        min_gain_ratio=cfg.reopt_min_gain_ratio,
        lcb_lambda=cfg.lcb_lambda,
        normalize_gradient=cfg.latent_gradient_normalize,
        gradient_steps=cfg.gradient_steps,
        gradient_lr=cfg.gradient_lr,
        max_gradient_norm=cfg.max_gradient_norm,
        uncertainty_kappa=cfg.uncertainty_kappa,
        elite_fraction=cfg.elite_fraction,
        immigrant_fraction=cfg.immigrant_fraction,
        crossover_alpha_min=cfg.crossover_alpha_min,
        crossover_alpha_max=cfg.crossover_alpha_max,
        mutation_sigma_base=cfg.mutation_sigma_base,
        mutation_sigma_min=cfg.mutation_sigma_min,
        mutation_sigma_max=cfg.mutation_sigma_max,
        seed=cfg.seed + 2000,
        acceptance="critic",
        evaluation_backend="critic",
        search_scope="episode",
        budget_seconds=cfg.online_search_budget_seconds,
    )
    args.update(plan.search_overrides)
    try:
        config = config_cls(**args)
    except TypeError as exc:
        raise RuntimeError("Install the episode/current-state Critic HybridSearchConfig") from exc
    config.validate()
    if (config.search_scope, config.evaluation_backend, config.acceptance) != ("episode", "critic", "critic"):
        raise RuntimeError("Resolved search config must use episode-scope Critic scoring/acceptance")
    return config


def validation_settings(cfg):
    resolve_ablation(cfg)
    if cfg.validation_protocol not in ("policy", "system"):
        raise ValueError("online validation requires policy/system")
    source = cfg.validation_initial_latent_source if cfg.validation_protocol == "system" else "common"
    if source not in ("common", "provided"):
        raise ValueError("invalid validation latent source")
    for name in ("seed", "validation_seed", "updates", "validation_seeds"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"invalid {name}")
    if cfg.updates < 1 or cfg.validation_seeds < 1:
        raise ValueError("invalid update/validation count")
    train = range(cfg.seed, cfg.seed + cfg.updates)
    validation = range(cfg.validation_seed, cfg.validation_seed + cfg.validation_seeds)
    if max(train.start, validation.start) < min(train.stop, validation.stop):
        raise ValueError("training/validation seeds overlap")
    return dict(
        protocol=cfg.validation_protocol,
        initial_latent_source=source,
        search_scope="episode",
        search_enabled=cfg.validation_protocol == "system" and cfg.mode == "evolution",
        execution_seed_start=cfg.validation_seed,
        execution_seed_count=cfg.validation_seeds,
        trigger=asdict(OnlineTriggerConfig.from_config(cfg)),
        training_warmup_applies=False,
        state_source="current execution state; no representative reset states",
    )


def ablation_manifest(cfg, search_config):
    plan = resolve_ablation(cfg)
    if (search_config.search_scope, search_config.evaluation_backend, search_config.acceptance) != (
        "episode",
        "critic",
        "critic",
    ):
        raise ValueError("manifest requires episode-scope Critic scoring/acceptance")
    return dict(
        plan=asdict(plan),
        resolved_search_config=asdict(search_config),
        validation_protocol=cfg.validation_protocol,
        validation_settings=validation_settings(cfg),
        online_trigger=asdict(OnlineTriggerConfig.from_config(cfg)),
        online_context_schema=1,
        online_context_dim=ONLINE_CONTEXT_DIM,
        online_context_fields=list(ONLINE_CONTEXT_FIELDS),
        value_target=("Recorded Actor-controller lambda-return targets with actual " "future online latent switches"),
        integration_stage="episode_online_critic_search",
        verification=(
            "Runtime contracts are checked by the trainer; this manifest "
            "alone does not certify end-to-end correctness"
        ),
    )


__all__ = [
    "AblationPlan",
    "VARIANTS",
    "TRUST_FLAGS",
    "resolve_ablation",
    "build_search_config",
    "validation_settings",
    "ablation_manifest",
]
