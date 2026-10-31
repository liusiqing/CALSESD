"""Held-out greedy validation for in-episode Critic-guided latent control.

policy: seeded common z and no search. system: common/provided initial z, then
this variant's periodic search during execution. No legacy/pre-episode search,
representatives, rollout evaluator or training-controller state is reused.
Search lifecycle is shared with ActorOnlyCollector._decision_state.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace, asdict
from numbers import Integral
import hashlib
import math

import torch

from CALSESD_internal.models import build_geometry_data
from CALSESD_internal.models.hybrid_latent_search import HybridLatentSearch
from .ablation_config import resolve_ablation, build_search_config
from .env_factory import make_env
from .online_trigger import OnlineTrigger, OnlineTriggerConfig
from .ppo_rollout import ActorOnlyCollector
from .inference_timing import InferenceTiming
from .rng_state import preserve_rng


SEARCH_SCOPE = "episode"
VALIDATION_SEMANTICS = "episode_online_validation_v1"


@contextmanager
def evaluation(*models):
    saved = []
    seen = set()
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if id(module) not in seen:
                seen.add(id(module))
                saved.append((module, module.training))
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield
    finally:
        for module, flag in saved:
            module.training = flag


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")
    return int(value)


def _domain_seed(domain, seed):
    payload = f"{domain}:{int(seed)}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


def _device(value):
    result = torch.device(value)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def _resolve_protocol(cfg, protocol, initial_latent_source):
    selected = cfg.validation_protocol if protocol is None else protocol
    if selected not in ("policy", "system"):
        raise ValueError("online validation supports policy/system only, not legacy/auto")
    source = (
        (cfg.validation_initial_latent_source if selected == "system" else "common")
        if initial_latent_source is None
        else initial_latent_source
    )
    if source not in ("common", "provided") or (selected == "policy" and source != "common"):
        raise ValueError("policy requires common z; system allows common/provided")
    return selected, source


def _check_seed_split(cfg):
    start = _integer("validation_seed", cfg.validation_seed)
    count = _integer("validation_seeds", cfg.validation_seeds, 1)
    training_start = _integer("seed", cfg.seed)
    updates = _integer("updates", cfg.updates, 1)
    if max(start, training_start) < min(start + count, training_start + updates):
        raise ValueError("validation execution instances overlap training instances")
    return start, count


def _check_z(cfg, z):
    if (
        not torch.is_tensor(z)
        or z.shape != (cfg.batch_size, cfg.latent_dim)
        or not z.is_floating_point()
        or not bool(torch.isfinite(z).all())
        or bool((z.abs() > cfg.latent_bound).any())
    ):
        raise ValueError("validation latent must be finite floating [B,D] within bounds")


def _initial_z(cfg, z, source, seed, device):
    if source == "provided":
        if z is None:
            raise ValueError("provided initialization requires z")
        return z.detach().clone().to(device), None
    latent_seed = _domain_seed("online-validation-latent", seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(latent_seed)
    latent = (2 * torch.rand(cfg.batch_size, cfg.latent_dim, device=device, generator=generator) - 1) * cfg.latent_bound
    return latent.detach(), latent_seed


def _emit(sink, event):
    if sink is not None:
        sink(event)


def _episode_metrics(cfg, env, distance, seed, update, episode_writer):
    done = env.terminated.detach().cpu().bool().tolist()
    cost = env.total_costs.detach().cpu().tolist()
    dist = distance.detach().cpu().tolist()
    remaining = (~env.visited_masks[:, env.customer_mask]).sum(1)
    penalty = cfg.failure_penalty * (remaining.float() + 1) * env.truncated.float()
    normalizer = torch.as_tensor(env.norm_dist, device=env.total_costs.device)
    if not bool(torch.isfinite(normalizer).all()) or bool((normalizer <= 0).any()):
        raise ValueError("validation norm_dist must be finite and positive")
    penalized = (env.total_costs / env.norm_dist + penalty).detach().cpu().tolist()
    lengths = env.steps.detach().cpu().tolist()
    if not all(math.isfinite(v) for v in cost + dist + penalized):
        raise ValueError("nonfinite validation execution statistics")
    if episode_writer is not None:
        for b in range(cfg.batch_size):
            episode_writer.writerow(
                dict(
                    update=update,
                    seed=seed,
                    batch_index=b,
                    completed=int(done[b]),
                    total_cost=cost[b],
                    geometric_distance=dist[b],
                    episode_length=lengths[b],
                    penalized_cost=penalized[b],
                )
            )
    return done, cost, dist, penalized


def _validate_episode(
    cfg,
    policy,
    critic,
    z,
    device,
    protocol,
    source,
    seed,
    update,
    episode_writer,
    sink,
):
    initial, latent_seed = _initial_z(cfg, z, source, seed, device)
    _check_z(cfg, initial)
    # The factory performs the physical reset; binding must not resample it.
    env = make_env(cfg, seed, device)
    if bool((env.steps != 0).any()) or bool(env.dones.any()):
        raise ValueError("validation factory must return a fresh unfinished batch")
    trigger_config = OnlineTriggerConfig.from_config(cfg)
    enabled = protocol == "system" and cfg.mode == "evolution"
    if not enabled:
        trigger_config = replace(trigger_config, mode="keep")
    trigger = OnlineTrigger(trigger_config)
    trigger.reset_episode(completed_updates=update, training=False)
    env.set_online_trigger(trigger)
    if getattr(env, "online_trigger", None) is not trigger:
        raise RuntimeError("validation requires native BatchCVRPEnv online integration, not the old proxy wrapper")
    state = env.get_batch_state()
    search_seed = _domain_seed("online-validation-search", seed)
    handler = None
    search_config = None
    if enabled:
        if critic is None:
            raise ValueError("system online search requires the online-controller Critic")
        search_config = replace(build_search_config(cfg), seed=search_seed)
        search_config.validate()
        search = HybridLatentSearch(search_config)
        handler = ActorOnlyCollector(cfg, env, policy, critic, search, online_trigger=trigger)
        handler.z = initial.detach().clone()
        handler.critic_updates = update
        timing = handler.inference_timing
    else:
        timing = InferenceTiming(device)
    with timing.measure("actor_geometry"):
        geometry = policy.encode_geometry(
            build_geometry_data(
                state, env.env.baseline_distance_matrix, coords=env.env.coords / max(1.0, float(env.env.grid_size))
            )
        )
    coords = torch.as_tensor(env.env.coords, dtype=torch.float32, device=device)
    matrix = torch.cdist(coords, coords, p=2)
    distance = torch.zeros(cfg.batch_size, device=device)
    current = initial
    counters = dict(
        search_calls=0,
        timeout_calls=0,
        due_rows=0,
        switched_rows=0,
        critic_forward_calls=0,
        gradient_evaluations=0,
        search_seconds=0.0,
    )
    reasons = Counter()
    events = []
    for _ in range(cfg.max_episode_steps):
        active = ~env.dones.clone()
        if not bool(active.any()):
            break
        if handler is not None:
            # Independent validation state; training search/trigger are not used.
            state, due, switched, diagnostics = handler._decision_state(state, update)
            current = handler.z
            reasons.update(handler.last_search_status.get("reasons", ()))
            if diagnostics is not None:
                counters["search_calls"] += 1
                counters["timeout_calls"] += int(bool(diagnostics.get("timed_out", False)))
                counters["due_rows"] += int(due.sum())
                counters["switched_rows"] += int(switched.sum())
                counters["critic_forward_calls"] += int(diagnostics.get("critic_forward_calls", 0))
                counters["gradient_evaluations"] += int(diagnostics.get("gradient_evaluations", 0))
                counters["search_seconds"] += float(diagnostics["collector_search_seconds"])
                # Validation work must not enter training-search CSV totals.
                event = dict(diagnostics)
                event.update(
                    type="validation_latent_search",
                    update=update,
                    seed=seed,
                    validation_protocol=protocol,
                )
                if sink is None:
                    events.append(event)
                else:
                    _emit(sink, event)
            handler.enhancement_events.clear()
        if getattr(state, "online_search_pending", None) is not False:
            raise RuntimeError("validation action state must be post-search, not pending")
        with timing.measure("actor_decision"):
            mask = env.action_mask()
            action = policy(state, geometry, current, mask).argmax(-1)
        if bool((active & ~mask.gather(1, action[:, None]).squeeze(1)).any()):
            raise ValueError("validation selected an illegal action")
        distance += torch.where(active, matrix[env.current_nodes, action], 0.0)
        state, _, _ = env.step(action, action_mask=mask)
    if not bool(env.dones.all()) or trigger.pending:
        raise RuntimeError("validation did not finish full finite episodes with a closed trigger")
    metrics = _episode_metrics(cfg, env, distance, seed, update, episode_writer)
    summary = dict(
        seed=seed,
        latent_seed=latent_seed,
        search_seed=search_seed if enabled else None,
        search_enabled=enabled,
        **counters,
        row_attempts=list(trigger.attempts),
        row_last_switch_steps=list(trigger.last_switch_step),
        batch_search_seconds=trigger.batch_seconds_used,
        decision_row_reasons=dict(reasons),
        inference_timing=timing.flush(),
    )
    return metrics, summary, events, asdict(search_config) if search_config is not None else None


@torch.no_grad()
def validate_controller(
    cfg,
    policy,
    critic,
    z,
    device,
    *,
    episode_writer=None,
    update=0,
    protocol=None,
    diagnostics_sink=None,
    initial_latent_source=None,
):
    """Greedy held-out execution with online search for system/evolution.

    Each execution episode gets its own initial-z clone, trigger and search.
    Training search RNG/EMA and controller state are not advanced. Models are
    not optimized. Do not wrap gradient-guided validation in inference_mode().
    Search enabled is intent; search called reports actual calls, which can be
    zero when episodes finish before the first periodic trigger.
    """
    if cfg.search_scope != "episode" or cfg.bootstrap_time_limits:
        raise ValueError("requires finite-task episode-scope configuration")
    _integer("batch_size", cfg.batch_size, 1)
    _integer("latent_dim", cfg.latent_dim, 1)
    _integer("max_episode_steps", cfg.max_episode_steps, 1)
    update = _integer("update", update)
    if not math.isfinite(cfg.latent_bound) or cfg.latent_bound <= 0:
        raise ValueError("invalid latent_bound")
    if z is not None:
        _check_z(cfg, z)
    if diagnostics_sink is not None and not callable(diagnostics_sink):
        raise TypeError("diagnostics_sink must be callable")
    plan = resolve_ablation(cfg)
    protocol, source = _resolve_protocol(cfg, protocol, initial_latent_source)
    if source == "provided" and z is None:
        raise ValueError("provided validation requires z")
    start, count = _check_seed_split(cfg)
    device = _device(device)
    enabled = protocol == "system" and cfg.mode == "evolution"
    completed = []
    costs = []
    distances = []
    penalized = []
    summaries = []
    events = []
    resolved = []
    with preserve_rng(), evaluation(policy, critic):
        try:
            for j in range(count):
                clear = getattr(critic, "clear_static_cache", None)
                if callable(clear):
                    clear()
                metrics, summary, episode_events, search_config = _validate_episode(
                    cfg,
                    policy,
                    critic,
                    z,
                    device,
                    protocol,
                    source,
                    start + j,
                    update,
                    episode_writer,
                    diagnostics_sink,
                )
                done, cost, dist, p_cost = metrics
                completed.extend(done)
                costs.extend(cost)
                distances.extend(dist)
                penalized.extend(p_cost)
                summaries.append(summary)
                events.extend(episode_events)
                if search_config is not None:
                    resolved.append(dict(seed=start + j, config=search_config))
        finally:
            clear = getattr(critic, "clear_static_cache", None)
            if callable(clear):
                clear()
        calls = sum(s["search_calls"] for s in summaries)
        timeouts = sum(s["timeout_calls"] for s in summaries)
        seconds = sum(s["search_seconds"] for s in summaries)
        details = dict(
            protocol=protocol,
            search_scope=SEARCH_SCOPE,
            validation_semantics=VALIDATION_SEMANTICS,
            latent_source=("common_seeded_prior" if source == "common" else "supplied_persistent_z_clone"),
            search_enabled=enabled,
            search_called=calls > 0,
            search_timed_out=timeouts > 0,
            search_calls=calls,
            timeout_calls=timeouts,
            search_seconds=seconds,
            due_rows=sum(s["due_rows"] for s in summaries),
            switched_rows=sum(s["switched_rows"] for s in summaries),
            critic_forward_calls=sum(s["critic_forward_calls"] for s in summaries),
            gradient_evaluations=sum(s["gradient_evaluations"] for s in summaries),
            ablation_family=plan.family,
            ablation_variant=plan.variant,
            seed_sets=dict(execution_seed_start=start, execution_seed_count=count),
            initial_latent_rule=(
                "per-execution-seed independent Generator"
                if source == "common"
                else "provided clone reset for each episode"
            ),
            search_state="fresh per execution episode; no training search/EMA/provider reused",
            search_schedule="same periodic trigger and budgets; no training-update warmup in validation",
            episode_summaries=summaries,
            search_events=events,
            resolved_search_configs=resolved,
        )
        good_cost = [v for v, d in zip(costs, completed) if d]
        good_distance = [v for v, d in zip(distances, completed) if d]

        def average(values):
            return sum(values) / len(values) if values else None

        result = dict(
            completion_rate=sum(completed) / len(completed),
            mean_penalized_cost=average(penalized),
            mean_completed_cost=average(good_cost),
            mean_completed_geometric_distance=average(good_distance),
            best_completed_geometric_distance=min(good_distance) if good_distance else None,
            validation_protocol=protocol,
            validation_search_scope=SEARCH_SCOPE,
            validation_search_enabled=enabled,
            validation_search_called=calls > 0,
            validation_search_timed_out=timeouts > 0,
            validation_search_seconds=seconds,
            validation_search_calls=calls,
            validation_search_timeouts=timeouts,
            validation_switched_rows=details["switched_rows"],
            # Preserve the existing logging field; both names use one counter.
            validation_search_timeout_calls=timeouts,
            validation_details=details,
        )
        if diagnostics_sink is not None:
            event = dict(details)
            event.update(type="validation_protocol", update=update)
            _emit(diagnostics_sink, event)
        return result


__all__ = [
    "evaluation",
    "validate_controller",
    "SEARCH_SCOPE",
    "VALIDATION_SEMANTICS",
]
