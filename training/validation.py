"""留出实例上的贪心验证(回合内 Critic 引导隐变量控制)。
policy:带种子的公共 z,不搜索。system:公共/给定的初始 z,执行过程中按本变体的周期触发器搜索。
搜索生命周期与 ActorOnlyCollector._decision_state 共用。"""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace, asdict
import hashlib
import torch
from CALSESD.models import build_geometry_data
from CALSESD.models.hybrid_latent_search import HybridLatentSearch
from .ablation_config import resolve_ablation, build_search_config
from .env_factory import make_env
from .online_trigger import OnlineTrigger, OnlineTriggerConfig
from .ppo_rollout import ActorOnlyCollector
from .inference_timing import InferenceTiming
from .rng_state import preserve_rng

SEARCH_SCOPE = "episode"
VALIDATION_SEMANTICS = "episode_online_validation_v1"


# 核心模块:临时切换为评估模式,退出时恢复每个子模块原来的 training 标志。
@contextmanager
def evaluation(*models):
    saved, seen = [], set()
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
    source = ((cfg.validation_initial_latent_source if selected == "system" else "common")
              if initial_latent_source is None else initial_latent_source)
    return selected, source


def _initial_z(cfg, z, source, seed, device):
    if source == "provided":
        return z.detach().clone().to(device), None
    latent_seed = _domain_seed("online-validation-latent", seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(latent_seed)
    latent = (2 * torch.rand(cfg.batch_size, cfg.latent_dim, device=device, generator=generator) - 1) * cfg.latent_bound
    return latent.detach(), latent_seed


def _emit(sink, event):
    if sink is not None:
        sink(event)


# 核心模块:单个验证实例的指标。失败回合按剩余客户数加惩罚:cost/norm_dist + failure_penalty*(remaining+1)。
def _episode_metrics(cfg, env, distance, seed, update, episode_writer):
    done = env.terminated.detach().cpu().bool().tolist()
    cost = env.total_costs.detach().cpu().tolist()
    dist = distance.detach().cpu().tolist()
    remaining = (~env.visited_masks[:, env.customer_mask]).sum(1)
    penalty = cfg.failure_penalty * (remaining.float() + 1) * env.truncated.float()
    penalized = (env.total_costs / env.norm_dist + penalty).detach().cpu().tolist()
    lengths = env.steps.detach().cpu().tolist()
    if episode_writer is not None:
        for b in range(cfg.batch_size):
            episode_writer.writerow(dict(update=update, seed=seed, batch_index=b, completed=int(done[b]),
                                         total_cost=cost[b], geometric_distance=dist[b],
                                         episode_length=lengths[b], penalized_cost=penalized[b]))
    return done, cost, dist, penalized


# 核心模块:一个验证种子的完整贪心执行。每个实例使用独立的初始 z、触发器与搜索器。
def _validate_episode(cfg, policy, critic, z, device, protocol, source, seed, update, episode_writer, sink):
    initial, latent_seed = _initial_z(cfg, z, source, seed, device)
    env = make_env(cfg, seed, device)
    trigger_config = OnlineTriggerConfig.from_config(cfg)
    enabled = protocol == "system" and cfg.mode == "evolution"
    if not enabled:
        trigger_config = replace(trigger_config, mode="keep")
    trigger = OnlineTrigger(trigger_config)
    trigger.reset_episode(completed_updates=update, training=False)
    env.set_online_trigger(trigger)
    state = env.get_batch_state()
    search_seed = _domain_seed("online-validation-search", seed)
    handler = None
    search_config = None
    if enabled:
        search_config = replace(build_search_config(cfg), seed=search_seed)
        handler = ActorOnlyCollector(cfg, env, policy, critic, HybridLatentSearch(search_config), online_trigger=trigger)
        handler.z = initial.detach().clone()
        handler.critic_updates = update
        timing = handler.inference_timing
    else:
        timing = InferenceTiming(device)
    with timing.measure("actor_geometry"):
        geometry = policy.encode_geometry(build_geometry_data(
            state, env.env.baseline_distance_matrix, coords=env.env.coords / max(1.0, float(env.env.grid_size))))
    coords = torch.as_tensor(env.env.coords, dtype=torch.float32, device=device)
    matrix = torch.cdist(coords, coords, p=2)
    distance = torch.zeros(cfg.batch_size, device=device)
    current = initial
    counters = dict(search_calls=0, timeout_calls=0, due_rows=0, switched_rows=0,
                    critic_forward_calls=0, gradient_evaluations=0, search_seconds=0.0)
    reasons = Counter()
    events = []
    for _ in range(cfg.max_episode_steps):
        active = ~env.dones.clone()
        if not bool(active.any()):
            break
        if handler is not None:
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
                event = dict(diagnostics)
                event.update(type="validation_latent_search", update=update, seed=seed, validation_protocol=protocol)
                if sink is None:
                    events.append(event)
                else:
                    _emit(sink, event)
            handler.enhancement_events.clear()
        with timing.measure("actor_decision"):
            mask = env.action_mask()
            action = policy(state, geometry, current, mask).argmax(-1)
        distance += torch.where(active, matrix[env.current_nodes, action], 0.0)
        state, _, _ = env.step(action, action_mask=mask)
    metrics = _episode_metrics(cfg, env, distance, seed, update, episode_writer)
    summary = dict(seed=seed, latent_seed=latent_seed, search_seed=search_seed if enabled else None,
                   search_enabled=enabled, **counters, row_attempts=list(trigger.attempts),
                   row_last_switch_steps=list(trigger.last_switch_step),
                   batch_search_seconds=trigger.batch_seconds_used, decision_row_reasons=dict(reasons),
                   inference_timing=timing.flush())
    return metrics, summary, events, asdict(search_config) if search_config is not None else None


# 核心模块:验证入口。逐个验证种子执行并汇总;不推进训练的搜索 RNG/EMA/触发器,不更新模型参数。
@torch.no_grad()
def validate_controller(cfg, policy, critic, z, device, *, episode_writer=None, update=0, protocol=None,
                        diagnostics_sink=None, initial_latent_source=None):
    plan = resolve_ablation(cfg)
    protocol, source = _resolve_protocol(cfg, protocol, initial_latent_source)
    start, count = cfg.validation_seed, cfg.validation_seeds
    device = _device(device)
    enabled = protocol == "system" and cfg.mode == "evolution"
    completed, costs, distances, penalized, summaries, events, resolved = [], [], [], [], [], [], []
    with preserve_rng(), evaluation(policy, critic):
        try:
            for j in range(count):
                critic.clear_static_cache()
                metrics, summary, episode_events, search_config = _validate_episode(
                    cfg, policy, critic, z, device, protocol, source, start + j, update, episode_writer,
                    diagnostics_sink)
                done, cost, dist, p_cost = metrics
                completed.extend(done); costs.extend(cost); distances.extend(dist); penalized.extend(p_cost)
                summaries.append(summary)
                events.extend(episode_events)
                if search_config is not None:
                    resolved.append(dict(seed=start + j, config=search_config))
        finally:
            critic.clear_static_cache()
        calls = sum(s["search_calls"] for s in summaries)
        timeouts = sum(s["timeout_calls"] for s in summaries)
        seconds = sum(s["search_seconds"] for s in summaries)
        details = dict(
            protocol=protocol, search_scope=SEARCH_SCOPE, validation_semantics=VALIDATION_SEMANTICS,
            latent_source="common_seeded_prior" if source == "common" else "supplied_persistent_z_clone",
            search_enabled=enabled, search_called=calls > 0, search_timed_out=timeouts > 0,
            search_calls=calls, timeout_calls=timeouts, search_seconds=seconds,
            due_rows=sum(s["due_rows"] for s in summaries), switched_rows=sum(s["switched_rows"] for s in summaries),
            critic_forward_calls=sum(s["critic_forward_calls"] for s in summaries),
            gradient_evaluations=sum(s["gradient_evaluations"] for s in summaries),
            ablation_family=plan.family, ablation_variant=plan.variant,
            seed_sets=dict(execution_seed_start=start, execution_seed_count=count),
            episode_summaries=summaries, search_events=events, resolved_search_configs=resolved)
        good_cost = [v for v, d in zip(costs, completed) if d]
        good_distance = [v for v, d in zip(distances, completed) if d]
        average = lambda values: sum(values) / len(values) if values else None
        result = dict(
            completion_rate=sum(completed) / len(completed), mean_penalized_cost=average(penalized),
            mean_completed_cost=average(good_cost), mean_completed_geometric_distance=average(good_distance),
            best_completed_geometric_distance=min(good_distance) if good_distance else None,
            validation_protocol=protocol, validation_search_scope=SEARCH_SCOPE, validation_search_enabled=enabled,
            validation_search_called=calls > 0, validation_search_timed_out=timeouts > 0,
            validation_search_seconds=seconds, validation_search_calls=calls, validation_search_timeouts=timeouts,
            validation_switched_rows=details["switched_rows"], validation_search_timeout_calls=timeouts,
            validation_details=details)
        if diagnostics_sink is not None:
            event = dict(details)
            event.update(type="validation_protocol", update=update)
            _emit(diagnostics_sink, event)
        return result


__all__ = ["evaluation", "validate_controller", "SEARCH_SCOPE", "VALIDATION_SEMANTICS"]
