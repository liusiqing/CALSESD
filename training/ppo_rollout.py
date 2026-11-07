"""完整回合的 Actor-only 采样,回合内按触发器执行周期隐空间搜索。
搜索基于当前待决策状态与 token.due;搜索结束后才刷新真实 PPO 状态。
上一步的 bootstrap 在下一次真实(搜索后)决策时用该决策的价值回填。只支持有限完整回合。"""
from __future__ import annotations
from copy import copy, deepcopy
from dataclasses import dataclass, field, asdict
import math
from time import perf_counter
import torch
from CALSESD.models import build_geometry_data
from CALSESD.models.hybrid_latent_search import search_stream_seed
from CALSESD.training.online_trigger import OnlineTrigger, OnlineTriggerConfig
from CALSESD.training.ablation_config import resolve_ablation
from CALSESD.training.inference_timing import InferenceTiming
from CALSESD.training.rng_state import preserve_rng

VALUE_SEMANTICS = "online_actor_controller_v1"
BOOTSTRAP_SEMANTICS = "online_post_trigger_next_decision_v1"


@dataclass
class Rollout:
    states: list = field(default_factory=list)
    geometry_inputs: list = field(default_factory=list)
    latents: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    old_logits: list = field(default_factory=list)
    valid_masks: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    boundaries: list = field(default_factory=list)
    values: list = field(default_factory=list)
    keep_values: list = field(default_factory=list)
    next_values: list = field(default_factory=list)
    latent_switch_masks: list = field(default_factory=list)
    search_due_masks: list = field(default_factory=list)
    completed_costs: list = field(default_factory=list)
    completed_lengths: list = field(default_factory=list)
    truncated_count: int = 0
    due_rows: int = 0
    switched_rows: int = 0
    latent_seconds: float = 0.0
    evolution_search_calls: int = 0
    predicted_gain: object = None
    ensemble_disagreement: object = None
    mutation_sigma: object = None
    latent_gradient_norm: object = None
    geometric_distances: object = None
    completed_mask: object = None
    ppo_env_steps: int = 0
    inference_metrics: dict = field(default_factory=dict)
    ablation_metrics: dict = field(default_factory=dict)
    search_diagnostics: dict = field(default_factory=dict)
    value_semantics: str = VALUE_SEMANTICS
    bootstrap_semantics: str = BOOTSTRAP_SEMANTICS
    _diagnostic_sums: dict = field(default_factory=dict, repr=False)
    _diagnostic_weights: dict = field(default_factory=dict, repr=False)


def _weighted(r, name, value, weight=1):
    if value is None or weight <= 0:
        return
    value = float(value)
    r._diagnostic_sums[name] = r._diagnostic_sums.get(name, 0.0) + value * weight
    r._diagnostic_weights[name] = r._diagnostic_weights.get(name, 0) + weight
    setattr(r, name, r._diagnostic_sums[name] / r._diagnostic_weights[name])


# 核心模块:把一次搜索调用的诊断累计进 Rollout(按 due 行数加权)。
def record_latent_diagnostics(r, d):
    n = int(d.get("due_rows", 0))
    r.due_rows += n
    r.switched_rows += int(d.get("switched_rows", 0))
    # 超时的调用没有可靠的增益估计,不能把默认 0 当作实测值
    if not bool(d.get("timed_out", False)):
        _weighted(r, "predicted_gain", d.get("predicted_gain_mean"), n)
    _weighted(r, "ensemble_disagreement", d.get("mean_incumbent_disagreement"), n)
    for generation in d.get("generation_stats") or []:
        _weighted(r, "mutation_sigma", (generation.get("mutation_sigma") or {}).get("mean"))
        _weighted(r, "latent_gradient_norm", generation.get("mean_gradient_norm"))
    r.search_diagnostics = deepcopy(d)


def _synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class ActorOnlyCollector:
    def __init__(self, cfg, env, policy, critic, search, episode_csv=None, step_csv=None, *, online_trigger=None):
        trigger = online_trigger if online_trigger is not None else getattr(env, "online_trigger", None)
        if trigger is None:
            trigger = OnlineTrigger(OnlineTriggerConfig.from_config(cfg))
        env.set_online_trigger(trigger)
        self.cfg, self.env, self.policy, self.critic = cfg, env, policy, critic
        self.search = search
        self.online_trigger = trigger
        self.ablation_plan = resolve_ablation(cfg)
        self.episode_csv, self.step_csv = episode_csv, step_csv
        self.global_step = 0
        self.critic_updates = 0
        self.z = None
        self.last_incumbent_z = None
        self.z_resampled_rows = 0
        self.last_search_status = {}
        self.last_search_diagnostics = {}
        self.enhancement_events = []
        self.inference_timing = InferenceTiming(env.device)

    # 核心模块:每个 update 开始时按概率 z_resample_prob 对各行 z 重新均匀采样(种子由 seed 与 update 决定,与是否 resume 无关)。
    def _resample_latents(self, update):
        c = self.cfg
        p = float(c.z_resample_prob)
        self.z_resampled_rows = 0
        if p <= 0.0:
            return
        device = torch.device(self.env.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        g = torch.Generator(device=device)
        g.manual_seed(search_stream_seed(c.seed, update, "z_resample"))
        rows = torch.rand(c.batch_size, device=device, generator=g) < p
        fresh = (2 * torch.rand(c.batch_size, c.latent_dim, device=device, generator=g) - 1) * c.latent_bound
        self.z = torch.where(rows[:, None], fresh.to(self.z.dtype), self.z).detach().clone()
        self.z_resampled_rows = int(rows.sum())

    # 核心模块:真实动作之前的在线搜索。
    # decide -> begin_attempt -> 搜索(Critic 评价候选,不推进真实环境) -> finish_attempt -> 刷新搜索后的真实状态。
    # 返回 (搜索后状态, due 掩码, 实际切换掩码, 诊断或 None)。无论搜索是否异常,trigger 都会被 finish_attempt 清理。
    @torch.no_grad()
    def _decision_state(self, state, update):
        e = self.env
        self.last_incumbent_z = None
        decision = e.online_decision()
        due = torch.as_tensor(decision.due, dtype=torch.bool, device=e.device)
        switches = torch.zeros_like(due)
        self.last_search_status = dict(schedule="periodic", steps=list(decision.steps),
                                       reasons=list(decision.reasons), due_rows=int(due.sum()), called=False)
        if not decision.should_search:
            return state, due, switches, None
        incumbent = self.z.detach().clone()
        _synchronize(e.device)
        started = perf_counter()
        self.last_incumbent_z = incumbent
        token = self.online_trigger.begin_attempt(decision)
        success = False
        timed_out = False
        candidate = incumbent
        report = None
        no_switches = (False,) * self.cfg.batch_size
        cleanup_switches = no_switches
        try:
            with self.inference_timing.measure("latent_search"), preserve_rng():
                preview = e.get_batch_state()
                choose_started = perf_counter()
                candidate, switches = self.search.choose(
                    self.critic, preview, incumbent, due=due, model_version=self.critic_updates,
                    deadline=token.deadline)
                _synchronize(e.device)
                choose_seconds = perf_counter() - choose_started
                timed_out = bool(self.search.last_diagnostics.get("timed_out", False))
                cleanup_switches = tuple(switches.detach().cpu().tolist())
                success = True
        finally:
            try:
                _synchronize(e.device)
            finally:
                if self.online_trigger.pending:
                    report = self.online_trigger.finish_attempt(
                        token, switched=cleanup_switches if success else no_switches, timed_out=timed_out)
                if not success:
                    self.last_search_status.update(called=True, failed=True, attempt_report=report)
        self.z = candidate.detach().clone()
        post = e.get_batch_state()
        _synchronize(e.device)
        seconds = perf_counter() - started
        diagnostics = deepcopy(self.search.last_diagnostics)
        diagnostics["gain_in_summary"] = not timed_out and diagnostics.get("predicted_gain_mean") is not None
        if timed_out:
            diagnostics["predicted_gain_mean"] = None
            diagnostics["selection_gain_mean"] = None
        diagnostics.update(
            update=update, episode_step=max(decision.steps), trigger_reason="periodic",
            due_rows=int(due.sum()), switched_rows=int(switches.sum()), trigger_due_mask=list(token.due),
            ablation_family=self.ablation_plan.family, ablation_variant=self.ablation_plan.variant,
            training_schedule="periodic", collector_search_seconds=seconds, choose_wall_seconds=choose_seconds,
            attempt_report=report)
        self.search.last_diagnostics = deepcopy(diagnostics)
        self.last_search_diagnostics = deepcopy(diagnostics)
        event = dict(diagnostics)
        event["type"] = "joint_latent_search"
        self.enhancement_events.append(event)
        self.last_search_status.update(called=True, timed_out=timed_out, switched_rows=int(switches.sum()),
                                       attempt_report=report)
        return post, due, switches, diagnostics

    # 核心模块:采集一个完整批量回合。每步:搜索(若触发) -> 记录价值 -> Actor 采样动作 -> 环境推进。
    def collect(self, update):
        c, e = self.cfg, self.env
        initial_step = self.global_step
        self.policy.eval()
        self.critic.eval()
        self.last_search_diagnostics = {}
        self.last_search_status = {}
        if c.vary_instance_seed:
            e.env.seed = c.seed + update - 1
            e.env._seed_rngs()
        state = e.reset(online_completed_updates=update - 1, online_training=True)
        if self.z is None:
            self.z = self.policy.sample_prior(c.batch_size, c.latent_dim, e.device, c.latent_bound).detach().clone()
        self._resample_latents(update)
        geometry_input = build_geometry_data(
            state, e.env.baseline_distance_matrix, coords=e.env.coords / max(1.0, float(e.env.grid_size))
        ).clone().detach()
        with torch.no_grad():
            with self.inference_timing.measure("actor_geometry"):
                geometry = self.policy.encode_geometry(geometry_input)
            coords = torch.as_tensor(e.env.coords, dtype=torch.float32, device=e.device)
            geometric_matrix = torch.cdist(coords, coords, p=2)
        r = Rollout()
        distance = torch.zeros(c.batch_size, device=e.device)
        for t in range(c.max_episode_steps):
            active = ~e.dones.clone()
            if not bool(active.any()):
                break
            state, due, switched, diagnostics = self._decision_state(state, update)
            if diagnostics is not None:
                r.latent_seconds += diagnostics["collector_search_seconds"]
                r.evolution_search_calls += 1
                record_latent_diagnostics(r, diagnostics)
            z = self.z.detach().clone()
            with torch.no_grad():
                with self.inference_timing.measure("critic_forward"):
                    features = self.critic.encode_state(state)
                    value = self.critic.evaluate_features(features, z).mean(0)
                    keep = value
                    if c.gae_keep_bootstrap and bool(switched.any()):
                        keep = torch.where(
                            switched, self.critic.evaluate_features(features, self.last_incumbent_z).mean(0), value)
                value = value.masked_fill(~active, 0.0)
                keep = keep.masked_fill(~active, 0.0)
                if r.next_values:
                    # 上一步 bootstrap = 本次(搜索之后、实际 z_t 下)的价值
                    continuing = r.valid_masks[-1] & ~r.boundaries[-1]
                    r.next_values[-1][continuing] = value[continuing]
                with self.inference_timing.measure("actor_decision"):
                    mask = e.action_mask()
                    logits = self.policy(state, geometry, z, mask)
                    distribution = torch.distributions.Categorical(logits=logits)
                    action = distribution.sample()
                    logp = distribution.log_prob(action)
            snapshot = copy(state)
            distance += torch.where(active, geometric_matrix[e.current_nodes, action], 0.0)
            next_state, reward, _ = e.step(action, action_mask=mask)
            done = e.dones.clone()
            terminated = e.terminated.clone()
            reward = reward.masked_fill(~active, 0.0)
            r.states.append(snapshot)
            r.geometry_inputs.append(geometry_input)
            r.latents.append(z)
            r.masks.append(mask.detach().clone())
            r.actions.append(action.detach().clone())
            r.log_probs.append(logp.detach().clone())
            r.old_logits.append(logits.detach().clone())
            r.valid_masks.append(active)
            r.rewards.append((reward * c.reward_scale).detach().clone())
            r.boundaries.append(done)
            r.values.append(value.detach().clone())
            r.keep_values.append(keep.detach().clone())
            r.next_values.append(torch.zeros_like(value))
            r.latent_switch_masks.append(switched.detach().clone())
            r.search_due_masks.append(due.detach().clone())
            self.global_step += int(active.sum())
            for b in torch.where(active & done)[0].tolist():
                completed = bool(terminated[b])
                if completed:
                    r.completed_costs.append(float(e.total_costs[b]))
                    r.completed_lengths.append(int(e.steps[b]))
                else:
                    r.truncated_count += 1
                if self.episode_csv is not None:
                    self.episode_csv.writerow(dict(
                        update=update, global_step=self.global_step, batch_index=b,
                        total_cost=float(e.total_costs[b]), geometric_distance=float(distance[b]),
                        episode_length=int(e.steps[b]),
                        served_customers=int(e.visited_masks[b, e.customer_mask].sum()),
                        done_reason="completed" if completed else "time_limit"))
            if self.step_csv is not None:
                for b in torch.where(active)[0].tolist():
                    self.step_csv.writerow(dict(
                        update=update, global_step=self.global_step, batch_index=b, action=int(action[b]),
                        reward=float(reward[b]), done=int(done[b]), z_switched=int(switched[b])))
            state = next_state
        r.geometric_distances = distance.detach().clone()
        r.completed_mask = e.terminated.detach().clone()
        r.ppo_env_steps = self.global_step - initial_step
        r.inference_metrics = self.inference_timing.flush()
        r.ablation_metrics = dict(
            family=self.ablation_plan.family, variant=self.ablation_plan.variant, search_scope="episode",
            schedule="periodic", ppo_env_steps=r.ppo_env_steps, search_calls=r.evolution_search_calls,
            attempted_rows=r.due_rows, switched_rows=r.switched_rows,
            search_status=deepcopy(self.last_search_status), row_attempts=list(self.online_trigger.attempts),
            batch_search_seconds=self.online_trigger.batch_seconds_used, z_resampled_rows=self.z_resampled_rows)
        return r


__all__ = ["Rollout", "ActorOnlyCollector", "record_latent_diagnostics", "VALUE_SEMANTICS", "BOOTSTRAP_SEMANTICS"]
