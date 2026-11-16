"""单文件 PPO：完整回合采样、在线隐空间搜索、GAE 与集成 Critic 更新。
只依赖 CALSESD.env/DCVRP，不依赖其他 training 模块。
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import copy
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import os
import random
import sys
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import csv
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from CALSESD.env import BatchCVRPEnv, CVRPGridEnvForLatent
from CALSESD.DCVRP import GeometryDynamicPointerPolicy, ValueNetwork, build_geometry_data
from CALSESD.DCVRP.hybrid_latent_search import HybridLatentSearch, HybridSearchConfig, search_stream_seed


@dataclass
class Config:
    output_dir: str = "outputs/ppo_single"
    checkpoints_dir: str = "checkpoints/ppo_single"
    seed: int = 42
    device: str = "auto"
    updates: int = 200
    resume: str = ""
    init_checkpoint: str = ""
    checkpoint_interval: int = 5
    grid_size: int = 12
    num_customers: int = 100
    vehicle_capacity: int = 15
    max_episode_steps: int = 200
    failure_penalty: float = 2.0
    demand_noise_std: float = 1.0
    traffic_noise_std: float = 1.0
    traffic_trend: float = 5.0
    traffic_beta: float = 0.07
    traffic_trend_mode: str = "relative" #additive
    traffic_trend_ratio: float = 0.3
    vary_instance_seed: bool = True
    batch_size: int = 32
    minibatch_size: int = 64
    ppo_epochs: int = 4
    learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    target_kl: float | None = 0.03
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    bootstrap_prob: float = 0.8
    reward_scale: float = 1
    gae_keep_bootstrap: bool = True
    hidden_dim: int = 128
    dynamic_dim: int = 64
    latent_dim: int = 64
    num_heads: int = 8
    static_gat_layers: int = 2
    static_transformer_layers: int = 2
    dropout: float = 0.0
    n_critics: int = 4
    critic_prior_scale: float = 0.5
    latent_bound: float = 1.0
    z_resample_prob: float = 0.5
    model: str ="keep"        #"evolution"
    search_warmup_updates: int = 1
    online_search_interval: int = 30
    online_cooldown_steps: int = 5
    online_max_attempts_per_episode: int = 3
    online_search_budget_seconds: float = 0.0
    online_batch_budget_seconds: float = 0.0
    online_min_remaining_customers: int = 2
    reopt_candidates: int = 64
    evolution_generations: int = 5
    gradient_steps: int = 1
    gradient_lr: float = 0.01
    latent_gradient_normalize: bool = True
    max_gradient_norm: float = 1.0
    uncertainty_kappa: float = 1.0
    reopt_min_gain: float = 0.0
    reopt_min_gain_ratio: float = 1.0
    lcb_lambda: float = 1.0
    elite_fraction: float = 0.25
    immigrant_fraction: float = 0.125
    mutation_sigma_base: float = 0.15
    mutation_sigma_min: float = 0.03
    mutation_sigma_max: float = 0.4
    crossover_alpha_min: float = 0.2
    crossover_alpha_max: float = 0.8
    refinement_mode: str = "value_gradient"
    adaptive_gradient_trust: bool = True
    adaptive_mutation: bool = True
    ablation_family: str = "none"
    ablation_variant: str = "default"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(Config):
        flag, value = "--" + field.name.replace("_", "-"), field.default
        if isinstance(value, bool):
            parser.add_argument(flag, default=value, action=argparse.BooleanOptionalAction)
        else:
            converter = (lambda x: None if x.lower() in ("none", "off") else float(x)) if field.name == "target_kl" else type(value)
            choices = {"model": ("keep", "evolution"), "ablation_family": ("none", "exp1", "exp3"),
                       "refinement_mode": ("none", "value_gradient", "random_matched")}.get(field.name)
            flags = (flag, "--mode") if field.name == "model" else (flag,)
            parser.add_argument(*flags, default=value, type=converter, choices=choices)
    return Config(**vars(parser.parse_args(argv)))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


# 周期触发器：保留尝试次数、冷却、软预算和 Critic 的六维在线上下文。
class OnlineTrigger:
    context_fields = ("period_remaining_fraction", "cooldown_remaining_fraction",
                      "attempts_remaining_fraction", "batch_budget_remaining_fraction",
                      "batch_budget_limited", "online_enabled")

    def __init__(self, config):
        self.config = config
        self.reset_episode()

    @property
    def pending(self):
        return self._pending is not None

    def reset_episode(self, *, completed_updates=0, training=True):
        c = self.config
        self.enabled = c.model == "evolution" and (not training or completed_updates >= c.search_warmup_updates)
        self.last_attempt = [0] * c.batch_size
        self.attempts = [0] * c.batch_size
        self.seconds_used = 0.0
        self._pending = None

    def remaining_seconds(self):
        cap = self.config.online_batch_budget_seconds
        used = self.seconds_used + (perf_counter() - self._pending.started if self.pending else 0)
        return max(0.0, cap - used) if cap else None

    def decide(self, *, steps, active, remaining_customers, legal_action_counts):
        c, available = self.config, self.remaining_seconds()
        due = []
        for b, (step, alive, remaining, legal) in enumerate(zip(
                steps.tolist(), active.tolist(), remaining_customers.tolist(), legal_action_counts.tolist())):
            gap = step - self.last_attempt[b]
            due.append(bool(alive and self.enabled and self.attempts[b] < c.online_max_attempts_per_episode
                            and (available is None or available > 0)
                            and remaining >= c.online_min_remaining_customers and legal >= 2
                            and gap >= c.online_cooldown_steps and gap >= c.online_search_interval))
        return SimpleNamespace(due=tuple(due), steps=tuple(steps.tolist()), should_search=any(due))

    def begin_attempt(self, decision):
        for b, due in enumerate(decision.due):
            if due:
                self.last_attempt[b] = decision.steps[b]
                self.attempts[b] += 1
        limits = [x for x in (self.config.online_search_budget_seconds, self.remaining_seconds())
                  if x is not None and x > 0]
        started = perf_counter()
        self._pending = SimpleNamespace(started=started, due=decision.due,
                                        deadline=started + min(limits) if limits else None)
        return self._pending

    def finish_attempt(self, attempt, *, switched=None, elapsed_seconds=None, timed_out=False):
        self.seconds_used += perf_counter() - attempt.started if elapsed_seconds is None else elapsed_seconds
        self._pending = None

    def context(self, steps):
        c = self.config
        remaining = self.remaining_seconds()
        fraction = 1.0 if remaining is None else remaining / c.online_batch_budget_seconds
        rows = []
        for b, step in enumerate(steps.tolist()):
            gap = step - self.last_attempt[b]
            period = max(0., min(1., (c.online_search_interval - gap) / c.online_search_interval))
            cooldown = max(0., min(1., (c.online_cooldown_steps - gap) / c.online_cooldown_steps)) if c.online_cooldown_steps else 0.
            attempts = max(0., (c.online_max_attempts_per_episode - self.attempts[b]) / c.online_max_attempts_per_episode)
            rows.append([period, cooldown, attempts, fraction,
                         float(c.online_batch_budget_seconds > 0), float(self.enabled)])
        return rows


# 环境与搜索配置都在本文件构造；模型及进化算子仍留在 models。
def make_env(c, device):
    base = CVRPGridEnvForLatent(
        grid_size=c.grid_size, num_customers=c.num_customers, vehicle_capacity=c.vehicle_capacity,
        seed=c.seed, device=device, max_steps=c.max_episode_steps,
        demand_noise_std=c.demand_noise_std, traffic_noise_std=c.traffic_noise_std,
        traffic_trend=c.traffic_trend, traffic_beta=c.traffic_beta,
        traffic_trend_mode=c.traffic_trend_mode, traffic_trend_ratio=c.traffic_trend_ratio)
    env = BatchCVRPEnv(base, c.batch_size, valid_reward=0., step_penalty=0., max_steps=c.max_episode_steps)
    env.failure_penalty = c.failure_penalty
    env.reset()
    env.set_online_trigger(OnlineTrigger(c))
    return env


def make_search(c):
    mapping = dict(candidates="reopt_candidates", generations="evolution_generations",
                   min_gain="reopt_min_gain", min_gain_ratio="reopt_min_gain_ratio",
                   normalize_gradient="latent_gradient_normalize", budget_seconds="online_search_budget_seconds")
    args = {field.name: getattr(c, mapping.get(field.name, field.name))
            for field in fields(HybridSearchConfig) if hasattr(c, mapping.get(field.name, field.name))}
    args.update(seed=c.seed + 2000, record_refinement_diagnostics=False)
    if c.ablation_family == "exp1":
        if c.ablation_variant not in ("none", "value_gradient", "random_matched") or c.model != "evolution":
            raise ValueError("exp1 requires evolution and none/value_gradient/random_matched")
        if c.ablation_variant != "none" and c.gradient_lr <= 0:
            raise ValueError("exp1 refinement requires gradient_lr > 0")
        args.update(refinement_mode=c.ablation_variant, gradient_steps=1,
                    adaptive_gradient_trust=False, adaptive_mutation=False)
    elif c.ablation_family == "exp3":
        flags = {"fixed": (False, False), "trust_only": (True, False),
                 "mutation_only": (False, True), "both": (True, True)}
        if c.model != "evolution" or c.gradient_steps < 1 or c.gradient_lr <= 0 or c.uncertainty_kappa <= 0:
            raise ValueError("exp3 requires evolution and effective gradient/uncertainty controls")
        trust, mutation = flags[c.ablation_variant]
        args.update(refinement_mode="value_gradient", adaptive_gradient_trust=trust, adaptive_mutation=mutation)
    return HybridLatentSearch(HybridSearchConfig(**args))


# 真实动作前完成搜索；无论成功与否都清理 pending，不推进环境做候选 rollout。
@torch.no_grad()
def search_before_action(c, env, critic, search, state, z, update):
    decision = env.online_decision()
    switched = torch.zeros(c.batch_size, dtype=torch.bool, device=env.device)
    if not decision.should_search:
        return state, z, switched
    synchronize(env.device)
    token = env.online_trigger.begin_attempt(decision)
    timed_out = False
    try:
        with preserve_rng():
            preview = env.get_batch_state()
            candidate, switched = search.choose(critic, preview, z, due=decision.due,
                                                model_version=update - 1, deadline=token.deadline)
            synchronize(env.device)
            timed_out = bool(search.last_diagnostics.get("timed_out", False))
    finally:
        synchronize(env.device)
        state, _ = env.finish_online_search(token, switched=switched, timed_out=timed_out)
    return state, candidate.detach().clone(), switched


def accumulate_geometric_distance(distance, coords, current_nodes, actions, valid):
    """累计真实动作的原始坐标欧氏距离；结束行不再计入，不含交通或归一化。"""
    step_distance = torch.linalg.vector_norm(coords[current_nodes] - coords[actions], dim=-1)
    return distance + torch.where(valid, step_distance, 0.)


@torch.no_grad()
def collect(c, env, policy, critic, search, z, update):
    policy.eval()
    critic.eval()
    if c.vary_instance_seed:
        env.env.seed = c.seed + update - 1
        env.env._seed_rngs()
    state = env.reset(online_completed_updates=update - 1, online_training=True)
    if z is None:
        z = policy.sample_prior(c.batch_size, c.latent_dim, env.device, c.latent_bound)
    if c.z_resample_prob > 0:
        generator = torch.Generator(device=env.device).manual_seed(search_stream_seed(c.seed, update, "z_resample"))
        rows = torch.rand(c.batch_size, device=env.device, generator=generator) < c.z_resample_prob
        fresh = (2 * torch.rand(c.batch_size, c.latent_dim, device=env.device, generator=generator) - 1) * c.latent_bound
        z = torch.where(rows[:, None], fresh.to(z), z)
    geometry_input = build_geometry_data(
        state, env.env.baseline_distance_matrix, coords=env.env.coords / max(1., float(env.env.grid_size)))
    geometry = policy.encode_geometry(geometry_input)
    # 独立记录几何路线长度，不改变环境成本、奖励或 PPO 目标。
    coords = torch.as_tensor(env.env.coords, dtype=torch.float32, device=env.device)
    geometric_distances = torch.zeros(c.batch_size, dtype=torch.float32, device=env.device)
    trajectory, switches = [], 0
    for _ in range(c.max_episode_steps):
        valid = ~env.dones.clone()
        if not bool(valid.any()):
            break
        old_z = z.detach().clone()
        state, z, switched = search_before_action(c, env, critic, search, state, z, update)
        z = z.detach().clone()
        features = critic.encode_state(state)
        value = critic.evaluate_features(features, z).mean(0).masked_fill(~valid, 0.)
        keep_value = value
        if c.gae_keep_bootstrap and bool(switched.any()):
            keep_value = torch.where(switched, critic.evaluate_features(features, old_z).mean(0), value).masked_fill(~valid, 0.)
        # 上一步 bootstrap 对应下一次搜索结束后的真实决策价值。
        if trajectory:
            previous = trajectory[-1]
            previous["next_value"] = torch.where(previous["valid"] & ~previous["done"], value, 0.)
        mask = env.action_mask()
        logits = policy(state, geometry, z, mask)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        record = dict(state=copy(state), z=z, mask=mask.detach().clone(), action=action,
                      logp=distribution.log_prob(action), logits=logits, valid=valid,
                      value=value, keep_value=keep_value, next_value=torch.zeros_like(value))
        # 必须在 step 更新 current_nodes 之前累计，含实际执行的回仓路段。
        geometric_distances = accumulate_geometric_distance(
            geometric_distances, coords, env.current_nodes, action, valid)
        state, reward, done = env.step(action, action_mask=mask)
        record.update(reward=reward.masked_fill(~valid, 0.) * c.reward_scale, done=done.clone())
        trajectory.append(record)
        switches += int(switched.sum())
    if not bool(env.dones.all()):
        raise RuntimeError("finite full-episode collection ended before the environment boundary")
    completed = env.terminated
    metrics = dict(completion=float(completed.float().mean()), switches=switches,
                   cost=float(env.total_costs[completed].mean()) if bool(completed.any()) else None,
                   mean_completed_geometric_distance=float(geometric_distances[completed].mean()) if bool(completed.any()) else None,
                   best_completed_geometric_distance=float(geometric_distances[completed].min()) if bool(completed.any()) else None,
                   mean_geometric_distance=float(geometric_distances.mean()),
                   steps=sum(int(row["valid"].sum()) for row in trajectory))
    return trajectory, geometry_input, z.detach().clone(), metrics


# GAE 只在环境结束处截断，隐变量切换处继续递推。
def compute_gae(rewards, boundaries, bootstrap_values, values, gamma, gae_lambda,
                valid_masks=None, carry=None):
    valid = torch.ones_like(boundaries) if valid_masks is None else valid_masks.bool()
    carry = torch.zeros_like(rewards) if carry is None else carry
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * bootstrap_values[t] - values[t]
        running = delta + gamma * gae_lambda * (~boundaries[t]).to(rewards.dtype) * (running + carry[t])
        running = torch.where(valid[t], running, torch.zeros_like(running))
        advantages[t] = running
    returns = torch.where(valid, advantages + values, torch.zeros_like(values))
    return advantages, returns


def make_bootstrap_mask(n_critics, valid, probability):
    T, B = valid.shape
    mask = (torch.rand(n_critics, T, B, device=valid.device) < probability) & valid.unsqueeze(0)
    positions = valid.nonzero()
    for head in range(n_critics):
        if not bool(mask[head].any()):
            t, b = positions[head % len(positions)]
            mask[head, t, b] = True
    return mask


# PPO：固定采样时的 state/z/mask；只更新 Actor，搜索过程不参与反向传播。
def update_ppo(c, policy, critic, optimizers, trajectory, geometry_input):
    policy.eval()
    critic.eval()
    actor_optimizer, value_optimizer = optimizers
    stack = lambda name: torch.stack([row[name] for row in trajectory])
    valid, done, values = stack("valid"), stack("done"), stack("value")
    bootstrap, carry = stack("next_value"), None
    if c.gae_keep_bootstrap:
        keep = stack("keep_value")
        continuing = valid[:-1] & ~done[:-1]
        bootstrap, carry = torch.zeros_like(values), torch.zeros_like(values)
        bootstrap[:-1] = torch.where(continuing, keep[1:], 0.)
        carry[:-1] = torch.where(continuing, values[1:] - keep[1:], 0.)
    advantages, returns = compute_gae(stack("reward"), done, bootstrap, values,
                                      c.gamma, c.gae_lambda, valid, carry)
    selected = advantages[valid]
    advantages = torch.where(valid, (advantages - selected.mean()) / selected.std(unbiased=False).clamp_min(1e-8), 0.)
    head_mask = make_bootstrap_mask(critic.n_critics, valid, c.bootstrap_prob)
    T, B = valid.shape
    time_batch = max(1, c.minibatch_size // B)
    totals = dict(policy_loss=0., entropy=0., kl=0., value_loss=0.)
    actor_weight = value_weight = 0
    stop_actor = False
    for _ in range(c.ppo_epochs):
        for indices in torch.randperm(T, device=valid.device).split(time_batch):
            chosen = indices.tolist()
            count = int(valid[chosen].sum())
            if not count:
                continue
            if not stop_actor:
                actor_optimizer.zero_grad(set_to_none=True)
                # 本 minibatch 内复用带梯度几何；参数更新后不跨 minibatch 复用。
                geometry = policy.encode_geometry(geometry_input)
                objective = torch.zeros((), device=valid.device)
                entropy = torch.zeros_like(objective)
                for t in chosen:
                    row, keep = trajectory[t], valid[t]
                    if not bool(keep.any()):
                        continue
                    dist = Categorical(logits=policy(row["state"], geometry, row["z"], row["mask"]))
                    ratio = (dist.log_prob(row["action"])[keep] - row["logp"][keep]).exp()
                    adv = advantages[t, keep]
                    objective -= torch.minimum(ratio * adv, ratio.clamp(1 - c.clip_coef, 1 + c.clip_coef) * adv).sum()
                    entropy += dist.entropy()[keep].sum()
                policy_loss, entropy = objective / count, entropy / count
                (policy_loss - c.entropy_coef * entropy).backward()
                clip_grad_norm_(policy.parameters(), c.max_grad_norm)
                actor_optimizer.step()
                del geometry
                # 完整 categorical KL 在更新后计算；停止 Actor 后仍继续训练 Critic。
                with torch.no_grad():
                    geometry = policy.encode_geometry(geometry_input)
                    kl = 0.
                    for t in chosen:
                        row, keep = trajectory[t], valid[t]
                        if bool(keep.any()):
                            old = Categorical(logits=row["logits"][keep])
                            new = Categorical(logits=policy(row["state"], geometry, row["z"], row["mask"])[keep])
                            kl += float(kl_divergence(old, new).sum())
                    kl /= count
                stop_actor = c.target_kl is not None and kl > c.target_kl
                for key, value in (("policy_loss", float(policy_loss.detach())), ("entropy", float(entropy.detach())), ("kl", kl)):
                    totals[key] += value * count
                actor_weight += count
            # 每个价值头只在自己的固定 bootstrap 子样本上回归 GAE return。
            value_optimizer.zero_grad(set_to_none=True)
            errors = torch.zeros(critic.n_critics, device=valid.device)
            counts = torch.zeros_like(errors)
            for t in chosen:
                row, keep = trajectory[t], valid[t]
                if not bool(keep.any()):
                    continue
                pred = critic.evaluate_features(critic.encode_state(row["state"]), row["z"])[:, keep]
                mask = head_mask[:, t, keep]
                error = F.smooth_l1_loss(pred, returns[t, keep][None, :].expand_as(pred), reduction="none")
                errors = errors + (error * mask).sum(1)
                counts = counts + mask.sum(1)
            present = counts > 0
            if bool(present.any()):
                loss = (errors[present] / counts[present]).mean()
                (c.value_coef * loss).backward()
                clip_grad_norm_(critic.parameters(), c.max_grad_norm)
                value_optimizer.step()
                totals["value_loss"] += float(loss.detach()) * count
                value_weight += count
    return {key: total / max(1, value_weight if key == "value_loss" else actor_weight)
            for key, total in totals.items()}


# 仅保存 update 边界状态；环境下一轮完整 reset，不归档轨迹或中途状态。
def save_checkpoint(path, c, policy, critic, optimizers, search, env, z, update, global_step):
    state = dict(single_file_version=1, config=asdict(c), update=update, global_step=global_step,
                 actor_state_dict=policy.state_dict(), hybrid_critic_state_dict=critic.state_dict(),
                 optimizers=[optimizer.state_dict() for optimizer in optimizers], search=search.state_dict(),
                 z=z.detach().cpu(), rng=rng_state(),
                 env_rng=(env.env.rng.getstate(), env.env.np_rng.get_state(), env.env._torch_generator.get_state()))
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, c, policy, critic, optimizers, search, env, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("single_file_version") != 1:
        raise ValueError("resume requires this single-file trainer's checkpoint; use init-checkpoint for old weights")
    ignored = {"updates", "output_dir", "checkpoints_dir", "resume", "init_checkpoint", "checkpoint_interval", "device"}
    saved_config = dict(state["config"])
    if "mode" in saved_config and "model" not in saved_config:
        saved_config["model"] = saved_config.pop("mode")
    if {k: v for k, v in asdict(c).items() if k not in ignored} != {k: v for k, v in saved_config.items() if k not in ignored}:
        raise ValueError("resume requires unchanged training hyperparameters")
    policy.load_state_dict(state["actor_state_dict"])
    critic.load_state_dict(state["hybrid_critic_state_dict"])
    for optimizer, saved in zip(optimizers, state["optimizers"]):
        optimizer.load_state_dict(saved)
    search.load_state_dict(state["search"])
    python_rng, numpy_rng, torch_rng = state["env_rng"]
    env.env.rng.setstate(python_rng)
    env.env.np_rng.set_state(numpy_rng)
    env.env._torch_generator.set_state(torch_rng.cpu())
    restore_rng(state["rng"])
    return state["update"] + 1, state["global_step"], state["z"].to(device)


def main(argv=None):
    c = parse_args(argv)
    random.seed(c.seed)
    np.random.seed(c.seed % (2 ** 32))
    torch.manual_seed(c.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if c.device == "auto" else c.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    env = make_env(c, device)
    policy = GeometryDynamicPointerPolicy(
        hidden_dim=c.hidden_dim, dynamic_dim=c.dynamic_dim, latent_dim=c.latent_dim,
        num_heads=c.num_heads, static_gat_layers=c.static_gat_layers,
        static_transformer_layers=c.static_transformer_layers, dropout=c.dropout).to(device)
    critic = ValueNetwork(hidden_dim=c.hidden_dim, latent_dim=c.latent_dim,
                          n_critics=c.n_critics, prior_scale=c.critic_prior_scale).to(device)
    optimizers = (torch.optim.Adam(policy.parameters(), lr=c.learning_rate),
                  torch.optim.Adam(critic.parameters(), lr=c.value_learning_rate))
    search = make_search(c)
    start, global_step, z = 1, 0, None
    if c.resume:
        start, global_step, z = load_checkpoint(c.resume, c, policy, critic, optimizers, search, env, device)
    elif c.init_checkpoint:
        weights = torch.load(c.init_checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(weights["actor_state_dict"])
        critic.load_state_dict(weights["hybrid_critic_state_dict"])
    directory = Path(c.output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    checkpoint_directory = Path(c.checkpoints_dir)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    # 每个 update 保存一行训练指标；续训时仅保留 checkpoint 之前的记录。
    csv_path = directory / "training_metrics.csv"
    columns = [
        "update",
        "global_step",
        "completion_rate",
        "mean_completed_cost",
        "mean_completed_geometric_distance",
        "best_completed_geometric_distance",
        "mean_geometric_distance",
        "latent_switches",
        "policy_loss",
        "value_loss",
        "entropy",
        "kl",
        "update_seconds",
    ]

    history = []
    if c.resume and csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            history = [
                {key: row.get(key, "") for key in columns}
                for row in csv.DictReader(file)
                if int(row["update"]) < start
            ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
        file.flush()

        for update in range(start, c.updates + 1):
            started = perf_counter()

            # 完整回合采样，包括真实动作之前的在线隐空间搜索。
            trajectory, geometry, z, metrics = collect(
                c, env, policy, critic, search, z, update
            )

            # PPO Actor 更新与集成 Critic 更新。
            losses = update_ppo(
                c, policy, critic, optimizers, trajectory, geometry
            )
            global_step += metrics["steps"]

            if (
                c.checkpoint_interval > 0
                and update % c.checkpoint_interval == 0
            ) or update == c.updates:
                save_checkpoint(
                    checkpoint_directory / "checkpoint_latest.pt",
                    c,
                    policy,
                    critic,
                    optimizers,
                    search,
                    env,
                    z,
                    update,
                    global_step,
                )

            elapsed = perf_counter() - started

            # CSV 与控制台使用同一组指标，不额外执行模型前向。
            writer.writerow({
                "update": update,
                "global_step": global_step,
                "completion_rate": metrics["completion"],
                "mean_completed_cost": metrics["cost"],
                "mean_completed_geometric_distance": metrics["mean_completed_geometric_distance"],
                "best_completed_geometric_distance": metrics["best_completed_geometric_distance"],
                "mean_geometric_distance": metrics["mean_geometric_distance"],
                "latent_switches": metrics["switches"],
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "kl": losses["kl"],
                "update_seconds": elapsed,
            })
            file.flush()

            cost = (
                "NA"
                if metrics["cost"] is None
                else f"{metrics['cost']:.3f}"
            )
            geometric = metrics["mean_completed_geometric_distance"]
            geometric_text = "NA" if geometric is None else f"{geometric:.3f}"
            print(
                f"update={update} "
                f"step={global_step} "
                f"complete={metrics['completion']:.1%} "
                f"cost={cost} "
                f"geom={geometric_text} "
                f"switch={metrics['switches']} "
                f"policy={losses['policy_loss']:.4f} "
                f"value={losses['value_loss']:.4f} "
                f"entropy={losses['entropy']:.4f} "
                f"kl={losses['kl']:.5f} "
                f"sec={elapsed:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
