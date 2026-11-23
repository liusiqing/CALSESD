"""单文件 PPO：完整回合采样、在线隐空间搜索、GAE 与集成 Critic 更新。
只依赖 CALSESD.env/DCVRP，不依赖其他 training 模块。
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import copy, deepcopy
from datetime import datetime
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
    # 运行、输出与断点
    output_dir: str = "outputs"          # CSV 训练指标的保存目录
    checkpoints_dir: str = "checkpoints"  # 模型及训练状态 .pt 文件的保存目录
    seed: int = 42                                       # 随机种子，用于初始化、环境实例和搜索随机流
    device: str = "auto"                                 # 计算设备：auto 自动选择，也可指定 cpu、cuda、cuda:0
    updates: int = 200                                   # 总更新轮数；每轮采集一个完整批量回合，再执行 PPO 更新
    resume: str = ""                                     # 续训 checkpoint 路径；恢复网络、优化器、z、搜索状态和随机数状态
    init_checkpoint: str = ""                            # 权重热启动路径；只加载 Actor/Critic，不恢复训练进度
    checkpoint_interval: int = 5                        # 每隔多少个 update 保存 latest；最后一个 update 也会保存

    # 同状态、不同 z 的首动作监督；设 coef=0 可运行纯 PPO 对照。
    specialization_coef: float = 0.05
    specialization_groups: int = 2
    specialization_stride: int = 20
    specialization_cost_tolerance: float = 0.02

    # 环境规模、需求与交通
    grid_size: int = 12                                  # 网格边长；当前网格环境共有 grid_size² 个节点
    num_customers: int = 100                             # 每个实例的客户数；其余非仓库节点不一定是客户
    vehicle_capacity: int = 15                          # 车辆容量；回仓后补满
    max_episode_steps: int = 200                         # 每个环境行的最大动作步数，达到后未完成则截断
    failure_penalty: float = 2.0                         # 失败惩罚系数；截断时按“剩余客户数 + 1”计入惩罚
    demand_noise_std: float = 1.0                        # 实际需求相对期望需求的采样噪声标准差
    traffic_noise_std: float = 1.0                       # 交通演化时加入的随机噪声标准差
    traffic_trend: float = 5.0                           # additive 交通模式使用的加性趋势参数
    traffic_beta: float = 0.07                           # 交通状态更新中的转移/混合系数，具体公式由交通模块定义
    traffic_trend_mode: str = "additive"                 # 交通趋势模式：additive 加性趋势，relative 相对趋势
    traffic_trend_ratio: float = 0.3                     # relative 模式使用的相对趋势比例
    vary_instance_seed: bool = True                     # 每轮按 seed + update - 1 重设环境随机流，生成可复现的变化实例

    # PPO 采样与更新
    batch_size: int = 16                                 # 同时采样的环境行数；当前批次共享实例几何和交通过程
    minibatch_size: int = 32                             # 目标 minibatch 大小；实际按整批时间片组织，时间片数约为该值 // batch_size
    ppo_epochs: int = 4                                  # 每份已采集轨迹重复用于参数更新的 epoch 数
    learning_rate: float = 3e-4                          # Actor 的 Adam 学习率
    value_learning_rate: float = 3e-4                    # Critic 的 Adam 学习率
    target_kl: float | None = 0.03                       # 更新后 KL 超过此值则停止本轮后续 Actor 更新；None 表示关闭
    gamma: float = 1.0                                   # 回报折扣因子；1.0 表示不对未来奖励进行折扣
    gae_lambda: float = 0.95                             # GAE 递推系数，控制多步优势估计的衰减
    clip_coef: float = 0.2                               # PPO 概率比裁剪范围为 [1 - clip_coef, 1 + clip_coef]
    value_coef: float = 0.5                              # Critic 损失反向传播时的乘数；当前使用独立价值优化器
    entropy_coef: float = 0.05                           # Actor 熵奖励系数，鼓励动作分布保持一定探索性
    max_grad_norm: float = 0.5                           # Actor/Critic 参数更新时的梯度范数裁剪上限
    bootstrap_prob: float = 0.8                          # 每个有效轨迹样本被各 Critic 价值头选入训练子集的概率
    reward_scale: float = 1                              # 环境奖励进入 GAE/Critic 目标前的固定缩放系数
    gae_keep_bootstrap: bool = True                      # 切换 z 时使用旧 z 的下一状态价值，并加入 carry 修正

    # Actor / Critic 网络
    hidden_dim: int = 128                                # 主要隐藏维度：静态节点表示、融合表示及价值网络隐藏层
    dynamic_dim: int = 64                                # Actor 动态特征分支的输出维度
    actor_latent_fusion: bool = False
    critic_latent_fusion: bool = False
    latent_node_rank: int = 0  # 0 keeps the original Actor; positive enables node-wise z residual
    latent_dim: int = 16                                 # 条件隐变量 z 的维度，也是进化搜索空间的维度
    num_heads: int = 8                                   # Actor 的 GAT、Transformer 和交叉注意力头数
    static_gat_layers: int = 2                           # Actor 静态几何编码中的残差 GATv2 层数
    static_transformer_layers: int = 2                   # Actor 静态几何编码中的 Transformer 层数
    dropout: float = 0.0                                 # Actor 注意力模块的 dropout；当前采样和 PPO 更新均使用 eval 模式
    n_critics: int = 4                                   # 集成 Critic 的价值头数量；状态编码器由各头共享
    critic_prior_scale: float = 0.0                      # 冻结随机先验输出的叠加系数；0 表示不启用随机先验
    latent_bound: float = 1.0                            # z 的搜索边界，随机采样及搜索更新通常限制在 [-bound, bound]
    z_resample_prob: float = 1.0                         # 每个完整回合开始时，各环境行重新均匀采样 z 的概率

    # 在线搜索触发
    model: str = "evolution"                                  # keep 关闭在线搜索；evolution 开启 Critic 引导隐空间进化
    search_warmup_updates: int = 1                       # 搜索前的训练预热轮数；1 表示第一个 update 不搜索
    online_search_interval: int = 20                     # 距上次搜索尝试至少经过多少个环境动作步，才允许再次搜索
    online_cooldown_steps: int = 5                       # 搜索尝试后的最小冷却步数，与 interval 条件同时满足
    online_max_attempts_per_episode: int = 3              # 每个环境行每回合最多搜索尝试次数；拒绝和超时也计次数
    online_search_budget_seconds: float = 0.0            # 单次共享搜索的软时间预算，单位秒；0 表示不设此上限
    online_batch_budget_seconds: float = 0.0             # 当前完整批量回合累计搜索的软时间预算；0 表示不设此上限
    online_min_remaining_customers: int = 2              # 剩余未服务客户少于该值时，不再触发搜索

    # 隐空间进化搜索
    reopt_candidates: int = 64                           # 每个到期环境行的候选种群大小
    evolution_generations: int = 3                       # 每次在线搜索的进化代数
    mutation_mode: str = "adaptive"                     # isotropic / guided / adaptive
    guided_alpha: float = 0.5                           # 各向同性方差比例的下限
    uncertainty_kappa: float = 1.0                       # 锚点分歧对引导强度的影响系数
    reopt_min_gain: float = 0.1                          # 接受新 z 所需的绝对预测增益阈值
    reopt_min_gain_ratio: float = 1.0                    # 不确定性尺度阈值系数；总阈值 = min_gain + ratio × scale
    lcb_lambda: float = 1.0                              # 保守评分系数：集成均值 - lambda × 集成标准差
    elite_fraction: float = 0.25                        # 每代保留的精英候选比例
    immigrant_fraction: float = 0.125                    # 每代随机移民数量占总种群的目标比例，受剩余位置限制
    mutation_sigma_base: float = 0.05                    # 固定变异尺度（改变方向分配，不改变总均方扰动）
    crossover_alpha_min: float = 0.2                     # 线性交叉系数 alpha 的采样下限
    crossover_alpha_max: float = 0.8                     # 线性交叉系数 alpha 的采样上限

                                                        # exp3: fixed/trust_only/mutation_only/both


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(Config):
        flag, value = "--" + field.name.replace("_", "-"), field.default
        if isinstance(value, bool):
            parser.add_argument(flag, default=value, action=argparse.BooleanOptionalAction)
        else:
            converter = (lambda x: None if x.lower() in ("none", "off") else float(x)) if field.name == "target_kl" else type(value)
            choices = {"model": ("keep", "evolution"),
                       "mutation_mode": ("isotropic", "guided", "adaptive")}.get(field.name)
            parser.add_argument(flag, default=value, type=converter, choices=choices)
    config = Config(**vars(parser.parse_args(argv)))
    if config.latent_node_rank < 0:
        parser.error("--latent-node-rank must be nonnegative")
    if (not np.isfinite(config.specialization_coef) or config.specialization_coef < 0
            or config.specialization_groups < 0 or config.specialization_stride < 1
            or not np.isfinite(config.specialization_cost_tolerance)
            or config.specialization_cost_tolerance < 0):
        parser.error("invalid specialization settings")
    if config.specialization_coef > 0 and config.specialization_groups > 0 and (
            not np.isfinite(config.latent_bound) or config.latent_bound <= 0 or config.latent_dim < 1):
        parser.error("specialization requires latent_dim >= 1 and a finite positive latent_bound")
    return config


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
                   budget_seconds="online_search_budget_seconds")
    args = {field.name: getattr(c, mapping.get(field.name, field.name))
            for field in fields(HybridSearchConfig) if hasattr(c, mapping.get(field.name, field.name))}
    return HybridLatentSearch(HybridSearchConfig(seed=c.seed + 2000, **args))


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


def specialization_pair(c, env, policy, geometry, state, mask, logits, z, row, remaining_steps):
    """同一批环境完整拷贝三次，不对真实环境执行强制动作。"""
    legal = mask[row].bool() & geometry.padding_mask[row]
    actions = logits[row].masked_fill(~legal, -torch.inf).topk(2).indices
    proto = z.new_zeros((2, z.size(-1)))
    proto[:, 0] = z.new_tensor([-0.75, 0.75]) * c.latent_bound
    initial_cost = float(env.total_costs[row])

    def continuation(code, first):
        with preserve_rng():
            try:
                branch = deepcopy(env)
            except Exception as exc:
                raise RuntimeError("Specialization requires a deepcopy-compatible BatchCVRPEnv; "
                                   "implement exact snapshot/restore or set --specialization-coef 0") from exc
            branch_state = branch.get_batch_state()
            branch_z = z.clone()
            branch_z[row] = code
            for step in range(remaining_steps):
                branch_mask = branch.action_mask()
                branch_legal = branch_mask.bool() & geometry.padding_mask
                if bool((~branch.dones & ~branch_legal.any(dim=-1)).any()):
                    raise RuntimeError("No legal action in active specialization branch")
                action = policy(branch_state, geometry, branch_z, branch_mask).argmax(dim=-1)
                if step == 0:
                    action[row] = first
                branch_state, _, _ = branch.step(action, action_mask=branch_mask)
                if bool(branch.dones[row]):
                    break
            cost = float(branch.total_costs[row]) - initial_cost
            return cost if bool(branch.terminated[row]) and np.isfinite(cost) else None

    baseline = continuation(z[row], actions[0])
    a = continuation(proto[0], actions[0])
    b = continuation(proto[1], actions[1])
    if a is None or b is None:
        return None
    # 两个标签都须接近组内最佳；若当前 z 能完成，它也参与质量门槛。
    best = min(a, b) if baseline is None else min(a, b, baseline)
    if max(a, b) > best + c.specialization_cost_tolerance * max(abs(best), 1e-8):
        return None
    # 使用两个完整批次的 z，重放时只监督目标环境行。
    codes = z.detach().clone().unsqueeze(0).repeat(2, 1, 1)
    codes[:, row] = proto
    return dict(state=deepcopy(state), mask=mask.detach().clone(), row=row,
                codes=codes, actions=actions.detach().clone())


def specialization_loss(policy, geometry, pairs):
    terms = []
    for pair in pairs:
        row = pair["row"]
        for code, action in zip(pair["codes"], pair["actions"]):
            logits = policy(pair["state"], geometry, code, pair["mask"])[row]
            terms.append(-Categorical(logits=logits).log_prob(action))
    return torch.stack(terms).mean()


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
    pairs, group_attempts = [], 0
    for step in range(c.max_episode_steps):
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
        missing = valid & ~(mask.bool() & geometry.padding_mask).any(dim=-1)
        if bool(missing.any()):
            rows = missing.nonzero(as_tuple=True)[0]
            raise RuntimeError(
                f"No legal action: update={update}, step={len(trajectory)}, "
                f"rows={rows.tolist()}, current_nodes={env.current_nodes[rows].tolist()}"
            )
        logits = policy(state, geometry, z, mask)
        if (c.specialization_coef > 0 and group_attempts < c.specialization_groups
                and (step + 1) % c.specialization_stride == 0):
            eligible = (valid & ((mask.bool() & geometry.padding_mask).sum(-1) >= 2)).nonzero(as_tuple=True)[0]
            if eligible.numel():
                row = int(eligible[(update - 1 + group_attempts) % eligible.numel()])
                group_attempts += 1
                pair = specialization_pair(c, env, policy, geometry, state, mask, logits,
                                           z, row, c.max_episode_steps - step)
                if pair is not None:
                    pairs.append(pair)
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
    metrics.update(specialization_attempts=group_attempts, specialization_pairs=len(pairs))
    return trajectory, geometry_input, z.detach().clone(), metrics, pairs


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
def update_ppo(c, policy, critic, optimizers, trajectory, geometry_input, pairs=()):
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
    auxiliary_sum, auxiliary_updates = 0.0, 0
    for _ in range(c.ppo_epochs):
        auxiliary_used = False
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
                loss = policy_loss - c.entropy_coef * entropy
                # 每个 PPO epoch 至多加一次，避免同一对标签在每个小批次反复强化。
                if pairs and not auxiliary_used:
                    auxiliary = specialization_loss(policy, geometry, pairs)
                    loss = loss + c.specialization_coef * auxiliary
                    auxiliary_sum += float(auxiliary.detach())
                    auxiliary_updates += 1
                    auxiliary_used = True
                loss.backward()
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
    result = {key: total / max(1, value_weight if key == "value_loss" else actor_weight)
            for key, total in totals.items()}
    result["specialization_loss"] = auxiliary_sum / auxiliary_updates if auxiliary_updates else None
    return result


# 仅保存 update 边界状态；环境下一轮完整 reset，不归档轨迹或中途状态。
def save_checkpoint(path, c, policy, critic, optimizers, search, env, z, update, global_step):
    state = dict(single_file_version=2, config=asdict(c), update=update, global_step=global_step,
                 actor_state_dict=policy.state_dict(), hybrid_critic_state_dict=critic.state_dict(),
                 optimizers=[optimizer.state_dict() for optimizer in optimizers], search=search.state_dict(),
                 z=z.detach().cpu(), rng=rng_state(),
                 env_rng=(env.env.rng.getstate(), env.env.np_rng.get_state(), env.env._torch_generator.get_state()))
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, c, policy, critic, optimizers, search, env, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("single_file_version") != 2:
        raise ValueError("resume requires this single-file trainer's checkpoint; use init-checkpoint for old weights")
    ignored = {"updates", "output_dir", "checkpoints_dir", "resume", "init_checkpoint", "checkpoint_interval", "device"}
    saved_config = dict(state["config"])
    saved_config.setdefault("latent_node_rank", 0)
    saved_config.setdefault("actor_latent_fusion", False)
    saved_config.setdefault("critic_latent_fusion", False)
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


def load_optional_latent_weights(model, weights):
    weights = weights.copy()
    optional = {key: value for key, value in model.state_dict().items()
                if key.startswith("latent_fusion_")}
    # Only supply new weights for a wholly absent branch; do not hide a partial checkpoint.
    if not any(key.startswith("latent_fusion_") for key in weights):
        weights.update(optional)
    model.load_state_dict(weights, strict=True)


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
        static_transformer_layers=c.static_transformer_layers, dropout=c.dropout,
        latent_node_rank=c.latent_node_rank, latent_fusion=c.actor_latent_fusion).to(device)
    critic = ValueNetwork(hidden_dim=c.hidden_dim, latent_dim=c.latent_dim,
                          n_critics=c.n_critics, prior_scale=c.critic_prior_scale,
                          latent_fusion=c.critic_latent_fusion).to(device)
    optimizers = (torch.optim.Adam(policy.parameters(), lr=c.learning_rate),
                  torch.optim.Adam(critic.parameters(), lr=c.value_learning_rate))
    search = make_search(c)
    start, global_step, z = 1, 0, None
    if c.resume:
        start, global_step, z = load_checkpoint(c.resume, c, policy, critic, optimizers, search, env, device)
    elif c.init_checkpoint:
        weights = torch.load(c.init_checkpoint, map_location=device, weights_only=False)
        actor_weights = weights["actor_state_dict"].copy()
        if c.latent_node_rank:
            residual_keys = ("latent_to_node.weight", "node_to_latent.weight")
            if all(key not in actor_weights for key in residual_keys):
                initialized = policy.state_dict()
                for key in residual_keys:
                    actor_weights[key] = initialized[key]
        load_optional_latent_weights(policy, actor_weights)
        load_optional_latent_weights(critic, weights["hybrid_critic_state_dict"])
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    directory = Path(c.output_dir) / stamp
    checkpoint_directory = Path(c.checkpoints_dir) / stamp
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint_directory.mkdir(parents=True, exist_ok=directory.resolve() == checkpoint_directory.resolve())
    print(f"CSV directory: {directory}\nCheckpoint directory: {checkpoint_directory}", flush=True)

    # 每次进程写入新目录；resume 延续 update 编号，不覆盖上一次的 CSV。
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
        "specialization_attempts",
        "specialization_pairs",
        "specialization_loss",
        "update_seconds",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        file.flush()

        for update in range(start, c.updates + 1):
            started = perf_counter()

            # 完整回合采样，包括真实动作之前的在线隐空间搜索。
            trajectory, geometry, z, metrics, pairs = collect(
                c, env, policy, critic, search, z, update
            )

            # PPO Actor 更新与集成 Critic 更新。
            losses = update_ppo(
                c, policy, critic, optimizers, trajectory, geometry, pairs
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
                "specialization_attempts": metrics["specialization_attempts"],
                "specialization_pairs": metrics["specialization_pairs"],
                "specialization_loss": losses["specialization_loss"],
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
