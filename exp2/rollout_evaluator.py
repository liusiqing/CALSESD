"""影子回合评价：固定候选 z 到结束；不是递归执行未来在线搜索。"""
from copy import deepcopy
import numpy as np
import torch
from CALSESD.env import BatchCVRPEnv
from CALSESD.models import build_geometry_data
from .scenario import seed_for

class RolloutEvaluator:
    def __init__(self, env, actor, cfg, chunk=8, repeats=1):
        self.env, self.actor, self.cfg = env, actor, cfg
        self.chunk, self.repeats = chunk, repeats
        self.counts = dict(rollout_episodes=0, rollout_env_steps=0, rollout_batch_steps=0)

    # 核心模块：独立物理环境，不复制/解绑真实 trigger；隐藏需求按原始行独立生成。
    def shadow(self, rows, copies, seed, call, repeat):
        live = self.env
        base = deepcopy(live.env)
        base.seed = seed_for(seed, call, repeat, "traffic")
        base._seed_rngs()
        shadow = BatchCVRPEnv(base, len(rows)*copies, valid_reward=live.valid_reward,
                              step_penalty=live.step_penalty, max_steps=live.max_steps)
        index = rows.repeat_interleave(copies)
        shadow.customer_mask = live.customer_mask.clone()
        shadow.failure_penalty = live.failure_penalty
        shadow.norm_capacity, shadow.norm_dist = live.norm_capacity, live.norm_dist
        for name in live._SNAPSHOT_TENSORS:
            setattr(shadow, name, getattr(live, name)[index].clone())
        # 不读取 unrevealed 的真实值；按与环境一致的 NumPy round/clip 分布采样。
        demands = []
        for row in rows.tolist():
            expected = live.expected_demands_tensor[row].detach().cpu().numpy()
            rng = np.random.RandomState(seed_for(seed, call, repeat, "demand", row))
            values = expected if getattr(base, "demand_mode", None)=="uchoa" else np.clip(
                np.rint(rng.normal(expected, base.demand_noise_std)), 1, base.vehicle_capacity)
            demands.append(values)
        fresh = torch.as_tensor(np.asarray(demands), device=live.device,
                                dtype=live.actual_demands_tensor.dtype).repeat_interleave(copies,0)
        fresh *= shadow.customer_mask[None,:]
        shadow.actual_demands_tensor = torch.where(shadow.revealed_masks,
                                                   shadow.actual_demands_tensor, fresh)
        shadow._update_edge_attr()
        return shadow

    # 核心模块：贪心 Actor 固定 z 前滚；无 Critic bootstrap，按有限任务截止计罚。
    @torch.no_grad()
    def run(self, rows, z, copies, seed, call, repeat):
        e = self.shadow(rows, copies, seed, call, repeat)
        state = e.get_batch_state()
        geometry = self.actor.encode_geometry(build_geometry_data(state,e.env.baseline_distance_matrix,
                                      coords=e.env.coords/max(1.,float(e.env.grid_size))))
        total = torch.zeros(e.batch_size,device=e.device)
        discount = 1.
        for _ in range(self.cfg.max_episode_steps):
            active = ~e.dones.clone()
            if not bool(active.any()): break
            mask = e.action_mask()
            actions = self.actor(state,geometry,z,mask).argmax(-1)
            state,reward,_ = e.step(actions,action_mask=mask)
            total += discount*reward*self.cfg.reward_scale
            discount *= self.cfg.gamma
            self.counts['rollout_env_steps'] += int(active.sum())
            self.counts['rollout_batch_steps'] += 1
        if not bool(e.dones.all()):
            raise RuntimeError('Shadow rollout did not reach finite-task termination')
        self.counts['rollout_episodes'] += e.batch_size
        return total

    @torch.no_grad()
    def score(self, rows, population, seed, call):
        results=[]
        for chunk in population.split(self.chunk):
            samples=[self.run(rows,chunk.repeat(len(rows),1),len(chunk),seed,call,r).view(len(rows),len(chunk))
                     for r in range(self.repeats)]
            results.append(torch.stack(samples).mean(0))
        return torch.cat(results,1)

    @torch.no_grad()
    def baseline(self, rows, z, seed, call):
        return torch.stack([self.run(rows,z,1,seed,call,r) for r in range(self.repeats)]).mean(0)
