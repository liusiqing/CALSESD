"""仅推进影子环境，固定候选 z 贪心前滚至有限任务结束。"""
from copy import deepcopy
import numpy as np
import torch
from CALSESD.env import BatchCVRPEnv
from CALSESD.models import build_geometry_data
from .scenario import seed_for

class RolloutEvaluator:
    def __init__(self,env,actor,cfg):
        self.env,self.actor,self.cfg=env,actor,cfg
        self.counts=dict(rollout_episodes=0,rollout_env_steps=0,rollout_batch_steps=0)

    # 核心模块：独立交通引擎，隐藏需求按原始行重新采样；不复制真实 trigger。
    def shadow(self,rows,copies,seed,call,repeat):
        live=self.env
        base=deepcopy(live.env)
        base.seed=seed_for(seed,call,repeat,'traffic')
        base._seed_rngs()
        e=BatchCVRPEnv(base,len(rows)*copies,valid_reward=live.valid_reward,
                       step_penalty=live.step_penalty,max_steps=live.max_steps)
        ix=rows.repeat_interleave(copies)
        for name in live._SNAPSHOT_TENSORS:
            setattr(e,name,getattr(live,name)[ix].clone())
        e.customer_mask=live.customer_mask.clone()
        e.failure_penalty=live.failure_penalty
        e.norm_capacity,e.norm_dist=live.norm_capacity,live.norm_dist
        values=[]
        for row in rows.tolist():
            expected=live.expected_demands_tensor[row].detach().cpu().numpy()
            rng=np.random.RandomState(seed_for(seed,call,repeat,'demand',row))
            sampled=expected if getattr(base,'demand_mode',None)=='uchoa' else np.clip(
                np.rint(rng.normal(expected,base.demand_noise_std)),1,base.vehicle_capacity)
            values.append(sampled)
        fresh=torch.as_tensor(np.asarray(values),device=e.device,
                               dtype=e.actual_demands_tensor.dtype).repeat_interleave(copies,0)
        fresh=fresh*e.customer_mask[None,:]
        e.actual_demands_tensor=torch.where(e.revealed_masks,e.actual_demands_tensor,fresh)
        e._update_edge_attr()
        return e

    # 核心模块：模拟动作不进入 PPO buffer；Actor 参数不更新。
    @torch.no_grad()
    def run(self,rows,z,copies,seed,call,repeat):
        e=self.shadow(rows,copies,seed,call,repeat)
        state=e.get_batch_state()
        geometry=self.actor.encode_geometry(build_geometry_data(state,e.env.baseline_distance_matrix,
                             coords=e.env.coords/max(1.,float(e.env.grid_size))))
        total=torch.zeros(e.batch_size,device=e.device)
        discount=1.
        for _ in range(self.cfg.max_episode_steps):
            active=~e.dones.clone()
            if not bool(active.any()): break
            mask=e.action_mask()
            actions=self.actor(state,geometry,z,mask).argmax(-1)
            state,reward,_=e.step(actions,action_mask=mask)
            total+=discount*reward*self.cfg.reward_scale
            discount*=self.cfg.gamma
            self.counts['rollout_env_steps']+=int(active.sum())
            self.counts['rollout_batch_steps']+=1
        if not bool(e.dones.all()):
            raise RuntimeError('Shadow rollout must reach the environment finite-task cutoff')
        self.counts['rollout_episodes']+=e.batch_size
        return total

    def population(self,rows,population,seed,call):
        result=[]
        for chunk in population.split(self.cfg.exp2_chunk):
            estimates=[self.run(rows,chunk.repeat(len(rows),1),len(chunk),seed,call,r).view(len(rows),len(chunk))
                       for r in range(self.cfg.exp2_repeats)]
            result.append(torch.stack(estimates).mean(0))
        return torch.cat(result,1)

    def baseline(self,rows,z,seed,call):
        return torch.stack([self.run(rows,z,1,seed,call,r)
                           for r in range(self.cfg.exp2_repeats)]).mean(0)
