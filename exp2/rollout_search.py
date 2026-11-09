"""纯评分来源对比：关闭梯度、自适应、LCB/分歧阈值，双方相同。"""
from dataclasses import replace
from time import perf_counter
import torch
from CALSESD.models.hybrid_latent_search import HybridLatentSearch
from CALSESD.training.ppo_rollout import ActorOnlyCollector
from .rollout_evaluator import RolloutEvaluator

class RolloutLatentSearch(HybridLatentSearch):
    def bind(self, env, actor, cfg, chunk, repeats):
        self.evaluator=RolloutEvaluator(env,actor,cfg,chunk,repeats)

    def _score(self, critic, features, population):
        self.last_diagnostics['population_evaluations'] += 1
        return self.evaluator.score(self.rows,population,self.config.seed,self.calls)

    def _individual_heads(self, critic, features, z):
        values=self.evaluator.baseline(self.rows,z,self.config.seed,self.calls)
        return values[None,:].expand(critic.n_critics,-1)

    # 核心模块：纯 rollout 组不调用 Critic 精修或用 Critic 分歧变异。
    def _refine_children(self, critic, features, children, joint_due, scale, call, generation, deadline):
        return children, dict(refinement_mode='none',mean_gradient_norm=0.,refinement_step_norm_mean=0.)

    def choose(self, critic, state, z, due=None, **kw):
        started=perf_counter()
        mask=torch.ones(z.size(0),dtype=torch.bool,device=z.device) if due is None else due
        self.rows=torch.nonzero(mask,as_tuple=False).flatten()
        before=dict(self.evaluator.counts)
        result=super().choose(critic,state,z,due=due,**kw)
        d=self.last_diagnostics
        d.update(evaluation_backend='rollout',acceptance='rollout',
                 elapsed_seconds=perf_counter()-started,rollout_counts_complete=True)
        d.update({k:v-before[k] for k,v in self.evaluator.counts.items()})
        d['continuation_policy']='greedy_fixed_z_no_future_search'
        return result

def factories(chunk,repeats):
    class BoundCollector(ActorOnlyCollector):
        def __init__(self,cfg,env,policy,critic,search,*args,**kw):
            super().__init__(cfg,env,policy,critic,search,*args,**kw)
            search.bind(env,policy,cfg,chunk,repeats)
    return RolloutLatentSearch,BoundCollector
