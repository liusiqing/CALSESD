"""训练与验证共用的 rollout 进化搜索及 collector。"""
from dataclasses import replace
from time import perf_counter
import torch
from CALSESD.models.hybrid_latent_search import HybridLatentSearch
from CALSESD.training.ablation_config import build_search_config
from CALSESD.training.ppo_rollout import ActorOnlyCollector
from .evaluator import RolloutEvaluator

# 核心模块：后端对比采用相同进化算子，关闭梯度、自适应、LCB 和分歧相对阈值。
def search_config(cfg):
    base=build_search_config(cfg)
    return replace(base,refinement_mode='none',gradient_steps=0,
        adaptive_gradient_trust=False,adaptive_mutation=False,
        record_refinement_diagnostics=False,lcb_lambda=0.,min_gain_ratio=0.,budget_seconds=0.)

class RolloutSearch(HybridLatentSearch):
    def bind(self,env,actor,cfg):
        self.evaluator=RolloutEvaluator(env,actor,cfg)

    def _score(self,critic,features,population):
        self.last_diagnostics['population_evaluations']+=1
        return self.evaluator.population(self.rows,population,self.config.seed,self.calls)

    def _individual_heads(self,critic,features,z):
        value=self.evaluator.baseline(self.rows,z,self.config.seed,self.calls)
        return value[None,:].expand(critic.n_critics,-1)

    def _refine_children(self,critic,features,children,joint_due,scale,call,generation,deadline):
        return children,dict(refinement_mode='none',mean_gradient_norm=0.,refinement_step_norm_mean=0.)

    @torch.no_grad()
    def choose(self,critic,state,z,due=None,**kw):
        mask=torch.ones(z.size(0),device=z.device,dtype=torch.bool) if due is None else due
        self.rows=mask.nonzero(as_tuple=False).flatten()
        started=perf_counter();before=dict(self.evaluator.counts)
        result=super().choose(critic,state,z,due=due,**kw)
        d=self.last_diagnostics
        d.update(evaluation_backend='rollout',acceptance='rollout',
                 elapsed_seconds=perf_counter()-started,rollout_counts_complete=True)
        d.update({k:v-before[k] for k,v in self.evaluator.counts.items()})
        d['continuation_policy']='greedy_fixed_z_no_future_search'
        return result

class Exp2Collector(ActorOnlyCollector):
    # 核心模块：搜索选择的 z 直接进入原采样器实际动作与 log_prob/value 记录。
    def __init__(self,cfg,env,policy,critic,search,*args,**kw):
        super().__init__(cfg,env,policy,critic,search,*args,**kw)
        if isinstance(search,RolloutSearch): search.bind(env,policy,cfg)
        self.ablation_plan=replace(self.ablation_plan,family='exp2',variant=cfg.exp2_backend)
