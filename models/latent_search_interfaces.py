"""Shared latent-search contracts, including pure full-rollout evaluation."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import math
from numbers import Integral, Real
from typing import Any, Optional, Protocol, TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor

@dataclass
class HybridSearchConfig:
    candidates: int = 64
    generations: int = 3
    latent_bound: float = 1.
    elite_fraction: float = .25
    immigrant_fraction: float = .125
    crossover_alpha_min: float = .2
    crossover_alpha_max: float = .8
    gradient_steps: int = 1
    gradient_lr: float = .05
    max_gradient_norm: float = 1.
    mutation_sigma_base: float = .15
    mutation_sigma_min: float = .03
    mutation_sigma_max: float = .4
    adaptive_gradient_trust: bool = True
    adaptive_mutation: bool = True
    uncertainty_kappa: float = 1.
    scale_decay: float = .95
    scale_floor: float = 1e-3
    scale_max_version_lag: int = 50
    candidate_chunk_size: int = 32
    min_gain: float = .1
    acceptance: str = 'critic'
    confirmation_scenarios: int = 4
    confirmation_min_gain: float = 0.
    confirmation_se_multiplier: float = 1.
    budget_seconds: float = 0.
    seed: int = 42
    refinement_mode: str = 'value_gradient'
    evaluation_backend: str = 'critic'  # critic / callback / rollout
    record_refinement_diagnostics: bool = True

    def validate(self):
        for key in ('candidates','generations','candidate_chunk_size','scale_max_version_lag'):
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,Integral) or v<1:raise ValueError(f'invalid {key}')
        for key in ('gradient_steps','seed'):
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,Integral) or v<0:raise ValueError(f'invalid {key}')
        v=self.confirmation_scenarios
        if isinstance(v,bool) or not isinstance(v,Integral) or v<2:raise ValueError('confirmation_scenarios must be >=2')
        for key in ('adaptive_gradient_trust','adaptive_mutation','record_refinement_diagnostics'):
            if not isinstance(getattr(self,key),bool):raise ValueError(f'invalid {key}')
        keys=('latent_bound','elite_fraction','immigrant_fraction','crossover_alpha_min','crossover_alpha_max',
              'gradient_lr','max_gradient_norm','mutation_sigma_base','mutation_sigma_min','mutation_sigma_max',
              'uncertainty_kappa','scale_decay','scale_floor','min_gain','confirmation_min_gain',
              'confirmation_se_multiplier','budget_seconds')
        for key in keys:
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v):raise ValueError(f'invalid {key}')
        if min(self.latent_bound,self.max_gradient_norm,self.scale_floor)<=0:raise ValueError('invalid scales')
        if not 0<self.elite_fraction<=1 or not 0<=self.immigrant_fraction<1:raise ValueError('invalid fractions')
        if not 0<=self.crossover_alpha_min<=self.crossover_alpha_max<=1:raise ValueError('invalid crossover')
        if not 0<self.mutation_sigma_min<=self.mutation_sigma_base<=self.mutation_sigma_max:raise ValueError('invalid mutation scales')
        if not 0<=self.scale_decay<1:raise ValueError('invalid EMA')
        if min(self.gradient_lr,self.uncertainty_kappa,self.min_gain,self.confirmation_min_gain,
               self.confirmation_se_multiplier,self.budget_seconds)<0:raise ValueError('negative scale')
        if self.acceptance not in ('critic','paired_rollout'):raise ValueError('invalid acceptance')
        if self.refinement_mode not in ('none','value_gradient','random_matched'):raise ValueError('invalid refinement mode')
        if self.evaluation_backend not in ('critic','callback','rollout'):raise ValueError('invalid evaluation backend')
        if self.evaluation_backend=='rollout':
            if self.refinement_mode!='none' or self.adaptive_gradient_trust or self.adaptive_mutation or self.record_refinement_diagnostics:
                raise ValueError('Pure rollout requires no refinement, no adaptive controls and no Critic diagnostics')
            if self.acceptance!='critic':raise ValueError('Pure rollout uses evaluator-score acceptance, not winner confirmation')

@dataclass(frozen=True)
class SearchEvaluationContext:
    states: tuple[Any,...]
    features: Optional[Tensor]  # None for pure rollout
    due: Tensor
    joint_due: Tensor
    batch_size: int
    task_count: int
    call_index: int
    phase: str
    generation: int
    deadline: Optional[float]

@dataclass
class PopulationEvaluation:
    """Higher-is-better [J,B,K]; evaluate_incumbents may return [J,B,1]."""
    scores: Tensor
    diagnostics: dict[str,Any] = field(default_factory=dict)

class PopulationEvaluator(Protocol):
    def __call__(self,*,critic:Any,population:Tensor,context:SearchEvaluationContext)->PopulationEvaluation: ...

def search_stream_seed(seed,call_index,stream,generation=-1,step=-1):
    payload=f'latent-search-stream-v2:{int(seed)}:{int(call_index)}:{stream}:{int(generation)}:{int(step)}'.encode()
    return int.from_bytes(hashlib.blake2b(payload,digest_size=8).digest(),'little')%(2**63-1)

__all__=['HybridSearchConfig','SearchEvaluationContext','PopulationEvaluation','PopulationEvaluator','search_stream_seed']
