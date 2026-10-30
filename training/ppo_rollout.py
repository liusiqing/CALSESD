"""Full-episode Actor-only collection with periodic in-episode latent search.

Search uses current pending state and token.due, never representatives. Real
PPO states are refreshed AFTER finish_attempt. The previous transition's
bootstrap is filled from the NEXT actual post-trigger decision value.
Only finite full episodes are supported; no Critic bootstrap at task cutoff.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass,field,asdict
import math
from numbers import Integral
from time import perf_counter
import torch
from CALSESD_internal.models import build_geometry_data
from CALSESD_internal.training.online_trigger import OnlineTrigger,OnlineTriggerConfig
from CALSESD_internal.training.ablation_config import resolve_ablation
from CALSESD_internal.training.inference_timing import InferenceTiming
from CALSESD_internal.training.rng_state import preserve_rng

VALUE_SEMANTICS='online_actor_controller_v1'
BOOTSTRAP_SEMANTICS='online_post_trigger_next_decision_v1'

@dataclass
class Rollout:
    states:list=field(default_factory=list)
    geometry_inputs:list=field(default_factory=list)
    latents:list=field(default_factory=list)
    masks:list=field(default_factory=list)
    actions:list=field(default_factory=list)
    log_probs:list=field(default_factory=list)
    old_logits:list=field(default_factory=list)
    valid_masks:list=field(default_factory=list)
    actor_masks:list=field(default_factory=list)
    rewards:list=field(default_factory=list)
    boundaries:list=field(default_factory=list)
    values:list=field(default_factory=list)
    next_values:list=field(default_factory=list)
    latent_switch_masks:list=field(default_factory=list)
    search_due_masks:list=field(default_factory=list)
    completed_costs:list=field(default_factory=list)
    completed_lengths:list=field(default_factory=list)
    truncated_count:int=0
    due_rows:int=0
    switched_rows:int=0
    latent_seconds:float=0.
    evolution_search_calls:int=0
    predicted_gain:object=None
    confirmation_gain:object=None  # No paired confirmation in this phase.
    ensemble_disagreement:object=None
    mutation_sigma:object=None
    latent_gradient_norm:object=None
    geometric_distances:object=None
    completed_mask:object=None
    ppo_env_steps:int=0
    rollout_episodes:int=0
    rollout_env_steps:int=0
    rollout_batch_steps:int=0
    rollout_counts_complete:bool=True
    inference_metrics:dict=field(default_factory=dict)
    ablation_metrics:dict=field(default_factory=dict)
    search_diagnostics:dict=field(default_factory=dict)  # Latest call; events retain all calls.
    value_semantics:str=VALUE_SEMANTICS
    bootstrap_semantics:str=BOOTSTRAP_SEMANTICS
    _diagnostic_sums:dict=field(default_factory=dict,repr=False)
    _diagnostic_weights:dict=field(default_factory=dict,repr=False)


def _weighted(r,name,value,weight=1):
    if value is None or weight<=0:return
    value=float(value)
    if not math.isfinite(value):raise ValueError(f'nonfinite search diagnostic {name}')
    r._diagnostic_sums[name]=r._diagnostic_sums.get(name,0.)+value*weight
    r._diagnostic_weights[name]=r._diagnostic_weights.get(name,0)+weight
    setattr(r,name,r._diagnostic_sums[name]/r._diagnostic_weights[name])


def record_latent_diagnostics(r,d):
    n=int(d.get('due_rows',0));r.due_rows+=n;r.switched_rows+=int(d.get('switched_rows',0))
    # A timeout may leave the searcher's initial gain=0 without a completed
    # estimate. Never treat that default as a measured zero improvement.
    if not bool(d.get('timed_out',False)):
        _weighted(r,'predicted_gain',d.get('predicted_gain_mean'),n)
    _weighted(r,'ensemble_disagreement',d.get('mean_incumbent_disagreement'),n)
    for generation in d.get('generation_stats') or []:
        _weighted(r,'mutation_sigma',(generation.get('mutation_sigma') or {}).get('mean'))
        _weighted(r,'latent_gradient_norm',generation.get('mean_gradient_norm'))
    r.search_diagnostics=deepcopy(d)


def _synchronize(device):
    device=torch.device(device)
    if device.type=='cuda':torch.cuda.synchronize(device)


class ActorOnlyCollector:
    def __init__(self,cfg,env,policy,critic,search,episode_csv=None,step_csv=None,*,online_trigger=None):
        if cfg.search_scope!='episode' or cfg.bootstrap_time_limits:
            raise ValueError('requires episode-scope finite-task configuration')
        if getattr(critic,'value_semantics',None)!=VALUE_SEMANTICS or getattr(critic,'online_context_dim',None)!=6:
            raise ValueError('install the online-controller Critic')
        if getattr(search.config,'search_scope',None)!='episode' or getattr(search.config,'evaluation_backend',None)!='critic':
            raise ValueError('install the current-state online Critic search')
        for name in ('set_online_trigger','online_decision','get_batch_state','reset','step','action_mask'):
            if not callable(getattr(env,name,None)):raise TypeError(f'native BatchCVRPEnv missing {name}')
        expected=OnlineTriggerConfig.from_config(cfg);expected.validate()
        trigger=online_trigger if online_trigger is not None else getattr(env,'online_trigger',None)
        if trigger is None:trigger=OnlineTrigger(expected)
        if asdict(trigger.config)!=asdict(expected):raise ValueError('trigger configuration does not match cfg')
        env.set_online_trigger(trigger)
        self.cfg,self.env,self.policy,self.critic=cfg,env,policy,critic
        self.search=search;self.online_trigger=trigger;self.ablation_plan=resolve_ablation(cfg)
        self.episode_csv,self.step_csv=episode_csv,step_csv
        self.global_step=0;self.critic_updates=0;self.z=None
        self.last_search_status={};self.last_search_diagnostics={};self.enhancement_events=[]
        self.inference_timing=InferenceTiming(env.device)

    def _validate_z(self,z):
        c=self.cfg
        if (not torch.is_tensor(z) or z.shape!=(c.batch_size,c.latent_dim) or not z.is_floating_point()
                or not bool(torch.isfinite(z).all()) or bool((z.abs()>c.latent_bound).any())):
            raise ValueError('invalid persistent/selected z')

    @torch.no_grad()
    def _decision_state(self,state,update):
        """Return non-pending state and actual due/switch masks for this action."""
        e=self.env
        decision=e.online_decision()
        due=torch.as_tensor(decision.due,dtype=torch.bool,device=e.device)
        switches=torch.zeros_like(due)
        self.last_search_status=dict(schedule='periodic',steps=list(decision.steps),
            reasons=list(decision.reasons),due_rows=int(due.sum()),called=False)
        if not decision.should_search:return state,due,switches,None
        incumbent=self.z.detach().clone();_synchronize(e.device);started=perf_counter()
        # Retain token BEFORE state construction, so an encoding/schema failure
        # cannot leave the trigger pending without a token available for cleanup.
        token=self.online_trigger.begin_attempt(decision)
        success=False;timed_out=False;candidate=incumbent;report=None
        # Cleanup must remain possible without allocating/copying CUDA tensors.
        no_switches=(False,)*self.cfg.batch_size
        cleanup_switches=no_switches
        try:
            with self.inference_timing.measure('latent_search'),preserve_rng():
                preview=e.get_batch_state()
                choose_started=perf_counter()
                candidate,switches=self.search.choose(self.critic,preview,incumbent,due=due,
                    model_version=self.critic_updates,deadline=token.deadline)
                _synchronize(e.device);choose_seconds=perf_counter()-choose_started
                self._validate_z(candidate)
                if candidate.device!=incumbent.device or candidate.dtype!=incumbent.dtype:
                    raise ValueError('search changed latent device/dtype')
                if (not torch.is_tensor(switches) or switches.shape!=due.shape or switches.dtype!=torch.bool
                        or switches.device!=due.device or bool((switches & ~due).any())):
                    raise ValueError('search switched a row outside token.due')
                if not torch.equal((candidate!=incumbent).any(1),switches):
                    raise ValueError('search switch mask does not match actual latent changes')
                timed_out=bool(self.search.last_diagnostics.get('timed_out',False))
                if timed_out and bool(switches.any()):raise ValueError('timeout must retain incumbent latents')
                cleanup_switches=tuple(switches.detach().cpu().tolist())
                success=True
        finally:
            # A synchronize failure must not skip the pure-Python trigger cleanup.
            # Fatal CUDA/search errors still propagate and terminate the run.
            try:
                _synchronize(e.device)
            except BaseException:
                success=False
                raise
            finally:
                try:
                    if self.online_trigger.pending:
                        report=self.online_trigger.finish_attempt(token,
                            switched=cleanup_switches if success else no_switches,
                            timed_out=timed_out)
                except BaseException:
                    success=False
                    raise
                finally:
                    if not success:
                        self.last_search_status.update(called=True,failed=True,attempt_report=report)
        self.z=candidate.detach().clone()
        post=e.get_batch_state()
        if getattr(post,'online_search_pending',None) is not False:
            raise RuntimeError('real decision state must be refreshed after finishing search')
        _synchronize(e.device);seconds=perf_counter()-started
        diagnostics=deepcopy(self.search.last_diagnostics)
        # Keep events/CSV consumers consistent with the aggregate policy.
        # Even a late timeout discards its candidate-gain estimate from summaries.
        diagnostics['gain_in_summary']=not timed_out and diagnostics.get('predicted_gain_mean') is not None
        if timed_out:
            diagnostics['predicted_gain_mean']=None
            diagnostics['selection_gain_mean']=None
        diagnostics.update(update=update,episode_step=max(decision.steps),trigger_reason='periodic',
            due_rows=int(due.sum()),switched_rows=int(switches.sum()),trigger_due_mask=list(token.due),
            ablation_family=self.ablation_plan.family,ablation_variant=self.ablation_plan.variant,
            training_schedule='periodic',collector_search_seconds=seconds,choose_wall_seconds=choose_seconds,
            attempt_report=report,rollout_episodes=0,rollout_env_steps=0,rollout_batch_steps=0,
            rollout_counts_complete=True,rollout_count_source='online_critic_no_rollout')
        self.search.last_diagnostics=deepcopy(diagnostics);self.last_search_diagnostics=deepcopy(diagnostics)
        event=dict(diagnostics);event['type']='joint_latent_search';self.enhancement_events.append(event)
        self.last_search_status.update(called=True,timed_out=timed_out,switched_rows=int(switches.sum()),attempt_report=report)
        return post,due,switches,diagnostics

    def collect(self,update):
        if isinstance(update,bool) or not isinstance(update,Integral) or update<1:raise ValueError('update must be >=1')
        c,e=self.cfg,self.env;initial_step=self.global_step
        self.policy.eval();self.critic.eval();self.last_search_diagnostics={};self.last_search_status={}
        if c.vary_instance_seed:e.env.seed=c.seed+update-1;e.env._seed_rngs()
        state=e.reset(online_completed_updates=update-1,online_training=True)
        if self.z is None:
            self.z=self.policy.sample_prior(c.batch_size,c.latent_dim,e.device,c.latent_bound).detach().clone()
        self._validate_z(self.z)
        geometry_input=build_geometry_data(state,e.env.baseline_distance_matrix,coords=e.env.coords).clone().detach()
        with torch.no_grad():
            with self.inference_timing.measure('actor_geometry'):geometry=self.policy.encode_geometry(geometry_input)
            coords=torch.as_tensor(e.env.coords,dtype=torch.float32,device=e.device)
            geometric_matrix=torch.cdist(coords,coords,p=2)
        r=Rollout();distance=torch.zeros(c.batch_size,device=e.device)
        for t in range(c.max_episode_steps):
            active=~e.dones.clone()
            if not bool(active.any()):break
            state,due,switched,diagnostics=self._decision_state(state,update)
            if diagnostics is not None:
                r.latent_seconds+=diagnostics['collector_search_seconds'];r.evolution_search_calls+=1
                record_latent_diagnostics(r,diagnostics)
            z=self.z.detach().clone()
            if getattr(state,'online_search_pending',None) is not False:raise RuntimeError('cannot execute a pending-search state')
            with torch.no_grad():
                with self.inference_timing.measure('critic_forward'):
                    value=self.critic.mean_value(state,z)
                if value.shape!=active.shape or not bool(torch.isfinite(value[active]).all()):raise ValueError('invalid current value')
                value=value.masked_fill(~active,0.)
                if r.next_values:
                    continuing=r.valid_masks[-1]&~r.boundaries[-1]
                    if bool((continuing & ~active).any()):raise RuntimeError('trajectory disappeared before next decision')
                    # Crucially uses THIS state, after any trigger and actual z_t.
                    r.next_values[-1][continuing]=value[continuing]
                with self.inference_timing.measure('actor_decision'):
                    mask=e.action_mask();logits=self.policy(state,geometry,z,mask)
                    distribution=torch.distributions.Categorical(logits=logits)
                    action=distribution.sample();logp=distribution.log_prob(action)
            snapshot=state.clone().detach()
            distance+=torch.where(active,geometric_matrix[e.current_nodes,action],0.)
            next_state,reward,_=e.step(action)
            done=e.dones.clone();terminated=e.terminated.clone()
            if t+1==c.max_episode_steps and bool((active & ~done).any()):
                raise RuntimeError('environment must apply its real finite-task cutoff/reward; no artificial truncation')
            if reward.shape!=active.shape or not bool(torch.isfinite(reward[active]).all()):raise ValueError('invalid environment reward')
            reward=reward.masked_fill(~active,0.)
            r.states.append(snapshot);r.geometry_inputs.append(geometry_input);r.latents.append(z)
            r.masks.append(mask.detach().clone());r.actions.append(action.detach().clone())
            r.log_probs.append(logp.detach().clone());r.old_logits.append(logits.detach().clone())
            r.valid_masks.append(active);r.actor_masks.append(active.clone())
            r.rewards.append(reward.detach().clone());r.boundaries.append(done)
            r.values.append(value.detach().clone())
            # Terminal/invalid entries remain zero; continuing entries are filled
            # at the next actual decision, never by prematurely searching twice.
            r.next_values.append(torch.zeros_like(value))
            r.latent_switch_masks.append(switched.detach().clone());r.search_due_masks.append(due.detach().clone())
            self.global_step+=int(active.sum())
            for b in torch.where(active & done)[0].tolist():
                completed=bool(terminated[b])
                if completed:r.completed_costs.append(float(e.total_costs[b]));r.completed_lengths.append(int(e.steps[b]))
                else:r.truncated_count+=1
                if self.episode_csv is not None:
                    self.episode_csv.writerow(dict(update=update,global_step=self.global_step,batch_index=b,
                        total_cost=float(e.total_costs[b]),geometric_distance=float(distance[b]),episode_length=int(e.steps[b]),
                        served_customers=int(e.visited_masks[b,e.customer_mask].sum()),done_reason='completed' if completed else 'time_limit'))
            if self.step_csv is not None:
                for b in torch.where(active)[0].tolist():
                    self.step_csv.writerow(dict(update=update,global_step=self.global_step,batch_index=b,
                        action=int(action[b]),reward=float(reward[b]),done=int(done[b]),z_switched=int(switched[b])))
            state=next_state
        if not r.states or not bool(e.dones.all()) or len(r.completed_costs)+r.truncated_count!=c.batch_size:
            raise RuntimeError('collector requires full completed/truncated batch episodes')
        r.geometric_distances=distance.detach().clone();r.completed_mask=e.terminated.detach().clone()
        r.ppo_env_steps=self.global_step-initial_step;r.inference_metrics=self.inference_timing.flush()
        r.ablation_metrics=dict(family=self.ablation_plan.family,variant=self.ablation_plan.variant,
            search_scope='episode',schedule='periodic',ppo_env_steps=r.ppo_env_steps,search_calls=r.evolution_search_calls,
            attempted_rows=r.due_rows,switched_rows=r.switched_rows,search_status=deepcopy(self.last_search_status),
            row_attempts=list(self.online_trigger.attempts),batch_search_seconds=self.online_trigger.batch_seconds_used,
            rollout_episodes=0,rollout_env_steps=0,rollout_batch_steps=0,rollout_counts_complete=True)
        return r

__all__=['Rollout','ActorOnlyCollector','record_latent_diagnostics','VALUE_SEMANTICS','BOOTSTRAP_SEMANTICS']
