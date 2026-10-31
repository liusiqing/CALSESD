"""Actor-only PPO entry for phase-one periodic in-episode latent search.

This entry refuses stale update-period trainer/validation/checkpoint/logging
modules. Search occurs before real actions at OnlineTrigger decision points;
no representative pool/provider, rollout candidate evaluator or route planner
is created here. It deliberately supports Critic scoring only in this phase.
"""
from __future__ import annotations
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime
from numbers import Integral
from pathlib import Path
import numpy as np
import torch
from copy import deepcopy

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


from CALSESD.models import GeometryDynamicPointerPolicy
from CALSESD.models.hybrid_latent_search import HybridLatentSearch
from CALSESD.models.value import ValueNetwork
from CALSESD.training.ppo_config import parse_args,validate_config
from CALSESD.training.online_trigger import OnlineTrigger
from CALSESD.training.ablation_config import (resolve_ablation,validation_settings,
    ablation_manifest,build_search_config)
from CALSESD.training.env_factory import make_env
from CALSESD.training.ppo_rollout import ActorOnlyCollector,Rollout,VALUE_SEMANTICS,BOOTSTRAP_SEMANTICS
import CALSESD.training.ppo_update as update_io
import CALSESD.training.validation as validation_io
import CALSESD.training.checkpoint as checkpoint_io
import CALSESD.training.compact_logging as logging_io

PROJECT_ROOT=Path(__file__).resolve().parents[1]
CORE_METRICS=('policy_loss','value_loss','entropy','approx_kl')


def _count(name,value):
    if isinstance(value,bool) or not isinstance(value,Integral) or value<0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _synchronize(device):
    device=torch.device(device)
    if device.type=='cuda':torch.cuda.synchronize(device)


def _check_contracts(cfg):
    validate_config(cfg)
    plan=resolve_ablation(cfg)
    if getattr(plan,'search_scope',None)!='episode':
        raise RuntimeError('Install the episode-scope ablation_config.py')
    if plan.evaluation_backend!='critic':
        raise RuntimeError('This phase-one entry supports Critic scoring only; rollout search is not wired into online continuation')
    required={'ppo_env_steps','latent_switch_masks','search_due_masks','value_semantics','bootstrap_semantics'}
    missing=required-set(getattr(Rollout,'__dataclass_fields__',{}))
    if missing:raise RuntimeError(f'Install the in-episode collector; missing fields {sorted(missing)}')
    if Rollout.__dataclass_fields__['value_semantics'].default!=VALUE_SEMANTICS or Rollout.__dataclass_fields__['bootstrap_semantics'].default!=BOOTSTRAP_SEMANTICS:
        raise RuntimeError('Collector value/bootstrap semantics mismatch')
    if not callable(getattr(update_io,'_validate_online_rollout',None)):
        raise RuntimeError('Install the online/deferred-bootstrap ppo_update.py')
    if getattr(validation_io,'SEARCH_SCOPE',None)!='episode':
        raise RuntimeError('Install validation.py using the same in-episode trigger protocol')
    signature=__import__('inspect').signature(validation_io.validate_controller)
    if not {'protocol','diagnostics_sink'}<=set(signature.parameters):
        raise RuntimeError('validation.py must accept protocol and diagnostics_sink')
    if getattr(validation_io, 'VALIDATION_SEMANTICS', None) != 'episode_online_validation_v1':
        raise RuntimeError('Install the episode-online validation.py')

    if (
        getattr(checkpoint_io, 'SCHEME', None) != 'online-critic-latent-actor-only-v1'
        or getattr(checkpoint_io, 'VALUE_SEMANTICS', None) != VALUE_SEMANTICS
        or getattr(checkpoint_io, 'BOOTSTRAP_SEMANTICS', None) != BOOTSTRAP_SEMANTICS
        or getattr(checkpoint_io, 'SUPPORTS_ONLINE_TRIGGER_RESUME', False) is not True
        or getattr(checkpoint_io, 'TRAINING_STATE_VERSION', 0) < 3
    ):
        raise RuntimeError('Install the matching online checkpoint.py')

    train_fields = set(getattr(logging_io, 'TRAIN_FIELDS', ()))
    validation_fields = set(getattr(logging_io, 'VALIDATION_FIELDS', ()))
    search_fields = set(getattr(logging_io, 'SEARCH_FIELDS', ()))

    needed_train = {
        'ablation_family', 'ablation_variant', 'ppo_env_steps', 'latent_search_seconds'
    }
    needed_validation = {
        'validation_protocol', 'validation_initial_latent_source',
        'validation_search_called', 'validation_search_timed_out',
        'validation_search_seconds',
    }
    needed_search = {
        'update', 'evaluation_backend', 'search_calls_counter',
        'due_rows', 'switched_rows', 'timed_out', 'core_search_seconds',
    }

    missing_fields = {
        'TRAIN_FIELDS': sorted(needed_train - train_fields),
        'VALIDATION_FIELDS': sorted(needed_validation - validation_fields),
        'SEARCH_FIELDS': sorted(needed_search - search_fields),
    }
    missing_fields = {
        name: fields
        for name, fields in missing_fields.items()
        if fields
    }

    if missing_fields:
        raise RuntimeError(
            'compact_logging.py schema is incompatible with online search; '
            f'module={logging_io.__name__}; '
            f'file={getattr(logging_io, "__file__", None)}; '
            f'missing={missing_fields}'
        )
    return plan


def _seed_everything(seed):
    seed=int(seed);random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def _device(value):
    d=torch.device('cuda' if value=='auto' and torch.cuda.is_available() else 'cpu' if value=='auto' else value)
    if d.type=='cuda' and d.index is None:d=torch.device('cuda',torch.cuda.current_device())
    return d


def _checkpoint_path(value):
    if not value:return None
    path=Path(value).expanduser().resolve()
    if not path.is_file():raise FileNotFoundError(f'Checkpoint does not exist: {path}')
    return path


def _write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def _output_base(value):
    path=Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT/path


def _build_search_config(cfg,plan):
    # Single source of truth; checkpoint._identity compares against this exact builder.
    return build_search_config(cfg)


def _configure_networks(cfg,actor,critic):
    if not hasattr(actor,'cache_static_fusion'):raise RuntimeError('Install the dual-stream Actor')
    if (getattr(critic,'value_semantics',None)!=VALUE_SEMANTICS
            or getattr(critic,'online_context_dim',None)!=6):raise RuntimeError('Install online-controller Critic')
    for name in ('encode_state','evaluate_features','mean_value','all_values','encode_static','configure_static_cache','clear_static_cache'):
        if not callable(getattr(critic,name,None)):raise RuntimeError(f'Critic API missing {name}')
    actor.cache_static_fusion=cfg.actor_cache_static_fusion
    critic.configure_static_cache(enabled=cfg.critic_cache_static,max_entries=cfg.critic_static_cache_size)


def _audit_rollout(r,step_delta):
    if not r.valid_masks:raise RuntimeError('empty rollout')
    valid=torch.stack(r.valid_masks).bool();actor=torch.stack(r.actor_masks).bool()
    if actor.shape!=valid.shape or not torch.equal(actor,valid):raise RuntimeError('every valid sample must be Actor generated')
    samples=int(valid.sum())
    if samples<1 or samples!=step_delta or samples!=_count('ppo_env_steps',r.ppo_env_steps):raise RuntimeError('PPO sample/step counter mismatch')
    if r.value_semantics!=VALUE_SEMANTICS or r.bootstrap_semantics!=BOOTSTRAP_SEMANTICS:raise RuntimeError('rollout semantics mismatch')
    return samples


def _audit_search_events(r,events,calls_before,calls_after,trigger):
    calls=_count('search calls',r.evolution_search_calls)
    actual=[e for e in events if e.get('type')=='joint_latent_search']
    if calls_after-calls_before!=calls or len(actual)!=calls:raise RuntimeError('search call/event counts disagree')
    ids=[_count('search call id',e.get('calls')) for e in actual]
    if ids!=list(range(calls_before+1,calls_after+1)):raise RuntimeError('search RNG call ids are not contiguous')
    due=switched=timeouts=0
    for event in actual:
        n=_count('due_rows',event.get('due_rows'));s=_count('switched_rows',event.get('switched_rows'))
        timeout=event.get('timed_out')
        if not isinstance(timeout,bool) or s>n or (timeout and s):raise RuntimeError('invalid search outcome')
        if event.get('search_scope')!='episode' or event.get('evaluation_backend')!='critic':raise RuntimeError('unexpected search backend/scope')
        if event.get('rollout_counts_complete') is not True or any(_count(k,event.get(k))!=0 for k in ('rollout_episodes','rollout_env_steps','rollout_batch_steps')):
            raise RuntimeError('Critic-only search reported simulated rollout')
        due+=n;switched+=s;timeouts+=int(timeout)
    if (due,switched)!=(r.due_rows,r.switched_rows):raise RuntimeError('search event and collector row totals disagree')
    if trigger.pending:raise RuntimeError('pending online search after collect')
    seconds=float(trigger.batch_seconds_used)
    if not math.isfinite(seconds) or seconds<0:raise RuntimeError('invalid trigger batch-time counter')
    return dict(search_calls=calls,search_due_rows=due,latent_switch_rows=switched,
        search_timeouts=timeouts,batch_search_seconds=seconds)


def _drain_search_events(logs,collector):
    for event in collector.enhancement_events:logs.event(event)
    collector.enhancement_events.clear()


def _check_metrics(metrics):
    bad=[]
    for name in CORE_METRICS:
        try:ok=metrics.get(name) is not None and math.isfinite(float(metrics[name]))
        except (TypeError,ValueError,OverflowError):ok=False
        if not ok:bad.append(name)
    if bad:raise FloatingPointError(f'nonfinite PPO/critic metrics: {bad}')


def _validation_key(values,settings,cfg):
    if values.get('validation_protocol')!=settings['protocol'] or values.get('validation_search_scope')!='episode':
        raise RuntimeError('validation protocol/scope mismatch')
    calls=_count('validation_search_calls',values.get('validation_search_calls'))
    timeouts=_count('validation_search_timeouts',values.get('validation_search_timeouts'))
    switched=_count('validation_switched_rows',values.get('validation_switched_rows'))
    called,timed=values.get('validation_search_called'),values.get('validation_search_timed_out')
    if not isinstance(called,bool) or called!=(calls>0):raise RuntimeError('validation call count/flag mismatch')
    if not isinstance(timed,bool) or timed!=(timeouts>0) or timeouts>calls:raise RuntimeError('validation timeout count/flag mismatch')
    if switched>calls*cfg.batch_size:raise RuntimeError('validation switch count exceeds search capacity')
    if not settings['search_enabled'] and calls:raise RuntimeError('policy-only validation unexpectedly searched')
    seconds=values.get('validation_search_seconds')
    if seconds is None or not math.isfinite(float(seconds)) or seconds<0:raise RuntimeError('invalid validation search seconds')
    details=values.get('validation_details') or {}
    expected={'common':{'common_seeded_prior'},'provided':{'supplied_persistent_z_clone','supplied_training_persistent_z'}}[settings['initial_latent_source']]
    if details.get('latent_source') not in expected:raise RuntimeError('validation latent source mismatch')
    rate,cost=values.get('completion_rate'),values.get('mean_penalized_cost')
    if rate is None or cost is None or not math.isfinite(float(rate)) or not math.isfinite(float(cost)) or not 0<=rate<=1:
        raise FloatingPointError('invalid validation result')
    return float(rate),-float(cost)


def main():
    cfg=parse_args();plan=_check_contracts(cfg)
    search_config=_build_search_config(cfg,plan)
    manifest=ablation_manifest(cfg,search_config)
    resume_path=_checkpoint_path(cfg.resume);init_path=_checkpoint_path(cfg.init_checkpoint)
    _seed_everything(cfg.seed);device=_device(cfg.device)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_base,ckpt_base=_output_base(cfg.output_dir),_output_base(cfg.checkpoints_dir)
    if out_base.resolve()==ckpt_base.resolve():raise ValueError('output and checkpoint directories must differ')
    out_dir,ckpt_dir=out_base/stamp,ckpt_base/stamp
    out_dir.mkdir(parents=True,exist_ok=False);ckpt_dir.mkdir(parents=True,exist_ok=False)
    env=make_env(cfg,cfg.seed,device)   # same construction path as validation
    trigger=OnlineTrigger.from_config(cfg);env.set_online_trigger(trigger)
    actor=GeometryDynamicPointerPolicy(dynamic_input_dim=6,geometry_input_dim=2,hidden_dim=cfg.hidden_dim,
        dynamic_dim=cfg.dynamic_dim,latent_dim=cfg.latent_dim,num_heads=cfg.num_heads,
        static_gat_layers=cfg.static_gat_layers,static_transformer_layers=cfg.static_transformer_layers,
        dropout=cfg.dropout).to(device)
     critic=ValueNetwork(hidden_dim=cfg.hidden_dim,latent_dim=cfg.latent_dim,n_critics=cfg.n_critics,prior_scale=cfg.critic_prior_scale).to(device)
    _configure_networks(cfg,actor,critic)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=cfg.learning_rate)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=cfg.value_learning_rate)
    search=HybridLatentSearch(search_config)
    if (search.state_dict().get('format_version')!=3
            or search.state_dict().get('value_semantics')!=VALUE_SEMANTICS):raise RuntimeError('online search state semantics mismatch')
    models=(actor,critic);optimizers=(actor_opt,critic_opt)
    _write_json(out_dir/'config.json',asdict(cfg));_write_json(out_dir/'online_protocol.json',manifest)
    with logging_io.LogBundle(out_dir,debug=cfg.save_step_csv) as logs:
        logs.event(dict(type='ablation_setup',plan={**asdict(plan),'search_scope':'episode'},
                        resolved_search_config=asdict(search_config)))
        collector=ActorOnlyCollector(cfg,env,actor,critic,search,logs.episode_csv,logs.step_csv,online_trigger=trigger)
        best=(0.,float('-inf'));start=1;history=[];origin='from_scratch'
        if resume_path is not None:
            start,best,history=checkpoint_io.load_checkpoint(resume_path,cfg,models,optimizers,collector,device);origin='strict_resume'
        elif init_path is not None:
            checkpoint_io.initialize_from_checkpoint(init_path,models,collector,device);origin='weight_warm_start'
            print('WARNING: init_checkpoint is a warm-start, not a scratch ablation run',flush=True)
        if collector.critic_updates!=start-1:raise RuntimeError('checkpoint update and critic-update counters disagree')
        info=dict(initialization=origin,seed=cfg.seed,device=str(device),search_scope='episode',
            start_update=start,total_updates=cfg.updates,global_step=collector.global_step,search_calls=search.calls,
            ablation_family=plan.family,ablation_variant=plan.variant,checkpoint_version=checkpoint_io.FORMAT_VERSION,
            checkpoint_training_state_version=checkpoint_io.TRAINING_STATE_VERSION)
        _write_json(out_dir/'run.json',info);logs.event(dict(type='training_start',**info))
        for update in range(start,cfg.updates+1):
            timer=logging_io.UpdateTimer(lambda:_synchronize(device))
            try:
                step_before=collector.global_step;calls_before=search.calls
                with timer.phase('collect'):r=collector.collect(update)
                actor_count=_audit_rollout(r,collector.global_step-step_before)
                events=[deepcopy(e) for e in collector.enhancement_events if e.get('type')=='joint_latent_search']
                search_stats=_audit_search_events(r,events,calls_before,search.calls,trigger)
                _drain_search_events(logs,collector)
                with timer.phase('ppo'):metrics=update_io.update_actor_critic(cfg,actor,critic,optimizers,r,device)
                if (_count('PPO search rows',metrics['search_attempt_rows'])!=search_stats['search_due_rows']
                        or _count('PPO switch rows',metrics['latent_switch_events'])!=search_stats['latent_switch_rows']):
                    raise RuntimeError('PPO trigger masks and emitted search events disagree')
                metrics.update(search_stats,search_scope='episode',ablation_family=plan.family,ablation_variant=plan.variant,
                    ppo_env_steps=r.ppo_env_steps,latent_search_seconds=r.latent_seconds,predicted_gain=r.predicted_gain,
                    ensemble_disagreement=r.ensemble_disagreement,mutation_sigma=r.mutation_sigma,latent_gradient_norm=r.latent_gradient_norm)
                _check_metrics(metrics)
                quality=float(metrics.get('explained_variance',float('nan')))
                if math.isfinite(quality):history.append(quality)
                history=history[-10:];collector.critic_updates+=1
                batch_event=dict(family=plan.family,variant=plan.variant,search_scope='episode',schedule='periodic',
                    **search_stats,ppo_env_steps=r.ppo_env_steps,row_attempts=list(trigger.attempts),
                    search_status=deepcopy(collector.last_search_status))
                logs.event(dict(type='ablation_training_update',update=update,**batch_event))
                validation=None;key=None
                if update%cfg.validation_interval==0:
                    with timer.phase('validation'):
                        validation=validation_io.validate_controller(cfg,actor,critic,collector.z,device,
                            episode_writer=logs.validation_episode_csv,update=update,protocol=cfg.validation_protocol,
                            diagnostics_sink=logs.event)
                        paired=None
                        if cfg.validation_paired and cfg.validation_protocol=='system':
                            paired=validation_io.validate_controller(cfg,actor,critic,collector.z,device,
                                episode_writer=None,update=update,protocol='policy',diagnostics_sink=None)
                    key=_validation_key(validation,manifest['validation_settings'],cfg)
                    logs.validation_csv.writerow(logging_io.build_validation_row(update,validation,timer.seconds['validation']))
                    if paired is not None:
                        sys_cost,pol_cost=validation.get('mean_penalized_cost'),paired.get('mean_penalized_cost')
                        logs.event(dict(type='paired_validation',update=update,
                            system_penalized_cost=sys_cost,policy_penalized_cost=pol_cost,
                            search_gain=(pol_cost-sys_cost) if None not in (sys_cost,pol_cost) else None,
                            system_completion_rate=validation.get('completion_rate'),
                            policy_completion_rate=paired.get('completion_rate'),
                            latent_source=validation['validation_details'].get('latent_source')))
                with timer.phase('checkpoint'):
                    if validation is not None and key>best:
                        best=key;checkpoint_io.save_checkpoint(ckpt_dir/'checkpoint_best.pt',cfg,models,optimizers,update,best,r,collector,history)
                    if update%cfg.latest_checkpoint_interval==0 or update==cfg.updates:
                        checkpoint_io.save_checkpoint(ckpt_dir/'checkpoint_latest.pt',cfg,models,optimizers,update,best,r,collector,history)
                    if update%cfg.checkpoint_interval==0:
                        checkpoint_io.save_checkpoint(ckpt_dir/f'checkpoint_update_{update:05d}.pt',cfg,models,optimizers,update,best,r,collector,history)
                row=logging_io.build_training_row(update=update,global_step=collector.global_step,batch_size=cfg.batch_size,
                    completed_costs=r.completed_costs,geometric_distances=r.geometric_distances.detach().cpu().tolist(),
                    completed_mask=r.completed_mask.detach().cpu().bool().tolist(),metrics=metrics,
                    update_seconds=timer.elapsed(),phase_seconds=timer.seconds)
                logs.training_csv.writerow(row);logs.event(dict(type='update_timing',update=update,**timer.seconds))
                logs.event(dict(type='inference_timing',update=update,batch_size=cfg.batch_size,**r.inference_metrics))
                if logs.debug:logs.event(dict(type='debug_update',update=update,metrics=metrics,actor_samples=actor_count))
                logs.flush()
                cost=row['mean_completed_cost'];cost_text=f'{cost:.3f}' if cost is not None else 'NA'
                print(f'update={update:04d} ablation={plan.family}/{plan.variant} complete={row["completion_rate"]:.1%} '
                    f'cost={cost_text} steps={r.ppo_env_steps} search={search_stats["search_calls"]} '
                    f'switch={search_stats["latent_switch_rows"]} timeout={search_stats["search_timeouts"]} '
                    f'sec={row["update_seconds"]:.2f}',flush=True)
            except Exception as exc:
                try:
                    _drain_search_events(logs,collector)
                    logs.event(dict(type='training_failed',update=update,ablation_family=plan.family,
                        ablation_variant=plan.variant,global_step=collector.global_step,
                        error_type=type(exc).__name__,error=str(exc),search_status=deepcopy(collector.last_search_status)))
                    logs.flush()
                except Exception as log_exc:print(f'Could not fully log failure: {log_exc}',file=sys.stderr,flush=True)
                raise

if __name__=='__main__':main()
