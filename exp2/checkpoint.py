"""exp2 专用 checkpoint，避免原 Critic checkpoint 被误当作 rollout 续训。"""
from dataclasses import asdict
from CALSESD.training.checkpoint import _atomic_save,_rng_state,_restore_rng
import torch

SCHEME='exp2-training-evolution-v1'
IGNORE={'updates','output_dir','checkpoints_dir','resume','init_checkpoint','device',
        'checkpoint_interval','latest_checkpoint_interval','save_step_csv'}

def save(path,cfg,actor,critic,optimizers,collector,update,best):
    _atomic_save(dict(scheme=SCHEME,config=asdict(cfg),update=update,best_key=best,
        actor_state_dict=actor.state_dict(),hybrid_critic_state_dict=critic.state_dict(),
        optimizer_states=[o.state_dict() for o in optimizers],search_state=collector.search.state_dict(),
        online_trigger_state=collector.online_trigger.state_dict(),
        collector_state=dict(persistent_z=collector.z.detach().cpu().clone(),
             global_step=collector.global_step,critic_updates=collector.critic_updates),rng=_rng_state()),path)

def resume(data,cfg,actor,critic,optimizers,collector,device):
    if data['scheme']!=SCHEME: raise ValueError('Not an exp2 training checkpoint; use init-checkpoint for weights')
    live={k:v for k,v in asdict(cfg).items() if k not in IGNORE}
    saved={k:v for k,v in data['config'].items() if k not in IGNORE}
    if live!=saved: raise ValueError('Resume configuration differs, including exp2 backend/chunk/repeats')
    actor.load_state_dict(data['actor_state_dict']);critic.load_state_dict(data['hybrid_critic_state_dict'])
    for o,s in zip(optimizers,data['optimizer_states']): o.load_state_dict(s)
    collector.search.load_state_dict(data['search_state'])
    collector.online_trigger.load_state_dict(data['online_trigger_state'])
    s=data['collector_state'];collector.z=s['persistent_z'].to(device)
    collector.global_step=s['global_step'];collector.critic_updates=s['critic_updates']
    critic.clear_static_cache();_restore_rng(data['rng'])
    return data['update']+1,tuple(data['best_key'])
