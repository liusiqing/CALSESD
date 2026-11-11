"""训练期 rollout 进化搜索：真实轨迹 PPO，不把模拟轨迹加入训练。"""
import sys,json,csv,random,time
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
import numpy as np
import torch
from CALSESD.training.env_factory import make_env
from CALSESD.training.online_trigger import OnlineTrigger
from CALSESD.training.ppo_update import update_actor_critic
from CALSESD.models.hybrid_latent_search import HybridLatentSearch
from CALSESD.exp2.config import parse
from CALSESD.exp2.common import models,device_for
from CALSESD.exp2.search import RolloutSearch,Exp2Collector,search_config
from CALSESD.exp2.validation import validate
from CALSESD.exp2.checkpoint import save,resume

def main():
    cfg=parse();device=device_for(cfg.device)
    random.seed(cfg.seed);np.random.seed(cfg.seed%2**32);torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    def base(p):
        p=Path(p).expanduser();return p if p.is_absolute() else ROOT/p
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=base(cfg.output_dir)/('exp2_'+cfg.exp2_backend)/stamp
    ckpt=base(cfg.checkpoints_dir)/('exp2_'+cfg.exp2_backend)/stamp
    out.mkdir(parents=True);ckpt.mkdir(parents=True)
    actor,critic=models(cfg,device)
    optimizers=(torch.optim.Adam(actor.parameters(),lr=cfg.learning_rate),
                torch.optim.Adam(critic.parameters(),lr=cfg.value_learning_rate))
    env=make_env(cfg,cfg.seed,device);trigger=OnlineTrigger.from_config(cfg);env.set_online_trigger(trigger)
    cls=RolloutSearch if cfg.exp2_backend=='rollout' else HybridLatentSearch
    collector=Exp2Collector(cfg,env,actor,critic,cls(search_config(cfg)),online_trigger=trigger)
    start,best=1,(0.,float('-inf'))
    if cfg.resume:
        data=torch.load(cfg.resume,map_location='cpu',weights_only=False)
        start,best=resume(data,cfg,actor,critic,optimizers,collector,device)
    elif cfg.init_checkpoint:
        data=torch.load(cfg.init_checkpoint,map_location='cpu',weights_only=False)
        actor.load_state_dict(data['actor_state_dict']);critic.load_state_dict(data['hybrid_critic_state_dict'])
    (out/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
    (out/'protocol.json').write_text(json.dumps(dict(family='exp2',training_backend=cfg.exp2_backend,
        resolved_search_config=asdict(collector.search.config),
        simulation_policy='greedy_fixed_z_no_future_search',real_training_policy='sampled_actor',
        validation_seeds=list(range(cfg.validation_seed,cfg.validation_seed+cfg.validation_seeds)),
        initialization='resume' if cfg.resume else 'warm_start' if cfg.init_checkpoint else 'scratch'),indent=2))
    fields=['update','global_step','backend','completion_rate','mean_completed_cost','policy_loss','value_loss',
            'entropy','approx_kl','search_calls','search_due_rows','switched_rows','rollout_episodes',
            'rollout_env_steps','rollout_batch_steps','update_seconds']
    with (out/'training_metrics.csv').open('w',newline='') as h,(out/'events.jsonl').open('w') as eh:
        writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader()
        def event(e): eh.write(json.dumps(e,allow_nan=False)+'\n');eh.flush()
        for update in range(start,cfg.updates+1):
            began=time.perf_counter()
            # 核心模块：真实动作前，rollout 回报参与候选排序、进化与按行接受。
            r=collector.collect(update)
            events=list(collector.enhancement_events);collector.enhancement_events.clear()
            for e in events: event(e)
            # 核心模块：仅真实 Actor 轨迹用于 PPO/GAE；Critic 仍作为价值基线训练。
            metrics=update_actor_critic(cfg,actor,critic,optimizers,r,device)
            collector.critic_updates+=1
            if update%cfg.validation_interval==0:
                result=validate(cfg,actor,critic,collector.z,device,update,sink=event)
                key=(result['completion_rate'],-result['mean_penalized_cost'])
                event(dict(type='exp2_validation',update=update,backend=cfg.exp2_backend,
                           completion_rate=result['completion_rate'],mean_penalized_cost=result['mean_penalized_cost']))
                if cfg.validation_paired:
                    policy=validate(cfg,actor,critic,collector.z,device,update,protocol='policy')
                    event(dict(type='paired_validation',update=update,
                        search_gain=policy['mean_penalized_cost']-result['mean_penalized_cost']))
                if key>best:
                    best=key;save(ckpt/'checkpoint_best.pt',cfg,actor,critic,optimizers,collector,update,best)
            if update%cfg.latest_checkpoint_interval==0 or update==cfg.updates:
                save(ckpt/'checkpoint_latest.pt',cfg,actor,critic,optimizers,collector,update,best)
            if update%cfg.checkpoint_interval==0:
                save(ckpt/f'checkpoint_update_{update:05d}.pt',cfg,actor,critic,optimizers,collector,update,best)
            if device.type=='cuda':torch.cuda.synchronize(device)
            row=dict(update=update,global_step=collector.global_step,backend=cfg.exp2_backend,
                completion_rate=len(r.completed_costs)/cfg.batch_size,
                mean_completed_cost=sum(r.completed_costs)/len(r.completed_costs) if r.completed_costs else None,
                **{k:metrics[k] for k in ['policy_loss','value_loss','entropy','approx_kl']},
                search_calls=r.evolution_search_calls,search_due_rows=r.due_rows,switched_rows=r.switched_rows,
                **{k:sum(int(e.get(k,0)) for e in events) for k in
                   ['rollout_episodes','rollout_env_steps','rollout_batch_steps']},
                update_seconds=time.perf_counter()-began)
            writer.writerow(row);h.flush();print(row,flush=True)
    print('checkpoint directory:',ckpt)

if __name__=='__main__':main()
