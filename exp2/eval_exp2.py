"""同一训练 checkpoint，使用 validation 原始种子比较评分后端。非 rollout 训练。"""
import argparse
import csv
import json
import sys
from pathlib import Path
from contextlib import contextmanager
from dataclasses import fields,replace,asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
import torch
from CALSESD.models import GeometryDynamicPointerPolicy,ValueNetwork
from CALSESD.training.ppo_config import PPOConfig
import CALSESD.training.validation as validation
from CALSESD.exp2.rollout_search import factories

@contextmanager
def backend(name,chunk,repeats):
    old=validation.HybridLatentSearch,validation.ActorOnlyCollector
    try:
        if name=='rollout':
            validation.HybridLatentSearch,validation.ActorOnlyCollector=factories(chunk,repeats)
        yield
    finally:
        validation.HybridLatentSearch,validation.ActorOnlyCollector=old

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out-dir',default=str(ROOT/'exp2'/'results'))
    p.add_argument('--chunk',type=int,default=8)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--validation-seeds',type=int)
    a=p.parse_args()
    if a.chunk<1 or a.repeats<1: p.error('chunk/repeats must be positive')
    data=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    keys={f.name for f in fields(PPOConfig)}
    cfg=PPOConfig(**{k:v for k,v in data['config'].items() if k in keys})
    # 核心模块：同一参数设置，唯一实验开关为评分后端；保留原验证 seed 起点。
    cfg=replace(cfg,mode='evolution',ablation_family='exp1',ablation_variant='none',
                ablation_fixed_sigma=None,ablation_min_gain=None,ablation_record_diagnostics=False,
                gradient_steps=0,lcb_lambda=0.,reopt_min_gain_ratio=0.,
                online_search_budget_seconds=0.,online_batch_budget_seconds=0.,
                validation_initial_latent_source='common')
    if a.validation_seeds is not None: cfg=replace(cfg,validation_seeds=a.validation_seeds)
    if cfg.validation_seeds<1: p.error('validation-seeds must be positive')
    device=torch.device(a.device)
    if device.type=='cuda' and device.index is None: device=torch.device('cuda',torch.cuda.current_device())
    actor=GeometryDynamicPointerPolicy(hidden_dim=cfg.hidden_dim,dynamic_dim=cfg.dynamic_dim,
        latent_dim=cfg.latent_dim,num_heads=cfg.num_heads,static_gat_layers=cfg.static_gat_layers,
        static_transformer_layers=cfg.static_transformer_layers,dropout=cfg.dropout).to(device)
    critic=ValueNetwork(hidden_dim=cfg.hidden_dim,latent_dim=cfg.latent_dim,n_critics=cfg.n_critics,
                        prior_scale=cfg.critic_prior_scale).to(device)
    actor.load_state_dict(data['actor_state_dict']);critic.load_state_dict(data['hybrid_critic_state_dict'])
    actor.cache_static_fusion=cfg.actor_cache_static_fusion
    critic.configure_static_cache(enabled=cfg.critic_cache_static,max_entries=cfg.critic_static_cache_size)
    actor.eval();critic.eval()
    z=data['collector_state']['persistent_z'].to(device)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(dict(checkpoint=str(Path(a.checkpoint).resolve()),
        update=data['update'],config=asdict(cfg),simulation_repeats=a.repeats,
        confirmation=False,training_backend='original_checkpoint',
        continuation_policy='greedy_fixed_z_no_future_search'),indent=2),encoding='utf-8')
    summary=[]; episodes={}
    columns=['update','seed','batch_index','completed','total_cost','geometric_distance','episode_length','penalized_cost']
    for name in ('policy','critic','rollout'):
        with (out/f'episodes_{name}.csv').open('w',newline='') as h, (out/f'events_{name}.jsonl').open('w') as eh:
            writer=csv.DictWriter(h,fieldnames=columns);writer.writeheader()
            def sink(e): eh.write(json.dumps(e)+'\n')
            with backend(name,a.chunk,a.repeats):
                result=validation.validate_controller(cfg,actor,critic,z,device,episode_writer=writer,
                    update=int(data['update']),protocol='policy' if name=='policy' else 'system',diagnostics_sink=sink)
        with (out/f'episodes_{name}.csv').open() as h: episodes[name]=list(csv.DictReader(h))
        summary.append(dict(method=name,completion_rate=result['completion_rate'],
            mean_penalized_cost=result['mean_penalized_cost'],search_calls=result['validation_search_calls'],
            switched_rows=result['validation_switched_rows'],search_seconds=result['validation_search_seconds']))
    with (out/'summary.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    # 核心模块：先按 seed 聚合 batch 行，再计算配对标准误，避免把共享交通的行当独立样本。
    import math
    paired=[]
    def by_seed(rows):
        d={}
        for r in rows: d.setdefault(int(r['seed']),[]).append(float(r['penalized_cost']))
        return {k:sum(v)/len(v) for k,v in d.items()}
    for name,base in [('critic','policy'),('rollout','policy'),('rollout','critic')]:
        x,y=by_seed(episodes[name]),by_seed(episodes[base])
        diff=[x[k]-y[k] for k in sorted(x)];n=len(diff);mean=sum(diff)/n
        se=math.sqrt(sum((v-mean)**2 for v in diff)/(n*(n-1))) if n>1 else None
        paired.append(dict(method=name,baseline=base,n_seeds=n,mean_cost_diff=mean,seed_cluster_se=se))
    with (out/'paired.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
