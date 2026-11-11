"""从训练期 exp2 checkpoint 读取真实训练后端，复用原 validation 种子。"""
import sys,json,csv,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent))
import torch
from CALSESD.exp2.config import from_saved
from CALSESD.exp2.common import models,device_for
from CALSESD.exp2.validation import validate
from CALSESD.exp2.checkpoint import SCHEME

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='auto');p.add_argument('--out-dir',default=str(ROOT/'exp2/results'))
    a=p.parse_args();data=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if data['scheme']!=SCHEME:raise ValueError('Use a checkpoint from exp2/train_exp2.py')
    cfg=from_saved(data['config']);device=device_for(a.device);actor,critic=models(cfg,device)
    actor.load_state_dict(data['actor_state_dict']);critic.load_state_dict(data['hybrid_critic_state_dict'])
    z=data['collector_state']['persistent_z'].to(device)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    columns=['update','seed','batch_index','completed','total_cost','geometric_distance','episode_length','penalized_cost']
    results={}
    for protocol in ['system','policy']:
        with (out/f'{protocol}_episodes.csv').open('w',newline='') as h,(out/f'{protocol}_events.jsonl').open('w') as eh:
            writer=csv.DictWriter(h,fieldnames=columns);writer.writeheader()
            def sink(e):eh.write(json.dumps(e)+'\n')
            r=validate(cfg,actor,critic,z,device,int(data['update']),protocol,writer,sink)
            results[protocol]={k:v for k,v in r.items() if k!='validation_details'}
    (out/'summary.json').write_text(json.dumps(dict(training_backend=cfg.exp2_backend,
        validation_seed=cfg.validation_seed,validation_seeds=cfg.validation_seeds,results=results),indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
