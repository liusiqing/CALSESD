"""在项目 PyTorch/PyG 环境运行：python -m unittest CALSESD.exp2.test_exp2 -v。"""
import unittest
from dataclasses import replace
import torch
from CALSESD.exp2.config import Exp2Config
from CALSESD.exp2.common import models
from CALSESD.exp2.evaluator import RolloutEvaluator
from CALSESD.exp2.scenario import seed_for
from CALSESD.training.env_factory import make_env
from CALSESD.training.rng_state import preserve_rng

class Exp2Tests(unittest.TestCase):
    def test_seed(self):
        self.assertEqual(seed_for(1,2,3,'demand',4),seed_for(1,2,3,'demand',4))
        self.assertNotEqual(seed_for(1,2,3,'demand',4),seed_for(1,2,3,'demand',5))

    def test_shadow_isolation_and_crn(self):
        cfg=Exp2Config(grid_size=4,num_customers=3,batch_size=2,max_episode_steps=15,
                       hidden_dim=16,dynamic_dim=8,num_heads=2,static_gat_layers=1,
                       static_transformer_layers=1,latent_dim=4,exp2_chunk=1)
        with preserve_rng():
            env=make_env(cfg,7,'cpu');actor,_=models(cfg,torch.device('cpu'));actor.eval()
            evaluator=RolloutEvaluator(env,actor,cfg)
            rows=torch.tensor([0,1]);population=torch.zeros(2,4)
            snap=env.snapshot()
            a=evaluator.shadow(rows,1,11,1,0)
            b=evaluator.shadow(torch.tensor([1]),1,11,1,0)
            self.assertTrue(torch.equal(a.actual_demands_tensor[1],b.actual_demands_tensor[0]))
            first=evaluator.population(rows,population,11,1)
            evaluator.cfg=replace(cfg,exp2_chunk=2)
            second=evaluator.population(rows,population,11,1)
            self.assertTrue(torch.allclose(first,second,rtol=1e-5,atol=1e-5))
            for name in env._SNAPSHOT_TENSORS:
                self.assertTrue(torch.equal(getattr(env,name),snap[name]),name)
            self.assertTrue((env.env.distance_matrix==snap['env']['distance_matrix']).all())
            fresh=evaluator.shadow(rows,1,11,1,0).actual_demands_tensor.clone()
            env.actual_demands_tensor[~env.revealed_masks]=0
            again=evaluator.shadow(rows,1,11,1,0).actual_demands_tensor
            self.assertTrue(torch.equal(fresh,again))

if __name__=='__main__':unittest.main()
