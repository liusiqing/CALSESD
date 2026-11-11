"""直接使用原 validation 的实例/z/搜索种子，按训练后端执行 system 验证。"""
from contextlib import contextmanager
from dataclasses import replace
import CALSESD.training.validation as original
from CALSESD.models.hybrid_latent_search import HybridLatentSearch
from .search import RolloutSearch,Exp2Collector,search_config

# 核心模块：仅在串行验证调用内替换依赖，退出恢复；不要并发使用此适配器。
@contextmanager
def adapter(cfg):
    saved=(original.HybridLatentSearch,original.ActorOnlyCollector,original.build_search_config)
    try:
        original.HybridLatentSearch=RolloutSearch if cfg.exp2_backend=='rollout' else HybridLatentSearch
        original.ActorOnlyCollector=Exp2Collector
        original.build_search_config=search_config
        yield
    finally:
        original.HybridLatentSearch,original.ActorOnlyCollector,original.build_search_config=saved

def validate(cfg,actor,critic,z,device,update,protocol='system',episode_writer=None,sink=None):
    with adapter(cfg):
        result=original.validate_controller(cfg,actor,critic,z,device,update=update,protocol=protocol,
                      episode_writer=episode_writer,diagnostics_sink=sink)
    details=result['validation_details']
    details.update(ablation_family='exp2',ablation_variant=cfg.exp2_backend)
    return result
