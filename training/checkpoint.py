"""断点保存/恢复:仅在完整批量回合(update 边界)保存;恢复后下一次 collect() 会做带种子的完整 reset。
只加载可信文件。"""
from dataclasses import asdict
from pathlib import Path
import os
import random
import tempfile
import numpy as np
import torch

FORMAT_VERSION = 6
TRAINING_STATE_VERSION = 3
SCHEME = "online-critic-latent-actor-only-v1"
ARCHITECTURE = "independent-actor-online-controller-critic-ensemble-v1"
_IGNORED_CONFIG = {"updates", "output_dir", "checkpoints_dir", "resume", "device", "save_step_csv",
                   "checkpoint_interval", "init_checkpoint", "latest_checkpoint_interval"}


# 核心模块:随机数状态(Python / NumPy / torch CPU / 所有 CUDA 设备)的保存与恢复。
def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state().clone().cpu(),
                cuda=[s.clone().cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None)


def _restore_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["cpu"].cpu())
    if rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])


# 核心模块:原子写入,先写临时文件再替换,避免中途中断留下损坏的 checkpoint。
def _atomic_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _clear_runtime(collector, critic):
    collector.enhancement_events.clear()
    collector.last_search_status = {}
    collector.last_search_diagnostics = {}
    critic.clear_static_cache()


# 核心模块:保存续训所需的全部状态:模型、优化器、搜索器(calls/EMA)、触发器、持久化隐变量 z、全局计数与 RNG。
# rollout 参数仅为兼容 train_ppo 的调用签名,不再归档轨迹。
def save_checkpoint(path, cfg, models, optimizers, update, best_key, rollout, collector, quality_history=()):
    actor, critic = models
    _atomic_save(dict(
        format_version=FORMAT_VERSION, training_state_version=TRAINING_STATE_VERSION,
        architecture=ARCHITECTURE, scheme=SCHEME, config=asdict(cfg), update=int(update),
        actor_state_dict=actor.state_dict(), hybrid_critic_state_dict=critic.state_dict(),
        optimizer_states=[o.state_dict() for o in optimizers],
        best_key=(float(best_key[0]), float(best_key[1])),
        search_state=collector.search.state_dict(), online_trigger_state=collector.online_trigger.state_dict(),
        collector_state=dict(persistent_z=collector.z.detach().cpu().clone(),
                             global_step=int(collector.global_step), critic_updates=int(collector.critic_updates)),
        critic_quality=dict(updates=int(collector.critic_updates), history=[float(v) for v in quality_history]),
        rng=_rng_state()), path)


# 核心模块:严格续训。返回 (下一个 update 编号, best_key, Critic 质量历史)。
# 仅保留两项检查:格式版本(旧的固定 z checkpoint 不能续训)与配置一致性(避免换超参数后静默续训)。
def load_checkpoint(path, cfg, models, optimizers, collector, device):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format_version") != FORMAT_VERSION or data.get("scheme") != SCHEME:
        raise ValueError("Strict resume requires an online checkpoint (outer v6)")
    live = {k: v for k, v in asdict(cfg).items() if k not in _IGNORED_CONFIG}
    saved = {k: v for k, v in data["config"].items() if k not in _IGNORED_CONFIG}
    if live != saved:
        keys = sorted(k for k in set(live) | set(saved) if live.get(k) != saved.get(k))
        raise ValueError(f"Resume configuration mismatch: {keys}")
    actor, critic = models
    actor.load_state_dict(data["actor_state_dict"])
    critic.load_state_dict(data["hybrid_critic_state_dict"])
    for optimizer, state in zip(optimizers, data["optimizer_states"]):
        optimizer.load_state_dict(state)
    state = data["collector_state"]
    collector.z = state["persistent_z"].to(device)
    collector.global_step = int(state["global_step"])
    collector.critic_updates = int(state["critic_updates"])
    collector.search.load_state_dict(data["search_state"])
    collector.online_trigger.load_state_dict(data["online_trigger_state"])
    _clear_runtime(collector, critic)
    _restore_rng(data["rng"])
    return int(data["update"]) + 1, tuple(data["best_key"]), list(data["critic_quality"]["history"])


# 核心模块:只迁移网络权重的热启动(不恢复优化器、计数、RNG)。
# Critic 用 strict=False,允许缺少 heads.*(改变 n_critics 时),缺失的头保持随机初始化。
def initialize_from_checkpoint(path, models, collector, device):
    data = torch.load(path, map_location="cpu", weights_only=False)
    actor, critic = models
    actor.load_state_dict(data["actor_state_dict"])
    critic.load_state_dict(data["hybrid_critic_state_dict"], strict=False)
    _clear_runtime(collector, critic)
