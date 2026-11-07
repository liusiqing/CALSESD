"""Actor-only PPO 入口:回合内周期 Critic 引导隐空间搜索(仅 Critic 评分)。
搜索发生在 OnlineTrigger 决策点、真实动作之前。"""
from __future__ import annotations
import json
import math
import random
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CALSESD.models import GeometryDynamicPointerPolicy
from CALSESD.models.hybrid_latent_search import HybridLatentSearch
from CALSESD.models.value import ValueNetwork
from CALSESD.training.ppo_config import parse_args
from CALSESD.training.online_trigger import OnlineTrigger
from CALSESD.training.ablation_config import resolve_ablation, ablation_manifest, build_search_config
from CALSESD.training.env_factory import make_env
from CALSESD.training.ppo_rollout import ActorOnlyCollector
import CALSESD.training.ppo_update as update_io
import CALSESD.training.validation as validation_io
import CALSESD.training.checkpoint as checkpoint_io
import CALSESD.training.compact_logging as logging_io

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _seed_everything(seed):
    seed = int(seed); random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(value):
    d = torch.device('cuda' if value == 'auto' and torch.cuda.is_available() else 'cpu' if value == 'auto' else value)
    if d.type == 'cuda' and d.index is None:
        d = torch.device('cuda', torch.cuda.current_device())
    return d


def _checkpoint_path(value):
    return Path(value).expanduser().resolve() if value else None


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _output_base(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _drain_search_events(logs, collector):
    for event in collector.enhancement_events:
        logs.event(event)
    collector.enhancement_events.clear()


# 核心模块:验证结果排序键。完成率优先,其次惩罚成本越低越好。
def _validation_key(values):
    return float(values['completion_rate']), -float(values['mean_penalized_cost'])


# 核心模块:网络缓存开关(Actor 静态融合缓存、Critic 几何缓存)。
def _configure_networks(cfg, actor, critic):
    actor.cache_static_fusion = cfg.actor_cache_static_fusion
    critic.configure_static_cache(enabled=cfg.critic_cache_static, max_entries=cfg.critic_static_cache_size)


def main():
    cfg = parse_args()
    plan = resolve_ablation(cfg)
    search_config = build_search_config(cfg)
    manifest = ablation_manifest(cfg, search_config)
    resume_path, init_path = _checkpoint_path(cfg.resume), _checkpoint_path(cfg.init_checkpoint)
    _seed_everything(cfg.seed); device = _device(cfg.device)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir = _output_base(cfg.output_dir) / stamp
    ckpt_dir = _output_base(cfg.checkpoints_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=False); ckpt_dir.mkdir(parents=True, exist_ok=False)

    # 核心模块:构造环境、Actor、Critic 集成、优化器与搜索器
    env = make_env(cfg, cfg.seed, device)
    trigger = OnlineTrigger.from_config(cfg); env.set_online_trigger(trigger)
    actor = GeometryDynamicPointerPolicy(
        dynamic_input_dim=6, geometry_input_dim=2, hidden_dim=cfg.hidden_dim, dynamic_dim=cfg.dynamic_dim,
        latent_dim=cfg.latent_dim, num_heads=cfg.num_heads, static_gat_layers=cfg.static_gat_layers,
        static_transformer_layers=cfg.static_transformer_layers, dropout=cfg.dropout).to(device)
    critic = ValueNetwork(hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim, n_critics=cfg.n_critics,
                          prior_scale=cfg.critic_prior_scale).to(device)
    _configure_networks(cfg, actor, critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.value_learning_rate)
    search = HybridLatentSearch(search_config)
    models, optimizers = (actor, critic), (actor_opt, critic_opt)
    _write_json(out_dir / 'config.json', asdict(cfg)); _write_json(out_dir / 'online_protocol.json', manifest)

    with logging_io.LogBundle(out_dir, debug=cfg.save_step_csv) as logs:
        logs.event(dict(type='ablation_setup', plan={**asdict(plan), 'search_scope': 'episode'},
                        resolved_search_config=asdict(search_config)))
        collector = ActorOnlyCollector(cfg, env, actor, critic, search, logs.episode_csv, logs.step_csv,
                                       online_trigger=trigger)
        best = (0., float('-inf')); start = 1; history = []; origin = 'from_scratch'
        if resume_path is not None:
            start, best, history = checkpoint_io.load_checkpoint(resume_path, cfg, models, optimizers, collector, device)
            origin = 'strict_resume'
        elif init_path is not None:
            checkpoint_io.initialize_from_checkpoint(init_path, models, collector, device)
            origin = 'weight_warm_start'
            print('WARNING: init_checkpoint is a warm-start, not a scratch ablation run', flush=True)
        info = dict(initialization=origin, seed=cfg.seed, device=str(device), search_scope='episode',
                    start_update=start, total_updates=cfg.updates, global_step=collector.global_step,
                    search_calls=search.calls, ablation_family=plan.family, ablation_variant=plan.variant,
                    checkpoint_version=checkpoint_io.FORMAT_VERSION,
                    checkpoint_training_state_version=checkpoint_io.TRAINING_STATE_VERSION)
        _write_json(out_dir / 'run.json', info); logs.event(dict(type='training_start', **info))

        for update in range(start, cfg.updates + 1):
            timer = logging_io.UpdateTimer(lambda: _synchronize(device))
            try:
                # 核心模块 1:完整批量回合采样(回合内按触发器搜索)
                with timer.phase('collect'):
                    r = collector.collect(update)
                events = [e for e in collector.enhancement_events if e.get('type') == 'joint_latent_search']
                search_stats = dict(
                    search_calls=r.evolution_search_calls, search_due_rows=r.due_rows,
                    latent_switch_rows=r.switched_rows,
                    search_timeouts=sum(bool(e['timed_out']) for e in events),
                    batch_search_seconds=float(trigger.batch_seconds_used))
                _drain_search_events(logs, collector)

                # 核心模块 2:Actor PPO + Critic 集成更新
                with timer.phase('ppo'):
                    metrics = update_io.update_actor_critic(cfg, actor, critic, optimizers, r, device)
                metrics.update(
                    search_stats, search_scope='episode', ablation_family=plan.family, ablation_variant=plan.variant,
                    ppo_env_steps=r.ppo_env_steps, latent_search_seconds=r.latent_seconds,
                    predicted_gain=r.predicted_gain, ensemble_disagreement=r.ensemble_disagreement,
                    mutation_sigma=r.mutation_sigma, latent_gradient_norm=r.latent_gradient_norm)
                quality = float(metrics.get('explained_variance', float('nan')))
                if math.isfinite(quality):
                    history.append(quality)
                history = history[-10:]; collector.critic_updates += 1
                logs.event(dict(type='ablation_training_update', update=update, family=plan.family,
                                variant=plan.variant, search_scope='episode', schedule='periodic', **search_stats,
                                ppo_env_steps=r.ppo_env_steps, row_attempts=list(trigger.attempts),
                                search_status=deepcopy(collector.last_search_status)))

                # 核心模块 3:周期验证(policy / system,可选配对比较)
                validation = key = None
                if update % cfg.validation_interval == 0:
                    with timer.phase('validation'):
                        validation = validation_io.validate_controller(
                            cfg, actor, critic, collector.z, device, episode_writer=logs.validation_episode_csv,
                            update=update, protocol=cfg.validation_protocol, diagnostics_sink=logs.event)
                        paired = None
                        if cfg.validation_paired and cfg.validation_protocol == 'system':
                            paired = validation_io.validate_controller(
                                cfg, actor, critic, collector.z, device, episode_writer=None, update=update,
                                protocol='policy', diagnostics_sink=None)
                    key = _validation_key(validation)
                    logs.validation_csv.writerow(
                        logging_io.build_validation_row(update, validation, timer.seconds['validation']))
                    if paired is not None:
                        sys_cost, pol_cost = validation.get('mean_penalized_cost'), paired.get('mean_penalized_cost')
                        logs.event(dict(
                            type='paired_validation', update=update, system_penalized_cost=sys_cost,
                            policy_penalized_cost=pol_cost,
                            search_gain=(pol_cost - sys_cost) if None not in (sys_cost, pol_cost) else None,
                            system_completion_rate=validation.get('completion_rate'),
                            policy_completion_rate=paired.get('completion_rate'),
                            latent_source=validation['validation_details'].get('latent_source')))

                # 核心模块 4:checkpoint(best / latest / 周期)
                with timer.phase('checkpoint'):
                    if validation is not None and key > best:
                        best = key
                        checkpoint_io.save_checkpoint(ckpt_dir / 'checkpoint_best.pt', cfg, models, optimizers,
                                                      update, best, r, collector, history)
                    if update % cfg.latest_checkpoint_interval == 0 or update == cfg.updates:
                        checkpoint_io.save_checkpoint(ckpt_dir / 'checkpoint_latest.pt', cfg, models, optimizers,
                                                      update, best, r, collector, history)
                    if update % cfg.checkpoint_interval == 0:
                        checkpoint_io.save_checkpoint(ckpt_dir / f'checkpoint_update_{update:05d}.pt', cfg, models,
                                                      optimizers, update, best, r, collector, history)

                row = logging_io.build_training_row(
                    update=update, global_step=collector.global_step, batch_size=cfg.batch_size,
                    completed_costs=r.completed_costs, geometric_distances=r.geometric_distances.detach().cpu().tolist(),
                    completed_mask=r.completed_mask.detach().cpu().bool().tolist(), metrics=metrics,
                    update_seconds=timer.elapsed(), phase_seconds=timer.seconds)
                logs.training_csv.writerow(row)
                logs.event(dict(type='update_timing', update=update, **timer.seconds))
                logs.event(dict(type='inference_timing', update=update, batch_size=cfg.batch_size, **r.inference_metrics))
                if logs.debug:
                    logs.event(dict(type='debug_update', update=update, metrics=metrics, actor_samples=r.ppo_env_steps))
                logs.flush()
                cost = row['mean_completed_cost']; cost_text = f'{cost:.3f}' if cost is not None else 'NA'
                print(f'update={update:04d} ablation={plan.family}/{plan.variant} complete={row["completion_rate"]:.1%} '
                      f'cost={cost_text} steps={r.ppo_env_steps} search={search_stats["search_calls"]} '
                      f'switch={search_stats["latent_switch_rows"]} timeout={search_stats["search_timeouts"]} '
                      f'sec={row["update_seconds"]:.2f}', flush=True)
            except Exception as exc:
                try:
                    _drain_search_events(logs, collector)
                    logs.event(dict(type='training_failed', update=update, ablation_family=plan.family,
                                    ablation_variant=plan.variant, global_step=collector.global_step,
                                    error_type=type(exc).__name__, error=str(exc),
                                    search_status=deepcopy(collector.last_search_status)))
                    logs.flush()
                except Exception as log_exc:
                    print(f'Could not fully log failure: {log_exc}', file=sys.stderr, flush=True)
                raise


if __name__ == '__main__':
    main()
