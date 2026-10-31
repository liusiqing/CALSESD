"""Episode-internal Actor-controller checkpoints. Load trusted files only.

Save/resume only at completed full-batch/update boundaries. The next collect()
MUST perform its seeded full reset before acting. No live environment or geometry
cache is serialized, and this is not a mid-episode continuation checkpoint.

Outer v6 keeps standard weight keys; online SCHEME/ARCHITECTURE and training
state v3 deliberately reject strict resume from old fixed-z checkpoints.
RNG restoration cannot guarantee identical future wall-clock timeouts or bitwise
CUDA results. State getters must not mutate the live controller/model state.
"""

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from numbers import Integral, Real
import math
import os
import random
import tempfile
import warnings
import numpy as np
import torch

FORMAT_VERSION = 6
TRAINING_STATE_VERSION = 3
SCHEME = "online-critic-latent-actor-only-v1"
ARCHITECTURE = "independent-actor-online-controller-critic-ensemble-v1"
VALUE_SEMANTICS = "online_actor_controller_v1"
BOOTSTRAP_SEMANTICS = "online_post_trigger_next_decision_v1"
SUPPORTS_ONLINE_TRIGGER_RESUME = True
SUPPORTS_ROLLOUT_PROVIDER_RESUME = False  # Pure-rollout online search is not in phase one.
_IGNORED_CONFIG = {
    "updates",
    "output_dir",
    "checkpoints_dir",
    "resume",
    "device",
    "save_step_csv",
    "checkpoint_interval",
    "init_checkpoint",
    "latest_checkpoint_interval",
}


def _count(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _quality_history(values):
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("Critic quality history must contain finite numbers")
        result.append(float(value))
    return result


def _best_key(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("best_key must contain completion rate and negative cost")
    rate, score = value
    if isinstance(rate, bool) or not isinstance(rate, Real) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Invalid completion rate in best_key")
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("Invalid cost score in best_key")
    if not math.isfinite(score) and not (score == float("-inf") and rate == 0):
        raise ValueError("Only the initial (0,-inf) best-key sentinel may be nonfinite")
    return float(rate), float(score)


def _rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state().clone().cpu(),
        cuda=[s.clone().cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    )


def _validate_rng(rng):
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    cpu = rng["cpu"]
    if not torch.is_tensor(cpu) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError("Invalid CPU RNG state")
    torch.Generator(device="cpu").set_state(cpu.cpu())
    saved = rng.get("cuda")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if saved is not None and not isinstance(saved, (list, tuple)):
        raise ValueError("Invalid CUDA RNG state list")
    if (0 if saved is None else len(saved)) != count:
        raise ValueError(
            "Strict resume requires the same visible CUDA device count; use --init-checkpoint for migration"
        )
    if saved is not None:
        for index, state in enumerate(saved):
            if not torch.is_tensor(state) or state.dtype != torch.uint8 or state.ndim != 1:
                raise ValueError("Invalid CUDA RNG state")
            torch.Generator(device=torch.device("cuda", index)).set_state(state.cpu())


def _restore_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["cpu"].cpu())
    if rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])


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


def _validate_latent(z, cfg):
    if not torch.is_tensor(z) or z.shape != (cfg.batch_size, cfg.latent_dim):
        raise ValueError("Persistent latent batch/dimension mismatch")
    if not z.is_floating_point() or not bool(torch.isfinite(z).all()):
        raise ValueError("Persistent latent must be finite floating point")
    if bool((z.abs() > cfg.latent_bound).any()):
        raise ValueError("Persistent latent outside configured bounds")


def _class_name(obj):
    cls = type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _check_weights(model, saved, label, *, allow_missing_heads=False, exact_dtype=True):
    if not isinstance(saved, dict):
        raise TypeError(f"{label} weights must be a state_dict")
    target = model.state_dict()
    missing, unexpected = set(target) - set(saved), set(saved) - set(target)
    if unexpected or (missing and not (allow_missing_heads and all(k.startswith("heads.") for k in missing))):
        raise ValueError(f"{label} keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key in set(target) & set(saved):
        src, dst = saved[key], target[key]
        if not torch.is_tensor(src) or src.shape != dst.shape:
            raise ValueError(f"{label} tensor shape mismatch: {key}")
        if exact_dtype and src.dtype != dst.dtype:
            raise ValueError(f"{label} tensor dtype mismatch: {key}")
        if (src.is_floating_point() or src.is_complex()) and not bool(torch.isfinite(src).all()):
            raise ValueError(f"{label} contains nonfinite tensor values: {key}")


def _model_execution(models):
    """Guard real device placement and intentional trainability, not cfg aliases."""
    return [
        dict(
            parameter_devices={name: str(p.device) for name, p in model.named_parameters()},
            buffer_devices={name: str(b.device) for name, b in model.named_buffers()},
            requires_grad={name: bool(p.requires_grad) for name, p in model.named_parameters()},
        )
        for model in models
    ]


def _optimizer_layout(models, optimizers):
    maps = [{id(p): name for name, p in model.named_parameters()} for model in models]
    if set(maps[0]) & set(maps[1]):
        raise ValueError("Actor and Critic must not share parameters")
    result = []
    for index, optimizer in enumerate(optimizers):
        groups, seen = [], set()
        for group in optimizer.param_groups:
            names = []
            for parameter in group["params"]:
                key = id(parameter)
                if key not in maps[index] or key in seen:
                    raise ValueError("Optimizer includes foreign or duplicate model parameters")
                seen.add(key)
                names.append(maps[index][key])
            groups.append(names)
        result.append(dict(class_name=_class_name(optimizer), parameter_groups=groups))
    return result


def _check_optimizer(optimizer, saved):
    if not isinstance(saved, dict):
        raise TypeError("Optimizer state must be a dictionary")
    groups = saved.get("param_groups", [])
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer parameter-group count mismatch")
    for live, group in zip(optimizer.param_groups, groups):
        if len(live["params"]) != len(group["params"]):
            raise ValueError("Optimizer parameter count mismatch")
        for parameter, saved_id in zip(live["params"], group["params"]):
            state = saved.get("state", {}).get(saved_id, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state:
                    moment = state[key]
                    if not torch.is_tensor(moment) or moment.shape != parameter.shape:
                        raise ValueError(f"Optimizer moment shape mismatch: {key}")
                    if not bool(torch.isfinite(moment).all()):
                        raise ValueError(f"Optimizer has nonfinite moments: {key}")


def _preflight_stateful(obj, state, label):
    # Only search/trigger objects; no Actor/Critic is copied here.
    try:
        probe = deepcopy(obj)
        probe.load_state_dict(deepcopy(state))
    except Exception as exc:
        raise ValueError(f"{label} state failed preflight; do not continue training") from exc


def _identity(cfg, collector, models):
    from .ablation_config import resolve_ablation, build_search_config, validation_settings
    from .online_trigger import OnlineTriggerConfig, ONLINE_CONTEXT_FIELDS

    if cfg.search_scope != "episode" or cfg.bootstrap_time_limits:
        raise ValueError("Requires episode-internal finite-task configuration")
    critic = models[1]
    if getattr(critic, "value_semantics", None) != VALUE_SEMANTICS or getattr(
        critic, "online_context_dim", None
    ) != len(ONLINE_CONTEXT_FIELDS):
        raise ValueError("Install the online-context Critic")
    plan = resolve_ablation(cfg)
    attached = getattr(collector, "ablation_plan", None)
    if not is_dataclass(attached) or asdict(attached) != asdict(plan):
        raise ValueError("Collector ablation plan differs from cfg")
    actual = getattr(collector.search, "config", None)
    if not is_dataclass(actual):
        raise TypeError("Search config must be a dataclass")
    expected = build_search_config(cfg, config_cls=type(actual))
    if asdict(expected) != asdict(actual):
        raise ValueError("Collector search config differs from the resolved online ablation")
    trigger = collector.online_trigger
    expected_trigger = OnlineTriggerConfig.from_config(cfg)
    expected_trigger.validate()
    if not is_dataclass(trigger.config) or asdict(trigger.config) != asdict(expected_trigger):
        raise ValueError("Online trigger configuration mismatch")
    if tuple(trigger.context_fields) != tuple(ONLINE_CONTEXT_FIELDS):
        raise ValueError("Online context schema/order mismatch")
    if getattr(collector.env, "online_trigger", None) is not trigger:
        raise ValueError("Environment and collector must own the SAME trigger")
    if trigger.pending:
        raise ValueError("Cannot save/restore while online search is pending")
    return dict(
        family=plan.family,
        variant=plan.variant,
        plan=asdict(plan),
        resolved_search_config=asdict(actual),
        trigger_config=asdict(expected_trigger),
        trigger_class=_class_name(trigger),
        search_class=_class_name(collector.search),
        online_context_fields=list(ONLINE_CONTEXT_FIELDS),
        value_semantics=VALUE_SEMANTICS,
        bootstrap_semantics=BOOTSTRAP_SEMANTICS,
        validation_settings={k: v for k, v in validation_settings(cfg).items() if k not in ("state_source",)},
    )


def _stack_detached(name, values):
    if not values or any(not torch.is_tensor(x) or x.requires_grad for x in values):
        raise ValueError(f"{name} must be a nonempty sequence of detached tensors")
    return torch.stack(values).detach().cpu()


def _archive_payload(rollout, collector, cfg, update):
    if (
        getattr(rollout, "value_semantics", None) != VALUE_SEMANTICS
        or getattr(rollout, "bootstrap_semantics", None) != BOOTSTRAP_SEMANTICS
    ):
        raise ValueError("Wrong collection/bootstrap semantics")
    _validate_latent(collector.z, cfg)
    T = len(rollout.states)
    if not T:
        raise ValueError("Cannot archive an empty rollout")
    names = (
        "valid_masks",
        "actor_masks",
        "boundaries",
        "latents",
        "latent_switch_masks",
        "search_due_masks",
        "values",
        "next_values",
        "rewards",
    )
    if any(len(getattr(rollout, name)) != T for name in names):
        raise ValueError("Online rollout lists have different lengths")
    contexts, steps, active = [], [], []
    for state in rollout.states:
        if getattr(state, "online_search_pending", None) is not False:
            raise ValueError("Archive states must be captured AFTER search completion")
        contexts.append(getattr(state, "online_context", None))
        steps.append(getattr(state, "episode_steps", None))
        active.append(getattr(state, "active_mask", None))
    archive = dict(
        latent_sequence=_stack_detached("latents", rollout.latents),
        valid_masks=_stack_detached("valid masks", rollout.valid_masks),
        actor_sample_masks=_stack_detached("actor masks", rollout.actor_masks),
        boundaries=_stack_detached("boundaries", rollout.boundaries),
        latent_switch_masks=_stack_detached("switch masks", rollout.latent_switch_masks),
        search_due_masks=_stack_detached("due masks", rollout.search_due_masks),
        online_context=_stack_detached("online context", contexts),
        episode_steps=_stack_detached("episode steps", steps),
        state_active_masks=_stack_detached("state active masks", active),
        values=_stack_detached("values", rollout.values),
        next_values=_stack_detached("next values", rollout.next_values),
        rewards=_stack_detached("rewards", rollout.rewards),
        ppo_env_steps=_count("PPO env steps", rollout.ppo_env_steps),
        search_calls_in_batch=_count("batch search calls", rollout.evolution_search_calls),
        source_update=update,
        collection_stage="before_ppo_update",
        value_semantics=VALUE_SEMANTICS,
        bootstrap_semantics=BOOTSTRAP_SEMANTICS,
    )
    valid = archive["valid_masks"]
    if valid.ndim != 2 or valid.shape[1] != cfg.batch_size:
        raise ValueError("Invalid valid-mask shape")
    if archive["latent_sequence"].shape != (T, cfg.batch_size, cfg.latent_dim):
        raise ValueError("Invalid latent archive shape")
    lengths = valid.long().sum(0)
    if not bool((lengths > 0).all()):
        raise ValueError("Cannot archive an empty episode row")
    rows = torch.arange(cfg.batch_size)
    archive["z_initial"] = archive["latent_sequence"][0].clone()
    archive["z_final"] = archive["latent_sequence"][lengths - 1, rows].clone()
    done = getattr(collector.env, "dones", None)
    if not torch.is_tensor(done) or done.shape != (cfg.batch_size,) or done.dtype != torch.bool or not bool(done.all()):
        raise ValueError("Save only after ALL batch rows complete or truncate")
    env_steps = getattr(collector.env, "steps", None)
    if not torch.is_tensor(env_steps) or not torch.equal(env_steps.detach().cpu(), lengths):
        raise ValueError("Environment episode lengths differ from the rollout")
    return archive


def _check_archive(archive, cfg, persistent_z, update):
    """Audit saved sequence, not an old episode-fixed-z assumption."""
    if (
        archive.get("source_update") != update
        or archive.get("value_semantics") != VALUE_SEMANTICS
        or archive.get("bootstrap_semantics") != BOOTSTRAP_SEMANTICS
    ):
        raise ValueError("Archive update/value/bootstrap identity mismatch")
    valid = archive["valid_masks"]
    if not torch.is_tensor(valid) or valid.ndim != 2 or valid.size(0) < 1 or valid.shape[1] != cfg.batch_size:
        raise ValueError("Invalid archived valid masks")
    T, B = valid.shape
    masks = {
        name: archive[name]
        for name in (
            "actor_sample_masks",
            "boundaries",
            "latent_switch_masks",
            "search_due_masks",
            "state_active_masks",
        )
    }
    if valid.dtype != torch.bool:
        raise ValueError("Valid masks must be bool")
    for name, x in masks.items():
        if not torch.is_tensor(x) or x.shape != valid.shape or x.dtype != torch.bool:
            raise ValueError(f"Invalid archived mask: {name}")
    actor = masks["actor_sample_masks"]
    bounds = masks["boundaries"]
    switched = masks["latent_switch_masks"]
    due = masks["search_due_masks"]
    if not torch.equal(actor, valid) or not torch.equal(masks["state_active_masks"], valid):
        raise ValueError("Actor/state valid-mask mismatch")
    if not bool(valid[0].all()) or bool((valid[-1] & ~bounds[-1]).any()):
        raise ValueError("Archive must contain full-batch finite episodes")
    if T > 1 and not torch.equal(valid[1:], valid[:-1] & ~bounds[:-1]):
        raise ValueError("Episode rows reset or disappeared inside the archive")
    if bool((due & ~valid).any()) or bool((switched & ~due).any()):
        raise ValueError("Invalid archived trigger/switch mask")
    if bool(due[0].any()) or bool(switched[0].any()):
        raise ValueError("Periodic phase must not search at reset")
    z = archive["latent_sequence"]
    if (
        not torch.is_tensor(z)
        or z.shape != (T, B, cfg.latent_dim)
        or not z.is_floating_point()
        or not bool(torch.isfinite(z).all())
        or bool((z.abs() > cfg.latent_bound).any())
    ):
        raise ValueError("Invalid archived latent sequence")
    if T > 1 and not torch.equal((z[1:] != z[:-1]).any(-1), switched[1:]):
        raise ValueError("Latent changes do not match actual switch masks")
    lengths = valid.long().sum(0)
    rows = torch.arange(B)
    final = z[lengths - 1, rows]
    persistent = persistent_z.detach().cpu()
    if not torch.equal(archive["z_initial"], z[0]) or not torch.equal(archive["z_final"], final):
        raise ValueError("Initial/final latent archive mismatch")
    if persistent.dtype != z.dtype or not torch.equal(final, persistent):
        raise ValueError("Persistent z differs from each row final actual latent")
    context = archive["online_context"]
    steps = archive["episode_steps"]
    if (
        not torch.is_tensor(context)
        or context.shape != (T, B, 6)
        or not context.is_floating_point()
        or not bool(torch.isfinite(context).all())
        or bool(((context < -1e-6) | (context > 1.0 + 1e-6)).any())
    ):
        raise ValueError("Invalid archived online context")
    if not torch.is_tensor(steps) or steps.shape != valid.shape or steps.dtype != torch.long:
        raise ValueError("Invalid archived episode steps")
    expected = torch.arange(T)[:, None].expand(T, B)
    if not torch.equal(steps[valid], expected[valid]):
        raise ValueError("Decision step history does not match full-episode collection")
    for name in ("values", "next_values", "rewards"):
        x = archive[name]
        if (
            not torch.is_tensor(x)
            or x.shape != valid.shape
            or not x.is_floating_point()
            or not bool(torch.isfinite(x).all())
        ):
            raise ValueError(f"Invalid archived {name}")
    values, nxt = archive["values"], archive["next_values"]
    if bool((nxt[valid & bounds] != 0).any()):
        raise ValueError("Terminal finite tasks must bootstrap zero")
    if T > 1:
        continuing = valid[:-1] & ~bounds[:-1]
        if not torch.equal(nxt[:-1][continuing], values[1:][continuing]):
            raise ValueError("Archive bootstrap is not the actual next post-trigger value")
    if _count("archived PPO steps", archive["ppo_env_steps"]) != int(valid.sum()):
        raise ValueError("Archived PPO step count mismatch")
    if _count("archived batch search calls", archive["search_calls_in_batch"]) != int(due.any(1).sum()):
        raise ValueError("Search calls do not match archived due decisions")
    return lengths


def _check_links(cfg, update, critic_updates, search_state, trigger_state, archive, lengths):
    if critic_updates != update:
        raise ValueError("critic_updates must count completed outer PPO updates")
    if search_state.get("format_version") != 3 or search_state.get("value_semantics") != VALUE_SEMANTICS:
        raise ValueError("Requires online search state v3")
    calls = _count("total search calls", search_state["calls"])
    local_calls = _count("trigger attempt number", trigger_state["attempt_number"])
    if local_calls != archive["search_calls_in_batch"] or calls < local_calls:
        raise ValueError("Global/batch search count mismatch")
    if trigger_state.get("training") is not True or trigger_state.get("completed_updates") != update - 1:
        raise ValueError("Trigger must describe the just-collected training episode")
    attempts = torch.as_tensor(trigger_state["attempts"], dtype=torch.long)
    due = archive["search_due_masks"]
    switched = archive["latent_switch_masks"]
    steps = archive["episode_steps"]
    if attempts.shape != (cfg.batch_size,) or not torch.equal(attempts, due.long().sum(0)):
        raise ValueError("Trigger per-row attempts differ from rollout due masks")
    if list(trigger_state["seen_steps"]) != lengths.tolist():
        raise ValueError("Trigger observed final steps differ from episode lengths")
    for b in range(cfg.batch_size):
        due_steps = steps[:, b][due[:, b]]
        switch_steps = steps[:, b][switched[:, b]]
        expected_attempt = int(due_steps[-1]) if due_steps.numel() else 0
        expected_switch = int(switch_steps[-1]) if switch_steps.numel() else 0
        if (
            trigger_state["last_attempt_step"][b] != expected_attempt
            or trigger_state["last_switch_step"][b] != expected_switch
        ):
            raise ValueError("Trigger last-attempt/switch steps differ from trajectory")


def _clear_runtime(collector, critic):
    collector.enhancement_events.clear()
    collector.last_search_status = {}
    collector.last_search_diagnostics = {}
    clear = getattr(critic, "clear_static_cache", None)
    if callable(clear):
        clear()


def save_checkpoint(path, cfg, models, optimizers, update, best_key, rollout, collector, quality_history=()):
    if len(models) != 2 or len(optimizers) != 2:
        raise ValueError("Expected Actor/Critic and two optimizers")
    update = _count("update", update)
    if update < 1:
        raise ValueError("Checkpoint requires a completed update")
    entry_rng = _rng_state()
    try:
        identity = _identity(cfg, collector, models)
        archive = _archive_payload(rollout, collector, cfg, update)
        lengths = _check_archive(archive, cfg, collector.z, update)
        search_state = deepcopy(collector.search.state_dict())
        trigger_state = deepcopy(collector.online_trigger.state_dict())
        if search_state.get("config") != identity["resolved_search_config"]:
            raise ValueError("Search config/state mismatch")
        critic_updates = _count("critic_updates", collector.critic_updates)
        _check_links(cfg, update, critic_updates, search_state, trigger_state, archive, lengths)
        _preflight_stateful(collector.search, search_state, "Search")
        _preflight_stateful(collector.online_trigger, trigger_state, "Online trigger")
        step = _count("global_step", collector.global_step)
        if step < archive["ppo_env_steps"]:
            raise ValueError("global_step is smaller than current PPO batch")
        actor, critic = models
        data = dict(
            format_version=FORMAT_VERSION,
            training_state_version=TRAINING_STATE_VERSION,
            architecture=ARCHITECTURE,
            scheme=SCHEME,
            config=asdict(cfg),
            update=update,
            ablation=identity,
            actor_state_dict=actor.state_dict(),
            hybrid_critic_state_dict=critic.state_dict(),
            optimizer_states=[o.state_dict() for o in optimizers],
            optimizer_layout=_optimizer_layout(models, optimizers),
            model_execution=_model_execution(models),
            best_key=_best_key(best_key),
            search_state=search_state,
            online_trigger_state=trigger_state,
            collector_state=dict(
                persistent_z=collector.z.detach().cpu().clone(), global_step=step, critic_updates=critic_updates
            ),
            critic_quality=dict(
                updates=critic_updates, history=_quality_history(() if quality_history is None else quality_history)
            ),
            latent_archive=archive,
            rng=entry_rng,
        )
        _atomic_save(data, path)
    finally:
        _restore_rng(entry_rng)


def load_checkpoint(path, cfg, models, optimizers, collector, device):
    if len(models) != 2 or len(optimizers) != 2:
        raise ValueError("Expected Actor/Critic and two optimizers")
    entry_rng = _rng_state()
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if (
            data.get("format_version") != FORMAT_VERSION
            or data.get("training_state_version") != TRAINING_STATE_VERSION
            or data.get("scheme") != SCHEME
            or data.get("architecture") != ARCHITECTURE
        ):
            raise ValueError(
                "Strict resume requires an online checkpoint: outer v6/training-state v3; old fixed-z runs are not continuation checkpoints"
            )
        if len(data["optimizer_states"]) != 2:
            raise ValueError("Optimizer count mismatch")
        live = {k: v for k, v in asdict(cfg).items() if k not in _IGNORED_CONFIG}
        saved = {k: v for k, v in data["config"].items() if k not in _IGNORED_CONFIG}
        if live != saved:
            keys = sorted(k for k in set(live) | set(saved) if k not in live or k not in saved or live[k] != saved[k])
            raise ValueError(f"Resume configuration mismatch: {keys}")
        if data.get("ablation") != _identity(cfg, collector, models):
            raise ValueError("Online ablation/controller identity mismatch")
        update = _count("saved update", data["update"])
        if update < 1 or cfg.updates < update:
            raise ValueError("Invalid saved/requested update count")
        state = data["collector_state"]
        z = state["persistent_z"].to(device)
        _validate_latent(z, cfg)
        step = _count("global_step", state["global_step"])
        critic_updates = _count("critic_updates", state["critic_updates"])
        lengths = _check_archive(data["latent_archive"], cfg, z, update)
        if step < data["latent_archive"]["ppo_env_steps"]:
            raise ValueError("Invalid global step")
        _check_links(
            cfg,
            update,
            critic_updates,
            data["search_state"],
            data["online_trigger_state"],
            data["latent_archive"],
            lengths,
        )
        history = _quality_history(data.get("critic_quality", {}).get("history", []))
        if _count("quality updates", data.get("critic_quality", {}).get("updates")) != critic_updates:
            raise ValueError("Critic diagnostic counter mismatch")
        if data.get("model_execution") != _model_execution(models):
            raise ValueError("Model device/trainability changed; use weight migration")
        if data.get("optimizer_layout") != _optimizer_layout(models, optimizers):
            raise ValueError("Optimizer type/parameter order changed")
        for model, key in zip(models, ("actor_state_dict", "hybrid_critic_state_dict")):
            _check_weights(model, data[key], key)
        for optimizer, state_dict in zip(optimizers, data["optimizer_states"]):
            _check_optimizer(optimizer, state_dict)
        key = _best_key(data["best_key"])
        _validate_rng(data["rng"])
        _preflight_stateful(collector.search, data["search_state"], "Search")
        _preflight_stateful(collector.online_trigger, data["online_trigger_state"], "Online trigger")
        # External loads may still fail: abort on error; do not continue with a
        # partially restored model. RNG is restored on both success and failure.
        for model, name in zip(models, ("actor_state_dict", "hybrid_critic_state_dict")):
            model.load_state_dict(data[name], strict=True)
        for optimizer, state_dict in zip(optimizers, data["optimizer_states"]):
            optimizer.load_state_dict(state_dict)
        collector.search.load_state_dict(data["search_state"])
        collector.online_trigger.load_state_dict(data["online_trigger_state"])
        collector.z = z.detach().clone()
        collector.global_step = step
        collector.critic_updates = critic_updates
        _clear_runtime(collector, models[1])
    except Exception:
        _restore_rng(entry_rng)
        raise
    _restore_rng(data["rng"])
    return update + 1, key, history


def initialize_from_checkpoint(path, models, collector, device):
    """Fresh experiment: matching weights and optional final z, not strict resume.

    Critic input/head shapes must match. Old context/head tensors are NOT silently
    dropped or padded. Use from-scratch online training for the main ablation.
    """
    if len(models) != 2:
        raise ValueError("Expected Actor/Critic")
    if (
        _count("global_step", collector.global_step)
        or _count("critic_updates", collector.critic_updates)
        or _count("search calls", collector.search.calls)
        or collector.online_trigger.pending
    ):
        raise ValueError("Weight migration requires a fresh collector/new optimizers")
    fresh = collector.online_trigger.state_dict()
    if fresh["attempt_number"] or any(fresh["attempts"]) or fresh["batch_seconds_used"]:
        raise ValueError("Weight migration requires a fresh trigger")
    entry_rng = _rng_state()
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("scheme") != SCHEME:
            warnings.warn(
                "Different controller/value semantics: only shape-compatible weights can initialize this online model.",
                RuntimeWarning,
            )
        actor, critic = models
        a = data.get("actor_state_dict", data.get("policy_state_dict"))
        v = data.get("hybrid_critic_state_dict", data.get("hybrid_value_state_dict"))
        if a is None or v is None:
            raise ValueError("Checkpoint must contain Actor/Critic weights")
        _check_weights(actor, a, "Actor migration", exact_dtype=False)
        _check_weights(critic, v, "Online Critic migration", allow_missing_heads=True, exact_dtype=False)
        z = data.get("collector_state", {}).get("persistent_z")
        if z is not None:
            z = z.to(device)
            _validate_latent(z, collector.cfg)
        identity = _identity(collector.cfg, collector, models)
        origin = data.get("ablation")
        if origin is not None and (origin.get("family"), origin.get("variant")) != (
            identity["family"],
            identity["variant"],
        ):
            warnings.warn("Cross-ablation warm start is not a from-scratch comparison.", RuntimeWarning)
        actor.load_state_dict(a, strict=True)
        critic.load_state_dict(v, strict=False)
        collector.z = None if z is None else z.detach().clone()
        collector.global_step = 0
        collector.critic_updates = 0
        collector.search.reset()
        collector.online_trigger.reset_episode(completed_updates=0, training=True)
        _clear_runtime(collector, critic)
    finally:
        _restore_rng(entry_rng)


__all__ = [
    "FORMAT_VERSION",
    "TRAINING_STATE_VERSION",
    "SCHEME",
    "ARCHITECTURE",
    "VALUE_SEMANTICS",
    "BOOTSTRAP_SEMANTICS",
    "SUPPORTS_ONLINE_TRIGGER_RESUME",
    "SUPPORTS_ROLLOUT_PROVIDER_RESUME",
    "save_checkpoint",
    "load_checkpoint",
    "initialize_from_checkpoint",
]
