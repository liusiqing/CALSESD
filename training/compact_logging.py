"""训练/验证/搜索三张 CSV 与一份 JSONL 诊断日志(无 Torch 依赖)。
所有行在数据齐备后直接写入:训练行在 PPO 更新后写,搜索行在事件到达时写。"""
from contextlib import ExitStack, contextmanager
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
import csv
import json
import math

TRAIN_FIELDS = (
    "update", "global_step", "ablation_family", "ablation_variant", "search_scope", "ppo_env_steps",
    "completion_rate", "mean_completed_cost", "mean_completed_geometric_distance",
    "best_completed_geometric_distance", "policy_loss", "value_loss", "entropy", "approx_kl",
    "update_seconds", "ppo_seconds", "latent_search_seconds", "search_calls", "search_due_rows",
    "latent_switch_rows", "search_timeouts", "batch_search_seconds", "predicted_gain",
    "ensemble_disagreement", "mutation_sigma", "latent_gradient_norm",
)
VALIDATION_FIELDS = (
    "update", "ablation_family", "ablation_variant", "validation_protocol",
    "validation_initial_latent_source", "validation_search_scope", "completion_rate",
    "mean_penalized_cost", "mean_completed_cost", "mean_completed_geometric_distance",
    "best_completed_geometric_distance", "validation_seconds", "validation_search_called",
    "validation_search_calls", "validation_search_timed_out", "validation_search_timeouts",
    "validation_switched_rows", "validation_search_seconds",
)
VALIDATION_EPISODE_FIELDS = ("update", "seed", "batch_index", "completed", "total_cost",
                             "geometric_distance", "episode_length", "penalized_cost")
EPISODE_FIELDS = ("update", "global_step", "batch_index", "total_cost", "geometric_distance",
                  "episode_length", "served_customers", "done_reason")
STEP_FIELDS = ("update", "global_step", "batch_index", "action", "reward", "done", "z_switched")
SEARCH_FIELDS = (
    "update", "search_calls_counter", "episode_step", "attempt_number", "ablation_family",
    "ablation_variant", "search_scope", "trigger_reason", "due_rows", "switched_rows", "timed_out",
    "status", "core_search_seconds", "collector_search_seconds", "attempt_seconds",
    "batch_search_seconds_used", "soft_budget_overrun", "critic_forward_calls", "gradient_evaluations",
    "predicted_gain", "gain_in_summary", "mutation_sigma", "population_diversity",
    "logged_search_calls_run_total", "logged_search_timeouts_run_total", "evaluation_backend",
)


def finite_or_none(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


# 核心模块:tensor / numpy / 非有限数 -> 可写入 CSV 与 JSON 的纯 Python 值(NaN/inf -> None)。
def _plain(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            return None
        return int(value) if isinstance(value, Integral) else float(value)
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return _plain(value.item())
    raise TypeError(f"Unsupported diagnostic type: {type(value).__name__}")


def _flag(value):
    return None if value is None else int(bool(value))


class CSVWriter:
    def __init__(self, handle, fields):
        self.fields = tuple(fields)
        self.writer = csv.DictWriter(handle, fieldnames=self.fields)
        self.writer.writeheader()

    def writerow(self, row):
        self.writer.writerow({name: _plain(row.get(name)) for name in self.fields})


# 核心模块:训练各阶段的墙钟计时;阶段边界先同步设备(CUDA)以保证计时准确。
class UpdateTimer:
    def __init__(self, synchronize=lambda: None, clock=perf_counter):
        self.synchronize, self.clock, self.seconds = synchronize, clock, {}
        synchronize()
        self.started = clock()

    @contextmanager
    def phase(self, name):
        self.synchronize()
        start = self.clock()
        try:
            yield
        finally:
            self.synchronize()
            self.seconds[name] = self.seconds.get(name, 0.0) + self.clock() - start

    def elapsed(self):
        self.synchronize()
        return self.clock() - self.started


# 核心模块:日志容器。训练/验证/搜索 CSV 与 search_diagnostics.jsonl;debug=True 时额外写逐回合、逐步 CSV。
class LogBundle:
    def __init__(self, directory, debug=False):
        self.directory, self.debug = Path(directory), bool(debug)
        self.stack, self.handles = ExitStack(), []
        self.episode_csv = self.step_csv = None
        self._calls = 0
        self._timeouts = 0

    def _open(self, name):
        handle = self.stack.enter_context((self.directory / name).open("w", newline="", encoding="utf-8"))
        self.handles.append(handle)
        return handle

    def _csv(self, name, fields):
        return CSVWriter(self._open(name), fields)

    def __enter__(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.training_csv = self._csv("training_metrics.csv", TRAIN_FIELDS)
            self.validation_csv = self._csv("validation.csv", VALIDATION_FIELDS)
            self.validation_episode_csv = self._csv("validation_episodes.csv", VALIDATION_EPISODE_FIELDS)
            self.search_csv = self._csv("search_metrics.csv", SEARCH_FIELDS)
            self.diagnostics = self._open("search_diagnostics.jsonl")
            if self.debug:
                self.episode_csv = self._csv("episode_metrics.csv", EPISODE_FIELDS)
                self.step_csv = self._csv("step_metrics.csv", STEP_FIELDS)
            return self
        except BaseException:
            self.stack.close()
            raise

    # 核心模块:一次 joint_latent_search 事件 -> search_metrics.csv 的一行。
    def _search_row(self, event):
        report = event.get("attempt_report") or {}
        timeout = _flag(event.get("timed_out"))
        generations = event.get("generation_stats") or []
        sigma = ((generations[-1] if generations else {}).get("mutation_sigma")) or {}
        gain = None if timeout else finite_or_none(event.get("predicted_gain_mean"))
        self._calls += 1
        self._timeouts += timeout or 0
        self.search_csv.writerow(dict(
            update=event["update"], search_calls_counter=event.get("calls"), episode_step=event.get("episode_step"),
            attempt_number=report.get("attempt_number"), ablation_family=event.get("ablation_family"),
            ablation_variant=event.get("ablation_variant"), search_scope=event.get("search_scope"),
            trigger_reason=event.get("trigger_reason"), due_rows=event.get("due_rows"),
            switched_rows=event.get("switched_rows"), timed_out=timeout,
            status="timed_out" if timeout == 1 else "completed" if timeout == 0 else "reported",
            core_search_seconds=event.get("elapsed_seconds"),
            collector_search_seconds=event.get("collector_search_seconds"),
            attempt_seconds=report.get("elapsed_seconds"),
            batch_search_seconds_used=report.get("batch_seconds_used"),
            soft_budget_overrun=_flag(report.get("soft_budget_overrun")),
            critic_forward_calls=event.get("critic_forward_calls"),
            gradient_evaluations=event.get("gradient_evaluations"),
            predicted_gain=gain, gain_in_summary=int(gain is not None),
            mutation_sigma=finite_or_none(sigma.get("mean")),
            population_diversity=finite_or_none(event.get("population_diversity")),
            logged_search_calls_run_total=self._calls, logged_search_timeouts_run_total=self._timeouts,
            evaluation_backend=event.get("evaluation_backend")))

    def event(self, event):
        event = _plain(event)
        if event.get("type") == "joint_latent_search":
            self._search_row(event)
        self.diagnostics.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def flush(self):
        for handle in self.handles:
            handle.flush()

    def __exit__(self, *args):
        try:
            self.flush()
        finally:
            self.stack.close()


# 核心模块:由一次 update 的采样结果与 PPO 指标构造 training_metrics.csv 的一行。
def build_training_row(*, update, global_step, batch_size, completed_costs, geometric_distances,
                       completed_mask, metrics, update_seconds, phase_seconds):
    costs = [float(v) for v in completed_costs]
    selected = [float(v) for v, done in zip(geometric_distances, completed_mask) if done]
    average = lambda values: sum(values) / len(values) if values else None
    row = dict(
        update=int(update), global_step=int(global_step),
        ablation_family=metrics.get("ablation_family"), ablation_variant=metrics.get("ablation_variant"),
        search_scope=metrics.get("search_scope"), ppo_env_steps=metrics.get("ppo_env_steps"),
        completion_rate=len(costs) / batch_size, mean_completed_cost=average(costs),
        mean_completed_geometric_distance=average(selected),
        best_completed_geometric_distance=min(selected) if selected else None,
        update_seconds=float(update_seconds), ppo_seconds=float(phase_seconds.get("ppo", 0.0)),
        search_calls=metrics.get("search_calls"), search_due_rows=metrics.get("search_due_rows"),
        latent_switch_rows=metrics.get("latent_switch_rows"), search_timeouts=metrics.get("search_timeouts"),
        batch_search_seconds=finite_or_none(metrics.get("batch_search_seconds")))
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "latent_search_seconds", "predicted_gain",
                "ensemble_disagreement", "mutation_sigma", "latent_gradient_norm"):
        row[key] = finite_or_none(metrics.get(key))
    return row


# 核心模块:由 validate_controller 的返回值构造 validation.csv 的一行。
def build_validation_row(update, values, seconds):
    details = values.get("validation_details") or {}
    source = "common" if str(details.get("latent_source", "")).startswith("common") else "provided"
    return dict(
        update=int(update), ablation_family=details.get("ablation_family"),
        ablation_variant=details.get("ablation_variant"), validation_protocol=values.get("validation_protocol"),
        validation_initial_latent_source=source, validation_search_scope=values.get("validation_search_scope"),
        completion_rate=finite_or_none(values.get("completion_rate")),
        mean_penalized_cost=finite_or_none(values.get("mean_penalized_cost")),
        mean_completed_cost=finite_or_none(values.get("mean_completed_cost")),
        mean_completed_geometric_distance=finite_or_none(values.get("mean_completed_geometric_distance")),
        best_completed_geometric_distance=finite_or_none(values.get("best_completed_geometric_distance")),
        validation_seconds=float(seconds), validation_search_called=_flag(values.get("validation_search_called")),
        validation_search_calls=values.get("validation_search_calls"),
        validation_search_timed_out=_flag(values.get("validation_search_timed_out")),
        validation_search_timeouts=values.get("validation_search_timeouts"),
        validation_switched_rows=values.get("validation_switched_rows"),
        validation_search_seconds=finite_or_none(values.get("validation_search_seconds")))
