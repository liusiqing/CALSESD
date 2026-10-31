"""Compact logs for periodic in-episode Critic latent search; no Torch import.

One update may contain many search calls. Training rows are deferred until
flush(), so events and the batch summary may arrive after writerow(). Per-call
search rows are immediate. All run totals count this logging segment only.
Validation search never changes training-search totals. Missing data stay blank.
Core elapsed time, collector wall time and charged trigger budget are distinct.
Current phase is Critic-only: obsolete rollout-work and paired-confirmation CSV
columns are removed. Full raw diagnostics are still retained in JSONL.
Send all search events/batch summary before logs.flush(); genuinely late events
raise instead of silently leaving an already-written update summary incorrect.
"""

from collections import deque
from contextlib import ExitStack, contextmanager
from pathlib import Path
import csv
import hashlib
import json
import math
from numbers import Integral, Real
from time import perf_counter
import warnings

STRICT_LOGGING = False

TRAIN_FIELDS = (
    "update",
    "global_step",
    "ablation_family",
    "ablation_variant",
    "search_scope",
    "ppo_env_steps",
    "completion_rate",
    "mean_completed_cost",
    "mean_completed_geometric_distance",
    "best_completed_geometric_distance",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "update_seconds",
    "ppo_seconds",
    "latent_search_seconds",
    "search_calls",
    "search_due_rows",
    "latent_switch_rows",
    "search_timeouts",
    "batch_search_seconds",
    "search_events_complete",
    "predicted_gain",
    "ensemble_disagreement",
    "mutation_sigma",
    "latent_gradient_norm",
)
VALIDATION_FIELDS = (
    "update",
    "ablation_family",
    "ablation_variant",
    "validation_protocol",
    "validation_initial_latent_source",
    "validation_search_scope",
    "completion_rate",
    "mean_penalized_cost",
    "mean_completed_cost",
    "mean_completed_geometric_distance",
    "best_completed_geometric_distance",
    "validation_seconds",
    "validation_search_called",
    "validation_search_calls",
    "validation_search_timed_out",
    "validation_search_timeouts",
    "validation_switched_rows",
    "validation_search_seconds",
)
VALIDATION_EPISODE_FIELDS = (
    "update",
    "seed",
    "batch_index",
    "completed",
    "total_cost",
    "geometric_distance",
    "episode_length",
    "penalized_cost",
)
EPISODE_FIELDS = (
    "update",
    "global_step",
    "batch_index",
    "total_cost",
    "geometric_distance",
    "episode_length",
    "served_customers",
    "done_reason",
)
STEP_FIELDS = ("update", "global_step", "batch_index", "action", "reward", "done", "z_switched")
SEARCH_FIELDS = (
    "update",
    "search_calls_counter",
    "episode_step",
    "attempt_number",
    "ablation_family",
    "ablation_variant",
    "search_scope",
    "trigger_reason",
    "due_rows",
    "switched_rows",
    "timed_out",
    "status",
    "core_search_seconds",
    "collector_search_seconds",
    "attempt_seconds",
    "batch_search_seconds_used",
    "soft_budget_overrun",
    "critic_forward_calls",
    "gradient_evaluations",
    "predicted_gain",
    "gain_in_summary",
    "mutation_sigma",
    "population_diversity",
    "logged_search_calls_run_total",
    "logged_search_timeouts_run_total",
    "evaluation_backend",
)


def _violation(message):
    if STRICT_LOGGING:
        raise ValueError(message)
    warnings.warn(f"logging invariant violated: {message}", RuntimeWarning, stacklevel=3)


def finite_or_none(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


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


def _count(name, value, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _flag(name, value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral) and value in (0, 1):
        return int(value)
    raise ValueError(f"{name} must be bool, 0, 1 or None")


def _time_or_none(name, value):
    result = finite_or_none(value)
    if result is not None and result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


def _coalesce(*values):
    return next((value for value in values if value is not None), None)


def _label(name, value):
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be string or None")
    return value


class MetricRow(dict):
    """Normal dict with optional JSON event metadata, never an extra CSV key."""

    def __init__(self, *args, diagnostic_event=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostic_event = diagnostic_event


class CSVWriter:
    def __init__(self, handle, fields, strict=False, *, enrich=None, on_row=None):
        self.fields, self.strict = tuple(fields), strict
        self.enrich, self.on_row = enrich, on_row
        self.writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction="raise")
        self.writer.writeheader()

    def writerow(self, row):
        values = dict(row)
        if self.enrich is not None:
            values = self.enrich(values)
        if self.strict and set(values) != set(self.fields):
            raise ValueError(
                f"CSV schema mismatch: missing={set(self.fields)-set(values)}, extra={set(values)-set(self.fields)}"
            )
        plain = {name: _plain(values.get(name)) for name in self.fields}
        if self.on_row is not None:
            self.on_row(row, plain)
        self.writer.writerow(plain)


class UpdateTimer:
    """Boundary-synchronized wall clock for training phases."""

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


class DeferredCSVWriter(CSVWriter):
    """Buffer training scalars only; LogBundle.flush drains in insertion order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = deque()

    def writerow(self, row):
        values = dict(row)
        if self.strict and set(values) != set(self.fields):
            raise ValueError(
                f"CSV schema mismatch: missing={set(self.fields)-set(values)}, extra={set(values)-set(self.fields)}"
            )
        self.pending.append({key: _plain(value) for key, value in values.items()})

    def drain(self):
        written = []
        while self.pending:
            row = self.pending.popleft()
            super().writerow(row)
            written.append(row["update"])
        return written


def _same_count(name, *values):
    values = [_count(name, value) for value in values if value is not None]
    if values and any(value != values[0] for value in values[1:]):
        _violation(f"Conflicting per-update {name}: {values}")
    return values[0] if values else None


def _same_time(name, *values):
    values = [_time_or_none(name, value) for value in values if value is not None]
    values = [value for value in values if value is not None]
    if values and any(not math.isclose(value, values[0], rel_tol=1e-6, abs_tol=1e-6) for value in values[1:]):
        _violation(f"Conflicting {name}; do not mix collector wall time with charged trigger time")
    return values[0] if values else None


def _add_known(total, value):
    return None if total is None or value is None else total + value


class LogBundle:
    def __init__(self, directory, debug=False):
        self.directory, self.debug = Path(directory), bool(debug)
        self.stack, self.handles = ExitStack(), []
        self.episode_csv = self.step_csv = None
        self.run_identity = {
            "ablation_family": None,
            "ablation_variant": None,
            "search_scope": None,
            "evaluation_backend": None,
        }
        self._last_global_step = None
        self._updates = {}
        self._seen_search = {}
        self._seen_validation = set()
        self._written_updates = set()
        self._logged_calls = 0
        self._logged_timeouts = 0

    def _csv(self, name, fields, strict=False, *, deferred=False, **kwargs):
        handle = self.stack.enter_context((self.directory / name).open("w", newline="", encoding="utf-8"))
        self.handles.append(handle)
        cls = DeferredCSVWriter if deferred else CSVWriter
        return cls(handle, fields, strict, **kwargs)

    def __enter__(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.training_csv = self._csv(
                "training_metrics.csv", TRAIN_FIELDS, True, deferred=True, enrich=self._training_context
            )
            self.validation_csv = self._csv(
                "validation.csv", VALIDATION_FIELDS, True, enrich=self._identity_context, on_row=self._validation_event
            )
            self.validation_episode_csv = self._csv("validation_episodes.csv", VALIDATION_EPISODE_FIELDS, True)
            self.search_csv = self._csv("search_metrics.csv", SEARCH_FIELDS, True)
            self.diagnostics = self.stack.enter_context(
                (self.directory / "search_diagnostics.jsonl").open("w", encoding="utf-8")
            )
            self.handles.append(self.diagnostics)
            if self.debug:
                self.episode_csv = self._csv("episode_metrics.csv", EPISODE_FIELDS)
                self.step_csv = self._csv("step_metrics.csv", STEP_FIELDS)
            return self
        except BaseException:
            self.stack.close()
            raise

    def _set_identity(self, family=None, variant=None, scope=None, backend=None):
        if scope is not None and scope != "episode":
            raise ValueError("This log schema is for episode-internal search, not old update-period experiments")
        if backend is not None and backend != "critic":
            raise ValueError("This phase-one logger expects Critic-scored search")
        incoming = dict(
            ablation_family=family, ablation_variant=variant, search_scope=scope, evaluation_backend=backend
        )
        for key, value in incoming.items():
            if value is None:
                continue
            _label(key, value)
            if self.run_identity[key] is not None and self.run_identity[key] != value:
                raise ValueError(f"One LogBundle cannot mix experimental identities: {key}")
            self.run_identity[key] = value

    def _identity_context(self, row):
        for key in ("ablation_family", "ablation_variant"):
            value = self.run_identity[key]
            if row.get(key) is None:
                row[key] = value
            elif value is not None and row[key] != value:
                raise ValueError(f"CSV identity mismatch: {key}")
        return row

    def _update(self, update):
        update = _count("update", update)
        if update not in self._updates:
            self._updates[update] = dict(
                events=0, due_rows=0, switched_rows=0, timeouts=0, attempt_seconds=0.0, summary=None
            )
        return self._updates[update]

    def _training_context(self, row):
        row = self._identity_context(row)
        scope = self.run_identity["search_scope"]
        if row["search_scope"] is None:
            row["search_scope"] = scope
        elif scope is not None and row["search_scope"] != scope:
            raise ValueError("training search scope mismatch")
        step = _count("global_step", row["global_step"])
        delta = _count("ppo_env_steps", row["ppo_env_steps"], optional=True)
        if self._last_global_step is not None:
            if step < self._last_global_step:
                _violation("global_step decreased within the logging segment")
            expected = step - self._last_global_step
            if delta is not None and delta != expected:
                _violation("ppo_env_steps disagrees with global_step delta")
            if delta is None:
                delta = expected
        elif delta is None and row["update"] == 1:
            delta = step
        row["ppo_env_steps"] = delta
        data = self._updates.get(row["update"])
        summary = data["summary"] if data and data["summary"] is not None else {}
        expected_calls = _same_count("search_calls", row["search_calls"], summary.get("search_calls"))
        event_count = data["events"] if data else 0
        if expected_calls is not None and event_count > expected_calls:
            _violation("More unique search events than the batch reports")
        complete = (event_count == expected_calls) if expected_calls is not None else None
        row["search_events_complete"] = None if complete is None else int(complete)
        row["search_calls"] = expected_calls if expected_calls is not None else event_count if event_count else None
        # Only compare event sums with batch totals when every call is present.
        use_events = complete is True or (expected_calls is None and event_count > 0)
        zeros = expected_calls == 0
        row["search_due_rows"] = _same_count(
            "search_due_rows",
            row["search_due_rows"],
            summary.get("attempted_rows"),
            data["due_rows"] if use_events and data else 0 if zeros else None,
        )
        row["latent_switch_rows"] = _same_count(
            "latent_switch_rows",
            row["latent_switch_rows"],
            summary.get("switched_rows"),
            data["switched_rows"] if use_events and data else 0 if zeros else None,
        )
        row["search_timeouts"] = _same_count(
            "search_timeouts",
            row["search_timeouts"],
            summary.get("search_timeouts"),
            data["timeouts"] if use_events and data else 0 if zeros else None,
        )
        row["batch_search_seconds"] = _same_time(
            "batch_search_seconds",
            row["batch_search_seconds"],
            summary.get("batch_search_seconds"),
            data["attempt_seconds"] if use_events and data else 0.0 if zeros else None,
        )
        self._last_global_step = step
        return row

    def _validation_event(self, source_row, plain_row):
        event = getattr(source_row, "diagnostic_event", None)
        if event is not None:
            self.event(event)

    def _record_search(self, event):
        update = _count("search update", event["update"])
        counter = _count("search calls counter", event.get("calls"), optional=True)
        report = event.get("attempt_report") or {}
        if not isinstance(report, dict):
            raise TypeError("attempt_report must be a dict")
        attempt = _count("attempt_number", report.get("attempt_number"), optional=True)
        token = json.dumps(event, sort_keys=True, allow_nan=False)
        signature = hashlib.sha256(token.encode()).hexdigest()
        if counter is not None:
            key = ("call", update, counter)
        elif attempt is not None:
            key = ("attempt", update, attempt)
        else:
            key = ("event", signature)
        if key in self._seen_search:
            if self._seen_search[key] != signature:
                _violation("Conflicting diagnostics for the same search identity")
            return False
        if update in self._written_updates:
            _violation("New search event arrived after its training row was flushed")
            return False
        timeout = _flag("timed_out", event.get("timed_out"))
        due = _count("due_rows", event.get("due_rows"), optional=True)
        switched = _count("switched_rows", event.get("switched_rows"), optional=True)
        if due is not None and switched is not None and switched > due:
            raise ValueError("switched rows exceed due rows")
        gain_allowed = timeout == 0 and event.get("gain_in_summary", True) is not False
        predicted = finite_or_none(event.get("predicted_gain_mean")) if gain_allowed else None
        generations = event.get("generation_stats") or []
        last = generations[-1] if generations else {}
        sigma = last.get("mutation_sigma") or {}
        duration = _time_or_none("attempt_seconds", report.get("elapsed_seconds"))
        data = self._update(update)
        row = dict(
            update=update,
            search_calls_counter=counter,
            episode_step=_count(
                "episode_step",
                event.get("episode_step"),
                optional=True,
            ),
            attempt_number=attempt,
            ablation_family=self.run_identity["ablation_family"],
            ablation_variant=self.run_identity["ablation_variant"],
            search_scope=self.run_identity["search_scope"],
            evaluation_backend=event.get("evaluation_backend"),
            trigger_reason=event.get("trigger_reason"),
            due_rows=due,
            switched_rows=switched,
            timed_out=timeout,
            status=("timed_out" if timeout == 1 else "completed" if timeout == 0 else "reported"),
            core_search_seconds=_time_or_none(
                "elapsed_seconds",
                event.get("elapsed_seconds"),
            ),
            collector_search_seconds=_time_or_none(
                "collector_search_seconds",
                event.get("collector_search_seconds"),
            ),
            attempt_seconds=duration,
            batch_search_seconds_used=_time_or_none(
                "batch_seconds_used",
                report.get("batch_seconds_used"),
            ),
            soft_budget_overrun=_flag(
                "soft_budget_overrun",
                report.get("soft_budget_overrun"),
            ),
            critic_forward_calls=_count(
                "critic_forward_calls",
                event.get("critic_forward_calls"),
                optional=True,
            ),
            gradient_evaluations=_count(
                "gradient_evaluations",
                event.get("gradient_evaluations"),
                optional=True,
            ),
            predicted_gain=predicted,
            gain_in_summary=int(gain_allowed and predicted is not None),
            mutation_sigma=finite_or_none(sigma.get("mean")),
            population_diversity=finite_or_none(
                event.get("population_diversity"),
            ),
            logged_search_calls_run_total=self._logged_calls + 1,
            logged_search_timeouts_run_total=(self._logged_timeouts + (timeout or 0)),
        )

        self.search_csv.writerow(row)
        self._seen_search[key] = signature
        self._logged_calls += 1
        self._logged_timeouts += timeout or 0
        data["events"] += 1
        data["due_rows"] = _add_known(data["due_rows"], due)
        data["switched_rows"] = _add_known(data["switched_rows"], switched)
        data["timeouts"] = _add_known(data["timeouts"], timeout)
        data["attempt_seconds"] = _add_known(data["attempt_seconds"], duration)
        return True

    def event(self, event):
        event = _plain(event)
        if not isinstance(event, dict):
            raise TypeError("log event must be a dict")
        kind = event.get("type")
        if kind == "ablation_setup":
            plan = event.get("plan") or {}
            search = event.get("resolved_search_config") or {}
            self._set_identity(
                plan.get("family"),
                plan.get("variant"),
                plan.get("search_scope", search.get("search_scope")),
                search.get("evaluation_backend", plan.get("evaluation_backend")),
            )
        elif kind in ("joint_latent_search", "ablation_training_update"):
            self._set_identity(
                _coalesce(event.get("ablation_family"), event.get("family")),
                _coalesce(event.get("ablation_variant"), event.get("variant")),
                event.get("search_scope"),
                event.get("evaluation_backend"),
            )
        if kind in ("joint_latent_search", "validation_protocol"):
            for key in ("ablation_family", "ablation_variant"):
                if event.get(key) is None:
                    event[key] = self.run_identity[key]
                elif self.run_identity[key] is not None and event[key] != self.run_identity[key]:
                    raise ValueError(f"event identity mismatch: {key}")
        if kind == "joint_latent_search" and not self._record_search(event):
            return
        if kind == "ablation_training_update":
            if event["update"] in self._written_updates:
                _violation("Batch summary arrived after its training row was flushed")
                return
            data = self._update(event["update"])
            if data["summary"] is not None and data["summary"] != event:
                _violation("Conflicting summaries for one update")
            data["summary"] = event
        if kind == "validation_protocol":
            token = hashlib.sha256(json.dumps(event, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if token in self._seen_validation:
                return
            self._seen_validation.add(token)
        self.diagnostics.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def flush(self):
        for update in self.training_csv.drain():
            self._written_updates.add(update)
            self._updates.pop(update, None)
        for handle in self.handles:
            handle.flush()

    def __exit__(self, *args):
        try:
            self.flush()
        except BaseException:
            self.stack.close()
            if args[0] is None:
                raise
            return False  # Keep an original training failure as the primary error.
        return self.stack.__exit__(*args)


def build_training_row(
    *,
    update,
    global_step,
    batch_size,
    completed_costs,
    geometric_distances,
    completed_mask,
    metrics,
    update_seconds,
    phase_seconds,
):
    _count("update", update)
    _count("global_step", global_step)
    _count("batch_size", batch_size)
    costs = [float(v) for v in completed_costs]
    distances, completed = list(geometric_distances), list(completed_mask)
    if batch_size < 1 or len(distances) != batch_size or len(completed) != batch_size:
        raise ValueError("invalid batch summary shape")
    if len(costs) != sum(bool(v) for v in completed):
        raise ValueError("completed cost/mask count mismatch")
    selected = [float(v) for v, done in zip(distances, completed) if done]
    if not all(math.isfinite(v) for v in costs + selected):
        raise ValueError("nonfinite completed execution statistics")
    for name, value in dict(update_seconds=update_seconds, ppo_seconds=phase_seconds.get("ppo", 0.0)).items():
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"invalid elapsed time: {name}")
    average = lambda values: sum(values) / len(values) if values else None
    row = dict(
        update=int(update),
        global_step=int(global_step),
        ablation_family=_label("ablation_family", metrics.get("ablation_family")),
        ablation_variant=_label("ablation_variant", metrics.get("ablation_variant")),
        search_scope=_label("search_scope", metrics.get("search_scope")),
        ppo_env_steps=_count("ppo_env_steps", metrics.get("ppo_env_steps"), optional=True),
        completion_rate=len(costs) / batch_size,
        mean_completed_cost=average(costs),
        mean_completed_geometric_distance=average(selected),
        best_completed_geometric_distance=min(selected) if selected else None,
        update_seconds=float(update_seconds),
        ppo_seconds=float(phase_seconds.get("ppo", 0.0)),
        search_calls=_count("search_calls", metrics.get("search_calls"), optional=True),
        search_due_rows=_count(
            "search_due_rows",
            _coalesce(metrics.get("search_due_rows"), metrics.get("search_attempt_rows")),
            optional=True,
        ),
        latent_switch_rows=_count(
            "latent_switch_rows",
            _coalesce(metrics.get("latent_switch_rows"), metrics.get("latent_switch_events")),
            optional=True,
        ),
        search_timeouts=_count("search_timeouts", metrics.get("search_timeouts"), optional=True),
        batch_search_seconds=_time_or_none("batch_search_seconds", metrics.get("batch_search_seconds")),
        search_events_complete=None,
    )
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "latent_search_seconds",
        "predicted_gain",
        "ensemble_disagreement",
        "mutation_sigma",
        "latent_gradient_norm",
    ):
        row[key] = finite_or_none(metrics.get(key))
    if set(row) != set(TRAIN_FIELDS):
        raise ValueError("internal training schema mismatch")
    return row


def build_validation_row(update, values, seconds):
    _count("update", update)
    if not math.isfinite(float(seconds)) or seconds < 0:
        raise ValueError("invalid validation time")
    details = values.get("validation_details")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise TypeError("validation_details must be a dict")
    calls = _count(
        "validation_search_calls",
        _coalesce(values.get("validation_search_calls"), details.get("search_calls")),
        optional=True,
    )
    timeouts = _count(
        "validation_search_timeouts",
        _coalesce(values.get("validation_search_timeouts"), details.get("search_timeouts")),
        optional=True,
    )
    called = _flag(
        "validation_search_called", _coalesce(values.get("validation_search_called"), details.get("search_called"))
    )
    timed_out = _flag(
        "validation_search_timed_out",
        _coalesce(values.get("validation_search_timed_out"), details.get("search_timed_out")),
    )
    if calls is not None:
        if called is not None and called != int(calls > 0):
            raise ValueError("validation call count/flag disagree")
        called = int(calls > 0)
    elif called == 0:
        calls = 0
    if timeouts is not None:
        if timed_out is not None and timed_out != int(timeouts > 0):
            raise ValueError("validation timeout count/flag disagree")
        timed_out = int(timeouts > 0)
    elif timed_out == 0:
        timeouts = 0
    if calls == 0 and (timed_out == 1 or (timeouts is not None and timeouts > 0)):
        raise ValueError("validation timed out without search calls")
    if calls is not None and timeouts is not None and timeouts > calls:
        raise ValueError("validation timeout count exceeds call count")
    row = {
        key: finite_or_none(values.get(key))
        for key in (
            "completion_rate",
            "mean_penalized_cost",
            "mean_completed_cost",
            "mean_completed_geometric_distance",
            "best_completed_geometric_distance",
        )
    }
    row.update(
        update=int(update),
        validation_seconds=float(seconds),
        ablation_family=_label(
            "ablation_family", _coalesce(values.get("ablation_family"), details.get("ablation_family"))
        ),
        ablation_variant=_label(
            "ablation_variant", _coalesce(values.get("ablation_variant"), details.get("ablation_variant"))
        ),
        validation_protocol=_label(
            "validation_protocol", _coalesce(values.get("validation_protocol"), details.get("protocol"))
        ),
        validation_initial_latent_source=_label(
            "latent_source", _coalesce(values.get("validation_initial_latent_source"), details.get("latent_source"))
        ),
        # Never infer actual validation scope merely from the training manifest.
        validation_search_scope=_label(
            "validation_search_scope", _coalesce(values.get("validation_search_scope"), details.get("search_scope"))
        ),
        validation_search_called=called,
        validation_search_calls=calls,
        validation_search_timed_out=timed_out,
        validation_search_timeouts=timeouts,
        validation_switched_rows=_count(
            "validation_switched_rows",
            _coalesce(values.get("validation_switched_rows"), details.get("switched_rows")),
            optional=True,
        ),
        validation_search_seconds=_time_or_none(
            "validation_search_seconds",
            _coalesce(values.get("validation_search_seconds"), details.get("search_seconds")),
        ),
    )
    if set(row) != set(VALIDATION_FIELDS):
        raise ValueError("internal validation schema mismatch")
    event = None
    if details or row["validation_protocol"] is not None:
        event = dict(details)
        event.update(type="validation_protocol", update=int(update))
        if "protocol" not in event:
            event["protocol"] = row["validation_protocol"]
    return MetricRow(row, diagnostic_event=event)


__all__ = [
    "TRAIN_FIELDS",
    "VALIDATION_FIELDS",
    "VALIDATION_EPISODE_FIELDS",
    "EPISODE_FIELDS",
    "STEP_FIELDS",
    "SEARCH_FIELDS",
    "finite_or_none",
    "CSVWriter",
    "LogBundle",
    "UpdateTimer",
    "build_training_row",
    "build_validation_row",
]
