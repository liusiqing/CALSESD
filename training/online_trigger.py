"""Periodic in-episode trigger. Pure Python; no model/environment mutation.

One decision -> begin_attempt -> search -> finish_attempt, before a real action.
Rejected proposals and handled timeouts count as attempts. A pending search
blocks another decision/reset. Time budgets are soft wall-clock budgets.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from numbers import Integral, Real
from time import perf_counter

ONLINE_CONTEXT_FIELDS = (
    "period_remaining_fraction",
    "cooldown_remaining_fraction",
    "attempts_remaining_fraction",
    "batch_budget_remaining_fraction",
    "batch_budget_limited",
    "online_enabled",
)
ONLINE_CONTEXT_DIM = len(ONLINE_CONTEXT_FIELDS)


def _int(name, x, minimum=0):
    if isinstance(x, bool) or not isinstance(x, Integral) or x < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")
    return int(x)


def _seconds(name, x):
    if isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x < 0:
        raise ValueError(f"invalid {name}")
    return float(x)


def _values(x, name, size, kind):
    # Accept Python/NumPy sequences and, by duck typing, small device tensors.
    # Tensor conversion synchronizes the device; profile this first version.
    if callable(getattr(x, "detach", None)):
        x = x.detach()
    if callable(getattr(x, "cpu", None)):
        x = x.cpu()
    if callable(getattr(x, "tolist", None)):
        x = x.tolist()
    if not isinstance(x, (list, tuple)) or len(x) != size:
        raise ValueError(f"{name} must have {size} rows")
    if kind is bool:
        if not all(isinstance(v, bool) for v in x):
            raise ValueError(f"{name} must be bool rows")
        return tuple(x)
    return tuple(_int(name, v) for v in x)


@dataclass(frozen=True)
class OnlineTriggerConfig:
    batch_size: int
    interval: int = 20
    cooldown: int = 5
    max_attempts: int = 3
    call_budget_seconds: float = 1.0
    batch_budget_seconds: float = 3.0
    min_remaining_customers: int = 2
    warmup_updates: int = 10
    mode: str = "evolution"

    @classmethod
    def from_config(cls, cfg):
        return cls(
            cfg.batch_size,
            cfg.online_search_interval,
            cfg.online_cooldown_steps,
            cfg.online_max_attempts_per_episode,
            cfg.online_search_budget_seconds,
            cfg.online_batch_budget_seconds,
            cfg.online_min_remaining_customers,
            cfg.search_warmup_updates,
            cfg.mode,
        )

    def validate(self):
        for name in ("batch_size", "interval", "max_attempts"):
            _int(name, getattr(self, name), 1)
        for name in ("cooldown", "min_remaining_customers", "warmup_updates"):
            _int(name, getattr(self, name))
        if self.cooldown > self.interval:
            raise ValueError("cooldown must not exceed interval")
        _seconds("call budget", self.call_budget_seconds)
        _seconds("batch budget", self.batch_budget_seconds)
        if self.mode not in ("keep", "evolution"):
            raise ValueError("invalid mode")


@dataclass(frozen=True)
class TriggerDecision:
    episode_id: int
    steps: tuple[int, ...]
    due: tuple[bool, ...]
    reasons: tuple[str, ...]
    budget_seconds: float

    @property
    def should_search(self):
        return any(self.due)


@dataclass(frozen=True)
class SearchAttempt:
    episode_id: int
    number: int
    steps: tuple[int, ...]
    due: tuple[bool, ...]
    started_at: float
    budget_seconds: float

    @property
    def deadline(self):
        return self.started_at + self.budget_seconds if self.budget_seconds > 0 else None


class OnlineTrigger:
    context_fields = ONLINE_CONTEXT_FIELDS

    def __init__(self, config, *, clock=perf_counter):
        self.config = config
        config.validate()
        self.clock = clock
        self.episode_id = 0
        self._pending = None
        self.reset_episode()

    @classmethod
    def from_config(cls, cfg, *, clock=perf_counter):
        return cls(OnlineTriggerConfig.from_config(cfg), clock=clock)

    @property
    def pending(self):
        return self._pending is not None

    def reset_episode(self, *, completed_updates=0, training=True):
        if self.pending:
            raise RuntimeError("finish the pending search before reset")
        if not isinstance(training, bool):
            raise ValueError("training must be bool")
        completed_updates = _int("completed_updates", completed_updates)
        self.episode_id += 1
        self.completed_updates = completed_updates
        self.training = training
        self.enabled = self.config.mode == "evolution" and (
            not training or completed_updates >= self.config.warmup_updates
        )
        B = self.config.batch_size
        self.last_attempt_step = [0] * B
        self.last_switch_step = [0] * B
        self.attempts = [0] * B
        self._seen_steps = [0] * B
        self.batch_seconds_used = 0.0
        self._attempt_number = 0
        self._decision = None

    def _steps(self, steps):
        values = _values(steps, "steps", self.config.batch_size, int)
        if any(now < old for now, old in zip(values, self._seen_steps)):
            raise ValueError("steps decreased; call reset_episode at the full-batch reset")
        self._seen_steps = list(values)
        return values

    def remaining_batch_seconds(self):
        cap = self.config.batch_budget_seconds
        if cap == 0:
            return None
        used = self.batch_seconds_used
        if self.pending:
            used += max(0.0, self.clock() - self._pending.started_at)
        return max(0.0, cap - used)

    def decide(self, *, steps, active, remaining_customers, legal_action_counts):
        if self.pending:
            raise RuntimeError("a search is already pending")
        c = self.config
        B = c.batch_size
        steps = self._steps(steps)
        active = _values(active, "active", B, bool)
        remaining = _values(remaining_customers, "remaining_customers", B, int)
        legal = _values(legal_action_counts, "legal_action_counts", B, int)
        available = self.remaining_batch_seconds()
        reasons = []
        due = []
        for b in range(B):
            gap = steps[b] - self.last_attempt_step[b]
            if not active[b]:
                reason = "inactive"
            elif not self.enabled:
                reason = "keep" if c.mode == "keep" else "warmup"
            elif self.attempts[b] >= c.max_attempts:
                reason = "attempt_cap"
            elif available is not None and available <= 0:
                reason = "batch_budget"
            elif remaining[b] < c.min_remaining_customers:
                reason = "few_customers"
            elif legal[b] < 2:
                reason = "single_legal_action"
            elif gap < c.cooldown:
                reason = "cooldown"
            elif gap < c.interval:
                reason = "interval"
            else:
                reason = "periodic"
            due.append(reason == "periodic")
            reasons.append(reason)
        limits = [v for v in (c.call_budget_seconds, available) if v is not None and v > 0]
        budget = min(limits) if limits else 0.0
        self._decision = TriggerDecision(self.episode_id, steps, tuple(due), tuple(reasons), budget)
        return self._decision

    def begin_attempt(self, decision):
        if self.pending:
            raise RuntimeError("a search is already pending")
        if (
            not isinstance(decision, TriggerDecision)
            or decision is not self._decision
            or decision.episode_id != self.episode_id
        ):
            raise ValueError("decision belongs to a different episode")
        if not decision.should_search:
            raise ValueError("decision has no due rows")
        if tuple(self._seen_steps) != decision.steps:
            raise ValueError("decision is stale")
        # Recheck mutable guards; a previously completed decision cannot be reused.
        available = self.remaining_batch_seconds()
        if available is not None and available <= 0:
            raise RuntimeError("batch budget exhausted")
        for b, is_due in enumerate(decision.due):
            if is_due and (
                not self.enabled
                or self.attempts[b] >= self.config.max_attempts
                or decision.steps[b] - self.last_attempt_step[b] < self.config.interval
            ):
                raise ValueError("decision is no longer eligible")
        for b, is_due in enumerate(decision.due):
            if is_due:
                self.last_attempt_step[b] = decision.steps[b]
                self.attempts[b] += 1
        limits = [v for v in (self.config.call_budget_seconds, available) if v is not None and v > 0]
        budget = min(limits) if limits else 0.0
        self._decision = None
        self._attempt_number += 1
        self._pending = SearchAttempt(
            self.episode_id, self._attempt_number, decision.steps, decision.due, self.clock(), budget
        )
        return self._pending

    def finish_attempt(self, attempt, *, switched=None, elapsed_seconds=None, timed_out=False):
        if self._pending is not attempt:
            raise ValueError("attempt is not the current pending search")
        if not isinstance(timed_out, bool):
            raise ValueError("timed_out must be bool")
        if switched is None:
            switched = (False,) * self.config.batch_size
        switched = _values(switched, "switched", self.config.batch_size, bool)
        if any(s and not d for s, d in zip(switched, attempt.due)):
            raise ValueError("inactive/non-due row switched")
        elapsed = _seconds(
            "elapsed_seconds", self.clock() - attempt.started_at if elapsed_seconds is None else elapsed_seconds
        )
        self.batch_seconds_used += elapsed
        for b, changed in enumerate(switched):
            if changed:
                self.last_switch_step[b] = attempt.steps[b]
        self._pending = None
        return dict(
            attempt_number=attempt.number,
            due_rows=sum(attempt.due),
            switched_rows=sum(switched),
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            batch_seconds_used=self.batch_seconds_used,
            soft_budget_overrun=attempt.budget_seconds > 0 and elapsed > attempt.budget_seconds,
        )

    def context(self, steps):
        """Graph-level [B,6] rows; attach separately, NEVER overwrite context[6].

        begin_attempt commits the same count/cooldown for incumbent and proposed z.
        Refresh after finish_attempt for actual Actor/value/bootstrap inputs.
        Pending budget reflects elapsed time so far, not a prediction of the
        final search cost; this is controller state, not a calibrated guarantee.
        """
        steps = self._steps(steps)
        c = self.config
        remaining = self.remaining_batch_seconds()
        limited = c.batch_budget_seconds > 0
        fraction = 1.0 if not limited else remaining / c.batch_budget_seconds
        rows = []
        for b in range(c.batch_size):
            gap = steps[b] - self.last_attempt_step[b]
            period = max(0.0, min(1.0, (c.interval - gap) / c.interval))
            cooldown = 0.0 if c.cooldown == 0 else max(0.0, min(1.0, (c.cooldown - gap) / c.cooldown))
            rows.append(
                [
                    period,
                    cooldown,
                    max(0.0, (c.max_attempts - self.attempts[b]) / c.max_attempts),
                    fraction,
                    float(limited),
                    float(self.enabled),
                ]
            )
        return rows

    def state_dict(self):
        if self.pending:
            raise RuntimeError("cannot checkpoint while search is pending")
        return dict(
            format_version=1,
            config=asdict(self.config),
            episode_id=self.episode_id,
            completed_updates=self.completed_updates,
            training=self.training,
            last_attempt_step=list(self.last_attempt_step),
            last_switch_step=list(self.last_switch_step),
            attempts=list(self.attempts),
            seen_steps=list(self._seen_steps),
            batch_seconds_used=self.batch_seconds_used,
            attempt_number=self._attempt_number,
        )

    def load_state_dict(self, state):
        if self.pending:
            raise RuntimeError("cannot restore while search is pending")
        if state.get("format_version") != 1 or state.get("config") != asdict(self.config):
            raise ValueError("trigger state/config mismatch")
        B = self.config.batch_size
        arrays = {
            name: list(_values(state[name], name, B, int))
            for name in ("last_attempt_step", "last_switch_step", "attempts", "seen_steps")
        }
        if any(v > self.config.max_attempts for v in arrays["attempts"]):
            raise ValueError("invalid attempt count")
        if any(
            s > a or a > seen
            for s, a, seen in zip(arrays["last_switch_step"], arrays["last_attempt_step"], arrays["seen_steps"])
        ):
            raise ValueError("inconsistent trigger step counters")
        if not isinstance(state["training"], bool):
            raise ValueError("invalid training flag")
        episode = _int("episode_id", state["episode_id"], 1)
        updates = _int("completed_updates", state["completed_updates"])
        seconds = _seconds("batch_seconds_used", state["batch_seconds_used"])
        number = _int("attempt_number", state["attempt_number"])
        if number < max(arrays["attempts"]) or number > sum(arrays["attempts"]):
            raise ValueError("attempt sequence disagrees with row counts")
        if any((a == 0) != (step == 0) for a, step in zip(arrays["attempts"], arrays["last_attempt_step"])):
            raise ValueError("row attempt count/step mismatch")
        enabled = self.config.mode == "evolution" and (not state["training"] or updates >= self.config.warmup_updates)
        if not enabled and number:
            raise ValueError("disabled episode contains search attempts")
        self.episode_id = episode
        self.completed_updates = updates
        self.training = state["training"]
        self.enabled = self.config.mode == "evolution" and (not self.training or updates >= self.config.warmup_updates)
        self.last_attempt_step = arrays["last_attempt_step"]
        self.last_switch_step = arrays["last_switch_step"]
        self.attempts = arrays["attempts"]
        self._seen_steps = arrays["seen_steps"]
        self.batch_seconds_used = seconds
        self._attempt_number = number
        self._decision = None


__all__ = [
    "ONLINE_CONTEXT_FIELDS",
    "ONLINE_CONTEXT_DIM",
    "OnlineTriggerConfig",
    "TriggerDecision",
    "SearchAttempt",
    "OnlineTrigger",
]
