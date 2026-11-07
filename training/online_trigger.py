"""回合内周期触发器(纯 Python,不修改模型/环境)。
流程:decide -> begin_attempt -> 搜索 -> finish_attempt,均发生在真实动作之前。
被拒绝的候选与已处理的超时都计为一次尝试;搜索 pending 时不能再决策或 reset;时间预算为软预算。"""
from __future__ import annotations
from dataclasses import dataclass, asdict
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


def _values(x, name=None, size=None, kind=None):
    """Tensor / ndarray / list -> tuple(Python 标量)。"""
    if callable(getattr(x, "detach", None)):
        x = x.detach()
    if callable(getattr(x, "cpu", None)):
        x = x.cpu()
    if callable(getattr(x, "tolist", None)):
        x = x.tolist()
    return tuple(x)


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
        return cls(cfg.batch_size, cfg.online_search_interval, cfg.online_cooldown_steps,
                   cfg.online_max_attempts_per_episode, cfg.online_search_budget_seconds,
                   cfg.online_batch_budget_seconds, cfg.online_min_remaining_customers,
                   cfg.search_warmup_updates, cfg.mode)


@dataclass(frozen=True)
class TriggerDecision:
    episode_id: int
    steps: tuple
    due: tuple
    reasons: tuple
    budget_seconds: float

    @property
    def should_search(self):
        return any(self.due)


@dataclass(frozen=True)
class SearchAttempt:
    episode_id: int
    number: int
    steps: tuple
    due: tuple
    started_at: float
    budget_seconds: float

    @property
    def deadline(self):
        return self.started_at + self.budget_seconds if self.budget_seconds > 0 else None


# 核心模块:周期触发器。决定哪些行在当前决策点需要搜索,并记录尝试次数、冷却与批量时间预算。
class OnlineTrigger:
    context_fields = ONLINE_CONTEXT_FIELDS

    def __init__(self, config, *, clock=perf_counter):
        self.config = config
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
        self.episode_id += 1
        self.completed_updates = completed_updates
        self.training = training
        self.enabled = self.config.mode == "evolution" and (
            not training or completed_updates >= self.config.warmup_updates)
        B = self.config.batch_size
        self.last_attempt_step = [0] * B
        self.last_switch_step = [0] * B
        self.attempts = [0] * B
        self._seen_steps = [0] * B
        self.batch_seconds_used = 0.0
        self._attempt_number = 0
        self._decision = None

    def remaining_batch_seconds(self):
        cap = self.config.batch_budget_seconds
        if cap == 0:
            return None
        used = self.batch_seconds_used
        if self.pending:
            used += max(0.0, self.clock() - self._pending.started_at)
        return max(0.0, cap - used)

    # 核心模块:逐行判定是否触发,原因按优先级依次为
    # inactive / keep|warmup / attempt_cap / batch_budget / few_customers / single_legal_action / cooldown / interval / periodic
    def decide(self, *, steps, active, remaining_customers, legal_action_counts):
        c = self.config
        steps = _values(steps)
        self._seen_steps = list(steps)
        active, remaining, legal = _values(active), _values(remaining_customers), _values(legal_action_counts)
        available = self.remaining_batch_seconds()
        reasons, due = [], []
        for b in range(c.batch_size):
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
        available = self.remaining_batch_seconds()
        for b, is_due in enumerate(decision.due):
            if is_due:
                self.last_attempt_step[b] = decision.steps[b]
                self.attempts[b] += 1
        limits = [v for v in (self.config.call_budget_seconds, available) if v is not None and v > 0]
        budget = min(limits) if limits else 0.0
        self._decision = None
        self._attempt_number += 1
        self._pending = SearchAttempt(self.episode_id, self._attempt_number, decision.steps,
                                      decision.due, self.clock(), budget)
        return self._pending

    def finish_attempt(self, attempt, *, switched=None, elapsed_seconds=None, timed_out=False):
        if switched is None:
            switched = (False,) * self.config.batch_size
        switched = _values(switched)
        elapsed = float(self.clock() - attempt.started_at if elapsed_seconds is None else elapsed_seconds)
        self.batch_seconds_used += elapsed
        for b, changed in enumerate(switched):
            if changed:
                self.last_switch_step[b] = attempt.steps[b]
        self._pending = None
        return dict(attempt_number=attempt.number, due_rows=sum(attempt.due), switched_rows=sum(switched),
                    elapsed_seconds=elapsed, timed_out=timed_out, batch_seconds_used=self.batch_seconds_used,
                    soft_budget_overrun=attempt.budget_seconds > 0 and elapsed > attempt.budget_seconds)

    # 核心模块:Critic 输入的 6 维控制器上下文(字段顺序见 ONLINE_CONTEXT_FIELDS)。
    def context(self, steps):
        steps = _values(steps)
        self._seen_steps = list(steps)
        c = self.config
        remaining = self.remaining_batch_seconds()
        limited = c.batch_budget_seconds > 0
        fraction = 1.0 if not limited else remaining / c.batch_budget_seconds
        rows = []
        for b in range(c.batch_size):
            gap = steps[b] - self.last_attempt_step[b]
            period = max(0.0, min(1.0, (c.interval - gap) / c.interval))
            cooldown = 0.0 if c.cooldown == 0 else max(0.0, min(1.0, (c.cooldown - gap) / c.cooldown))
            rows.append([period, cooldown, max(0.0, (c.max_attempts - self.attempts[b]) / c.max_attempts),
                         fraction, float(limited), float(self.enabled)])
        return rows

    def state_dict(self):
        return dict(format_version=1, config=asdict(self.config), episode_id=self.episode_id,
                    completed_updates=self.completed_updates, training=self.training,
                    last_attempt_step=list(self.last_attempt_step), last_switch_step=list(self.last_switch_step),
                    attempts=list(self.attempts), seen_steps=list(self._seen_steps),
                    batch_seconds_used=self.batch_seconds_used, attempt_number=self._attempt_number)

    def load_state_dict(self, state):
        self.episode_id = state["episode_id"]
        self.completed_updates = state["completed_updates"]
        self.training = state["training"]
        self.enabled = self.config.mode == "evolution" and (
            not self.training or self.completed_updates >= self.config.warmup_updates)
        self.last_attempt_step = list(state["last_attempt_step"])
        self.last_switch_step = list(state["last_switch_step"])
        self.attempts = list(state["attempts"])
        self._seen_steps = list(state["seen_steps"])
        self.batch_seconds_used = float(state["batch_seconds_used"])
        self._attempt_number = state["attempt_number"]
        self._decision = None


__all__ = ["ONLINE_CONTEXT_FIELDS", "ONLINE_CONTEXT_DIM", "OnlineTriggerConfig", "TriggerDecision",
           "SearchAttempt", "OnlineTrigger"]
