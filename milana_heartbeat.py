"""Autonomous, schedule-aware heartbeat for the standalone Milana service."""

from __future__ import annotations

import asyncio
import inspect
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Awaitable, Callable, Mapping

from milana_state import HeartbeatJob, MilanaStateStore, RecoveryWindow


MIN_HEARTBEAT_INTERVAL_SECONDS = 30 * 60
MAX_HEARTBEAT_INTERVAL_SECONDS = 90 * 60
MAX_WAKEUP_HORIZON = timedelta(days=30)
REFLECTIVE_REASONS = frozenset({"heartbeat", "schedule_transition", "delayed_result"})
SLEEP_DEFERRED_REASONS = REFLECTIVE_REASONS | {"schedule_wakeup"}
RECOVERY_SUPERSEDED_REASONS = frozenset(
    {
        "heartbeat",
        "schedule_transition",
        "schedule_wakeup",
        "delayed_result",
    }
)


class HeartbeatReason(StrEnum):
    HEARTBEAT = "heartbeat"
    SCHEDULE_TRANSITION = "schedule_transition"
    SCHEDULE_WAKEUP = "schedule_wakeup"
    RECOVERY = "recovery"
    MANUAL_WAKE = "manual_wake"
    DELAYED_RESULT = "delayed_result"


@dataclass(frozen=True)
class HeartbeatTrigger:
    reason: HeartbeatReason
    scheduled_at: datetime
    fired_at: datetime
    payload: Mapping[str, Any] = field(default_factory=dict)
    recovery: RecoveryWindow | None = None
    logical_id: str | None = None


ExecuteHeartbeat = Callable[[HeartbeatTrigger], Awaitable[None] | None]
SleepCheck = Callable[[datetime], bool]
TimeLookup = Callable[[datetime], datetime | None]
RecoveryContext = Callable[
    [RecoveryWindow],
    Mapping[str, Any] | Awaitable[Mapping[str, Any]],
]


def _now() -> datetime:
    return datetime.now().astimezone()


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("Время должно быть datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class MilanaHeartbeat:
    """Persist and execute Milana's reflective wakes.

    The scheduler does not know about a specific LLM, Telegram or schedule
    implementation.  Those concerns are injected as callbacks, which keeps
    the service testable and lets schedule transitions and sleep be computed by
    ``milana_schedule`` without creating a circular dependency.
    """

    def __init__(
        self,
        state: MilanaStateStore,
        execute: ExecuteHeartbeat,
        *,
        now: Callable[[], datetime] = _now,
        randint: Callable[[int, int], int] = random.randint,
        is_sleeping: SleepCheck | None = None,
        next_awake_at: TimeLookup | None = None,
        next_transition_at: TimeLookup | None = None,
        on_recovery: ExecuteHeartbeat | None = None,
        recovery_context: RecoveryContext | None = None,
        recovery_threshold: timedelta = timedelta(minutes=5),
        poll_interval_seconds: float = 30.0,
        max_attempts: int = 5,
        dev_mode: bool = False,
    ) -> None:
        if not isinstance(state, MilanaStateStore):
            raise TypeError("state должен быть MilanaStateStore")
        if not callable(execute):
            raise TypeError("execute должен быть вызываемым")
        if poll_interval_seconds <= 0:
            raise ValueError("Интервал проверки heartbeat должен быть положительным")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("Число попыток heartbeat должно быть положительным")
        if not isinstance(recovery_threshold, timedelta) or recovery_threshold.total_seconds() < 0:
            raise ValueError("recovery_threshold должен быть неотрицательным timedelta")
        self.state = state
        self.execute = execute
        self._now = now
        self._randint = randint
        self._is_sleeping = is_sleeping or (lambda _: False)
        self._next_awake_at = next_awake_at or (lambda _: None)
        self._next_transition_at = next_transition_at or (lambda _: None)
        self._on_recovery = on_recovery
        self._recovery_context = recovery_context
        self.recovery_threshold = recovery_threshold
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.max_attempts = max_attempts
        self._changed = asyncio.Event()
        self._recovery_checked = False
        self._last_error: str | None = None
        if dev_mode:
            # Delayed actions and explicit wakes are separate and remain active.
            self.state.set_heartbeat_paused(True, at=self._now())

    @property
    def paused(self) -> bool:
        return self.state.get_agent_state().heartbeat_paused

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def pause(self) -> None:
        """Pause only random/schedule-transition reflection."""
        self.state.set_heartbeat_paused(True, at=self._now())
        self._changed.set()

    def resume(self) -> None:
        """Resume reflection and ensure a future random heartbeat exists."""
        current = _aware(self._now())
        self.state.set_heartbeat_paused(False, at=current)
        next_at = self.state.get_agent_state().next_heartbeat_at
        if next_at is None or _aware(next_at) <= current:
            self.state.set_next_heartbeat(self._random_next(current), at=current)
        self._changed.set()

    def wake(
        self,
        reason: HeartbeatReason | str = HeartbeatReason.MANUAL_WAKE,
        *,
        at: datetime | None = None,
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> HeartbeatJob:
        """Persist an explicit wake and notify the running scheduler."""
        normalized = self._reason(reason)
        job = self.state.schedule_heartbeat_job(
            normalized.value,
            _aware(at or self._now()),
            payload=payload,
            idempotency_key=idempotency_key,
        )
        self._changed.set()
        return job

    def schedule_wakeup(
        self,
        at: datetime,
        *,
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> HeartbeatJob:
        current = _aware(self._now())
        due = _aware(at)
        if due <= current:
            raise ValueError("Пробуждение должно быть в будущем")
        if due - current > MAX_WAKEUP_HORIZON:
            raise ValueError("Пробуждение можно назначить максимум на 30 дней вперёд")
        return self.wake(
            HeartbeatReason.SCHEDULE_WAKEUP,
            at=due,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    def wake_now(
        self,
        *,
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> HeartbeatJob:
        """Panel-friendly explicit name for an immediate manual wake."""
        return self.wake(
            HeartbeatReason.MANUAL_WAKE,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    def notify_delayed_result(
        self,
        payload: Mapping[str, Any],
        *,
        at: datetime | None = None,
        idempotency_key: str | None = None,
    ) -> HeartbeatJob:
        return self.wake(
            HeartbeatReason.DELAYED_RESULT,
            at=at,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    def _random_next(self, after: datetime) -> datetime:
        seconds = self._randint(
            MIN_HEARTBEAT_INTERVAL_SECONDS,
            MAX_HEARTBEAT_INTERVAL_SECONDS,
        )
        if isinstance(seconds, bool) or not isinstance(seconds, int):
            raise ValueError("randint должен вернуть целое число")
        if not MIN_HEARTBEAT_INTERVAL_SECONDS <= seconds <= MAX_HEARTBEAT_INTERVAL_SECONDS:
            raise ValueError("randint вернул интервал вне диапазона 30–90 минут")
        return _aware(after) + timedelta(seconds=seconds)

    @staticmethod
    def _reason(value: HeartbeatReason | str) -> HeartbeatReason:
        try:
            return value if isinstance(value, HeartbeatReason) else HeartbeatReason(value)
        except ValueError as exc:
            raise ValueError(f"Неизвестная причина heartbeat: {value!r}") from exc

    def _sleep_postpone_at(self, now: datetime) -> datetime:
        awake_at = self._next_awake_at(now)
        if awake_at is None:
            # A defensive fallback prevents a busy loop when a schedule adapter
            # can detect sleep but has no transition data.
            return now + timedelta(minutes=30)
        awake = _aware(awake_at)
        if awake <= now:
            raise ValueError("next_awake_at должен вернуть будущее время")
        return awake

    def _ensure_transition_job(self, now: datetime) -> None:
        transition = self._next_transition_at(now)
        if transition is None:
            return
        transition = _aware(transition)
        # Some schedule adapters report the exact current boundary for one
        # sampling tick.  It is safe: the idempotency key makes it one-shot.
        key = f"schedule-transition:{transition.astimezone(timezone.utc).isoformat()}"
        self.state.schedule_heartbeat_job(
            HeartbeatReason.SCHEDULE_TRANSITION.value,
            transition,
            idempotency_key=key,
        )

    async def _run_recovery_once(self, now: datetime) -> int:
        if self._recovery_checked:
            self.state.touch_service(now)
            return 0
        window = self.state.begin_recovery(
            now,
            minimum_gap=self.recovery_threshold,
        )
        if window is None:
            self._recovery_checked = True
            return 0
        payload: Mapping[str, Any] = {
            "downtime_seconds": window.duration_seconds,
            "started_at": window.started_at.isoformat(),
            "ended_at": window.ended_at.isoformat(),
        }
        if self._recovery_context is not None:
            supplied = await _await_if_needed(self._recovery_context(window))
            if not isinstance(supplied, Mapping):
                raise TypeError("recovery_context должен вернуть объект")
            payload = {**payload, **dict(supplied)}
        trigger = HeartbeatTrigger(
            reason=HeartbeatReason.RECOVERY,
            scheduled_at=window.ended_at,
            fired_at=now,
            payload=payload,
            recovery=window,
            logical_id=(
                "recovery:"
                + _aware(window.started_at).astimezone(timezone.utc).isoformat()
                + ":"
                + _aware(window.ended_at).astimezone(timezone.utc).isoformat()
            ),
        )
        callback = self._on_recovery or self.execute
        try:
            await _await_if_needed(callback(trigger))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - recovery remains pending for retry
            self._last_error = f"{type(exc).__name__}: {exc}"
            return 0
        self.state.complete_recovery(window, at=now)
        self.state.cancel_stale_heartbeat_jobs(
            window.ended_at,
            kinds=RECOVERY_SUPERSEDED_REASONS,
        )
        self._recovery_checked = True
        self._last_error = None
        self._record_success(now)
        return 1

    def _record_success(self, completed_at: datetime) -> None:
        self.state.record_heartbeat(
            completed_at=completed_at,
            next_at=self._random_next(completed_at),
        )

    async def _execute_job(self, job: HeartbeatJob, now: datetime) -> bool:
        reason = self._reason(job.kind)
        state = self.state.get_agent_state()
        if reason.value in REFLECTIVE_REASONS:
            if state.heartbeat_paused:
                self.state.reschedule_heartbeat_job(
                    job.id,
                    now + timedelta(seconds=self.poll_interval_seconds),
                )
                return False
        if reason.value in SLEEP_DEFERRED_REASONS and self._is_sleeping(now):
            self.state.reschedule_heartbeat_job(
                job.id,
                self._sleep_postpone_at(now),
            )
            return False
        trigger = HeartbeatTrigger(
            reason=reason,
            scheduled_at=job.due_at,
            fired_at=now,
            payload=job.payload,
            logical_id=(
                "random-heartbeat:"
                + str(job.payload["retry_of"])
                if reason == HeartbeatReason.HEARTBEAT
                and isinstance(job.payload.get("retry_of"), str)
                and bool(str(job.payload["retry_of"]).strip())
                else f"heartbeat-job:{job.id}"
            ),
        )
        try:
            await _await_if_needed(self.execute(trigger))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - each job owns its retry state
            self._last_error = f"{type(exc).__name__}: {exc}"
            delay = min(5 * (2 ** max(0, job.attempts - 1)), 300)
            self.state.retry_heartbeat_job(
                job.id,
                error=self._last_error,
                retry_at=now + timedelta(seconds=delay),
                max_attempts=self.max_attempts,
            )
            return False
        self.state.complete_heartbeat_job(job.id, completed_at=now)
        self._last_error = None
        self._record_success(now)
        return True

    async def _run_random_heartbeat(self, now: datetime) -> int:
        state = self.state.get_agent_state()
        if state.next_heartbeat_at is None:
            self.state.set_next_heartbeat(self._random_next(now), at=now)
            return 0
        if _aware(state.next_heartbeat_at) > now or state.heartbeat_paused:
            return 0
        if self._is_sleeping(now):
            self.state.set_next_heartbeat(self._sleep_postpone_at(now), at=now)
            return 0
        trigger = HeartbeatTrigger(
            reason=HeartbeatReason.HEARTBEAT,
            scheduled_at=state.next_heartbeat_at,
            fired_at=now,
            logical_id=(
                "random-heartbeat:"
                + _aware(state.next_heartbeat_at)
                .astimezone(timezone.utc)
                .isoformat()
            ),
        )
        try:
            await _await_if_needed(self.execute(trigger))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - persist a retry instead of stopping loop
            self._last_error = f"{type(exc).__name__}: {exc}"
            retry_at = now + timedelta(seconds=5)
            self.state.schedule_heartbeat_job(
                HeartbeatReason.HEARTBEAT.value,
                retry_at,
                payload={"retry_of": state.next_heartbeat_at.isoformat()},
                idempotency_key=(
                    "heartbeat-retry:"
                    + state.next_heartbeat_at.astimezone(timezone.utc).isoformat()
                ),
            )
            self.state.set_next_heartbeat(retry_at, at=now)
            return 0
        self._last_error = None
        self._record_success(now)
        return 1

    async def run_once(self) -> int:
        """Process recovery, all due explicit jobs and one random heartbeat."""
        now = _aware(self._now())
        processed = await self._run_recovery_once(now)
        if processed:
            # Recovery already contains the missed schedule summary and sets a
            # fresh random heartbeat.  Do not replay stale internal wakes in
            # the same scheduler cycle.
            return processed
        self._ensure_transition_job(now)
        jobs = self.state.claim_due_heartbeat_jobs(now, limit=20)
        for job in jobs:
            processed += int(await self._execute_job(job, now))
        processed += await self._run_random_heartbeat(now)
        return processed

    def _next_timeout(self, now: datetime) -> float:
        candidates: list[datetime] = []
        state = self.state.get_agent_state()
        if not state.heartbeat_paused and state.next_heartbeat_at is not None:
            candidates.append(_aware(state.next_heartbeat_at))
        job_due = self.state.next_heartbeat_job_due_at()
        if job_due is not None:
            candidates.append(_aware(job_due))
        timeout = self.poll_interval_seconds
        if candidates:
            seconds = (min(candidates) - now).total_seconds()
            timeout = min(timeout, max(0.05, seconds))
        return timeout

    async def run(self) -> None:
        """Run until cancelled, waking no later than the next persisted job."""
        while True:
            await self.run_once()
            self._changed.clear()
            timeout = self._next_timeout(_aware(self._now()))
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=timeout)
            except TimeoutError:
                pass
