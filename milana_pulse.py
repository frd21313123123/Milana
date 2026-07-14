"""Persistent delayed actions and the background pulse that executes them."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from milana_memory import MilanaMemoryStore, PulseTask


MAX_SCHEDULE_DELAY_SECONDS = 365 * 24 * 60 * 60
MAX_SCHEDULED_MESSAGE_LENGTH = 4_000


def _current_time() -> datetime:
    return datetime.now().astimezone()


SCHEDULE_MESSAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "schedule_message",
    "description": (
        "Поставить одно отложенное сообщение текущему собеседнику. Используй, когда "
        "собеседник просит Милану написать ему через указанное время. delay_seconds — "
        "задержка от текущего момента до отправки (например, 5 минут = 300 секунд). "
        "message — именно тот естественный текст от Миланы, который будет отправлен позже. "
        "Не используй для обычного немедленного ответа."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "delay_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SCHEDULE_DELAY_SECONDS,
                "description": "Через сколько секунд отправить сообщение.",
            },
            "message": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_SCHEDULED_MESSAGE_LENGTH,
                "description": "Готовый текст будущего Telegram-сообщения.",
            },
        },
        "required": ["delay_seconds", "message"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class StagedScheduledMessage:
    """Validated delayed send, not persisted until the model answer is accepted."""

    delay_seconds: int
    message: str


def validate_scheduled_message(delay_seconds: Any, message: Any) -> StagedScheduledMessage:
    if isinstance(delay_seconds, bool) or not isinstance(delay_seconds, int):
        raise TypeError("delay_seconds должен быть целым числом")
    if not 1 <= delay_seconds <= MAX_SCHEDULE_DELAY_SECONDS:
        raise ValueError(
            f"delay_seconds должен быть от 1 до {MAX_SCHEDULE_DELAY_SECONDS}"
        )
    if not isinstance(message, str):
        raise TypeError("Текст отложенного сообщения должен быть строкой")
    clean_message = message.strip()
    if not clean_message:
        raise ValueError("Текст отложенного сообщения не может быть пустым")
    if len(clean_message) > MAX_SCHEDULED_MESSAGE_LENGTH:
        raise ValueError(
            "Текст отложенного сообщения не может быть длиннее "
            f"{MAX_SCHEDULED_MESSAGE_LENGTH} символов"
        )
    return StagedScheduledMessage(delay_seconds, clean_message)


class DelayedActionDispatcher:
    """Wake near the next due action and retry temporary delivery failures."""

    def __init__(
        self,
        memory: MilanaMemoryStore,
        execute: Callable[[PulseTask], Awaitable[None]],
        *,
        now: Callable[[], datetime] = _current_time,
        idle_interval_seconds: float = 30.0,
        max_attempts: int = 5,
    ) -> None:
        if idle_interval_seconds <= 0:
            raise ValueError("Интервал пульса должен быть положительным")
        if max_attempts <= 0:
            raise ValueError("Число попыток пульса должно быть положительным")
        self.memory = memory
        self.execute = execute
        self._now = now
        self.idle_interval_seconds = float(idle_interval_seconds)
        self.max_attempts = max_attempts
        self._changed = asyncio.Event()

    def wake(self) -> None:
        """Notify the pulse that an earlier task may have appeared."""
        self._changed.set()

    async def run_once(self) -> int:
        """Claim and execute all tasks currently due; return the claimed count."""
        now = self._now()
        tasks = self.memory.claim_due_pulse_tasks(now, limit=20)
        for task in tasks:
            try:
                await self.execute(task)
            except asyncio.CancelledError:
                # The lease lets another process recover the task after a crash/restart.
                raise
            except Exception as exc:  # noqa: BLE001 - each task owns its retry state
                retry_delay = min(
                    5 * (2 ** min(max(0, task.attempts - 1), 6)),
                    300,
                )
                # A promised send must wait for Telegram to reconnect.  Keep
                # malformed/non-retryable actions bounded, but do not turn a
                # prolonged transport outage into a permanently failed promise.
                retry_limit = (
                    2_147_483_647
                    if isinstance(exc, (ConnectionError, TimeoutError, OSError))
                    else self.max_attempts
                )
                self.memory.retry_pulse_task(
                    task.id,
                    error=f"{type(exc).__name__}: {exc}",
                    retry_at=self._now() + timedelta(seconds=retry_delay),
                    max_attempts=retry_limit,
                )
            else:
                self.memory.complete_pulse_task(task.id, completed_at=self._now())
        return len(tasks)

    async def run(self) -> None:
        """Run until cancelled, sleeping no later than the nearest persisted task."""
        while True:
            await self.run_once()
            # Clear before reading SQLite: a task committed after this point sets
            # the event again, while a task committed just before it is visible
            # to next_pulse_due_at(). This avoids losing a concurrent wake-up.
            self._changed.clear()
            next_due = self.memory.next_pulse_due_at()
            timeout = self.idle_interval_seconds
            if next_due is not None:
                seconds = (next_due - self._now()).total_seconds()
                timeout = min(timeout, max(0.05, seconds))
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=timeout)
            except TimeoutError:
                pass


# Backward-compatible public name.  The old "pulse" is a durable delayed-action
# dispatcher; Milana's autonomous heartbeat lives in ``milana_heartbeat.py``.
MilanaPulse = DelayedActionDispatcher
