import unittest
from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import AsyncMock

from milana_memory import MilanaMemoryStore
from milana_pulse import MilanaPulse, validate_scheduled_message


class PulseStorageTests(unittest.TestCase):
    def test_task_survives_reopen_and_is_only_claimed_when_due(self) -> None:
        now = datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.sqlite3"
            first = MilanaMemoryStore(path)
            created = first.schedule_pulse_message(
                100,
                "Я снова здесь",
                due_at=now + timedelta(minutes=5),
                source_message_id=7,
            )
            first.close()

            second = MilanaMemoryStore(path)
            try:
                self.assertEqual(second.claim_due_pulse_tasks(now), [])
                claimed = second.claim_due_pulse_tasks(now + timedelta(minutes=5))
                self.assertEqual([task.id for task in claimed], [created.id])
                self.assertEqual(claimed[0].message, "Я снова здесь")
                self.assertEqual(claimed[0].attempts, 1)
            finally:
                second.close()

    def test_schedule_arguments_are_strictly_validated(self) -> None:
        self.assertEqual(
            validate_scheduled_message(300, "  Напоминаю  ").message,
            "Напоминаю",
        )
        with self.assertRaises(TypeError):
            validate_scheduled_message(True, "текст")
        with self.assertRaises(ValueError):
            validate_scheduled_message(0, "текст")
        with self.assertRaises(ValueError):
            validate_scheduled_message(10, "   ")


class PulseRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_due_task_is_delivered_and_completed(self) -> None:
        now = datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)
        memory = MilanaMemoryStore()
        memory.schedule_pulse_message(100, "Пять минут прошло", due_at=now)
        execute = AsyncMock()
        pulse = MilanaPulse(memory, execute, now=lambda: now)

        self.assertEqual(await pulse.run_once(), 1)

        execute.assert_awaited_once()
        self.assertEqual(memory.get_pulse_tasks()[0].status, "completed")

    async def test_temporary_failure_is_retried_with_backoff(self) -> None:
        clock = [datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)]
        memory = MilanaMemoryStore()
        memory.schedule_pulse_message(100, "Попробуй ещё раз", due_at=clock[0])
        execute = AsyncMock(side_effect=[OSError("Telegram недоступен"), None])
        pulse = MilanaPulse(memory, execute, now=lambda: clock[0])

        await pulse.run_once()
        pending = memory.get_pulse_tasks()[0]
        self.assertEqual(pending.status, "pending")
        self.assertEqual(pending.attempts, 1)

        clock[0] += timedelta(seconds=5)
        await pulse.run_once()
        completed = memory.get_pulse_tasks()[0]
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.attempts, 2)

    async def test_transport_outage_does_not_exhaust_promised_delivery(self) -> None:
        now = datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)
        memory = MilanaMemoryStore()
        memory.schedule_pulse_message(100, "дождусь связи", due_at=now)
        pulse = MilanaPulse(
            memory,
            AsyncMock(side_effect=ConnectionError("host offline")),
            now=lambda: now,
            max_attempts=1,
        )

        await pulse.run_once()

        task = memory.get_pulse_tasks()[0]
        self.assertEqual(task.status, "pending")
        self.assertEqual(task.attempts, 1)

    async def test_non_transport_failure_still_respects_attempt_limit(self) -> None:
        now = datetime(2026, 7, 13, 18, 0, tzinfo=timezone.utc)
        memory = MilanaMemoryStore()
        memory.schedule_pulse_message(100, "некорректная задача", due_at=now)
        pulse = MilanaPulse(
            memory,
            AsyncMock(side_effect=ValueError("bad action")),
            now=lambda: now,
            max_attempts=1,
        )

        await pulse.run_once()

        self.assertEqual(memory.get_pulse_tasks()[0].status, "failed")


if __name__ == "__main__":
    unittest.main()
