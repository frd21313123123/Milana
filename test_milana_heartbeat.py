import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from milana_heartbeat import (
    MAX_HEARTBEAT_INTERVAL_SECONDS,
    MIN_HEARTBEAT_INTERVAL_SECONDS,
    HeartbeatReason,
    MilanaHeartbeat,
)
from milana_state import MilanaStateStore


NOW = datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc)


class HeartbeatSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clock = [NOW]
        self.store = MilanaStateStore()
        self.execute = AsyncMock()

    async def asyncTearDown(self) -> None:
        self.store.close()

    def scheduler(self, **kwargs):
        return MilanaHeartbeat(
            self.store,
            self.execute,
            now=lambda: self.clock[0],
            randint=lambda low, high: low,
            **kwargs,
        )

    async def test_due_reflection_runs_and_schedules_30_to_90_minutes(self) -> None:
        self.store.set_next_heartbeat(NOW - timedelta(seconds=1), at=NOW)
        heartbeat = self.scheduler()

        self.assertEqual(await heartbeat.run_once(), 1)

        trigger = self.execute.await_args.args[0]
        self.assertEqual(trigger.reason, HeartbeatReason.HEARTBEAT)
        state = self.store.get_agent_state()
        delay = (state.next_heartbeat_at - NOW).total_seconds()
        self.assertGreaterEqual(delay, MIN_HEARTBEAT_INTERVAL_SECONDS)
        self.assertLessEqual(delay, MAX_HEARTBEAT_INTERVAL_SECONDS)

    async def test_pause_blocks_reflection_but_not_manual_wake(self) -> None:
        self.store.set_next_heartbeat(NOW - timedelta(seconds=1), at=NOW)
        heartbeat = self.scheduler()
        heartbeat.pause()
        self.assertEqual(await heartbeat.run_once(), 0)
        self.execute.assert_not_awaited()

        heartbeat.wake(payload={"source": "panel"})
        self.assertEqual(await heartbeat.run_once(), 1)
        trigger = self.execute.await_args.args[0]
        self.assertEqual(trigger.reason, HeartbeatReason.MANUAL_WAKE)
        self.assertEqual(trigger.payload["source"], "panel")

    async def test_sleep_defers_reflection_until_injected_wake_time(self) -> None:
        wake_at = NOW + timedelta(hours=7)
        self.store.set_next_heartbeat(NOW - timedelta(seconds=1), at=NOW)
        heartbeat = self.scheduler(
            is_sleeping=lambda _: True,
            next_awake_at=lambda _: wake_at,
        )

        self.assertEqual(await heartbeat.run_once(), 0)

        self.execute.assert_not_awaited()
        self.assertEqual(self.store.get_agent_state().next_heartbeat_at, wake_at)

    async def test_schedule_transition_is_persisted_and_idempotent(self) -> None:
        heartbeat = self.scheduler(next_transition_at=lambda _: NOW)

        self.assertEqual(await heartbeat.run_once(), 1)
        self.assertEqual(
            self.execute.await_args.args[0].reason,
            HeartbeatReason.SCHEDULE_TRANSITION,
        )
        self.execute.reset_mock()
        self.assertEqual(await heartbeat.run_once(), 0)
        self.execute.assert_not_awaited()

    async def test_recovery_turn_runs_once_with_injected_context(self) -> None:
        self.store.touch_service(NOW)
        self.clock[0] = NOW + timedelta(hours=3)
        recovery = AsyncMock()
        heartbeat = self.scheduler(
            on_recovery=recovery,
            recovery_context=lambda window: {"missed": "сон → завтрак"},
        )

        self.assertEqual(await heartbeat.run_once(), 1)
        trigger = recovery.await_args.args[0]
        self.assertEqual(trigger.reason, HeartbeatReason.RECOVERY)
        self.assertEqual(trigger.payload["missed"], "сон → завтрак")
        self.assertEqual(trigger.payload["downtime_seconds"], 3 * 60 * 60)
        self.assertIsNone(self.store.get_pending_recovery())

        recovery.reset_mock()
        self.assertEqual(await heartbeat.run_once(), 0)
        recovery.assert_not_awaited()

    async def test_schedule_wakeup_has_30_day_horizon_and_survives_pause(self) -> None:
        heartbeat = self.scheduler()
        heartbeat.pause()
        with self.assertRaises(ValueError):
            heartbeat.schedule_wakeup(NOW + timedelta(days=31))

        heartbeat.schedule_wakeup(
            NOW + timedelta(hours=1),
            payload={"promise": "проверить планы"},
        )
        self.clock[0] += timedelta(hours=1)
        self.assertEqual(await heartbeat.run_once(), 1)
        trigger = self.execute.await_args.args[0]
        self.assertEqual(trigger.reason, HeartbeatReason.SCHEDULE_WAKEUP)

    async def test_schedule_wakeup_is_deferred_during_sleep(self) -> None:
        wake_at = NOW + timedelta(hours=7)
        heartbeat = self.scheduler(
            is_sleeping=lambda _: True,
            next_awake_at=lambda _: wake_at,
        )
        heartbeat.schedule_wakeup(NOW + timedelta(minutes=1))
        self.clock[0] += timedelta(minutes=1)

        self.assertEqual(await heartbeat.run_once(), 0)

        self.execute.assert_not_awaited()
        pending = self.store.list_heartbeat_jobs(statuses=("pending",))
        self.assertEqual(pending[0].due_at, wake_at)

    async def test_recovery_supersedes_missed_reflective_jobs(self) -> None:
        self.store.touch_service(NOW)
        self.store.schedule_heartbeat_job(
            HeartbeatReason.SCHEDULE_TRANSITION.value,
            NOW + timedelta(minutes=10),
        )
        self.store.schedule_heartbeat_job(
            HeartbeatReason.SCHEDULE_WAKEUP.value,
            NOW + timedelta(minutes=20),
        )
        self.clock[0] = NOW + timedelta(hours=3)
        heartbeat = self.scheduler()

        self.assertEqual(await heartbeat.run_once(), 1)
        self.assertEqual(self.execute.await_count, 1)
        self.assertEqual(
            self.execute.await_args.args[0].reason,
            HeartbeatReason.RECOVERY,
        )
        cancelled = self.store.list_heartbeat_jobs(statuses=("cancelled",))
        self.assertEqual(len(cancelled), 2)

        self.assertEqual(await heartbeat.run_once(), 0)
        self.assertEqual(self.execute.await_count, 1)


if __name__ == "__main__":
    unittest.main()
