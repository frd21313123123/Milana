import asyncio
import json
import unittest
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from milana import TurnTrigger, empty_turn_payload
from milana_future_actions import FutureActionStore, validate_operations
from milana_heartbeat import MilanaHeartbeat, HeartbeatReason
from milana_life import LifePlanner
from milana_schedule import load_routine
from milana_state import MilanaStateStore
from milana_web import start_web_server
import test_milana_service as service_tests
from test_milana_service import (
    NOW, _final, _call, _direct_telegram_payload,
    _production_telegram_trigger,
)


def operation(**fields):
    return {"arguments_json": json.dumps(fields)}


def intention(**changes):
    return {"intent": "написать после учебы", "trigger_type": "delay",
            "delay_seconds": 60, "is_promise": True, **changes}


class FutureActionStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "state.sqlite"
        self.state = MilanaStateStore(self.path)
        self.store = FutureActionStore(self.state)

    def tearDown(self):
        self.doCleanups()
        self.state.close()
        self.tmp.cleanup()

    def create(self, **changes):
        return self.store.create(intention(**changes), target_id=77, now=NOW)

    def test_create_restart_due_overdue_and_claim_once(self):
        action_id = self.create()
        self.assertIsNone(self.store.claim_due(NOW))
        self.state.close()
        self.state = MilanaStateStore(self.path)
        self.store = FutureActionStore(self.state)
        row = self.store.get(action_id)
        self.assertEqual(row["target_id"], "77")
        self.assertEqual(datetime.fromisoformat(row["due_at"]), NOW + timedelta(minutes=1))
        later = NOW + timedelta(minutes=48)
        self.assertEqual(self.store.snapshot(later)["overdue_count"], 1)
        claimed = self.store.claim_due(later)
        self.assertEqual(claimed["overdue_seconds"], 47 * 60)
        self.assertIsNone(self.store.claim_due(later))

    def test_reschedule_cancel_expiration_and_own_intention(self):
        action_id = self.create(is_promise=False)
        self.assertEqual(self.store.snapshot(NOW + timedelta(days=1))["overdue_count"], 0)
        self.store.reschedule_future_action(action_id, NOW + timedelta(hours=1), now=NOW)
        self.assertIsNone(self.store.claim_due(NOW + timedelta(minutes=5)))
        self.store.cancel(action_id, now=NOW)
        self.assertEqual(self.store.get(action_id)["status"], "cancelled")
        action_id = self.create(expires_at=(NOW + timedelta(minutes=2)).isoformat())
        self.assertIsNone(self.store.claim_due(NOW + timedelta(minutes=3)))
        self.assertEqual(self.store.get(action_id)["status"], "expired")

    def test_all_trigger_types_and_opportunity_conditions(self):
        for trigger in ("datetime", "activity_end", "schedule_event", "opportunity"):
            action_id = self.store.create(intention(trigger_type=trigger, due_at=(NOW + timedelta(minutes=1)).isoformat()),
                target_id=77, now=NOW, activity_end=NOW + timedelta(minutes=2), schedule_end=NOW + timedelta(minutes=3))
            self.assertEqual(self.store.get(action_id)["trigger_type"], trigger)
            self.store.cancel(action_id, now=NOW)
        action_id = self.create(trigger_type="opportunity", earliest_at=NOW.isoformat())
        self.assertIsNone(self.store.claim_due(NOW, busy=True))
        self.assertIsNotNone(self.store.claim_due(NOW + timedelta(minutes=6)))
        self.assertEqual(self.store.get(action_id)["attempts"], 1)

    def test_schedule_event_tracks_linked_life_plan_event(self):
        routine = load_routine()
        planner = LifePlanner(self.state, routine, now=lambda: NOW, enabled=False)
        event = planner.advance(NOW).current
        action_id = self.store.create(
            intention(trigger_type="schedule_event", due_at=(NOW + timedelta(minutes=1)).isoformat()),
            target_id=77, now=NOW, schedule_end=event.actual_end,
            origin={"planned_event_id": event.event_id},
        )
        moved_end = event.actual_end + timedelta(minutes=20)
        with self.state.transaction() as db:
            db.execute("UPDATE planned_events SET actual_end=?,updated_at=? WHERE event_id=?",
                       (moved_end.timestamp(), NOW.timestamp(), event.event_id))
        self.store.refresh_activity_ends()
        self.assertEqual(datetime.fromisoformat(self.store.get(action_id)["due_at"]), moved_end)

        cancelled_at = NOW + timedelta(minutes=2)
        with self.state.transaction() as db:
            db.execute("UPDATE planned_events SET status='cancelled',updated_at=? WHERE event_id=?",
                       (cancelled_at.timestamp(), event.event_id))
        self.store.refresh_activity_ends()
        self.assertEqual(datetime.fromisoformat(self.store.get(action_id)["due_at"]), cancelled_at)

    def test_invalid_output(self):
        invalid = [None, {}, [{"intent": "hi"}], [operation(trigger_type="unknown", intent="hi")],
                   [operation(**intention(is_promise="yes"))], [operation(**intention(delay_seconds=True))],
                   [operation(**intention(due_at="2026-01-01"))], [operation(operation="reschedule", id="x")],
                   [operation(**intention(message="prewritten"))], [operation(**intention(conditions={"eval": "bad"}))]]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_operations(raw)

    def test_migration_preserves_legacy_tables_and_is_repeatable(self):
        # A pre-feature StateStore contains memory/world/outbox tables but no
        # future-action schema. Migrate that database, rather than an empty DB.
        from milana_memory import MilanaMemoryStore
        legacy_path = Path(self.tmp.name) / "legacy.sqlite"
        memory = MilanaMemoryStore(legacy_path)
        memory.add_message(77, "user", "old conversation", telegram_message_id=1)
        legacy = MilanaStateStore(legacy_path)
        self.addCleanup(memory.close)
        self.addCleanup(legacy.close)
        with legacy.transaction() as db:
            db.execute("CREATE TABLE legacy_data(value TEXT)")
            db.execute("INSERT INTO legacy_data VALUES ('keep me')")
            tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            self.assertNotIn("future_actions", tables)
            before = {table: list(map(tuple, db.execute('SELECT * FROM "' + table + '"'))) for table in tables}
        FutureActionStore(legacy)
        FutureActionStore(legacy)
        with legacy.transaction() as db:
            after = {table: list(map(tuple, db.execute('SELECT * FROM "' + table + '"'))) for table in tables}
            self.assertEqual(before, after)
            self.assertEqual(db.execute("SELECT value FROM legacy_data").fetchone()[0], "keep me")
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")


class FutureActionHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_timeout_retry_restart_and_completion(self):
        state = MilanaStateStore()
        self.addCleanup(state.close)
        store = FutureActionStore(state)
        now = [NOW]
        seen = []
        action_id = store.create(intention(delay_seconds=10), target_id=77, now=NOW)

        async def execute(trigger):
            seen.append(trigger)
            if len(seen) == 1:
                raise TimeoutError("temporary Telegram error")
            store.prepare_plan("finish", {}, [{"operation": "complete", "id": action_id}], 77)
            store.finish_plan("finish", now[0], result={"telegram_message_ids": [123]})

        heartbeat = MilanaHeartbeat(state, execute, future_actions=store, now=lambda: now[0], dev_mode=True)
        self.assertEqual(heartbeat._next_timeout(NOW), 10)
        now[0] += timedelta(seconds=10)
        await heartbeat.run_once()
        self.assertEqual(store.get(action_id)["status"], "pending")
        now[0] += timedelta(seconds=6)
        await heartbeat.run_once()
        self.assertEqual(store.get(action_id)["status"], "completed")
        self.assertEqual(store.get(action_id)["result"]["telegram_message_ids"], [123])
        self.assertEqual(seen[0].reason, HeartbeatReason.FUTURE_ACTION)
        self.assertEqual(seen[0].logical_id, seen[1].logical_id)
        store.recover()
        await heartbeat.run_once()
        self.assertEqual(len(seen), 2)


class FutureActionServiceTests(unittest.IsolatedAsyncioTestCase):
    setUp = service_tests.MilanaServiceTests.setUp
    tearDown = service_tests.MilanaServiceTests.tearDown
    service = service_tests.MilanaServiceTests.service

    async def test_model_output_creates_promise_and_incoming_can_reschedule_cancel(self):
        payload = _direct_telegram_payload("напишу через минуту")
        payload["future_actions"] = [operation(**intention())]
        service = self.service([_final(payload)])
        await service._execute_turn(_production_telegram_trigger())
        actions = service.future_actions.list()
        self.assertEqual(len(actions), 1)
        row = actions[0]
        self.assertEqual(row["origin_user_message_id"], 9)
        self.assertEqual(row["origin_milana_message_id"], 10)
        self.assertEqual(row["target_id"], "77")
        self.assertEqual(len(service.model_client.responses.requests), 1)
        for message_id, kind in [(10, "reschedule"), (11, "cancel")]:
            fields = {"operation": kind, "id": row["id"]}
            if kind == "reschedule":
                fields["due_at"] = (NOW + timedelta(hours=2)).isoformat()
            payload = _direct_telegram_payload("хорошо")
            payload["future_actions"] = [operation(**fields)]
            service.model_client.responses.values.append(_final(payload))
            await service._execute_turn(_production_telegram_trigger(message_id))
        self.assertEqual(service.future_actions.get(row["id"])["status"], "cancelled")

    async def test_failed_send_does_not_publish_promise_and_replays_without_model(self):
        payload = _direct_telegram_payload("потом напишу")
        payload["future_actions"] = [operation(**intention())]
        service = self.service([_final(payload)])
        original = self.supervisor.request

        async def failed(method, params, **options):
            if method == "telegram.execute" and params["action"] == "send_messages":
                raise TimeoutError("lost response")
            return await original(method, params, **options)

        self.supervisor.request = failed
        with self.assertRaises(TimeoutError):
            await service._execute_turn(_production_telegram_trigger())
        self.assertEqual(service.future_actions.list(), [])
        self.supervisor.request = original
        restarted = self.service()
        await restarted._execute_turn(_production_telegram_trigger())
        self.assertEqual(len(restarted.future_actions.list()), 1)
        self.assertEqual(len(restarted.model_client.responses.requests), 0)

    async def test_future_execution_retry_uses_same_outbox_and_no_second_model(self):
        service = self.service()
        action_id = service.future_actions.create(intention(delay_seconds=1), target_id=77, now=NOW)
        action = service.future_actions.claim_due(NOW + timedelta(seconds=1))
        payload = empty_turn_payload(telegram=True)
        payload["telegram"] = _direct_telegram_payload("я освободилась")["telegram"]
        payload["future_actions"] = [operation(operation="complete", id=action_id)]
        service.model_client.responses.values.extend([
            _call("open_skill", {"skill_id": "telegram"}, "open"), _final(payload)])
        def trigger():
            return TurnTrigger(kind="future_action", occurred_at=NOW,
                revision=self.state.get_agent_state().revision,
                metadata={"future_action": action, "_telegram_target_ref": 77,
                          "_logical_action_scope": f"future_action:{action_id}:execute:0"})
        original = self.supervisor.request
        keys = []

        async def flaky(method, params, **options):
            if method == "telegram.execute" and params["action"] == "send_messages":
                keys.append(params["arguments"]["batch_id"])
                if len(keys) == 1:
                    raise TimeoutError("lost result")
            return await original(method, params, **options)

        self.supervisor.request = flaky
        with self.assertRaises(TimeoutError):
            await service._execute_turn(trigger())
        self.assertNotEqual(service.future_actions.get(action_id)["status"], "completed")
        restarted = self.service()
        await restarted._execute_turn(trigger())
        await restarted._execute_turn(trigger())
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(len(keys), 2)
        self.assertEqual(restarted.future_actions.get(action_id)["status"], "completed")
        self.assertEqual(len(restarted.model_client.responses.requests), 0)

    async def test_wrong_target_and_invalid_model_output_have_no_effect(self):
        for raw in ([operation(**intention(target_token="forged"))], [operation(**intention(trigger_type="invalid"))]):
            payload = _direct_telegram_payload("обещаю")
            payload["future_actions"] = raw
            service = self.service([_final(payload)])
            with self.assertRaises((PermissionError, ValueError)):
                await service._execute_turn(_production_telegram_trigger())
            self.assertEqual(service.future_actions.list(), [])
            self.assertFalse(any(m == "telegram.execute" and p["action"] == "send_messages" for m, p, _ in self.supervisor.calls))

    async def test_activity_end_uses_scene_transition_and_schedule_end_is_frozen(self):
        service = self.service()
        scene = service.scene.current()
        for index, trigger_type in enumerate(("activity_end", "schedule_event")):
            payload = _direct_telegram_payload("напишу после занятия")
            payload["future_actions"] = [operation(**intention(trigger_type=trigger_type))]
            service.model_client.responses.values.append(_final(payload))
            await service._execute_turn(_production_telegram_trigger(20 + index))
        rows = {a["trigger_type"]: a for a in service.future_actions.list()}
        self.assertEqual(datetime.fromisoformat(rows["schedule_event"]["due_at"]), service._next_transition_at(NOW))
        self.assertEqual(rows["activity_end"]["origin_scene_id"], scene.scene_id)
        service.scene.end()
        service.future_actions.refresh_activity_ends()
        self.assertEqual(datetime.fromisoformat(service.future_actions.get(rows["activity_end"]["id"])["due_at"]), NOW)

    async def test_full_heartbeat_queue_flow_and_model_postpone(self):
        service = self.service()
        action_id = service.future_actions.create(intention(trigger_type="datetime", due_at=NOW.isoformat()), target_id=77, now=NOW)
        payload = empty_turn_payload()
        payload["future_actions"] = [operation(operation="postpone", id=action_id, due_at=(NOW + timedelta(hours=1)).isoformat())]
        service.model_client.responses.values.append(_final(payload))
        # The production callback routes through the existing per-chat worker.
        queue_task = asyncio.create_task(service._queue_loop())
        try:
            self.assertEqual(await asyncio.wait_for(service.heartbeat._run_future_action(NOW), 5), 1)
            action = service.future_actions.get(action_id)
            self.assertEqual(action["status"], "pending")
            self.assertEqual(datetime.fromisoformat(action["due_at"]), NOW + timedelta(hours=1))
            self.assertEqual(action["version"], 1)
            self.assertEqual(len(service.model_client.responses.requests), 1)
        finally:
            queue_task.cancel()
            workers = list(service._worker_tasks.values())
            for task in workers:
                task.cancel()
            await asyncio.gather(queue_task, *workers, return_exceptions=True)

    async def test_revision_failure_does_not_save_plan_or_promise(self):
        service = self.service()
        payload = empty_turn_payload(telegram=True)
        payload["telegram"] = _direct_telegram_payload("потом напишу")["telegram"]
        payload["future_actions"] = [operation(**intention())]
        service.model_client.responses.values.extend([
            _call("open_skill", {"skill_id": "telegram"}, "open"), _final(payload)])
        trigger = TurnTrigger(kind="heartbeat", occurred_at=NOW, revision=999)
        from milana_state import StateConflictError
        with self.assertRaises(StateConflictError):
            await service._execute_turn(trigger)
        self.assertEqual(service.future_actions.list(), [])
        with self.state.transaction() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM future_action_turns").fetchone()[0], 0)

    async def test_completion_and_outbox_roll_back_together(self):
        service = self.service()
        action_id = service.future_actions.create(intention(), target_id=77, now=NOW)
        key = "transaction-test"
        service.future_actions.prepare_plan(key, {}, [{"operation": "complete", "id": action_id}], 77)
        self.state.prepare_telegram_outbox(key, 77, [], ["test"])
        def fail(db, entry):
            service.future_actions.finish_plan(key, NOW, db=db)
            raise OSError("disk failed")
        with self.assertRaises(OSError):
            self.state.advance_telegram_outbox(key, sent_part_indexes=[0], sent_message_ids=[123],
                next_part_index=1, complete=True, on_complete=fail)
        self.assertEqual(service.future_actions.get(action_id)["status"], "pending")
        self.assertEqual(self.state.prepare_telegram_outbox(key, 77, [], ["test"]).status, "pending")

    async def test_noop_and_cancel_need_no_telegram_and_lost_delivery_cannot_be_rescheduled(self):
        service = self.service()
        action_id = service.future_actions.create(intention(), target_id=77, now=NOW)
        action = service.future_actions.claim_due(NOW + timedelta(minutes=1))
        for version, operations in [(0, []), (1, [operation(operation="cancel", id=action_id)])]:
            payload = empty_turn_payload()
            payload["future_actions"] = operations
            service.model_client.responses.values.append(_final(payload))
            trigger = TurnTrigger(kind="future_action", occurred_at=NOW,
                revision=self.state.get_agent_state().revision,
                metadata={"future_action": action, "_logical_action_scope": f"future:{action_id}:{version}"})
            await service._execute_turn(trigger)
            if version == 0:
                self.assertEqual(service.future_actions.get(action_id)["status"], "executing")
        self.assertEqual(service.future_actions.get(action_id)["status"], "cancelled")
        other = service.future_actions.create(intention(), target_id=77, now=NOW)
        service.future_actions.prepare_plan("ambiguous-send", {}, [{"operation": "complete", "id": other}], 77)
        with self.assertRaises(ValueError):
            service.future_actions.reschedule_future_action(other, NOW, now=NOW)
        with self.assertRaises(ValueError):
            service.future_actions.cancel(other, now=NOW)


class FutureActionWebTests(unittest.TestCase):
    def test_read_cancel_execute_and_reschedule(self):
        state = MilanaStateStore()
        store = FutureActionStore(state)
        action_id = store.create(intention(), target_id=77, now=NOW)
        panel = start_web_server(port=0, state_store=state, callbacks={
            "future_actions": lambda: store.snapshot(NOW),
            "cancel_future_action": lambda b: store.cancel(b["id"], now=NOW),
            "reschedule_future_action": lambda b: store.reschedule_future_action(b["id"], b["due_at"], now=NOW),
            "execute_future_action": lambda b: store.reschedule_future_action(b["id"], NOW, now=NOW),
        })
        try:
            with urllib.request.urlopen(panel.url + "api/future-actions") as response:
                self.assertEqual(json.load(response)["actions"][0]["id"], action_id)
            for verb in ("reschedule", "execute", "cancel"):
                body = {"id": action_id, "due_at": (NOW + timedelta(hours=2)).isoformat()}
                req = urllib.request.Request(panel.url + "api/future-actions/" + verb, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req) as response:
                    self.assertTrue(json.load(response)["ok"])
            self.assertEqual(store.get(action_id)["status"], "cancelled")
        finally:
            panel.stop()
            state.close()


if __name__ == "__main__":
    unittest.main()
