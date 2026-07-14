import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

from milana import ToolResult, TurnTrigger, empty_turn_payload
from milana_heartbeat import HeartbeatReason, HeartbeatTrigger
from milana_ipc import MediaPathValidator
from milana_memory import MilanaMemoryStore
from milana_schedule import load_routine
from milana_service import MilanaService, TurnPreemptedError, build_heartbeat_changes
from milana_state import MilanaStateStore, StateConflictError, TelegramTurnMetric
from telegram_client import AIConfig, MessageFlowConfig, TelegramFastPathConfig


NOW = datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc)


class _Responses:
    def __init__(self, values):
        self.values = list(values)
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        return self.values.pop(0)


class _Model:
    def __init__(self, values=()):
        self.responses = _Responses(values)


class _Supervisor:
    def __init__(self):
        self.calls = []
        self.open_text = "привет"

    async def request(self, method, params, **options):
        self.calls.append((method, dict(params), dict(options)))
        if method == "telegram.open":
            incoming = "target_ref" not in params
            return {
                "turn_id": params["turn_id"],
                "target_token": "token-for-turn",
                "target_ref": 77,
                "messages": ([
                    {
                        "message_id": 9,
                        "timestamp": NOW.isoformat(),
                        "sender": {"id": 88, "display_name": "Лера"},
                        "text": self.open_text,
                        "media_type": "text",
                    }
                ] if incoming else []),
                "history": [],
            }
        if method == "telegram.execute":
            action = params["action"]
            if action == "open_sticker_picker":
                return {
                    "status": "ok",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "view": "pack",
                                    "stickers": [
                                        {"sticker_id": "P001:S001", "emoji": "🙂"}
                                    ],
                                }
                            ),
                        }
                    ],
                }
            if action == "send_messages":
                messages = params["arguments"]["messages"]
                start = params["arguments"].get("start_index", 0)
                indexes = list(range(start, len(messages)))
                return {
                    "status": "sent",
                    "sent_message_ids": [10 + index for index in indexes],
                    "sent_part_indexes": indexes,
                    "next_part_index": len(messages),
                    "total_parts": len(messages),
                    "first_send_elapsed_ms": 1.0 if indexes else None,
                }
            if action == "send_sticker":
                return {"status": "sent", "message_id": 11}
            return {"status": "ok"}
        if method == "telegram.presence":
            return {"online": params["online"]}
        if method == "telegram.cleanup_turn":
            return {"cleaned": True}
        raise AssertionError(method)

    async def start(self):
        return None

    async def stop(self):
        return None

    def status(self):
        return {"connected": True}


def _call(name, arguments, call_id):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=json.dumps(arguments),
                call_id=call_id,
            )
        ],
        output_text="",
    )


def _final(payload):
    return SimpleNamespace(output=[], output_text=json.dumps(payload, ensure_ascii=False))


def _direct_telegram_payload(*messages, token="token-for-turn"):
    return {
        "telegram": {
            "target_token": token,
            "messages": list(messages),
            "reaction": None,
            "blacklist_sender": False,
        }
    }


def _production_telegram_trigger(message_id=9):
    notice_id = f"tg:77:{message_id}"
    notice = {
        "source": "telegram",
        "notice_id": notice_id,
        "chat_id": 77,
        "message_id": message_id,
        "timestamp": NOW.isoformat(),
        "sender": {"id": 88, "display_name": "Лера"},
        "media_type": "text",
    }
    return TurnTrigger(
        kind="telegram_notice",
        source_skill="telegram",
        occurred_at=NOW,
        revision=0,
        metadata={
            "chat_id": 77,
            "notice_ids": [notice_id],
            "notices": [notice],
        },
    )


class MilanaServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        path = Path(self.tmp.name) / "milana.sqlite3"
        self.memory = MilanaMemoryStore(path)
        self.state = MilanaStateStore(path)
        self.supervisor = _Supervisor()

    def tearDown(self):
        self.state.close()
        self.memory.close()
        self.tmp.cleanup()

    def service(
        self,
        responses=(),
        *,
        model=None,
        dev_mode=True,
        fast_max_reply_messages=1,
    ):
        config = AIConfig(
            api_key="test",
            model="fake",
            instructions="персона без телеграм инструкций",
            temperature=0.7,
            max_output_tokens=1200,
            message_flow=MessageFlowConfig(
                input_quiet_seconds=0,
                input_max_wait_seconds=0,
                max_reply_messages=5,
                inter_message_min_delay_seconds=0,
                inter_message_max_delay_seconds=0,
            ),
            telegram_fast_path=TelegramFastPathConfig(
                dev_chat_only=False,
                max_reply_messages=fast_max_reply_messages,
            ),
        )
        return MilanaService(
            config=config,
            model_client=model or _Model(responses),
            memory=self.memory,
            state=self.state,
            routine=load_routine(),
            rpc_server=SimpleNamespace(),
            supervisor=self.supervisor,
            dev_mode=dev_mode,
            now=lambda: NOW,
        )

    def test_absolute_need_change_is_bounded_to_fifteen(self):
        payload = empty_turn_payload()
        payload["state_update"]["social"] = 65
        changes = build_heartbeat_changes(payload, self.state.get_agent_state())
        self.assertEqual(changes.need_deltas, {"social": 15})
        payload["state_update"]["social"] = 66
        with self.assertRaises(ValueError):
            build_heartbeat_changes(payload, self.state.get_agent_state())

    async def test_production_ordinary_text_notice_stays_one_call_fast_path(self):
        self.supervisor.open_text = "Как у тебя дела?"
        fast_final = {
            "telegram": {
                "target_token": "token-for-turn",
                "messages": ["всё хорошо"],
                "reaction": None,
                "blacklist_sender": False,
            },
        }
        service = self.service([_final(fast_final)], dev_mode=False)

        result = await service._execute_turn(_production_telegram_trigger())

        self.assertTrue(service.telegram_fast_path_enabled)
        self.assertEqual(len(service.model_client.responses.requests), 1)
        request = service.model_client.responses.requests[0]
        self.assertEqual(request["tools"], [])
        self.assertEqual(
            set(request["text"]["format"]["schema"]["properties"]),
            {"telegram"},
        )
        self.assertNotIn("Доступные корневые навыки", request["instructions"])
        self.assertNotIn("requires_tools", result.trigger.metadata)

    def test_status_is_not_green_when_fast_window_contains_missing_first_sends(self):
        service = self.service()

        def record(
            turn_id: str,
            *,
            outcome: str,
            first_sent: bool,
            offset: int,
        ) -> None:
            started_at = NOW + timedelta(seconds=offset)
            self.state.record_telegram_turn_metric(
                TelegramTurnMetric(
                    turn_id=turn_id,
                    chat_id="77",
                    outcome=outcome,
                    context_ms=1.0,
                    provider_queue_ms=2.0,
                    model_ms=500.0,
                    send_ms=10.0 if first_sent else 0.0,
                    generation_to_first_send_ms=(
                        1_000.0 if first_sent else 300_000.0
                    ),
                    model_rounds=1,
                    context_messages=2,
                    context_characters=20,
                    started_at=started_at,
                    first_sent_at=(
                        started_at + timedelta(seconds=1) if first_sent else None
                    ),
                    sla_eligible=True,
                )
            )

        record("ok", outcome="sent", first_sent=True, offset=0)
        record("error", outcome="error:AgyError", first_sent=False, offset=1)
        record("no-first-send", outcome="sent", first_sent=False, offset=2)

        latency = service.status()["telegram_latency"]

        self.assertEqual(latency["ordinary_text_turns"], 3)
        self.assertEqual(latency["sample_size"], 1)
        self.assertEqual(latency["censored_turns"], 2)
        self.assertEqual(latency["failed_turns"], 2)
        self.assertEqual(latency["error_turns"], 1)
        self.assertEqual(latency["no_first_send_turns"], 1)
        self.assertEqual(latency["delivery_rate"], 1 / 3)
        self.assertTrue(latency["slo_evaluable"])
        self.assertFalse(latency["slo_met"])

    async def test_production_textual_sticker_command_uses_sticker_tools(self):
        self.supervisor.open_text = "Милана, пришли мне стикер, пожалуйста"
        service = self.service(
            [
                _call("open_sticker_picker", {"pack_id": None}, "picker"),
                _call("send_sticker", {"sticker_id": "P001:S001"}, "send"),
                _final(_direct_telegram_payload()),
            ]
        )

        result = await service._execute_turn(_production_telegram_trigger())

        requests = service.model_client.responses.requests
        self.assertEqual(len(requests), 3)
        self.assertEqual(
            {tool["name"] for tool in requests[0]["tools"]},
            {"open_sticker_picker", "send_sticker", "schedule_sticker"},
        )
        self.assertNotIn("state_update", requests[0]["text"]["format"]["schema"]["properties"])
        executed = [
            call[1].get("action")
            for call in self.supervisor.calls
            if call[0] == "telegram.execute"
        ]
        self.assertIn("open_sticker_picker", executed)
        self.assertIn("send_sticker", executed)
        self.assertTrue(result.trigger.metadata["_telegram_sticker_tools"])

    async def test_production_textual_reminder_command_stays_direct(self):
        self.supervisor.open_text = "Remind me in an hour to drink water"
        final = {
            "telegram": {
                "target_token": "token-for-turn",
                "messages": ["я не могу поставить напоминание из этого ответа"],
                "reaction": None,
                "blacklist_sender": False,
            },
        }
        service = self.service([_final(final)])

        result = await service._execute_turn(_production_telegram_trigger())

        requests = service.model_client.responses.requests
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["tools"], [])
        self.assertEqual(result.staged_actions, ())
        self.assertNotIn("_telegram_sticker_tools", result.trigger.metadata)

    async def test_revision_conflict_keeps_every_staged_write_inert(self):
        service = self.service()
        trigger = TurnTrigger(
            kind="heartbeat", occurred_at=NOW, revision=0, id="stale-turn"
        )
        stage = service.staging.begin(trigger)
        stage.add_action("write_diary", {"entry": "не должна сохраниться"})
        stage = service.staging.finish(trigger.id)
        self.state.apply_heartbeat_changes(
            build_heartbeat_changes(empty_turn_payload(), self.state.get_agent_state()),
            expected_revision=0,
            at=NOW,
        )
        result = SimpleNamespace(payload=empty_turn_payload(), trigger=trigger)
        with self.assertRaises(StateConflictError):
            await service._commit_turn(result, stage)
        self.assertEqual(self.memory.get_diary(), [])
        self.assertEqual(self.supervisor.calls, [])

    async def test_failed_notice_is_released_for_unread_backfill(self):
        service = self.service()
        service.agent.run_turn = AsyncMock(side_effect=RuntimeError("model offline"))
        service._seen_notices.add("tg:77:9")
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={"chat_id": 77, "notice_ids": ["tg:77:9"], "notices": []},
        )

        with self.assertRaises(RuntimeError):
            await service._execute_turn(trigger)

        self.assertNotIn("tg:77:9", service._seen_notices)

    async def test_poison_materialization_failure_does_not_block_healthy_notice(self):
        service = self.service()
        notice_ids = ("tg:77:9", "tg:77:10")
        for message_id, notice_id in zip((9, 10), notice_ids, strict=True):
            self.state.record_telegram_notice(
                {
                    "source": "telegram",
                    "notice_id": notice_id,
                    "chat_id": 77,
                    "message_id": message_id,
                    "timestamp": NOW.isoformat(),
                    "sender": {"id": 88, "display_name": "Лера"},
                    "media_type": "text",
                },
                received_at=NOW,
            )
            service._seen_notices.add(notice_id)
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={
                "chat_id": 77,
                "notice_ids": list(notice_ids),
                "notices": [
                    {"notice_id": notice_id, "media_type": "text"}
                    for notice_id in notice_ids
                ],
            },
        )

        service._defer_failed_telegram_turn(
            trigger,
            "Telegram notice could not be materialized: tg:77:9",
        )

        healthy = service._turn_queue.get_nowait()
        self.assertEqual(healthy.metadata["notice_ids"], ["tg:77:10"])
        self.assertEqual(
            [item["notice_id"] for item in healthy.metadata["notices"]],
            ["tg:77:10"],
        )
        self.assertEqual(
            self.state.telegram_notice_attempt_count(["tg:77:9"]), 1
        )
        self.assertEqual(
            self.state.telegram_notice_attempt_count(["tg:77:10"]), 0
        )
        self.assertNotIn("tg:77:9", service._seen_notices)
        self.assertIn("tg:77:10", service._seen_notices)

    async def test_pending_notice_is_restored_from_durable_journal(self):
        service = self.service()
        service.dev_mode = True
        payload = {
            "source": "telegram",
            "notice_id": "tg:77:12",
            "chat_id": 77,
            "message_id": 12,
            "timestamp": NOW.isoformat(),
            "sender": {"id": 88, "display_name": "Лера"},
            "media_type": "text",
        }
        self.state.record_telegram_notice(payload, received_at=NOW)

        await service._restore_pending_telegram_notices()
        trigger = await asyncio.wait_for(service._turn_queue.get(), timeout=1)

        self.assertEqual(trigger.metadata["notice_ids"], ["tg:77:12"])
        self.assertEqual(trigger.metadata["notices"], [payload])
        service._turn_queue.task_done()

    async def test_new_notice_does_not_cancel_commit_phase(self):
        service = self.service()
        release = asyncio.Event()

        async def committing():
            await release.wait()

        active_task = asyncio.create_task(committing())
        active_trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=0,
            metadata={
                "chat_id": 77,
                "notice_ids": ["tg:77:8"],
                "notices": [{"notice_id": "tg:77:8"}],
            },
        )
        service._active_turn_tasks["77"] = active_task
        service._active_triggers["77"] = active_trigger
        service._turn_phases["77"] = "commit"
        params = {
            "source": "telegram",
            "notice_id": "tg:77:9",
            "chat_id": 77,
            "message_id": 9,
            "timestamp": NOW.isoformat(),
            "sender": {"id": 88, "display_name": "Лера"},
            "media_type": "text",
        }

        await service._rpc_telegram_notice(params, SimpleNamespace())
        quiet_task = service._notice_tasks["77"]
        quiet_task.cancel()
        await asyncio.gather(quiet_task, return_exceptions=True)

        self.assertFalse(active_task.cancelled())
        self.assertEqual(
            [item["notice_id"] for item in service._notice_buffers["77"]],
            ["tg:77:9"],
        )
        release.set()
        await active_task

    async def test_new_notice_preempts_generation_and_retries_heartbeat(self):
        service = self.service()
        started = asyncio.Event()

        async def blocking_turn(_trigger):
            started.set()
            await asyncio.Event().wait()

        service.agent.run_turn = AsyncMock(side_effect=blocking_turn)
        queue = asyncio.Queue()
        completion = asyncio.get_running_loop().create_future()
        trigger = TurnTrigger(
            kind="heartbeat",
            occurred_at=NOW,
            revision=0,
            metadata={"_completion_future": completion},
        )
        worker = asyncio.create_task(service._worker_loop("__life__", queue))
        await queue.put(trigger)
        await asyncio.wait_for(started.wait(), timeout=1)

        params = {
            "source": "telegram",
            "notice_id": "tg:77:10",
            "chat_id": 77,
            "message_id": 10,
            "timestamp": NOW.isoformat(),
            "sender": {"id": 88, "display_name": "Лера"},
            "media_type": "text",
        }
        await service._rpc_telegram_notice(params, SimpleNamespace())
        quiet_task = service._notice_tasks["77"]
        quiet_task.cancel()
        await asyncio.gather(quiet_task, return_exceptions=True)

        with self.assertRaises(TurnPreemptedError):
            await asyncio.wait_for(completion, timeout=1)
        self.assertNotIn("__life__", service._active_turn_tasks)

        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    async def test_failed_telegram_generation_is_deferred_without_immediate_retry(self):
        service = self.service()
        calls = []
        failed = asyncio.Event()

        async def execute(trigger):
            calls.append(trigger)
            failed.set()
            raise ValueError("invalid telegram draft")

        service._execute_turn = execute
        queue = asyncio.Queue()
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=0,
            metadata={
                "chat_id": 77,
                "notice_ids": ["tg:77:10"],
                "notices": [{"notice_id": "tg:77:10"}],
            },
        )
        self.state.record_telegram_notice(
            {
                "source": "telegram",
                "notice_id": "tg:77:10",
                "chat_id": 77,
                "message_id": 10,
                "timestamp": NOW.isoformat(),
                "sender": {"id": 88, "display_name": "Лера"},
                "media_type": "text",
            },
            received_at=NOW,
        )
        worker = asyncio.create_task(service._worker_loop("77", queue))
        await queue.put(trigger)
        await asyncio.wait_for(failed.wait(), timeout=1)
        for _ in range(20):
            if self.state.telegram_notice_attempt_count(["tg:77:10"]) == 1:
                break
            await asyncio.sleep(0)

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            self.state.telegram_notice_attempt_count(["tg:77:10"]), 1
        )
        self.assertNotIn("tg:77:10", service._seen_notices)

        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    async def test_notice_opens_telegram_then_commits_reply_and_read(self):
        final = _direct_telegram_payload("приветик")
        service = self.service(
            [
                _final(final),
            ]
        )
        notice_payload = {
            "source": "telegram",
            "notice_id": "tg:77:9",
            "chat_id": 77,
            "message_id": 9,
            "timestamp": NOW.isoformat(),
            "sender": {"id": 88, "display_name": "Лера"},
            "media_type": "text",
        }
        self.state.record_telegram_notice(notice_payload, received_at=NOW)
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={"chat_id": 77, "notice_ids": ["tg:77:9"], "notices": []},
        )
        result = await service._execute_turn(trigger)

        self.assertEqual(result.active_skills, ("telegram",))
        self.assertIsNotNone(result.validated_changes)
        self.assertEqual(result.staged_actions, ())
        self.assertEqual(len(service.model_client.responses.requests), 1)
        methods = [item[0] for item in self.supervisor.calls]
        self.assertEqual(methods[0], "telegram.open")
        actions = [
            item[1].get("action")
            for item in self.supervisor.calls
            if item[0] == "telegram.execute"
        ]
        self.assertEqual(actions, ["typing", "send_messages", "acknowledge"])
        history = self.memory.get_chat_history(77)
        self.assertEqual([item.role for item in history], ["user", "assistant"])
        self.assertEqual(history[0].content, "привет")
        self.assertEqual(history[1].content, "приветик")
        audit = self.state.list_skill_audit(turn_id=trigger.id)
        # Direct application routing is infrastructure, not a model skill call.
        self.assertEqual(audit, [])
        self.assertEqual(self.state.list_pending_telegram_notices(), [])
        self.assertEqual(
            self.state.record_telegram_notice(notice_payload, received_at=NOW),
            "handled",
        )

    async def test_lost_ack_rpc_is_recovered_without_regenerating_answer(self):
        class LostAckSupervisor(_Supervisor):
            def __init__(self):
                super().__init__()
                self.ack_attempts = 0
                self.remote_ack_applied = False
                self.host_restarts = 0

            async def request(self, method, params, **options):
                if method == "telegram.open" and str(params["turn_id"]).startswith(
                    "ack-recovery-"
                ):
                    self.host_restarts += 1
                if (
                    method == "telegram.execute"
                    and params.get("action") == "acknowledge"
                ):
                    self.calls.append((method, dict(params), dict(options)))
                    self.ack_attempts += 1
                    if self.ack_attempts == 1:
                        # Telegram advanced the read marker, then the host died
                        # before its JSON-RPC result reached the service.
                        self.remote_ack_applied = True
                        raise TimeoutError("ack response lost; host exited")
                    self.remote_ack_applied = True
                    return {"status": "acknowledged", "through_message_id": 9}
                return await super().request(method, params, **options)

        final = _direct_telegram_payload("один настоящий ответ")
        notice_payload = {
            "source": "telegram",
            "notice_id": "tg:77:9",
            "chat_id": 77,
            "message_id": 9,
            "timestamp": NOW.isoformat(),
            "sender": {"id": 88, "display_name": "Лера"},
            "media_type": "text",
        }
        self.state.record_telegram_notice(notice_payload, received_at=NOW)
        self.supervisor = LostAckSupervisor()
        service = self.service([_final(final)])
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={
                "chat_id": 77,
                "notice_ids": ["tg:77:9"],
                "notices": [],
            },
        )

        await service._execute_turn(trigger)
        for _ in range(100):
            if not self.state.list_pending_telegram_ack_intents():
                break
            await asyncio.sleep(0.01)

        self.assertTrue(self.supervisor.remote_ack_applied)
        self.assertEqual(self.supervisor.ack_attempts, 2)
        self.assertEqual(self.supervisor.host_restarts, 1)
        self.assertEqual(len(service.model_client.responses.requests), 1)
        send_calls = [
            call
            for call in self.supervisor.calls
            if call[0] == "telegram.execute"
            and call[1].get("action") == "send_messages"
        ]
        self.assertEqual(len(send_calls), 1)
        self.assertEqual(
            [
                item.content
                for item in self.memory.get_chat_history(77)
                if item.role == "assistant"
            ],
            ["один настоящий ответ"],
        )
        self.assertEqual(self.state.list_pending_telegram_notices(), [])
        self.assertEqual(self.state.list_pending_telegram_ack_intents(), [])
        self.assertEqual(
            self.state.record_telegram_notice(notice_payload, received_at=NOW),
            "handled",
        )

    async def test_telegram_activation_reveals_existing_durable_chat_memory(self):
        self.memory.add_message(
            77,
            "user",
            "я давно люблю зелёный чай",
            telegram_message_id=3,
            sender_name="Лера",
        )
        service = self.service(
            [
                _final(_direct_telegram_payload()),
            ]
        )
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={"chat_id": 77, "notice_ids": ["tg:77:9"], "notices": []},
        )

        await service._execute_turn(trigger)

        first_input = service.model_client.responses.requests[0]["input"]
        self.assertIn("я давно люблю зелёный чай", str(first_input))

    async def test_direct_sticker_route_does_not_record_model_skill_activation(self):
        self.supervisor.open_text = "пришли стикер"
        service = self.service([_final(_direct_telegram_payload())])
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={
                "chat_id": 77,
                "notice_ids": ["tg:77:9"],
                "notices": [],
            },
        )

        await service._execute_turn(trigger)

        self.assertEqual(self.state.list_skill_audit(turn_id=trigger.id), [])

    async def test_heartbeat_updates_life_without_opening_telegram(self):
        payload = empty_turn_payload()
        payload["state_update"]["mood_label"] = "спокойное"
        service = self.service([_final(payload)])
        trigger = TurnTrigger(
            kind="heartbeat",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
        )
        result = await service._execute_turn(trigger)
        self.assertEqual(result.active_skills, ())
        self.assertEqual(self.supervisor.calls, [])
        self.assertEqual(self.state.get_agent_state().last_heartbeat_at, NOW)

    async def test_heartbeat_can_open_telegram_and_initiate_once(self):
        self.state.create_entity(
            "person", "Лера", entity_id="telegram:77", is_real=True, at=NOW
        )
        self.state.upsert_relationship("telegram:77", at=NOW)
        final = empty_turn_payload(telegram=True)
        final["telegram"] = {
            "target_token": "token-for-turn",
            "messages": ["как ты там"],
            "reaction": None,
            "blacklist_sender": False,
        }
        service = self.service(
            [
                _call("open_skill", {"skill_id": "telegram"}, "open"),
                _final(final),
            ]
        )
        trigger = TurnTrigger(
            kind="heartbeat",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={"_telegram_target_ref": 77},
        )
        await service._execute_turn(trigger)
        self.assertTrue(self.state.get_relationship("telegram:77").awaiting_reply)
        open_call = next(call for call in self.supervisor.calls if call[0] == "telegram.open")
        self.assertEqual(open_call[1]["target_ref"], 77)

    async def test_initiative_resumes_pending_target_outbox_after_restart(self):
        self.state.create_entity(
            "person", "Лера", entity_id="telegram:77", is_real=True, at=NOW
        )
        self.state.upsert_relationship("telegram:77", at=NOW)
        self.state.prepare_telegram_outbox(
            "initiative-before-restart", 77, [], ["старое принятое сообщение"]
        )
        generated = empty_turn_payload(telegram=True)
        generated["telegram"] = {
            "target_token": "token-for-turn",
            "messages": ["новый черновик нельзя отправлять"],
            "reaction": None,
            "blacklist_sender": False,
        }
        service = self.service(
            [
                _call("open_skill", {"skill_id": "telegram"}, "open"),
                _final(generated),
            ]
        )

        await service._execute_turn(
            TurnTrigger(
                kind="heartbeat",
                occurred_at=NOW,
                revision=self.state.get_agent_state().revision,
                metadata={"_telegram_target_ref": 77},
            )
        )

        send = next(
            item
            for item in self.supervisor.calls
            if item[0] == "telegram.execute"
            and item[1].get("action") == "send_messages"
        )
        self.assertEqual(
            send[1]["arguments"]["messages"], ["старое принятое сообщение"]
        )
        self.assertEqual(
            send[1]["arguments"]["batch_id"], "initiative-before-restart"
        )
        self.assertIsNone(
            self.state.find_pending_telegram_outbox_for_target(77)
        )

    async def test_telegram_then_stickers_picker_then_send(self):
        final = _direct_telegram_payload()
        self.supervisor.open_text = "пришли мне стикер"
        service = self.service(
            [
                _call("open_sticker_picker", {"pack_id": None}, "packs"),
                _call("open_sticker_picker", {"pack_id": "P001"}, "picker"),
                _call("send_sticker", {"sticker_id": "P001:S001"}, "send"),
                _final(final),
            ]
        )
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={
                "chat_id": 77,
                "notice_ids": ["tg:77:9"],
                "notices": [{"media_type": "sticker"}],
            },
        )
        result = await service._execute_turn(trigger)
        self.assertEqual(result.active_skills, ("telegram", "telegram.stickers"))
        actions = [
            call[1].get("action")
            for call in self.supervisor.calls
            if call[0] == "telegram.execute"
        ]
        self.assertEqual(
            actions,
            [
                "typing",
                "open_sticker_picker",
                "open_sticker_picker",
                "send_sticker",
                "acknowledge",
            ],
        )
        self.assertEqual(
            [item.role for item in self.memory.get_chat_history(77)],
            ["user"],
        )

    async def test_sticker_only_initiative_waits_for_reply(self):
        self.state.create_entity(
            "person", "Лера", entity_id="telegram:77", is_real=True, at=NOW
        )
        self.state.upsert_relationship("telegram:77", at=NOW)
        service = self.service(
            [
                _call("open_skill", {"skill_id": "telegram"}, "tg"),
                _call(
                    "open_skill", {"skill_id": "telegram.stickers"}, "stickers"
                ),
                _call("open_sticker_picker", {"pack_id": None}, "packs"),
                _call("open_sticker_picker", {"pack_id": "P001"}, "picker"),
                _call("send_sticker", {"sticker_id": "P001:S001"}, "send"),
                _final(empty_turn_payload(telegram=True)),
            ]
        )
        trigger = TurnTrigger(
            kind="heartbeat",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={"_telegram_target_ref": 77},
        )

        await service._execute_turn(trigger)

        self.assertTrue(self.state.get_relationship("telegram:77").awaiting_reply)

    async def test_lost_initiative_sticker_response_reuses_heartbeat_action_key(self):
        class LostStickerResponseSupervisor(_Supervisor):
            def __init__(self):
                super().__init__()
                self.sticker_keys = []
                self.remote_sticker_count = 0

            async def request(self, method, params, **options):
                if (
                    method == "telegram.execute"
                    and params.get("action") == "send_sticker"
                ):
                    self.calls.append((method, dict(params), dict(options)))
                    key = options.get("idempotency_key")
                    self.sticker_keys.append(key)
                    if len(self.sticker_keys) == 1:
                        self.remote_sticker_count += 1
                        raise TimeoutError("sticker was sent; RPC response was lost")
                    if key != self.sticker_keys[0]:
                        self.remote_sticker_count += 1
                    return {"status": "sent", "deduplicated": True}
                return await super().request(method, params, **options)

        def sticker_turn(prefix, *, insert_unrelated_action=False):
            calls = [
                _call("open_skill", {"skill_id": "telegram"}, f"{prefix}-tg"),
                _call(
                    "open_skill",
                    {"skill_id": "telegram.stickers"},
                    f"{prefix}-stickers",
                ),
                _call(
                    "open_sticker_picker", {"pack_id": None}, f"{prefix}-picker"
                ),
            ]
            if insert_unrelated_action:
                calls.append(
                    _call(
                        "write_diary",
                        {"entry": "несвязанное действие второго хода"},
                        f"{prefix}-diary",
                    )
                )
            calls.extend(
                [
                _call(
                    "send_sticker",
                    {"sticker_id": "P001:S001"},
                    f"{prefix}-send",
                ),
                _final(empty_turn_payload(telegram=True)),
                ]
            )
            return calls

        self.state.create_entity(
            "person", "Лера", entity_id="telegram:77", is_real=True, at=NOW
        )
        self.state.upsert_relationship("telegram:77", at=NOW)
        self.supervisor = LostStickerResponseSupervisor()
        service = self.service(
            sticker_turn("first")
            + sticker_turn("retry", insert_unrelated_action=True)
        )
        job = service.heartbeat.wake_now(
            idempotency_key="initiative-sticker-job"
        )
        first_claim = self.state.claim_due_heartbeat_jobs(NOW)
        self.assertEqual([item.id for item in first_claim], [job.id])
        router = asyncio.create_task(service._queue_loop())
        try:
            self.assertFalse(
                await service.heartbeat._execute_job(first_claim[0], NOW)
            )
            retry_at = NOW + timedelta(seconds=5)
            retry_claim = self.state.claim_due_heartbeat_jobs(retry_at)
            self.assertEqual([item.id for item in retry_claim], [job.id])

            self.assertTrue(
                await service.heartbeat._execute_job(retry_claim[0], retry_at)
            )

            self.assertEqual(len(self.supervisor.sticker_keys), 2)
            self.assertEqual(
                self.supervisor.sticker_keys[0], self.supervisor.sticker_keys[1]
            )
            self.assertEqual(self.supervisor.remote_sticker_count, 1)
            self.assertTrue(
                self.state.get_relationship("telegram:77").awaiting_reply
            )
        finally:
            service._stopping = True
            router.cancel()
            for task in service._worker_tasks.values():
                task.cancel()
            await asyncio.gather(
                router, *service._worker_tasks.values(), return_exceptions=True
            )

    async def test_deliberate_no_reply_still_records_and_acknowledges_notice(self):
        service = self.service(
            [
                _final(_direct_telegram_payload()),
            ]
        )
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={"chat_id": 77, "notice_ids": ["tg:77:9"], "notices": []},
        )

        await service._execute_turn(trigger)

        actions = [
            call[1].get("action")
            for call in self.supervisor.calls
            if call[0] == "telegram.execute"
        ]
        self.assertEqual(actions, ["typing", "acknowledge"])
        self.assertEqual(self.memory.get_chat_history(77)[0].content, "привет")

    async def test_hanging_presence_never_delays_network_send(self):
        service = self.service()
        never = asyncio.Event()

        async def hanging_presence():
            await never.wait()

        service._show_online = hanging_presence
        trigger = TurnTrigger(kind="heartbeat", occurred_at=NOW, revision=0)
        stage = service.staging.begin(trigger)
        try:
            outcome = await asyncio.wait_for(
                service._host_action(
                    stage,
                    "token-for-turn",
                    "send_messages",
                    {"messages": ["сразу"]},
                    "presence-does-not-block",
                ),
                timeout=0.1,
            )
            self.assertEqual(outcome["status"], "sent")
        finally:
            service.staging.discard(trigger.id)
            for task in tuple(service._cosmetic_tasks):
                task.cancel()
            await asyncio.gather(
                *tuple(service._cosmetic_tasks), return_exceptions=True
            )

    async def test_web_host_restart_cannot_outlive_service_stop(self):
        class RacingSupervisor(_Supervisor):
            def __init__(self):
                super().__init__()
                self.restart_stop_entered = asyncio.Event()
                self.restart_stop_cancelled = asyncio.Event()
                self.stop_calls = 0
                self.start_calls = 0

            async def stop(self):
                self.stop_calls += 1
                if self.stop_calls != 1:
                    return
                self.restart_stop_entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.restart_stop_cancelled.set()
                    raise

            async def start(self):
                self.start_calls += 1

        self.supervisor = RacingSupervisor()
        service = self.service()
        service.rpc_server = SimpleNamespace(close=AsyncMock())

        service._queue_telegram_host_restart()
        await asyncio.wait_for(
            self.supervisor.restart_stop_entered.wait(), timeout=1
        )
        restart_task = next(iter(service._management_tasks))

        await asyncio.wait_for(service.stop(), timeout=1)

        self.assertTrue(self.supervisor.restart_stop_cancelled.is_set())
        self.assertTrue(restart_task.cancelled())
        self.assertEqual(self.supervisor.start_calls, 0)
        self.assertEqual(service._management_tasks, set())

        # A callback already queued by the web thread after shutdown is a no-op.
        service._queue_telegram_host_restart()
        await asyncio.sleep(0)
        self.assertEqual(self.supervisor.start_calls, 0)
        self.assertEqual(service._management_tasks, set())

    async def test_partial_send_resumes_after_restart_without_provider(self):
        class PartialThenResumeSupervisor(_Supervisor):
            def __init__(self):
                super().__init__()
                self.send_attempts = 0

            async def request(self, method, params, **options):
                if (
                    method == "telegram.execute"
                    and params.get("action") == "send_messages"
                ):
                    self.calls.append((method, dict(params), dict(options)))
                    self.send_attempts += 1
                    messages = params["arguments"]["messages"]
                    start = params["arguments"]["start_index"]
                    if self.send_attempts == 1:
                        return {
                            "status": "partial",
                            "sent_message_ids": [501],
                            "sent_part_indexes": [0],
                            "deduplicated_part_indexes": [],
                            "next_part_index": 1,
                            "total_parts": len(messages),
                            "error": "connection dropped after part 0",
                        }
                    if start != 1:
                        raise AssertionError(f"resume started at part {start}, not 1")
                    return {
                        "status": "sent",
                        "sent_message_ids": [502],
                        "sent_part_indexes": [1],
                        "deduplicated_part_indexes": [],
                        "next_part_index": len(messages),
                        "total_parts": len(messages),
                    }
                return await super().request(method, params, **options)

        first = _direct_telegram_payload("первая часть", "вторая часть")
        self.supervisor = PartialThenResumeSupervisor()
        initial_model = _Model([_final(first)])
        service = self.service(model=initial_model, fast_max_reply_messages=2)

        def trigger():
            return TurnTrigger(
                kind="telegram_notice",
                occurred_at=NOW,
                revision=self.state.get_agent_state().revision,
                metadata={
                    "chat_id": 77,
                    "notice_ids": ["tg:77:9"],
                    # Use two configured semantic parts to exercise durable resume.
                    "notices": [{"media_type": "sticker"}],
                },
            )

        with self.assertRaisesRegex(RuntimeError, "connection dropped"):
            await service._execute_turn(trigger())

        pending = self.state.find_telegram_outbox_for_notice_ids(["tg:77:9"])
        self.assertIsNotNone(pending)
        self.assertEqual(pending.status, "pending")
        self.assertEqual(pending.next_part_index, 1)
        self.assertEqual(pending.messages, ("первая часть", "вторая часть"))
        self.assertEqual(pending.message_id_for_part(0), 501)

        # Simulate a process restart while Gemini is unavailable.  Recovery
        # must use the durable frozen outbox and never reach the provider.
        offline_model = _Model()
        offline_model.responses.create = AsyncMock(
            side_effect=RuntimeError("provider offline")
        )
        service = self.service(model=offline_model, fast_max_reply_messages=2)
        await service._execute_turn(trigger())
        offline_model.responses.create.assert_not_awaited()
        self.assertEqual(len(initial_model.responses.requests), 1)

        sent = self.state.find_telegram_outbox_for_notice_ids(["tg:77:9"])
        self.assertIsNotNone(sent)
        self.assertEqual(sent.status, "sent")
        self.assertEqual(sent.next_part_index, 2)
        self.assertEqual(
            [sent.message_id_for_part(index) for index in range(2)],
            [501, 502],
        )
        send_calls = [
            call
            for call in self.supervisor.calls
            if call[0] == "telegram.execute"
            and call[1].get("action") == "send_messages"
        ]
        self.assertEqual(
            [call[1]["arguments"]["start_index"] for call in send_calls],
            [0, 1],
        )
        self.assertEqual(
            [call[1]["arguments"]["messages"] for call in send_calls],
            [
                ["первая часть", "вторая часть"],
                ["первая часть", "вторая часть"],
            ],
        )
        self.assertEqual(
            [
                item.content
                for item in self.memory.get_chat_history(77)
                if item.role == "assistant"
            ],
            ["первая часть", "вторая часть"],
        )

    async def test_lost_send_rpc_reuses_batch_and_accepts_host_deduplication(self):
        class LostResponseSupervisor(_Supervisor):
            def __init__(self):
                super().__init__()
                self.send_attempts = 0

            async def request(self, method, params, **options):
                if (
                    method == "telegram.execute"
                    and params.get("action") == "send_messages"
                ):
                    self.calls.append((method, dict(params), dict(options)))
                    self.send_attempts += 1
                    if self.send_attempts == 1:
                        # Telegram accepted the deterministic random_id, but the
                        # RPC result was lost before the service could persist it.
                        raise TimeoutError("lost RPC response")
                    return {
                        "status": "sent",
                        "sent_message_ids": [],
                        "sent_part_indexes": [0],
                        "deduplicated_part_indexes": [0],
                        "next_part_index": 1,
                        "total_parts": 1,
                    }
                return await super().request(method, params, **options)

        original = _direct_telegram_payload("единственный настоящий ответ")
        regenerated = _direct_telegram_payload("новый текст после таймаута")
        self.supervisor = LostResponseSupervisor()
        service = self.service([_final(original), _final(regenerated)])

        def trigger():
            return TurnTrigger(
                kind="telegram_notice",
                occurred_at=NOW,
                revision=self.state.get_agent_state().revision,
                metadata={
                    "chat_id": 77,
                    "notice_ids": ["tg:77:9"],
                    "notices": [],
                },
            )

        with self.assertRaisesRegex(TimeoutError, "lost RPC response"):
            await service._execute_turn(trigger())

        pending = self.state.find_telegram_outbox_for_notice_ids(["tg:77:9"])
        self.assertIsNotNone(pending)
        self.assertEqual(pending.next_part_index, 0)
        self.assertEqual(pending.messages, ("единственный настоящий ответ",))

        await service._execute_turn(trigger())

        send_calls = [
            call
            for call in self.supervisor.calls
            if call[0] == "telegram.execute"
            and call[1].get("action") == "send_messages"
        ]
        self.assertEqual(len(send_calls), 2)
        self.assertEqual(
            [call[1]["arguments"]["start_index"] for call in send_calls],
            [0, 0],
        )
        self.assertEqual(
            send_calls[0][1]["arguments"]["batch_id"],
            send_calls[1][1]["arguments"]["batch_id"],
        )
        self.assertEqual(
            send_calls[1][1]["arguments"]["messages"],
            ["единственный настоящий ответ"],
        )
        outbox = self.state.find_telegram_outbox_for_notice_ids(["tg:77:9"])
        self.assertIsNotNone(outbox)
        self.assertEqual(outbox.status, "sent")
        self.assertIsNone(outbox.message_id_for_part(0))
        assistant = [
            item
            for item in self.memory.get_chat_history(77)
            if item.role == "assistant"
        ]
        self.assertEqual(
            [item.content for item in assistant],
            ["единственный настоящий ответ"],
        )
        self.assertIsInstance(assistant[0].telegram_message_id, int)
        self.assertLess(assistant[0].telegram_message_id, 0)

    def test_tool_result_media_is_validated_and_encoded_for_the_model(self):
        service = self.service()
        root = Path(self.tmp.name) / "turn-media"
        root.mkdir()
        image = root / "photo.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        service.media_paths = MediaPathValidator(root)
        result = ToolResult(
            name="open_skill",
            output={
                "context": {
                    "messages": [
                        {
                            "media_path": str(image),
                            "media_mime_type": "image/png",
                        }
                    ]
                }
            },
        )

        content = service._tool_result_media(result)

        self.assertEqual(content[0]["type"], "input_image")
        self.assertTrue(content[0]["image_url"].startswith("data:image/png;base64,"))

    async def test_delayed_message_survives_memory_reopen_and_is_delivered(self):
        path = Path(self.memory.path)
        scheduled = self.memory.schedule_pulse_message(77, "я не потерялась", due_at=NOW)
        self.memory.close()
        self.memory = MilanaMemoryStore(path)
        claimed = self.memory.claim_due_pulse_tasks(NOW)
        self.assertEqual([task.id for task in claimed], [scheduled.id])
        service = self.service()
        await service._deliver_delayed_action(claimed[0])
        execute = next(
            call
            for call in self.supervisor.calls
            if call[0] == "telegram.execute"
            and call[1].get("action") == "send_messages"
        )
        self.assertEqual(execute[1]["arguments"]["messages"], ["я не потерялась"])
        self.assertEqual(execute[2]["idempotency_key"], f"delayed:{scheduled.id}")

    async def test_retried_schedule_reuses_original_due_at(self):
        service = self.service()
        clock = [NOW]
        service._now = lambda: clock[0]
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=0,
            metadata={"notice_ids": ["tg:77:9"]},
        )
        stage = service.staging.begin(trigger)
        action = stage.add_action(
            "schedule_message",
            {
                "target": {"target_ref": 77, "messages": []},
                "message": "напомню позже",
                "delay_seconds": 120,
            },
        )
        try:
            await service._commit_staged_action(stage, action)
            first_due = self.memory.get_pulse_task(action.idempotency_key).due_at
            clock[0] = NOW + timedelta(minutes=1)

            await service._commit_staged_action(stage, action)

            retried = self.memory.get_pulse_task(action.idempotency_key)
            self.assertEqual(retried.due_at, first_due)
            self.assertEqual(first_due, NOW + timedelta(seconds=120))
        finally:
            service.staging.discard(trigger.id)

    async def test_heartbeat_retry_keeps_delayed_action_key_across_turn_ids(self):
        service = self.service()
        clock = [NOW]
        service._now = lambda: clock[0]
        keys = []
        for turn_id in ("initiative-attempt-one", "initiative-attempt-two"):
            trigger = TurnTrigger(
                kind="heartbeat",
                occurred_at=clock[0],
                revision=self.state.get_agent_state().revision,
                metadata={
                    "notice_ids": [],
                    "_logical_action_scope": "heartbeat-job:persisted-42",
                },
                id=turn_id,
            )
            stage = service.staging.begin(trigger)
            try:
                action = stage.add_action(
                    "schedule_message",
                    {
                        "target": {"target_ref": 77, "messages": []},
                        "message": "напомню позже",
                        "delay_seconds": 120,
                    },
                )
                keys.append(action.idempotency_key)
                await service._commit_staged_action(stage, action)
            finally:
                service.staging.discard(trigger.id)
            clock[0] += timedelta(minutes=1)

        self.assertEqual(keys[0], keys[1])
        tasks = self.memory.get_pulse_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].id, keys[0])
        self.assertEqual(tasks[0].due_at, NOW + timedelta(seconds=120))

    async def test_failed_host_status_keeps_delayed_action_retryable(self):
        class FailedSupervisor(_Supervisor):
            async def request(self, method, params, **options):
                if method == "telegram.execute":
                    self.calls.append((method, dict(params), dict(options)))
                    return {"status": "failed", "error": "network lost"}
                return await super().request(method, params, **options)

        scheduled = self.memory.schedule_pulse_message(
            77, "дождусь связи", due_at=NOW
        )
        task = self.memory.claim_due_pulse_tasks(NOW)[0]
        self.supervisor = FailedSupervisor()
        service = self.service()

        with self.assertRaisesRegex(RuntimeError, "network lost"):
            await service._deliver_delayed_action(task)

        self.assertEqual(
            self.state.list_heartbeat_jobs(statuses=("pending",)),
            [],
        )
        self.memory.retry_pulse_task(
            scheduled.id,
            error="network lost",
            retry_at=NOW,
            max_attempts=5,
        )
        self.assertEqual(self.memory.get_pulse_tasks()[0].status, "pending")

    async def test_large_backfill_is_split_into_host_safe_turns(self):
        service = self.service()
        service.dev_mode = True
        loop = asyncio.get_running_loop()
        notices = [
            {
                "source": "telegram",
                "notice_id": f"tg:77:{index}",
                "chat_id": 77,
                "message_id": index,
                "timestamp": NOW.isoformat(),
                "sender": {"id": 88, "display_name": "Лера"},
                "media_type": "text",
            }
            for index in range(1, 206)
        ]
        service._notice_buffers["77"] = notices
        service._notice_first_at["77"] = loop.time()

        await service._flush_notices("77")

        turns = [service._turn_queue.get_nowait() for _ in range(3)]
        self.assertEqual(
            [len(turn.metadata["notice_ids"]) for turn in turns],
            [100, 100, 5],
        )
        self.assertEqual(turns[0].metadata["notice_ids"][0], "tg:77:1")
        self.assertEqual(turns[-1].metadata["notice_ids"][-1], "tg:77:205")

    async def test_pending_outbox_notice_is_not_merged_with_fresh_notice(self):
        service = self.service()
        loop = asyncio.get_running_loop()
        old_notice = {
            "source": "telegram",
            "notice_id": "tg:77:9",
            "chat_id": 77,
            "message_id": 9,
            "timestamp": NOW.isoformat(),
            "sender": {"id": 88, "display_name": "Лера"},
            "media_type": "text",
        }
        fresh_notice = {**old_notice, "notice_id": "tg:77:10", "message_id": 10}
        self.state.prepare_telegram_outbox(
            "outbox-for-old-notice",
            77,
            [old_notice["notice_id"]],
            ["уже подготовленный ответ"],
        )
        service._notice_buffers["77"] = [old_notice, fresh_notice]
        service._notice_first_at["77"] = loop.time()

        await service._flush_notices("77")

        turns = [service._turn_queue.get_nowait() for _ in range(2)]
        self.assertEqual(
            [turn.metadata["notice_ids"] for turn in turns],
            [["tg:77:9"], ["tg:77:10"]],
        )

    async def test_recovery_does_not_prepare_retroactive_initiative(self):
        self.state.create_entity(
            "person", "Лера", entity_id="telegram:77", is_real=True, at=NOW
        )
        self.state.upsert_relationship("telegram:77", at=NOW)
        service = self.service()
        heartbeat_trigger = HeartbeatTrigger(
            reason=HeartbeatReason.RECOVERY,
            scheduled_at=NOW,
            fired_at=NOW,
            payload={"downtime_seconds": 3600},
        )

        running = asyncio.create_task(service._on_heartbeat(heartbeat_trigger))
        turn = await asyncio.wait_for(service._turn_queue.get(), timeout=1)
        self.assertNotIn("_telegram_target_ref", turn.metadata)
        turn.metadata["_completion_future"].set_result(None)
        await running

    async def test_concurrent_telegram_chats_do_not_regenerate_on_state_revision(self):
        class ConcurrentTelegramResponses:
            def __init__(self):
                self.started = 0
                self.both_started = asyncio.Event()
                self.release = asyncio.Event()

            async def create(self, **_request):
                self.started += 1
                if self.started == 2:
                    self.both_started.set()
                await self.release.wait()
                return _final(
                    {
                        "telegram": {
                            "target_token": "token-for-turn",
                            "messages": ["один готовый ответ"],
                            "reaction": None,
                            "blacklist_sender": False,
                        },
                    }
                )

        class PerChatSupervisor(_Supervisor):
            async def request(self, method, params, **options):
                if method != "telegram.open":
                    return await super().request(method, params, **options)
                self.calls.append((method, dict(params), dict(options)))
                notice_id = params["notice_ids"][0]
                chat_id = int(notice_id.split(":")[1])
                message_id = int(notice_id.split(":")[2])
                return {
                    "turn_id": params["turn_id"],
                    "target_token": "token-for-turn",
                    "target_ref": chat_id,
                    "messages": [
                        {
                            "message_id": message_id,
                            "timestamp": NOW.isoformat(),
                            "sender": {
                                "id": chat_id + 100,
                                "display_name": f"chat {chat_id}",
                            },
                            "text": "привет",
                            "media_type": "text",
                        }
                    ],
                    "history": [],
                }

        responses = ConcurrentTelegramResponses()
        self.supervisor = PerChatSupervisor()
        service = self.service(model=SimpleNamespace(responses=responses))
        router = asyncio.create_task(service._queue_loop())
        completions = []
        try:
            for chat_id in (71, 72):
                completion = asyncio.get_running_loop().create_future()
                completions.append(completion)
                notice_id = f"tg:{chat_id}:1"
                await service._turn_queue.put(
                    TurnTrigger(
                        kind="telegram_notice",
                        occurred_at=NOW,
                        source_skill="telegram",
                        revision=0,
                        metadata={
                            "chat_id": chat_id,
                            "notice_ids": [notice_id],
                            "notices": [
                                {"notice_id": notice_id, "media_type": "text"}
                            ],
                            "_completion_future": completion,
                        },
                    )
                )
            await asyncio.wait_for(responses.both_started.wait(), timeout=2)
            responses.release.set()
            await asyncio.wait_for(asyncio.gather(*completions), timeout=2)

            self.assertEqual(responses.started, 2)
            self.assertEqual(self.state.get_agent_state().revision, 2)
            for chat_id in (71, 72):
                self.assertEqual(
                    [item.role for item in self.memory.get_chat_history(chat_id)],
                    ["user", "assistant"],
                )
        finally:
            service._stopping = True
            router.cancel()
            for task in service._worker_tasks.values():
                task.cancel()
            await asyncio.gather(
                router, *service._worker_tasks.values(), return_exceptions=True
            )

    async def test_different_chat_workers_generate_concurrently(self):
        class ConcurrentResponses:
            def __init__(self):
                self.started = 0
                self.both_started = asyncio.Event()
                self.release = asyncio.Event()

            async def create(self, **_request):
                self.started += 1
                if self.started <= 2:
                    if self.started == 2:
                        self.both_started.set()
                    await self.release.wait()
                return _final(empty_turn_payload())

        responses = ConcurrentResponses()
        service = self.service(model=SimpleNamespace(responses=responses))
        router = asyncio.create_task(service._queue_loop())
        try:
            for chat_id in (1, 2):
                await service._turn_queue.put(
                    TurnTrigger(
                        kind="manual_wake",
                        occurred_at=NOW,
                        revision=0,
                        metadata={"chat_id": chat_id},
                    )
                )
            await asyncio.wait_for(responses.both_started.wait(), timeout=2)
            self.assertEqual(responses.started, 2)
            responses.release.set()
            for _ in range(200):
                if self.state.get_agent_state().revision == 2:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(self.state.get_agent_state().revision, 2)
            self.assertGreaterEqual(responses.started, 3)  # one revision retry
        finally:
            service._stopping = True
            router.cancel()
            for task in service._worker_tasks.values():
                task.cancel()
            await asyncio.gather(
                router, *service._worker_tasks.values(), return_exceptions=True
            )


if __name__ == "__main__":
    unittest.main()
