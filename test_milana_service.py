import asyncio
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

from milana import ToolResult, TurnTrigger, empty_turn_payload
from milana_heartbeat import HeartbeatReason, HeartbeatTrigger
from milana_ipc import MediaPathValidator
from milana_memory import MilanaMemoryStore
from milana_schedule import load_routine
from milana_service import MilanaService, build_heartbeat_changes
from milana_state import MilanaStateStore, StateConflictError
from telegram_client import AIConfig, MessageFlowConfig


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
                        "text": "привет",
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
                return {"status": "sent", "sent_message_ids": [10]}
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

    def service(self, responses=(), *, model=None):
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
        )
        return MilanaService(
            config=config,
            model_client=model or _Model(responses),
            memory=self.memory,
            state=self.state,
            routine=load_routine(),
            rpc_server=SimpleNamespace(),
            supervisor=self.supervisor,
            dev_mode=False,
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

    async def test_notice_opens_telegram_then_commits_reply_and_read(self):
        final = empty_turn_payload(telegram=True)
        final["telegram"] = {
            "target_token": "token-for-turn",
            "messages": ["приветик"],
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
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={"chat_id": 77, "notice_ids": ["tg:77:9"], "notices": []},
        )
        result = await service._execute_turn(trigger)

        self.assertEqual(result.active_skills, ("telegram",))
        self.assertIsNotNone(result.validated_changes)
        self.assertEqual(result.staged_actions, ())
        methods = [item[0] for item in self.supervisor.calls]
        self.assertEqual(methods[0], "telegram.open")
        actions = [
            item[1].get("action")
            for item in self.supervisor.calls
            if item[0] == "telegram.execute"
        ]
        self.assertEqual(actions, ["send_messages", "acknowledge"])
        history = self.memory.get_chat_history(77)
        self.assertEqual([item.role for item in history], ["user", "assistant"])
        self.assertEqual(history[0].content, "привет")
        self.assertEqual(history[1].content, "приветик")
        audit = self.state.list_skill_audit(turn_id=trigger.id)
        self.assertEqual([item.skill_id for item in audit], ["telegram"])

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
                _call("open_skill", {"skill_id": "telegram"}, "open"),
                _final(empty_turn_payload(telegram=True)),
            ]
        )
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={"chat_id": 77, "notice_ids": ["tg:77:9"], "notices": []},
        )

        await service._execute_turn(trigger)

        second_input = service.model_client.responses.requests[1]["input"]
        outputs = [
            item["output"]
            for item in second_input
            if isinstance(item, dict)
            and item.get("type") == "function_call_output"
        ]
        self.assertIn("я давно люблю зелёный чай", "\n".join(outputs))

    async def test_skill_audit_records_parent_denial_and_success(self):
        service = self.service(
            [
                _call(
                    "open_skill",
                    {"skill_id": "telegram.stickers"},
                    "denied-child",
                ),
                _call("open_skill", {"skill_id": "telegram"}, "open-parent"),
                _final(empty_turn_payload(telegram=True)),
            ]
        )
        trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=NOW,
            revision=self.state.get_agent_state().revision,
            metadata={"chat_id": 77, "notice_ids": ["tg:77:9"], "notices": []},
        )

        await service._execute_turn(trigger)

        rows = list(reversed(self.state.list_skill_audit(turn_id=trigger.id)))
        self.assertEqual(
            [(row.skill_id, row.action, row.success) for row in rows],
            [
                ("telegram.stickers", "activation_denied", False),
                ("telegram", "activate", True),
            ],
        )

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

    async def test_telegram_then_stickers_picker_then_send(self):
        final = empty_turn_payload(telegram=True)
        service = self.service(
            [
                _call("open_skill", {"skill_id": "telegram"}, "tg"),
                _call(
                    "open_skill", {"skill_id": "telegram.stickers"}, "stickers"
                ),
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
            metadata={"chat_id": 77, "notice_ids": ["tg:77:9"], "notices": []},
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

    async def test_deliberate_no_reply_still_records_and_acknowledges_notice(self):
        service = self.service(
            [
                _call("open_skill", {"skill_id": "telegram"}, "open"),
                _final(empty_turn_payload(telegram=True)),
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
        self.assertEqual(actions, ["acknowledge"])
        self.assertEqual(self.memory.get_chat_history(77)[0].content, "привет")

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
