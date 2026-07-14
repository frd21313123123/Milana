import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from milana import ToolCall, TurnTrigger, bind_telegram_skill_tree, load_default_registry
from milana.runtime import (
    CoreSkillExecutor,
    StickerSkillExecutor,
    TelegramSkillExecutor,
    TurnStagingArea,
)
from milana_schedule import load_routine


class _Gateway:
    def __init__(self):
        self.calls = []

    async def request(self, method, params, **options):
        self.calls.append((method, params, options))
        if method == "telegram.open":
            return {"target_token": "only-this-turn", "chat_id": 42, "messages": []}
        if method == "telegram.execute" and params.get("action") == "open_sticker_picker":
            return {
                "stickers": [
                    {
                        "sticker_id": "s1",
                        "document_id": 101,
                        "set_id": 202,
                        "set_access_hash": 303,
                        "set_short_name": "pack",
                        "pack_title": "Pack",
                        "emoji": "🙂",
                    }
                ]
            }
        raise AssertionError(method)


class RuntimeStagingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc)
        self.trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=self.now,
            revision=4,
        )
        self.staging = TurnStagingArea()
        self.stage = self.staging.begin(self.trigger)
        self.gateway = _Gateway()
        self.telegram = TelegramSkillExecutor(self.staging, self.gateway)
        self.stickers = StickerSkillExecutor(self.staging, self.gateway)
        registry = load_default_registry()
        bind_telegram_skill_tree(
            registry,
            telegram_executor=self.telegram,
            sticker_executor=self.stickers,
            telegram_on_activate=self.telegram.activate,
        )
        self.session = registry.new_session(
            turn_id=self.trigger.id,
            core_executor=CoreSkillExecutor(
                self.staging, load_routine(), now=lambda: self.now
            ),
        )

    async def test_writes_are_staged_and_picker_is_read_only(self):
        await self.session.execute_tool(
            ToolCall.from_arguments("open_skill", {"skill_id": "telegram"})
        )
        await self.session.execute_tool(
            ToolCall.from_arguments(
                "schedule_message", {"delay_seconds": 60, "message": "позже"}
            )
        )
        await self.session.execute_tool(
            ToolCall.from_arguments("open_skill", {"skill_id": "telegram.stickers"})
        )
        await self.session.execute_tool(
            ToolCall.from_arguments("open_sticker_picker", {"pack_id": None})
        )
        await self.session.execute_tool(
            ToolCall.from_arguments("send_sticker", {"sticker_id": "s1"})
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(
            [action.kind for action in self.stage.actions],
            ["schedule_message", "send_sticker"],
        )
        self.assertEqual(
            [call[0] for call in self.gateway.calls],
            ["telegram.open", "telegram.execute", "telegram.execute"],
        )
        self.assertIs(self.gateway.calls[0][1]["include_history"], False)
        self.assertCountEqual(
            [
                call[1].get("action")
                for call in self.gateway.calls
                if call[0] == "telegram.execute"
            ],
            ["typing", "open_sticker_picker"],
        )

    async def test_target_ref_activation_always_sends_empty_notice_array(self):
        trigger = TurnTrigger(
            kind="heartbeat",
            occurred_at=self.now,
            revision=4,
            metadata={"_telegram_target_ref": 991},
        )
        staging = TurnStagingArea()
        staging.begin(trigger)
        telegram = TelegramSkillExecutor(staging, self.gateway)
        registry = load_default_registry()
        bind_telegram_skill_tree(
            registry,
            telegram_executor=telegram,
            sticker_executor=self.stickers,
            telegram_on_activate=telegram.activate,
        )
        session = registry.new_session(turn_id=trigger.id)

        await session.execute_tool(
            ToolCall.from_arguments("open_skill", {"skill_id": "telegram"})
        )

        method, params, _ = self.gateway.calls[0]
        self.assertEqual(method, "telegram.open")
        self.assertEqual(params["notice_ids"], [])
        self.assertEqual(params["target_ref"], 991)
        self.assertIs(params["include_history"], True)

    async def test_typing_is_nonblocking_and_uses_half_second_timeout(self):
        blocker = asyncio.Event()

        class SlowGateway(_Gateway):
            async def request(inner_self, method, params, **options):
                inner_self.calls.append((method, params, options))
                if method == "telegram.open":
                    return {
                        "target_token": "only-this-turn",
                        "chat_id": 42,
                        "messages": [],
                    }
                if method == "telegram.execute" and params.get("action") == "typing":
                    await blocker.wait()
                    return {"ok": True}
                raise AssertionError(method)

        gateway = SlowGateway()
        telegram = TelegramSkillExecutor(self.staging, gateway)
        registry = load_default_registry()
        bind_telegram_skill_tree(
            registry,
            telegram_executor=telegram,
            sticker_executor=self.stickers,
            telegram_on_activate=telegram.activate,
        )
        session = registry.new_session(turn_id=self.trigger.id)

        await asyncio.wait_for(
            session.execute_tool(
                ToolCall.from_arguments("open_skill", {"skill_id": "telegram"})
            ),
            timeout=0.1,
        )
        await asyncio.sleep(0)

        typing_call = next(
            call
            for call in gateway.calls
            if call[0] == "telegram.execute" and call[1].get("action") == "typing"
        )
        self.assertEqual(typing_call[2]["timeout"], 0.5)
        typing_tasks = tuple(telegram._typing_tasks)
        for task in typing_tasks:
            task.cancel()
        await asyncio.gather(*typing_tasks, return_exceptions=True)

    async def test_unknown_sticker_is_rejected_without_rpc_or_staging(self):
        await self.session.execute_tool(
            ToolCall.from_arguments("open_skill", {"skill_id": "telegram"})
        )
        await self.session.execute_tool(
            ToolCall.from_arguments("open_skill", {"skill_id": "telegram.stickers"})
        )
        before = len(self.gateway.calls)
        with self.assertRaises(PermissionError):
            await self.session.execute_tool(
                ToolCall.from_arguments("send_sticker", {"sticker_id": "hidden"})
            )
        self.assertEqual(len(self.gateway.calls), before)
        self.assertEqual(self.stage.actions, [])

    async def test_diary_and_wakeup_are_not_persisted_by_executor(self):
        await self.session.execute_tool(
            ToolCall.from_arguments("write_diary", {"entry": "важный день"})
        )
        result = await self.session.execute_tool(
            ToolCall.from_arguments(
                "schedule_wakeup", {"delay_seconds": 300, "reason": "проверить"}
            )
        )
        self.assertTrue(result.ok)
        self.assertEqual(
            [action.kind for action in self.stage.actions],
            ["write_diary", "schedule_wakeup"],
        )
        self.assertEqual(self.stage.actions[1].payload["due_at"], "2026-07-14T10:05:00+00:00")
        self.assertEqual(self.stage.actions[1].payload["delay_seconds"], 300)

    def test_stage_lifetime_is_explicit(self):
        finished = self.staging.finish(self.trigger.id)
        self.assertIs(finished, self.stage)
        with self.assertRaises(RuntimeError):
            self.staging.get(self.trigger.id)

    def test_notice_action_keys_survive_turn_regeneration(self):
        first_trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=self.now,
            revision=1,
            metadata={"notice_ids": ["tg:42:7", "tg:42:8"]},
        )
        second_trigger = TurnTrigger(
            kind="telegram_notice",
            occurred_at=self.now,
            revision=2,
            metadata={"notice_ids": ["tg:42:7", "tg:42:8"]},
        )
        first = TurnStagingArea().begin(first_trigger)
        second = TurnStagingArea().begin(second_trigger)

        first_key = first.add_action("send_sticker", {"sticker_id": "s1"}).idempotency_key
        second.add_action("write_diary", {"entry": "новая несвязанная запись"})
        second_key = second.add_action("send_sticker", {"sticker_id": "s1"}).idempotency_key

        self.assertEqual(first_key, second_key)
        self.assertNotIn(first_trigger.id, first_key)

    def test_exact_duplicate_actions_have_stable_occurrence_keys(self):
        metadata = {"notice_ids": ["tg:42:7"]}
        first = TurnStagingArea().begin(
            TurnTrigger(
                kind="telegram_notice",
                occurred_at=self.now,
                revision=1,
                metadata=metadata,
            )
        )
        second = TurnStagingArea().begin(
            TurnTrigger(
                kind="telegram_notice",
                occurred_at=self.now,
                revision=2,
                metadata=metadata,
            )
        )

        first_keys = [
            first.add_action("send_sticker", {"sticker_id": "s1"}).idempotency_key,
            first.add_action("send_sticker", {"sticker_id": "s1"}).idempotency_key,
        ]
        second.add_action("write_diary", {"entry": "unrelated"})
        second_keys = [
            second.add_action("send_sticker", {"sticker_id": "s1"}).idempotency_key,
            second.add_action("send_sticker", {"sticker_id": "s1"}).idempotency_key,
        ]

        self.assertEqual(first_keys, second_keys)
        self.assertNotEqual(first_keys[0], first_keys[1])

    def test_heartbeat_logical_scope_survives_turn_regeneration(self):
        metadata = {"notice_ids": [], "_logical_action_scope": "heartbeat-job:42"}
        first_trigger = TurnTrigger(
            kind="heartbeat",
            occurred_at=self.now,
            revision=1,
            metadata=metadata,
        )
        second_trigger = TurnTrigger(
            kind="heartbeat",
            occurred_at=self.now,
            revision=2,
            metadata=metadata,
        )
        first = TurnStagingArea().begin(first_trigger)
        second = TurnStagingArea().begin(second_trigger)

        first_keys = [
            first.add_action("write_diary", {"entry": "день"}).idempotency_key,
            first.add_action("schedule_message", {"message": "позже"}).idempotency_key,
        ]
        second_keys = [
            second.add_action("write_diary", {"entry": "день"}).idempotency_key,
            second.add_action("schedule_message", {"message": "позже"}).idempotency_key,
        ]

        self.assertEqual(first_keys, second_keys)
        self.assertTrue(all(key.startswith("logical:") for key in first_keys))
        self.assertTrue(all(first_trigger.id not in key for key in first_keys))


if __name__ == "__main__":
    unittest.main()
