import unittest
import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from milana_ipc import INVALID_PARAMS, JsonRpcError, MediaPathError, RequestContext
from telegram_skill_host import (
    TelegramNotice,
    TelegramSkillHost,
    TelethonTelegramAdapter,
)


class _Adapter:
    def __init__(self):
        self.executed = []
        self.cleaned = []
        self.callback = None
        self.materialized = None
        self.backfill_notices = ()

    async def start(self, callback):
        self.callback = callback

    async def stop(self):
        return None

    async def backfill(self, limit):
        return self.backfill_notices

    async def materialize(self, notice_ids, *, turn_id, turn_dir, target_ref=None):
        if self.materialized is not None:
            return self.materialized
        return {
            "_target": target_ref if target_ref is not None else 10,
            "_message_ids": [7],
            "_sender_ids": [20],
            "messages": [{"message_id": 7, "text": "секретный текст"}],
            "history": [],
        }

    async def execute_action(
        self, action, arguments, *, turn_id, turn_dir, request
    ):
        self.executed.append((action, dict(arguments), turn_id))
        return {"status": "ok"}

    async def cleanup_turn(self, turn_id):
        self.cleaned.append(turn_id)

    async def set_presence(self, online):
        self.online = online


def _request(method, *, key=None):
    return RequestContext(
        peer=SimpleNamespace(),
        request_id=1,
        method=method,
        idempotency_key=key,
    )


class TelegramSkillHostTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.adapter = _Adapter()
        self.host = TelegramSkillHost(self.adapter, self.tmp.name)

    async def asyncTearDown(self):
        for turn_id in tuple(self.host._turn_dirs):
            await self.host.cleanup_turn(turn_id)
        self.tmp.cleanup()

    async def test_open_issues_turn_scoped_token_then_authorizes_action(self):
        opened = await self.host._handle_open(
            {"turn_id": "turn-1", "notice_ids": ["tg:10:7"]},
            _request("telegram.open"),
        )
        self.assertEqual(opened["target_ref"], 10)
        self.assertEqual(opened["messages"][0]["text"], "секретный текст")
        token = opened["target_token"]

        with self.assertRaises(JsonRpcError) as missing_key:
            await self.host._handle_execute(
                {
                    "turn_id": "turn-1",
                    "target_token": token,
                    "action": "send_messages",
                    "arguments": {"messages": ["ответ"]},
                },
                _request("telegram.execute"),
            )
        self.assertEqual(missing_key.exception.code, INVALID_PARAMS)
        self.assertEqual(self.adapter.executed, [])

        result = await self.host._handle_execute(
            {
                "turn_id": "turn-1",
                "target_token": token,
                "action": "send_messages",
                "arguments": {"messages": ["ответ"]},
            },
            _request("telegram.execute", key="turn-1:send"),
        )
        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(self.adapter.executed[0][1]["_target"], 10)

        with self.assertRaises(JsonRpcError):
            await self.host._handle_execute(
                {
                    "turn_id": "another-turn",
                    "target_token": token,
                    "action": "send_messages",
                    "arguments": {"messages": ["нельзя"]},
                },
                _request("telegram.execute", key="other"),
            )
        self.assertEqual(len(self.adapter.executed), 1)

    async def test_typing_signal_is_turn_scoped_but_needs_no_idempotency_key(self):
        opened = await self.host._handle_open(
            {"turn_id": "typing-turn", "notice_ids": ["tg:10:7"]},
            _request("telegram.open"),
        )
        result = await self.host._handle_execute(
            {
                "turn_id": "typing-turn",
                "target_token": opened["target_token"],
                "action": "typing",
                "arguments": {"active": True},
            },
            _request("telegram.execute"),
        )

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(self.adapter.executed[0][0], "typing")
        self.assertEqual(self.adapter.executed[0][1]["_target"], 10)

    def test_startup_removes_media_left_by_a_crashed_host(self):
        stale = Path(self.tmp.name) / "turns" / "old-turn" / "photo.png"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"old")

        TelegramSkillHost(self.adapter, self.tmp.name)

        self.assertFalse(stale.exists())

    async def test_reaction_and_block_are_limited_to_opened_ids(self):
        opened = await self.host._handle_open(
            {"turn_id": "turn", "notice_ids": ["tg:10:7"]},
            _request("telegram.open"),
        )
        token = opened["target_token"]
        for action, arguments in (
            ("reaction", {"message_id": 999, "reaction": "👍"}),
            ("blacklist_sender", {"sender_id": 999}),
        ):
            with self.subTest(action=action), self.assertRaises(JsonRpcError):
                await self.host._handle_execute(
                    {
                        "turn_id": "turn",
                        "target_token": token,
                        "action": action,
                        "arguments": arguments,
                    },
                    _request("telegram.execute", key=action),
                )
        self.assertEqual(self.adapter.executed, [])

    async def test_adapter_media_escape_is_rejected_and_cleanup_removes_turn(self):
        outside = Path(self.tmp.name).parent / "outside.bin"
        outside.write_bytes(b"x")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        self.adapter.materialized = {
            "_target": 10,
            "_message_ids": [],
            "_sender_ids": [],
            "messages": [{"media_path": str(outside)}],
        }
        with self.assertRaises(MediaPathError):
            await self.host._handle_open(
                {"turn_id": "escape", "notice_ids": ["tg:10:7"]},
                _request("telegram.open"),
            )
        await self.host.cleanup_turn("escape")
        self.assertIn("escape", self.adapter.cleaned)

    def test_notice_payload_is_metadata_only(self):
        notice = TelegramNotice(
            notice_id="tg:10:7",
            chat_id=10,
            message_id=7,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="photo",
        )
        payload = notice.to_payload()
        self.assertEqual(
            set(payload),
            {
                "source",
                "notice_id",
                "chat_id",
                "message_id",
                "timestamp",
                "sender",
                "media_type",
            },
        )
        self.assertNotIn("text", payload)
        self.assertNotIn("path", payload)

    async def test_start_backfills_unread_metadata_after_reconnect(self):
        notice = TelegramNotice(
            notice_id="tg:10:8",
            chat_id=10,
            message_id=8,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="text",
        )
        self.adapter.backfill_notices = (notice,)

        class Peer:
            closed = False

            def __init__(self):
                self.notifications = []

            async def notify(self, method, params, **options):
                self.notifications.append((method, params, options))

        peer = Peer()
        self.host.peer = peer
        await self.host.start()
        self.assertEqual(peer.notifications[0][0], "telegram.notice")
        self.assertNotIn("text", peer.notifications[0][1])

    async def test_periodic_backfill_recovers_a_missed_live_notice(self):
        notice = TelegramNotice(
            notice_id="tg:10:9",
            chat_id=10,
            message_id=9,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="text",
        )
        self.adapter.backfill_notices = (notice,)

        class Peer:
            closed = False

            def __init__(self):
                self.notifications = []

            async def notify(self, method, params, **options):
                self.notifications.append((method, params, options))

        self.host.peer = Peer()
        self.host._started = True
        self.host.backfill_poll_seconds = 0.01
        task = asyncio.create_task(self.host._backfill_loop())
        try:
            for _ in range(50):
                if self.host.peer.notifications:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(
                self.host.peer.notifications[0][1]["notice_id"], "tg:10:9"
            )
        finally:
            self.host._started = False
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_acknowledge_publishes_the_next_unread_backfill_page(self):
        opened = await self.host._handle_open(
            {"turn_id": "turn", "notice_ids": ["tg:10:7"]},
            _request("telegram.open"),
        )
        next_notice = TelegramNotice(
            notice_id="tg:10:8",
            chat_id=10,
            message_id=8,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="text",
        )
        self.adapter.backfill_notices = (next_notice,)

        class Peer:
            closed = False

            def __init__(self):
                self.notifications = []

            async def notify(self, method, params, **options):
                self.notifications.append((method, params, options))

        peer = Peer()
        self.host.peer = peer
        await self.host._handle_execute(
            {
                "turn_id": "turn",
                "target_token": opened["target_token"],
                "action": "acknowledge",
                "arguments": {"message_ids": [7]},
            },
            _request("telegram.execute", key="turn:ack"),
        )
        self.assertEqual(
            [item[1]["notice_id"] for item in peer.notifications], ["tg:10:8"]
        )

    async def test_presence_is_forwarded_to_the_adapter(self):
        result = await self.host._handle_presence(
            {"online": True}, _request("telegram.presence")
        )
        self.assertEqual(result, {"online": True})
        self.assertTrue(self.adapter.online)


class TelethonBackfillTests(unittest.IsolatedAsyncioTestCase):
    async def test_typing_lives_until_turn_cleanup(self):
        class TypingContext:
            def __init__(self):
                self.entered = False
                self.exited = False

            async def __aenter__(self):
                self.entered = True
                return self

            async def __aexit__(self, _type, _value, _traceback):
                self.exited = True

        class Client:
            def __init__(self):
                self.context = TypingContext()
                self.action_call = None

            def action(self, target, action):
                self.action_call = (target, action)
                return self.context

        adapter = object.__new__(TelethonTelegramAdapter)
        adapter.client = Client()
        adapter._typing_contexts = {}
        adapter._sticker_sessions = {}

        result = await adapter.execute_action(
            "typing",
            {"_target": 10, "active": True},
            turn_id="turn",
            turn_dir=Path("."),
            request=_request("telegram.execute"),
        )

        self.assertEqual(result, {"status": "typing", "active": True})
        self.assertEqual(adapter.client.action_call, (10, "typing"))
        self.assertTrue(adapter.client.context.entered)
        self.assertFalse(adapter.client.context.exited)
        await adapter.cleanup_turn("turn")
        self.assertTrue(adapter.client.context.exited)

    async def test_backfill_reads_a_contiguous_oldest_unread_prefix(self):
        class Client:
            def __init__(self):
                self.iteration = None

            async def iter_dialogs(self):
                yield SimpleNamespace(
                    entity="chat",
                    unread_count=900,
                    dialog=SimpleNamespace(read_inbox_max_id=40),
                )

            def iter_messages(self, entity, limit, **options):
                self.iteration = (entity, limit, options)

                async def messages():
                    for message_id in (41, 42, 43):
                        yield SimpleNamespace(id=message_id, out=False)

                return messages()

        adapter = object.__new__(TelethonTelegramAdapter)
        adapter.client = Client()
        adapter._messages = OrderedDict()

        async def notice_from_message(message):
            return TelegramNotice(
                notice_id=f"tg:10:{message.id}",
                chat_id=10,
                message_id=message.id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                sender={"id": 20, "display_name": "Лера"},
                media_type="text",
            )

        adapter._notice_from_message = notice_from_message
        notices = await adapter.backfill(3)

        self.assertEqual([item.message_id for item in notices], [41, 42, 43])
        self.assertEqual(
            adapter.client.iteration,
            ("chat", 3, {"min_id": 40, "reverse": True}),
        )


if __name__ == "__main__":
    unittest.main()
