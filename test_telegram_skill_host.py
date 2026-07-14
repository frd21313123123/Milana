import unittest
import asyncio
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

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
        self.materialize_calls = []
        self.terminal_acknowledged = []

    async def start(self, callback):
        self.callback = callback

    async def stop(self):
        return None

    async def backfill(self, limit):
        return self.backfill_notices

    async def acknowledge_terminal_notice(self, notice):
        self.terminal_acknowledged.append(notice.notice_id)
        return True

    async def materialize(
        self,
        notice_ids,
        *,
        turn_id,
        turn_dir,
        target_ref=None,
        include_history=True,
    ):
        self.materialize_calls.append(
            {
                "notice_ids": tuple(notice_ids),
                "target_ref": target_ref,
                "include_history": include_history,
            }
        )
        if self.materialized is not None:
            return self.materialized
        has_notice = bool(notice_ids)
        return {
            "_target": target_ref if target_ref is not None else 10,
            "_message_ids": [7] if has_notice else [],
            "_sender_ids": [20] if has_notice else [],
            "messages": (
                [{"message_id": 7, "text": "секретный текст"}]
                if has_notice
                else []
            ),
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

    async def test_target_ref_only_open_defaults_missing_notice_ids_to_empty(self):
        opened = await self.host._handle_open(
            {
                "turn_id": "initiative",
                "target_ref": 10,
                "include_history": False,
            },
            _request("telegram.open"),
        )

        self.assertEqual(opened["target_ref"], 10)
        self.assertEqual(opened["messages"], [])
        self.assertEqual(
            self.adapter.materialize_calls,
            [
                {
                    "notice_ids": (),
                    "target_ref": 10,
                    "include_history": False,
                }
            ],
        )

    async def test_open_rejects_a_notice_the_adapter_did_not_materialize(self):
        self.adapter.materialized = {
            "_target": 10,
            "_message_ids": [],
            "_sender_ids": [],
            "messages": [],
            "history": [],
        }

        with self.assertRaises(JsonRpcError) as missing:
            await self.host._handle_open(
                {"turn_id": "missing", "notice_ids": ["tg:10:7"]},
                _request("telegram.open"),
            )

        self.assertEqual(missing.exception.code, INVALID_PARAMS)
        self.assertIn("tg:10:7", str(missing.exception))

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
            "_message_ids": [7],
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
                self.requests = []

            async def request(self, method, params, **options):
                self.requests.append((method, params, options))
                return {
                    "accepted": True,
                    "safe_to_ack": False,
                    "terminal": False,
                }

        peer = Peer()
        self.host.peer = peer
        await self.host.start()
        self.assertEqual(peer.requests[0][0], "telegram.notice")
        self.assertNotIn("text", peer.requests[0][1])
        self.assertNotIn("idempotency_key", peer.requests[0][2])
        self.assertEqual(self.adapter.terminal_acknowledged, [])

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
                self.requests = []

            async def request(self, method, params, **options):
                self.requests.append((method, params, options))
                return {
                    "accepted": True,
                    "safe_to_ack": False,
                    "terminal": False,
                }

        self.host.peer = Peer()
        self.host._started = True
        self.host.backfill_poll_seconds = 0.01
        task = asyncio.create_task(self.host._backfill_loop())
        try:
            for _ in range(50):
                if self.host.peer.requests:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(
                self.host.peer.requests[0][1]["notice_id"], "tg:10:9"
            )
        finally:
            self.host._started = False
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_backfill_retires_only_confirmed_terminal_prefixes(self):
        def notice(chat_id, message_id):
            return TelegramNotice(
                notice_id=f"tg:{chat_id}:{message_id}",
                chat_id=chat_id,
                message_id=message_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                sender={"id": 20, "display_name": "Лера"},
                media_type="text",
            )

        self.adapter.backfill_notices = (
            notice(10, 8),
            notice(10, 9),
            notice(10, 10),
            notice(11, 3),
        )
        receipts = {
            "tg:10:8": {
                "accepted": True,
                "safe_to_ack": True,
                "terminal": True,
            },
            "tg:10:9": {
                "accepted": True,
                "safe_to_ack": False,
                "terminal": False,
            },
            "tg:10:10": {
                "accepted": True,
                "safe_to_ack": True,
                "terminal": True,
            },
            "tg:11:3": {
                "accepted": True,
                "safe_to_ack": True,
                "terminal": True,
            },
        }

        class Peer:
            closed = False

            def __init__(self):
                self.requested = []

            async def request(self, _method, params, **_options):
                notice_id = params["notice_id"]
                self.requested.append(notice_id)
                return receipts[notice_id]

        peer = Peer()
        self.host.peer = peer
        await self.host._publish_backfill()

        self.assertEqual(
            peer.requested,
            ["tg:10:8", "tg:10:9", "tg:10:10", "tg:11:3"],
        )
        # Chat 10 stops at the last terminal notice before the deferred gap;
        # chat 11 remains independent.
        self.assertEqual(
            self.adapter.terminal_acknowledged, ["tg:10:8", "tg:11:3"]
        )

    async def test_backfill_advances_multiple_terminal_notices_oldest_first(self):
        notices = tuple(
            TelegramNotice(
                notice_id=f"tg:10:{message_id}",
                chat_id=10,
                message_id=message_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                sender={"id": 20, "display_name": "Лера"},
                media_type="text",
            )
            for message_id in (8, 9)
        )
        self.adapter.backfill_notices = notices
        unread = {8, 9}

        async def gap_safe_ack(notice):
            if min(unread) != notice.message_id:
                return False
            unread.remove(notice.message_id)
            self.adapter.terminal_acknowledged.append(notice.notice_id)
            return True

        self.adapter.acknowledge_terminal_notice = gap_safe_ack

        class Peer:
            closed = False

            async def request(self, _method, _params, **_options):
                return {
                    "accepted": True,
                    "safe_to_ack": True,
                    "terminal": True,
                }

        self.host.peer = Peer()
        await self.host._publish_backfill()

        self.assertEqual(
            self.adapter.terminal_acknowledged, ["tg:10:8", "tg:10:9"]
        )
        self.assertEqual(unread, set())

    async def test_old_notification_only_peer_never_advances_terminal_read_state(self):
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
        await self.host._publish_backfill()

        self.assertEqual(peer.notifications[0][1]["notice_id"], "tg:10:8")
        self.assertEqual(self.adapter.terminal_acknowledged, [])

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
        for _ in range(20):
            if peer.notifications:
                break
            await asyncio.sleep(0)
        self.assertEqual(
            [item[1]["notice_id"] for item in peer.notifications], ["tg:10:8"]
        )

    async def test_acknowledge_reconciles_a_missing_predecessor_before_read_ack(self):
        opened = await self.host._handle_open(
            {"turn_id": "turn", "notice_ids": ["tg:10:7"]},
            _request("telegram.open"),
        )
        predecessor = TelegramNotice(
            notice_id="tg:10:6",
            chat_id=10,
            message_id=6,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="text",
        )
        current = TelegramNotice(
            notice_id="tg:10:7",
            chat_id=10,
            message_id=7,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="text",
        )
        events = []

        async def reconcile(target, through_message_id):
            events.append(("reconcile", target, through_message_id))
            return (predecessor, current)

        original_execute = self.adapter.execute_action

        async def execute(action, arguments, **options):
            events.append(("execute", action))
            return await original_execute(action, arguments, **options)

        class Peer:
            closed = False

            async def request(self, method, params, **_options):
                events.append(("notice", method, params["notice_id"]))
                return {
                    "accepted": True,
                    "safe_to_ack": True,
                    "terminal": True,
                }

        self.adapter.backfill_before_ack = reconcile
        self.adapter.execute_action = execute
        self.host.peer = Peer()

        await self.host._handle_execute(
            {
                "turn_id": "turn",
                "target_token": opened["target_token"],
                "action": "acknowledge",
                "arguments": {"message_ids": [7]},
            },
            _request("telegram.execute", key="turn:ack:reconciled"),
        )

        self.assertEqual(
            events[:3],
            [
                ("reconcile", 10, 7),
                ("notice", "telegram.notice", "tg:10:6"),
                ("execute", "acknowledge"),
            ],
        )

    async def test_acknowledge_does_not_cross_a_deferred_gap_notice(self):
        opened = await self.host._handle_open(
            {"turn_id": "turn", "notice_ids": ["tg:10:7"]},
            _request("telegram.open"),
        )
        predecessor = TelegramNotice(
            notice_id="tg:10:6",
            chat_id=10,
            message_id=6,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="text",
        )

        async def reconcile(_target, _through_message_id):
            return (predecessor,)

        class Peer:
            closed = False

            async def request(self, _method, _params, **_options):
                return {
                    "accepted": True,
                    "safe_to_ack": False,
                    "terminal": False,
                }

        self.adapter.backfill_before_ack = reconcile
        self.host.peer = Peer()

        with self.assertRaisesRegex(RuntimeError, "tg:10:6"):
            await self.host._handle_execute(
                {
                    "turn_id": "turn",
                    "target_token": opened["target_token"],
                    "action": "acknowledge",
                    "arguments": {"message_ids": [7]},
                },
                _request("telegram.execute", key="turn:ack:deferred-gap"),
            )

        self.assertEqual(self.adapter.executed, [])

    async def test_acknowledge_does_not_advance_when_gap_notice_is_not_accepted(self):
        opened = await self.host._handle_open(
            {"turn_id": "turn", "notice_ids": ["tg:10:7"]},
            _request("telegram.open"),
        )
        predecessor = TelegramNotice(
            notice_id="tg:10:6",
            chat_id=10,
            message_id=6,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="text",
        )

        async def reconcile(_target, _through_message_id):
            return (predecessor,)

        class Peer:
            closed = False

            async def request(self, _method, _params, **_options):
                raise ConnectionError("service disconnected before persistence")

        self.adapter.backfill_before_ack = reconcile
        self.host.peer = Peer()

        with self.assertRaises(ConnectionError):
            await self.host._handle_execute(
                {
                    "turn_id": "turn",
                    "target_token": opened["target_token"],
                    "action": "acknowledge",
                    "arguments": {"message_ids": [7]},
                },
                _request("telegram.execute", key="turn:ack:rejected"),
            )

        self.assertEqual(self.adapter.executed, [])

    async def test_acknowledge_result_survives_backfill_refresh_failure(self):
        opened = await self.host._handle_open(
            {"turn_id": "turn", "notice_ids": ["tg:10:7"]},
            _request("telegram.open"),
        )

        async def broken_backfill(_limit):
            raise OSError("temporary backfill failure")

        self.adapter.backfill = broken_backfill
        result = await self.host._handle_execute(
            {
                "turn_id": "turn",
                "target_token": opened["target_token"],
                "action": "acknowledge",
                "arguments": {"message_ids": [7]},
            },
            _request("telegram.execute", key="turn:ack:failure"),
        )
        await asyncio.sleep(0)

        self.assertEqual(result, {"status": "ok"})

    async def test_presence_is_forwarded_to_the_adapter(self):
        result = await self.host._handle_presence(
            {"online": True}, _request("telegram.presence")
        )
        self.assertEqual(result, {"online": True})
        self.assertTrue(self.adapter.online)

    async def test_presence_timeout_is_reported_without_hanging_the_request(self):
        async def slow_presence(_online):
            await asyncio.sleep(60)

        self.adapter.set_presence = slow_presence
        with patch("telegram_skill_host.SIGNAL_TIMEOUT_SECONDS", 0.01):
            result = await self.host._handle_presence(
                {"online": True}, _request("telegram.presence")
            )

        self.assertEqual(
            result, {"online": True, "applied": False, "timed_out": True}
        )


class TelethonBackfillTests(unittest.IsolatedAsyncioTestCase):
    async def test_materialize_without_history_uses_cached_message_and_sender(self):
        sender = SimpleNamespace(
            id=20,
            first_name="Лера",
            last_name="",
            title="",
            username="lera",
        )

        class Message:
            id = 7
            chat_id = 10
            date = datetime.now(timezone.utc)
            raw_text = "быстрый контекст"
            media = None
            sticker = gif = voice = video = photo = audio = document = None

            def __init__(self):
                self.sender_calls = 0

            async def get_sender(self):
                self.sender_calls += 1
                return sender

        class Client:
            def __init__(self):
                self.history_requested = False

            def iter_messages(self, *_args, **_kwargs):
                self.history_requested = True
                raise AssertionError("history must not be requested")

        message = Message()
        adapter = object.__new__(TelethonTelegramAdapter)
        adapter.client = Client()
        adapter._messages = OrderedDict([("tg:10:7", message)])
        adapter._senders = OrderedDict([("tg:10:7", sender)])
        adapter._render_sticker_png = lambda *_args: b""

        with TemporaryDirectory() as tmp:
            result = await adapter.materialize(
                ["tg:10:7"],
                turn_id="turn",
                turn_dir=Path(tmp),
                include_history=False,
            )

        self.assertEqual(result["messages"][0]["text"], "быстрый контекст")
        self.assertEqual(result["history"], [])
        self.assertEqual(message.sender_calls, 0)
        self.assertFalse(adapter.client.history_requested)

    async def test_send_messages_reports_partial_index_and_resumes_stably(self):
        adapter = object.__new__(TelethonTelegramAdapter)
        adapter.client = SimpleNamespace()
        adapter._dev_chat = True
        adapter._typing_contexts = {}
        adapter._sticker_sessions = {}
        attempts = []
        fail_part_one = True

        async def send_text(target, text, *, random_id):
            nonlocal fail_part_one
            attempts.append((target, text, random_id))
            if text == "два" and fail_part_one:
                raise OSError("network lost")
            if text == "два":
                return SimpleNamespace(id=None, deduplicated=True)
            return SimpleNamespace(id={"один": 101, "два": 102, "три": 103}[text])

        adapter._send_text = send_text
        arguments = {
            "_target": 10,
            "messages": ["один", "два", "три"],
            "inter_message_min_delay_seconds": 0,
            "inter_message_max_delay_seconds": 0,
        }
        request = _request("telegram.execute", key="turn:messages")

        first = await adapter.execute_action(
            "send_messages",
            arguments,
            turn_id="turn",
            turn_dir=Path("."),
            request=request,
        )
        fail_part_one = False
        resumed = await adapter.execute_action(
            "send_messages",
            {**arguments, "start_index": first["next_part_index"]},
            turn_id="turn",
            turn_dir=Path("."),
            request=_request("telegram.execute", key="turn:messages:resume:1"),
        )

        self.assertEqual(first["status"], "partial")
        self.assertEqual(first["sent_message_ids"], [101])
        self.assertEqual(first["sent_part_indexes"], [0])
        self.assertEqual(first["deduplicated_part_indexes"], [])
        self.assertEqual(first["next_part_index"], 1)
        self.assertEqual(first["first_sent_message_id"], 101)
        self.assertEqual(first["first_sent_part_index"], 0)
        self.assertIsInstance(first["first_send_elapsed_ms"], float)
        self.assertEqual(resumed["status"], "sent")
        self.assertEqual(resumed["sent_message_ids"], [103])
        self.assertEqual(resumed["sent_part_indexes"], [1, 2])
        self.assertEqual(resumed["deduplicated_part_indexes"], [1])
        self.assertEqual(resumed["next_part_index"], 3)
        self.assertEqual(resumed["total_parts"], 3)
        self.assertEqual(first["batch_id"], resumed["batch_id"])
        self.assertEqual(attempts[1][2], attempts[2][2])
        self.assertEqual(len({item[2] for item in attempts}), 3)

    async def test_raw_text_send_passes_stable_random_id_to_telethon(self):
        class Sender:
            def __init__(self, owner):
                self.owner = owner

            def send(self, request):
                async def response():
                    self.owner.request = request
                    if self.owner.duplicate:
                        from telethon.errors import RandomIdDuplicateError

                        raise RandomIdDuplicateError(request)
                    return object()

                return response()

        class Client:
            def __init__(self):
                self.request = None
                self.duplicate = False
                self._sender = Sender(self)

            async def get_input_entity(self, target):
                return target

            async def _parse_message_text(self, text, parse_mode):
                self.parse_mode = parse_mode
                return text, []

            async def __call__(self, _request):
                raise AssertionError("the retrying high-level call must not be used")

            def _get_response_message(self, request, _result, _entity):
                return SimpleNamespace(id=77, request=request)

        adapter = object.__new__(TelethonTelegramAdapter)
        adapter.client = Client()

        sent = await adapter._send_text(10, "привет", random_id=-123456789)
        adapter.client.duplicate = True
        duplicate = await adapter._send_text(
            10, "привет", random_id=-123456789
        )

        self.assertEqual(sent.id, 77)
        self.assertEqual(adapter.client.request.random_id, -123456789)
        self.assertIsNone(duplicate.id)
        self.assertTrue(duplicate.deduplicated)

    async def test_sticker_actions_deduplicate_after_host_restart(self):
        reference = SimpleNamespace(
            set_id=100,
            set_access_hash=200,
            set_short_name="milana_pack",
            document_id=300,
            pack_title="Milana",
            emoji="🙂",
        )
        accepted_random_ids = set()
        attempted_random_ids = []

        class Sender:
            def send(self, telegram_request):
                async def response():
                    from telethon.errors import RandomIdDuplicateError

                    random_id = telegram_request.random_id
                    attempted_random_ids.append(random_id)
                    if random_id in accepted_random_ids:
                        raise RandomIdDuplicateError(telegram_request)
                    accepted_random_ids.add(random_id)
                    return object()

                return response()

        class Client:
            def __init__(self):
                self._sender = Sender()

            async def get_input_entity(self, target):
                return target

            async def _file_to_media(self, document):
                self.document = document
                return None, SimpleNamespace(document=document), False

            async def __call__(self, _request):
                raise AssertionError("the retrying high-level call must not be used")

            def _get_response_message(self, request, _result, _entity):
                return SimpleNamespace(id=77, request=request)

        class Session:
            def choose(self, sticker_id):
                self.sticker_id = sticker_id
                return SimpleNamespace(document="picker-document", reference=reference)

        class StickerSkill:
            async def resolve_reference(self, resolved_reference):
                self.resolved_reference = resolved_reference
                return SimpleNamespace(document="reference-document")

            def new_session(self):
                return Session()

        def fresh_adapter():
            adapter = object.__new__(TelethonTelegramAdapter)
            adapter.client = Client()
            adapter._sticker_skill = StickerSkill()
            adapter._sticker_sessions = {}
            adapter._typing_contexts = {}
            adapter._dev_chat = True
            return adapter

        reference_arguments = {
            "_target": 10,
            "sticker": {
                "set_id": reference.set_id,
                "set_access_hash": reference.set_access_hash,
                "set_short_name": reference.set_short_name,
                "document_id": reference.document_id,
                "pack_title": reference.pack_title,
                "emoji": reference.emoji,
            },
        }
        picker_arguments = {"_target": 10, "sticker_id": "choice-1"}

        for action, arguments in (
            ("send_sticker_reference", reference_arguments),
            ("send_sticker", picker_arguments),
        ):
            key = f"durable-outbox:{action}:1"
            first = await fresh_adapter().execute_action(
                action,
                arguments,
                turn_id="turn-before-restart",
                turn_dir=Path("."),
                request=_request("telegram.execute", key=key),
            )
            # The second adapter has no in-memory RPC cache: this models a host
            # restart after Telegram accepted the first request but its response
            # was lost on the way back to the service.
            replay = await fresh_adapter().execute_action(
                action,
                arguments,
                turn_id="turn-after-restart",
                turn_dir=Path("."),
                request=_request("telegram.execute", key=key),
            )

            self.assertEqual(first["message_id"], 77)
            self.assertFalse(first["deduplicated"])
            self.assertIsNone(replay["message_id"])
            self.assertTrue(replay["deduplicated"])

        self.assertEqual(attempted_random_ids[0], attempted_random_ids[1])
        self.assertEqual(attempted_random_ids[2], attempted_random_ids[3])
        self.assertNotEqual(attempted_random_ids[0], attempted_random_ids[2])
        self.assertNotIn(0, attempted_random_ids)
        self.assertEqual(len(accepted_random_ids), 2)

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

    async def test_slow_typing_signal_times_out_without_leaking_context(self):
        class TypingContext:
            def __init__(self):
                self.exited = False

            async def __aenter__(self):
                await asyncio.sleep(60)

            async def __aexit__(self, _type, _value, _traceback):
                self.exited = True

        class Client:
            def __init__(self):
                self.context = TypingContext()

            def action(self, _target, _action):
                return self.context

        adapter = object.__new__(TelethonTelegramAdapter)
        adapter.client = Client()
        adapter._typing_contexts = {}
        adapter._sticker_sessions = {}

        with patch("telegram_skill_host.SIGNAL_TIMEOUT_SECONDS", 0.01):
            result = await adapter.execute_action(
                "typing",
                {"_target": 10, "active": True},
                turn_id="turn",
                turn_dir=Path("."),
                request=_request("telegram.execute"),
            )

        self.assertEqual(
            result, {"status": "typing", "active": False, "timed_out": True}
        )
        self.assertEqual(adapter._typing_contexts, {})
        self.assertTrue(adapter.client.context.exited)

    async def test_turn_cleanup_does_not_wait_for_stuck_typing_exit(self):
        release = asyncio.Event()

        class TypingContext:
            async def __aexit__(self, _type, _value, _traceback):
                await release.wait()

        adapter = object.__new__(TelethonTelegramAdapter)
        adapter._typing_contexts = {"turn": TypingContext()}
        adapter._sticker_sessions = {"turn": object()}

        with patch("telegram_skill_host.SIGNAL_TIMEOUT_SECONDS", 0.01):
            await asyncio.wait_for(adapter.cleanup_turn("turn"), timeout=0.1)

        self.assertEqual(adapter._typing_contexts, {})
        self.assertEqual(adapter._sticker_sessions, {})
        release.set()
        await asyncio.sleep(0)

    async def test_backfill_reads_a_contiguous_oldest_unread_prefix(self):
        class Client:
            def __init__(self):
                self.iteration = None

            async def iter_dialogs(self):
                yield SimpleNamespace(
                    entity="chat",
                    unread_count=900,
                    dialog=SimpleNamespace(read_inbox_max_id=40, top_message=43),
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
            ("chat", None, {"min_id": 40, "reverse": True, "max_id": 44}),
        )

    async def test_backfill_outgoing_messages_do_not_consume_unread_limit(self):
        class Client:
            def __init__(self):
                self.iteration = None

            async def iter_dialogs(self):
                yield SimpleNamespace(
                    entity="chat",
                    unread_count=1,
                    dialog=SimpleNamespace(read_inbox_max_id=5, top_message=7),
                )

            def iter_messages(self, entity, limit, **options):
                self.iteration = (entity, limit, options)

                async def messages():
                    yield SimpleNamespace(id=6, out=True)
                    yield SimpleNamespace(id=7, out=False)

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
        notices = await adapter.backfill(1)

        self.assertEqual([item.message_id for item in notices], [7])
        self.assertEqual(
            adapter.client.iteration,
            ("chat", None, {"min_id": 5, "reverse": True, "max_id": 8}),
        )

    async def test_pre_ack_backfill_reads_the_complete_crossed_window(self):
        class Client:
            def __init__(self):
                self.iteration = None

            async def iter_dialogs(self):
                yield SimpleNamespace(
                    id=10,
                    entity="chat",
                    dialog=SimpleNamespace(read_inbox_max_id=5),
                )

            def iter_messages(self, entity, limit, **options):
                self.iteration = (entity, limit, options)

                async def messages():
                    for message_id in (6, 7, 8):
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
        notices = await adapter.backfill_before_ack(10, 7)

        self.assertEqual([item.message_id for item in notices], [6, 7])
        self.assertEqual(
            adapter.client.iteration,
            (
                "chat",
                None,
                {"min_id": 5, "max_id": 8, "reverse": True},
            ),
        )

    async def test_terminal_ack_refuses_to_cross_an_older_unread_message(self):
        class Client:
            def __init__(self):
                self.acknowledged = []

            async def iter_dialogs(self):
                yield SimpleNamespace(
                    id=10,
                    entity="chat",
                    dialog=SimpleNamespace(read_inbox_max_id=5),
                )

            def iter_messages(self, _entity, limit, **_options):
                async def messages():
                    for message_id in (6, 7):
                        yield SimpleNamespace(id=message_id, out=False)

                return messages()

            async def send_read_acknowledge(self, target, *, max_id):
                self.acknowledged.append((target, max_id))

        adapter = object.__new__(TelethonTelegramAdapter)
        adapter.client = Client()
        terminal = TelegramNotice(
            notice_id="tg:10:7",
            chat_id=10,
            message_id=7,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="text",
        )

        acknowledged = await adapter.acknowledge_terminal_notice(terminal)

        self.assertFalse(acknowledged)
        self.assertEqual(adapter.client.acknowledged, [])

    async def test_terminal_ack_advances_when_it_is_oldest_unread(self):
        class Client:
            def __init__(self):
                self.acknowledged = []

            async def iter_dialogs(self):
                yield SimpleNamespace(
                    id=10,
                    entity="chat",
                    dialog=SimpleNamespace(read_inbox_max_id=5),
                )

            def iter_messages(self, _entity, limit, **_options):
                async def messages():
                    yield SimpleNamespace(id=6, out=False)

                return messages()

            async def send_read_acknowledge(self, target, *, max_id):
                self.acknowledged.append((target, max_id))

        adapter = object.__new__(TelethonTelegramAdapter)
        adapter.client = Client()
        terminal = TelegramNotice(
            notice_id="tg:10:6",
            chat_id=10,
            message_id=6,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sender={"id": 20, "display_name": "Лера"},
            media_type="text",
        )

        acknowledged = await adapter.acknowledge_terminal_notice(terminal)

        self.assertTrue(acknowledged)
        self.assertEqual(adapter.client.acknowledged, [("chat", 6)])


if __name__ == "__main__":
    unittest.main()
