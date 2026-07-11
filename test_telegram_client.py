import argparse
import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from openai import BadRequestError

from milana_schedule import load_routine
from telegram_client import (
    AIConfig,
    MessageFlowConfig,
    MilanaMessageResponder,
    MilanaPresenceController,
    ai_number,
    ai_positive_int,
    ai_string,
    display_name,
    load_env_file,
    load_ai_settings,
    load_message_flow_config,
    message_text,
    normalize_target,
    positive_int,
    split_telegram_text,
    telegram_image_mime_type,
)


YEKT = timezone(timedelta(hours=5))


class AsyncContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None


class AdvancingClock:
    def __init__(self, value: datetime) -> None:
        self.value = value
        self.delays: list[float] = []

    def now(self) -> datetime:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.value += timedelta(seconds=seconds)


class GatedClock(AdvancingClock):
    def __init__(self, value: datetime) -> None:
        super().__init__(value)
        self.sleep_calls: asyncio.Queue[tuple[float, asyncio.Event]] = asyncio.Queue()

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)
        release = asyncio.Event()
        await self.sleep_calls.put((seconds, release))
        await release.wait()
        self.value += timedelta(seconds=seconds)


def structured_response(*messages: str, output=None):
    return SimpleNamespace(
        output_text=json.dumps({"messages": list(messages)}, ensure_ascii=False),
        output=[] if output is None else output,
    )


def make_responder(
    clock: AdvancingClock,
    *,
    memory=None,
    randint=None,
    message_flow: MessageFlowConfig | None = None,
):
    client = MagicMock()
    client.side_effect = AsyncMock(return_value=None)
    client.send_read_acknowledge = AsyncMock()
    client.send_message = AsyncMock()
    client.action.return_value = AsyncContext()

    openai_client = MagicMock()
    openai_client.responses.create = AsyncMock(
        return_value=structured_response("Готовый ответ")
    )
    config = AIConfig(
        api_key="test-key",
        model="test-model",
        instructions="Тестовая инструкция",
        temperature=0.2,
        max_output_tokens=100,
        message_flow=message_flow or MessageFlowConfig(),
    )
    responder = MilanaMessageResponder(
        client,
        openai_client,
        config,
        load_routine(),
        memory=memory,
        now=clock.now,
        sleep=clock.sleep,
        randint=randint or (lambda minimum, maximum: minimum),
    )
    return responder, client, openai_client


def make_event(
    value: datetime,
    *,
    text: str = "Привет",
    chat_id: int = 100,
    sender_id: int = 200,
    message_id: int = 300,
    photo: bool = False,
    mime_type: str | None = None,
    image_bytes: bytes = b"test-image",
):
    message = SimpleNamespace(date=value, photo=object() if photo else None)
    message.download_media = AsyncMock(return_value=image_bytes)
    event = SimpleNamespace(
        chat_id=chat_id,
        sender_id=sender_id,
        id=message_id,
        raw_text=text,
        photo=message.photo,
        file=SimpleNamespace(mime_type=mime_type) if mime_type else None,
        message=message,
        get_input_chat=AsyncMock(return_value="peer"),
        get_sender=AsyncMock(return_value=None),
        reply=AsyncMock(),
    )
    return event


class SplitTelegramTextTests(unittest.TestCase):
    def test_empty_text_returns_no_parts(self) -> None:
        self.assertEqual(split_telegram_text("   \n"), [])

    def test_short_text_is_unchanged(self) -> None:
        self.assertEqual(split_telegram_text("  Привет  "), ["Привет"])

    def test_long_unbroken_text_respects_limit(self) -> None:
        parts = split_telegram_text("a" * 8001)
        self.assertEqual([len(part) for part in parts], [4000, 4000, 1])

    def test_prefers_newlines_and_spaces_when_splitting(self) -> None:
        self.assertEqual(split_telegram_text("первая строка\nвторая", 14), ["первая строка", "вторая"])
        self.assertEqual(split_telegram_text("один два три", 9), ["один два", "три"])

    def test_load_ai_settings_reads_json_object(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ai_config.json"
            path.write_text('{"model": "test-model", "temperature": 0.2}', encoding="utf-8")

            self.assertEqual(
                load_ai_settings(path),
                {"model": "test-model", "temperature": 0.2},
            )

    def test_load_ai_settings_rejects_non_object(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ai_config.json"
            path.write_text('[]', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "JSON-объект"):
                load_ai_settings(path)

    def test_max_output_tokens_must_be_an_integer(self) -> None:
        with self.assertRaisesRegex(ValueError, "целым числом"):
            ai_positive_int({"max_output_tokens": 1.5}, "max_output_tokens", 1200)

    def test_config_value_helpers_validate_types_and_ranges(self) -> None:
        self.assertEqual(ai_string({}, "model", "  default  ", "model"), "default")
        self.assertEqual(ai_number({"temperature": 1}, "temperature", 0.7, 0, 2), 1.0)
        with self.assertRaises(ValueError):
            ai_string({"model": "   "}, "model", "default", "model")
        for value in (True, float("inf"), -0.1, 2.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ai_number({"temperature": value}, "temperature", 0.7, 0, 2)

    def test_message_flow_config_defaults_custom_values_and_validation(self) -> None:
        self.assertEqual(load_message_flow_config({}), MessageFlowConfig())
        configured = load_message_flow_config(
            {
                "message_flow": {
                    "input_quiet_seconds": 1.5,
                    "input_max_wait_seconds": 6,
                    "max_reply_messages": 3,
                    "inter_message_min_delay_seconds": 0.5,
                    "inter_message_max_delay_seconds": 2,
                }
            }
        )
        self.assertEqual(configured.input_quiet_seconds, 1.5)
        self.assertEqual(configured.max_reply_messages, 3)

        invalid_settings = [
            {"message_flow": []},
            {"message_flow": {"unknown": 1}},
            {"message_flow": {"input_quiet_seconds": -0.1}},
            {"message_flow": {"input_max_wait_seconds": float("inf")}},
            {
                "message_flow": {
                    "input_quiet_seconds": 3,
                    "input_max_wait_seconds": 2,
                }
            },
            {"message_flow": {"max_reply_messages": True}},
            {
                "message_flow": {
                    "inter_message_min_delay_seconds": 3,
                    "inter_message_max_delay_seconds": 1,
                }
            },
        ]
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                load_message_flow_config(settings)

    def test_env_loader_handles_comments_quotes_and_does_not_override_environment(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# comment\nFIRST='one'\nSECOND=two=three\n",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"FIRST": "existing"}, clear=False):
                values = load_env_file(path)
                self.assertEqual(values, {"FIRST": "one", "SECOND": "two=three"})
                self.assertEqual(__import__("os").environ["FIRST"], "existing")

            invalid = Path(directory) / "invalid.env"
            invalid.write_text("BROKEN", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "строка 1"):
                load_env_file(invalid)

        self.assertEqual(load_env_file(Path(directory) / "missing.env"), {})

    def test_telegram_value_helpers_cover_ids_names_and_message_kinds(self) -> None:
        self.assertEqual(normalize_target(" -123 "), -123)
        self.assertEqual(normalize_target(" @name "), "@name")
        with self.assertRaises(ValueError):
            normalize_target("   ")
        self.assertEqual(positive_int("3"), 3)
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")
        self.assertEqual(display_name(None), "неизвестно")
        with patch("telegram_client.utils.get_display_name", return_value=""):
            self.assertEqual(display_name(SimpleNamespace(username="anna", id=7)), "anna")
            self.assertEqual(display_name(SimpleNamespace(username=None, id=7)), "7")
        self.assertEqual(
            message_text(SimpleNamespace(raw_text="a\r\nb", media=None)),
            "a  ⏎ b",
        )
        self.assertEqual(message_text(SimpleNamespace(raw_text="", media=True)), "[медиа без подписи]")
        self.assertEqual(message_text(SimpleNamespace(raw_text="", media=None)), "[служебное сообщение]")

    def test_detects_telegram_photos_and_supported_image_documents(self) -> None:
        photo = SimpleNamespace(
            photo=object(),
            file=None,
            message=SimpleNamespace(photo=None, file=None),
        )
        document = SimpleNamespace(
            photo=None,
            file=SimpleNamespace(mime_type="image/png"),
            message=None,
        )
        unsupported = SimpleNamespace(
            photo=None,
            file=SimpleNamespace(mime_type="application/pdf"),
            message=None,
        )

        self.assertEqual(telegram_image_mime_type(photo), "image/jpeg")
        self.assertEqual(telegram_image_mime_type(document), "image/png")
        self.assertIsNone(telegram_image_mime_type(unsupported))


class MilanaMessageResponderTests(unittest.IsolatedAsyncioTestCase):
    async def test_typing_action_covers_generation_and_sending(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, openai_client = make_responder(clock)
        event = make_event(clock.value)
        action_active = False

        class TrackedTypingAction:
            async def __aenter__(self) -> None:
                nonlocal action_active
                action_active = True

            async def __aexit__(self, exc_type, exc, traceback) -> None:
                nonlocal action_active
                action_active = False

        client.action.return_value = TrackedTypingAction()

        async def generate_while_typing(**kwargs):
            self.assertTrue(action_active)
            return structured_response("Готовый ответ")

        async def reply_while_typing(text: str):
            self.assertTrue(action_active)
            return SimpleNamespace(id=301)

        openai_client.responses.create.side_effect = generate_while_typing
        event.reply.side_effect = reply_while_typing

        with patch("builtins.print"):
            await responder.process(event)

        client.action.assert_called_once_with("peer", "typing")
        self.assertFalse(action_active)

    async def test_different_chats_generate_answers_concurrently(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        both_generations_started = asyncio.Event()
        release_generations = asyncio.Event()
        started_chats: set[str] = set()

        async def generate_after_both_started(**kwargs):
            started_chats.add(str(kwargs["input"][-1]["content"]))
            if len(started_chats) == 2:
                both_generations_started.set()
            await release_generations.wait()
            return structured_response("Готовый ответ")

        openai_client.responses.create.side_effect = generate_after_both_started
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        responder._import_existing_history = AsyncMock()

        first = make_event(clock.value, text="Чат A", chat_id=100, message_id=300)
        second = make_event(
            clock.value,
            text="Чат B",
            chat_id=200,
            sender_id=201,
            message_id=301,
        )
        tasks = [
            asyncio.create_task(responder.process(first)),
            asyncio.create_task(responder.process(second)),
        ]
        try:
            with patch("builtins.print"):
                await asyncio.wait_for(both_generations_started.wait(), timeout=1)
                self.assertEqual(len(started_chats), 2)
                release_generations.set()
                await asyncio.gather(*tasks)
        finally:
            release_generations.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_two_messages_during_quiet_window_share_one_model_call(self) -> None:
        clock = GatedClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, openai_client = make_responder(clock)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        first = make_event(clock.value, text="Первое сообщение", message_id=300)
        second = make_event(clock.value, text="Второе сообщение", message_id=301)

        with patch("builtins.print"):
            worker = await responder.submit(first)
            _, first_quiet = await asyncio.wait_for(clock.sleep_calls.get(), timeout=1)
            same_worker = await responder.submit(second)
            _, final_quiet = await asyncio.wait_for(clock.sleep_calls.get(), timeout=1)
            self.assertIs(worker, same_worker)
            self.assertFalse(first_quiet.is_set())
            final_quiet.set()
            await asyncio.wait_for(worker, timeout=1)

        openai_client.responses.create.assert_awaited_once()
        request_input = openai_client.responses.create.await_args.kwargs["input"]
        serialized = str(request_input)
        self.assertEqual(serialized.count("Первое сообщение"), 1)
        self.assertEqual(serialized.count("Второе сообщение"), 1)
        self.assertLess(serialized.index("Первое сообщение"), serialized.index("Второе сообщение"))
        first.reply.assert_not_awaited()
        second.reply.assert_awaited_once_with("Готовый ответ")
        client.send_read_acknowledge.assert_awaited_once_with(
            "peer",
            message=second.message,
            max_id=second.id,
        )

    async def test_input_quiet_window_never_exceeds_maximum_deadline(self) -> None:
        started_at = datetime(2026, 7, 13, 21, 0, tzinfo=YEKT)
        clock = GatedClock(started_at)
        responder, _, openai_client = make_responder(clock)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()

        with patch("builtins.print"):
            worker = await responder.submit(
                make_event(clock.value, text="batch-0", message_id=300)
            )
            requested_delay, _ = await asyncio.wait_for(
                clock.sleep_calls.get(), timeout=1
            )
            self.assertEqual(requested_delay, 2)

            # Every arrival comes before the current two-second quiet period ends,
            # so it resets quiet time while the absolute eight-second cap stays put.
            for index, elapsed in enumerate((1.5, 3.0, 4.5, 6.0, 7.5), start=1):
                clock.value = started_at + timedelta(seconds=elapsed)
                same_worker = await responder.submit(
                    make_event(
                        clock.value,
                        text=f"batch-{index}",
                        message_id=300 + index,
                    )
                )
                self.assertIs(worker, same_worker)
                requested_delay, release = await asyncio.wait_for(
                    clock.sleep_calls.get(), timeout=1
                )

            self.assertEqual(requested_delay, 0.5)
            release.set()
            await asyncio.wait_for(worker, timeout=1)

        self.assertEqual(clock.value, started_at + timedelta(seconds=8))
        self.assertEqual(clock.delays, [2, 2, 2, 2, 2, 0.5])
        openai_client.responses.create.assert_awaited_once()

    async def test_stale_telegram_date_still_resets_quiet_from_local_arrival(self) -> None:
        started_at = datetime(2026, 7, 13, 21, 0, tzinfo=YEKT)
        clock = GatedClock(started_at)
        responder, _, openai_client = make_responder(clock)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        first = make_event(started_at, text="Первая часть", message_id=300)

        with patch("builtins.print"):
            worker = await responder.submit(first)
            first_delay, _ = await asyncio.wait_for(
                clock.sleep_calls.get(), timeout=1
            )
            self.assertEqual(first_delay, 2)

            clock.value = started_at + timedelta(seconds=1.9)
            second = make_event(
                started_at,
                text="Задержанная в доставке часть",
                message_id=301,
            )
            await responder.submit(second)
            reset_delay, finish_reset_quiet = await asyncio.wait_for(
                clock.sleep_calls.get(), timeout=1
            )
            self.assertEqual(reset_delay, 2)
            finish_reset_quiet.set()
            await asyncio.wait_for(worker, timeout=1)

        openai_client.responses.create.assert_awaited_once()
        second.reply.assert_awaited_once_with("Готовый ответ")

    async def test_message_during_quiet_cleanup_starts_a_fresh_quiet_wait(self) -> None:
        class SlowCancelEvent(asyncio.Event):
            def __init__(self) -> None:
                super().__init__()
                self.cleanup_started = asyncio.Event()
                self.release_cleanup = asyncio.Event()

            async def wait(self):
                try:
                    return await super().wait()
                except asyncio.CancelledError:
                    self.cleanup_started.set()
                    await self.release_cleanup.wait()
                    raise

        clock = GatedClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        schedule_started = asyncio.Event()
        release_schedule = asyncio.Event()

        async def hold_schedule(*args, **kwargs):
            schedule_started.set()
            await release_schedule.wait()

        responder._wait_before_reading = AsyncMock(side_effect=hold_schedule)
        responder._wait_for_full_online_window = AsyncMock()
        first = make_event(clock.value, text="Первая часть", message_id=300)

        with patch("builtins.print"):
            worker = await responder.submit(first)
            await asyncio.wait_for(schedule_started.wait(), timeout=1)
            slow_signal = SlowCancelEvent()
            async with responder._chat_states_lock:
                responder._chat_states[100].changed = slow_signal
            release_schedule.set()

            first_delay, finish_first_quiet = await asyncio.wait_for(
                clock.sleep_calls.get(), timeout=1
            )
            self.assertEqual(first_delay, 2)
            finish_first_quiet.set()
            await asyncio.wait_for(slow_signal.cleanup_started.wait(), timeout=1)

            second = make_event(
                clock.value,
                text="Пришла во время cleanup",
                message_id=301,
            )
            await responder.submit(second)
            slow_signal.release_cleanup.set()

            second_delay, finish_second_quiet = await asyncio.wait_for(
                clock.sleep_calls.get(), timeout=1
            )
            self.assertEqual(second_delay, 2)
            openai_client.responses.create.assert_not_awaited()
            finish_second_quiet.set()
            await asyncio.wait_for(worker, timeout=1)

        openai_client.responses.create.assert_awaited_once()
        second.reply.assert_awaited_once_with("Готовый ответ")

    async def test_large_out_of_order_batch_is_sorted_deduplicated_and_multimodal(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, client, openai_client = make_responder(clock, message_flow=flow)
        schedule_started = asyncio.Event()
        release_schedule = asyncio.Event()

        async def hold_first_batch(*args, **kwargs) -> None:
            schedule_started.set()
            await release_schedule.wait()

        responder._wait_before_reading = AsyncMock(side_effect=hold_first_batch)
        responder._wait_for_full_online_window = AsyncMock()

        texts_by_index = [f"batch-{index:04d}" for index in range(35)]
        events_by_id = {
            1000 + index: make_event(
                clock.value + timedelta(seconds=40 if index == 33 else index),
                text=text,
                message_id=1000 + index,
                photo=index in {4, 31},
                image_bytes=f"image-{index}".encode(),
            )
            for index, text in enumerate(texts_by_index)
        }
        submitted = [events_by_id[message_id] for message_id in reversed(events_by_id)]
        duplicate = make_event(
            clock.value + timedelta(seconds=99),
            text="duplicate-must-not-reach-model",
            message_id=1012,
        )

        with patch("builtins.print"):
            worker = await responder.submit(submitted[0])
            await asyncio.wait_for(schedule_started.wait(), timeout=1)
            for event in [*submitted[1:], duplicate]:
                self.assertIs(worker, await responder.submit(event))
            release_schedule.set()
            await asyncio.wait_for(worker, timeout=2)

        openai_client.responses.create.assert_awaited_once()
        request_input = openai_client.responses.create.await_args.kwargs["input"]
        observed_texts: list[str] = []
        image_items = 0
        for item in request_input:
            if item.get("role") != "user":
                continue
            content = item["content"]
            if isinstance(content, str):
                user_text = content
            else:
                image_items += sum(
                    part.get("type") == "input_image" for part in content
                )
                user_text = next(
                    part["text"] for part in content if part.get("type") == "input_text"
                )
            observed_texts.append(user_text.split(": ", 1)[-1])

        expected_texts = [*texts_by_index[:33], texts_by_index[34], texts_by_index[33]]
        self.assertEqual(observed_texts, expected_texts)
        self.assertEqual(image_items, 2)
        self.assertNotIn("duplicate-must-not-reach-model", str(request_input))
        events_by_id[1033].reply.assert_awaited_once()
        for message_id, event in events_by_id.items():
            if message_id != 1033:
                event.reply.assert_not_awaited()
        client.send_read_acknowledge.assert_awaited_once_with(
            "peer",
            message=events_by_id[1034].message,
            max_id=1034,
        )
        events_by_id[1004].message.download_media.assert_awaited_once()
        events_by_id[1031].message.download_media.assert_awaited_once()

    async def test_one_broken_photo_does_not_discard_the_rest_of_the_batch(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, _, openai_client = make_responder(clock, message_flow=flow)
        schedule_started = asyncio.Event()
        release_schedule = asyncio.Event()

        async def hold_batch(*args, **kwargs):
            schedule_started.set()
            await release_schedule.wait()

        responder._wait_before_reading = AsyncMock(side_effect=hold_batch)
        responder._wait_for_full_online_window = AsyncMock()
        first = make_event(clock.value, text="До фото", message_id=300)
        broken_photo = make_event(
            clock.value + timedelta(seconds=1),
            text="Подпись у недоступного фото",
            message_id=301,
            photo=True,
        )
        broken_photo.message.download_media.side_effect = OSError("download failed")
        last = make_event(
            clock.value + timedelta(seconds=2),
            text="После фото",
            message_id=302,
        )

        with patch("builtins.print"):
            worker = await responder.submit(first)
            await asyncio.wait_for(schedule_started.wait(), timeout=1)
            await responder.submit(broken_photo)
            await responder.submit(last)
            release_schedule.set()
            await asyncio.wait_for(worker, timeout=1)

        openai_client.responses.create.assert_awaited_once()
        request_input = openai_client.responses.create.await_args.kwargs["input"]
        serialized = str(request_input)
        for text in ("До фото", "Подпись у недоступного фото", "После фото"):
            self.assertEqual(serialized.count(text), 1)
        self.assertNotIn("input_image", serialized)
        last.reply.assert_awaited_once_with("Готовый ответ")

    async def test_message_received_during_schedule_wait_joins_the_same_batch(self) -> None:
        clock = GatedClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        first = make_event(clock.value, text="Первая часть", message_id=300)
        second = make_event(clock.value, text="Вторая часть", message_id=301)

        with patch("builtins.print"):
            worker = await responder.submit(first)
            _, schedule_release = await asyncio.wait_for(
                clock.sleep_calls.get(), timeout=1
            )
            same_worker = await responder.submit(second)
            self.assertIs(worker, same_worker)
            schedule_release.set()
            quiet_delay, quiet_release = await asyncio.wait_for(
                clock.sleep_calls.get(), timeout=1
            )
            self.assertEqual(quiet_delay, 2)
            openai_client.responses.create.assert_not_awaited()
            quiet_release.set()
            await asyncio.wait_for(worker, timeout=1)

        openai_client.responses.create.assert_awaited_once()
        request_input = str(openai_client.responses.create.await_args.kwargs["input"])
        self.assertEqual(request_input.count("Первая часть"), 1)
        self.assertEqual(request_input.count("Вторая часть"), 1)
        first.reply.assert_not_awaited()
        second.reply.assert_awaited_once_with("Готовый ответ")

    async def test_shutdown_cancels_waiting_workers_and_rejects_new_events(self) -> None:
        clock = GatedClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        event = make_event(clock.value)

        with patch("builtins.print"):
            worker = await responder.submit(event)
            await asyncio.wait_for(clock.sleep_calls.get(), timeout=1)
            await responder.shutdown()

        self.assertTrue(worker.done())
        self.assertEqual(responder._chat_states, {})
        openai_client.responses.create.assert_not_awaited()
        with self.assertRaises(RuntimeError):
            await responder.submit(make_event(clock.value, message_id=301))

    async def test_message_during_generation_cancels_stale_draft_and_diary(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, _, openai_client = make_responder(clock, message_flow=flow)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        stale_generation_started = asyncio.Event()
        stale_generation_cancelled = asyncio.Event()
        release_stale_generation = asyncio.Event()
        diary_call = SimpleNamespace(
            type="function_call",
            name="write_diary",
            arguments='{"content":"Устаревшая запись"}',
            call_id="stale-diary",
        )
        call_number = 0

        async def generate(**kwargs):
            nonlocal call_number
            call_number += 1
            if call_number == 1:
                return structured_response("Черновик", output=[diary_call])
            if call_number == 2:
                stale_generation_started.set()
                try:
                    await release_stale_generation.wait()
                except asyncio.CancelledError:
                    stale_generation_cancelled.set()
                    raise
                return structured_response("Устаревший ответ")
            return structured_response("Актуальный ответ")

        openai_client.responses.create.side_effect = generate
        first = make_event(clock.value, text="Первый вопрос", message_id=300)
        second = make_event(clock.value, text="Дополнение", message_id=301)
        second.reply.return_value = SimpleNamespace(id=401)

        with patch("builtins.print"):
            worker = await responder.submit(first)
            try:
                await asyncio.wait_for(stale_generation_started.wait(), timeout=1)
                await responder.submit(second)
                await asyncio.wait_for(worker, timeout=1)
            finally:
                release_stale_generation.set()
                if not worker.done():
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)

        self.assertTrue(stale_generation_cancelled.is_set())
        self.assertEqual(openai_client.responses.create.await_count, 3)
        fresh_input = openai_client.responses.create.await_args_list[-1].kwargs["input"]
        serialized = str(fresh_input)
        self.assertEqual(serialized.count("Первый вопрос"), 1)
        self.assertEqual(serialized.count("Дополнение"), 1)
        self.assertNotIn("stale-diary", serialized)
        first.reply.assert_not_awaited()
        second.reply.assert_awaited_once_with("Актуальный ответ")
        self.assertEqual(responder.memory.get_diary(), [])
        self.assertEqual(
            [
                item.content
                for item in responder.memory.get_chat_history(100)
                if item.role == "assistant"
            ],
            ["Актуальный ответ"],
        )

    async def test_structured_semantic_messages_are_sent_and_stored_separately(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, client, openai_client = make_responder(clock, message_flow=flow)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        openai_client.responses.create.return_value = structured_response(
            "Первая мысль",
            "Вторая мысль",
        )
        event = make_event(clock.value)
        event.reply.return_value = SimpleNamespace(id=401)
        client.send_message.return_value = SimpleNamespace(id=402)

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_awaited_once_with("Первая мысль")
        client.send_message.assert_awaited_once_with(100, "Вторая мысль")
        request = openai_client.responses.create.await_args.kwargs
        self.assertEqual(
            request["text"]["format"]["schema"]["properties"]["messages"]["maxItems"],
            responder.config.message_flow.max_reply_messages,
        )
        assistant_messages = [
            item
            for item in responder.memory.get_chat_history(100)
            if item.role == "assistant"
        ]
        self.assertEqual(
            [(item.content, item.telegram_message_id) for item in assistant_messages],
            [("Первая мысль", 401), ("Вторая мысль", 402)],
        )

    async def test_new_input_during_inter_part_delay_stops_series_and_continues(self) -> None:
        clock = GatedClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=1,
            inter_message_max_delay_seconds=1,
        )
        responder, client, openai_client = make_responder(clock, message_flow=flow)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        openai_client.responses.create.side_effect = [
            structured_response("Отправленный префикс", "Устаревший остаток"),
            structured_response("Новое продолжение"),
        ]
        first = make_event(clock.value, text="Начальный вопрос", message_id=300)
        second = make_event(clock.value, text="Уточнение", message_id=301)
        first.reply.return_value = SimpleNamespace(id=501)
        second.reply.return_value = SimpleNamespace(id=502)

        with patch("builtins.print"):
            worker = await responder.submit(first)
            _, inter_part_release = await asyncio.wait_for(
                clock.sleep_calls.get(), timeout=1
            )
            try:
                await responder.submit(second)
                await asyncio.wait_for(worker, timeout=1)
            finally:
                inter_part_release.set()
                if not worker.done():
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)

        first.reply.assert_awaited_once_with("Отправленный префикс")
        second.reply.assert_awaited_once_with("Новое продолжение")
        client.send_message.assert_not_awaited()
        self.assertEqual(openai_client.responses.create.await_count, 2)
        continuation_input = openai_client.responses.create.await_args_list[-1].kwargs[
            "input"
        ]
        self.assertEqual(str(continuation_input).count("Отправленный префикс"), 1)
        assistant_parts = [
            item.content
            for item in responder.memory.get_chat_history(100)
            if item.role == "assistant"
        ]
        self.assertEqual(assistant_parts, ["Отправленный префикс", "Новое продолжение"])

    async def test_partial_send_stores_only_successful_prefix(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, client, openai_client = make_responder(clock, message_flow=flow)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        responder.presence.finish_response = AsyncMock(
            wraps=responder.presence.finish_response
        )
        openai_client.responses.create.return_value = structured_response(
            "Успешный префикс",
            "Неотправленный остаток",
        )
        event = make_event(clock.value)
        event.reply.return_value = SimpleNamespace(id=601)
        client.send_message.side_effect = OSError("Telegram недоступен")

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_awaited_once_with("Успешный префикс")
        client.send_message.assert_awaited_once_with(100, "Неотправленный остаток")
        responder.presence.finish_response.assert_awaited_once_with(answered=True)
        assistant_messages = [
            item
            for item in responder.memory.get_chat_history(100)
            if item.role == "assistant"
        ]
        self.assertEqual(
            [(item.content, item.telegram_message_id) for item in assistant_messages],
            [("Успешный префикс", 601)],
        )

    async def test_waits_before_reading_and_uses_current_schedule_prompt(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, openai_client = make_responder(clock)
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        self.assertEqual(clock.delays, [2, 2])
        client.send_read_acknowledge.assert_awaited_once()
        openai_client.responses.create.assert_awaited_once()
        event.reply.assert_awaited_once_with("Готовый ответ")
        instructions = openai_client.responses.create.await_args.kwargs["instructions"]
        self.assertIn("состояние «Свободное время»", instructions)

    async def test_sleep_keeps_message_unread_until_morning_delay_finishes(self) -> None:
        class BlockingClock(AdvancingClock):
            def __init__(self, value: datetime) -> None:
                super().__init__(value)
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def sleep(self, seconds: float) -> None:
                self.delays.append(seconds)
                self.started.set()
                await self.release.wait()
                self.value += timedelta(seconds=seconds)

        clock = BlockingClock(datetime(2026, 7, 13, 23, 45, tzinfo=YEKT))
        responder, client, openai_client = make_responder(clock)
        event = make_event(clock.value)

        with patch("builtins.print"):
            task = asyncio.create_task(responder.process(event))
            await clock.started.wait()

            client.send_read_acknowledge.assert_not_awaited()
            openai_client.responses.create.assert_not_awaited()
            event.reply.assert_not_awaited()

            clock.release.set()
            await task

        self.assertEqual(clock.delays, [7 * 60 * 60 + 45 * 60 + 15, 2])
        client.send_read_acknowledge.assert_awaited_once()
        event.reply.assert_awaited_once_with("Готовый ответ")
        instructions = openai_client.responses.create.await_args.kwargs["instructions"]
        self.assertIn("состояние «Утренние сборы»", instructions)

    async def test_media_without_caption_uses_read_delay_but_gets_no_reply(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, openai_client = make_responder(clock)
        event = make_event(clock.value, text="")

        with patch("builtins.print"):
            await responder.process(event)

        self.assertEqual(clock.delays, [2, 2])
        client.send_read_acknowledge.assert_awaited_once()
        openai_client.responses.create.assert_not_awaited()
        event.reply.assert_not_awaited()

    async def test_photo_without_caption_is_sent_to_model_and_gets_reply(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        event = make_event(clock.value, text="", photo=True, image_bytes=b"jpeg")

        with patch("builtins.print"):
            await responder.process(event)

        event.message.download_media.assert_awaited_once_with(file=bytes)
        request_input = openai_client.responses.create.await_args.kwargs["input"]
        content = request_input[-1]["content"]
        self.assertEqual(
            content[0],
            {"type": "input_text", "text": "неизвестно: [фото без подписи]"},
        )
        self.assertEqual(content[1]["type"], "input_image")
        self.assertEqual(content[1]["image_url"], "data:image/jpeg;base64,anBlZw==")
        event.reply.assert_awaited_once_with("Готовый ответ")

    async def test_photo_caption_is_included_with_image(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        event = make_event(
            clock.value,
            text="Как тебе?",
            mime_type="image/png",
            image_bytes=b"png",
        )

        with patch("builtins.print"):
            await responder.process(event)

        content = openai_client.responses.create.await_args.kwargs["input"][-1]["content"]
        self.assertEqual(content[0]["text"], "неизвестно: Как тебе?")
        self.assertEqual(content[1]["image_url"], "data:image/png;base64,cG5n")

    async def test_message_received_while_online_uses_at_most_ten_seconds(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, _ = make_responder(
            clock,
            randint=lambda minimum, maximum: maximum,
        )

        with patch("builtins.print"):
            await responder.process(make_event(clock.value, message_id=300))
            await responder.process(make_event(clock.value, message_id=301))

        self.assertEqual(clock.delays, [60, 2, 10, 2])

    async def test_online_delay_finishes_before_reading_and_generation(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, openai_client = make_responder(
            clock,
            randint=lambda minimum, maximum: maximum,
        )
        await responder.presence.begin_response()
        await responder.presence.finish_response(answered=True)
        event = make_event(clock.value, message_id=302)

        read_at: list[datetime] = []
        generation_started_at: list[datetime] = []

        async def acknowledge_after_timer(*args, **kwargs):
            read_at.append(clock.value)

        async def generate_immediately(**kwargs):
            generation_started_at.append(clock.value)
            return structured_response("Готовый ответ")

        client.send_read_acknowledge.side_effect = acknowledge_after_timer
        openai_client.responses.create.side_effect = generate_immediately

        with patch("builtins.print"):
            await responder.process(event)

        expected_start = event.message.date + timedelta(seconds=12)
        self.assertEqual(read_at, [expected_start])
        self.assertEqual(generation_started_at, [expected_start])
        self.assertEqual(clock.delays, [10, 2])
        client.send_read_acknowledge.assert_awaited_once()
        event.reply.assert_awaited_once_with("Готовый ответ")

    async def test_answer_keeps_presence_online_for_thirty_to_sixty_seconds(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        client = MagicMock()
        update_status = AsyncMock(return_value=None)
        client.side_effect = update_status
        presence = MilanaPresenceController(
            client,
            load_routine(),
            now=clock.now,
            sleep=clock.sleep,
            randint=lambda minimum, maximum: maximum,
        )

        with patch("builtins.print"):
            await presence.begin_response()
            seconds = await presence.finish_response(answered=True)

        self.assertEqual(seconds, 60)
        self.assertTrue(presence.is_online())
        self.assertEqual(presence.online_until, clock.value + timedelta(seconds=60))
        clock.value += timedelta(seconds=60)
        self.assertFalse(presence.is_online())

    async def test_answer_defers_sleep_for_thirty_minutes(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 23, 25, tzinfo=YEKT))
        client = MagicMock()
        client.side_effect = AsyncMock(return_value=None)
        presence = MilanaPresenceController(
            client,
            load_routine(),
            now=clock.now,
            sleep=clock.sleep,
            randint=lambda minimum, maximum: minimum,
        )

        await presence.begin_response()
        await presence.finish_response(answered=True)

        expected = clock.value + timedelta(minutes=30)
        self.assertEqual(presence.sleep_deferred_until, expected)
        self.assertTrue(presence.is_sleep_deferred(expected - timedelta(seconds=1)))
        self.assertFalse(presence.is_sleep_deferred(expected))

    async def test_message_during_deferred_sleep_extends_timer_from_new_reply(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 23, 25, tzinfo=YEKT))
        responder, _, _ = make_responder(clock)

        with patch("builtins.print"):
            await responder.process(make_event(clock.value, message_id=300))
            first_deadline = responder.presence.sleep_deferred_until
            self.assertIsNotNone(first_deadline)

            clock.value = datetime(2026, 7, 13, 23, 35, tzinfo=YEKT)
            await responder.process(make_event(clock.value, message_id=301))

        self.assertEqual(clock.delays[-2:], [1, 2])
        self.assertEqual(
            responder.presence.sleep_deferred_until,
            clock.value + timedelta(minutes=30),
        )
        self.assertGreater(responder.presence.sleep_deferred_until, first_deadline)

    async def test_message_received_just_before_timer_expiry_still_gets_reply(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 23, 25, tzinfo=YEKT))
        responder, _, _ = make_responder(clock)

        with patch("builtins.print"):
            await responder.process(make_event(clock.value, message_id=300))
            deadline = responder.presence.sleep_deferred_until
            self.assertIsNotNone(deadline)
            clock.value = deadline - timedelta(milliseconds=500)
            event = make_event(clock.value, message_id=301)
            await responder.process(event)

        event.reply.assert_awaited_once_with("Готовый ответ")
        self.assertGreater(responder.presence.sleep_deferred_until, deadline)

    async def test_presence_spontaneously_goes_online_for_a_few_minutes(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        client = MagicMock()
        update_status = AsyncMock(return_value=None)
        client.side_effect = update_status
        presence = MilanaPresenceController(
            client,
            load_routine(),
            now=clock.now,
            sleep=clock.sleep,
            randint=lambda minimum, maximum: minimum,
        )

        self.assertIsNone(await presence.refresh())
        self.assertFalse(presence.is_online())
        clock.value += timedelta(minutes=15)
        self.assertEqual(await presence.refresh(), 120)
        self.assertTrue(presence.is_online())
        clock.value += timedelta(minutes=2)
        self.assertIsNone(await presence.refresh())
        self.assertFalse(presence.is_online())

        offline_flags = [
            call.args[0].offline for call in update_status.await_args_list
        ]
        self.assertEqual(offline_flags, [True, False, True])

    async def test_spontaneous_online_ends_before_pre_sleep_buffer(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 23, 13, tzinfo=YEKT))
        client = MagicMock()
        client.side_effect = AsyncMock(return_value=None)
        presence = MilanaPresenceController(
            client,
            load_routine(),
            now=clock.now,
            sleep=clock.sleep,
            randint=lambda minimum, maximum: minimum,
        )

        await presence.refresh()
        clock.value += timedelta(minutes=15)
        self.assertEqual(await presence.refresh(), 60)
        self.assertEqual(
            presence.online_until,
            datetime(2026, 7, 13, 23, 29, tzinfo=YEKT),
        )

    async def test_presence_does_not_spontaneously_go_online_during_sleep(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 1, 0, tzinfo=YEKT))
        client = MagicMock()
        client.side_effect = AsyncMock(return_value=None)
        presence = MilanaPresenceController(
            client,
            load_routine(),
            now=clock.now,
            sleep=clock.sleep,
            randint=lambda minimum, maximum: minimum,
        )

        await presence.refresh()
        clock.value += timedelta(minutes=15)
        self.assertIsNone(await presence.refresh())
        self.assertFalse(presence.is_online())

    async def test_second_message_receives_history_from_the_same_chat(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        first = make_event(clock.value, text="Меня зовут Лена", message_id=300)
        second = make_event(clock.value, text="Как меня зовут?", message_id=301)

        with patch("builtins.print"):
            await responder.process(first)
            await responder.process(second)

        second_input = openai_client.responses.create.await_args_list[1].kwargs["input"]
        self.assertIn(
            {"role": "user", "content": "неизвестно: Меня зовут Лена"},
            second_input,
        )
        self.assertIn(
            {"role": "assistant", "content": "Готовый ответ"},
            second_input,
        )
        self.assertEqual(second_input[-1]["content"], "неизвестно: Как меня зовут?")

    async def test_private_history_does_not_cross_chat_boundary(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)

        with patch("builtins.print"):
            await responder.process(
                make_event(clock.value, text="Секрет чата A", chat_id=100)
            )
            await responder.process(
                make_event(
                    clock.value,
                    text="Сообщение чата B",
                    chat_id=200,
                    sender_id=201,
                )
            )

        second_input = openai_client.responses.create.await_args_list[1].kwargs["input"]
        self.assertNotIn("Секрет чата A", str(second_input))
        self.assertIn("Сообщение чата B", str(second_input))

    async def test_model_can_write_shared_diary_with_a_function_call(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        diary_call = SimpleNamespace(
            type="function_call",
            name="write_diary",
            arguments='{"content":"Лена предпочитает зелёный чай"}',
            call_id="call-1",
        )
        openai_client.responses.create.side_effect = [
            structured_response("Записываю", output=[diary_call]),
            structured_response("Запомнила"),
            structured_response("Ответ в другом чате"),
        ]

        with patch("builtins.print"):
            await responder.process(
                make_event(clock.value, text="Я люблю зелёный чай", chat_id=100)
            )

        self.assertEqual(
            [entry.content for entry in responder.memory.get_diary()],
            ["Лена предпочитает зелёный чай"],
        )
        tool_result_input = openai_client.responses.create.await_args_list[1].kwargs["input"]
        self.assertEqual(tool_result_input[-1]["type"], "function_call_output")
        self.assertEqual(tool_result_input[-1]["call_id"], "call-1")

        with patch("builtins.print"):
            await responder.process(
                make_event(
                    clock.value,
                    text="Что важно помнить?",
                    chat_id=200,
                    sender_id=201,
                )
            )
        other_chat_instructions = (
            openai_client.responses.create.await_args_list[2].kwargs["instructions"]
        )
        self.assertIn("Лена предпочитает зелёный чай", other_chat_instructions)

    async def test_failed_send_does_not_store_assistant_turn(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, _ = make_responder(clock)
        event = make_event(clock.value)
        event.reply.side_effect = OSError("Telegram недоступен")

        with patch("builtins.print"):
            await responder.process(event)

        history = responder.memory.get_chat_history(event.chat_id)
        self.assertEqual([item.role for item in history], ["user"])

    async def test_duplicate_event_is_not_answered_twice(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)
            await responder.process(event)

        openai_client.responses.create.assert_awaited_once()
        event.reply.assert_awaited_once()

    async def test_imports_existing_telegram_history_before_first_answer(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, openai_client = make_responder(clock)
        historical = [
            SimpleNamespace(
                id=11,
                raw_text="Ранее отвечала Милана",
                out=True,
                date=clock.value,
            ),
            SimpleNamespace(
                id=10,
                raw_text="Старый вопрос",
                out=False,
                sender_id=200,
                date=clock.value,
                get_sender=AsyncMock(return_value=None),
            ),
        ]

        async def iter_history():
            for message in historical:
                yield message

        client.iter_messages.return_value = iter_history()
        event = make_event(clock.value, text="Новый вопрос", message_id=12)

        with patch("builtins.print"):
            await responder.process(event)

        request_input = openai_client.responses.create.await_args.kwargs["input"]
        self.assertEqual(
            request_input[:2],
            [
                {"role": "user", "content": "неизвестно: Старый вопрос"},
                {"role": "assistant", "content": "Ранее отвечала Милана"},
            ],
        )
        client.iter_messages.assert_called_once_with(100, limit=40, max_id=12)

    async def test_retries_without_temperature_when_model_rejects_it(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        error = BadRequestError(
            "Unsupported parameter: 'temperature' is not supported with this model.",
            response=httpx.Response(400, request=request),
            body={"param": "temperature"},
        )
        openai_client.responses.create.side_effect = [
            error,
            structured_response("Ответ без temperature"),
        ]

        event = make_event(clock.value)
        with patch("builtins.print"):
            await responder.process(event)

        self.assertIn(
            "temperature",
            openai_client.responses.create.await_args_list[0].kwargs,
        )
        self.assertNotIn(
            "temperature",
            openai_client.responses.create.await_args_list[1].kwargs,
        )
        event.reply.assert_awaited_once_with("Ответ без temperature")

    async def test_retries_as_one_plain_message_when_structured_output_is_unsupported(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        error = BadRequestError(
            "Structured Outputs are not supported with this model.",
            response=httpx.Response(400, request=request),
            body={"param": "text.format"},
        )
        openai_client.responses.create.side_effect = [
            error,
            SimpleNamespace(output_text="Обычный ответ", output=[]),
        ]
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        self.assertIn("text", openai_client.responses.create.await_args_list[0].kwargs)
        self.assertNotIn("text", openai_client.responses.create.await_args_list[1].kwargs)
        self.assertIn(
            "Верни только один готовый текст Telegram-сообщения без JSON",
            openai_client.responses.create.await_args_list[1].kwargs["instructions"],
        )
        event.reply.assert_awaited_once_with("Обычный ответ")

    async def test_structured_schema_error_is_not_misread_as_unsupported(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        error = BadRequestError(
            "Invalid schema: required must include every property.",
            response=httpx.Response(400, request=request),
            body={"param": "text.format"},
        )
        openai_client.responses.create.side_effect = error
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        openai_client.responses.create.assert_awaited_once()
        self.assertIn("text", openai_client.responses.create.await_args.kwargs)
        self.assertIsNone(responder._supports_structured_reply)
        event.reply.assert_not_awaited()

    async def test_shutdown_cancels_in_flight_generation(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, _, openai_client = make_responder(clock, message_flow=flow)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        started = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()

        async def blocked_generation(**kwargs):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return structured_response("Уже не будет отправлено")

        openai_client.responses.create.side_effect = blocked_generation
        event = make_event(clock.value)

        with patch("builtins.print"):
            worker = await responder.submit(event)
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                await responder.shutdown()
            finally:
                release.set()
                if not worker.done():
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)

        self.assertTrue(worker.done())
        self.assertTrue(cancelled.is_set())
        self.assertEqual(responder._chat_states, {})
        event.reply.assert_not_awaited()

    async def test_shutdown_cancels_in_flight_summary(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, _, _ = make_responder(clock, message_flow=flow)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        summary_started = asyncio.Event()
        summary_cancelled = asyncio.Event()
        release_summary = asyncio.Event()

        async def blocked_summary(chat_key):
            summary_started.set()
            try:
                await release_summary.wait()
            except asyncio.CancelledError:
                summary_cancelled.set()
                raise

        responder._maybe_update_chat_summary = blocked_summary
        event = make_event(clock.value)
        event.reply.return_value = SimpleNamespace(id=401)

        with patch("builtins.print"):
            worker = await responder.submit(event)
            try:
                await asyncio.wait_for(summary_started.wait(), timeout=1)
                await responder.shutdown()
            finally:
                release_summary.set()
                if not worker.done():
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)

        self.assertTrue(worker.done())
        self.assertTrue(summary_cancelled.is_set())
        event.reply.assert_awaited_once_with("Готовый ответ")

    async def test_schedule_closing_during_begin_response_prevents_first_send(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, _, openai_client = make_responder(clock, message_flow=flow)
        responder._wait_before_reading = AsyncMock()
        schedule_open = True
        waiting_for_reopen = asyncio.Event()
        reopen = asyncio.Event()

        async def wait_for_open_window(*, continues_conversation=False):
            if not schedule_open:
                waiting_for_reopen.set()
                await reopen.wait()

        async def close_schedule():
            nonlocal schedule_open
            schedule_open = False

        responder._wait_for_full_online_window = wait_for_open_window
        responder._full_online_window_is_open = MagicMock(
            side_effect=lambda **kwargs: schedule_open
        )
        responder.presence.begin_response = AsyncMock(side_effect=close_schedule)
        responder.presence.finish_response = AsyncMock(return_value=None)
        event = make_event(clock.value)

        with patch("builtins.print"):
            worker = await responder.submit(event)
            try:
                await asyncio.wait_for(waiting_for_reopen.wait(), timeout=1)
                event.reply.assert_not_awaited()
                responder.presence.finish_response.assert_awaited_once_with(
                    answered=False
                )
                await responder.shutdown()
            finally:
                reopen.set()
                if not worker.done():
                    worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)

        openai_client.responses.create.assert_awaited_once()
        event.reply.assert_not_awaited()

    async def test_post_send_memory_and_diary_errors_still_count_as_answered(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, _, _ = make_responder(clock, message_flow=flow)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        responder.presence.finish_response = AsyncMock(
            wraps=responder.presence.finish_response
        )
        original_add_message = responder.memory.add_message

        def fail_only_assistant(chat_id, role, content, **kwargs):
            if role == "assistant":
                raise RuntimeError("SQLite write failed")
            return original_add_message(chat_id, role, content, **kwargs)

        event = make_event(clock.value)
        event.reply.return_value = SimpleNamespace(id=401)

        with (
            patch.object(
                responder.memory,
                "add_message",
                side_effect=fail_only_assistant,
            ),
            patch.object(
                responder,
                "_commit_staged_diary",
                side_effect=RuntimeError("Diary write failed"),
            ) as commit_diary,
            patch("builtins.print"),
        ):
            await responder.process(event)

        event.reply.assert_awaited_once_with("Готовый ответ")
        responder.presence.finish_response.assert_awaited_once_with(answered=True)
        commit_diary.assert_called_once()
        self.assertEqual(
            [item.role for item in responder.memory.get_chat_history(100)],
            ["user"],
        )

    async def test_structured_refusal_is_sent_as_one_message(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        openai_client.responses.create.return_value = SimpleNamespace(
            output_text="",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="refusal", refusal="Не могу помочь")],
                )
            ],
        )
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_awaited_once_with("Не могу помочь")

    async def test_blank_structured_items_are_removed_before_sending(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        flow = MessageFlowConfig(
            input_quiet_seconds=0,
            input_max_wait_seconds=0,
            max_reply_messages=2,
            inter_message_min_delay_seconds=0,
            inter_message_max_delay_seconds=0,
        )
        responder, client, openai_client = make_responder(clock, message_flow=flow)
        responder._wait_before_reading = AsyncMock()
        responder._wait_for_full_online_window = AsyncMock()
        openai_client.responses.create.return_value = SimpleNamespace(
            output_text=json.dumps(
                {"messages": ["  Первая часть  ", "", "   ", "Вторая часть"]},
                ensure_ascii=False,
            ),
            output=[],
        )
        event = make_event(clock.value)
        event.reply.return_value = SimpleNamespace(id=401)
        client.send_message.return_value = SimpleNamespace(id=402)

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_awaited_once_with("Первая часть")
        client.send_message.assert_awaited_once_with(100, "Вторая часть")

    async def test_invalid_structured_json_is_not_sent(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        openai_client.responses.create.return_value = SimpleNamespace(
            output_text="not-json",
            output=[],
        )
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        openai_client.responses.create.assert_awaited_once()
        event.reply.assert_not_awaited()
        self.assertEqual(
            [item.role for item in responder.memory.get_chat_history(100)],
            ["user"],
        )

    async def test_incomplete_structured_output_is_not_sent_or_stored(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        diary_call = SimpleNamespace(
            type="function_call",
            name="write_diary",
            arguments='{"content":"Не должна сохраниться"}',
            call_id="incomplete-diary",
        )
        openai_client.responses.create.return_value = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            output_text='{"messages":["Обрезано',
            output=[diary_call],
        )
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_not_awaited()
        openai_client.responses.create.assert_awaited_once()
        self.assertEqual(responder.memory.get_diary(), [])
        self.assertEqual(
            [item.role for item in responder.memory.get_chat_history(100)],
            ["user"],
        )

    async def test_invalid_diary_call_returns_tool_error_without_writing(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, _ = make_responder(clock)
        call = SimpleNamespace(arguments="not-json")

        result = json.loads(
            responder._execute_diary_call(call, chat_key=100, source_message_id=5)
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(responder.memory.get_diary(), [])

    async def test_repeated_diary_call_reports_existing_entry(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, _ = make_responder(clock)
        call = SimpleNamespace(arguments='{"content":"Один факт"}')

        first = json.loads(responder._execute_diary_call(call, chat_key=1, source_message_id=1))
        second = json.loads(responder._execute_diary_call(call, chat_key=2, source_message_id=2))

        self.assertEqual(first["status"], "stored")
        self.assertEqual(second["status"], "already_exists")

    async def test_empty_model_answer_is_not_sent_or_stored_as_assistant(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)
        openai_client.responses.create.return_value = structured_response("   ")
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_not_awaited()
        self.assertEqual([item.role for item in responder.memory.get_chat_history(100)], ["user"])

    async def test_multi_part_answer_uses_reply_then_send_message(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, openai_client = make_responder(clock)
        openai_client.responses.create.return_value = structured_response("a" * 4001)
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_awaited_once_with("a" * 4000)
        client.send_message.assert_awaited_once_with(100, "a")
        assistant_parts = [
            item.content
            for item in responder.memory.get_chat_history(100)
            if item.role == "assistant"
        ]
        self.assertEqual(assistant_parts, ["a" * 4000, "a"])

    async def test_read_acknowledgement_failure_still_allows_answer(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, _ = make_responder(clock)
        client.send_read_acknowledge.side_effect = OSError("offline")
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_awaited_once_with("Готовый ответ")

    async def test_dynamic_summary_triggers_on_60_user_messages_and_prepends_context(self) -> None:
        """When ~60 user messages accumulate the summarizer runs and Milana receives summary + recent 30."""
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, _, openai_client = make_responder(clock)

        chat = 777
        # Pre-populate just under the window (59 users). Interleave a few assistant turns.
        for i in range(59):
            responder.memory.add_message(chat, "user", f"u{i}", sender_name="Тест")
            if i % 2 == 0:
                responder.memory.add_message(chat, "assistant", f"a{i}")

        # The next (60th) user message should cross the threshold inside process
        event = make_event(clock.value, text="Юбилейное сообщение 60", chat_id=chat, message_id=5000, sender_id=123)

        # Structured answer calls and the plain-text summarizer share one client.
        async def answer_or_summarize(**kwargs):
            if "text" in kwargs:
                return structured_response("Поняла, кофе и отпуск.")
            return SimpleNamespace(
                output_text="Ключевые темы: имя Тест, любит кофе, планирует отпуск."
            )

        openai_client.responses.create.side_effect = answer_or_summarize

        with patch("builtins.print"):
            await responder.process(event)

        # Two model calls: first summarizer, second the actual answer
        self.assertGreaterEqual(openai_client.responses.create.call_count, 2)

        # Summary was persisted
        info = responder.memory.get_chat_summary_info(chat)
        self.assertIsNotNone(info)
        self.assertIn("кофе", (info.summary if info else ""))

        answer_calls = [
            call
            for call in openai_client.responses.create.await_args_list
            if "text" in call.kwargs
        ]
        self.assertIn("Юбилейное сообщение 60", str(answer_calls[0].kwargs["input"]))

        # Once the worker is idle, the persisted summary is available to the next answer.
        follow_up = make_event(
            clock.value,
            text="Что ты помнишь?",
            chat_id=chat,
            message_id=5001,
            sender_id=123,
        )
        with patch("builtins.print"):
            await responder.process(follow_up)

        answer_calls = [
            call
            for call in openai_client.responses.create.await_args_list
            if "text" in call.kwargs
        ]
        joined = str(answer_calls[-1].kwargs["input"])
        self.assertIn("Ключевые темы", joined)
        self.assertIn("Что ты помнишь?", joined)

        # Covered count advanced so that active user window is reset toward 30
        self.assertLessEqual(info.covered_user_messages if info else 0, 59)


if __name__ == "__main__":
    unittest.main()
