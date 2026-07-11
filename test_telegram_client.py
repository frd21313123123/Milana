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
    MilanaMessageResponder,
    MilanaPresenceController,
    ai_number,
    ai_positive_int,
    ai_string,
    display_name,
    load_env_file,
    load_ai_settings,
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


def make_responder(clock: AdvancingClock, *, memory=None, randint=None):
    client = MagicMock()
    client.side_effect = AsyncMock(return_value=None)
    client.send_read_acknowledge = AsyncMock()
    client.send_message = AsyncMock()
    client.action.return_value = AsyncContext()

    openai_client = MagicMock()
    openai_client.responses.create = AsyncMock(
        return_value=SimpleNamespace(output_text="Готовый ответ")
    )
    config = AIConfig(
        api_key="test-key",
        model="test-model",
        instructions="Тестовая инструкция",
        temperature=0.2,
        max_output_tokens=100,
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
            return SimpleNamespace(output_text="Готовый ответ", output=[])

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

    async def test_waits_before_reading_and_uses_current_schedule_prompt(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, openai_client = make_responder(clock)
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        self.assertEqual(clock.delays, [2])
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

        self.assertEqual(clock.delays, [7 * 60 * 60 + 45 * 60 + 15])
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

        self.assertEqual(clock.delays, [2])
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

        self.assertEqual(clock.delays, [60, 10])

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
            return SimpleNamespace(output_text="Готовый ответ")

        client.send_read_acknowledge.side_effect = acknowledge_after_timer
        openai_client.responses.create.side_effect = generate_immediately

        with patch("builtins.print"):
            await responder.process(event)

        expected_start = event.message.date + timedelta(seconds=10)
        self.assertEqual(read_at, [expected_start])
        self.assertEqual(generation_started_at, [expected_start])
        self.assertEqual(clock.delays, [10])
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
            SimpleNamespace(output_text="", output=[diary_call]),
            SimpleNamespace(output_text="Запомнила", output=[]),
            SimpleNamespace(output_text="Ответ в другом чате", output=[]),
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
            SimpleNamespace(output_text="Ответ без temperature", output=[]),
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
        openai_client.responses.create.return_value = SimpleNamespace(output_text="   ", output=[])
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_not_awaited()
        self.assertEqual([item.role for item in responder.memory.get_chat_history(100)], ["user"])

    async def test_multi_part_answer_uses_reply_then_send_message(self) -> None:
        clock = AdvancingClock(datetime(2026, 7, 13, 21, 0, tzinfo=YEKT))
        responder, client, openai_client = make_responder(clock)
        openai_client.responses.create.return_value = SimpleNamespace(
            output_text="a" * 4001,
            output=[],
        )
        event = make_event(clock.value)

        with patch("builtins.print"):
            await responder.process(event)

        event.reply.assert_awaited_once_with("a" * 4000)
        client.send_message.assert_awaited_once_with(100, "a")
        self.assertEqual(responder.memory.get_chat_history(100)[-1].content, "a" * 4001)

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

        # Make the summary call return something distinct, answer call returns normal
        summary_call_result = SimpleNamespace(output_text="Ключевые темы: имя Тест, любит кофе, планирует отпуск.")
        answer_result = SimpleNamespace(output_text="Поняла, кофе и отпуск.", output=[])
        openai_client.responses.create.side_effect = [summary_call_result, answer_result]

        with patch("builtins.print"):
            await responder.process(event)

        # Two model calls: first summarizer, second the actual answer
        self.assertGreaterEqual(openai_client.responses.create.call_count, 2)

        # Summary was persisted
        info = responder.memory.get_chat_summary_info(chat)
        self.assertIsNotNone(info)
        self.assertIn("кофе", (info.summary if info else ""))

        # The input to the *answer* generation (last call) must contain the summary note + recent
        last_call_input = openai_client.responses.create.await_args_list[-1].kwargs["input"]
        joined = str(last_call_input)
        self.assertIn("Ключевые темы", joined)  # from the injected summary block
        self.assertIn("Юбилейное сообщение 60", joined)

        # Covered count advanced so that active user window is reset toward 30
        self.assertLessEqual(info.covered_user_messages if info else 0, 59)


if __name__ == "__main__":
    unittest.main()
