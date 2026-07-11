import asyncio
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
    ai_positive_int,
    load_ai_settings,
    split_telegram_text,
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


def make_responder(clock: AdvancingClock, *, memory=None):
    client = MagicMock()
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
        randint=lambda minimum, maximum: minimum,
    )
    return responder, client, openai_client


def make_event(
    value: datetime,
    *,
    text: str = "Привет",
    chat_id: int = 100,
    sender_id: int = 200,
    message_id: int = 300,
):
    event = SimpleNamespace(
        chat_id=chat_id,
        sender_id=sender_id,
        id=message_id,
        raw_text=text,
        message=SimpleNamespace(date=value),
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


class MilanaMessageResponderTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
