import asyncio
import base64
import copy
import json
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

from agy_provider import (
    AgyAuthError,
    AgyError,
    AgyModelClient,
    AgyQuotaError,
    strip_ansi,
)
from milana.builtin_skills import SCHEDULE_MESSAGE_TOOL
from milana.session import WRITE_DIARY_TOOL
from milana_stickers import OPEN_STICKER_PICKER_TOOL, SEND_STICKER_TOOL


class StripAnsiTests(unittest.TestCase):
    def test_removes_terminal_sequences_and_normalizes_line_endings(self) -> None:
        value = (
            "\x1b]0;Antigravity\x07"
            "\x1b[32mпервая\x1b[0m\r\n"
            "\x1b[2Kвторая\r"
        )

        self.assertEqual(strip_ansi(value), "первая\nвторая")

    def test_empty_text_stays_empty(self) -> None:
        self.assertEqual(strip_ansi(""), "")


class AgyModelClientTests(unittest.TestCase):
    def test_defaults_match_gemini_flash_configuration(self) -> None:
        client = AgyModelClient()

        self.assertEqual(client.model, "gemini-3.5-flash")
        self.assertEqual(client.timeout_seconds, 300)
        self.assertEqual(client.executable, "agy")
        self.assertTrue(callable(client.responses.create))

    def test_rejects_blank_model_and_non_positive_timeout(self) -> None:
        with self.assertRaises(ValueError):
            AgyModelClient(model="   ")
        with self.assertRaises(ValueError):
            AgyModelClient(timeout_seconds=0)

    def test_error_details_hide_oauth_url_and_report_auth_failure(self) -> None:
        details = AgyModelClient._safe_error_details(
            "Authentication required. Visit https://accounts.google.com/secret\n"
            "Error: authentication timed out."
        )

        self.assertEqual(
            details,
            "Antigravity CLI не авторизован или срок авторизации истёк",
        )
        self.assertNotIn("https://", details)

    def test_quota_errors_are_classified_separately(self) -> None:
        for details in (
            "RESOURCE_EXHAUSTED: quota exceeded",
            "Rate limit exceeded for gemini-3.5-flash",
            "You have remaining: 0 messages",
        ):
            with self.subTest(details=details):
                self.assertTrue(AgyModelClient._is_quota_failure(details))
                self.assertIs(AgyModelClient._error_type(details), AgyQuotaError)

        self.assertFalse(AgyModelClient._is_quota_failure("temporary network error"))
        self.assertIs(AgyModelClient._error_type("temporary network error"), AgyError)

    def test_successful_process_output_containing_quota_error_is_rejected(self) -> None:
        client = AgyModelClient()
        with (
            patch("agy_provider.platform.system", return_value="Linux"),
            patch.object(
                client,
                "_run_direct",
                return_value="RESOURCE_EXHAUSTED: quota exceeded",
            ),
        ):
            with self.assertRaises(AgyQuotaError):
                client._query({"input": []})

    def test_missing_executable_is_reported_as_agy_error(self) -> None:
        client = AgyModelClient(executable="missing-agy")
        with (
            patch("agy_provider.platform.system", return_value="Linux"),
            patch.object(client, "_run_direct", side_effect=FileNotFoundError),
        ):
            with self.assertRaisesRegex(AgyError, "не найдена"):
                client._query({"input": []})

    def test_timeout_is_reported_as_agy_error(self) -> None:
        client = AgyModelClient(timeout_seconds=17)
        timeout = subprocess.TimeoutExpired(["agy"], 17)
        with (
            patch("agy_provider.platform.system", return_value="Linux"),
            patch.object(client, "_run_direct", side_effect=timeout),
        ):
            with self.assertRaisesRegex(AgyError, "17 секунд"):
                client._query({"input": []})

    def test_command_uses_selected_model_and_keeps_prompt_last(self) -> None:
        client = AgyModelClient(model="gemini-3.5-flash", timeout_seconds=42)
        workspace = Path("temporary-workspace")

        command = client._command("короткий prompt", workspace)

        self.assertEqual(
            command[:3], ["agy", "--model", "Gemini 3.5 Flash (Medium)"]
        )
        self.assertIn("--sandbox", command)
        self.assertIn("--dangerously-skip-permissions", command)
        self.assertEqual(command[-2:], ["-p", "короткий prompt"])
        self.assertIn("42s", command)

    def test_query_uses_inline_text_without_request_file_or_dangerous_flag(self) -> None:
        client = AgyModelClient()
        observed: dict[str, Any] = {}

        def fake_run_windows(
            command: list[str],
            workspace: Path,
            cancel_event: threading.Event | None,
            *,
            stop_on_structured_output: bool,
        ) -> str:
            observed["command"] = command
            observed["request_exists"] = (workspace / "request.json").exists()
            observed["cancel_event"] = cancel_event
            observed["stop_on_structured_output"] = stop_on_structured_output
            return '{"messages":["готово"],"reaction":null}'

        request = {
            "instructions": "Отвечай кратко",
            "input": [{"role": "user", "content": "Привет"}],
            "text": {"format": {}},
        }
        with (
            patch("agy_provider.platform.system", return_value="Windows"),
            patch.object(client, "_run_windows", side_effect=fake_run_windows),
        ):
            answer = client._query(request)

        command = observed["command"]
        self.assertEqual(answer, '{"messages":["готово"],"reaction":null}')
        self.assertFalse(observed["request_exists"])
        self.assertIsNone(observed["cancel_event"])
        self.assertTrue(observed["stop_on_structured_output"])
        self.assertNotIn("--dangerously-skip-permissions", command)
        self.assertEqual(command[-2], "-p")
        self.assertIn("REQUEST_JSON:", command[-1])
        self.assertIn('"instructions":"Отвечай кратко"', command[-1])
        self.assertNotIn("Read the request file", command[-1])

    def test_query_falls_back_to_request_file_for_oversized_text(self) -> None:
        client = AgyModelClient()
        observed: dict[str, Any] = {}

        def fake_run_windows(
            command: list[str],
            workspace: Path,
            cancel_event: threading.Event | None,
            *,
            stop_on_structured_output: bool,
        ) -> str:
            request_path = workspace / "request.json"
            observed["command"] = command
            observed["request_exists"] = request_path.exists()
            observed["payload"] = json.loads(request_path.read_text(encoding="utf-8"))
            observed["stop_on_structured_output"] = stop_on_structured_output
            return "готово"

        long_text = "я" * 25_000
        with (
            patch("agy_provider.platform.system", return_value="Windows"),
            patch.object(client, "_run_windows", side_effect=fake_run_windows),
        ):
            answer = client._query(
                {"input": [{"role": "user", "content": long_text}]}
            )

        command = observed["command"]
        self.assertEqual(answer, "готово")
        self.assertTrue(observed["request_exists"])
        self.assertEqual(observed["payload"]["input"][0]["content"], long_text)
        self.assertFalse(observed["stop_on_structured_output"])
        self.assertIn("--dangerously-skip-permissions", command)
        self.assertIn("Read the request file", command[-1])

    def test_query_falls_back_to_request_file_for_image(self) -> None:
        client = AgyModelClient()
        image_bytes = b"\x89PNG\r\n\x1a\nimage"
        observed: dict[str, Any] = {}

        def fake_run_windows(
            command: list[str],
            workspace: Path,
            cancel_event: threading.Event | None,
            *,
            stop_on_structured_output: bool,
        ) -> str:
            request_path = workspace / "request.json"
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            image_item = payload["input"][0]["content"][1]
            observed["command"] = command
            observed["request_exists"] = request_path.exists()
            observed["image_item"] = image_item
            observed["image_bytes"] = Path(image_item["local_path"]).read_bytes()
            return "описание изображения"

        request = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Что здесь?"},
                        {
                            "type": "input_image",
                            "image_url": (
                                "data:image/png;base64,"
                                + base64.b64encode(image_bytes).decode("ascii")
                            ),
                        },
                    ],
                }
            ]
        }
        with (
            patch("agy_provider.platform.system", return_value="Windows"),
            patch.object(client, "_run_windows", side_effect=fake_run_windows),
        ):
            answer = client._query(request)

        command = observed["command"]
        self.assertEqual(answer, "описание изображения")
        self.assertTrue(observed["request_exists"])
        self.assertEqual(observed["image_bytes"], image_bytes)
        self.assertNotIn("image_url", observed["image_item"])
        self.assertIn("--dangerously-skip-permissions", command)
        self.assertIn("Read the request file", command[-1])

    def test_query_falls_back_to_request_file_for_video(self) -> None:
        client = AgyModelClient()
        video_bytes = b"\x1aE\xdf\xa3webm-video"
        observed: dict[str, Any] = {}

        def fake_run_windows(
            command: list[str],
            workspace: Path,
            cancel_event: threading.Event | None,
            *,
            stop_on_structured_output: bool,
        ) -> str:
            request_path = workspace / "request.json"
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            video_item = payload["input"][0]["content"][1]
            observed["command"] = command
            observed["request_exists"] = request_path.exists()
            observed["video_item"] = video_item
            observed["video_bytes"] = Path(video_item["local_path"]).read_bytes()
            return "описание видео"

        request = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Что происходит?"},
                        {
                            "type": "input_video",
                            "video_url": (
                                "data:video/webm;base64,"
                                + base64.b64encode(video_bytes).decode("ascii")
                            ),
                        },
                    ],
                }
            ]
        }
        with (
            patch("agy_provider.platform.system", return_value="Windows"),
            patch.object(client, "_run_windows", side_effect=fake_run_windows),
        ):
            answer = client._query(request)

        command = observed["command"]
        self.assertEqual(answer, "описание видео")
        self.assertTrue(observed["request_exists"])
        self.assertEqual(observed["video_bytes"], video_bytes)
        self.assertNotIn("video_url", observed["video_item"])
        self.assertEqual(observed["video_item"]["mime_type"], "video/webm")
        self.assertIn("--dangerously-skip-permissions", command)
        self.assertIn("Read the request file", command[-1])

    def test_inline_budget_counts_full_quoted_command_in_utf16_units(self) -> None:
        command = ["agy", "-p", 'ответ 😀 с "кавычками"']
        command_line = subprocess.list2cmdline(command)
        utf16_units = len(command_line.encode("utf-16-le")) // 2 + 1

        self.assertGreater(utf16_units, len(command_line) + 1)
        with (
            patch("agy_provider.platform.system", return_value="Windows"),
            patch("agy_provider.WINDOWS_INLINE_COMMAND_MAX_UNITS", utf16_units),
        ):
            self.assertTrue(AgyModelClient._inline_command_fits(command))
        with (
            patch("agy_provider.platform.system", return_value="Windows"),
            patch(
                "agy_provider.WINDOWS_INLINE_COMMAND_MAX_UNITS", utf16_units - 1
            ),
        ):
            self.assertFalse(AgyModelClient._inline_command_fits(command))

    def test_auth_retry_environment_values_and_validation(self) -> None:
        with patch.dict(
            os.environ,
            {"AGY_AUTH_RETRIES": "3", "AGY_AUTH_RETRY_DELAY_SECONDS": "0.25"},
        ):
            client = AgyModelClient()

        self.assertEqual(client.auth_retries, 3)
        self.assertEqual(client.auth_retry_delay_seconds, 0.25)

        invalid_values = (
            ("AGY_AUTH_RETRIES", "many", "целым числом"),
            ("AGY_AUTH_RETRIES", "6", "от 0 до 5"),
            ("AGY_AUTH_RETRY_DELAY_SECONDS", "fast", "должен быть числом"),
            ("AGY_AUTH_RETRY_DELAY_SECONDS", "31", "от 0 до 30"),
        )
        for name, value, expected_message in invalid_values:
            environment = {
                "AGY_AUTH_RETRIES": "2",
                "AGY_AUTH_RETRY_DELAY_SECONDS": "1",
                name: value,
            }
            with self.subTest(name=name, value=value):
                with patch.dict(os.environ, environment):
                    with self.assertRaisesRegex(ValueError, expected_message):
                        AgyModelClient()

    def test_launcher_prompt_contains_absolute_request_path(self) -> None:
        client = AgyModelClient()
        with TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"

            prompt = client._launcher_prompt(request_path, structured=True)

        self.assertIn(f'"{request_path.resolve().as_posix()}"', prompt)
        self.assertIn("Return only one JSON object", prompt)
        self.assertIn("local media files", prompt)

    def test_request_payload_materializes_image_data_url(self) -> None:
        client = AgyModelClient()
        image_bytes = b"\x89PNG\r\n\x1a\nimage"
        request = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": (
                                "data:image/png;base64,"
                                + base64.b64encode(image_bytes).decode("ascii")
                            ),
                        }
                    ],
                }
            ]
        }

        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            payload = client._request_payload(request, workspace)
            image_item = payload["input"][0]["content"][0]

            self.assertNotIn("image_url", image_item)
            self.assertEqual(image_item["mime_type"], "image/png")
            local_path = Path(image_item["local_path"])
            self.assertTrue(local_path.is_absolute())
            self.assertEqual(local_path.read_bytes(), image_bytes)

    def test_request_payload_materializes_video_data_url(self) -> None:
        client = AgyModelClient()
        video_bytes = b"\x00\x00\x00\x18ftypmp42video"
        request = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_video",
                            "video_url": (
                                "data:video/mp4;base64,"
                                + base64.b64encode(video_bytes).decode("ascii")
                            ),
                        }
                    ],
                }
            ]
        }

        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            payload = client._request_payload(request, workspace)
            video_item = payload["input"][0]["content"][0]

            self.assertNotIn("video_url", video_item)
            self.assertEqual(video_item["mime_type"], "video/mp4")
            local_path = Path(video_item["local_path"])
            self.assertTrue(local_path.is_absolute())
            self.assertEqual(local_path.suffix, ".mp4")
            self.assertEqual(local_path.read_bytes(), video_bytes)

    def test_request_payload_materializes_audio_data_url(self) -> None:
        client = AgyModelClient()
        audio_bytes = b"OggS-opus-voice"
        request = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "audio_url": (
                                "data:audio/ogg;base64,"
                                + base64.b64encode(audio_bytes).decode("ascii")
                            ),
                        }
                    ],
                }
            ]
        }

        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            payload = client._request_payload(request, workspace)
            audio_item = payload["input"][0]["content"][0]

            self.assertTrue(client._contains_media(request["input"]))
            self.assertNotIn("audio_url", audio_item)
            self.assertEqual(audio_item["mime_type"], "audio/ogg")
            local_path = Path(audio_item["local_path"])
            self.assertTrue(local_path.is_absolute())
            self.assertEqual(local_path.suffix, ".ogg")
            self.assertEqual(local_path.read_bytes(), audio_bytes)

    def test_request_payload_rejects_invalid_or_mismatched_video_data(self) -> None:
        client = AgyModelClient()
        invalid_items = (
            (
                {
                    "type": "input_video",
                    "video_url": "data:video/webm;base64,",
                },
                "Некорректный data URL видео",
            ),
            (
                {
                    "type": "input_video",
                    "video_url": "data:video/webm;base64,not-base64!",
                },
                "Некорректные данные видео",
            ),
            (
                {
                    "type": "input_video",
                    "video_url": "data:image/png;base64,aW1hZ2U=",
                },
                "не соответствует input_video",
            ),
        )

        for item, expected_message in invalid_items:
            with self.subTest(item=item):
                with TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(AgyError, expected_message):
                        client._request_payload(
                            {
                                "input": [
                                    {"role": "user", "content": [item]}
                                ]
                            },
                            Path(directory),
                        )

    def test_request_payload_adds_diary_output_without_mutating_request(self) -> None:
        client = AgyModelClient()
        request = {
            "instructions": "Системная инструкция",
            "input": [{"role": "user", "content": "Запомни это"}],
            "tools": [
                {
                    "type": "function",
                    "name": "write_diary",
                    "description": "Добавить запись в дневник",
                    "parameters": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "telegram_reply",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "reaction": {"type": ["string", "null"]},
                        },
                        "required": ["messages", "reaction"],
                        "additionalProperties": False,
                    },
                }
            },
        }
        original = copy.deepcopy(request)

        with TemporaryDirectory() as directory:
            payload = client._request_payload(request, Path(directory))

        self.assertEqual(request, original)
        response_schema = payload["response_format"]["schema"]
        self.assertEqual(
            response_schema["properties"]["diary_entries"],
            {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 4,
            },
        )
        self.assertIn("diary_entries", response_schema["required"])
        self.assertIn("diary_entries", payload["instructions"])

    def test_milana_agent_turn_uses_only_universal_tool_calls(self) -> None:
        client = AgyModelClient()
        request = {
            "instructions": "Системная инструкция",
            "input": [],
            "tools": [
                WRITE_DIARY_TOOL,
                SCHEDULE_MESSAGE_TOOL,
                OPEN_STICKER_PICKER_TOOL,
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "milana_agent_turn",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"state_update": {"type": "object"}},
                        "required": ["state_update"],
                    },
                }
            },
        }

        with TemporaryDirectory() as directory:
            payload = client._request_payload(request, Path(directory))

        properties = payload["response_format"]["schema"]["properties"]
        self.assertIn("tool_calls", properties)
        self.assertNotIn("diary_entries", properties)
        self.assertNotIn("scheduled_messages", properties)
        self.assertNotIn("sticker_actions", properties)

    @staticmethod
    def _fake_winpty_module(process: SimpleNamespace) -> tuple[Any, MagicMock]:
        spawn = MagicMock(return_value=process)
        module = SimpleNamespace(PtyProcess=SimpleNamespace(spawn=spawn))
        return module, spawn

    @staticmethod
    def _alive_process() -> tuple[SimpleNamespace, dict[str, bool]]:
        state = {"alive": True}
        pty = SimpleNamespace(isalive=MagicMock(side_effect=lambda: state["alive"]))
        process = SimpleNamespace(
            pty=pty,
            fileobj=object(),
            exitstatus=0,
            read=MagicMock(),
            terminate=MagicMock(
                side_effect=lambda force=False: state.__setitem__("alive", False)
            ),
            close=MagicMock(),
        )
        return process, state

    def test_windows_pty_times_out_without_output_or_blocking_read(self) -> None:
        client = AgyModelClient(timeout_seconds=1)
        process, _ = self._alive_process()
        winpty_module, spawn = self._fake_winpty_module(process)

        with (
            TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"winpty": winpty_module}),
            patch("agy_provider.select.select", return_value=([], [], [])) as select_call,
            patch("agy_provider.time.monotonic", side_effect=[0.0, 0.0, 12.0]),
        ):
            workspace = Path(directory)
            with self.assertRaises(subprocess.TimeoutExpired):
                client._run_windows(["agy"], workspace)

        spawn.assert_called_once_with(["agy"], cwd=str(workspace), backend=0)
        select_call.assert_called_once()
        process.read.assert_not_called()
        process.terminate.assert_called_once_with(force=True)
        process.close.assert_called_once_with(force=True)

    def test_windows_pty_cancellation_terminates_process_without_reading(self) -> None:
        client = AgyModelClient(timeout_seconds=10)
        process, _ = self._alive_process()
        winpty_module, _ = self._fake_winpty_module(process)
        cancel_event = threading.Event()
        cancel_event.set()

        with (
            TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"winpty": winpty_module}),
            patch("agy_provider.select.select") as select_call,
            patch("agy_provider.time.monotonic", side_effect=[0.0, 0.0]),
        ):
            with self.assertRaisesRegex(AgyError, "отменён"):
                client._run_windows(["agy"], Path(directory), cancel_event)

        select_call.assert_not_called()
        process.read.assert_not_called()
        process.terminate.assert_called_once_with(force=True)
        process.close.assert_called_once_with(force=True)

    def test_windows_pty_nonzero_exit_uses_agy_log_diagnostic(self) -> None:
        client = AgyModelClient(timeout_seconds=10)
        pty = SimpleNamespace(isalive=MagicMock(return_value=False))
        process = SimpleNamespace(
            pty=pty,
            fileobj=object(),
            exitstatus=7,
            read=MagicMock(),
            terminate=MagicMock(),
            close=MagicMock(),
        )
        winpty_module, _ = self._fake_winpty_module(process)

        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "agy.log").write_text(
                "E0000 log.go:398] model output error: request.json not found\n",
                encoding="utf-8",
            )
            with (
                patch.dict(sys.modules, {"winpty": winpty_module}),
                patch("agy_provider.select.select", return_value=([], [], [])),
                patch(
                    "agy_provider.time.monotonic",
                    side_effect=[0.0, 0.0, 0.31],
                ),
            ):
                with self.assertRaisesRegex(
                    AgyError, "кодом 7: model output error: request.json not found"
                ):
                    client._run_windows(["agy"], workspace)

        process.read.assert_not_called()
        process.terminate.assert_not_called()
        process.close.assert_called_once_with(force=True)

    def test_windows_pty_returns_complete_json_before_process_exits(self) -> None:
        client = AgyModelClient(timeout_seconds=10)
        process, state = self._alive_process()
        process.exitstatus = None
        process.read.return_value = '{"messages":["быстро"],"reaction":null}'
        winpty_module, _ = self._fake_winpty_module(process)

        with (
            TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"winpty": winpty_module}),
            patch(
                "agy_provider.select.select",
                return_value=([process.fileobj], [], []),
            ) as select_call,
            patch("agy_provider.time.monotonic", side_effect=[0.0, 0.0]),
        ):
            output = client._run_windows(
                ["agy"],
                Path(directory),
                stop_on_structured_output=True,
            )

        self.assertEqual(output, '{"messages":["быстро"],"reaction":null}')
        self.assertFalse(state["alive"])
        select_call.assert_called_once()
        process.read.assert_called_once()
        process.terminate.assert_called_once_with(force=True)
        process.close.assert_called_once_with(force=True)


class AgyResponsesTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_delegates_blocking_query_to_thread(self) -> None:
        client = AgyModelClient()
        request = {"model": "ignored-by-adapter", "input": [{"role": "user"}]}
        client._query = MagicMock()  # type: ignore[method-assign]

        with patch(
            "agy_provider.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value="готовый ответ",
        ) as to_thread:
            response = await client.responses.create(**request)

        to_thread.assert_awaited_once()
        query, submitted_request, cancel_event = to_thread.await_args.args
        self.assertIs(query, client._query)
        self.assertEqual(submitted_request, request)
        self.assertIsInstance(cancel_event, threading.Event)
        self.assertFalse(cancel_event.is_set())
        client._query.assert_not_called()
        self.assertEqual(response.output_text, "готовый ответ")
        self.assertEqual(response.output, [])
        self.assertEqual(response.status, "completed")

    async def test_create_retries_auth_errors_with_incremental_delays(self) -> None:
        client = AgyModelClient(
            auth_retries=2,
            auth_retry_delay_seconds=0.25,
        )
        request = {"input": [{"role": "user", "content": "Привет"}]}

        with (
            patch(
                "agy_provider.asyncio.to_thread",
                new_callable=AsyncMock,
                side_effect=[
                    AgyAuthError("первая попытка"),
                    AgyAuthError("вторая попытка"),
                    "готовый ответ",
                ],
            ) as to_thread,
            patch(
                "agy_provider.asyncio.sleep", new_callable=AsyncMock
            ) as retry_sleep,
        ):
            response = await client.responses.create(**request)

        self.assertEqual(response.output_text, "готовый ответ")
        self.assertEqual(to_thread.await_count, 3)
        self.assertEqual(
            retry_sleep.await_args_list,
            [call(0.25), call(0.5)],
        )

    async def test_auth_retry_holds_lock_across_concurrent_requests(self) -> None:
        client = AgyModelClient(
            auth_retries=1,
            auth_retry_delay_seconds=0.25,
        )
        retry_sleep_started = asyncio.Event()
        allow_retry = asyncio.Event()
        attempts: dict[str, int] = {"first": 0, "second": 0}
        call_order: list[str] = []

        async def fake_create_once(request: dict[str, Any]) -> str:
            label = request["input"][0]["content"]
            attempts[label] += 1
            call_order.append(label)
            if label == "first" and attempts[label] == 1:
                raise AgyAuthError("нужен повтор")
            return f"answer-{label}"

        async def hold_retry_sleep(delay: float) -> None:
            self.assertEqual(delay, 0.25)
            retry_sleep_started.set()
            await allow_retry.wait()

        with (
            patch.object(
                client.responses,
                "_create_once",
                new_callable=AsyncMock,
                side_effect=fake_create_once,
            ),
            patch("agy_provider.asyncio.sleep", side_effect=hold_retry_sleep),
        ):
            first = asyncio.create_task(
                client.responses.create(
                    input=[{"role": "user", "content": "first"}]
                )
            )
            await asyncio.wait_for(retry_sleep_started.wait(), timeout=1)
            second = asyncio.create_task(
                client.responses.create(
                    input=[{"role": "user", "content": "second"}]
                )
            )

            done, _ = await asyncio.wait({second}, timeout=0.02)
            self.assertFalse(done)
            self.assertEqual(call_order, ["first"])

            allow_retry.set()
            first_result, second_result = await asyncio.gather(first, second)

        self.assertEqual(first_result, "answer-first")
        self.assertEqual(second_result, "answer-second")
        self.assertEqual(call_order, ["first", "first", "second"])
        self.assertEqual(attempts, {"first": 2, "second": 1})

    async def test_plain_request_preserves_fenced_json_as_plain_text(self) -> None:
        client = AgyModelClient()
        raw = '```json\n{"messages":["привет"],"reaction":null}\n```'
        client._query = MagicMock(return_value=raw)  # type: ignore[method-assign]

        response = await client.responses.create(input=[])

        self.assertEqual(response.output_text, raw)

    async def test_structured_request_normalizes_fenced_json(self) -> None:
        client = AgyModelClient()
        client._query = MagicMock(  # type: ignore[method-assign]
            return_value=(
                "```json\n"
                '{"messages":["привет","как дела?"],"reaction":"👍"}'
                "\n```"
            )
        )

        response = await client.responses.create(input=[], text={"format": {}})

        self.assertEqual(
            json.loads(response.output_text),
            {"messages": ["привет", "как дела?"], "reaction": "👍"},
        )

    async def test_structured_diary_entries_are_exposed_outside_output_text(self) -> None:
        client = AgyModelClient()
        client._query = MagicMock(  # type: ignore[method-assign]
            return_value=json.dumps(
                {
                    "messages": ["запомнила"],
                    "reaction": None,
                    "diary_entries": ["  Лена любит зелёный чай  ", "", "живёт в Перми"],
                },
                ensure_ascii=False,
            )
        )

        response = await client.responses.create(input=[], text={"format": {}})

        self.assertEqual(
            response.agy_diary_entries,
            ("Лена любит зелёный чай", "живёт в Перми"),
        )
        self.assertEqual(
            json.loads(response.output_text),
            {"messages": ["запомнила"], "reaction": None},
        )
        self.assertNotIn("diary_entries", response.output_text)

    async def test_structured_scheduled_messages_are_exposed_outside_output_text(self) -> None:
        client = AgyModelClient()
        client._query = MagicMock(  # type: ignore[method-assign]
            return_value=json.dumps(
                {
                    "messages": ["напишу через пять минут"],
                    "reaction": None,
                    "scheduled_messages": [
                        {"delay_seconds": 300, "message": "Пять минут прошло"}
                    ],
                },
                ensure_ascii=False,
            )
        )

        response = await client.responses.create(input=[], text={"format": {}})

        self.assertEqual(
            response.agy_scheduled_messages,
            ({"delay_seconds": 300, "message": "Пять минут прошло"},),
        )
        self.assertNotIn("scheduled_messages", response.output_text)

    async def test_structured_request_wraps_unstructured_answer(self) -> None:
        client = AgyModelClient()
        client._query = MagicMock(  # type: ignore[method-assign]
            return_value="обычный ответ без json"
        )

        response = await client.responses.create(input=[], text={"format": {}})

        self.assertEqual(
            json.loads(response.output_text),
            {"messages": ["обычный ответ без json"], "reaction": None},
        )

    async def test_structured_read_only_sentinel_becomes_empty_reply(self) -> None:
        client = AgyModelClient()
        client._query = MagicMock(  # type: ignore[method-assign]
            return_value="[[READ_ONLY]]"
        )

        response = await client.responses.create(input=[], text={"format": {}})

        self.assertEqual(
            json.loads(response.output_text),
            {"messages": [], "reaction": None},
        )

    async def test_empty_plain_answer_raises_agy_error(self) -> None:
        client = AgyModelClient()
        client._query = MagicMock(return_value=" \r\n ")  # type: ignore[method-assign]

        with self.assertRaisesRegex(AgyError, "пустой ответ"):
            await client.responses.create(input=[])

    async def test_event_loop_keeps_running_while_query_is_in_thread(self) -> None:
        import threading

        client = AgyModelClient()
        started = asyncio.Event()
        release = threading.Event()

        def blocking_query(request, cancel_event):
            del request
            del cancel_event
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=2)
            return "ответ"

        loop = asyncio.get_running_loop()
        client._query = blocking_query  # type: ignore[method-assign]
        task = asyncio.create_task(client.responses.create(input=[]))
        await asyncio.wait_for(started.wait(), timeout=1)

        # This coroutine can still make progress while the provider is blocked.
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release.set()
        response = await asyncio.wait_for(task, timeout=1)
        self.assertEqual(response.output_text, "ответ")

    async def test_cancelling_create_signals_blocking_query(self) -> None:
        client = AgyModelClient()
        started = asyncio.Event()
        cancellation_seen = threading.Event()
        loop = asyncio.get_running_loop()

        def blocking_query(request, cancel_event):
            del request
            loop.call_soon_threadsafe(started.set)
            if not cancel_event.wait(timeout=2):
                raise AssertionError("cancel_event не был установлен")
            cancellation_seen.set()
            raise AgyError("Запрос Gemini отменён")

        client._query = blocking_query  # type: ignore[method-assign]
        task = asyncio.create_task(client.responses.create(input=[]))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)

        self.assertTrue(cancellation_seen.is_set())


class AgyStickerActionTests(unittest.IsolatedAsyncioTestCase):
    async def test_milana_agent_tool_step_excludes_final_and_legacy_channels(self) -> None:
        client = AgyModelClient()
        client._query = MagicMock(  # type: ignore[method-assign]
            return_value=json.dumps(
                {
                    "state_update": {},
                    "entity_updates": [],
                    "life_events": [],
                    "goal_updates": [],
                    "relationship_updates": [],
                    "tool_calls": [
                        {
                            "name": "write_diary",
                            "arguments_json": json.dumps(
                                {"entry": "важная мысль"}, ensure_ascii=False
                            ),
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        response = await client.responses.create(
            input=[],
            tools=[],
            text={
                "format": {
                    "name": "milana_agent_turn",
                    "schema": {"type": "object", "properties": {}},
                }
            },
        )

        self.assertEqual(response.output_text, "")
        self.assertEqual(response.agy_tool_calls[0]["name"], "write_diary")
        self.assertEqual(response.agy_diary_entries, ())
        self.assertEqual(response.agy_scheduled_messages, ())
        self.assertEqual(response.agy_sticker_actions, ())

    async def test_generic_tool_calls_are_exposed_and_keep_legacy_bridge(self) -> None:
        client = AgyModelClient()
        client._query = MagicMock(  # type: ignore[method-assign]
            return_value=json.dumps(
                {
                    "messages": [],
                    "reaction": None,
                    "blacklist_sender": False,
                    "tool_calls": [
                        {
                            "name": "open_skill",
                            "arguments_json": json.dumps(
                                {"path": "telegram"}, ensure_ascii=False
                            ),
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        open_skill = {
            "type": "function",
            "name": "open_skill",
            "description": "Открыть навык",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        }
        response = await client.responses.create(
            input=[],
            tools=[open_skill],
            text={
                "format": {
                    "name": "milana_turn",
                    "schema": {
                        "type": "object",
                        "properties": {"messages": {"type": "array"}},
                        "required": ["messages"],
                    },
                }
            },
        )

        self.assertEqual(
            response.agy_tool_calls,
            (
                {
                    "name": "open_skill",
                    "arguments_json": '{"path": "telegram"}',
                },
            ),
        )
        self.assertNotIn("tool_calls", response.output_text)

    async def test_sticker_actions_are_extracted_from_structured_result(self) -> None:
        client = AgyModelClient()
        client._query = MagicMock(  # type: ignore[method-assign]
            return_value=json.dumps(
                {
                    "messages": [],
                    "reaction": None,
                    "blacklist_sender": False,
                    "sticker_actions": [
                        {
                            "name": "open_sticker_picker",
                            "pack_id": None,
                            "sticker_id": None,
                            "delay_seconds": None,
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )

        response = await client.responses.create(
            input=[],
            tools=[OPEN_STICKER_PICKER_TOOL, SEND_STICKER_TOOL],
            text={
                "format": {
                    "name": "milana_telegram_reply",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "messages": {"type": "array"},
                            "reaction": {},
                            "blacklist_sender": {"type": "boolean"},
                        },
                        "required": ["messages", "reaction", "blacklist_sender"],
                    },
                }
            },
        )

        self.assertEqual(
            response.agy_sticker_actions,
            ({"name": "open_sticker_picker", "arguments": {}},),
        )
        self.assertNotIn("sticker_actions", response.output_text)

    async def test_generic_initiative_json_is_not_wrapped_as_telegram_reply(self) -> None:
        client = AgyModelClient()
        payload = {
            "should_write": False,
            "contact_id": None,
            "message": None,
            "note": "не хочу писать",
            "sticker_actions": [],
        }
        client._query = MagicMock(return_value=json.dumps(payload, ensure_ascii=False))  # type: ignore[method-assign]

        response = await client.responses.create(
            input=[],
            tools=[OPEN_STICKER_PICKER_TOOL],
            text={
                "format": {
                    "name": "milana_initiative_decision",
                    "schema": {
                        "type": "object",
                        "properties": {"should_write": {"type": "boolean"}},
                        "required": ["should_write"],
                    },
                }
            },
        )

        self.assertEqual(json.loads(response.output_text)["should_write"], False)
        self.assertNotIn("messages", response.output_text)


if __name__ == "__main__":
    unittest.main()
