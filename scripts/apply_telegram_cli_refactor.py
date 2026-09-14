"""One-shot safe refactor for extracting Telegram CLI helpers."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "telegram_client.py"
MODULE_PATH = ROOT / "milana" / "telegram_cli.py"

IMPORT_ANCHOR = "from milana_memory import (\n"
CLI_IMPORT = '''from milana.telegram_cli import (
    build_parser,
    display_name,
    message_text,
    normalize_target,
    positive_int,
)
'''

MODULE_SOURCE = '''"""Command-line parsing and simple Telegram presentation helpers."""

from __future__ import annotations

import argparse
from typing import Any

from telethon import utils

from milana_schedule import DAY_KEYS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Чтение и отправка сообщений через ваш Telegram-аккаунт"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("login", help="Войти в аккаунт или проверить сессию")
    subparsers.add_parser("me", help="Показать подключённый аккаунт")

    dialogs = subparsers.add_parser("dialogs", help="Показать список диалогов")
    dialogs.add_argument("--limit", type=positive_int, default=20)

    read = subparsers.add_parser("read", help="Прочитать последние сообщения")
    read.add_argument("target", help="@username, ID чата, номер телефона или me")
    read.add_argument("--limit", type=positive_int, default=20)

    send = subparsers.add_parser("send", help="Отправить текстовое сообщение")
    send.add_argument("target", help="@username, ID чата, номер телефона или me")
    send.add_argument("message", nargs="+", help="Текст сообщения")

    listen = subparsers.add_parser("listen", help="Показывать новые сообщения")
    listen.add_argument(
        "target",
        nargs="?",
        help="Необязательно: слушать только этот @username или ID чата",
    )

    ai_bot = subparsers.add_parser(
        "ai-bot",
        help="Отвечать через выбранную LLM на все входящие сообщения",
    )
    ai_bot.add_argument(
        "--dev-chat",
        action="store_true",
        help=(
            "Режим прямого общения: отвечать сразу, без расписания, "
            "симуляции присутствия и искусственных пауз"
        ),
    )

    schedule = subparsers.add_parser(
        "schedule",
        help="Показать текущее состояние расписания Миланы",
    )
    schedule.add_argument(
        "--brief",
        action="store_true",
        help="Вывести текущее состояние одной строкой",
    )
    schedule.add_argument(
        "--day",
        choices=DAY_KEYS,
        help="Показать расписание выбранного дня (mon–sun)",
    )

    return parser


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return number


def normalize_target(value: str) -> str | int:
    value = value.strip()
    if not value:
        raise ValueError("Адресат не может быть пустым")
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def display_name(entity: Any) -> str:
    if entity is None:
        return "неизвестно"
    name = utils.get_display_name(entity)
    return (
        name
        or getattr(entity, "username", None)
        or str(getattr(entity, "id", "неизвестно"))
    )


def message_text(message: Any) -> str:
    text = message.raw_text or ""
    if text:
        return text.replace("\\r", " ").replace("\\n", " ⏎ ")
    if message.media:
        return "[медиа без подписи]"
    return "[служебное сообщение]"


__all__ = [
    "build_parser",
    "display_name",
    "message_text",
    "normalize_target",
    "positive_int",
]
'''


def main() -> None:
    text = CLIENT_PATH.read_text(encoding="utf-8")
    original_lines = len(text.splitlines())

    if "from milana.telegram_cli import (" in text:
        raise SystemExit("telegram_client.py CLI helpers are already refactored")
    if IMPORT_ANCHOR not in text:
        raise RuntimeError("Telegram CLI import anchor not found")

    text = text.replace(IMPORT_ANCHOR, CLI_IMPORT + IMPORT_ANCHOR, 1)

    start_marker = "def build_parser() -> argparse.ArgumentParser:"
    end_marker = "def telegram_image_mime_type(event: Any) -> str | None:"
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    text = text[:start] + text[end:]

    for name in (
        "build_parser",
        "positive_int",
        "normalize_target",
        "display_name",
        "message_text",
    ):
        if name not in text:
            raise RuntimeError(f"compatibility export disappeared: {name}")

    new_lines = len(text.splitlines())
    removed = original_lines - new_lines
    if removed < 75:
        raise RuntimeError(f"CLI refactor removed only {removed} lines; expected >= 75")

    MODULE_PATH.write_text(MODULE_SOURCE, encoding="utf-8")
    CLIENT_PATH.write_text(text, encoding="utf-8")
    print(
        f"Extracted Telegram CLI helpers: telegram_client.py {original_lines} -> "
        f"{new_lines} lines; created {MODULE_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
