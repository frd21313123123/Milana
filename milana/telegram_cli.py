"""Command-line parsing and simple Telegram presentation helpers."""

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
        return text.replace("\r", " ").replace("\n", " ⏎ ")
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
