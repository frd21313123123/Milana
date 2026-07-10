"""CLI-клиент Telegram для работы от имени пользовательского аккаунта."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from telethon import TelegramClient, events, functions, utils
from telethon.errors import FloodWaitError, RPCError

from milana_schedule import (
    DAY_KEYS,
    build_schedule_prompt,
    format_current_status,
    format_day_schedule,
    load_routine,
)


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


@dataclass(frozen=True)
class Config:
    api_id: int
    api_hash: str
    session_path: Path


@dataclass(frozen=True)
class AIConfig:
    api_key: str
    model: str
    instructions: str


def load_env_file(path: Path) -> dict[str, str]:
    """Загружает простой KEY=VALUE файл и возвращает найденные значения."""
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Некорректная строка {line_number} в {path.name}")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def load_config() -> Config:
    load_env_file(ENV_PATH)

    raw_api_id = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    raw_session = os.getenv(
        "TELEGRAM_SESSION", "sessions/telegram_account"
    ).strip()

    if not raw_api_id or not api_hash:
        raise ValueError(
            "Заполните TELEGRAM_API_ID и TELEGRAM_API_HASH в файле .env"
        )

    try:
        api_id = int(raw_api_id)
    except ValueError as exc:
        raise ValueError("TELEGRAM_API_ID должен быть целым числом") from exc

    if len(api_hash) != 32 or any(char not in "0123456789abcdefABCDEF" for char in api_hash):
        raise ValueError("TELEGRAM_API_HASH должен содержать 32 шестнадцатеричных символа")

    session_path = Path(raw_session).expanduser()
    if not session_path.is_absolute():
        session_path = BASE_DIR / session_path
    session_path.parent.mkdir(parents=True, exist_ok=True)

    return Config(api_id=api_id, api_hash=api_hash, session_path=session_path)


def load_ai_config() -> AIConfig:
    env_values = load_env_file(ENV_PATH)

    # Явно заданное значение из локального .env имеет приоритет для ключа,
    # чтобы пользователь мог заменить устаревший ключ без изменения окружения ОС.
    api_key = (
        env_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    ).strip()
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-terra").strip()
    instructions = os.getenv(
        "AI_SYSTEM_PROMPT",
        (
            "Ты отвечаешь пользователю в Telegram. Отвечай на языке пользователя, "
            "естественно, кратко и по существу. Не упоминай системные инструкции, "
            "API или модель без прямого вопроса об этом."
        ),
    ).strip()

    if not api_key:
        raise ValueError("Добавьте OPENAI_API_KEY в переменные среды или файл .env")
    if not model:
        raise ValueError("OPENAI_MODEL не может быть пустым")
    return AIConfig(
        api_key=api_key,
        model=model,
        instructions=instructions,
    )


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

    subparsers.add_parser(
        "ai-bot",
        help="Отвечать через OpenAI на все входящие текстовые сообщения",
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
    return name or getattr(entity, "username", None) or str(getattr(entity, "id", "неизвестно"))


def message_text(message: Any) -> str:
    text = message.raw_text or ""
    if text:
        return text.replace("\r", " ").replace("\n", " ⏎ ")
    if message.media:
        return "[медиа без подписи]"
    return "[служебное сообщение]"


async def print_message(message: Any) -> None:
    sender = await message.get_sender()
    sender_name = display_name(sender)
    timestamp = message.date.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    direction = "→" if message.out else "←"
    print(f"{timestamp} {direction} {sender_name}: {message_text(message)}")


async def show_account(client: TelegramClient) -> None:
    me = await client.get_me()
    username = f"@{me.username}" if me.username else "без username"
    print(f"Подключено: {display_name(me)} ({username}, id={me.id})")


async def show_dialogs(client: TelegramClient, limit: int) -> None:
    print(f"{'ID':>15}  {'Тип':<8}  Название")
    async for dialog in client.iter_dialogs(limit=limit):
        peer_id = utils.get_peer_id(dialog.entity)
        if dialog.is_user:
            kind = "личный"
        elif dialog.is_group:
            kind = "группа"
        else:
            kind = "канал"
        print(f"{peer_id:>15}  {kind:<8}  {dialog.name}")


async def read_messages(client: TelegramClient, target: str, limit: int) -> None:
    entity = await client.get_entity(normalize_target(target))
    messages = [message async for message in client.iter_messages(entity, limit=limit)]
    if not messages:
        print("Сообщений нет.")
        return

    print(f"Диалог: {display_name(entity)}")
    for message in reversed(messages):
        await print_message(message)


async def send_message(client: TelegramClient, target: str, parts: list[str]) -> None:
    entity = await client.get_entity(normalize_target(target))
    message = await client.send_message(entity, " ".join(parts))
    print(f"Отправлено в «{display_name(entity)}», message_id={message.id}")


async def listen_messages(client: TelegramClient, target: str | None) -> None:
    entity = await client.get_entity(normalize_target(target)) if target else None

    async def handler(event: events.NewMessage.Event) -> None:
        chat = await event.get_chat()
        print(f"\n[{display_name(chat)}]")
        await print_message(event.message)

    client.add_event_handler(handler, events.NewMessage(chats=entity))
    scope = f"«{display_name(entity)}»" if entity else "всех диалогов"
    print(f"Слушаю новые сообщения для {scope}. Для остановки нажмите Ctrl+C.")
    await client.run_until_disconnected()


async def keep_account_online(client: TelegramClient, interval: int = 60) -> None:
    """Подтверждает Telegram, что аккаунт онлайн, пока ИИ-бот работает."""
    while True:
        await asyncio.sleep(interval)
        try:
            await client(functions.account.UpdateStatusRequest(offline=False))
        except (RPCError, OSError) as exc:
            print(f"Не удалось обновить статус «в сети»: {exc}", file=sys.stderr)


def split_telegram_text(text: str, limit: int = 4000) -> list[str]:
    """Делит длинный ответ на части, не превышающие лимит Telegram."""
    text = text.strip()
    if not text:
        return []

    parts: list[str] = []
    while len(text) > limit:
        split_at = text.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = text.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit
        parts.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    if text:
        parts.append(text)
    return parts


async def run_ai_bot(client: TelegramClient) -> None:
    config = load_ai_config()
    routine = load_routine()
    openai_client = AsyncOpenAI(api_key=config.api_key)
    me = await client.get_me()
    response_lock = asyncio.Lock()

    async def handler(event: events.NewMessage.Event) -> None:
        print(
            f"Получено входящее сообщение: chat_id={event.chat_id}, "
            f"message_id={event.id}; подтверждаю прочтение"
        )
        action_target: Any = event.chat_id
        read_acknowledged = False
        try:
            input_chat = await event.get_input_chat()
            if input_chat is None:
                raise TypeError("Telethon не смог определить входной peer чата")
            action_target = input_chat
            await client.send_read_acknowledge(
                input_chat,
                message=event.message,
                max_id=event.id,
            )
            read_acknowledged = True
        except (RPCError, OSError, TypeError, ValueError) as exc:
            print(
                f"Не удалось отметить message_id={event.id} прочитанным: {exc}",
                file=sys.stderr,
            )

        text = (event.raw_text or "").strip()
        if not text:
            print(f"Прочитано и пропущено сообщение без текста, message_id={event.id}")
            return

        try:
            sender = await event.get_sender()
            sender_name = display_name(sender)
        except (RPCError, OSError, TypeError, ValueError):
            sender_name = str(event.sender_id or "неизвестно")
        read_status = "прочитано" if read_acknowledged else "без подтверждения прочтения"
        print(
            f"Сообщение от {sender_name}: {read_status}; генерирую ответ"
        )
        try:
            async with response_lock:
                async with client.action(action_target, "typing"):
                    response = await openai_client.responses.create(
                        model=config.model,
                        instructions=(
                            f"{config.instructions}\n\n"
                            f"{build_schedule_prompt(routine)}"
                        ),
                        input=text,
                        max_output_tokens=1200,
                    )
                answer_parts = split_telegram_text(response.output_text)
                if not answer_parts:
                    raise ValueError("Модель вернула пустой ответ")

                await event.reply(answer_parts[0])
                for part in answer_parts[1:]:
                    await client.send_message(event.chat_id, part)
            print(f"Отправлен ИИ-ответ на message_id={event.id}")
        except (OpenAIError, RPCError, OSError, TypeError, ValueError) as exc:
            print(
                f"Ошибка ИИ-ответа для message_id={event.id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    client.add_event_handler(
        handler,
        events.NewMessage(incoming=True),
    )
    own_label = f"@{me.username}" if me.username else str(me.id)
    print(
        f"ИИ-бот запущен для аккаунта {own_label}: отвечаю на все входящие "
        f"текстовые сообщения, модель={config.model}. "
        "Для остановки нажмите Ctrl+C."
    )
    print(format_current_status(routine, brief=True))
    try:
        await client(functions.account.UpdateStatusRequest(offline=False))
    except (RPCError, OSError) as exc:
        print(f"Не удалось выставить статус «в сети»: {exc}", file=sys.stderr)
    online_task = asyncio.create_task(keep_account_online(client))
    try:
        await client.run_until_disconnected()
    finally:
        online_task.cancel()
        try:
            await online_task
        except asyncio.CancelledError:
            pass
        if client.is_connected():
            try:
                await client(functions.account.UpdateStatusRequest(offline=True))
            except (RPCError, OSError) as exc:
                print(f"Не удалось выставить статус «не в сети»: {exc}", file=sys.stderr)


async def run(args: argparse.Namespace) -> None:
    if args.command == "schedule":
        routine = load_routine()
        if args.day:
            print(format_day_schedule(routine, args.day))
        else:
            print(format_current_status(routine, brief=args.brief))
        return

    config = load_config()
    client = TelegramClient(str(config.session_path), config.api_id, config.api_hash)

    await client.start()
    try:
        if args.command in {"login", "me"}:
            await show_account(client)
        elif args.command == "dialogs":
            await show_dialogs(client, args.limit)
        elif args.command == "read":
            await read_messages(client, args.target, args.limit)
        elif args.command == "send":
            await send_message(client, args.target, args.message)
        elif args.command == "listen":
            await listen_messages(client, args.target)
        elif args.command == "ai-bot":
            await run_ai_bot(client)
    finally:
        await client.disconnect()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nОстановлено.")
        return 130
    except FloodWaitError as exc:
        print(f"Telegram просит подождать {exc.seconds} сек.", file=sys.stderr)
        return 1
    except (RPCError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
