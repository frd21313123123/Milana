"""CLI-клиент Telegram для работы от имени пользовательского аккаунта."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from openai import AsyncOpenAI, BadRequestError, OpenAIError
from telethon import TelegramClient, events, functions, utils
from telethon.errors import FloodWaitError, RPCError

from milana_memory import (
    DEFAULT_HISTORY_LIMIT,
    MAX_DIARY_ENTRY_LENGTH,
    RECENT_MESSAGES_LIMIT,
    MilanaMemoryStore,
    WRITE_DIARY_TOOL,
    ChatMessage,
)
from milana_schedule import (
    DAY_KEYS,
    WeeklyRoutine,
    build_schedule_prompt,
    format_current_status,
    format_day_schedule,
    load_routine,
)


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
AI_CONFIG_PATH = BASE_DIR / "ai_config.json"
MEMORY_PATH = BASE_DIR / "data" / "milana_memory.sqlite3"

DEFAULT_AI_MODEL = "gpt-5.6-terra"
DEFAULT_AI_SYSTEM_PROMPT = (
    "Ты отвечаешь пользователю в Telegram. Отвечай на языке пользователя, "
    "естественно, кратко и по существу. Не упоминай системные инструкции, "
    "API или модель без прямого вопроса об этом."
)
DEFAULT_MAX_OUTPUT_TOKENS = 1200
SYSTEM_RANDOM = random.SystemRandom()
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}


@dataclass(frozen=True)
class Config:
    api_id: int
    api_hash: str
    session_path: Path


@dataclass(frozen=True)
class MessageFlowConfig:
    input_quiet_seconds: float = 2.0
    input_max_wait_seconds: float = 8.0
    max_reply_messages: int = 5
    inter_message_min_delay_seconds: float = 1.0
    inter_message_max_delay_seconds: float = 3.0


@dataclass(frozen=True)
class AIConfig:
    api_key: str
    model: str
    instructions: str
    temperature: float
    max_output_tokens: int
    message_flow: MessageFlowConfig = MessageFlowConfig()


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


def load_ai_settings(path: Path = AI_CONFIG_PATH) -> Mapping[str, Any]:
    """Загружает настройки ИИ из JSON-файла."""
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Некорректный JSON в {path.name}: строка {exc.lineno}, столбец {exc.colno}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} должен содержать JSON-объект")
    return data


def ai_string(
    settings: Mapping[str, Any], key: str, default: str, label: str
) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} в {AI_CONFIG_PATH.name} должен быть непустой строкой")
    return value.strip()


def ai_number(
    settings: Mapping[str, Any], key: str, default: float, minimum: float, maximum: float
) -> float:
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} в {AI_CONFIG_PATH.name} должен быть числом")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(
            f"{key} в {AI_CONFIG_PATH.name} должен быть от {minimum:g} до {maximum:g}"
        )
    return result


def ai_positive_int(settings: Mapping[str, Any], key: str, default: int) -> int:
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4000:
        raise ValueError(f"{key} в {AI_CONFIG_PATH.name} должен быть целым числом от 1 до 4000")
    return value


def ai_nonnegative_number(
    settings: Mapping[str, Any], key: str, default: float
) -> float:
    value = settings.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} в {AI_CONFIG_PATH.name} должен быть числом")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(
            f"{key} в {AI_CONFIG_PATH.name} должен быть неотрицательным числом"
        )
    return result


def load_message_flow_config(settings: Mapping[str, Any]) -> MessageFlowConfig:
    raw = settings.get("message_flow", {})
    if not isinstance(raw, dict):
        raise ValueError(f"message_flow в {AI_CONFIG_PATH.name} должен быть JSON-объектом")

    allowed = {
        "input_quiet_seconds",
        "input_max_wait_seconds",
        "max_reply_messages",
        "inter_message_min_delay_seconds",
        "inter_message_max_delay_seconds",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"Неизвестные параметры message_flow в {AI_CONFIG_PATH.name}: "
            + ", ".join(unknown)
        )

    input_quiet_seconds = ai_nonnegative_number(raw, "input_quiet_seconds", 2.0)
    input_max_wait_seconds = ai_nonnegative_number(
        raw, "input_max_wait_seconds", 8.0
    )
    if input_max_wait_seconds < input_quiet_seconds:
        raise ValueError(
            "input_max_wait_seconds в ai_config.json не может быть меньше "
            "input_quiet_seconds"
        )

    max_reply_messages = raw.get("max_reply_messages", 5)
    if (
        isinstance(max_reply_messages, bool)
        or not isinstance(max_reply_messages, int)
        or not 1 <= max_reply_messages <= 6
    ):
        raise ValueError(
            "max_reply_messages в ai_config.json должен быть целым числом от 1 до 6"
        )

    min_delay = ai_nonnegative_number(
        raw, "inter_message_min_delay_seconds", 1.0
    )
    max_delay = ai_nonnegative_number(
        raw, "inter_message_max_delay_seconds", 3.0
    )
    if min_delay > max_delay:
        raise ValueError(
            "inter_message_min_delay_seconds в ai_config.json не может быть больше "
            "inter_message_max_delay_seconds"
        )

    return MessageFlowConfig(
        input_quiet_seconds=input_quiet_seconds,
        input_max_wait_seconds=input_max_wait_seconds,
        max_reply_messages=max_reply_messages,
        inter_message_min_delay_seconds=min_delay,
        inter_message_max_delay_seconds=max_delay,
    )


def load_ai_config() -> AIConfig:
    env_values = load_env_file(ENV_PATH)
    settings = load_ai_settings()

    # Явно заданное значение из локального .env имеет приоритет для ключа,
    # чтобы пользователь мог заменить устаревший ключ без изменения окружения ОС.
    api_key = (
        env_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    ).strip()
    model = ai_string(
        settings,
        "model",
        os.getenv("OPENAI_MODEL", DEFAULT_AI_MODEL),
        "model",
    )
    instructions = ai_string(
        settings,
        "system_prompt",
        os.getenv("AI_SYSTEM_PROMPT", DEFAULT_AI_SYSTEM_PROMPT),
        "system_prompt",
    )
    temperature = ai_number(settings, "temperature", 0.7, 0, 2)
    max_output_tokens = ai_positive_int(
        settings, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS
    )
    message_flow = load_message_flow_config(settings)

    if not api_key:
        raise ValueError("Добавьте OPENAI_API_KEY в переменные среды или файл .env")
    if not model:
        raise ValueError("OPENAI_MODEL не может быть пустым")
    return AIConfig(
        api_key=api_key,
        model=model,
        instructions=instructions,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        message_flow=message_flow,
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


def telegram_image_mime_type(event: Any) -> str | None:
    """Возвращает MIME-тип поддерживаемого изображения из Telegram-события."""
    message = getattr(event, "message", None)
    photo = getattr(event, "photo", None) or getattr(message, "photo", None)
    file_info = getattr(event, "file", None) or getattr(message, "file", None)
    mime_type = getattr(file_info, "mime_type", None)

    # Обычные Telegram-фото всегда отдаются как JPEG, даже если у File нет MIME.
    if photo is not None:
        return mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else "image/jpeg"
    if mime_type in SUPPORTED_IMAGE_MIME_TYPES:
        return mime_type
    return None


async def telegram_image_data_url(event: Any, mime_type: str) -> str:
    """Скачивает Telegram-изображение в память и кодирует для Responses API."""
    message = getattr(event, "message", None)
    download_media = getattr(message, "download_media", None)
    if not callable(download_media):
        download_media = getattr(event, "download_media", None)
    if not callable(download_media):
        raise ValueError("Telegram не предоставил способ скачать изображение")

    image_bytes = await download_media(file=bytes)
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("Не удалось скачать изображение из Telegram")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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


class MilanaPresenceController:
    """Управляет короткими и правдоподобными окнами статуса «в сети»."""

    def __init__(
        self,
        client: TelegramClient,
        routine: WeeklyRoutine,
        *,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        randint: Callable[[int, int], int] = SYSTEM_RANDOM.randint,
    ) -> None:
        self.client = client
        self.routine = routine
        self._now = now
        self._sleep = sleep
        self._randint = randint
        self._online_until: datetime | None = None
        self._last_outgoing_at: datetime | None = None
        self._sleep_deferred_until: datetime | None = None
        self._next_spontaneous_online_at: datetime | None = None
        self._active_responses = 0
        self._last_offline: bool | None = None
        self._lock = asyncio.Lock()

    def current_time(self) -> datetime:
        value = self._now() if self._now is not None else None
        return self.routine.normalize_datetime(value)

    @property
    def online_until(self) -> datetime | None:
        return self._online_until

    @property
    def sleep_deferred_until(self) -> datetime | None:
        """Момент, после которого Милана может уснуть после переписки."""
        return self._sleep_deferred_until

    def is_sleep_deferred(self, value: datetime | None = None) -> bool:
        moment = self.routine.normalize_datetime(value) if value else self.current_time()
        return (
            self._last_outgoing_at is not None
            and self._sleep_deferred_until is not None
            and self._last_outgoing_at <= moment
            and moment < self._sleep_deferred_until
        )

    def can_respond(self, value: datetime | None = None) -> bool:
        """Разрешает продолжить активную переписку после планового отбоя."""
        moment = self.routine.normalize_datetime(value) if value else self.current_time()
        return (
            self.routine.response_policy_at(moment).available
            or self.is_sleep_deferred(moment)
        )

    def is_online(self, value: datetime | None = None) -> bool:
        moment = self.routine.normalize_datetime(value) if value else self.current_time()
        if self._active_responses > 0:
            return True
        if not self.can_respond(moment):
            return False
        grace_is_active = (
            self._online_until is not None and moment < self._online_until
        )
        return grace_is_active

    async def _publish_locked(self) -> None:
        offline = not self.is_online()
        if offline == self._last_offline:
            return
        try:
            await self.client(functions.account.UpdateStatusRequest(offline=offline))
            self._last_offline = offline
        except (RPCError, OSError) as exc:
            label = "не в сети" if offline else "в сети"
            print(
                f"Не удалось обновить статус «{label}»: {exc}",
                file=sys.stderr,
            )

    async def begin_response(self) -> None:
        """Сразу показывает online перед чтением и отправкой ответа."""
        async with self._lock:
            self._active_responses += 1
            await self._publish_locked()

    async def finish_response(self, *, answered: bool) -> int | None:
        """После ответа оставляет аккаунт online на случайные 30–60 секунд."""
        async with self._lock:
            self._active_responses = max(0, self._active_responses - 1)
            online_seconds: int | None = None
            if answered:
                behavior = self.routine.online_behavior
                answered_at = self.current_time()
                self._last_outgoing_at = answered_at
                sleep_candidate = answered_at + timedelta(
                    seconds=behavior.conversation_sleep_delay_seconds
                )
                if (
                    self._sleep_deferred_until is None
                    or sleep_candidate > self._sleep_deferred_until
                ):
                    self._sleep_deferred_until = sleep_candidate
                online_seconds = self._randint(
                    behavior.post_reply_online_min_seconds,
                    behavior.post_reply_online_max_seconds,
                )
                candidate = self.current_time() + timedelta(seconds=online_seconds)
                if self._online_until is None or candidate > self._online_until:
                    self._online_until = candidate
            await self._publish_locked()
            return online_seconds

    def _schedule_spontaneous_online_locked(self, from_time: datetime) -> None:
        behavior = self.routine.online_behavior
        delay = self._randint(
            behavior.spontaneous_online_interval_min_seconds,
            behavior.spontaneous_online_interval_max_seconds,
        )
        self._next_spontaneous_online_at = from_time + timedelta(seconds=delay)

    async def refresh(self) -> int | None:
        """Обновляет статус и при необходимости начинает фоновое online-окно."""
        async with self._lock:
            now = self.current_time()
            if self._next_spontaneous_online_at is None:
                self._schedule_spontaneous_online_locked(now)

            online_seconds: int | None = None
            if (
                self._next_spontaneous_online_at is not None
                and now >= self._next_spontaneous_online_at
                and self.routine.response_policy_at(now).available
            ):
                behavior = self.routine.online_behavior
                online_seconds = self._randint(
                    behavior.spontaneous_online_duration_min_seconds,
                    behavior.spontaneous_online_duration_max_seconds,
                )
                candidate = now + timedelta(seconds=online_seconds)
                state = self.routine.state_at(now)
                if (
                    state.next_at is not None
                    and state.next_activity is not None
                    and state.next_activity.kind == "sleep"
                ):
                    # Фоновый online должен закончиться заранее, чтобы статус
                    # «в сети» никогда не появлялся там, где уже нельзя успеть
                    # ответить и затем сохранить post-reply окно.
                    safe_until = state.next_at - timedelta(
                        seconds=behavior.sleep_buffer_seconds
                    )
                    candidate = min(candidate, safe_until)
                online_seconds = max(0, int((candidate - now).total_seconds()))
                if self._online_until is None or candidate > self._online_until:
                    if candidate > now:
                        self._online_until = candidate
                if online_seconds == 0:
                    online_seconds = None
                self._schedule_spontaneous_online_locked(candidate)

            await self._publish_locked()
            return online_seconds

    async def run(self, interval: float = 1.0) -> None:
        """Управляет ответами и случайными короткими появлениями в сети."""
        while True:
            online_seconds = await self.refresh()
            if online_seconds is not None:
                print(
                    "Милана ненадолго зашла в сеть; выйдет примерно через "
                    f"{online_seconds // 60} мин."
                )
            await self._sleep(interval)

    async def force_offline(self) -> None:
        async with self._lock:
            self._active_responses = 0
            self._online_until = None
            self._last_outgoing_at = None
            self._sleep_deferred_until = None
            self._next_spontaneous_online_at = None
            try:
                await self.client(functions.account.UpdateStatusRequest(offline=True))
                self._last_offline = True
            except (RPCError, OSError) as exc:
                print(f"Не удалось выставить статус «не в сети»: {exc}", file=sys.stderr)


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


@dataclass(frozen=True)
class IncomingEnvelope:
    event: Any
    received_at: datetime
    queued_at: datetime
    received_while_online: bool
    continues_conversation: bool


@dataclass(frozen=True)
class PreparedIncoming:
    event: Any
    received_at: datetime
    sender_name: str
    text: str
    image_data_url: str | None


@dataclass(frozen=True)
class GeneratedReply:
    messages: tuple[str, ...]
    staged_diary_entries: tuple[str, ...] = ()


@dataclass
class ChatWorkerState:
    chat_key: int | str
    pending: list[IncomingEnvelope] = field(default_factory=list)
    seen_message_ids: set[int] = field(default_factory=set)
    revision: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    worker: asyncio.Task[None] | None = None


@dataclass(frozen=True)
class SendOutcome:
    sent_count: int
    interrupted: bool


class MilanaMessageResponder:
    """Объединяет входящие по чатам и отвечает в ритме текущего занятия Миланы."""

    def __init__(
        self,
        client: TelegramClient,
        openai_client: AsyncOpenAI,
        config: AIConfig,
        routine: WeeklyRoutine,
        *,
        memory: MilanaMemoryStore | None = None,
        presence: MilanaPresenceController | None = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        randint: Callable[[int, int], int] = SYSTEM_RANDOM.randint,
    ) -> None:
        self.client = client
        self.openai_client = openai_client
        self.config = config
        self.routine = routine
        self.memory = memory or MilanaMemoryStore()
        self.history_limit = history_limit
        self._now = now
        self._sleep = sleep
        self._randint = randint
        self.presence = presence or MilanaPresenceController(
            client,
            routine,
            now=now,
            sleep=sleep,
            randint=randint,
        )
        self._chat_states: dict[int | str, ChatWorkerState] = {}
        self._chat_states_lock = asyncio.Lock()
        self._closing = False
        self._supports_temperature: bool | None = None
        self._supports_structured_reply: bool | None = None

    async def _import_existing_history(
        self, event: events.NewMessage.Event, chat_key: int | str
    ) -> None:
        """Import the initial Telegram tail and fill gaps after bot downtime."""
        latest_id = self.memory.latest_telegram_message_id(chat_key)
        if latest_id is not None and event.id <= latest_id + 1:
            return

        try:
            query: dict[str, Any] = {
                "limit": self.history_limit,
                "max_id": event.id,
            }
            if latest_id is not None:
                query["min_id"] = latest_id
            messages = [
                message
                async for message in self.client.iter_messages(
                    event.chat_id,
                    **query,
                )
            ]
        except (RPCError, OSError, TypeError, ValueError, AttributeError) as exc:
            print(f"Не удалось импортировать историю chat_id={event.chat_id}: {exc}", file=sys.stderr)
            return

        for message in reversed(messages):
            text = (getattr(message, "raw_text", None) or "").strip()
            if not text:
                continue
            outgoing = bool(getattr(message, "out", False))
            sender_name = "Милана" if outgoing else None
            if not outgoing:
                try:
                    sender = await message.get_sender()
                    sender_name = display_name(sender)
                except (RPCError, OSError, TypeError, ValueError, AttributeError):
                    sender_name = str(getattr(message, "sender_id", None) or "неизвестно")
            created = getattr(message, "date", None)
            self.memory.add_message(
                chat_key,
                "assistant" if outgoing else "user",
                text,
                telegram_message_id=getattr(message, "id", None),
                sender_name=sender_name,
                created_at=created.isoformat() if isinstance(created, datetime) else None,
            )

        if messages:
            print(
                f"Импортирована история chat_id={event.chat_id}: "
                f"до {min(len(messages), self.history_limit)} сообщений"
            )

    def _response_request(
        self,
        *,
        instructions: str,
        input_items: list[Any],
        structured_reply: bool = True,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.config.model,
            "instructions": instructions,
            "input": input_items,
            "tools": [WRITE_DIARY_TOOL],
            "tool_choice": "auto",
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self._supports_temperature is not False:
            request["temperature"] = self.config.temperature
        if structured_reply and self._supports_structured_reply is not False:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "milana_telegram_reply",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": self.config.message_flow.max_reply_messages,
                            }
                        },
                        "required": ["messages"],
                        "additionalProperties": False,
                    },
                }
            }
        return request

    @staticmethod
    def _temperature_is_unsupported(exc: BadRequestError) -> bool:
        body = getattr(exc, "body", None)
        parameter = body.get("param") if isinstance(body, dict) else None
        message = str(exc).lower()
        return parameter == "temperature" or (
            "temperature" in message and "unsupported parameter" in message
        )

    @staticmethod
    def _structured_reply_is_unsupported(exc: BadRequestError) -> bool:
        body = getattr(exc, "body", None)
        parameter = (
            str(body.get("param", "")).lower() if isinstance(body, dict) else ""
        )
        message = str(exc).lower()
        mentions_format = parameter in {"text", "text.format", "response_format"} or (
            "json_schema" in message
            or "structured output" in message
            or "text.format" in message
        )
        return mentions_format and any(
            marker in message
            for marker in (
                "unsupported",
                "not support",
                "unknown parameter",
                "unrecognized parameter",
            )
        )

    @staticmethod
    def _plain_reply_instructions(instructions: str) -> str:
        return (
            f"{instructions}\n\n"
            "Structured Outputs для этой модели недоступны. Верни только один готовый "
            "текст Telegram-сообщения без JSON, массива messages, Markdown-блока кода "
            "и служебных пояснений."
        )

    async def _create_model_response(
        self, *, instructions: str, input_items: list[Any]
    ) -> tuple[Any, bool]:
        request = self._response_request(
            instructions=instructions,
            input_items=input_items,
        )
        if "text" not in request:
            request["instructions"] = self._plain_reply_instructions(instructions)
        while True:
            try:
                response = await self.openai_client.responses.create(**request)
                if "temperature" in request:
                    self._supports_temperature = True
                if "text" in request:
                    self._supports_structured_reply = True
                return response, "text" in request
            except BadRequestError as exc:
                if "temperature" in request and self._temperature_is_unsupported(exc):
                    self._supports_temperature = False
                    request.pop("temperature")
                    print(
                        f"Модель {self.config.model} не поддерживает temperature; "
                        "повторяю запрос без этого параметра"
                    )
                    continue
                if "text" in request and self._structured_reply_is_unsupported(exc):
                    self._supports_structured_reply = False
                    request.pop("text")
                    request["instructions"] = self._plain_reply_instructions(
                        instructions
                    )
                    print(
                        f"Модель {self.config.model} не поддерживает Structured Outputs; "
                        "повторяю запрос с одним обычным текстовым ответом"
                    )
                    continue
                raise

    def _execute_diary_call(
        self,
        call: Any,
        *,
        chat_key: int | str,
        source_message_id: int | None,
    ) -> str:
        """Validate and execute one model-requested diary write."""
        try:
            arguments = json.loads(call.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("arguments должен быть объектом")
            content = arguments.get("content")
            stored = self.memory.add_diary_entry(
                content,
                source_chat_id=chat_key,
                source_message_id=source_message_id,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return json.dumps(
                {"status": "error", "message": str(exc)}, ensure_ascii=False
            )
        return json.dumps(
            {"status": "stored" if stored else "already_exists"},
            ensure_ascii=False,
        )

    @staticmethod
    def _staged_diary_call(call: Any, staged: list[str]) -> str:
        """Validate a diary call without mutating memory until the reply is committed."""
        try:
            arguments = json.loads(call.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("arguments должен быть объектом")
            content = arguments.get("content")
            if not isinstance(content, str):
                raise TypeError("Запись дневника должна быть строкой")
            content = content.strip()
            if not content:
                raise ValueError("Запись дневника не может быть пустой")
            if len(content) > MAX_DIARY_ENTRY_LENGTH:
                raise ValueError(
                    f"Запись дневника не может быть длиннее {MAX_DIARY_ENTRY_LENGTH} символов"
                )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return json.dumps(
                {"status": "error", "message": str(exc)}, ensure_ascii=False
            )

        if content not in staged:
            staged.append(content)
            status = "accepted"
        else:
            status = "already_accepted"
        return json.dumps({"status": status}, ensure_ascii=False)

    def _commit_staged_diary(
        self,
        entries: tuple[str, ...],
        *,
        chat_key: int | str,
        source_message_id: int | None,
    ) -> None:
        for content in entries:
            self.memory.add_diary_entry(
                content,
                source_chat_id=chat_key,
                source_message_id=source_message_id,
            )

    # --- Summarization for long-term per-chat context + dynamic user window ---

    async def _generate_summary(
        self, *, current_summary: str, new_messages: list[ChatMessage]
    ) -> str:
        """Call the model to produce or update a concise chat summary (incremental)."""
        if not new_messages:
            return current_summary.strip()

        lines: list[str] = []
        for m in new_messages:
            who = m.sender_name or ("Милана" if m.role == "assistant" else "Собеседник")
            lines.append(f"{who}: {m.content}")
        transcript = "\n".join(lines)

        instructions = (
            "Ты — модель сжатия истории диалога. Создай или обнови КРАТКИЙ пересказ "
            "основных моментов разговора между собеседником и Миланой. "
            "Выдели только существенное:\n"
            "- имя и ключевые факты о собеседнике (предпочтения, важные детали жизни)\n"
            "- темы обсуждений и устойчивые факты\n"
            "- важные события, договорённости, обещания, решения\n"
            "- текущее положение дел в чате (если релевантно)\n\n"
            "Будь очень краток (5–15 пунктов или короткий связный текст). "
            "Отвечай на языке пользователя (в основном русский). "
            "Не выдумывай. Не используй дословные длинные цитаты. "
            "Если есть предыдущий обзор — интегрируй в него новую информацию, сохраняя лаконичность.\n"
            "ВЫВОДИ ТОЛЬКО сам пересказ, без вступлений и пояснений."
        )

        prev = f"Предыдущий обзор:\n{current_summary}\n\n" if current_summary.strip() else ""
        input_items: list[Any] = [
            {
                "role": "user",
                "content": (
                    prev
                    + "Новый фрагмент диалога, который нужно учесть в обзоре:\n\n"
                    + transcript
                    + "\n\nОбновлённый краткий обзор основных моментов:"
                ),
            }
        ]

        # Use a direct call (no diary tools). Reuse temperature-fallback logic.
        max_tokens = min(700, self.config.max_output_tokens)
        request: dict[str, Any] = {
            "model": self.config.model,
            "instructions": instructions,
            "input": input_items,
            "max_output_tokens": max_tokens,
        }
        if self._supports_temperature is not False:
            request["temperature"] = 0.25

        try:
            response = await self.openai_client.responses.create(**request)
            if "temperature" in request:
                self._supports_temperature = True
            text = str(getattr(response, "output_text", "") or "").strip()
            return text or current_summary.strip()
        except BadRequestError as exc:
            if "temperature" not in request or not self._temperature_is_unsupported(exc):
                print(f"Ошибка summarizer: {exc}", file=sys.stderr)
                return current_summary.strip()
            self._supports_temperature = False
            request.pop("temperature", None)
            try:
                response = await self.openai_client.responses.create(**request)
                text = str(getattr(response, "output_text", "") or "").strip()
                return text or current_summary.strip()
            except Exception as inner:  # noqa: BLE001
                print(f"Ошибка summarizer (повтор без temperature): {inner}", file=sys.stderr)
                return current_summary.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка summarizer: {exc}", file=sys.stderr)
            return current_summary.strip()

    async def _maybe_update_chat_summary(self, chat_key: int | str) -> None:
        """If the dynamic user-message window reached 60, summarize older part (except last ~30)."""
        try:
            total_users = self.memory.count_user_messages(chat_key)
            info = self.memory.get_chat_summary_info(chat_key)
            covered = info.covered_user_messages if info else 0
            last_covered_id = info.last_covered_message_id if info else 0
            current_summary = info.summary if info else ""

            if total_users - covered < 60:
                return

            user_cutoff = self.memory.get_nth_last_user_message_id(chat_key, 30)
            total_cutoff = self.memory.get_nth_last_message_id(chat_key, 30)
            cutoff = None
            if user_cutoff is not None and total_cutoff is not None:
                cutoff = min(user_cutoff, total_cutoff)
            elif user_cutoff is not None:
                cutoff = user_cutoff
            elif total_cutoff is not None:
                cutoff = total_cutoff

            if cutoff is None or cutoff <= last_covered_id:
                # Nothing new to cover; just advance the covered count
                if total_users - 30 > covered:
                    self.memory.set_chat_summary(
                        chat_key,
                        current_summary or "Диалог начат.",
                        covered_user_messages=total_users - 30,
                        last_covered_message_id=cutoff or last_covered_id,
                    )
                return

            batch = self.memory.get_messages_in_id_range(
                chat_key, last_covered_id + 1, cutoff
            )
            if not batch:
                return

            new_summary = await self._generate_summary(
                current_summary=current_summary, new_messages=batch
            )
            if new_summary:
                self.memory.set_chat_summary(
                    chat_key,
                    new_summary,
                    covered_user_messages=total_users - 30,
                    last_covered_message_id=cutoff,
                )
                print(f"Обновлён обзор чата chat_id={chat_key} (покрыто пользователей: {total_users - 30})")
        except Exception as exc:  # noqa: BLE001
            # Never break the main flow because of summarization
            print(f"Не удалось обновить обзор чата {chat_key}: {exc}", file=sys.stderr)

    @staticmethod
    def _response_refusal(response: Any) -> str | None:
        for output in list(getattr(response, "output", None) or []):
            if getattr(output, "type", None) != "message":
                continue
            for item in list(getattr(output, "content", None) or []):
                if getattr(item, "type", None) == "refusal":
                    refusal = str(getattr(item, "refusal", "") or "").strip()
                    if refusal:
                        return refusal
        return None

    @staticmethod
    def _raise_if_incomplete(response: Any) -> None:
        if getattr(response, "status", None) != "incomplete":
            return
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None) or "unknown"
        raise ValueError(f"Модель вернула незавершённый ответ: {reason}")

    def _parse_generated_reply(
        self,
        response: Any,
        *,
        structured: bool,
        staged_diary_entries: list[str],
    ) -> GeneratedReply:
        self._raise_if_incomplete(response)

        refusal = self._response_refusal(response)
        if refusal:
            return GeneratedReply((refusal,), tuple(staged_diary_entries))

        output_text = str(getattr(response, "output_text", "") or "").strip()
        if not structured:
            if not output_text:
                raise ValueError("Модель вернула пустой ответ")
            return GeneratedReply((output_text,), tuple(staged_diary_entries))

        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Модель вернула некорректный структурированный ответ") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            raise ValueError("Структурированный ответ не содержит массив messages")

        raw_messages = payload["messages"]
        if any(not isinstance(message, str) for message in raw_messages):
            raise ValueError("Каждая часть структурированного ответа должна быть строкой")
        messages = tuple(message.strip() for message in raw_messages if message.strip())
        if len(messages) > self.config.message_flow.max_reply_messages:
            raise ValueError("Модель превысила максимальное число сообщений в ответе")
        if not messages:
            raise ValueError("Модель вернула пустой ответ")
        return GeneratedReply(messages, tuple(staged_diary_entries))

    async def _generate_answer(
        self,
        *,
        chat_key: int | str,
        history_input: list[dict[str, str]],
        messages: list[PreparedIncoming],
    ) -> GeneratedReply:
        max_parts = self.config.message_flow.max_reply_messages
        instructions = (
            f"{self.config.instructions}\n\n"
            f"{build_schedule_prompt(self.routine, self.current_time())}\n\n"
            f"{self.memory.diary_instructions()}\n\n"
            "Сформируй готовый ответ для Telegram как от Миланы. Самостоятельно реши, "
            f"нужна одна реплика или естественная серия до {max_parts} реплик. "
            "Не дроби цельную мысль искусственно, но можешь отделить короткое приветствие, "
            "реакцию или продолжение так, как люди пишут в живом чате. Каждая строка массива "
            "messages будет отправлена отдельным сообщением; не добавляй служебные пояснения."
        )
        input_items: list[Any] = [*history_input]
        for message in messages:
            current_text = f"{message.sender_name}: {message.text}"
            if message.image_data_url is None:
                input_items.append({"role": "user", "content": current_text})
            else:
                input_items.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": current_text},
                            {"type": "input_image", "image_url": message.image_data_url},
                        ],
                    }
                )

        staged_diary_entries: list[str] = []
        for _ in range(4):
            response, structured = await self._create_model_response(
                instructions=instructions,
                input_items=input_items,
            )
            self._raise_if_incomplete(response)
            output = list(getattr(response, "output", None) or [])
            calls = [
                item
                for item in output
                if getattr(item, "type", None) == "function_call"
                and getattr(item, "name", None) == "write_diary"
            ]
            if not calls:
                return self._parse_generated_reply(
                    response,
                    structured=structured,
                    staged_diary_entries=staged_diary_entries,
                )

            # The Responses API expects the model output followed by one result
            # for every function call on the next request.
            input_items.extend(output)
            for call in calls:
                result = self._staged_diary_call(call, staged_diary_entries)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result,
                    }
                )

        raise ValueError("Модель превысила лимит последовательных записей в дневник")

    def current_time(self) -> datetime:
        value = self._now() if self._now is not None else None
        return self.routine.normalize_datetime(value)

    def received_time(self, event: events.NewMessage.Event) -> datetime:
        value = getattr(event.message, "date", None)
        if isinstance(value, datetime):
            return self.routine.normalize_datetime(value)
        return self.current_time()

    async def _sleep_until(self, target: datetime) -> None:
        delay = (target - self.current_time()).total_seconds()
        if delay > 0:
            await self._sleep(delay)

    async def _wait_before_reading(
        self,
        received_at: datetime,
        *,
        received_while_online: bool,
        continues_conversation: bool,
    ) -> None:
        if received_while_online or continues_conversation:
            behavior = self.routine.online_behavior
            fast_delay = self._randint(
                behavior.online_response_min_seconds,
                behavior.online_response_max_seconds,
            )
            fast_target = received_at + timedelta(seconds=fast_delay)
            print(
                f"Сообщение получено, пока Милана в сети; "
                f"чтение запланировано на {fast_target:%d.%m %H:%M:%S} "
                f"(через {fast_delay} сек.)"
            )
            now = self.current_time()
            if continues_conversation or self.presence.can_respond(now):
                await self._sleep_until(fast_target)
                now = self.current_time()
                if continues_conversation or self.presence.can_respond(now):
                    return
                # Если после короткой задержки окно закрылось (редко),
                # планируем чтение заново от текущего момента.
                received_at = now

        plan = self.routine.plan_response(received_at, randint=self._randint)
        while True:
            now = self.current_time()
            delay = max(0.0, (plan.respond_at - now).total_seconds())
            print(
                f"Чтение запланировано на {plan.respond_at:%d.%m %H:%M:%S} "
                f"(через {math.ceil(delay)} сек.; {plan.policy.label})"
            )
            await self._sleep_until(plan.respond_at)
            now = self.current_time()
            if self.presence.can_respond(now):
                return None

            # Системные часы или расписание могли измениться во время ожидания.
            # Во сне по-прежнему ничего не читаем и строим новый план от «сейчас».
            plan = self.routine.plan_response(now, randint=self._randint)

    async def _wait_out_sleep(self, *, continues_conversation: bool = False) -> None:
        while True:
            now = self.current_time()
            if continues_conversation or self.presence.can_respond(now):
                return
            plan = self.routine.plan_response(now, randint=self._randint)
            print(
                "Ответ готов, но Милана спит; отправка перенесена на "
                f"{plan.respond_at:%d.%m %H:%M:%S}"
            )
            await self._sleep_until(plan.respond_at)

    async def _wait_for_full_online_window(
        self, *, continues_conversation: bool = False
    ) -> None:
        """Не начинает ответ, если до сна осталось меньше минуты."""
        while True:
            await self._wait_out_sleep(
                continues_conversation=continues_conversation
            )
            if continues_conversation:
                return
            now = self.current_time()
            state = self.routine.state_at(now)
            seconds_to_next = (
                (state.next_at - now).total_seconds()
                if state.next_at is not None
                else None
            )
            next_is_sleep = (
                state.next_activity is not None
                and state.next_activity.kind == "sleep"
            )
            if (
                not next_is_sleep
                or seconds_to_next is None
                or seconds_to_next >= self.routine.online_behavior.sleep_buffer_seconds
            ):
                return
            plan = self.routine.plan_response(state.next_at, randint=self._randint)
            print(
                "До сна осталось меньше минуты; ответ перенесён на "
                f"{plan.respond_at:%d.%m %H:%M:%S}"
            )
            await self._sleep_until(plan.respond_at)

    @staticmethod
    def _chat_key(event: Any) -> int | str:
        chat_key: int | str | None = getattr(event, "chat_id", None)
        if chat_key is None:
            chat_key = str(getattr(event, "sender_id", None) or "unknown")
        return chat_key

    @staticmethod
    def _envelope_sort_key(envelope: IncomingEnvelope) -> tuple[datetime, int]:
        message_id = getattr(envelope.event, "id", 0)
        return envelope.received_at, message_id if isinstance(message_id, int) else 0

    @staticmethod
    def _report_worker_error(done: asyncio.Task[None]) -> None:
        if done.cancelled():
            return
        error = done.exception()
        if error is not None:
            print(f"Необработанная ошибка worker чата: {error}", file=sys.stderr)

    async def submit(self, event: events.NewMessage.Event) -> asyncio.Task[None]:
        """Quickly enqueue an event and ensure exactly one worker owns its chat."""
        received_at = self.received_time(event)
        envelope = IncomingEnvelope(
            event=event,
            received_at=received_at,
            queued_at=self.current_time(),
            received_while_online=self.presence.is_online(received_at),
            continues_conversation=self.presence.is_sleep_deferred(received_at),
        )
        chat_key = self._chat_key(event)
        message_id = getattr(event, "id", None)

        async with self._chat_states_lock:
            if self._closing:
                raise RuntimeError("ИИ-ответчик уже останавливается")
            state = self._chat_states.get(chat_key)
            if state is None:
                state = ChatWorkerState(chat_key=chat_key)
                self._chat_states[chat_key] = state

            if isinstance(message_id, int) and message_id in state.seen_message_ids:
                if state.worker is None:
                    state.worker = asyncio.create_task(
                        self._chat_worker(state),
                        name=f"milana-chat-{chat_key}",
                    )
                    state.worker.add_done_callback(self._report_worker_error)
                return state.worker

            if isinstance(message_id, int):
                state.seen_message_ids.add(message_id)
            state.pending.append(envelope)
            state.revision += 1
            state.changed.set()
            if state.worker is None or state.worker.done():
                state.worker = asyncio.create_task(
                    self._chat_worker(state),
                    name=f"milana-chat-{chat_key}",
                )
                state.worker.add_done_callback(self._report_worker_error)
            worker = state.worker

        print(
            f"Получено входящее сообщение: chat_id={getattr(event, 'chat_id', None)}, "
            f"message_id={message_id}; добавлено в очередь чата"
        )
        return worker

    async def process(self, event: events.NewMessage.Event) -> None:
        """Compatibility helper: enqueue one event and wait until its chat is idle."""
        worker = await self.submit(event)
        await asyncio.shield(worker)

    async def shutdown(self) -> None:
        """Stop all chat workers and wait until their cancellation is complete."""
        async with self._chat_states_lock:
            self._closing = True
            workers = [
                state.worker
                for state in self._chat_states.values()
                if state.worker is not None and not state.worker.done()
            ]
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        async with self._chat_states_lock:
            self._chat_states.clear()

    async def _first_pending(self, state: ChatWorkerState) -> IncomingEnvelope | None:
        async with self._chat_states_lock:
            if not state.pending:
                return None
            return min(state.pending, key=self._envelope_sort_key)

    async def _revision(self, state: ChatWorkerState) -> int:
        async with self._chat_states_lock:
            return state.revision

    async def _generation_revision(self, state: ChatWorkerState) -> int | None:
        """Capture a revision only when every queued event is already in the batch."""
        async with self._chat_states_lock:
            if state.pending:
                return None
            state.changed.clear()
            return state.revision

    async def _sleep_or_changed(
        self, state: ChatWorkerState, seconds: float
    ) -> bool:
        if seconds <= 0:
            return state.changed.is_set()
        sleep_task = asyncio.create_task(self._sleep(seconds))
        changed_task = asyncio.create_task(state.changed.wait())
        try:
            done, _ = await asyncio.wait(
                {sleep_task, changed_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            changed_won = changed_task in done
        finally:
            for task in (sleep_task, changed_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sleep_task, changed_task, return_exceptions=True)
        # An event can arrive while the losing task is being cancelled above.
        # Check the signal after cleanup so that arrival starts a fresh quiet wait.
        return changed_won or state.changed.is_set()

    async def _wait_for_input_quiet(
        self, state: ChatWorkerState
    ) -> list[IncomingEnvelope]:
        """Wait for quiet (within the cap) and atomically claim the pending batch."""
        flow = self.config.message_flow
        started_at = self.current_time()
        deadline = started_at + timedelta(seconds=flow.input_max_wait_seconds)
        earliest_quiet_end = started_at + timedelta(seconds=flow.input_quiet_seconds)
        while True:
            async with self._chat_states_lock:
                if not state.pending:
                    return []
                latest = max(item.queued_at for item in state.pending)
                state.changed.clear()
            target = min(
                max(
                    latest + timedelta(seconds=flow.input_quiet_seconds),
                    earliest_quiet_end,
                ),
                deadline,
            )
            delay = (target - self.current_time()).total_seconds()
            if delay > 0 and await self._sleep_or_changed(state, delay):
                continue

            # Recheck and drain under one lock. A submit racing with the end of
            # sleep either extends quiet here or remains pending for the next pass.
            async with self._chat_states_lock:
                if not state.pending:
                    return []
                now = self.current_time()
                latest = max(item.queued_at for item in state.pending)
                current_target = min(
                    max(
                        latest + timedelta(seconds=flow.input_quiet_seconds),
                        earliest_quiet_end,
                    ),
                    deadline,
                )
                if current_target > now:
                    continue
                pending = sorted(state.pending, key=self._envelope_sort_key)
                state.pending.clear()
                state.changed.clear()
                return pending

    async def _prepare_envelopes(
        self,
        state: ChatWorkerState,
        envelopes: list[IncomingEnvelope],
    ) -> list[PreparedIncoming]:
        if not envelopes:
            return []

        envelopes_with_ids = [
            envelope
            for envelope in envelopes
            if isinstance(getattr(envelope.event, "id", None), int)
        ]
        history_anchor = (
            min(envelopes_with_ids, key=lambda item: item.event.id)
            if envelopes_with_ids
            else envelopes[0]
        )
        acknowledge_through = (
            max(envelopes_with_ids, key=lambda item: item.event.id)
            if envelopes_with_ids
            else envelopes[-1]
        )
        await self._import_existing_history(history_anchor.event, state.chat_key)

        try:
            input_chat = await acknowledge_through.event.get_input_chat()
            if input_chat is None:
                raise TypeError("Telethon не смог определить входной peer чата")
            await self.client.send_read_acknowledge(
                input_chat,
                message=acknowledge_through.event.message,
                max_id=acknowledge_through.event.id,
            )
        except (RPCError, OSError, TypeError, ValueError) as exc:
            print(
                "Не удалось отметить пакет до "
                f"message_id={getattr(acknowledge_through.event, 'id', None)} "
                f"прочитанным: {exc}",
                file=sys.stderr,
            )

        prepared: list[PreparedIncoming] = []
        for envelope in envelopes:
            event = envelope.event
            text = (getattr(event, "raw_text", "") or "").strip()
            image_mime_type = telegram_image_mime_type(event)
            if not text and image_mime_type is None:
                print(
                    f"Прочитано и пропущено сообщение без текста, message_id={event.id}"
                )
                continue

            image_data_url: str | None = None
            if image_mime_type is not None:
                try:
                    image_data_url = await telegram_image_data_url(
                        event, image_mime_type
                    )
                except (RPCError, OSError, TypeError, ValueError) as exc:
                    print(
                        f"Не удалось загрузить изображение message_id={event.id}: {exc}",
                        file=sys.stderr,
                    )
                    if not text:
                        continue
            stored_text = text or "[фото без подписи]"
            try:
                sender = await event.get_sender()
                sender_name = display_name(sender)
            except (RPCError, OSError, TypeError, ValueError, AttributeError):
                sender_name = str(getattr(event, "sender_id", None) or "неизвестно")

            try:
                is_new = self.memory.add_message(
                    state.chat_key,
                    "user",
                    stored_text,
                    telegram_message_id=event.id,
                    sender_name=sender_name,
                    created_at=envelope.received_at.isoformat(),
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Не удалось сохранить входящее message_id={event.id}: {exc}",
                    file=sys.stderr,
                )
                is_new = True
            if not is_new:
                print(
                    f"Повторное событие пропущено: chat_id={getattr(event, 'chat_id', None)}, "
                    f"message_id={event.id}"
                )
                continue
            prepared.append(
                PreparedIncoming(
                    event=event,
                    received_at=envelope.received_at,
                    sender_name=sender_name,
                    text=stored_text,
                    image_data_url=image_data_url,
                )
            )
        return prepared

    async def _action_target(self, event: Any) -> Any:
        try:
            input_chat = await event.get_input_chat()
            if input_chat is None:
                raise TypeError("Telethon не смог определить входной peer чата")
            return input_chat
        except (RPCError, OSError, TypeError, ValueError, AttributeError):
            return getattr(event, "chat_id", None)

    def _full_online_window_is_open(self, *, continues_conversation: bool) -> bool:
        if continues_conversation:
            return True
        now = self.current_time()
        if not self.presence.can_respond(now):
            return False
        routine_state = self.routine.state_at(now)
        if (
            routine_state.next_at is None
            or routine_state.next_activity is None
            or routine_state.next_activity.kind != "sleep"
        ):
            return True
        seconds_to_sleep = (routine_state.next_at - now).total_seconds()
        return seconds_to_sleep >= self.routine.online_behavior.sleep_buffer_seconds

    async def _generate_or_change(
        self,
        state: ChatWorkerState,
        *,
        revision: int,
        chat_key: int | str,
        history_input: list[dict[str, str]],
        active: list[PreparedIncoming],
    ) -> GeneratedReply | None:
        generation_task = asyncio.create_task(
            self._generate_answer(
                chat_key=chat_key,
                history_input=history_input,
                messages=active,
            )
        )
        changed_task = asyncio.create_task(state.changed.wait())
        try:
            done, _ = await asyncio.wait(
                {generation_task, changed_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if generation_task in done:
                result = await generation_task
                if await self._revision(state) != revision:
                    return None
                return result
            generation_task.cancel()
            await asyncio.gather(generation_task, return_exceptions=True)
            return None
        finally:
            for task in (generation_task, changed_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(generation_task, changed_task, return_exceptions=True)

    def _physical_reply_parts(self, reply: GeneratedReply) -> list[str]:
        parts: list[str] = []
        for message in reply.messages:
            parts.extend(split_telegram_text(message))
        if not parts:
            raise ValueError("Модель вернула пустой ответ")
        return parts

    async def _send_generated_reply(
        self,
        state: ChatWorkerState,
        *,
        revision: int,
        active: list[PreparedIncoming],
        reply: GeneratedReply,
        continues_conversation: bool,
    ) -> SendOutcome:
        parts = self._physical_reply_parts(reply)
        reply_event = active[-1].event
        sent_count = 0
        diary_committed = False

        for index, part in enumerate(parts):
            if index > 0:
                flow = self.config.message_flow
                minimum_ms = round(flow.inter_message_min_delay_seconds * 1000)
                maximum_ms = round(flow.inter_message_max_delay_seconds * 1000)
                delay = self._randint(minimum_ms, maximum_ms) / 1000
                if await self._sleep_or_changed(state, delay):
                    return SendOutcome(sent_count=sent_count, interrupted=True)

            if await self._revision(state) != revision:
                return SendOutcome(sent_count=sent_count, interrupted=True)
            if index == 0 and not self._full_online_window_is_open(
                continues_conversation=continues_conversation
            ):
                return SendOutcome(sent_count=sent_count, interrupted=True)

            try:
                if index == 0:
                    sent = await reply_event.reply(part)
                else:
                    sent = await self.client.send_message(reply_event.chat_id, part)
            except (RPCError, OSError, TypeError, ValueError) as exc:
                print(
                    f"Ошибка отправки части ответа на message_id={reply_event.id}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return SendOutcome(sent_count=sent_count, interrupted=False)

            sent_count += 1
            candidate_id = getattr(sent, "id", None)
            try:
                self.memory.add_message(
                    state.chat_key,
                    "assistant",
                    part,
                    telegram_message_id=(
                        candidate_id if isinstance(candidate_id, int) else None
                    ),
                    sender_name="Милана",
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Ответ отправлен, но не сохранён в памяти: {exc}",
                    file=sys.stderr,
                )
            if not diary_committed:
                try:
                    self._commit_staged_diary(
                        reply.staged_diary_entries,
                        chat_key=state.chat_key,
                        source_message_id=getattr(reply_event, "id", None),
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"Ответ отправлен, но записи дневника не сохранены: {exc}",
                        file=sys.stderr,
                    )
                diary_committed = True

        return SendOutcome(sent_count=sent_count, interrupted=False)

    async def _update_summary_while_idle(self, state: ChatWorkerState) -> None:
        async with self._chat_states_lock:
            if state.pending:
                return
            state.changed.clear()
        summary_task = asyncio.create_task(self._maybe_update_chat_summary(state.chat_key))
        changed_task = asyncio.create_task(state.changed.wait())
        try:
            done, _ = await asyncio.wait(
                {summary_task, changed_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if changed_task in done and not summary_task.done():
                summary_task.cancel()
            await asyncio.gather(summary_task, return_exceptions=True)
        finally:
            for task in (summary_task, changed_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(summary_task, changed_task, return_exceptions=True)

    async def _retire_if_idle(self, state: ChatWorkerState) -> bool:
        async with self._chat_states_lock:
            if state.pending:
                return False
            if self._chat_states.get(state.chat_key) is state:
                self._chat_states.pop(state.chat_key, None)
            return True

    async def _chat_worker(self, state: ChatWorkerState) -> None:
        active: list[PreparedIncoming] = []
        context: IncomingEnvelope | None = None
        skip_schedule_once = False
        try:
            while True:
                if not active:
                    context = await self._first_pending(state)
                    if context is None:
                        if await self._retire_if_idle(state):
                            return
                        continue
                    if not skip_schedule_once:
                        await self._wait_before_reading(
                            context.received_at,
                            received_while_online=context.received_while_online,
                            continues_conversation=context.continues_conversation,
                        )
                    skip_schedule_once = False

                envelopes = await self._wait_for_input_quiet(state)
                if envelopes:
                    try:
                        active.extend(await self._prepare_envelopes(state, envelopes))
                    except (RPCError, OSError, TypeError, ValueError) as exc:
                        latest_id = getattr(envelopes[-1].event, "id", None)
                        print(
                            f"Ошибка подготовки пакета до message_id={latest_id}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                        )
                active.sort(key=lambda item: (item.received_at, getattr(item.event, "id", 0)))
                if not active:
                    context = None
                    continue
                assert context is not None

                await self._wait_for_full_online_window(
                    continues_conversation=context.continues_conversation
                )
                active_ids = {
                    item.event.id
                    for item in active
                    if isinstance(getattr(item.event, "id", None), int)
                }
                history_input = self.memory.response_input_with_summary(
                    state.chat_key,
                    recent_limit=RECENT_MESSAGES_LIMIT,
                    exclude_user_message_ids=active_ids,
                )
                revision = await self._generation_revision(state)
                if revision is None:
                    continue
                reply_event = active[-1].event
                action_target = await self._action_target(reply_event)
                presence_started = False
                outcome: SendOutcome | None = None

                try:
                    # Telethon renews the action while this context is active, so
                    # «печатает…» covers generation, pauses and every sent part.
                    async with self.client.action(action_target, "typing"):
                        reply = await self._generate_or_change(
                            state,
                            revision=revision,
                            chat_key=state.chat_key,
                            history_input=history_input,
                            active=active,
                        )
                        if reply is None:
                            continue
                        if not self._full_online_window_is_open(
                            continues_conversation=context.continues_conversation
                        ):
                            continue
                        if await self._revision(state) != revision:
                            continue

                        await self.presence.begin_response()
                        presence_started = True
                        outcome = await self._send_generated_reply(
                            state,
                            revision=revision,
                            active=active,
                            reply=reply,
                            continues_conversation=context.continues_conversation,
                        )
                except (OpenAIError, RPCError, OSError, TypeError, ValueError) as exc:
                    print(
                        f"Ошибка ИИ-ответа для message_id={reply_event.id}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                finally:
                    if presence_started:
                        answered = outcome is not None and outcome.sent_count > 0
                        online_seconds = await self.presence.finish_response(answered=answered)
                        if online_seconds is not None:
                            print(
                                "После ответа Милана останется в сети ещё "
                                f"{online_seconds} сек."
                            )

                if outcome is not None and outcome.interrupted and outcome.sent_count == 0:
                    # No part crossed the commit boundary: keep the old inputs and
                    # merge the newly queued messages into the same response.
                    continue

                sent_count = outcome.sent_count if outcome is not None else 0
                interrupted = bool(outcome and outcome.interrupted and sent_count > 0)
                if sent_count:
                    print(
                        f"Отправлено частей ИИ-ответа: {sent_count}; "
                        f"последний входящий message_id={reply_event.id}"
                    )
                active = []
                context = None
                skip_schedule_once = interrupted
                if sent_count:
                    await self._update_summary_while_idle(state)
        finally:
            async with self._chat_states_lock:
                if self._chat_states.get(state.chat_key) is state:
                    self._chat_states.pop(state.chat_key, None)


async def run_ai_bot(client: TelegramClient) -> None:
    config = load_ai_config()
    routine = load_routine()
    openai_client = AsyncOpenAI(api_key=config.api_key)
    memory = MilanaMemoryStore(MEMORY_PATH)
    me = await client.get_me()
    presence = MilanaPresenceController(client, routine)
    responder = MilanaMessageResponder(
        client,
        openai_client,
        config,
        routine,
        memory=memory,
        presence=presence,
    )
    async def handler(event: events.NewMessage.Event) -> None:
        try:
            await responder.submit(event)
        except RuntimeError as exc:
            print(f"Входящее сообщение пропущено при остановке: {exc}", file=sys.stderr)

    client.add_event_handler(
        handler,
        events.NewMessage(incoming=True),
    )
    own_label = f"@{me.username}" if me.username else str(me.id)
    print(
        f"ИИ-бот запущен для аккаунта {own_label}: отвечаю на все входящие "
        f"текстовые сообщения и фото, модель={config.model}. "
        "Для остановки нажмите Ctrl+C."
    )
    print(format_current_status(routine, brief=True))
    presence_task = asyncio.create_task(
        presence.run(),
        name="milana-presence",
    )
    try:
        await client.run_until_disconnected()
    finally:
        await responder.shutdown()
        presence_task.cancel()
        await asyncio.gather(presence_task, return_exceptions=True)
        if client.is_connected():
            await presence.force_offline()
        memory.close()


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
