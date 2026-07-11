"""CLI-клиент Telegram для работы от имени пользовательского аккаунта."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from openai import AsyncOpenAI, BadRequestError, OpenAIError
from telethon import TelegramClient, events, functions, utils
from telethon.errors import FloodWaitError, RPCError

from milana_memory import (
    DEFAULT_HISTORY_LIMIT,
    MilanaMemoryStore,
    WRITE_DIARY_TOOL,
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
    temperature: float
    max_output_tokens: int


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


async def keep_schedule_presence(
    client: TelegramClient,
    routine: WeeklyRoutine,
    interval: int = 60,
) -> None:
    """Держит аккаунт офлайн во сне и доступным в остальное время."""
    last_offline: bool | None = None
    while True:
        offline = not routine.response_policy_at().available
        if offline != last_offline:
            try:
                await client(functions.account.UpdateStatusRequest(offline=offline))
                last_offline = offline
            except (RPCError, OSError) as exc:
                label = "не в сети" if offline else "в сети"
                print(
                    f"Не удалось обновить статус «{label}»: {exc}",
                    file=sys.stderr,
                )
        await asyncio.sleep(interval)


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


class MilanaMessageResponder:
    """Читает и обрабатывает одно сообщение в ритме текущего занятия Миланы."""

    def __init__(
        self,
        client: TelegramClient,
        openai_client: AsyncOpenAI,
        config: AIConfig,
        routine: WeeklyRoutine,
        *,
        memory: MilanaMemoryStore | None = None,
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
        self._chat_locks: dict[int | str, asyncio.Lock] = {}
        self._generation_lock = asyncio.Lock()
        self._supports_temperature: bool | None = None

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
        return request

    @staticmethod
    def _temperature_is_unsupported(exc: BadRequestError) -> bool:
        body = getattr(exc, "body", None)
        parameter = body.get("param") if isinstance(body, dict) else None
        message = str(exc).lower()
        return parameter == "temperature" or (
            "temperature" in message and "unsupported parameter" in message
        )

    async def _create_model_response(
        self, *, instructions: str, input_items: list[Any]
    ) -> Any:
        request = self._response_request(
            instructions=instructions,
            input_items=input_items,
        )
        try:
            response = await self.openai_client.responses.create(**request)
            if "temperature" in request:
                self._supports_temperature = True
            return response
        except BadRequestError as exc:
            if "temperature" not in request or not self._temperature_is_unsupported(exc):
                raise
            self._supports_temperature = False
            request.pop("temperature")
            print(
                f"Модель {self.config.model} не поддерживает temperature; "
                "повторяю запрос без этого параметра"
            )
            return await self.openai_client.responses.create(**request)

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

    async def _generate_answer(
        self,
        *,
        chat_key: int | str,
        message_id: int | None,
        sender_name: str,
        text: str,
        history_input: list[dict[str, str]],
    ) -> str:
        instructions = (
            f"{self.config.instructions}\n\n"
            f"{build_schedule_prompt(self.routine, self.current_time())}\n\n"
            f"{self.memory.diary_instructions()}"
        )
        input_items: list[Any] = [*history_input]
        input_items.append({"role": "user", "content": f"{sender_name}: {text}"})

        for _ in range(4):
            response = await self._create_model_response(
                instructions=instructions,
                input_items=input_items,
            )
            output = list(getattr(response, "output", None) or [])
            calls = [
                item
                for item in output
                if getattr(item, "type", None) == "function_call"
                and getattr(item, "name", None) == "write_diary"
            ]
            if not calls:
                return str(getattr(response, "output_text", "") or "")

            # The Responses API expects the model output followed by one result
            # for every function call on the next request.
            input_items.extend(output)
            for call in calls:
                result = self._execute_diary_call(
                    call,
                    chat_key=chat_key,
                    source_message_id=message_id,
                )
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

    async def _wait_before_reading(self, received_at: datetime) -> None:
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
            if self.routine.response_policy_at(now).available:
                return

            # Системные часы или расписание могли измениться во время ожидания.
            # Во сне по-прежнему ничего не читаем и строим новый план от «сейчас».
            plan = self.routine.plan_response(now, randint=self._randint)

    async def _wait_out_sleep(self) -> None:
        while True:
            now = self.current_time()
            if self.routine.response_policy_at(now).available:
                return
            plan = self.routine.plan_response(now, randint=self._randint)
            print(
                "Ответ готов, но Милана спит; отправка перенесена на "
                f"{plan.respond_at:%d.%m %H:%M:%S}"
            )
            await self._sleep_until(plan.respond_at)

    async def process(self, event: events.NewMessage.Event) -> None:
        """Последовательно обрабатывает сообщения одного чата, не блокируя другие."""
        chat_key: int | str = event.chat_id
        if chat_key is None:
            chat_key = str(event.sender_id or "unknown")
        chat_lock = self._chat_locks.setdefault(chat_key, asyncio.Lock())

        async with chat_lock:
            await self._process_locked(event)

    async def _process_locked(self, event: events.NewMessage.Event) -> None:
        received_at = self.received_time(event)
        print(
            f"Получено входящее сообщение: chat_id={event.chat_id}, "
            f"message_id={event.id}; ожидаю подходящего момента для чтения"
        )

        try:
            await self._wait_before_reading(received_at)

            action_target: Any = event.chat_id
            read_acknowledged = False
            try:
                input_chat = await event.get_input_chat()
                if input_chat is None:
                    raise TypeError("Telethon не смог определить входной peer чата")
                action_target = input_chat
                await self.client.send_read_acknowledge(
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
                print(
                    f"Прочитано и пропущено сообщение без текста, message_id={event.id}"
                )
                return

            try:
                sender = await event.get_sender()
                sender_name = display_name(sender)
            except (RPCError, OSError, TypeError, ValueError):
                sender_name = str(event.sender_id or "неизвестно")
            read_status = (
                "прочитано" if read_acknowledged else "без подтверждения прочтения"
            )
            print(f"Сообщение от {sender_name}: {read_status}; генерирую ответ")

            chat_key: int | str = event.chat_id
            if chat_key is None:
                chat_key = str(event.sender_id or "unknown")
            await self._import_existing_history(event, chat_key)
            history_input = self.memory.response_input(
                chat_key, limit=self.history_limit
            )
            is_new = self.memory.add_message(
                chat_key,
                "user",
                text,
                telegram_message_id=event.id,
                sender_name=sender_name,
                created_at=received_at.isoformat(),
            )
            if not is_new:
                print(
                    f"Повторное событие пропущено: chat_id={event.chat_id}, "
                    f"message_id={event.id}"
                )
                return

            async with self._generation_lock:
                async with self.client.action(action_target, "typing"):
                    answer = await self._generate_answer(
                        chat_key=chat_key,
                        message_id=event.id,
                        sender_name=sender_name,
                        text=text,
                        history_input=history_input,
                    )

            answer_parts = split_telegram_text(answer)
            if not answer_parts:
                raise ValueError("Модель вернула пустой ответ")

            sent_message_id: int | None = None
            for index, part in enumerate(answer_parts):
                await self._wait_out_sleep()
                if index == 0:
                    sent = await event.reply(part)
                else:
                    sent = await self.client.send_message(event.chat_id, part)
                candidate_id = getattr(sent, "id", None)
                if sent_message_id is None and isinstance(candidate_id, int):
                    sent_message_id = candidate_id
            self.memory.add_message(
                chat_key,
                "assistant",
                answer,
                telegram_message_id=sent_message_id,
                sender_name="Милана",
            )
            print(f"Отправлен ИИ-ответ на message_id={event.id}")
        except (OpenAIError, RPCError, OSError, TypeError, ValueError) as exc:
            print(
                f"Ошибка ИИ-ответа для message_id={event.id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )


async def run_ai_bot(client: TelegramClient) -> None:
    config = load_ai_config()
    routine = load_routine()
    openai_client = AsyncOpenAI(api_key=config.api_key)
    memory = MilanaMemoryStore(MEMORY_PATH)
    me = await client.get_me()
    responder = MilanaMessageResponder(
        client, openai_client, config, routine, memory=memory
    )
    pending_tasks: set[asyncio.Task[None]] = set()

    async def handler(event: events.NewMessage.Event) -> None:
        task = asyncio.create_task(
            responder.process(event),
            name=f"milana-message-{event.chat_id}-{event.id}",
        )
        pending_tasks.add(task)

        def finish_message(done: asyncio.Task[None]) -> None:
            pending_tasks.discard(done)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                print(
                    f"Необработанная ошибка фонового ответа: {error}",
                    file=sys.stderr,
                )

        task.add_done_callback(finish_message)

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
    presence_task = asyncio.create_task(
        keep_schedule_presence(client, routine),
        name="milana-presence",
    )
    try:
        await client.run_until_disconnected()
    finally:
        for task in tuple(pending_tasks):
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        presence_task.cancel()
        await asyncio.gather(presence_task, return_exceptions=True)
        if client.is_connected():
            try:
                await client(functions.account.UpdateStatusRequest(offline=True))
            except (RPCError, OSError) as exc:
                print(f"Не удалось выставить статус «не в сети»: {exc}", file=sys.stderr)
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
