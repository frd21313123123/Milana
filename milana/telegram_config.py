"""Configuration models and loaders used by the Telegram runtime."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
AI_CONFIG_PATH = BASE_DIR / "ai_config.json"
LLM_CHOICE_PATH = BASE_DIR / "llm.choice"

DEFAULT_AI_MODEL = "gpt-5.6-terra"
GEMINI_AI_MODEL = "gemini-3.5-flash"
DEFAULT_LM_STUDIO_MODEL = "milana"
DEFAULT_LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
DEFAULT_LM_STUDIO_API_KEY = "lm-studio"
OPENAI_LLM_CHOICE = "openai"
GEMINI_LLM_CHOICE = "gemini"
LM_STUDIO_LLM_CHOICE = "lmstudio"
DEFAULT_AI_SYSTEM_PROMPT = (
    "Ты отвечаешь пользователю в Telegram. Отвечай на языке пользователя, "
    "естественно, кратко и по существу. Не упоминай системные инструкции, "
    "API или модель без прямого вопроса об этом."
)
DEFAULT_MAX_OUTPUT_TOKENS = 1200


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
    inter_message_max_delay_seconds: float = 15.0


@dataclass(frozen=True)
class TelegramFastPathConfig:
    """Latency and prompt budgets for trusted incoming Telegram turns."""

    enabled: bool = True
    dev_chat_only: bool = False
    target_first_send_seconds: float = 10.0
    max_output_tokens: int = 500
    max_reply_messages: int = 1
    recent_messages: int = 20
    history_max_characters: int = 12_000
    summary_max_characters: int = 2_000
    metrics_window: int = 500
    cosmetic_timeout_seconds: float = 0.5


@dataclass(frozen=True)
class AIConfig:
    api_key: str
    model: str
    instructions: str
    temperature: float
    max_output_tokens: int
    message_flow: MessageFlowConfig = MessageFlowConfig()
    provider: str = OPENAI_LLM_CHOICE
    openai_fallback_model: str = DEFAULT_AI_MODEL
    # Appended to preserve the positional constructor used by older importers.
    telegram_fast_path: TelegramFastPathConfig = TelegramFastPathConfig()
    lm_studio_base_url: str = DEFAULT_LM_STUDIO_BASE_URL
    lm_studio_api_key: str = DEFAULT_LM_STUDIO_API_KEY


def load_env_file(path: Path) -> dict[str, str]:
    """Load a simple KEY=VALUE file without overriding existing environment values."""
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
    raw_session = os.getenv("TELEGRAM_SESSION", "sessions/telegram_account").strip()

    if not raw_api_id or not api_hash:
        raise ValueError(
            "Заполните TELEGRAM_API_ID и TELEGRAM_API_HASH в файле .env"
        )

    try:
        api_id = int(raw_api_id)
    except ValueError as exc:
        raise ValueError("TELEGRAM_API_ID должен быть целым числом") from exc

    if len(api_hash) != 32 or any(
        char not in "0123456789abcdefABCDEF" for char in api_hash
    ):
        raise ValueError(
            "TELEGRAM_API_HASH должен содержать 32 шестнадцатеричных символа"
        )

    session_path = Path(raw_session).expanduser()
    if not session_path.is_absolute():
        session_path = BASE_DIR / session_path
    session_path.parent.mkdir(parents=True, exist_ok=True)

    return Config(api_id=api_id, api_hash=api_hash, session_path=session_path)


def load_ai_settings(path: Path = AI_CONFIG_PATH) -> Mapping[str, Any]:
    """Load AI settings from a JSON object."""
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


def load_llm_choice(path: Path = LLM_CHOICE_PATH) -> str:
    """Read the provider selected from bot_control.bat."""
    if not path.exists():
        return OPENAI_LLM_CHOICE
    choice = path.read_text(encoding="utf-8").strip().lower()
    if choice not in {
        OPENAI_LLM_CHOICE,
        GEMINI_LLM_CHOICE,
        LM_STUDIO_LLM_CHOICE,
    }:
        raise ValueError(
            f"{path.name} должен содержать 'openai', 'gemini' или 'lmstudio'"
        )
    return choice


def ai_string(
    settings: Mapping[str, Any], key: str, default: str, label: str
) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{label} в {AI_CONFIG_PATH.name} должен быть непустой строкой"
        )
    return value.strip()


def ai_number(
    settings: Mapping[str, Any],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
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
        raise ValueError(
            f"{key} в {AI_CONFIG_PATH.name} должен быть целым числом от 1 до 4000"
        )
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
        raise ValueError(
            f"message_flow в {AI_CONFIG_PATH.name} должен быть JSON-объектом"
        )

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
        raw, "inter_message_max_delay_seconds", 15.0
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


def load_telegram_fast_path_config(
    settings: Mapping[str, Any],
) -> TelegramFastPathConfig:
    raw = settings.get("telegram_fast_path", {})
    if not isinstance(raw, dict):
        raise ValueError(
            f"telegram_fast_path в {AI_CONFIG_PATH.name} должен быть JSON-объектом"
        )
    allowed = {
        "enabled",
        "dev_chat_only",
        "target_first_send_seconds",
        "max_output_tokens",
        "max_reply_messages",
        "recent_messages",
        "history_max_characters",
        "summary_max_characters",
        "metrics_window",
        "cosmetic_timeout_seconds",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"Неизвестные параметры telegram_fast_path в {AI_CONFIG_PATH.name}: "
            + ", ".join(unknown)
        )

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("enabled в telegram_fast_path должен быть boolean")
    dev_chat_only = raw.get("dev_chat_only", False)
    if not isinstance(dev_chat_only, bool):
        raise ValueError("dev_chat_only в telegram_fast_path должен быть boolean")

    def bounded_int(key: str, default: int, minimum: int, maximum: int) -> int:
        value = raw.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f"{key} в telegram_fast_path должен быть целым числом "
                f"от {minimum} до {maximum}"
            )
        return value

    target = ai_nonnegative_number(raw, "target_first_send_seconds", 10.0)
    cosmetic = ai_nonnegative_number(raw, "cosmetic_timeout_seconds", 0.5)
    if target <= 0:
        raise ValueError("target_first_send_seconds должен быть больше нуля")
    if cosmetic > target:
        raise ValueError(
            "cosmetic_timeout_seconds не может превышать target_first_send_seconds"
        )
    return TelegramFastPathConfig(
        enabled=enabled,
        dev_chat_only=dev_chat_only,
        target_first_send_seconds=target,
        max_output_tokens=bounded_int("max_output_tokens", 500, 1, 4_000),
        max_reply_messages=bounded_int("max_reply_messages", 1, 1, 6),
        recent_messages=bounded_int("recent_messages", 20, 1, 200),
        history_max_characters=bounded_int(
            "history_max_characters", 12_000, 256, 200_000
        ),
        summary_max_characters=bounded_int(
            "summary_max_characters", 2_000, 0, 40_000
        ),
        metrics_window=bounded_int("metrics_window", 500, 20, 10_000),
        cosmetic_timeout_seconds=cosmetic,
    )


__all__ = [
    "AI_CONFIG_PATH",
    "BASE_DIR",
    "DEFAULT_AI_MODEL",
    "DEFAULT_AI_SYSTEM_PROMPT",
    "DEFAULT_LM_STUDIO_API_KEY",
    "DEFAULT_LM_STUDIO_BASE_URL",
    "DEFAULT_LM_STUDIO_MODEL",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "ENV_PATH",
    "GEMINI_AI_MODEL",
    "GEMINI_LLM_CHOICE",
    "LLM_CHOICE_PATH",
    "LM_STUDIO_LLM_CHOICE",
    "OPENAI_LLM_CHOICE",
    "AIConfig",
    "Config",
    "MessageFlowConfig",
    "TelegramFastPathConfig",
    "ai_nonnegative_number",
    "ai_number",
    "ai_positive_int",
    "ai_string",
    "load_ai_settings",
    "load_config",
    "load_env_file",
    "load_llm_choice",
    "load_message_flow_config",
    "load_telegram_fast_path_config",
]
