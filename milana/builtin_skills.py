"""Data schemas and explicit host bindings for the built-in Telegram skill tree."""

from __future__ import annotations

from typing import Any

from .registry import ActivationHook, SkillRegistry
from .types import SkillExecutor


MAX_TELEGRAM_SCHEDULE_DELAY_SECONDS = 365 * 24 * 60 * 60

SCHEDULE_MESSAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "schedule_message",
    "description": (
        "Поставить готовое сообщение в отложенную отправку текущему собеседнику. "
        "Не используй для обычного немедленного ответа."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "delay_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TELEGRAM_SCHEDULE_DELAY_SECONDS,
            },
            "message": {"type": "string", "minLength": 1, "maxLength": 4_000},
        },
        "required": ["delay_seconds", "message"],
        "additionalProperties": False,
    },
}

OPEN_STICKER_PICKER_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "open_sticker_picker",
    "description": (
        "Открыть внутренний визуальный выбор стикеров. Сначала вызови с "
        "pack_id=null для индекса наборов, затем с показанным pack_id для просмотра "
        "стикеров набора."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "pack_id": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": "Показанный ID набора либо null для индекса.",
            }
        },
        "required": ["pack_id"],
        "additionalProperties": False,
    },
}

SEND_STICKER_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "send_sticker",
    "description": (
        "Выбрать для немедленной отправки стикер, уже показанный через "
        "open_sticker_picker в текущем ходе."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {"sticker_id": {"type": "string"}},
        "required": ["sticker_id"],
        "additionalProperties": False,
    },
}

SCHEDULE_STICKER_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "schedule_sticker",
    "description": (
        "Поставить уже показанный стикер в отложенную отправку текущему собеседнику."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "sticker_id": {"type": "string"},
            "delay_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_TELEGRAM_SCHEDULE_DELAY_SECONDS,
            },
        },
        "required": ["sticker_id", "delay_seconds"],
        "additionalProperties": False,
    },
}

TELEGRAM_TOOLS = (SCHEDULE_MESSAGE_TOOL,)
STICKER_TOOLS = (
    OPEN_STICKER_PICKER_TOOL,
    SEND_STICKER_TOOL,
    SCHEDULE_STICKER_TOOL,
)


def bind_telegram_skill_tree(
    registry: SkillRegistry,
    *,
    telegram_executor: SkillExecutor,
    sticker_executor: SkillExecutor,
    telegram_on_activate: ActivationHook | None = None,
) -> SkillRegistry:
    """Bind manifests to trusted objects supplied by MilanaService.

    The manifest files remain declarative and cannot select/import Python code.
    Returning ``registry`` keeps service construction concise.
    """

    registry.bind(
        "telegram",
        tools=TELEGRAM_TOOLS,
        executor=telegram_executor,
        on_activate=telegram_on_activate,
    )
    registry.bind(
        "telegram.stickers",
        tools=STICKER_TOOLS,
        executor=sticker_executor,
    )
    return registry
