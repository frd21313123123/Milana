"""One-shot safe extraction of Telegram persistence methods from MilanaStateStore."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "milana_state.py"
MODULE_PATH = ROOT / "milana" / "state_telegram.py"

SCHEMA_IMPORT = "from milana.state_schema import create_state_schema\n"
MIXIN_IMPORT = "from milana.state_telegram import TelegramStateStoreMixin\n"

MODULE_HEADER = '''"""Durable Telegram notice, outbox, acknowledgement, and latency persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from milana.state_models import (
    StateConflictError,
    TelegramAckIntent,
    TelegramOutboxEntry,
    TelegramOutboxSentPart,
    TelegramTurnMetric,
    _clean_text,
    _identifier,
    _integer,
    _json_dump,
    _json_load,
    _now,
    _parse_timestamp,
    _timestamp,
)


class TelegramStateStoreMixin:
    """Telegram-specific persistence mixed into the main SQLite state store."""

'''

MODULE_EXPORTS = '''\n\n__all__ = ["TelegramStateStoreMixin"]\n'''


def main() -> None:
    text = STATE_PATH.read_text(encoding="utf-8")
    original_lines = len(text.splitlines())

    if MIXIN_IMPORT in text or MODULE_PATH.exists():
        raise SystemExit("Telegram state persistence is already refactored")
    if SCHEMA_IMPORT not in text:
        raise RuntimeError("state schema extraction must land first")

    start_marker = "    def record_telegram_notice(\n"
    end_marker = '\n    def __enter__(self) -> "MilanaStateStore":\n'
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    methods = text[start:end].rstrip()

    required_methods = (
        "def record_telegram_notice(",
        "def prepare_telegram_ack_intent(",
        "def prepare_telegram_outbox(",
        "def record_telegram_turn_metric(",
        "def telegram_latency_summary(",
    )
    for marker in required_methods:
        if marker not in methods:
            raise RuntimeError(f"Telegram persistence extraction lost {marker}")

    text = text[:start] + text[end:]
    text = text.replace(SCHEMA_IMPORT, SCHEMA_IMPORT + MIXIN_IMPORT, 1)
    old_class = "class MilanaStateStore:\n"
    new_class = "class MilanaStateStore(TelegramStateStoreMixin):\n"
    if old_class not in text:
        raise RuntimeError("MilanaStateStore class declaration not found")
    text = text.replace(old_class, new_class, 1)

    removed = original_lines - len(text.splitlines())
    if removed < 850:
        raise RuntimeError(f"Telegram persistence extraction removed only {removed} lines")

    MODULE_PATH.write_text(
        MODULE_HEADER + methods + MODULE_EXPORTS,
        encoding="utf-8",
    )
    STATE_PATH.write_text(text, encoding="utf-8")
    print(
        f"Extracted Telegram persistence: milana_state.py {original_lines} -> "
        f"{len(text.splitlines())} lines; created {MODULE_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
