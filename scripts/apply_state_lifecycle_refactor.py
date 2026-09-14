"""One-shot safe extraction of agent lifecycle and heartbeat persistence.

The guarded workflow commits the generated module only after compile, lint, and tests pass.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "milana_state.py"
MODULE_PATH = ROOT / "milana" / "state_lifecycle.py"

TELEGRAM_IMPORT = "from milana.state_telegram import TelegramStateStoreMixin\n"
MIXIN_IMPORT = "from milana.state_lifecycle import LifecycleStateStoreMixin\n"

MODULE_HEADER = '''"""Agent state, recovery, and heartbeat-job persistence."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping
from uuid import uuid4

from milana.state_models import (
    MAX_NEED_DELTA,
    NEED_NAMES,
    AgentState,
    HeartbeatJob,
    RecoveryWindow,
    StateConflictError,
    _bounded_delta,
    _clean_text,
    _identifier,
    _integer,
    _json_dump,
    _json_load,
    _now,
    _parse_timestamp,
    _timestamp,
    _utc_datetime,
)


class LifecycleStateStoreMixin:
    """Agent-state, recovery, and heartbeat persistence for MilanaStateStore."""

'''

MODULE_EXPORTS = '''\n\n__all__ = ["LifecycleStateStoreMixin"]\n'''


def main() -> None:
    text = STATE_PATH.read_text(encoding="utf-8")
    original_lines = len(text.splitlines())

    if MIXIN_IMPORT in text or MODULE_PATH.exists():
        raise SystemExit("state lifecycle is already refactored")
    if TELEGRAM_IMPORT not in text:
        raise RuntimeError("Telegram state extraction must land first")

    start_marker = "    @staticmethod\n    def _agent_state_from_row(row: sqlite3.Row) -> AgentState:\n"
    end_marker = "    @staticmethod\n    def _entity_from_row(row: sqlite3.Row) -> WorldEntity:\n"
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    methods = text[start:end].rstrip()

    required = (
        "def get_agent_state(",
        "def apply_need_deltas(",
        "def begin_recovery(",
        "def schedule_heartbeat_job(",
        "def claim_due_heartbeat_jobs(",
        "def next_heartbeat_job_due_at(",
    )
    for marker in required:
        if marker not in methods:
            raise RuntimeError(f"lifecycle extraction lost {marker}")

    text = text[:start] + text[end:]
    text = text.replace(TELEGRAM_IMPORT, TELEGRAM_IMPORT + MIXIN_IMPORT, 1)
    old_class = "class MilanaStateStore(TelegramStateStoreMixin):\n"
    new_class = "class MilanaStateStore(TelegramStateStoreMixin, LifecycleStateStoreMixin):\n"
    if old_class not in text:
        raise RuntimeError("MilanaStateStore mixin declaration not found")
    text = text.replace(old_class, new_class, 1)

    removed = original_lines - len(text.splitlines())
    if removed < 450:
        raise RuntimeError(f"lifecycle extraction removed only {removed} lines")

    MODULE_PATH.write_text(MODULE_HEADER + methods + MODULE_EXPORTS, encoding="utf-8")
    STATE_PATH.write_text(text, encoding="utf-8")
    print(
        f"Extracted lifecycle persistence: milana_state.py {original_lines} -> "
        f"{len(text.splitlines())} lines; created {MODULE_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
