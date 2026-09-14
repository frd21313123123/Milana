"""One-shot safe extraction of world-model persistence from MilanaStateStore.

The guarded workflow commits the generated module only after compile, lint, and tests pass.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "milana_state.py"
MODULE_PATH = ROOT / "milana" / "state_world.py"

LIFECYCLE_IMPORT = "from milana.state_lifecycle import LifecycleStateStoreMixin\n"
MIXIN_IMPORT = "from milana.state_world import WorldStateStoreMixin\n"

MODULE_HEADER = '''"""World entities, facts, events, goals, relationships, and atomic state updates."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from milana.state_models import (
    MAX_ACTIVE_GOALS,
    MAX_HEARTBEAT_CHANGES,
    MAX_NEED_DELTA,
    MAX_RELATIONSHIP_DELTA,
    NEED_NAMES,
    AgentState,
    FactSeed,
    Goal,
    GoalChange,
    GoalLimitError,
    HeartbeatChanges,
    LifeEvent,
    LockedFactError,
    NewEntity,
    NewLifeEvent,
    Relationship,
    RelationshipDelta,
    SkillAuditRecord,
    StateConflictError,
    WorldContext,
    WorldEntity,
    WorldFact,
    WorldSummary,
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
    adaptive_initiative_cooldown,
    initiative_allowed,
)


class WorldStateStoreMixin:
    """World-model persistence and atomic state reducers for MilanaStateStore."""

'''

MODULE_EXPORTS = '''\n\n__all__ = ["WorldStateStoreMixin"]\n'''


def main() -> None:
    text = STATE_PATH.read_text(encoding="utf-8")
    original_lines = len(text.splitlines())

    if MIXIN_IMPORT in text or MODULE_PATH.exists():
        raise SystemExit("world state persistence is already refactored")
    if LIFECYCLE_IMPORT not in text:
        raise RuntimeError("lifecycle state extraction must land first")

    start_marker = "    @staticmethod\n    def _entity_from_row(row: sqlite3.Row) -> WorldEntity:\n"
    end_marker = "\n\n# Shorter name for dependency injection while retaining the explicit public one.\n"
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    methods = text[start:end].rstrip()

    required = (
        "def create_entity(",
        "def set_fact(",
        "def add_life_event(",
        "def create_goal(",
        "def adjust_relationship(",
        "def add_world_summary(",
        "def apply_heartbeat_changes(",
        "def load_world_context(",
    )
    for marker in required:
        if marker not in methods:
            raise RuntimeError(f"world-state extraction lost {marker}")

    text = text[:start] + text[end:]
    text = text.replace(LIFECYCLE_IMPORT, LIFECYCLE_IMPORT + MIXIN_IMPORT, 1)
    old_class = "class MilanaStateStore(TelegramStateStoreMixin, LifecycleStateStoreMixin):\n"
    new_class = (
        "class MilanaStateStore("
        "TelegramStateStoreMixin, LifecycleStateStoreMixin, WorldStateStoreMixin):\n"
    )
    if old_class not in text:
        raise RuntimeError("MilanaStateStore lifecycle declaration not found")
    text = text.replace(old_class, new_class, 1)

    removed = original_lines - len(text.splitlines())
    if removed < 1_100:
        raise RuntimeError(f"world-state extraction removed only {removed} lines")

    MODULE_PATH.write_text(MODULE_HEADER + methods + MODULE_EXPORTS, encoding="utf-8")
    STATE_PATH.write_text(text, encoding="utf-8")
    print(
        f"Extracted world persistence: milana_state.py {original_lines} -> "
        f"{len(text.splitlines())} lines; created {MODULE_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
