"""One-shot safe refactor for extracting persistent state models and policies."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "milana_state.py"
MODULE_PATH = ROOT / "milana" / "state_models.py"

IMPORT_ANCHOR = "from uuid import uuid4\n"
MODEL_IMPORT = '''from milana.state_models import (
    MAX_ACTIVE_GOALS,
    MAX_HEARTBEAT_CHANGES,
    MAX_INITIATIVE_COOLDOWN_HOURS,
    MAX_NEED_DELTA,
    MAX_RELATIONSHIP_DELTA,
    MIN_INITIATIVE_COOLDOWN_HOURS,
    NEED_NAMES,
    AgentState,
    FactSeed,
    Goal,
    GoalChange,
    GoalLimitError,
    HeartbeatChanges,
    HeartbeatJob,
    LifeEvent,
    LockedFactError,
    NewEntity,
    NewLifeEvent,
    RecoveryWindow,
    Relationship,
    RelationshipDelta,
    SkillAuditRecord,
    StateConflictError,
    TelegramAckIntent,
    TelegramOutboxEntry,
    TelegramOutboxSentPart,
    TelegramTurnMetric,
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
    calculate_adaptive_cooldown,
    initiative_allowed,
)
'''

MODULE_HEADER = '''"""Data models, validation primitives and policies for Milana persistent state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


'''

PUBLIC_EXPORTS = '''

__all__ = [
    "MAX_ACTIVE_GOALS",
    "MAX_HEARTBEAT_CHANGES",
    "MAX_INITIATIVE_COOLDOWN_HOURS",
    "MAX_NEED_DELTA",
    "MAX_RELATIONSHIP_DELTA",
    "MIN_INITIATIVE_COOLDOWN_HOURS",
    "NEED_NAMES",
    "AgentState",
    "FactSeed",
    "Goal",
    "GoalChange",
    "GoalLimitError",
    "HeartbeatChanges",
    "HeartbeatJob",
    "LifeEvent",
    "LockedFactError",
    "NewEntity",
    "NewLifeEvent",
    "RecoveryWindow",
    "Relationship",
    "RelationshipDelta",
    "SkillAuditRecord",
    "StateConflictError",
    "TelegramAckIntent",
    "TelegramOutboxEntry",
    "TelegramOutboxSentPart",
    "TelegramTurnMetric",
    "WorldContext",
    "WorldEntity",
    "WorldFact",
    "WorldSummary",
    "adaptive_initiative_cooldown",
    "calculate_adaptive_cooldown",
    "initiative_allowed",
]
'''


def main() -> None:
    text = STATE_PATH.read_text(encoding="utf-8")
    original_lines = len(text.splitlines())

    if "from milana.state_models import (" in text:
        raise SystemExit("milana_state.py models are already refactored")
    if IMPORT_ANCHOR not in text:
        raise RuntimeError("state models import anchor not found")

    start_marker = 'NEED_NAMES = ("social", "rest", "novelty", "achievement")'
    end_marker = 'class MilanaStateStore:'
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    models_source = text[start:end].rstrip()
    text = text[:start] + text[end:]
    text = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + MODEL_IMPORT, 1)

    # Existing callers deliberately keep importing these names from milana_state.
    for name in (
        "AgentState",
        "HeartbeatChanges",
        "Relationship",
        "TelegramOutboxEntry",
        "StateConflictError",
        "adaptive_initiative_cooldown",
        "initiative_allowed",
        "_utc_datetime",
        "_timestamp",
    ):
        if name not in text:
            raise RuntimeError(f"compatibility import disappeared: {name}")

    new_lines = len(text.splitlines())
    removed = original_lines - new_lines
    if removed < 360:
        raise RuntimeError(f"state models extraction removed only {removed} lines")

    MODULE_PATH.write_text(
        MODULE_HEADER + models_source + PUBLIC_EXPORTS,
        encoding="utf-8",
    )
    STATE_PATH.write_text(text, encoding="utf-8")
    print(
        f"Extracted state models: milana_state.py {original_lines} -> {new_lines} lines; "
        f"created {MODULE_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
