"""Persistent rich state and world model for the standalone Milana service.

The store intentionally uses its own SQLite connection.  Pointing it at the
same path as :class:`milana_memory.MilanaMemoryStore` performs an additive
migration and leaves all legacy chat, diary and delayed-action tables intact.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4
from milana.state_schema import create_state_schema
from milana.state_telegram import TelegramStateStoreMixin
from milana.state_lifecycle import LifecycleStateStoreMixin
from milana.state_world import WorldStateStoreMixin
from milana.state_models import (
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


class MilanaStateStore(TelegramStateStoreMixin, LifecycleStateStoreMixin, WorldStateStoreMixin):
    """Thread-safe additive SQLite repository for Milana's lived state."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            timeout=5.0,
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            if self.path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            create_state_schema(self._connection)

    @classmethod
    def from_memory(cls, memory: Any) -> "MilanaStateStore":
        path = getattr(memory, "path", None)
        if path is None:
            raise TypeError("memory должен предоставлять путь SQLite в поле path")
        if str(path) == ":memory:":
            raise ValueError(
                "Независимое подключение не может разделить SQLite :memory:; "
                "передайте файловый путь"
            )
        return cls(path)


    def close(self) -> None:
        with self._lock:
            self._connection.close()


    def __enter__(self) -> "MilanaStateStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()



# Shorter name for dependency injection while retaining the explicit public one.
StateStore = MilanaStateStore
