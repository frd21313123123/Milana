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


NEED_NAMES = ("social", "rest", "novelty", "achievement")
MAX_ACTIVE_GOALS = 20
MAX_HEARTBEAT_CHANGES = 3
MAX_NEED_DELTA = 15
MAX_RELATIONSHIP_DELTA = 10
MIN_INITIATIVE_COOLDOWN_HOURS = 2
MAX_INITIATIVE_COOLDOWN_HOURS = 72


class StateConflictError(RuntimeError):
    """The caller tried to commit against an obsolete agent-state revision."""


class LockedFactError(ValueError):
    """An immutable persona/world fact was about to be overwritten."""


class GoalLimitError(ValueError):
    """The maximum number of active autonomous goals would be exceeded."""


@dataclass(frozen=True)
class AgentState:
    revision: int
    mood: str
    valence: int
    arousal: int
    social: int
    rest: int
    novelty: int
    achievement: int
    current_intention: str | None
    last_heartbeat_at: datetime | None
    next_heartbeat_at: datetime | None
    heartbeat_paused: bool
    last_service_seen_at: datetime | None
    recovery_pending_from: datetime | None
    recovery_pending_to: datetime | None
    recovery_completed_through: datetime | None
    updated_at: datetime

    @property
    def needs(self) -> dict[str, int]:
        return {
            "social": self.social,
            "rest": self.rest,
            "novelty": self.novelty,
            "achievement": self.achievement,
        }


@dataclass(frozen=True)
class HeartbeatJob:
    id: str
    kind: str
    due_at: datetime
    status: str
    payload: Mapping[str, Any]
    attempts: int
    idempotency_key: str | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True)
class FactSeed:
    key: str
    value: Any
    locked: bool = False
    source: str | None = None


@dataclass(frozen=True)
class NewEntity:
    kind: str
    name: str
    description: str = ""
    is_real: bool = False
    entity_id: str | None = None
    facts: tuple[FactSeed, ...] = ()


@dataclass(frozen=True)
class WorldEntity:
    id: str
    kind: str
    name: str
    description: str
    is_real: bool
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class WorldFact:
    id: int
    entity_id: str
    key: str
    value: Any
    locked: bool
    version: int
    source: str | None
    valid_from: datetime
    superseded_at: datetime | None


@dataclass(frozen=True)
class NewLifeEvent:
    title: str
    description: str
    kind: str = "life"
    importance: int = 50
    entity_ids: tuple[str, ...] = ()
    happened_at: datetime | None = None
    raw_payload: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LifeEvent:
    id: str
    kind: str
    title: str
    description: str
    importance: int
    entity_ids: tuple[str, ...]
    happened_at: datetime
    status: str
    raw_payload: Mapping[str, Any] | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class GoalChange:
    operation: str
    goal_id: str | None = None
    title: str | None = None
    description: str = ""
    horizon: str | None = None
    progress: int | None = None


@dataclass(frozen=True)
class Goal:
    id: str
    horizon: str
    title: str
    description: str
    status: str
    progress: int
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True)
class RelationshipDelta:
    entity_id: str
    closeness: int = 0
    reciprocity: int = 0
    tension: int = 0
    awaiting_reply: bool | None = None
    blocked: bool | None = None
    interacted_at: datetime | None = None


@dataclass(frozen=True)
class Relationship:
    entity_id: str
    closeness: int
    reciprocity: int
    tension: int
    awaiting_reply: bool
    blocked: bool
    last_interaction_at: datetime | None
    last_initiative_at: datetime | None
    updated_at: datetime


@dataclass(frozen=True)
class WorldSummary:
    id: str
    period_start: datetime
    period_end: datetime
    content: str
    created_at: datetime


@dataclass(frozen=True)
class SkillAuditRecord:
    id: int
    turn_id: str
    skill_id: str
    action: str
    success: bool
    detail: Mapping[str, Any]
    created_at: datetime


@dataclass(frozen=True)
class RecoveryWindow:
    started_at: datetime
    ended_at: datetime

    @property
    def duration(self) -> timedelta:
        return self.ended_at - self.started_at

    @property
    def duration_seconds(self) -> float:
        return self.duration.total_seconds()


@dataclass(frozen=True)
class HeartbeatChanges:
    """One bounded, atomic world update produced by a heartbeat turn."""

    entities: tuple[NewEntity, ...] = ()
    events: tuple[NewLifeEvent, ...] = ()
    goals: tuple[GoalChange, ...] = ()
    need_deltas: Mapping[str, int] = field(default_factory=dict)
    relationships: tuple[RelationshipDelta, ...] = ()
    mood: str | None = None
    valence: int | None = None
    arousal: int | None = None
    current_intention: str | None = None


@dataclass(frozen=True)
class WorldContext:
    state: AgentState
    goals: tuple[Goal, ...]
    entities: tuple[WorldEntity, ...]
    events: tuple[LifeEvent, ...]
    relationships: tuple[Relationship, ...]
    summaries: tuple[WorldSummary, ...]


def _utc_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("Время должно быть datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return _utc_datetime(value).isoformat(timespec="microseconds")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean_text(
    value: Any,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} должен быть строкой")
    result = value.strip()
    if not result and not allow_empty:
        raise ValueError(f"{label} не может быть пустым")
    if len(result) > maximum:
        raise ValueError(f"{label} не может быть длиннее {maximum} символов")
    return result


def _identifier(value: Any, label: str = "ID") -> str:
    return _clean_text(value, label, maximum=255)


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} должен быть целым числом")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} должен быть от {minimum} до {maximum}")
    return value


def _json_dump(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("Значение должно сериализоваться в JSON") from exc


def _json_load(value: str) -> Any:
    return json.loads(value)


def _bounded_delta(value: Any, label: str, maximum: int) -> int:
    return _integer(value, label, -maximum, maximum)


def adaptive_initiative_cooldown(
    relationship: Relationship,
    *,
    now: datetime | None = None,
) -> timedelta:
    """Calculate an explainable 2–72 hour initiative cooldown.

    Closeness and reciprocity shorten it, tension lengthens it, and a stale
    relationship gradually becomes safer to revisit.  The final hard bounds
    are part of the public policy and are enforced even for corrupt inputs.
    """

    if not isinstance(relationship, Relationship):
        raise TypeError("relationship должен быть Relationship")
    current = _utc_datetime(now or _now())
    closeness = min(100, max(0, relationship.closeness))
    reciprocity = min(100, max(0, relationship.reciprocity))
    tension = min(100, max(0, relationship.tension))
    affinity = (closeness + reciprocity) / 200.0
    hours = 72.0 - 60.0 * affinity + 36.0 * (tension / 100.0)
    if relationship.last_interaction_at is not None:
        age_days = max(
            0.0,
            (current - _utc_datetime(relationship.last_interaction_at)).total_seconds()
            / 86_400.0,
        )
        hours -= min(10.0, age_days)
    hours = min(
        float(MAX_INITIATIVE_COOLDOWN_HOURS),
        max(float(MIN_INITIATIVE_COOLDOWN_HOURS), hours),
    )
    return timedelta(seconds=round(hours * 3_600))


# A discoverable alternative name for callers that group policy helpers.
calculate_adaptive_cooldown = adaptive_initiative_cooldown


def initiative_allowed(
    relationship: Relationship,
    *,
    now: datetime | None = None,
    sleeping: bool = False,
) -> bool:
    if sleeping or relationship.blocked or relationship.awaiting_reply:
        return False
    if relationship.last_initiative_at is None:
        return True
    current = _utc_datetime(now or _now())
    return current >= (
        _utc_datetime(relationship.last_initiative_at)
        + adaptive_initiative_cooldown(relationship, now=current)
    )


class MilanaStateStore:
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
            self._create_schema()

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

    def _create_schema(self) -> None:
        # executescript normally commits before execution.  Keeping BEGIN and
        # COMMIT inside the script makes this additive migration atomic.
        script = """
        BEGIN IMMEDIATE;

        CREATE TABLE IF NOT EXISTS agent_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
            mood TEXT NOT NULL DEFAULT 'спокойное',
            valence INTEGER NOT NULL DEFAULT 0 CHECK (valence BETWEEN -100 AND 100),
            arousal INTEGER NOT NULL DEFAULT 50 CHECK (arousal BETWEEN 0 AND 100),
            social_need INTEGER NOT NULL DEFAULT 50 CHECK (social_need BETWEEN 0 AND 100),
            rest_need INTEGER NOT NULL DEFAULT 50 CHECK (rest_need BETWEEN 0 AND 100),
            novelty_need INTEGER NOT NULL DEFAULT 50 CHECK (novelty_need BETWEEN 0 AND 100),
            achievement_need INTEGER NOT NULL DEFAULT 50 CHECK (achievement_need BETWEEN 0 AND 100),
            current_intention TEXT,
            last_heartbeat_at TEXT,
            next_heartbeat_at TEXT,
            heartbeat_paused INTEGER NOT NULL DEFAULT 0 CHECK (heartbeat_paused IN (0, 1)),
            last_service_seen_at TEXT,
            recovery_pending_from TEXT,
            recovery_pending_to TEXT,
            recovery_completed_through TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS heartbeat_jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            due_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
            payload_json TEXT NOT NULL DEFAULT '{}',
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            idempotency_key TEXT UNIQUE,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_heartbeat_jobs_status_due
            ON heartbeat_jobs(status, due_at);

        CREATE TABLE IF NOT EXISTS world_entities (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            is_real INTEGER NOT NULL DEFAULT 0 CHECK (is_real IN (0, 1)),
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'archived')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_world_entities_status_kind
            ON world_entities(status, kind);

        CREATE TABLE IF NOT EXISTS world_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT NOT NULL REFERENCES world_entities(id) ON DELETE CASCADE,
            fact_key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            locked INTEGER NOT NULL DEFAULT 0 CHECK (locked IN (0, 1)),
            version INTEGER NOT NULL CHECK (version >= 1),
            source TEXT,
            valid_from TEXT NOT NULL,
            superseded_at TEXT,
            UNIQUE (entity_id, fact_key, version)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_world_facts_current
            ON world_facts(entity_id, fact_key)
            WHERE superseded_at IS NULL;

        CREATE TABLE IF NOT EXISTS life_events (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            importance INTEGER NOT NULL DEFAULT 50 CHECK (importance BETWEEN 0 AND 100),
            entity_ids_json TEXT NOT NULL DEFAULT '[]',
            happened_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'archived')),
            raw_payload_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_life_events_status_happened
            ON life_events(status, happened_at DESC);

        CREATE TABLE IF NOT EXISTS goals (
            id TEXT PRIMARY KEY,
            horizon TEXT NOT NULL CHECK (horizon IN ('short', 'long')),
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active', 'completed', 'archived')),
            progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_goals_status_horizon
            ON goals(status, horizon, updated_at DESC);

        CREATE TABLE IF NOT EXISTS relationships (
            entity_id TEXT PRIMARY KEY REFERENCES world_entities(id) ON DELETE CASCADE,
            closeness INTEGER NOT NULL DEFAULT 50 CHECK (closeness BETWEEN 0 AND 100),
            reciprocity INTEGER NOT NULL DEFAULT 50 CHECK (reciprocity BETWEEN 0 AND 100),
            tension INTEGER NOT NULL DEFAULT 0 CHECK (tension BETWEEN 0 AND 100),
            awaiting_reply INTEGER NOT NULL DEFAULT 0 CHECK (awaiting_reply IN (0, 1)),
            blocked INTEGER NOT NULL DEFAULT 0 CHECK (blocked IN (0, 1)),
            last_interaction_at TEXT,
            last_initiative_at TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS world_summaries (
            id TEXT PRIMARY KEY,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (period_start, period_end),
            CHECK (period_end > period_start)
        );
        CREATE INDEX IF NOT EXISTS idx_world_summaries_period
            ON world_summaries(period_end DESC);

        CREATE TABLE IF NOT EXISTS skill_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            action TEXT NOT NULL,
            success INTEGER NOT NULL CHECK (success IN (0, 1)),
            detail_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_skill_audit_turn_created
            ON skill_audit(turn_id, created_at);

        CREATE TABLE IF NOT EXISTS telegram_notice_journal (
            notice_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'handled')),
            received_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            handled_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_telegram_notice_journal_status_received
            ON telegram_notice_journal(status, received_at);

        INSERT OR IGNORE INTO agent_state (id, updated_at)
        VALUES (1, strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'));

        COMMIT;
        """
        try:
            self._connection.executescript(script)
        except Exception:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def record_telegram_notice(
        self,
        payload: Mapping[str, Any],
        *,
        received_at: datetime | None = None,
    ) -> str:
        """Durably accept a metadata-only Telegram notice before scheduling it."""

        if not isinstance(payload, Mapping):
            raise TypeError("Telegram notice payload должен быть mapping")
        notice_id = _identifier(payload.get("notice_id"), "Telegram notice ID")
        serialized = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
        timestamp = _timestamp(received_at or _now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT status FROM telegram_notice_journal WHERE notice_id = ?",
                    (notice_id,),
                ).fetchone()
                if row is not None and row["status"] == "handled":
                    self._connection.commit()
                    return "handled"
                if row is None:
                    self._connection.execute(
                        """
                        INSERT INTO telegram_notice_journal (
                            notice_id, payload_json, status, received_at, updated_at
                        ) VALUES (?, ?, 'pending', ?, ?)
                        """,
                        (notice_id, serialized, timestamp, timestamp),
                    )
                    result = "created"
                else:
                    self._connection.execute(
                        """
                        UPDATE telegram_notice_journal
                        SET payload_json = ?, updated_at = ?
                        WHERE notice_id = ? AND status = 'pending'
                        """,
                        (serialized, timestamp, notice_id),
                    )
                    result = "pending"
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def list_pending_telegram_notices(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        limit = _integer(limit, "Лимит Telegram notices", 1, 10_000)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload_json FROM telegram_notice_journal
                WHERE status = 'pending'
                ORDER BY received_at ASC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        notices: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            if isinstance(payload, dict):
                notices.append(payload)
        return notices

    def complete_telegram_notices(
        self,
        notice_ids: Iterable[str],
        *,
        handled_at: datetime | None = None,
    ) -> int:
        normalized = tuple(
            dict.fromkeys(_identifier(item, "Telegram notice ID") for item in notice_ids)
        )
        if not normalized:
            return 0
        timestamp = _timestamp(handled_at or _now())
        placeholders = ", ".join("?" for _ in normalized)
        with self._lock:
            cursor = self._connection.execute(
                f"""
                UPDATE telegram_notice_journal
                SET status = 'handled', handled_at = ?, updated_at = ?
                WHERE notice_id IN ({placeholders}) AND status = 'pending'
                """,
                (timestamp, timestamp, *normalized),
            )
            self._connection.commit()
            return int(cursor.rowcount)

    def __enter__(self) -> "MilanaStateStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _agent_state_from_row(row: sqlite3.Row) -> AgentState:
        return AgentState(
            revision=int(row["revision"]),
            mood=str(row["mood"]),
            valence=int(row["valence"]),
            arousal=int(row["arousal"]),
            social=int(row["social_need"]),
            rest=int(row["rest_need"]),
            novelty=int(row["novelty_need"]),
            achievement=int(row["achievement_need"]),
            current_intention=(
                str(row["current_intention"])
                if row["current_intention"] is not None
                else None
            ),
            last_heartbeat_at=_parse_timestamp(row["last_heartbeat_at"]),
            next_heartbeat_at=_parse_timestamp(row["next_heartbeat_at"]),
            heartbeat_paused=bool(row["heartbeat_paused"]),
            last_service_seen_at=_parse_timestamp(row["last_service_seen_at"]),
            recovery_pending_from=_parse_timestamp(row["recovery_pending_from"]),
            recovery_pending_to=_parse_timestamp(row["recovery_pending_to"]),
            recovery_completed_through=_parse_timestamp(
                row["recovery_completed_through"]
            ),
            updated_at=_parse_timestamp(row["updated_at"]) or _now(),
        )

    def get_agent_state(self) -> AgentState:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agent_state WHERE id = 1"
            ).fetchone()
        assert row is not None
        return self._agent_state_from_row(row)

    def update_agent_state(
        self,
        *,
        mood: str | None = None,
        valence: int | None = None,
        arousal: int | None = None,
        current_intention: str | None = None,
        clear_intention: bool = False,
        expected_revision: int | None = None,
        at: datetime | None = None,
    ) -> AgentState:
        if mood is not None:
            mood = _clean_text(mood, "Настроение", maximum=120)
        if valence is not None:
            valence = _integer(valence, "Valence", -100, 100)
        if arousal is not None:
            arousal = _integer(arousal, "Arousal", 0, 100)
        if clear_intention and current_intention is not None:
            raise ValueError("Нельзя одновременно задать и очистить намерение")
        if current_intention is not None:
            current_intention = _clean_text(
                current_intention,
                "Намерение",
                maximum=1_000,
            )
        changes: dict[str, Any] = {}
        if mood is not None:
            changes["mood"] = mood
        if valence is not None:
            changes["valence"] = valence
        if arousal is not None:
            changes["arousal"] = arousal
        if current_intention is not None or clear_intention:
            changes["current_intention"] = current_intention
        if not changes:
            return self.get_agent_state()
        return self._update_agent_columns(
            changes,
            expected_revision=expected_revision,
            at=at,
        )

    def _update_agent_columns(
        self,
        changes: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
        at: datetime | None = None,
        increment_revision: bool = True,
    ) -> AgentState:
        changed_at = _timestamp(at or _now())
        assignments = [f"{column} = ?" for column in changes]
        values = list(changes.values())
        if increment_revision:
            assignments.append("revision = revision + 1")
        assignments.append("updated_at = ?")
        values.append(changed_at)
        where = "id = 1"
        if expected_revision is not None:
            if (
                isinstance(expected_revision, bool)
                or not isinstance(expected_revision, int)
                or expected_revision < 0
            ):
                raise ValueError("expected_revision должен быть неотрицательным целым")
            where += " AND revision = ?"
            values.append(expected_revision)
        with self._lock:
            cursor = self._connection.execute(
                f"UPDATE agent_state SET {', '.join(assignments)} WHERE {where}",
                values,
            )
            if cursor.rowcount != 1:
                self._connection.rollback()
                raise StateConflictError("Состояние Миланы уже изменилось")
            self._connection.commit()
        return self.get_agent_state()

    def apply_need_deltas(
        self,
        deltas: Mapping[str, int],
        *,
        expected_revision: int | None = None,
        at: datetime | None = None,
    ) -> AgentState:
        if not isinstance(deltas, Mapping):
            raise TypeError("deltas должен быть объектом")
        unknown = set(deltas) - set(NEED_NAMES)
        if unknown:
            raise ValueError("Неизвестные потребности: " + ", ".join(sorted(unknown)))
        validated = {
            name: _bounded_delta(value, f"Изменение {name}", MAX_NEED_DELTA)
            for name, value in deltas.items()
        }
        if not validated:
            return self.get_agent_state()
        state = self.get_agent_state()
        changes = {
            f"{name}_need": min(100, max(0, state.needs[name] + delta))
            for name, delta in validated.items()
        }
        return self._update_agent_columns(
            changes,
            expected_revision=(
                state.revision if expected_revision is None else expected_revision
            ),
            at=at,
        )

    def set_heartbeat_paused(
        self,
        paused: bool,
        *,
        at: datetime | None = None,
    ) -> AgentState:
        if not isinstance(paused, bool):
            raise TypeError("paused должен быть bool")
        return self._update_agent_columns(
            {"heartbeat_paused": int(paused)},
            at=at,
            increment_revision=False,
        )

    def set_next_heartbeat(
        self,
        value: datetime | None,
        *,
        at: datetime | None = None,
    ) -> AgentState:
        timestamp = _timestamp(value) if value is not None else None
        return self._update_agent_columns(
            {"next_heartbeat_at": timestamp},
            at=at,
            increment_revision=False,
        )

    def record_heartbeat(
        self,
        *,
        completed_at: datetime,
        next_at: datetime | None,
    ) -> AgentState:
        completed = _utc_datetime(completed_at)
        if next_at is not None and _utc_datetime(next_at) <= completed:
            raise ValueError("Следующий heartbeat должен быть позже завершённого")
        return self._update_agent_columns(
            {
                "last_heartbeat_at": _timestamp(completed),
                "next_heartbeat_at": _timestamp(next_at) if next_at else None,
            },
            at=completed,
            increment_revision=False,
        )

    def touch_service(self, at: datetime | None = None) -> None:
        current = _utc_datetime(at or _now())
        self._update_agent_columns(
            {"last_service_seen_at": _timestamp(current)},
            at=current,
            increment_revision=False,
        )

    def begin_recovery(
        self,
        at: datetime | None = None,
        *,
        minimum_gap: timedelta = timedelta(minutes=5),
    ) -> RecoveryWindow | None:
        current = _utc_datetime(at or _now())
        if not isinstance(minimum_gap, timedelta) or minimum_gap.total_seconds() < 0:
            raise ValueError("minimum_gap должен быть неотрицательным timedelta")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM agent_state WHERE id = 1"
                ).fetchone()
                assert row is not None
                pending_from = _parse_timestamp(row["recovery_pending_from"])
                pending_to = _parse_timestamp(row["recovery_pending_to"])
                if pending_from is not None and pending_to is not None:
                    self._connection.execute(
                        "UPDATE agent_state SET last_service_seen_at = ?, updated_at = ? WHERE id = 1",
                        (_timestamp(current), _timestamp(current)),
                    )
                    self._connection.commit()
                    return RecoveryWindow(pending_from, pending_to)

                previous = _parse_timestamp(row["last_service_seen_at"])
                values: list[Any] = [_timestamp(current), _timestamp(current)]
                recovery: RecoveryWindow | None = None
                if previous is not None and current - previous >= minimum_gap:
                    recovery = RecoveryWindow(previous, current)
                    self._connection.execute(
                        """
                        UPDATE agent_state
                        SET last_service_seen_at = ?, recovery_pending_from = ?,
                            recovery_pending_to = ?, updated_at = ?
                        WHERE id = 1
                        """,
                        (
                            _timestamp(current),
                            _timestamp(previous),
                            _timestamp(current),
                            _timestamp(current),
                        ),
                    )
                else:
                    self._connection.execute(
                        "UPDATE agent_state SET last_service_seen_at = ?, updated_at = ? WHERE id = 1",
                        values,
                    )
                self._connection.commit()
                return recovery
            except Exception:
                self._connection.rollback()
                raise

    # Service-oriented synonym used by the entrypoint.
    record_service_start = begin_recovery

    def get_pending_recovery(self) -> RecoveryWindow | None:
        state = self.get_agent_state()
        if (
            state.recovery_pending_from is None
            or state.recovery_pending_to is None
        ):
            return None
        return RecoveryWindow(
            state.recovery_pending_from,
            state.recovery_pending_to,
        )

    def complete_recovery(
        self,
        window: RecoveryWindow | datetime,
        *,
        at: datetime | None = None,
    ) -> bool:
        through = window.ended_at if isinstance(window, RecoveryWindow) else window
        through_timestamp = _timestamp(through)
        changed_at = _timestamp(at or _now())
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE agent_state
                SET recovery_completed_through = ?, recovery_pending_from = NULL,
                    recovery_pending_to = NULL, updated_at = ?
                WHERE id = 1 AND recovery_pending_to = ?
                """,
                (through_timestamp, changed_at, through_timestamp),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    @staticmethod
    def _heartbeat_job_from_row(row: sqlite3.Row) -> HeartbeatJob:
        return HeartbeatJob(
            id=str(row["id"]),
            kind=str(row["kind"]),
            due_at=_parse_timestamp(row["due_at"]) or _now(),
            status=str(row["status"]),
            payload=_json_load(str(row["payload_json"])),
            attempts=int(row["attempts"]),
            idempotency_key=(
                str(row["idempotency_key"])
                if row["idempotency_key"] is not None
                else None
            ),
            last_error=(
                str(row["last_error"]) if row["last_error"] is not None else None
            ),
            created_at=_parse_timestamp(row["created_at"]) or _now(),
            updated_at=_parse_timestamp(row["updated_at"]) or _now(),
            completed_at=_parse_timestamp(row["completed_at"]),
        )

    def schedule_heartbeat_job(
        self,
        kind: str,
        due_at: datetime,
        *,
        payload: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        job_id: str | None = None,
    ) -> HeartbeatJob:
        clean_kind = _clean_text(kind, "Тип heartbeat-задачи", maximum=100)
        due_timestamp = _timestamp(due_at)
        payload_json = _json_dump(dict(payload or {}))
        clean_key = (
            _clean_text(idempotency_key, "Idempotency key", maximum=255)
            if idempotency_key is not None
            else None
        )
        clean_id = _identifier(job_id, "ID heartbeat-задачи") if job_id else uuid4().hex
        created = _timestamp(_now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if clean_key is not None:
                    row = self._connection.execute(
                        "SELECT * FROM heartbeat_jobs WHERE idempotency_key = ?",
                        (clean_key,),
                    ).fetchone()
                    if row is not None:
                        self._connection.commit()
                        return self._heartbeat_job_from_row(row)
                self._connection.execute(
                    """
                    INSERT INTO heartbeat_jobs (
                        id, kind, due_at, status, payload_json, attempts,
                        idempotency_key, created_at, updated_at
                    ) VALUES (?, ?, ?, 'pending', ?, 0, ?, ?, ?)
                    """,
                    (
                        clean_id,
                        clean_kind,
                        due_timestamp,
                        payload_json,
                        clean_key,
                        created,
                        created,
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM heartbeat_jobs WHERE id = ?",
                    (clean_id,),
                ).fetchone()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        assert row is not None
        return self._heartbeat_job_from_row(row)

    def claim_due_heartbeat_jobs(
        self,
        now: datetime,
        *,
        limit: int = 20,
        lease_seconds: int = 300,
    ) -> list[HeartbeatJob]:
        limit = _integer(limit, "Лимит задач", 1, 1_000)
        lease_seconds = _integer(lease_seconds, "Срок аренды", 1, 86_400)
        current = _utc_datetime(now)
        current_timestamp = _timestamp(current)
        expired_timestamp = _timestamp(current - timedelta(seconds=lease_seconds))
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    """
                    UPDATE heartbeat_jobs
                    SET status = 'pending', updated_at = ?
                    WHERE status = 'running' AND updated_at <= ?
                    """,
                    (current_timestamp, expired_timestamp),
                )
                rows = self._connection.execute(
                    """
                    SELECT id FROM heartbeat_jobs
                    WHERE status = 'pending' AND due_at <= ?
                    ORDER BY due_at ASC, created_at ASC
                    LIMIT ?
                    """,
                    (current_timestamp, limit),
                ).fetchall()
                ids = [str(row["id"]) for row in rows]
                if ids:
                    placeholders = ", ".join("?" for _ in ids)
                    self._connection.execute(
                        f"""
                        UPDATE heartbeat_jobs
                        SET status = 'running', attempts = attempts + 1,
                            updated_at = ?
                        WHERE id IN ({placeholders}) AND status = 'pending'
                        """,
                        (current_timestamp, *ids),
                    )
                    claimed = self._connection.execute(
                        f"SELECT * FROM heartbeat_jobs WHERE id IN ({placeholders})",
                        ids,
                    ).fetchall()
                else:
                    claimed = []
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        by_id = {
            str(row["id"]): self._heartbeat_job_from_row(row) for row in claimed
        }
        return [by_id[job_id] for job_id in ids if job_id in by_id]

    def complete_heartbeat_job(
        self,
        job_id: str,
        *,
        completed_at: datetime | None = None,
    ) -> bool:
        completed = _timestamp(completed_at or _now())
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE heartbeat_jobs
                SET status = 'completed', completed_at = ?, updated_at = ?,
                    last_error = NULL
                WHERE id = ? AND status = 'running'
                """,
                (completed, completed, _identifier(job_id)),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def retry_heartbeat_job(
        self,
        job_id: str,
        *,
        error: str,
        retry_at: datetime,
        max_attempts: int = 5,
    ) -> bool:
        clean_error = _clean_text(error, "Ошибка heartbeat", maximum=2_000)
        max_attempts = _integer(max_attempts, "Максимум попыток", 1, 100)
        retry_timestamp = _timestamp(retry_at)
        with self._lock:
            row = self._connection.execute(
                "SELECT attempts, status FROM heartbeat_jobs WHERE id = ?",
                (_identifier(job_id),),
            ).fetchone()
            if row is None or row["status"] != "running":
                return False
            failed = int(row["attempts"]) >= max_attempts
            cursor = self._connection.execute(
                """
                UPDATE heartbeat_jobs
                SET status = ?, due_at = ?, last_error = ?, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    "failed" if failed else "pending",
                    retry_timestamp,
                    clean_error,
                    _timestamp(_now()),
                    job_id,
                ),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def reschedule_heartbeat_job(
        self,
        job_id: str,
        due_at: datetime,
        *,
        preserve_attempt: bool = True,
    ) -> bool:
        """Return a claimed job to pending, normally without spending an attempt.

        This is used when a reflective wake lands inside sleep or while the
        heartbeat is paused; neither situation is a delivery failure.
        """

        if not isinstance(preserve_attempt, bool):
            raise TypeError("preserve_attempt должен быть bool")
        attempt_sql = "attempts = MAX(0, attempts - 1)," if preserve_attempt else ""
        with self._lock:
            cursor = self._connection.execute(
                f"""
                UPDATE heartbeat_jobs
                SET status = 'pending', due_at = ?, {attempt_sql}
                    updated_at = ?, last_error = NULL
                WHERE id = ? AND status = 'running'
                """,
                (
                    _timestamp(due_at),
                    _timestamp(_now()),
                    _identifier(job_id),
                ),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def cancel_heartbeat_job(self, job_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE heartbeat_jobs
                SET status = 'cancelled', updated_at = ?
                WHERE id = ? AND status IN ('pending', 'running')
                """,
                (_timestamp(_now()), _identifier(job_id)),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def cancel_stale_heartbeat_jobs(
        self,
        through: datetime,
        *,
        kinds: Iterable[str],
    ) -> int:
        """Cancel missed reflective jobs already summarized by recovery."""

        normalized = tuple(
            dict.fromkeys(
                _clean_text(kind, "Тип heartbeat-задачи", maximum=100)
                for kind in kinds
            )
        )
        if not normalized:
            return 0
        placeholders = ", ".join("?" for _ in normalized)
        changed_at = _timestamp(_now())
        with self._lock:
            cursor = self._connection.execute(
                f"""
                UPDATE heartbeat_jobs
                SET status = 'cancelled', updated_at = ?
                WHERE status IN ('pending', 'running')
                  AND due_at <= ? AND kind IN ({placeholders})
                """,
                (changed_at, _timestamp(through), *normalized),
            )
            self._connection.commit()
            return int(cursor.rowcount)

    def list_heartbeat_jobs(
        self,
        *,
        statuses: Iterable[str] | None = None,
        limit: int = 100,
    ) -> list[HeartbeatJob]:
        limit = _integer(limit, "Лимит задач", 1, 10_000)
        parameters: list[Any] = []
        where = ""
        if statuses is not None:
            values = tuple(dict.fromkeys(_identifier(value, "Статус") for value in statuses))
            if not values:
                return []
            placeholders = ", ".join("?" for _ in values)
            where = f"WHERE status IN ({placeholders})"
            parameters.extend(values)
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM heartbeat_jobs {where}
                ORDER BY due_at ASC, created_at ASC LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._heartbeat_job_from_row(row) for row in rows]

    def next_heartbeat_job_due_at(self) -> datetime | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT due_at FROM heartbeat_jobs
                WHERE status = 'pending' ORDER BY due_at ASC LIMIT 1
                """
            ).fetchone()
        return _parse_timestamp(row["due_at"]) if row is not None else None

    @staticmethod
    def _entity_from_row(row: sqlite3.Row) -> WorldEntity:
        return WorldEntity(
            id=str(row["id"]),
            kind=str(row["kind"]),
            name=str(row["name"]),
            description=str(row["description"]),
            is_real=bool(row["is_real"]),
            status=str(row["status"]),
            created_at=_parse_timestamp(row["created_at"]) or _now(),
            updated_at=_parse_timestamp(row["updated_at"]) or _now(),
        )

    @staticmethod
    def _fact_from_row(row: sqlite3.Row) -> WorldFact:
        return WorldFact(
            id=int(row["id"]),
            entity_id=str(row["entity_id"]),
            key=str(row["fact_key"]),
            value=_json_load(str(row["value_json"])),
            locked=bool(row["locked"]),
            version=int(row["version"]),
            source=str(row["source"]) if row["source"] is not None else None,
            valid_from=_parse_timestamp(row["valid_from"]) or _now(),
            superseded_at=_parse_timestamp(row["superseded_at"]),
        )

    def _insert_entity(
        self,
        entity: NewEntity,
        *,
        at: datetime,
    ) -> WorldEntity:
        if not isinstance(entity, NewEntity):
            raise TypeError("entity должен быть NewEntity")
        kind = _clean_text(entity.kind, "Тип сущности", maximum=100)
        name = _clean_text(entity.name, "Имя сущности", maximum=255)
        description = _clean_text(
            entity.description,
            "Описание сущности",
            maximum=4_000,
            allow_empty=True,
        )
        if not isinstance(entity.is_real, bool):
            raise TypeError("is_real должен быть bool")
        entity_id = (
            _identifier(entity.entity_id, "ID сущности")
            if entity.entity_id is not None
            else uuid4().hex
        )
        timestamp = _timestamp(at)
        self._connection.execute(
            """
            INSERT INTO world_entities (
                id, kind, name, description, is_real, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (
                entity_id,
                kind,
                name,
                description,
                int(entity.is_real),
                timestamp,
                timestamp,
            ),
        )
        for fact in entity.facts:
            self._set_fact(entity_id, fact, at=at)
        row = self._connection.execute(
            "SELECT * FROM world_entities WHERE id = ?",
            (entity_id,),
        ).fetchone()
        assert row is not None
        return self._entity_from_row(row)

    def _apply_heartbeat_entity(
        self,
        entity: NewEntity,
        *,
        at: datetime,
    ) -> WorldEntity:
        """Create a new entity or version facts of an existing stable ID."""

        if entity.entity_id is not None:
            entity_id = _identifier(entity.entity_id, "ID сущности")
            existing = self._connection.execute(
                "SELECT * FROM world_entities WHERE id = ?", (entity_id,)
            ).fetchone()
            if existing is not None:
                for fact in entity.facts:
                    # Model-originated facts always arrive unlocked; _set_fact
                    # rejects conflicts with persona/world seed facts.
                    self._set_fact(entity_id, fact, at=at)
                self._connection.execute(
                    "UPDATE world_entities SET updated_at = ? WHERE id = ?",
                    (_timestamp(at), entity_id),
                )
                updated = self._connection.execute(
                    "SELECT * FROM world_entities WHERE id = ?", (entity_id,)
                ).fetchone()
                assert updated is not None
                return self._entity_from_row(updated)
        return self._insert_entity(entity, at=at)

    def create_entity(
        self,
        kind: str,
        name: str,
        *,
        description: str = "",
        is_real: bool = False,
        entity_id: str | None = None,
        facts: Sequence[FactSeed] = (),
        at: datetime | None = None,
    ) -> WorldEntity:
        entity = NewEntity(
            kind=kind,
            name=name,
            description=description,
            is_real=is_real,
            entity_id=entity_id,
            facts=tuple(facts),
        )
        changed_at = _utc_datetime(at or _now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                result = self._insert_entity(entity, at=changed_at)
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def get_entity(self, entity_id: str) -> WorldEntity | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM world_entities WHERE id = ?",
                (_identifier(entity_id, "ID сущности"),),
            ).fetchone()
        return self._entity_from_row(row) if row is not None else None

    def list_entities(
        self,
        *,
        include_archived: bool = False,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[WorldEntity]:
        limit = _integer(limit, "Лимит сущностей", 1, 10_000)
        clauses: list[str] = []
        parameters: list[Any] = []
        if not include_archived:
            clauses.append("status = 'active'")
        if kind is not None:
            clauses.append("kind = ?")
            parameters.append(_clean_text(kind, "Тип сущности", maximum=100))
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM world_entities {where}
                ORDER BY updated_at DESC, id ASC LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._entity_from_row(row) for row in rows]

    def archive_entity(self, entity_id: str, *, at: datetime | None = None) -> bool:
        timestamp = _timestamp(at or _now())
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE world_entities SET status = 'archived', updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (timestamp, _identifier(entity_id, "ID сущности")),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def _set_fact(
        self,
        entity_id: str,
        fact: FactSeed,
        *,
        at: datetime,
    ) -> WorldFact:
        if not isinstance(fact, FactSeed):
            raise TypeError("fact должен быть FactSeed")
        clean_entity_id = _identifier(entity_id, "ID сущности")
        key = _clean_text(fact.key, "Ключ факта", maximum=120)
        value_json = _json_dump(fact.value)
        if not isinstance(fact.locked, bool):
            raise TypeError("locked должен быть bool")
        source = (
            _clean_text(fact.source, "Источник факта", maximum=255)
            if fact.source is not None
            else None
        )
        if self._connection.execute(
            "SELECT 1 FROM world_entities WHERE id = ?",
            (clean_entity_id,),
        ).fetchone() is None:
            raise KeyError(f"Сущность не найдена: {clean_entity_id}")
        current = self._connection.execute(
            """
            SELECT * FROM world_facts
            WHERE entity_id = ? AND fact_key = ? AND superseded_at IS NULL
            """,
            (clean_entity_id, key),
        ).fetchone()
        if current is not None:
            if str(current["value_json"]) == value_json:
                if fact.locked and not bool(current["locked"]):
                    self._connection.execute(
                        "UPDATE world_facts SET locked = 1 WHERE id = ?",
                        (int(current["id"]),),
                    )
                    current = self._connection.execute(
                        "SELECT * FROM world_facts WHERE id = ?",
                        (int(current["id"]),),
                    ).fetchone()
                assert current is not None
                return self._fact_from_row(current)
            if bool(current["locked"]):
                raise LockedFactError(
                    f"Факт {clean_entity_id}.{key} заблокирован и не может быть изменён"
                )
            version = int(current["version"]) + 1
            self._connection.execute(
                "UPDATE world_facts SET superseded_at = ? WHERE id = ?",
                (_timestamp(at), int(current["id"])),
            )
        else:
            version = 1
        cursor = self._connection.execute(
            """
            INSERT INTO world_facts (
                entity_id, fact_key, value_json, locked, version, source, valid_from
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_entity_id,
                key,
                value_json,
                int(fact.locked),
                version,
                source,
                _timestamp(at),
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM world_facts WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        assert row is not None
        return self._fact_from_row(row)

    def set_fact(
        self,
        entity_id: str,
        key: str,
        value: Any,
        *,
        locked: bool = False,
        source: str | None = None,
        at: datetime | None = None,
    ) -> WorldFact:
        changed_at = _utc_datetime(at or _now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                result = self._set_fact(
                    entity_id,
                    FactSeed(key, value, locked, source),
                    at=changed_at,
                )
                self._connection.execute(
                    "UPDATE world_entities SET updated_at = ? WHERE id = ?",
                    (_timestamp(changed_at), entity_id),
                )
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    set_world_fact = set_fact

    def seed_locked_facts(
        self,
        entity_id: str,
        facts: Mapping[str, Any],
        *,
        source: str = "persona",
        at: datetime | None = None,
    ) -> list[WorldFact]:
        if not isinstance(facts, Mapping):
            raise TypeError("facts должен быть объектом")
        changed_at = _utc_datetime(at or _now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                result = [
                    self._set_fact(
                        entity_id,
                        FactSeed(key, value, True, source),
                        at=changed_at,
                    )
                    for key, value in facts.items()
                ]
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def get_facts(
        self,
        entity_id: str,
        *,
        include_history: bool = False,
    ) -> list[WorldFact]:
        history_clause = "" if include_history else "AND superseded_at IS NULL"
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM world_facts
                WHERE entity_id = ? {history_clause}
                ORDER BY fact_key ASC, version ASC
                """,
                (_identifier(entity_id, "ID сущности"),),
            ).fetchall()
        return [self._fact_from_row(row) for row in rows]

    get_world_facts = get_facts

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> LifeEvent:
        payload = (
            _json_load(str(row["raw_payload_json"]))
            if row["raw_payload_json"] is not None
            else None
        )
        return LifeEvent(
            id=str(row["id"]),
            kind=str(row["kind"]),
            title=str(row["title"]),
            description=str(row["description"]),
            importance=int(row["importance"]),
            entity_ids=tuple(_json_load(str(row["entity_ids_json"]))),
            happened_at=_parse_timestamp(row["happened_at"]) or _now(),
            status=str(row["status"]),
            raw_payload=payload,
            created_at=_parse_timestamp(row["created_at"]) or _now(),
            updated_at=_parse_timestamp(row["updated_at"]) or _now(),
        )

    def _insert_life_event(
        self,
        event: NewLifeEvent,
        *,
        at: datetime,
    ) -> LifeEvent:
        if not isinstance(event, NewLifeEvent):
            raise TypeError("event должен быть NewLifeEvent")
        title = _clean_text(event.title, "Заголовок события", maximum=255)
        description = _clean_text(
            event.description,
            "Описание события",
            maximum=8_000,
        )
        kind = _clean_text(event.kind, "Тип события", maximum=100)
        importance = _integer(event.importance, "Важность события", 0, 100)
        entity_ids = tuple(_identifier(item, "ID связанной сущности") for item in event.entity_ids)
        for entity_id in entity_ids:
            if self._connection.execute(
                "SELECT 1 FROM world_entities WHERE id = ?", (entity_id,)
            ).fetchone() is None:
                raise KeyError(f"Сущность не найдена: {entity_id}")
        happened_at = _utc_datetime(event.happened_at or at)
        event_id = uuid4().hex
        timestamp = _timestamp(at)
        self._connection.execute(
            """
            INSERT INTO life_events (
                id, kind, title, description, importance, entity_ids_json,
                happened_at, status, raw_payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (
                event_id,
                kind,
                title,
                description,
                importance,
                _json_dump(entity_ids),
                _timestamp(happened_at),
                _json_dump(dict(event.raw_payload)) if event.raw_payload is not None else None,
                timestamp,
                timestamp,
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM life_events WHERE id = ?", (event_id,)
        ).fetchone()
        assert row is not None
        return self._event_from_row(row)

    def add_life_event(
        self,
        title: str,
        description: str,
        *,
        kind: str = "life",
        importance: int = 50,
        entity_ids: Sequence[str] = (),
        happened_at: datetime | None = None,
        raw_payload: Mapping[str, Any] | None = None,
        at: datetime | None = None,
    ) -> LifeEvent:
        changed_at = _utc_datetime(at or _now())
        event = NewLifeEvent(
            title=title,
            description=description,
            kind=kind,
            importance=importance,
            entity_ids=tuple(entity_ids),
            happened_at=happened_at,
            raw_payload=raw_payload,
        )
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                result = self._insert_life_event(event, at=changed_at)
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def list_life_events(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[LifeEvent]:
        limit = _integer(limit, "Лимит событий", 1, 10_000)
        where = "" if include_archived else "WHERE status = 'active'"
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM life_events {where}
                ORDER BY happened_at DESC, created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def archive_life_event(
        self,
        event_id: str,
        *,
        at: datetime | None = None,
    ) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE life_events SET status = 'archived', updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (_timestamp(at or _now()), _identifier(event_id, "ID события")),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    @staticmethod
    def _goal_from_row(row: sqlite3.Row) -> Goal:
        return Goal(
            id=str(row["id"]),
            horizon=str(row["horizon"]),
            title=str(row["title"]),
            description=str(row["description"]),
            status=str(row["status"]),
            progress=int(row["progress"]),
            created_at=_parse_timestamp(row["created_at"]) or _now(),
            updated_at=_parse_timestamp(row["updated_at"]) or _now(),
            completed_at=_parse_timestamp(row["completed_at"]),
        )

    def _active_goal_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS c FROM goals WHERE status = 'active'"
        ).fetchone()
        return int(row["c"]) if row is not None else 0

    def _insert_goal(
        self,
        *,
        title: str,
        description: str,
        horizon: str,
        progress: int,
        at: datetime,
        goal_id: str | None = None,
    ) -> Goal:
        if self._active_goal_count() >= MAX_ACTIVE_GOALS:
            raise GoalLimitError(
                f"Одновременно может быть не больше {MAX_ACTIVE_GOALS} активных целей"
            )
        clean_title = _clean_text(title, "Название цели", maximum=255)
        clean_description = _clean_text(
            description,
            "Описание цели",
            maximum=4_000,
            allow_empty=True,
        )
        if horizon not in {"short", "long"}:
            raise ValueError("Горизонт цели должен быть short или long")
        clean_progress = _integer(progress, "Прогресс цели", 0, 100)
        clean_id = _identifier(goal_id, "ID цели") if goal_id else uuid4().hex
        timestamp = _timestamp(at)
        self._connection.execute(
            """
            INSERT INTO goals (
                id, horizon, title, description, status, progress,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
            """,
            (
                clean_id,
                horizon,
                clean_title,
                clean_description,
                clean_progress,
                timestamp,
                timestamp,
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM goals WHERE id = ?", (clean_id,)
        ).fetchone()
        assert row is not None
        return self._goal_from_row(row)

    def create_goal(
        self,
        title: str,
        *,
        description: str = "",
        horizon: str = "short",
        progress: int = 0,
        goal_id: str | None = None,
        at: datetime | None = None,
    ) -> Goal:
        changed_at = _utc_datetime(at or _now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                goal = self._insert_goal(
                    title=title,
                    description=description,
                    horizon=horizon,
                    progress=progress,
                    at=changed_at,
                    goal_id=goal_id,
                )
                self._connection.commit()
                return goal
            except Exception:
                self._connection.rollback()
                raise

    def get_goal(self, goal_id: str) -> Goal | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM goals WHERE id = ?",
                (_identifier(goal_id, "ID цели"),),
            ).fetchone()
        return self._goal_from_row(row) if row is not None else None

    def list_goals(
        self,
        *,
        statuses: Iterable[str] | None = ("active",),
        limit: int = 100,
    ) -> list[Goal]:
        limit = _integer(limit, "Лимит целей", 1, 10_000)
        parameters: list[Any] = []
        where = ""
        if statuses is not None:
            values = tuple(dict.fromkeys(_identifier(item, "Статус цели") for item in statuses))
            if not values:
                return []
            invalid = set(values) - {"active", "completed", "archived"}
            if invalid:
                raise ValueError("Неизвестный статус цели: " + ", ".join(sorted(invalid)))
            placeholders = ", ".join("?" for _ in values)
            where = f"WHERE status IN ({placeholders})"
            parameters.extend(values)
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM goals {where}
                ORDER BY updated_at DESC, created_at DESC LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._goal_from_row(row) for row in rows]

    def _change_goal(self, change: GoalChange, *, at: datetime) -> Goal:
        if not isinstance(change, GoalChange):
            raise TypeError("Изменение цели должно быть GoalChange")
        if change.operation == "create":
            if change.title is None:
                raise ValueError("Для создания цели нужно название")
            return self._insert_goal(
                title=change.title,
                description=change.description,
                horizon=change.horizon or "short",
                progress=change.progress or 0,
                at=at,
                goal_id=change.goal_id,
            )
        if change.operation not in {"update", "complete", "archive"}:
            raise ValueError("Операция цели должна быть create/update/complete/archive")
        if change.goal_id is None:
            raise ValueError("Для изменения цели нужен goal_id")
        goal_id = _identifier(change.goal_id, "ID цели")
        row = self._connection.execute(
            "SELECT * FROM goals WHERE id = ?", (goal_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Цель не найдена: {goal_id}")
        if row["status"] != "active":
            raise ValueError("Завершённую или архивную цель нельзя изменять")
        assignments: dict[str, Any] = {}
        if change.title is not None:
            assignments["title"] = _clean_text(
                change.title, "Название цели", maximum=255
            )
        if change.description:
            assignments["description"] = _clean_text(
                change.description,
                "Описание цели",
                maximum=4_000,
                allow_empty=True,
            )
        if change.horizon is not None:
            if change.horizon not in {"short", "long"}:
                raise ValueError("Горизонт цели должен быть short или long")
            if change.operation == "update" and change.horizon != row["horizon"]:
                assignments["horizon"] = change.horizon
        if change.progress is not None:
            assignments["progress"] = _integer(
                change.progress, "Прогресс цели", 0, 100
            )
        if change.operation == "complete":
            assignments.update(
                status="completed",
                progress=100,
                completed_at=_timestamp(at),
            )
        elif change.operation == "archive":
            assignments.update(status="archived", completed_at=_timestamp(at))
        if not assignments:
            return self._goal_from_row(row)
        assignments["updated_at"] = _timestamp(at)
        clause = ", ".join(f"{column} = ?" for column in assignments)
        self._connection.execute(
            f"UPDATE goals SET {clause} WHERE id = ?",
            (*assignments.values(), goal_id),
        )
        updated = self._connection.execute(
            "SELECT * FROM goals WHERE id = ?", (goal_id,)
        ).fetchone()
        assert updated is not None
        return self._goal_from_row(updated)

    def change_goal(
        self,
        change: GoalChange,
        *,
        at: datetime | None = None,
    ) -> Goal:
        changed_at = _utc_datetime(at or _now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                goal = self._change_goal(change, at=changed_at)
                self._connection.commit()
                return goal
            except Exception:
                self._connection.rollback()
                raise

    def update_goal(
        self,
        goal_id: str,
        *,
        title: str | None = None,
        description: str = "",
        horizon: str | None = None,
        progress: int | None = None,
        at: datetime | None = None,
    ) -> Goal:
        return self.change_goal(
            GoalChange(
                "update",
                goal_id=goal_id,
                title=title,
                description=description,
                horizon=horizon,
                progress=progress,
            ),
            at=at,
        )

    def complete_goal(self, goal_id: str, *, at: datetime | None = None) -> Goal:
        return self.change_goal(GoalChange("complete", goal_id=goal_id), at=at)

    def archive_goal(self, goal_id: str, *, at: datetime | None = None) -> Goal:
        return self.change_goal(GoalChange("archive", goal_id=goal_id), at=at)

    @staticmethod
    def _relationship_from_row(row: sqlite3.Row) -> Relationship:
        return Relationship(
            entity_id=str(row["entity_id"]),
            closeness=int(row["closeness"]),
            reciprocity=int(row["reciprocity"]),
            tension=int(row["tension"]),
            awaiting_reply=bool(row["awaiting_reply"]),
            blocked=bool(row["blocked"]),
            last_interaction_at=_parse_timestamp(row["last_interaction_at"]),
            last_initiative_at=_parse_timestamp(row["last_initiative_at"]),
            updated_at=_parse_timestamp(row["updated_at"]) or _now(),
        )

    def upsert_relationship(
        self,
        entity_id: str,
        *,
        closeness: int = 50,
        reciprocity: int = 50,
        tension: int = 0,
        awaiting_reply: bool = False,
        blocked: bool = False,
        last_interaction_at: datetime | None = None,
        at: datetime | None = None,
    ) -> Relationship:
        clean_id = _identifier(entity_id, "ID сущности")
        closeness = _integer(closeness, "Близость", 0, 100)
        reciprocity = _integer(reciprocity, "Взаимность", 0, 100)
        tension = _integer(tension, "Напряжение", 0, 100)
        if not isinstance(awaiting_reply, bool) or not isinstance(blocked, bool):
            raise TypeError("awaiting_reply и blocked должны быть bool")
        timestamp = _timestamp(at or _now())
        interaction = (
            _timestamp(last_interaction_at) if last_interaction_at is not None else None
        )
        with self._lock:
            if self._connection.execute(
                "SELECT 1 FROM world_entities WHERE id = ?", (clean_id,)
            ).fetchone() is None:
                raise KeyError(f"Сущность не найдена: {clean_id}")
            self._connection.execute(
                """
                INSERT INTO relationships (
                    entity_id, closeness, reciprocity, tension, awaiting_reply,
                    blocked, last_interaction_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    closeness = excluded.closeness,
                    reciprocity = excluded.reciprocity,
                    tension = excluded.tension,
                    awaiting_reply = excluded.awaiting_reply,
                    blocked = excluded.blocked,
                    last_interaction_at = COALESCE(
                        excluded.last_interaction_at, relationships.last_interaction_at
                    ),
                    updated_at = excluded.updated_at
                """,
                (
                    clean_id,
                    closeness,
                    reciprocity,
                    tension,
                    int(awaiting_reply),
                    int(blocked),
                    interaction,
                    timestamp,
                ),
            )
            self._connection.commit()
            row = self._connection.execute(
                "SELECT * FROM relationships WHERE entity_id = ?", (clean_id,)
            ).fetchone()
        assert row is not None
        return self._relationship_from_row(row)

    def get_relationship(self, entity_id: str) -> Relationship | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM relationships WHERE entity_id = ?",
                (_identifier(entity_id, "ID сущности"),),
            ).fetchone()
        return self._relationship_from_row(row) if row is not None else None

    def list_relationships(self, *, limit: int = 100) -> list[Relationship]:
        limit = _integer(limit, "Лимит отношений", 1, 10_000)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM relationships
                ORDER BY updated_at DESC, entity_id ASC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._relationship_from_row(row) for row in rows]

    def _adjust_relationship(
        self,
        delta: RelationshipDelta,
        *,
        at: datetime,
    ) -> Relationship:
        if not isinstance(delta, RelationshipDelta):
            raise TypeError("Изменение отношений должно быть RelationshipDelta")
        entity_id = _identifier(delta.entity_id, "ID сущности")
        values = {
            "closeness": _bounded_delta(
                delta.closeness, "Изменение близости", MAX_RELATIONSHIP_DELTA
            ),
            "reciprocity": _bounded_delta(
                delta.reciprocity, "Изменение взаимности", MAX_RELATIONSHIP_DELTA
            ),
            "tension": _bounded_delta(
                delta.tension, "Изменение напряжения", MAX_RELATIONSHIP_DELTA
            ),
        }
        if delta.awaiting_reply is not None and not isinstance(delta.awaiting_reply, bool):
            raise TypeError("awaiting_reply должен быть bool")
        if delta.blocked is not None and not isinstance(delta.blocked, bool):
            raise TypeError("blocked должен быть bool")
        row = self._connection.execute(
            "SELECT * FROM relationships WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            if self._connection.execute(
                "SELECT 1 FROM world_entities WHERE id = ?", (entity_id,)
            ).fetchone() is None:
                raise KeyError(f"Сущность не найдена: {entity_id}")
            self._connection.execute(
                """
                INSERT INTO relationships (
                    entity_id, closeness, reciprocity, tension, awaiting_reply,
                    blocked, last_interaction_at, updated_at
                ) VALUES (?, 50, 50, 0, 0, 0, NULL, ?)
                """,
                (entity_id, _timestamp(at)),
            )
            row = self._connection.execute(
                "SELECT * FROM relationships WHERE entity_id = ?", (entity_id,)
            ).fetchone()
        assert row is not None
        assignments: dict[str, Any] = {
            name: min(100, max(0, int(row[name]) + change))
            for name, change in values.items()
        }
        if delta.awaiting_reply is not None:
            assignments["awaiting_reply"] = int(delta.awaiting_reply)
        if delta.blocked is not None:
            assignments["blocked"] = int(delta.blocked)
        if delta.interacted_at is not None:
            assignments["last_interaction_at"] = _timestamp(delta.interacted_at)
        assignments["updated_at"] = _timestamp(at)
        clause = ", ".join(f"{column} = ?" for column in assignments)
        self._connection.execute(
            f"UPDATE relationships SET {clause} WHERE entity_id = ?",
            (*assignments.values(), entity_id),
        )
        updated = self._connection.execute(
            "SELECT * FROM relationships WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        assert updated is not None
        return self._relationship_from_row(updated)

    def adjust_relationship(
        self,
        entity_id: str,
        *,
        closeness: int = 0,
        reciprocity: int = 0,
        tension: int = 0,
        awaiting_reply: bool | None = None,
        blocked: bool | None = None,
        interacted_at: datetime | None = None,
        at: datetime | None = None,
    ) -> Relationship:
        changed_at = _utc_datetime(at or _now())
        delta = RelationshipDelta(
            entity_id=entity_id,
            closeness=closeness,
            reciprocity=reciprocity,
            tension=tension,
            awaiting_reply=awaiting_reply,
            blocked=blocked,
            interacted_at=interacted_at,
        )
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                result = self._adjust_relationship(delta, at=changed_at)
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def mark_initiative(
        self,
        entity_id: str,
        *,
        at: datetime | None = None,
    ) -> Relationship:
        current = _utc_datetime(at or _now())
        relationship = self.get_relationship(entity_id)
        if relationship is None:
            relationship = self.upsert_relationship(entity_id, at=current)
        if not initiative_allowed(relationship, now=current):
            raise ValueError("Инициативный контакт сейчас запрещён политикой отношений")
        with self._lock:
            self._connection.execute(
                """
                UPDATE relationships
                SET awaiting_reply = 1, last_initiative_at = ?,
                    last_interaction_at = ?, updated_at = ?
                WHERE entity_id = ?
                """,
                (
                    _timestamp(current),
                    _timestamp(current),
                    _timestamp(current),
                    entity_id,
                ),
            )
            self._connection.commit()
        result = self.get_relationship(entity_id)
        assert result is not None
        return result

    def mark_reply_received(
        self,
        entity_id: str,
        *,
        at: datetime | None = None,
    ) -> Relationship:
        current = _utc_datetime(at or _now())
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE relationships
                SET awaiting_reply = 0, last_interaction_at = ?, updated_at = ?
                WHERE entity_id = ?
                """,
                (_timestamp(current), _timestamp(current), _identifier(entity_id)),
            )
            self._connection.commit()
        if cursor.rowcount != 1:
            raise KeyError(f"Отношения не найдены: {entity_id}")
        result = self.get_relationship(entity_id)
        assert result is not None
        return result

    def can_initiate(
        self,
        entity_id: str,
        *,
        now: datetime | None = None,
        sleeping: bool = False,
    ) -> bool:
        relationship = self.get_relationship(entity_id)
        return bool(
            relationship is not None
            and initiative_allowed(relationship, now=now, sleeping=sleeping)
        )

    @staticmethod
    def _summary_from_row(row: sqlite3.Row) -> WorldSummary:
        return WorldSummary(
            id=str(row["id"]),
            period_start=_parse_timestamp(row["period_start"]) or _now(),
            period_end=_parse_timestamp(row["period_end"]) or _now(),
            content=str(row["content"]),
            created_at=_parse_timestamp(row["created_at"]) or _now(),
        )

    def add_world_summary(
        self,
        period_start: datetime,
        period_end: datetime,
        content: str,
        *,
        at: datetime | None = None,
    ) -> WorldSummary:
        start = _utc_datetime(period_start)
        end = _utc_datetime(period_end)
        if end <= start:
            raise ValueError("Конец периода должен быть позже начала")
        clean_content = _clean_text(content, "Сводка мира", maximum=16_000)
        summary_id = uuid4().hex
        created = _timestamp(at or _now())
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO world_summaries (
                    id, period_start, period_end, content, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(period_start, period_end) DO UPDATE SET
                    content = excluded.content,
                    created_at = excluded.created_at
                """,
                (
                    summary_id,
                    _timestamp(start),
                    _timestamp(end),
                    clean_content,
                    created,
                ),
            )
            self._connection.commit()
            row = self._connection.execute(
                """
                SELECT * FROM world_summaries
                WHERE period_start = ? AND period_end = ?
                """,
                (_timestamp(start), _timestamp(end)),
            ).fetchone()
        assert row is not None
        return self._summary_from_row(row)

    def list_world_summaries(self, *, limit: int = 12) -> list[WorldSummary]:
        limit = _integer(limit, "Лимит сводок", 1, 1_000)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM world_summaries
                ORDER BY period_end DESC, created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    @staticmethod
    def _skill_audit_from_row(row: sqlite3.Row) -> SkillAuditRecord:
        return SkillAuditRecord(
            id=int(row["id"]),
            turn_id=str(row["turn_id"]),
            skill_id=str(row["skill_id"]),
            action=str(row["action"]),
            success=bool(row["success"]),
            detail=_json_load(str(row["detail_json"])),
            created_at=_parse_timestamp(row["created_at"]) or _now(),
        )

    def record_skill_audit(
        self,
        turn_id: str,
        skill_id: str,
        action: str,
        *,
        success: bool,
        detail: Mapping[str, Any] | None = None,
        at: datetime | None = None,
    ) -> SkillAuditRecord:
        if not isinstance(success, bool):
            raise TypeError("success должен быть bool")
        values = (
            _identifier(turn_id, "ID хода"),
            _clean_text(skill_id, "ID навыка", maximum=255),
            _clean_text(action, "Действие навыка", maximum=120),
            int(success),
            _json_dump(dict(detail or {})),
            _timestamp(at or _now()),
        )
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO skill_audit (
                    turn_id, skill_id, action, success, detail_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            self._connection.commit()
            row = self._connection.execute(
                "SELECT * FROM skill_audit WHERE id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
        assert row is not None
        return self._skill_audit_from_row(row)

    # More natural alias at the skill boundary.
    record_skill_activation = record_skill_audit

    def list_skill_audit(
        self,
        *,
        turn_id: str | None = None,
        limit: int = 100,
    ) -> list[SkillAuditRecord]:
        limit = _integer(limit, "Лимит аудита", 1, 10_000)
        where = ""
        parameters: list[Any] = []
        if turn_id is not None:
            where = "WHERE turn_id = ?"
            parameters.append(_identifier(turn_id, "ID хода"))
        parameters.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT * FROM skill_audit {where}
                ORDER BY id DESC LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._skill_audit_from_row(row) for row in rows]

    def apply_heartbeat_changes(
        self,
        changes: HeartbeatChanges,
        *,
        expected_revision: int | None = None,
        at: datetime | None = None,
        record_heartbeat: bool = True,
    ) -> AgentState:
        """Atomically apply one bounded heartbeat result.

        No partial world update becomes visible when validation, a locked fact,
        the active-goal cap or a revision check fails.
        """

        if not isinstance(changes, HeartbeatChanges):
            raise TypeError("changes должен быть HeartbeatChanges")
        if not isinstance(record_heartbeat, bool):
            raise TypeError("record_heartbeat должен быть bool")
        collections = {
            "новых сущностей": changes.entities,
            "событий": changes.events,
            "изменений целей": changes.goals,
            "изменений отношений": changes.relationships,
        }
        for label, values in collections.items():
            if len(values) > MAX_HEARTBEAT_CHANGES:
                raise ValueError(
                    f"За heartbeat допускается не больше {MAX_HEARTBEAT_CHANGES} {label}"
                )
        if not isinstance(changes.need_deltas, Mapping):
            raise TypeError("need_deltas должен быть объектом")
        unknown = set(changes.need_deltas) - set(NEED_NAMES)
        if unknown:
            raise ValueError("Неизвестные потребности: " + ", ".join(sorted(unknown)))
        need_deltas = {
            name: _bounded_delta(value, f"Изменение {name}", MAX_NEED_DELTA)
            for name, value in changes.need_deltas.items()
        }
        mood = (
            _clean_text(changes.mood, "Настроение", maximum=120)
            if changes.mood is not None
            else None
        )
        valence = (
            _integer(changes.valence, "Valence", -100, 100)
            if changes.valence is not None
            else None
        )
        arousal = (
            _integer(changes.arousal, "Arousal", 0, 100)
            if changes.arousal is not None
            else None
        )
        intention = (
            _clean_text(changes.current_intention, "Намерение", maximum=1_000)
            if changes.current_intention is not None
            else None
        )
        changed_at = _utc_datetime(at or _now())

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                state_row = self._connection.execute(
                    "SELECT * FROM agent_state WHERE id = 1"
                ).fetchone()
                assert state_row is not None
                actual_revision = int(state_row["revision"])
                if expected_revision is not None and actual_revision != expected_revision:
                    raise StateConflictError("Состояние Миланы уже изменилось")

                for entity in changes.entities:
                    self._apply_heartbeat_entity(entity, at=changed_at)
                for event in changes.events:
                    self._insert_life_event(event, at=changed_at)
                for goal_change in changes.goals:
                    self._change_goal(goal_change, at=changed_at)
                for relationship_delta in changes.relationships:
                    self._adjust_relationship(relationship_delta, at=changed_at)

                assignments: dict[str, Any] = {}
                for name, delta in need_deltas.items():
                    column = f"{name}_need"
                    assignments[column] = min(
                        100,
                        max(0, int(state_row[column]) + delta),
                    )
                if mood is not None:
                    assignments["mood"] = mood
                if valence is not None:
                    assignments["valence"] = valence
                if arousal is not None:
                    assignments["arousal"] = arousal
                if intention is not None:
                    assignments["current_intention"] = intention
                if record_heartbeat:
                    assignments["last_heartbeat_at"] = _timestamp(changed_at)
                assignments["revision"] = actual_revision + 1
                assignments["updated_at"] = _timestamp(changed_at)
                clause = ", ".join(f"{column} = ?" for column in assignments)
                cursor = self._connection.execute(
                    f"""
                    UPDATE agent_state SET {clause}
                    WHERE id = 1 AND revision = ?
                    """,
                    (*assignments.values(), actual_revision),
                )
                if cursor.rowcount != 1:
                    raise StateConflictError("Состояние Миланы уже изменилось")
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_agent_state()

    # Service code historically used the singular spelling while the model
    # adapter emits an array.  Keep both names deliberately.
    apply_heartbeat_update = apply_heartbeat_changes

    def load_world_context(
        self,
        *,
        entity_limit: int = 40,
        event_limit: int = 30,
        summary_limit: int = 4,
    ) -> WorldContext:
        return WorldContext(
            state=self.get_agent_state(),
            goals=tuple(self.list_goals(statuses=("active",), limit=MAX_ACTIVE_GOALS)),
            entities=tuple(self.list_entities(limit=entity_limit)),
            events=tuple(self.list_life_events(limit=event_limit)),
            relationships=tuple(self.list_relationships(limit=entity_limit)),
            summaries=tuple(self.list_world_summaries(limit=summary_limit)),
        )


# Shorter name for dependency injection while retaining the explicit public one.
StateStore = MilanaStateStore
