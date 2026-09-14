"""Data models, validation primitives and policies for Milana persistent state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


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
    current_intention_updated_at: datetime | None
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
class TelegramOutboxSentPart:
    """Durable delivery result for one immutable outbox message part.

    ``message_id`` is intentionally optional: Telegram can confirm a repeated
    MTProto ``random_id`` without returning the ID of the message created by
    the original, lost RPC response.
    """

    part_index: int
    message_id: int | None


@dataclass(frozen=True)
class TelegramOutboxEntry:
    action_key: str
    target_ref: str
    notice_ids: tuple[str, ...]
    messages: tuple[str, ...]
    next_part_index: int
    sent_parts: tuple[TelegramOutboxSentPart, ...]
    status: str
    created_at: datetime
    updated_at: datetime
    first_sent_at: datetime | None

    @property
    def sent_message_ids(self) -> tuple[int, ...]:
        """Known Telegram IDs, retained for compatibility with old callers."""

        return tuple(
            part.message_id
            for part in self.sent_parts
            if part.message_id is not None
        )

    def message_id_for_part(self, part_index: int) -> int | None:
        """Return the ID bound to ``part_index`` (or ``None`` when unknown)."""

        for part in self.sent_parts:
            if part.part_index == part_index:
                return part.message_id
        return None


@dataclass(frozen=True)
class TelegramAckIntent:
    """Durable, idempotently recoverable Telegram read acknowledgement.

    The notice journal becomes terminal in the same transaction that creates
    this intent.  That transaction is only started after the incoming content
    was materialized and every user-visible reply part was durably sent.
    """

    action_key: str
    target_ref: str
    notice_ids: tuple[str, ...]
    message_ids: tuple[int, ...]
    status: str
    attempts: int
    last_error: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass(frozen=True)
class TelegramTurnMetric:
    turn_id: str
    chat_id: str
    outcome: str
    context_ms: float
    provider_queue_ms: float
    model_ms: float
    send_ms: float
    generation_to_first_send_ms: float
    model_rounds: int
    context_messages: int
    context_characters: int
    started_at: datetime
    first_sent_at: datetime | None
    # ``None`` is reserved for rows produced before the additive migration or
    # lightweight callers that do not know whether the turn was an ordinary
    # text fast-path attempt.  The service always writes an explicit value.
    sla_eligible: bool | None = None


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
