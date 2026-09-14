"""Provider-neutral state payload validation for MilanaService."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping

from milana_state import (
    FactSeed,
    GoalChange,
    HeartbeatChanges,
    NewEntity,
    NewLifeEvent,
    RelationshipDelta,
)


def _parse_datetime(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be an ISO datetime string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _patch_rows(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw = payload.get(key, [])
    if not isinstance(raw, list) or len(raw) > 3:
        raise ValueError(f"{key} must be an array with at most three entries")
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"arguments_json"}:
            raise ValueError(f"Each {key} entry must contain arguments_json")
        encoded = item["arguments_json"]
        if not isinstance(encoded, str):
            raise TypeError(f"{key}.arguments_json must be a string")
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {key} arguments_json") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"{key} patch must decode to an object")
        result.append(decoded)
    return result


def _facts(raw: Any) -> tuple[FactSeed, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or len(raw) > 20:
        raise ValueError("entity facts must be an array with at most 20 entries")
    result: list[FactSeed] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
            raise ValueError("Each entity fact needs a string key")
        # Only the trusted seed path may create immutable facts.
        result.append(
            FactSeed(
                key=item["key"],
                value=item.get("value"),
                locked=False,
                source="milana",
            )
        )
    return tuple(result)


def build_heartbeat_changes(
    payload: Mapping[str, Any],
    current_state: Any,
) -> HeartbeatChanges:
    """Validate the provider-neutral final payload into one atomic reducer input."""

    state_update = payload.get("state_update")
    if not isinstance(state_update, Mapping):
        raise ValueError("state_update must be an object")
    need_deltas: dict[str, int] = {}
    for name in ("social", "rest", "novelty", "achievement"):
        target = state_update.get(name)
        if target is None:
            continue
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= 100:
            raise ValueError(f"state_update.{name} must be 0..100 or null")
        delta = target - int(getattr(current_state, name))
        if not -15 <= delta <= 15:
            raise ValueError(f"state_update.{name} changes a need by more than 15")
        need_deltas[name] = delta

    entities: list[NewEntity] = []
    for item in _patch_rows(payload, "entity_updates"):
        entities.append(
            NewEntity(
                kind=item.get("kind", "person"),
                name=item.get("name", ""),
                description=item.get("description", ""),
                is_real=item.get("is_real", False),
                entity_id=item.get("entity_id"),
                facts=_facts(item.get("facts")),
            )
        )

    events: list[NewLifeEvent] = []
    for item in _patch_rows(payload, "life_events"):
        entity_ids = item.get("entity_ids", [])
        if not isinstance(entity_ids, list) or not all(
            isinstance(value, str) for value in entity_ids
        ):
            raise ValueError("life event entity_ids must be an array of strings")
        happened_at = item.get("happened_at")
        events.append(
            NewLifeEvent(
                title=item.get("title", ""),
                description=item.get("description", ""),
                kind=item.get("kind", "life"),
                importance=item.get("importance", 50),
                entity_ids=tuple(entity_ids),
                happened_at=(
                    _parse_datetime(happened_at, field_name="happened_at")
                    if happened_at is not None
                    else None
                ),
                raw_payload=item,
            )
        )

    goals = tuple(
        GoalChange(
            operation=item.get("operation", "create"),
            goal_id=item.get("goal_id"),
            title=item.get("title"),
            description=item.get("description", ""),
            horizon=item.get("horizon", "short"),
            progress=item.get("progress"),
        )
        for item in _patch_rows(payload, "goal_updates")
    )

    relationships: list[RelationshipDelta] = []
    for item in _patch_rows(payload, "relationship_updates"):
        interacted_at = item.get("interacted_at")
        relationships.append(
            RelationshipDelta(
                entity_id=item.get("entity_id", ""),
                closeness=item.get("closeness", 0),
                reciprocity=item.get("reciprocity", 0),
                tension=item.get("tension", 0),
                awaiting_reply=item.get("awaiting_reply"),
                blocked=item.get("blocked"),
                interacted_at=(
                    _parse_datetime(interacted_at, field_name="interacted_at")
                    if interacted_at is not None
                    else None
                ),
            )
        )

    mood = state_update.get("mood_label")
    intention = state_update.get("current_intention")
    return HeartbeatChanges(
        entities=tuple(entities),
        events=tuple(events),
        goals=goals,
        need_deltas=need_deltas,
        relationships=tuple(relationships),
        mood=mood if isinstance(mood, str) and mood.strip() else None,
        valence=state_update.get("valence"),
        arousal=state_update.get("arousal"),
        current_intention=(
            intention if isinstance(intention, str) and intention.strip() else None
        ),
    )

__all__ = ["build_heartbeat_changes"]
