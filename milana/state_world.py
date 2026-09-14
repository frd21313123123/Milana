"""World entities, facts, events, goals, relationships, and atomic state updates."""

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
        idempotency_key: str | None = None,
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
        clean_idempotency_key = (
            _identifier(idempotency_key, "Ключ применения состояния")
            if idempotency_key is not None
            else None
        )

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if clean_idempotency_key is not None:
                    applied = self._connection.execute(
                        "SELECT 1 FROM state_change_ledger WHERE action_key = ?",
                        (clean_idempotency_key,),
                    ).fetchone()
                    if applied is not None:
                        self._connection.commit()
                        return self.get_agent_state()
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
                    assignments["current_intention_updated_at"] = _timestamp(
                        changed_at
                    )
                elif record_heartbeat:
                    # Intentions are deliberately ephemeral.  A reflective
                    # heartbeat must explicitly renew one; otherwise the old
                    # desire is removed instead of leaking into future chats.
                    assignments["current_intention"] = None
                    assignments["current_intention_updated_at"] = None
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
                if clean_idempotency_key is not None:
                    self._connection.execute(
                        "INSERT INTO state_change_ledger (action_key, applied_at) VALUES (?, ?)",
                        (clean_idempotency_key, _timestamp(changed_at)),
                    )
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

__all__ = ["WorldStateStoreMixin"]
