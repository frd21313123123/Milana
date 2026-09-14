"""Agent state, recovery, and heartbeat-job persistence."""

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
            current_intention_updated_at=_parse_timestamp(
                row["current_intention_updated_at"]
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
            changes["current_intention_updated_at"] = (
                _timestamp(at or _now()) if current_intention is not None else None
            )
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

__all__ = ["LifecycleStateStoreMixin"]
