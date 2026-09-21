"""Durable, model-directed phone usage sessions.

The store deliberately contains no Telegram or model code.  It is the durable
state machine used by :class:`MilanaService`; network effects remain in the
existing Telegram outbox/acknowledgement paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping
from uuid import uuid4


PHONE_DECISIONS = frozenset({"continue", "put_away", "go_to_sleep"})
PHONE_INTENTS = frozenset({"read", "reply", "react"})


@dataclass(frozen=True, slots=True)
class PhoneVisit:
    target_ref: str
    intent: str


@dataclass(frozen=True, slots=True)
class PhoneSession:
    id: str
    status: str
    reason: str
    started_at: datetime
    next_decision_at: datetime
    selected_chat: str | None
    plan: tuple[PhoneVisit, ...]
    plan_cursor: int
    last_sleep_reminder_at: datetime | None
    ended_at: datetime | None = None
    end_reason: str | None = None

    @property
    def remaining_plan(self) -> tuple[PhoneVisit, ...]:
        return self.plan[self.plan_cursor :]


PHONE_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["continue", "put_away", "go_to_sleep"],
        },
        "reconsider_seconds": {
            "type": "integer",
            "minimum": 120,
            "maximum": 480,
        },
        "visits": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "target_ref": {"type": "string", "minLength": 1, "maxLength": 128},
                    "intent": {
                        "type": "string",
                        "enum": ["read", "reply", "react"],
                    },
                },
                "required": ["target_ref", "intent"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["decision", "reconsider_seconds", "visits"],
    "additionalProperties": False,
}


def _aware(value: str | datetime | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("PhoneSession timestamps must include a timezone")
    return parsed


class PhoneSessionStore:
    """Additive SQLite repository sharing ``MilanaStateStore`` transactions."""

    def __init__(self, state: Any) -> None:
        self.state = state
        with state.transaction() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS phone_sessions (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('active','ended')),
                    reason TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    next_decision_at TEXT NOT NULL,
                    selected_chat TEXT,
                    plan_json TEXT NOT NULL DEFAULT '[]',
                    plan_cursor INTEGER NOT NULL DEFAULT 0 CHECK (plan_cursor >= 0),
                    results_json TEXT NOT NULL DEFAULT '[]',
                    last_sleep_reminder_at TEXT,
                    ended_at TEXT,
                    end_reason TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_phone_sessions_one_active
                    ON phone_sessions(status) WHERE status='active';
                CREATE INDEX IF NOT EXISTS idx_phone_sessions_started
                    ON phone_sessions(started_at DESC);
                """
            )

    @staticmethod
    def _row(row: Any) -> PhoneSession | None:
        if row is None:
            return None
        raw_plan = json.loads(row["plan_json"])
        plan = tuple(
            PhoneVisit(str(item["target_ref"]), str(item["intent"]))
            for item in raw_plan
            if isinstance(item, Mapping)
        )
        return PhoneSession(
            id=str(row["id"]),
            status=str(row["status"]),
            reason=str(row["reason"]),
            started_at=_aware(row["started_at"]),  # type: ignore[arg-type]
            next_decision_at=_aware(row["next_decision_at"]),  # type: ignore[arg-type]
            selected_chat=row["selected_chat"],
            plan=plan,
            plan_cursor=int(row["plan_cursor"]),
            last_sleep_reminder_at=_aware(row["last_sleep_reminder_at"]),
            ended_at=_aware(row["ended_at"]),
            end_reason=row["end_reason"],
        )

    def active(self) -> PhoneSession | None:
        with self.state.transaction() as db:
            return self._row(
                db.execute(
                    "SELECT * FROM phone_sessions WHERE status='active' LIMIT 1"
                ).fetchone()
            )

    def recover(self, at: datetime) -> None:
        with self.state.transaction() as db:
            db.execute(
                """UPDATE phone_sessions
                   SET status='ended',ended_at=?,end_reason='interrupted'
                   WHERE status='active'""",
                (at.isoformat(),),
            )

    def start(self, reason: str, at: datetime, duration_seconds: int) -> PhoneSession:
        if not 120 <= duration_seconds <= 480:
            raise ValueError("PhoneSession duration must be between 120 and 480 seconds")
        session_id = uuid4().hex
        next_at = at + timedelta(seconds=duration_seconds)
        with self.state.transaction() as db:
            current = db.execute(
                "SELECT * FROM phone_sessions WHERE status='active' LIMIT 1"
            ).fetchone()
            if current is not None:
                return self._row(current)  # type: ignore[return-value]
            db.execute(
                """INSERT INTO phone_sessions
                   (id,status,reason,started_at,next_decision_at)
                   VALUES (?,'active',?,?,?)""",
                (session_id, reason[:80], at.isoformat(), next_at.isoformat()),
            )
        return self.active()  # type: ignore[return-value]

    @staticmethod
    def validate_plan(
        payload: Mapping[str, Any],
    ) -> tuple[str, int, tuple[PhoneVisit, ...]]:
        decision = payload.get("decision")
        seconds = payload.get("reconsider_seconds")
        raw_visits = payload.get("visits")
        if decision not in PHONE_DECISIONS:
            raise ValueError("Unknown PhoneSession decision")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, int)
            or not 120 <= seconds <= 480
        ):
            raise ValueError("PhoneSession reconsider_seconds must be 120..480")
        if not isinstance(raw_visits, list) or len(raw_visits) > 12:
            raise ValueError("PhoneSession visits must be an array of at most 12 items")
        visits: list[PhoneVisit] = []
        for item in raw_visits:
            if not isinstance(item, Mapping) or set(item) != {"target_ref", "intent"}:
                raise ValueError("Invalid PhoneSession visit")
            target = item["target_ref"]
            intent = item["intent"]
            if (
                not isinstance(target, str)
                or not target.strip()
                or len(target) > 128
            ):
                raise ValueError("Invalid PhoneSession target_ref")
            if intent not in PHONE_INTENTS:
                raise ValueError("Invalid PhoneSession intent")
            visits.append(PhoneVisit(target.strip(), str(intent)))
        return str(decision), seconds, tuple(visits)

    def apply_plan(
        self, session_id: str, payload: Mapping[str, Any], at: datetime
    ) -> PhoneSession:
        decision, seconds, visits = self.validate_plan(payload)
        if decision != "continue":
            return self.end(session_id, decision, at)
        raw = [{"target_ref": item.target_ref, "intent": item.intent} for item in visits]
        with self.state.transaction() as db:
            changed = db.execute(
                """UPDATE phone_sessions
                   SET next_decision_at=?,selected_chat=NULL,plan_json=?,plan_cursor=0
                   WHERE id=? AND status='active'""",
                (
                    (at + timedelta(seconds=seconds)).isoformat(),
                    json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                    session_id,
                ),
            ).rowcount
            if not changed:
                raise ValueError("PhoneSession is no longer active")
        return self.active()  # type: ignore[return-value]

    def select_visit(self, session_id: str, target_ref: str) -> None:
        with self.state.transaction() as db:
            db.execute(
                "UPDATE phone_sessions SET selected_chat=? WHERE id=? AND status='active'",
                (str(target_ref), session_id),
            )

    def finish_visit(
        self,
        session_id: str,
        *,
        target_ref: str,
        intent: str,
        outcome: str,
        at: datetime,
    ) -> None:
        with self.state.transaction() as db:
            row = db.execute(
                """SELECT plan_cursor,results_json FROM phone_sessions
                   WHERE id=? AND status='active'""",
                (session_id,),
            ).fetchone()
            if row is None:
                return
            results = json.loads(row["results_json"])
            results.append(
                {
                    "target_ref": str(target_ref),
                    "intent": intent,
                    "outcome": outcome,
                    "at": at.isoformat(),
                }
            )
            results = results[-50:]
            db.execute(
                """UPDATE phone_sessions
                   SET selected_chat=NULL,plan_cursor=plan_cursor+1,results_json=?
                   WHERE id=?""",
                (json.dumps(results, ensure_ascii=False, separators=(",", ":")), session_id),
            )

    def mark_sleep_reminder(self, session_id: str, at: datetime) -> None:
        with self.state.transaction() as db:
            db.execute(
                """UPDATE phone_sessions SET last_sleep_reminder_at=?
                   WHERE id=? AND status='active'""",
                (at.isoformat(), session_id),
            )

    def end(self, session_id: str, reason: str, at: datetime) -> PhoneSession:
        with self.state.transaction() as db:
            db.execute(
                """UPDATE phone_sessions
                   SET status='ended',ended_at=?,end_reason=?,selected_chat=NULL
                   WHERE id=? AND status='active'""",
                (at.isoformat(), reason[:80], session_id),
            )
            row = db.execute("SELECT * FROM phone_sessions WHERE id=?", (session_id,)).fetchone()
        session = self._row(row)
        if session is None:
            raise ValueError("Unknown PhoneSession")
        return session

    def snapshot(self) -> dict[str, Any]:
        with self.state.transaction() as db:
            row = db.execute(
                "SELECT * FROM phone_sessions ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return {"status": "idle", "session": None, "recent_actions": []}
            results = json.loads(row["results_json"])
        session = self._row(row)
        assert session is not None
        return {
            "status": session.status,
            "session": {
                "id": session.id,
                "reason": session.reason,
                "started_at": session.started_at.isoformat(),
                "next_decision_at": session.next_decision_at.isoformat(),
                "selected_chat": session.selected_chat,
                "remaining_plan": [
                    {"target_ref": item.target_ref, "intent": item.intent}
                    for item in session.remaining_plan
                ],
                "last_sleep_reminder_at": (
                    session.last_sleep_reminder_at.isoformat()
                    if session.last_sleep_reminder_at
                    else None
                ),
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "end_reason": session.end_reason,
            },
            "recent_actions": results[-20:],
        }


__all__ = [
    "PHONE_PLAN_SCHEMA",
    "PhoneSession",
    "PhoneSessionStore",
    "PhoneVisit",
]
