"""Durable, rule-based deviations layered over the weekly routine."""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from milana_schedule import Activity, ResponsePlan, ScheduleState, WeeklyRoutine


SCENARIOS = {
    "overslept": ("Проспала", "sleep", "Проспала и встала позже"),
    "commute_delay": ("Задержалась в дороге", "commute", "Дорога заняла больше времени"),
    "coffee_stop": ("Зашла за кофе", "food", "По пути решила взять кофе"),
    "class_cancelled": ("Пару отменили", "personal", "Последнюю пару отменили"),
    "walk_extended": ("Гуляет дольше", "walk", "Решила погулять подольше"),
    "store_visit": ("Зашла в магазин", "chores", "По дороге зашла в магазин"),
    "series_late": ("Смотрит сериал допоздна", "personal", "Увлеклась сериалом и засиделась"),
}

DEFAULT_PROBABILITIES = {
    "overslept": .10, "commute_delay": .15, "coffee_stop": .15,
    "class_cancelled": .05, "walk_extended": .20, "store_visit": .15,
    "series_late": .10,
}
DEFAULT_DURATIONS = {
    "overslept": (10, 30), "commute_delay": (5, 20), "coffee_stop": (10, 20),
    "class_cancelled": (90, 90), "walk_extended": (15, 40),
    "store_visit": (15, 30), "series_late": (20, 60),
}
TERMINAL = {"completed", "cancelled"}


@dataclass(frozen=True)
class PlannedEvent:
    event_id: str
    source_key: str
    scenario: str | None
    title: str
    kind: str
    location: str | None
    original_start: datetime
    original_end: datetime
    actual_start: datetime
    actual_end: datetime
    status: str
    reason: str | None
    custom: bool = False
    inferred: bool = False

    @property
    def start(self) -> int:
        return self.actual_start.hour * 60 + self.actual_start.minute

    @property
    def end(self) -> int:
        return self.actual_end.hour * 60 + self.actual_end.minute

    @property
    def duration(self) -> int:
        return max(0, round((self.actual_end - self.actual_start).total_seconds() / 60))

    def contains(self, at: datetime) -> bool:
        return self.actual_start <= at < self.actual_end and self.status != "cancelled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id, "source_key": self.source_key,
            "scenario": self.scenario, "title": self.title, "kind": self.kind,
            "location": self.location, "original_start": self.original_start.isoformat(),
            "original_end": self.original_end.isoformat(),
            "actual_start": self.actual_start.isoformat(), "actual_end": self.actual_end.isoformat(),
            "status": self.status, "reason": self.reason, "custom": self.custom,
            "inferred": self.inferred,
        }


@dataclass(frozen=True)
class LifePlanState:
    now: datetime
    day_key: str
    current: PlannedEvent | None
    next_activity: PlannedEvent | None
    next_at: datetime | None
    metrics: Any


class LifePlanner:
    """Persist a concrete plan and make each random decision exactly once."""

    def __init__(self, state: Any, routine: WeeklyRoutine, *, now: Callable[[], datetime] | None = None,
                 enabled: bool | None = None, rng: random.Random | None = None) -> None:
        self.store = state
        self.routine = routine
        self.timezone = routine.timezone
        self.timezone_name = routine.timezone_name
        self.online_behavior = routine.online_behavior
        self._now = now or (lambda: datetime.now(routine.timezone))
        config = routine.life_planner
        self.enabled = bool(config.get("enabled", False) if enabled is None else enabled)
        self.daily_limit = self._integer(config.get("daily_deviation_limit", 2), 0, 10, "daily_deviation_limit")
        self.probabilities = self._probabilities(config.get("probabilities", {}))
        self.durations = self._durations(config.get("durations_minutes", {}))
        self.rng = rng or random.Random()
        with state.transaction() as db:
            self._create_schema(db)
            self._sync_config(db, self.normalize_datetime(self._now()))

    @staticmethod
    def _integer(value: Any, low: int, high: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"life_planner.{label} должен быть целым числом от {low} до {high}")
        return value

    @staticmethod
    def _probabilities(raw: Any) -> dict[str, float]:
        if not isinstance(raw, Mapping):
            raise ValueError("life_planner.probabilities должен быть объектом")
        result = dict(DEFAULT_PROBABILITIES)
        for key, value in raw.items():
            if key not in SCENARIOS or isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"Некорректная вероятность life_planner.probabilities.{key}")
            result[key] = float(value)
        return result

    @staticmethod
    def _durations(raw: Any) -> dict[str, tuple[int, int]]:
        if not isinstance(raw, Mapping):
            raise ValueError("life_planner.durations_minutes должен быть объектом")
        result = dict(DEFAULT_DURATIONS)
        for key, value in raw.items():
            if key not in SCENARIOS or not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"Некорректный диапазон life_planner.durations_minutes.{key}")
            low, high = value
            if any(isinstance(x, bool) or not isinstance(x, int) for x in value) or not 1 <= low <= high <= 180:
                raise ValueError(f"Некорректный диапазон life_planner.durations_minutes.{key}")
            result[key] = (low, high)
        return result

    @staticmethod
    def _create_schema(db: sqlite3.Connection) -> None:
        db.execute("""CREATE TABLE IF NOT EXISTS planned_events (
            event_id TEXT PRIMARY KEY, source_key TEXT NOT NULL, scenario TEXT,
            title TEXT NOT NULL, kind TEXT NOT NULL, location TEXT,
            original_start REAL NOT NULL, original_end REAL NOT NULL,
            actual_start REAL NOT NULL, actual_end REAL NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('planned','active','completed','cancelled')),
            reason TEXT, custom INTEGER NOT NULL DEFAULT 0, inferred INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL)""")
        db.execute("CREATE INDEX IF NOT EXISTS planned_events_time ON planned_events(actual_start,actual_end)")
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS planned_one_active ON planned_events ((1)) WHERE status='active'")
        db.execute("""CREATE TABLE IF NOT EXISTS life_opportunities (
            opportunity_key TEXT PRIMARY KEY, local_day TEXT NOT NULL, scenario TEXT NOT NULL,
            source_event_id TEXT NOT NULL, selected INTEGER NOT NULL, reason TEXT,
            checked_at REAL NOT NULL)""")
        db.execute("CREATE TABLE IF NOT EXISTS life_planner_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS life_materialized_days (local_day TEXT PRIMARY KEY,created_at REAL NOT NULL)")

    def _config_signature(self) -> str:
        days = {
            key: [(item.title, item.kind, item.start, item.end, item.custom)
                  for item in self.routine.days[key]]
            for key in self.routine.days
        }
        payload = {"enabled": self.enabled, "limit": self.daily_limit,
                   "probabilities": self.probabilities, "durations": self.durations,
                   "days": days}
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def _sync_config(self, db: sqlite3.Connection, now: datetime) -> None:
        signature = self._config_signature()
        row = db.execute("SELECT value FROM life_planner_meta WHERE key='config_signature'").fetchone()
        if row is not None and row[0] != signature:
            db.execute("""UPDATE planned_events SET status='cancelled',reason='План изменён в конфигурации',updated_at=?
                WHERE status='planned' AND scenario IS NOT NULL AND actual_start>?""",
                (now.timestamp(), now.timestamp()))
            db.execute("DELETE FROM planned_events WHERE status='planned' AND scenario IS NULL AND actual_start>?",
                       (now.timestamp(),))
            db.execute("DELETE FROM life_materialized_days WHERE local_day>=?", (now.date().isoformat(),))
        db.execute("INSERT OR REPLACE INTO life_planner_meta VALUES ('config_signature',?)", (signature,))

    def normalize_datetime(self, value: datetime | None = None) -> datetime:
        return self.routine.normalize_datetime(value)

    @staticmethod
    def _location(activity: Activity) -> str:
        if activity.kind == "study":
            return "home" if "самообуч" in activity.title.lower() else "university"
        return {"work": "workplace", "commute": "transit", "walk": "outside"}.get(activity.kind, "home")

    @staticmethod
    def _event_id(start: datetime, activity: Activity) -> str:
        raw = f"{start.isoformat()}|{activity.kind}|{activity.title}|{activity.start}|{activity.custom}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _materialize_day(self, db: sqlite3.Connection, day: datetime, created_at: datetime) -> None:
        start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        day_key = start.date().isoformat()
        if db.execute("SELECT 1 FROM life_materialized_days WHERE local_day=?", (day_key,)).fetchone():
            return
        cursor = start
        while cursor < end:
            state = self.routine.state_at(cursor)
            schedule_end = state.next_at or end
            boundary = min(schedule_end, end)
            activity = state.current
            if activity is not None and schedule_end > cursor and schedule_end > created_at:
                minute = cursor.hour * 60 + cursor.minute
                elapsed = (minute - activity.start) % (24 * 60)
                event_start = cursor.replace(second=0, microsecond=0) - timedelta(minutes=elapsed)
                existing_current = db.execute("""SELECT 1 FROM planned_events WHERE status!='cancelled'
                    AND actual_start<=? AND actual_end>? LIMIT 1""",
                    (created_at.timestamp(), created_at.timestamp())).fetchone()
                if event_start < created_at and existing_current:
                    cursor = boundary if boundary > cursor else cursor + timedelta(minutes=1)
                    continue
                event_id = self._event_id(event_start, activity)
                db.execute("""INSERT OR IGNORE INTO planned_events VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    event_id, event_id, None, activity.title, activity.kind, self._location(activity),
                    event_start.timestamp(), schedule_end.timestamp(), event_start.timestamp(), schedule_end.timestamp(),
                    "planned", None, int(activity.custom), 0, created_at.timestamp(), created_at.timestamp(),
                ))
            cursor = boundary if boundary > cursor else cursor + timedelta(minutes=1)
        db.execute("INSERT OR REPLACE INTO life_materialized_days VALUES (?,?)",
                   (day_key, created_at.timestamp()))

    def _row(self, row: sqlite3.Row) -> PlannedEvent:
        stamp = lambda value: datetime.fromtimestamp(value, self.timezone)
        return PlannedEvent(
            event_id=row["event_id"], source_key=row["source_key"], scenario=row["scenario"],
            title=row["title"], kind=row["kind"], location=row["location"],
            original_start=stamp(row["original_start"]), original_end=stamp(row["original_end"]),
            actual_start=stamp(row["actual_start"]), actual_end=stamp(row["actual_end"]),
            status=row["status"], reason=row["reason"], custom=bool(row["custom"]),
            inferred=bool(row["inferred"]),
        )

    def _candidates(self, event: PlannedEvent, energy: float) -> list[str]:
        result: list[str] = []
        if event.kind == "sleep": result.append("overslept")
        if event.kind == "commute": result.extend(("commute_delay", "coffee_stop"))
        if event.kind == "study" and not event.custom and event.title == "Учёба" and event.duration >= 90:
            result.append("class_cancelled")
        if event.kind == "walk" and energy >= 30: result.append("walk_extended")
        if event.kind == "personal": result.append("store_visit")
        if event.kind == "personal" and event.actual_start.hour >= 18 and energy >= 40:
            result.append("series_late")
        return result

    def _daily_count(self, db: sqlite3.Connection, day: str) -> int:
        return db.execute("SELECT count(*) FROM life_opportunities WHERE local_day=? AND selected=1", (day,)).fetchone()[0]

    def _already_selected(self, db: sqlite3.Connection, day: str, scenario: str) -> bool:
        return db.execute("SELECT 1 FROM life_opportunities WHERE local_day=? AND scenario=? AND selected=1", (day, scenario)).fetchone() is not None

    def _record_decision(self, db: sqlite3.Connection, key: str, day: str, scenario: str,
                         event_id: str, selected: bool, reason: str, at: datetime) -> None:
        db.execute("INSERT OR IGNORE INTO life_opportunities VALUES (?,?,?,?,?,?,?)",
                   (key, day, scenario, event_id, int(selected), reason, at.timestamp()))

    def _trim_overlaps(self, db: sqlite3.Connection, event_id: str, start: datetime, end: datetime, at: datetime) -> None:
        rows = db.execute("""SELECT * FROM planned_events WHERE event_id<>? AND status='planned'
            AND actual_start<? AND actual_end>? ORDER BY actual_start""", (event_id, end.timestamp(), start.timestamp())).fetchall()
        for row in rows:
            other = self._row(row)
            if other.custom:
                continue
            new_start = max(other.actual_start, end)
            status = "cancelled" if new_start >= other.actual_end else "planned"
            db.execute("UPDATE planned_events SET actual_start=?,status=?,reason=COALESCE(reason,?),updated_at=? WHERE event_id=?",
                       (new_start.timestamp(), status, "Сокращено из-за изменения плана", at.timestamp(), other.event_id))

    def _insert_deviation(self, db: sqlite3.Connection, source: PlannedEvent, scenario: str,
                          start: datetime, end: datetime, at: datetime, *, location: str | None = None) -> PlannedEvent:
        title, kind, reason = SCENARIOS[scenario]
        event_id = hashlib.sha256(f"{source.event_id}|{scenario}".encode()).hexdigest()[:24]
        db.execute("""INSERT OR REPLACE INTO planned_events VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            event_id, source.event_id, scenario, title, kind, location or source.location,
            start.timestamp(), end.timestamp(), start.timestamp(), end.timestamp(), "planned",
            reason, 0, 0, at.timestamp(), at.timestamp(),
        ))
        return self._row(db.execute("SELECT * FROM planned_events WHERE event_id=?", (event_id,)).fetchone())

    def _insert_service_event(self, db: sqlite3.Connection, source: PlannedEvent, suffix: str,
                              title: str, kind: str, location: str, start: datetime,
                              end: datetime, at: datetime, reason: str) -> PlannedEvent:
        event_id = hashlib.sha256(f"{source.event_id}|{suffix}".encode()).hexdigest()[:24]
        db.execute("""INSERT OR REPLACE INTO planned_events VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            event_id, source.event_id, None, title, kind, location,
            start.timestamp(), end.timestamp(), start.timestamp(), end.timestamp(),
            "planned", reason, 0, 0, at.timestamp(), at.timestamp(),
        ))
        return self._row(db.execute("SELECT * FROM planned_events WHERE event_id=?", (event_id,)).fetchone())

    def _apply(self, db: sqlite3.Connection, event: PlannedEvent, scenario: str, minutes: int,
               at: datetime) -> bool:
        delta = timedelta(minutes=minutes)
        reason = SCENARIOS[scenario][2]
        if scenario in {"overslept", "commute_delay", "walk_extended"}:
            desired = event.actual_end + delta
            fixed = db.execute("""SELECT MIN(actual_start) FROM planned_events
                WHERE status='planned' AND (custom=1 OR kind='sleep') AND actual_start>=?""",
                (event.actual_end.timestamp(),)).fetchone()[0]
            if fixed is not None:
                desired = min(desired, datetime.fromtimestamp(fixed, self.timezone))
            if desired <= event.actual_end:
                return False
            db.execute("UPDATE planned_events SET actual_end=?,scenario=?,reason=?,updated_at=? WHERE event_id=?",
                       (desired.timestamp(), scenario, reason, at.timestamp(), event.event_id))
            self._trim_overlaps(db, event.event_id, event.actual_end, desired, at)
            return True
        if scenario == "coffee_stop":
            if event.duration <= minutes + 5:
                return False
            start = event.actual_end - delta
            db.execute("UPDATE planned_events SET actual_end=?,updated_at=? WHERE event_id=?",
                       (start.timestamp(), at.timestamp(), event.event_id))
            self._insert_deviation(db, event, scenario, start, event.actual_end, at, location="cafe")
            return True
        if scenario == "class_cancelled":
            if event.duration < 90:
                return False
            start = event.actual_end - timedelta(minutes=90)
            db.execute("UPDATE planned_events SET actual_end=?,updated_at=? WHERE event_id=?",
                       (start.timestamp(), at.timestamp(), event.event_id))
            self._insert_deviation(db, event, scenario, start, event.actual_end, at, location="university")
            return True
        if scenario == "store_visit":
            travel = timedelta(minutes=10)
            end = min(event.actual_end, event.actual_start + travel + delta + travel)
            if (end - event.actual_start) < timedelta(minutes=35):
                return False
            db.execute("UPDATE planned_events SET actual_start=?,updated_at=? WHERE event_id=?",
                       (end.timestamp(), at.timestamp(), event.event_id))
            db.execute("UPDATE planned_events SET status='planned' WHERE event_id=?", (event.event_id,))
            self._insert_service_event(db, event, "store-out", "Дорога в магазин", "commute", "transit",
                                       event.actual_start, event.actual_start + travel, at,
                                       "По дороге зашла в магазин")
            self._insert_deviation(db, event, scenario, event.actual_start + travel,
                                   end - travel, at, location="shop")
            self._insert_service_event(db, event, "store-back", "Дорога домой", "commute", "transit",
                                       end - travel, end, at, "Возвращается из магазина")
            return True
        if scenario == "series_late":
            sleep_row = db.execute("""SELECT * FROM planned_events WHERE kind='sleep' AND status='planned'
                AND actual_start>=? ORDER BY actual_start LIMIT 1""", (event.actual_end.timestamp(),)).fetchone()
            if sleep_row is None:
                return False
            sleep = self._row(sleep_row)
            wake_row = db.execute("""SELECT MAX(actual_end) FROM planned_events WHERE kind='sleep'
                AND status='planned' AND actual_start>=? AND actual_start<?""",
                (sleep.actual_start.timestamp(), (sleep.actual_start + timedelta(hours=12)).timestamp())).fetchone()[0]
            next_wake = datetime.fromtimestamp(wake_row, self.timezone) if wake_row else sleep.actual_end
            duration = min(delta, next_wake - sleep.actual_start - timedelta(hours=6))
            if duration < timedelta(minutes=20):
                return False
            end = sleep.actual_start + duration
            db.execute("UPDATE planned_events SET actual_start=?,scenario=?,reason=?,updated_at=? WHERE event_id=?",
                       (end.timestamp(), scenario, reason, at.timestamp(), sleep.event_id))
            inserted = self._insert_deviation(db, event, scenario, sleep.actual_start, end, at, location="home")
            self._trim_overlaps(db, inserted.event_id, sleep.actual_start, end, at)
            return True
        return False

    def _advance_db(self, db: sqlite3.Connection, now: datetime, energy: float) -> None:
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        self._materialize_day(db, midnight, now)
        self._materialize_day(db, midnight + timedelta(days=1), now)
        db.execute("UPDATE planned_events SET status='completed',updated_at=? WHERE status='active' AND actual_end<=?",
                   (now.timestamp(), now.timestamp()))
        db.execute("UPDATE planned_events SET status='completed',inferred=1,updated_at=? WHERE status='planned' AND actual_end<=?",
                   (now.timestamp(), now.timestamp()))
        current_row = db.execute("""SELECT * FROM planned_events WHERE status IN ('planned','active')
            AND actual_start<=? AND actual_end>? ORDER BY custom DESC,actual_start DESC LIMIT 1""",
            (now.timestamp(), now.timestamp())).fetchone()
        if current_row:
            current = self._row(current_row)
            reason = "Завершено пользовательским событием" if current.custom else "Завершено переходом плана"
            db.execute("""UPDATE planned_events SET actual_end=?,status='completed',reason=COALESCE(reason,?),updated_at=?
                WHERE status='active' AND event_id<>?""",
                (now.timestamp(), reason, now.timestamp(), current.event_id))
            db.execute("UPDATE planned_events SET status='active',updated_at=? WHERE event_id=?",
                       (now.timestamp(), current.event_id))
            if self.enabled and current.scenario is None:
                day = current.actual_start.date().isoformat()
                for scenario in self._candidates(current, energy):
                    key = f"{current.event_id}:{scenario}"
                    if db.execute("SELECT 1 FROM life_opportunities WHERE opportunity_key=?", (key,)).fetchone():
                        continue
                    preplanned = current_row["created_at"] < current.original_start.timestamp()
                    eligible = current.original_start == now or (
                        preplanned and current.original_start <= now < current.original_start + timedelta(minutes=1)
                    )
                    eligible = eligible and self._daily_count(db, day) < self.daily_limit
                    eligible = eligible and not self._already_selected(db, day, scenario)
                    probability = .20 if scenario == "overslept" and energy <= 30 else self.probabilities[scenario]
                    selected = eligible and self.rng.random() < probability
                    minutes = self.rng.randint(*self.durations[scenario]) if selected else 0
                    applied = selected and self._apply(db, current, scenario, minutes, now)
                    self._record_decision(db, key, day, scenario, current.event_id, applied,
                                          SCENARIOS[scenario][2] if applied else "Отклонение не выбрано", now)
                    if applied:
                        break
        if not self.enabled:
            db.execute("UPDATE planned_events SET status='cancelled',reason='LifePlanner отключён',updated_at=? WHERE status='planned' AND scenario IS NOT NULL",
                       (now.timestamp(),))

    def advance(self, value: datetime | None = None, *, energy: float = 65.0,
                db: sqlite3.Connection | None = None) -> LifePlanState:
        now = self.normalize_datetime(value or self._now())
        if db is not None:
            self._advance_db(db, now, energy)
            return self.state_at(now, db=db)
        with self.store.transaction() as connection:
            self._advance_db(connection, now, energy)
            return self.state_at(now, db=connection)

    def state_at(self, value: datetime | None = None, *, db: sqlite3.Connection | None = None) -> LifePlanState:
        now = self.normalize_datetime(value)
        def read(connection: sqlite3.Connection):
            row = connection.execute("""SELECT * FROM planned_events WHERE status!='cancelled'
                AND actual_start<=? AND actual_end>? ORDER BY custom DESC,actual_start DESC LIMIT 1""",
                (now.timestamp(), now.timestamp())).fetchone()
            next_row = connection.execute("""SELECT * FROM planned_events WHERE status='planned' AND actual_start>?
                ORDER BY actual_start,custom DESC LIMIT 1""", (now.timestamp(),)).fetchone()
            bounds = connection.execute("SELECT MIN(original_start),MAX(actual_end) FROM planned_events").fetchone()
            return row, next_row, bounds
        if db is None:
            with self.store.transaction() as connection:
                row, next_row, bounds = read(connection)
        else:
            row, next_row, bounds = read(db)
        if bounds[0] is None or now.timestamp() < bounds[0] or now.timestamp() >= bounds[1]:
            base = self.routine.state_at(now)
            return LifePlanState(base.now, base.day_key, base.current, base.next_activity,
                                 base.next_at, base.metrics)
        current = self._row(row) if row else None
        following = self._row(next_row) if next_row else None
        if current is None and following is None:
            base = self.routine.state_at(now)
            return LifePlanState(base.now, base.day_key, None, None, base.next_at, base.metrics)
        return LifePlanState(now, ("mon","tue","wed","thu","fri","sat","sun")[now.weekday()],
                             current, following, current.actual_end if current else following.actual_start,
                             self.routine.state_at(now).metrics)

    def response_policy_at(self, value: datetime | None = None):
        state = self.state_at(value)
        return self.routine._response_policy_for_activity(state.current)

    def attentive_response_policy(self, policy, value, last_attentive_at):
        return self.routine.attentive_response_policy(policy, value, last_attentive_at)

    def plan_response(self, value: datetime | None = None, randint: Callable[[int, int], int] = random.randint,
                      *, last_attentive_at: datetime | None = None) -> ResponsePlan:
        received = self.normalize_datetime(value)
        cursor = received
        for _ in range(200):
            state = self.state_at(cursor)
            policy = self.routine._response_policy_for_activity(state.current)
            if policy.available:
                policy = self.routine.attentive_response_policy(policy, cursor, last_attentive_at)
                due = cursor + timedelta(seconds=randint(policy.min_delay_seconds, policy.max_delay_seconds))
                if state.next_at is None or due < state.next_at:
                    return ResponsePlan(received, due, policy)
            if state.next_at is None or state.next_at <= cursor:
                break
            cursor = state.next_at + timedelta(microseconds=1)
        return self.routine.plan_response(received, randint, last_attentive_at=last_attentive_at)

    def snapshot(self, value: datetime | None = None) -> dict[str, Any]:
        now = self.normalize_datetime(value or self._now())
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        with self.store.transaction() as db:
            rows = db.execute("""SELECT * FROM planned_events WHERE actual_start<? AND actual_end>?
                ORDER BY actual_start,custom DESC""", ((start + timedelta(days=1)).timestamp(), start.timestamp())).fetchall()
            decisions = db.execute("SELECT * FROM life_opportunities WHERE local_day=? ORDER BY checked_at",
                                   (start.date().isoformat(),)).fetchall()
        state = self.state_at(now)
        return {
            "enabled": self.enabled,
            "current": state.current.to_dict() if state.current else None,
            "next": state.next_activity.to_dict() if state.next_activity else None,
            "events": [self._row(row).to_dict() for row in rows],
            "decisions": [dict(row) for row in decisions],
            "daily_deviation_limit": self.daily_limit,
        }

    def model_context(self, value: datetime | None = None) -> str:
        state = self.state_at(value)
        lines = ["<life_plan>Актуальный сохранённый план; будущие пункты — намерения, а не уже произошедшие факты."]
        if state.current:
            lines.append(f"Сейчас: {state.current.title} до {state.current.actual_end:%H:%M}." +
                         (f" Причина отклонения: {state.current.reason}." if state.current.reason else ""))
        if state.next_activity:
            lines.append(f"Дальше: {state.next_activity.title} с {state.next_activity.actual_start:%H:%M}.")
        lines.append("</life_plan>")
        return "\n".join(lines)
