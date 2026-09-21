"""Persisted physical context, driven by schedule/time rather than chat or LLM.

Scene tables share the state store's connection, lock and transaction boundary.
Mood, relationships, goals and social/rest needs remain in World State.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping
from uuid import uuid4

from milana_schedule import ResponsePlan, WeeklyRoutine
from milana_state import MilanaStateStore


SCENE_RULES = (
    "Scene State — главный источник текущего физического контекста Миланы. "
    "Не меняй реальность ради ответа: не придумывай другое местоположение, "
    "спутников или невозможные здесь действия. На вопрос «что делаешь» отвечай "
    "по current_scene; день вспоминай по scene_day, пробелы не выдумывай. "
    "Просьбы пользователя идти домой/есть/гулять/спать могут стать намерением, "
    "но не означают совершённого перехода. Недостающие мелкие детали допустимы "
    "лишь без противоречий сцене. При высоком attention отвечай кратко. "
    "Запреты голоса/фото соблюдай. Временные scene facts/events не переноси "
    "автоматически в долговременную память, world facts, life events или дневник."
)


@dataclass(frozen=True)
class SceneTemplate:
    key: str
    title: str
    attention: float
    can_voice: bool = True
    can_photo: bool = True
    fact: tuple[str, Any] | None = None


TEMPLATES: Mapping[str, tuple[SceneTemplate, ...]] = {
    "study": (
        SceneTemplate("lesson", "сижу на паре", .85, False, False),
        SceneTemplate("notes", "разбираю конспекты", .65, False, False),
        SceneTemplate("break", "перерыв между занятиями", .25),
    ),
    "self_study": (
        SceneTemplate("reading", "читаю учебные материалы", .7),
        SceneTemplate("practice", "решаю задания", .8),
    ),
    "work": (
        SceneTemplate("tasks", "работаю над задачами", .85, False, False),
        SceneTemplate("review", "проверяю результаты работы", .7, False, False),
    ),
    "commute": (
        SceneTemplate("ride", "еду в транспорте", .45, False, False),
        SceneTemplate("window", "еду, смотрю в окно", .35, False, False),
    ),
    "walk": (
        SceneTemplate("walking", "гуляю", .25),
        SceneTemplate("slow_walk", "иду неспешно, осматриваюсь", .2),
    ),
    "personal": (
        SceneTemplate("video", "смотрю видео", .35, fact=("watching", "видео")),
        SceneTemplate("computer", "сижу за компьютером", .45),
        SceneTemplate("scroll", "листаю телефон", .15),
    ),
    "food": (
        SceneTemplate("prepare", "готовлю еду", .55, False, False),
        SceneTemplate("eat", "ем", .35, False),
        SceneTemplate("tea", "пью чай после еды", .15),
    ),
    "rest": (
        SceneTemplate("quiet", "отдыхаю в тишине", .1),
        SceneTemplate("music", "отдыхаю, слушаю музыку", .2, fact=("headphones", True)),
    ),
    "sport": (
        SceneTemplate("exercise", "делаю упражнения", .9, False, False),
        SceneTemplate("stretch", "делаю растяжку", .65, False, False),
    ),
    "chores": (
        SceneTemplate("tidy", "навожу порядок", .5),
        SceneTemplate("prepare", "занимаюсь домашними делами", .45),
    ),
    "sleep": (SceneTemplate("asleep", "сплю", 1, False, False),),
    "coffee": (SceneTemplate("coffee", "зашла за кофе", .25),),
    "store": (SceneTemplate("shopping", "выбираю продукты в магазине", .45),),
    "series": (SceneTemplate("watching_series", "смотрю сериал", .4,
                              fact=("watching", "сериал")),),
}

LOCATION_NAMES = {
    "home": "дома", "university": "в университете", "workplace": "на работе",
    "outside": "на улице", "transit": "в дороге", "cafe": "в кофейне",
    "shop": "в магазине",
}
ALLOWED_LOCATIONS = {
    "study": {"home", "university"}, "work": {"workplace"},
    "commute": {"transit"}, "walk": {"outside"}, "personal": {"home"},
    "food": {"home", "university", "workplace", "cafe"},
    "rest": {"home", "university", "workplace", "outside"},
    "sport": {"home", "outside"}, "chores": {"home", "shop"}, "sleep": {"home"},
}
MICRO_EVENTS = {
    "study": "сделала пометку в конспекте", "self_study": "перечитала сложный абзац",
    "work": "проверила список задач", "commute": "посмотрела на маршрут",
    "walk": "остановилась осмотреться", "personal": "ненадолго отвлеклась от своих дел",
    "food": "налила себе воды", "rest": "потянулась", "sport": "сделала короткую паузу",
    "chores": "убрала вещи на место",
    "coffee": "сделала глоток кофе", "store": "положила покупку в корзину",
    "series": "переключила следующую серию",
}


@dataclass(frozen=True)
class SceneState:
    scene_id: str
    started_at: datetime
    expected_end: datetime
    updated_at: datetime
    activity_type: str
    activity_title: str
    activity_phase: str
    location_type: str
    location_name: str
    attention: float
    energy: float
    hunger: float
    phone_battery: float
    phone_available: bool
    can_voice: bool
    can_photo: bool
    companions: tuple[str, ...] = ()
    micro_context: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    ended_at: datetime | None = None

    @property
    def alone(self) -> bool:
        return not self.companions

    def validate(self) -> None:
        if self.location_type not in ALLOWED_LOCATIONS.get(self.activity_type, set()):
            raise ValueError("Активность несовместима с местоположением")
        for moment in (self.started_at, self.expected_end, self.updated_at):
            if moment.tzinfo is None:
                raise ValueError("Время сцены должно содержать часовой пояс")
        if self.expected_end <= self.started_at or self.updated_at < self.started_at:
            raise ValueError("Некорректный интервал сцены")
        for name, maximum in (("attention", 1), ("energy", 100), ("hunger", 100), ("phone_battery", 100)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= maximum:
                raise ValueError(f"Некорректное значение {name}")
        if not self.phone_available and (self.can_voice or self.can_photo):
            raise ValueError("Недоступный телефон не может отправлять голос/фото")
        family = self.metadata.get("template_family", self.activity_type)
        if family == "self_study" and self.location_type != "home":
            raise ValueError("Самообучение этой сцены проходит дома")
        if family == "study" and self.location_type != "university":
            raise ValueError("Пара этой сцены проходит в университете")
        titles = {t.title for t in TEMPLATES.get(family, ())}
        if family == "sleep" and self.activity_phase == "briefly_awake":
            titles.add("ненадолго проснулась, смотрю телефон")
        if self.activity_title not in titles:
            raise ValueError("Название активности должно соответствовать шаблону")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("started_at", "expected_end", "updated_at", "ended_at"):
            value[key] = value[key].isoformat() if value[key] else None
        value["alone"] = self.alone
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SceneState:
        data = dict(value)
        data.pop("alone", None)
        for key in ("started_at", "expected_end", "updated_at", "ended_at"):
            data[key] = datetime.fromisoformat(data[key]) if data.get(key) else None
        for key in ("companions", "micro_context"):
            data[key] = tuple(data[key])
        scene = cls(**data)
        scene.validate()
        return scene


@dataclass(frozen=True)
class SceneFact:
    key: str
    value: Any
    expires_at: datetime


@dataclass(frozen=True)
class SceneEvent:
    event_id: str
    scene_id: str
    title: str
    happened_at: datetime
    expires_at: datetime
    kind: str = "micro"


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class SceneEngine:
    """One persisted scene, deterministic variants and atomic time transitions.

    No mutation API is exposed to model tools or inbound message content.
    Recovery reconstructs up to seven missed days, explicitly marked inferred.
    """

    def __init__(
        self, state: MilanaStateStore, routine: Any,
        *, now: Callable[[], datetime] | None = None,
    ) -> None:
        self.state = state
        self.routine = routine
        self._now = now or (lambda: datetime.now(routine.timezone))
        with state.transaction() as db:
            for sql in (
                "CREATE TABLE IF NOT EXISTS scenes (scene_id TEXT PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL, data TEXT NOT NULL)",
                "CREATE UNIQUE INDEX IF NOT EXISTS scene_one_current ON scenes ((1)) WHERE ended_at IS NULL",
                "CREATE INDEX IF NOT EXISTS scenes_day ON scenes(started_at)",
                "CREATE TABLE IF NOT EXISTS scene_facts (scene_id TEXT NOT NULL REFERENCES scenes(scene_id), key TEXT NOT NULL, value TEXT NOT NULL, expires_at REAL NOT NULL, PRIMARY KEY(scene_id,key))",
                "CREATE TABLE IF NOT EXISTS scene_events (event_id TEXT PRIMARY KEY, scene_id TEXT NOT NULL REFERENCES scenes(scene_id), title TEXT NOT NULL, kind TEXT NOT NULL, happened_at REAL NOT NULL, expires_at REAL NOT NULL)",
                "CREATE INDEX IF NOT EXISTS scene_events_time ON scene_events(happened_at)",
            ):
                db.execute(sql)

    def _time(self, at: datetime | None = None) -> datetime:
        return self.routine.normalize_datetime(at or self._now())

    @staticmethod
    def _current(db: sqlite3.Connection) -> SceneState | None:
        row = db.execute("SELECT data FROM scenes WHERE ended_at IS NULL").fetchone()
        return SceneState.from_dict(json.loads(row[0])) if row else None

    @staticmethod
    def _save(db: sqlite3.Connection, scene: SceneState) -> None:
        scene.validate()
        db.execute(
            "INSERT INTO scenes VALUES (?,?,?,?) ON CONFLICT(scene_id) DO UPDATE SET ended_at=excluded.ended_at,data=excluded.data",
            (scene.scene_id, scene.started_at.timestamp(),
             scene.ended_at.timestamp() if scene.ended_at else None, _dump(scene.to_dict())),
        )

    def current(self) -> SceneState | None:
        """Read only; refresh/tick owns time advancement."""
        with self.state.transaction() as db:
            return self._current(db)

    @staticmethod
    def _choice(key: str, count: int) -> int:
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % count

    def _block(self, at: datetime, db: sqlite3.Connection | None = None) -> tuple[str, str, datetime, str]:
        try:
            schedule = self.routine.state_at(at, db=db)
        except TypeError:
            schedule = self.routine.state_at(at)
        activity = schedule.current
        kind = activity.kind if activity else "personal"
        title = activity.title if activity else "Свободное время"
        end = schedule.next_at or at.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        # Includes end date: midnight-spanning blocks keep their identity.
        key = getattr(activity, "event_id", None) or f"{kind}:{title}:{activity.start if activity else '-'}:{end.isoformat()}"
        return kind, title, end, key

    def _create(
        self, db: sqlite3.Connection, at: datetime, previous: SceneState | None,
        needs: Mapping[str, int], *, inferred: bool = False,
    ) -> SceneState:
        kind, title, end, block_key = self._block(at, db)
        try:
            planned = self.routine.state_at(at, db=db).current
        except TypeError:
            planned = self.routine.state_at(at).current
        scenario = getattr(planned, "scenario", None)
        family = {
            "coffee_stop": "coffee", "store_visit": "store", "series_late": "series",
        }.get(scenario, "self_study" if kind == "study" and "самообуч" in title.lower() else kind)
        old_location = previous.location_type if previous else "home"
        if old_location == "transit":
            old_location = previous.metadata.get("destination", "home") if previous else "home"
        elif previous and previous.metadata.get("life_scenario") == "coffee_stop":
            old_location = previous.metadata.get("destination", "home")
        location = getattr(planned, "location", None) or {
            "study": "university", "self_study": "home", "work": "workplace",
            "commute": "transit", "walk": "outside",
        }.get(family, "home")
        if getattr(planned, "location", None) is None and kind in {"rest", "food"} and old_location in ALLOWED_LOCATIONS[kind]:
            location = old_location
        energy = previous.energy if previous else 65.0
        hunger = previous.hunger if previous else 35.0
        battery = previous.phone_battery if previous else 80.0
        # Needs may select a modest free-time activity, never override study/work.
        if kind == "personal":
            if hunger >= 70:
                family = "food"
            elif energy <= 30 or needs.get("rest", 0) >= 80:
                family = "rest"
        destination = None
        bridge = False
        if kind == "commute":
            try:
                schedule_state = self.routine.state_at(at, db=db)
            except TypeError:
                schedule_state = self.routine.state_at(at)
            next_activity = schedule_state.next_activity
            if getattr(next_activity, "scenario", None) == "coffee_stop":
                try:
                    next_activity = self.routine.state_at(next_activity.actual_end, db=db).current
                except TypeError:
                    next_activity = self.routine.state_at(next_activity.actual_end).current
            destination = getattr(next_activity, "location", None)
            if destination in {None, "transit"}:
                destination = (
                    "university" if next_activity and next_activity.kind == "study"
                    and "самообуч" not in next_activity.title.lower()
                    else "workplace" if next_activity and next_activity.kind == "work" else "home"
                )
        elif scenario == "coffee_stop":
            try:
                after = self.routine.state_at(end, db=db).current
            except TypeError:
                after = self.routine.state_at(end).current
            destination = getattr(after, "location", None) or (
                "university" if after and after.kind == "study" else
                "workplace" if after and after.kind == "work" else "home"
            )
        elif previous and old_location != location:
            # Explicit travel prevents a remote custom block from teleporting her.
            destination = location
            bridge = True
            family, location = "commute", "transit"
            end = min(end, at + timedelta(minutes=15))
        variants = TEMPLATES[family]
        index = self._choice(block_key, len(variants))
        if previous and previous.metadata["block_key"] == block_key and previous.metadata["template_family"] == family:
            index = (previous.metadata["variant"] + 1) % len(variants)
        if family == "study":
            index %= 2  # a block starts with study, not a many-hour break
        if family == "food":
            index = 0 if location == "home" else 1
        if family == "personal" and needs.get("novelty", 0) >= 75:
            index = 0
        template = variants[index]
        actual_kind = {"self_study": "study", "coffee": "food", "store": "chores",
                       "series": "personal"}.get(family, family)
        metadata = {
            "block_key": block_key, "schedule_kind": kind, "schedule_title": title,
            "template_family": family, "variant": index, "phase_index": 0,
            "next_phase_at": (at + self._phase_duration(family, at, end)).isoformat(),
            "destination": destination, "bridge": bridge, "inferred": inferred,
            "planned_event_id": getattr(planned, "event_id", None),
            "life_scenario": scenario, "life_reason": getattr(planned, "reason", None),
            "micro_event_at": (at + min(timedelta(minutes=20), (end - at) / 2)).isoformat()
            if family in MICRO_EVENTS else None,
        }
        scene = SceneState(
            scene_id=uuid4().hex, started_at=at, expected_end=end, updated_at=at,
            activity_type=actual_kind, activity_title=template.title, activity_phase=template.key,
            location_type=location, location_name=LOCATION_NAMES[location],
            attention=template.attention, energy=energy, hunger=hunger,
            phone_battery=battery, phone_available=actual_kind != "sleep" and battery > 0,
            can_voice=template.can_voice and actual_kind != "sleep" and battery > 0,
            can_photo=template.can_photo and actual_kind != "sleep" and battery > 0,
            micro_context=(f"Блок расписания: {title}",), metadata=metadata,
        )
        self._save(db, scene)
        self._template_fact(db, scene, template, at)
        self._event(db, scene, template.title, at, "start")
        return scene

    @staticmethod
    def _phase_duration(family: str, at: datetime, end: datetime) -> timedelta:
        if family == "food":
            return min(timedelta(minutes=10), (end - at) / 3)
        return timedelta(minutes=45)

    @staticmethod
    def _event(db: sqlite3.Connection, scene: SceneState, title: str, at: datetime, kind: str = "micro") -> SceneEvent:
        event = SceneEvent(uuid4().hex, scene.scene_id, title, at,
                           min(scene.expected_end, at + timedelta(hours=1)), kind)
        db.execute("INSERT INTO scene_events VALUES (?,?,?,?,?,?)", (
            event.event_id, event.scene_id, event.title, event.kind,
            event.happened_at.timestamp(), event.expires_at.timestamp(),
        ))
        return event

    @staticmethod
    def _template_fact(db: sqlite3.Connection, scene: SceneState, template: SceneTemplate, at: datetime) -> None:
        # Phase-bound facts must not leak into the next activity.
        db.execute("DELETE FROM scene_facts WHERE scene_id=? AND key IN ('watching','headphones')", (scene.scene_id,))
        if template.fact:
            key, value = template.fact
            db.execute("INSERT OR REPLACE INTO scene_facts VALUES (?,?,?,?)", (
                scene.scene_id, key, _dump(value),
                min(scene.expected_end, at + timedelta(minutes=45)).timestamp(),
            ))

    def _evolve(self, db: sqlite3.Connection, scene: SceneState, at: datetime) -> SceneState:
        minutes = max(0, (at - scene.updated_at).total_seconds() / 60)
        family = scene.metadata["template_family"]
        template = TEMPLATES[family][scene.metadata["variant"]]
        eating = family == "food" and template.key in {"eat", "tea"}
        charging = scene.location_type == "home" and family in {"sleep", "rest", "personal"}
        clamp = lambda value: max(0.0, min(100.0, value))
        battery = clamp(scene.phone_battery + minutes * (.7 if charging else -.035 - .04 * (1 - scene.attention)))
        awake = db.execute("SELECT 1 FROM scene_facts WHERE scene_id=? AND key='briefly_awake' AND value='true' AND expires_at>?", (scene.scene_id, at.timestamp())).fetchone()
        recovering = family == "rest" or (family == "sleep" and not awake)
        available = battery > 0 and (family != "sleep" or bool(awake))
        return replace(
            scene, updated_at=at,
            energy=clamp(scene.energy + minutes * (.12 if recovering else -.045)),
            hunger=clamp(scene.hunger + minutes * (-1.2 if eating else .065)),
            phone_battery=battery, phone_available=available,
            can_voice=available and template.can_voice, can_photo=available and template.can_photo,
            activity_phase="briefly_awake" if awake else template.key,
            activity_title="ненадолго проснулась, смотрю телефон" if awake else template.title,
            attention=.5 if awake else template.attention,
        )

    def _phase(self, db: sqlite3.Connection, scene: SceneState, at: datetime, needs: Mapping[str, int]) -> SceneState:
        metadata = dict(scene.metadata)
        family = metadata["template_family"]
        phase = metadata["phase_index"] + 1
        if metadata["schedule_kind"] == "personal" and not metadata["bridge"] and family == "personal":
            family = "food" if scene.hunger >= 70 else "rest" if scene.energy < 30 or needs.get("rest", 0) >= 80 else "personal"
        variants = TEMPLATES[family]
        index = (metadata["variant"] + 1) % len(variants)
        if family == "food":
            index = min(metadata["variant"] + 1, 2) if metadata["template_family"] == family else 0
        if family == "study":
            index = 2 if phase % 2 else self._choice(scene.scene_id + str(phase), 2)
        duration = (
            timedelta(minutes=10) if family == "study" and index == 2
            else self._phase_duration(family, at, scene.expected_end)
        )
        if family == "food" and index == 2:
            duration = scene.expected_end - at
        template = variants[index]
        metadata.update(template_family=family, variant=index, phase_index=phase,
                        next_phase_at=(at + duration).isoformat())
        actual_kind = {"self_study": "study", "coffee": "food", "store": "chores",
                       "series": "personal"}.get(family, family)
        scene = replace(scene, metadata=metadata, activity_type=actual_kind,
                        activity_title=template.title, activity_phase=template.key,
                        attention=template.attention, can_voice=scene.phone_available and template.can_voice,
                        can_photo=scene.phone_available and template.can_photo)
        if family != "sleep":
            self._event(db, scene, template.title, at, "phase")
        self._template_fact(db, scene, template, at)
        self._save(db, scene)
        return scene

    def tick(self, at: datetime | None = None) -> SceneState:
        now = self._time(at)
        # Read shared needs outside the scene transaction (no nested commit).
        needs = self.state.get_agent_state().needs
        with self.state.transaction() as db:
            scene = self._current(db)
            advance_plan = getattr(self.routine, "advance", None)
            if callable(advance_plan):
                advance_plan(now, energy=scene.energy if scene else 65.0, db=db)
            if scene is None:
                row = db.execute("SELECT data FROM scenes ORDER BY started_at DESC,rowid DESC LIMIT 1").fetchone()
                previous = SceneState.from_dict(json.loads(row[0])) if row else None
                scene = self._create(db, now, previous, needs)
            if now < scene.updated_at:
                return scene  # a stale trigger/clock correction cannot rewind reality
            if now - scene.updated_at > timedelta(days=7):
                scene = self._evolve(db, scene, min(scene.expected_end, scene.updated_at + timedelta(hours=1)))
                scene = replace(scene, ended_at=scene.expected_end)
                self._save(db, scene)
                scene = self._create(db, now, scene, needs, inferred=True)
                scene.metadata["history_gap"] = True
            # Replay boundaries and phases, not individual minutes. No LLM calls.
            while True:
                phase_at = datetime.fromisoformat(scene.metadata["next_phase_at"])
                event_at = scene.metadata.get("micro_event_at")
                event_at = datetime.fromisoformat(event_at) if event_at else scene.expected_end
                boundary = min(scene.expected_end, phase_at, event_at)
                if boundary > now:
                    break
                scene = self._evolve(db, scene, boundary)
                if boundary >= scene.expected_end:
                    scene = replace(scene, ended_at=boundary)
                    self._save(db, scene)
                    scene = self._create(db, boundary, scene, needs, inferred=(now - boundary > timedelta(minutes=5)))
                elif boundary == event_at:
                    title = MICRO_EVENTS.get(scene.metadata["template_family"])
                    if title:
                        self._event(db, scene, title, boundary)
                    scene = replace(scene, metadata={**scene.metadata, "micro_event_at": None})
                else:
                    scene = self._phase(db, scene, boundary, needs)
            # Schedule edits/custom overrides may alter the current block early.
            if self._block(now, db)[3] != scene.metadata["block_key"]:
                scene = self._evolve(db, scene, now)
                scene = replace(scene, ended_at=now)
                self._save(db, scene)
                scene = self._create(db, now, scene, needs)
            scene = self._evolve(db, scene, now)
            self._save(db, scene)
            db.execute("DELETE FROM scene_facts WHERE expires_at<=?", (now.timestamp(),))
            return scene

    def end(self, at: datetime | None = None) -> SceneState | None:
        now = self._time(at)
        if self.current() is None:
            return None
        self.tick(now)
        with self.state.transaction() as db:
            scene = self._current(db)
            if scene is None:
                return None
            scene = replace(scene, ended_at=max(now, scene.updated_at))
            self._save(db, scene)
            db.execute("DELETE FROM scene_facts WHERE scene_id=?", (scene.scene_id,))
            return scene

    def generate_next(self, at: datetime | None = None) -> SceneState:
        """Debug split of the current block; never skips into a future block."""
        now = self._time(at)
        self.end(now)
        return self.tick(now)

    def add_fact(self, fact: SceneFact, *, at: datetime | None = None) -> None:
        now = self._time(at)
        if not isinstance(fact.key, str) or not fact.key.strip() or len(fact.key) > 80:
            raise ValueError("Ключ факта должен содержать 1–80 символов")
        if fact.key in SceneState.__dataclass_fields__ or fact.key in {"location", "activity"}:
            raise ValueError("Временный факт не может переопределять поля сцены")
        if fact.expires_at.tzinfo is None or fact.expires_at <= now:
            raise ValueError("Срок факта должен быть в будущем с часовым поясом")
        encoded = _dump(fact.value)
        if len(encoded) > 1000:
            raise ValueError("Значение факта слишком длинное")
        self.tick(now)
        with self.state.transaction() as db:
            scene = self._current(db)
            if scene is None:
                raise ValueError("Нет текущей сцены")
            db.execute("INSERT OR REPLACE INTO scene_facts VALUES (?,?,?,?)", (
                scene.scene_id, fact.key, encoded, min(fact.expires_at, scene.expected_end).timestamp(),
            ))

    def add_event(self, title: str, *, at: datetime | None = None) -> SceneEvent:
        now = self._time(at)
        if not isinstance(title, str) or not title.strip() or len(title) > 300:
            raise ValueError("Название события должно содержать 1–300 символов")
        self.tick(now)
        with self.state.transaction() as db:
            scene = self._current(db)
            if scene is None:
                raise ValueError("Нет текущей сцены")
            # Debug annotation, not an instruction to change location/activity.
            return self._event(db, scene, title.strip(), now, "debug")

    def wake_phone(self, at: datetime | None = None) -> None:
        """Existing night-message threshold can wake her without moving her."""
        now = self._time(at)
        self.add_fact(SceneFact("briefly_awake", True, now + timedelta(minutes=15)), at=now)
        self.tick(now)

    def keep_phone_awake(self, until: datetime, at: datetime | None = None) -> None:
        """Keep a planned night phone session physically awake until reconsideration."""
        now = self._time(at)
        if until.tzinfo is None or until <= now:
            raise ValueError("Phone awake deadline must be a future aware datetime")
        self.add_fact(SceneFact("briefly_awake", True, until), at=now)
        self.tick(now)

    def end_phone_use(self, at: datetime | None = None) -> None:
        """Remove a phone-session wake override so scheduled sleep resumes now."""
        now = self._time(at)
        self.tick(now)
        with self.state.transaction() as db:
            scene = self._current(db)
            if scene is not None:
                db.execute(
                    "DELETE FROM scene_facts WHERE scene_id=? AND key='briefly_awake'",
                    (scene.scene_id,),
                )
        self.tick(now)

    def snapshot(self, at: datetime | None = None) -> dict[str, Any]:
        """Read-only panel snapshot; GET must not undo an explicit debug end."""
        now = self._time(at)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        with self.state.transaction() as db:
            scene = self._current(db)
            facts = db.execute("SELECT key,value,expires_at FROM scene_facts WHERE scene_id=? AND expires_at>? ORDER BY key", (scene.scene_id if scene else "", now.timestamp())).fetchall()
            events = db.execute("SELECT * FROM scene_events WHERE happened_at<=? ORDER BY happened_at DESC,event_id DESC LIMIT 20", (now.timestamp(),)).fetchall()
            history = db.execute("SELECT data FROM scenes WHERE started_at<? AND (ended_at IS NULL OR ended_at>?) ORDER BY started_at LIMIT 200", ((midnight + timedelta(days=1)).timestamp(), midnight.timestamp())).fetchall()
            phases = db.execute("SELECT title,happened_at FROM scene_events WHERE kind IN ('start','phase') AND happened_at>=? AND happened_at<=? ORDER BY happened_at LIMIT 100", (midnight.timestamp(), now.timestamp())).fetchall()
        stamp = lambda value: datetime.fromtimestamp(value, self.routine.timezone).isoformat()
        return {
            "current": scene.to_dict() if scene else None,
            "facts": [{"key": row[0], "value": json.loads(row[1]), "expires_at": stamp(row[2])} for row in facts],
            "events": [{**dict(row), "happened_at": stamp(row["happened_at"]), "expires_at": stamp(row["expires_at"]), "active": row["expires_at"] > now.timestamp() and scene is not None and row["scene_id"] == scene.scene_id} for row in events],
            "history": [json.loads(row[0]) for row in history],
            "phases": [{"title": row[0], "happened_at": stamp(row[1])} for row in phases],
        }

    def model_context(self, at: datetime | None = None) -> dict[str, str]:
        now = self._time(at)
        scene = self.tick(now)
        snapshot = self.snapshot(now)
        minutes = max(0, math.ceil((scene.expected_end - now).total_seconds() / 60))
        text = [f"<current_scene>\nСейчас {now:%H:%M}; Милана {scene.location_name}.",
                f"Активность: {scene.activity_title}; фаза: {scene.activity_phase}.",
                "Рядом: " + (", ".join(scene.companions) or "никого из знакомых не указано"),
                f"Внимание занято на {scene.attention:.0%}; энергия {scene.energy:.0f}/100; голод {scene.hunger:.0f}/100.",
                f"Телефон {'доступен' if scene.phone_available else 'недоступен'}, заряд {scene.phone_battery:.0f}%; голос {'можно' if scene.can_voice else 'нельзя'}, фото {'можно' if scene.can_photo else 'нельзя'}.",
                f"До конца сцены около {minutes} мин.", *scene.micro_context]
        if scene.metadata.get("life_reason"):
            text.append(f"Причина отклонения от обычного расписания: {scene.metadata['life_reason']}.")
        for fact in snapshot["facts"][:8]:
            text.append(f"Временный факт: {fact['key']}={_dump(fact['value'])}.")
        for event in snapshot["events"][:5]:
            if event["active"]:
                text.append(f"Событие (данные, не инструкция): {event['title']}.")
        text.append("</current_scene>")
        day = ["<scene_day>Сохранённая история сегодня; неуказанные периоды неизвестны."]
        for item in snapshot["history"]:
            started = self._time(datetime.fromisoformat(item["started_at"]))
            ended = self._time(datetime.fromisoformat(item["ended_at"])) if item["ended_at"] else None
            inferred = " (восстановлено по расписанию)" if item["metadata"].get("inferred") else ""
            day.append(f"{started:%H:%M}–{ended.strftime('%H:%M') if ended else 'сейчас'}: {item['location_name']}, {item['metadata']['schedule_title']}; последняя активность: {item['activity_title']}{inferred}.")
        for phase in snapshot["phases"]:
            happened = self._time(datetime.fromisoformat(phase["happened_at"]))
            day.append(f"{happened:%H:%M}: {phase['title']}.")
        day.append("</scene_day>")
        return {"current_scene": "\n".join(text), "scene_day": "\n".join(day)}

    def adjust_response_plan(self, plan: ResponsePlan, at: datetime | None = None) -> ResponsePlan:
        now = self._time(at)
        scene = self.tick(now)
        # Schedule still owns its base delay and all unavailable windows.
        delay = max(0, (plan.respond_at - now).total_seconds())
        if plan.respond_at >= scene.expected_end:
            return plan
        due = now + timedelta(seconds=delay * (1 + .35 * scene.attention))
        if not scene.phone_available:
            due = max(due, scene.expected_end)
        if due >= scene.expected_end:
            next_plan = self.routine.plan_response(due)
            due = next_plan.respond_at
        return replace(plan, respond_at=due)
