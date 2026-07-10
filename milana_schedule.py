"""Недельное расписание Миланы и расчёт её текущего состояния."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_DIR = Path(__file__).resolve().parent
SCHEDULE_PATH = BASE_DIR / "milana_schedule.json"

DAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DAY_NAMES = {
    "mon": "Понедельник",
    "tue": "Вторник",
    "wed": "Среда",
    "thu": "Четверг",
    "fri": "Пятница",
    "sat": "Суббота",
    "sun": "Воскресенье",
}
ACTIVITY_NAMES = {
    "sleep": "Сон",
    "study": "Учёба",
    "work": "Работа",
    "walk": "Прогулка",
    "personal": "Личные дела",
    "food": "Еда",
    "sport": "Спорт",
    "chores": "Быт",
    "commute": "Дорога",
    "rest": "Перерыв",
}
ACTIVITY_EFFECTS = {
    "sleep": "восстановление энергии",
    "study": "рост навыков",
    "work": "результат и нагрузка",
    "walk": "снижение стресса",
    "personal": "баланс и отдых",
    "food": "поддержка энергии",
    "sport": "здоровье и тонус",
    "chores": "порядок",
    "commute": "затраты времени",
    "rest": "восстановление внимания",
}
VALID_MODES = {"student", "worker", "balanced", "loaded"}


@dataclass(frozen=True)
class DaySettings:
    wake: int
    sleep: int
    main_hours: float
    walk_hours: float
    personal_hours: float
    commute_minutes: int
    breaks_minutes: int


@dataclass(frozen=True)
class Activity:
    title: str
    kind: str
    start: int
    end: int
    duration: int
    custom: bool = False

    def contains(self, minute: int) -> bool:
        if self.duration <= 0:
            return False
        if self.end > self.start:
            return self.start <= minute < self.end
        return minute >= self.start or minute < self.end


@dataclass(frozen=True)
class DayMetrics:
    energy: int
    stress: int
    productivity: int
    balance: int
    sleep: float
    main: float
    walk: float
    personal: float
    sport: float
    commute: float
    free: float


@dataclass(frozen=True)
class WeekMetrics:
    balance: int
    energy: int
    stress: int
    productivity: int
    sleep: float
    free: float
    main: float
    walk: float
    personal: float
    sport: float


@dataclass(frozen=True)
class ScheduleState:
    now: datetime
    day_key: str
    current: Activity | None
    next_activity: Activity | None
    next_at: datetime | None
    metrics: DayMetrics


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} должен быть объектом")
    return value


def _number(
    data: Mapping[str, Any],
    key: str,
    label: str,
    minimum: float,
    maximum: float,
) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} должен быть числом")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{label} должен быть от {minimum:g} до {maximum:g}")
    return number


def time_to_minutes(value: Any, label: str = "время") -> int:
    if not isinstance(value, str):
        raise ValueError(f"{label} должно быть строкой ЧЧ:ММ")
    parts = value.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"{label} должно быть записано как ЧЧ:ММ")
    hours, minutes = map(int, parts)
    if not 0 <= hours <= 23 or not 0 <= minutes <= 59:
        raise ValueError(f"{label} содержит недопустимое время: {value}")
    return hours * 60 + minutes


def minutes_to_time(value: int) -> str:
    value %= 24 * 60
    return f"{value // 60:02d}:{value % 60:02d}"


def duration_between(start: int, end: int) -> int:
    return end - start if end >= start else 24 * 60 - start + end


def parse_utc_offset(value: Any) -> timezone:
    if (
        not isinstance(value, str)
        or len(value) != 6
        or value[0] not in "+-"
        or value[3] != ":"
    ):
        raise ValueError("timezone.utc_offset должен иметь вид +05:00")
    try:
        hours = int(value[1:3])
        minutes = int(value[4:6])
    except ValueError as exc:
        raise ValueError("timezone.utc_offset должен иметь вид +05:00") from exc
    if hours > 14 or minutes > 59 or (hours == 14 and minutes):
        raise ValueError("timezone.utc_offset выходит за допустимый диапазон")
    total = hours * 60 + minutes
    if value[0] == "-":
        total = -total
    return timezone(timedelta(minutes=total))


def _load_day_settings(
    data: Mapping[str, Any], label: str, *, weekend: bool
) -> DaySettings:
    wake = time_to_minutes(data.get("wake"), f"{label}.wake")
    sleep = time_to_minutes(data.get("sleep"), f"{label}.sleep")
    return DaySettings(
        wake=wake,
        sleep=sleep,
        main_hours=_number(data, "main_hours", f"{label}.main_hours", 0, 12),
        walk_hours=_number(
            data, "walk_hours", f"{label}.walk_hours", 0, 6 if weekend else 4
        ),
        personal_hours=_number(
            data,
            "personal_hours",
            f"{label}.personal_hours",
            0,
            10 if weekend else 8,
        ),
        commute_minutes=int(
            _number(data, "commute_minutes", f"{label}.commute_minutes", 0, 240)
        ),
        breaks_minutes=int(
            _number(data, "breaks_minutes", f"{label}.breaks_minutes", 0, 240)
        ),
    )


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


def _round_hour(value: float) -> float:
    return math.floor(value * 10 + 0.5) / 10


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def calculate_day_metrics(activities: Sequence[Activity]) -> DayMetrics:
    totals: dict[str, int] = {}
    for activity in activities:
        totals[activity.kind] = totals.get(activity.kind, 0) + activity.duration

    sleep = totals.get("sleep", 0) / 60
    main = (totals.get("study", 0) + totals.get("work", 0)) / 60
    walk = totals.get("walk", 0) / 60
    personal = totals.get("personal", 0) / 60
    sport = totals.get("sport", 0) / 60
    commute = totals.get("commute", 0) / 60
    rest = totals.get("rest", 0) / 60
    chores = totals.get("chores", 0) / 60
    busy = sum(activity.duration for activity in activities)
    free = max(0.0, 24 - busy / 60)

    energy = (
        52
        + min(35, sleep * 4.5)
        + min(10, sport * 4)
        + min(8, walk * 3)
        + min(8, rest * 2)
        - max(0, main - 6) * 5
        - commute * 4
        - max(0, 7 - sleep) * 8
    )
    stress = (
        28
        + max(0, main - 6) * 8
        + commute * 6
        + max(0, chores - 1) * 3
        - walk * 8
        - personal * 4
        - rest * 2
        + max(0, 7 - sleep) * 9
    )
    energy = _clamp(energy, 0, 100)
    stress = _clamp(stress, 0, 100)
    productivity = (
        35
        + min(45, main * 6)
        + min(10, sleep)
        + min(8, rest * 2)
        - max(0, stress - 60) * 0.45
        - max(0, main - 9) * 4
    )
    productivity = _clamp(productivity, 0, 100)
    balance = _js_round(
        energy * 0.35
        + (100 - stress) * 0.25
        + productivity * 0.25
        + min(100, (walk + personal + sport) * 12) * 0.15
    )
    return DayMetrics(
        energy=_js_round(energy),
        stress=_js_round(stress),
        productivity=_js_round(productivity),
        balance=balance,
        sleep=_round_hour(sleep),
        main=_round_hour(main),
        walk=_round_hour(walk),
        personal=_round_hour(personal),
        sport=_round_hour(sport),
        commute=_round_hour(commute),
        free=_round_hour(free),
    )


class WeeklyRoutine:
    """Сгенерированная неделя по правилам HTML-шаблона."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        mode = config.get("mode")
        if not isinstance(mode, str) or mode not in VALID_MODES:
            raise ValueError(
                "mode должен быть одним из: student, worker, balanced, loaded"
            )
        self.mode = str(mode)

        timezone_data = _mapping(config.get("timezone"), "timezone")
        self.timezone = parse_utc_offset(timezone_data.get("utc_offset"))
        timezone_name = timezone_data.get("name", "UTC")
        if not isinstance(timezone_name, str) or not timezone_name.strip():
            raise ValueError("timezone.name должен быть непустой строкой")
        self.timezone_name = timezone_name.strip()

        self.weekday = _load_day_settings(
            _mapping(config.get("weekday"), "weekday"), "weekday", weekend=False
        )
        self.weekend = _load_day_settings(
            _mapping(config.get("weekend"), "weekend"), "weekend", weekend=True
        )
        self.sport_hours = _number(
            config, "sport_hours", "sport_hours", 0, 3
        )
        self.custom_events = self._load_custom_events(
            config.get("custom_events", {})
        )
        self.days = {
            day_key: self._generate_day(day_key)
            for day_key in DAY_KEYS
        }

    @staticmethod
    def _load_custom_events(value: Any) -> Mapping[str, tuple[Activity, ...]]:
        data = _mapping(value, "custom_events")
        unknown_days = set(data) - set(DAY_KEYS)
        if unknown_days:
            raise ValueError(
                "Неизвестные дни в custom_events: " + ", ".join(sorted(unknown_days))
            )

        result: dict[str, tuple[Activity, ...]] = {}
        for day_key in DAY_KEYS:
            raw_events = data.get(day_key, [])
            if not isinstance(raw_events, list):
                raise ValueError(f"custom_events.{day_key} должен быть списком")
            events: list[Activity] = []
            for index, raw_event in enumerate(raw_events, start=1):
                label = f"custom_events.{day_key}[{index}]"
                event = _mapping(raw_event, label)
                title = event.get("title")
                kind = event.get("type")
                if not isinstance(title, str) or not title.strip():
                    raise ValueError(f"{label}.title должен быть непустой строкой")
                if not isinstance(kind, str) or kind not in ACTIVITY_NAMES:
                    raise ValueError(
                        f"{label}.type должен быть одним из: "
                        + ", ".join(ACTIVITY_NAMES)
                    )
                start = time_to_minutes(event.get("start"), f"{label}.start")
                end = time_to_minutes(event.get("end"), f"{label}.end")
                duration = duration_between(start, end)
                if duration:
                    events.append(
                        Activity(
                            title=title.strip(),
                            kind=str(kind),
                            start=start,
                            end=end,
                            duration=duration,
                            custom=True,
                        )
                    )
            result[day_key] = tuple(events)
        return result

    @staticmethod
    def _add(
        activities: list[Activity],
        title: str,
        kind: str,
        start: int,
        duration: float,
    ) -> None:
        rounded_duration = _js_round(duration)
        if rounded_duration <= 0:
            return
        normalized_start = start % (24 * 60)
        activities.append(
            Activity(
                title=title,
                kind=kind,
                start=normalized_start,
                end=(normalized_start + rounded_duration) % (24 * 60),
                duration=rounded_duration,
            )
        )

    def _generate_day(self, day_key: str) -> tuple[Activity, ...]:
        is_weekend = day_key in {"sat", "sun"}
        settings = self.weekend if is_weekend else self.weekday
        main_kind = "work" if self.mode == "worker" else "study"

        main_minutes = settings.main_hours * 60
        commute_minutes = settings.commute_minutes
        breaks_minutes = settings.breaks_minutes

        activities: list[Activity] = []
        self._add(
            activities,
            "Ночной сон",
            "sleep",
            settings.sleep,
            duration_between(settings.sleep, settings.wake),
        )
        cursor = settings.wake
        morning_duration = 45 if is_weekend else 35
        self._add(activities, "Утренние сборы", "chores", cursor, morning_duration)
        cursor += morning_duration
        self._add(activities, "Завтрак", "food", cursor, 25)
        cursor += 25
        if commute_minutes:
            self._add(activities, "Дорога", "commute", cursor, commute_minutes)
            cursor += commute_minutes
        if main_minutes:
            main_title = (
                "Работа"
                if main_kind == "work"
                else "Самообучение" if is_weekend else "Учёба"
            )
            self._add(activities, main_title, main_kind, cursor, main_minutes)
            cursor += _js_round(main_minutes)
        if breaks_minutes:
            self._add(
                activities, "Перерывы и отдых", "rest", cursor, breaks_minutes
            )
            cursor += breaks_minutes
        if commute_minutes and not is_weekend:
            self._add(
                activities, "Дорога домой", "commute", cursor, commute_minutes
            )
            cursor += commute_minutes
        meal_duration = 45 if is_weekend else 40
        self._add(
            activities,
            "Обед" if is_weekend else "Обед / ужин",
            "food",
            cursor,
            meal_duration,
        )
        cursor += meal_duration
        walk_minutes = settings.walk_hours * 60
        if walk_minutes:
            self._add(
                activities,
                "Длинная прогулка" if is_weekend else "Прогулка",
                "walk",
                cursor,
                walk_minutes,
            )
            cursor += _js_round(walk_minutes)
        sport_minutes = self.sport_hours * 60
        if sport_minutes:
            self._add(activities, "Спорт", "sport", cursor, sport_minutes)
            cursor += _js_round(sport_minutes)
        personal_minutes = settings.personal_hours * 60
        if personal_minutes:
            self._add(
                activities,
                "Личные дела и отдых" if is_weekend else "Личные дела",
                "personal",
                cursor,
                personal_minutes,
            )
            cursor += _js_round(personal_minutes)
        free_minutes = duration_between(cursor % (24 * 60), settings.sleep)
        if 20 < free_minutes < 720:
            self._add(
                activities, "Свободное время", "personal", cursor, free_minutes
            )

        activities.extend(self.custom_events[day_key])
        activities.sort(key=lambda activity: (activity.start, -activity.duration))
        return tuple(activities)

    def activity_at(self, day_key: str, minute: int) -> Activity | None:
        matches = [
            (index, activity)
            for index, activity in enumerate(self.days[day_key])
            if activity.contains(minute)
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda item: (item[1].custom, item[1].start, item[0]),
        )[1]

    def normalize_datetime(self, value: datetime | None = None) -> datetime:
        if value is None:
            return datetime.now(self.timezone)
        if value.tzinfo is None:
            return value.replace(tzinfo=self.timezone)
        return value.astimezone(self.timezone)

    def state_at(self, value: datetime | None = None) -> ScheduleState:
        now = self.normalize_datetime(value)
        day_key = DAY_KEYS[now.weekday()]
        current = self.activity_at(day_key, now.hour * 60 + now.minute)
        current_identity = _activity_identity(current)
        base = now.replace(second=0, microsecond=0)
        next_activity: Activity | None = None
        next_at: datetime | None = None

        for offset in range(1, 8 * 24 * 60 + 1):
            candidate = base + timedelta(minutes=offset)
            candidate_day = DAY_KEYS[candidate.weekday()]
            candidate_activity = self.activity_at(
                candidate_day, candidate.hour * 60 + candidate.minute
            )
            if _activity_identity(candidate_activity) != current_identity:
                next_activity = candidate_activity
                next_at = candidate
                break

        return ScheduleState(
            now=now,
            day_key=day_key,
            current=current,
            next_activity=next_activity,
            next_at=next_at,
            metrics=calculate_day_metrics(self.days[day_key]),
        )

    def week_metrics(self) -> WeekMetrics:
        days = [calculate_day_metrics(self.days[day_key]) for day_key in DAY_KEYS]

        def average(field: str) -> int:
            return _js_round(sum(getattr(day, field) for day in days) / len(days))

        def average_hours(field: str) -> float:
            return _round_hour(sum(getattr(day, field) for day in days) / len(days))

        def sum_hours(field: str) -> float:
            return _round_hour(sum(getattr(day, field) for day in days))

        return WeekMetrics(
            balance=average("balance"),
            energy=average("energy"),
            stress=average("stress"),
            productivity=average("productivity"),
            sleep=average_hours("sleep"),
            free=average_hours("free"),
            main=sum_hours("main"),
            walk=sum_hours("walk"),
            personal=sum_hours("personal"),
            sport=sum_hours("sport"),
        )


def _activity_identity(activity: Activity | None) -> tuple[str, str] | None:
    if activity is None:
        return None
    return activity.title, activity.kind


def load_routine(path: Path = SCHEDULE_PATH) -> WeeklyRoutine:
    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Файл расписания не найден: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Ошибка JSON в {path.name}, строка {exc.lineno}: {exc.msg}"
        ) from exc
    config = _mapping(raw_config, path.name)
    return WeeklyRoutine(config)


def format_duration(minutes: int) -> str:
    minutes = max(0, minutes)
    hours, remainder = divmod(minutes, 60)
    if not hours:
        return f"{remainder} мин"
    if not remainder:
        return f"{hours} ч"
    return f"{hours} ч {remainder} мин"


def format_activity_range(activity: Activity) -> str:
    return f"{minutes_to_time(activity.start)}–{minutes_to_time(activity.end)}"


def _activity_label(activity: Activity | None) -> str:
    return activity.title if activity else "Свободное время вне расписания"


def format_current_status(
    routine: WeeklyRoutine,
    value: datetime | None = None,
    *,
    brief: bool = False,
) -> str:
    state = routine.state_at(value)
    current_label = _activity_label(state.current)
    current_range = (
        f" ({format_activity_range(state.current)})" if state.current else ""
    )
    if state.next_at is not None:
        remaining = math.ceil(
            max(0.0, (state.next_at - state.now).total_seconds()) / 60
        )
        next_text = (
            f"далее «{_activity_label(state.next_activity)}» "
            f"в {state.next_at:%H:%M}"
        )
        remaining_text = f", ещё {format_duration(remaining)}"
    else:
        next_text = "следующее занятие не найдено"
        remaining_text = ""

    if brief:
        return (
            f"Расписание Миланы: сейчас «{current_label}»{current_range}"
            f"{remaining_text}; {next_text}. "
            f"Состояние дня: энергия {state.metrics.energy}%, "
            f"стресс {state.metrics.stress}%, "
            f"продуктивность {state.metrics.productivity}%, "
            f"баланс {state.metrics.balance}%."
        )

    offset = state.now.strftime("%z")
    offset_label = f"UTC{offset[:3]}:{offset[3:]}"
    metrics = state.metrics
    week = routine.week_metrics()
    lines = [
        (
            f"Расписание Миланы — {DAY_NAMES[state.day_key]}, "
            f"{state.now:%d.%m.%Y %H:%M} "
            f"({routine.timezone_name}, {offset_label})"
        ),
        f"Сейчас: {current_label}{current_range}{remaining_text}.",
        f"Далее: {_activity_label(state.next_activity)}"
        + (f" в {state.next_at:%H:%M}." if state.next_at else "."),
        (
            "Состояние дня: "
            f"энергия {metrics.energy}%, стресс {metrics.stress}%, "
            f"продуктивность {metrics.productivity}%, баланс {metrics.balance}%."
        ),
        f"Баланс недели: {week.balance}%.",
        "Сегодня:",
    ]
    for activity in routine.days[state.day_key]:
        marker = "→" if activity is state.current else " "
        lines.append(
            f"{marker} {format_activity_range(activity)}  {activity.title} "
            f"({ACTIVITY_NAMES[activity.kind]})"
        )
    return "\n".join(lines)


def format_day_schedule(routine: WeeklyRoutine, day_key: str) -> str:
    metrics = calculate_day_metrics(routine.days[day_key])
    lines = [
        f"Расписание Миланы — {DAY_NAMES[day_key]}",
        (
            f"Энергия {metrics.energy}%, стресс {metrics.stress}%, "
            f"продуктивность {metrics.productivity}%, баланс {metrics.balance}%."
        ),
    ]
    for activity in routine.days[day_key]:
        lines.append(
            f"{format_activity_range(activity)}  {activity.title} "
            f"({ACTIVITY_NAMES[activity.kind]}: {ACTIVITY_EFFECTS[activity.kind]})"
        )
    return "\n".join(lines)


def build_schedule_prompt(
    routine: WeeklyRoutine, value: datetime | None = None
) -> str:
    state = routine.state_at(value)
    current = _activity_label(state.current)
    current_range = (
        format_activity_range(state.current) if state.current else "без точного интервала"
    )
    metrics = state.metrics
    if state.next_at:
        next_line = (
            f"Следующее состояние: «{_activity_label(state.next_activity)}» "
            f"с {state.next_at:%H:%M}."
        )
    else:
        next_line = "Следующее состояние не определено."
    return (
        "Ты — Милана. Ниже твой актуальный бытовой контекст из недельного "
        "расписания:\n"
        f"Сейчас {DAY_NAMES[state.day_key]}, {state.now:%d.%m.%Y %H:%M %z}; "
        f"состояние «{current}», интервал {current_range}.\n"
        f"{next_line}\n"
        f"Показатели дня: энергия {metrics.energy}%, стресс {metrics.stress}%, "
        f"продуктивность {metrics.productivity}%, баланс {metrics.balance}%.\n"
        "Считай расписание фактическим контекстом. Если тебя спрашивают, что ты "
        "сейчас делаешь, свободна ли ты или как себя чувствуешь, отвечай "
        "согласованно с этим состоянием. Не перечисляй показатели без необходимости."
    )


def _parse_datetime(value: str, zone: tzinfo) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            "--at должен быть датой ISO, например 2026-07-13T10:00"
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=zone)
    return parsed.astimezone(zone)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Показать расписание Миланы")
    parser.add_argument(
        "--brief", action="store_true", help="вывести одну строку для BAT-файла"
    )
    parser.add_argument(
        "--day", choices=DAY_KEYS, help="показать весь выбранный день (mon–sun)"
    )
    parser.add_argument(
        "--at", help="показать состояние в момент ISO вместо текущего времени"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        routine = load_routine()
        value = _parse_datetime(args.at, routine.timezone) if args.at else None
        if args.day:
            print(format_day_schedule(routine, args.day))
        else:
            print(format_current_status(routine, value, brief=args.brief))
    except (OSError, ValueError) as exc:
        print(f"Ошибка расписания: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
