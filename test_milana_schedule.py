import json
import unittest
from datetime import datetime, timedelta, timezone

from milana_schedule import (
    SCHEDULE_PATH,
    WeeklyRoutine,
    build_schedule_prompt,
    calculate_day_metrics,
    load_routine,
)


YEKT = timezone(timedelta(hours=5))


class MilanaScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.routine = load_routine()

    def test_weekday_state_and_next_activity(self) -> None:
        state = self.routine.state_at(datetime(2026, 7, 13, 10, 0, tzinfo=YEKT))

        self.assertEqual(state.day_key, "mon")
        self.assertIsNotNone(state.current)
        self.assertEqual(state.current.title, "Учёба")
        self.assertEqual((state.current.start, state.current.end), (550, 910))
        self.assertIsNotNone(state.next_activity)
        self.assertEqual(state.next_activity.title, "Перерывы и отдых")
        self.assertEqual(state.next_at, datetime(2026, 7, 13, 15, 10, tzinfo=YEKT))

    def test_cross_midnight_sleep_is_active(self) -> None:
        state = self.routine.state_at(datetime(2026, 7, 13, 0, 15, tzinfo=YEKT))

        self.assertIsNotNone(state.current)
        self.assertEqual(state.current.title, "Ночной сон")
        self.assertEqual(state.next_at, datetime(2026, 7, 13, 7, 30, tzinfo=YEKT))

    def test_weekend_uses_template_timing(self) -> None:
        state = self.routine.state_at(datetime(2026, 7, 11, 1, 0, tzinfo=YEKT))

        self.assertEqual(state.day_key, "sat")
        self.assertIsNotNone(state.current)
        self.assertEqual(state.current.title, "Ночной сон")
        self.assertEqual((state.current.start, state.current.end), (30, 570))
        self.assertEqual(state.next_at, datetime(2026, 7, 11, 9, 30, tzinfo=YEKT))

    def test_metrics_match_student_preset_formula(self) -> None:
        weekday = calculate_day_metrics(self.routine.days["mon"])
        weekend = calculate_day_metrics(self.routine.days["sat"])

        self.assertEqual(
            (weekday.energy, weekday.stress, weekday.productivity, weekday.balance),
            (89, 8, 81, 85),
        )
        self.assertEqual(
            (weekend.energy, weekend.stress, weekend.productivity, weekend.balance),
            (98, 0, 47, 86),
        )

    def test_custom_event_overrides_base_activity_for_current_state(self) -> None:
        config = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
        config["custom_events"]["mon"].append(
            {
                "title": "Приём у врача",
                "type": "personal",
                "start": "10:00",
                "end": "11:00",
            }
        )
        routine = WeeklyRoutine(config)

        state = routine.state_at(datetime(2026, 7, 13, 10, 30, tzinfo=YEKT))

        self.assertIsNotNone(state.current)
        self.assertEqual(state.current.title, "Приём у врача")
        self.assertEqual(state.next_at, datetime(2026, 7, 13, 11, 0, tzinfo=YEKT))
        self.assertIsNotNone(state.next_activity)
        self.assertEqual(state.next_activity.title, "Учёба")

    def test_prompt_contains_dynamic_schedule_context(self) -> None:
        prompt = build_schedule_prompt(
            self.routine, datetime(2026, 7, 13, 10, 0, tzinfo=YEKT)
        )

        self.assertIn("Ты — Милана", prompt)
        self.assertIn("состояние «Учёба»", prompt)
        self.assertIn("Следующее состояние: «Перерывы и отдых» с 15:10", prompt)
        self.assertIn("энергия 89%", prompt)


if __name__ == "__main__":
    unittest.main()
