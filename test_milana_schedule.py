import json
import unittest
from datetime import datetime, timedelta, timezone

from milana_schedule import (
    ResponsePolicy,
    SCHEDULE_PATH,
    WeeklyRoutine,
    build_schedule_prompt,
    calculate_day_metrics,
    format_current_status,
    format_day_schedule,
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

    def test_response_policy_uses_title_kind_and_default_precedence(self) -> None:
        morning = self.routine.response_policy_at(
            datetime(2026, 7, 13, 7, 45, tzinfo=YEKT)
        )
        commute = self.routine.response_policy_at(
            datetime(2026, 7, 13, 8, 45, tzinfo=YEKT)
        )
        self.assertEqual(morning, ResponsePolicy(True, 15, 120))
        self.assertEqual(commute, ResponsePolicy(True, 15, 180))

        config = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
        del config["response_behavior"]["by_kind"]["study"]
        routine = WeeklyRoutine(config)
        default = routine.response_policy_at(
            datetime(2026, 7, 13, 10, 0, tzinfo=YEKT)
        )
        self.assertEqual(default, ResponsePolicy(True, 10, 240))

    def test_custom_event_response_policy_falls_back_to_kind(self) -> None:
        config = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
        config["custom_events"]["mon"].append(
            {
                "title": "Встреча",
                "type": "personal",
                "start": "10:00",
                "end": "11:00",
            }
        )
        routine = WeeklyRoutine(config)

        policy = routine.response_policy_at(
            datetime(2026, 7, 13, 10, 30, tzinfo=YEKT)
        )

        self.assertEqual(policy, ResponsePolicy(True, 10, 240))

    def test_plan_response_uses_injected_minimum_and_maximum(self) -> None:
        received_at = datetime(2026, 7, 13, 19, 10, tzinfo=YEKT)

        minimum = self.routine.plan_response(
            received_at, randint=lambda low, high: low
        )
        maximum = self.routine.plan_response(
            received_at, randint=lambda low, high: high
        )

        self.assertEqual(minimum.received_at, received_at)
        self.assertEqual(minimum.respond_at, received_at + timedelta(seconds=60))
        self.assertEqual(maximum.respond_at, received_at + timedelta(seconds=600))
        self.assertEqual(minimum.policy, ResponsePolicy(True, 60, 600))
        self.assertEqual(maximum.policy, minimum.policy)

    def test_sleep_plan_waits_for_wake_up_and_morning_delay(self) -> None:
        received_at = datetime(2026, 7, 13, 1, 0, tzinfo=YEKT)

        plan = self.routine.plan_response(
            received_at, randint=lambda low, high: low
        )

        self.assertEqual(
            plan.respond_at,
            datetime(2026, 7, 13, 7, 30, 15, tzinfo=YEKT),
        )
        self.assertEqual(plan.policy, ResponsePolicy(True, 15, 120))
        self.assertEqual(
            self.routine.response_policy_at(received_at),
            ResponsePolicy(False, 0, 0),
        )

    def test_plan_recalculates_policy_when_delay_crosses_transition(self) -> None:
        config = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
        config["response_behavior"]["by_title"]["Утренние сборы"] = {
            "available": True,
            "min_delay_seconds": 100,
            "max_delay_seconds": 200,
        }
        config["response_behavior"]["by_title"]["Завтрак"] = {
            "available": True,
            "min_delay_seconds": 1,
            "max_delay_seconds": 2,
        }
        routine = WeeklyRoutine(config)
        sampled = iter([200, 1])

        def deterministic_randint(low: int, high: int) -> int:
            value = next(sampled)
            self.assertLessEqual(low, value)
            self.assertLessEqual(value, high)
            return value

        plan = routine.plan_response(
            datetime(2026, 7, 13, 8, 4, 50, tzinfo=YEKT),
            randint=deterministic_randint,
        )

        self.assertEqual(
            plan.respond_at,
            datetime(2026, 7, 13, 8, 5, 1, tzinfo=YEKT),
        )
        self.assertEqual(plan.policy, ResponsePolicy(True, 1, 2))

    def test_response_behavior_validation(self) -> None:
        invalid_updates = (
            (
                lambda behavior: behavior["default"].update(available="yes"),
                "available должен быть true или false",
            ),
            (
                lambda behavior: behavior["default"].update(
                    min_delay_seconds=1.5
                ),
                "должен быть целым числом",
            ),
            (
                lambda behavior: behavior["default"].update(
                    min_delay_seconds=601,
                    max_delay_seconds=600,
                ),
                "не может быть больше",
            ),
            (
                lambda behavior: behavior["by_kind"].update(
                    unknown={
                        "available": True,
                        "min_delay_seconds": 1,
                        "max_delay_seconds": 2,
                    }
                ),
                "Неизвестный вид занятия",
            ),
        )
        for update, expected_error in invalid_updates:
            with self.subTest(expected_error=expected_error):
                config = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
                update(config["response_behavior"])
                with self.assertRaisesRegex(ValueError, expected_error):
                    WeeklyRoutine(config)

    def test_response_behavior_is_visible_in_schedule_output(self) -> None:
        sleeping_at = datetime(2026, 7, 13, 1, 0, tzinfo=YEKT)

        current = format_current_status(self.routine, sleeping_at, brief=True)
        day = format_day_schedule(self.routine, "mon")

        self.assertIn("Сообщения: не читает и не отвечает", current)
        self.assertIn("читает и отвечает через 4 мин–24 мин", day)
        self.assertEqual(
            self.routine.response_policy_at(sleeping_at).label,
            "не читает и не отвечает",
        )


if __name__ == "__main__":
    unittest.main()
