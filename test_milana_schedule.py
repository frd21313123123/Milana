import json
import json
import unittest
from datetime import datetime, timedelta, timezone

from milana_schedule import (
    Activity,
    OnlineBehavior,
    ResponsePolicy,
    SCHEDULE_PATH,
    WeeklyRoutine,
    build_schedule_prompt,
    calculate_day_metrics,
    format_current_status,
    format_day_schedule,
    format_duration,
    format_response_delay,
    format_response_policy,
    load_routine,
    minutes_to_time,
    parse_utc_offset,
    time_to_minutes,
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

    def test_online_behavior_is_loaded_and_validated(self) -> None:
        self.assertEqual(
            self.routine.online_behavior,
            OnlineBehavior(
                online_response_min_seconds=1,
                online_response_max_seconds=10,
                post_reply_online_min_seconds=30,
                post_reply_online_max_seconds=60,
                spontaneous_online_interval_min_seconds=900,
                spontaneous_online_interval_max_seconds=2700,
                spontaneous_online_duration_min_seconds=120,
                spontaneous_online_duration_max_seconds=300,
                sleep_buffer_seconds=60,
            ),
        )

        invalid_updates = (
            (
                lambda behavior: behavior.update(
                    online_response_min_seconds=11,
                    online_response_max_seconds=10,
                ),
                "online_response_min_seconds не может быть больше",
            ),
            (
                lambda behavior: behavior.update(
                    post_reply_online_min_seconds=61,
                    post_reply_online_max_seconds=60,
                ),
                "post_reply_online_min_seconds не может быть больше",
            ),
            (
                lambda behavior: behavior.update(
                    spontaneous_online_interval_min_seconds=2701,
                ),
                "spontaneous_online_interval_min_seconds не может",
            ),
            (
                lambda behavior: behavior.update(
                    spontaneous_online_duration_min_seconds=301,
                ),
                "spontaneous_online_duration_min_seconds не может",
            ),
            (
                lambda behavior: behavior.update(sleep_buffer_seconds=59),
                "sleep_buffer_seconds должен быть не меньше",
            ),
        )
        for update, expected_error in invalid_updates:
            with self.subTest(expected_error=expected_error):
                config = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
                update(config["online_behavior"])
                with self.assertRaisesRegex(ValueError, expected_error):
                    WeeklyRoutine(config)

    def test_time_parsing_and_cross_midnight_activity_boundaries(self) -> None:
        self.assertEqual(time_to_minutes("00:00"), 0)
        self.assertEqual(time_to_minutes("23:59"), 1439)
        self.assertEqual(minutes_to_time(1441), "00:01")
        self.assertEqual(parse_utc_offset("+05:30").utcoffset(None), timedelta(hours=5, minutes=30))

        sleep = Activity("Сон", "sleep", 23 * 60, 7 * 60, 8 * 60)
        self.assertTrue(sleep.contains(23 * 60))
        self.assertTrue(sleep.contains(6 * 60 + 59))
        self.assertFalse(sleep.contains(12 * 60))

    def test_invalid_time_and_timezone_values_are_rejected(self) -> None:
        invalid_times = (None, "24:00", "12:60", "12-00")
        for value in invalid_times:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    time_to_minutes(value)

        for value in (None, "UTC+5", "+15:00", "+05:60"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_utc_offset(value)

    def test_human_readable_duration_and_response_policy_formats(self) -> None:
        self.assertEqual(format_duration(-5), "0 мин")
        self.assertEqual(format_duration(125), "2 ч 5 мин")
        self.assertEqual(format_response_delay(3661), "1 ч 1 мин 1 сек")
        self.assertEqual(
            format_response_policy(ResponsePolicy(True, 0, 0)),
            "читает и отвечает сразу",
        )
        self.assertEqual(
            format_response_policy(ResponsePolicy(True, 30, 30)),
            "читает и отвечает примерно через 30 сек",
        )

    def test_randint_must_return_an_integer_inside_policy_range(self) -> None:
        received_at = datetime(2026, 7, 13, 21, 0, tzinfo=YEKT)
        invalid_results = (True, 0, 601, 1.5)

        for result in invalid_results:
            with self.subTest(result=result):
                with self.assertRaisesRegex(ValueError, "randint"):
                    self.routine.plan_response(
                        received_at,
                        randint=lambda low, high, result=result: result,
                    )

    def test_load_routine_reports_missing_and_invalid_json(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with self.assertRaisesRegex(ValueError, "не найден"):
                load_routine(missing)

            invalid = Path(directory) / "invalid.json"
            invalid.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Ошибка JSON"):
                load_routine(invalid)


if __name__ == "__main__":
    unittest.main()
