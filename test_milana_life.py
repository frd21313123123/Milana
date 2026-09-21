import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from milana_life import LifePlanner
from milana_schedule import Activity, load_routine
from milana_scene import SceneEngine
from milana_state import MilanaStateStore


ZONE = timezone(timedelta(hours=5))


class FixedRandom:
    def __init__(self, roll=0.0, duration=None):
        self.roll = roll
        self.duration = duration

    def random(self):
        return self.roll

    def randint(self, low, high):
        return self.duration if self.duration is not None else low


class FailingRandom(FixedRandom):
    def random(self):
        raise RuntimeError("simulated random failure")


def activity(kind, start, end, title=None, custom=False):
    return Activity(title or kind, kind, start, end, (end - start) % 1440, custom)


class LifePlannerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.state = MilanaStateStore(Path(self.tmp.name) / "state.sqlite3")
        self.routine = load_routine()
        self.routine.life_planner = {
            "enabled": True,
            "daily_deviation_limit": 2,
            "probabilities": {key: 1.0 for key in (
                "overslept", "commute_delay", "coffee_stop", "class_cancelled",
                "walk_extended", "store_visit", "series_late",
            )},
        }

    def tearDown(self):
        self.state.close()
        self.tmp.cleanup()

    def planner(self, blocks, now, duration=None):
        self.routine.days = {key: tuple(blocks) for key in self.routine.days}
        return LifePlanner(self.state, self.routine, now=lambda: now,
                           rng=FixedRandom(duration=duration))

    def test_first_start_inside_block_does_not_invent_past_deviation(self):
        now = datetime(2026, 9, 21, 10, 0, 30, tzinfo=ZONE)
        planner = self.planner([activity("study", 600, 720, "Учёба")], now)
        state = planner.advance(now)
        self.assertEqual(state.current.title, "Учёба")
        self.assertFalse(any(row["selected"] for row in planner.snapshot(now)["decisions"]))

    def test_commute_delay_is_durable_and_shortens_following_study(self):
        now = datetime(2026, 9, 21, 8, tzinfo=ZONE)
        planner = self.planner([
            activity("commute", 480, 520, "Дорога"),
            activity("study", 520, 700, "Учёба"),
        ], now, duration=20)
        current = planner.advance(now).current
        self.assertEqual(current.actual_end, now + timedelta(minutes=60))
        snapshot = planner.snapshot(now)
        study = next(row for row in snapshot["events"] if row["title"] == "Учёба")
        self.assertEqual(datetime.fromisoformat(study["actual_start"]), now + timedelta(minutes=60))
        restarted = LifePlanner(self.state, self.routine, now=lambda: now,
                                rng=FixedRandom(roll=1.0))
        self.assertEqual(restarted.state_at(now).current.actual_end, current.actual_end)

    def test_coffee_replaces_end_of_commute_without_late_arrival(self):
        self.routine.life_planner["probabilities"]["commute_delay"] = 0.0
        now = datetime(2026, 9, 21, 8, tzinfo=ZONE)
        planner = self.planner([
            activity("commute", 480, 520, "Дорога"),
            activity("study", 520, 700, "Учёба"),
        ], now, duration=10)
        planner.advance(now)
        coffee = next(row for row in planner.snapshot(now)["events"] if row["scenario"] == "coffee_stop")
        self.assertEqual(coffee["location"], "cafe")
        self.assertEqual(datetime.fromisoformat(coffee["actual_end"]), now.replace(hour=8, minute=40))

    def test_scene_executes_coffee_and_reaches_study_without_extra_bridge(self):
        self.routine.life_planner["probabilities"]["commute_delay"] = 0.0
        now = datetime(2026, 9, 21, 8, tzinfo=ZONE)
        planner = self.planner([
            activity("commute", 480, 520, "Дорога"),
            activity("study", 520, 700, "Учёба"),
        ], now, duration=10)
        planner.advance(now)
        scene = SceneEngine(self.state, planner, now=lambda: now)
        self.assertEqual(scene.tick(now).activity_type, "commute")
        coffee_at = now + timedelta(minutes=30)
        planner.advance(coffee_at)
        coffee = scene.tick(coffee_at)
        self.assertEqual(coffee.activity_title, "зашла за кофе")
        self.assertEqual(coffee.location_type, "cafe")
        study_at = now + timedelta(minutes=40)
        planner.advance(study_at)
        study = scene.tick(study_at)
        self.assertEqual(study.activity_type, "study")
        self.assertEqual(study.location_type, "university")

    def test_cancelled_class_creates_ninety_minute_university_window(self):
        now = datetime(2026, 9, 21, 9, tzinfo=ZONE)
        planner = self.planner([activity("study", 540, 900, "Учёба")], now)
        planner.advance(now)
        free = next(row for row in planner.snapshot(now)["events"] if row["scenario"] == "class_cancelled")
        self.assertEqual((datetime.fromisoformat(free["actual_end"]) - datetime.fromisoformat(free["actual_start"])).seconds, 5400)
        self.assertEqual(free["location"], "university")

    def test_store_replaces_start_of_free_time(self):
        now = datetime(2026, 9, 21, 18, tzinfo=ZONE)
        self.routine.life_planner["probabilities"]["series_late"] = 0.0
        planner = self.planner([activity("personal", 1080, 1200, "Личные дела")], now, duration=15)
        self.assertEqual(planner.advance(now).current.kind, "commute")
        store = planner.state_at(now + timedelta(minutes=10)).current
        self.assertEqual(store.scenario, "store_visit")
        self.assertEqual(store.location, "shop")

    def test_extended_walk_and_oversleep_do_not_overlap_following_blocks(self):
        for kind, start, blocks, scenario in (
            ("walk", 600, [activity("walk", 600, 660), activity("personal", 660, 720)], "walk_extended"),
            ("sleep", 0, [activity("sleep", 0, 480), activity("chores", 480, 540)], "overslept"),
        ):
            with self.subTest(scenario=scenario):
                self.state.close()
                self.state = MilanaStateStore(Path(self.tmp.name) / f"{scenario}.sqlite3")
                now = datetime(2026, 9, 21, start // 60, start % 60, tzinfo=ZONE)
                planner = self.planner(blocks, now, duration=20)
                planner.advance(now)
                events = [e for e in planner.snapshot(now)["events"] if e["status"] != "cancelled"]
                intervals = sorted((datetime.fromisoformat(e["actual_start"]), datetime.fromisoformat(e["actual_end"])) for e in events)
                self.assertTrue(all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:])))

    def test_series_delays_sleep_but_keeps_six_hours(self):
        self.routine.life_planner["probabilities"]["store_visit"] = 0.0
        now = datetime(2026, 9, 21, 22, tzinfo=ZONE)
        planner = self.planner([
            activity("personal", 1320, 1380, "Личные дела"),
            activity("sleep", 1380, 420, "Ночной сон"),
        ], now, duration=60)
        planner.advance(now, energy=60)
        series = next(row for row in planner.snapshot(now)["events"] if row["scenario"] == "series_late")
        sleep = planner.state_at(now + timedelta(hours=2)).current.to_dict()
        self.assertEqual(datetime.fromisoformat(series["actual_end"]), now.replace(hour=0, minute=0) + timedelta(days=1))
        self.assertGreaterEqual(datetime.fromisoformat(sleep["actual_end"]) - datetime.fromisoformat(sleep["actual_start"]), timedelta(hours=6))

    def test_daily_limit_and_read_only_snapshot(self):
        now = datetime(2026, 9, 21, 8, tzinfo=ZONE)
        planner = self.planner([activity("commute", 480, 520), activity("walk", 600, 660)], now, duration=10)
        planner.daily_limit = 1
        planner.advance(now)
        before = planner.snapshot(now)
        planner.snapshot(now)
        planner.advance(now.replace(hour=10))
        after = planner.snapshot(now.replace(hour=10))
        self.assertEqual(sum(bool(x["selected"]) for x in after["decisions"]), 1)
        self.assertEqual(before["decisions"], planner.snapshot(now)["decisions"][:len(before["decisions"])])

    def test_concurrent_advance_makes_one_durable_decision(self):
        now = datetime(2026, 9, 21, 8, tzinfo=ZONE)
        planner = self.planner([activity("commute", 480, 520)], now, duration=10)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: planner.advance(now), range(8)))
        self.assertTrue(all(result.current.event_id == results[0].current.event_id for result in results))
        selected = [row for row in planner.snapshot(now)["decisions"] if row["selected"]]
        self.assertEqual(len(selected), 1)

    def test_failed_decision_rolls_back_materialization(self):
        now = datetime(2026, 9, 21, 8, tzinfo=ZONE)
        planner = self.planner([activity("commute", 480, 520)], now)
        planner.rng = FailingRandom()
        with self.assertRaises(RuntimeError):
            planner.advance(now)
        with self.state.transaction() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM planned_events").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT count(*) FROM life_opportunities").fetchone()[0], 0)
        planner.rng = FixedRandom(duration=10)
        self.assertIsNotNone(planner.advance(now).current)

    def test_config_change_preserves_active_event_and_rebuilds_future(self):
        now = datetime(2026, 9, 21, 10, tzinfo=ZONE)
        planner = self.planner([
            activity("study", 540, 660, "Учёба"),
            activity("personal", 660, 720, "Старый отдых"),
        ], now)
        current_id = planner.advance(now).current.event_id
        self.routine.days = {key: (
            activity("study", 540, 660, "Учёба"),
            activity("walk", 660, 720, "Новая прогулка"),
        ) for key in self.routine.days}
        restarted = LifePlanner(self.state, self.routine, now=lambda: now, rng=FixedRandom(roll=1.0))
        restarted.advance(now)
        self.assertEqual(restarted.state_at(now).current.event_id, current_id)
        titles = [row["title"] for row in restarted.snapshot(now)["events"] if row["status"] != "cancelled"]
        self.assertIn("Новая прогулка", titles)
        self.assertNotIn("Старый отдых", titles)


if __name__ == "__main__":
    unittest.main()
