import sqlite3
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock

from milana_heartbeat import MilanaHeartbeat
from milana_memory import MilanaMemoryStore
from milana_scene import SceneEngine, SceneFact, SceneState
from milana_schedule import Activity, ResponsePlan, ResponsePolicy, load_routine
from milana_state import MilanaStateStore


ZONE = timezone(timedelta(hours=5))
NOW = datetime(2026, 9, 21, 10, tzinfo=ZONE)


def routine_with(*blocks):
    routine = load_routine()
    routine.days = {key: tuple(blocks) for key in routine.days}
    return routine


def block(kind, start, end, title=None, custom=False):
    return Activity(title or kind, kind, start, end, (end-start) % 1440, custom)


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'memory.sqlite3'
        self.state = MilanaStateStore(self.path)
        self.clock = NOW
        self.routine = routine_with(
            block('sleep', 1380, 480), block('study', 480, 720),
            block('commute', 720, 750), block('personal', 750, 1380),
        )
        self.engine = SceneEngine(self.state, self.routine, now=lambda: self.clock)

    def tearDown(self):
        self.state.close()
        self.tmp.cleanup()

    def test_create_absent_state_and_round_trip(self):
        self.assertIsNone(self.engine.current())
        scene = self.engine.tick()
        self.assertEqual(scene.location_type, 'university')
        self.assertEqual(scene.activity_type, 'study')
        self.assertFalse(scene.can_voice)
        self.assertFalse(scene.can_photo)
        self.assertEqual(SceneState.from_dict(scene.to_dict()), scene)
        self.assertEqual(self.engine.current(), scene)

    def test_restart_keeps_identity_phase_and_facts(self):
        scene = self.engine.tick()
        self.engine.add_fact(SceneFact('notebook', True, NOW+timedelta(minutes=20)))
        self.state.close()
        self.state = MilanaStateStore(self.path)
        restarted = SceneEngine(self.state, self.routine, now=lambda: NOW+timedelta(seconds=30))
        current = restarted.tick()
        self.assertEqual(current.scene_id, scene.scene_id)
        self.assertEqual(current.location_type, scene.location_type)
        self.assertEqual(current.activity_phase, scene.activity_phase)
        self.assertEqual(restarted.snapshot()['facts'][0]['key'], 'notebook')

    def test_schedule_boundaries_recover_day_without_teleport(self):
        first = self.engine.tick()
        self.clock = NOW.replace(hour=13)
        current = self.engine.tick()
        self.assertEqual(current.location_type, 'home')
        history = self.engine.snapshot()['history']
        self.assertEqual([s['location_type'] for s in history], ['university', 'transit', 'home'])
        self.assertEqual(history[0]['scene_id'], first.scene_id)
        self.assertEqual(datetime.fromisoformat(history[0]['ended_at']), NOW.replace(hour=12))
        self.assertTrue(history[-1]['metadata']['inferred'])
        self.assertIn('восстановлено по расписанию', self.engine.model_context()['scene_day'])

    def test_missing_commute_inserts_bridge_before_home(self):
        self.routine.days = {key: (block('study', 480, 720), block('personal', 720, 1380)) for key in self.routine.days}
        self.engine.tick()
        self.clock = NOW.replace(hour=12)
        travelling = self.engine.tick()
        self.assertEqual(travelling.activity_type, 'commute')
        self.assertEqual(travelling.metadata['destination'], 'home')
        self.clock += timedelta(minutes=15)
        self.assertEqual(self.engine.tick().location_type, 'home')

    def test_custom_schedule_override_still_requires_travel(self):
        self.engine.tick()
        for key in self.routine.days:
            self.routine.days[key] += (block('personal', 610, 680, custom=True),)
        self.clock += timedelta(minutes=10)
        self.assertEqual(self.engine.tick().location_type, 'transit')

    def test_end_and_generate_next_stay_in_current_block(self):
        scene = self.engine.tick()
        ended = self.engine.end()
        self.assertEqual(ended.ended_at, NOW)
        self.assertIsNone(self.engine.snapshot()['current'])
        next_scene = self.engine.tick()
        self.assertNotEqual(scene.scene_id, next_scene.scene_id)
        self.assertEqual(next_scene.activity_type, 'study')
        self.assertEqual(self.engine.generate_next().location_type, 'university')

    def test_phase_changes_keep_scene_and_record_events(self):
        first = self.engine.tick()
        self.clock += timedelta(minutes=45)
        scene = self.engine.tick()
        self.assertEqual(scene.scene_id, first.scene_id)
        self.assertEqual(scene.location_type, first.location_type)
        self.assertEqual(scene.activity_phase, 'break')
        self.assertTrue(scene.can_voice)
        self.assertEqual(len(self.engine.snapshot()['events']), 3)
        self.engine.tick()
        self.assertEqual(len(self.engine.snapshot()['events']), 3)
        self.clock += timedelta(minutes=10)
        self.assertFalse(self.engine.tick().can_voice)
        self.assertIn('перерыв между занятиями', self.engine.model_context()['scene_day'])

    def test_repeated_end_does_not_create_phantom_scenes(self):
        self.assertIsNone(self.engine.end())
        self.engine.tick()
        self.engine.end()
        self.assertIsNone(self.engine.end())
        self.engine.generate_next()
        with self.state.transaction() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM scenes').fetchone()[0], 2)

    def test_facts_expire_at_exact_boundary_and_do_not_enter_world(self):
        self.engine.add_fact(SceneFact('waiting_for_bus', True, NOW+timedelta(minutes=2)))
        self.assertEqual(len(self.engine.snapshot()['facts']), 1)
        self.clock += timedelta(minutes=2)
        self.assertEqual(self.engine.snapshot()['facts'], [])
        self.engine.tick()
        with self.state.transaction() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM scene_facts').fetchone()[0], 0)
        self.assertEqual(self.state.list_life_events(), [])
        with self.assertRaises(ValueError):
            self.engine.add_fact(SceneFact('location', 'home', self.clock+timedelta(minutes=1)))

    def test_event_is_temporary_and_survives_restart(self):
        event = self.engine.add_event('купила кофе')
        other = SceneEngine(self.state, self.routine, now=lambda: self.clock)
        saved = next(e for e in other.snapshot()['events'] if e['event_id'] == event.event_id)
        self.assertTrue(saved['active'])
        self.clock += timedelta(hours=1)
        self.assertFalse(other.snapshot()['events'][0]['active'])
        self.assertEqual(self.state.list_life_events(), [])
        with self.assertRaises(ValueError):
            self.engine.add_event('')

    def test_context_has_physical_limits_and_bounded_history(self):
        context = self.engine.model_context()
        self.assertIn('<current_scene>', context['current_scene'])
        self.assertIn('в университете', context['current_scene'])
        self.assertIn('голос нельзя', context['current_scene'])
        self.assertIn('<scene_day>', context['scene_day'])
        self.assertNotIn('metadata', context['current_scene'])

    def test_validation_rejects_incompatible_title_location_and_numbers(self):
        scene = self.engine.tick()
        for invalid in (
            replace(scene, location_type='home'), replace(scene, activity_title='лежу дома'),
            replace(scene, attention=2), replace(scene, energy=float('nan')),
            replace(scene, phone_available=False, can_voice=True),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    invalid.validate()

    def test_same_block_varies_only_on_explicit_next(self):
        first = self.engine.tick()
        for minute in range(1, 10):
            self.assertEqual(self.engine.tick(NOW+timedelta(minutes=minute)).scene_id, first.scene_id)
        self.clock += timedelta(minutes=10)
        variants = {self.engine.generate_next().activity_phase for _ in range(20)}
        self.assertGreater(len(variants), 1)

    def test_short_meal_reaches_eating_and_reduces_hunger(self):
        self.routine.days = {key: (block('food', 600, 625),) for key in self.routine.days}
        scene = self.engine.tick()
        self.assertEqual(scene.activity_phase, 'prepare')
        self.clock += timedelta(minutes=20)
        after = self.engine.tick()
        self.assertEqual(after.scene_id, scene.scene_id)
        self.assertEqual(after.activity_phase, 'tea')
        self.assertLess(after.hunger, scene.hunger)

    def test_free_time_uses_hunger_and_rest_needs_without_moving(self):
        self.routine.days = {key: (block('personal', 480, 1380),) for key in self.routine.days}
        first = self.engine.tick()
        with self.state.transaction() as db:
            self.engine._save(db, replace(first, hunger=85))
        self.clock += timedelta(minutes=45)
        self.assertEqual(self.engine.tick().activity_type, 'food')
        self.assertEqual(self.engine.current().location_type, 'home')

    def test_sleep_midnight_wake_is_temporary_and_does_not_move(self):
        self.clock = NOW.replace(hour=23, minute=30)
        sleeping = self.engine.tick()
        self.assertFalse(sleeping.phone_available)
        self.engine.wake_phone()
        awake = self.engine.current()
        self.assertEqual(awake.scene_id, sleeping.scene_id)
        self.assertTrue(awake.phone_available)
        self.assertIn('проснулась', awake.activity_title)
        self.assertEqual(awake.location_type, 'home')
        self.clock += timedelta(minutes=15)
        self.assertFalse(self.engine.tick().phone_available)
        self.assertEqual(self.engine.current().activity_title, 'сплю')
        self.clock = (NOW+timedelta(days=1)).replace(hour=0, minute=1)
        self.assertEqual(self.engine.tick().scene_id, sleeping.scene_id)

    def test_empty_schedule_does_not_recreate_on_every_message(self):
        self.routine.days = {key: () for key in self.routine.days}
        first = self.engine.tick()
        self.clock += timedelta(seconds=30)
        self.assertEqual(self.engine.tick().scene_id, first.scene_id)

    def test_response_delay_preserves_base_and_respects_boundary(self):
        scene = self.engine.tick()
        plan = ResponsePlan(NOW, NOW+timedelta(seconds=100), ResponsePolicy(True, 100, 100))
        adjusted = self.engine.adjust_response_plan(plan)
        self.assertGreater(adjusted.respond_at, plan.respond_at)
        self.assertLessEqual(adjusted.respond_at, NOW+timedelta(seconds=135))
        later = replace(plan, respond_at=scene.expected_end+timedelta(minutes=1))
        self.assertEqual(self.engine.adjust_response_plan(later), later)

    def test_concurrent_refresh_creates_only_one_current_scene(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(lambda _: self.engine.tick().scene_id, range(20)))
        self.assertEqual(len(set(ids)), 1)
        with self.state.transaction() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM scenes WHERE ended_at IS NULL').fetchone()[0], 1)

    def test_long_gap_is_bounded_and_clock_does_not_rewind(self):
        self.engine.tick()
        self.clock += timedelta(days=30)
        current = self.engine.tick()
        self.assertTrue(current.metadata['history_gap'])
        self.assertEqual(self.engine.tick(NOW), current)


class SceneMigrationTests(unittest.TestCase):
    def test_legacy_memory_database_is_preserved_and_migration_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'old.sqlite3'
            memory = MilanaMemoryStore(path)
            memory.add_message(7, 'user', 'старое сообщение')
            memory.add_diary_entry('запись дневника')
            state = MilanaStateStore(path)
            try:
                one = SceneEngine(state, load_routine(), now=lambda: NOW).tick()
                two = SceneEngine(state, load_routine(), now=lambda: NOW).tick()
                self.assertEqual(one.scene_id, two.scene_id)
                self.assertEqual(memory.get_chat_history(7)[0].content, 'старое сообщение')
                self.assertEqual(memory.get_diary()[0].content, 'запись дневника')
                with closing(sqlite3.connect(path)) as db:
                    self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            finally:
                state.close()
                memory.close()


class SceneHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    async def test_paused_heartbeat_advances_scene_without_model_calls(self):
        state = MilanaStateStore()
        clock = [NOW]
        routine = routine_with(block('study', 480, 720), block('commute', 720, 750))
        engine = SceneEngine(state, routine, now=lambda: clock[0])
        execute = AsyncMock()
        heartbeat = MilanaHeartbeat(state, execute, now=lambda: clock[0], on_tick=engine.tick, dev_mode=True)
        try:
            await heartbeat.run_once()
            original = engine.current()
            clock[0] += timedelta(minutes=45)
            await heartbeat.run_once()
            self.assertEqual(engine.current().scene_id, original.scene_id)
            self.assertEqual(engine.current().activity_phase, 'break')
            clock[0] = NOW.replace(hour=12)
            await heartbeat.run_once()
            self.assertEqual(engine.current().activity_type, 'commute')
            execute.assert_not_awaited()
        finally:
            state.close()


if __name__ == '__main__':
    unittest.main()
