import sqlite3
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from milana_memory import MilanaMemoryStore
from milana_state import (
    MAX_ACTIVE_GOALS,
    FactSeed,
    GoalChange,
    GoalLimitError,
    HeartbeatChanges,
    LockedFactError,
    MilanaStateStore,
    NewEntity,
    NewLifeEvent,
    Relationship,
    RelationshipDelta,
    StateConflictError,
    adaptive_initiative_cooldown,
    initiative_allowed,
)


NOW = datetime(2026, 7, 14, 10, 0, tzinfo=timezone.utc)


class StateMigrationTests(unittest.TestCase):
    def test_schema_is_additive_to_existing_memory_database(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "milana.sqlite3"
            memory = MilanaMemoryStore(path)
            task = memory.schedule_pulse_message(10, "не потерять", due_at=NOW)
            memory.close()

            state = MilanaStateStore(path)
            state.create_goal("Новая цель")
            state.close()

            reopened = MilanaMemoryStore(path)
            try:
                self.assertEqual(reopened.get_pulse_tasks()[0].id, task.id)
            finally:
                reopened.close()
            connection = sqlite3.connect(path)
            try:
                names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                connection.close()
            self.assertTrue(
                {
                    "agent_state",
                    "heartbeat_jobs",
                    "world_entities",
                    "world_facts",
                    "life_events",
                    "goals",
                    "relationships",
                    "world_summaries",
                    "skill_audit",
                    "chat_messages",
                    "pulse_tasks",
                }.issubset(names)
            )


class WorldStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MilanaStateStore()

    def tearDown(self) -> None:
        self.store.close()

    def test_locked_fact_cannot_change_and_unlocked_fact_is_versioned(self) -> None:
        milana = self.store.create_entity(
            "person",
            "Милана",
            entity_id="milana",
            facts=(FactSeed("city", "Екатеринбург", locked=True),),
            at=NOW,
        )
        with self.assertRaises(LockedFactError):
            self.store.set_fact(milana.id, "city", "Москва", at=NOW)

        self.store.set_fact(milana.id, "project", "alpha", at=NOW)
        self.store.set_fact(
            milana.id,
            "project",
            "beta",
            at=NOW + timedelta(minutes=1),
        )
        history = [
            fact
            for fact in self.store.get_facts(milana.id, include_history=True)
            if fact.key == "project"
        ]
        self.assertEqual([fact.version for fact in history], [1, 2])
        self.assertIsNotNone(history[0].superseded_at)
        self.assertIsNone(history[1].superseded_at)

    def test_heartbeat_can_version_existing_fact_but_not_locked_seed(self) -> None:
        self.store.create_entity(
            "person",
            "Лера",
            entity_id="lera",
            facts=(
                FactSeed("favorite_drink", "кофе"),
                FactSeed("city", "Пермь", locked=True),
            ),
            at=NOW,
        )
        revision = self.store.get_agent_state().revision
        self.store.apply_heartbeat_changes(
            HeartbeatChanges(
                entities=(
                    NewEntity(
                        "person",
                        "Лера",
                        entity_id="lera",
                        facts=(FactSeed("favorite_drink", "чай"),),
                    ),
                )
            ),
            expected_revision=revision,
            at=NOW + timedelta(minutes=1),
        )
        versions = [
            fact
            for fact in self.store.get_facts("lera", include_history=True)
            if fact.key == "favorite_drink"
        ]
        self.assertEqual([fact.value for fact in versions], ["кофе", "чай"])
        self.assertEqual([fact.version for fact in versions], [1, 2])

        before = self.store.get_agent_state()
        with self.assertRaises(LockedFactError):
            self.store.apply_heartbeat_changes(
                HeartbeatChanges(
                    entities=(
                        NewEntity(
                            "person",
                            "Лера",
                            entity_id="lera",
                            facts=(FactSeed("city", "Москва"),),
                        ),
                    ),
                    mood="не должно сохраниться",
                ),
                expected_revision=before.revision,
                at=NOW + timedelta(minutes=2),
            )
        self.assertEqual(self.store.get_agent_state(), before)

    def test_active_goal_limit_requires_archiving_before_next_create(self) -> None:
        goals = [
            self.store.create_goal(f"Цель {index}", at=NOW)
            for index in range(MAX_ACTIVE_GOALS)
        ]
        with self.assertRaises(GoalLimitError):
            self.store.create_goal("Лишняя", at=NOW)
        self.store.archive_goal(goals[0].id, at=NOW)
        self.assertEqual(self.store.create_goal("Новая", at=NOW).status, "active")

    def test_heartbeat_update_is_bounded_atomic_and_revision_checked(self) -> None:
        state = self.store.get_agent_state()
        changes = HeartbeatChanges(
            entities=(
                NewEntity("person", "Лера", entity_id="lera", is_real=True),
            ),
            events=(
                NewLifeEvent(
                    "Познакомились",
                    "Милана познакомилась с Лерой",
                    entity_ids=("lera",),
                ),
            ),
            goals=(GoalChange("create", title="Позвать Леру гулять"),),
            need_deltas={"social": 15, "rest": -15},
            relationships=(RelationshipDelta("lera", closeness=10),),
            mood="воодушевлённое",
            valence=70,
            arousal=65,
            current_intention="договориться о прогулке",
        )
        committed = self.store.apply_heartbeat_changes(
            changes,
            expected_revision=state.revision,
            at=NOW,
        )
        self.assertEqual(committed.revision, state.revision + 1)
        self.assertEqual(committed.social, 65)
        self.assertEqual(committed.rest, 35)
        self.assertEqual(self.store.get_relationship("lera").closeness, 60)

        with self.assertRaises(StateConflictError):
            self.store.apply_heartbeat_changes(
                HeartbeatChanges(mood="устаревшее"),
                expected_revision=state.revision,
                at=NOW,
            )
        with self.assertRaises(ValueError):
            self.store.apply_heartbeat_changes(
                HeartbeatChanges(need_deltas={"social": 16}),
                at=NOW,
            )
        with self.assertRaises(ValueError):
            self.store.apply_heartbeat_changes(
                HeartbeatChanges(
                    entities=tuple(
                        NewEntity("place", f"Место {index}") for index in range(4)
                    )
                ),
                at=NOW,
            )
        self.assertEqual(len(self.store.list_entities()), 1)

    def test_non_heartbeat_turn_can_commit_without_changing_heartbeat_time(self) -> None:
        state = self.store.get_agent_state()
        committed = self.store.apply_heartbeat_changes(
            HeartbeatChanges(mood="заинтересованное"),
            expected_revision=state.revision,
            at=NOW,
            record_heartbeat=False,
        )
        self.assertEqual(committed.mood, "заинтересованное")
        self.assertIsNone(committed.last_heartbeat_at)

    def test_relationship_delta_is_limited_per_interaction(self) -> None:
        entity = self.store.create_entity("person", "Аня", entity_id="anya")
        self.store.upsert_relationship(entity.id, closeness=95)
        adjusted = self.store.adjust_relationship(entity.id, closeness=10)
        self.assertEqual(adjusted.closeness, 100)
        with self.assertRaises(ValueError):
            self.store.adjust_relationship(entity.id, tension=11)

    def test_recovery_window_is_persistent_and_completed_once(self) -> None:
        self.store.touch_service(NOW)
        later = NOW + timedelta(hours=2)
        window = self.store.begin_recovery(later, minimum_gap=timedelta(minutes=5))
        self.assertIsNotNone(window)
        repeated = self.store.begin_recovery(
            later + timedelta(minutes=1), minimum_gap=timedelta(minutes=5)
        )
        self.assertEqual(repeated, window)
        self.assertTrue(self.store.complete_recovery(window, at=later))
        self.assertFalse(self.store.complete_recovery(window, at=later))
        self.assertIsNone(self.store.get_pending_recovery())

    def test_adaptive_initiative_policy_has_hard_bounds_and_guards(self) -> None:
        distant = Relationship(
            "distant",
            closeness=0,
            reciprocity=0,
            tension=100,
            awaiting_reply=False,
            blocked=False,
            last_interaction_at=NOW,
            last_initiative_at=None,
            updated_at=NOW,
        )
        close = Relationship(
            "close",
            closeness=100,
            reciprocity=100,
            tension=0,
            awaiting_reply=False,
            blocked=False,
            last_interaction_at=NOW - timedelta(days=20),
            last_initiative_at=None,
            updated_at=NOW,
        )
        self.assertEqual(adaptive_initiative_cooldown(distant, now=NOW), timedelta(hours=72))
        self.assertEqual(adaptive_initiative_cooldown(close, now=NOW), timedelta(hours=2))
        self.assertTrue(initiative_allowed(close, now=NOW))
        self.assertFalse(
            initiative_allowed(
                Relationship(
                    **{
                        **close.__dict__,
                        "awaiting_reply": True,
                    }
                ),
                now=NOW,
            )
        )
        self.assertFalse(initiative_allowed(close, now=NOW, sleeping=True))


if __name__ == "__main__":
    unittest.main()
