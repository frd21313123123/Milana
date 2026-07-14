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
    TelegramOutboxSentPart,
    TelegramTurnMetric,
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
                    "telegram_notice_journal",
                    "telegram_outbox",
                    "telegram_outbox_notice_owners",
                    "telegram_ack_intents",
                    "telegram_turn_metrics",
                    "state_change_ledger",
                    "chat_messages",
                    "pulse_tasks",
                }.issubset(names)
            )

    def test_latency_sla_marker_is_added_to_legacy_metrics_table(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    """
                    CREATE TABLE telegram_turn_metrics (
                        turn_id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        context_ms REAL NOT NULL,
                        provider_queue_ms REAL NOT NULL,
                        model_ms REAL NOT NULL,
                        send_ms REAL NOT NULL,
                        generation_to_first_send_ms REAL NOT NULL,
                        model_rounds INTEGER NOT NULL,
                        context_messages INTEGER NOT NULL,
                        context_characters INTEGER NOT NULL,
                        started_at TEXT NOT NULL,
                        first_sent_at TEXT
                    )
                    """
                )
                connection.commit()
            finally:
                connection.close()

            state = MilanaStateStore(path)
            state.close()
            connection = sqlite3.connect(path)
            try:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(telegram_turn_metrics)"
                    )
                }
            finally:
                connection.close()
            self.assertIn("sla_eligible", columns)

    def test_pending_telegram_notice_survives_reopen_until_handled(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "milana.sqlite3"
            payload = {
                "source": "telegram",
                "notice_id": "tg:10:7",
                "chat_id": 10,
                "message_id": 7,
                "timestamp": NOW.isoformat(),
                "sender": {"id": 20, "display_name": "Лера"},
                "media_type": "text",
            }
            store = MilanaStateStore(path)
            self.assertEqual(
                store.record_telegram_notice(payload, received_at=NOW), "created"
            )
            store.close()

            reopened = MilanaStateStore(path)
            try:
                self.assertEqual(reopened.list_pending_telegram_notices(), [payload])
                self.assertEqual(
                    reopened.record_telegram_notice(payload, received_at=NOW),
                    "pending",
                )
                self.assertEqual(
                    reopened.complete_telegram_notices(["tg:10:7"], handled_at=NOW),
                    1,
                )
                self.assertEqual(reopened.list_pending_telegram_notices(), [])
                self.assertEqual(
                    reopened.record_telegram_notice(payload, received_at=NOW),
                    "handled",
                )
            finally:
                reopened.close()

    def test_ack_intent_atomically_terminals_notice_and_survives_reopen(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "milana.sqlite3"
            payload = {
                "source": "telegram",
                "notice_id": "tg:10:7",
                "chat_id": 10,
                "message_id": 7,
                "timestamp": NOW.isoformat(),
                "sender": {"id": 20, "display_name": "Лера"},
                "media_type": "text",
            }
            store = MilanaStateStore(path)
            store.record_telegram_notice(payload, received_at=NOW)

            intent = store.prepare_telegram_ack_intent(
                "turn-7:ack", 10, ["tg:10:7"], [7], at=NOW
            )

            self.assertEqual(intent.status, "pending")
            self.assertEqual(intent.notice_ids, ("tg:10:7",))
            self.assertEqual(intent.message_ids, (7,))
            self.assertEqual(store.list_pending_telegram_notices(), [])
            self.assertEqual(
                store.record_telegram_notice(payload, received_at=NOW), "handled"
            )
            store.close()

            reopened = MilanaStateStore(path)
            pending = reopened.list_pending_telegram_ack_intents()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].action_key, "turn-7:ack")
            self.assertTrue(
                reopened.fail_telegram_ack_intent(
                    "turn-7:ack", "host response was lost"
                )
            )
            with self.assertRaises(StateConflictError):
                reopened.prepare_telegram_ack_intent(
                    "turn-7:ack", 11, ["tg:10:7"], [7], at=NOW
                )
            self.assertTrue(
                reopened.complete_telegram_ack_intent("turn-7:ack", at=NOW)
            )
            self.assertEqual(reopened.list_pending_telegram_ack_intents(), [])
            reopened.close()

    def test_notice_attempts_survive_reopen_and_poison_notice_stops(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "milana.sqlite3"
            payload = {
                "source": "telegram",
                "notice_id": "tg:10:poison",
                "chat_id": 10,
                "message_id": 8,
            }
            retry_at = NOW - timedelta(seconds=1)

            store = MilanaStateStore(path)
            self.assertEqual(store.record_telegram_notice(payload), "created")
            self.assertEqual(
                store.fail_telegram_notices(
                    ["tg:10:poison"], "first failure", retry_at=retry_at
                ),
                1,
            )
            store.close()

            reopened = MilanaStateStore(path)
            self.assertEqual(
                reopened.telegram_notice_attempt_count(["tg:10:poison"]), 1
            )
            self.assertEqual(reopened.list_pending_telegram_notices(), [payload])
            self.assertEqual(
                reopened.fail_telegram_notices(
                    ["tg:10:poison"], "second failure", retry_at=retry_at
                ),
                1,
            )
            self.assertEqual(
                reopened.fail_telegram_notices(
                    ["tg:10:poison"], "third failure", retry_at=retry_at
                ),
                1,
            )
            reopened.close()

            poisoned = MilanaStateStore(path)
            try:
                self.assertEqual(
                    poisoned.telegram_notice_attempt_count(["tg:10:poison"]), 3
                )
                self.assertEqual(poisoned.list_pending_telegram_notices(), [])
                self.assertEqual(
                    poisoned.record_telegram_notice(payload, received_at=NOW), "dead"
                )
            finally:
                poisoned.close()

    def test_outbox_partial_progress_resumes_after_reopen(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "milana.sqlite3"
            store = MilanaStateStore(path)
            prepared = store.prepare_telegram_outbox(
                "turn-1:send", 10, ["tg:10:8"], ["one", "two", "three"]
            )
            self.assertEqual(prepared.next_part_index, 0)
            partial = store.advance_telegram_outbox(
                "turn-1:send",
                sent_part_indexes=[0],
                sent_message_ids=[101],
                next_part_index=1,
                complete=False,
                first_sent_at=NOW,
            )
            self.assertEqual(partial.status, "pending")
            store.close()

            reopened = MilanaStateStore(path)
            resumed = reopened.prepare_telegram_outbox(
                "turn-1:send", 10, ["tg:10:8"], ["one", "two", "three"]
            )
            self.assertEqual(resumed.messages, ("one", "two", "three"))
            self.assertEqual(resumed.next_part_index, 1)
            self.assertEqual(resumed.sent_message_ids, (101,))
            self.assertEqual(resumed.first_sent_at, NOW)
            completed = reopened.advance_telegram_outbox(
                "turn-1:send",
                sent_part_indexes=[1, 2],
                sent_message_ids=[102, 103],
                next_part_index=3,
                complete=True,
                first_sent_at=NOW + timedelta(seconds=5),
            )
            self.assertEqual(completed.status, "sent")
            self.assertEqual(completed.sent_message_ids, (101, 102, 103))
            self.assertEqual(completed.first_sent_at, NOW)
            reopened.close()

            final = MilanaStateStore(path)
            try:
                persisted = final.prepare_telegram_outbox(
                    "turn-1:send", 10, ["tg:10:8"], ["one", "two", "three"]
                )
                self.assertEqual(persisted.status, "sent")
                self.assertEqual(persisted.next_part_index, 3)
            finally:
                final.close()

    def test_outbox_rejects_action_key_payload_collisions_and_cursor_jumps(self) -> None:
        store = MilanaStateStore()
        original = store.prepare_telegram_outbox(
            "turn-immutable:send", 10, ["tg:10:8"], ["one", "two"]
        )
        self.assertEqual(
            store.prepare_telegram_outbox(
                "turn-immutable:send", "10", ["tg:10:8"], ["one", "two"]
            ),
            original,
        )
        collisions = (
            (11, ["tg:10:8"], ["one", "two"]),
            (10, ["tg:10:9"], ["one", "two"]),
            (10, ["tg:10:8"], ["changed", "two"]),
        )
        for target, notices, messages in collisions:
            with self.subTest(target=target, notices=notices, messages=messages):
                with self.assertRaises(StateConflictError):
                    store.prepare_telegram_outbox(
                        "turn-immutable:send", target, notices, messages
                    )

        invalid_progress = (
            {
                "sent_part_indexes": [1],
                "sent_message_ids": [102],
                "next_part_index": 2,
                "complete": True,
            },
            {
                "sent_part_indexes": [1, 0],
                "sent_message_ids": [102, 101],
                "next_part_index": 2,
                "complete": True,
            },
            {
                "sent_part_indexes": [0],
                "sent_message_ids": [],
                "next_part_index": 1,
                "complete": False,
            },
            {
                "sent_part_indexes": [0],
                "sent_message_ids": [101],
                "next_part_index": 1,
                "complete": True,
            },
        )
        for arguments in invalid_progress:
            with self.subTest(arguments=arguments):
                with self.assertRaises((StateConflictError, ValueError)):
                    store.advance_telegram_outbox(
                        "turn-immutable:send", **arguments
                    )
        unchanged = store.prepare_telegram_outbox(
            "turn-immutable:send", 10, ["tg:10:8"], ["one", "two"]
        )
        self.assertEqual(unchanged.next_part_index, 0)
        self.assertEqual(unchanged.sent_parts, ())
        store.close()

    def test_outbox_lost_rpc_response_records_deduplicated_part_without_id(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "milana.sqlite3"
            store = MilanaStateStore(path)
            store.prepare_telegram_outbox(
                "stable-batch", 10, ["tg:10:20"], ["first", "second"]
            )
            # The first Telegram RPC succeeded, but its response was lost.  No
            # local progress was committed before the process restarted.
            store.close()

            reopened = MilanaStateStore(path)
            retry = reopened.prepare_telegram_outbox(
                "stable-batch", 10, ["tg:10:20"], ["first", "second"]
            )
            self.assertEqual(retry.next_part_index, 0)
            deduplicated = reopened.advance_telegram_outbox(
                "stable-batch",
                sent_part_indexes=[0],
                sent_message_ids=[],
                deduplicated_part_indexes=[0],
                next_part_index=1,
                complete=False,
                first_sent_at=NOW,
            )
            self.assertEqual(
                deduplicated.sent_parts, (TelegramOutboxSentPart(0, None),)
            )
            self.assertIsNone(deduplicated.message_id_for_part(0))
            reopened.close()

            final = MilanaStateStore(path)
            resumed = final.prepare_telegram_outbox(
                "stable-batch", 10, ["tg:10:20"], ["first", "second"]
            )
            self.assertEqual(resumed.sent_parts, (TelegramOutboxSentPart(0, None),))
            completed = final.advance_telegram_outbox(
                "stable-batch",
                sent_part_indexes=[1],
                sent_message_ids=[202],
                next_part_index=2,
                complete=True,
            )
            self.assertEqual(
                completed.sent_parts,
                (
                    TelegramOutboxSentPart(0, None),
                    TelegramOutboxSentPart(1, 202),
                ),
            )
            self.assertEqual(completed.sent_message_ids, (202,))
            self.assertEqual(completed.message_id_for_part(1), 202)
            final.close()

    def test_outbox_notice_owner_lookup_includes_pending_and_sent(self) -> None:
        store = MilanaStateStore()
        store.prepare_telegram_outbox("batch-a", 10, ["notice-a"], ["sent"])
        store.advance_telegram_outbox(
            "batch-a",
            sent_part_indexes=[0],
            sent_message_ids=[100],
            next_part_index=1,
            complete=True,
        )
        store.prepare_telegram_outbox("batch-b", 10, ["notice-b"], ["pending"])

        owner_a = store.find_telegram_outbox_for_notice_ids(["notice-a"])
        owner_b = store.find_telegram_outbox_for_notice_ids(["notice-b"])
        self.assertIsNotNone(owner_a)
        self.assertIsNotNone(owner_b)
        self.assertEqual(owner_a.action_key, "batch-a")
        self.assertEqual(owner_a.status, "sent")
        self.assertEqual(owner_b.action_key, "batch-b")
        self.assertEqual(owner_b.status, "pending")
        self.assertIsNone(store.find_telegram_outbox_for_notice_ids(["missing"]))
        with self.assertRaisesRegex(
            StateConflictError, "batch-a, batch-b"
        ):
            store.find_telegram_outbox_for_notice_ids(["notice-b", "notice-a"])
        with self.assertRaises(StateConflictError):
            store.prepare_telegram_outbox(
                "batch-c", 10, ["notice-a", "notice-c"], ["collision"]
            )
        store.close()

    def test_pending_initiative_outbox_is_target_scoped_until_sent(self) -> None:
        store = MilanaStateStore()
        pending = store.prepare_telegram_outbox(
            "initiative-old", 77, [], ["не потеряй меня"]
        )

        self.assertEqual(
            store.find_pending_telegram_outbox_for_target(77), pending
        )
        self.assertIsNone(store.find_pending_telegram_outbox_for_target(78))

        completed = store.advance_telegram_outbox(
            pending.action_key,
            sent_part_indexes=[0],
            sent_message_ids=[707],
            next_part_index=1,
            complete=True,
        )
        self.assertEqual(completed.status, "sent")
        self.assertIsNone(store.find_pending_telegram_outbox_for_target(77))
        store.close()

    def test_multiple_pending_initiatives_for_target_are_rejected(self) -> None:
        store = MilanaStateStore()
        store.prepare_telegram_outbox("initiative-a", 77, [], ["one"])
        store.prepare_telegram_outbox("initiative-b", 77, [], ["two"])

        with self.assertRaisesRegex(StateConflictError, "initiative-a, initiative-b"):
            store.find_pending_telegram_outbox_for_target(77)
        store.close()

    def test_outbox_additive_migration_binds_legacy_progress_by_part(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-state.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE telegram_outbox (
                    action_key TEXT PRIMARY KEY,
                    target_ref TEXT NOT NULL,
                    notice_ids_json TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    next_part_index INTEGER NOT NULL DEFAULT 0,
                    sent_message_ids_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    first_sent_at TEXT
                );
                INSERT INTO telegram_outbox (
                    action_key, target_ref, notice_ids_json, messages_json,
                    next_part_index, sent_message_ids_json, status,
                    created_at, updated_at, first_sent_at
                ) VALUES (
                    'legacy', '10', '["notice-old"]', '["one","two","three"]',
                    2, '[501]', 'pending',
                    '2026-07-14T10:00:00+00:00',
                    '2026-07-14T10:00:00+00:00',
                    '2026-07-14T10:00:00+00:00'
                );
                """
            )
            connection.close()

            store = MilanaStateStore(path)
            migrated = store.prepare_telegram_outbox(
                "legacy", 10, ["notice-old"], ["one", "two", "three"]
            )
            self.assertEqual(
                migrated.sent_parts,
                (
                    TelegramOutboxSentPart(0, 501),
                    TelegramOutboxSentPart(1, None),
                ),
            )
            completed = store.advance_telegram_outbox(
                "legacy",
                sent_part_indexes=[2],
                sent_message_ids=[503],
                next_part_index=3,
                complete=True,
            )
            self.assertEqual(completed.message_id_for_part(2), 503)
            store.close()

            connection = sqlite3.connect(path)
            try:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(telegram_outbox)"
                    )
                }
            finally:
                connection.close()
            self.assertIn("sent_parts_json", columns)

    def test_latency_percentiles_use_only_successful_first_sends(self) -> None:
        store = MilanaStateStore()

        def record(
            turn_id: str,
            duration_ms: float,
            *,
            outcome: str = "sent",
            first_sent: bool = True,
            offset: int = 0,
            model_rounds: int = 1,
        ) -> None:
            started_at = NOW + timedelta(seconds=offset)
            store.record_telegram_turn_metric(
                TelegramTurnMetric(
                    turn_id=turn_id,
                    chat_id="10",
                    outcome=outcome,
                    context_ms=float(offset),
                    provider_queue_ms=float(offset * 2),
                    model_ms=float(offset * 3),
                    send_ms=float(offset * 4),
                    generation_to_first_send_ms=duration_ms,
                    model_rounds=model_rounds,
                    context_messages=20,
                    context_characters=12_000,
                    started_at=started_at,
                    first_sent_at=(
                        started_at + timedelta(milliseconds=duration_ms)
                        if first_sent
                        else None
                    ),
                )
            )

        for index in range(1, 21):
            record(
                f"sent-{index}",
                float(index * 1_000),
                offset=index,
                model_rounds=1 if index <= 10 else 2,
            )
        record("error", 999_000.0, outcome="error", offset=21)
        record("no-first-send", 888_000.0, first_sent=False, offset=22)
        record("resumed", 777_000.0, outcome="resumed", offset=23)
        record("media", 666_000.0, outcome="sent:media", offset=24)

        summary = store.telegram_latency_summary(limit=100, target_seconds=10)
        self.assertEqual(summary["turn_count"], 24)
        self.assertEqual(summary["sample_size"], 20)
        self.assertEqual(summary["p50_ms"], 10_000.0)
        self.assertEqual(summary["p95_ms"], 19_000.0)
        self.assertEqual(summary["p99_ms"], 20_000.0)
        self.assertEqual(summary["breaches"], 10)
        self.assertEqual(summary["breach_rate"], 0.5)
        self.assertEqual(summary["exceedances"], 10)
        self.assertEqual(summary["exceed_rate"], 0.5)
        self.assertEqual(
            summary["generation_to_first_send_ms"],
            {"average": 10_500.0, "p50": 10_000.0, "p95": 19_000.0, "p99": 20_000.0},
        )
        self.assertEqual(summary["average_model_rounds"], 1.5)
        self.assertEqual(
            summary["llm_calls"],
            {"total": 30, "average": 1.5, "p50": 1.0, "p95": 2.0, "p99": 2.0},
        )
        self.assertEqual(
            summary["phases"]["context_ms"],
            {"average": 10.5, "p50": 10.0, "p95": 19.0, "p99": 20.0},
        )
        self.assertEqual(summary["phases"]["provider_queue_ms"]["p95"], 38.0)
        self.assertEqual(summary["phases"]["model_ms"]["p99"], 60.0)
        self.assertEqual(summary["phases"]["send_ms"]["average"], 42.0)
        self.assertEqual(
            summary["outcomes"],
            {"error": 1, "resumed": 1, "sent": 21, "sent:media": 1},
        )
        store.close()

    def test_empty_latency_window_has_no_false_zero_percent_success(self) -> None:
        store = MilanaStateStore()
        try:
            summary = store.telegram_latency_summary(limit=100, target_seconds=10)
            self.assertEqual(summary["sample_size"], 0)
            self.assertIsNone(summary["p95_ms"])
            self.assertIsNone(summary["breach_rate"])
            self.assertIsNone(summary["exceed_rate"])
            self.assertIsNone(summary["llm_calls"]["average"])
            self.assertIsNone(summary["phases"]["model_ms"]["p95"])
        finally:
            store.close()

    def test_latency_delivery_completeness_includes_fast_errors_and_censoring(self) -> None:
        store = MilanaStateStore()
        try:
            store.record_telegram_turn_metric(
                TelegramTurnMetric(
                    turn_id="delivered",
                    chat_id="10",
                    outcome="sent",
                    context_ms=1.0,
                    provider_queue_ms=2.0,
                    model_ms=500.0,
                    send_ms=10.0,
                    generation_to_first_send_ms=1_000.0,
                    model_rounds=1,
                    context_messages=2,
                    context_characters=20,
                    started_at=NOW,
                    first_sent_at=NOW + timedelta(seconds=1),
                    sla_eligible=True,
                )
            )
            for index in range(4):
                store.record_telegram_turn_metric(
                    TelegramTurnMetric(
                        turn_id=f"failed-{index}",
                        chat_id="10",
                        outcome="error:AgyError",
                        context_ms=1.0,
                        provider_queue_ms=2.0,
                        model_ms=500.0,
                        send_ms=0.0,
                        generation_to_first_send_ms=300_000.0,
                        model_rounds=1,
                        context_messages=2,
                        context_characters=20,
                        started_at=NOW + timedelta(seconds=index + 1),
                        first_sent_at=None,
                        sla_eligible=True,
                    )
                )

            summary = store.telegram_latency_summary(limit=20, target_seconds=10)

            self.assertEqual(summary["sample_size"], 1)
            self.assertEqual(summary["ordinary_text_turns"], 5)
            self.assertEqual(summary["delivered_turns"], 1)
            self.assertEqual(summary["failed_turns"], 4)
            self.assertEqual(summary["censored_turns"], 4)
            self.assertEqual(summary["delivery_rate"], 0.2)
            self.assertEqual(summary["completeness_rate"], 0.2)
            self.assertEqual(
                summary["delivery"],
                {
                    "attempts": 5,
                    "delivered": 1,
                    "failed": 4,
                    "errors": 4,
                    "no_first_send": 0,
                    "censored": 4,
                    "rate": 0.2,
                },
            )
            self.assertFalse(summary["completeness"]["complete"])
            self.assertEqual(summary["completeness"]["missing_samples"], 4)
        finally:
            store.close()


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

    def test_heartbeat_changes_idempotency_survives_reopen(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "milana.sqlite3"
            store = MilanaStateStore(path)
            initial = store.get_agent_state()
            changes = HeartbeatChanges(
                entities=(NewEntity("person", "Лера", entity_id="lera"),),
                need_deltas={"social": 10},
                mood="радостное",
            )
            first = store.apply_heartbeat_changes(
                changes,
                expected_revision=initial.revision,
                record_heartbeat=False,
                idempotency_key="telegram:turn-42:state",
                at=NOW,
            )
            repeated = store.apply_heartbeat_changes(
                changes,
                expected_revision=initial.revision,
                record_heartbeat=False,
                idempotency_key="telegram:turn-42:state",
                at=NOW + timedelta(seconds=1),
            )
            self.assertEqual(repeated, first)
            self.assertEqual(first.social, initial.social + 10)
            self.assertEqual(len(store.list_entities()), 1)
            store.close()

            reopened = MilanaStateStore(path)
            try:
                persisted = reopened.apply_heartbeat_changes(
                    changes,
                    expected_revision=initial.revision,
                    record_heartbeat=False,
                    idempotency_key="telegram:turn-42:state",
                    at=NOW + timedelta(seconds=2),
                )
                self.assertEqual(persisted, first)
                self.assertEqual(len(reopened.list_entities()), 1)
            finally:
                reopened.close()

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
