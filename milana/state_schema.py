"""SQLite schema bootstrap and additive migrations for Milana state."""

from __future__ import annotations

import sqlite3

from milana.state_models import _json_load


def create_state_schema(connection: sqlite3.Connection) -> None:
    # executescript normally commits before execution.  Keeping BEGIN and
    # COMMIT inside the script makes this additive migration atomic.
    script = """
    BEGIN IMMEDIATE;

    CREATE TABLE IF NOT EXISTS agent_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        revision INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
        mood TEXT NOT NULL DEFAULT 'спокойное',
        valence INTEGER NOT NULL DEFAULT 0 CHECK (valence BETWEEN -100 AND 100),
        arousal INTEGER NOT NULL DEFAULT 50 CHECK (arousal BETWEEN 0 AND 100),
        social_need INTEGER NOT NULL DEFAULT 50 CHECK (social_need BETWEEN 0 AND 100),
        rest_need INTEGER NOT NULL DEFAULT 50 CHECK (rest_need BETWEEN 0 AND 100),
        novelty_need INTEGER NOT NULL DEFAULT 50 CHECK (novelty_need BETWEEN 0 AND 100),
        achievement_need INTEGER NOT NULL DEFAULT 50 CHECK (achievement_need BETWEEN 0 AND 100),
        current_intention TEXT,
        current_intention_updated_at TEXT,
        last_heartbeat_at TEXT,
        next_heartbeat_at TEXT,
        heartbeat_paused INTEGER NOT NULL DEFAULT 0 CHECK (heartbeat_paused IN (0, 1)),
        last_service_seen_at TEXT,
        recovery_pending_from TEXT,
        recovery_pending_to TEXT,
        recovery_completed_through TEXT,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS heartbeat_jobs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        due_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
        payload_json TEXT NOT NULL DEFAULT '{}',
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        idempotency_key TEXT UNIQUE,
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_heartbeat_jobs_status_due
        ON heartbeat_jobs(status, due_at);

    CREATE TABLE IF NOT EXISTS world_entities (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        is_real INTEGER NOT NULL DEFAULT 0 CHECK (is_real IN (0, 1)),
        status TEXT NOT NULL DEFAULT 'active'
            CHECK (status IN ('active', 'archived')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_world_entities_status_kind
        ON world_entities(status, kind);

    CREATE TABLE IF NOT EXISTS world_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_id TEXT NOT NULL REFERENCES world_entities(id) ON DELETE CASCADE,
        fact_key TEXT NOT NULL,
        value_json TEXT NOT NULL,
        locked INTEGER NOT NULL DEFAULT 0 CHECK (locked IN (0, 1)),
        version INTEGER NOT NULL CHECK (version >= 1),
        source TEXT,
        valid_from TEXT NOT NULL,
        superseded_at TEXT,
        UNIQUE (entity_id, fact_key, version)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_world_facts_current
        ON world_facts(entity_id, fact_key)
        WHERE superseded_at IS NULL;

    CREATE TABLE IF NOT EXISTS life_events (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        importance INTEGER NOT NULL DEFAULT 50 CHECK (importance BETWEEN 0 AND 100),
        entity_ids_json TEXT NOT NULL DEFAULT '[]',
        happened_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active'
            CHECK (status IN ('active', 'archived')),
        raw_payload_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_life_events_status_happened
        ON life_events(status, happened_at DESC);

    CREATE TABLE IF NOT EXISTS goals (
        id TEXT PRIMARY KEY,
        horizon TEXT NOT NULL CHECK (horizon IN ('short', 'long')),
        title TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'active'
            CHECK (status IN ('active', 'completed', 'archived')),
        progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_goals_status_horizon
        ON goals(status, horizon, updated_at DESC);

    CREATE TABLE IF NOT EXISTS relationships (
        entity_id TEXT PRIMARY KEY REFERENCES world_entities(id) ON DELETE CASCADE,
        closeness INTEGER NOT NULL DEFAULT 50 CHECK (closeness BETWEEN 0 AND 100),
        reciprocity INTEGER NOT NULL DEFAULT 50 CHECK (reciprocity BETWEEN 0 AND 100),
        tension INTEGER NOT NULL DEFAULT 0 CHECK (tension BETWEEN 0 AND 100),
        awaiting_reply INTEGER NOT NULL DEFAULT 0 CHECK (awaiting_reply IN (0, 1)),
        blocked INTEGER NOT NULL DEFAULT 0 CHECK (blocked IN (0, 1)),
        last_interaction_at TEXT,
        last_initiative_at TEXT,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS world_summaries (
        id TEXT PRIMARY KEY,
        period_start TEXT NOT NULL,
        period_end TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (period_start, period_end),
        CHECK (period_end > period_start)
    );
    CREATE INDEX IF NOT EXISTS idx_world_summaries_period
        ON world_summaries(period_end DESC);

    CREATE TABLE IF NOT EXISTS skill_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id TEXT NOT NULL,
        skill_id TEXT NOT NULL,
        action TEXT NOT NULL,
        success INTEGER NOT NULL CHECK (success IN (0, 1)),
        detail_json TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_skill_audit_turn_created
        ON skill_audit(turn_id, created_at);

    CREATE TABLE IF NOT EXISTS telegram_notice_journal (
        notice_id TEXT PRIMARY KEY,
        payload_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'handled')),
        received_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        handled_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_telegram_notice_journal_status_received
        ON telegram_notice_journal(status, received_at);

    CREATE TABLE IF NOT EXISTS telegram_outbox (
        action_key TEXT PRIMARY KEY,
        target_ref TEXT NOT NULL,
        notice_ids_json TEXT NOT NULL,
        messages_json TEXT NOT NULL,
        next_part_index INTEGER NOT NULL DEFAULT 0,
        sent_message_ids_json TEXT NOT NULL DEFAULT '[]',
        sent_parts_json TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'sent')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        first_sent_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_telegram_outbox_status_updated
        ON telegram_outbox(status, updated_at);

    CREATE TABLE IF NOT EXISTS telegram_outbox_notice_owners (
        notice_id TEXT NOT NULL,
        action_key TEXT NOT NULL,
        PRIMARY KEY (notice_id, action_key),
        FOREIGN KEY (action_key) REFERENCES telegram_outbox(action_key)
            ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_telegram_outbox_notice_owner_action
        ON telegram_outbox_notice_owners(action_key);

    CREATE TABLE IF NOT EXISTS telegram_notice_action_owners (
        notice_id TEXT NOT NULL,
        action_key TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (notice_id, action_key)
    );
    CREATE INDEX IF NOT EXISTS idx_telegram_notice_action_owner_action
        ON telegram_notice_action_owners(action_key);

    CREATE TABLE IF NOT EXISTS telegram_ack_intents (
        action_key TEXT PRIMARY KEY,
        target_ref TEXT NOT NULL,
        notice_ids_json TEXT NOT NULL,
        message_ids_json TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'completed')),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        last_error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        completed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_telegram_ack_intents_status_updated
        ON telegram_ack_intents(status, updated_at);

    CREATE TABLE IF NOT EXISTS telegram_turn_metrics (
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
        first_sent_at TEXT,
        sla_eligible INTEGER CHECK (sla_eligible IN (0, 1))
    );
    CREATE INDEX IF NOT EXISTS idx_telegram_turn_metrics_started
        ON telegram_turn_metrics(started_at DESC);

    CREATE TABLE IF NOT EXISTS state_change_ledger (
        action_key TEXT PRIMARY KEY,
        applied_at TEXT NOT NULL
    );

    INSERT OR IGNORE INTO agent_state (id, updated_at)
    VALUES (1, strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now'));

    COMMIT;
    """
    try:
        connection.executescript(script)
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(telegram_notice_journal)"
            ).fetchall()
        }
        for name, declaration in (
            ("attempts", "INTEGER NOT NULL DEFAULT 0"),
            ("next_attempt_at", "TEXT"),
            ("last_error", "TEXT"),
        ):
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE telegram_notice_journal ADD COLUMN {name} {declaration}"
                )
        outbox_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(telegram_outbox)"
            ).fetchall()
        }
        if "sent_parts_json" not in outbox_columns:
            connection.execute(
                "ALTER TABLE telegram_outbox "
                "ADD COLUMN sent_parts_json TEXT NOT NULL DEFAULT '[]'"
            )
        metric_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(telegram_turn_metrics)"
            ).fetchall()
        }
        if "sla_eligible" not in metric_columns:
            connection.execute(
                "ALTER TABLE telegram_turn_metrics "
                "ADD COLUMN sla_eligible INTEGER "
                "CHECK (sla_eligible IN (0, 1))"
            )
        agent_state_columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(agent_state)"
            ).fetchall()
        }
        if "current_intention_updated_at" not in agent_state_columns:
            connection.execute(
                "ALTER TABLE agent_state "
                "ADD COLUMN current_intention_updated_at TEXT"
            )
            connection.execute(
                """
                UPDATE agent_state
                SET current_intention_updated_at = COALESCE(
                    last_heartbeat_at,
                    updated_at
                )
                WHERE current_intention IS NOT NULL
                """
            )
        # Backfill the normalized ownership index for databases created by
        # the original JSON-only outbox schema.  Keep all conflicting old
        # owners so lookup can reject ambiguity instead of choosing one.
        outbox_rows = connection.execute(
            "SELECT action_key, notice_ids_json FROM telegram_outbox "
            "ORDER BY action_key"
        ).fetchall()
        for row in outbox_rows:
            for notice_id in _json_load(row["notice_ids_json"]):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO telegram_outbox_notice_owners (
                        notice_id, action_key
                    ) VALUES (?, ?)
                    """,
                    (str(notice_id), str(row["action_key"])),
                )
        connection.commit()
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise

__all__ = ["create_state_schema"]
