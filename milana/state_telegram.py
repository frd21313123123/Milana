"""Durable Telegram notice, outbox, acknowledgement, and latency persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from milana.state_models import (
    StateConflictError,
    TelegramAckIntent,
    TelegramOutboxEntry,
    TelegramOutboxSentPart,
    TelegramTurnMetric,
    _clean_text,
    _identifier,
    _integer,
    _json_dump,
    _json_load,
    _now,
    _parse_timestamp,
    _timestamp,
)


class TelegramStateStoreMixin:
    """Telegram-specific persistence mixed into the main SQLite state store."""

    def record_telegram_notice(
        self,
        payload: Mapping[str, Any],
        *,
        received_at: datetime | None = None,
    ) -> str:
        """Durably accept a metadata-only Telegram notice before scheduling it."""

        if not isinstance(payload, Mapping):
            raise TypeError("Telegram notice payload должен быть mapping")
        notice_id = _identifier(payload.get("notice_id"), "Telegram notice ID")
        serialized = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
        timestamp = _timestamp(received_at or _now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    """
                    SELECT status, attempts, next_attempt_at
                    FROM telegram_notice_journal WHERE notice_id = ?
                    """,
                    (notice_id,),
                ).fetchone()
                if row is not None and row["status"] == "handled":
                    self._connection.commit()
                    return "handled"
                if row is not None and row["next_attempt_at"] is not None:
                    retry_at = _parse_timestamp(str(row["next_attempt_at"]))
                    if retry_at is not None and retry_at > (received_at or _now()):
                        self._connection.commit()
                        return "deferred"
                if row is None:
                    self._connection.execute(
                        """
                        INSERT INTO telegram_notice_journal (
                            notice_id, payload_json, status, received_at, updated_at
                        ) VALUES (?, ?, 'pending', ?, ?)
                        """,
                        (notice_id, serialized, timestamp, timestamp),
                    )
                    result = "created"
                else:
                    self._connection.execute(
                        """
                        UPDATE telegram_notice_journal
                        SET payload_json = ?, updated_at = ?
                        WHERE notice_id = ? AND status = 'pending'
                        """,
                        (serialized, timestamp, notice_id),
                    )
                    result = "pending"
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def list_pending_telegram_notices(
        self, *, limit: int = 1000, include_deferred: bool = False
    ) -> list[dict[str, Any]]:
        limit = _integer(limit, "Лимит Telegram notices", 1, 10_000)
        if not isinstance(include_deferred, bool):
            raise TypeError("include_deferred должен быть bool")
        retry_filter = "" if include_deferred else """
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
        """
        parameters: tuple[Any, ...] = (
            (limit,) if include_deferred else (_timestamp(_now()), limit)
        )
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT payload_json FROM telegram_notice_journal
                WHERE status = 'pending'
                {retry_filter}
                ORDER BY received_at ASC LIMIT ?
                """,
                parameters,
            ).fetchall()
        notices: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"]))
            if isinstance(payload, dict):
                notices.append(payload)
        return notices

    def complete_telegram_notices(
        self,
        notice_ids: Iterable[str],
        *,
        handled_at: datetime | None = None,
    ) -> int:
        normalized = tuple(
            dict.fromkeys(_identifier(item, "Telegram notice ID") for item in notice_ids)
        )
        if not normalized:
            return 0
        timestamp = _timestamp(handled_at or _now())
        placeholders = ", ".join("?" for _ in normalized)
        with self._lock:
            cursor = self._connection.execute(
                f"""
                UPDATE telegram_notice_journal
                SET status = 'handled', handled_at = ?, updated_at = ?,
                    next_attempt_at = NULL, last_error = NULL
                WHERE notice_id IN ({placeholders}) AND status = 'pending'
                """,
                (timestamp, timestamp, *normalized),
            )
            self._connection.commit()
            return int(cursor.rowcount)

    def fail_telegram_notices(
        self,
        notice_ids: Iterable[str],
        error: str,
        *,
        retry_at: datetime,
    ) -> int:
        """Persist a failed attempt so backfill cannot reset a poison notice."""

        normalized = tuple(
            dict.fromkeys(_identifier(item, "Telegram notice ID") for item in notice_ids)
        )
        if not normalized:
            return 0
        clean_error = _clean_text(
            error or "unknown Telegram generation error",
            "Ошибка Telegram notice",
            maximum=2_000,
        )
        retry_timestamp = _timestamp(retry_at)
        updated_at = _timestamp(_now())
        placeholders = ", ".join("?" for _ in normalized)
        with self._lock:
            cursor = self._connection.execute(
                f"""
                UPDATE telegram_notice_journal
                SET attempts = attempts + 1,
                    next_attempt_at = ?, last_error = ?, updated_at = ?
                WHERE notice_id IN ({placeholders}) AND status = 'pending'
                """,
                (retry_timestamp, clean_error, updated_at, *normalized),
            )
            self._connection.commit()
            return int(cursor.rowcount)

    def telegram_notice_attempt_count(self, notice_ids: Iterable[str]) -> int:
        normalized = tuple(
            dict.fromkeys(_identifier(item, "Telegram notice ID") for item in notice_ids)
        )
        if not normalized:
            return 0
        placeholders = ", ".join("?" for _ in normalized)
        with self._lock:
            row = self._connection.execute(
                f"""
                SELECT MAX(attempts) AS attempts
                FROM telegram_notice_journal
                WHERE notice_id IN ({placeholders})
                """,
                normalized,
            ).fetchone()
        return int(row["attempts"] or 0) if row is not None else 0

    @staticmethod
    def _telegram_ack_intent_from_row(row: sqlite3.Row) -> TelegramAckIntent:
        raw_message_ids = _json_load(str(row["message_ids_json"]))
        if not isinstance(raw_message_ids, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in raw_message_ids
        ):
            raise StateConflictError("Telegram ack intent содержит повреждённые message IDs")
        return TelegramAckIntent(
            action_key=str(row["action_key"]),
            target_ref=str(row["target_ref"]),
            notice_ids=tuple(
                str(item) for item in _json_load(str(row["notice_ids_json"]))
            ),
            message_ids=tuple(raw_message_ids),
            status=str(row["status"]),
            attempts=int(row["attempts"] or 0),
            last_error=(
                str(row["last_error"]) if row["last_error"] is not None else None
            ),
            created_at=_parse_timestamp(row["created_at"]) or _now(),
            updated_at=_parse_timestamp(row["updated_at"]) or _now(),
            completed_at=_parse_timestamp(row["completed_at"]),
        )

    def prepare_telegram_ack_intent(
        self,
        action_key: str,
        target_ref: str | int,
        notice_ids: Iterable[str],
        message_ids: Iterable[int],
        *,
        at: datetime | None = None,
    ) -> TelegramAckIntent:
        """Persist read-ack recovery and make its notices terminal atomically.

        Callers must invoke this only after full materialization and durable
        outbound delivery.  Once it commits, a lost RPC response cannot put
        the same incoming notice back through model generation; the remaining
        read marker is an idempotent side effect recovered from this row.
        """

        key = _identifier(action_key, "Telegram acknowledge action key")
        if isinstance(target_ref, bool) or not isinstance(target_ref, (str, int)):
            raise TypeError("Telegram acknowledge target должен быть строкой или числом")
        target = _clean_text(
            str(target_ref), "Telegram acknowledge target", maximum=255
        )
        notices = tuple(
            dict.fromkeys(
                _identifier(item, "Telegram notice ID") for item in notice_ids
            )
        )
        if not notices:
            raise ValueError("Telegram ack intent требует notice IDs")
        normalized_messages: list[int] = []
        for message_id in message_ids:
            if (
                isinstance(message_id, bool)
                or not isinstance(message_id, int)
                or message_id <= 0
            ):
                raise ValueError("Telegram ack intent содержит неверный message ID")
            if message_id not in normalized_messages:
                normalized_messages.append(message_id)
        messages = tuple(normalized_messages)
        if not messages:
            raise ValueError("Telegram ack intent требует message IDs")
        timestamp = _timestamp(at or _now())
        notices_json = _json_dump(list(notices))
        messages_json = _json_dump(list(messages))
        placeholders = ", ".join("?" for _ in notices)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM telegram_ack_intents WHERE action_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        """
                        INSERT INTO telegram_ack_intents (
                            action_key, target_ref, notice_ids_json,
                            message_ids_json, status, attempts, last_error,
                            created_at, updated_at, completed_at
                        ) VALUES (?, ?, ?, ?, 'pending', 0, NULL, ?, ?, NULL)
                        """,
                        (
                            key,
                            target,
                            notices_json,
                            messages_json,
                            timestamp,
                            timestamp,
                        ),
                    )
                else:
                    existing = self._telegram_ack_intent_from_row(row)
                    expected = (target, notices, messages)
                    actual = (
                        existing.target_ref,
                        existing.notice_ids,
                        existing.message_ids,
                    )
                    if actual != expected:
                        raise StateConflictError(
                            "Telegram acknowledge action key сменил payload"
                        )
                # This is intentionally in the same transaction as the intent.
                # A host retry may now safely see these notices as terminal.
                self._connection.execute(
                    f"""
                    UPDATE telegram_notice_journal
                    SET status = 'handled', handled_at = ?, updated_at = ?,
                        next_attempt_at = NULL, last_error = NULL
                    WHERE notice_id IN ({placeholders}) AND status = 'pending'
                    """,
                    (timestamp, timestamp, *notices),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            result = self._connection.execute(
                "SELECT * FROM telegram_ack_intents WHERE action_key = ?", (key,)
            ).fetchone()
        assert result is not None
        return self._telegram_ack_intent_from_row(result)

    def list_pending_telegram_ack_intents(
        self, *, limit: int = 1_000
    ) -> list[TelegramAckIntent]:
        limit = _integer(limit, "Лимит Telegram ack intents", 1, 10_000)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM telegram_ack_intents
                WHERE status = 'pending'
                ORDER BY created_at, action_key LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._telegram_ack_intent_from_row(row) for row in rows]

    def fail_telegram_ack_intent(self, action_key: str, error: str) -> bool:
        key = _identifier(action_key, "Telegram acknowledge action key")
        clean_error = _clean_text(
            error or "unknown Telegram acknowledge error",
            "Ошибка Telegram acknowledge",
            maximum=2_000,
        )
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE telegram_ack_intents
                SET attempts = attempts + 1, last_error = ?, updated_at = ?
                WHERE action_key = ? AND status = 'pending'
                """,
                (clean_error, _timestamp(_now()), key),
            )
            self._connection.commit()
        return cursor.rowcount == 1

    def complete_telegram_ack_intent(
        self, action_key: str, *, at: datetime | None = None
    ) -> bool:
        key = _identifier(action_key, "Telegram acknowledge action key")
        timestamp = _timestamp(at or _now())
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE telegram_ack_intents
                SET status = 'completed', completed_at = ?, updated_at = ?,
                    last_error = NULL
                WHERE action_key = ? AND status = 'pending'
                """,
                (timestamp, timestamp, key),
            )
            self._connection.commit()
        return cursor.rowcount == 1

    @staticmethod
    def _telegram_outbox_from_row(row: sqlite3.Row) -> TelegramOutboxEntry:
        next_part_index = int(row["next_part_index"])
        legacy_message_ids = tuple(
            int(item) for item in _json_load(row["sent_message_ids_json"])
        )
        raw_sent_parts: Any = []
        if "sent_parts_json" in row.keys():
            raw_sent_parts = _json_load(row["sent_parts_json"])
        sent_parts: list[TelegramOutboxSentPart] = []
        if raw_sent_parts:
            if not isinstance(raw_sent_parts, list):
                raise StateConflictError("Telegram outbox содержит повреждённый прогресс")
            for raw_part in raw_sent_parts:
                if not isinstance(raw_part, Mapping):
                    raise StateConflictError(
                        "Telegram outbox содержит повреждённый прогресс"
                    )
                part_index = raw_part.get("part_index")
                message_id = raw_part.get("message_id")
                if (
                    isinstance(part_index, bool)
                    or not isinstance(part_index, int)
                    or part_index < 0
                    or (
                        message_id is not None
                        and (
                            isinstance(message_id, bool)
                            or not isinstance(message_id, int)
                        )
                    )
                ):
                    raise StateConflictError(
                        "Telegram outbox содержит повреждённый прогресс"
                    )
                sent_parts.append(TelegramOutboxSentPart(part_index, message_id))
        elif next_part_index:
            # Additive migration from the original unbound ID array.  Old rows
            # were written in part order; if an ID was unavailable, preserve
            # the delivered part with an explicit unknown ID.
            sent_parts.extend(
                TelegramOutboxSentPart(
                    part_index,
                    (
                        legacy_message_ids[part_index]
                        if part_index < len(legacy_message_ids)
                        else None
                    ),
                )
                for part_index in range(next_part_index)
            )
        expected_indexes = list(range(next_part_index))
        if [part.part_index for part in sent_parts] != expected_indexes:
            raise StateConflictError("Telegram outbox содержит разрыв прогресса")
        return TelegramOutboxEntry(
            action_key=str(row["action_key"]),
            target_ref=str(row["target_ref"]),
            notice_ids=tuple(str(item) for item in _json_load(row["notice_ids_json"])),
            messages=tuple(str(item) for item in _json_load(row["messages_json"])),
            next_part_index=next_part_index,
            sent_parts=tuple(sent_parts),
            status=str(row["status"]),
            created_at=_parse_timestamp(row["created_at"]) or _now(),
            updated_at=_parse_timestamp(row["updated_at"]) or _now(),
            first_sent_at=_parse_timestamp(row["first_sent_at"]),
        )

    def find_telegram_outbox_for_notice_ids(
        self, notice_ids: Iterable[str]
    ) -> TelegramOutboxEntry | None:
        """Find the sole durable outbox owner intersecting ``notice_ids``.

        Both pending and completed rows remain owners.  A request intersecting
        more than one action is ambiguous and is rejected deterministically;
        silently choosing one could resend or acknowledge only half a batch.
        """

        normalized = tuple(
            dict.fromkeys(_identifier(item, "Telegram notice ID") for item in notice_ids)
        )
        if not normalized:
            return None
        placeholders = ", ".join("?" for _ in normalized)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT DISTINCT outbox.*
                FROM telegram_outbox AS outbox
                JOIN telegram_outbox_notice_owners AS owner
                  ON owner.action_key = outbox.action_key
                WHERE owner.notice_id IN ({placeholders})
                ORDER BY outbox.action_key
                """,
                normalized,
            ).fetchall()
        owners = [self._telegram_outbox_from_row(row) for row in rows]
        if len(owners) > 1:
            action_keys = ", ".join(entry.action_key for entry in owners)
            raise StateConflictError(
                "Telegram notices принадлежат разным outbox actions: " + action_keys
            )
        return owners[0] if owners else None

    def find_telegram_notice_action_owner(
        self, notice_ids: Iterable[str]
    ) -> str | None:
        """Find the durable side-effect batch intersecting ``notice_ids``."""

        normalized = tuple(
            dict.fromkeys(_identifier(item, "Telegram notice ID") for item in notice_ids)
        )
        if not normalized:
            return None
        placeholders = ", ".join("?" for _ in normalized)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT DISTINCT action_key
                FROM telegram_notice_action_owners
                WHERE notice_id IN ({placeholders})
                ORDER BY action_key
                """,
                normalized,
            ).fetchall()
        owners = [str(row["action_key"]) for row in rows]
        if len(owners) > 1:
            raise StateConflictError(
                "Telegram notices принадлежат разным side-effect actions: "
                + ", ".join(owners)
            )
        return owners[0] if owners else None

    def prepare_telegram_notice_action_owner(
        self, action_key: str, notice_ids: Iterable[str]
    ) -> str:
        """Bind a notice batch before its first non-outbox side effect."""

        key = _identifier(action_key, "Telegram side-effect action key")
        normalized = tuple(
            dict.fromkeys(_identifier(item, "Telegram notice ID") for item in notice_ids)
        )
        if not normalized:
            raise ValueError("Telegram side-effect owner требует notice IDs")
        placeholders = ", ".join("?" for _ in normalized)
        now = _timestamp(_now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                rows = self._connection.execute(
                    f"""
                    SELECT DISTINCT action_key
                    FROM telegram_notice_action_owners
                    WHERE notice_id IN ({placeholders}) AND action_key != ?
                    ORDER BY action_key
                    """,
                    (*normalized, key),
                ).fetchall()
                foreign_owners = [str(row["action_key"]) for row in rows]
                if foreign_owners:
                    raise StateConflictError(
                        "Telegram notices уже принадлежат side-effect action: "
                        + ", ".join(foreign_owners)
                    )
                self._connection.executemany(
                    """
                    INSERT OR IGNORE INTO telegram_notice_action_owners (
                        notice_id, action_key, created_at
                    ) VALUES (?, ?, ?)
                    """,
                    ((notice_id, key, now) for notice_id in normalized),
                )
                self._connection.commit()
                return key
            except Exception:
                self._connection.rollback()
                raise

    def find_pending_telegram_outbox_for_target(
        self, target_ref: int | str
    ) -> TelegramOutboxEntry | None:
        """Return the sole unfinished initiative send for a target.

        Initiative/heartbeat turns do not own incoming notice IDs, so their
        durable retry identity is the pending target-scoped row itself. Sent
        rows are deliberately ignored so later legitimate initiatives remain
        possible.
        """

        target = str(target_ref)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM telegram_outbox
                WHERE target_ref = ? AND status = 'pending'
                  AND notice_ids_json = '[]'
                ORDER BY created_at, action_key
                """,
                (target,),
            ).fetchall()
        owners = [self._telegram_outbox_from_row(row) for row in rows]
        if len(owners) > 1:
            action_keys = ", ".join(entry.action_key for entry in owners)
            raise StateConflictError(
                "Telegram target имеет несколько pending initiative outbox actions: "
                + action_keys
            )
        return owners[0] if owners else None

    def prepare_telegram_outbox(
        self,
        action_key: str,
        target_ref: int | str,
        notice_ids: Iterable[str],
        messages: Sequence[str],
    ) -> TelegramOutboxEntry:
        """Create an immutable send payload or return its prior retry state."""

        key = _identifier(action_key, "Telegram outbox action key")
        target = str(target_ref)
        normalized_notices = tuple(
            dict.fromkeys(_identifier(item, "Telegram notice ID") for item in notice_ids)
        )
        normalized_messages = tuple(
            _clean_text(item, "Telegram outbox message", maximum=4_096)
            for item in messages
        )
        if not normalized_messages:
            raise ValueError("Telegram outbox требует хотя бы одно сообщение")
        now = _timestamp(_now())
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if normalized_notices:
                    placeholders = ", ".join("?" for _ in normalized_notices)
                    owner_rows = self._connection.execute(
                        f"""
                        SELECT DISTINCT action_key
                        FROM telegram_outbox_notice_owners
                        WHERE notice_id IN ({placeholders}) AND action_key != ?
                        ORDER BY action_key
                        """,
                        (*normalized_notices, key),
                    ).fetchall()
                else:
                    owner_rows = []
                foreign_owners = [str(row["action_key"]) for row in owner_rows]
                if foreign_owners:
                    raise StateConflictError(
                        "Telegram notices уже принадлежат outbox action: "
                        + ", ".join(foreign_owners)
                    )
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO telegram_outbox (
                        action_key, target_ref, notice_ids_json, messages_json,
                        next_part_index, sent_message_ids_json, sent_parts_json,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, '[]', '[]', 'pending', ?, ?)
                    """,
                    (
                        key,
                        target,
                        _json_dump(normalized_notices),
                        _json_dump(normalized_messages),
                        now,
                        now,
                    ),
                )
                self._connection.executemany(
                    """
                    INSERT OR IGNORE INTO telegram_outbox_notice_owners (
                        notice_id, action_key
                    ) VALUES (?, ?)
                    """,
                    ((notice_id, key) for notice_id in normalized_notices),
                )
                row = self._connection.execute(
                    "SELECT * FROM telegram_outbox WHERE action_key = ?", (key,)
                ).fetchone()
                assert row is not None
                entry = self._telegram_outbox_from_row(row)
                immutable_values = (
                    (entry.target_ref, target, "получателя"),
                    (entry.notice_ids, normalized_notices, "notice_ids"),
                    (entry.messages, normalized_messages, "messages"),
                )
                for actual, expected, label in immutable_values:
                    if actual != expected:
                        raise StateConflictError(
                            f"Telegram outbox action key сменил {label}"
                        )
                self._connection.commit()
                return entry
            except Exception:
                self._connection.rollback()
                raise

    def advance_telegram_outbox(
        self,
        action_key: str,
        *,
        sent_part_indexes: Sequence[int],
        sent_message_ids: Sequence[int],
        next_part_index: int,
        complete: bool,
        first_sent_at: datetime | None = None,
        deduplicated_part_indexes: Sequence[int] = (),
    ) -> TelegramOutboxEntry:
        """Persist one contiguous host result starting at the durable cursor.

        ``sent_message_ids`` correspond, in order, to delivered indexes not
        listed in ``deduplicated_part_indexes``.  A deduplicated part is still
        durable progress even though its original Telegram message ID is not
        recoverable from the repeated RPC.
        """

        key = _identifier(action_key, "Telegram outbox action key")
        if isinstance(next_part_index, bool) or not isinstance(next_part_index, int):
            raise TypeError("next_part_index Telegram outbox должен быть целым")
        if not isinstance(complete, bool):
            raise TypeError("complete Telegram outbox должен быть bool")

        def normalize_indexes(values: Sequence[int], label: str) -> tuple[int, ...]:
            normalized: list[int] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{label} должен содержать целые индексы")
                normalized.append(value)
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{label} содержит повторяющиеся индексы")
            return tuple(normalized)

        part_indexes = normalize_indexes(sent_part_indexes, "sent_part_indexes")
        deduplicated_indexes = normalize_indexes(
            deduplicated_part_indexes, "deduplicated_part_indexes"
        )
        message_ids: list[int] = []
        for message_id in sent_message_ids:
            if isinstance(message_id, bool) or not isinstance(message_id, int):
                raise TypeError("sent_message_ids должен содержать целые ID")
            message_ids.append(message_id)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT * FROM telegram_outbox WHERE action_key = ?", (key,)
                ).fetchone()
                if row is None:
                    raise KeyError(key)
                entry = self._telegram_outbox_from_row(row)
                if not entry.next_part_index <= next_part_index <= len(entry.messages):
                    raise ValueError("next_part_index Telegram outbox вне диапазона")
                expected_indexes = tuple(
                    range(entry.next_part_index, next_part_index)
                )
                if part_indexes != expected_indexes:
                    raise StateConflictError(
                        "Telegram outbox принимает только непрерывный прогресс от текущего cursor"
                    )
                if any(index not in expected_indexes for index in deduplicated_indexes):
                    raise ValueError(
                        "deduplicated_part_indexes должен быть подмножеством отправленных частей"
                    )
                known_indexes = tuple(
                    index
                    for index in expected_indexes
                    if index not in set(deduplicated_indexes)
                )
                if len(message_ids) != len(known_indexes):
                    raise ValueError(
                        "sent_message_ids не соответствует известным отправленным частям"
                    )
                should_be_complete = next_part_index == len(entry.messages)
                if complete != should_be_complete:
                    raise StateConflictError(
                        "Статус Telegram outbox не соответствует next_part_index"
                    )
                ids_by_index = dict(zip(known_indexes, message_ids, strict=True))
                new_parts = tuple(
                    TelegramOutboxSentPart(
                        index,
                        None if index in deduplicated_indexes else ids_by_index[index],
                    )
                    for index in expected_indexes
                )
                all_parts = (*entry.sent_parts, *new_parts)
                first_timestamp = (
                    _timestamp(first_sent_at)
                    if first_sent_at is not None and entry.first_sent_at is None
                    else (
                        row["first_sent_at"]
                        if row["first_sent_at"] is not None
                        else None
                    )
                )
                serialized_parts = tuple(
                    {
                        "part_index": part.part_index,
                        "message_id": part.message_id,
                    }
                    for part in all_parts
                )
                known_message_ids = tuple(
                    part.message_id
                    for part in all_parts
                    if part.message_id is not None
                )
                self._connection.execute(
                    """
                    UPDATE telegram_outbox
                    SET next_part_index = ?, sent_message_ids_json = ?,
                        sent_parts_json = ?, status = ?, first_sent_at = ?,
                        updated_at = ?
                    WHERE action_key = ?
                    """,
                    (
                        next_part_index,
                        _json_dump(known_message_ids),
                        _json_dump(serialized_parts),
                        "sent" if complete else "pending",
                        first_timestamp,
                        _timestamp(_now()),
                        key,
                    ),
                )
                updated = self._connection.execute(
                    "SELECT * FROM telegram_outbox WHERE action_key = ?", (key,)
                ).fetchone()
                assert updated is not None
                result = self._telegram_outbox_from_row(updated)
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def record_telegram_turn_metric(self, metric: TelegramTurnMetric) -> None:
        if not isinstance(metric, TelegramTurnMetric):
            raise TypeError("metric должен быть TelegramTurnMetric")
        values = (
            metric.context_ms,
            metric.provider_queue_ms,
            metric.model_ms,
            metric.send_ms,
            metric.generation_to_first_send_ms,
        )
        if any(value < 0 for value in values):
            raise ValueError("Telegram latency metrics не могут быть отрицательными")
        if metric.sla_eligible is not None and not isinstance(
            metric.sla_eligible, bool
        ):
            raise TypeError("sla_eligible должен быть boolean или null")
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO telegram_turn_metrics (
                    turn_id, chat_id, outcome, context_ms, provider_queue_ms,
                    model_ms, send_ms, generation_to_first_send_ms, model_rounds,
                    context_messages, context_characters, started_at, first_sent_at,
                    sla_eligible
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metric.turn_id,
                    metric.chat_id,
                    metric.outcome,
                    *values,
                    metric.model_rounds,
                    metric.context_messages,
                    metric.context_characters,
                    _timestamp(metric.started_at),
                    _timestamp(metric.first_sent_at) if metric.first_sent_at else None,
                    (
                        int(metric.sla_eligible)
                        if metric.sla_eligible is not None
                        else None
                    ),
                ),
            )
            # Keep the additive metrics table bounded while retaining a much
            # larger history than the default status window.
            self._connection.execute(
                """
                DELETE FROM telegram_turn_metrics
                WHERE turn_id NOT IN (
                    SELECT turn_id FROM telegram_turn_metrics
                    ORDER BY started_at DESC LIMIT 10000
                )
                """
            )
            self._connection.commit()

    def telegram_latency_summary(
        self, *, limit: int = 500, target_seconds: float = 10.0
    ) -> Mapping[str, Any]:
        limit = _integer(limit, "Окно Telegram latency", 1, 10_000)
        target_ms = max(0.0, float(target_seconds) * 1_000.0)
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT context_ms, provider_queue_ms, model_ms, send_ms,
                       generation_to_first_send_ms, model_rounds, outcome,
                       first_sent_at, sla_eligible
                FROM telegram_turn_metrics
                ORDER BY started_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        def legacy_sla_eligibility(row: sqlite3.Row) -> bool | None:
            marker = row["sla_eligible"]
            if marker is not None:
                return bool(marker)
            outcome = str(row["outcome"])
            if outcome == "sent":
                # Before the marker existed, only ordinary successful text
                # turns used this exact outcome.
                return True
            if outcome == "resumed" or outcome.startswith("sent:"):
                return False
            # Old generic errors cannot safely be classified as text or media.
            # Keep the latency distribution intact, but make the live SLO
            # explicitly unevaluable instead of optimistically green.
            return None

        eligible_rows = [
            row for row in rows if legacy_sla_eligibility(row) is True
        ]
        unknown_eligibility = sum(
            1 for row in rows if legacy_sla_eligibility(row) is None
        )
        # ``sent`` plus a Telegram ID is the deliberately narrow measured SLA
        # sample.  Eligible errors and no-first-send rows are censored delivery
        # attempts and are surfaced separately below.
        measured_rows = [
            row
            for row in eligible_rows
            if row["first_sent_at"] is not None and str(row["outcome"]) == "sent"
        ]
        durations = sorted(
            float(row["generation_to_first_send_ms"]) for row in measured_rows
        )

        def percentile(values: Sequence[float], percent: int) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            index = max(0, ((len(ordered) * percent + 99) // 100) - 1)
            return round(ordered[min(index, len(ordered) - 1)], 3)

        def distribution(values: Sequence[float]) -> Mapping[str, float | None]:
            return {
                "average": (
                    round(sum(values) / len(values), 3) if values else None
                ),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "p99": percentile(values, 99),
            }

        phase_names = (
            "context_ms",
            "provider_queue_ms",
            "model_ms",
            "send_ms",
        )
        phases = {
            name: distribution([float(row[name]) for row in measured_rows])
            for name in phase_names
        }
        model_calls = [int(row["model_rounds"]) for row in measured_rows]

        latency_distribution = distribution(durations)
        breaches = sum(1 for value in durations if value > target_ms)
        breach_rate = breaches / len(durations) if durations else None
        ordinary_turns = len(eligible_rows)
        delivered_turns = len(measured_rows)
        error_turns = sum(
            1
            for row in eligible_rows
            if str(row["outcome"]).startswith("error:")
        )
        censored_turns = ordinary_turns - delivered_turns
        failed_turns = censored_turns
        no_first_send_turns = sum(
            1
            for row in eligible_rows
            if str(row["outcome"]) == "sent" and row["first_sent_at"] is None
        )
        delivery_rate = (
            delivered_turns / ordinary_turns if ordinary_turns else None
        )
        completeness_rate = delivery_rate
        complete_window = (
            unknown_eligibility == 0
            and ordinary_turns > 0
            and censored_turns == 0
        )

        return {
            "sample_size": len(measured_rows),
            "turn_count": len(rows),
            "ordinary_text_turns": ordinary_turns,
            "delivered_turns": delivered_turns,
            "failed_turns": failed_turns,
            "error_turns": error_turns,
            "no_first_send_turns": no_first_send_turns,
            "censored_turns": censored_turns,
            "delivery_rate": delivery_rate,
            "complete_samples": delivered_turns,
            "missing_samples": censored_turns,
            "completeness_rate": completeness_rate,
            "eligibility_unknown": unknown_eligibility,
            "delivery": {
                "attempts": ordinary_turns,
                "delivered": delivered_turns,
                "failed": failed_turns,
                "errors": error_turns,
                "no_first_send": no_first_send_turns,
                "censored": censored_turns,
                "rate": delivery_rate,
            },
            "completeness": {
                "expected_samples": ordinary_turns,
                "measured_samples": delivered_turns,
                "missing_samples": censored_turns,
                "unknown_eligibility": unknown_eligibility,
                "rate": completeness_rate,
                "complete": complete_window,
            },
            "target_ms": target_ms,
            "p50_ms": latency_distribution["p50"],
            "p95_ms": latency_distribution["p95"],
            "p99_ms": latency_distribution["p99"],
            "generation_to_first_send_ms": latency_distribution,
            "breaches": breaches,
            "breach_rate": breach_rate,
            # Explicit names make the web/status API self-describing while
            # retaining the original breach_* keys for compatibility.
            "exceedances": breaches,
            "exceed_rate": breach_rate,
            "average_model_rounds": (
                sum(model_calls) / len(model_calls) if model_calls else 0.0
            ),
            "llm_calls": {
                "total": sum(model_calls),
                **distribution([float(value) for value in model_calls]),
            },
            "phases": phases,
            "outcomes": {
                outcome: sum(1 for row in rows if row["outcome"] == outcome)
                for outcome in sorted({str(row["outcome"]) for row in rows})
            },
        }

__all__ = ["TelegramStateStoreMixin"]
