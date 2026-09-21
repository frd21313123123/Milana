"""Durable intentions, using the StateStore transaction and Telegram outbox.

An intention contains meaning, never a prewritten message. Turn plans are a
delivery journal: only a validated execution turn may put text in that journal.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from uuid import uuid4


TRIGGERS = {"datetime", "delay", "schedule_event", "activity_end", "opportunity"}
TERMINAL = {"completed", "cancelled", "expired"}
RULES = """
future_actions — массив {"arguments_json":"JSON операции"}, по умолчанию [].
Создавай намерение только когда действительно собираешься сделать что-то позже,
а не на каждое упоминание будущего. Каждое обещание написать/напомнить/рассказать
позже подкрепляй намерением is_promise=true. Храни смысл в intent, не готовый текст.
Создание: operation=create, action_type=telegram_message, intent, is_promise,
trigger_type=datetime|delay|schedule_event|activity_end|opportunity,
due_at (ISO с timezone для datetime), delay_seconds (для delay), earliest_at,
expires_at, context, priority (0..100). target_token — только выданный Telegram.
schedule_event означает конец текущего блока расписания; activity_end — конец
текущей активности сцены. opportunity: ближайший свободный момент, можно задать
earliest_at и conditions={not_sleeping:true,not_busy:true}.
Изменение существующего намерения: operation=reschedule, id, due_at; отмена:
operation=cancel, id. Для текущего future_action: complete только вместе с
реальным Telegram сообщением; postpone с due_at; cancel если потеряло смысл.
Если ничего не делать сейчас, верни [] — система повторит проверку позже.
Учитывай overdue_seconds: просроченное обещание требует естественного признания
задержки, если это уместно. Не создавай копию выполняемого намерения.
"""


def timestamp(value: Any) -> str:
    try:
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Expected ISO datetime with timezone") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Datetime must include timezone")
    return dt.astimezone(timezone.utc).isoformat()


def validate_operations(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) > 8:
        raise ValueError("future_actions must be an array of at most 8 operations")
    result = []
    allowed = {"operation", "id", "action_type", "trigger_type", "intent", "context",
               "is_promise", "priority", "target_token", "due_at", "earliest_at",
               "expires_at", "delay_seconds", "conditions"}
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"arguments_json"}:
            raise ValueError("future_actions items must contain arguments_json")
        try:
            op = json.loads(item["arguments_json"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid future action JSON") from exc
        if not isinstance(op, dict) or set(op) - allowed:
            raise ValueError("Unknown future action fields")
        kind = op.get("operation", "create")
        if kind not in {"create", "cancel", "reschedule", "postpone", "complete"}:
            raise ValueError("Unknown future action operation")
        op["operation"] = kind
        for key in ("id", "intent", "context", "target_token"):
            if key in op and (not isinstance(op[key], str) or not op[key].strip() or len(op[key]) > 2000):
                raise ValueError(f"Invalid future action {key}")
        for key in ("due_at", "earliest_at", "expires_at"):
            if key in op:
                op[key] = timestamp(op[key])
        if kind == "create":
            if not op.get("intent") or op.get("trigger_type") not in TRIGGERS:
                raise ValueError("Intention and supported trigger are required")
            if op.get("action_type", "telegram_message") != "telegram_message":
                raise ValueError("Unsupported action type")
            if not isinstance(op.get("is_promise", False), bool):
                raise ValueError("is_promise must be boolean")
            priority = op.get("priority", 50 if op.get("is_promise") else 30)
            if isinstance(priority, bool) or not isinstance(priority, int) or not 0 <= priority <= 100:
                raise ValueError("priority must be 0..100")
            if op["trigger_type"] == "datetime" and "due_at" not in op:
                raise ValueError("datetime requires due_at")
            if op["trigger_type"] == "delay":
                delay = op.get("delay_seconds")
                if isinstance(delay, bool) or not isinstance(delay, int) or not 1 <= delay <= 366 * 86400:
                    raise ValueError("delay_seconds must be 1..31622400")
            conditions = op.get("conditions", {})
            if (not isinstance(conditions, dict) or set(conditions) - {"not_sleeping", "not_busy"}
                    or any(not isinstance(v, bool) for v in conditions.values())):
                raise ValueError("Invalid opportunity conditions")
        else:
            if not op.get("id"):
                raise ValueError("Action id required")
            if set(op) - {"operation", "id", "due_at"}:
                raise ValueError("Only id and due_at may be used for updates")
            if kind in {"reschedule", "postpone"} and "due_at" not in op:
                raise ValueError("Reschedule requires due_at")
        result.append(op)
    return result


class FutureActionStore:
    def __init__(self, state):
        self.state = state
        with state.transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS future_actions (
                id TEXT PRIMARY KEY, target_id TEXT NOT NULL, status TEXT NOT NULL,
                due_at TEXT NOT NULL, earliest_at TEXT NOT NULL, expires_at TEXT,
                retry_at TEXT, is_promise INTEGER NOT NULL, priority INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, last_attempt_at TEXT, completed_at TEXT,
                cancelled_at TEXT, record_json TEXT NOT NULL, result_json TEXT,
                last_error TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS future_actions_due ON future_actions(status, retry_at, due_at)")
            db.execute("CREATE INDEX IF NOT EXISTS future_actions_target ON future_actions(target_id, status)")
            db.execute("""CREATE TABLE IF NOT EXISTS future_action_turns (
                action_key TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                operations_json TEXT NOT NULL, target_id TEXT,
                status TEXT NOT NULL DEFAULT 'prepared')""")

    @staticmethod
    def _row(row):
        if row is None:
            return None
        value = dict(row)
        value.update(json.loads(value.pop("record_json")))
        value["result"] = json.loads(value.pop("result_json") or "null")
        value["is_promise"] = bool(value["is_promise"])
        return value

    def get(self, action_id):
        with self.state.transaction() as db:
            return self._row(db.execute("SELECT * FROM future_actions WHERE id=?", (action_id,)).fetchone())

    def list(self, *, status=None, target_id=None, overdue_at=None, limit=200):
        clauses, args = [], []
        if status:
            clauses.append("status=?")
            args.append(status)
        if target_id is not None:
            clauses.append("target_id=?")
            args.append(str(target_id))
        if overdue_at is not None:
            clauses.append("is_promise=1 AND status IN ('pending','executing') AND due_at<?")
            args.append(timestamp(overdue_at))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.state.transaction() as db:
            return [self._row(r) for r in db.execute(
                "SELECT * FROM future_actions" + where + " ORDER BY priority DESC, due_at, id LIMIT ?",
                (*args, limit))]

    def snapshot(self, now):
        with self.state.transaction() as db:
            count = db.execute("SELECT COUNT(*) FROM future_actions WHERE is_promise=1 AND status IN ('pending','executing') AND due_at<?", (timestamp(now),)).fetchone()[0]
        return {"actions": self.list(), "overdue": self.list(overdue_at=now), "overdue_count": count}

    def list_due(self, now, *, limit=100):
        at = timestamp(now)
        with self.state.transaction() as db:
            return [self._row(row) for row in db.execute("""SELECT * FROM future_actions
                WHERE status='pending' AND due_at<=? AND earliest_at<=?
                AND (retry_at IS NULL OR retry_at<=?) AND (expires_at IS NULL OR expires_at>?)
                ORDER BY priority DESC,due_at LIMIT ?""", (at, at, at, at, limit))]

    def create(self, spec, *, target_id, now, action_id=None, origin=None,
               activity_end=None, schedule_end=None, db=None):
        # Reuse validation for both service and administrative callers.
        spec = validate_operations([{"arguments_json": json.dumps(spec)}])[0]
        if spec["operation"] != "create":
            raise ValueError("Expected create operation")
        if isinstance(target_id, bool) or not isinstance(target_id, (str, int)) or not str(target_id).strip():
            raise ValueError("Telegram target is required")
        now_s = timestamp(now)
        due = spec.get("due_at")
        trigger = spec["trigger_type"]
        if trigger == "delay":
            due = timestamp(now + timedelta(seconds=spec["delay_seconds"]))
        elif trigger in {"activity_end", "schedule_event"}:
            end = activity_end if trigger == "activity_end" else schedule_end
            if end is None:
                raise ValueError("No current activity/schedule end")
            due = timestamp(end)
        elif trigger == "opportunity":
            due = spec.get("earliest_at", timestamp(now + timedelta(minutes=5)))
        earliest = spec.get("earliest_at", now_s)
        expires = spec.get("expires_at")
        if expires and expires <= max(due, earliest):
            raise ValueError("expires_at must follow due_at and earliest_at")
        record = {"action_type": "telegram_message", "trigger_type": trigger,
                  "intent": spec["intent"], "context": spec.get("context", ""),
                  "conditions": spec.get("conditions", {"not_sleeping": True, "not_busy": trigger == "opportunity"}),
                  "source": "telegram" if (origin or {}).get("origin_chat_id") else "autonomous",
                  **(origin or {})}
        action_id = action_id or str(uuid4())
        values = (action_id, str(target_id), due, earliest, expires,
                  int(spec.get("is_promise", False)), spec.get("priority", 50 if spec.get("is_promise") else 30),
                  now_s, json.dumps(record, ensure_ascii=False))
        def insert(conn):
            conn.execute("""INSERT OR IGNORE INTO future_actions
                (id,target_id,status,due_at,earliest_at,expires_at,is_promise,priority,created_at,record_json)
                VALUES (?,?,'pending',?,?,?,?,?,?,?)""", values)
        if db is None:
            with self.state.transaction() as conn:
                insert(conn)
        else:
            insert(db)
        return action_id

    def stage_turn(self, stage, payload, *, now, scene, schedule_end,
                   schedule_event_id=None):
        """Validate capabilities and freeze relative triggers before any effects."""
        key = stage.action_key("final:messages")
        existing = self.plan(key)
        if existing is not None:
            return
        raw = payload.get("future_actions", [])
        operations = validate_operations(raw)
        if not operations and stage.trigger.kind != "future_action":
            return
        telegram = payload.get("telegram") or {}
        bound = []
        target_id = None
        if stage.default_target_token:
            _, target = stage.require_target(telegram.get("target_token"))
            target_id = str(target["target_ref"])
        current = stage.trigger.metadata.get("future_action", {}).get("id")
        seen = set()
        for index, op in enumerate(operations):
            kind = op["operation"]
            if kind == "create":
                _, target = stage.require_target(op.get("target_token") or telegram.get("target_token"))
                recipient = str(target["target_ref"])
                if target_id is not None and recipient != target_id:
                    raise ValueError("Future action must belong to the selected Telegram target")
                target_id = recipient
                spec = dict(op)
                # Freeze delay now so retry never moves the promised date.
                due = spec.get("due_at")
                if spec["trigger_type"] == "delay":
                    due = timestamp(now + timedelta(seconds=spec["delay_seconds"]))
                elif spec["trigger_type"] == "activity_end":
                    due = timestamp(scene.expected_end)
                elif spec["trigger_type"] == "schedule_event":
                    if schedule_end is None:
                        raise ValueError("No schedule event available")
                    due = timestamp(schedule_end)
                elif spec["trigger_type"] == "opportunity":
                    due = spec.get("earliest_at", timestamp(now + timedelta(minutes=5)))
                if spec.get("expires_at") and spec["expires_at"] <= max(due, spec.get("earliest_at", timestamp(now))):
                    raise ValueError("Expiration precedes trigger")
                messages = target.get("messages", [])
                source_ids = [m.get("message_id") for m in messages if isinstance(m, dict) and isinstance(m.get("message_id"), int)]
                context = "\n".join(str(m.get("text") or m.get("content") or "")[:500] for m in messages[-3:] if isinstance(m, dict))
                if not spec.get("context") and context.strip():
                    spec["context"] = context[:1500]
                bound.append({"operation": "create", "id": f"{key}:future:{index}",
                              "spec": spec, "target_id": recipient, "created_at": timestamp(now),
                              "activity_end": timestamp(scene.expected_end),
                              "schedule_end": timestamp(schedule_end) if schedule_end else None,
                              "origin": {"origin_chat_id": recipient,
                                         "source": stage.trigger.kind,
                                         "origin_user_message_id": max(source_ids) if source_ids else None,
                                         "origin_turn_id": stage.turn_id,
                                         "origin_scene_id": scene.scene_id,
                                         "origin_activity": scene.activity_title,
                                         "planned_event_id": schedule_event_id}})
            else:
                action = self.get(op["id"])
                if op["id"] in seen:
                    raise ValueError("An action can only be changed once per turn")
                seen.add(op["id"])
                if action is None or action["status"] in TERMINAL:
                    raise ValueError("Action is not active")
                if op["id"] != current:
                    _, target = stage.require_target()
                    if str(target["target_ref"]) != action["target_id"] or action["status"] != "pending":
                        raise PermissionError("Action does not belong to this Telegram contact")
                    with self.state.transaction() as db:
                        self._require_mutable(db, op["id"])
                if kind == "complete":
                    if op["id"] != current or not telegram.get("messages") or target_id != action["target_id"]:
                        raise ValueError("Completion requires a message to the future action recipient")
                if kind in {"postpone", "reschedule"}:
                    if op["due_at"] <= timestamp(now):
                        raise ValueError("Reschedule must be in the future")
                    if action["expires_at"] and op["due_at"] >= action["expires_at"]:
                        raise ValueError("Reschedule exceeds expiration")
                bound.append(op)
        if current and telegram.get("messages") and not any(op["operation"] == "complete" and op.get("id") == current for op in bound):
            raise ValueError("A future action message requires explicit completion")
        # Store token-free meaning/validated output for replay with fresh grants.
        saved = dict(payload)
        saved["_future_staged_actions"] = [
            {"kind": action.kind, "payload": dict(action.payload), "idempotency_key": action.idempotency_key}
            for action in stage.actions
        ]
        if saved.get("telegram") is not None:
            saved["telegram"] = {**saved["telegram"], "_notice_ids": list(stage.trigger.metadata.get("notice_ids", []))}
        return (key, saved, bound, target_id)

    def reschedule_future_action(self, action_id, due_at, *, now):
        due = timestamp(due_at)
        with self.state.transaction() as db:
            self._require_mutable(db, action_id)
            row = db.execute("SELECT * FROM future_actions WHERE id=?", (action_id,)).fetchone()
            if row is None or row["status"] != "pending":
                raise ValueError("Only pending actions can be rescheduled")
            if row["expires_at"] and due >= row["expires_at"]:
                raise ValueError("New date is beyond expiration")
            db.execute("UPDATE future_actions SET due_at=?,earliest_at=?,retry_at=NULL,version=version+1 WHERE id=?",
                       (due, min(timestamp(now), due), action_id))
        return self.get(action_id)

    def cancel(self, action_id, *, now):
        with self.state.transaction() as db:
            self._require_mutable(db, action_id)
            changed = db.execute("UPDATE future_actions SET status='cancelled',cancelled_at=? WHERE id=? AND status='pending'",
                                 (timestamp(now), action_id)).rowcount
            if not changed:
                raise ValueError("Only pending actions can be cancelled")
        return self.get(action_id)

    @staticmethod
    def _require_mutable(db, action_id):
        row = db.execute("""SELECT 1 FROM future_action_turns, json_each(operations_json) AS op
            WHERE status='prepared' AND json_extract(op.value,'$.operation')='complete'
            AND json_extract(op.value,'$.id')=? LIMIT 1""", (action_id,)).fetchone()
        if row:
            raise ValueError("Delivery is already prepared; retry must reconcile its result first")

    def next_due_at(self):
        with self.state.transaction() as db:
            row = db.execute("SELECT MIN(MAX(due_at,earliest_at,COALESCE(retry_at,due_at))) FROM future_actions WHERE status='pending'").fetchone()
        return datetime.fromisoformat(row[0]) if row[0] else None

    def claim_due(self, now, *, sleeping=False, busy=False):
        at = timestamp(now)
        with self.state.transaction() as db:
            db.execute("""UPDATE future_actions SET status='expired' WHERE status='pending' AND expires_at<=?
                AND NOT EXISTS (SELECT 1 FROM future_action_turns,json_each(operations_json) AS op
                    WHERE future_action_turns.status='prepared'
                    AND json_extract(op.value,'$.operation')='complete'
                    AND json_extract(op.value,'$.id')=future_actions.id)""", (at,))
            rows = db.execute("""SELECT * FROM future_actions WHERE status='pending'
                AND due_at<=? AND earliest_at<=? AND (retry_at IS NULL OR retry_at<=?)
                ORDER BY priority DESC,due_at LIMIT 20""", (at, at, at)).fetchall()
            for row in rows:
                action = self._row(row)
                conditions = action["conditions"]
                if (sleeping and conditions.get("not_sleeping", True)) or (busy and conditions.get("not_busy", True)):
                    db.execute("UPDATE future_actions SET retry_at=? WHERE id=?", (timestamp(now + timedelta(minutes=5)), action["id"]))
                    continue
                db.execute("UPDATE future_actions SET status='executing',attempts=attempts+1,last_attempt_at=? WHERE id=?", (at, action["id"]))
                action.update(status="executing", attempts=action["attempts"] + 1, last_attempt_at=at)
                action["overdue_seconds"] = max(0, int((now - datetime.fromisoformat(action["due_at"])).total_seconds()))
                return action
        return None

    def refresh_activity_ends(self):
        """SceneEngine owns transitions, including early completion and recovery."""
        with self.state.transaction() as db:
            rows = db.execute("SELECT id,record_json FROM future_actions WHERE status='pending' AND version=0").fetchall()
            for row in rows:
                record = json.loads(row["record_json"])
                if record.get("trigger_type") == "activity_end" and record.get("origin_scene_id"):
                    scene = db.execute("SELECT ended_at FROM scenes WHERE scene_id=?", (record["origin_scene_id"],)).fetchone()
                    if scene and scene[0] is not None:
                        due = timestamp(datetime.fromtimestamp(scene[0], timezone.utc))
                        db.execute("UPDATE future_actions SET due_at=? WHERE id=?", (due, row["id"]))
                elif record.get("trigger_type") == "schedule_event" and record.get("planned_event_id"):
                    event = db.execute("SELECT actual_end,status,updated_at FROM planned_events WHERE event_id=?",
                                       (record["planned_event_id"],)).fetchone()
                    if event:
                        due_at = event[2] if event[1] == "cancelled" else event[0]
                        due = timestamp(datetime.fromtimestamp(due_at, timezone.utc))
                        db.execute("UPDATE future_actions SET due_at=? WHERE id=?", (due, row["id"]))

    def retry(self, action_id, now, error=None, *, postpone=False):
        with self.state.transaction() as db:
            row = db.execute("SELECT attempts FROM future_actions WHERE id=? AND status='executing'", (action_id,)).fetchone()
            if row:
                seconds = 300 if postpone else min(300, 5 * 2 ** min(6, row[0] - 1))
                db.execute("UPDATE future_actions SET status='pending',retry_at=?,last_error=?,version=version+? WHERE id=?",
                           (timestamp(now + timedelta(seconds=seconds)), error, int(postpone), action_id))

    def recover(self):
        # Called once by the owning service at startup, not by panel readers.
        with self.state.transaction() as db:
            db.execute("UPDATE future_actions SET status='pending' WHERE status='executing'")

    def plan(self, key):
        with self.state.transaction() as db:
            row = db.execute("SELECT * FROM future_action_turns WHERE action_key=?", (key,)).fetchone()
        if row is None:
            return None
        return {**dict(row), "payload": json.loads(row["payload_json"]), "operations": json.loads(row["operations_json"])}

    def prepare_plan(self, key, payload, operations, target_id, *, db=None):
        def prepare(conn):
            conn.execute("INSERT OR IGNORE INTO future_action_turns(action_key,payload_json,operations_json,target_id) VALUES (?,?,?,?)",
                       (key, json.dumps(payload, ensure_ascii=False), json.dumps(operations, ensure_ascii=False), str(target_id) if target_id is not None else None))
        if db is not None:
            prepare(db)
        else:
            with self.state.transaction() as conn:
                prepare(conn)

    def finish_plan(self, key, now, *, result=None, db=None):
        def finish(conn):
            row = conn.execute("SELECT * FROM future_action_turns WHERE action_key=? AND status='prepared'", (key,)).fetchone()
            if row is None:
                return
            for op in json.loads(row["operations_json"]):
                kind = op["operation"]
                if kind == "create":
                    origin = dict(op["origin"])
                    ids = (result or {}).get("telegram_message_ids", [])
                    origin["origin_milana_message_id"] = ids[-1] if ids else None
                    self.create(op["spec"], target_id=op["target_id"], now=datetime.fromisoformat(op["created_at"]),
                                action_id=op["id"], origin=origin,
                                activity_end=op.get("activity_end"), schedule_end=op.get("schedule_end"), db=conn)
                elif kind in {"reschedule", "postpone"}:
                    conn.execute("UPDATE future_actions SET status='pending',due_at=?,earliest_at=?,retry_at=NULL,version=version+1 WHERE id=? AND status IN ('pending','executing')",
                                 (op["due_at"], op["due_at"], op["id"]))
                else:
                    status = "completed" if kind == "complete" else "cancelled"
                    column = "completed_at" if kind == "complete" else "cancelled_at"
                    conn.execute(f"UPDATE future_actions SET status=?,{column}=?,result_json=? WHERE id=? AND status IN ('pending','executing')",
                                 (status, timestamp(now), json.dumps(result), op["id"]))
            conn.execute("UPDATE future_action_turns SET status='committed' WHERE action_key=?", (key,))
        if db is not None:
            finish(db)
        else:
            with self.state.transaction() as conn:
                finish(conn)
