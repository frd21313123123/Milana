"""The single user-facing process that owns Milana's life and skills.

``MilanaService`` is the only SQLite owner and the only model loop.  Telegram
runs as a supervised child skill-host and sees no prompt, state database or
model credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import random
import sys
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI

from agy_provider import AgyModelClient
from milana import (
    MilanaAgent,
    TurnResult,
    TurnTrigger,
    bind_telegram_skill_tree,
    load_default_registry,
)
from milana.host_supervisor import SkillHostSupervisor
from milana.runtime import (
    CoreSkillExecutor,
    StagedAction,
    StagedTurn,
    StickerSkillExecutor,
    TelegramSkillExecutor,
    TurnStagingArea,
)
from milana_heartbeat import (
    HeartbeatReason,
    HeartbeatTrigger,
    MilanaHeartbeat,
)
from milana_ipc import (
    JsonRpcServer,
    MediaPathValidator,
    RequestContext,
    load_or_create_auth_token,
)
from milana_memory import MilanaMemoryStore, PulseTask
from milana_pulse import DelayedActionDispatcher
from milana_schedule import WeeklyRoutine, load_routine
from milana_state import (
    FactSeed,
    GoalChange,
    HeartbeatChanges,
    MilanaStateStore,
    NewEntity,
    NewLifeEvent,
    RelationshipDelta,
    StateConflictError,
)
from telegram_client import (
    GEMINI_LLM_CHOICE,
    MEMORY_PATH,
    AIConfig,
    GeminiQuotaFallbackClient,
    load_ai_config,
)


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "data" / "runtime"
TOKEN_FILE = RUNTIME_DIR / "telegram-host.token"
HOST_MEDIA_DIR = RUNTIME_DIR / "telegram-media"
DEFAULT_WEB_PORT = 8765
PID_FILE = BASE_DIR / "bot.pid"
MODE_FILE = BASE_DIR / "bot.mode"
MAX_TELEGRAM_NOTICES_PER_TURN = 100
_WORKER_STOP = object()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _parse_datetime(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be an ISO datetime string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _patch_rows(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw = payload.get(key, [])
    if not isinstance(raw, list) or len(raw) > 3:
        raise ValueError(f"{key} must be an array with at most three entries")
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"arguments_json"}:
            raise ValueError(f"Each {key} entry must contain arguments_json")
        encoded = item["arguments_json"]
        if not isinstance(encoded, str):
            raise TypeError(f"{key}.arguments_json must be a string")
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {key} arguments_json") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"{key} patch must decode to an object")
        result.append(decoded)
    return result


def _facts(raw: Any) -> tuple[FactSeed, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or len(raw) > 20:
        raise ValueError("entity facts must be an array with at most 20 entries")
    result: list[FactSeed] = []
    for item in raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
            raise ValueError("Each entity fact needs a string key")
        # Only the trusted seed path may create immutable facts.
        result.append(
            FactSeed(
                key=item["key"],
                value=item.get("value"),
                locked=False,
                source="milana",
            )
        )
    return tuple(result)


def build_heartbeat_changes(
    payload: Mapping[str, Any],
    current_state: Any,
) -> HeartbeatChanges:
    """Validate the provider-neutral final payload into one atomic reducer input."""

    state_update = payload.get("state_update")
    if not isinstance(state_update, Mapping):
        raise ValueError("state_update must be an object")
    need_deltas: dict[str, int] = {}
    for name in ("social", "rest", "novelty", "achievement"):
        target = state_update.get(name)
        if target is None:
            continue
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target <= 100:
            raise ValueError(f"state_update.{name} must be 0..100 or null")
        delta = target - int(getattr(current_state, name))
        if not -15 <= delta <= 15:
            raise ValueError(f"state_update.{name} changes a need by more than 15")
        need_deltas[name] = delta

    entities: list[NewEntity] = []
    for item in _patch_rows(payload, "entity_updates"):
        entities.append(
            NewEntity(
                kind=item.get("kind", "person"),
                name=item.get("name", ""),
                description=item.get("description", ""),
                is_real=item.get("is_real", False),
                entity_id=item.get("entity_id"),
                facts=_facts(item.get("facts")),
            )
        )

    events: list[NewLifeEvent] = []
    for item in _patch_rows(payload, "life_events"):
        entity_ids = item.get("entity_ids", [])
        if not isinstance(entity_ids, list) or not all(
            isinstance(value, str) for value in entity_ids
        ):
            raise ValueError("life event entity_ids must be an array of strings")
        happened_at = item.get("happened_at")
        events.append(
            NewLifeEvent(
                title=item.get("title", ""),
                description=item.get("description", ""),
                kind=item.get("kind", "life"),
                importance=item.get("importance", 50),
                entity_ids=tuple(entity_ids),
                happened_at=(
                    _parse_datetime(happened_at, field_name="happened_at")
                    if happened_at is not None
                    else None
                ),
                raw_payload=item,
            )
        )

    goals = tuple(
        GoalChange(
            operation=item.get("operation", "create"),
            goal_id=item.get("goal_id"),
            title=item.get("title"),
            description=item.get("description", ""),
            horizon=item.get("horizon", "short"),
            progress=item.get("progress"),
        )
        for item in _patch_rows(payload, "goal_updates")
    )

    relationships: list[RelationshipDelta] = []
    for item in _patch_rows(payload, "relationship_updates"):
        interacted_at = item.get("interacted_at")
        relationships.append(
            RelationshipDelta(
                entity_id=item.get("entity_id", ""),
                closeness=item.get("closeness", 0),
                reciprocity=item.get("reciprocity", 0),
                tension=item.get("tension", 0),
                awaiting_reply=item.get("awaiting_reply"),
                blocked=item.get("blocked"),
                interacted_at=(
                    _parse_datetime(interacted_at, field_name="interacted_at")
                    if interacted_at is not None
                    else None
                ),
            )
        )

    mood = state_update.get("mood_label")
    intention = state_update.get("current_intention")
    return HeartbeatChanges(
        entities=tuple(entities),
        events=tuple(events),
        goals=goals,
        need_deltas=need_deltas,
        relationships=tuple(relationships),
        mood=mood if isinstance(mood, str) and mood.strip() else None,
        valence=state_update.get("valence"),
        arousal=state_update.get("arousal"),
        current_intention=(
            intention if isinstance(intention, str) and intention.strip() else None
        ),
    )


def _target_ref(value: Any) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Telegram target is missing from the activated skill context")
    if isinstance(value, str):
        clean = value.strip()
        if clean.lstrip("-").isdigit():
            return int(clean)
        if not clean:
            raise ValueError("Telegram target cannot be empty")
        return clean
    return value


class MilanaService:
    """Own the model, memory, world, schedule, heartbeat and skill registry."""

    def __init__(
        self,
        *,
        config: AIConfig,
        model_client: Any,
        memory: MilanaMemoryStore,
        state: MilanaStateStore,
        routine: WeeklyRoutine,
        rpc_server: JsonRpcServer,
        supervisor: SkillHostSupervisor,
        dev_mode: bool = False,
        now: Any = _now,
    ) -> None:
        self.config = config
        self.model_client = model_client
        self.memory = memory
        self.state = state
        self.routine = routine
        self.rpc_server = rpc_server
        self.supervisor = supervisor
        self.dev_mode = bool(dev_mode)
        self._now = now
        self.staging = TurnStagingArea()
        self.core_executor = CoreSkillExecutor(self.staging, routine, now=now)
        self.telegram_executor = TelegramSkillExecutor(
            self.staging,
            supervisor,
            context_enricher=self._telegram_memory_context,
        )
        self.sticker_executor = StickerSkillExecutor(self.staging, supervisor)
        HOST_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        self.media_paths = MediaPathValidator(HOST_MEDIA_DIR)
        registry = load_default_registry()
        bind_telegram_skill_tree(
            registry,
            telegram_executor=self.telegram_executor,
            sticker_executor=self.sticker_executor,
            telegram_on_activate=self.telegram_executor.activate,
        )
        self.registry = registry
        self.agent = MilanaAgent(
            model_client,
            model=config.model,
            persona=config.instructions,
            registry=registry,
            core_executor=self.core_executor,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            max_reply_messages=config.message_flow.max_reply_messages,
            state_context=self._state_context,
            tool_result_content=self._tool_result_media,
        )
        self.heartbeat = MilanaHeartbeat(
            state,
            self._on_heartbeat,
            now=now,
            is_sleeping=self._is_sleeping,
            next_awake_at=self._next_awake_at,
            next_transition_at=self._next_transition_at,
            recovery_context=self._recovery_context,
            dev_mode=dev_mode,
        )
        self.delayed_dispatcher = DelayedActionDispatcher(
            memory,
            self._deliver_delayed_action,
            now=now,
        )
        self._turn_queue: asyncio.Queue[TurnTrigger] = asyncio.Queue()
        self._notice_buffers: dict[str, list[dict[str, Any]]] = {}
        self._notice_first_at: dict[str, float] = {}
        self._notice_tasks: dict[str, asyncio.Task[None]] = {}
        self._seen_notices: set[str] = set()
        self._night_thresholds: dict[str, int] = {}
        self._worker_queues: dict[str, asyncio.Queue[TurnTrigger]] = {}
        self._worker_tasks: dict[str, asyncio.Task[None]] = {}
        self._active_turn_tasks: dict[str, asyncio.Task[TurnResult]] = {}
        self._active_triggers: dict[str, TurnTrigger] = {}
        self._turn_phases: dict[str, str] = {}
        self._tasks: list[asyncio.Task[Any]] = []
        self._stopping = False
        self._web_panel: Any | None = None
        self.last_turn_error: str | None = None
        self.last_turn_at: datetime | None = None
        self._random = random.SystemRandom()
        self._presence_lock = asyncio.Lock()
        self._presence_online = False
        self._attention_until: datetime | None = None

    @classmethod
    async def create_default(
        cls,
        *,
        dev_mode: bool = False,
    ) -> "MilanaService":
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        HOST_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        token = load_or_create_auth_token(TOKEN_FILE)
        rpc_server = JsonRpcServer(token, port=0, request_timeout=30.0)
        await rpc_server.start()
        supervisor = SkillHostSupervisor(
            rpc_server,
            token_file=TOKEN_FILE,
            runtime_dir=HOST_MEDIA_DIR,
            dev_mode=dev_mode,
        )
        config = load_ai_config()
        if config.provider == GEMINI_LLM_CHOICE:
            gemini = AgyModelClient(model=config.model)
            model_client: Any = (
                GeminiQuotaFallbackClient(
                    gemini,
                    AsyncOpenAI(api_key=config.api_key),
                    openai_model=config.openai_fallback_model,
                )
                if config.api_key
                else gemini
            )
        else:
            model_client = AsyncOpenAI(api_key=config.api_key)
        memory = MilanaMemoryStore(MEMORY_PATH)
        state = MilanaStateStore(MEMORY_PATH)
        service = cls(
            config=config,
            model_client=model_client,
            memory=memory,
            state=state,
            routine=load_routine(),
            rpc_server=rpc_server,
            supervisor=supervisor,
            dev_mode=dev_mode,
        )
        rpc_server.register_method("telegram.notice", service._rpc_telegram_notice)
        return service

    def _seed_world(self) -> None:
        if self.state.get_entity("milana") is None:
            self.state.create_entity(
                "person",
                "милана",
                description="сама милана",
                is_real=True,
                entity_id="milana",
                facts=(
                    FactSeed("name", "милана", True, "persona"),
                    FactSeed("age", 21, True, "persona"),
                    FactSeed("city", "пермь", True, "persona"),
                ),
            )
        else:
            self.state.seed_locked_facts(
                "milana", {"name": "милана", "age": 21, "city": "пермь"}
            )

    async def start(self, *, web_port: int | None = DEFAULT_WEB_PORT) -> None:
        self._seed_world()
        await self.supervisor.start()
        self._tasks = [
            asyncio.create_task(self._queue_loop(), name="milana-turn-queue"),
            asyncio.create_task(self.heartbeat.run(), name="milana-heartbeat"),
            asyncio.create_task(
                self.delayed_dispatcher.run(), name="milana-delayed-actions"
            ),
        ]
        if not self.dev_mode:
            self._tasks.append(
                asyncio.create_task(self._presence_loop(), name="milana-presence")
            )
        if web_port is not None:
            self._start_web_panel(web_port)

    async def run_forever(self, *, web_port: int | None = DEFAULT_WEB_PORT) -> None:
        await self.start(web_port=web_port)
        try:
            await asyncio.Event().wait()
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        for task in tuple(self._notice_tasks.values()):
            task.cancel()
        for task in self._active_turn_tasks.values():
            task.cancel()
        # A worker may have consumed an earlier cancellation while unwinding a
        # superseded model turn.  Explicitly wake every per-chat queue as well
        # so Windows' Proactor loop cannot leave it parked in Queue.get().
        for queue in self._worker_queues.values():
            queue.put_nowait(_WORKER_STOP)
        for task in self._worker_tasks.values():
            task.cancel()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(
            *self._notice_tasks.values(),
            *self._tasks,
            *self._worker_tasks.values(),
            *self._active_turn_tasks.values(),
            return_exceptions=True,
        )
        if self._web_panel is not None:
            self._web_panel.stop()
            self._web_panel = None
        if not self.dev_mode:
            try:
                await self._set_presence(False)
            except Exception:
                pass
        await self.supervisor.stop()
        await self.rpc_server.close()
        self.state.touch_service(self._now())
        self.state.close()
        self.memory.close()

    async def _rpc_telegram_notice(
        self, params: Any, _request: RequestContext
    ) -> Mapping[str, Any]:
        if not isinstance(params, Mapping):
            raise TypeError("telegram.notice params must be an object")
        allowed = {
            "source",
            "notice_id",
            "chat_id",
            "message_id",
            "timestamp",
            "sender",
            "media_type",
        }
        if set(params) - allowed:
            raise PermissionError("Pre-activation Telegram notice contains full content")
        required = allowed
        if not required.issubset(params):
            raise ValueError("Telegram notice is missing metadata fields")
        notice_id = params["notice_id"]
        chat_id = params["chat_id"]
        if not isinstance(notice_id, str) or isinstance(chat_id, bool) or not isinstance(
            chat_id, (str, int)
        ):
            raise ValueError("Telegram notice IDs are invalid")
        if notice_id in self._seen_notices:
            return {"accepted": True, "duplicate": True}
        self._seen_notices.add(notice_id)
        chat_key = str(chat_id)
        self._notice_buffers.setdefault(chat_key, []).append(dict(params))
        self._merge_queued_notices(chat_key)
        loop = asyncio.get_running_loop()
        self._notice_first_at.setdefault(chat_key, loop.time())
        previous = self._notice_tasks.get(chat_key)
        if previous is not None:
            previous.cancel()
        active_task = self._active_turn_tasks.get(chat_key)
        if (
            active_task is not None
            and not active_task.done()
            and self._turn_phases.get(chat_key) == "generation"
        ):
            active = self._active_triggers.get(chat_key)
            if active is not None:
                prior = active.metadata.get("notices", [])
                if isinstance(prior, list):
                    known = {
                        item.get("notice_id")
                        for item in self._notice_buffers.get(chat_key, [])
                        if isinstance(item, Mapping)
                    }
                    self._notice_buffers.setdefault(chat_key, [])[:0] = [
                        dict(item)
                        for item in prior
                        if isinstance(item, Mapping)
                        and item.get("notice_id") not in known
                    ]
            active_task.cancel()
        self._notice_tasks[chat_key] = asyncio.create_task(
            self._flush_notices(chat_key), name=f"telegram-quiet:{chat_key}"
        )
        return {"accepted": True, "duplicate": False}

    def _merge_queued_notices(self, chat_key: str) -> None:
        """Fold not-yet-started chat notices back into the quiet buffer."""

        queue = self._worker_queues.get(chat_key)
        if queue is None or queue.empty():
            return
        pending_notices: list[dict[str, Any]] = []
        retained: list[TurnTrigger] = []
        while True:
            try:
                trigger = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            queue.task_done()
            raw = trigger.metadata.get("notices", ())
            if trigger.kind == "telegram_notice" and isinstance(raw, (list, tuple)):
                pending_notices.extend(
                    dict(item) for item in raw if isinstance(item, Mapping)
                )
            else:
                retained.append(trigger)
        for trigger in retained:
            queue.put_nowait(trigger)
        if not pending_notices:
            return
        current = self._notice_buffers.setdefault(chat_key, [])
        known = {
            item.get("notice_id")
            for item in current
            if isinstance(item, Mapping)
        }
        prefix = [
            item for item in pending_notices if item.get("notice_id") not in known
        ]
        current[:0] = prefix

    async def _flush_notices(self, chat_key: str) -> None:
        flow = self.config.message_flow
        loop = asyncio.get_running_loop()
        elapsed = loop.time() - self._notice_first_at[chat_key]
        quiet = 0.0 if self.dev_mode else flow.input_quiet_seconds
        remaining_max = max(0.0, flow.input_max_wait_seconds - elapsed)
        try:
            await asyncio.sleep(min(quiet, remaining_max))
            notices = list(self._notice_buffers.get(chat_key, []))
            if not notices:
                return
            if not self.dev_mode:
                received = self._now()
                active_conversation = self._chat_recently_active(
                    notices[-1]["chat_id"], received
                )
                night_wake = False
                if self._is_sleeping(received) and not active_conversation:
                    threshold = self._night_thresholds.setdefault(
                        chat_key, self._random.randint(3, 8)
                    )
                    night_wake = len(notices) >= threshold
                if not night_wake and not active_conversation:
                    plan = self.routine.plan_response(
                        received,
                        last_attentive_at=self.memory.get_last_attentive_at(),
                    )
                    delay = max(0.0, (plan.respond_at - received).total_seconds())
                    if delay:
                        await asyncio.sleep(delay)
            notices = self._notice_buffers.pop(chat_key, [])
            self._notice_first_at.pop(chat_key, None)
            self._notice_tasks.pop(chat_key, None)
            if not notices:
                return
            self._night_thresholds.pop(chat_key, None)
            # telegram.open intentionally caps one materialization at 100
            # notices so its JSON/media frame cannot balloon past the IPC
            # limit.  Backfill can be larger, therefore preserve chronological
            # order and route it as consecutive per-chat turns.
            for offset in range(0, len(notices), MAX_TELEGRAM_NOTICES_PER_TURN):
                batch = notices[offset : offset + MAX_TELEGRAM_NOTICES_PER_TURN]
                state = self.state.get_agent_state()
                await self._turn_queue.put(
                    TurnTrigger(
                        kind="telegram_notice",
                        occurred_at=self._now(),
                        source_skill="telegram",
                        revision=state.revision,
                        metadata={
                            "chat_id": batch[-1]["chat_id"],
                            "notice_ids": [item["notice_id"] for item in batch],
                            "notices": batch,
                        },
                    )
                )
        except asyncio.CancelledError:
            raise

    async def _queue_loop(self) -> None:
        while True:
            trigger = await self._turn_queue.get()
            key = self._turn_key(trigger)
            queue = self._worker_queues.setdefault(key, asyncio.Queue())
            await queue.put(trigger)
            worker = self._worker_tasks.get(key)
            if worker is None or worker.done():
                self._worker_tasks[key] = asyncio.create_task(
                    self._worker_loop(key, queue), name=f"milana-worker:{key}"
                )
            self._turn_queue.task_done()

    @staticmethod
    def _turn_key(trigger: TurnTrigger) -> str:
        chat = trigger.metadata.get("chat_id")
        return str(chat) if chat is not None else "__life__"

    async def _worker_loop(
        self, key: str, queue: asyncio.Queue[Any]
    ) -> None:
        while True:
            trigger = await queue.get()
            if trigger is _WORKER_STOP or self._stopping:
                queue.task_done()
                return
            completion = trigger.metadata.get("_completion_future")
            task = asyncio.create_task(
                self._execute_turn(trigger), name=f"milana-turn:{trigger.id}"
            )
            self._active_turn_tasks[key] = task
            self._active_triggers[key] = trigger
            self._turn_phases[key] = "generation"
            try:
                result = await task
                if isinstance(completion, asyncio.Future) and not completion.done():
                    completion.set_result(result)
            except asyncio.CancelledError:
                if self._stopping:
                    raise
                # A newer notice reinserted the cancelled input into its buffer.
            except StateConflictError:
                await queue.put(
                    TurnTrigger(
                        kind=trigger.kind,
                        occurred_at=self._now(),
                        source_skill=trigger.source_skill,
                        revision=self.state.get_agent_state().revision,
                        metadata=trigger.metadata,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - keep other workers alive
                self.last_turn_error = f"{type(exc).__name__}: {exc}"
                print(
                    f"Ход Миланы завершился ошибкой: {self.last_turn_error}",
                    file=sys.stderr,
                )
                if isinstance(completion, asyncio.Future) and not completion.done():
                    completion.set_exception(exc)
            finally:
                if self._active_turn_tasks.get(key) is task:
                    self._active_turn_tasks.pop(key, None)
                    self._active_triggers.pop(key, None)
                    self._turn_phases.pop(key, None)
                queue.task_done()

    async def _execute_turn(self, trigger: TurnTrigger) -> TurnResult:
        self.staging.begin(trigger)
        active_skills: tuple[str, ...] = ()
        try:
            result = await self.agent.run_turn(trigger)
            active_skills = result.active_skills
            stage = self.staging.finish(trigger.id)
            turn_key = self._turn_key(trigger)
            if self._active_triggers.get(turn_key) is trigger:
                self._turn_phases[turn_key] = "commit"
            result = await self._commit_turn(result, stage)
            self.last_turn_error = None
            self.last_turn_at = self._now()
            self._record_skill_audit(result)
            return result
        except BaseException as exc:
            self.staging.discard(trigger.id)
            # Successful handling ends with acknowledge, so an unread notice
            # may safely stay in the in-memory duplicate cache.  A failed turn
            # must be eligible for host backfill after reconnect.  Cancellation
            # by a newer notice is different: _rpc_telegram_notice already put
            # the cancelled input back into that chat's quiet buffer.
            if (
                trigger.kind == "telegram_notice"
                and not isinstance(exc, (asyncio.CancelledError, StateConflictError))
            ):
                raw_notice_ids = trigger.metadata.get("notice_ids", ())
                if isinstance(raw_notice_ids, (list, tuple)):
                    for notice_id in raw_notice_ids:
                        if isinstance(notice_id, str):
                            self._seen_notices.discard(notice_id)
            raise
        finally:
            if "telegram" in active_skills or trigger.kind == "telegram_notice":
                try:
                    await self.supervisor.request(
                        "telegram.cleanup_turn",
                        {"turn_id": trigger.id},
                        timeout=5.0,
                    )
                except Exception:
                    pass

    def _record_skill_audit(self, result: TurnResult) -> None:
        activated: set[str] = set()
        for tool_result in result.tool_results:
            if tool_result.name != "open_skill":
                continue
            skill_id = "unknown"
            if isinstance(tool_result.arguments_json, str):
                try:
                    arguments = json.loads(tool_result.arguments_json)
                except json.JSONDecodeError:
                    arguments = {}
                candidate = (
                    arguments.get("skill_id")
                    if isinstance(arguments, Mapping)
                    else None
                )
                if isinstance(candidate, str) and candidate.strip():
                    skill_id = candidate.strip()
            action = "activate"
            if tool_result.ok and skill_id in activated:
                action = "activate_idempotent"
            elif not tool_result.ok:
                action = "activation_denied"
            detail: dict[str, Any] = {}
            if tool_result.error:
                detail["error"] = tool_result.error
            self.state.record_skill_activation(
                result.turn_id,
                skill_id,
                action,
                success=tool_result.ok,
                detail=detail,
                at=self._now(),
            )
            if tool_result.ok:
                activated.add(skill_id)

    async def _commit_turn(self, result: TurnResult, stage: StagedTurn) -> TurnResult:
        current = self.state.get_agent_state()
        changes = build_heartbeat_changes(result.payload, current)
        self.state.apply_heartbeat_changes(
            changes,
            expected_revision=stage.expected_revision,
            at=self._now(),
            record_heartbeat=result.trigger.kind != "telegram_notice",
        )
        if result.trigger.kind != "telegram_notice":
            self._maybe_create_weekly_summary(self._now())
        outbound_sent = False
        for action in stage.actions:
            outbound_sent = (
                await self._commit_staged_action(stage, action) or outbound_sent
            )
        telegram = result.payload.get("telegram")
        if telegram is not None:
            outbound_sent = (
                await self._commit_telegram_final(stage, telegram) or outbound_sent
            )
        elif (
            result.trigger.kind == "telegram_notice"
            and stage.default_target_token is not None
        ):
            # A sticker-only response, a scheduled promise, or a deliberate
            # no-reply still constitutes handling the opened incoming message.
            # Store its context and acknowledge it only after every staged
            # action has committed successfully.
            outbound_sent = (
                await self._commit_telegram_final(
                    stage,
                    {
                        "target_token": stage.default_target_token,
                        "messages": [],
                        "reaction": None,
                        "blacklist_sender": False,
                    },
                )
                or outbound_sent
            )
        if (
            outbound_sent
            and result.trigger.kind != "telegram_notice"
            and stage.default_target_token is not None
        ):
            _, target = stage.require_target(stage.default_target_token)
            target_ref = _target_ref(target.get("target_ref"))
            relationship_id = f"telegram:{target_ref}"
            relationship = self.state.get_relationship(relationship_id)
            if relationship is not None and not relationship.awaiting_reply:
                try:
                    self.state.mark_initiative(relationship_id, at=self._now())
                except ValueError:
                    pass
        if isinstance(result, TurnResult):
            return replace(
                result,
                validated_changes=changes,
                staged_actions=tuple(stage.actions),
            )
        return result

    async def _commit_staged_action(
        self, stage: StagedTurn, action: StagedAction
    ) -> bool:
        payload = action.payload
        if action.kind == "write_diary":
            self.memory.add_diary_entry(payload["entry"])
            return False
        if action.kind == "schedule_wakeup":
            self.heartbeat.schedule_wakeup(
                _parse_datetime(payload["due_at"], field_name="due_at"),
                payload={"reason": payload["reason"]},
                idempotency_key=action.idempotency_key,
            )
            return False
        if action.kind == "schedule_message":
            target = _target_ref(payload["target"].get("target_ref"))
            self.memory.schedule_pulse_message(
                target,
                payload["message"],
                due_at=self._now() + timedelta(seconds=payload["delay_seconds"]),
                source_message_id=self._latest_message_id(payload["target"]),
            )
            self.delayed_dispatcher.wake()
            return False
        if action.kind == "schedule_sticker":
            target = _target_ref(payload["target"].get("target_ref"))
            sticker = payload["sticker"]
            self.memory.schedule_pulse_sticker(
                target,
                due_at=self._now() + timedelta(seconds=payload["delay_seconds"]),
                set_id=sticker["set_id"],
                set_access_hash=sticker["set_access_hash"],
                set_short_name=sticker["set_short_name"],
                document_id=sticker["document_id"],
                pack_title=sticker["pack_title"],
                emoji=sticker["emoji"],
                source_message_id=self._latest_message_id(payload["target"]),
            )
            self.delayed_dispatcher.wake()
            return False
        if action.kind == "send_sticker":
            outcome = await self._host_action(
                stage,
                payload["target_token"],
                "send_sticker",
                {"sticker_id": payload["sticker_id"]},
                action.idempotency_key,
            )
            if outcome.get("status") != "sent":
                raise RuntimeError(
                    str(outcome.get("error") or "Telegram did not send the sticker")
                )
            return True
        raise ValueError(f"Unknown staged action: {action.kind}")

    async def _commit_telegram_final(
        self, stage: StagedTurn, telegram: Any
    ) -> bool:
        if not isinstance(telegram, Mapping):
            raise ValueError("telegram final branch must be an object or null")
        token = telegram.get("target_token")
        messages = telegram.get("messages", [])
        reaction = telegram.get("reaction")
        blacklist = telegram.get("blacklist_sender", False)
        if not isinstance(messages, list) or not all(
            isinstance(message, str) and message.strip() for message in messages
        ):
            raise ValueError("Telegram messages must be non-empty strings")
        token, target = stage.require_target(token)
        target_ref = _target_ref(target.get("target_ref"))
        self._store_incoming_context(target_ref, target)
        relationship_id = self._record_incoming_relationship(target_ref, target)
        sent_count = 0
        if messages:
            outcome = await self._host_action(
                stage,
                token,
                "send_messages",
                {
                    "messages": messages,
                    "inter_message_min_delay_seconds": (
                        0
                        if self.dev_mode
                        else self.config.message_flow.inter_message_min_delay_seconds
                    ),
                    "inter_message_max_delay_seconds": (
                        0
                        if self.dev_mode
                        else self.config.message_flow.inter_message_max_delay_seconds
                    ),
                },
                stage.action_key("final:messages"),
            )
            sent_ids = (
                outcome.get("sent_message_ids", [])
                if isinstance(outcome, Mapping)
                else []
            )
            sent_count = len(sent_ids) if isinstance(sent_ids, list) else 0
            status = outcome.get("status") if isinstance(outcome, Mapping) else None
            if status == "failed" or (status == "partial" and sent_count == 0):
                raise RuntimeError(
                    str(outcome.get("error") or "Telegram did not send the reply")
                )
            if (
                sent_count == 0
                and isinstance(outcome, Mapping)
                and outcome.get("status") == "sent"
            ):
                sent_count = len(messages)
            for index, message in enumerate(messages[:sent_count]):
                message_id = sent_ids[index] if index < len(sent_ids) else None
                self.memory.add_message(
                    target_ref,
                    "assistant",
                    message,
                    telegram_message_id=(
                        message_id if isinstance(message_id, int) else None
                    ),
                    sender_name="Милана",
                )
        latest_id = self._latest_message_id(target)
        if reaction is not None and latest_id is not None:
            await self._host_action(
                stage,
                token,
                "reaction",
                {"message_id": latest_id, "reaction": reaction},
                stage.action_key("final:reaction"),
            )
        sender_id = self._latest_sender_id(target)
        if blacklist and sender_id is not None:
            await self._host_action(
                stage,
                token,
                "blacklist_sender",
                {"sender_id": sender_id},
                stage.action_key("final:blacklist"),
            )
            if relationship_id is not None:
                self.state.adjust_relationship(
                    relationship_id, blocked=True, awaiting_reply=False, at=self._now()
                )
        message_ids = self._message_ids(target)
        if message_ids:
            await self._host_action(
                stage,
                token,
                "acknowledge",
                {"message_ids": message_ids},
                stage.action_key("final:acknowledge"),
            )
        self.memory.set_last_attentive_at(self._now())
        return sent_count > 0

    async def _host_action(
        self,
        stage: StagedTurn,
        token: str,
        action: str,
        arguments: Mapping[str, Any],
        key: str,
    ) -> Mapping[str, Any]:
        if action in {"send_messages", "send_sticker", "send_sticker_reference"}:
            await self._show_online()
        result = await self.supervisor.request(
            "telegram.execute",
            {
                "turn_id": stage.turn_id,
                "target_token": token,
                "action": action,
                "arguments": dict(arguments),
            },
            timeout=30.0,
            idempotency_key=key,
        )
        if not isinstance(result, Mapping):
            raise TypeError("Telegram host action must return an object")
        return result

    async def _set_presence(self, online: bool) -> None:
        if self.dev_mode:
            return
        async with self._presence_lock:
            if self._presence_online == online:
                return
            await self.supervisor.request(
                "telegram.presence", {"online": online}, timeout=10.0
            )
            self._presence_online = online

    async def _show_online(self) -> None:
        if self.dev_mode:
            return
        behavior = self.routine.online_behavior
        seconds = self._random.randint(
            behavior.post_reply_online_min_seconds,
            behavior.post_reply_online_max_seconds,
        )
        self._attention_until = max(
            self._attention_until or self._now(),
            self._now() + timedelta(seconds=seconds),
        )
        await self._set_presence(True)

    async def _presence_loop(self) -> None:
        behavior = self.routine.online_behavior
        next_spontaneous = self._now() + timedelta(
            seconds=self._random.randint(
                behavior.spontaneous_online_interval_min_seconds,
                behavior.spontaneous_online_interval_max_seconds,
            )
        )
        spontaneous_until: datetime | None = None
        while True:
            now = self._now()
            should_be_online = False
            if not self._is_sleeping(now):
                if self._attention_until is not None and now < self._attention_until:
                    should_be_online = True
                elif spontaneous_until is not None and now < spontaneous_until:
                    should_be_online = True
                elif now >= next_spontaneous:
                    spontaneous_until = now + timedelta(
                        seconds=self._random.randint(
                            behavior.spontaneous_online_duration_min_seconds,
                            behavior.spontaneous_online_duration_max_seconds,
                        )
                    )
                    should_be_online = True
            else:
                spontaneous_until = None
                self._attention_until = None
            if spontaneous_until is not None and now >= spontaneous_until:
                spontaneous_until = None
                next_spontaneous = now + timedelta(
                    seconds=self._random.randint(
                        behavior.spontaneous_online_interval_min_seconds,
                        behavior.spontaneous_online_interval_max_seconds,
                    )
                )
            try:
                await self._set_presence(should_be_online)
            except Exception as exc:  # host supervisor handles recovery
                self.last_turn_error = f"presence: {type(exc).__name__}: {exc}"
                self._presence_online = False
            await asyncio.sleep(5.0)

    async def _deliver_delayed_action(self, task: PulseTask) -> None:
        turn_id = f"delayed:{task.id}"
        target = _target_ref(task.chat_id)
        opened = await self.supervisor.request(
            "telegram.open",
            {"turn_id": turn_id, "notice_ids": [], "target_ref": target},
            timeout=20.0,
        )
        if not isinstance(opened, Mapping) or not isinstance(
            opened.get("target_token"), str
        ):
            raise RuntimeError("Telegram host did not issue a delayed-action token")
        try:
            await self._show_online()
            if task.action == "send_message":
                action = "send_messages"
                arguments = {"messages": [task.message]}
            elif task.action == "send_sticker":
                action = "send_sticker_reference"
                arguments = {
                    "sticker": {
                        "set_id": task.sticker_set_id,
                        "set_access_hash": task.sticker_set_access_hash,
                        "set_short_name": task.sticker_set_short_name,
                        "document_id": task.sticker_document_id,
                        "pack_title": task.sticker_pack_title,
                        "emoji": task.sticker_emoji,
                    }
                }
            else:
                raise ValueError(f"Unsupported delayed action: {task.action}")
            outcome = await self.supervisor.request(
                "telegram.execute",
                {
                    "turn_id": turn_id,
                    "target_token": opened["target_token"],
                    "action": action,
                    "arguments": arguments,
                },
                timeout=30.0,
                idempotency_key=f"delayed:{task.id}",
            )
            if not isinstance(outcome, Mapping):
                raise TypeError("Telegram delayed action must return an object")
            if outcome.get("status") != "sent":
                raise RuntimeError(
                    str(
                        outcome.get("error")
                        or f"Telegram delayed action was not sent: {outcome.get('status')}"
                    )
                )
            self.heartbeat.notify_delayed_result(
                {"task_id": task.id, "action": task.action, "status": "sent"},
                idempotency_key=f"delayed-result:{task.id}",
            )
        finally:
            try:
                await self.supervisor.request(
                    "telegram.cleanup_turn", {"turn_id": turn_id}, timeout=5.0
                )
            except Exception:
                pass

    async def _on_heartbeat(self, trigger: HeartbeatTrigger) -> None:
        kind = {
            HeartbeatReason.SCHEDULE_TRANSITION: "schedule_transition",
            HeartbeatReason.RECOVERY: "recovery",
            HeartbeatReason.MANUAL_WAKE: "manual_wake",
        }.get(trigger.reason, "heartbeat")
        metadata = dict(trigger.payload)
        initiative = (
            None
            if trigger.reason == HeartbeatReason.RECOVERY
            else self._initiative_target()
        )
        if initiative is not None:
            metadata["_telegram_target_ref"] = initiative
        completion = asyncio.get_running_loop().create_future()
        metadata["_completion_future"] = completion
        await self._turn_queue.put(
            TurnTrigger(
                kind=kind,
                occurred_at=trigger.fired_at,
                revision=self.state.get_agent_state().revision,
                metadata=metadata,
            )
        )
        await completion

    def _initiative_target(self) -> str | int | None:
        now = self._now()
        if self._is_sleeping(now):
            return None
        for relationship in self.state.list_relationships(limit=100):
            if not self.state.can_initiate(relationship.entity_id, now=now):
                continue
            entity_id = relationship.entity_id
            if entity_id.startswith("telegram:"):
                return _target_ref(entity_id.split(":", 1)[1])
        return None

    async def _state_context(self, trigger: TurnTrigger) -> Mapping[str, Any]:
        world = self.state.load_world_context()
        return {
            "world": _json_ready(world),
            "schedule": self._schedule_context(trigger.occurred_at),
            "diary": self.memory.diary_instructions(limit=12),
            "turn_policy": {
                "max_new_entities": 3,
                "max_new_events": 3,
                "max_goal_changes": 3,
                "max_relationship_changes": 3,
                "need_change_limit": 15,
                "relationship_change_limit": 10,
                "max_active_goals": 20,
                "one_initiative_contact": True,
            },
            "state_patch_formats": {
                "entity_updates": {
                    "encoding": "arguments_json",
                    "fields": [
                        "entity_id (stable ID; omit only when creating)",
                        "kind",
                        "name",
                        "description",
                        "is_real",
                        "facts: [{key, value}]",
                    ],
                    "existing_entity": (
                        "When entity_id already exists, only supplied unlocked facts are "
                        "versioned; locked persona/world facts cannot change."
                    ),
                },
                "life_events": {
                    "encoding": "arguments_json",
                    "fields": [
                        "title",
                        "description",
                        "kind",
                        "importance 0..100",
                        "entity_ids",
                        "happened_at ISO datetime or omit",
                    ],
                },
                "goal_updates": {
                    "encoding": "arguments_json",
                    "operations": ["create", "update", "complete", "archive"],
                    "fields": [
                        "operation",
                        "goal_id",
                        "title",
                        "description",
                        "horizon short|long",
                        "progress 0..100",
                    ],
                },
                "relationship_updates": {
                    "encoding": "arguments_json",
                    "fields": [
                        "entity_id",
                        "closeness delta -10..10",
                        "reciprocity delta -10..10",
                        "tension delta -10..10",
                        "awaiting_reply",
                        "blocked",
                        "interacted_at ISO datetime",
                    ],
                },
            },
        }

    def _telegram_memory_context(
        self,
        _stage: StagedTurn,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Reveal durable chat memory only after Telegram is activated."""

        target = context.get("target_ref")
        if isinstance(target, bool) or not isinstance(target, (str, int)):
            return {}
        current_ids = self._message_ids(context)
        history = self.memory.summary_context(
            target,
            exclude_user_message_ids=current_ids,
        )
        if not history:
            return {}
        return {"durable_memory": history}

    def _tool_result_media(self, result: Any) -> Sequence[Mapping[str, Any]]:
        found: list[tuple[str, str | None]] = []

        def visit(value: Any, inherited_mime: str | None = None) -> None:
            if isinstance(value, Mapping):
                mime = value.get("media_mime_type", inherited_mime)
                if not isinstance(mime, str):
                    mime = inherited_mime
                for key, nested in value.items():
                    if key in {"media_path", "path"} and isinstance(nested, str):
                        found.append((nested, mime))
                    else:
                        visit(nested, mime)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, inherited_mime)

        visit(getattr(result, "output", None))
        content: list[Mapping[str, Any]] = []
        seen: set[Path] = set()
        for raw_path, declared_mime in found:
            path = self.media_paths.validate(raw_path)
            if path in seen:
                continue
            seen.add(path)
            mime = declared_mime or mimetypes.guess_type(path.name)[0]
            if not isinstance(mime, str):
                continue
            family = mime.split("/", 1)[0].lower()
            if family not in {"image", "video", "audio"}:
                continue
            if family != "image" and self.config.provider != GEMINI_LLM_CHOICE:
                continue
            data = path.read_bytes()
            if not data or len(data) > 20 * 1024 * 1024:
                continue
            encoded = base64.b64encode(data).decode("ascii")
            url = f"data:{mime};base64,{encoded}"
            if family == "image":
                content.append(
                    {"type": "input_image", "image_url": url, "detail": "original"}
                )
            elif family == "video":
                content.append({"type": "input_video", "video_url": url})
            else:
                content.append({"type": "input_audio", "audio_url": url})
        return content

    def _schedule_context(self, at: datetime) -> Mapping[str, Any]:
        value = self.routine.state_at(at)
        return {
            "now": value.now.isoformat(),
            "current": (
                {"title": value.current.title, "kind": value.current.kind}
                if value.current
                else None
            ),
            "next": (
                {"title": value.next_activity.title, "kind": value.next_activity.kind}
                if value.next_activity
                else None
            ),
            "next_at": value.next_at.isoformat() if value.next_at else None,
            "energy": value.metrics.energy,
            "stress": value.metrics.stress,
            "productivity": value.metrics.productivity,
        }

    def _is_sleeping(self, at: datetime) -> bool:
        current = self.routine.state_at(at).current
        return current is not None and current.kind == "sleep"

    def _next_transition_at(self, at: datetime) -> datetime | None:
        return self.routine.state_at(at).next_at

    def _next_awake_at(self, at: datetime) -> datetime | None:
        cursor = at
        for _ in range(32):
            schedule = self.routine.state_at(cursor)
            if schedule.current is None or schedule.current.kind != "sleep":
                return cursor
            if schedule.next_at is None:
                return None
            cursor = schedule.next_at + timedelta(seconds=1)
        return None

    async def _recovery_context(self, window: Any) -> Mapping[str, Any]:
        cursor = window.started_at
        missed: list[dict[str, Any]] = []
        while cursor < window.ended_at and len(missed) < 64:
            schedule = self.routine.state_at(cursor)
            if schedule.current is not None:
                item = {
                    "title": schedule.current.title,
                    "kind": schedule.current.kind,
                    "at": cursor.isoformat(),
                }
                if not missed or missed[-1]["title"] != item["title"]:
                    missed.append(item)
            if schedule.next_at is None or schedule.next_at <= cursor:
                break
            cursor = schedule.next_at + timedelta(seconds=1)
        return {"missed_activities": missed}

    @staticmethod
    def _messages(context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        raw = context.get("messages", [])
        return [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []

    @classmethod
    def _message_ids(cls, context: Mapping[str, Any]) -> list[int]:
        return [
            item["message_id"]
            for item in cls._messages(context)
            if isinstance(item.get("message_id"), int)
        ]

    @classmethod
    def _latest_message_id(cls, context: Mapping[str, Any]) -> int | None:
        ids = cls._message_ids(context)
        return ids[-1] if ids else None

    @classmethod
    def _latest_sender_id(cls, context: Mapping[str, Any]) -> str | int | None:
        messages = cls._messages(context)
        for item in reversed(messages):
            sender = item.get("sender")
            if isinstance(sender, Mapping) and isinstance(sender.get("id"), (str, int)):
                return sender["id"]
        return None

    def _store_incoming_context(
        self, target_ref: str | int, context: Mapping[str, Any]
    ) -> None:
        for item in self._messages(context):
            text = str(item.get("text", "") or "").strip()
            media_type = item.get("media_type")
            if not text:
                text = f"[{media_type or 'media'}]"
            sender = item.get("sender")
            sender_name = (
                sender.get("display_name") if isinstance(sender, Mapping) else None
            )
            created_at = item.get("timestamp")
            self.memory.add_message(
                target_ref,
                "user",
                text,
                telegram_message_id=(
                    item.get("message_id")
                    if isinstance(item.get("message_id"), int)
                    else None
                ),
                sender_name=sender_name if isinstance(sender_name, str) else None,
                created_at=created_at if isinstance(created_at, str) else None,
            )

    def _record_incoming_relationship(
        self, target_ref: str | int, context: Mapping[str, Any]
    ) -> str | None:
        messages = self._messages(context)
        if not messages:
            return f"telegram:{target_ref}" if self.state.get_entity(
                f"telegram:{target_ref}"
            ) else None
        sender = messages[-1].get("sender")
        name = (
            sender.get("display_name")
            if isinstance(sender, Mapping)
            else None
        )
        entity_id = f"telegram:{target_ref}"
        if self.state.get_entity(entity_id) is None:
            self.state.create_entity(
                "person",
                name if isinstance(name, str) and name.strip() else str(target_ref),
                description="собеседник из telegram",
                is_real=True,
                entity_id=entity_id,
                facts=(FactSeed("telegram_target", target_ref, False, "telegram"),),
                at=self._now(),
            )
            self.state.upsert_relationship(
                entity_id, last_interaction_at=self._now(), at=self._now()
            )
        else:
            relationship = self.state.get_relationship(entity_id)
            if relationship is None:
                self.state.upsert_relationship(
                    entity_id, last_interaction_at=self._now(), at=self._now()
                )
            else:
                self.state.mark_reply_received(entity_id, at=self._now())
        return entity_id

    def _maybe_create_weekly_summary(self, at: datetime) -> None:
        current = at.astimezone(timezone.utc)
        this_week = (current - timedelta(days=current.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        period_start = this_week - timedelta(days=7)
        existing = self.state.list_world_summaries(limit=2)
        if any(
            item.period_start == period_start and item.period_end == this_week
            for item in existing
        ):
            return
        events = [
            event
            for event in self.state.list_life_events(include_archived=True, limit=500)
            if period_start <= event.happened_at < this_week
        ]
        goals = self.state.list_goals(statuses=None, limit=100)
        content = json.dumps(
            {
                "events": [
                    {
                        "title": event.title,
                        "importance": event.importance,
                        "status": event.status,
                    }
                    for event in events[:40]
                ],
                "active_goals": [goal.title for goal in goals if goal.status == "active"],
                "completed_goals": [
                    goal.title
                    for goal in goals
                    if goal.status == "completed"
                    and period_start <= goal.updated_at < this_week
                ],
                "ending_state": _json_ready(self.state.get_agent_state()),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.state.add_world_summary(period_start, this_week, content, at=current)

    def _chat_recently_active(
        self, chat_id: str | int, at: datetime
    ) -> bool:
        history = self.memory.get_chat_history(chat_id, limit=1)
        if not history or history[-1].role != "assistant":
            return False
        try:
            sent_at = _parse_datetime(
                history[-1].created_at, field_name="history.created_at"
            )
        except (TypeError, ValueError):
            return False
        return timedelta(0) <= at - sent_at <= timedelta(minutes=30)

    def _start_web_panel(self, port: int) -> None:
        try:
            from milana_web import start_web_server
        except (ImportError, AttributeError):
            return
        loop = asyncio.get_running_loop()

        def on_loop(callback: Any) -> Any:
            loop.call_soon_threadsafe(callback)
            return {"queued": True}

        def require_id(body: Mapping[str, Any], label: str) -> str:
            value = body.get("id")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Нужен ID: {label}")
            return value.strip()

        def update_state(body: Mapping[str, Any]) -> Any:
            state = self.state.get_agent_state()
            needs = body.get("needs", {})
            if needs is not None and not isinstance(needs, Mapping):
                raise TypeError("needs должен быть объектом")
            deltas: dict[str, int] = {}
            for name in ("social", "rest", "novelty", "achievement"):
                if name not in (needs or {}):
                    continue
                target = needs[name]
                if isinstance(target, bool) or not isinstance(target, int):
                    raise TypeError(f"needs.{name} должен быть целым числом")
                deltas[name] = target - state.needs[name]
            if deltas:
                state = self.state.apply_need_deltas(
                    deltas, expected_revision=state.revision
                )
            return self.state.update_agent_state(
                mood=body.get("mood"),
                valence=body.get("valence"),
                arousal=body.get("arousal"),
                current_intention=body.get("current_intention"),
                clear_intention=bool(body.get("clear_intention", False)),
                expected_revision=state.revision,
            )

        def create_goal(body: Mapping[str, Any]) -> Any:
            return self.state.create_goal(
                body.get("title"),
                description=body.get("description", ""),
                horizon=body.get("horizon", "short"),
                progress=body.get("progress", 0),
            )

        def update_goal(body: Mapping[str, Any]) -> Any:
            return self.state.update_goal(
                require_id(body, "цели"),
                title=body.get("title"),
                description=body.get("description", ""),
                horizon=body.get("horizon"),
                progress=body.get("progress"),
            )

        def create_event(body: Mapping[str, Any]) -> Any:
            entity_ids = body.get("entity_ids", [])
            if not isinstance(entity_ids, list):
                raise TypeError("entity_ids должен быть массивом")
            return self.state.add_life_event(
                body.get("title"),
                body.get("description", ""),
                kind=body.get("kind", "life"),
                importance=body.get("importance", 50),
                entity_ids=entity_ids,
            )

        def create_entity(body: Mapping[str, Any]) -> Any:
            facts = body.get("facts", {})
            if not isinstance(facts, Mapping):
                raise TypeError("facts должен быть объектом")
            return self.state.create_entity(
                body.get("kind", "person"),
                body.get("name"),
                description=body.get("description", ""),
                is_real=bool(body.get("is_real", False)),
                entity_id=body.get("id"),
                facts=tuple(
                    FactSeed(str(key), value, False, "web-panel")
                    for key, value in facts.items()
                ),
            )

        def set_entity_fact(body: Mapping[str, Any]) -> Any:
            entity_id = require_id(body, "сущности")
            key = body.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Нужен ключ факта")
            return self.state.set_fact(
                entity_id, key, body.get("value"), source="web-panel"
            )

        def update_relationship(body: Mapping[str, Any]) -> Any:
            entity_id = require_id(body, "сущности")
            current = self.state.get_relationship(entity_id)
            return self.state.upsert_relationship(
                entity_id,
                closeness=body.get(
                    "closeness", current.closeness if current else 50
                ),
                reciprocity=body.get(
                    "reciprocity", current.reciprocity if current else 50
                ),
                tension=body.get("tension", current.tension if current else 0),
                awaiting_reply=body.get(
                    "awaiting_reply", current.awaiting_reply if current else False
                ),
                blocked=body.get("blocked", current.blocked if current else False),
            )

        async def restart_telegram_host() -> None:
            await self.supervisor.stop()
            await self.supervisor.start()

        self._web_panel = start_web_server(
            port=port,
            state_store=self.state,
            callbacks={
                "pause_heartbeat": lambda: on_loop(self.heartbeat.pause),
                "resume_heartbeat": lambda: on_loop(self.heartbeat.resume),
                "wake_now": lambda: on_loop(
                    lambda: self.heartbeat.wake(HeartbeatReason.MANUAL_WAKE)
                ),
                "cancel_heartbeat_job": lambda body: self.state.cancel_heartbeat_job(
                    require_id(body, "heartbeat-задачи")
                ),
                "update_state": update_state,
                "create_goal": create_goal,
                "update_goal": update_goal,
                "complete_goal": lambda body: self.state.complete_goal(
                    require_id(body, "цели")
                ),
                "archive_goal": lambda body: self.state.archive_goal(
                    require_id(body, "цели")
                ),
                "create_event": create_event,
                "archive_event": lambda body: self.state.archive_life_event(
                    require_id(body, "события")
                ),
                "create_entity": create_entity,
                "set_entity_fact": set_entity_fact,
                "archive_entity": lambda body: self.state.archive_entity(
                    require_id(body, "сущности")
                ),
                "update_relationship": update_relationship,
                "restart_telegram_host": lambda: on_loop(
                    lambda: asyncio.create_task(restart_telegram_host())
                ),
            },
            status_provider=self.status,
        )

    def status(self) -> Mapping[str, Any]:
        state = self.state.get_agent_state()
        return {
            "service": "running",
            "dev_mode": self.dev_mode,
            "telegram_host": self.supervisor.status(),
            "skills": [item["id"] for item in self.registry.root_catalog()],
            "turn_queue_size": self._turn_queue.qsize(),
            "worker_queue_size": sum(queue.qsize() for queue in self._worker_queues.values()),
            "active_chats": [key for key in self._active_turn_tasks if key != "__life__"],
            "last_turn_at": self.last_turn_at.isoformat() if self.last_turn_at else None,
            "last_turn_error": self.last_turn_error,
            "heartbeat_paused": state.heartbeat_paused,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Отдельный сервис живой Миланы")
    parser.add_argument("--dev-chat", action="store_true")
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT)
    return parser


async def _main_async(args: argparse.Namespace) -> None:
    pid = os.getpid()
    PID_FILE.write_text(str(pid), encoding="ascii")
    MODE_FILE.write_text(
        f"{pid} {'DEV' if args.dev_chat else 'NORMAL'}", encoding="ascii"
    )
    try:
        service = await MilanaService.create_default(dev_mode=args.dev_chat)
        print(
            "MilanaService запущен: Telegram является дочерним навыком, "
            f"режим={'DEV' if args.dev_chat else 'обычный'}, модель={service.config.model}."
        )
        await service.run_forever(web_port=None if args.no_web else args.web_port)
    finally:
        try:
            if PID_FILE.read_text(encoding="ascii").strip() == str(pid):
                PID_FILE.unlink(missing_ok=True)
                MODE_FILE.unlink(missing_ok=True)
        except OSError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI reports startup diagnostics
        print(f"MilanaService error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MilanaService",
    "build_heartbeat_changes",
    "build_parser",
    "main",
]
