"""Per-turn staging and trusted executors used by :mod:`milana_service`.

Read-only skill operations may contact a host while the model is thinking.
Every write is represented as a :class:`StagedAction` and remains inert until
the service validates the state's revision and commits the complete turn.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Mapping, Protocol

from milana_pulse import validate_scheduled_message
from milana_schedule import WeeklyRoutine

from .session import MAX_WAKEUP_DELAY_SECONDS, SkillSession
from .types import ToolCall, ToolResult


class SkillHostGateway(Protocol):
    """Small interface shared by the real supervisor and test fakes."""

    async def request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float | None = None,
        idempotency_key: str | None = None,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class StagedAction:
    kind: str
    payload: Mapping[str, Any]
    idempotency_key: str


@dataclass(slots=True)
class StagedTurn:
    turn_id: str
    trigger: Any
    expected_revision: int
    actions: list[StagedAction] = field(default_factory=list)
    target_tokens: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    default_target_token: str | None = None
    sticker_references: dict[str, Mapping[str, Any]] = field(default_factory=dict)

    def action_key(self, suffix: str) -> str:
        """Return a retry-stable key for one logical action.

        Regenerating an unread Telegram notice gets a fresh turn ID, but it is
        still the same input.  Basing the key on its immutable notice IDs lets
        the host suppress a duplicate whose first response was lost.
        """

        metadata = getattr(self.trigger, "metadata", {})
        raw_ids = metadata.get("notice_ids", ()) if isinstance(metadata, Mapping) else ()
        notice_ids = (
            tuple(item for item in raw_ids if isinstance(item, str) and item)
            if isinstance(raw_ids, (list, tuple))
            else ()
        )
        if notice_ids:
            digest = hashlib.sha256("\x1f".join(notice_ids).encode("utf-8")).hexdigest()[:24]
            scope = f"notices:{digest}"
        else:
            scope = f"turn:{self.turn_id}"
        clean_suffix = str(suffix).strip().replace(" ", "-")
        if not clean_suffix:
            raise ValueError("Action idempotency suffix cannot be empty")
        return f"{scope}:{clean_suffix}"

    def add_action(self, kind: str, payload: Mapping[str, Any]) -> StagedAction:
        action = StagedAction(
            kind=kind,
            payload=dict(payload),
            idempotency_key=self.action_key(f"{len(self.actions)}:{kind}"),
        )
        self.actions.append(action)
        return action

    def register_target(self, token: str, target: Mapping[str, Any]) -> None:
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Telegram host returned an empty target_token")
        clean = token.strip()
        self.target_tokens[clean] = dict(target)
        if self.default_target_token is None:
            self.default_target_token = clean

    def require_target(self, token: str | None = None) -> tuple[str, Mapping[str, Any]]:
        chosen = token or self.default_target_token
        if chosen is None or chosen not in self.target_tokens:
            raise PermissionError(
                "Telegram target_token was not issued during the current turn"
            )
        return chosen, self.target_tokens[chosen]


class TurnStagingArea:
    """Own all ephemeral write intents and channel capabilities for active turns."""

    def __init__(self) -> None:
        self._turns: dict[str, StagedTurn] = {}

    def begin(self, trigger: Any) -> StagedTurn:
        turn_id = getattr(trigger, "id", None)
        revision = getattr(trigger, "revision", None)
        if not isinstance(turn_id, str) or not turn_id:
            raise TypeError("trigger must expose a non-empty string id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise TypeError("trigger must expose a non-negative integer revision")
        if turn_id in self._turns:
            raise RuntimeError(f"Turn {turn_id!r} is already active")
        staged = StagedTurn(turn_id, trigger, revision)
        self._turns[turn_id] = staged
        return staged

    def get(self, turn_id: str | None) -> StagedTurn:
        if not isinstance(turn_id, str) or turn_id not in self._turns:
            raise RuntimeError("No staging context exists for this model turn")
        return self._turns[turn_id]

    def finish(self, turn_id: str) -> StagedTurn:
        try:
            return self._turns.pop(turn_id)
        except KeyError as exc:
            raise RuntimeError(f"Turn {turn_id!r} is not active") from exc

    def discard(self, turn_id: str) -> None:
        self._turns.pop(turn_id, None)


def _aware_now() -> datetime:
    return datetime.now(timezone.utc)


def _schedule_snapshot(routine: WeeklyRoutine, now: datetime) -> dict[str, Any]:
    state = routine.state_at(now)

    def activity(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "title": value.title,
            "kind": value.kind,
            "start_minute": value.start,
            "end_minute": value.end,
        }

    return {
        "now": state.now.isoformat(),
        "day": state.day_key,
        "current": activity(state.current),
        "next": activity(state.next_activity),
        "next_transition_at": state.next_at.isoformat() if state.next_at else None,
        "energy": state.metrics.energy,
        "stress": state.metrics.stress,
        "productivity": state.metrics.productivity,
    }


class CoreSkillExecutor:
    """Stage diary/wakeup writes and answer schedule inspection locally."""

    def __init__(
        self,
        staging: TurnStagingArea,
        routine: WeeklyRoutine,
        *,
        now: Any = _aware_now,
    ) -> None:
        self.staging = staging
        self.routine = routine
        self._now = now

    async def execute(self, call: ToolCall, *, session: SkillSession) -> ToolResult:
        stage = self.staging.get(session.turn_id)
        arguments = call.parse_arguments()
        if call.name == "inspect_schedule":
            if arguments:
                raise ValueError("inspect_schedule does not accept arguments")
            return ToolResult.success(call, _schedule_snapshot(self.routine, self._now()))
        if call.name == "write_diary":
            if set(arguments) != {"entry"} or not isinstance(arguments["entry"], str):
                raise ValueError("write_diary requires one string entry")
            entry = arguments["entry"].strip()
            if not entry:
                raise ValueError("Diary entry cannot be empty")
            action = stage.add_action("write_diary", {"entry": entry})
            return ToolResult.success(call, {"staged": True, "key": action.idempotency_key})
        if call.name == "schedule_wakeup":
            if set(arguments) != {"delay_seconds", "reason"}:
                raise ValueError("schedule_wakeup requires delay_seconds and reason")
            delay = arguments["delay_seconds"]
            reason = arguments["reason"]
            if (
                isinstance(delay, bool)
                or not isinstance(delay, int)
                or not 1 <= delay <= MAX_WAKEUP_DELAY_SECONDS
            ):
                raise ValueError("schedule_wakeup delay is outside the 30 day horizon")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("schedule_wakeup reason cannot be empty")
            due_at = self._now() + timedelta(seconds=delay)
            action = stage.add_action(
                "schedule_wakeup",
                {"due_at": due_at.isoformat(), "reason": reason.strip()},
            )
            return ToolResult.success(
                call,
                {
                    "staged": True,
                    "due_at": due_at.isoformat(),
                    "key": action.idempotency_key,
                },
            )
        raise LookupError(f"Unsupported core tool: {call.name}")


class TelegramSkillExecutor:
    """Materialize Telegram lazily and stage delayed sends."""

    def __init__(
        self,
        staging: TurnStagingArea,
        gateway: SkillHostGateway,
        *,
        context_enricher: Callable[
            [StagedTurn, Mapping[str, Any]],
            Mapping[str, Any] | Awaitable[Mapping[str, Any]],
        ]
        | None = None,
    ) -> None:
        self.staging = staging
        self.gateway = gateway
        self.context_enricher = context_enricher

    async def activate(self, _spec: Any, session: SkillSession) -> Mapping[str, Any]:
        stage = self.staging.get(session.turn_id)
        trigger = stage.trigger
        payload = {
            "turn_id": stage.turn_id,
            "trigger": trigger.model_payload(),
        }
        metadata = getattr(trigger, "metadata", {})
        notice_ids = metadata.get("notice_ids", []) if isinstance(metadata, Mapping) else []
        if notice_ids:
            payload["notice_ids"] = list(notice_ids)
        target_ref = (
            metadata.get("_telegram_target_ref")
            if isinstance(metadata, Mapping)
            else None
        )
        if target_ref is not None:
            payload["target_ref"] = target_ref
        raw_context = await self.gateway.request("telegram.open", payload, timeout=20.0)
        if not isinstance(raw_context, Mapping):
            raise TypeError("telegram.open must return an object")
        context = dict(raw_context)
        self._register_targets(stage, context)
        if self.context_enricher is not None:
            extra = self.context_enricher(stage, context)
            if inspect.isawaitable(extra):
                extra = await extra
            if not isinstance(extra, Mapping):
                raise TypeError("Telegram context enricher must return an object")
            for key, value in extra.items():
                if key in context:
                    raise ValueError(f"Telegram context enricher repeated field {key!r}")
                context[str(key)] = value
        return context

    @staticmethod
    def _register_targets(stage: StagedTurn, context: Mapping[str, Any]) -> None:
        token = context.get("target_token")
        if isinstance(token, str):
            stage.register_target(token, context)
        targets = context.get("targets")
        if targets is not None:
            if not isinstance(targets, list):
                raise TypeError("telegram.open targets must be an array")
            for target in targets:
                if not isinstance(target, Mapping) or not isinstance(
                    target.get("target_token"), str
                ):
                    raise TypeError("Each Telegram target must contain target_token")
                stage.register_target(str(target["target_token"]), target)

    async def execute(self, call: ToolCall, *, session: SkillSession) -> ToolResult:
        stage = self.staging.get(session.turn_id)
        if call.name != "schedule_message":
            raise LookupError(f"Unsupported Telegram tool: {call.name}")
        args = call.parse_arguments()
        if set(args) != {"delay_seconds", "message"}:
            raise ValueError("schedule_message requires delay_seconds and message")
        message = validate_scheduled_message(args["delay_seconds"], args["message"])
        token, target = stage.require_target()
        action = stage.add_action(
            "schedule_message",
            {
                "target_token": token,
                "target": dict(target),
                "delay_seconds": message.delay_seconds,
                "message": message.message,
            },
        )
        return ToolResult.success(call, {"staged": True, "key": action.idempotency_key})


class StickerSkillExecutor:
    """Keep sticker discovery read-only; stage every selected sticker action."""

    def __init__(self, staging: TurnStagingArea, gateway: SkillHostGateway) -> None:
        self.staging = staging
        self.gateway = gateway

    async def execute(self, call: ToolCall, *, session: SkillSession) -> ToolResult:
        stage = self.staging.get(session.turn_id)
        args = call.parse_arguments()
        if call.name == "open_sticker_picker":
            if set(args) != {"pack_id"} or not (
                args["pack_id"] is None or isinstance(args["pack_id"], str)
            ):
                raise ValueError("open_sticker_picker requires nullable pack_id")
            token, _ = stage.require_target()
            result = await self.gateway.request(
                "telegram.execute",
                {
                    "turn_id": stage.turn_id,
                    "target_token": token,
                    "action": "open_sticker_picker",
                    "arguments": {"pack_id": args["pack_id"]},
                },
                timeout=30.0,
            )
            if not isinstance(result, Mapping):
                raise TypeError("Sticker picker must return an object")
            stickers = self._picker_stickers(result)
            for reference in stickers:
                if isinstance(reference, Mapping) and isinstance(
                    reference.get("sticker_id"), str
                ):
                    stage.sticker_references[str(reference["sticker_id"])] = dict(reference)
            return ToolResult.success(call, dict(result))

        if call.name not in {"send_sticker", "schedule_sticker"}:
            raise LookupError(f"Unsupported sticker tool: {call.name}")
        expected = {"sticker_id"} | (
            {"delay_seconds"} if call.name == "schedule_sticker" else set()
        )
        if set(args) != expected or not isinstance(args.get("sticker_id"), str):
            raise ValueError(f"Invalid arguments for {call.name}")
        sticker_id = args["sticker_id"]
        reference = stage.sticker_references.get(sticker_id)
        if reference is None:
            raise PermissionError(
                "Sticker was not exposed by open_sticker_picker in this turn"
            )
        token, target = stage.require_target()
        payload: dict[str, Any] = {
            "target_token": token,
            "target": dict(target),
            "sticker_id": sticker_id,
        }
        if call.name == "schedule_sticker":
            delay = args["delay_seconds"]
            if isinstance(delay, bool) or not isinstance(delay, int) or delay <= 0:
                raise ValueError("schedule_sticker delay_seconds must be positive")
            payload["delay_seconds"] = delay
            resolved = await self.gateway.request(
                "telegram.execute",
                {
                    "turn_id": stage.turn_id,
                    "target_token": token,
                    "action": "schedule_sticker",
                    "arguments": {
                        "sticker_id": sticker_id,
                        "delay_seconds": delay,
                    },
                },
                timeout=20.0,
            )
            if not isinstance(resolved, Mapping) or not isinstance(
                resolved.get("sticker"), Mapping
            ):
                raise TypeError("Telegram host did not resolve scheduled sticker")
            payload["sticker"] = dict(resolved["sticker"])
        action = stage.add_action(call.name, payload)
        return ToolResult.success(call, {"staged": True, "key": action.idempotency_key})

    @staticmethod
    def _picker_stickers(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        direct = result.get("stickers")
        if isinstance(direct, list):
            return [item for item in direct if isinstance(item, Mapping)]
        content = result.get("content")
        if not isinstance(content, list):
            return []
        import json

        for item in content:
            if not isinstance(item, Mapping) or item.get("type") != "input_text":
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping) and isinstance(payload.get("stickers"), list):
                return [
                    value for value in payload["stickers"] if isinstance(value, Mapping)
                ]
        return []


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


__all__ = [
    "CoreSkillExecutor",
    "SkillHostGateway",
    "StagedAction",
    "StagedTurn",
    "StickerSkillExecutor",
    "TelegramSkillExecutor",
    "TurnStagingArea",
]
