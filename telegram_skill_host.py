"""Dependent Telegram skill host for :mod:`milana_service`.

The host owns the Telegram connection but deliberately does not own Milana's
model, memory, schedule, or durable delayed-action queue.  Before the Telegram
skill is opened it sends only small notification envelopes.  Full message
content and temporary media paths are exposed by ``telegram.open`` for one
turn, together with an unguessable target token.  Every external action must
then use that token and an IPC idempotency key.

The transport is the authenticated, length-prefixed JSON-RPC implementation in
``milana_ipc``.  The adapter boundary keeps all protocol and security behaviour
testable without importing or connecting Telethon.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import inspect
import json
import mimetypes
import os
import secrets
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence

from milana_ipc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    ConnectionClosedError,
    JsonRpcError,
    JsonRpcPeer,
    MediaPathError,
    MediaPathValidator,
    RequestContext,
    connect_json_rpc,
)


RPC_NOTICE = "telegram.notice"
RPC_OPEN = "telegram.open"
RPC_MATERIALIZE = "telegram.materialize"  # compatibility/readability alias
RPC_EXECUTE = "telegram.execute"
RPC_BACKFILL = "telegram.backfill"
RPC_CLEANUP_TURN = "telegram.cleanup_turn"
RPC_HEALTH = "telegram.health"
RPC_PRESENCE = "telegram.presence"

NOTICE_SOURCE = "telegram"
MAX_NOTICE_CACHE = 4096
MAX_BACKFILL_NOTICES = 500
DEFAULT_BACKFILL_POLL_SECONDS = 20.0
SIGNAL_TIMEOUT_SECONDS = 0.4
MAX_OUTGOING_MESSAGES = 10
MAX_MESSAGE_LENGTH = 4096
SIDE_EFFECT_ACTIONS = frozenset(
    {
        "send_messages",
        "send_media",
        "reaction",
        "blacklist_sender",
        "acknowledge",
        "send_sticker",
        "send_sticker_reference",
    }
)
LOCAL_STAGED_ACTIONS = frozenset({"schedule_message", "schedule_sticker"})
READ_ACTIONS = frozenset({"open_sticker_picker"})
SIGNAL_ACTIONS = frozenset({"typing"})
ALLOWED_ACTIONS = (
    SIDE_EFFECT_ACTIONS | LOCAL_STAGED_ACTIONS | READ_ACTIONS | SIGNAL_ACTIONS
)


def _utc_iso(value: datetime | str | None = None) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp cannot be empty")
        return text
    current = value if isinstance(value, datetime) else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).isoformat()


def _json_scalar_id(value: Any, *, field_name: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field_name} must be a string or integer")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value


@dataclass(frozen=True)
class TelegramNotice:
    """Metadata-only notification safe to send before skill activation."""

    notice_id: str
    chat_id: str | int
    message_id: int
    timestamp: str
    sender: Mapping[str, Any]
    media_type: str
    source: str = NOTICE_SOURCE

    def __post_init__(self) -> None:
        if not isinstance(self.notice_id, str) or not self.notice_id.strip():
            raise ValueError("notice_id must be a non-empty string")
        _json_scalar_id(self.chat_id, field_name="chat_id")
        if isinstance(self.message_id, bool) or not isinstance(self.message_id, int):
            raise ValueError("message_id must be an integer")
        if not isinstance(self.timestamp, str) or not self.timestamp.strip():
            raise ValueError("timestamp must be a non-empty string")
        if not isinstance(self.sender, Mapping):
            raise ValueError("sender must be an object")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ValueError("media_type must be a non-empty string")
        if self.source != NOTICE_SOURCE:
            raise ValueError("unsupported notice source")

    def to_payload(self) -> dict[str, Any]:
        # Keep this explicit: adding message text/history/files to an adapter
        # object can never accidentally expand the pre-activation envelope.
        return {
            "source": self.source,
            "notice_id": self.notice_id,
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "sender": dict(self.sender),
            "media_type": self.media_type,
        }


NoticeCallback = Callable[[TelegramNotice], Awaitable[None]]


class TelegramAdapter(Protocol):
    """Narrow boundary implemented by Telethon and test doubles."""

    async def start(self, on_notice: NoticeCallback) -> None: ...

    async def stop(self) -> None: ...

    async def backfill(self, limit: int) -> Sequence[TelegramNotice]: ...

    async def acknowledge_terminal_notice(self, notice: TelegramNotice) -> bool: ...

    async def materialize(
        self,
        notice_ids: Sequence[str],
        *,
        turn_id: str,
        turn_dir: Path,
        target_ref: str | int | None = None,
        include_history: bool = True,
    ) -> Mapping[str, Any]: ...

    async def execute_action(
        self,
        action: str,
        arguments: Mapping[str, Any],
        *,
        turn_id: str,
        turn_dir: Path,
        request: RequestContext,
    ) -> Mapping[str, Any]: ...

    async def cleanup_turn(self, turn_id: str) -> None: ...

    async def set_presence(self, online: bool) -> None: ...


@dataclass
class _TargetGrant:
    target: str | int
    message_ids: frozenset[int] = field(default_factory=frozenset)
    sender_ids: frozenset[str | int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class _SendReceipt:
    id: int | None
    deduplicated: bool = False


@dataclass(frozen=True)
class _NoticeReceipt:
    """Durable disposition returned by the Milana notice journal."""

    accepted: bool
    safe_to_ack: bool
    terminal: bool


class TelegramSkillHost:
    """JSON-RPC host enforcing turn scope, target grants and media sandboxing."""

    def __init__(
        self,
        adapter: TelegramAdapter,
        runtime_root: str | os.PathLike[str],
        *,
        max_backfill: int = MAX_BACKFILL_NOTICES,
        backfill_poll_seconds: float = DEFAULT_BACKFILL_POLL_SECONDS,
    ) -> None:
        if isinstance(max_backfill, bool) or not 0 < max_backfill <= MAX_BACKFILL_NOTICES:
            raise ValueError(f"max_backfill must be between 1 and {MAX_BACKFILL_NOTICES}")
        if (
            isinstance(backfill_poll_seconds, bool)
            or not isinstance(backfill_poll_seconds, (int, float))
            or backfill_poll_seconds <= 0
        ):
            raise ValueError("backfill_poll_seconds must be positive")
        root = Path(runtime_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.runtime_root = root.resolve(strict=True)
        self.media_validator = MediaPathValidator(self.runtime_root)
        stale_turns = self.runtime_root / "turns"
        if stale_turns.exists():
            validated_stale_turns = self.media_validator.validate(
                stale_turns, allow_directory=True
            )
            shutil.rmtree(validated_stale_turns)
        self.adapter = adapter
        self.max_backfill = max_backfill
        self.backfill_poll_seconds = float(backfill_poll_seconds)
        self.peer: JsonRpcPeer | None = None
        self._started = False
        self._backfill_lock = asyncio.Lock()
        self._backfill_task: asyncio.Task[None] | None = None
        self._backfill_refresh_task: asyncio.Task[None] | None = None
        self._turn_files: dict[str, set[Path]] = {}
        self._turn_dirs: dict[str, Path] = {}
        self._grants: dict[str, dict[str, _TargetGrant]] = {}
        self._handlers = {
            RPC_OPEN: self._handle_open,
            RPC_MATERIALIZE: self._handle_open,
            RPC_EXECUTE: self._handle_execute,
            RPC_BACKFILL: self._handle_backfill,
            RPC_CLEANUP_TURN: self._handle_cleanup_turn,
            RPC_HEALTH: self._handle_health,
            RPC_PRESENCE: self._handle_presence,
        }

    @property
    def handlers(self) -> Mapping[str, Callable[..., Awaitable[Any]]]:
        return dict(self._handlers)

    async def connect(
        self,
        host: str,
        port: int,
        token: str,
        *,
        request_timeout: float = 15.0,
    ) -> JsonRpcPeer:
        if self.peer is not None and not self.peer.closed:
            return self.peer
        peer = await connect_json_rpc(
            host,
            port,
            token,
            handlers=self._handlers,
            request_timeout=request_timeout,
            name="telegram-skill-host",
        )
        self.peer = peer
        try:
            await self.start()
        except BaseException:
            self.peer = None
            await peer.close()
            raise
        return peer

    async def start(self) -> None:
        if self._started:
            return
        if self.peer is None or self.peer.closed:
            raise ConnectionClosedError("Telegram host has no authenticated service peer")
        self._started = True
        try:
            await self.adapter.start(self.publish_notice)
            await self._publish_backfill()
        except BaseException:
            self._started = False
            raise

    async def _publish_backfill(self) -> None:
        """Publish one oldest-first unread page and retire safe terminal items.

        A request/response exchange is required when the connected peer supports
        it.  Merely notifying the service is not enough evidence to advance a
        Telegram read marker: the notice may have been deferred or may still be
        waiting for generation.  Terminal notices (already handled or declared
        poison by the durable journal) are acknowledged only through the
        adapter's gap-safe terminal acknowledgement primitive.
        """

        async with self._backfill_lock:
            terminal_prefix: dict[str, list[TelegramNotice]] = {}
            blocked_chats: set[str] = set()
            for notice in await self.adapter.backfill(self.max_backfill):
                receipt = await self.publish_notice(
                    notice, wait_for_acceptance=True
                )
                chat_key = str(notice.chat_id)
                if receipt is None:
                    # Compatibility with an old peer that only implements
                    # notifications.  Delivery still happens, but read state is
                    # deliberately left untouched until a confirmable poll.
                    blocked_chats.add(chat_key)
                    continue
                if chat_key in blocked_chats:
                    continue
                if not receipt.accepted or not receipt.safe_to_ack:
                    blocked_chats.add(chat_key)
                    continue
                if receipt.terminal:
                    # Retain every safe item in the contiguous terminal prefix.
                    # The real adapter deliberately refuses to acknowledge a
                    # newer terminal item while an older incoming item is still
                    # unread, so advancing only to the highest candidate would
                    # deadlock a prefix containing two handled/poison notices.
                    terminal_prefix.setdefault(chat_key, []).append(notice)

            for chat_key, notices in terminal_prefix.items():
                for notice in sorted(notices, key=lambda item: item.message_id):
                    acknowledged = await self._acknowledge_terminal_notice(notice)
                    if not acknowledged:
                        blocked_chats.add(chat_key)
                        break

    async def _backfill_loop(self) -> None:
        """Recover a live NewMessage event that Telethon did not deliver."""

        while self._started:
            await asyncio.sleep(self.backfill_poll_seconds)
            peer = self.peer
            if peer is None or peer.closed:
                return
            try:
                await self._publish_backfill()
            except ConnectionClosedError:
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep live updates running
                print(
                    f"Telegram backfill poll failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    async def run_until_closed(self) -> None:
        if self.peer is None:
            raise RuntimeError("Telegram host is not connected")
        if self._backfill_task is None or self._backfill_task.done():
            self._backfill_task = asyncio.create_task(
                self._backfill_loop(), name="telegram-backfill-poll"
            )
        try:
            await self.peer.wait_closed()
        finally:
            await self.stop()

    async def stop(self) -> None:
        backfill_task, self._backfill_task = self._backfill_task, None
        if backfill_task is not None and not backfill_task.done():
            backfill_task.cancel()
            await asyncio.gather(backfill_task, return_exceptions=True)
        refresh_task, self._backfill_refresh_task = (
            self._backfill_refresh_task,
            None,
        )
        if refresh_task is not None and not refresh_task.done():
            refresh_task.cancel()
            await asyncio.gather(refresh_task, return_exceptions=True)
        if self._started:
            self._started = False
            await self.adapter.stop()
        for turn_id in tuple(self._turn_dirs):
            await self.cleanup_turn(turn_id)
        peer, self.peer = self.peer, None
        if peer is not None and not peer.closed:
            await peer.close()

    async def publish_notice(
        self, notice: TelegramNotice, *, wait_for_acceptance: bool = False
    ) -> _NoticeReceipt | None:
        peer = self.peer
        if peer is None or peer.closed:
            # The adapter must not mark the message read.  Its startup backfill
            # will surface it when the service/host connection is restored.
            raise ConnectionClosedError("Milana service is unavailable")
        payload = notice.to_payload()
        idempotency_key = f"notice:{notice.notice_id}"
        request = getattr(peer, "request", None)
        if wait_for_acceptance and callable(request):
            # A disposition is mutable (pending -> handled/dead).  Reusing the
            # live-delivery idempotency key would make the RPC cache replay a
            # stale ``safe_to_ack=False`` forever.  The service notice journal
            # already supplies durable input idempotency, so probes deliberately
            # bypass the transport response cache.
            result = await request(
                RPC_NOTICE,
                payload,
            )
            if not isinstance(result, Mapping) or not isinstance(
                result.get("accepted"), bool
            ):
                raise RuntimeError(
                    "Milana service returned an invalid Telegram notice receipt"
                )
            # Older services returned only ``accepted``.  Missing safety fields
            # fail closed: the notice was delivered, but no read marker may move.
            safe_to_ack = result.get("safe_to_ack", False)
            terminal = result.get("terminal", False)
            if not isinstance(safe_to_ack, bool) or not isinstance(terminal, bool):
                raise RuntimeError(
                    "Milana service returned an invalid Telegram notice receipt"
                )
            return _NoticeReceipt(
                accepted=result["accepted"],
                safe_to_ack=safe_to_ack,
                terminal=terminal,
            )
        await peer.notify(RPC_NOTICE, payload, idempotency_key=idempotency_key)
        return None

    async def _acknowledge_terminal_notice(self, notice: TelegramNotice) -> bool:
        """Advance a poison/handled notice only through a gap-safe adapter API."""

        acknowledge = getattr(self.adapter, "acknowledge_terminal_notice", None)
        if not callable(acknowledge):
            # Old adapters remain usable, but cannot automatically retire poison
            # notices because their generic max-id acknowledgement is unsafe.
            return False
        result = acknowledge(notice)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, bool):
            raise RuntimeError(
                "Telegram terminal acknowledgement returned a non-boolean result"
            )
        return result

    def _turn_dir(self, turn_id: str) -> Path:
        existing = self._turn_dirs.get(turn_id)
        if existing is not None:
            return existing
        digest = hashlib.sha256(turn_id.encode("utf-8")).hexdigest()[:32]
        candidate = self.runtime_root / "turns" / digest
        candidate.mkdir(parents=True, exist_ok=True)
        resolved = self.media_validator.validate(candidate, allow_directory=True)
        self._turn_dirs[turn_id] = resolved
        self._turn_files.setdefault(turn_id, set())
        return resolved

    def _track_media_paths(self, turn_id: str, value: Any) -> None:
        """Recursively validate adapter-returned ``*_path`` values."""

        if isinstance(value, Mapping):
            for key, nested in value.items():
                if isinstance(key, str) and (key == "path" or key.endswith("_path")):
                    if not isinstance(nested, str):
                        raise MediaPathError(f"{key} must be a string")
                    resolved = self.media_validator.validate(nested)
                    turn_dir = self._turn_dirs.get(turn_id)
                    if turn_dir is None:
                        raise MediaPathError("No runtime directory exists for this turn")
                    try:
                        resolved.relative_to(turn_dir)
                    except ValueError as exc:
                        raise MediaPathError("Media path belongs to another turn") from exc
                    self._turn_files.setdefault(turn_id, set()).add(resolved)
                else:
                    self._track_media_paths(turn_id, nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                self._track_media_paths(turn_id, nested)

    async def cleanup_turn(self, turn_id: str) -> None:
        try:
            result = self.adapter.cleanup_turn(turn_id)
            if inspect.isawaitable(result):
                await result
        finally:
            self._grants.pop(turn_id, None)
            paths = self._turn_files.pop(turn_id, set())
            for path in paths:
                try:
                    validated = self.media_validator.validate(path)
                    validated.unlink(missing_ok=True)
                except (FileNotFoundError, OSError, MediaPathError):
                    pass
            turn_dir = self._turn_dirs.pop(turn_id, None)
            if turn_dir is not None:
                try:
                    validated_dir = self.media_validator.validate(
                        turn_dir, allow_directory=True
                    )
                    shutil.rmtree(validated_dir)
                except (FileNotFoundError, OSError, MediaPathError):
                    pass

    async def _handle_open(
        self, params: Any, request: RequestContext
    ) -> Mapping[str, Any]:
        payload = _params_object(params)
        turn_id = _required_string(payload, "turn_id", max_length=256)
        raw_notice_ids = payload.get("notice_ids", [])
        if not isinstance(raw_notice_ids, list) or not all(
            isinstance(item, str) and item for item in raw_notice_ids
        ):
            raise JsonRpcError(INVALID_PARAMS, "notice_ids must be an array of strings")
        if len(raw_notice_ids) > 100:
            raise JsonRpcError(INVALID_PARAMS, "At most 100 notices can be opened per turn")
        target_ref = payload.get("target_ref")
        if target_ref is not None:
            try:
                target_ref = _json_scalar_id(target_ref, field_name="target_ref")
            except ValueError as exc:
                raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc
        if not raw_notice_ids and target_ref is None:
            raise JsonRpcError(
                INVALID_PARAMS, "telegram.open needs notice_ids or target_ref"
            )
        include_history = payload.get("include_history", True)
        if not isinstance(include_history, bool):
            raise JsonRpcError(INVALID_PARAMS, "include_history must be boolean")

        turn_dir = self._turn_dir(turn_id)
        try:
            result = dict(
                await self.adapter.materialize(
                    tuple(raw_notice_ids),
                    turn_id=turn_id,
                    turn_dir=turn_dir,
                    target_ref=target_ref,
                    include_history=include_history,
                )
            )
        except MediaPathError:
            raise
        except ValueError as exc:
            raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc
        except Exception as exc:
            # Materialization failures otherwise become an unhelpful generic
            # JSON-RPC error.  The service needs the exception text to defer
            # only the affected notice and leave the rest of the chat queue
            # available for processing.
            raise JsonRpcError(
                INTERNAL_ERROR,
                "Telegram materialization failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        if "_target" not in result:
            raise JsonRpcError(INVALID_PARAMS, "Adapter did not resolve a Telegram target")
        try:
            target = _json_scalar_id(result.pop("_target"), field_name="_target")
        except ValueError as exc:
            raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc
        message_ids = _internal_id_set(result.pop("_message_ids", ()), "message")
        sender_ids = _internal_scalar_id_set(result.pop("_sender_ids", ()), "sender")
        expected_message_ids: set[int] = set()
        for notice_id in raw_notice_ids:
            try:
                _, message_id = _parse_notice_id(notice_id)
            except ValueError as exc:
                raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc
            expected_message_ids.add(message_id)
        missing_message_ids = expected_message_ids.difference(message_ids)
        if missing_message_ids:
            missing = ", ".join(
                notice_id
                for notice_id in raw_notice_ids
                if _parse_notice_id(notice_id)[1] in missing_message_ids
            )
            raise JsonRpcError(
                INVALID_PARAMS,
                f"Telegram notices were not materialized: {missing}",
            )
        self._track_media_paths(turn_id, result)

        token = secrets.token_urlsafe(24)
        self._grants.setdefault(turn_id, {})[token] = _TargetGrant(
            target=target,
            message_ids=message_ids,
            sender_ids=sender_ids,
        )
        result["target_token"] = token
        result["target_ref"] = target
        result["turn_id"] = turn_id
        return result

    async def _handle_execute(
        self, params: Any, request: RequestContext
    ) -> Mapping[str, Any]:
        payload = _params_object(params)
        turn_id = _required_string(payload, "turn_id", max_length=256)
        target_token = _required_string(payload, "target_token", max_length=256)
        action = _required_string(payload, "action", max_length=64)
        if action not in ALLOWED_ACTIONS:
            raise JsonRpcError(INVALID_PARAMS, f"Unsupported Telegram action: {action}")
        if action in SIDE_EFFECT_ACTIONS and not request.idempotency_key:
            raise JsonRpcError(
                INVALID_PARAMS,
                f"Telegram action {action} requires an idempotency key",
            )
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise JsonRpcError(INVALID_PARAMS, "arguments must be an object")
        grant = self._grants.get(turn_id, {}).get(target_token)
        if grant is None:
            raise JsonRpcError(
                INVALID_PARAMS, "target_token is invalid or belongs to another turn"
            )

        normalized = dict(arguments)
        normalized["_target"] = grant.target
        self._validate_granted_action(action, normalized, grant)
        if action == "acknowledge":
            await self._reconcile_before_ack(grant, normalized["message_ids"])
        turn_dir = self._turn_dir(turn_id)
        if action == "send_media":
            path = normalized.get("media_path")
            if not isinstance(path, str):
                raise JsonRpcError(INVALID_PARAMS, "send_media needs media_path")
            try:
                normalized["media_path"] = str(self.media_validator.validate(path))
            except MediaPathError as exc:
                raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc
        result = dict(
            await self.adapter.execute_action(
                action,
                normalized,
                turn_id=turn_id,
                turn_dir=turn_dir,
                request=request,
            )
        )
        self._track_media_paths(turn_id, result)
        if action == "acknowledge":
            # A startup page is bounded.  Once its oldest prefix is read, reveal
            # the next unread page immediately so a later live-message max_id
            # can never skip historical messages that Milana has not opened.
            self._schedule_backfill_refresh()
        return result

    async def _reconcile_before_ack(
        self, grant: _TargetGrant, message_ids: Sequence[int]
    ) -> None:
        """Publish every unread message that a Telegram max-id ack would cross.

        ``send_read_acknowledge(max_id=N)`` also reads N-1.  A delayed NewMessage
        callback for N-1 would therefore disappear from the normal unread
        backfill if the read marker advanced first.  The real adapter exposes a
        targeted, unbounded scan for this window.  It is detected dynamically so
        older/test adapters keep their existing protocol compatibility.

        Reconciliation is deliberately part of the acknowledgement transaction:
        if the service cannot durably accept a recovered notice, the read marker
        is not advanced and startup/periodic backfill can try again later.
        """

        reconcile = getattr(self.adapter, "backfill_before_ack", None)
        if not callable(reconcile):
            return
        through_message_id = max(message_ids)
        async with self._backfill_lock:
            notices = reconcile(grant.target, through_message_id)
            if inspect.isawaitable(notices):
                notices = await notices
            if not isinstance(notices, Sequence):
                raise RuntimeError("Telegram pre-ack reconciliation returned invalid notices")
            explicitly_acknowledged = frozenset(message_ids)
            for notice in notices:
                if not isinstance(notice, TelegramNotice):
                    raise RuntimeError(
                        "Telegram pre-ack reconciliation returned an invalid notice"
                    )
                if (
                    str(notice.chat_id) != str(grant.target)
                    or notice.message_id > through_message_id
                ):
                    raise RuntimeError(
                        "Telegram pre-ack reconciliation escaped the acknowledged window"
                    )
                if notice.message_id in explicitly_acknowledged:
                    # The current turn is the authority for its own opened
                    # notice.  Its journal row remains pending until this action
                    # succeeds, so asking the service for ``safe_to_ack`` here
                    # would deadlock every normal acknowledgement.
                    continue
                receipt = await self.publish_notice(
                    notice, wait_for_acceptance=True
                )
                if (
                    receipt is None
                    or not receipt.accepted
                    or not receipt.safe_to_ack
                ):
                    raise RuntimeError(
                        "Telegram acknowledgement is blocked by an unconfirmed "
                        f"notice: {notice.notice_id}"
                    )

    def _schedule_backfill_refresh(self) -> None:
        """Reveal the next unread page without coupling it to read-ack success."""

        current = self._backfill_refresh_task
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._publish_backfill(), name="telegram-backfill-after-ack"
        )
        self._backfill_refresh_task = task

        def completed(done: asyncio.Task[None]) -> None:
            if self._backfill_refresh_task is done:
                self._backfill_refresh_task = None
            if done.cancelled():
                return
            try:
                done.result()
            except (ConnectionClosedError, asyncio.CancelledError):
                return
            except Exception as exc:  # noqa: BLE001 - periodic poll will retry
                print(
                    f"Telegram post-ack backfill failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        task.add_done_callback(completed)

    def _validate_granted_action(
        self, action: str, arguments: Mapping[str, Any], grant: _TargetGrant
    ) -> None:
        if action == "reaction":
            message_id = arguments.get("message_id")
            if (
                isinstance(message_id, bool)
                or not isinstance(message_id, int)
                or message_id not in grant.message_ids
            ):
                raise JsonRpcError(
                    INVALID_PARAMS, "reaction message_id was not opened in this turn"
                )
        elif action == "blacklist_sender":
            sender_id = arguments.get("sender_id")
            if sender_id not in grant.sender_ids:
                raise JsonRpcError(
                    INVALID_PARAMS, "sender_id was not opened in this turn"
                )
        elif action == "acknowledge":
            raw_ids = arguments.get("message_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                raise JsonRpcError(INVALID_PARAMS, "acknowledge needs message_ids")
            if any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item not in grant.message_ids
                for item in raw_ids
            ):
                raise JsonRpcError(
                    INVALID_PARAMS, "acknowledge contains an unopened message_id"
                )

    async def _handle_backfill(
        self, params: Any, request: RequestContext
    ) -> Mapping[str, Any]:
        payload = _params_object(params)
        limit = payload.get("limit", self.max_backfill)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.max_backfill:
            raise JsonRpcError(
                INVALID_PARAMS, f"limit must be between 1 and {self.max_backfill}"
            )
        notices = await self.adapter.backfill(limit)
        return {"notices": [notice.to_payload() for notice in notices]}

    async def _handle_cleanup_turn(
        self, params: Any, request: RequestContext
    ) -> Mapping[str, Any]:
        turn_id = _required_string(_params_object(params), "turn_id", max_length=256)
        await self.cleanup_turn(turn_id)
        return {"cleaned": True, "turn_id": turn_id}

    async def _handle_health(
        self, params: Any, request: RequestContext
    ) -> Mapping[str, Any]:
        return {
            "ok": self._started and self.peer is not None and not self.peer.closed,
            "skill": "telegram",
            "active_turns": len(self._grants),
        }

    async def _handle_presence(
        self, params: Any, request: RequestContext
    ) -> Mapping[str, Any]:
        payload = _params_object(params)
        online = payload.get("online")
        if not isinstance(online, bool):
            raise JsonRpcError(INVALID_PARAMS, "online must be boolean")
        callback = getattr(self.adapter, "set_presence", None)
        if not callable(callback):
            raise JsonRpcError(INVALID_PARAMS, "Adapter does not support presence")
        result = callback(online)
        if inspect.isawaitable(result):
            try:
                await asyncio.wait_for(result, timeout=SIGNAL_TIMEOUT_SECONDS)
            except TimeoutError:
                return {"online": online, "applied": False, "timed_out": True}
        return {"online": online}


def _params_object(params: Any) -> Mapping[str, Any]:
    if not isinstance(params, Mapping):
        raise JsonRpcError(INVALID_PARAMS, "params must be an object")
    return params


def _consume_background_task(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


def _required_string(
    payload: Mapping[str, Any], field_name: str, *, max_length: int
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise JsonRpcError(
            INVALID_PARAMS,
            f"{field_name} must be a non-empty string up to {max_length} characters",
        )
    return value


def _internal_id_set(values: Any, label: str) -> frozenset[int]:
    if not isinstance(values, (tuple, list, set, frozenset)):
        raise JsonRpcError(INVALID_PARAMS, f"Adapter {label} IDs are invalid")
    result: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise JsonRpcError(INVALID_PARAMS, f"Adapter {label} ID is invalid")
        result.add(value)
    return frozenset(result)


def _internal_scalar_id_set(values: Any, label: str) -> frozenset[str | int]:
    if not isinstance(values, (tuple, list, set, frozenset)):
        raise JsonRpcError(INVALID_PARAMS, f"Adapter {label} IDs are invalid")
    result: set[str | int] = set()
    for value in values:
        try:
            result.add(_json_scalar_id(value, field_name=f"{label}_id"))
        except ValueError as exc:
            raise JsonRpcError(INVALID_PARAMS, str(exc)) from exc
    return frozenset(result)


class TelethonTelegramAdapter:
    """Real adapter, imported lazily so protocol tests need no Telegram I/O."""

    def __init__(self, *, dev_chat: bool = False) -> None:
        # These imports are intentionally local.  telegram_client remains the
        # compatibility CLI and the established helpers stay unmodified.
        from telethon import TelegramClient, events
        from telegram_client import load_config, render_sticker_png
        from milana_stickers import MilanaStickerSkill

        config = load_config()
        self.client = TelegramClient(
            str(config.session_path), config.api_id, config.api_hash
        )
        self._events = events
        self._notice_callback: NoticeCallback | None = None
        self._handler: Any | None = None
        self._messages: OrderedDict[str, Any] = OrderedDict()
        self._senders: OrderedDict[str, Any] = OrderedDict()
        self._dev_chat = bool(dev_chat)
        self._render_sticker_png = render_sticker_png
        self._sticker_skill = MilanaStickerSkill(
            self.client, animated_renderer=render_sticker_png
        )
        self._sticker_sessions: dict[str, Any] = {}
        self._typing_contexts: dict[str, Any] = {}

    async def start(self, on_notice: NoticeCallback) -> None:
        self._notice_callback = on_notice
        await self.client.start()
        await self._ensure_connected()

        async def incoming(event: Any) -> None:
            notice = await self._notice_from_message(event.message, event=event)
            self._remember(notice.notice_id, event.message)
            try:
                await on_notice(notice)
            except ConnectionClosedError:
                # It remains unread and is found by backfill after restart.
                return

        self._handler = incoming
        self.client.add_event_handler(incoming, self._events.NewMessage(incoming=True))

    async def stop(self) -> None:
        for turn_id in tuple(self._typing_contexts):
            await self._stop_typing(turn_id)
        if self._handler is not None:
            self.client.remove_event_handler(self._handler)
            self._handler = None
        if self.client.is_connected():
            await self.set_presence(False)
            await self.client.disconnect()

    async def set_presence(self, online: bool) -> None:
        if not self.client.is_connected():
            return
        from telethon import functions

        await self.client(functions.account.UpdateStatusRequest(offline=not online))

    async def _ensure_connected(self) -> None:
        """Reconnect the persistent Telethon session before an RPC operation.

        The skill host can stay alive while Telegram briefly drops its MTProto
        transport.  ``TelegramClient.start()`` authenticates at launch, but it
        does not guarantee that a later materialization or send still has an
        active transport.  Reconnect on demand so a queued notice reaches the
        model instead of failing before the provider fallback can run.

        Tiny protocol-test doubles do not model a network connection, so they
        intentionally bypass this production-only guard.
        """

        is_connected = getattr(self.client, "is_connected", None)
        if not callable(is_connected) or is_connected():
            return
        connect = getattr(self.client, "connect", None)
        if not callable(connect):
            raise ConnectionError("Telegram client cannot reconnect")
        await connect()
        if not is_connected():
            raise ConnectionError("Telegram reconnect did not establish a session")

    async def backfill(self, limit: int) -> Sequence[TelegramNotice]:
        await self._ensure_connected()
        notices: list[TelegramNotice] = []
        async for dialog in self.client.iter_dialogs():
            unread = int(getattr(dialog, "unread_count", 0) or 0)
            if unread <= 0:
                continue
            remaining = limit - len(notices)
            if remaining <= 0:
                break
            dialog_state = getattr(dialog, "dialog", None)
            read_inbox_max_id = getattr(dialog_state, "read_inbox_max_id", 0)
            if isinstance(read_inbox_max_id, bool) or not isinstance(
                read_inbox_max_id, int
            ):
                read_inbox_max_id = 0
            wanted_incoming = min(unread, remaining)
            top_message_id = getattr(dialog_state, "top_message", None)
            if isinstance(top_message_id, bool) or not isinstance(
                top_message_id, int
            ):
                top_message_id = getattr(
                    getattr(dialog, "message", None), "id", None
                )
            options: dict[str, Any] = {
                "min_id": max(0, read_inbox_max_id),
                "reverse": True,
            }
            if (
                isinstance(top_message_id, int)
                and not isinstance(top_message_id, bool)
                and top_message_id > read_inbox_max_id
            ):
                # Freeze a finite ID window.  Telethon pages through it lazily;
                # outgoing messages do not consume the incoming notice budget.
                options["max_id"] = top_message_id + 1
            messages = self.client.iter_messages(
                dialog.entity,
                limit=None,
                **options,
            )
            incoming_count = 0
            async for message in messages:
                if bool(getattr(message, "out", False)):
                    continue
                notice = await self._notice_from_message(message)
                self._remember(notice.notice_id, message)
                notices.append(notice)
                incoming_count += 1
                if incoming_count >= wanted_incoming:
                    break
        return tuple(notices[:limit])

    async def backfill_before_ack(
        self, target: str | int, through_message_id: int
    ) -> Sequence[TelegramNotice]:
        """Return the complete unread prefix that ``max_id`` would mark read."""

        if (
            isinstance(through_message_id, bool)
            or not isinstance(through_message_id, int)
            or through_message_id <= 0
        ):
            raise ValueError("through_message_id must be a positive integer")
        await self._ensure_connected()

        matched_dialog: Any | None = None
        async for dialog in self.client.iter_dialogs():
            if str(getattr(dialog, "id", "")) == str(target):
                matched_dialog = dialog
                break
        if matched_dialog is None:
            # Failing closed is important here: acknowledging without knowing the
            # current inbox marker could permanently hide an older unread message.
            raise RuntimeError("Telegram target is absent from the dialog list")

        dialog_state = getattr(matched_dialog, "dialog", None)
        read_inbox_max_id = getattr(dialog_state, "read_inbox_max_id", 0)
        if isinstance(read_inbox_max_id, bool) or not isinstance(
            read_inbox_max_id, int
        ):
            read_inbox_max_id = 0
        if read_inbox_max_id >= through_message_id:
            return ()

        notices: list[TelegramNotice] = []
        messages = self.client.iter_messages(
            matched_dialog.entity,
            limit=None,
            min_id=max(0, read_inbox_max_id),
            max_id=through_message_id + 1,
            reverse=True,
        )
        async for message in messages:
            message_id = getattr(message, "id", None)
            if (
                isinstance(message_id, bool)
                or not isinstance(message_id, int)
                or message_id <= read_inbox_max_id
                or message_id > through_message_id
                or bool(getattr(message, "out", False))
            ):
                continue
            notice = await self._notice_from_message(message)
            self._remember(notice.notice_id, message)
            notices.append(notice)
        return tuple(notices)

    async def acknowledge_terminal_notice(self, notice: TelegramNotice) -> bool:
        """Mark one handled/poison notice read without crossing an unread gap.

        Telegram's read API advances a per-chat max-id marker.  Before using it,
        re-read the complete interval between the current marker and the terminal
        notice.  Any older incoming message blocks the operation; it must first
        receive its own durable service disposition on a later backfill pass.
        """

        target = notice.chat_id
        through_message_id = notice.message_id
        await self._ensure_connected()
        matched_dialog: Any | None = None
        async for dialog in self.client.iter_dialogs():
            if str(getattr(dialog, "id", "")) == str(target):
                matched_dialog = dialog
                break
        if matched_dialog is None:
            raise RuntimeError("Telegram target is absent from the dialog list")

        dialog_state = getattr(matched_dialog, "dialog", None)
        read_inbox_max_id = getattr(dialog_state, "read_inbox_max_id", 0)
        if isinstance(read_inbox_max_id, bool) or not isinstance(
            read_inbox_max_id, int
        ):
            read_inbox_max_id = 0
        if read_inbox_max_id >= through_message_id:
            return True

        messages = self.client.iter_messages(
            matched_dialog.entity,
            limit=None,
            min_id=max(0, read_inbox_max_id),
            max_id=through_message_id + 1,
            reverse=True,
        )
        async for message in messages:
            if bool(getattr(message, "out", False)):
                continue
            message_id = getattr(message, "id", None)
            if isinstance(message_id, bool) or not isinstance(message_id, int):
                raise RuntimeError(
                    "Telegram returned an incoming message without a stable ID"
                )
            if read_inbox_max_id < message_id < through_message_id:
                return False

        await self.client.send_read_acknowledge(
            matched_dialog.entity, max_id=through_message_id
        )
        return True

    async def materialize(
        self,
        notice_ids: Sequence[str],
        *,
        turn_id: str,
        turn_dir: Path,
        target_ref: str | int | None = None,
        include_history: bool = True,
    ) -> Mapping[str, Any]:
        await self._ensure_connected()
        selected: list[tuple[str, Any]] = []
        target: str | int | None = target_ref
        for notice_id in notice_ids:
            message = self._messages.get(notice_id)
            if message is None:
                chat_id, message_id = _parse_notice_id(notice_id)
                message = await self.client.get_messages(chat_id, ids=message_id)
            if message is None:
                raise ValueError(
                    f"Telegram notice could not be materialized: {notice_id}"
                )
            message_chat = getattr(message, "chat_id", None)
            if message_chat is None:
                message_chat, _ = _parse_notice_id(notice_id)
            if target is None:
                target = message_chat
            if str(message_chat) != str(target):
                raise ValueError("All opened notices must belong to one Telegram chat")
            self._remember(notice_id, message)
            selected.append((notice_id, message))
        if target is None:
            raise ValueError("Telegram target could not be resolved")

        materialized: list[dict[str, Any]] = []
        message_ids: set[int] = set()
        sender_ids: set[str | int] = set()
        for notice_id, message in selected:
            sender = await self._sender_for_message(message, notice_id=notice_id)
            item = await self._message_content(
                message, turn_dir=turn_dir, sender=sender
            )
            materialized.append(item)
            message_id = item.get("message_id")
            if isinstance(message_id, int):
                message_ids.add(message_id)
            sender = item.get("sender")
            if isinstance(sender, Mapping) and isinstance(sender.get("id"), (str, int)):
                sender_ids.add(sender["id"])

        history: list[dict[str, Any]] = []
        if include_history:
            async for message in self.client.iter_messages(target, limit=30):
                message_id = getattr(message, "id", None)
                if message_id in message_ids:
                    continue
                sender = await self._sender_for_message(message)
                history.append(
                    {
                        "message_id": message_id,
                        "timestamp": _utc_iso(getattr(message, "date", None)),
                        "sender": _sender_payload(sender),
                        "outgoing": bool(getattr(message, "out", False)),
                        "text": str(getattr(message, "raw_text", "") or ""),
                        "media_type": _media_type(message),
                    }
                )
            history.reverse()
        return {
            "_target": target,
            "_message_ids": sorted(message_ids),
            "_sender_ids": list(sender_ids),
            "messages": materialized,
            "history": history,
        }

    async def execute_action(
        self,
        action: str,
        arguments: Mapping[str, Any],
        *,
        turn_id: str,
        turn_dir: Path,
        request: RequestContext,
    ) -> Mapping[str, Any]:
        await self._ensure_connected()
        target = arguments["_target"]
        if action == "typing":
            active = arguments.get("active")
            if not isinstance(active, bool):
                raise ValueError("typing active must be boolean")
            deadline = asyncio.get_running_loop().time() + SIGNAL_TIMEOUT_SECONDS
            await self._stop_typing(turn_id, timeout=SIGNAL_TIMEOUT_SECONDS)
            if active:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return {
                        "status": "typing",
                        "active": False,
                        "timed_out": True,
                    }
                context = self.client.action(target, "typing")
                try:
                    await asyncio.wait_for(
                        context.__aenter__(), timeout=remaining
                    )
                except TimeoutError:
                    await self._close_typing_context(context, timeout=0)
                    return {
                        "status": "typing",
                        "active": False,
                        "timed_out": True,
                    }
                except BaseException:
                    await self._close_typing_context(context, timeout=0)
                    raise
                self._typing_contexts[turn_id] = context
            return {"status": "typing", "active": active}

        if action == "send_messages":
            messages = arguments.get("messages")
            if not isinstance(messages, list) or not messages or len(messages) > MAX_OUTGOING_MESSAGES:
                raise ValueError(
                    f"messages must contain 1-{MAX_OUTGOING_MESSAGES} strings"
                )
            if any(
                not isinstance(text, str)
                or not text.strip()
                or len(text) > MAX_MESSAGE_LENGTH
                for text in messages
            ):
                raise ValueError("Outgoing Telegram messages are invalid")
            sent_ids: list[int] = []
            sent_part_indexes: list[int] = []
            deduplicated_part_indexes: list[int] = []
            batch_id = arguments.get("batch_id", f"{turn_id}:send_messages")
            if (
                not isinstance(batch_id, str)
                or not batch_id.strip()
                or len(batch_id) > 256
            ):
                raise ValueError(
                    "batch_id must be a non-empty string up to 256 characters"
                )
            start_index = arguments.get("start_index", 0)
            if (
                isinstance(start_index, bool)
                or not isinstance(start_index, int)
                or not 0 <= start_index <= len(messages)
            ):
                raise ValueError(
                    "start_index must identify a message part or completion"
                )
            minimum = arguments.get("inter_message_min_delay_seconds", 1.0)
            maximum = arguments.get("inter_message_max_delay_seconds", 15.0)
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
                for value in (minimum, maximum)
            ) or minimum > maximum:
                raise ValueError("inter-message delay bounds are invalid")
            from telegram_client import inter_message_typing_delay

            started_at = time.perf_counter()
            first_sent_message_id: int | None = None
            first_sent_part_index: int | None = None
            first_send_elapsed_ms: float | None = None
            for index in range(start_index, len(messages)):
                text = messages[index]
                request.raise_if_cancelled()
                if index > start_index and not self._dev_chat:
                    delay = inter_message_typing_delay(
                        text,
                        minimum_seconds=float(minimum),
                        maximum_seconds=float(maximum),
                    )
                    await asyncio.sleep(delay)
                    request.raise_if_cancelled()
                try:
                    random_id = _part_random_id(
                        batch_id,
                        target=target,
                        part_index=index,
                        text=text,
                    )
                    sent = await self._send_text(
                        target, text, random_id=random_id
                    )
                except Exception as exc:  # report already-sent parts explicitly
                    return {
                        "status": "partial" if sent_part_indexes else "failed",
                        "batch_id": batch_id,
                        "sent_message_ids": sent_ids,
                        "sent_part_indexes": sent_part_indexes,
                        "deduplicated_part_indexes": deduplicated_part_indexes,
                        "next_part_index": index,
                        "total_parts": len(messages),
                        "first_sent_message_id": first_sent_message_id,
                        "first_sent_part_index": first_sent_part_index,
                        "first_send_elapsed_ms": first_send_elapsed_ms,
                        "error": str(exc),
                    }
                candidate = getattr(sent, "id", None)
                if isinstance(candidate, int):
                    sent_ids.append(candidate)
                sent_part_indexes.append(index)
                if bool(getattr(sent, "deduplicated", False)):
                    deduplicated_part_indexes.append(index)
                if first_send_elapsed_ms is None:
                    first_sent_part_index = index
                    first_send_elapsed_ms = round(
                        (time.perf_counter() - started_at) * 1000.0, 3
                    )
                    if isinstance(candidate, int):
                        first_sent_message_id = candidate
            return {
                "status": "sent",
                "batch_id": batch_id,
                "sent_message_ids": sent_ids,
                "sent_part_indexes": sent_part_indexes,
                "deduplicated_part_indexes": deduplicated_part_indexes,
                "next_part_index": len(messages),
                "total_parts": len(messages),
                "first_sent_message_id": first_sent_message_id,
                "first_sent_part_index": first_sent_part_index,
                "first_send_elapsed_ms": first_send_elapsed_ms,
            }

        if action == "send_media":
            caption = arguments.get("caption")
            if caption is not None and not isinstance(caption, str):
                raise ValueError("caption must be a string or null")
            sent = await self.client.send_file(
                target, arguments["media_path"], caption=caption
            )
            return {"status": "sent", "message_id": getattr(sent, "id", None)}

        if action == "reaction":
            from telethon import functions, types

            reaction = arguments.get("reaction")
            if not isinstance(reaction, str) or not reaction:
                raise ValueError("reaction must be a non-empty string")
            peer = await self.client.get_input_entity(target)
            await self.client(
                functions.messages.SendReactionRequest(
                    peer=peer,
                    msg_id=int(arguments["message_id"]),
                    reaction=[types.ReactionEmoji(emoticon=reaction)],
                )
            )
            return {"status": "reacted"}

        if action == "blacklist_sender":
            from telethon import functions

            entity = await self.client.get_input_entity(arguments["sender_id"])
            await self.client(functions.contacts.BlockRequest(id=entity))
            return {"status": "blocked"}

        if action == "acknowledge":
            message_ids = list(arguments["message_ids"])
            await self.client.send_read_acknowledge(target, max_id=max(message_ids))
            return {"status": "acknowledged", "through_message_id": max(message_ids)}

        if action == "schedule_message":
            delay = _positive_delay(arguments.get("delay_seconds"))
            message = arguments.get("message")
            if not isinstance(message, str) or not message.strip() or len(message) > 4000:
                raise ValueError("scheduled message is invalid")
            return {
                "status": "staged",
                "action": "send_message",
                "target": target,
                "delay_seconds": delay,
                "message": message,
            }

        if action == "send_sticker_reference":
            from milana_stickers import StickerReference

            raw = arguments.get("sticker")
            if not isinstance(raw, Mapping):
                raise ValueError("send_sticker_reference needs sticker")
            reference = StickerReference(
                set_id=int(raw["set_id"]),
                set_access_hash=int(raw["set_access_hash"]),
                set_short_name=str(raw["set_short_name"]),
                document_id=int(raw["document_id"]),
                pack_title=str(raw["pack_title"]),
                emoji=str(raw["emoji"]),
            )
            resolved = await self._sticker_skill.resolve_reference(reference)
            sent = await self._send_sticker(
                target,
                resolved.document,
                random_id=_idempotency_random_id(request.idempotency_key),
            )
            return {
                "status": "sent",
                "message_id": getattr(sent, "id", None),
                "deduplicated": bool(getattr(sent, "deduplicated", False)),
                "sticker": _sticker_reference_payload(reference),
            }
        session = self._sticker_sessions.setdefault(
            turn_id, self._sticker_skill.new_session()
        )
        if action == "open_sticker_picker":
            picker = await session.open(arguments.get("pack_id"))
            content = await self._externalize_picker_content(
                picker.content, turn_dir=turn_dir
            )
            return {"status": "ok", "content": content}

        sticker_id = arguments.get("sticker_id")
        choice = session.choose(sticker_id)
        if action == "send_sticker":
            sent = await self._send_sticker(
                target,
                choice.document,
                random_id=_idempotency_random_id(request.idempotency_key),
            )
            return {
                "status": "sent",
                "message_id": getattr(sent, "id", None),
                "deduplicated": bool(getattr(sent, "deduplicated", False)),
                "sticker": _sticker_reference_payload(choice.reference),
            }
        if action == "schedule_sticker":
            return {
                "status": "staged",
                "action": "send_sticker",
                "target": target,
                "delay_seconds": _positive_delay(arguments.get("delay_seconds")),
                "sticker": _sticker_reference_payload(choice.reference),
            }
        raise ValueError(f"Unsupported action: {action}")

    async def cleanup_turn(self, turn_id: str) -> None:
        # Telegram's action context is cosmetic. A stuck __aexit__ must not
        # serialize the next chat behind cleanup of the previous turn.
        await self._stop_typing(turn_id, timeout=SIGNAL_TIMEOUT_SECONDS)
        self._sticker_sessions.pop(turn_id, None)

    async def _stop_typing(
        self, turn_id: str, *, timeout: float | None = None
    ) -> None:
        context = self._typing_contexts.pop(turn_id, None)
        if context is not None:
            await self._close_typing_context(context, timeout=timeout)

    @staticmethod
    async def _close_typing_context(
        context: Any, *, timeout: float | None = SIGNAL_TIMEOUT_SECONDS
    ) -> None:
        task = asyncio.create_task(
            context.__aexit__(None, None, None), name="telegram-stop-typing"
        )
        try:
            if timeout is None:
                await task
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except TimeoutError:
            pass
        finally:
            if not task.done():
                task.add_done_callback(_consume_background_task)

    async def _send_text(
        self, target: str | int, text: str, *, random_id: int
    ) -> Any:
        """Send text with a caller-supplied MTProto deduplication ID."""

        from telethon import functions, types, utils
        from telethon.errors import RandomIdDuplicateError

        entity = await self.client.get_input_entity(target)
        parsed_text, formatting_entities = await self.client._parse_message_text(
            text, ()
        )
        request = functions.messages.SendMessageRequest(
            peer=entity,
            message=parsed_text,
            entities=formatting_entities,
            no_webpage=False,
            random_id=random_id,
        )
        try:
            sender = getattr(self.client, "_sender", None)
            if sender is None:
                result = await self.client(request)
            else:
                # TelegramClient._call retries every ServerError five times,
                # including RANDOM_ID_DUPLICATE.  A direct one-shot future
                # makes a lost-response retry complete immediately while the
                # stable random_id still prevents a second message.
                result = await sender.send(request)
                session = getattr(self.client, "session", None)
                process_entities = getattr(session, "process_entities", None)
                if callable(process_entities):
                    await utils.maybe_async(process_entities(result))
        except RandomIdDuplicateError:
            return _SendReceipt(id=None, deduplicated=True)
        if isinstance(result, types.UpdateShortSentMessage):
            return result
        return self.client._get_response_message(request, result, entity)

    async def _send_sticker(
        self, target: str | int, document: Any, *, random_id: int
    ) -> Any:
        """Send one sticker with a durable caller-supplied MTProto ID.

        ``TelegramClient.send_file`` creates a fresh random ID internally, so a
        host restart after Telegram accepted the sticker but before the JSON-RPC
        response arrived could send it twice.  Building ``SendMediaRequest``
        directly gives retries the same ID and lets Telegram reject the replay.
        """

        from telethon import functions, utils
        from telethon.errors import RandomIdDuplicateError

        entity = await self.client.get_input_entity(target)
        _file_handle, media, _image = await self.client._file_to_media(document)
        if media is None:
            raise TypeError("Cannot convert sticker document to Telegram media")
        telegram_request = functions.messages.SendMediaRequest(
            peer=entity,
            media=media,
            message="",
            entities=[],
            random_id=random_id,
        )
        try:
            sender = getattr(self.client, "_sender", None)
            if sender is None:
                result = await self.client(telegram_request)
            else:
                # Avoid TelegramClient._call's automatic retries of
                # RANDOM_ID_DUPLICATE; this one-shot path turns the duplicate
                # into a successful idempotent receipt immediately.
                result = await sender.send(telegram_request)
                session = getattr(self.client, "session", None)
                process_entities = getattr(session, "process_entities", None)
                if callable(process_entities):
                    await utils.maybe_async(process_entities(result))
        except RandomIdDuplicateError:
            return _SendReceipt(id=None, deduplicated=True)
        return self.client._get_response_message(
            telegram_request, result, entity
        )

    async def _notice_from_message(
        self, message: Any, *, event: Any | None = None
    ) -> TelegramNotice:
        chat_id = getattr(message, "chat_id", None)
        message_id = getattr(message, "id", None)
        if chat_id is None or not isinstance(message_id, int):
            raise ValueError("Telegram message has no stable chat/message ID")
        sender = await (
            event.get_sender() if event is not None else message.get_sender()
        )
        notice_id = _notice_id(chat_id, message_id)
        self._remember_sender(notice_id, sender)
        return TelegramNotice(
            notice_id=notice_id,
            chat_id=chat_id,
            message_id=message_id,
            timestamp=_utc_iso(getattr(message, "date", None)),
            sender=_sender_payload(sender),
            media_type=_media_type(message),
        )

    async def _message_content(
        self, message: Any, *, turn_dir: Path, sender: Any | None = None
    ) -> dict[str, Any]:
        if sender is None:
            sender = await self._sender_for_message(message)
        result: dict[str, Any] = {
            "message_id": getattr(message, "id", None),
            "timestamp": _utc_iso(getattr(message, "date", None)),
            "sender": _sender_payload(sender),
            "text": str(getattr(message, "raw_text", "") or ""),
            "media_type": _media_type(message),
        }
        if getattr(message, "media", None) is not None:
            suffix = _media_suffix(message)
            destination = turn_dir / f"message-{int(message.id)}{suffix}"
            downloaded = await message.download_media(file=str(destination))
            if downloaded:
                resolved = Path(downloaded).resolve(strict=True)
                file_info = getattr(message, "file", None)
                declared_mime = getattr(file_info, "mime_type", None)
                mime = (
                    declared_mime
                    if isinstance(declared_mime, str) and declared_mime
                    else mimetypes.guess_type(resolved.name)[0]
                )
                if bool(getattr(message, "sticker", None)) and mime in {
                    "application/x-tgsticker",
                    "video/webm",
                }:
                    try:
                        png = await asyncio.to_thread(
                            self._render_sticker_png,
                            resolved.read_bytes(),
                            mime,
                        )
                    except ValueError:
                        pass
                    else:
                        preview = turn_dir / f"message-{int(message.id)}-sticker.png"
                        preview.write_bytes(png)
                        resolved = preview.resolve(strict=True)
                        mime = "image/png"
                result["media_path"] = str(resolved)
                if mime:
                    result["media_mime_type"] = mime
        return result

    async def _externalize_picker_content(
        self, content: Sequence[Mapping[str, Any]], *, turn_dir: Path
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        image_index = 0
        for item in content:
            normalized = dict(item)
            data_url = normalized.get("image_url")
            if normalized.get("type") == "input_image" and isinstance(data_url, str):
                prefix = "data:image/png;base64,"
                if not data_url.startswith(prefix):
                    raise ValueError("Sticker picker returned an unsupported image")
                image_index += 1
                raw = base64.b64decode(data_url[len(prefix) :], validate=True)
                path = turn_dir / f"sticker-picker-{image_index:03d}.png"
                path.write_bytes(raw)
                normalized.pop("image_url", None)
                normalized["path"] = str(path.resolve(strict=True))
            result.append(normalized)
        return result

    def _remember(self, notice_id: str, message: Any) -> None:
        self._messages[notice_id] = message
        self._messages.move_to_end(notice_id)
        while len(self._messages) > MAX_NOTICE_CACHE:
            evicted_notice_id, _ = self._messages.popitem(last=False)
            senders = getattr(self, "_senders", None)
            if senders is not None:
                senders.pop(evicted_notice_id, None)

    def _remember_sender(self, notice_id: str, sender: Any) -> None:
        if sender is None:
            return
        senders = getattr(self, "_senders", None)
        if senders is None:
            senders = OrderedDict()
            self._senders = senders
        senders[notice_id] = sender
        senders.move_to_end(notice_id)
        while len(senders) > MAX_NOTICE_CACHE:
            senders.popitem(last=False)

    async def _sender_for_message(
        self, message: Any, *, notice_id: str | None = None
    ) -> Any:
        if notice_id is None:
            chat_id = getattr(message, "chat_id", None)
            message_id = getattr(message, "id", None)
            if chat_id is not None and isinstance(message_id, int):
                notice_id = _notice_id(chat_id, message_id)
        senders = getattr(self, "_senders", None)
        if notice_id is not None and senders is not None:
            sender = senders.get(notice_id)
            if sender is not None:
                senders.move_to_end(notice_id)
                return sender
        sender = await message.get_sender()
        if notice_id is not None:
            self._remember_sender(notice_id, sender)
        return sender


def _notice_id(chat_id: str | int, message_id: int) -> str:
    return f"tg:{chat_id}:{message_id}"


def _parse_notice_id(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 3 or parts[0] != "tg":
        raise ValueError("Invalid Telegram notice ID")
    try:
        return int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError("Invalid Telegram notice ID") from exc


def _part_random_id(
    batch_id: str,
    *,
    target: str | int,
    part_index: int,
    text: str,
) -> int:
    """Derive one stable signed 64-bit MTProto ID per outgoing part."""

    material = (
        f"{batch_id}\x1f{type(target).__name__}:{target}"
        f"\x1f{part_index}\x1f{text}"
    ).encode("utf-8")
    digest = hashlib.blake2b(
        material, digest_size=8, person=b"milana-tg-send"
    ).digest()
    value = int.from_bytes(digest, "little", signed=True)
    return value or 1


def _idempotency_random_id(idempotency_key: str | None) -> int:
    """Map one durable JSON-RPC action key to one signed MTProto random ID."""

    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise ValueError("Sticker send requires a non-empty idempotency key")
    digest = hashlib.blake2b(
        idempotency_key.encode("utf-8"),
        digest_size=8,
        person=b"milana-tg-stkr",
    ).digest()
    value = int.from_bytes(digest, "little", signed=True)
    return value or 1


def _sender_payload(sender: Any) -> dict[str, Any]:
    sender_id = getattr(sender, "id", None)
    first = str(getattr(sender, "first_name", "") or "").strip()
    last = str(getattr(sender, "last_name", "") or "").strip()
    title = str(getattr(sender, "title", "") or "").strip()
    display_name = " ".join(part for part in (first, last) if part) or title
    return {
        "id": sender_id,
        "display_name": display_name or str(sender_id or "unknown"),
        "username": getattr(sender, "username", None),
    }


def _media_type(message: Any) -> str:
    if bool(getattr(message, "sticker", None)):
        return "sticker"
    if bool(getattr(message, "gif", None)):
        return "gif"
    if bool(getattr(message, "voice", None)):
        return "voice"
    if bool(getattr(message, "video", None)):
        return "video"
    if bool(getattr(message, "photo", None)):
        return "photo"
    if bool(getattr(message, "audio", None)):
        return "audio"
    if bool(getattr(message, "document", None)):
        return "document"
    if getattr(message, "media", None) is not None:
        return "media"
    return "text"


def _media_suffix(message: Any) -> str:
    file_info = getattr(message, "file", None)
    name = getattr(file_info, "name", None)
    if isinstance(name, str) and Path(name).suffix:
        return Path(name).suffix[:16]
    mime_type = getattr(file_info, "mime_type", None)
    suffix = mimetypes.guess_extension(mime_type) if isinstance(mime_type, str) else None
    return suffix or ".bin"


def _positive_delay(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("delay_seconds must be a positive integer")
    return value


def _sticker_reference_payload(reference: Any) -> dict[str, Any]:
    return {
        "set_id": int(reference.set_id),
        "set_access_hash": int(reference.set_access_hash),
        "set_short_name": str(reference.set_short_name),
        "document_id": int(reference.document_id),
        "pack_title": str(reference.pack_title),
        "emoji": str(reference.emoji),
    }


def _existing_token(path: str | os.PathLike[str]) -> str:
    token_path = Path(path)
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Cannot read Telegram host token file: {token_path}") from exc
    if not token:
        raise ValueError(f"Telegram host token file is empty: {token_path}")
    return token


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dependent Telegram skill host")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--dev-chat", action="store_true")
    return parser


async def run_host(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    token = _existing_token(args.token_file)
    adapter = TelethonTelegramAdapter(dev_chat=args.dev_chat)
    host = TelegramSkillHost(adapter, args.runtime_dir)
    await host.connect("127.0.0.1", args.port, token)
    await host.run_until_closed()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    try:
        asyncio.run(run_host(args))
    except KeyboardInterrupt:
        return 130
    except (ConnectionError, OSError, ValueError) as exc:
        print(f"Telegram skill host error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_ACTIONS",
    "LOCAL_STAGED_ACTIONS",
    "READ_ACTIONS",
    "RPC_BACKFILL",
    "RPC_CLEANUP_TURN",
    "RPC_EXECUTE",
    "RPC_HEALTH",
    "RPC_MATERIALIZE",
    "RPC_NOTICE",
    "RPC_OPEN",
    "RPC_PRESENCE",
    "SIDE_EFFECT_ACTIONS",
    "SIGNAL_ACTIONS",
    "TelegramAdapter",
    "TelegramNotice",
    "TelegramSkillHost",
    "TelethonTelegramAdapter",
    "build_parser",
    "main",
    "run_host",
]
