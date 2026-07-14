"""Authenticated local JSON-RPC transport used by Milana skill hosts.

The module deliberately has no Telegram or model dependencies.  A connection is
represented by :class:`JsonRpcPeer`; both ends of the connection can issue
requests, notifications and cancellation messages.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import math
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, TypeVar


JSONRPC_VERSION = "2.0"
MAX_FRAME = 1024 * 1024
HANDSHAKE_METHOD = "rpc.handshake"
CANCEL_METHOD = "$/cancelRequest"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
AUTHENTICATION_FAILED = -32001
IDEMPOTENCY_CONFLICT = -32009
REQUEST_CANCELLED = -32800

JsonValue = Any
RequestId = str | int | float | None
T = TypeVar("T")


class JsonRpcError(Exception):
    """An application or remote JSON-RPC error."""

    def __init__(self, code: int, message: str, data: JsonValue = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def as_dict(self) -> dict[str, JsonValue]:
        error: dict[str, JsonValue] = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


class JsonRpcProtocolError(Exception):
    """Raised when a peer sends an invalid frame or response."""


class FrameTooLargeError(JsonRpcProtocolError):
    """Raised when an incoming or outgoing payload exceeds ``MAX_FRAME``."""


class AuthenticationError(JsonRpcProtocolError):
    """Raised when the mandatory connection handshake is rejected."""


class ConnectionClosedError(ConnectionError):
    """Raised for pending requests when their transport goes away."""


class MediaPathError(ValueError):
    """Raised when a media path escapes the configured runtime directory."""


class IdempotencyConflictError(JsonRpcError):
    """The same idempotency key was reused for a different operation."""

    def __init__(self, key: str) -> None:
        super().__init__(
            IDEMPOTENCY_CONFLICT,
            "Idempotency key was already used with different arguments",
            {"key": key},
        )


def _validate_frame_limit(max_frame: int) -> None:
    if (
        isinstance(max_frame, bool)
        or not isinstance(max_frame, int)
        or max_frame <= 0
        or max_frame > MAX_FRAME
    ):
        raise ValueError(f"max_frame must be an integer between 1 and {MAX_FRAME}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def encode_frame(message: JsonValue, *, max_frame: int = MAX_FRAME) -> bytes:
    """Encode a JSON value with a four-byte big-endian length prefix."""

    _validate_frame_limit(max_frame)
    try:
        payload = json.dumps(
            message,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JsonRpcProtocolError("Message is not valid JSON") from exc
    if len(payload) > max_frame:
        raise FrameTooLargeError(
            f"JSON payload is {len(payload)} bytes; limit is {max_frame} bytes"
        )
    return struct.pack(">I", len(payload)) + payload


async def read_frame(
    reader: asyncio.StreamReader, *, max_frame: int = MAX_FRAME
) -> JsonValue:
    """Read and decode one length-prefixed JSON frame."""

    _validate_frame_limit(max_frame)
    header = await reader.readexactly(4)
    (length,) = struct.unpack(">I", header)
    if length == 0:
        raise JsonRpcProtocolError("Empty JSON frame")
    if length > max_frame:
        raise FrameTooLargeError(
            f"Incoming JSON payload is {length} bytes; limit is {max_frame} bytes"
        )
    payload = await reader.readexactly(length)
    try:
        return json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise JsonRpcProtocolError("Invalid UTF-8 JSON frame") from exc


async def write_frame(
    writer: asyncio.StreamWriter,
    message: JsonValue,
    *,
    max_frame: int = MAX_FRAME,
) -> None:
    """Write one complete frame and apply stream backpressure."""

    writer.write(encode_frame(message, max_frame=max_frame))
    await writer.drain()


def load_or_create_auth_token(path: str | os.PathLike[str]) -> str:
    """Load a token or atomically create a new 256-bit runtime token.

    The caller chooses an ignored runtime path.  Parent directories must already
    exist so this helper cannot accidentally create runtime state elsewhere.
    """

    token_path = Path(path)
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(token_path, flags, 0o600)
        except FileExistsError:
            token = token_path.read_text(encoding="utf-8").strip()
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(token + "\n")
    if not token:
        raise AuthenticationError(f"Empty IPC token file: {token_path}")
    return token


class MediaPathValidator:
    """Resolve media paths while preventing traversal and symlink escapes."""

    def __init__(self, runtime_root: str | os.PathLike[str]) -> None:
        root = Path(runtime_root).expanduser()
        try:
            self.runtime_root = root.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise MediaPathError(f"Runtime media root does not exist: {root}") from exc
        if not self.runtime_root.is_dir():
            raise MediaPathError(f"Runtime media root is not a directory: {root}")

    def validate(
        self,
        path: str | os.PathLike[str],
        *,
        must_exist: bool = True,
        allow_directory: bool = False,
    ) -> Path:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("Media path must be a string or PathLike")
        if isinstance(path, str) and not path.strip():
            raise MediaPathError("Media path cannot be empty")

        untrusted = Path(path).expanduser()
        candidate = untrusted if untrusted.is_absolute() else self.runtime_root / untrusted
        try:
            resolved = candidate.resolve(strict=must_exist)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise MediaPathError(f"Invalid media path: {path}") from exc

        try:
            resolved.relative_to(self.runtime_root)
        except ValueError as exc:
            raise MediaPathError("Media path escapes the runtime root") from exc
        if resolved == self.runtime_root and not allow_directory:
            raise MediaPathError("Runtime root itself is not a media file")
        if must_exist:
            if resolved.is_dir() and not allow_directory:
                raise MediaPathError("Media path points to a directory")
            if not resolved.is_file() and not allow_directory:
                raise MediaPathError("Media path is not a regular file")
        return resolved


def validate_media_path(
    path: str | os.PathLike[str],
    runtime_root: str | os.PathLike[str],
    *,
    must_exist: bool = True,
    allow_directory: bool = False,
) -> Path:
    """One-shot convenience wrapper around :class:`MediaPathValidator`."""

    return MediaPathValidator(runtime_root).validate(
        path,
        must_exist=must_exist,
        allow_directory=allow_directory,
    )


@dataclass
class _IdempotencyEntry:
    fingerprint: str
    future: asyncio.Future[tuple[bool, Any]]
    expires_at: float


class IdempotencyCache:
    """Async duplicate suppression with conflict detection and a bounded TTL."""

    def __init__(self, *, ttl: float = 300.0, max_entries: int = 2048) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.ttl = float(ttl)
        self.max_entries = max_entries
        self._entries: dict[str, _IdempotencyEntry] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        key: str,
        fingerprint: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        if not isinstance(key, str) or not key:
            raise ValueError("Idempotency key must be a non-empty string")
        loop = asyncio.get_running_loop()
        owner = False
        async with self._lock:
            self._prune(loop.time())
            entry = self._entries.get(key)
            if entry is not None:
                if not hmac.compare_digest(entry.fingerprint, fingerprint):
                    raise IdempotencyConflictError(key)
            else:
                entry = _IdempotencyEntry(
                    fingerprint=fingerprint,
                    future=loop.create_future(),
                    expires_at=math.inf,
                )
                self._entries[key] = entry
                owner = True

        if not owner:
            succeeded, value = await asyncio.shield(entry.future)
            if succeeded:
                return value
            raise value

        try:
            value = await operation()
        except BaseException as exc:
            async with self._lock:
                if self._entries.get(key) is entry:
                    self._entries.pop(key, None)
                if not entry.future.done():
                    entry.future.set_result((False, exc))
            raise
        else:
            async with self._lock:
                entry.expires_at = loop.time() + self.ttl
                if not entry.future.done():
                    entry.future.set_result((True, value))
                self._trim()
            return value

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    def _prune(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.future.done() and entry.expires_at <= now
        ]
        for key in expired:
            self._entries.pop(key, None)

    def _trim(self) -> None:
        overflow = len(self._entries) - self.max_entries
        if overflow <= 0:
            return
        completed = sorted(
            (
                (entry.expires_at, key)
                for key, entry in self._entries.items()
                if entry.future.done()
            ),
            key=lambda item: item[0],
        )
        for _, key in completed[:overflow]:
            self._entries.pop(key, None)


CancelCallback = Callable[[], Awaitable[None] | None]


@dataclass
class RequestContext:
    """Metadata and cooperative cancellation hooks for an inbound call."""

    peer: "JsonRpcPeer"
    request_id: RequestId
    method: str
    idempotency_key: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    _cancel_callbacks: list[CancelCallback] = field(default_factory=list, repr=False)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def add_cancel_callback(self, callback: CancelCallback) -> None:
        if not callable(callback):
            raise TypeError("Cancellation callback must be callable")
        if self.cancelled:
            result = callback()
            if inspect.isawaitable(result):
                asyncio.create_task(result)
            return
        self._cancel_callbacks.append(callback)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError

    def _signal_cancel(self) -> None:
        if self.cancelled:
            return
        self.cancel_event.set()
        callbacks, self._cancel_callbacks = self._cancel_callbacks, []
        for callback in callbacks:
            try:
                result = callback()
                if inspect.isawaitable(result):
                    asyncio.create_task(result)
            except Exception:
                # Cancellation must continue even if an optional observer fails.
                pass


Handler = Callable[[JsonValue, RequestContext], Awaitable[JsonValue] | JsonValue]


def _valid_request_id(value: Any) -> bool:
    if value is None or isinstance(value, str):
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return not isinstance(value, float) or math.isfinite(value)


def _fingerprint(method: str, params: JsonValue) -> str:
    canonical = json.dumps(
        {"method": method, "params": params},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class JsonRpcPeer:
    """One authenticated, bidirectional JSON-RPC connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        handlers: Mapping[str, Handler] | None = None,
        request_timeout: float = 10.0,
        max_frame: int = MAX_FRAME,
        idempotency_cache: IdempotencyCache | None = None,
        name: str = "peer",
        on_close: Callable[["JsonRpcPeer"], None] | None = None,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        _validate_frame_limit(max_frame)
        self.reader = reader
        self.writer = writer
        self.request_timeout = float(request_timeout)
        self.max_frame = max_frame
        self.name = name
        self._handlers = dict(handlers or {})
        self._idempotency_cache = idempotency_cache or IdempotencyCache()
        self._on_close = on_close
        self._send_lock = asyncio.Lock()
        self._pending: dict[RequestId, asyncio.Future[JsonValue]] = {}
        self._incoming: dict[RequestId, tuple[asyncio.Task[None], RequestContext]] = {}
        self._notifications: set[asyncio.Task[None]] = set()
        self._next_id = 1
        self._read_task: asyncio.Task[None] | None = None
        self._closed = asyncio.Event()
        self._terminated = False

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    @property
    def peername(self) -> Any:
        return self.writer.get_extra_info("peername")

    def start(self) -> "JsonRpcPeer":
        if self._read_task is not None:
            return self
        if self.closed:
            raise ConnectionClosedError("Cannot start a closed JSON-RPC peer")
        self._read_task = asyncio.create_task(
            self._read_loop(), name=f"json-rpc-reader:{self.name}"
        )
        return self

    def register_method(self, method: str, handler: Handler) -> None:
        if not isinstance(method, str) or not method:
            raise ValueError("Method name must be a non-empty string")
        if not callable(handler):
            raise TypeError("Handler must be callable")
        self._handlers[method] = handler

    def unregister_method(self, method: str) -> None:
        self._handlers.pop(method, None)

    async def request(
        self,
        method: str,
        params: JsonValue = None,
        *,
        timeout: float | None = None,
        idempotency_key: str | None = None,
    ) -> JsonValue:
        self._validate_outgoing(method, params, idempotency_key)
        effective_timeout = self.request_timeout if timeout is None else timeout
        if effective_timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.closed:
            raise ConnectionClosedError(f"JSON-RPC {self.name} is closed")

        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        message: dict[str, JsonValue] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "method": method,
            "params": params,
        }
        if idempotency_key is not None:
            message["idempotency_key"] = idempotency_key

        try:
            await self._send(message)
            return await asyncio.wait_for(asyncio.shield(future), effective_timeout)
        except TimeoutError:
            await self._best_effort_cancel(request_id)
            raise
        except asyncio.CancelledError:
            await self._best_effort_cancel(request_id)
            raise
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                # Mark a transport exception as observed if sending failed before
                # this request got as far as awaiting its response future.
                future.exception()

    async def notify(
        self,
        method: str,
        params: JsonValue = None,
        *,
        idempotency_key: str | None = None,
    ) -> None:
        self._validate_outgoing(method, params, idempotency_key)
        message: dict[str, JsonValue] = {
            "jsonrpc": JSONRPC_VERSION,
            "method": method,
            "params": params,
        }
        if idempotency_key is not None:
            message["idempotency_key"] = idempotency_key
        await self._send(message)

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def close(self) -> None:
        tasks: set[asyncio.Task[Any]] = set(self._notifications)
        tasks.update(task for task, _ in self._incoming.values())
        if self._read_task is not None:
            tasks.add(self._read_task)
        self._terminate(ConnectionClosedError(f"JSON-RPC {self.name} closed"))
        current_task = asyncio.current_task()
        wait_for = [task for task in tasks if task is not current_task]
        for task in wait_for:
            if not task.done():
                task.cancel()
        if wait_for:
            await asyncio.gather(*wait_for, return_exceptions=True)
        try:
            await self.writer.wait_closed()
        except (ConnectionError, OSError):
            pass

    async def __aenter__(self) -> "JsonRpcPeer":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    @staticmethod
    def _validate_outgoing(
        method: str, params: JsonValue, idempotency_key: str | None
    ) -> None:
        if not isinstance(method, str) or not method:
            raise ValueError("Method name must be a non-empty string")
        if params is not None and not isinstance(params, (dict, list)):
            raise TypeError("JSON-RPC params must be an object, array, or None")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > 256
        ):
            raise ValueError("Idempotency key must contain 1 to 256 characters")

    async def _send(self, message: Mapping[str, JsonValue]) -> None:
        if self.closed:
            raise ConnectionClosedError(f"JSON-RPC {self.name} is closed")
        async with self._send_lock:
            try:
                await write_frame(self.writer, message, max_frame=self.max_frame)
            except (ConnectionError, OSError) as exc:
                self._terminate(ConnectionClosedError(str(exc)))
                raise ConnectionClosedError(str(exc)) from exc

    async def _best_effort_cancel(self, request_id: RequestId) -> None:
        if self.closed:
            return
        try:
            await self.notify(CANCEL_METHOD, {"id": request_id})
        except (ConnectionError, JsonRpcProtocolError):
            pass

    async def _read_loop(self) -> None:
        failure: BaseException = ConnectionClosedError(
            f"JSON-RPC {self.name} closed by remote peer"
        )
        try:
            while True:
                message = await read_frame(self.reader, max_frame=self.max_frame)
                await self._route_message(message)
        except asyncio.CancelledError:
            failure = ConnectionClosedError(f"JSON-RPC {self.name} reader stopped")
        except asyncio.IncompleteReadError:
            pass
        except (ConnectionError, OSError, JsonRpcProtocolError) as exc:
            failure = exc
        finally:
            self._terminate(failure)

    async def _route_message(self, message: JsonValue) -> None:
        if not isinstance(message, dict) or message.get("jsonrpc") != JSONRPC_VERSION:
            request_id = message.get("id") if isinstance(message, dict) else None
            await self._send_error(
                request_id,
                JsonRpcError(INVALID_REQUEST, "Invalid JSON-RPC request"),
            )
            return

        if "method" in message:
            await self._route_call(message)
            return
        self._route_response(message)

    async def _route_call(self, message: dict[str, JsonValue]) -> None:
        method = message.get("method")
        has_id = "id" in message
        request_id = message.get("id")
        params = message.get("params")
        idempotency_key = message.get("idempotency_key")
        if (
            not isinstance(method, str)
            or not method
            or (has_id and not _valid_request_id(request_id))
            or (params is not None and not isinstance(params, (dict, list)))
            or (
                idempotency_key is not None
                and (
                    not isinstance(idempotency_key, str)
                    or not idempotency_key
                    or len(idempotency_key) > 256
                )
            )
        ):
            if has_id:
                await self._send_error(
                    request_id,
                    JsonRpcError(INVALID_REQUEST, "Invalid JSON-RPC request"),
                )
            return

        if method == CANCEL_METHOD:
            self._handle_cancel(params)
            if has_id:
                await self._send_result(request_id, None)
            return

        context = RequestContext(
            peer=self,
            request_id=request_id if has_id else None,
            method=method,
            idempotency_key=idempotency_key,
        )
        if has_id:
            if request_id in self._incoming:
                await self._send_error(
                    request_id,
                    JsonRpcError(INVALID_REQUEST, "Duplicate active request id"),
                )
                return
            task = asyncio.create_task(
                self._execute_call(method, params, context, True),
                name=f"json-rpc-call:{method}:{request_id}",
            )
            self._incoming[request_id] = (task, context)

            def remove_incoming(
                completed: asyncio.Task[None], *, call_id: RequestId = request_id
            ) -> None:
                current = self._incoming.get(call_id)
                if current is not None and current[0] is completed:
                    self._incoming.pop(call_id, None)
                self._consume_task_result(completed)

            task.add_done_callback(remove_incoming)
        else:
            task = asyncio.create_task(
                self._execute_call(method, params, context, False),
                name=f"json-rpc-notification:{method}",
            )
            self._notifications.add(task)
            task.add_done_callback(self._notification_finished)

    def _route_response(self, message: dict[str, JsonValue]) -> None:
        if "id" not in message or not _valid_request_id(message.get("id")):
            return
        request_id = message.get("id")
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        has_result = "result" in message
        has_error = "error" in message
        if has_result == has_error:
            future.set_exception(JsonRpcProtocolError("Malformed JSON-RPC response"))
            return
        if has_result:
            future.set_result(message.get("result"))
            return
        error = message.get("error")
        if not isinstance(error, dict):
            future.set_exception(JsonRpcProtocolError("Malformed JSON-RPC error"))
            return
        code = error.get("code")
        description = error.get("message")
        if isinstance(code, bool) or not isinstance(code, int) or not isinstance(
            description, str
        ):
            future.set_exception(JsonRpcProtocolError("Malformed JSON-RPC error"))
            return
        future.set_exception(JsonRpcError(code, description, error.get("data")))

    def _handle_cancel(self, params: JsonValue) -> None:
        if not isinstance(params, dict) or "id" not in params:
            return
        incoming = self._incoming.get(params.get("id"))
        if incoming is None:
            return
        task, context = incoming
        context._signal_cancel()
        task.cancel()

    def _notification_finished(self, task: asyncio.Task[None]) -> None:
        self._notifications.discard(task)
        self._consume_task_result(task)

    @staticmethod
    def _consume_task_result(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    async def _execute_call(
        self,
        method: str,
        params: JsonValue,
        context: RequestContext,
        respond: bool,
    ) -> None:
        try:
            handler = self._handlers.get(method)
            if handler is None:
                raise JsonRpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

            async def invoke() -> JsonValue:
                result = handler(params, context)
                if inspect.isawaitable(result):
                    return await result
                return result

            if context.idempotency_key is not None:
                result = await self._idempotency_cache.execute(
                    context.idempotency_key,
                    _fingerprint(method, params),
                    invoke,
                )
            else:
                result = await invoke()
            if respond and not self.closed:
                await self._send_result(context.request_id, result)
        except asyncio.CancelledError:
            context._signal_cancel()
            if respond and not self.closed:
                try:
                    await self._send_error(
                        context.request_id,
                        JsonRpcError(REQUEST_CANCELLED, "Request cancelled"),
                    )
                except (ConnectionError, JsonRpcProtocolError):
                    pass
        except JsonRpcError as exc:
            if respond and not self.closed:
                await self._send_error(context.request_id, exc)
        except Exception:
            if respond and not self.closed:
                await self._send_error(
                    context.request_id,
                    JsonRpcError(INTERNAL_ERROR, "Internal JSON-RPC error"),
                )

    async def _send_result(self, request_id: RequestId, result: JsonValue) -> None:
        await self._send(
            {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}
        )

    async def _send_error(self, request_id: RequestId, error: JsonRpcError) -> None:
        await self._send(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "error": error.as_dict(),
            }
        )

    def _terminate(self, failure: BaseException) -> None:
        if self._terminated:
            return
        self._terminated = True
        self.writer.close()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(ConnectionClosedError(str(failure)))
        for task, context in self._incoming.values():
            context._signal_cancel()
            task.cancel()
        for task in self._notifications:
            task.cancel()
        self._closed.set()
        if self._on_close is not None:
            try:
                self._on_close(self)
            except Exception:
                pass


class JsonRpcServer:
    """Authenticated JSON-RPC server bound strictly to the loopback interface."""

    def __init__(
        self,
        token: str,
        *,
        port: int = 0,
        handlers: Mapping[str, Handler] | None = None,
        request_timeout: float = 10.0,
        handshake_timeout: float = 5.0,
        max_frame: int = MAX_FRAME,
        idempotency_cache: IdempotencyCache | None = None,
    ) -> None:
        if not isinstance(token, str) or not token:
            raise ValueError("Authentication token must be a non-empty string")
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if handshake_timeout <= 0:
            raise ValueError("handshake_timeout must be positive")
        _validate_frame_limit(max_frame)
        self.token = token
        self.port = port
        self.request_timeout = request_timeout
        self.handshake_timeout = handshake_timeout
        self.max_frame = max_frame
        self._handlers = dict(handlers or {})
        self._idempotency_cache = idempotency_cache or IdempotencyCache()
        self._server: asyncio.AbstractServer | None = None
        self._peers: set[JsonRpcPeer] = set()
        self._new_peers: asyncio.Queue[JsonRpcPeer] = asyncio.Queue()
        self._accept_tasks: set[asyncio.Task[None]] = set()
        self._preauth_writers: set[asyncio.StreamWriter] = set()

    @property
    def peers(self) -> tuple[JsonRpcPeer, ...]:
        return tuple(peer for peer in self._peers if not peer.closed)

    @property
    def bound_port(self) -> int:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("JSON-RPC server has not been started")
        return int(self._server.sockets[0].getsockname()[1])

    async def start(self) -> "JsonRpcServer":
        if self._server is not None:
            return self
        self._server = await asyncio.start_server(
            self._client_connected,
            host="127.0.0.1",
            port=self.port,
            start_serving=True,
        )
        return self

    def register_method(self, method: str, handler: Handler) -> None:
        if not isinstance(method, str) or not method:
            raise ValueError("Method name must be a non-empty string")
        if not callable(handler):
            raise TypeError("Handler must be callable")
        self._handlers[method] = handler
        for peer in self._peers:
            peer.register_method(method, handler)

    async def wait_for_peer(self, *, timeout: float | None = None) -> JsonRpcPeer:
        try:
            if timeout is None:
                return await self._new_peers.get()
            return await asyncio.wait_for(self._new_peers.get(), timeout)
        except TimeoutError:
            raise TimeoutError("Timed out waiting for authenticated JSON-RPC peer") from None

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()

        accept_tasks = tuple(self._accept_tasks)
        for task in accept_tasks:
            task.cancel()
        for writer in tuple(self._preauth_writers):
            writer.close()
        if accept_tasks:
            await asyncio.gather(*accept_tasks, return_exceptions=True)
        await asyncio.gather(
            *(peer.close() for peer in tuple(self._peers)),
            return_exceptions=True,
        )

    async def __aenter__(self) -> "JsonRpcServer":
        return await self.start()

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    def _client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._preauth_writers.add(writer)
        task = asyncio.create_task(
            self._authenticate(reader, writer), name="json-rpc-handshake"
        )
        self._accept_tasks.add(task)
        task.add_done_callback(self._accept_tasks.discard)

    async def _authenticate(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        handed_off = False
        try:
            message = await asyncio.wait_for(
                read_frame(reader, max_frame=self.max_frame),
                self.handshake_timeout,
            )
            request_id = message.get("id") if isinstance(message, dict) else None
            params = message.get("params") if isinstance(message, dict) else None
            candidate = params.get("token") if isinstance(params, dict) else None
            valid = (
                isinstance(message, dict)
                and message.get("jsonrpc") == JSONRPC_VERSION
                and message.get("method") == HANDSHAKE_METHOD
                and "id" in message
                and _valid_request_id(request_id)
                and isinstance(candidate, str)
                and hmac.compare_digest(candidate, self.token)
            )
            if not valid:
                await write_frame(
                    writer,
                    {
                        "jsonrpc": JSONRPC_VERSION,
                        "id": request_id,
                        "error": {
                            "code": AUTHENTICATION_FAILED,
                            "message": "IPC authentication failed",
                        },
                    },
                    max_frame=self.max_frame,
                )
                return

            await write_frame(
                writer,
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": request_id,
                    "result": {"authenticated": True},
                },
                max_frame=self.max_frame,
            )
            peer = JsonRpcPeer(
                reader,
                writer,
                handlers=self._handlers,
                request_timeout=self.request_timeout,
                max_frame=self.max_frame,
                idempotency_cache=self._idempotency_cache,
                name="server-peer",
                on_close=self._peers.discard,
            )
            self._peers.add(peer)
            self._new_peers.put_nowait(peer)
            handed_off = True
            peer.start()
        except (
            TimeoutError,
            asyncio.IncompleteReadError,
            ConnectionError,
            OSError,
            JsonRpcProtocolError,
        ):
            pass
        finally:
            self._preauth_writers.discard(writer)
            if not handed_off:
                writer.close()
                try:
                    await writer.wait_closed()
                except (ConnectionError, OSError):
                    pass


async def connect_json_rpc(
    host: str,
    port: int,
    token: str,
    *,
    handlers: Mapping[str, Handler] | None = None,
    request_timeout: float = 10.0,
    handshake_timeout: float = 5.0,
    max_frame: int = MAX_FRAME,
    idempotency_cache: IdempotencyCache | None = None,
    name: str = "client-peer",
) -> JsonRpcPeer:
    """Connect to a local Milana IPC server and complete authentication."""

    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("Milana IPC clients may only connect to localhost")
    if not isinstance(token, str) or not token:
        raise ValueError("Authentication token must be a non-empty string")
    _validate_frame_limit(max_frame)
    reader, writer = await asyncio.open_connection(host, port)
    handshake_id = secrets.randbits(63)
    try:
        await write_frame(
            writer,
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": handshake_id,
                "method": HANDSHAKE_METHOD,
                "params": {"token": token},
            },
            max_frame=max_frame,
        )
        response = await asyncio.wait_for(
            read_frame(reader, max_frame=max_frame), handshake_timeout
        )
        authenticated = (
            isinstance(response, dict)
            and response.get("jsonrpc") == JSONRPC_VERSION
            and response.get("id") == handshake_id
            and response.get("result") == {"authenticated": True}
        )
        if not authenticated:
            raise AuthenticationError("IPC authentication failed")
    except BaseException:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionError, OSError):
            pass
        raise

    return JsonRpcPeer(
        reader,
        writer,
        handlers=handlers,
        request_timeout=request_timeout,
        max_frame=max_frame,
        idempotency_cache=idempotency_cache,
        name=name,
    ).start()


__all__ = [
    "AUTHENTICATION_FAILED",
    "CANCEL_METHOD",
    "ConnectionClosedError",
    "FrameTooLargeError",
    "HANDSHAKE_METHOD",
    "IdempotencyCache",
    "IdempotencyConflictError",
    "JsonRpcError",
    "JsonRpcPeer",
    "JsonRpcProtocolError",
    "JsonRpcServer",
    "MAX_FRAME",
    "MediaPathError",
    "MediaPathValidator",
    "REQUEST_CANCELLED",
    "RequestContext",
    "AuthenticationError",
    "connect_json_rpc",
    "encode_frame",
    "load_or_create_auth_token",
    "read_frame",
    "validate_media_path",
    "write_frame",
]
