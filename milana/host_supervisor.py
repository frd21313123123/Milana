"""Lifecycle supervision for Milana's dependent Telegram skill host."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from milana_ipc import ConnectionClosedError, JsonRpcPeer, JsonRpcServer


RESTART_BACKOFF_SECONDS = (1, 2, 5, 10, 30, 60)
STABLE_CONNECTION_SECONDS = 60.0


def restart_delay(attempt: int) -> int:
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise ValueError("attempt must be a non-negative integer")
    return RESTART_BACKOFF_SECONDS[min(attempt, len(RESTART_BACKOFF_SECONDS) - 1)]


ProcessFactory = Callable[[Sequence[str]], Awaitable[Any]]


async def _default_process_factory(command: Sequence[str]) -> Any:
    return await asyncio.create_subprocess_exec(*command)


class SkillHostSupervisor:
    """Start, reconnect and restart one child host owned by MilanaService."""

    def __init__(
        self,
        server: JsonRpcServer,
        *,
        token_file: str | Path,
        runtime_dir: str | Path,
        host_script: str | Path | None = None,
        python_executable: str | Path | None = None,
        dev_mode: bool = False,
        process_factory: ProcessFactory = _default_process_factory,
    ) -> None:
        self.server = server
        self.token_file = Path(token_file).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()
        self.host_script = Path(
            host_script
            or Path(__file__).resolve().parent.parent / "telegram_skill_host.py"
        ).resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.dev_mode = bool(dev_mode)
        self._process_factory = process_factory
        self._peer: JsonRpcPeer | None = None
        self._connected = asyncio.Event()
        self._stopping = asyncio.Event()
        self._spawn_requested = asyncio.Event()
        self._monitor_task: asyncio.Task[None] | None = None
        self._accept_task: asyncio.Task[None] | None = None
        self._stable_reset_task: asyncio.Task[None] | None = None
        self._process: Any | None = None
        self._restart_attempt = 0
        self.last_exit_code: int | None = None
        self.last_error: str | None = None

    @property
    def peer(self) -> JsonRpcPeer | None:
        peer = self._peer
        return peer if peer is not None and not peer.closed else None

    @property
    def connected(self) -> bool:
        return self.peer is not None

    @property
    def process_running(self) -> bool:
        process = self._process
        return process is not None and getattr(process, "returncode", None) is None

    @property
    def command(self) -> tuple[str, ...]:
        command = [
            self.python_executable,
            "-u",
            str(self.host_script),
            "--port",
            str(self.server.bound_port),
            "--token-file",
            str(self.token_file),
            "--runtime-dir",
            str(self.runtime_dir),
        ]
        if self.dev_mode:
            command.append("--dev-chat")
        return tuple(command)

    async def start(self) -> None:
        if self._monitor_task is not None and not self._monitor_task.done():
            return
        # Accessing bound_port here gives a clear error when service ordering is wrong.
        _ = self.server.bound_port
        self._stopping.clear()
        self._spawn_requested.set()
        self._accept_task = asyncio.create_task(
            self._accept_loop(), name="telegram-host-accept"
        )
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="telegram-host-supervisor"
        )

    async def ensure_running(self) -> None:
        await self.start()
        if not self.process_running:
            self._spawn_requested.set()

    def attach_peer(self, peer: JsonRpcPeer) -> None:
        if not isinstance(peer, JsonRpcPeer):
            raise TypeError("peer must be JsonRpcPeer")
        old = self.peer
        self._peer = peer
        self._connected.set()
        self.last_error = None
        if old is not None and old is not peer:
            asyncio.create_task(old.close())

        if self._stable_reset_task is not None:
            self._stable_reset_task.cancel()

        async def reset_after_stable_connection() -> None:
            try:
                await asyncio.sleep(STABLE_CONNECTION_SECONDS)
                if self._peer is peer and not peer.closed and self.process_running:
                    self._restart_attempt = 0
            except asyncio.CancelledError:
                pass

        self._stable_reset_task = asyncio.create_task(
            reset_after_stable_connection(),
            name="telegram-host-stable-reset",
        )

        async def watch() -> None:
            await peer.wait_closed()
            if self._peer is peer:
                self._peer = None
                self._connected.clear()

        asyncio.create_task(watch(), name="telegram-host-peer-watch")

    async def _accept_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                peer = await self.server.wait_for_peer(timeout=1.0)
            except TimeoutError:
                continue
            self.attach_peer(peer)

    async def _monitor_loop(self) -> None:
        while not self._stopping.is_set():
            await self._spawn_requested.wait()
            self._spawn_requested.clear()
            if self._stopping.is_set():
                return
            if self.process_running:
                continue
            try:
                self._process = await self._process_factory(self.command)
                exit_code = await self._process.wait()
                self.last_exit_code = int(exit_code)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - supervisor must stay alive
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._process = None
            if self._stopping.is_set():
                return
            delay = restart_delay(self._restart_attempt)
            self._restart_attempt += 1
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)
                return
            except TimeoutError:
                self._spawn_requested.set()

    async def request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        await self.ensure_running()
        effective_timeout = 20.0 if timeout is None else float(timeout)
        if effective_timeout <= 0:
            raise ValueError("timeout must be positive")
        try:
            await asyncio.wait_for(self._connected.wait(), effective_timeout)
        except TimeoutError:
            raise ConnectionClosedError(
                "Telegram skill host did not connect before the request timeout"
            ) from None
        peer = self.peer
        if peer is None:
            raise ConnectionClosedError("Telegram skill host disconnected")
        try:
            return await peer.request(
                method,
                dict(params),
                timeout=effective_timeout,
                idempotency_key=idempotency_key,
            )
        except ConnectionClosedError:
            if self._peer is peer:
                self._peer = None
                self._connected.clear()
            self._spawn_requested.set()
            raise

    async def stop(self) -> None:
        self._stopping.set()
        self._spawn_requested.set()
        peer, self._peer = self._peer, None
        self._connected.clear()
        if peer is not None and not peer.closed:
            await peer.close()
        process = self._process
        if process is not None and getattr(process, "returncode", None) is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 5.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        tasks = [task for task in (self._monitor_task, self._accept_task) if task]
        if self._stable_reset_task is not None:
            self._stable_reset_task.cancel()
            tasks.append(self._stable_reset_task)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._monitor_task = None
        self._accept_task = None
        self._stable_reset_task = None
        self._process = None

    def status(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "process_running": self.process_running,
            "restart_attempt": self._restart_attempt,
            "last_exit_code": self.last_exit_code,
            "last_error": self.last_error,
        }


__all__ = [
    "RESTART_BACKOFF_SECONDS",
    "STABLE_CONNECTION_SECONDS",
    "SkillHostSupervisor",
    "restart_delay",
]
