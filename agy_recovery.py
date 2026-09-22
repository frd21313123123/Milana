"""Small, fail-closed helpers for recovering the Antigravity unlocker.

The Telegram channel is treated as untrusted input.  A key is accepted only
when it is an exact-looking unlocker key and appears next to the installed
unlocker version marker.  The key is kept in memory only and is never passed
as a command-line argument or written to a file.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import select
import shutil
import socket
import stat
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_UNLOCKER_PATH = Path("/root/.local/share/agunlocker/ag_unlocker")
DEFAULT_UNLOCKER_SETTINGS_PATH = Path(
    "/root/.local/share/agunlocker/settings.json"
)
KEY_RE = re.compile(r"(?<![A-Z0-9])[A-Z0-9]{24}(?![A-Z0-9])")
VERSION_RE = re.compile(
    r"\bversion\s*:\s*([0-9]+(?:[._][0-9]+)+)\b", re.I
)
UNLOCKER_MESSAGE_LINK_RE = re.compile(
    r"https?://t\.me/nova_txt/(?:\d+/)?(\d+)", re.I
)
AGY_PROXY_VARIABLE_ORIGINAL = b"https_proxy"
AGY_PROXY_VARIABLE_PATCHED = b"AG_LS_PROXY"
AGY_ELIGIBILITY_ORIGINAL = b"ineligible"
AGY_ELIGIBILITY_PATCHED = b"inexigible"


def is_unlocker_recoverable_error(error: BaseException) -> bool:
    """Return true only for region/eligibility failures the unlocker fixes."""

    text = str(error).casefold()
    if not text:
        return False
    markers = (
        "eligibility",
        "not eligible",
        "location is not supported",
        "unsupported location",
        "сетевом регионе",
        "сетевого региона",
        "регионе",
        "регион",
    )
    return any(marker in text for marker in markers)


def installed_unlocker_version(binary: Path) -> str | None:
    """Read the version banner without starting the interactive TUI."""

    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = VERSION_RE.search(f"{result.stdout}\n{result.stderr}")
    return match.group(1) if match else None


def extract_unlocker_key(
    messages: Iterable[Mapping[str, Any]], version: str | None
) -> str | None:
    """Extract a key only from a nearby version-labelled Telegram post."""

    if not version:
        return None
    expected = re.compile(
        rf"\bv\s*{re.escape(version)}(?:\s*\+)?\b", re.IGNORECASE
    )
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        text = message.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    for index, text in enumerate(texts):
        start = max(0, index - 2)
        end = min(len(texts), index + 3)
        window = "\n".join(texts[start:end])
        if "ключ" not in window.casefold() or not expected.search(window):
            continue
        match = KEY_RE.search(window.upper())
        if match:
            return match.group(0)
    return None


def extract_unlocker_message_ids(
    messages: Iterable[Mapping[str, Any]], *, limit: int = 20
) -> tuple[int, ...]:
    """Return exact message IDs referenced as the pinned unlocker key."""

    found: list[int] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        text = message.get("text")
        if not isinstance(text, str):
            continue
        for match in UNLOCKER_MESSAGE_LINK_RE.finditer(text):
            message_id = int(match.group(1))
            if message_id > 0 and message_id not in found:
                found.append(message_id)
                if len(found) >= limit:
                    return tuple(found)
    return tuple(found)


def _user_systemd_environment() -> dict[str, str]:
    environment = os.environ.copy()
    runtime_dir = Path("/run/user/0")
    if runtime_dir.is_dir():
        environment.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))
    return environment


def prefer_unlocker_relay(
    settings_path: Path = DEFAULT_UNLOCKER_SETTINGS_PATH,
) -> bool:
    """Persist the upstream relay route used when built-in exits are denied."""

    try:
        current: dict[str, Any] = {}
        if settings_path.is_file():
            parsed = json.loads(settings_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                return False
            current = parsed
        current.update(
            {
                "local_proxy": True,
                "builtin_exits": False,
                "own_proxy_enabled": False,
            }
        )
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = settings_path.with_name(
            f".{settings_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(current, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o600)
        temporary.replace(settings_path)
        return True
    except (OSError, TypeError, ValueError):
        return False


def ensure_agy_proxy_patch(executable: Path | None = None) -> bool:
    """Apply both same-length upstream Antigravity edits to the agy binary."""

    if executable is None:
        found = shutil.which("agy")
        if not found:
            return False
        executable = Path(found)
    try:
        executable = executable.resolve(strict=True)
        data = executable.read_bytes()
        if (
            AGY_PROXY_VARIABLE_ORIGINAL not in data
            and AGY_ELIGIBILITY_ORIGINAL not in data
        ):
            return (
                AGY_PROXY_VARIABLE_PATCHED in data
                and AGY_ELIGIBILITY_PATCHED in data
            )

        updated = data.replace(
            AGY_PROXY_VARIABLE_ORIGINAL, AGY_PROXY_VARIABLE_PATCHED
        )
        updated = updated.replace(
            AGY_ELIGIBILITY_ORIGINAL, AGY_ELIGIBILITY_PATCHED
        )
        if updated == data:
            return False
        backup = executable.with_name(f"{executable.name}.before-ag-ls-proxy")
        if not backup.exists():
            shutil.copy2(executable, backup)
        temporary = executable.with_name(
            f".{executable.name}.{os.getpid()}.proxy-patch"
        )
        temporary.write_bytes(updated)
        temporary.chmod(stat.S_IMODE(executable.stat().st_mode))
        temporary.replace(executable)
        installed = executable.read_bytes()
        return (
            AGY_PROXY_VARIABLE_PATCHED in installed
            and AGY_ELIGIBILITY_PATCHED in installed
        )
    except (OSError, ValueError):
        return False


def _run_unlocker_tui(binary: Path, key: str) -> bool:
    """Feed the key to the unlocker's TUI through a private pseudo-terminal."""

    if os.name == "nt" or not binary.is_file() or not os.access(binary, os.X_OK):
        return False
    try:
        import fcntl
        import pty
        import termios

        master, slave = pty.openpty()
        # systemd services have neither TERM nor a terminal window size.  The
        # ratatui/crossterm UI needs both before it starts accepting keys.
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        environment = os.environ.copy()
        environment.setdefault("TERM", "xterm-256color")
        process = subprocess.Popen(
            [str(binary), "--tui"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=environment,
            close_fds=True,
            start_new_session=True,
        )
    except (ImportError, OSError, subprocess.SubprocessError):
        return False
    finally:
        try:
            os.close(slave)
        except (UnboundLocalError, OSError):
            pass

    try:
        # Drain terminal output so the TUI does not block on a full PTY buffer.
        deadline = time.monotonic() + 35.0
        started_at = time.monotonic()
        sent_key = False
        sent_enable = False
        sent_quit = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            ready, _, _ = select.select([master], [], [], 0.25)
            if ready:
                try:
                    os.read(master, 8192)
                except OSError:
                    pass
            elapsed = time.monotonic() - started_at
            if not sent_key and elapsed >= 0.5:
                os.write(master, key.encode("ascii") + b"\n")
                sent_key = True
            elif sent_key and not sent_enable and elapsed >= 2.5:
                os.write(master, b"a")
                sent_enable = True
            elif sent_enable and not sent_quit and elapsed >= 20.0:
                os.write(master, b"q")
                sent_quit = True
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return process.returncode == 0
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass
        return False
    finally:
        try:
            os.close(master)
        except OSError:
            pass


def activate_unlocker(binary: Path, key: str) -> bool:
    """Activate the unlocker and make its local proxy persistent."""

    if not _run_unlocker_tui(binary, key):
        return False
    if not prefer_unlocker_relay():
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "enable", "--now", "ag-unlocker-proxy"],
            env=_user_systemd_environment(),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def restart_unlocker_proxy() -> bool:
    """Reset the Linux proxy route and wait until its local listener returns."""

    if os.name == "nt":
        return False
    executable_value = os.environ.get("AGY_EXECUTABLE")
    executable = Path(executable_value).expanduser() if executable_value else None
    if not ensure_agy_proxy_patch(executable):
        return False
    settings_path = Path(
        os.environ.get(
            "AG_UNLOCKER_SETTINGS", str(DEFAULT_UNLOCKER_SETTINGS_PATH)
        )
    ).expanduser()
    if not prefer_unlocker_relay(settings_path):
        return False
    try:
        result = subprocess.run(
            ["systemctl", "--user", "restart", "ag-unlocker-proxy"],
            env=_user_systemd_environment(),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 53129), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


async def activate_unlocker_async(binary: Path, key: str) -> bool:
    return await asyncio.to_thread(activate_unlocker, binary, key)


async def restart_unlocker_proxy_async() -> bool:
    return await asyncio.to_thread(restart_unlocker_proxy)


__all__ = [
    "DEFAULT_UNLOCKER_PATH",
    "DEFAULT_UNLOCKER_SETTINGS_PATH",
    "activate_unlocker",
    "activate_unlocker_async",
    "extract_unlocker_key",
    "extract_unlocker_message_ids",
    "ensure_agy_proxy_patch",
    "installed_unlocker_version",
    "is_unlocker_recoverable_error",
    "prefer_unlocker_relay",
    "restart_unlocker_proxy",
    "restart_unlocker_proxy_async",
]
