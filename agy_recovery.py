"""Small, fail-closed helpers for recovering the Antigravity unlocker.

The Telegram channel is treated as untrusted input.  A key is accepted only
when it is an exact-looking unlocker key and appears next to the installed
unlocker version marker.  The key is kept in memory only and is never passed
as a command-line argument or written to a file.
"""

from __future__ import annotations

import asyncio
import os
import re
import select
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_UNLOCKER_PATH = Path("/root/.local/share/agunlocker/ag_unlocker")
KEY_RE = re.compile(r"(?<![A-Z0-9])[A-Z0-9]{24}(?![A-Z0-9])")
VERSION_RE = re.compile(r"\bversion\s*:\s*([0-9]+\.[0-9]+\.[0-9]+)\b", re.I)


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


def _user_systemd_environment() -> dict[str, str]:
    environment = os.environ.copy()
    runtime_dir = Path("/run/user/0")
    if runtime_dir.is_dir():
        environment.setdefault("XDG_RUNTIME_DIR", str(runtime_dir))
    return environment


def _run_unlocker_tui(binary: Path, key: str) -> bool:
    """Feed the key to the unlocker's TUI through a private pseudo-terminal."""

    if os.name == "nt" or not binary.is_file() or not os.access(binary, os.X_OK):
        return False
    try:
        import pty

        master, slave = pty.openpty()
        process = subprocess.Popen(
            [str(binary), "--tui"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
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
        deadline = time.monotonic() + 20.0
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
            elapsed = 20.0 - max(0.0, deadline - time.monotonic())
            if not sent_key and elapsed >= 0.5:
                os.write(master, key.encode("ascii") + b"\n")
                sent_key = True
            elif sent_key and not sent_enable and elapsed >= 1.5:
                os.write(master, b"a")
                sent_enable = True
            elif sent_enable and not sent_quit and elapsed >= 3.0:
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


async def activate_unlocker_async(binary: Path, key: str) -> bool:
    return await asyncio.to_thread(activate_unlocker, binary, key)


__all__ = [
    "DEFAULT_UNLOCKER_PATH",
    "activate_unlocker",
    "activate_unlocker_async",
    "extract_unlocker_key",
    "installed_unlocker_version",
    "is_unlocker_recoverable_error",
]
