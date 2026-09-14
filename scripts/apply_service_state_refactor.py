"""One-shot safe refactor for extracting MilanaService state payload validation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = ROOT / "milana_service.py"
MODULE_PATH = ROOT / "milana" / "service_state.py"
PYPROJECT_PATH = ROOT / "pyproject.toml"

IMPORT_ANCHOR = "from milana.host_supervisor import SkillHostSupervisor\n"
STATE_IMPORT = "from milana.service_state import _parse_datetime, build_heartbeat_changes\n"
CONFIG_ANCHOR = "from milana.runtime import (\n"
CONFIG_IMPORT = "from milana.telegram_config import GEMINI_LLM_CHOICE\n"

MODULE_HEADER = '''"""Provider-neutral state payload validation for MilanaService."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping

from milana_state import (
    FactSeed,
    GoalChange,
    HeartbeatChanges,
    NewEntity,
    NewLifeEvent,
    RelationshipDelta,
)


'''

MODULE_EXPORTS = '''

__all__ = ["build_heartbeat_changes"]
'''

RUFF_IGNORE_BLOCK = '''\n[tool.ruff.lint.per-file-ignores]\n# Pre-existing legacy debt: milana_service.py references GEMINI_LLM_CHOICE without\n# importing it. Keep F821 enabled everywhere else while the large service module\n# is split into smaller components in the next refactor step.\n"milana_service.py" = ["F821"]\n'''


def main() -> None:
    text = SERVICE_PATH.read_text(encoding="utf-8")
    original_lines = len(text.splitlines())

    if STATE_IMPORT in text:
        raise SystemExit("milana_service.py state reducer is already refactored")
    if IMPORT_ANCHOR not in text:
        raise RuntimeError("service state import anchor not found")
    if CONFIG_ANCHOR not in text:
        raise RuntimeError("service config import anchor not found")

    start_marker = "def _parse_datetime(value: Any, *, field_name: str) -> datetime:"
    end_marker = "def _target_ref(value: Any) -> str | int:"
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    reducer_source = text[start:end].rstrip()
    text = text[:start] + text[end:]
    text = text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + STATE_IMPORT, 1)
    text = text.replace(CONFIG_ANCHOR, CONFIG_IMPORT + CONFIG_ANCHOR, 1)

    if "build_heartbeat_changes(" not in text:
        raise RuntimeError("service no longer calls build_heartbeat_changes")
    if "_parse_datetime(" not in text:
        raise RuntimeError("service no longer calls _parse_datetime")

    new_lines = len(text.splitlines())
    removed = original_lines - new_lines
    if removed < 150:
        raise RuntimeError(f"state reducer extraction removed only {removed} lines")

    MODULE_PATH.write_text(MODULE_HEADER + reducer_source + MODULE_EXPORTS, encoding="utf-8")
    SERVICE_PATH.write_text(text, encoding="utf-8")

    pyproject = PYPROJECT_PATH.read_text(encoding="utf-8")
    if RUFF_IGNORE_BLOCK not in pyproject:
        raise RuntimeError("expected legacy Ruff ignore block not found")
    pyproject = pyproject.replace(RUFF_IGNORE_BLOCK, "\n", 1)
    PYPROJECT_PATH.write_text(pyproject, encoding="utf-8")

    print(
        f"Extracted service state reducer: milana_service.py {original_lines} -> "
        f"{new_lines} lines; created {MODULE_PATH.relative_to(ROOT)}; enabled F821 globally"
    )


if __name__ == "__main__":
    main()
