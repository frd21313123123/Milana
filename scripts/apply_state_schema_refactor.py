"""One-shot safe extraction of MilanaStateStore schema bootstrap.

The guarded workflow commits the generated module only after compile, lint, and tests pass.
This script is intentionally temporary and is removed after the extraction lands.
"""

from __future__ import annotations

import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "milana_state.py"
MODULE_PATH = ROOT / "milana" / "state_schema.py"

IMPORT_ANCHOR = "from milana.state_models import (\n"
SCHEMA_IMPORT = "from milana.state_schema import create_state_schema\n"

MODULE_HEADER = '''"""SQLite schema bootstrap and additive migrations for Milana state."""

from __future__ import annotations

import sqlite3

from milana.state_models import _json_load


'''

MODULE_EXPORTS = '''\n\n__all__ = ["create_state_schema"]\n'''


def main() -> None:
    text = STATE_PATH.read_text(encoding="utf-8")
    original_lines = len(text.splitlines())

    if SCHEMA_IMPORT in text or MODULE_PATH.exists():
        raise SystemExit("state schema is already refactored")
    if IMPORT_ANCHOR not in text:
        raise RuntimeError("state-model import anchor not found")

    start_marker = "    def _create_schema(self) -> None:\n"
    end_marker = "\n    def close(self) -> None:\n"
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    method_source = text[start:end]
    function_source = textwrap.dedent(method_source)
    function_source = function_source.replace(
        "def _create_schema(self) -> None:",
        "def create_state_schema(connection: sqlite3.Connection) -> None:",
        1,
    )
    function_source = function_source.replace("self._connection", "connection")

    if "CREATE TABLE IF NOT EXISTS agent_state" not in function_source:
        raise RuntimeError("schema extraction lost agent_state DDL")
    if "ALTER TABLE telegram_turn_metrics" not in function_source:
        raise RuntimeError("schema extraction lost additive metric migration")
    if "telegram_outbox_notice_owners" not in function_source:
        raise RuntimeError("schema extraction lost ownership backfill")

    text = text[:start] + text[end:]
    text = text.replace(IMPORT_ANCHOR, SCHEMA_IMPORT + IMPORT_ANCHOR, 1)
    old_call = "            self._create_schema()\n"
    new_call = "            create_state_schema(self._connection)\n"
    if old_call not in text:
        raise RuntimeError("schema call site not found")
    text = text.replace(old_call, new_call, 1)

    removed = original_lines - len(text.splitlines())
    if removed < 250:
        raise RuntimeError(f"schema extraction removed only {removed} lines")

    MODULE_PATH.write_text(
        MODULE_HEADER + function_source.rstrip() + MODULE_EXPORTS,
        encoding="utf-8",
    )
    STATE_PATH.write_text(text, encoding="utf-8")
    print(
        f"Extracted SQLite schema bootstrap: milana_state.py {original_lines} -> "
        f"{len(text.splitlines())} lines; created {MODULE_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
