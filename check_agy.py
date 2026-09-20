"""Check Milana's AGY adapter using the environment of the launching terminal."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone

from agy_provider import AgyModelClient
from telegram_client import (
    ENV_PATH,
    load_agy_model,
    load_agy_reasoning_effort,
    load_ai_settings,
    load_env_file,
)


BASE_DIR = Path(__file__).resolve().parent
REPORT_PATH = BASE_DIR / 'data' / 'agy-check.json'


def safe_error(exc: Exception) -> str:
    message = str(exc)
    message = re.sub(r'https?://\S+', '[URL hidden]', message)
    message = re.sub(r'[\w.+-]+@[\w.-]+', '[email hidden]', message)
    return message[-1500:]


async def check() -> int:
    report: dict[str, object] = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'success': False,
        'python': sys.executable,
        'cwd': str(Path.cwd()),
    }
    try:
        env_values = load_env_file(ENV_PATH)
        settings = load_ai_settings()
        model = load_agy_model(settings, env_values)
        effort = load_agy_reasoning_effort(settings, env_values)
        executable = shutil.which('agy')
        if not executable:
            candidate = Path(os.environ.get('LOCALAPPDATA', '')) / 'agy/bin/agy.exe'
            if candidate.is_file():
                executable = str(candidate.resolve())
        if not executable:
            raise RuntimeError('agy executable not found')
        report.update(executable=executable, model=model, effort=effort)
        report['proxy_variables_present'] = [
            name for name in ('AG_LS_PROXY', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')
            if os.environ.get(name)
        ]
        print(f'AGY: {executable}', flush=True)
        print(f'Model: {model}; effort: {effort}', flush=True)
        print('Testing generation through the Milana adapter (up to 55 seconds)...', flush=True)
        client = AgyModelClient(
            model=model,
            reasoning_effort=effort,
            executable=executable,
            timeout_seconds=45,
            auth_retries=0,
        )
        result = await client.responses.create(
            input=[{
                'role': 'user',
                'content': 'Compute 739 plus 184. Reply only with the integer sum. Do not use tools.',
            }],
        )
        # A completed model response is required; merely listing models is insufficient.
        answer = result.output_text.strip()
        if answer != '923':
            raise RuntimeError('AGY responded, but did not return the expected test answer (923).')
        report['success'] = True
        print('OK: Milana received the expected model answer: 923')
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        report['error'] = safe_error(exc)
        print(f'FAILED: {report["error"]}')
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Report: {REPORT_PATH}')
    return 0 if report['success'] else 1


if __name__ == '__main__':
    if sys.stdout is not None:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    raise SystemExit(asyncio.run(check()))
