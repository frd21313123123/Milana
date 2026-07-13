#!/usr/bin/env python3
"""Локальная веб-панель управления Миланой (альтернатива bot_control.bat).

Запуск:
  python milana_web.py
  # или
  python milana_web.py --port 8765 --no-browser
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# --- пути (как в bot_control.bat) ---
BASE_DIR = Path(__file__).resolve().parent
PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
SCRIPT = BASE_DIR / "telegram_client.py"
SCHEDULE_SCRIPT = BASE_DIR / "milana_schedule.py"
PID_FILE = BASE_DIR / "bot.pid"
MODE_FILE = BASE_DIR / "bot.mode"
LLM_FILE = BASE_DIR / "llm.choice"
OUT_LOG = BASE_DIR / "bot-output.log"
ERR_LOG = BASE_DIR / "bot-error.log"
BAT_FILE = BASE_DIR / "bot_control.bat"

PS = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")

PORT = 8765

# --- утилиты ---
def _read_text(path: Path) -> str | None:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        pass
    return None


def _read_first_line(path: Path) -> str | None:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8", errors="replace") as f:
                line = f.readline().strip()
                return line or None
    except Exception:
        pass
    return None


def _tail(path: Path, n: int = 25) -> list[str]:
    try:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [ln.rstrip("\n\r") for ln in lines[-n:]]
    except Exception:
        return []


def _run_ps(cmd: str, timeout: float = 6.0) -> str:
    try:
        out = subprocess.check_output(
            [PS, "-NoProfile", "-Command", cmd],
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return out.decode("utf-8", errors="replace")
    except Exception:
        return ""


def is_pid_running(pid: int) -> bool:
    if not pid:
        return False
    cmd = (
        f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        f"if ($p -and $p.ProcessName -match '^pythonw?$') {{ exit 0 }}; exit 1"
    )
    res = subprocess.run([PS, "-NoProfile", "-Command", cmd], capture_output=True)
    return res.returncode == 0


def find_bot_pids() -> list[int]:
    pids: list[int] = []

    # 1. Сохранённый PID
    saved = _read_first_line(PID_FILE)
    if saved and saved.isdigit():
        pid = int(saved)
        if is_pid_running(pid):
            pids.append(pid)

    # 2. Поиск по командной строке (как в bat)
    if not pids:
        ps_query = (
            "$processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue; "
            "foreach ($p in $processes) { "
            "if ($p.CommandLine -and $p.CommandLine -match '(?i)telegram_client\\.py' "
            "-and $p.CommandLine -match '(?i)ai-bot') { $p.ProcessId } }"
        )
        raw = _run_ps(ps_query)
        for tok in raw.split():
            tok = tok.strip()
            if tok.isdigit():
                pid = int(tok)
                if pid not in pids and is_pid_running(pid):
                    pids.append(pid)

    return pids


def get_llm_choice() -> str:
    val = _read_first_line(LLM_FILE)
    if val and val.lower() == "gemini":
        return "gemini"
    return "openai"


def resolve_mode(pids: list[int]) -> str:
    """DEV / NORMAL / UNKNOWN / OFF"""
    if not pids:
        return "OFF"

    # Смотрим сохранённый режим
    mode_line = _read_first_line(MODE_FILE)
    if mode_line:
        parts = mode_line.split()
        if len(parts) >= 2:
            saved_pid = parts[0]
            saved_val = parts[1].upper()
            if saved_pid.isdigit() and int(saved_pid) in pids:
                if saved_val == "DEV":
                    return "DEV"
                if saved_val == "NORMAL":
                    return "NORMAL"

    # Определяем по командной строке
    found_dev = False
    found_normal = False
    for pid in pids:
        ps = (
            f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}' "
            "-ErrorAction SilentlyContinue; "
            "if (-not $p -or [string]::IsNullOrWhiteSpace($p.CommandLine)) { exit 2 }; "
            "if ($p.CommandLine -match '(?i)(?:^|\\s)--dev-chat(?:\\s|$)') { exit 0 }; exit 1"
        )
        res = subprocess.run([PS, "-NoProfile", "-Command", ps], capture_output=True)
        if res.returncode == 0:
            found_dev = True
        elif res.returncode == 1:
            found_normal = True

    if found_dev and found_normal:
        return "MIXED"
    if found_dev:
        return "DEV"
    if found_normal:
        return "NORMAL"
    return "UNKNOWN"


def get_process_info(pid: int) -> dict[str, Any]:
    """Возвращает краткую информацию о процессе (как в bat)."""
    ps = (
        f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
        "if (-not $p) { exit 1 }; "
        "$now = Get-Date; "
        "$uptime = $now - $p.StartTime; "
        "$uptimeText = if ($uptime.Days -gt 0) { "
        "  '{0} д {1:00}:{2:00}:{3:00}' -f $uptime.Days,$uptime.Hours,$uptime.Minutes,$uptime.Seconds "
        "} else { "
        "  '{0:00}:{1:00}:{2:00}' -f ([int]$uptime.TotalHours),$uptime.Minutes,$uptime.Seconds "
        "}; "
        "Write-Output ('NAME=' + $p.ProcessName); "
        "Write-Output ('PID=' + $p.Id); "
        "Write-Output ('STARTED=' + $p.StartTime.ToString('dd.MM.yyyy HH:mm:ss')); "
        "Write-Output ('UPTIME=' + $uptimeText); "
        "Write-Output ('CPU=' + ('{0:N1}' -f $p.CPU)); "
        "Write-Output ('MEM=' + ('{0:N1}' -f ($p.WorkingSet64 / 1MB))); "
        "Write-Output ('THREADS=' + $p.Threads.Count)"
    )
    raw = _run_ps(ps)
    info: dict[str, Any] = {"pid": pid}
    for line in raw.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.lower()] = v.strip()
    return info


# --- импорт расписания ---
def get_schedule_state() -> dict[str, Any]:
    """Возвращает текущее состояние расписания в удобном для UI виде."""
    if not PYTHON.exists() or not SCHEDULE_SCRIPT.exists():
        return {"available": False, "text": "Расписание недоступно (нет .venv или milana_schedule.py)"}

    try:
        # Импортируем напрямую — быстро и без запуска отдельного процесса
        sys.path.insert(0, str(BASE_DIR))
        from milana_schedule import (
            DAY_NAMES,
            format_current_status,
            load_routine,
        )

        routine = load_routine()
        text = format_current_status(routine)
        state = routine.state_at()

        return {
            "available": True,
            "text": text,
            "current": _activity_label(state.current) if state.current else "Свободное время",
            "day": DAY_NAMES.get(state.day_key, state.day_key),
            "time": state.now.strftime("%H:%M"),
            "metrics": {
                "energy": state.metrics.energy,
                "stress": state.metrics.stress,
                "productivity": state.metrics.productivity,
                "balance": state.metrics.balance,
            },
        }
    except Exception as exc:
        return {"available": False, "text": f"Ошибка чтения расписания: {exc}"}


def _activity_label(act: Any) -> str:
    try:
        return getattr(act, "title", "Занятие")
    except Exception:
        return "Занятие"


# --- управление через bat (для一致ности логики) ---
def run_bat_command(args: list[str], timeout: float = 35.0) -> dict[str, Any]:
    """Запускает bot_control.bat с аргументами и возвращает результат."""
    if not BAT_FILE.exists():
        return {"ok": False, "message": "bot_control.bat не найден"}

    try:
        proc = subprocess.run(
            [str(BAT_FILE)] + args,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return {
            "ok": proc.returncode == 0,
            "message": output.strip() or ("Успешно" if proc.returncode == 0 else "Ошибка"),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "message": "Команда выполнялась слишком долго"}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def do_start(dev: bool = False) -> dict[str, Any]:
    arg = "dev" if dev else "start"
    return run_bat_command([arg])


def do_stop() -> dict[str, Any]:
    return run_bat_command(["stop"])


def do_restart() -> dict[str, Any]:
    return run_bat_command(["restart"])


def do_set_model(choice: str) -> dict[str, Any]:
    ch = "gemini" if choice.lower() == "gemini" else "openai"
    return run_bat_command(["model", ch])


# --- сбор полного статуса ---
def collect_status() -> dict[str, Any]:
    pids = find_bot_pids()
    running = bool(pids)
    mode = resolve_mode(pids) if running else "OFF"
    llm = get_llm_choice()

    llm_label = "Gemini 3.5 Flash" if llm == "gemini" else "OpenAI (ai_config.json)"

    processes: list[dict[str, Any]] = []
    for pid in pids:
        try:
            processes.append(get_process_info(pid))
        except Exception:
            processes.append({"pid": pid})

    schedule = get_schedule_state()

    # Короткие логи для превью
    out_tail = _tail(OUT_LOG, 6)
    err_tail = _tail(ERR_LOG, 4)

    status_text = "ЗАПУЩЕНА" if running else "НЕ ЗАПУЩЕНА"

    return {
        "running": running,
        "status_text": status_text,
        "pids": pids,
        "mode": mode,
        "llm": llm,
        "llm_label": llm_label,
        "processes": processes,
        "schedule": schedule,
        "logs_preview": {
            "output": out_tail,
            "errors": err_tail,
        },
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


def collect_logs() -> dict[str, Any]:
    return {
        "output": _tail(OUT_LOG, 40),
        "errors": _tail(ERR_LOG, 30),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


# --- HTTP обработчик ---
INDEX_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Милана — Панель управления</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=Space+Grotesk:wght@500;600&amp;display=swap');
    
    :root {
      --accent: 167 139 250;
    }
    
    body {
      font-family: 'Inter', system_ui, sans-serif;
    }
    
    .font-display {
      font-family: 'Space Grotesk', 'Inter', system_ui, sans-serif;
    }

    .status-badge {
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .milana-card {
      transition: transform 0.2s cubic-bezier(0.4, 0.0, 0.2, 1), 
                  box-shadow 0.2s cubic-bezier(0.4, 0.0, 0.2, 1);
    }

    .milana-card:hover {
      transform: translateY(-1px);
    }

    .action-btn {
      transition: all 0.1s ease;
    }
    
    .action-btn:active {
      transform: scale(0.985);
    }

    .log-pre {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 12.5px;
      line-height: 1.45;
    }

    .section-title {
      font-size: 13px;
      letter-spacing: 0.5px;
      font-weight: 600;
      text-transform: uppercase;
      color: #64748b;
    }

    .metric {
      transition: all 0.3s ease;
    }

    .nav-active {
      background-color: rgb(241 245 249);
      color: rgb(15 23 42);
      font-weight: 600;
    }

    .toast {
      animation: toast-pop 0.2s ease forwards;
    }

    @keyframes toast-pop {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-200">
  <div class="max-w-5xl mx-auto px-4 py-8">
    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <div class="flex items-center gap-x-3">
        <div class="w-11 h-11 bg-violet-500 rounded-2xl flex items-center justify-center shadow-inner">
          <span class="text-white text-3xl leading-none mt-0.5">🦋</span>
        </div>
        <div>
          <div class="font-display text-3xl font-semibold tracking-tighter">Милана</div>
          <div class="text-violet-400 text-sm -mt-1">Панель управления</div>
        </div>
      </div>
      
      <div class="flex items-center gap-x-2 text-sm">
        <div id="connection-dot" 
             class="w-2.5 h-2.5 bg-emerald-400 rounded-full animate-pulse"></div>
        <div class="text-slate-400">Локально • <span id="server-time">--:--:--</span></div>
      </div>
    </div>

    <!-- Main Status -->
    <div class="milana-card bg-slate-900 border border-slate-800 rounded-3xl p-6 mb-6 shadow-xl shadow-black/30">
      <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-y-4">
        <div>
          <div class="flex items-center gap-x-3">
            <div id="status-badge" 
                 class="status-badge inline-flex items-center px-4 py-1.5 rounded-2xl text-sm font-semibold tracking-wide bg-slate-700 text-slate-300">
              ○ Загрузка...
            </div>
            <div id="pids-text" class="text-sm text-slate-400 font-mono"></div>
          </div>
          <div id="milana-state" class="mt-2 text-base font-semibold text-slate-400">Милана: Проверяем состояние...</div>
          <div id="mode-text" class="mt-1.5 text-xl font-semibold"></div>
          <div id="llm-text" class="text-sm text-violet-300 mt-0.5"></div>
        </div>

        <div class="flex gap-2">
          <button onclick="refreshAll(false)" 
                  class="px-4 py-2 text-sm bg-slate-800 hover:bg-slate-700 active:bg-slate-900 transition rounded-2xl border border-slate-700 flex items-center gap-x-2">
            <span>⟳</span>
            <span>Обновить</span>
          </button>
        </div>
      </div>

      <!-- Process details -->
      <div id="process-details" class="mt-5 grid grid-cols-1 md:grid-cols-2 gap-3 text-sm"></div>
    </div>

    <!-- Actions -->
    <div class="mb-6">
      <div class="section-title mb-2 px-1">Управление</div>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
        <button onclick="startBot(false)"
                class="action-btn flex items-center justify-center gap-x-2 bg-emerald-600 hover:bg-emerald-500 active:bg-emerald-700 text-white font-semibold py-3.5 rounded-3xl shadow">
          <span class="text-lg">▶</span>
          <span>Запустить (обычный)</span>
        </button>
        
        <button onclick="startBot(true)"
                class="action-btn flex items-center justify-center gap-x-2 bg-sky-600 hover:bg-sky-500 active:bg-sky-700 text-white font-semibold py-3.5 rounded-3xl shadow">
          <span class="text-lg">⚡</span>
          <span>Запустить DEV</span>
        </button>
        
        <button onclick="restartBot()"
                class="action-btn flex items-center justify-center gap-x-2 bg-amber-500 hover:bg-amber-400 active:bg-amber-600 text-slate-900 font-semibold py-3.5 rounded-3xl shadow">
          <span class="text-lg">⟳</span>
          <span>Перезапустить</span>
        </button>
        
        <button onclick="stopBot()"
                class="action-btn flex items-center justify-center gap-x-2 bg-rose-600 hover:bg-rose-500 active:bg-rose-700 text-white font-semibold py-3.5 rounded-3xl shadow">
          <span class="text-lg">■</span>
          <span>Остановить</span>
        </button>
      </div>
    </div>

    <!-- LLM -->
    <div class="mb-6">
      <div class="section-title mb-2 px-1">Модель ИИ</div>
      <div class="flex gap-3">
        <button onclick="setModel('openai')"
                id="btn-openai"
                class="flex-1 action-btn px-5 py-3 rounded-3xl border border-slate-700 hover:border-slate-500 font-medium flex items-center justify-center gap-x-2">
          <span>OpenAI</span>
          <span class="text-xs opacity-60">(ai_config.json)</span>
        </button>
        <button onclick="setModel('gemini')"
                id="btn-gemini"
                class="flex-1 action-btn px-5 py-3 rounded-3xl border border-slate-700 hover:border-slate-500 font-medium flex items-center justify-center gap-x-2">
          <span>Gemini 3.5 Flash</span>
        </button>
      </div>
      <div class="text-[10px] text-slate-500 mt-1.5 px-1">После смены модели рекомендуется перезапустить бота.</div>
    </div>

    <!-- Current State -->
    <div class="mb-6">
      <div class="flex items-center justify-between mb-2 px-1">
        <div class="section-title">Текущее состояние Миланы</div>
        <div id="schedule-time" class="text-xs text-slate-500"></div>
      </div>
      
      <div class="milana-card bg-slate-900 border border-slate-800 rounded-3xl p-5">
        <!-- Launch state inside "Текущее состояние Миланы" for visibility -->
        <div id="schedule-launch-state" class="mb-3 text-sm font-semibold"></div>
        <!-- Metrics -->
        <div id="metrics-row" class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4"></div>
        
        <div id="schedule-text"
             class="log-pre bg-slate-950 border border-slate-800 text-slate-300 p-4 rounded-2xl whitespace-pre-wrap text-sm leading-snug min-h-[92px]">
          Загрузка...
        </div>
      </div>
    </div>

    <!-- Logs -->
    <div>
      <div class="flex items-center justify-between mb-2 px-1">
        <div class="section-title">Последние события</div>
        <button onclick="refreshLogs()" 
                class="text-xs px-3 py-1 bg-slate-800 hover:bg-slate-700 rounded-2xl border border-slate-700 transition">Обновить логи</button>
      </div>
      
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div>
          <div class="text-emerald-400 text-xs font-medium mb-1 px-1">bot-output.log</div>
          <pre id="log-output" 
               class="log-pre bg-slate-900 border border-slate-800 text-emerald-200/90 p-4 rounded-2xl h-44 overflow-auto"></pre>
        </div>
        <div>
          <div class="text-rose-400 text-xs font-medium mb-1 px-1">bot-error.log</div>
          <pre id="log-error" 
               class="log-pre bg-slate-900 border border-slate-800 text-rose-300/90 p-4 rounded-2xl h-44 overflow-auto"></pre>
        </div>
      </div>
    </div>

    <div class="mt-8 text-center text-[10px] text-slate-500">
      Локальная панель. Всё работает так же, как и через <span class="font-mono">bot_control.bat</span>.
      Закройте вкладку, когда закончите. Сервер можно остановить в консоли (Ctrl+C).
    </div>
  </div>

  <!-- Toast -->
  <div id="toast-container" class="fixed bottom-4 right-4 flex flex-col gap-2 z-50"></div>

  <script>
    let polling = null;
    let isActing = false;

    function showToast(message, type = 'info') {
      const container = document.getElementById('toast-container');
      const colors = {
        info: 'bg-slate-800 border-slate-700 text-slate-200',
        success: 'bg-emerald-900/90 border-emerald-700 text-emerald-100',
        error: 'bg-rose-900/90 border-rose-700 text-rose-100'
      };
      
      const el = document.createElement('div');
      el.className = `toast px-4 py-2.5 rounded-2xl shadow-xl border text-sm max-w-xs ${colors[type] || colors.info}`;
      el.innerHTML = `<div>${message}</div>`;
      
      container.appendChild(el);
      
      setTimeout(() => {
        el.style.transition = 'all 0.2s ease';
        el.style.opacity = '0';
        setTimeout(() => el.remove(), 120);
      }, 2800);
    }

    function setInitialLoadingState() {
      const badge = document.getElementById('status-badge');
      if (badge) {
        badge.className = 'status-badge inline-flex items-center px-4 py-1.5 rounded-2xl text-sm font-semibold tracking-wide bg-slate-700 text-slate-300';
        badge.textContent = '○ Загрузка...';
      }
      const stateEl = document.getElementById('milana-state');
      if (stateEl) {
        stateEl.innerHTML = '<span class="text-slate-400">Милана: Проверяем состояние...</span>';
      }
      const pidsEl = document.getElementById('pids-text');
      if (pidsEl) pidsEl.textContent = '';
      const modeEl = document.getElementById('mode-text');
      if (modeEl) modeEl.textContent = '';
      const llmEl = document.getElementById('llm-text');
      if (llmEl) llmEl.textContent = 'Модель: ...';
      const schedLaunch = document.getElementById('schedule-launch-state');
      if (schedLaunch) schedLaunch.textContent = 'Милана: Проверяем...';
    }

    function setErrorState() {
      const badge = document.getElementById('status-badge');
      if (badge) {
        badge.className = 'status-badge inline-flex items-center px-4 py-1.5 rounded-2xl text-sm font-semibold tracking-wide bg-rose-900 text-rose-200';
        badge.textContent = '⚠ ОШИБКА СВЯЗИ';
      }
      const stateEl = document.getElementById('milana-state');
      if (stateEl) {
        stateEl.innerHTML = '<span class="text-rose-400">Милана: Состояние недоступно (нажмите Обновить)</span>';
      }
      const dot = document.getElementById('connection-dot');
      if (dot) {
        dot.className = 'w-2.5 h-2.5 bg-rose-500 rounded-full';
      }
      const schedLaunch = document.getElementById('schedule-launch-state');
      if (schedLaunch) schedLaunch.innerHTML = '<span class="text-rose-400">Милана: Состояние недоступно</span>';
    }

    function setBadge(running, text) {
      const badge = document.getElementById('status-badge');
      if (!badge) return;
      
      if (running) {
        badge.className = 'status-badge inline-flex items-center px-4 py-1.5 rounded-2xl text-sm font-semibold tracking-wide bg-emerald-500 text-emerald-950';
        badge.textContent = '● ' + (text || 'ЗАПУЩЕНА');
      } else {
        badge.className = 'status-badge inline-flex items-center px-4 py-1.5 rounded-2xl text-sm font-semibold tracking-wide bg-slate-700 text-slate-300';
        badge.textContent = '○ ' + (text || 'НЕ ЗАПУЩЕНА');
      }
    }

    function updateLLMButtons(llm) {
      const btnO = document.getElementById('btn-openai');
      const btnG = document.getElementById('btn-gemini');
      
      if (!btnO || !btnG) return;
      
      btnO.classList.remove('!border-violet-400', 'bg-slate-800');
      btnG.classList.remove('!border-violet-400', 'bg-slate-800');
      
      if (llm === 'gemini') {
        btnG.classList.add('!border-violet-400', 'bg-slate-800');
      } else {
        btnO.classList.add('!border-violet-400', 'bg-slate-800');
      }
    }

    function renderMetrics(metrics) {
      const row = document.getElementById('metrics-row');
      if (!row) return;
      row.innerHTML = '';
      
      if (!metrics) {
        row.innerHTML = `<div class="col-span-4 text-xs text-slate-500 px-1">Метрики недоступны</div>`;
        return;
      }
      
      const items = [
        { label: 'Энергия', value: metrics.energy, unit: '%', color: 'text-emerald-400' },
        { label: 'Стресс', value: metrics.stress, unit: '%', color: 'text-amber-400' },
        { label: 'Продуктивность', value: metrics.productivity, unit: '%', color: 'text-sky-400' },
        { label: 'Баланс', value: metrics.balance, unit: '%', color: 'text-violet-400' },
      ];
      
      items.forEach(it => {
        const div = document.createElement('div');
        div.className = 'bg-slate-950 border border-slate-800 rounded-2xl px-3 py-2.5';
        div.innerHTML = `
          <div class="text-[10px] uppercase tracking-widest text-slate-500">${it.label}</div>
          <div class="text-2xl font-semibold tabular-nums mt-0.5 ${it.color}">${it.value}<span class="text-sm font-normal text-slate-400">${it.unit}</span></div>
        `;
        row.appendChild(div);
      });
    }

    function renderProcessDetails(processes) {
      const el = document.getElementById('process-details');
      if (!el) return;
      el.innerHTML = '';
      
      if (!processes || processes.length === 0) {
        el.innerHTML = `<div class="text-sm text-slate-500 col-span-2">Нет активных процессов</div>`;
        return;
      }
      
      processes.forEach(p => {
        const div = document.createElement('div');
        div.className = 'bg-slate-950/70 border border-slate-800 px-3 py-2 rounded-2xl text-xs mono';
        const lines = [];
        if (p.name) lines.push(`Процесс: ${p.name}`);
        lines.push(`PID: ${p.pid}`);
        if (p.started) lines.push(`Запущен: ${p.started}`);
        if (p.uptime) lines.push(`Uptime: ${p.uptime}`);
        if (p.mem) lines.push(`Память: ${p.mem} MB`);
        div.innerHTML = lines.join('<br>');
        el.appendChild(div);
      });
    }

    function updateUI(data) {
      if (!data) return;
      
      // badge
      setBadge(data.running, data.status_text);
      
      // pids
      const pidsEl = document.getElementById('pids-text');
      if (pidsEl) pidsEl.textContent = data.pids && data.pids.length ? 'PID: ' + data.pids.join(', ') : '';
      
      // mode
      const modeEl = document.getElementById('mode-text');
      let modeLabel = '';
      if (data.mode === 'DEV') modeLabel = 'DEV CHAT — мгновенные ответы (расписание отключено)';
      else if (data.mode === 'NORMAL') modeLabel = 'Обычный режим (по расписанию)';
      else if (data.mode === 'MIXED') modeLabel = 'СМЕШАННЫЙ РЕЖИМ (внимание!)';
      else if (data.mode === 'UNKNOWN') modeLabel = 'Режим неизвестен';
      if (modeEl) modeEl.textContent = modeLabel;
      
      // llm
      const llmEl = document.getElementById('llm-text');
      if (llmEl) llmEl.textContent = 'Модель: ' + (data.llm_label || data.llm);
      updateLLMButtons(data.llm);
      
      // explicit launch state for "состояние Миланы (Запущена или нет)"
      const stateEl = document.getElementById('milana-state');
      if (stateEl) {
        if (data.running) {
          stateEl.innerHTML = '<span class="text-emerald-400">Милана: Запущена</span>';
        } else {
          stateEl.innerHTML = '<span class="text-slate-400">Милана: Не запущена</span>';
        }
      }
      
      // also show in the "Текущее состояние Миланы" section
      const schedLaunch = document.getElementById('schedule-launch-state');
      if (schedLaunch) {
        if (data.running) {
          schedLaunch.innerHTML = '<span class="text-emerald-400">● Милана запущена</span>';
        } else {
          schedLaunch.innerHTML = '<span class="text-slate-400">○ Милана не запущена</span>';
        }
      }
      
      // processes
      renderProcessDetails(data.processes);
      
      // schedule
      const sched = data.schedule || {};
      const schedText = document.getElementById('schedule-text');
      if (schedText) {
        schedText.textContent = sched.text || 'Состояние расписания недоступно';
      }
      
      const schedTime = document.getElementById('schedule-time');
      if (schedTime) schedTime.textContent = sched.day ? `${sched.day} • ${sched.time || ''}` : '';
      
      renderMetrics(sched.metrics);
      
      // server time
      if (data.timestamp) {
        const timeEl = document.getElementById('server-time');
        if (timeEl) timeEl.textContent = data.timestamp;
      }
    }

    async function refreshStatus(silent = false) {
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          const res = await fetch('/api/status');
          if (!res.ok) throw new Error('HTTP ' + res.status);
          const data = await res.json();
          updateUI(data);
          // restore dot if it was errored
          const dot = document.getElementById('connection-dot');
          if (dot) dot.className = 'w-2.5 h-2.5 bg-emerald-400 rounded-full animate-pulse';
          return data;
        } catch (e) {
          if (attempt < 2) {
            await new Promise(r => setTimeout(r, 350));
            continue;
          }
          if (!silent) showToast('Не удалось подключиться к серверу. Попробуйте обновить страницу.', 'error');
          console.error(e);
          setErrorState();
        }
      }
    }

    async function refreshLogs() {
      try {
        const res = await fetch('/api/logs');
        const data = await res.json();
        
        const outEl = document.getElementById('log-output');
        const errEl = document.getElementById('log-error');
        
        outEl.textContent = (data.output || []).join('\n') || 'Нет записей';
        errEl.textContent = (data.errors || []).join('\n') || 'Нет записей';
        
        if (data.timestamp) {
          const timeEl = document.getElementById('server-time');
          if (timeEl) timeEl.textContent = data.timestamp;
        }
      } catch (e) {
        showToast('Не удалось загрузить логи', 'error');
      }
    }

    async function refreshAll(silent = false) {
      if (!silent) setInitialLoadingState();
      await refreshStatus(silent);
      await refreshLogs();
    }

    async function performAction(path, body = {}) {
      if (isActing) return;
      isActing = true;
      
      const btns = document.querySelectorAll('button');
      btns.forEach(b => b.disabled = true);
      
      try {
        const res = await fetch(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        
        if (data.message) {
          showToast(data.message.slice(0, 220), data.ok ? 'success' : 'error');
        }
        
        // Подождать немного, чтобы процессы успели запуститься/остановиться
        await new Promise(r => setTimeout(r, 650));
        await refreshAll(true);
      } catch (e) {
        showToast('Ошибка выполнения действия', 'error');
      } finally {
        btns.forEach(b => b.disabled = false);
        isActing = false;
      }
    }

    function startBot(dev) {
      performAction('/api/start', { dev: !!dev });
    }
    
    function stopBot() {
      performAction('/api/stop');
    }
    
    function restartBot() {
      performAction('/api/restart');
    }
    
    function setModel(choice) {
      performAction('/api/model', { choice });
    }

    function startPolling() {
      if (polling) clearInterval(polling);
      polling = setInterval(() => {
        refreshStatus(true);
      }, 4800);
    }

    async function init() {
      // Tailwind script already loaded via CDN
      setInitialLoadingState();
      await refreshAll(true);
      startPolling();
      
      // initial logs load
      setTimeout(() => refreshLogs(), 800);
      
      // Keyboard hint
      document.addEventListener('keydown', (e) => {
        if (e.key.toLowerCase() === 'r' && (e.metaKey || e.ctrlKey)) {
          e.preventDefault();
          refreshAll();
        }
      });
      
      console.log('%c[Milana] Web control ready', 'color:#64748b');
    }

    // Boot
    window.addEventListener('load', init);
  </script>
</body>
</html>
"""


class MilanaHandler(BaseHTTPRequestHandler):
    def _send_headers(self, code: int = 200, content_type: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def send_json(self, payload: dict[str, Any], code: int = 200):
        self._send_headers(code, "application/json; charset=utf-8")
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.wfile.write(body)

    def send_html(self, html: str):
        self._send_headers(200, "text/html; charset=utf-8")
        self.wfile.write(html.encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # --- GET ---
    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html", "/ui"):
            self.send_html(INDEX_HTML)
            return

        if path == "/api/status":
            data = collect_status()
            self.send_json(data)
            return

        if path == "/api/logs":
            data = collect_logs()
            self.send_json(data)
            return

        self.send_error(404, "Not Found")

    # --- POST ---
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            body = {}

        path = self.path.split("?")[0]

        if path == "/api/start":
            dev = bool(body.get("dev"))
            result = do_start(dev)
            result["action"] = "start"
            self.send_json(result)
            return

        if path == "/api/stop":
            result = do_stop()
            result["action"] = "stop"
            self.send_json(result)
            return

        if path == "/api/restart":
            result = do_restart()
            result["action"] = "restart"
            self.send_json(result)
            return

        if path == "/api/model":
            choice = body.get("choice", "openai")
            result = do_set_model(choice)
            result["action"] = "model"
            self.send_json(result)
            return

        self.send_json({"ok": False, "message": "Unknown action"}, 404)


def run_server(port: int = PORT, open_browser: bool = True):
    server_address = ("127.0.0.1", port)
    httpd = None
    url = f"http://127.0.0.1:{port}/"

    try:
        httpd = HTTPServer(server_address, MilanaHandler)
    except OSError as e:
        print(f"Не удалось запустить сервер на порту {port}: {e}")
        print(f"Возможно, панель уже запущена. Открываю браузер: {url}")
        if open_browser:
            try:
                webbrowser.open_new_tab(url)
            except Exception:
                pass
        return

    print("=" * 56)
    print("  Милана — локальная панель управления")
    print(f"  Открыто: {url}")
    print("  (Ctrl+C — остановить сервер)")
    print("=" * 56)

    # Server is now listening
    print("Server ready and listening on", url)

    if open_browser:
        def _open():
            # Small delay so the browser doesn't race the very first request
            time.sleep(1.3)
            try:
                print("Opening browser...")
                webbrowser.open_new_tab(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка сервера...")
    finally:
        if httpd:
            httpd.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Локальная веб-панель управления Миланой")
    parser.add_argument("--port", type=int, default=PORT, help=f"Порт (по умолчанию {PORT})")
    parser.add_argument("--no-browser", action="store_true", help="Не открывать браузер автоматически")
    args = parser.parse_args()

    run_server(port=args.port, open_browser=not args.no_browser)
