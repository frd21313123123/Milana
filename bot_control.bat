@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "PYTHONW=%ROOT%.venv\Scripts\pythonw.exe"
set "SCRIPT=%ROOT%milana_service.py"
set "SCHEDULE_SCRIPT=%ROOT%milana_schedule.py"
set "PID_FILE=%ROOT%bot.pid"
set "MODE_FILE=%ROOT%bot.mode"
set "LLM_FILE=%ROOT%llm.choice"
set "AGY_MODEL_FILE=%ROOT%agy.model"
set "AGY_EFFORT_FILE=%ROOT%agy.effort"
set "OUT_LOG=%ROOT%bot-output.log"
set "ERR_LOG=%ROOT%bot-error.log"
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

rem The Antigravity installer updates the user PATH, but terminals that were
rem already open keep the old environment. Prefer the standard CLI location so
rem a freshly installed agy is available to this controller immediately.
if exist "%LOCALAPPDATA%\agy\bin\agy.exe" set "PATH=%LOCALAPPDATA%\agy\bin;%PATH%"

if "%~1"=="" (
    set "INTERACTIVE=1"
    goto menu
)

if /I "%~1"=="start" if "%~2"=="" goto start
if /I "%~1"=="start" if /I "%~2"=="dev" goto start_dev
if /I "%~1"=="start" goto invalid_start_mode
if /I "%~1"=="dev" goto start_dev
if /I "%~1"=="start-dev" goto start_dev
if /I "%~1"=="restart" goto restart
if /I "%~1"=="model" goto model_command
if /I "%~1"=="agy" goto agy_command
if /I "%~1"=="stop" goto stop
if /I "%~1"=="status" goto status
if /I "%~1"=="logs" goto logs
if /I "%~1"=="web" goto open_web
if /I "%~1"=="site" goto open_web
if /I "%~1"=="ui" goto open_web
if /I "%~1"=="open" goto open_web

echo Unknown command: %~1
echo Use: bot_control.bat [start [dev]^|dev^|start-dev^|restart^|model [openai^|gemini^|lmstudio]^|agy [model ID^|effort low^|medium^|high]^|stop^|status^|logs^|web]
exit /b 2

:invalid_start_mode
echo Unknown start mode: %~2
echo Use: bot_control.bat start [dev]
exit /b 2

:menu
cls
title Milana AI control
echo Milana AI control
echo.
call :show_full_status
echo.
echo 1. Start bot normally (schedule enabled)
echo 2. Start DEV chat (immediate replies)
echo 3. Choose LLM model
echo 4. Configure Antigravity model and reasoning
echo 5. Restart bot (keep current mode)
echo 6. Stop bot
echo 7. Show status
echo 8. Show recent logs
echo 9. Открыть сайт (веб-панель управления Миланой)
echo 0. Exit
echo.
set /p "CHOICE=Choose an action: "
if "%CHOICE%"=="1" goto start
if "%CHOICE%"=="2" goto start_dev
if "%CHOICE%"=="3" goto model_menu
if "%CHOICE%"=="4" goto agy_menu
if "%CHOICE%"=="5" goto restart
if "%CHOICE%"=="6" goto stop
if "%CHOICE%"=="7" goto status
if "%CHOICE%"=="8" goto logs
if "%CHOICE%"=="9" goto open_web
if "%CHOICE%"=="0" goto done
echo Invalid choice.
goto menu_pause

:start
set "DEV_CHAT_ARG="
set "START_MODE_KEY=NORMAL"
set "START_MODE=normal (schedule enabled)"
goto start_common

:start_dev
set "DEV_CHAT_ARG=--dev-chat"
set "START_MODE_KEY=DEV"
set "START_MODE=DEV chat (immediate replies)"
goto start_common

:start_common
call :load_llm_choice
if /I "%LLM_CHOICE%"=="gemini" (
    where agy >nul 2>&1
    if errorlevel 1 (
        echo Cannot start with Antigravity: the "agy" command was not found in PATH.
        echo Install and configure agy, or switch back with: bot_control.bat model openai
        goto action_done
    )
)
if not exist "%PYTHON%" (
    echo Python environment not found: %PYTHON%
    goto action_done
)
if not exist "%PYTHONW%" set "PYTHONW=%PYTHON%"
if not exist "%SCRIPT%" (
    echo Bot script not found: %SCRIPT%
    goto action_done
)

call :find_bot_pids
if defined BOT_PIDS (
    echo Bot is already running. PIDs:%BOT_PIDS%
    goto action_done
)
if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
if exist "%MODE_FILE%" del /q "%MODE_FILE%" >nul 2>&1

if defined DEV_CHAT_ARG (
    start "" "%PYTHONW%" -u "%SCRIPT%" --dev-chat 1>"%OUT_LOG%" 2>"%ERR_LOG%"
) else (
    start "" "%PYTHONW%" -u "%SCRIPT%" 1>"%OUT_LOG%" 2>"%ERR_LOG%"
)
if errorlevel 1 (
    echo Failed to start the bot.
    call :cleanup_failed_start
    goto action_done
)

rem MilanaService records the PID of the real interpreter itself. This avoids
rem saving the venv launcher PID on Windows.  Cold Windows imports and the
rem launcher handoff can take longer than five seconds, so allow up to twenty
rem seconds before declaring startup failure.
for /L %%N in (1,1,80) do (
    if not exist "%PID_FILE%" "%PS%" -NoProfile -Command "Start-Sleep -Milliseconds 250"
)

call :read_pid
call :is_running %BOT_PID% %BOT_START_TICKS%
if errorlevel 1 (
    echo Bot process exited during startup. Recent errors:
    if exist "%ERR_LOG%" "%PS%" -NoProfile -Command "Get-Content -Encoding utf8 -LiteralPath '%ERR_LOG%' -Tail 20"
    if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
    if exist "%MODE_FILE%" del /q "%MODE_FILE%" >nul 2>&1
    goto action_done
)
echo Bot started. PID: %BOT_PID%
echo Mode: %START_MODE%
echo Output log: %OUT_LOG%
echo Error log: %ERR_LOG%
echo.
call :show_full_status
goto action_done

:model_command
if "%~2"=="" (
    call :show_llm_choice
    echo Use: bot_control.bat model [openai^|gemini^|lmstudio]
    exit /b 0
)
if not "%~3"=="" goto invalid_model
if /I "%~2"=="openai" (
    call :set_llm_choice openai
    goto action_done
)
if /I "%~2"=="gemini" (
    call :set_llm_choice gemini
    goto action_done
)
if /I "%~2"=="lmstudio" (
    call :set_llm_choice lmstudio
    goto action_done
)

:invalid_model
echo Unknown LLM model: %~2
echo Use: bot_control.bat model [openai^|gemini^|lmstudio]
exit /b 2

:model_menu
cls
echo Choose LLM model
echo.
call :show_llm_choice
echo.
echo 1. OpenAI (model configured in ai_config.json)
echo 2. Antigravity (agy CLI; model and reasoning configured separately)
echo 3. LM Studio (local OpenAI-compatible server)
echo 0. Back
echo.
set "MODEL_CHOICE="
set /p "MODEL_CHOICE=Choose a model: "
if "%MODEL_CHOICE%"=="1" (
    call :set_llm_choice openai
    goto menu_pause
)
if "%MODEL_CHOICE%"=="2" (
    call :set_llm_choice gemini
    goto menu_pause
)
if "%MODEL_CHOICE%"=="3" (
    call :set_llm_choice lmstudio
    goto menu_pause
)
if "%MODEL_CHOICE%"=="0" goto menu
echo Invalid choice.
goto menu_pause

:agy_command
if "%~2"=="" (
    call :show_agy_choice
    echo Use: bot_control.bat agy [model MODEL_ID^|effort low^|medium^|high]
    exit /b 0
)
if /I "%~2"=="model" (
    if "%~3"=="" goto invalid_agy_command
    if not "%~4"=="" goto invalid_agy_command
    call :set_agy_model "%~3"
    if errorlevel 1 exit /b 2
    goto action_done
)
if /I "%~2"=="effort" (
    if "%~3"=="" goto invalid_agy_command
    if not "%~4"=="" goto invalid_agy_command
    call :set_agy_effort "%~3"
    if errorlevel 1 exit /b 2
    goto action_done
)

:invalid_agy_command
echo Use: bot_control.bat agy [model MODEL_ID^|effort low^|medium^|high]
exit /b 2

:agy_menu
cls
echo Configure Antigravity
echo.
call :show_agy_choice
echo.
echo 1. Choose model
echo 2. Choose reasoning effort
echo 0. Back
echo.
set "AGY_CHOICE="
set /p "AGY_CHOICE=Choose an action: "
if "%AGY_CHOICE%"=="1" goto agy_model_menu
if "%AGY_CHOICE%"=="2" goto agy_effort_menu
if "%AGY_CHOICE%"=="0" goto menu
echo Invalid choice.
goto menu_pause

:agy_model_menu
cls
echo Choose Antigravity model
echo.
call :show_agy_choice
echo.
echo  1. Gemini 3.8 Flash High
echo  2. Gemini 3.8 Flash Medium
echo  3. Gemini 3.8 Flash Low
echo  4. Gemini 3.7 Flash High
echo  5. Gemini 3.7 Flash Medium
echo  6. Gemini 3.7 Flash Low
echo  7. Gemini 3.6 Flash High
echo  8. Gemini 3.6 Flash Medium
echo  9. Gemini 3.6 Flash Low
echo 10. Gemini 3.1 Pro High
echo 11. Gemini 3.1 Pro Low
echo 12. Claude Sonnet 4.6 Thinking
echo 13. Claude Opus 4.6 Thinking
echo 14. GPT-OSS 120B Medium
echo  0. Back
echo.
set "AGY_MODEL_CHOICE="
set /p "AGY_MODEL_CHOICE=Choose a model: "
if "%AGY_MODEL_CHOICE%"=="1" call :set_agy_model gemini-3.8-flash-high
if "%AGY_MODEL_CHOICE%"=="2" call :set_agy_model gemini-3.8-flash-medium
if "%AGY_MODEL_CHOICE%"=="3" call :set_agy_model gemini-3.8-flash-low
if "%AGY_MODEL_CHOICE%"=="4" call :set_agy_model gemini-3.7-flash-high
if "%AGY_MODEL_CHOICE%"=="5" call :set_agy_model gemini-3.7-flash-medium
if "%AGY_MODEL_CHOICE%"=="6" call :set_agy_model gemini-3.7-flash-low
if "%AGY_MODEL_CHOICE%"=="7" call :set_agy_model gemini-3.6-flash-high
if "%AGY_MODEL_CHOICE%"=="8" call :set_agy_model gemini-3.6-flash-medium
if "%AGY_MODEL_CHOICE%"=="9" call :set_agy_model gemini-3.6-flash-low
if "%AGY_MODEL_CHOICE%"=="10" call :set_agy_model gemini-3.1-pro-high
if "%AGY_MODEL_CHOICE%"=="11" call :set_agy_model gemini-3.1-pro-low
if "%AGY_MODEL_CHOICE%"=="12" call :set_agy_model claude-sonnet-4-6
if "%AGY_MODEL_CHOICE%"=="13" call :set_agy_model claude-opus-4-6-thinking
if "%AGY_MODEL_CHOICE%"=="14" call :set_agy_model gpt-oss-120b-medium
if "%AGY_MODEL_CHOICE%"=="0" goto agy_menu
if "%AGY_MODEL_CHOICE%"=="1" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="2" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="3" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="4" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="5" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="6" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="7" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="8" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="9" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="10" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="11" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="12" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="13" goto menu_pause
if "%AGY_MODEL_CHOICE%"=="14" goto menu_pause
echo Invalid choice.
goto menu_pause

:agy_effort_menu
cls
echo Choose Antigravity reasoning effort
echo.
call :show_agy_choice
echo.
echo 1. Low    - fastest, least deliberation
echo 2. Medium - balanced (default)
echo 3. High   - slower, most deliberation
echo 0. Back
echo.
set "AGY_EFFORT_CHOICE="
set /p "AGY_EFFORT_CHOICE=Choose an effort: "
if "%AGY_EFFORT_CHOICE%"=="1" call :set_agy_effort low
if "%AGY_EFFORT_CHOICE%"=="2" call :set_agy_effort medium
if "%AGY_EFFORT_CHOICE%"=="3" call :set_agy_effort high
if "%AGY_EFFORT_CHOICE%"=="0" goto agy_menu
if "%AGY_EFFORT_CHOICE%"=="1" goto menu_pause
if "%AGY_EFFORT_CHOICE%"=="2" goto menu_pause
if "%AGY_EFFORT_CHOICE%"=="3" goto menu_pause
echo Invalid choice.
goto menu_pause

:show_agy_choice
call :load_agy_model
call :load_agy_effort
echo Antigravity model: %AGY_MODEL%
echo Reasoning effort:  %AGY_EFFORT%
exit /b 0

:load_agy_model
set "AGY_MODEL=gemini-3.8-flash-medium"
if exist "%AGY_MODEL_FILE%" set /p "AGY_MODEL=" < "%AGY_MODEL_FILE%"
call :is_valid_agy_model "%AGY_MODEL%"
if errorlevel 1 set "AGY_MODEL=gemini-3.8-flash-medium"
exit /b 0

:load_agy_effort
set "AGY_EFFORT=medium"
if exist "%AGY_EFFORT_FILE%" set /p "AGY_EFFORT=" < "%AGY_EFFORT_FILE%"
call :is_valid_agy_effort "%AGY_EFFORT%"
if errorlevel 1 set "AGY_EFFORT=medium"
exit /b 0

:set_agy_model
call :is_valid_agy_model "%~1"
if errorlevel 1 (
    echo Unknown Antigravity model: %~1
    exit /b 1
)
> "%AGY_MODEL_FILE%" <nul set /p "=%~1"
set "AGY_MODEL=%~1"
echo Antigravity model: %AGY_MODEL%
call :show_agy_restart_hint
exit /b 0

:set_agy_effort
call :is_valid_agy_effort "%~1"
if errorlevel 1 (
    echo Unknown reasoning effort: %~1
    echo Use: low, medium, or high
    exit /b 1
)
> "%AGY_EFFORT_FILE%" <nul set /p "=%~1"
set "AGY_EFFORT=%~1"
echo Reasoning effort: %AGY_EFFORT%
call :show_agy_restart_hint
exit /b 0

:show_agy_restart_hint
call :find_bot_pids
if defined BOT_PIDS echo The bot is running. Restart it to use the new Antigravity settings.
exit /b 0

:is_valid_agy_model
if /I "%~1"=="gemini-3.8-flash-high" exit /b 0
if /I "%~1"=="gemini-3.8-flash-medium" exit /b 0
if /I "%~1"=="gemini-3.8-flash-low" exit /b 0
if /I "%~1"=="gemini-3.7-flash-high" exit /b 0
if /I "%~1"=="gemini-3.7-flash-medium" exit /b 0
if /I "%~1"=="gemini-3.7-flash-low" exit /b 0
if /I "%~1"=="gemini-3.6-flash-high" exit /b 0
if /I "%~1"=="gemini-3.6-flash-medium" exit /b 0
if /I "%~1"=="gemini-3.6-flash-low" exit /b 0
if /I "%~1"=="gemini-3.1-pro-high" exit /b 0
if /I "%~1"=="gemini-3.1-pro-low" exit /b 0
if /I "%~1"=="claude-sonnet-4-6" exit /b 0
if /I "%~1"=="claude-opus-4-6-thinking" exit /b 0
if /I "%~1"=="gpt-oss-120b-medium" exit /b 0
exit /b 1

:is_valid_agy_effort
if /I "%~1"=="low" exit /b 0
if /I "%~1"=="medium" exit /b 0
if /I "%~1"=="high" exit /b 0
exit /b 1

:restart
call :find_bot_pids
if not defined BOT_PIDS (
    if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
    if exist "%MODE_FILE%" del /q "%MODE_FILE%" >nul 2>&1
    echo Bot is not running. Start it normally or in DEV mode first.
    goto action_done
)

call :resolve_bot_mode
if /I "%BOT_MODE%"=="MIXED" (
    echo Cannot restart: normal and DEV bot processes are running together.
    echo Stop them, then start the required mode explicitly.
    goto action_done
)
if /I "%BOT_MODE%"=="UNKNOWN" (
    echo Cannot restart: the current bot mode could not be determined.
    echo Stop it, then start the required mode explicitly.
    goto action_done
)

set "RESTART_PIDS=%BOT_PIDS%"
set "RESTART_MODE=%BOT_MODE%"
for %%P in (%RESTART_PIDS%) do (
    rem Do not use /T here: when restart is requested by the embedded web
    rem panel this controller is itself a child of MilanaService. Killing the
    rem whole tree would kill the controller before it can start the new copy.
    taskkill /PID %%P /F >nul 2>&1
)
for /f "delims=" %%H in ('%PS% -NoProfile -Command "$processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue; foreach ($process in $processes) { if ($process.CommandLine -and $process.CommandLine -match '(?i)telegram_skill_host\.py') { $process.ProcessId } }"') do (
    taskkill /PID %%H /F >nul 2>&1
)
if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
if exist "%MODE_FILE%" del /q "%MODE_FILE%" >nul 2>&1
echo Bot stopped for restart. PID(s):%RESTART_PIDS%
if /I "%RESTART_MODE%"=="DEV" goto start_dev
goto start

:stop
call :find_bot_pids
if not defined BOT_PIDS (
    if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
    if exist "%MODE_FILE%" del /q "%MODE_FILE%" >nul 2>&1
    echo Bot is not running.
    echo.
    call :show_full_status
    goto action_done
)

for %%P in (%BOT_PIDS%) do (
    taskkill /PID %%P /T /F >nul 2>&1
)
del /q "%PID_FILE%" >nul 2>&1
if exist "%MODE_FILE%" del /q "%MODE_FILE%" >nul 2>&1
echo Bot stopped. PID(s):%BOT_PIDS%
echo.
call :show_full_status
goto action_done

:status
call :show_full_status
goto action_done

:show_full_status
set "BOT_MODE=OFF"
call :find_bot_pids
if defined BOT_PIDS call :resolve_bot_mode
if not defined BOT_PIDS (
    if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
    if exist "%MODE_FILE%" del /q "%MODE_FILE%" >nul 2>&1
    echo ============================================================
    echo MILANA: NOT RUNNING
    echo ============================================================
) else (
    echo ============================================================
    echo MILANA: RUNNING. PIDs:%BOT_PIDS%
    echo ============================================================
    for %%P in (%BOT_PIDS%) do call :show_process_details %%P
    call :show_bot_mode
)
call :show_llm_choice
if /I "%LLM_CHOICE%"=="gemini" call :show_agy_choice
echo.
call :show_log_status
echo.
call :show_detailed_state
exit /b 0

:load_llm_choice
set "LLM_CHOICE=openai"
if not exist "%LLM_FILE%" exit /b 0
set "SAVED_LLM_CHOICE="
set /p "SAVED_LLM_CHOICE=" < "%LLM_FILE%"
if /I "%SAVED_LLM_CHOICE%"=="gemini" set "LLM_CHOICE=gemini"
if /I "%SAVED_LLM_CHOICE%"=="lmstudio" set "LLM_CHOICE=lmstudio"
exit /b 0

:show_llm_choice
call :load_llm_choice
if /I "%LLM_CHOICE%"=="gemini" (
    echo Configured LLM: Antigravity ^(agy CLI^)
) else if /I "%LLM_CHOICE%"=="lmstudio" (
    echo Configured LLM: LM Studio - local model configured in ai_config.json
) else (
    echo Configured LLM: OpenAI - model configured in ai_config.json
)
exit /b 0

:set_llm_choice
call :load_llm_choice
set "PREVIOUS_LLM_CHOICE=%LLM_CHOICE%"
> "%LLM_FILE%" <nul set /p "=%~1"
set "LLM_CHOICE=%~1"
call :show_llm_choice
if /I "%PREVIOUS_LLM_CHOICE%"=="%LLM_CHOICE%" exit /b 0
call :find_bot_pids
if defined BOT_PIDS echo The bot is running. Restart it to use the newly selected LLM.
exit /b 0

:resolve_bot_mode
set "BOT_MODE=UNKNOWN"
set "SAVED_MODE_PID="
set "SAVED_MODE_VALUE="
if exist "%MODE_FILE%" for /f "usebackq tokens=1,2" %%A in ("%MODE_FILE%") do (
    set "SAVED_MODE_PID=%%A"
    set "SAVED_MODE_VALUE=%%B"
)
set "MODE_PID_MATCH="
for %%P in (%BOT_PIDS%) do call :match_mode_pid %%P
if defined MODE_PID_MATCH if /I "%SAVED_MODE_VALUE%"=="DEV" (
    set "BOT_MODE=DEV"
    exit /b 0
)
if defined MODE_PID_MATCH if /I "%SAVED_MODE_VALUE%"=="NORMAL" (
    set "BOT_MODE=NORMAL"
    exit /b 0
)
set "FOUND_DEV="
set "FOUND_NORMAL="
for %%P in (%BOT_PIDS%) do call :detect_process_mode %%P
if defined FOUND_DEV if defined FOUND_NORMAL set "BOT_MODE=MIXED"
if defined FOUND_DEV if not defined FOUND_NORMAL set "BOT_MODE=DEV"
if defined FOUND_NORMAL if not defined FOUND_DEV set "BOT_MODE=NORMAL"
exit /b 0

:match_mode_pid
if "%~1"=="%SAVED_MODE_PID%" set "MODE_PID_MATCH=1"
exit /b 0

:detect_process_mode
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-CimInstance Win32_Process -Filter 'ProcessId = %~1' -ErrorAction SilentlyContinue; if (-not $p -or [string]::IsNullOrWhiteSpace($p.CommandLine)) { exit 2 }; if ($p.CommandLine -match '(?i)(?:^|\s)--dev-chat(?:\s|$)') { exit 0 }; exit 1" >nul 2>&1
if errorlevel 2 exit /b 0
if errorlevel 1 (
    set "FOUND_NORMAL=1"
    exit /b 0
)
set "FOUND_DEV=1"
exit /b 0

:show_detailed_state
echo ---------------- CURRENT DETAILED STATE ----------------
if /I "%BOT_MODE%"=="DEV" (
    echo DEV CHAT: ACTIVE
    echo Replies are generated immediately. Artificial response/presence delays are disabled; reflective heartbeat starts paused.
    exit /b 0
)
if /I "%BOT_MODE%"=="MIXED" echo WARNING: normal and DEV bot processes are running together.
if /I "%BOT_MODE%"=="UNKNOWN" echo WARNING: running bot mode could not be determined; schedule is shown for reference only.
if not exist "%PYTHON%" (
    echo State unavailable: Python environment not found: %PYTHON%
) else if not exist "%SCHEDULE_SCRIPT%" (
    echo State unavailable: schedule script not found: %SCHEDULE_SCRIPT%
) else (
    "%PYTHON%" "%SCHEDULE_SCRIPT%"
    if errorlevel 1 echo Failed to read Milana state.
)
exit /b 0

:show_bot_mode
if /I "%BOT_MODE%"=="DEV" (
    echo Mode: DEV CHAT - immediate replies, schedule bypassed
) else if /I "%BOT_MODE%"=="NORMAL" (
    echo Mode: NORMAL - schedule enabled
) else if /I "%BOT_MODE%"=="MIXED" (
    echo Mode: MIXED - normal and DEV processes detected
) else (
    echo Mode: UNKNOWN - command line is unavailable
)
exit /b 0

:show_process_details
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-Process -Id %~1 -ErrorAction SilentlyContinue; if (-not $p) { Write-Host 'Process details unavailable.'; exit }; $now = Get-Date; $uptime = $now - $p.StartTime; $uptimeText = if ($uptime.Days -gt 0) { '{0} d {1:00}:{2:00}:{3:00}' -f $uptime.Days,$uptime.Hours,$uptime.Minutes,$uptime.Seconds } else { '{0:00}:{1:00}:{2:00}' -f ([int]$uptime.TotalHours),$uptime.Minutes,$uptime.Seconds }; Write-Host ('Process:      {0} (PID {1})' -f $p.ProcessName,$p.Id); Write-Host ('Started:      {0:dd.MM.yyyy HH:mm:ss}' -f $p.StartTime); Write-Host ('Uptime:       {0}' -f $uptimeText); Write-Host ('CPU time:     {0:N1} sec' -f $p.CPU); Write-Host ('Memory:       {0:N1} MB RAM' -f ($p.WorkingSet64 / 1MB)); Write-Host ('Threads:      {0}' -f $p.Threads.Count)"
exit /b 0

:show_log_status
"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "$out = Get-Item -LiteralPath '%OUT_LOG%' -ErrorAction SilentlyContinue; $err = Get-Item -LiteralPath '%ERR_LOG%' -ErrorAction SilentlyContinue; Write-Host 'Logs:'; if ($out) { Write-Host ('  Output: {0:N0} bytes, updated {1:dd.MM.yyyy HH:mm:ss}' -f $out.Length,$out.LastWriteTime) } else { Write-Host '  Output: not created yet' }; if ($err) { $label = if ($err.Length -gt 0) { 'HAS ERRORS' } else { 'empty (no recorded errors)' }; Write-Host ('  Errors: {0:N0} bytes, updated {1:dd.MM.yyyy HH:mm:ss} - {2}' -f $err.Length,$err.LastWriteTime,$label) } else { Write-Host '  Errors: not created yet' }; if ($out -and $out.Length -gt 0) { Write-Host 'Recent events:'; Get-Content -Encoding UTF8 -LiteralPath $out.FullName -Tail 3 | ForEach-Object { Write-Host ('  ' + $_) } }; if ($err -and $err.Length -gt 0) { Write-Host 'Recent errors:'; Get-Content -Encoding UTF8 -LiteralPath $err.FullName -Tail 5 | ForEach-Object { Write-Host ('  ' + $_) } }"
exit /b 0

:logs
if exist "%ERR_LOG%" (
    echo --- Errors ---
    "%PS%" -NoProfile -Command "Get-Content -Encoding utf8 -LiteralPath '%ERR_LOG%' -Tail 20"
)
if exist "%OUT_LOG%" (
    echo --- Output ---
    "%PS%" -NoProfile -Command "Get-Content -Encoding utf8 -LiteralPath '%OUT_LOG%' -Tail 20"
)
if not exist "%ERR_LOG%" if not exist "%OUT_LOG%" echo No logs yet.
goto action_done

:open_web
call :find_bot_pids
if not defined BOT_PIDS (
    echo MilanaService is not running. Starting it now...
    call "%~f0" start
    call :find_bot_pids
    if not defined BOT_PIDS (
        echo Could not start MilanaService. Check bot-error.log.
        goto action_done
    )
)

rem The web panel is embedded in MilanaService. This command only opens it.
start "" "http://127.0.0.1:8765/"

echo Opened: http://127.0.0.1:8765/
goto action_done

:read_pid
set "BOT_PID="
set "BOT_START_TICKS="
if exist "%PID_FILE%" for /f "usebackq tokens=1,2" %%A in ("%PID_FILE%") do (
    set "BOT_PID=%%A"
    set "BOT_START_TICKS=%%B"
)
exit /b 0

:cleanup_failed_start
call :read_pid
call :is_running %BOT_PID% %BOT_START_TICKS%
if not errorlevel 1 taskkill /PID %BOT_PID% /T /F >nul 2>&1
if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
if exist "%MODE_FILE%" del /q "%MODE_FILE%" >nul 2>&1
exit /b 0

:is_running
if "%~1"=="" exit /b 1
"%PS%" -NoProfile -Command "$p = Get-Process -Id %~1 -ErrorAction SilentlyContinue; if (-not $p -or $p.ProcessName -notmatch '^pythonw?$') { exit 1 }; $expected = '%~2'; if ($expected -match '^[0-9]+$') { if ($p.StartTime.ToUniversalTime().ToFileTimeUtc().ToString() -eq $expected) { exit 0 }; exit 1 }; $w = Get-CimInstance Win32_Process -Filter 'ProcessId = %~1' -ErrorAction SilentlyContinue; $script = [IO.Path]::GetFullPath($env:SCRIPT); if ($w.CommandLine -and $w.CommandLine.IndexOf($script, [StringComparison]::OrdinalIgnoreCase) -ge 0) { exit 0 }; exit 1" >nul 2>&1
exit /b %errorlevel%

:find_bot_pids
setlocal EnableDelayedExpansion
set "FOUND_PIDS="

rem Prefer the PID recorded when this controller started the bot. Reading the
rem command line through CIM can fail for a perfectly healthy process.
if exist "%PID_FILE%" (
    set "SAVED_PID="
    set "SAVED_START_TICKS="
    for /f "usebackq tokens=1,2" %%A in ("%PID_FILE%") do (
        set "SAVED_PID=%%A"
        set "SAVED_START_TICKS=%%B"
    )
    echo(!SAVED_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if not errorlevel 1 (
        call :is_running !SAVED_PID! !SAVED_START_TICKS!
        if not errorlevel 1 set "FOUND_PIDS= !SAVED_PID!"
    )
)

rem Fall back to discovery for bots started outside this controller.
if not defined FOUND_PIDS (
for /f "delims=" %%P in ('%PS% -NoProfile -Command "$script = [IO.Path]::GetFullPath($env:SCRIPT); $processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue; foreach ($process in $processes) { if ($process.CommandLine -and $process.CommandLine.IndexOf($script, [StringComparison]::OrdinalIgnoreCase) -ge 0) { $process.ProcessId } }"') do (
    set "FOUND_PIDS=!FOUND_PIDS! %%P"
)
)
endlocal & set "BOT_PIDS=%FOUND_PIDS%"
exit /b 0

:menu_pause
if defined INTERACTIVE pause
goto menu

:action_done
if defined INTERACTIVE goto menu_pause
exit /b 0

:done
exit /b 0
