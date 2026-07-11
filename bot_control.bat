@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SCRIPT=%ROOT%telegram_client.py"
set "SCHEDULE_SCRIPT=%ROOT%milana_schedule.py"
set "PID_FILE=%ROOT%bot.pid"
set "OUT_LOG=%ROOT%bot-output.log"
set "ERR_LOG=%ROOT%bot-error.log"
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"

if "%~1"=="" (
    set "INTERACTIVE=1"
    goto menu
)

if /I "%~1"=="start" goto start
if /I "%~1"=="stop" goto stop
if /I "%~1"=="status" goto status
if /I "%~1"=="logs" goto logs

echo Unknown command: %~1
echo Use: bot_control.bat [start^|stop^|status^|logs]
exit /b 2

:menu
cls
title Milana AI control
echo Milana AI control
echo.
call :show_full_status
echo.
echo 1. Start bot
echo 2. Stop bot
echo 3. Show status
echo 4. Show recent logs
echo 0. Exit
echo.
set /p "CHOICE=Choose an action: "
if "%CHOICE%"=="1" goto start
if "%CHOICE%"=="2" goto stop
if "%CHOICE%"=="3" goto status
if "%CHOICE%"=="4" goto logs
if "%CHOICE%"=="0" goto done
echo Invalid choice.
goto menu_pause

:start
if not exist "%PYTHON%" (
    echo Python environment not found: %PYTHON%
    goto action_done
)
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

"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $pathValue = [Environment]::GetEnvironmentVariable('Path'); [Environment]::SetEnvironmentVariable('PATH', $null); [Environment]::SetEnvironmentVariable('Path', $pathValue); $p = Start-Process -FilePath '%PYTHON%' -ArgumentList '-u', '%SCRIPT%', 'ai-bot' -WorkingDirectory '%ROOT%' -WindowStyle Hidden -RedirectStandardOutput '%OUT_LOG%' -RedirectStandardError '%ERR_LOG%' -PassThru; Set-Content -NoNewline -Encoding ascii -Path '%PID_FILE%' -Value $p.Id"
if errorlevel 1 (
    echo Failed to start the bot. Check PowerShell availability.
    goto action_done
)

call :read_pid
call :is_running %BOT_PID%
if errorlevel 1 (
    echo Bot process exited during startup. Recent errors:
    if exist "%ERR_LOG%" "%PS%" -NoProfile -Command "Get-Content -Encoding utf8 -LiteralPath '%ERR_LOG%' -Tail 20"
    if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
    goto action_done
)
echo Bot started. PID: %BOT_PID%
echo Output log: %OUT_LOG%
echo Error log: %ERR_LOG%
echo.
call :show_full_status
goto action_done

:stop
call :find_bot_pids
if not defined BOT_PIDS (
    if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
    echo Bot is not running.
    echo.
    call :show_full_status
    goto action_done
)

for %%P in (%BOT_PIDS%) do (
    taskkill /PID %%P /T /F >nul 2>&1
)
del /q "%PID_FILE%" >nul 2>&1
echo Bot stopped. PID(s):%BOT_PIDS%
echo.
call :show_full_status
goto action_done

:status
call :show_full_status
goto action_done

:show_full_status
call :find_bot_pids
if not defined BOT_PIDS (
    if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
    echo ============================================================
    echo MILANA: NOT RUNNING
    echo ============================================================
) else (
    echo ============================================================
    echo MILANA: RUNNING. PIDs:%BOT_PIDS%
    echo ============================================================
    for %%P in (%BOT_PIDS%) do call :show_process_details %%P
)
echo.
call :show_log_status
echo.
echo ---------------- CURRENT DETAILED STATE ----------------
if not exist "%PYTHON%" (
    echo State unavailable: Python environment not found: %PYTHON%
) else if not exist "%SCHEDULE_SCRIPT%" (
    echo State unavailable: schedule script not found: %SCHEDULE_SCRIPT%
) else (
    "%PYTHON%" "%SCHEDULE_SCRIPT%"
    if errorlevel 1 echo Failed to read Milana state.
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

:read_pid
set "BOT_PID="
if exist "%PID_FILE%" set /p "BOT_PID=" < "%PID_FILE%"
exit /b 0

:is_running
if "%~1"=="" exit /b 1
"%PS%" -NoProfile -Command "$p = Get-Process -Id %~1 -ErrorAction SilentlyContinue; if ($p -and $p.ProcessName -match '^pythonw?$') { exit 0 }; exit 1" >nul 2>&1
exit /b %errorlevel%

:find_bot_pids
setlocal EnableDelayedExpansion
set "FOUND_PIDS="

rem Prefer the PID recorded when this controller started the bot. Reading the
rem command line through CIM can fail for a perfectly healthy process.
if exist "%PID_FILE%" (
    set "SAVED_PID="
    set /p "SAVED_PID=" < "%PID_FILE%"
    echo(!SAVED_PID!| findstr /r "^[0-9][0-9]*$" >nul
    if not errorlevel 1 (
        "%PS%" -NoProfile -Command "$p = Get-Process -Id !SAVED_PID! -ErrorAction SilentlyContinue; if ($p -and $p.ProcessName -match '^pythonw?$') { exit 0 }; exit 1" >nul 2>&1
        if not errorlevel 1 set "FOUND_PIDS= !SAVED_PID!"
    )
)

rem Fall back to discovery for bots started outside this controller.
if not defined FOUND_PIDS (
for /f "delims=" %%P in ('%PS% -NoProfile -Command "$processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue; foreach ($process in $processes) { if ($process.CommandLine -and $process.CommandLine -match '(?i)telegram_client\.py' -and $process.CommandLine -match '(?i)ai-bot') { $process.ProcessId } }"') do (
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
