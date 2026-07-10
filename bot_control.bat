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
call "%ROOT%show_schedule.bat"
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
    goto done
)
if not exist "%SCRIPT%" (
    echo Bot script not found: %SCRIPT%
    goto done
)

call :find_bot_pids
if defined BOT_PIDS (
    echo Bot is already running. PID(s):%BOT_PIDS%
    goto done
)
if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1

"%PS%" -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath '%PYTHON%' -ArgumentList '-u', '%SCRIPT%', 'ai-bot' -WorkingDirectory '%ROOT%' -WindowStyle Hidden -RedirectStandardOutput '%OUT_LOG%' -RedirectStandardError '%ERR_LOG%' -PassThru; Set-Content -NoNewline -Encoding ascii -Path '%PID_FILE%' -Value $p.Id"
if errorlevel 1 (
    echo Failed to start the bot. Check PowerShell availability.
    goto done
)

call :read_pid
echo Bot started. PID: %BOT_PID%
echo Output log: %OUT_LOG%
echo Error log: %ERR_LOG%
echo.
call "%ROOT%show_schedule.bat"
goto done

:stop
call :find_bot_pids
if not defined BOT_PIDS (
    if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
    echo Bot is not running.
    echo.
    call "%ROOT%show_schedule.bat"
    goto done
)

for %%P in (%BOT_PIDS%) do (
    taskkill /PID %%P /T /F >nul 2>&1
)
del /q "%PID_FILE%" >nul 2>&1
echo Bot stopped. PID(s):%BOT_PIDS%
echo.
call "%ROOT%show_schedule.bat"
goto done

:status
call :find_bot_pids
if not defined BOT_PIDS (
    if exist "%PID_FILE%" del /q "%PID_FILE%" >nul 2>&1
    echo Bot status: stopped.
    echo.
    call "%ROOT%show_schedule.bat"
    goto done
)
echo Bot status: running. PID(s):%BOT_PIDS%
echo.
call "%ROOT%show_schedule.bat"
goto done

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
goto done

:read_pid
set "BOT_PID="
if exist "%PID_FILE%" set /p "BOT_PID=" < "%PID_FILE%"
exit /b 0

:is_running
"%PS%" -NoProfile -Command "if (Get-Process -Id %~1 -ErrorAction SilentlyContinue) { exit 0 }; exit 1" >nul 2>&1
exit /b %errorlevel%

:find_bot_pids
setlocal EnableDelayedExpansion
set "FOUND_PIDS="
for /f "delims=" %%P in ('%PS% -NoProfile -Command "$processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue; foreach ($process in $processes) { if ($process.CommandLine -and $process.CommandLine -match '(?i)telegram_client\.py' -and $process.CommandLine -match '(?i)ai-bot') { $process.ProcessId } }"') do (
    set "FOUND_PIDS=!FOUND_PIDS! %%P"
)
endlocal & set "BOT_PIDS=%FOUND_PIDS%"
exit /b 0

:menu_pause
if defined INTERACTIVE pause
goto menu

:done
if defined INTERACTIVE pause
exit /b 0
