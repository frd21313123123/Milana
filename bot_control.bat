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
call :show_schedule
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

call :read_pid
if defined BOT_PID (
    call :is_running %BOT_PID%
    if not errorlevel 1 (
        echo Bot is already running. PID: %BOT_PID%
        goto done
    )
    del /q "%PID_FILE%" >nul 2>&1
)

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
call :show_schedule
goto done

:stop
call :read_pid
if not defined BOT_PID (
    echo Bot is not running under this control file.
    echo.
    call :show_schedule
    goto done
)

call :is_running %BOT_PID%
if errorlevel 1 (
    echo Stale PID file removed.
    del /q "%PID_FILE%" >nul 2>&1
    echo.
    call :show_schedule
    goto done
)

taskkill /PID %BOT_PID% /T /F >nul 2>&1
if errorlevel 1 (
    echo Failed to stop PID %BOT_PID%.
    goto done
)
del /q "%PID_FILE%" >nul 2>&1
echo Bot stopped.
echo.
call :show_schedule
goto done

:status
call :read_pid
if not defined BOT_PID (
    echo Bot status: stopped.
    echo.
    call :show_schedule
    goto done
)

call :is_running %BOT_PID%
if errorlevel 1 (
    echo Bot status: stopped ^(stale PID file removed^).
    del /q "%PID_FILE%" >nul 2>&1
    echo.
    call :show_schedule
    goto done
)
echo Bot status: running. PID: %BOT_PID%
echo.
call :show_schedule
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

:show_schedule
if not exist "%PYTHON%" (
    echo Schedule unavailable: Python environment not found.
    exit /b 0
)
if not exist "%SCHEDULE_SCRIPT%" (
    echo Schedule unavailable: %SCHEDULE_SCRIPT%
    exit /b 0
)
"%PYTHON%" "%SCHEDULE_SCRIPT%" --brief
if errorlevel 1 echo Failed to read Milana schedule.
exit /b 0

:menu_pause
if defined INTERACTIVE pause
goto menu

:done
if defined INTERACTIVE pause
exit /b 0
