@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
set "SCHEDULE_SCRIPT=%ROOT%milana_schedule.py"

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
