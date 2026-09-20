@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Python environment not found. Set up .venv first.
    pause
    exit /b 1
)
"%PYTHON%" "%ROOT%check_agy.py"
set "CHECK_RESULT=%ERRORLEVEL%"
pause
exit /b %CHECK_RESULT%
