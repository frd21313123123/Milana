@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "SSH=%SystemRoot%\System32\OpenSSH\ssh.exe"
if not exist "%SSH%" set "SSH=ssh"
set "SERVER=root@ocenochka-server"

echo.
echo === 1/2: Telegram ===
echo Enter your phone number, Telegram code, and 2FA password only in the SSH window.
echo Do not paste those values into chat.
echo.
"%SSH%" -tt %SERVER% "cd /opt/milana && /opt/milana/.venv/bin/python /opt/milana/telegram_client.py login"
if errorlevel 1 (
    echo.
    echo Telegram login did not finish. Fix the error and run this file again.
    pause
    exit /b 1
)

echo.
echo === 2/2: Google Antigravity ===
echo Open the URL shown by Antigravity in your browser, sign in to Google,
echo and paste the one-time code back into the SSH window. Exit agy after login.
echo.
"%SSH%" -tt %SERVER% "/root/.local/bin/agy"

echo.
echo After both logins finish, return to Codex and say: ready
pause
