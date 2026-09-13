@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo venv missing. First-time setup:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -e .
    pause
    exit /b 1
)
".venv\Scripts\python.exe" "dashboard\server.py"
pause
