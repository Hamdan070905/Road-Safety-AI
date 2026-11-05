@echo off
setlocal
cd /d "%~dp0"

echo Checking for existing processes on port 8000...
for /f "tokens=5" %%a in ('netstat -ano -p tcp ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Freeing port 8000 held by PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

if exist ".venv312\Scripts\python.exe" (
    call .venv312\Scripts\activate.bat
) else if exist ".venv\Scripts\python.exe" (
    call .venv\Scripts\activate.bat
) else (
    echo No virtual environment found.
    pause
    exit /b 1
)

echo Starting RoadGuard AI server on http://127.0.0.1:8000 ...
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
