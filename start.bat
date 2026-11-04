@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv312" (
    echo Creating virtual environment with Python 3.12...
    py -3.12 -m venv .venv312
)
call .venv312\Scripts\activate.bat
python -m pip install -r requirements.txt
echo Starting RoadGuard AI server on http://127.0.0.1:8000 ...
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
