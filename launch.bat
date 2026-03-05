@echo off
cd /d "%~dp0"

echo AutoRegCT - Setup and Launch
echo ==============================

if not exist "venv\Scripts\python.exe" (
    echo Setting up virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Python not found. Make sure Python 3.10+ is installed.
        pause
        exit /b 1
    )
    echo Installing dependencies - this will take a few minutes...
    venv\Scripts\pip install -r requirements.txt --quiet
    echo Done.
)

echo Starting AutoRegCT...
set PYTHONPATH=%~dp0
venv\Scripts\python -m streamlit run app.py --server.headless false

pause
