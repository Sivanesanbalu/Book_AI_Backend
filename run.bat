@echo off
REM ==============================================================================
REM 📚 BookNest AI — 1-Click Windows Server Runner
REM ==============================================================================
title BookNest AI Server

echo =================================================
echo   🚀 Starting BookNest AI — Visual Book Sanctuary
echo =================================================

if not exist data mkdir data
if not exist static\avatars mkdir static\avatars

echo 📦 Checking Python environment...
python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Installing required dependencies...
    pip install -r requirements.txt
)

echo ✨ Starting server on http://localhost:8000 ...
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
