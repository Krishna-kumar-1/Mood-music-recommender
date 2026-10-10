@echo off
setlocal EnableDelayedExpansion
title Mood-Based Music Recommender - One-Click Launcher
cd /d "%~dp0"

echo ====================================================================
echo          MOOD-BASED MUSIC RECOMMENDATION SYSTEM LAUNCHER
echo ====================================================================
echo.

:: Store virtual environment in LocalAppData to avoid OneDrive file lock issues
set "VENV_DIR=%LOCALAPPDATA%\MoodMusicRecommender\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

:: Step 1: Check if virtual environment already exists
if exist "%VENV_PYTHON%" (
    echo [INFO] Found existing Python environment.
    goto :RUN_APP
)

echo [SETUP] First-time setup detected! Preparing Python environment...
if not exist "%LOCALAPPDATA%\MoodMusicRecommender" mkdir "%LOCALAPPDATA%\MoodMusicRecommender"

:: Prefer uv with Python 3.11 for best MediaPipe/TensorFlow compatibility
where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [SETUP] Creating Python 3.11 virtual environment using uv...
    uv venv "%VENV_DIR%" --python 3.11
    if exist "%VENV_PYTHON%" goto :RUN_APP
)

:: Try Python Launcher with 3.11 or 3.10
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py -3.11 -m venv "%VENV_DIR%" 2>nul
    if exist "%VENV_PYTHON%" goto :RUN_APP
    py -3.10 -m venv "%VENV_DIR%" 2>nul
    if exist "%VENV_PYTHON%" goto :RUN_APP
    py -3 -m venv "%VENV_DIR%" 2>nul
    if exist "%VENV_PYTHON%" goto :RUN_APP
)

:: Fallback to system python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [SETUP] Creating virtual environment using system Python...
    python -m venv "%VENV_DIR%"
    if exist "%VENV_PYTHON%" goto :RUN_APP
)

echo [ERROR] Python is not installed or not found in PATH.
echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
pause
exit /b 1

:RUN_APP
:: Run the smart Python launcher (verifies/installs requirements & starts the web app)
"%VENV_PYTHON%" "%~dp0run.py" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Application exited with an error. Check messages above.
    pause
)
endlocal
