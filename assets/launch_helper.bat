@echo off
setlocal EnableDelayedExpansion
title Mood Music Recommender
cd /d "%~dp0.."

set "VENV_DIR=%LOCALAPPDATA%\MoodMusicRecommender\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" "%~dp0..\app.py" %*
    if !ERRORLEVEL! NEQ 0 pause
    exit /b 0
)

echo ====================================================================
echo          MOOD-BASED MUSIC RECOMMENDER - INITIAL SETUP
echo ====================================================================
echo [SETUP] Preparing Python environment and installing requirements...
if not exist "%LOCALAPPDATA%\MoodMusicRecommender" mkdir "%LOCALAPPDATA%\MoodMusicRecommender"

where uv >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    uv venv "%VENV_DIR%" --python 3.11
    if exist "%VENV_PYTHON%" goto :LAUNCH
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py -3.11 -m venv "%VENV_DIR%" 2>nul
    if exist "%VENV_PYTHON%" goto :LAUNCH
    py -3.10 -m venv "%VENV_DIR%" 2>nul
    if exist "%VENV_PYTHON%" goto :LAUNCH
    py -3 -m venv "%VENV_DIR%" 2>nul
    if exist "%VENV_PYTHON%" goto :LAUNCH
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -m venv "%VENV_DIR%"
    if exist "%VENV_PYTHON%" goto :LAUNCH
)

echo [ERROR] Python was not found. Please install Python 3.10 or 3.11.
pause
exit /b 1

:LAUNCH
"%VENV_PYTHON%" "%~dp0..\app.py" %*
if %ERRORLEVEL% NEQ 0 pause
endlocal
