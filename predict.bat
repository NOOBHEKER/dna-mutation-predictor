@echo off
setlocal
title dna-predict CLI Bot

rem ===========================================================
rem  dna-predict - portable launcher (Windows)
rem  Works no matter where the repo is cloned. Creates a local
rem  venv + installs deps on first run, then launches the bot.
rem ===========================================================

rem --- locate this script's directory (repo root) ---
set "ROOT=%~dp0"
cd /d "%ROOT%"

rem --- locate a Python interpreter ---
set "PY="
for %%G in (py python python3) do (
    where %%G >nul 2>nul && set "PY=%%G" && goto :found_py
)
goto :no_python
:found_py

rem --- create venv if missing ---
set "VENV=%ROOT%venv"
if not exist "%VENV%\Scripts\python.exe" (
    echo.
    echo   [setup] Creating virtual environment...
    "%PY%" -m venv "%VENV%"
    if errorlevel 1 goto :venv_failed
)

set "VPY=%VENV%\Scripts\python.exe"
set "VPIP=%VENV%\Scripts\python.exe -m pip"

rem --- auto-install dependencies on first run ---
if not exist "%VENV%\Scripts\marker.lib.txt" (
    echo.
    echo   [setup] Installing dependencies (first run - this may take a while)...
    "%VPIP%" install --upgrade pip >nul 2>nul
    "%VPIP%" install -r "%ROOT%requirements.txt"
    if errorlevel 1 goto :install_failed
    echo ok > "%VENV%\Scripts\marker.lib.txt"
)

rem --- warn if model/data files are missing ---
set "MISSING_DATA="
if not exist "%ROOT%data\scaler.pkl" set "MISSING_DATA=! scaler.pkl (trained model)"
if not exist "%ROOT%data\variants.csv" set "MISSING_DATA=%MISSING_DATA%  ! variants.csv (variant index)"
if defined MISSING_DATA (
    echo.
    echo   [warn] Some data files are missing:%MISSING_DATA%
    echo          Suggestions and gene info will be limited.
    echo          See README.md for how to obtain the model.
    echo.
)

rem --- launch ---
echo.
echo   Starting dna-predict CLI Bot...
echo.
"%VPY%" -m src.cli
set "EXIT=%ERRORLEVEL%"

echo.
pause
exit /b %EXIT%

rem ===========================================================
:no_python
echo.
echo   [error] Python 3 is required but was not found on PATH.
echo          Install it from https://www.python.org/downloads/
echo          and tick "Add Python to PATH", then re-run predict.bat
echo.
pause
exit /b 1

:venv_failed
echo.
echo   [error] Could not create virtual environment.
echo          Check that the "venv" module is available.
echo.
pause
exit /b 1

:install_failed
echo.
echo   [error] Dependency install failed. Try running manually:
echo          %VPY% -m pip install -r "%ROOT%requirements.txt"
echo.
pause
exit /b 1
