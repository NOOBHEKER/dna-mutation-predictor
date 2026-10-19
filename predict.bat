@echo off
setlocal EnableExtensions EnableDelayedExpansion
title DNA Mutation Predictor - CLI

rem ===========================================================
rem  Portable launcher (Windows)
rem  Creates a local venv, installs dependencies on first run,
rem  then starts the interactive CLI. Works from any location.
rem ===========================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"

rem --- locate Python ---
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY goto :no_python

rem --- create venv ---
set "VENV=%ROOT%venv"
set "VPY=%VENV%\Scripts\python.exe"
if not exist "%VPY%" (
    echo.
    echo   [setup] Creating virtual environment...
    %PY% -m venv "%VENV%"
    if errorlevel 1 goto :venv_failed
)

rem --- install dependencies once ---
if not exist "%VENV%\.dna-predict-installed" (
    echo.
    echo   [setup] Installing dependencies ^(first run, this can take a while^)...
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install -r "%ROOT%requirements.txt"
    if errorlevel 1 goto :install_failed
    echo ok > "%VENV%\.dna-predict-installed"
)

rem --- warn about missing data ---
set "MISSING="
if not exist "%ROOT%data\scaler.pkl"    set "MISSING=!MISSING!  - scaler.pkl (trained head)"
if not exist "%ROOT%data\variants.csv"  set "MISSING=!MISSING!  - variants.csv (variant index)"
if not exist "%ROOT%data\tabular.pkl"   set "MISSING=!MISSING!  - tabular.pkl (population head)"
if defined MISSING (
    echo.
    echo   [warn] Missing data files:
    echo          !MISSING!
    echo          Predictions will be limited. See data/README.md.
    echo.
)

echo.
echo   Starting DNA Mutation Predictor...
echo.
"%VPY%" -m src.cli %*
set "EXITCODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXITCODE%

:no_python
echo.
echo   [error] Python 3.10+ was not found on PATH.
echo          Install it from https://www.python.org/downloads/
echo          and tick "Add Python to PATH", then re-run predict.bat.
echo.
pause
exit /b 1

:venv_failed
echo.
echo   [error] Could not create the virtual environment.
echo          Ensure the "venv" module is available.
echo.
pause
exit /b 1

:install_failed
echo.
echo   [error] Dependency installation failed.
echo          Try manually:
echo            "%VPY%" -m pip install -r "%ROOT%requirements.txt"
echo.
pause
exit /b 1
