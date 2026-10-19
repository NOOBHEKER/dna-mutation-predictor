@echo off
setlocal EnableExtensions EnableDelayedExpansion
title DNA Mutation Predictor - Web

rem ===========================================================
rem  Portable Streamlit launcher (Windows)
rem  Uses the same venv as predict.bat.
rem ===========================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY goto :no_python

set "VENV=%ROOT%venv"
set "VPY=%VENV%\Scripts\python.exe"
if not exist "%VPY%" (
    echo.
    echo   [setup] Creating virtual environment...
    %PY% -m venv "%VENV%"
    if errorlevel 1 goto :venv_failed
)

if not exist "%VENV%\.dna-predict-web-installed" (
    echo.
    echo   [setup] Installing web dependencies ^(first run^)...
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install -r "%ROOT%requirements-web.txt"
    if errorlevel 1 goto :install_failed
    echo ok > "%VENV%\.dna-predict-web-installed"
)

echo.
echo   Starting dashboard at http://localhost:8501
echo   Press Ctrl+C to stop.
echo.
"%VPY%" -m streamlit run "%ROOT%webapp\app.py" %*
set "EXITCODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXITCODE%

:no_python
echo.
echo   [error] Python 3.10+ was not found on PATH.
echo          Install it from https://www.python.org/downloads/
echo          and tick "Add Python to PATH", then re-run start-web.bat.
echo.
pause
exit /b 1

:venv_failed
echo.
echo   [error] Could not create the virtual environment.
echo.
pause
exit /b 1

:install_failed
echo.
echo   [error] Web dependency installation failed.
echo          Try manually:
echo            "%VPY%" -m pip install -r "%ROOT%requirements-web.txt"
echo.
pause
exit /b 1
