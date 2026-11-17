@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Nectivon Release - Start

cd /d "%~dp0"

rem ---- Locate Python interpreter ----
set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%P"
)
if not defined PYTHON_EXE (
    echo [ERROR] Python interpreter not found.
    echo Expected: .venv\Scripts\python.exe or Python 3.11+ in PATH.
    echo Working directory: %CD%
    pause
    exit /b 3
)

rem ---- Verify Python version (>= 3.11) ----
"%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Nectivon requires Python 3.11 or newer.
    echo Current interpreter: %PYTHON_EXE%
    pause
    exit /b 3
)

rem ---- Verify runtime dependencies ----
"%PYTHON_EXE%" -c "import streamlit, fitz, PIL, dotenv, pydantic_settings, jieba, rapidfuzz" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Runtime dependencies are incomplete.
    echo Install with: "%PYTHON_EXE%" -m pip install -r "%~dp0requirements\requirements.txt"
    pause
    exit /b 5
)

rem ---- Launch release instance (formal config, port 8501) ----
rem Post-exam entry was explicitly authorized for the formal deployment.
rem This enables the page, not automatic library creation or AI calls.
if not defined EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED set "EKB_STUDY_LOOP_ACCEPTANCE_CONFIRMED=yes"
if not defined EKB_STUDY_LOOP_ROOT for %%I in ("%~dp0..\Nectivon-Study-Library") do set "EKB_STUDY_LOOP_ROOT=%%~fI"
set "NECTIVON_RUNTIME_ROOT=%~dp0"
set "NECTIVON_PYTHON=%PYTHON_EXE%"
set "NECTIVON_ACTION=start"
set "NECTIVON_INSTANCE=stable"
call "%~dp0_nectivon_launcher.cmd"

endlocal & exit /b %ERRORLEVEL%
