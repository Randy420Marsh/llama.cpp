@echo off
setlocal
cd /d "%~dp0"

rem ===== whisper-asr-server launcher (Windows) ==========================
rem First run: creates the uv venv (Python 3.11) and installs requirements
rem (faster-whisper + CUDA cuBLAS/cuDNN wheels). Later runs: skip the install
rem (stamp file) and start the OpenAI-compatible transcription server.
rem   run.bat --setup-only        ...do the setup steps and exit (no launch)
rem   run.bat --port 8178 --idle-unload 300   ...extra args go to asr_server.py

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv not found on PATH. Install: https://docs.astral.sh/uv/
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [Setup] Creating Python 3.11 venv with uv...
    uv venv venv --python 3.11 || exit /b 1
)

call venv\Scripts\activate.bat

rem ===== Requirements: install only on first run, when requirements.txt changes,
rem or when the stamp exists but the packages are missing (interrupted install)
fc /b requirements.txt "venv\.requirements.stamp" >nul 2>nul
if errorlevel 1 goto :reqs_install
python -c "import importlib.util as u, sys; sys.exit(any(u.find_spec(m) is None for m in ('faster_whisper', 'fastapi', 'uvicorn', 'multipart')))" >nul 2>nul
if not errorlevel 1 goto :reqs_ok
echo [Setup] Packages missing from venv - reinstalling...
:reqs_install
echo [Setup] Installing requirements (first run / requirements.txt changed)...
uv pip install -r requirements.txt || exit /b 1
copy /y requirements.txt "venv\.requirements.stamp" >nul
:reqs_ok
echo [Setup] venv and requirements OK.

if /i "%~1"=="--setup-only" (
    echo [OK] Setup complete - venv and requirements ready.
    exit /b 0
)

echo [Start] asr_server.py %*
set PYTHONUNBUFFERED=1
python -u asr_server.py %*
