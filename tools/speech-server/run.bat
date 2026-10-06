@echo off
setlocal
cd /d "%~dp0"

rem ===== Speech server for the llama.cpp web UI (Windows) ================
rem Text-to-speech, your own (cloned) voices, model downloads, speech-to-text.
rem First run: creates the uv venv and installs requirements. Later runs:
rem skip all that (stamp file) and start the server immediately.
rem   run.bat --no-https --lan    ...what the command center runs: http, LAN, port 8179
rem   run.bat                     ...https on 127.0.0.1:8179 (TLS default)
rem   run.bat --setup-only        ...set up, then exit
rem   run.bat --port 8179 --engine piper   ...forwarded to server.py
rem GPU engines (Kokoro, Chatterbox, Orpheus): run-tts-setup.bat once.
rem The llama.cpp engine needs no Python packages, only llama-tts-server.exe.

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

rem ===== Requirements: install only on first run or when requirements.txt changes
fc /b requirements.txt "venv\.requirements.stamp" >nul 2>nul
if not errorlevel 1 goto :reqs_ok
echo [Setup] Installing requirements (first run / requirements.txt changed)...
uv pip install -r requirements.txt || exit /b 1
copy /y requirements.txt "venv\.requirements.stamp" >nul
:reqs_ok

if /i "%~1"=="--setup-only" (
    python server.py --setup-only || exit /b 1
    echo [OK] Setup complete. Models: venv\Scripts\python setup\fetch_voices.py --list
    exit /b 0
)

echo [Setup] Starting server. The URL to open is printed below.
python server.py %*
