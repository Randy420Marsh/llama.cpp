@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"

rem ===== Speech server for the llama.cpp web UI (Windows) ================
rem Text-to-speech, your own (cloned) voices, model downloads, speech-to-text.
rem First run: creates the uv venv and installs requirements. Later runs:
rem skip all that (stamp file) and start the server immediately.
rem   run.bat --lan               ...what the command center runs: https, LAN, port 8179
rem   run.bat                     ...https on 127.0.0.1:8179 (TLS default)
rem The certificate is llama-server's (LLAMA_ARG_SSL_*), defaulting to %AI_ROOT%\ssl
rem (AI_ROOT: C:\AI by default, %USERPROFILE%\AI otherwise; see the block below).
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

rem AI_ROOT: the shared AI folder (C:\AI by default; set it, or SSL_DIR, to override).
if not defined AI_ROOT (
    if exist "C:\AI" (set "AI_ROOT=C:\AI") else (set "AI_ROOT=%USERPROFILE%\AI")
)
if not defined SSL_DIR set "SSL_DIR=%AI_ROOT%\ssl"
if not defined LLAMA_ARG_SSL_CERT_FILE if exist "%SSL_DIR%\cert.pem" set "LLAMA_ARG_SSL_CERT_FILE=%SSL_DIR%\cert.pem"
if not defined LLAMA_ARG_SSL_KEY_FILE if exist "%SSL_DIR%\key.pem" set "LLAMA_ARG_SSL_KEY_FILE=%SSL_DIR%\key.pem"

echo [Setup] Starting server. The URL to open is printed below.
python server.py %*
