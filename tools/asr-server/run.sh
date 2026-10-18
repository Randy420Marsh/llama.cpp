#!/usr/bin/env bash
# ===== whisper-asr-server launcher (Linux / Git Bash) =====================
# First run: creates the uv venv (Python 3.11) and installs requirements
# (faster-whisper + CUDA cuBLAS/cuDNN wheels). Later runs: skip the install
# (stamp file) and start the OpenAI-compatible transcription server.
#   ./run.sh --setup-only                    ...do the setup steps and exit
#   ./run.sh --port 8178 --idle-unload 300   ...extra args go to asr_server.py
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv not found on PATH. Install: https://docs.astral.sh/uv/"
    exit 1
fi

if [ ! -x venv/bin/python ] && [ ! -f venv/Scripts/python.exe ]; then
    echo "[Setup] Creating Python 3.11 venv with uv..."
    uv venv venv --python 3.11
fi

if [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
else
    # shellcheck disable=SC1091
    source venv/Scripts/activate
fi

# Requirements: install on first run, when requirements.txt changes, or when the
# stamp exists but the packages are missing (interrupted install)
if ! cmp -s requirements.txt venv/.requirements.stamp 2>/dev/null \
    || ! python -c "import importlib.util as u, sys; sys.exit(any(u.find_spec(m) is None for m in ('faster_whisper', 'fastapi', 'uvicorn', 'multipart')))" 2>/dev/null; then
    echo "[Setup] Installing requirements (first run / requirements.txt changed / packages missing)..."
    uv pip install -r requirements.txt
    cp requirements.txt venv/.requirements.stamp
fi
echo "[Setup] venv and requirements OK."

if [ "${1:-}" = "--setup-only" ]; then
    echo "[OK] Setup complete - venv and requirements ready."
    exit 0
fi

# https: the server refuses plain http unless --no-https is given. Use the same
# certificate as llama-server (LLAMA_ARG_SSL_*), defaulting to C:/AI/ssl on Windows.
if [ -z "${LLAMA_ARG_SSL_CERT_FILE:-}" ] && [ -f C:/AI/ssl/cert.pem ]; then
    export LLAMA_ARG_SSL_CERT_FILE=C:/AI/ssl/cert.pem
fi
if [ -z "${LLAMA_ARG_SSL_KEY_FILE:-}" ] && [ -f C:/AI/ssl/key.pem ]; then
    export LLAMA_ARG_SSL_KEY_FILE=C:/AI/ssl/key.pem
fi

# Whisper models live in the one shared model folder (the Windows layout, mirrored
# on Linux): /home/john/AI/tts-models/whisper. Override with ASR_MODELS_DIR.
if [ -z "${ASR_MODELS_DIR:-}" ] && [ -d /home/john/AI/tts-models/whisper ]; then
    export ASR_MODELS_DIR=/home/john/AI/tts-models/whisper
fi
echo "[Start] asr_server.py $*"
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
exec python -u asr_server.py "$@"
