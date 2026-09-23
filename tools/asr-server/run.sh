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

echo "[Start] asr_server.py $*"
export PYTHONUNBUFFERED=1
exec python -u asr_server.py "$@"
