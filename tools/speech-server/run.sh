#!/usr/bin/env bash
# ===== Speech server for the llama.cpp web UI (Linux / macOS / Git Bash) =====
# Text-to-speech, your own (cloned) voices, model downloads, speech-to-text.
# First run: creates the uv venv and installs requirements. Later runs: skip
# all that (stamp file) and start the server.
#   ./run.sh --no-https --lan          http on the LAN, port 8179 (what the UI expects)
#   ./run.sh                           https on 127.0.0.1:8179 (TLS default)
#   ./run.sh --setup-only              set up, then exit
#   ./run.sh --port 8179 --engine piper   forwarded to server.py
# GPU engines (Kokoro, Chatterbox, Orpheus): ./run-tts-setup.sh once.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv not found on PATH. Install: https://docs.astral.sh/uv/"
    exit 1
fi

if [ ! -x venv/bin/python ] && [ ! -x venv/Scripts/python.exe ]; then
    echo "[Setup] Creating Python 3.11 venv with uv..."
    uv venv venv --python 3.11
fi

# shellcheck disable=SC1091
if [ -f venv/bin/activate ]; then source venv/bin/activate; else source venv/Scripts/activate; fi

# ===== Requirements: install only on first run or when requirements.txt changes
if ! cmp -s requirements.txt venv/.requirements.stamp 2>/dev/null; then
    echo "[Setup] Installing requirements (first run / requirements.txt changed)..."
    uv pip install -r requirements.txt
    cp requirements.txt venv/.requirements.stamp
fi

if [ "${1:-}" = "--setup-only" ]; then
    python server.py --setup-only
    echo "[OK] Setup complete. Models: python setup/fetch_voices.py --list"
    exit 0
fi

echo "[Setup] Starting server. The URL to open is printed below."
exec python server.py "$@"
