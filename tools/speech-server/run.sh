#!/usr/bin/env bash
# ===== Speech server for the llama.cpp web UI (Linux / macOS / Git Bash) =====
# Text-to-speech, your own (cloned) voices, model downloads, speech-to-text.
# First run: creates the uv venv and installs requirements. Later runs: skip
# all that (stamp file) and start the server.
#   ./run.sh --lan                     https on the LAN, port 8179 (what the UI expects)
#   ./run.sh                           https on 127.0.0.1:8179 (TLS default)
# The certificate is llama-server's (LLAMA_ARG_SSL_*), defaulting to $AI_ROOT/ssl
# (AI_ROOT: C:/AI on Windows, $HOME/AI on Linux; see the block below).
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

# AI_ROOT: the shared AI folder. Defaults to C:/AI (the usual Windows layout,
# mirrored on Linux), else $HOME/AI. Override with AI_ROOT, SSL_DIR or TTS_CACHE.
if [ -z "${AI_ROOT:-}" ]; then
    if [ -d C:/AI ]; then AI_ROOT=C:/AI; else AI_ROOT="$HOME/AI"; fi
fi
SSL_DIR="${SSL_DIR:-$AI_ROOT/ssl}"

if [ -z "${LLAMA_ARG_SSL_CERT_FILE:-}" ] && [ -f "$SSL_DIR/cert.pem" ]; then
    export LLAMA_ARG_SSL_CERT_FILE="$SSL_DIR/cert.pem"
fi
if [ -z "${LLAMA_ARG_SSL_KEY_FILE:-}" ] && [ -f "$SSL_DIR/key.pem" ]; then
    export LLAMA_ARG_SSL_KEY_FILE="$SSL_DIR/key.pem"
fi

# All speech models live in the one shared model folder: $AI_ROOT/tts-models.
# Override with TTS_CACHE.
if [ -z "${TTS_CACHE:-}" ] && [ -d "$AI_ROOT/tts-models" ]; then
    export TTS_CACHE="$AI_ROOT/tts-models"
fi

echo "[Setup] Starting server. The URL to open is printed below."
export PYTHONUTF8=1
exec python server.py "$@"
