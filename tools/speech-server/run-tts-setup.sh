#!/usr/bin/env bash
# ===== Install the optional GPU speech engines =========================
# Kokoro-82M, Chatterbox and Orpheus 3B need torch. Roughly 4 GB of wheels;
# model weights go where speech_paths.py says (TTS_CACHE).
# The llama.cpp engine (Qwen3-TTS, voice cloning) needs none of this.
#   ./run-tts-setup.sh                    install everything
#   TTS_CACHE=/data/tts ./run-tts-setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv not found on PATH. Install: https://docs.astral.sh/uv/"
    exit 1
fi
PY=venv/bin/python; [ -x "$PY" ] || PY=venv/Scripts/python.exe
if [ ! -x "$PY" ]; then
    echo "[ERROR] run ./run.sh once first to create the venv."
    exit 1
fi

# torch must come from the CUDA index -- a plain PyPI install is CPU-only.
# cu128 is what an RTX 50-series (Blackwell, sm_120) needs.
echo "[Setup] installing torch + torchaudio (cu128) ..."
uv pip install --python "$PY" torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

echo "[Setup] installing engine requirements ..."
uv pip install --python "$PY" -r requirements-tts.txt

# chatterbox-tts pins torch==2.6.0, which cannot target sm_120. Its deps are
# already in requirements-tts.txt, so install the package alone.
echo "[Setup] installing chatterbox-tts (no deps, to protect the CUDA torch) ..."
uv pip install --python "$PY" --no-deps chatterbox-tts

echo
echo "[Setup] verifying CUDA ..."
"$PY" -c "import torch;print('  torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo
echo "[Setup] downloading the recommended model of each engine ..."
"$PY" setup/fetch_voices.py

echo "[OK] Engines installed. More models: Settings -> Voice -> Models in the"
echo "     llama.cpp web UI, or $PY setup/fetch_voices.py --list"
