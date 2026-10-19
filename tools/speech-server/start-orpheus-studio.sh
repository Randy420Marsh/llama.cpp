#!/usr/bin/env bash
# Start the Orpheus 3B LoRA fine-tuning studio (Gradio) on native Linux.
#
# The GUI is a thin Gradio app; it does no heavy work itself -- it shells out to
# the speech-server torch venv for dataset building, LoRA training, GGUF
# conversion and synthesis. So the GUI runs in a *gradio-only* environment,
# which is why it does not fight the torch-nightly-vs-gradio segfault.
#
#   default gradio env : $ORPHEUS_GUI_ENV, else ~/AI/sol/gui-env (gradio 5, proven)
#   fallback           : <speech-server>/gui-env (created on first run, gradio 5)
#   torch venv         : $ORPHEUS_TTS_VENV, else <speech-server>/venv
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH" UV_LINK_MODE=copy

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../tools/speech-server
LLAMA="$(cd "$HERE/../.." && pwd)"                            # the llama.cpp root

TTS_VENV="$HERE/venv"
[[ -x "$TTS_VENV/bin/python" ]] || { echo "torch venv missing: $TTS_VENV (run run-tts-setup.sh)" >&2; exit 1; }

# --- the gradio environment (GUI only, no torch) ------------------------------
GUI_ENV="${ORPHEUS_GUI_ENV:-$HOME/AI/sol/gui-env}"
if [[ ! -x "$GUI_ENV/bin/python" ]] || ! "$GUI_ENV/bin/python" -c "import gradio" 2>/dev/null; then
  GUI_ENV="$HERE/gui-env"
  if [[ ! -x "$GUI_ENV/bin/python" ]] || ! "$GUI_ENV/bin/python" -c "import gradio" 2>/dev/null; then
    echo "First start: creating a gradio env in $GUI_ENV ..."
    uv venv --allow-existing --python 3.11 "$GUI_ENV"
    uv pip install --python "$GUI_ENV/bin/python" "gradio>=5,<6"
  fi
fi

export ORPHEUS_TTS_VENV="$TTS_VENV"
export ORPHEUS_LLAMA_CPP="$LLAMA"
export GRADIO_ANALYTICS_ENABLED=False
export ORPHEUS_HOST="${ORPHEUS_HOST:-127.0.0.1}"
export ORPHEUS_PORT="${ORPHEUS_PORT:-7861}"

echo "Orpheus LoRA studio: http://$ORPHEUS_HOST:$ORPHEUS_PORT"
echo "  GUI env : $GUI_ENV"
echo "  torch   : $TTS_VENV"
echo "  llama   : $LLAMA"
exec "$GUI_ENV/bin/python" "$HERE/orpheus/app.py"
