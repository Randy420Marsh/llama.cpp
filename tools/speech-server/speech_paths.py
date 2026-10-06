"""
Where every speech model and voice lives. One place, so the engines, the voice
library, the model downloader and the documentation all agree.

    %TTS_CACHE%                       default: C:\\AI\\tts-models if it exists,
                                      else models\\ next to server.py
      piper\\        <voice>.onnx + <voice>.onnx.json      Piper voices
      gguf\\
        qwen3-tts\\  <model>.gguf + mmproj-<model>.gguf    llama.cpp speech (llama-tts-server)
        orpheus\\    orpheus-3b-0.1-ft-<Q>.gguf,
                    orpheus-3b-0.1-pretrained-<Q>.gguf    Orpheus via llama-server
      voices\\       <id>\\reference.wav + transcript.txt   your own (cloned) voices
      hf\\           Hugging Face cache: Kokoro, Chatterbox, SNAC
      whisper\\      faster-whisper models (local speech-to-text fallback)

Older locations are still searched, so nothing has to be moved:
<server>\\voices\\*.onnx for Piper; ORPHEUS_GGUF_DIR / LLAMA_TTS_MODEL_DIRS add
more model folders (;-separated on Windows, :-separated elsewhere).

The llama.cpp binaries (llama-tts-server, llama-server) come from LLAMA_BIN,
by default this repo's build\\bin\\Release (Windows) or build/bin.
"""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent


def _cache_root() -> Path:
    """TTS_CACHE, else an existing shared C:\\AI\\tts-models, else models\\ beside this file."""
    env = os.environ.get("TTS_CACHE", "").strip()
    if env:
        return Path(env).expanduser()
    shared = Path("C:/AI/tts-models")
    if os.name == "nt" and shared.is_dir():
        return shared
    return APP_ROOT / "models"


def _llama_bin() -> Path:
    """LLAMA_BIN, else this repo's build output (tools/speech-server -> build/bin)."""
    env = os.environ.get("LLAMA_BIN", "").strip()
    if env:
        return Path(env).expanduser()
    repo = APP_ROOT.parent.parent
    candidates = [repo / "build" / "bin" / "Release", repo / "build" / "bin",
                  Path("C:/AI/llama.cpp/build/bin/Release")]
    for c in candidates:
        if (c / ("llama-server.exe" if os.name == "nt" else "llama-server")).exists():
            return c
    return candidates[0] if os.name == "nt" else candidates[1]


CACHE_ROOT = _cache_root()
LLAMA_BIN = _llama_bin()

PIPER_DIR = CACHE_ROOT / "piper"
LEGACY_PIPER_DIR = APP_ROOT / "voices"

GGUF_ROOT = CACHE_ROOT / "gguf"
NATIVE_DIR = GGUF_ROOT / "qwen3-tts"
ORPHEUS_DIR = GGUF_ROOT / "orpheus"

VOICES_LIB = Path(os.environ.get("TTS_VOICES_DIR") or CACHE_ROOT / "voices").expanduser()
WHISPER_DIR = CACHE_ROOT / "whisper"


def env_dirs(var: str, default: list[Path]) -> list[Path]:
    """`var` as a ;-separated list of folders, else the defaults."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    return [Path(p).expanduser() for p in raw.split(os.pathsep if os.name != "nt" else ";") if p.strip()]


def describe() -> dict:
    return {
        "cache": str(CACHE_ROOT),
        "llama_bin": str(LLAMA_BIN),
        "piper": str(PIPER_DIR),
        "native_tts": str(NATIVE_DIR),
        "orpheus": str(ORPHEUS_DIR),
        "voices": str(VOICES_LIB),
        "hf": str(CACHE_ROOT / "hf"),
        "whisper": str(WHISPER_DIR),
    }
