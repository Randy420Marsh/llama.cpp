"""
Speech-to-text for the mic button and clip transcripts.

When the shared ASR server answers (C:\\AI\\whisper-asr-server, the llama.cpp
fork's tools/asr-server, faster-whisper large-v3, the one llama-server's
--asr-url uses), audio goes there: multilingual, more accurate, and no second
Whisper model in VRAM. Otherwise a local faster-whisper model is loaded here,
sized for a card that is already mostly full: the default is a small int8
model that costs a few hundred MB and still transcribes a spoken sentence in
well under a second.  Override with:

    set STT_MODEL=distil-large-v3      more accurate local model, ~1.5 GB
    set STT_DEVICE=cpu                 keep the GPU entirely free
    set STT_USE_SERVER=off             never use the ASR server
    set ASR_URL=http://host:port       where the ASR server listens
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request
import uuid
from pathlib import Path

from speech_paths import CACHE_ROOT, WHISPER_DIR

os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))

MODEL = os.environ.get("STT_MODEL", "small.en")
DEVICE = os.environ.get("STT_DEVICE", "")          # "" = auto
COMPUTE = os.environ.get("STT_COMPUTE", "")        # "" = int8 variants

ASR_URL = os.environ.get("ASR_URL", "http://127.0.0.1:8178")
USE_SERVER = os.environ.get("STT_USE_SERVER", "auto").lower() != "off"

_model = None
_lock = threading.Lock()


def server_up() -> bool:
    if not USE_SERVER:
        return False
    try:
        with urllib.request.urlopen(ASR_URL + "/health", timeout=1.0) as r:
            return r.status == 200
    except Exception:
        return False


def available() -> tuple[bool, str]:
    if server_up():
        return True, ""
    try:
        import faster_whisper  # noqa: F401
        return True, ""
    except Exception as e:
        return False, f"faster-whisper not installed and no ASR server at {ASR_URL}: {e}"


def _transcribe_server(path: str, language: str) -> dict:
    """POST the file as multipart/form-data (the server decodes any container itself)."""
    boundary = uuid.uuid4().hex
    fields = [("response_format", "verbose_json")] + ([("language", language)] if language else [])
    body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                    for k, v in fields)
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{Path(path).name}"\r\n'
             f"Content-Type: application/octet-stream\r\n\r\n").encode()
    body += Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    rq = urllib.request.Request(ASR_URL + "/v1/audio/transcriptions", data=body,
                                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(rq, timeout=900) as r:
        doc = json.loads(r.read())
    return {"text": (doc.get("text") or "").strip(),
            "language": doc.get("language") or "",
            "duration": round(float(doc.get("duration") or 0.0), 2),
            "via": "asr-server"}


def _pick_device() -> tuple[str, str]:
    if DEVICE:
        return DEVICE, (COMPUTE or ("int8_float16" if DEVICE == "cuda" else "int8"))
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            # Whisper small needs ~500 MB; don't elbow the chat model off the
            # card for a mic button. Fall back to CPU when the GPU is full.
            if free > 1.5 * 1024 ** 3:
                return "cuda", (COMPUTE or "int8_float16")
    except Exception:
        pass
    return "cpu", (COMPUTE or "int8")


def load():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            dev, comp = _pick_device()
            print(f"[STT] loading {MODEL} on {dev} ({comp}) ...", flush=True)
            _model = WhisperModel(MODEL, device=dev, compute_type=comp,
                                  download_root=str(WHISPER_DIR))
            print("[STT] ready", flush=True)
        return _model


def unload():
    global _model
    with _lock:
        if _model is None:
            return
        print("[STT] unloading", flush=True)
        _model = None
        try:
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def loaded() -> bool:
    return _model is not None


def transcribe(path: str, language: str = "") -> dict:
    if server_up():
        try:
            return _transcribe_server(path, language)
        except Exception as e:
            print(f"[STT] ASR server failed ({e}); using the local model", flush=True)
    m = load()
    segments, info = m.transcribe(
        path,
        language=language or None,
        beam_size=1,                 # greedy: this is a chat box, not a subtitle job
        vad_filter=True,             # drop the silence around a push-to-talk clip
        vad_parameters={"min_silence_duration_ms": 300},
    )
    text = "".join(s.text for s in segments).strip()
    return {"text": text,
            "language": getattr(info, "language", "") or "",
            "duration": round(getattr(info, "duration", 0.0), 2)}


def describe() -> dict:
    ok, why = available()
    if server_up():
        return {"installed": ok, "detail": why, "model": "ASR server", "server": ASR_URL,
                "device": "server", "compute": "", "loaded": True}
    dev, comp = _pick_device()
    return {"installed": ok, "detail": why, "model": MODEL,
            "device": dev, "compute": comp, "loaded": loaded()}
