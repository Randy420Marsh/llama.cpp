"""
Your own voices: reference clips for zero-shot voice cloning.

Nothing is trained. A voice is one short recording of the speaker (and what
was said in it); the cloning engines condition on it at synthesis time:

    llamacpp    Qwen3-TTS in llama.cpp (llama-tts-server) -- speaker embedding
                from the clip, 10 languages
    chatterbox  the clip as audio prompt (when the GPU engines are installed)
    orpheus     the clip + its transcript as an in-context example for the
                Orpheus *pretrained* model (when a pretrained GGUF is present)

Layout, shared with llama-tts-server's --tts-voices-dir and with the
llama.cpp-server-tts app:

    %TTS_CACHE%\\voices\\<id>\\reference.wav    24 kHz mono 16-bit, 3-30 s
                            transcript.txt   what is said in the clip
                            voice.json       {"name", "language", "created", ...}

A folder dropped in by hand with just a reference.wav (or a bare <id>.wav)
works too; the missing transcript is then simply empty.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

from speech_paths import VOICES_LIB as LIB
MIN_SECONDS, MAX_SECONDS = 3.0, 30.0
SAMPLE_RATE = 24000
AUDIO_EXTS = (".wav", ".flac", ".mp3")

_lock = threading.Lock()

# A sentence set that covers most English sounds; shown in the UI as something
# to read aloud when recording a voice (the classic "Rainbow Passage").
READING_PASSAGE = (
    "When the sunlight strikes raindrops in the air, they act as a prism and form "
    "a rainbow. The rainbow is a division of white light into many beautiful "
    "colors. These take the shape of a long round arch, with its path high above, "
    "and its two ends apparently beyond the horizon.")


def slug(name: str) -> str:
    """Folder-safe id from a display name: letters, digits, '-', '_' and '.'."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", (name or "").strip()).strip("-.")
    return s[:48]


def _duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:
        return 0.0


def _reference(folder: Path) -> Path | None:
    for ext in AUDIO_EXTS:
        p = folder / f"reference{ext}"
        if p.is_file():
            return p
    return None


def _entry(vid: str, ref: Path, folder: Path | None) -> dict:
    meta: dict = {}
    transcript = ""
    if folder is not None:
        try:
            meta = json.loads((folder / "voice.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        try:
            transcript = (folder / "transcript.txt").read_text(encoding="utf-8").strip()
        except Exception:
            transcript = ""
    return {
        "id": vid,
        "name": meta.get("name") or vid,
        "language": meta.get("language") or "",
        "transcript": transcript,
        "duration_s": meta.get("duration_s") or _duration(ref),
        "created": meta.get("created") or "",
        "file": str(ref),
        "bytes": ref.stat().st_size if ref.exists() else 0,
    }


def list_voices() -> list[dict]:
    if not LIB.is_dir():
        return []
    out = []
    for p in sorted(LIB.iterdir(), key=lambda q: q.name.lower()):
        if p.is_dir():
            ref = _reference(p)
            if ref:
                out.append(_entry(p.name, ref, p))
        elif p.suffix.lower() in AUDIO_EXTS and p.is_file():
            out.append(_entry(p.stem, p, None))
    return out


def get(vid: str) -> dict | None:
    """Look a voice up by id (or display name). Never builds a path from input."""
    for v in list_voices():
        if v["id"] == vid or v["name"] == vid:
            return v
    return None


def reference_path(vid: str) -> Path | None:
    v = get(vid)
    return Path(v["file"]) if v else None


def ffmpeg_exe() -> str | None:
    return shutil.which("ffmpeg")


def _normalise(src: Path, dst: Path) -> float:
    """Any audio the browser can record -> 24 kHz mono 16-bit WAV, trimmed and levelled.

    Leading/trailing silence goes (a reference that is half silence clones the
    room, not the voice), loudness is brought to a common level, and anything
    past MAX_SECONDS is cut: longer clips add encode time, not quality.
    """
    exe = ffmpeg_exe()
    if exe is None:
        if src.suffix.lower() != ".wav":
            raise RuntimeError("ffmpeg is not on PATH, so only .wav uploads can be used")
        shutil.copyfile(src, dst)
        return _duration(dst)
    trim = "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.15"
    af = f"{trim},areverse,{trim},areverse,loudnorm=I=-20:TP=-2:LRA=11"
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
           "-vn", "-ac", "1", "-af", af, "-ar", str(SAMPLE_RATE),
           "-t", str(MAX_SECONDS), "-c:a", "pcm_s16le", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError("could not read that audio: " + (r.stderr or "").strip()[:300])
    return _duration(dst)


def add_voice(name: str, data: bytes, filename: str = "", transcript: str = "",
              language: str = "", transcriber=None) -> dict:
    """Store a new voice. `transcriber(path, language) -> text` fills a missing transcript."""
    vid = slug(name)
    if not vid:
        raise ValueError("give the voice a name (letters, digits, - or _)")
    if not data:
        raise ValueError("the audio file is empty")
    with _lock:
        LIB.mkdir(parents=True, exist_ok=True)
        folder = LIB / vid
        if folder.exists() or (LIB / f"{vid}.wav").exists():
            raise FileExistsError(f"a voice called '{vid}' already exists")
        tmp = LIB / f".upload-{vid}-{int(time.time() * 1000)}{Path(filename or 'clip.webm').suffix or '.webm'}"
        tmp.write_bytes(data)
        staging = LIB / f".new-{vid}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        try:
            dur = _normalise(tmp, staging / "reference.wav")
            if dur < MIN_SECONDS:
                raise ValueError(f"only {dur:.1f} s of speech after trimming silence; "
                                 f"record at least {MIN_SECONDS:.0f} s (6-15 s works best)")
            text = (transcript or "").strip()
            how = "given"
            if not text and transcriber is not None:
                try:
                    text = (transcriber(str(staging / "reference.wav"), language) or "").strip()
                    how = "transcribed"
                except Exception as e:
                    print(f"[voices] transcription failed for {vid}: {e}", flush=True)
                    how = "none"
            (staging / "transcript.txt").write_text(text, encoding="utf-8")
            meta = {"name": (name or vid).strip()[:64], "language": (language or "").strip()[:8],
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "duration_s": dur,
                    "source": Path(filename).name if filename else "", "transcript_from": how}
            (staging / "voice.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                                encoding="utf-8")
            staging.rename(folder)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        finally:
            tmp.unlink(missing_ok=True)
    print(f"[voices] added '{vid}' ({dur:.1f} s)", flush=True)
    return get(vid)


def update_voice(vid: str, transcript: str | None = None, language: str | None = None,
                 name: str | None = None) -> dict:
    v = get(vid)
    if v is None:
        raise KeyError(vid)
    folder = LIB / v["id"]
    with _lock:
        if not folder.is_dir():
            # a bare <id>.wav: give it a folder so it can carry metadata
            folder.mkdir(parents=True)
            Path(v["file"]).rename(folder / "reference.wav")
        meta_p = folder / "voice.json"
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if transcript is not None:
            (folder / "transcript.txt").write_text(transcript.strip(), encoding="utf-8")
            meta["transcript_from"] = "edited"
        if language is not None:
            meta["language"] = language.strip()[:8]
        if name is not None and name.strip():
            meta["name"] = name.strip()[:64]
        meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return get(v["id"])


def delete_voice(vid: str) -> bool:
    v = get(vid)
    if v is None:
        return False
    with _lock:
        folder = LIB / v["id"]
        if folder.is_dir():
            shutil.rmtree(folder)
        else:
            Path(v["file"]).unlink(missing_ok=True)
    print(f"[voices] deleted '{v['id']}'", flush=True)
    return True
