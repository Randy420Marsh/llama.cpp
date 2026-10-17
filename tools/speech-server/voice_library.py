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

                            original.wav     the clip before the first trim,
                            original.txt     and its transcript (Restore puts
                                             both back)

A folder dropped in by hand with just a reference.wav (or a bare <id>.wav,
.mp3 or .flac) works too; the missing transcript is then simply empty. Such a
clip is converted to the format above the first time the library is listed
(the file as dropped is kept as source.<ext>): Orpheus reads 16-bit PCM only,
and every engine then hears the same clip.
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
# clips that could not be converted, by path -> mtime: not retried on every listing
_unconvertible: dict[str, float] = {}

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


def _canonical(path: Path) -> bool:
    """Is it already 24 kHz mono 16-bit PCM WAV?"""
    if path.suffix.lower() != ".wav":
        return False
    try:
        with wave.open(str(path), "rb") as w:
            return (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SAMPLE_RATE, 1, 2)
    except Exception:
        # float or extensible WAVs that the wave module cannot open
        return False


def _reference(folder: Path) -> Path | None:
    for ext in AUDIO_EXTS:
        p = folder / f"reference{ext}"
        if p.is_file():
            return p
    return None


def _canonicalise(vid: str, ref: Path, folder: Path | None) -> Path:
    """Convert a hand-dropped clip to reference.wav in its folder; keep the file as source.<ext>."""
    target = LIB / vid
    staging = LIB / f".fix-{vid}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        dur = _normalise(ref, staging / "reference.wav")
        if not _canonical(staging / "reference.wav"):
            raise RuntimeError("ffmpeg is needed to convert it")
        if folder is None:
            target.mkdir()
        source = target / f"source{ref.suffix.lower()}"
        if not source.exists():
            ref.rename(source)
        else:
            ref.unlink()
        (staging / "reference.wav").replace(target / "reference.wav")
        meta_p = target / "voice.json"
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta["duration_s"] = dur
        meta.setdefault("source", source.name)
        meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[voices] converted '{vid}' to {SAMPLE_RATE // 1000} kHz mono 16-bit ({dur:.1f} s)",
              flush=True)
        return target / "reference.wav"
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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
        "has_original": folder is not None and (folder / "original.wav").is_file(),
    }


def _scan() -> list[tuple[str, Path, Path | None]]:
    found = []
    for p in sorted(LIB.iterdir(), key=lambda q: q.name.lower()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            ref = _reference(p)
            if ref:
                found.append((p.name, ref, p))
        elif p.suffix.lower() in AUDIO_EXTS and p.is_file():
            found.append((p.stem, p, None))
    return found


def _needs_converting(vid: str, ref: Path, folder: Path | None) -> bool:
    if _canonical(ref) or (folder is None and (LIB / vid).exists()):
        return False
    try:
        return _unconvertible.get(str(ref)) != ref.stat().st_mtime
    except OSError:
        return False


def list_voices() -> list[dict]:
    if not LIB.is_dir():
        return []
    found = _scan()
    if any(_needs_converting(*f) for f in found):
        with _lock:
            for vid, ref, folder in _scan():
                if not _needs_converting(vid, ref, folder):
                    continue
                try:
                    _canonicalise(vid, ref, folder)
                except Exception as e:
                    # still listed as it is; Qwen3-TTS reads any wav/mp3/flac itself
                    print(f"[voices] could not convert '{vid}': {e}", flush=True)
                    try:
                        _unconvertible[str(ref)] = ref.stat().st_mtime
                    except OSError:
                        pass
            found = _scan()
    return [_entry(vid, ref, folder) for vid, ref, folder in found]


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


def _write_meta(folder: Path, **changes) -> None:
    meta_p = folder / "voice.json"
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    meta.update(changes)
    meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def _folder_of(vid: str) -> tuple[dict, Path]:
    v = get(vid)
    if v is None:
        raise KeyError(vid)
    folder = LIB / v["id"]
    if not folder.is_dir() or Path(v["file"]).name != "reference.wav" or not _canonical(Path(v["file"])):
        raise ValueError("this clip could not be converted to 24 kHz WAV, so it cannot be edited")
    return v, folder


def trim_voice(vid: str, start: float = 0.0, end: float | None = None, tighten: bool = False,
               retranscribe: bool = False, transcriber=None) -> dict:
    """Keep start..end seconds of the clip; `tighten` shortens pauses inside it to 0.25 s.

    The clip before the first trim is kept (original.wav / original.txt) for restore_voice.
    `transcriber(path, language) -> text` rewrites the transcript to match the new clip.
    """
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("ffmpeg is not on PATH; it is needed to trim")
    v, folder = _folder_of(vid)
    ref = folder / "reference.wav"
    with _lock:
        total = _duration(ref)
        start = max(0.0, float(start))
        end = total if end is None else min(float(end), total)
        if not (end - start >= MIN_SECONDS):
            raise ValueError(f"keep at least {MIN_SECONDS:.0f} s ({start:.2f}-{end:.2f} s is too short)")
        out = folder / ".trim.wav"
        # a short fade at both cut points, so a cut mid-sound does not click
        af = ["afade=t=in:d=0.02", "areverse", "afade=t=in:d=0.03", "areverse"]
        if tighten:
            af.insert(0, "silenceremove=stop_periods=-1:stop_duration=0.4:"
                         "stop_threshold=-45dB:stop_silence=0.25")
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(ref),
               "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-af", ",".join(af),
               "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)]
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode != 0 or not out.exists():
            out.unlink(missing_ok=True)
            raise RuntimeError("ffmpeg could not trim it: " + (r.stderr or "").strip()[:300])
        dur = _duration(out)
        if dur < MIN_SECONDS:
            out.unlink(missing_ok=True)
            raise ValueError(f"only {dur:.1f} s would be left; keep at least {MIN_SECONDS:.0f} s")
        if not (folder / "original.wav").exists():
            shutil.copyfile(ref, folder / "original.wav")
            transcript = folder / "transcript.txt"
            (folder / "original.txt").write_text(
                transcript.read_text(encoding="utf-8") if transcript.exists() else "", encoding="utf-8")
        out.replace(ref)
        _write_meta(folder, duration_s=dur)
    print(f"[voices] trimmed '{v['id']}' to {start:.2f}-{end:.2f} s -> {dur:.1f} s", flush=True)
    if retranscribe and transcriber is not None:
        try:
            text = (transcriber(str(ref), v["language"]) or "").strip()
        except Exception as e:
            print(f"[voices] transcription failed for {v['id']}: {e}", flush=True)
            text = ""
        if text:
            with _lock:
                (folder / "transcript.txt").write_text(text, encoding="utf-8")
                _write_meta(folder, transcript_from="transcribed")
    return get(v["id"])


def restore_voice(vid: str) -> dict:
    """Put back the clip (and transcript) from before the first trim."""
    v, folder = _folder_of(vid)
    original = folder / "original.wav"
    if not original.is_file():
        raise ValueError("this voice has not been trimmed")
    with _lock:
        original.replace(folder / "reference.wav")
        text = folder / "original.txt"
        if text.exists():
            text.replace(folder / "transcript.txt")
        _write_meta(folder, duration_s=_duration(folder / "reference.wav"))
    print(f"[voices] restored '{v['id']}'", flush=True)
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
