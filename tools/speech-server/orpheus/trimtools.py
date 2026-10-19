#!/usr/bin/env python3
"""Clip staging helpers for the Orpheus studio's trim/export tool.

Turns dropped videos or audio into training-ready clips:
  * extract the audio track from video
  * trim an A..B segment (seek on a synced preview in the GUI)
  * resample to 24 kHz mono 16-bit WAV (the SNAC codec's rate, best quality
    for Orpheus training data)
  * normalize to -23 LUFS with ffmpeg's loudnorm (2-pass, so a long source
    and a short clip both land at the same perceived level)

Only the system ffmpeg/ffprobe are needed (both ship the loudnorm filter).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

LUF_TARGET = -23.0   # EBU R128, the standard speech loudness
TRUE_PEAK = -1.5     # leave headroom
SR = 24000           # SNAC codec rate
REC_MIN, REC_MAX = 6.0, 15.0   # recommended clip length (Orpheus-TTS authors)


def _ffmpeg(*args: str, binary: bool = False) -> subprocess.CompletedProcess:
    """Run ffmpeg. binary=True keeps stdout raw (for -f f32le pipes);
    otherwise stdout/stderr are decoded text (for reading filter logs)."""
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", *args],
                          capture_output=True, text=not binary, timeout=600)


def probe(path: Path) -> dict:
    """duration / has_audio / has_video / width / height via ffprobe."""
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                        "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, timeout=120)
    info = json.loads(r.stdout or "{}")
    out = {"duration": 0.0, "has_audio": False, "has_video": False,
           "width": 0, "height": 0}
    if (info.get("format") or {}).get("duration"):
        out["duration"] = float(info["format"]["duration"])
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            out["has_audio"] = True
        elif s.get("codec_type") == "video":
            out["has_video"] = True
            out["width"] = int(s.get("width", 0))
            out["height"] = int(s.get("height", 0))
    return out


def peaks(path: Path, buckets: int = 4000) -> list[float]:
    """Per-bucket peak amplitudes (0..1) of the whole file, for a waveform view.

    Decodes to 8 kHz mono f32 (plenty for a level meter) and takes the max
    absolute sample per bucket -- fast even for long movies.
    """
    r = _ffmpeg("-i", str(path), "-map", "0:a:0", "-af", "aresample=8000",
                "-ac", "1", "-f", "f32le", "-", binary=True)
    if r.returncode != 0 or not r.stdout:
        return []
    import numpy as np
    x = np.frombuffer(r.stdout, dtype="<f4")
    if x.size == 0:
        return []
    n = min(buckets, max(1, x.size // 4))
    cut = (x.size // n) * n
    x = x[:cut].reshape(n, -1)
    return [round(float(v), 3) for v in x.max(axis=1)]


def _loudnorm_measured(src: Path, a: float | None, b: float | None) -> dict | None:
    """Pass 1: measure the (trimmed) segment's loudness stats for the 2nd pass."""
    cmd = ["-i", str(src), "-map", "0:a:0"]
    if a is not None:
        cmd += ["-ss", f"{max(a, 0):.3f}"]
        if b is not None:
            cmd += ["-t", f"{max(b - a, 0.05):.3f}"]
    cmd += ["-af", "aresample=24000,loudnorm=I=%s:print_format=json" % LUF_TARGET,
            "-ac", "1", "-f", "null", "-"]
    r = _ffmpeg(*cmd)
    m = re.findall(r"\{.*input_i.*\}", r.stderr, re.S)
    if not m:
        return None
    try:
        return json.loads(m[-1])
    except json.JSONDecodeError:
        return None


def export_clip(src: Path, dst: Path, a: float | None = None,
                b: float | None = None) -> dict:
    """Cut [a, b) (or the whole file), resample 24 kHz mono, normalize to
    -23 LUFS (2-pass), write 16-bit WAV. Returns {path, duration, measured_i}."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    meas = _loudnorm_measured(src, a, b)
    filt = f"aresample={SR}"
    if meas:
        # 2-pass: feed the measured stats back so the 2nd pass applies a flat,
        # exact gain (the standard ffmpeg loudnorm 2-pass recipe).
        filt += (",loudnorm=I=%s:TP=%s:measured_I=%s:measured_TP=%s"
                 ":measured_LRA=%s:measured_thresh=%s"
                 % (LUF_TARGET, TRUE_PEAK, meas.get("input_i", LUF_TARGET),
                    meas.get("input_tp", TRUE_PEAK), meas.get("input_lra", 0),
                    meas.get("input_thresh", -20)))
    else:
        filt += f",loudnorm=I={LUF_TARGET}:TP={TRUE_PEAK}"
    cmd = ["-i", str(src), "-map", "0:a:0"]
    if a is not None:
        cmd += ["-ss", f"{max(a, 0):.3f}"]
        if b is not None:
            cmd += ["-t", f"{max(b - a, 0.05):.3f}"]
    cmd += ["-af", filt, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le",
            "-y", str(dst)]
    r = _ffmpeg(*cmd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")
    dur = probe(dst)["duration"]
    return {"path": str(dst), "duration": dur,
            "measured_i": (meas or {}).get("input_i")}


def next_clip_name(voice_dir: Path) -> Path:
    """The next free NNNN.wav in a voice's clips folder."""
    voice_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in voice_dir.glob("*.wav"):
        if re.fullmatch(r"\d{4}\.wav", p.name):
            n = max(n, int(p.name[:-4]))
    return voice_dir / f"{n + 1:04d}.wav"


AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus", ".wma"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mpg",
             ".mpeg", ".3gp"}


def kind_of(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in VIDEO_EXT:
        return "video"
    if suf in AUDIO_EXT:
        return "audio"
    # unknown: let ffprobe decide
    return "video" if probe(path)["has_video"] else "audio"


def ensure_ffmpeg() -> str | None:
    """None when usable, else a human error message."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return None
    return "ffmpeg/ffprobe not found on PATH -- the trim tool needs them " \
           "(apt install ffmpeg, or export PATH to a build)"
