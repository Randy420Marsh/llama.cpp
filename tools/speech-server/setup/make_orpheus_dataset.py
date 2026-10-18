#!/usr/bin/env python3
"""
Build a fine-tuning dataset for Orpheus 3B from your own clips.

The Orpheus models (canopylabs) train on a Hugging Face dataset with a `text`
and an `audio` column -- plus a `source` (speaker name) column for multi-speaker
training. See the sample dataset:
https://huggingface.co/datasets/canopylabs/zac-sample-dataset

This tool takes a folder of clips you recorded (Settings -> Voice -> Your
voices, a microphone, a phone) and produces exactly that format:

    ./build-orpheus-dataset clips/ --out myvoice-dataset
        -> myvoice-dataset/data-00000-of-00001.arrow  (Hugging Face dataset)
        -> myvoice-dataset/dataset_info.json
    or, for use with load_dataset("json", ...):
    ./build-orpheus-dataset clips/ --jsonl myvoice.jsonl
        -> one line per clip: {"text": "...", "audio": "clips/0001.wav"}

How to pair transcripts with clips, in order of preference:
  1. a sidecar file per clip: 0001.wav -> 0001.txt
  2. a single transcripts.txt in the clips folder, one line per clip, in
     the same order as the clips sorted by name
  3. none: the audio is kept but the text column is empty -- you can fill
     it in later, or leave it: the model mostly follows the *audio*, the
     text teaches it what is being said so it can say new words in that
     voice

Data recommendations (from the Orpheus-TTS authors):
  * about 50 clips is enough to see your voice; aim for 300 per speaker
    for a strong result
  * 6-15 s of natural speech per clip, one speaker, a quiet room
  * vary the content (reading, talking, different emotions) so the model
    learns the voice, not one particular phrase
  * avoid synthetic / TTS-generated audio: it maps to the same audio
    tokens every time and uses the codebook poorly

Audio is resampled to 24 kHz mono (the SNAC codec's rate) when a resampler
is available (scipy or soundfile); otherwise the original clips are used
and the training script resamples on the fly.

Requires only Python 3.11 + (optionally) numpy/scipy/soundfile; the output
is read back by train_orpheus.py (which needs unsloth + snac).
"""
import argparse
import json
import re
import sys
import wave
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"}


def _find_clips(d: Path) -> list[Path]:
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            out.append(p)
    return out


def _transcripts(clips: list[Path], d: Path, explicit: Path | None) -> dict[Path, str]:
    """Pair each clip with its text (sidecar .txt, then transcripts.txt)."""
    got: dict[Path, str] = {}
    for c in clips:
        for cand in (c.with_suffix(".txt"), c.with_name(c.stem + ".txt")):
            if cand.is_file():
                got[c] = cand.read_text(encoding="utf-8").strip()
                break
    if explicit and explicit.is_file():
        for c, line in zip(clips, explicit.read_text(encoding="utf-8").splitlines()):
            got.setdefault(c, line.strip())
    else:
        t = d / "transcripts.txt"
        if t.is_file():
            for c, line in zip(clips, t.read_text(encoding="utf-8").splitlines()):
                got.setdefault(c, line.strip())
    return got


def _resample_24k_mono(src: Path, dst: Path) -> bool:
    """24 kHz mono WAV copy when a resampler exists; False when it must stay raw."""
    try:
        import numpy as np
    except ImportError:
        return False
    if src.suffix.lower() == ".wav":
        with wave.open(str(src), "rb") as wf:
            sr, ch = wf.getframerate(), wf.getnchannels()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype("float32") / 32768.0
            if ch > 1:
                pcm = pcm.reshape(-1, ch).mean(axis=1)
        if sr != 24000:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, 24000)
                pcm = resample_poly(pcm, 24000 // g, sr // g)
            except Exception:
                x = np.linspace(0, len(pcm) - 1, int(len(pcm) * 24000 / sr))
                pcm = np.interp(x, np.arange(len(pcm)), pcm)
        pcm = (np.clip(pcm, -1, 1) * 32767).astype("<i2")
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(pcm.tobytes())
        return True
    try:  # non-wav: soundfile reads mp3/flac/ogg/m4a
        import soundfile as sf
        data, sr = sf.read(str(src), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 24000:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, 24000)
                data = resample_poly(data, 24000 // g, sr // g)
            except Exception:
                x = np.linspace(0, len(data) - 1, int(len(data) * 24000 / sr))
                data = np.interp(x, np.arange(len(data)), data)
        data = (np.clip(data, -1, 1) * 32767).astype("<i2")
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(data.tobytes())
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an Orpheus 3B fine-tuning dataset from clips.")
    ap.add_argument("clips", help="folder of clips (wav/mp3/flac/...), recursively")
    ap.add_argument("--out", help="Hugging Face dataset folder (default: <clips>-dataset)")
    ap.add_argument("--jsonl", help="also write a .jsonl the trainer can load directly")
    ap.add_argument("--transcripts", help="a transcripts.txt to use for clips without sidecars")
    ap.add_argument("--speaker", help="speaker name; adds the multi-speaker `source` column")
    args = ap.parse_args()

    d = Path(args.clips)
    clips = _find_clips(d)
    if not clips:
        print(f"no audio files in {d}", file=sys.stderr)
        return 1
    texts = _transcripts(clips, d, Path(args.transcripts) if args.transcripts else None)
    n_missing = sum(1 for c in clips if not texts.get(c))
    print(f"{len(clips)} clips, {n_missing} without a transcript")
    if len(clips) < 50:
        print("note: the authors report ~50 clips for a usable result, 300 for a strong one")

    rows = []
    workdir = d / ".orpheus-24k"
    workdir.mkdir(exist_ok=True)
    for i, c in enumerate(clips):
        rel = "audio/clip-%04d%s" % (i, c.suffix.lower())
        dst = workdir / rel.split("/", 1)[1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if _resample_24k_mono(c, dst):
            src = str(dst)
        else:
            src = str(c)          # keep the original; the trainer resamples on the fly
        row = {"text": texts.get(c, ""), "audio": src}
        if args.speaker:
            row["source"] = args.speaker
        rows.append(row)

    jsonl = Path(args.jsonl) if args.jsonl else Path(args.out or (str(d) + "-dataset")) / "data.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"jsonl: {jsonl}")

    if args.out or jsonl.name != "data.jsonl":
        try:
            from datasets import Dataset
            ds = Dataset.from_list(rows)
            out = Path(args.out or (str(d) + "-dataset"))
            ds.save_to_disk(str(out))
            print(f"hugging face dataset: {out} ({len(ds)} rows, columns {ds.column_names})")
        except ImportError:
            print("note: `pip install datasets` to also save a Hugging Face dataset folder")
    print(f"next: train_orpheus.py --dataset {jsonl}"
          + (f" --speaker {args.speaker}" if args.speaker else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
