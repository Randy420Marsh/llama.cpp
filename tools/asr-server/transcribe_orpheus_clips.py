#!/usr/bin/env python3
"""
Transcribe a folder of voice clips into the transcripts that
make_orpheus_dataset.py pairs with the audio.

Uses the local Whisper large-v3 (faster-whisper) -- the most accurate
model on this box, the same one the ASR server (tools/asr-server) serves.
Run it with the ASR server's venv (it has faster-whisper):

    tools/asr-server/venv/bin/python tools/asr-server/transcribe_orpheus_clips.py clips/

Output, in order of preference:
  * sidecar files, one per clip:  0001.wav -> 0001.txt
    (make_orpheus_dataset.py picks these up automatically)
  * --combined transcripts.txt    one line per clip, sorted by name
    (the dataset builder's second pairing method)
  * --server http://...          if the ASR server is already running,
    transcribe through it instead of loading the model here

Language: pass --language en (or fi, de, ...) to force it; by default
each clip's language is auto-detected, which is fine for mixed folders
but slightly less accurate than a forced language on clean recordings.

Accuracy notes:
  * large-v3 is the right call for voice training data: the transcripts
    become the "text" side of what the model learns to say, so errors
    here leak straight into the fine-tuned voice.
  * keep clips 6-15 s of natural speech, one speaker, quiet room
    (see make_orpheus_dataset.py's data notes)
  * review the .txt files before training -- a 10-second listen per clip
    beats a wrong transcript
  * large-v3-turbo is ~4x faster but a little less accurate; pass
    --model large-v3-turbo if you have a lot of audio (it must be under
    ~/AI/tts-models/whisper/, downloaded by the ASR server on first use)
"""
import argparse
import json
import os
import ssl
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"}
DEFAULT_MODEL = "large-v3"
DEFAULT_MODELS_DIR = "/home/john/AI/tts-models/whisper"


def _find_clips(d: Path) -> list[Path]:
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            out.append(p)
    return out


def _resample(x, sr: int) -> "np.ndarray":
    import numpy as np
    if sr == 16000:
        return x
    n = int(len(x) * 16000 / sr)
    # float32: faster-whisper's VAD (ONNX) rejects float64
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")


def _load_audio_16k(path: Path):
    """Mono 16 kHz float32 numpy array (faster-whisper's input format).

    soundfile (mp3/flac/...) -> stdlib wave (wav) -> ffmpeg (anything).
    """
    import numpy as np
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return _resample(data.mean(axis=1), sr), 16000
    except ImportError:
        pass
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wf:
            sr, ch = wf.getframerate(), wf.getnchannels()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype("float32") / 32768.0
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)
        return _resample(pcm, sr), 16000
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                         "-f", "f32le", "-"], capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"cannot decode {path.name} (need soundfile, wav, or ffmpeg): {out.stderr.decode()[:200]}")
    return np.frombuffer(out.stdout, dtype="<f4"), 16000


def _transcribe_server(url: str, clip: Path, language: str | None) -> str:
    boundary = "----orpheusclip"
    body = bytearray()

    def part(name: str, value: str, content: bytes | None = None, ctype: str | None = None) -> None:
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"".encode()
        if content is not None:
            body += f"; filename=\"{clip.name}\"".encode()
            if ctype:
                body += f"\r\nContent-Type: {ctype}".encode()
            body += b"\r\n\r\n" + content + b"\r\n"
        else:
            body += b"\r\n\r\n" + value.encode() + b"\r\n"

    part("file", "", clip.read_bytes())
    part("model", "")  # the server's loaded model
    if language:
        part("language", language)
    part("response_format", "json")
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/audio/transcriptions", data=bytes(body), method="POST",
                                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    ctx = ssl.create_default_context()
    if url.startswith("https"):
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, context=ctx, timeout=600) as r:
        return json.loads(r.read().decode())["text"].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", help="folder of clips (searched recursively)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"faster-whisper model (default {DEFAULT_MODEL}; the most accurate local one)")
    ap.add_argument("--models-dir", default=os.environ.get("ASR_MODELS_DIR", DEFAULT_MODELS_DIR))
    ap.add_argument("--language", default=None, help="force a language code (en, fi, ...); default: auto per clip")
    ap.add_argument("--combined", metavar="FILE",
                    help="also write one line per clip (sorted) to FILE (transcripts.txt format)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="don't re-transcribe clips that already have a sidecar .txt")
    ap.add_argument("--server", metavar="URL",
                    help="transcribe through a running ASR server (http[s]://host:8178) instead of loading the model")
    args = ap.parse_args()

    clips = _find_clips(Path(args.clips))
    if not clips:
        print(f"no audio clips found in {args.clips}", file=sys.stderr)
        return 1

    if args.server:
        print(f"transcribing {len(clips)} clip(s) through {args.server} ...")
        fn = lambda c: _transcribe_server(args.server, c, args.language)
    else:
        import cuda_dll_dirs  # preload cuBLAS/cuDNN from the nvidia wheels (CTranslate2)
        from faster_whisper import WhisperModel
        model_dir = Path(args.models_dir) / args.model.replace("/", "--")
        if (model_dir / "model.bin").is_file():
            src = str(model_dir)
        else:
            src = args.model  # let faster-whisper download into its default cache
        print(f"loading {args.model} (one-time load ~30-60 s, ~4 GB VRAM) ...")
        model = WhisperModel(src, device="cuda", compute_type="float16")

        def fn(c: Path) -> str:
            audio, sr = _load_audio_16k(c)
            segments, _info = model.transcribe(audio, language=args.language,
                                               vad_filter=True, beam_size=5)
            return "".join(s.text for s in segments).strip()

    ok, skipped = 0, 0
    for i, c in enumerate(clips, 1):
        sidecar = c.with_suffix(".txt")
        if args.skip_existing and sidecar.is_file():
            skipped += 1
            print(f"[{i}/{len(clips)}] {c.name}  (existing sidecar, skipped)")
            continue
        try:
            text = fn(c)
        except Exception as exc:
            print(f"[{i}/{len(clips)}] {c.name}  FAILED: {exc}", file=sys.stderr)
            continue
        sidecar.write_text(text + "\n", encoding="utf-8")
        ok += 1
        print(f"[{i}/{len(clips)}] {c.name}  ->  {text[:70]}")

    combined = None
    if args.combined:
        combined = Path(args.combined)
        combined.parent.mkdir(parents=True, exist_ok=True)
        # one line per clip in name order; clips without a sidecar come out empty
        lines = []
        for c in clips:
            sc = c.with_suffix(".txt")
            lines.append(sc.read_text(encoding="utf-8").strip() if sc.is_file() else "")
        combined.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\ndone: {ok} transcribed, {skipped} skipped"
          + (f", combined -> {combined}" if combined else ""))
    if ok:
        print("next: review the .txt files, then build the dataset:")
        print(f"  make_orpheus_dataset.py {args.clips} --out <dataset> --jsonl <dataset>/data.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
