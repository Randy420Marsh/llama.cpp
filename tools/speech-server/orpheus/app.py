#!/usr/bin/env python3
"""Orpheus 3B LoRA fine-tuning studio (Gradio).

A thin GUI that drives the real work as subprocesses in the speech-server
torch venv, so it can run from a minimal gradio environment (the torch
nightly and gradio 6 segfault on import next to each other here):

    Dataset   build a .jsonl training set from clips + transcripts
    Train     LoRA fine-tune (plain PyTorch, no HF Trainer) -> adapters
    Publish   convert the adapters to a GGUF (llama-server --lora)
    Synthesize  zero-shot clone + LoRA through the base model -> WAV
    Status    GPU / tooling health

Set ORPHEUS_TTS_VENV (default: <speech-server>/venv) to the torch venv and
ORPHEUS_LLAMA_CPP (default: the repo root) for the GGUF tools.
"""
from __future__ import annotations

import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Generator

import gradio as gr

SS = Path(__file__).resolve().parent.parent          # .../tools/speech-server
VENV = Path(os.environ.get("ORPHEUS_TTS_VENV", str(SS / "venv")))
PY = VENV / ("bin/python" if os.name != "nt" else "Scripts/python.exe")
LLAMA = Path(os.environ.get("ORPHEUS_LLAMA_CPP", str(SS.parent.parent)))
ORPHEUS_DIR = Path(os.environ.get("ORPHEUS_OUT", "/home/john/AI/tts-models/orpheus"))
ORPHEUS_DIR.mkdir(parents=True, exist_ok=True)
VOICES = Path(os.environ.get("ORPHEUS_VOICES", "/home/john/AI/tts-models/voices"))
ASR = Path(os.environ.get("ORPHEUS_ASR", str(SS.parent / "asr-server")))
import trimtools  # same dir: ffmpeg probe/peaks/trim/-23 LUFS export
ASR_PY = ASR / "venv" / ("bin/python" if os.name != "nt" else "Scripts/python.exe")

ENV = dict(os.environ)
ENV["HF_HOME"] = os.environ.get("HF_HOME", "/home/john/AI/tts-models/hf")
ENV["GRADIO_ANALYTICS_ENABLED"] = "False"


def _run(cmd: list[str], cwd: str | Path = ".", env: dict | None = None) -> str:
    """Run a command, capturing combined output (for quick, non-streamed jobs)."""
    r = subprocess.run(cmd, cwd=str(cwd), env=env or ENV,
                       capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).splitlines()[-40:])
    return f"exit {r.returncode}\n{tail}"


def stream(cmd: list[str], cwd: str | Path = ".", env: dict | None = None) -> Generator[tuple[str, str], None, None]:
    """Stream a command's output to the UI line by line (long jobs)."""
    buf = ""
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env or ENV,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        buf += line
        yield buf, buf
    rc = proc.wait()
    buf += f"\n--- exit {rc} ---\n"
    yield buf, buf


def _voice_choices():
    out = []
    if VOICES.is_dir():
        for p in sorted(VOICES.iterdir()):
            ref = next((p / f"reference{e}" for e in (".wav", ".mp3", ".flac")
                        if (p / f"reference{e}").is_file()), None)
            if not ref:
                continue
            tr = p / "transcript.txt"
            label = p.name
            if tr.is_file():
                t = tr.read_text(encoding="utf-8", errors="replace").strip()
                if t:
                    label += f"  (\"{t[:20]}{'...' if len(t) > 20 else ''}\")"
            out.append((label, str(ref)))
    return out


def _voice_names() -> list[str]:
    """Folder names under VOICES that have a reference clip (for the trim tab)."""
    return [Path(ref).parent.name for _label, ref in _voice_choices()]


# --- job runners -------------------------------------------------------------

def job_dataset(clips: str, transcripts: str, out: str, speaker: str) -> Generator[tuple[str, str], None, None]:
    if not (out or True):
        out = str(SS / "orpheus-dataset")
    cmd = [str(PY), str(SS / "setup/make_orpheus_dataset.py"), clips,
           "--out", out, "--jsonl", Path(out) / "data.jsonl"]
    if transcripts:
        cmd += ["--transcripts", transcripts]
    if speaker:
        cmd += ["--speaker", speaker]
    yield from stream(cmd)


def stage_clips(files, target: str) -> tuple[str, str]:
    """Copy dropped clips into a folder the transcribe/dataset jobs can use.

    Gradio hands us the files' *server-side temp paths*; we copy them (preserving
    the original names) into a clips folder so the rest of the flow works on real
    files. Returns the folder path (into the clips textbox) plus a summary.
    """
    import shutil
    paths = []
    for f in files or []:
        p = f.name if hasattr(f, "name") else str(f)
        if p:
            paths.append(Path(p))
    if not paths:
        return target, "No files dropped yet."
    folder = Path(target) if target else ORPHEUS_DIR / "clips"
    folder.mkdir(parents=True, exist_ok=True)
    n, skipped = 0, 0
    for src in paths:
        dst = folder / src.name
        if dst.exists():
            skipped += 1
            continue
        shutil.copy2(src, dst)
        n += 1
    msg = f"{n} clip(s) staged into {folder}" + (f" ({skipped} same-name skipped)" if skipped else "")
    return str(folder), msg


def job_transcribe(clips: str, language: str, skip: bool) -> Generator[tuple[str, str], None, None]:
    """Transcribe a clips folder with local Whisper large-v3 (most accurate on this box).

    Writes a sidecar .txt per clip (what make_orpheus_dataset.py pairs by) so you can
    review/correct each before building the dataset. Runs in the ASR server's venv
    (the one with faster-whisper); the model loads ~30-60 s once, then each clip is
    a few seconds.
    """
    tool = ASR / "transcribe_orpheus_clips.py"
    if not ASR_PY.is_file() or not tool.is_file():
        msg = (f"ASR venv or transcribe tool not found:\n  {ASR_PY}\n  {tool}\n"
               "run tools/asr-server/run.sh once to set it up")
        yield msg, "transcribe"
        return
    env = dict(ENV)
    env["ASR_MODELS_DIR"] = env.get("ASR_MODELS_DIR", "/home/john/AI/tts-models/whisper")
    cmd = [str(ASR_PY), str(tool), clips, "--model", "large-v3",
           "--combined", str(Path(clips) / "transcripts.txt")]
    if language:
        cmd += ["--language", language]
    if skip:
        cmd.append("--skip-existing")
    msg = "loading Whisper large-v3 (one-time ~30-60 s, ~4 GB VRAM) ...\n"
    yield msg, msg
    yield from stream(cmd, env=env)


def job_train(dataset: str, out: str, rank: int, epochs: int, lr: float,
              batch: int, seq_len: int, max_steps: int) -> Generator[tuple[str, str], None, None]:
    out = out or str(ORPHEUS_DIR / "lora")
    cmd = [str(PY), str(SS / "orpheus/train_lora.py"),
           "--dataset", dataset, "--out", out,
           "--rank", str(rank), "--epochs", str(epochs), "--lr", str(lr),
           "--batch", str(batch), "--seq-len", str(seq_len)]
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    yield from stream(cmd)


def _base_config_dir() -> str | None:
    """The local HF snapshot that holds the base model's config.json/tokenizer.

    The base model (canopylabs/orpheus-3b-0.1-ft) is gated, so the converter is
    pointed at the cached snapshot and run offline instead of pulling from the hub.
    """
    import glob
    cands = glob.glob(str(Path(ENV["HF_HOME"]) / "hub/models--canopylabs--orpheus-3b-0.1-ft/snapshots/*/"))
    return cands[0] if cands else None


def job_publish(lora: str, outfile: str) -> Generator[tuple[str, None], None, None]:
    outfile = outfile or str(ORPHEUS_DIR / "lora.gguf")
    env = dict(ENV)
    env["PYTHONPATH"] = f"{LLAMA}{os.pathsep}{SS}"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    cmd = [str(PY), str(LLAMA / "convert_lora_to_gguf.py"), lora,
           "--outfile", outfile, "--outtype", "f32"]
    base = _base_config_dir()
    if base:
        cmd += ["--base", base]
    else:
        yield "no local base-model snapshot found; the converter will try the hub (needs HF_TOKEN for the gated model)\n"
    buf = ""
    proc = subprocess.Popen(cmd, cwd=str(SS), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        buf += line
        yield buf
    buf += f"\n--- exit {proc.wait()} ---\n"
    yield buf


def job_synth(lora: str, ref: str, ref_text: str, text: str, out: str,
              no_lora: bool) -> Generator[tuple[str, str | None], None, None]:
    out = out or str(ORPHEUS_DIR / "out.wav")
    cmd = [str(PY), str(SS / "orpheus/test_synthesis.py"),
           "--lora", lora, "--ref", ref, "--ref-text", ref_text,
           "--text", text, "--out", out]
    if no_lora:
        cmd.append("--no-lora")
    buf = ""
    proc = subprocess.Popen(cmd, cwd=str(SS), env=ENV,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        buf += line
        yield buf, None
    buf += f"\n--- exit {proc.wait()} ---\n"
    wav = out if Path(out).is_file() else None
    yield buf, wav


def status() -> tuple[str, str]:
    lines = []
    lines.append(f"torch venv : {VENV} ({'ok' if PY.is_file() else 'MISSING'})")
    lines.append(f"llama.cpp  : {LLAMA} ({'ok' if (LLAMA / 'convert_lora_to_gguf.py').is_file() else 'MISSING'})")
    lines.append(f"GGUF out   : {ORPHEUS_DIR}")
    lines.append(f"voices     : {VOICES} ({len(_voice_choices())} reference(s))")
    try:
        r = subprocess.run([str(PY), "-c",
                            "import torch;f,t=torch.cuda.mem_get_info();print(f'GPU {f/1e9:.1f}/{t/1e9:.1f} GB free; torch {torch.__version__}')"],
                           capture_output=True, text=True, timeout=120)
        lines.append(r.stdout.strip() or r.stderr.strip().splitlines()[-1])
    except Exception as e:
        lines.append(f"GPU check failed: {str(e).splitlines()[0]}")
    return "\n".join(lines), ""


# --- Trim & Export tool ------------------------------------------------------
# Queue of dropped videos/audio: trim an A-B segment (or the whole file when the
# toggle is off) and export 24 kHz mono 16-bit WAV normalized to -23 LUFS into
# the chosen voice's clips folder (VOICES/<voice>/).

_TRIM_QUEUE: list[dict] = []
_trim_seq = itertools.count(1)


def _trim_fmt_t(s: float) -> str:
    s = max(0.0, s)
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _trim_queue_rows() -> list[list[str]]:
    return [[str(i), it["name"], it["kind"], f"{it['dur']:.1f}",
             f"{it.get('exports', 0)}", it["status"]]
            for i, it in enumerate(_TRIM_QUEUE)]


def _trim_pending() -> dict | None:
    for it in _TRIM_QUEUE:
        if it["status"] == "pending":
            return it
    return None


def _trim_wave(path: str | None, a: float, b: float, dur: float):
    """PIL waveform: full-file peak bars, A-B selection outlined in green."""
    from PIL import Image, ImageDraw
    W, H = 900, 160
    img = Image.new("RGB", (W, H), (16, 20, 28))
    d = ImageDraw.Draw(img)
    if path:
        try:
            pk = trimtools.peaks(Path(path), W)
        except Exception:
            pk = []
        for i, v in enumerate(pk):
            hgt = max(2, int(v * (H - 24)))
            d.rectangle([i, (H - hgt) // 2, i + 1, (H + hgt) // 2], fill=(90, 200, 250))
    if dur > 0:
        x1 = int(max(0.0, min(a, dur)) / dur * W)
        x2 = int(max(a, min(b, dur)) / dur * W)
        d.rectangle([x1, 0, max(x2, x1 + 2), H], outline=(80, 255, 120), width=2)
    return img


def _trim_len_hint(a: float, b: float) -> str:
    L = max(0.0, b - a)
    ok = trimtools.REC_MIN <= L <= trimtools.REC_MAX
    tip = "in the 6-15 s recommendation" if ok else f"outside the 6-{trimtools.REC_MAX:.0f} s recommendation"
    return f"{L:.2f} s  ({tip})"


def trim_stage(files, voice: str, do_trim: bool) -> tuple:
    """Add dropped media to the queue. Toggle off -> export the full clips now
    (already trimmed); on -> they wait for one-by-one trimming ('Load next')."""
    empty = ("", None, _trim_wave(None, 0.0, 0.0, 0.0), "", 0.0, 0.0, "", "")
    err = trimtools.ensure_ffmpeg()
    if err:
        return _trim_queue_rows(), err, *empty
    staged = ORPHEUS_DIR / "staging"
    staged.mkdir(parents=True, exist_ok=True)
    added = 0
    for f in files or []:
        p = Path(getattr(f, "name", str(f)))
        if not p.is_file():
            continue
        dst = staged / f"{next(_trim_seq):02d}_{p.name}"
        try:
            shutil.copy2(p, dst)
            info = trimtools.probe(dst)
        except Exception:
            continue
        if not info["has_audio"]:
            continue
        kind = "video" if info["has_video"] else "audio"
        audio = None
        if kind == "video":
            # extract the audio track for the synced preview player (fast, 128k m4a)
            audio = dst.with_suffix(".m4a")
            r = trimtools._ffmpeg("-i", str(dst), "-map", "0:a:0",
                                  "-c:a", "aac", "-b:a", "128k", "-y", str(audio))
            if r.returncode != 0 or not audio.is_file():
                audio = None
        _TRIM_QUEUE.append({"name": p.name, "path": str(dst), "exports": 0,
                            "kind": kind, "audio": str(audio) if audio else None,
                            "dur": info["duration"], "status": "pending", "out": ""})
        added += 1
    if not added:
        return _trim_queue_rows(), "No readable audio/video files in the drop.", *empty
    if do_trim:
        return _trim_queue_rows(), \
            f"{added} item(s) queued. Trim ON: click 'Load next', set A/B, export.", *empty
    msg = f"{added} item(s). Trim OFF - exporting full clips to {VOICES / (voice or 'clips')}:"
    target = VOICES / (voice or "clips")
    for it in _TRIM_QUEUE:
        if it["status"] != "pending":
            continue
        try:
            out = trimtools.next_clip_name(target)
            r = trimtools.export_clip(Path(it["path"]), out, None, None)
            it["status"], it["out"], it["exports"] = "done", r["path"], 1
            msg += f"\n  {it['name']} -> {out.name} ({r['duration']:.1f} s, -23 LUFS)"
        except Exception as exc:
            it["status"] = "error"
            msg += f"\n  {it['name']} FAILED: {exc}"
    return _trim_queue_rows(), msg, *empty


# Preview output-group order, returned by every queue action:
#   video, audio, waveform, info, A, B, length-hint, current-source-path
def _trim_preview_empty() -> tuple:
    return ("", None, _trim_wave(None, 0.0, 0.0, 0.0),
            "Queue finished. Drop more files, or Clear to start over.",
            0.0, 0.0, "", "")


def _trim_preview_of(it: dict, a: float, b: float) -> tuple:
    dur = it["dur"]
    info = (f"#{_TRIM_QUEUE.index(it)}  {it['name']}  ({it['kind']}, {dur:.1f} s) - "
            f"seek the video, set A and B on the waveform, then **Trim & export**. "
            f"That saves the clip to the voice folder and keeps this item loaded, so "
            f"you can cut several clips out of one long video: adjust A/B and export "
            f"again. When done with this item, press **Next item**.")
    return (it["path"], it.get("audio") or None,
            _trim_wave(it["path"], a, b, dur), info, a, b,
            _trim_len_hint(a, b), it["path"])


def trim_load() -> tuple:
    """Load the next pending item into the preview (video + waveform + A/B)."""
    it = _trim_pending()
    if not it:
        return "No pending items left.", *_trim_preview_empty()
    dur = it["dur"]
    a, b = 0.0, (min(trimtools.REC_MAX, dur) if dur else 0.0)
    return f"Now trimming item #{_TRIM_QUEUE.index(it)}.", *_trim_preview_of(it, a, b)


def _trim_find(curpath: str) -> dict | None:
    for it in _TRIM_QUEUE:
        if curpath and it["path"] == curpath:
            return it
    return None


def trim_marks(curpath: str, a: float, b: float) -> tuple:
    """A/B moved: redraw the selection on the waveform + the length hint."""
    it = _trim_find(curpath)
    dur = it["dur"] if it else 0.0
    return _trim_wave(curpath or None, a or 0.0, b or 0.0, dur), \
        _trim_len_hint(a or 0.0, b or 0.0)


def trim_export(a: float, b: float, voice: str, curpath: str) -> tuple:
    """Save [a, b) to the voice's clips folder. Keeps the item loaded, so you
    can cut several clips from one long video; 'Next item' then advances."""
    it = _trim_find(curpath)
    if not it:
        return _trim_queue_rows(), "Load an item first (Load next / Add to queue).", \
            *_trim_preview_empty()
    a, b = float(a or 0.0), float(b or 0.0)
    if b - a < 0.3:
        return _trim_queue_rows(), "B must be at least 0.3 s after A.", \
            *_trim_preview_empty()
    try:
        out = trimtools.next_clip_name(VOICES / (voice or "clips"))
        r = trimtools.export_clip(Path(it["path"]), out, a, b)
        it["exports"] = it.get("exports", 0) + 1
        it["out"] = r["path"]
        it["status"] = "done"
        msg = (f"Saved {it['name']} [{_trim_fmt_t(a)}-{_trim_fmt_t(b)}] -> "
               f"{out.name} ({r['duration']:.1f} s, -23 LUFS, 24 kHz mono). "
               f"{it['exports']} clip(s) from this item so far. "
               f"Adjust A/B and export again, or press Next item.")
    except Exception as exc:
        msg = f"Export failed: {exc}"
    # keep the item loaded (re-draw the waveform with the same A/B) so several
    # clips can be cut from one long video; 'Next item' advances the queue.
    return _trim_queue_rows(), msg, _trim_preview_of(it, a, b)


def trim_next(curpath: str) -> tuple:
    """Mark the current item processed and load the next pending one."""
    it = _trim_find(curpath)
    if it and it["status"] == "pending":
        it["status"] = "done" if it.get("exports", 0) > 0 else "skipped"
    _status, *prev = trim_load()
    note = f"Marked {it['name']} {'processed' if it.get('exports', 0) else 'skipped'}." \
        if it else "No current item."
    return _trim_queue_rows(), note, *prev


def trim_clear() -> tuple:
    _TRIM_QUEUE.clear()
    staged = ORPHEUS_DIR / "staging"
    if staged.is_dir():
        for p in staged.iterdir():
            try:
                p.unlink()
            except OSError:
                pass
    return _trim_queue_rows(), "Queue cleared (staging files removed).", \
        *_trim_preview_empty()


# --- UI ----------------------------------------------------------------------

with gr.Blocks(title="Orpheus 3B LoRA Studio") as demo:
    gr.Markdown("# Orpheus 3B — LoRA fine-tuning studio\n"
                "Build a voice dataset, train a LoRA on it, publish it as a GGUF adapter "
                "(`llama-server --lora`) and A/B-test it against the base model.")

    with gr.Tab("Status"):
        st_txt = gr.Textbox(label="Environment", lines=7)
        st_btn = gr.Button("Check")
        st_btn.click(status, None, [st_txt, gr.Textbox(visible=False)])

    with gr.Tab("Dataset"):
        d_clips = gr.Textbox(label="Folder of clips (wav/mp3/flac, recursive)", value="")
        d_files = gr.File(label="…or drag & drop clips here (any mix of wav/mp3/flac/ogg/m4a/webm/opus)",
                         file_count="multiple", file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"])
        d_stage = gr.Button("Stage dropped clips into a folder")
        d_stage_out = gr.Textbox(label="Staged to", visible=True, value="")
        d_stage.click(stage_clips, [d_files, d_clips], [d_clips, d_stage_out])
        gr.Markdown("*Drag & drop your clips above, hit **Stage** (they're copied into a folder and "
                    "the folder path is filled in for you), or just type a folder path. Then "
                    "**Transcribe** to get the `.txt` files, **Build dataset** to make the `.jsonl`.*")
        d_trans = gr.Textbox(label="Optional transcripts.txt (one line per clip, in folder order)")
        d_speaker = gr.Textbox(label="Optional speaker name (multi-speaker `source` column)")
        d_out = gr.Textbox(label="Output dataset folder", value="")
        d_log = gr.Textbox(label="Output", lines=10)
        d_btn = gr.Button("Build dataset", variant="primary")
        d_btn.click(job_dataset, [d_clips, d_trans, d_speaker, d_out], [d_log, gr.Textbox(visible=False)])
        d_lang = gr.Textbox(label="Transcribe: language code (blank = auto-detect)")
        d_skip = gr.Checkbox(label="Skip clips that already have a .txt", value=False)
        d_tbtn = gr.Button("Transcribe clips (Whisper large-v3)", variant="secondary")
        d_tbtn.click(job_transcribe, [d_clips, d_lang, d_skip], [d_log, gr.Textbox(visible=False)])
        gr.Markdown("*Transcribe* writes a `.txt` next to each clip (and `transcripts.txt`), "
                    "which *Build dataset* then pairs automatically. Review the texts first -- "
                    "Whisper is accurate but not perfect, and the transcript becomes the voice's "
                    "training target.")

    with gr.Tab("Trim & Export"):
        gr.Markdown("**Extract audio from videos, trim 6-15 s clips, export as 24 kHz mono WAV @ -23 LUFS** "
                    "into the chosen voice's clips folder (for Orpheus training).\n"
                    "Drop video or audio files. **Trim OFF** = export the whole clips straight away. "
                    "**Trim ON** = each item is queued; click **Load next**, seek the video "
                    "(audio + waveform stay in sync), set **A/B** on the waveform, then **Trim & export** "
                    "one by one until the queue is done. The green box is your A-B selection; "
                    "exporting the same source repeatedly lets you cut several clips out of one long video.")
        with gr.Row():
            _vnames = _voice_names() or ["clips"]
            tv_voice = gr.Dropdown(_vnames, label="Voice model (clips saved to its folder)",
                                  value=_vnames[0])
            tv_trim = gr.Checkbox(value=False, label="Trim tool ON (ask one by one)")
        with gr.Row():
            tv_files = gr.File(label="Drop videos / audio here (mp4 mkv mov avi webm ts / wav mp3 flac ogg m4a opus wma)",
                               file_count="multiple",
                               file_types=[".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts",
                                           ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".wma"])
            tv_stage = gr.Button("Add to queue / export", variant="primary")
            tv_clear = gr.Button("Clear queue")
        tv_msg = gr.Textbox(label="Status", lines=4)
        tv_table = gr.Dataframe(_trim_queue_rows(),
                                headers=["#", "file", "kind", "dur(s)", "clips", "status"],
                                label="Queue", interactive=False)
        with gr.Row():
            with gr.Column(scale=2):
                tv_video = gr.Video(label="Preview (videos only)")
                tv_audio = gr.Audio(label="Synced audio (the extracted track / the file itself)")
            with gr.Column(scale=3):
                tv_wave = gr.Image(label="Waveform (green box = A-B selection)", width=900, height=160)
                with gr.Row():
                    tv_a = gr.Number(value=0.0, label="A (s)", precision=2, scale=1)
                    tv_b = gr.Number(value=0.0, label="B (s)", precision=2, scale=1)
                    tv_len = gr.Textbox(label="Clip length", interactive=False, scale=2)
                with gr.Row():
                    tv_load = gr.Button("Load next", variant="secondary")
                    tv_export = gr.Button("Trim & export", variant="primary")
                    tv_next = gr.Button("Next item")
        tv_info = gr.Textbox(label="Item help", value="", interactive=False,
                             lines=2, visible=True)
        tv_cur = gr.Textbox(label="Current source", visible=False, value="")
        _tv_prev = [tv_video, tv_audio, tv_wave, tv_info, tv_a, tv_b, tv_len, tv_cur]
        tv_stage.click(trim_stage, [tv_files, tv_voice, tv_trim], [tv_table, tv_msg] + _tv_prev)
        tv_load.click(trim_load, None, [tv_msg] + _tv_prev)
        tv_export.click(trim_export, [tv_a, tv_b, tv_voice, tv_cur], [tv_table, tv_msg] + _tv_prev)
        tv_next.click(trim_next, [tv_cur], [tv_table, tv_msg] + _tv_prev)
        tv_clear.click(trim_clear, None, [tv_table, tv_msg] + _tv_prev)
        tv_a.change(trim_marks, [tv_cur, tv_a, tv_b], [tv_wave, tv_len])
        tv_b.change(trim_marks, [tv_cur, tv_a, tv_b], [tv_wave, tv_len])

    with gr.Tab("Train (LoRA)"):
        t_ds = gr.Textbox(label="Dataset (.jsonl or HF folder)", value="")
        t_out = gr.Textbox(label="Output adapter folder", value="")
        with gr.Row():
            t_rank = gr.Slider(4, 64, 16, step=4, label="LoRA rank")
            t_ep = gr.Slider(1, 20, 2, step=1, label="epochs")
        with gr.Row():
            t_lr = gr.Number(2e-4, label="learning rate")
            t_bs = gr.Slider(1, 8, 2, step=1, label="batch (seqs/step)")
        with gr.Row():
            t_seq = gr.Slider(256, 4096, 1024, step=256, label="max sequence length")
            t_max = gr.Number(0, label="max steps (0 = all epochs)")
        t_log = gr.Textbox(label="Output", lines=12)
        t_btn = gr.Button("Train LoRA", variant="primary")
        t_btn.click(job_train, [t_ds, t_out, t_rank, t_ep, t_lr, t_bs, t_seq, t_max], [t_log, gr.Textbox(visible=False)])

    with gr.Tab("Publish (GGUF)"):
        p_lora = gr.Textbox(label="Trained LoRA folder (with adapter_model.safetensors)", value="")
        p_out = gr.Textbox(label="Output GGUF adapter path", value="")
        p_log = gr.Textbox(label="Output", lines=8)
        p_btn = gr.Button("Convert LoRA -> GGUF", variant="primary")
        p_btn.click(job_publish, [p_lora, p_out], [p_log])

    with gr.Tab("Synthesize (test)"):
        s_lora = gr.Textbox(label="LoRA folder (or leave for base model)", value="")
        _vc = [c[1] for c in _voice_choices()]
        s_ref = gr.Dropdown(_vc or ["(pick a reference clip)"], label="Reference voice clip (or paste a path)",
                            value=(_vc[0] if _vc else ""), allow_custom_value=True)
        s_ref_text = gr.Textbox(label="What the reference clip says")
        s_text = gr.Textbox(label="Text to speak", value="")
        s_out = gr.Textbox(label="Output WAV path", value="")
        s_nol = gr.Checkbox(label="Base model only (A/B control)", value=False)
        s_log = gr.Textbox(label="Output", lines=8)
        s_wav = gr.Audio(label="Result (24 kHz)")
        s_btn = gr.Button("Synthesize", variant="primary")
        s_btn.click(job_synth, [s_lora, s_ref, s_ref_text, s_text, s_out, s_nol], [s_log, s_wav])

if __name__ == "__main__":
    demo.queue().launch(server_name=os.environ.get("ORPHEUS_HOST", "127.0.0.1"),
                        server_port=int(os.environ.get("ORPHEUS_PORT", "7861")))
