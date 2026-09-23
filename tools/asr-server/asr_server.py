"""OpenAI-compatible speech-to-text server backed by faster-whisper.

Serves POST /v1/audio/transcriptions (multipart: file, model, language, prompt, response_format,
temperature, timestamp_granularities[]) so llama-server and ninfer-serve can transcribe the audio
track of video/audio inputs for models without a native audio encoder.

Everything stays in memory: uploaded bytes are decoded from RAM (PyAV via faster-whisper), the
Whisper model stays resident in VRAM, nothing is written to disk. With --idle-unload the model is
released after N idle seconds and reloaded on the next request.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path


def _add_cuda_dll_dirs() -> None:
    # CTranslate2 needs cuBLAS/cuDNN; the nvidia-* wheels ship them outside the default DLL path.
    if sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    roots = [Path(p) for p in nvidia.__path__]
    for root in roots:
        for sub in root.glob("*/bin"):
            os.add_dll_directory(str(sub))
            os.environ["PATH"] = str(sub) + os.pathsep + os.environ.get("PATH", "")


_add_cuda_dll_dirs()

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402
from faster_whisper.utils import _MODELS  # noqa: E402

log = logging.getLogger("asr")

# files faster-whisper needs from a model repo (same list as faster_whisper.utils.download_model)
MODEL_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
MODEL_SIZES = {"large-v3": "~3 GB", "large-v3-turbo": "~1.6 GB", "distil-large-v3.5": "~1.5 GB"}
# VRAM a loaded float16 model takes, weights + decoding buffers (GB); unknown models assume large-v3
MODEL_VRAM_GB = {"large-v3": 4.0, "large-v3-turbo": 2.5, "distil-large-v3.5": 2.5}
# CPU has no float16: fall back to full precision (int8 costs accuracy), not to int8
CPU_COMPUTE = {"float16": "float32", "bfloat16": "float32",
               "int8_float16": "int8_float32", "int8_bfloat16": "int8_float32"}


def free_vram_gb() -> float | None:
    """Free VRAM of the emptiest GPU in GB, or None if nvidia-smi is unavailable."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return max(float(x) for x in out) / 1024 if out else None
    except Exception:
        return None


class ModelHolder:
    """One resident Whisper model; transcriptions are serialized on the GPU."""

    def __init__(self, name: str, device: str, compute_type: str, idle_unload: float,
                 models_dir: Path, vram_margin: float = 2.0) -> None:
        self.name = name
        self.device_request = device
        self.compute_request = compute_type
        self.device = device  # what the loaded model actually runs on
        self.compute_type = compute_type
        self.idle_unload = idle_unload
        self.models_dir = models_dir
        self.vram_margin = vram_margin
        self.state = "not loaded"  # not loaded | downloading | loading | loaded | error: ...
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()
        self._last_use = time.monotonic()
        if idle_unload > 0:
            threading.Thread(target=self._reaper, daemon=True).start()

    def _resolve(self) -> str:
        """Local model directory, downloading it on first use.

        Downloads go into a plain folder (models_dir/<name>): the default Hugging Face cache uses
        symlinks, which Windows refuses without Developer Mode (WinError 1314). A model that is
        already there is used as-is, without contacting Hugging Face."""
        if Path(self.name).is_dir():
            return self.name
        target = self.models_dir / self.name.replace("/", "--")
        if (target / "model.bin").is_file():
            return str(target)
        repo = _MODELS.get(self.name, self.name)
        if "/" not in repo:
            raise ValueError(f"unknown model {self.name!r}: use one of {', '.join(_MODELS)}, "
                             f"a Hugging Face repo id, or a local CTranslate2 model directory")
        import huggingface_hub
        self.state = "downloading"
        # faster-whisper's own download_model hides the progress bar; show it, since this is
        # a multi-GB download that otherwise looks like a hang
        log.info("downloading %s (%s) from %s to %s - first run only, progress below ...", self.name,
                 MODEL_SIZES.get(self.name, "size unknown"), repo, target)
        path = huggingface_hub.snapshot_download(repo, local_dir=str(target), allow_patterns=MODEL_FILES)
        log.info("download complete: %s", path)
        return path

    def _pick_device(self) -> str:
        """--device auto: the GPU only when the model fits with vram_margin GB to spare (so it
        does not squeeze a llama-server sharing the card), else the CPU. Re-evaluated on every
        load, so a model reloaded after --idle-unload adapts to what is free by then."""
        if self.device_request != "auto":
            return self.device_request
        need = MODEL_VRAM_GB.get(self.name, 4.0) + self.vram_margin
        free = free_vram_gb()
        if free is None or free >= need:
            return "cuda"
        log.warning("only %.1f GB VRAM free, %s needs ~%.1f GB incl. %.1f GB margin - using the CPU",
                    free, self.name, need, self.vram_margin)
        return "cpu"

    def _create(self, path: str, device: str) -> WhisperModel:
        compute = self.compute_request
        if device == "cpu" and compute in CPU_COMPUTE:
            compute = CPU_COMPUTE[compute]
            log.info("CPU has no %s: using %s (full precision rather than lossy int8)",
                     self.compute_request, compute)
        log.info("loading %s on %s (%s) ...", self.name, device, compute)
        model = WhisperModel(path, device=device, compute_type=compute)
        self.device, self.compute_type = device, compute
        return model

    def _load(self) -> WhisperModel:
        if self._model is None:
            try:
                path = self._resolve()
                self.state = "loading"
                t = time.monotonic()
                device = self._pick_device()
                try:
                    self._model = self._create(path, device)
                except Exception as exc:
                    if self.device_request != "auto" or device != "cuda":
                        raise
                    # CUDA libraries missing / out of memory: still serve, from the CPU
                    log.warning("GPU load failed (%s) - falling back to the CPU", exc)
                    self._model = self._create(path, "cpu")
            except Exception as exc:
                self.state = f"error: {exc}"
                raise
            self.state = "loaded"
            log.info("loaded %s on %s (%s) in %.1fs", self.name, self.device, self.compute_type,
                     time.monotonic() - t)
        return self._model

    def _reaper(self) -> None:
        while True:
            time.sleep(5)
            with self._lock:
                if self._model is not None and time.monotonic() - self._last_use > self.idle_unload:
                    self._model = None
                    self.state = "not loaded"
                    log.info("unloaded %s after %.0fs idle (reloads on the next request)",
                             self.name, self.idle_unload)

    def transcribe(self, audio, **kwargs):
        """audio: 16 kHz mono float32 samples (numpy) or encoded bytes."""
        if isinstance(audio, (bytes, bytearray)):
            audio = io.BytesIO(audio)
        with self._lock:
            model = self._load()
            segments, info = model.transcribe(audio, **kwargs)
            segments = list(segments)  # the generator runs the model; finish under the lock
            self._last_use = time.monotonic()
            return segments, info


# Stock phrases Whisper invents over music or silence (learned from subtitle credits in its training data)
HALLUCINATIONS = {
    "thanks for watching", "thank you for watching", "thank you", "thank you very much", "you",
    "please subscribe", "like and subscribe", "subtitles by the amaraorg community",
    "subtitles by amaraorg", "bye", "the end",
}


def is_hallucination(seg) -> bool:
    """A stock phrase on a segment the model itself rates as probably not speech."""
    import re
    text = re.sub(r"[^a-z ]", "", seg.text.lower()).strip()
    return seg.no_speech_prob > 0.5 and text in HALLUCINATIONS


def active_fraction(samples, sample_rate: int = 16000, floor_dbfs: float = -40.0) -> float:
    """Share of 1 s windows louder than floor_dbfs: tells music/speech from silence."""
    import numpy as np
    n = len(samples) // sample_rate
    if n == 0:
        return 0.0
    windows = samples[: n * sample_rate].reshape(n, sample_rate)
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
    return float(np.mean(rms > 10 ** (floor_dbfs / 20)))


def build_app(holder: ModelHolder) -> FastAPI:
    app = FastAPI(title="whisper-asr-server")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": holder.name, "loaded": holder._model is not None,
                "state": holder.state, "device": holder.device, "compute_type": holder.compute_type}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": holder.name, "object": "model", "owned_by": "faster-whisper"}]}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str | None = Form(None),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        vad_filter: str = Form("auto"),
        word_timestamps: bool = Form(False),
    ):
        audio = await file.read()
        if not audio:
            raise HTTPException(400, "empty audio file")
        started = time.monotonic()
        vad = vad_filter.strip().lower()
        if vad not in ("auto", "true", "false", "1", "0", "on", "off"):
            raise HTTPException(400, "vad_filter must be auto, true or false")
        # temperature 0 means "Whisper's default schedule": greedy first, re-decoded hotter only when a
        # window comes out repetitive or unlikely (compression ratio / log-prob thresholds)
        temps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if temperature == 0.0 else temperature
        common = dict(language=language or None, initial_prompt=prompt or None, temperature=temps,
                      word_timestamps=word_timestamps, beam_size=5)
        mode = "vad"
        try:
            samples = decode_audio(io.BytesIO(audio), sampling_rate=16000)
            segments, info = holder.transcribe(samples, vad_filter=vad not in ("false", "0", "off"), **common)
            if vad == "auto":
                # VAD treats singing over instruments as non-speech and Whisper then invents a stock
                # phrase ("Thanks for watching!"). If VAD kept little of a track that is clearly not
                # silent, transcribe the whole track instead, without the previous-text conditioning
                # that makes Whisper loop on repeated lines.
                kept = sum(s.end - s.start for s in segments) / max(info.duration, 1e-6)
                active = active_fraction(samples)
                if kept < 0.2 and active > 0.5:
                    mode = f"no-vad (VAD kept {kept:.0%} of a {active:.0%} active track)"
                    segments, info = holder.transcribe(samples, vad_filter=False,
                                                       condition_on_previous_text=False, **common)
        except Exception as exc:  # decode or model failure: report as a client-visible error
            log.exception("transcription failed")
            raise HTTPException(400, f"transcription failed: {exc}") from exc
        dropped = [s.text.strip() for s in segments if is_hallucination(s)]
        if dropped:
            segments = [s for s in segments if not is_hallucination(s)]
            mode += f", dropped {len(dropped)} stock phrase(s): {dropped}"
        text = "".join(s.text for s in segments).strip()
        log.info("%.1fs audio -> %d segments in %.1fs (lang=%s, %s)", info.duration, len(segments),
                 time.monotonic() - started, info.language, mode)
        if response_format == "text":
            return PlainTextResponse(text)
        if response_format != "verbose_json":
            return {"text": text}
        out_segments = []
        for i, s in enumerate(segments):
            seg = {"id": i, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(),
                   "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                   "compression_ratio": s.compression_ratio, "temperature": s.temperature}
            if word_timestamps and s.words:
                seg["words"] = [{"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)} for w in s.words]
            out_segments.append(seg)
        return JSONResponse({"task": "transcribe", "language": info.language, "duration": info.duration,
                             "text": text, "segments": out_segments})

    return app


class _Server(uvicorn.Server):
    """uvicorn server that announces when it is actually listening (uvicorn runs at log level
    'warning' to hide per-request access lines, which also hides its own startup message)."""

    def __init__(self, config: uvicorn.Config, holder: ModelHolder) -> None:
        super().__init__(config)
        self.holder = holder

    async def startup(self, sockets=None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            url = f"http://{self.config.host}:{self.config.port}"
            loaded = ("model loaded" if self.holder._model is not None
                      else "model loads on the first request (use --preload to load it now)")
            log.info("READY - listening on %s (%s)", url, loaded)
            log.info("  health: %s/health    use with: llama-server ... --asr-url %s", url, url)


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8178)
    parser.add_argument("--model", default="large-v3",
                        help="faster-whisper model name or CTranslate2 model directory. large-v3 (default, "
                             "multilingual, most accurate); large-v3-turbo (multilingual, faster, weaker); "
                             "distil-large-v3.5 (English only, fastest)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                        help="auto (default): GPU when the model fits with --vram-margin GB to spare, "
                             "else CPU (also when the GPU load fails); cuda / cpu force one")
    parser.add_argument("--compute-type", default="float16",
                        help="CTranslate2 compute type: float16 (default, ~3 GB VRAM for large-v3), "
                             "int8_float16 (~1.6 GB, less accurate). On the CPU float16 becomes float32")
    parser.add_argument("--vram-margin", type=float, default=2.0,
                        help="GB of VRAM that must stay free besides the model for --device auto to "
                             "use the GPU (default 2)")
    parser.add_argument("--idle-unload", type=float, default=0.0,
                        help="release the model from VRAM after N idle seconds (0 = keep resident)")
    parser.add_argument("--preload", action="store_true", help="load the model at startup")
    parser.add_argument("--models-dir", default=os.environ.get("ASR_MODELS_DIR",
                                                               str(Path(__file__).parent / "models")),
                        help="where models are downloaded to / looked up in (default: ./models, "
                             "or the ASR_MODELS_DIR env var)")
    args = parser.parse_args()

    # stderr: unbuffered even when a GUI reads the output through a pipe
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [asr] %(message)s", force=True)
    log.info("starting: model=%s device=%s compute=%s port=%d models-dir=%s", args.model, args.device,
             args.compute_type, args.port, args.models_dir)
    # fail fast, before spending a minute loading a 3 GB model onto the GPU
    if _port_in_use(args.host, args.port):
        log.error("port %d on %s is already in use - is another asr-server running? "
                  "Check http://%s:%d/health or pick another --port", args.port, args.host, args.host, args.port)
        sys.exit(1)
    holder = ModelHolder(args.model, args.device, args.compute_type, args.idle_unload,
                         Path(args.models_dir), args.vram_margin)
    if args.preload:
        try:
            with holder._lock:
                holder._load()
        except Exception as exc:
            log.error("could not load %s on %s (%s): %s", args.model, holder.device, holder.compute_type, exc)
            if args.device == "cuda":
                log.error("hint: no usable GPU / CUDA libraries? use --device auto or --device cpu")
            sys.exit(1)
    config = uvicorn.Config(build_app(holder), host=args.host, port=args.port, log_level="warning")
    _Server(config, holder).run()


if __name__ == "__main__":
    main()
