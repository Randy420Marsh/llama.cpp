"""OpenAI-compatible speech-to-text server backed by faster-whisper.

Serves POST /v1/audio/transcriptions (multipart: file, model, language, prompt, response_format,
temperature, timestamp_granularities[]) so llama-server and ninfer-serve can transcribe the audio
track of video/audio inputs, replacing a model's native audio when the caller opts in.

Everything stays in memory: uploaded bytes are decoded from RAM (PyAV via faster-whisper), the
Whisper model stays resident in VRAM, nothing is written to disk. With --idle-unload a model is
released after N idle seconds and reloaded on the next request.

Several models can be served at once (--models): a request's `model` field (or the first one, the
default) selects which is used, so one server can offer a fast, small and an accurate Whisper and
the caller picks per call. The default model is large-v3-turbo, the smallest of the accurate,
multilingual, optimized Whisper variants (large-v3-turbo); large-v3 is the most accurate but the
largest; distil-large-v3.5 is English-only and the fastest.
"""

from __future__ import annotations

import argparse
import hashlib
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
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    roots = [Path(p) for p in nvidia.__path__]
    if sys.platform != "win32":
        # LD_LIBRARY_PATH is read only at process start, so load the libraries now: a later
        # dlopen("libcublas.so.12") by CTranslate2 then finds the one already loaded
        import ctypes
        for root in roots:
            for lib in sorted(root.glob("*/lib/lib*.so.*")):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        return
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
                             capture_output=True, encoding="utf-8", errors="replace", timeout=5).stdout.split()
        return max(float(x) for x in out) / 1024 if out else None
    except Exception:
        return None


class ModelHolder:
    """Resident Whisper model(s); transcriptions are serialized on the GPU.

    `models` lists every model this server offers (first = the default). A request's OpenAI
    `model` field selects which one to use, or the default when it is absent, so one server can
    offer a small fast Whisper and an accurate one and the caller picks per call. A per-model
    device / compute / state is tracked; `name`, `device`, `compute_type` and `state` report the
    default model (what the health endpoint and the logs use).
    """

    def __init__(self, models, device: str, compute_type: str, idle_unload: float,
                 models_dir: Path, vram_margin: float = 2.0) -> None:
        self.models = [m for m in (models if isinstance(models, (list, tuple)) else [models])]
        self.default = self.models[0]
        self.device_request = device
        self.device = device  # what a loaded model actually runs on
        self.compute_type = compute_type
        self.idle_unload = idle_unload
        self.models_dir = models_dir
        self.vram_margin = vram_margin
        self._state = {n: "not loaded" for n in self.models}  # not loaded | downloading | loading | loaded | error: ...
        self._model = {}  # name -> WhisperModel
        self._last_use = {}
        self._lock = threading.Lock()  # one transcription on the GPU at a time
        self._load_lock = {n: threading.Lock() for n in self.models}
        if idle_unload > 0:
            threading.Thread(target=self._reaper, daemon=True).start()

    # -- properties for the default model (health endpoint, logs) ----------------
    @property
    def name(self) -> str:
        return self.default

    @property
    def state(self) -> str:
        return self._state[self.default]

    def model_state(self, name: str) -> str:
        return self._state.get(name, "not loaded")

    def _verify_local(self, target: Path) -> bool:
        """Is the local model folder complete and uncorrupted?

        A model is trusted only when every regular file in its folder matches the sha256 recorded in
        the manifest the download step writes (models_dir/<name>.sha256.json). Without a manifest, a
        folder that already has a model.bin is hashed once to create the baseline, so a verified
        local model is then used as-is, with no network contact at all."""
        import json
        if not (target / "model.bin").is_file():
            return False
        manifest = self.models_dir / (target.name + ".sha256.json")
        recorded = None
        if manifest.is_file():
            try:
                recorded = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt manifest means re-verify from scratch
                recorded = None
        files = {}
        for f in sorted(target.rglob("*")):
            if f.is_file() and ".cache" not in f.parts:
                files[f.relative_to(target).as_posix()] = f

        def sha256_of(f: Path) -> str:
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        if recorded is None:
            hashes = {rel: sha256_of(f) for rel, f in files.items()}
            manifest.write_text(json.dumps(hashes, indent=0), encoding="utf-8")
            log.info("recorded sha256 for %d local file(s) of %s in %s (no prior manifest)",
                     len(hashes), target.name, manifest)
            return True
        for rel, f in files.items():
            if rel not in recorded:
                log.warning("local model %s has an unrecorded file %s - re-downloading", target.name, rel)
                return False
            if sha256_of(f) != recorded[rel]:
                log.warning("local model %s is corrupt (sha256 mismatch on %s) - re-downloading", target.name, rel)
                return False
        for rel in recorded:
            if rel not in files:
                log.warning("local model %s is missing %s - re-downloading", target.name, rel)
                return False
        return True

    def _download(self, name: str, target: Path) -> str:
        """Download the model into models_dir/<name> and record the sha256 of what landed."""
        import json
        import huggingface_hub
        repo = _MODELS.get(name, name)
        if "/" not in repo:
            raise ValueError(f"unknown model {name!r}: use one of {', '.join(_MODELS)}, "
                             "a Hugging Face repo id, or a local CTranslate2 model directory")
        self._state[name] = "downloading"
        # faster-whisper's own download_model hides the progress bar; show it, since this is
        # a multi-GB download that otherwise looks like a hang
        log.info("downloading %s (%s) from %s to %s - first run only, progress below ...", name,
                 MODEL_SIZES.get(name, "size unknown"), repo, target)
        huggingface_hub.snapshot_download(repo, local_dir=str(target), allow_patterns=MODEL_FILES)
        hashes = {}
        for f in sorted(target.rglob("*")):
            if f.is_file() and ".cache" not in f.parts:
                h = hashlib.sha256()
                with open(f, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                hashes[f.relative_to(target).as_posix()] = h.hexdigest()
        manifest = self.models_dir / (target.name + ".sha256.json")
        manifest.write_text(json.dumps(hashes, indent=0), encoding="utf-8")
        self._state[name] = "not loaded"
        log.info("download complete: %s (recorded sha256 for %d file(s) in %s)", target, len(hashes), manifest)
        return str(target)

    def _resolve(self, name: str) -> str:
        """Local model directory, downloading it only when no verified local copy exists.

        The local folder is checked first and used as-is when its files match the recorded sha256,
        so a verified model never triggers a network request. Downloads go into a plain folder
        (models_dir/<name>): the default Hugging Face cache uses symlinks, which Windows refuses
        without Developer Mode (WinError 1314)."""
        if Path(name).is_dir():
            return name
        target = self.models_dir / name.replace("/", "--")
        if self._verify_local(target):
            return str(target)
        return self._download(name, target)

    def _pick_device(self, name: str) -> str:
        """--device auto: the GPU only when the model fits with vram_margin GB to spare (so it
        does not squeeze a llama-server sharing the card), else the CPU. Re-evaluated on every
        load, so a model reloaded after --idle-unload adapts to what is free by then."""
        if self.device_request != "auto":
            return self.device_request
        need = MODEL_VRAM_GB.get(name, 4.0) + self.vram_margin
        free = free_vram_gb()
        if free is None or free >= need:
            return "cuda"
        log.warning("only %.1f GB VRAM free, %s needs ~%.1f GB incl. %.1f GB margin - using the CPU",
                    free, name, need, self.vram_margin)
        return "cpu"

    def _create(self, name: str, path: str, device: str) -> WhisperModel:
        compute = self.compute_type
        if device == "cpu" and compute in CPU_COMPUTE:
            compute = CPU_COMPUTE[compute]
            log.info("CPU has no %s: using %s (full precision rather than lossy int8)",
                     self.compute_type, compute)
        log.info("loading %s on %s (%s) ...", name, device, compute)
        model = WhisperModel(path, device=device, compute_type=compute)
        self.device, self.compute_type = device, compute
        return model

    def _load(self, name: str) -> WhisperModel:
        if self._model.get(name) is None:
            with self._load_lock[name]:
                if self._model.get(name) is None:
                    try:
                        path = self._resolve(name)
                        self._state[name] = "loading"
                        t = time.monotonic()
                        device = self._pick_device(name)
                        try:
                            self._model[name] = self._create(name, path, device)
                        except Exception as exc:
                            if self.device_request != "auto" or device != "cuda":
                                raise
                            # CUDA libraries missing / out of memory: still serve, from the CPU
                            log.warning("GPU load failed for %s (%s) - falling back to the CPU", name, exc)
                            self._model[name] = self._create(name, path, "cpu")
                    except Exception as exc:
                        self._state[name] = f"error: {exc}"
                        raise
                    self._state[name] = "loaded"
                    log.info("loaded %s on %s (%s) in %.1fs", name, self.device, self.compute_type,
                             time.monotonic() - t)
        return self._model[name]

    def _reaper(self) -> None:
        while True:
            time.sleep(5)
            with self._lock:
                for name in self.models:
                    last = self._last_use.get(name)
                    if self._model.get(name) is not None and last is not None \
                            and time.monotonic() - last > self.idle_unload:
                        self._model[name] = None
                        self._state[name] = "not loaded"
                        log.info("unloaded %s after %.0fs idle (reloads on the next request)",
                                 name, self.idle_unload)

    def transcribe(self, name: str, audio, **kwargs):
        """audio: 16 kHz mono float32 samples (numpy) or encoded bytes."""
        if isinstance(audio, (bytes, bytearray)):
            audio = io.BytesIO(audio)
        with self._lock:
            model = self._load(name)
            segments, info = model.transcribe(audio, **kwargs)
            segments = list(segments)  # the generator is consumed; finish under the lock
            self._last_use[name] = time.monotonic()
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
        return {"status": "ok", "default_model": holder.default,
                "loaded": holder._model.get(holder.default) is not None,
                "state": holder.state, "device": holder.device, "compute_type": holder.compute_type,
                "models": {n: holder.model_state(n) for n in holder.models}}

    @app.get("/v1/models")
    def models():
        return {"object": "list",
                "data": [{"id": n, "object": "model", "owned_by": "faster-whisper"} for n in holder.models]}

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
        # select which of the served models to use: an explicit `model` that is served, else the default
        if model:
            if model not in holder.models:
                raise HTTPException(400, f"model {model!r} is not served here; use one of "
                                         f"{', '.join(holder.models)} or leave it blank for the default")
            sel = model
        else:
            sel = holder.default
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
            segments, info = holder.transcribe(sel, samples, vad_filter=vad not in ("false", "0", "off"), **common)
            if vad == "auto":
                # VAD treats singing over instruments as non-speech and Whisper then invents a stock
                # phrase ("Thanks for watching!"). If VAD kept little of a track that is clearly not
                # silent, transcribe the whole track instead, without the previous-text conditioning
                # that makes Whisper loop on repeated lines.
                kept = sum(s.end - s.start for s in segments) / max(info.duration, 1e-6)
                active = active_fraction(samples)
                if kept < 0.2 and active > 0.5:
                    mode = f"no-vad (VAD kept {kept:.0%} of a {active:.0%} active track)"
                    segments, info = holder.transcribe(sel, samples, vad_filter=False,
                                                       condition_on_previous_text=False, **common)
        except Exception as exc:  # decode or model failure: report as a client-visible error
            log.exception("transcription failed")
            raise HTTPException(400, f"transcription failed: {exc}") from exc
        dropped = [s.text.strip() for s in segments if is_hallucination(s)]
        if dropped:
            segments = [s for s in segments if not is_hallucination(s)]
            mode += f", dropped {len(dropped)} stock phrase(s): {dropped}"
        text = "".join(s.text for s in segments).strip()
        log.info("%.1fs audio -> %d segments in %.1fs (model=%s, lang=%s, %s)", info.duration, len(segments),
                 time.monotonic() - started, sel, info.language, mode)
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
            scheme = "https" if self.config.ssl_certfile else "http"
            url = f"{scheme}://{self.config.host}:{self.config.port}"
            loaded = ("models: " + ", ".join(self.holder.models) + " loaded"
                      if any(self.holder._model.get(n) is not None for n in self.holder.models)
                      else "models load on the first request (use --preload to load them now)")
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
    parser.add_argument("--model", default="large-v3-turbo",
                        help="faster-whisper model name or CTranslate2 model directory, and the default for "
                             "requests that name no model. large-v3-turbo (default: the optimized, multilingual "
                             "variant - the smallest of the accurate ones); large-v3 (most accurate, largest); "
                             "distil-large-v3.5 (English only, fastest)")
    parser.add_argument("--models", default="",
                        help="comma-separated list of extra ASR models to serve alongside --model; the first "
                             "listed is the default and the rest are selected per request with the OpenAI "
                             "`model` field. So one server can offer a small fast model and a larger accurate "
                             "one, and the caller picks. e.g. --model large-v3-turbo --models large-v3 "
                             "(serves large-v3-turbo by default, large-v3 on request). default: just --model")
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
    # same flags and env vars as llama-server, so one certificate serves both
    parser.add_argument("--ssl-cert-file", default=os.environ.get("LLAMA_ARG_SSL_CERT_FILE"),
                        help="PEM certificate to serve https with (default: LLAMA_ARG_SSL_CERT_FILE)")
    parser.add_argument("--ssl-key-file", default=os.environ.get("LLAMA_ARG_SSL_KEY_FILE"),
                        help="PEM private key to serve https with (default: LLAMA_ARG_SSL_KEY_FILE)")
    parser.add_argument("--no-https", action="store_true",
                        help="serve plain http (https is required otherwise)")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    # stderr: unbuffered even when a GUI reads the output through a pipe
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [asr] %(message)s", force=True)
    ssl_kw: dict = {}
    if not args.no_https:
        if not args.ssl_cert_file or not args.ssl_key_file:
            log.error("https needs --ssl-cert-file and --ssl-key-file (or LLAMA_ARG_SSL_CERT_FILE / "
                      "LLAMA_ARG_SSL_KEY_FILE); pass --no-https to serve plain http")
            sys.exit(1)
        for p in (args.ssl_cert_file, args.ssl_key_file):
            if not Path(p).is_file():
                log.error("TLS file not found: %s", p)
                sys.exit(1)
        ssl_kw = {"ssl_certfile": args.ssl_cert_file, "ssl_keyfile": args.ssl_key_file}
    scheme = "https" if ssl_kw else "http"
    # the models served: the default first, then any extras from --models (comma-split, deduped, order kept)
    extras = [m for m in (args.models.split(",") if args.models else []) if m.strip()]
    served = [args.model.strip()] + [m.strip() for m in extras]
    seen = set()
    served = [m for m in served if m and not (m in seen or seen.add(m))]
    log.info("starting: models=%s (default=%s) device=%s compute=%s port=%d %s models-dir=%s",
             ", ".join(served), served[0], args.device, args.compute_type, args.port, scheme, args.models_dir)
    if ssl_kw:
        log.info("TLS certificate: %s", args.ssl_cert_file)
    # fail fast, before spending a minute loading a 3 GB model onto the GPU
    if _port_in_use(args.host, args.port):
        log.error("port %d on %s is already in use - is another asr-server running? "
                  "Check %s://%s:%d/health or pick another --port", args.port, args.host, scheme, args.host,
                  args.port)
        sys.exit(1)
    holder = ModelHolder(served, args.device, args.compute_type, args.idle_unload,
                         Path(args.models_dir), args.vram_margin)
    if args.preload:
        for name in served:
            try:
                holder._load(name)  # _load takes the per-model lock itself
            except Exception as exc:
                log.error("could not load %s on %s (%s): %s", name, holder.device, holder.compute_type, exc)
                if args.device == "cuda":
                    log.error("hint: no usable GPU / CUDA libraries? use --device auto or --device cpu")
                sys.exit(1)
    config = uvicorn.Config(build_app(holder), host=args.host, port=args.port, log_level="warning", **ssl_kw)
    _Server(config, holder).run()


if __name__ == "__main__":
    main()
