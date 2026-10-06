"""
Pluggable text-to-speech engines.

Every engine turns a short string into mono 16-bit PCM.  Readers send one
sentence at a time, so the number that matters is time-to-first-audio on a
*short* string, not throughput on a paragraph.

    llamacpp    Qwen3-TTS inside llama.cpp (llama-tts-server, a fork tool):
                zero-shot cloning of your own voices, 10 languages, ~5x realtime
    piper       CPU, ~40 ms, robotic but never stutters
    kokoro      82M, ~1 GB VRAM, the quality/latency sweet spot
    chatterbox  350M, ~3 GB, cloning from a reference clip + emotion
    orpheus     3B via llama.cpp, most natural; its *pretrained* model clones
                a voice from a clip and its transcript

Only one GPU engine stays resident at a time: llama-server already owns most
of the card, so loading a second engine evicts the first (see `select`), and
an engine that has not spoken for TTS_IDLE_UNLOAD seconds (default 600) is
unloaded on its own.

Where every model lives is in speech_paths.py.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import proc_guard
import voice_library
from speech_paths import (APP_ROOT, CACHE_ROOT, LEGACY_PIPER_DIR, LLAMA_BIN, NATIVE_DIR,
                          ORPHEUS_DIR, PIPER_DIR, VOICES_LIB, env_dirs)

ROOT = APP_ROOT
VOICES_DIR = LEGACY_PIPER_DIR          # old name, kept for importers
LOG_DIR = APP_ROOT / "logs"

# Offline by default. huggingface_hub otherwise revalidates every cached repo
# over HTTPS on load -- the weights are not re-fetched, but a live connection
# to the CDN is opened, which is not what "runs locally" should mean.
# Downloads you start yourself (Settings -> Voice -> Models, /api/models/download)
# are explicit and run regardless; this flag only stops *implicit* network use.
OFFLINE = os.environ.get("TTS_OFFLINE", "1") != "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"        # never phone home, either way
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
if OFFLINE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))

IDLE_UNLOAD_S = float(os.environ.get("TTS_IDLE_UNLOAD", "600"))
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_lock = threading.Lock()


# --------------------------------------------------------------- wav helpers
def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_to_pcm(data: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


def silence(sample_rate: int = 22050, ms: int = 40) -> bytes:
    return wav_bytes(b"\x00\x00" * int(sample_rate * ms / 1000), sample_rate)


# A sentence of "---" or ". . ." has nothing to pronounce.  Piper emits zero
# audio chunks for those and then dies writing the wav header, so every engine
# gets this guard rather than each one growing its own.
# str.isalnum() is Unicode-aware, which matters: an allowlist of Latin/Greek/
# Cyrillic ranges silently classified Japanese, Chinese and Devanagari as
# unpronounceable, so Kokoro ja/zh/hi voices returned 40 ms of silence.
def is_speakable(text: str) -> bool:
    return any(ch.isalnum() for ch in (text or ""))


def float_to_pcm16(audio) -> bytes:
    """torch tensor / numpy float array in [-1, 1] -> int16 little-endian."""
    import numpy as np
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype="float32").squeeze()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)          # downmix to mono
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return (audio * 32767.0).clip(-32768, 32767).astype("<i2").tobytes()


def guess_language(text: str, fallback: str = "") -> str:
    """Script-based guess for the cases a language token gets badly wrong.

    Latin-script languages cannot be told apart this cheaply; those fall back
    to the voice's own language (voice.json) or English.
    """
    for ch in text or "":
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF:
            return "ja"
        if 0xAC00 <= o <= 0xD7AF:
            return "ko"
    if any(0x4E00 <= ord(c) <= 0x9FFF for c in text or ""):
        return "zh"
    if any(0x0400 <= ord(c) <= 0x04FF for c in text or ""):
        return "ru"
    return fallback


def _http_json(url: str, body: dict | None = None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else None
    rq = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        return json.loads(r.read())


def _wait_health(url: str, proc, seconds: int, log: Path | None) -> None:
    for _ in range(seconds * 2):
        if proc is not None and proc.poll() is not None:
            tail = ""
            if log and log.exists():
                tail = log.read_text(encoding="utf-8", errors="replace")[-1200:]
            raise RuntimeError(f"server exited with code {proc.returncode} before it was ready\n{tail}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"no answer from {url} after {seconds} s")


def _same_file(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a).replace("\\", "/").lower() == str(b).replace("\\", "/").lower()


# ------------------------------------------------------------------ base
class Engine:
    name = "base"
    label = "Base"
    gpu = False
    vram_mb = 0
    note = ""
    speed_native = False        # the engine itself changes the speaking rate
    cloning = False             # can speak in a voice from the voice library
    cloning_experimental = False  # ...but often misses the voice
    languages: list[str] = []   # [] = whatever the voice speaks

    def __init__(self) -> None:
        self._model = None
        self.last_used = 0.0
        self._synth_lock = threading.Lock()

    # -- to implement
    def voices(self) -> list[str]:
        return []

    def default_voice(self) -> str:
        v = self.voices()
        return v[0] if v else ""

    def models(self) -> list[dict]:
        """Quality options on disk: [{"id", "label", ...}]. [] = nothing to choose."""
        return []

    def default_model(self) -> str:
        m = self.models()
        return m[0]["id"] if m else ""

    def _load(self):
        raise NotImplementedError

    def _synth(self, text: str, voice: str, speed: float, model: str = "",
               language: str = "") -> tuple[bytes, int]:
        raise NotImplementedError

    # -- shared
    @property
    def loaded(self) -> bool:
        return self._model is not None

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def load(self):
        if self._model is None:
            print(f"[TTS] loading {self.label} ...", flush=True)
            self._model = self._load()
            print(f"[TTS] {self.label} ready", flush=True)
        self.touch()
        return self._model

    def unload(self):
        if self._model is None:
            return
        print(f"[TTS] unloading {self.label}", flush=True)
        self._model = None
        if self.gpu:
            try:
                import gc, torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

    def synth_wav(self, text: str, voice: str = "", speed: float = 1.0,
                  model: str = "", language: str = "") -> bytes:
        text = (text or "").strip()
        if not is_speakable(text):
            return silence()
        self.load()
        with self._synth_lock:
            pcm, sr = self._synth(text, voice or self.default_voice(), speed, model, language)
        self.touch()
        if not pcm:
            return silence()
        return wav_bytes(pcm, sr)

    def available(self) -> tuple[bool, str]:
        """Is the engine usable right now? (False, why-not) if not."""
        return True, ""

    def voice_status(self) -> dict[str, bool]:
        """{voice: is its data actually on disk}.

        Engines that download per-voice data lie by omission otherwise: the
        dropdown lists a name, synthesis fails, and the server quietly falls
        back to Piper, so every 'missing' voice sounds identical. Reporting
        presence up front is what stops that being invisible.
        """
        return {v: True for v in self.voices()}

    def missing_voices(self) -> list[str]:
        return sorted(v for v, ok in self.voice_status().items() if not ok)


def library_voice_ids() -> list[str]:
    try:
        return [v["id"] for v in voice_library.list_voices()]
    except Exception:
        return []


# ------------------------------------------------------------------ piper
PIPER_URL = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
             "{family}/{code}/{name}/{quality}/{code}-{name}-{quality}{ext}")


def piper_parts(voice: str) -> dict | None:
    """'en_GB-northern_english_male-medium' -> code, name, quality, family."""
    m = re.fullmatch(r"([a-z]{2,3}_[A-Z]{2})-(.+)-(x_low|low|medium|high)", voice or "")
    if not m:
        return None
    return {"code": m.group(1), "name": m.group(2), "quality": m.group(3),
            "family": m.group(1).split("_")[0]}


class PiperEngine(Engine):
    name, label = "piper", "Piper (CPU)"
    note = "instant, robotic; never stutters"
    speed_native = True

    KNOWN = ["en_US-lessac-medium", "en_US-ryan-high", "en_US-amy-medium",
             "en_GB-alba-medium", "fi_FI-harri-medium"]

    def __init__(self) -> None:
        super().__init__()
        self._cache: dict[str, object] = {}

    @staticmethod
    def dirs() -> list[Path]:
        return [PIPER_DIR, LEGACY_PIPER_DIR]

    @classmethod
    def installed(cls) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for d in cls.dirs():
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.onnx")):
                if p.with_suffix(".onnx.json").exists():
                    out.setdefault(p.stem, p)
        return out

    def voices(self) -> list[str]:
        return sorted(set(self.KNOWN) | set(self.installed()))

    def default_voice(self) -> str:
        have = self.installed()
        return "en_US-lessac-medium" if ("en_US-lessac-medium" in have or not have) else sorted(have)[0]

    def voice_status(self):
        have = self.installed()
        return {v: (v in have) for v in self.voices()}

    def available(self):
        try:
            import piper  # noqa: F401
            return True, ""
        except Exception as e:
            return False, f"piper not installed: {e}"

    def _download(self, name: str) -> Path:
        p = self.installed().get(name)
        if p:
            return p
        if OFFLINE:
            raise RuntimeError(
                f"Piper voice '{name}' is not in {PIPER_DIR}. Download it in "
                "Settings -> Voice -> Models, or re-run once with TTS_OFFLINE=0.")
        job = start_download("piper", name, wait=True)
        if job.get("error"):
            raise RuntimeError(job["error"])
        p = self.installed().get(name)
        if not p:
            raise RuntimeError(f"voice '{name}' did not download to {PIPER_DIR}")
        return p

    def _voice(self, name: str):
        if name not in self._cache:
            from piper import PiperVoice
            p = self._download(name)
            print(f"[TTS] loading voice {p.name} ...", flush=True)
            self._cache[name] = PiperVoice.load(str(p))
        return self._cache[name]

    def _load(self):
        return self._voice(self.default_voice())

    def synth_wav(self, text: str, voice: str = "", speed: float = 1.0,
                  model: str = "", language: str = "") -> bytes:
        # Piper is per-voice rather than one resident model, so it overrides
        # the shared path to avoid loading a model it will not use.
        text = (text or "").strip()
        if not is_speakable(text):
            return silence()
        name = voice or self.default_voice()
        try:
            v = self._voice(name)
        except Exception:
            if name == self.default_voice():
                raise
            v = self._voice(self.default_voice())
        kw = {}
        if speed and abs(speed - 1.0) > 0.01:
            try:
                from piper.config import SynthesisConfig
                kw["syn_config"] = SynthesisConfig(length_scale=1.0 / max(0.25, min(4.0, speed)))
            except Exception:
                pass
        sr = int(getattr(v.config, "sample_rate", 22050))
        pcm = bytearray()
        with self._synth_lock:
            for chunk in v.synthesize(text, **kw):
                pcm += getattr(chunk, "audio_int16_bytes", b"")
        self.touch()
        return wav_bytes(bytes(pcm), sr) if pcm else silence(sr)


# ----------------------------------------------------------------- kokoro
class KokoroEngine(Engine):
    name, label = "kokoro", "Kokoro-82M"
    gpu, vram_mb = True, 950          # measured 916 MB resident on a 5090
    note = "best quality-per-millisecond; preset voices"
    speed_native = True

    # Kokoro ships one small .pt per voice and fetches them individually, so
    # what the dropdown offers has to come from the cache, not a hardcoded
    # list -- otherwise it advertises voices that cannot be synthesised.
    REPO = "hexgrad/Kokoro-82M"
    FALLBACK_VOICES = ["af_heart", "af_bella", "am_michael", "bf_emma", "bm_george"]

    @classmethod
    def voice_dir(cls) -> Path | None:
        root = CACHE_ROOT / "hf" / "hub" / "models--hexgrad--Kokoro-82M" / "snapshots"
        if not root.is_dir():
            return None
        for snap in sorted(root.iterdir(), reverse=True):
            d = snap / "voices"
            if d.is_dir():
                return d
        return None

    @classmethod
    def cached_voices(cls) -> list[str]:
        d = cls.voice_dir()
        return sorted(p.stem for p in d.glob("*.pt")) if d else []

    def voices(self):
        return self.cached_voices() or list(self.FALLBACK_VOICES)

    def voice_status(self):
        have = set(self.cached_voices())
        names = have or set(self.FALLBACK_VOICES)
        return {v: (v in have) for v in sorted(names)}

    def default_voice(self):
        have = self.cached_voices()
        return "af_heart" if ("af_heart" in have or not have) else have[0]

    def available(self):
        try:
            import kokoro  # noqa: F401
            return True, ""
        except Exception as e:
            return False, f"kokoro not installed: {e} (run run-tts-setup.bat)"

    # Kokoro's voice prefix IS its language: af_/am_ American, bf_/bm_ British,
    # then Spanish, French, Hindi, Italian, Japanese, Portuguese, Chinese.
    # Running a Japanese voice through the English G2P produces confident
    # nonsense, so each language gets its own pipeline.
    LANGS = {"a": "American English", "b": "British English", "e": "Spanish",
             "f": "French", "h": "Hindi", "i": "Italian", "j": "Japanese",
             "p": "Portuguese", "z": "Mandarin"}

    @staticmethod
    def lang_of(voice: str) -> str:
        c = (voice or "a")[:1]
        return c if c in KokoroEngine.LANGS else "a"

    def _load(self):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # Pipelines are built lazily per language: the ja/zh ones pull extra
        # g2p dependencies that may not be installed, and building them all up
        # front would make an English-only setup fail for no reason.
        return {"_pipes": {}, "_dev": dev}

    def _pipeline(self, lang: str):
        from kokoro import KPipeline
        m = self.load()
        pipes = m["_pipes"]
        if lang not in pipes:
            try:
                pipes[lang] = KPipeline(lang_code=lang, device=m["_dev"])
            except Exception as e:
                if lang == "a":
                    raise
                raise RuntimeError(
                    f"Kokoro {self.LANGS.get(lang, lang)} voices need extra g2p "
                    f"support that is not installed ({e}). English voices "
                    f"(af_/am_/bf_/bm_) work regardless.") from e
        return pipes[lang]

    def _synth(self, text, voice, speed, model="", language=""):
        import numpy as np
        pipe = self._pipeline(self.lang_of(voice))
        chunks = [audio for _, _, audio in pipe(text, voice=voice, speed=speed)
                  if audio is not None]
        if not chunks:
            return b"", 24000
        audio = np.concatenate([np.asarray(c.detach().cpu() if hasattr(c, "detach") else c)
                                for c in chunks])
        return float_to_pcm16(audio), 24000


# ------------------------------------------------------------- chatterbox
class ChatterboxEngine(Engine):
    name, label = "chatterbox", "Chatterbox"
    gpu, vram_mb = True, 3000
    note = "clones your voices from the reference clip; emotion control"
    cloning = True

    def voices(self):
        # Built-in default, your voice library, and any clip still dropped in
        # the old <server>\voices\clone\ folder.
        legacy = LEGACY_PIPER_DIR / "clone"
        old = [f"clone:{p.stem}" for p in sorted(legacy.glob("*.wav"))] if legacy.exists() else []
        return ["default"] + library_voice_ids() + old

    def default_voice(self):
        return "default"

    def available(self):
        try:
            import chatterbox  # noqa: F401
            return True, ""
        except Exception as e:
            return False, f"chatterbox-tts not installed: {e} (run run-tts-setup.bat)"

    def _load(self):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # Turbo ships as a separate checkpoint in newer releases; fall back to
        # the standard one so this keeps working on either version.
        try:
            from chatterbox.tts import ChatterboxTTS
        except ImportError:
            from chatterbox import ChatterboxTTS       # older layout
        for loader in ("from_pretrained_turbo", "from_pretrained"):
            fn = getattr(ChatterboxTTS, loader, None)
            if fn is None:
                continue
            try:
                return fn(device=dev)
            except TypeError:
                return fn(dev)
        raise RuntimeError("ChatterboxTTS exposes no from_pretrained loader")

    @staticmethod
    def reference(voice: str) -> Path | None:
        if voice.startswith("clone:"):
            p = LEGACY_PIPER_DIR / "clone" / f"{voice.split(':', 1)[1]}.wav"
            return p if p.exists() else None
        if voice and voice != "default":
            return voice_library.reference_path(voice)
        return None

    def _synth(self, text, voice, speed, model="", language=""):
        model_ = self.load()
        kw = {}
        ref = self.reference(voice)
        if voice not in ("", "default") and ref is None:
            raise RuntimeError(f"unknown voice '{voice}'")
        if ref is not None:
            kw["audio_prompt_path"] = str(ref)
        wav = model_.generate(text, **kw)
        return float_to_pcm16(wav), int(getattr(model_, "sr", 24000))


# ------------------------------------------------------------ llama.cpp native
def split_quant(stem: str) -> tuple[str, str]:
    """'Qwen3-TTS-12Hz-1.7B-Base-Q8_0' -> ('Qwen3-TTS-12Hz-1.7B-Base', 'Q8_0')."""
    m = re.fullmatch(r"(.+?)[-.]((?:UD-)?(?:[QqIiTt]Q?\d[\w]*)|[Bb][Ff]16|[Ff]16|[Ff]32)", stem)
    return (m.group(1), m.group(2)) if m else (stem, "")


def _mmproj_rank(p: Path) -> int:
    q = split_quant(p.stem)[1].lower()
    return 0 if q in ("bf16", "f16", "f32") else 1 if q.startswith("q8") else 2


class LlamaCppEngine(Engine):
    """Speech models that llama.cpp runs natively (libmtmd audio generation).

    A llama-tts-server child (built from the llama.cpp fork, tools/tts) holds
    the model; your voice library is its --tts-voices-dir, so a cloned voice
    is just a folder name. Started on first use, swapped when another model is
    picked, stopped by unload / idle timeout / shutdown.
    """
    name, label = "llamacpp", "llama.cpp (Qwen3-TTS)"
    gpu, vram_mb = True, 5000
    note = "clones your voices (zero-shot, no training); 10 languages; runs in llama.cpp"
    cloning = True
    languages = ["en", "zh", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"]

    EXE = Path(os.environ.get("LLAMA_TTS_SERVER_EXE")
               or (LLAMA_BIN / ("llama-tts-server.exe" if os.name == "nt" else "llama-tts-server")))
    PORT = int(os.environ.get("LLAMA_TTS_PORT", "8181"))
    MODEL_DIRS = env_dirs("LLAMA_TTS_MODEL_DIRS", [NATIVE_DIR])
    EXTRA_ARGS = os.environ.get("LLAMA_TTS_ARGS", "").split()

    _proc = None
    _proc_model = None
    _srv_lock = threading.Lock()

    @classmethod
    def model_files(cls) -> dict[str, dict]:
        """Backbone ggufs that have a matching mmproj next to them."""
        found: dict[str, dict] = {}
        for d in cls.MODEL_DIRS:
            if not d.is_dir():
                continue
            ggufs = sorted(list(d.glob("*.gguf")) + list(d.glob("*/*.gguf")))
            mmprojs = [p for p in ggufs if p.name.lower().startswith("mmproj")]
            for b in ggufs:
                if b in mmprojs or b.name.endswith(".part"):
                    continue
                base, quant = split_quant(b.stem)
                cands = [m for m in mmprojs if split_quant(m.stem[len("mmproj-"):])[0] == base]
                if not cands:
                    continue
                mm = sorted(cands, key=_mmproj_rank)[0]
                found.setdefault(b.stem, {
                    "id": b.stem, "label": f"{base} {quant}".strip(), "quant": quant,
                    "model": b, "mmproj": mm,
                    "size_mb": round((b.stat().st_size + mm.stat().st_size) / 1e6)})
        return found

    def models(self):
        return [{"id": k, "label": v["label"], "quant": v["quant"], "size_mb": v["size_mb"],
                 "path": str(v["model"]), "mmproj": str(v["mmproj"])}
                for k, v in sorted(self.model_files().items())]

    def default_model(self):
        files = self.model_files()
        for q in ("Q8_0", "bf16", "BF16", "F16", "Q6_K", "Q4_K_M"):
            for k, v in files.items():
                if v["quant"] == q:
                    return k
        return next(iter(files), "")

    def voices(self):
        return ["default"] + library_voice_ids()

    def default_voice(self):
        return "default"

    def available(self):
        if not self.EXE.exists():
            return False, (f"{self.EXE} not found: build the llama.cpp fork's llama-tts-server "
                           "target, or set LLAMA_TTS_SERVER_EXE")
        return True, ""

    @property
    def loaded(self) -> bool:
        p = type(self)._proc
        return p is not None and p.poll() is None

    @staticmethod
    def _serving(url: str, entry: dict) -> bool:
        try:
            doc = _http_json(url + "/v1/models", timeout=2)
            meta = (doc.get("data") or [{}])[0].get("meta") or {}
            return _same_file(meta.get("model", ""), entry["model"])
        except Exception:
            return False

    def _ensure_server(self, model: str) -> str:
        cls = type(self)
        with cls._srv_lock:
            files = self.model_files()
            if not files:
                raise RuntimeError(
                    "no llama.cpp speech model found in "
                    + ", ".join(str(d) for d in self.MODEL_DIRS)
                    + ": download one in Settings -> Voice -> Models, or put <model>.gguf "
                      "and mmproj-<model>.gguf there")
            entry = files.get(model) or files[self.default_model()]
            url = f"http://127.0.0.1:{cls.PORT}"
            if self.loaded and cls._proc_model == entry["id"]:
                return url
            if cls._proc is None and self._serving(url, entry):
                print(f"[TTS] reusing llama-tts-server already on {url}", flush=True)
                cls._proc_model = entry["id"]
                return url
            self._stop()
            ok, why = self.available()
            if not ok:
                raise RuntimeError(why)
            VOICES_LIB.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            log = LOG_DIR / "llama-tts-server.log"
            args = [str(self.EXE), "-m", str(entry["model"]), "-mm", str(entry["mmproj"]),
                    "-ngl", "99", "-c", "4096", "--tts-voices-dir", str(VOICES_LIB),
                    "--port", str(cls.PORT)] + self.EXTRA_ARGS
            print(f"[TTS] starting llama-tts-server: {entry['label']} on {url}", flush=True)
            with open(log, "wb") as fh:
                cls._proc = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT,
                                             creationflags=_NO_WINDOW, preexec_fn=proc_guard.PREEXEC)
            proc_guard.adopt(cls._proc)
            cls._proc_model = entry["id"]
            try:
                _wait_health(url, cls._proc, 180, log)
            except Exception:
                self._stop()
                raise
            return url

    @classmethod
    def _stop(cls):
        p = cls._proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=15)
            except Exception:
                p.kill()
        cls._proc = cls._proc_model = None

    def _load(self):
        return {"url": self._ensure_server(self.default_model())}

    def unload(self):
        if self.loaded:
            print(f"[TTS] unloading {self.label} (stopping llama-tts-server)", flush=True)
        self._stop()
        self._model = None

    def _synth(self, text, voice, speed, model="", language=""):
        url = self._ensure_server(model or (type(self)._proc_model or self.default_model()))
        v = "" if voice in ("", "default") else voice
        lv = None
        if v:
            lv = voice_library.get(v)
            if lv is None:
                raise RuntimeError(f"unknown voice '{voice}'")
            v = lv["id"]
        lang = (language or "").strip().lower()
        if lang in ("", "auto"):
            lang = guess_language(text, (lv or {}).get("language", "")) or "auto"
        body = json.dumps({"input": text, "voice": v, "language": lang}).encode()
        rq = urllib.request.Request(url + "/v1/audio/speech", data=body,
                                    headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(rq, timeout=600) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error", {}).get("message", "")
            except Exception:
                msg = ""
            raise RuntimeError(f"llama-tts-server HTTP {e.code}: {msg}") from e
        return wav_to_pcm(data)


# ---------------------------------------------------------------- orpheus
class OrpheusEngine(Engine):
    name, label = "orpheus", "Orpheus 3B"
    gpu, vram_mb = True, 4200          # Q8_0 in llama-server + SNAC here
    # Measured on a 5090, warm, one sentence:
    #   transformers bf16  RTF 1.53   (unusable -- slower than playback)
    #   llama.cpp F16      RTF 0.51
    #   llama.cpp Q8_0     RTF 0.33   <- the sweet spot
    #   llama.cpp Q4_K_M   RTF 0.26
    #   llama.cpp Q1_0     silent -- emits no valid audio codes at all
    # The project's own ~200 ms figure assumes vLLM, which has no usable
    # Windows support; llama.cpp is what makes this practical here.
    note = ("most natural; 8 preset voices (finetuned model). Your voices go through the "
            "pretrained model and often drift to a generic voice: use llama.cpp to clone")
    # Measured with one 7-8 s reference each: speaker similarity 0.55-0.83 to the
    # reference, one of two voices missed entirely (llama.cpp Qwen3-TTS: 0.93-0.95).
    # The model was never trained to clone; Canopy Labs say more examples help.
    cloning = True
    cloning_experimental = True

    # The official weights are gated: they 401 unless HF_TOKEN belongs to an
    # account that clicked through at
    #   https://huggingface.co/canopylabs/orpheus-3b-0.1-ft
    # Everything is Apache-2.0, so ungated community mirrors are perfectly
    # legal to use -- but silently falling back to one would route around a
    # consent step the authors deliberately put up, so that is opt-in:
    #   set ORPHEUS_REPO=audo/orpheus-3b-0.1-ft
    OFFICIAL = "canopylabs/orpheus-3b-0.1-ft"
    REPOS = [os.environ.get("ORPHEUS_REPO") or OFFICIAL]
    # Point at an already-running llama-server to skip the quant management:
    #   set ORPHEUS_SERVER=http://127.0.0.1:8180
    # Otherwise every orpheus*.gguf in GGUF_DIRS becomes a selectable quant and
    # we start/stop llama-server for it on demand, so quants can be compared by ear.
    SERVER = (os.environ.get("ORPHEUS_SERVER") or "").rstrip("/")
    # plus a model collection folder that already holds Orpheus quants, if present
    GGUF_DIRS = env_dirs("ORPHEUS_GGUF_DIR", [ORPHEUS_DIR] + [p for p in (Path("E:/LLAMA_GGUF"),)
                                                              if os.name == "nt" and p.is_dir()])
    GGUF_DIR = GGUF_DIRS[0]            # where downloads go
    LLAMA_SERVER_EXE = os.environ.get(
        "LLAMA_SERVER_EXE", str(LLAMA_BIN / ("llama-server.exe" if os.name == "nt" else "llama-server")))
    # 8081 is a second chat model, 8084 the command center's embedding server;
    # never share a port with another llama-server: Windows lets both bind it
    # and requests go to either.
    GGUF_PORT = int(os.environ.get("ORPHEUS_GGUF_PORT", "8180"))
    TOKENIZER_REPO = REPOS[0]
    VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

    _proc = None            # llama-server we started
    _proc_quant = None      # which gguf it holds
    _codes_cache: dict = {}

    @classmethod
    def quants(cls) -> dict[str, Path]:
        """{'Q8_0': path, 'pretrained-Q8_0': path, ...} for every Orpheus gguf we can serve."""
        out: dict[str, Path] = {}
        for d in cls.GGUF_DIRS:
            if not d.is_dir():
                continue
            for p in sorted(d.glob("orpheus*.gguf")):
                m = re.search(r"-(F16|BF16|Q\d[^.]*|IQ\d[^.]*|TQ\d[^.]*|UD-[^.]*)\.gguf$", p.name, re.I)
                q = m.group(1) if m else p.stem
                if "pretrained" in p.name.lower():
                    q = f"pretrained-{q}"
                out.setdefault(q, p)
        return out

    @classmethod
    def ft_quants(cls) -> dict[str, Path]:
        return {k: v for k, v in cls.quants().items() if not k.startswith("pretrained-")}

    @classmethod
    def pt_quants(cls) -> dict[str, Path]:
        return {k: v for k, v in cls.quants().items() if k.startswith("pretrained-")}

    def models(self):
        out = []
        for q, p in self.quants().items():
            pt = q.startswith("pretrained-")
            out.append({"id": q, "label": (f"{q[11:]} pretrained (your voices)" if pt
                                           else f"{q} finetuned (preset voices)"),
                        "kind": "pretrained" if pt else "finetuned", "path": str(p),
                        "size_mb": round(p.stat().st_size / 1e6)})
        return out

    def default_model(self):
        qs = self.ft_quants()
        return next((q for q in ("Q8_0", "Q6_K", "Q4_K_M") if q in qs), next(iter(qs), ""))

    def _ensure_server(self, quant: str) -> str:
        """Start (or swap) a llama-server holding the requested quant."""
        if self.SERVER:                       # caller manages it themselves
            return self.SERVER
        quants = self.quants()
        path = quants.get(quant) or (next(iter(self.ft_quants().values()), None)
                                     or next(iter(quants.values()), None))
        if path is None:
            raise RuntimeError("no orpheus*.gguf in " + ", ".join(str(d) for d in self.GGUF_DIRS))
        quant = next((k for k, v in quants.items() if v == path), quant)
        url = f"http://127.0.0.1:{self.GGUF_PORT}"

        cls = type(self)
        if cls._proc_quant == quant and cls._proc and cls._proc.poll() is None:
            return url
        # Someone may already be serving an Orpheus gguf on that port (a
        # hand-started llama-server, or a previous run of this process). Adopt
        # it instead of trying to bind the port a second time and failing.
        if cls._proc is None and self._is_orpheus_server(url, path):
            print(f"[TTS] reusing llama-server already on {url}", flush=True)
            cls._proc_quant = quant
            return url
        if cls._proc and cls._proc.poll() is None:
            print(f"[TTS] swapping orpheus gguf -> {quant}", flush=True)
            cls.stop_server()
        if not Path(self.LLAMA_SERVER_EXE).exists():
            raise RuntimeError(f"llama-server not found: {self.LLAMA_SERVER_EXE}")

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = LOG_DIR / "orpheus-llama-server.log"
        print(f"[TTS] starting llama-server for orpheus {quant} on {url}", flush=True)
        with open(log, "wb") as fh:
            # 8192: a cloned voice puts its reference clip (~83 tokens a second) in the prompt
            cls._proc = subprocess.Popen(
                [self.LLAMA_SERVER_EXE, "-m", str(path), "--port", str(self.GGUF_PORT),
                 "-ngl", "99", "-c", "8192", "--no-webui"],
                stdout=fh, stderr=subprocess.STDOUT,
                creationflags=_NO_WINDOW, preexec_fn=proc_guard.PREEXEC)
        proc_guard.adopt(cls._proc)
        cls._proc_quant = quant
        try:
            _wait_health(url, cls._proc, 120, log)
        except Exception:
            cls.stop_server()
            raise
        return url

    @staticmethod
    def _is_orpheus_server(url: str, path: Path | None = None) -> bool:
        """True if something healthy on `url` is serving that Orpheus model."""
        try:
            blob = _http_json(url + "/v1/models", timeout=2)
        except Exception:
            return False
        names = [str(m.get("model") or m.get("name") or m.get("id") or "")
                 for m in (blob.get("models") or blob.get("data") or [])]
        if path is not None:
            return any(Path(n).name.lower() == path.name.lower() or n.lower() == path.stem.lower()
                       for n in names)
        return any("orpheus" in n.lower() for n in names)

    @classmethod
    def stop_server(cls):
        if cls._proc and cls._proc.poll() is None:
            cls._proc.terminate()
            try:
                cls._proc.wait(timeout=15)
            except Exception:
                cls._proc.kill()
        cls._proc = cls._proc_quant = None

    # Token ids from the Orpheus reference decoder / training data layout.
    START_HUMAN, EOT, END_HUMAN = 128259, 128009, 128260
    START_AI, END_AI = 128261, 128262
    START_AUDIO, EOS = 128257, 128258
    CODE_OFFSET = 128266

    def voices(self):
        # Preset voices belong to the finetuned model; your own voices need the
        # pretrained one (it learned to continue whatever voice it is shown).
        out = list(self.VOICES)
        if self.pt_quants() or self.SERVER:
            out += library_voice_ids()
        return out

    def default_voice(self):
        return "tara"

    @staticmethod
    def split_voice(voice: str) -> tuple[str, str]:
        """Old 'tara [Q8_0]' style values still work: voice and quant in one."""
        m = re.match(r"\s*([\w.-]+)\s*\[([^\]]+)\]\s*$", voice or "")
        return (m.group(1), m.group(2)) if m else ((voice or "tara").strip(), "")

    def available(self):
        missing = []
        for mod in ("torch", "snac"):
            try:
                __import__(mod)
            except Exception:
                missing.append(mod)
        if not missing and not (self.quants() or self.SERVER):
            try:
                __import__("transformers")
            except Exception:
                missing.append("transformers (or an orpheus gguf)")
        return (not missing), ("missing: " + ", ".join(missing) + " (run run-tts-setup.bat)"
                               if missing else "")

    def _load(self):
        import torch
        from snac import SNAC
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(dev)

        # Preferred path: a llama-server holding an Orpheus GGUF. Only the SNAC
        # codec stays in this process (~80 MB); the server tokenizes the text
        # and llama.cpp's decode loop is what makes this fast enough to use.
        if self.SERVER or self.quants():
            where = self.SERVER or f"{len(self.quants())} gguf quant(s)"
            print(f"[TTS] orpheus via llama.cpp: {where}", flush=True)
            return {"snac": snac, "dev": dev, "gguf": True}

        from transformers import AutoModelForCausalLM, AutoTokenizer
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
        errors = []
        for repo in self.REPOS:
            try:
                tok = AutoTokenizer.from_pretrained(repo)
                try:                            # transformers >= 5 renamed it
                    lm = AutoModelForCausalLM.from_pretrained(repo, dtype=dt)
                except TypeError:
                    lm = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dt)
                print(f"[TTS] orpheus weights from {repo}", flush=True)
                break
            except Exception as e:
                errors.append(f"{repo}: {str(e).splitlines()[0]}")
        else:
            raise RuntimeError(
                "could not load Orpheus weights.\n  " + "\n  ".join(errors) +
                f"\n{self.OFFICIAL} is gated. Accept the licence at "
                f"https://huggingface.co/{self.OFFICIAL} and set HF_TOKEN "
                "(the model is Apache-2.0; the gate is just a click-through). "
                "Or download a GGUF quant in Settings -> Voice -> Models.")
        lm.to(dev).eval()
        return {"tok": tok, "lm": lm, "snac": snac, "dev": dev}

    @staticmethod
    def _tokenize(url: str, text: str) -> list[int]:
        """Text -> token ids by the model's own tokenizer, BOS included."""
        doc = _http_json(url + "/tokenize", {"content": text, "add_special": True}, timeout=30)
        return [t if isinstance(t, int) else t.get("id") for t in doc.get("tokens", [])]

    def _ref_codes(self, m, path: Path) -> list[int]:
        """SNAC-encode a reference clip into Orpheus audio tokens (cached per file)."""
        import numpy as np
        import torch
        key = (str(path), path.stat().st_mtime, path.stat().st_size)
        hit = type(self)._codes_cache.get(key)
        if hit is not None:
            return hit
        with wave.open(str(path), "rb") as wf:
            sr, ch = wf.getframerate(), wf.getnchannels()
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype("float32") / 32768.0
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)
        if sr != 24000:
            from math import gcd
            try:
                from scipy.signal import resample_poly
                g = gcd(sr, 24000)
                pcm = resample_poly(pcm, 24000 // g, sr // g).astype("float32")
            except Exception:
                x = np.linspace(0, len(pcm) - 1, int(len(pcm) * 24000 / sr))
                pcm = np.interp(x, np.arange(len(pcm)), pcm).astype("float32")
        t = torch.from_numpy(pcm).to(m["dev"]).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            l1, l2, l3 = [c[0].tolist() for c in m["snac"].encode(t)]
        o = self.CODE_OFFSET
        frames: list[list[int]] = []
        for i in range(len(l1)):
            if 4 * i + 3 >= len(l3) or 2 * i + 1 >= len(l2):
                break
            f = [l1[i] + o, l2[2 * i] + o + 4096, l3[4 * i] + o + 2 * 4096,
                 l3[4 * i + 1] + o + 3 * 4096, l2[2 * i + 1] + o + 4 * 4096,
                 l3[4 * i + 2] + o + 5 * 4096, l3[4 * i + 3] + o + 6 * 4096]
            # the training data dropped a frame whose first code repeats the
            # previous frame's (remove_duplicate_frames in the data prep):
            # show the model the kind of sequence it learned from
            if frames and f[0] == frames[-1][0]:
                continue
            frames.append(f)
        codes = [c for f in frames for c in f]
        type(self)._codes_cache[key] = codes
        return codes

    def _clone_prompt(self, m, url: str, text: str, voice: str) -> list[int]:
        """Zero-shot prompt for the pretrained model: one finished example turn in
        the target voice (its transcript, then its audio tokens), then the new text
        with the speech turn already opened -- the model continues in that voice."""
        v = voice_library.get(voice)
        if v is None:
            raise RuntimeError(f"unknown voice '{voice}'")
        if not v["transcript"]:
            raise RuntimeError(f"voice '{voice}' has no transcript; Orpheus cloning needs one "
                               "(add it in Settings -> Voice -> My voices)")
        codes = self._ref_codes(m, Path(v["file"]))
        ref = self._tokenize(url, v["transcript"])
        tgt = self._tokenize(url, text)
        return ([self.START_HUMAN] + ref + [self.EOT, self.END_HUMAN, self.START_AI, self.START_AUDIO]
                + codes + [self.EOS, self.END_AI]
                + [self.START_HUMAN] + tgt + [self.EOT, self.END_HUMAN, self.START_AI, self.START_AUDIO])

    def _codes_via_llamacpp(self, url: str, ids: list[int], text: str,
                            temperature: float = 0.6) -> list[int]:
        """Generate audio codes on a llama-server holding an Orpheus GGUF.

        The prompt is sent as raw token ids rather than a string: the special
        frame tokens have no stable text form, and passing text would let the
        server add its own BOS on top of the tokenizer's -- the double-BOS
        that makes Orpheus output garble.
        """
        body = json.dumps({
            "prompt": ids, "n_predict": min(2600, 200 + 75 * len(text.split())),
            "temperature": temperature, "top_p": 0.95,
            "repeat_penalty": 1.1,          # >=1.1 required for stable output
            "cache_prompt": True, "stream": False,
        }).encode()
        rq = urllib.request.Request(url + "/completion", data=body,
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(rq, timeout=300) as r:
            out = json.loads(r.read()).get("content", "")
        # Audio tokens come back detokenised as <custom_token_N>, the vocab
        # placing them 10 above the code value. The stream opens with control
        # markers (<custom_token_5>, <custom_token_1>) that would decode to
        # negative codes, so anything below the offset is dropped rather than
        # allowed to poison the frame alignment.
        return [c for c in (int(n) - 10 for n in
                            re.findall(r"<custom_token_(\d+)>", out)) if c >= 0]

    def _synth(self, text, voice, speed, model="", language=""):
        import torch
        name, quant = self.split_voice(voice)
        quant = model or quant
        m = self.load()
        snac, dev = m["snac"], m["dev"]

        if m.get("gguf"):                   # llama.cpp backend
            if name in self.VOICES:
                if quant.startswith("pretrained-"):
                    quant = ""              # presets live in the finetuned model
                url = self._ensure_server(quant or self.default_model())
                tok = self._tokenize(url, f"{name}: {text}")
                ids = [self.START_HUMAN] + tok + [self.EOT, self.END_HUMAN]
                codes = self._codes_via_llamacpp(url, ids, text)
            else:
                pts = self.pt_quants()
                if not pts and not self.SERVER:
                    raise RuntimeError("cloning a voice with Orpheus needs the pretrained model: "
                                       "download it in Settings -> Voice -> Models")
                if not quant.startswith("pretrained-"):
                    quant = next((q for q in ("pretrained-Q8_0", "pretrained-Q6_K",
                                              "pretrained-Q4_K_M") if q in pts), next(iter(pts), ""))
                url = self._ensure_server(quant)
                ids = self._clone_prompt(m, url, text, name)
                codes = self._codes_via_llamacpp(url, ids, text, temperature=0.5)
            return self._decode_codes(codes, snac, dev)

        tok, lm = m["tok"], m["lm"]
        if name not in self.VOICES:
            raise RuntimeError("Orpheus cloning runs on a GGUF quant of the pretrained model")
        ids = tok(f"{name}: {text}", return_tensors="pt").input_ids
        ids = torch.cat([torch.tensor([[self.START_HUMAN]]), ids,
                         torch.tensor([[self.EOT, self.END_HUMAN]])], dim=1).to(dev)
        # ~134 audio tokens buy a second of speech, so budget from the text
        # length instead of always burning the full window on a short line.
        budget = min(2000, 200 + 75 * len(text.split()))
        with torch.no_grad():
            out = lm.generate(ids, max_new_tokens=budget, do_sample=True,
                              temperature=0.6, top_p=0.95, repetition_penalty=1.1,
                              eos_token_id=self.EOS,
                              pad_token_id=tok.eos_token_id or self.EOS)

        row = out[0].tolist()
        if self.START_AUDIO in row:                   # keep only the audio span
            row = row[len(row) - 1 - row[::-1].index(self.START_AUDIO) + 1:]
        codes = [t - self.CODE_OFFSET for t in row if t != self.EOS]
        return self._decode_codes(codes, snac, dev)

    def _decode_codes(self, codes, snac, dev) -> tuple[bytes, int]:
        import torch
        codes = codes[:len(codes) // 7 * 7]            # 7 codes per SNAC frame
        if not codes:
            return b"", 24000

        # Orpheus interleaves the three SNAC layers 1/2/3/3/2/3/3 per frame.
        l1, l2, l3 = [], [], []
        for i in range(len(codes) // 7):
            f = codes[7 * i:7 * i + 7]
            l1.append(f[0])
            l2 += [f[1] - 4096, f[4] - 4 * 4096]
            l3 += [f[2] - 2 * 4096, f[3] - 3 * 4096, f[5] - 5 * 4096, f[6] - 6 * 4096]
        if min(min(l1), min(l2), min(l3)) < 0:         # malformed generation
            return b"", 24000

        layers = [torch.tensor(x, device=dev).unsqueeze(0) for x in (l1, l2, l3)]
        with torch.no_grad():
            audio = snac.decode(layers)
        return float_to_pcm16(audio), 24000

    def unload(self):
        super().unload()
        self.stop_server()


# ---------------------------------------------------------------- registry
ENGINES: dict[str, Engine] = {e.name: e for e in
                              (LlamaCppEngine(), PiperEngine(), KokoroEngine(),
                               ChatterboxEngine(), OrpheusEngine())}
DEFAULT_ENGINE = os.environ.get("TTS_ENGINE", "piper")


def get(name: str = "") -> Engine:
    return ENGINES.get(name or DEFAULT_ENGINE) or ENGINES["piper"]


def select(name: str) -> Engine:
    """Make `name` the resident GPU engine, evicting any other one."""
    eng = get(name)
    with _lock:
        if eng.gpu:
            for other in ENGINES.values():
                if other is not eng and other.gpu and other.loaded:
                    other.unload()
        eng.load()
    return eng


def synth_wav(text: str, voice: str = "", engine: str = "", speed: float = 1.0,
              model: str = "", language: str = "") -> bytes:
    eng = get(engine)
    if eng.gpu and not eng.loaded:
        select(eng.name)
    return eng.synth_wav(text, voice, speed, model, language)


def unload_all() -> list[str]:
    freed = []
    for e in ENGINES.values():
        if e.loaded:
            freed.append(e.name)
        e.unload()
    return freed


def describe() -> list[dict]:
    lib = {v["id"]: v for v in voice_library.list_voices()}
    out = []
    for e in ENGINES.values():
        ok, why = e.available()
        try:
            status = e.voice_status()
        except Exception:
            status = {}
        try:
            voices = e.voices()
        except Exception:
            voices = []
        try:
            models = e.models()
        except Exception:
            models = []
        vlist = []
        for v in voices:
            lv = lib.get(v)
            vlist.append({"id": v, "label": (lv["name"] if lv else v),
                          "kind": "custom" if lv else "preset",
                          "installed": bool(status.get(v, True))})
        out.append({"name": e.name, "label": e.label, "gpu": e.gpu,
                    "vram_mb": e.vram_mb, "note": e.note,
                    "installed": ok, "detail": why, "loaded": e.loaded,
                    "cloning": e.cloning, "cloning_experimental": e.cloning_experimental,
                    "speed_native": e.speed_native,
                    "languages": e.languages,
                    "voices": vlist, "voice_status": status,
                    "missing": sorted(v for v, k in status.items() if not k),
                    "default_voice": e.default_voice(),
                    "models": models, "default_model": e.default_model()})
    return out


# ------------------------------------------------------------ idle unloading
def _idle_watch():
    while True:
        time.sleep(30)
        if IDLE_UNLOAD_S <= 0:
            continue
        now = time.monotonic()
        for e in list(ENGINES.values()):
            try:
                if e.gpu and e.loaded and e.last_used and now - e.last_used > IDLE_UNLOAD_S:
                    print(f"[TTS] {e.label} idle for {IDLE_UNLOAD_S:.0f} s -> unloading", flush=True)
                    with _lock:
                        e.unload()
            except Exception as ex:
                print(f"[TTS] idle unload of {e.name} failed: {ex}", flush=True)


threading.Thread(target=_idle_watch, name="tts-idle-unload", daemon=True).start()


# ---------------------------------------------------------------- downloads
# What can be fetched from Settings -> Voice -> Models. Each item lands where
# speech_paths.py says, so a hand-placed copy and a downloaded one look the same.
_Q3 = "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
_Q3B = "Qwen3-TTS-12Hz-1.7B-Base"
CATALOG: list[dict] = [
    {"engine": "llamacpp", "id": f"{_Q3B}-Q8_0", "label": "Qwen3-TTS 1.7B Q8_0", "repo": _Q3,
     "dir": NATIVE_DIR, "files": [f"{_Q3B}-Q8_0.gguf", f"mmproj-{_Q3B}-bf16.gguf"],
     "size_mb": 2517, "recommended": True, "note": "best balance; ~4.5-5.5 GB VRAM, ~5x realtime"},
    {"engine": "llamacpp", "id": f"{_Q3B}-Q4_K_M", "label": "Qwen3-TTS 1.7B Q4_K_M", "repo": _Q3,
     "dir": NATIVE_DIR, "files": [f"{_Q3B}-Q4_K_M.gguf", f"mmproj-{_Q3B}-Q8_0.gguf"],
     "size_mb": 1482, "note": "smallest download and VRAM"},
    {"engine": "llamacpp", "id": f"{_Q3B}-bf16", "label": "Qwen3-TTS 1.7B bf16", "repo": _Q3,
     "dir": NATIVE_DIR, "files": [f"{_Q3B}-bf16.gguf", f"mmproj-{_Q3B}-bf16.gguf"],
     "size_mb": 4142, "note": "full precision"},
    {"engine": "orpheus", "id": "Q8_0", "label": "Orpheus 3B finetuned Q8_0",
     "repo": "unsloth/orpheus-3b-0.1-ft-GGUF", "dir": ORPHEUS_DIR,
     "files": ["orpheus-3b-0.1-ft-Q8_0.gguf"], "size_mb": 3516, "recommended": True,
     "note": "8 preset voices; community GGUF of canopylabs/orpheus-3b-0.1-ft"},
    {"engine": "orpheus", "id": "Q4_K_M", "label": "Orpheus 3B finetuned Q4_K_M",
     "repo": "unsloth/orpheus-3b-0.1-ft-GGUF", "dir": ORPHEUS_DIR,
     "files": ["orpheus-3b-0.1-ft-Q4_K_M.gguf"], "size_mb": 2093,
     "note": "faster, slightly rougher"},
    {"engine": "orpheus", "id": "pretrained-Q8_0", "label": "Orpheus 3B pretrained Q8_0",
     "repo": "HorizonNexusAI/orpheus-3b-0.1-pretrained-GGUF", "dir": ORPHEUS_DIR,
     "files": [("unsloth.Q8_0.gguf", "orpheus-3b-0.1-pretrained-Q8_0.gguf")], "size_mb": 3516,
     "note": "clones your voices (needs the clip's transcript); community GGUF of "
             "canopylabs/orpheus-3b-0.1-pretrained"},
    {"engine": "orpheus", "id": "snac", "label": "SNAC 24 kHz codec (Orpheus needs it)",
     "hf_snapshot": "hubertsiuzdak/snac_24khz", "size_mb": 80},
    {"engine": "kokoro", "id": "kokoro", "label": "Kokoro-82M weights + all voices",
     "hf_snapshot": "hexgrad/Kokoro-82M", "size_mb": 330, "recommended": True},
    {"engine": "chatterbox", "id": "chatterbox", "label": "Chatterbox weights",
     "hf_snapshot": "ResembleAI/chatterbox", "size_mb": 3000, "recommended": True},
]
PIPER_CATALOG = [
    ("en_US-lessac-medium", 63), ("en_US-ryan-high", 121), ("en_US-amy-medium", 63),
    ("en_US-libritts_r-medium", 79), ("en_GB-alba-medium", 63),
    ("en_GB-northern_english_male-medium", 63), ("fi_FI-harri-medium", 63),
    ("de_DE-thorsten-high", 114), ("fr_FR-siwis-medium", 63), ("es_ES-davefx-medium", 63),
    ("it_IT-paola-medium", 63), ("sv_SE-nst-medium", 63),
]
for _v, _mb in PIPER_CATALOG:
    _p = piper_parts(_v)
    CATALOG.append({"engine": "piper", "id": _v, "label": _v, "dir": PIPER_DIR, "size_mb": _mb,
                    "urls": [PIPER_URL.format(ext=e, **_p) for e in (".onnx", ".onnx.json")],
                    "recommended": _v in ("en_US-lessac-medium", "en_US-ryan-high")})

_DL = {"running": False, "engine": "", "id": "", "file": "", "done": 0, "total": 0,
       "error": "", "finished": False, "started": 0.0, "cancel": False}
_dl_lock = threading.Lock()


def _hf_token() -> str:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    for p in (CACHE_ROOT / "hf" / "token", Path.home() / ".cache" / "huggingface" / "token"):
        try:
            t = p.read_text().strip()
            if t:
                return t
        except OSError:
            pass
    return ""


def _installed_item(item: dict) -> bool:
    e = item["engine"]
    if e == "piper":
        return item["id"] in PiperEngine.installed()
    if "hf_snapshot" in item:
        snap = CACHE_ROOT / "hf" / "hub" / ("models--" + item["hf_snapshot"].replace("/", "--")) / "snapshots"
        return snap.is_dir() and any(snap.iterdir())
    if e == "llamacpp":
        return item["id"] in LlamaCppEngine.model_files()
    if e == "orpheus":
        return item["id"] in OrpheusEngine.quants()
    return False


def catalog() -> dict:
    """Everything on disk plus everything downloadable, per engine."""
    engines = []
    for e in ENGINES.values():
        ok, why = e.available()
        items = []
        known = set()
        for it in CATALOG:
            if it["engine"] != e.name:
                continue
            known.add(it["id"])
            items.append({"id": it["id"], "label": it["label"], "size_mb": it["size_mb"],
                          "installed": _installed_item(it), "recommended": bool(it.get("recommended")),
                          "note": it.get("note", ""),
                          "source": it.get("repo") or it.get("hf_snapshot") or "rhasspy/piper-voices",
                          "dest": str(it.get("dir") or (CACHE_ROOT / "hf"))})
        for m in e.models():                       # hand-placed models not in the catalog
            if m["id"] not in known:
                items.append({"id": m["id"], "label": m.get("label", m["id"]),
                              "size_mb": m.get("size_mb", 0), "installed": True,
                              "recommended": False, "note": "found on disk",
                              "source": "local", "dest": m.get("path", "")})
        if e.name == "piper":
            for v, p in PiperEngine.installed().items():
                if v not in known:
                    items.append({"id": v, "label": v, "size_mb": round(p.stat().st_size / 1e6),
                                  "installed": True, "recommended": False, "note": "found on disk",
                                  "source": "local", "dest": str(p.parent)})
        engines.append({"engine": e.name, "label": e.label, "installed": ok, "detail": why,
                        "items": items})
    from speech_paths import describe as paths
    return {"paths": paths(), "engines": engines, "download": download_status(),
            "offline": OFFLINE}


def _fetch(url: str, dest: Path, use_token: bool) -> None:
    d = _DL
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "tts-server/2"}
    tok = _hf_token() if use_token else ""
    if tok and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {tok}"
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        resumed = r.status == 206
        if not resumed:
            start = 0
        total = start + int(r.headers.get("Content-Length") or 0)
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            total = int(cr.rsplit("/", 1)[1])
        d["file"], d["done"], d["total"] = dest.name, start, total
        with open(part, "ab" if resumed else "wb") as f:
            while chunk := r.read(4 << 20):
                if d["cancel"]:
                    raise RuntimeError("cancelled")
                f.write(chunk)
                d["done"] += len(chunk)
    got = part.stat().st_size
    if total and got < total:
        raise RuntimeError(f"incomplete download of {dest.name} ({got} of {total} bytes)")
    part.replace(dest)


def _hf_snapshot(repo: str, size_mb: int) -> None:
    """huggingface_hub in a child process: this one stays HF_HUB_OFFLINE."""
    d = _DL
    env = dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0",
               HF_HOME=str(CACHE_ROOT / "hf"))
    code = ("import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(sys.argv[1])")
    d["file"], d["total"] = repo, size_mb * 1_000_000
    p = subprocess.Popen([sys.executable, "-c", code, repo], env=env, creationflags=_NO_WINDOW,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    root = CACHE_ROOT / "hf" / "hub" / ("models--" + repo.replace("/", "--"))
    while p.poll() is None:
        if d["cancel"]:
            p.terminate()
            raise RuntimeError("cancelled")
        try:
            d["done"] = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        except Exception:
            pass
        time.sleep(1)
    out = (p.stdout.read() if p.stdout else "") or ""
    if p.returncode != 0:
        raise RuntimeError(out.strip().splitlines()[-1][:300] if out.strip() else f"exit {p.returncode}")
    d["done"] = d["total"]


def _run_download(item: dict) -> None:
    d = _DL
    try:
        if "hf_snapshot" in item:
            _hf_snapshot(item["hf_snapshot"], item["size_mb"])
        else:
            dest_dir = Path(item["dir"])
            dest_dir.mkdir(parents=True, exist_ok=True)
            jobs = []
            for u in item.get("urls", []):
                jobs.append((u, dest_dir / u.rsplit("/", 1)[1].split("?")[0]))
            for f in item.get("files", []):
                src, dst = (f, f) if isinstance(f, str) else f
                jobs.append((f"https://huggingface.co/{item['repo']}/resolve/main/{src}", dest_dir / dst))
            for url, dest in jobs:
                if dest.exists():
                    continue
                _fetch(url, dest, use_token=True)
            if item["engine"] == "orpheus" and not _installed_item(
                    next(c for c in CATALOG if c["id"] == "snac")):
                _hf_snapshot("hubertsiuzdak/snac_24khz", 80)
        d["finished"] = True
        print(f"[TTS] downloaded {item['engine']}/{item['id']}", flush=True)
    except Exception as e:
        d["error"] = str(e).splitlines()[0][:300] if str(e) else type(e).__name__
        d["finished"] = True
        print(f"[TTS] download of {item['engine']}/{item['id']} failed: {e}", flush=True)
    finally:
        d["running"] = False


def start_download(engine: str, item_id: str, wait: bool = False) -> dict:
    item = next((c for c in CATALOG if c["engine"] == engine and c["id"] == item_id), None)
    if item is None:
        return {"ok": False, "error": f"nothing called {engine}/{item_id} in the catalog"}
    with _dl_lock:
        if _DL["running"]:
            return {"ok": False, "error": f"already downloading {_DL['engine']}/{_DL['id']}"}
        _DL.update(running=True, engine=engine, id=item_id, file="", done=0, total=0,
                   error="", finished=False, started=time.time(), cancel=False)
    t = threading.Thread(target=_run_download, args=(item,), daemon=True)
    t.start()
    if wait:
        t.join()
        return {"ok": not _DL["error"], **download_status()}
    return {"ok": True, "engine": engine, "id": item_id}


def cancel_download() -> dict:
    _DL["cancel"] = True
    return download_status()


def download_status() -> dict:
    return {k: v for k, v in _DL.items() if k != "cancel"}
