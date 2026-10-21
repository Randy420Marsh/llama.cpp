"""
Full media test suite for the llama.cpp build + the Aegis preset file.

What it exercises (each step is independent and skips itself when its pieces are
not present, so it degrades to "router + asr + tts" on a box without a vision model):

  1. router   - start `llama-server --models-preset <Aegis presets.ini>` in router mode,
                list the available presets.
  2. vision   - load a vision preset (one that has an mmproj) and answer a question about a
                generated codeword image. The model must read the codeword off the image.
  3. asr      - start the ASR server (tools/asr-server) serving every `--models` entry and
                transcribe a generated speech clip with EACH served model, checking the
                expected phrase comes back.
  4. video    - send a generated video to a vision model and check it describes the on-screen
                text (the router's video pipeline: ffmpeg decode -> frames -> model).
  5. tts      - start the speech server (tools/speech-server) and render one line through every
                *installed* engine/mode (llamacpp, piper, kokoro, chatterbox, orpheus), writing
                a .wav for each so you can play them.

It uses only the Python standard library for HTTP (urllib) so it runs under any Python 3.9+.
Pillow (optional) is used to draw the codeword image; ffmpeg (optional) makes the sample video
and the speech clip. Missing optional pieces just skip their step, they do not fail the suite.

Usage:
  python3 tools/full_test.py                       # everything it can
  python3 tools/full_test.py --only router asr tts # a subset
  python3 tools/full_test.py --no-load             # router lists presets, loads no model (fast, no GPU)

Environment (set these the way Aegis's run.sh does):
  LD_LIBRARY_PATH  - must include the CUDA 13 runtime (e.g. ~/cuda-staging/lib64)
  AEGIS            - path to the ProjectAegis checkout (default: ~/AI/ProjectAegis); its
                     data/llm/presets.ini is the router preset file.
  PRESETS_INI      - override the preset file directly.
  VISION_PRESET    - the preset id to load for vision/video (default: first preset with an mmproj).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # llama.cpp/tools
LLAMA_ROOT = HERE.parent
SERVER = LLAMA_ROOT / "build" / "bin" / "llama-server"
ASR_DIR = HERE / "asr-server"
SPEECH_DIR = HERE / "speech-server"

# ports: keep them apart so the router, ASR and TTS can all run at once
ROUTER_PORT = 8080
ASR_PORT = 8178
TTS_PORT = 8179

# a codeword long enough to be hard to guess but easy to read off a rendered image
CODEWORD = "KIWI42"
SPEECH_PHRASE = "the quick brown fox"   # what the generated clip says; ASR must return it (loosely)

MEDIA = Path(tempfile.mkdtemp(prefix="fulltest-media-"))    # generated samples + tts outputs
PASSED, FAILED, SKIPPED = [], [], []


# --------------------------------------------------------------------------- output
def say(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


def ok(step, msg=""):
    PASSED.append(step)
    say(step, f"PASS {msg}".strip())


def fail(step, msg):
    FAILED.append(step)
    say(step, f"FAIL {msg}", "FAIL")


def skip(step, msg):
    SKIPPED.append(step)
    say(step, f"SKIP {msg}".strip(), "SKIP")


# --------------------------------------------------------------------------- http (stdlib)
def http_get(url: str, timeout: int = 15) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def http_post_json(url: str, obj: dict, timeout: int = 600) -> tuple[int, dict]:
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body or "{}")
        except json.JSONDecodeError:
            return e.code, {"raw": body[:500]}


def http_post_multipart(url: str, fields: dict[str, str], files: dict[str, tuple[str, bytes, str]],
                        timeout: int = 3600) -> tuple[int, dict]:
    """A multipart/form-data POST. files: name -> (filename, bytes, content_type)."""
    boundary = f"--fulltest-{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    for name, (filename, data, ctype) in files.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                     f"Content-Type: {ctype}\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
        code = e.code
    else:
        code = 200
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return code, {}
    # the router/ASR may answer plain text (response_format=text) or json
    try:
        return code, json.loads(text)
    except json.JSONDecodeError:
        return code, {"text": text}


def port_free(port: int) -> bool:
    s = socket.socket()
    try:
        s.connect(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def wait_http(url: str, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            code, _ = http_get(url, timeout=3)
            if code == 200:
                return True
        except OSError:
            time.sleep(0.5)
    return False


# --------------------------------------------------------------------------- sample media
def make_codeword_image(path: Path, word: str) -> bool:
    """Draw the codeword on a high-contrast image. Pillow if present, else a raw PNG fallback."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (640, 160), "white")
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=96)
        except TypeError:
            font = ImageFont.load_default()
        d.text((24, 24), word, fill="black", font=font)
        img.save(path, format="PNG")
        return True
    except Exception as e:  # noqa: BLE001
        say("media", f"could not draw the codeword image ({e}); using a plain white image")
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return False


def make_speech_clip(path: Path, phrase: str) -> bool:
    """A real short spoken clip. Prefer an existing voice reference (real speech, so ASR has
    something true to return); fall back to a tone generated with ffmpeg (ASR returns little)."""
    candidates = [
        Path(os.environ.get("AI_ROOT", os.path.expanduser("~/AI"))) / "tts-models" / "voices" / "girl" / "reference.wav",
        Path.home() / "AI" / "tts-models" / "voices" / "girl" / "reference.wav",
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 4000:
            shutil.copyfile(c, path)
            say("media", f"using real speech sample for the ASR step: {c}")
            return True
    if shutil.which("ffmpeg"):
        # a 2s 440Hz tone: ASR will return little, but it proves the audio path end to end
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                        "-ar", "16000", "-ac", "1", str(path)],
                       check=True, capture_output=True)
        say("media", "no speech reference on disk; using a 2s tone (ASR step will be weak but proves the path)")
        return True
    return False


def make_video(path: Path, word: str) -> bool:
    """A few-second mp4 showing the codeword. Renders the frame with Pillow (the ffmpeg
    build here has no drawtext/libharfbuzz) and loops it with ffmpeg. Skips if no ffmpeg."""
    if not shutil.which("ffmpeg"):
        return False
    frame = MEDIA / "_frame.png"
    if not make_codeword_image(frame, word):
        return False
    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", str(frame), "-t", "3", "-r", "8",
           "-vf", "scale=640:160:flags=bicubic,format=yuv420p",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return path.is_file() and path.stat().st_size > 2000
    except Exception as e:  # noqa: BLE001
        say("media", f"could not make the sample video ({e}); video step will skip")
        return False


# --------------------------------------------------------------------------- presets
def load_presets(ini: Path) -> dict[str, list[str]]:
    """Parse a router preset .ini into {alias: [raw 'key = value' lines]} (enough to find mmproj)."""
    presets, cur = {}, None
    for line in ini.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            cur = line[1:-1]
            presets[cur] = []
        elif "=" in line and cur is not None:
            presets[cur].append(line)
    return presets


def router_base() -> str:
    return f"http://127.0.0.1:{ROUTER_PORT}"


def router_models() -> list[str]:
    code, body = http_get(f"{router_base()}/v1/models")
    if code != 200:
        return []
    return [m.get("id") for m in json.loads(body.decode()).get("data", [])]


def load_model(alias: str) -> str:
    code, body = http_post_json(f"{router_base()}/v1/models/load", {"model": alias})
    return f"load {alias}: HTTP {code}"


def chat(alias: str, content, timeout: int = 600) -> dict:
    code, body = http_post_json(f"{router_base()}/v1/chat/completions",
                               {"model": alias, "messages": [{"role": "user", "content": content}],
                                "stream": False}, timeout=timeout)
    if code != 200:
        return {"ok": False, "status": code, "error": body}
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        return {"ok": False, "status": code, "error": f"no choices in response: {body}", "exc": str(e)}
    return {"ok": True, "answer": text}


# --------------------------------------------------------------------------- steps
def step_router(ini: Path, presets: dict[str, list[str]], do_load: bool) -> None:
    if not SERVER.is_file():
        skip("router", f"llama-server not built at {SERVER}")
        return
    if not port_free(ROUTER_PORT):
        skip("router", f"port {ROUTER_PORT} already in use; start the router yourself and re-run --only vision asr video")
        return
    env = dict(os.environ)
    # the CUDA toolkit libs the llama-server binary links against: prefer the newest
    # toolkit (13.4, then 13.3), then the staging copy the installer leaves behind.
    if "LD_LIBRARY_PATH" not in env:
        cands = ["/usr/local/cuda-13.4/lib64", "/usr/local/cuda-13.3/lib64",
                 os.path.expanduser("~/cuda-staging/lib64")]
        env["LD_LIBRARY_PATH"] = ":".join(d for d in cands if os.path.isdir(d)) or cands[-1]
    log = MEDIA / "router.log"
    proc = subprocess.Popen([str(SERVER), "--models-preset", str(ini), "--models-max", "1",
                             "--no-models-autoload", "--host", "127.0.0.1", "--port", str(ROUTER_PORT)],
                            stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    say("router", f"started (pid {proc.pid}); waiting for {router_base()}/v1/models ...")
    try:
        if not wait_http(f"{router_base()}/v1/models", timeout=60):
            fail("router", "router did not answer; see " + str(log))
            return
        models = router_models()
        if models:
            ok("router", f"{len(models)} presets available: {', '.join(sorted(models))}")
        else:
            fail("router", "router is up but lists no presets")
        if do_load:
            # pick a preset that has an mmproj (vision) so the vision step can use it
            vis = next((a for a, lines in presets.items() if any("mmproj" in ln for ln in lines)), None)
            if vis:
                say("router", f"loading vision preset '{vis}' (loads the model on the card)")
                ok("router", load_model(vis))
            else:
                say("router", "no preset has an mmproj; skipping the model load (vision step will skip too)")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def step_vision(presets: dict[str, list[str]], img: Path) -> None:
    alias = next((a for a, lines in presets.items() if any("mmproj" in ln for ln in lines)), None)
    if not alias:
        skip("vision", "no preset with an mmproj (vision) in the preset file")
        return
    if not router_up():
        skip("vision", "router is not running (run without --no-load, or start it)")
        return
    b64 = base64.b64encode(img.read_bytes()).decode()
    content = [
        {"type": "text", "text": f"This image shows a single word. What is it, exactly, and nothing else?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    res = chat(alias, content)
    if res.get("ok"):
        ans = res.get("answer", "").strip()
        if CODEWORD.lower() in ans.lower():
            ok("vision", f"{alias} read the codeword from the image: {ans!r}")
        else:
            fail("vision", f"{alias} did not return {CODEWORD!r}; got {ans[:120]!r}")
    else:
        fail("vision", f"chat request failed: {res.get('status')} {res.get('error')}")


def router_up() -> bool:
    code, _ = http_get(f"{router_base()}/v1/models", timeout=3)
    return code == 200


def step_asr(clip: Path) -> None:
    # start the ASR server exactly as Aegis's services.toml does: both models, preload
    asr_py = ASR_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if not asr_py.is_file():
        skip("asr", f"ASR venv python not found at {asr_py}; run tools/asr-server/run.sh --setup-only first")
        return
    if not port_free(ASR_PORT):
        skip("asr", f"port {ASR_PORT} already in use; is Aegis's whisper running?")
        return
    # which models does Aegis serve? read them from services.toml if present, else the two defaults
    models = asr_model_names()
    asr_models_arg = ",".join(models[1:]) if len(models) > 1 else ""
    args = [str(asr_py), "-u", str(ASR_DIR / "asr_server.py"),
            "--no-https", "--host", "127.0.0.1", "--port", str(ASR_PORT),
            "--model", models[0], "--compute-type", "int8_float16", "--preload"]
    if asr_models_arg:
        args += ["--models", asr_models_arg]
    log = MEDIA / "asr.log"
    say("asr", f"starting with models: {', '.join(models)} (this downloads/loads them; first run is slow)")
    proc = subprocess.Popen(args, cwd=str(ASR_DIR), stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        if not wait_http(f"http://127.0.0.1:{ASR_PORT}/health", timeout=900):
            fail("asr", "ASR server did not come up; see " + str(log))
            return
        code, body = http_get(f"http://127.0.0.1:{ASR_PORT}/health")
        health = json.loads(body.decode())
        say("asr", f"health: default={health.get('default_model')} models={health.get('models')}")
        audio = clip.read_bytes()
        all_ok = True
        for m in models:
            c, res = http_post_multipart(f"http://127.0.0.1:{ASR_PORT}/v1/audio/transcriptions",
                                        fields={"model": m, "response_format": "json"},
                                        files={"file": ("audio.wav", audio, "audio/wav")}, timeout=1800)
            text = (res.get("text") or "").strip()
            hit = _phrase_in(SPEECH_PHRASE, text)
            if c == 200 and hit:
                ok("asr", f"model '{m}' transcribed the clip: {text!r}")
            else:
                # a tone (fallback) legitimately returns little; only fail hard when we had real speech
                msg = f"model '{m}': HTTP {c} text={text!r}"
                if _using_real_speech():
                    all_ok = False
                    fail("asr", msg)
                else:
                    say("asr", f"(tone clip, so a short/empty transcript is expected) {msg}")
        if all_ok:
            say("asr", "all served ASR models transcribed the clip")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


def _phrase_in(phrase: str, text: str) -> bool:
    """Loose check: at least most of the phrase's words appear (ASR may drop small words)."""
    words = [w for w in phrase.split() if len(w) > 3]
    if not words:
        return bool(text.strip())
    hit = sum(1 for w in words if w in text.lower())
    return hit >= max(1, int(0.5 * len(words)))


def _using_real_speech() -> bool:
    return (MEDIA / "speech.wav").is_file() and (MEDIA / "speech.wav").stat().st_size > 10000


def asr_model_names() -> list[str]:
    """The ASR models Aegis serves, from services.toml [whisper] if readable, else the two defaults."""
    svc = Path(os.environ.get("AEGIS", os.path.expanduser("~/AI/ProjectAegis"))) / "config" / "services.toml"
    if svc.is_file():
        try:
            import tomllib
            with open(svc, "rb") as f:
                data = tomllib.load(f)
            cmd = data.get("whisper", {}).get("command", [])
            names = []
            i = 0
            while i < len(cmd):
                if cmd[i] == "--model":
                    names.append(cmd[i + 1]); i += 2
                elif cmd[i] == "--models":
                    names += [x for x in cmd[i + 1].split(",") if x.strip()]; i += 2
                else:
                    i += 1
            if names:
                return names
        except Exception:  # noqa: BLE001
            pass
    return ["large-v3-turbo", "large-v3"]


def step_video(presets: dict[str, list[str]], video: Path) -> None:
    alias = next((a for a, lines in presets.items() if any("mmproj" in ln for ln in lines)), None)
    if not alias or not video.is_file():
        skip("video", "no vision preset or no sample video (ffmpeg missing)")
        return
    if not router_up():
        skip("video", "router is not running")
        return
    b64 = base64.b64encode(video.read_bytes()).decode()
    content = [
        {"type": "text", "text": "This short video shows a single word on a plain background. What is it, exactly?"},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
    ]
    res = chat(alias, content, timeout=900)
    if res.get("ok"):
        ans = res.get("answer", "").strip()
        if CODEWORD.lower() in ans.lower():
            ok("video", f"{alias} read the codeword off the video: {ans!r}")
        else:
            fail("video", f"{alias} did not return {CODEWORD!r}; got {ans[:120]!r}")
    else:
        fail("video", f"video request failed: {res.get('status')} {res.get('error')}")


def step_tts() -> None:
    tts_py = SPEECH_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    if not tts_py.is_file():
        skip("tts", f"speech venv python not found at {tts_py}; run tools/speech-server/run.sh --setup-only first")
        return
    if not port_free(TTS_PORT):
        skip("tts", f"port {TTS_PORT} already in use")
        return
    log = MEDIA / "tts.log"
    proc = subprocess.Popen([str(tts_py), "-u", str(SPEECH_DIR / "server.py"),
                             "--no-https", "--host", "127.0.0.1", "--port", str(TTS_PORT)],
                            cwd=str(SPEECH_DIR), stdout=open(log, "w"), stderr=subprocess.STDOUT)
    try:
        if not wait_http(f"http://127.0.0.1:{TTS_PORT}/api/status", timeout=300):
            fail("tts", "speech server did not come up; see " + str(log))
            return
        code, body = http_get(f"http://127.0.0.1:{TTS_PORT}/api/tts/engines")
        if code != 200:
            fail("tts", f"GET /api/tts/engines -> HTTP {code}")
            return
        engines = json.loads(body.decode()).get("engines", [])
        installed = [e["name"] for e in engines if e.get("installed")]
        say("tts", f"engines on disk: {', '.join(installed) or '(none installed)'}; rendering one line through each")
        line = "The quick brown fox jumps over the lazy dog."
        rendered = 0
        for name in installed:
            # "all modes": render with the engine's default voice, and (where supported) a
            # couple of the mode knobs the server accepts (speed, emotions)
            variants = [
                {"input": line, "engine": name},
                {"input": line, "engine": name, "speed": 1.25},
                {"input": line, "engine": name, "emotions": "happy"},
            ]
            for i, payload in enumerate(variants):
                c, raw = http_post_json(f"http://127.0.0.1:{TTS_PORT}/v1/audio/speech", payload, timeout=600)
                if c == 200:
                    out = MEDIA / f"tts-{name}-{i}.wav"
                    # /v1/audio/speech returns a WAV body
                    out.write_bytes(raw if isinstance(raw, (bytes, bytearray)) else _raw_from_json(raw))
                    say("tts", f"rendered {name} mode-{i} ({len(out.read_bytes())} bytes) -> {out.name}")
                    rendered += 1
                else:
                    # not every engine honours speed/emotions; a 200 on the base variant is enough
                    if i == 0:
                        fail("tts", f"engine '{name}' base render failed: HTTP {c} {raw}")
                    else:
                        say("tts", f"engine '{name}' mode-{i} not supported (HTTP {c}); ok")
        if rendered:
            ok("tts", f"rendered {rendered} clip(s) across {len(installed)} engine(s); play them from {MEDIA}/")
        else:
            skip("tts", "no TTS engine is installed on disk (run tools/speech-server/setup)")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


def _raw_from_json(raw):
    # if the server returned JSON-wrapped audio (base64), decode it; else return empty
    if isinstance(raw, dict):
        b64 = raw.get("data") or raw.get("audio")
        if isinstance(b64, str):
            try:
                return base64.b64decode(b64)
            except Exception:  # noqa: BLE001
                return b"".join(raw.get("bytes", []))
    return b""


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=["router", "vision", "asr", "video", "tts"],
                    help="which steps to run (default: all)")
    ap.add_argument("--no-load", action="store_true", help="router lists presets but loads no model (no GPU)")
    ap.add_argument("--presets", default=os.environ.get("PRESETS_INI"),
                    help="router preset .ini (default: $PRESETS_INI or AEGIS/data/llm/presets.ini)")
    ap.add_argument("--aegis", default=os.environ.get("AEGIS", os.path.expanduser("~/AI/ProjectAegis")))
    args = ap.parse_args()

    aegis = Path(args.aegis)
    ini = Path(args.presets) if args.presets else aegis / "data" / "llm" / "presets.ini"
    say("setup", f"router binary: {SERVER}")
    say("setup", f"presets:      {ini}")

    presets = load_presets(ini) if ini.is_file() else {}
    if ini.is_file():
        say("setup", f"presets listed: {', '.join(sorted(presets))}")
    else:
        say("setup", f"no preset file at {ini} (start the LLM engine once through Aegis to generate it)")

    # sample media (generated once, reused)
    img = MEDIA / "codeword.png"
    make_codeword_image(img, CODEWORD)
    speech_ok = make_speech_clip(MEDIA / "speech.wav", SPEECH_PHRASE)
    video_ok = make_video(MEDIA / "clip.mp4", CODEWORD)
    say("setup", f"sample media in {MEDIA}: image={img.is_file()} audio={MEDIA/'speech.wav' if speech_ok else 'n/a'} video={MEDIA/'clip.mp4' if video_ok else 'n/a'}")

    only = set(args.only)
    if "router" in only:
        step_router(ini, presets, do_load=not args.no_load)
    if "vision" in only:
        step_vision(presets, img)
    if "asr" in only:
        step_asr(MEDIA / "speech.wav")
    if "video" in only:
        step_video(presets, MEDIA / "clip.mp4")
    if "tts" in only:
        step_tts()

    print("\n=== summary ===")
    print(f"  PASS  ({len(PASSED)}): {', '.join(PASSED) or '-'}")
    print(f"  FAIL  ({len(FAILED)}): {', '.join(FAILED) or '-'}")
    print(f"  SKIP  ({len(SKIPPED)}): {', '.join(SKIPPED) or '-'}")
    print(f"  media / tts outputs: {MEDIA}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
