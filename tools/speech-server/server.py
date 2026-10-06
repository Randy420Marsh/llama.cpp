"""
Local speech server: text-to-speech, speech-to-text and your own voices.

The companion of the llama.cpp web UI (Settings -> Voice): that page lists the
engines, voices and models from here, records new voices into the library,
and sends every read-aloud sentence to /v1/audio/speech. Any other client can
use the same API. The full chat app is the separate llama.cpp-server-tts
project; this server only does speech.

    engines   llamacpp (Qwen3-TTS in llama.cpp, clones your voices), piper,
              kokoro, chatterbox, orpheus -- see tts_engines.py
    voices    %TTS_CACHE%\\voices\\<id>\\reference.wav (+ transcript) -- see
              voice_library.py
    models    where each one lives: speech_paths.py

    python server.py --no-https --lan    what the command center runs (port 8179)
    python server.py                     https on 127.0.0.1:8179 (TLS default)

API
    GET  /health                          liveness (the command center probes it)
    GET  /api/status                      engines, speech-to-text, folders, port
    POST /v1/audio/speech                 {input, engine, voice, model, language, speed} -> WAV
    GET  /api/tts/engines                 engines with their voices and models
    POST /api/tts/select | /api/tts/unload   load one engine / free VRAM
    GET  /api/voices/library              your voices
    POST /api/voices/library              multipart: name, file, transcript?, language?
    GET  /api/voices/library/{id}/audio   the reference clip
    PATCH /api/voices/library/{id}        {transcript?, language?, name?}
    DELETE /api/voices/library/{id}
    GET  /api/models                      everything on disk + downloadable, per engine
    POST /api/models/download             {engine, id}; poll GET /api/models/download/status
    POST /api/stt/transcribe              multipart file (+ language) -> {text}
    POST /v1/audio/transcriptions         the same, OpenAI-shaped
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

import speech_paths
import stt
import tts_engines
import voice_library

# The Windows console is cp1252 by default, so printing a Japanese or Chinese
# error message raises UnicodeEncodeError *inside* the error handler and turns
# a recoverable TTS failure into a 500. Never let logging be the thing that
# breaks the request.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
API_VERSION = 2
DEFAULT_PORT = 8179          # next to the speech-to-text server on 8178

app = FastAPI(title="local speech server")
# CORS: the llama.cpp web UI (and other browser tools) call this cross-origin,
# e.g. http://192.168.50.11:8080 -> http://192.168.50.11:8179. The X-TTS-*
# headers must be exposed or the page cannot read which voice really spoke.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-TTS-Engine", "X-TTS-Voice", "X-TTS-Model", "X-TTS-Speed",
                    "X-TTS-Fallback", "X-TTS-Error"],
)
_shutdown_done = False
_port = DEFAULT_PORT


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": msg}})


# ------------------------------------------------------------------ health
@app.get("/health")
def health():
    return {"status": "ok", "service": "tts-server", "api": API_VERSION,
            "engine": tts_engines.DEFAULT_ENGINE}


@app.get("/api/status")
def api_status():
    engines = [{"name": e.name, "label": e.label, "installed": e.available()[0],
                "loaded": e.loaded, "gpu": e.gpu} for e in tts_engines.ENGINES.values()]
    return {"service": "tts-server", "api": API_VERSION, "port": _port,
            "default_engine": tts_engines.DEFAULT_ENGINE, "offline": tts_engines.OFFLINE,
            "engines": engines, "stt": stt.describe(), "paths": speech_paths.describe(),
            "voices": len(voice_library.list_voices()),
            "reading_passage": voice_library.READING_PASSAGE}


# ------------------------------------------------------------------ speech
@app.post("/v1/audio/speech")
async def speech(req: Request):
    """OpenAI-compatible text-to-speech: {input, voice, engine, model, language, speed} -> WAV."""
    try:
        body = await req.json()
    except Exception:
        return _err(400, "body must be JSON")
    text = (body.get("input") or "").strip()
    if not text:
        return _err(400, "missing 'input'")
    engine = (body.get("engine") or tts_engines.DEFAULT_ENGINE).strip()
    if engine not in tts_engines.ENGINES:
        return _err(400, f"unknown engine '{engine}'; GET /api/tts/engines lists them")
    voice = (body.get("voice") or "").strip()
    model = (body.get("model") or body.get("quality") or "").strip()
    language = (body.get("language") or "").strip()
    try:
        speed = float(body.get("speed") or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    eng = tts_engines.get(engine)
    headers = {"X-TTS-Engine": engine, "X-TTS-Voice": voice or "(default)",
               "X-TTS-Model": model or "(default)",
               "X-TTS-Speed": "native" if eng.speed_native else "client"}
    try:
        wav = await _run(tts_engines.synth_wav, text, voice, engine, speed, model, language)
    except Exception as e:
        # Never let one bad sentence kill the read-aloud queue -- but say so
        # loudly. A silent fallback makes every broken voice sound like Piper's
        # default, which is indistinguishable from "the voice picker is a lie".
        print(f"[TTS] {engine}/{voice or 'default'} FAILED ({e}); using piper", flush=True)
        try:
            wav = await _run(tts_engines.get("piper").synth_wav, text, "", speed)
        except Exception as e2:
            return _err(500, f"{engine}: {e} / piper: {e2}")
        headers.update({"X-TTS-Fallback": "piper", "X-TTS-Speed": "native",
                        "X-TTS-Error": " ".join(str(e).split())[:300]})
    return Response(content=wav, media_type="audio/wav", headers=headers)


async def _run(fn, *args):
    """Synthesis blocks for seconds; keep it off the event loop."""
    import anyio
    return await anyio.to_thread.run_sync(lambda: fn(*args))


@app.get("/api/voices")
def api_voices(engine: str = ""):
    e = tts_engines.get(engine)
    return {"installed": [v for v, ok in e.voice_status().items() if ok],
            "available": e.voices(), "default": e.default_voice(), "engine": e.name}


@app.get("/api/tts/engines")
def api_tts_engines():
    return {"engines": tts_engines.describe(), "default": tts_engines.DEFAULT_ENGINE}


@app.post("/api/tts/select")
async def api_tts_select(req: Request):
    """Preload an engine (and evict any other GPU one) before reading starts."""
    body = await req.json()
    name = body.get("engine") or "piper"
    try:
        e = await _run(tts_engines.select, name)
        return {"ok": True, "engine": e.name, "loaded": e.loaded}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})


@app.post("/api/tts/unload")
async def api_tts_unload(req: Request):
    """Free VRAM: one engine ({"engine": name}), "stt", or everything ({})."""
    try:
        body = await req.json()
    except Exception:
        body = {}
    name = (body.get("engine") or "").strip()
    if name and name != "stt":
        tts_engines.get(name).unload()
        freed = [name]
    elif not name:
        freed = tts_engines.unload_all()
    else:
        freed = []
    if not name or name == "stt":
        stt.unload()
        freed.append("stt")
    return {"ok": True, "unloaded": freed}


# ------------------------------------------------------------- your voices
@app.get("/api/voices/library")
def api_library():
    return {"voices": voice_library.list_voices(), "dir": str(voice_library.LIB),
            "reading_passage": voice_library.READING_PASSAGE,
            "min_seconds": voice_library.MIN_SECONDS, "max_seconds": voice_library.MAX_SECONDS}


def _transcriber(path: str, language: str) -> str:
    ok, _ = stt.available()
    if not ok:
        return ""
    return (stt.transcribe(path, language) or {}).get("text", "")


@app.post("/api/voices/library")
async def api_library_add(name: str = Form(...), file: UploadFile = File(...),
                          transcript: str = Form(""), language: str = Form("")):
    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        return _err(413, "the clip is larger than 50 MB; 6-30 s of speech is all a voice needs")
    try:
        v = await _run(voice_library.add_voice, name, data, file.filename or "", transcript,
                       language, _transcriber)
        return {"ok": True, "voice": v}
    except FileExistsError as e:
        return _err(409, str(e))
    except ValueError as e:
        return _err(400, str(e))
    except Exception as e:
        return _err(500, str(e))


@app.get("/api/voices/library/{vid}/audio")
def api_library_audio(vid: str):
    p = voice_library.reference_path(vid)
    if p is None or not p.exists():
        return _err(404, f"no voice '{vid}'")
    return FileResponse(p, media_type="audio/wav" if p.suffix.lower() == ".wav" else None,
                        headers={"Cache-Control": "no-cache"})


@app.patch("/api/voices/library/{vid}")
async def api_library_update(vid: str, req: Request):
    body = await req.json()
    try:
        v = voice_library.update_voice(vid, transcript=body.get("transcript"),
                                       language=body.get("language"), name=body.get("name"))
        return {"ok": True, "voice": v}
    except KeyError:
        return _err(404, f"no voice '{vid}'")


@app.delete("/api/voices/library/{vid}")
def api_library_delete(vid: str):
    if not voice_library.delete_voice(vid):
        return _err(404, f"no voice '{vid}'")
    return {"ok": True}


# ------------------------------------------------------------------ models
@app.get("/api/models")
def api_models():
    return tts_engines.catalog()


@app.post("/api/models/download")
async def api_models_download(req: Request):
    body = await req.json()
    res = tts_engines.start_download((body.get("engine") or "").strip(), (body.get("id") or "").strip())
    return JSONResponse(status_code=200 if res.get("ok") else 409, content=res)


@app.get("/api/models/download/status")
def api_models_download_status():
    return tts_engines.download_status()


@app.post("/api/models/download/cancel")
def api_models_download_cancel():
    return tts_engines.cancel_download()


# Older Orpheus-only routes (the first version of the llama.cpp UI used them).
@app.get("/api/orpheus/models")
def api_orpheus_models():
    cat = tts_engines.catalog()
    e = next(x for x in cat["engines"] if x["engine"] == "orpheus")
    return {"local_dir": str(tts_engines.OrpheusEngine.GGUF_DIR),
            "quants": [{"quant": i["id"], "size_mb": i["size_mb"], "installed": i["installed"],
                        "path": i["dest"], "recommended": i["recommended"]}
                       for i in e["items"] if i["id"] != "snac"]}


@app.post("/api/orpheus/download")
async def api_orpheus_download(req: Request):
    body = await req.json()
    res = tts_engines.start_download("orpheus", (body.get("quant") or "").strip())
    return JSONResponse(status_code=200 if res.get("ok") else 409, content=res)


@app.get("/api/orpheus/download/status")
def api_orpheus_download_status():
    s = tts_engines.download_status()
    return {**s, "quant": s.get("id", "")}


# -------------------------------------------------------------------- stt
@app.get("/api/stt/status")
def api_stt_status():
    return stt.describe()


async def _transcribe_upload(file: UploadFile, language: str):
    ok, why = stt.available()
    if not ok:
        return _err(503, why)
    suffix = Path(file.filename or "clip.webm").suffix or ".webm"
    tmp = Path(tempfile.gettempdir()) / f"stt-{uuid.uuid4().hex}{suffix}"
    try:
        tmp.write_bytes(await file.read())
        return await _run(stt.transcribe, str(tmp), language)
    except Exception as e:
        return _err(500, str(e))
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/api/stt/transcribe")
async def api_stt(file: UploadFile = File(...), language: str = Form("")):
    return await _transcribe_upload(file, language)


@app.post("/v1/audio/transcriptions")
async def api_transcriptions(file: UploadFile = File(...), language: str = Form(""),
                             model: str = Form(""), response_format: str = Form("json")):
    res = await _transcribe_upload(file, language)
    if isinstance(res, JSONResponse) or response_format != "text":
        return res
    return Response(content=res.get("text", ""), media_type="text/plain")


# ----------------------------------------------------------------- control
@app.post("/api/shutdown")
async def api_shutdown():
    """Stop the server (used by managers that would otherwise have to kill it)."""
    global _shutdown_done
    if not _shutdown_done:
        _shutdown_done = True

        def _die():
            try:
                tts_engines.unload_all()
            except Exception:
                pass
            os._exit(0)
        threading.Thread(target=_die, daemon=True).start()
    return {"ok": True}


# ----------------------------------------------------------------- static
# The root is a page about this server: what it is, how the llama.cpp UI uses
# it, where the models go. no-cache means "revalidate every time" (cheap 304s),
# so an edited page is never served stale from heuristic browser caching.
_PAGE_HEADERS = {"Cache-Control": "no-cache, must-revalidate"}


@app.get("/")
def index():
    return FileResponse(ROOT / "speech.html", headers=_PAGE_HEADERS)


@app.get("/speech")
def speech_page():
    """Alias of the root page (kept for anything already linked to it)."""
    return FileResponse(ROOT / "speech.html", headers=_PAGE_HEADERS)


# ------------------------------------------------------------------ setup
def report_voices() -> None:
    """Print what can actually be spoken, and what only looks available.

    A voice listed but not on disk is worse than one that is absent: it
    synthesises as Piper and the user hears the same default voice for every
    name they pick.
    """
    print("[Setup] engines:", flush=True)
    for e in tts_engines.ENGINES.values():
        ok, why = e.available()
        if not ok:
            print(f"    {e.label:24} not installed ({why})")
            continue
        try:
            status = e.voice_status()
        except Exception as ex:
            print(f"    {e.label:24} voice list unavailable ({ex})")
            continue
        have = [v for v, k in status.items() if k]
        missing = [v for v, k in status.items() if not k]
        models = e.models()
        print(f"    {e.label:24} {len(have):3d} voices" +
              (f", {len(models)} model(s)" if models else "") +
              (f", {len(missing)} not downloaded" if missing else ""))
    print(f"[Setup] your voices: {len(voice_library.list_voices())} in {voice_library.LIB}")
    if tts_engines.OFFLINE:
        print("[Setup] TTS_OFFLINE=1: no implicit Hub calls. Downloads you start in "
              "Settings -> Voice -> Models still work.", flush=True)


def lan_address() -> str:
    """Best-effort outward-facing IPv4, purely so we can print a usable URL."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect() transmits nothing -- it only asks the routing table
        # which local interface would be used. TEST-NET-1 (RFC 5737) is
        # reserved and unroutable, so this cannot reach anything even in
        # principle; it just has to be off-link to select the default route.
        s.connect(("192.0.2.1", 80))
        return s.getsockname()[0]
    except Exception:
        return "<this-pc-ip>"
    finally:
        s.close()


def _listen_socket(host: str, port: int):
    """A listening socket that accepts IPv6 *and* IPv4 when serving the LAN.

    uvicorn's host="0.0.0.0" binds IPv4 only. A Windows PC publishes AAAA
    records for its own name, and browsers prefer IPv6, so http://<pcname>:port
    resolves to an IPv6 address and the connection is refused -- while the
    literal IPv4 address works. Binding "::" with IPV6_V6ONLY cleared accepts
    both, so every form of the address behaves the same.

    Returns None when the caller should let uvicorn bind normally.
    """
    if host not in ("0.0.0.0", "::"):
        return None
    if not socket.has_dualstack_ipv6():
        print("[warn] no dual-stack support: IPv6 clients (often the PC's own "
              "hostname) will not connect; use the IPv4 address")
        return None
    return socket.create_server(("::", port), family=socket.AF_INET6,
                                dualstack_ipv6=True, backlog=2048)


def _port_taken(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


class QuietServer(uvicorn.Server):
    """uvicorn, minus the connection-reset traceback spam.

    Pointing a browser at http:// when the server is serving TLS sends a
    plaintext request into a TLS listener. The handshake cannot parse it and
    the socket is torn down, which asyncio's Windows proactor reports as an
    unhandled ConnectionResetError (WinError 10054): a six-line traceback per
    attempt that looks like a crash and never mentions the actual cause.

    The connection is dead either way. The only thing worth printing is what
    the person did wrong, once.
    """

    hint = ""

    async def serve(self, sockets=None):
        import asyncio
        warned = [False]

        def handler(loop, ctx):
            if isinstance(ctx.get("exception"),
                          (ConnectionResetError, ConnectionAbortedError)):
                if self.hint and not warned[0]:
                    warned[0] = True
                    print(self.hint, flush=True)
                return
            loop.default_exception_handler(ctx)

        asyncio.get_running_loop().set_exception_handler(handler)
        return await super().serve(sockets=sockets)


def main() -> None:
    global _port
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=None,
                    help="bind address (default 127.0.0.1, or 0.0.0.0 with --lan)")
    ap.add_argument("--lan", action="store_true",
                    help="serve to the local network so phones/tablets can connect")
    # TLS is on by default: browsers only expose the microphone on a secure
    # origin, so plain http silently kills voice input everywhere except
    # localhost. --no-https is there for the case where a client cannot cope
    # with a self-signed certificate (and for the llama.cpp UI on plain http,
    # which cannot call an https server with an untrusted certificate).
    ap.add_argument("--https", action="store_true", default=True,
                    help="serve TLS with a self-signed certificate, generated "
                         "on first use (default)")
    ap.add_argument("--no-https", dest="https", action="store_false",
                    help="serve plain http instead")
    ap.add_argument("--cert", default=None, help="PEM certificate (implies --https)")
    ap.add_argument("--key", default=None, help="PEM private key (implies --https)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("TTS_PORT", DEFAULT_PORT)))
    ap.add_argument("--engine", default=os.environ.get("TTS_ENGINE", "piper"),
                    choices=sorted(tts_engines.ENGINES),
                    help="engine for requests that name none (the llama.cpp UI always names one)")
    ap.add_argument("--voice", default="en_US-lessac-medium", help="Piper voice to warm up")
    ap.add_argument("--setup-only", action="store_true")
    args = ap.parse_args()
    tts_engines.DEFAULT_ENGINE = args.engine
    _port = args.port
    # Explicit --host always wins; --lan only supplies the default.
    host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")

    # Warm Piper -- it is the fallback when a GPU engine errors. A missing
    # voice must not stop the server: everything else still works.
    try:
        tts_engines.get("piper").synth_wav("ready", args.voice)
    except Exception as e:
        print(f"[warn] Piper warm-up failed ({e}); download a Piper voice in "
              "Settings -> Voice -> Models", flush=True)
    report_voices()
    if args.setup_only:
        print("[OK] setup done")
        return

    if _port_taken(args.port):
        print(f"[ERROR] port {args.port} is already in use. Another speech server "
              f"(or the llama.cpp-server-tts app) may be running; pick another with "
              f"--port.", flush=True)
        sys.exit(2)

    ssl_kw: dict = {}
    scheme = "http"
    if args.https or args.cert or args.key:
        if args.cert and args.key:
            cert, key = Path(args.cert), Path(args.key)
            note = "supplied"
        else:
            import certs_util
            cert, key, note = certs_util.ensure_cert()
        ssl_kw = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        scheme = "https"
        print(f"[OK] TLS certificate {note}: {cert}", flush=True)

    if host in ("0.0.0.0", "::"):
        print(f"[OK] serving to the LAN on {scheme}://{lan_address()}:{args.port}", flush=True)
        print(f"     this PC:   {scheme}://127.0.0.1:{args.port}", flush=True)
        print(f"     by name:   {scheme}://{socket.gethostname().lower()}:{args.port}", flush=True)
    else:
        print(f"[OK] serving on {scheme}://{host}:{args.port}", flush=True)
    if scheme == "https":
        print("     the certificate is self-signed, so every browser shows a "
              "warning once -- accept it and the mic will work off-localhost")

    config = uvicorn.Config(app, host=host, port=args.port,
                            log_level="warning", **ssl_kw)
    server = QuietServer(config)
    if scheme == "https":
        server.hint = (f"[hint] a client connected without TLS -- this server is "
                       f"https only. Use https://127.0.0.1:{args.port}, "
                       f"not http://")
    sock = _listen_socket(host, args.port)
    try:
        server.run(sockets=[sock] if sock else None)
    finally:
        tts_engines.unload_all()


if __name__ == "__main__":
    main()
