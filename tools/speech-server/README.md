# Speech server

Text-to-speech, your own (cloned) voices, model downloads and speech-to-text
for the llama.cpp web UI. The UI's **Settings → Voice** and **Settings →
Microphone** pages talk to this server; any other program can use the same
HTTP API.

| | |
|---|---|
| Port | **8179** (the speech-to-text server is 8178) |
| Engines | llama.cpp **Qwen3-TTS** (clones voices), Piper, Kokoro-82M, Chatterbox, Orpheus 3B |
| Voice cloning | zero-shot: record or upload 6–15 s of a voice, no training |
| Speech-to-text | the ASR server (`tools/asr-server`, Whisper large-v3) when it runs, else a small local Whisper |

---

## 1. Quick start

Requirements: [uv](https://docs.astral.sh/uv/) on `PATH`, Python 3.11 (uv
fetches it), and a llama.cpp build that has `llama-tts-server` (this fork:
`cmake --build build --config Release --target llama-tts-server`).

```bat
run.bat --no-https --lan          :: Windows
```
```bash
./run.sh --no-https --lan         # Linux / macOS
```

The first run creates `venv\` and installs the small base requirements (Piper,
FastAPI, faster-whisper). Then open the llama.cpp web UI → **Settings →
Voice**: it finds this server on the same host, port 8179, and lists what can
be spoken. Download the Qwen3-TTS model there (2.5 GB, one click) and create a
voice under **Your voices**.

`--no-https` because the llama.cpp web UI is usually served over plain http,
and a page cannot call an https server whose certificate it does not trust.
Without `--lan` the server only answers on this PC.

GPU engines that need PyTorch (Kokoro, Chatterbox, Orpheus) are optional:

```bat
run-tts-setup.bat                 :: torch (CUDA 12.8) + engine packages, ~4 GB
```

The llama.cpp engine needs none of that: it runs `llama-tts-server`.

---

## 2. Engines

| Engine | Runs on | Voices | Speed | Notes |
|---|---|---|---|---|
| **llama.cpp (Qwen3-TTS)** | GPU via llama.cpp | default + **your voices** | ~5× realtime on an RTX 5090 | 10 languages (en zh de it pt es ja ko fr ru); ~4.5–5.5 GB VRAM while loaded |
| Piper | CPU | 12 downloadable presets | instant | robotic; the fallback when another engine fails |
| Kokoro-82M | GPU (torch) | 54 presets | 20×+ realtime | needs `run-tts-setup` |
| Chatterbox | GPU (torch) | default + **your voices** | ~2× realtime | needs `run-tts-setup` |
| Orpheus 3B | GPU via llama.cpp + SNAC (torch) | 8 presets; your voices *experimental* | ~3× realtime (Q8_0) | needs `run-tts-setup` |

Only one GPU engine is loaded at a time, and a GPU engine that has not spoken
for 10 minutes unloads itself (`TTS_IDLE_UNLOAD`), so the chat model keeps the
card. **Settings → Voice → Models → Unload** frees it at once.

### Voice cloning, measured

Two reference voices, one 7–8 s clip each, the same sentence spoken by each
engine; similarity of the result to each reference, measured with a speaker
encoder that took no part in the generation (1.0 = identical):

| Engine | clone of voice A → A / B | clone of voice B → A / B |
|---|---|---|
| llama.cpp Qwen3-TTS | **0.925** / 0.628 | 0.558 / **0.946** |
| Orpheus pretrained | 0.551 / 0.681 (missed) | 0.630 / 0.753 |

Whisper transcribed every Qwen3-TTS output word for word. Orpheus' pretrained
model was never trained to clone (its authors say more examples help); it
stays available but is marked experimental. Use the llama.cpp engine.

---

## 3. Your voices

A voice is one recording of the speaker, 3–30 s, ideally 6–15 s of clear
speech, plus what is said in it. Nothing is trained: the cloning engines read
the clip every time they speak.

* **Record**: Settings → Voice → Your voices → *Record*, read the passage on
  screen, *Stop*, name it, *Save voice*.
* **Upload**: *Upload a clip*: WAV, MP3, FLAC, OGG, WebM, M4A (anything ffmpeg
  reads).

On save the clip is converted to 24 kHz mono, silence is trimmed from both ends,
loudness is levelled, and it is transcribed if you left the transcript empty.
The transcript only matters to Orpheus.

On disk, one folder per voice:

```
<voices>/<id>/reference.wav     24 kHz mono 16-bit
              transcript.txt    what is said in the clip (optional)
              voice.json        {"name", "language", "created", "duration_s"}
```

Putting a folder there by hand works the same (a bare `<id>.wav` too). The same
folder is `llama-tts-server`'s `--tts-voices-dir`, so a voice is simply a
folder name.

---

## 4. Where the models go

Everything lives under one cache folder, `TTS_CACHE`. Default: `models\` next
to `server.py`, or `C:\AI\tts-models` when that exists. Settings → Voice →
Models shows the folder of every engine and downloads into it; files you copy
in by hand are found the same way (press *Refresh*).

| What | Folder | Files |
|---|---|---|
| llama.cpp speech models | `<cache>/gguf/qwen3-tts/` | `<name>-<quant>.gguf` **and** `mmproj-<name>-<quant>.gguf` (a pair with the same `<name>`) |
| Orpheus | `<cache>/gguf/orpheus/` | `orpheus-3b-0.1-ft-<quant>.gguf` (preset voices), `orpheus-3b-0.1-pretrained-<quant>.gguf` (your voices) |
| Piper | `<cache>/piper/` | `<voice>.onnx` + `<voice>.onnx.json` |
| Kokoro, Chatterbox, SNAC | `<cache>/hf/` | Hugging Face cache, filled by the download buttons |
| Your voices | `<cache>/voices/` (or `TTS_VOICES_DIR`) | see §3 |
| Local Whisper | `<cache>/whisper/` | faster-whisper models |

Download sources:

| Model | From |
|---|---|
| Qwen3-TTS 1.7B (Q8_0, Q4_K_M, bf16 + mmproj) | `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF` |
| Orpheus finetuned | `unsloth/orpheus-3b-0.1-ft-GGUF` (community GGUF of `canopylabs/orpheus-3b-0.1-ft`) |
| Orpheus pretrained | `HorizonNexusAI/orpheus-3b-0.1-pretrained-GGUF` (community GGUF of `canopylabs/orpheus-3b-0.1-pretrained`) |
| Piper | `rhasspy/piper-voices` |
| Kokoro / Chatterbox / SNAC | `hexgrad/Kokoro-82M`, `ResembleAI/chatterbox`, `hubertsiuzdak/snac_24khz` |

Licences: Qwen3-TTS, Orpheus and Kokoro are Apache-2.0, Chatterbox and SNAC
MIT; each Piper voice has its own (see the `MODEL_CARD` next to it on
Hugging Face). The Orpheus GGUFs are community conversions: the official
repositories publish safetensors only (and are gated behind a click-through),
so the download button names its source and nothing is fetched without that
click.

From the command line, the same catalog:

```bash
python setup/fetch_voices.py --list        # what is on disk, what can be downloaded
python setup/fetch_voices.py               # the recommended model of each usable engine
python setup/fetch_voices.py llamacpp/Qwen3-TTS-12Hz-1.7B-Base-Q8_0 piper/fi_FI-harri-medium
```

---

## 5. Speech-to-text

The mic button and voice creation transcribe through
`POST /api/stt/transcribe`. When the ASR server (Whisper large-v3,
`tools/asr-server`, port 8178) answers, it does the work: multilingual and
accurate, and llama-server's `--asr-url` uses the same one. Otherwise a small
local Whisper (`small.en`, CPU int8 when the GPU is busy) is used. It is
English only, so start the ASR server for other languages.

What a recording becomes is set in **Settings → Microphone → Voice input
becomes**:

| Mode | Result |
|---|---|
| Auto (default) | text in the message box when this server answers, else the audio itself |
| Text | always transcribed here |
| Audio attachment | the recording is sent; a model with an audio encoder hears it, other models get llama-server's `--asr-url` transcript |

A recording becomes text *or* an attachment, never both. Sending both made
the words arrive twice when llama-server also transcribed the attachment.

---

## 6. Ports

| Port | What |
|---|---|
| 8179 | this server |
| 8178 | ASR server (speech-to-text) |
| 8181 | `llama-tts-server`, started on demand for the llama.cpp engine |
| 8180 | `llama-server` with an Orpheus GGUF, started on demand |

The two children never share a port with anything else: on Windows a second
`llama-server` can bind a port that is already in use, and requests then go to
either process. The server refuses to start if 8179 is taken.

---

## 7. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TTS_CACHE` | `models\` next to server.py (or `C:\AI\tts-models` if present) | every model and voice |
| `TTS_VOICES_DIR` | `<cache>/voices` | your voices |
| `TTS_PORT` | `8179` | listening port (`--port` wins) |
| `TTS_ENGINE` | `piper` | engine for requests that name none |
| `TTS_OFFLINE` | `1` | no implicit Hugging Face calls; downloads you start still work |
| `TTS_IDLE_UNLOAD` | `600` | seconds before an idle GPU engine unloads (0 = never) |
| `LLAMA_BIN` | this repo's `build/bin/Release` (Windows) or `build/bin` | where `llama-tts-server` and `llama-server` are |
| `LLAMA_TTS_SERVER_EXE` | `<LLAMA_BIN>/llama-tts-server` | override the binary |
| `LLAMA_TTS_PORT` / `LLAMA_TTS_ARGS` | `8181` / — | port / extra flags for `llama-tts-server` |
| `LLAMA_TTS_MODEL_DIRS` | `<cache>/gguf/qwen3-tts` | more folders with speech GGUFs (`;`-separated on Windows) |
| `ORPHEUS_GGUF_DIR` | `<cache>/gguf/orpheus` | Orpheus quant folders (`;`-separated) |
| `ORPHEUS_GGUF_PORT` / `ORPHEUS_SERVER` | `8180` / — | its llama-server port / use one you run yourself |
| `ASR_URL` / `STT_USE_SERVER` | `http://127.0.0.1:8178` / `auto` | the ASR server / `off` to always use the local model |
| `STT_MODEL` / `STT_DEVICE` | `small.en` / auto | local Whisper fallback |
| `HF_TOKEN` | — | only for gated Hugging Face repositories |

`server.py` flags: `--host`, `--lan`, `--port`, `--no-https` / `--https`,
`--cert`, `--key`, `--engine`, `--setup-only`.

---

## 8. HTTP API

| Method | Path | |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/api/status` | engines, speech-to-text, folders, port |
| POST | `/v1/audio/speech` | `{input, engine, voice, model, language, speed}` → WAV. Headers `X-TTS-Engine`, `X-TTS-Voice`, `X-TTS-Speed` (`native`: the engine applied the speed; `client`: the player should), `X-TTS-Fallback` + `X-TTS-Error` when Piper had to step in |
| GET | `/api/tts/engines` | every engine with its voices and models |
| POST | `/api/tts/select` · `/api/tts/unload` | load one engine · free VRAM (`{}` = all, `{"engine": "stt"}`) |
| GET / POST | `/api/voices/library` | list · create (multipart `name`, `file`, `transcript`, `language`) |
| GET | `/api/voices/library/{id}/audio` | the reference clip |
| PATCH / DELETE | `/api/voices/library/{id}` | edit transcript / language / name · delete |
| GET | `/api/models` | on disk + downloadable, per engine, with folders |
| POST | `/api/models/download` | `{engine, id}`; poll `GET /api/models/download/status`, cancel with `POST …/cancel` |
| POST | `/api/stt/transcribe` | multipart `file` (+ `language`) → `{text, language, duration}` |
| POST | `/v1/audio/transcriptions` | the same, OpenAI-shaped |

```bash
curl -s localhost:8179/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input":"Hello there.","engine":"llamacpp","voice":"my-voice"}' -o hello.wav
```

---

## 9. Troubleshooting

**Settings → Voice says "No speech server".** The server is not running, or
not on this page's host: start it with `--lan` when the UI is opened from
another device, or put its URL in the field.

**The mic button is greyed out on a phone.** Browsers allow the microphone on
`https://` or `localhost` only. Serve the web UI over https (llama-server
`--ssl-key-file/--ssl-cert-file`), then run this server with https too and
accept its certificate once (open `https://<pc>:8179`).

**"port 8179 is already in use".** Another speech server is running (the
command center starts one). Stop it, or use `--port`.

**A GPU engine is "not installed".** Run `run-tts-setup.bat` / `.sh`. It
installs torch from the CUDA 12.8 index first: `chatterbox-tts` pins an old
CPU torch and is therefore installed with `--no-deps`. Never install it
plainly into this venv.

**Read aloud uses Piper instead of the chosen voice.** The engine failed for
that sentence; the toast and the `X-TTS-Error` header say why (model missing,
VRAM full, unknown voice). The server log has the details.
