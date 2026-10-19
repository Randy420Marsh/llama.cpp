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
run.bat --lan                     :: Windows
```
```bash
./run.sh --lan                    # Linux / macOS
```

The first run creates `venv\` and installs the small base requirements (Piper,
FastAPI, faster-whisper). Then open the llama.cpp web UI → **Settings →
Voice**: it finds this server on the same host, port 8179, and lists what can
be spoken. Download the Qwen3-TTS model there (2.5 GB, one click) and create a
voice under **Your voices**.

The server speaks https with the same certificate as `llama-server`
(`--ssl-cert-file` / `--ssl-key-file`, or the `LLAMA_ARG_SSL_CERT_FILE` /
`LLAMA_ARG_SSL_KEY_FILE` environment variables; `run.bat` / `run.sh` default to
`%AI_ROOT%\ssl\cert.pem` and `key.pem` when they exist, where AI_ROOT is `C:\AI` on Windows or `$HOME/AI` on Linux). The web UI calls the
speech server with its own scheme, so an https `llama-server` needs an https
speech server, and one trusted certificate covers both. Without a certificate
it falls back to a self-signed one, which the browser must accept once at
`https://<pc>:8179`. `--no-https` serves plain http, only for a web UI that is
itself plain http. Without `--lan` the server only answers on this PC.

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
model clones zero-shot -- from the clip and its transcript, no fine-tuning -- but
is the weakest of the two here (one of two voices missed entirely); it stays
available but is marked experimental. Use the llama.cpp engine for your own
voices.

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
to `server.py`, or `%AI_ROOT%\tts-models` when that exists (AI_ROOT: `C:\AI` on Windows, `$HOME/AI` on Linux; override with the `AI_ROOT` / `TTS_CACHE` variables). Settings → Voice →
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

Both children listen on 127.0.0.1 only and use https with this server's
certificate whenever it has one (`--cert`/`--key` or the `LLAMA_ARG_SSL_*`
variables); the self-signed fallback counts too. They are started with an
environment stripped of every `LLAMA_ARG_*` variable, so a shell set up for the
chat server cannot hand them its model, draft model, mmproj or host. To use an
Orpheus server you run yourself, set `ORPHEUS_SERVER` to its full URL, scheme
included (`https://127.0.0.1:8180`).

---

## 7. Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TTS_CACHE` | `models\` next to server.py (or `%AI_ROOT%\tts-models` if present) | every model and voice |
| `AI_ROOT` | `C:\AI` (Windows) / `$HOME/AI` (Linux) | the shared AI folder the defaults live under |
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
| `ORPHEUS_GGUF_PORT` / `ORPHEUS_SERVER` | `8180` / — | its llama-server port / use one you run yourself (full URL) |
| `ORPHEUS_SNAC_DEVICE` | `auto` | where Orpheus' SNAC decoder runs: `cuda`, `cpu`, or `auto` = CPU when under 8 GB VRAM is free at load |
| `ASR_URL` / `STT_USE_SERVER` | `https://127.0.0.1:8178` / `auto` | the ASR server / `off` to always use the local model |
| `STT_MODEL` / `STT_DEVICE` | `small.en` / auto | local Whisper fallback |
| `HF_TOKEN` | — | only for gated Hugging Face repositories |

`server.py` flags: `--host`, `--lan`, `--port`, `--https` (default) / `--no-https`,
`--ssl-cert-file` (`--cert`), `--ssl-key-file` (`--key`), `--engine`, `--setup-only`.

---

## 8. HTTP API

| Method | Path | |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/api/status` | engines, speech-to-text, folders, port |
| POST | `/v1/audio/speech` | `{input, engine, voice, model, language, speed, emotions}` → WAV. `emotions` (`"auto"` default / `"off"`): for engines that understand emotion tags, insert the `<laugh>`, `<sigh>` ... the text implies (below). Headers `X-TTS-Engine`, `X-TTS-Voice`, `X-TTS-Speed` (`native`: the engine applied the speed; `client`: the player should), `X-TTS-Emotions` (what was applied), `X-TTS-Fallback` + `X-TTS-Error` when Piper had to step in |
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

## 8a. Emotion tags (Orpheus)

The Orpheus finetuned model speaks paralinguistic tags in the text -- `<laugh>`,
`<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>` --
which makes it the natural engine for roleplay. Roleplay text rarely carries the
tags, so the server adds them on request:

- **Automatic (the default).** `POST /v1/audio/speech` accepts `"emotions"`;
  `"auto"` (or omitted) runs [emotion_tags.py](emotion_tags.py) over the text
  first: it scans each sentence for cues -- `sighed`, `hehe`, `laughed`, `ugh`,
  `(coughs)`, `whoa`, ... -- and inserts one tag per sentence, before the cue
  word, so the model *performs* the line. Sentences already tagged by hand are
  left alone. `"off"` sends the text unchanged.
- **Manual.** any client (the web UI, llama.cpp's chat page, your own program)
  can just write the tags into the text itself; the model reads them.

```bash
curl -s localhost:8179/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input":"He sighed. \"Again.\"","engine":"orpheus","voice":"tara","emotions":"auto"}' \
  -o reply.wav
```

The web UI's read-aloud sends `"emotions": "auto"` too (see
`tools/ui/src/lib/utils/tts.ts`). Only the Orpheus engine understands the tags;
other engines receive them as ordinary text, which is why the switch is per
request rather than global.

## 8b. Orpheus: zero-shot cloning, fine-tuning, and other local options

The two Orpheus models and where they come from ([canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS),
[canopylabs/orpheus-3b-0.1-ft](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)):

- **Finetuned (`-ft`)** -- 8 preset voices (`tara`, `leah`, `jess`, `leo`, `dan`,
  `mia`, `zac`, `zoe`, each language has its own set) and the emotion tags
  above. This is the one this server's Orpheus engine speaks by default.
- **Pretrained** -- zero-shot cloning: conditioned on one or more
  text-speech pairs in the prompt, it continues in that voice without any
  fine-tuning (the more pairs, the more reliably it holds the voice). This
  server's "your voices" path uses it, one finished example turn per voice.

Everything can stay fully local:

- **llama.cpp (this server's default):** an Orpheus GGUF behind a
  `llama-server` (section 4 lists the file names) plus the SNAC codec -- no
  Python model server, the whole thing runs off the GGUF you already have.
- **`orpheus-speech` (vLLM):** the upstream package, faster streaming on big
  cards: `pip install orpheus-speech`, then `OrpheusModel(model_name=
  "canopylabs/orpheus-tts-0.1-finetune-prod").generate_speech(prompt=...,
  voice="tara")` (24 kHz WAV; if the KV-cache error or missing
  `max_model_len` shows up, point `sys.path` at the repo's `orpheus_tts_pypi`).
  `repetition_penalty >= 1.1` is required for stable output either way.
- **No-GPU inference with llama.cpp** is described in the upstream repo's docs;
  this server is that path, already wired up.

**Fine-tuning (unsloth):** a text-only SFT on sequences of length 8192 in the
same dataset format as pretraining; the upstream notes recommend starting from
the finetuned model (not the pretrained one) for a new language or style, and
avoid synthetic training audio (it maps to the same tokens and uses the
codebook poorly). The `canopylabs/orpheus-3b-0.1-ft` model is the fine-tuned
release.

To train one locally on your own recordings, use the two scripts in
[setup/](setup/): `make_orpheus_dataset.py` turns a folder of clips (+
transcripts) into the Hugging Face dataset the model trains on (the authors
report ~50 clips for a usable result, ~300 per speaker for a strong one), and
`train_orpheus.py` runs the Unsloth LoRA training on a CUDA GPU and prints
the GGUF commands to follow. The full guide, with data recommendations and
how the result plugs back into this server, is
[wiki/Voice.md → Training your own Orpheus voice](../../wiki/Voice.md).

**Transcribing your clips:** the training data needs a `text` per clip. If
you recorded your own clips without writing them down, transcribe them with
local **Whisper large-v3** (the most accurate model on this box, the same one
the ASR server serves) — it writes a sidecar `.txt` next to each clip (what
`make_orpheus_dataset.py` pairs by) plus a `transcripts.txt`:

    tools/asr-server/venv/bin/python tools/asr-server/transcribe_orpheus_clips.py clips/ \
        --language en --combined clips/transcripts.txt

(Use `--model large-v3-turbo` for ~4× faster but slightly less accurate;
`--server http://127.0.0.1:8178` to reuse a running ASR server;
`--skip-existing` to fill only the gaps.) Then **review the `.txt` files**
before training — a wrong transcript becomes the voice's target, and 10
seconds of listening per clip beats a bad one.

**How much data:** the Orpheus-TTS authors report **~50 clips for a usable
result, ~300 per speaker for a strong one**; **6–15 s of natural speech per
clip**, one speaker, quiet room, varied content (reading/talking/emotions) so
it learns the *voice* not one phrase, and avoid synthetic/TTS audio.

**GUI:** [orpheus/app.py](orpheus/app.py) is a Gradio studio that wraps this
whole flow in one window — *Dataset* (transcribe the clips with Whisper, then
build the `.jsonl`), *Train* (a plain-PyTorch LoRA pass in
[orpheus/train_lora.py](orpheus/train_lora.py), no HF `Trainer`/`datasets`
so it sidesteps the pyarrow crash on the torch nightly), *Publish* (adapter →
GGUF via the repo's `convert_lora_to_gguf.py`, pointed at the cached base
config so the gated model stays offline), and *Synthesize* (zero-shot clone +
the adapter through the base model, in
[orpheus/test_synthesis.py](orpheus/test_synthesis.py)).
Start it with `start-orpheus-studio.sh` (or `.bat`); the GUI runs in a
gradio-only env and shells the heavy work out to the torch venv.

The studio's **Trim & Export** tab turns videos or audio into training-ready
clips: drop files (mp4/mkv/mov/avi/webm/ts or wav/mp3/flac/…), pick a voice
model, and toggle the **trim tool** — off exports the whole clip, on queues
it for one-by-one trimming. For each queued item you get a synced video
preview (native player) plus its extracted audio track, a waveform with a
green A–B selection and a clip-length hint against the 6–15 s recommendation,
and A/B fields. **Trim & export** cuts `[A, B)` into the voice's folder as a
`NNNN.wav` — 24 kHz mono 16-bit, normalized to **-23 LUFS** (2-pass ffmpeg
`loudnorm`, the SNAC codec's rate) — and keeps the item loaded so you can cut
several clips from one long TV/movie scene; **Next item** marks it processed
and advances, and the queue table tracks how many clips each source produced.
Needs `ffmpeg`/`ffprobe` on PATH (the `loudnorm` filter is built in). The
engine lives in [orpheus/trimtools.py](orpheus/trimtools.py).

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

**Orpheus plays slower than real time, or slowed down.** Orpheus itself runs
at about 3× real time on an RTX 5090 (Q8_0), and still about 2× while the
chat model generates at the same time. When it falls below real time, the card
is out of VRAM and Windows is paging into system memory (sysmem fallback),
which slows every model on the card. Measured next to a 27B chat model
(23.9 GB): the Orpheus child adds 4.8 GB at Q8_0, 4.0 GB at Q6_K and 3.4 GB at
Q4_K_M, and SNAC on the GPU 1.2 GB. With the desktop that is ~30.6 of 32 GB
before the ASR server or a browser. Fixes, in order:
`ORPHEUS_SNAC_DEVICE=auto` (the default) already moves SNAC to the CPU on a
full card, at ~0.4 s per 10 s of speech. Pick Q6_K (0.8 GB smaller and slightly
faster), or turn off NVIDIA's *CUDA – Sysmem Fallback
Policy* for llama-server so the problem shows up as an out-of-memory error
instead of a slowdown. The speed slider is applied by the player (Orpheus has
no native speed), so 1.3 plays faster, not slower.

Each request retries once when the model hits its token limit without ending
the speech. A frame with an out-of-range code is dropped alone, so the rest
of the sentence still plays. The first SNAC decode is done at load, so the
first sentence does not pay CUDA's ~4 s kernel set-up.

**A GPU engine is "not installed".** Run `run-tts-setup.bat` / `.sh`. It
installs torch from the CUDA 12.8 index first: `chatterbox-tts` pins an old
CPU torch and is therefore installed with `--no-deps`. Never install it
plainly into this venv.

**Read aloud uses Piper instead of the chosen voice.** The engine failed for
that sentence; the toast and the `X-TTS-Error` header say why (model missing,
VRAM full, unknown voice). The server log has the details.
