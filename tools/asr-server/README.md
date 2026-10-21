# asr-server: speech-to-text for llama-server

OpenAI-compatible transcription server (`POST /v1/audio/transcriptions`) backed by
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). `llama-server` uses it behind
`--asr-url` to turn the audio track of videos and standalone audio files into timestamped
transcripts for models without a native audio encoder. See
[wiki/Video-and-Audio.md](../../wiki/Video-and-Audio.md) section 7 for the full guide.

Everything stays in memory: uploaded audio is decoded from RAM, nothing is written to disk.

## Setup

The launcher (`run.bat` on Windows, `run.sh` on Linux / Git Bash) is idempotent:

1. First run: creates a `uv` venv (Python 3.11) in `venv/` and installs `requirements.txt`
   (`faster-whisper`, `fastapi`, `uvicorn`, `python-multipart`, and the CUDA cuBLAS/cuDNN
   wheels on Windows and Linux). Requires [`uv`](https://docs.astral.sh/uv/) on `PATH`.
2. Later runs: skip the install unless `requirements.txt` changed (stamp file in `venv/`).
3. The Whisper model (default `large-v3`, ~3 GB) is downloaded from Hugging Face into
   `models/` on first use, so the first request is slower. `--preload` loads it at startup.

```bat
rem Windows
tools\asr-server\run.bat --port 8178 --model large-v3 --compute-type int8_float16 --preload
```

```bash
# Linux / Git Bash
bash tools/asr-server/run.sh --port 8178 --model large-v3 --compute-type int8_float16 --preload
```

`--setup-only` does the venv/requirements steps and exits without starting the server.

## Options (passed to `asr_server.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--port` | `8178` | listen port |
| `--model` | `large-v3` | faster-whisper model name or a local CTranslate2 model directory |
| `--device` | `auto` | `auto`: the GPU when the model fits with `--vram-margin` GB to spare, otherwise (or when the GPU load fails) the CPU; checked again on every (re)load. `cuda` / `cpu` force one |
| `--compute-type` | `int8_float16` | CTranslate2 compute type; `int8_float16` (int8 weights, ~1.6 GB VRAM) leaves room for the LLM on the card, `float16` (~3 GB) is marginally more accurate. The CPU has no float16, so there the float16 part becomes `float32` |
| `--vram-margin GB` | `2` | VRAM that must stay free besides the model for `--device auto` to pick the GPU, so a llama-server sharing the card is not squeezed |
| `--idle-unload N` | `0` | release the model from VRAM after N idle seconds (0 = keep resident) |
| `--preload` | off | load the model at startup |
| `--models-dir DIR` | `./models` | where models are downloaded to / looked up in (env `ASR_MODELS_DIR`); point it at an existing download to avoid fetching the model again |

## Model choice and optimization

The default is multilingual `large-v3`. From testing on this machine (see
[wiki/Video-and-Audio.md](../../wiki/Video-and-Audio.md) section 8):

- **`large-v3`** is the accuracy reference - multilingual, the most accurate model in this
  family. `large-v3-turbo` was measurably less accurate, and the `distil-*` models are
  English-only, so they are not the right choice for a server that transcribes whatever the
  user says.
- **GPU, and only when it fits.** `--device auto` loads `large-v3` on the GPU only when at
  least `--vram-margin` (default 2 GB) of VRAM stays free afterwards, so a `llama-server` that
  shares the card is not squeezed into a re-quantization or a slowdown; otherwise - or when the
  GPU load fails (missing CUDA libraries, for example) - it runs on the CPU. The check repeats
  on every (re)load, and `GET /health` reports the device and compute type the model actually
  runs on.
- **Precision.** The default `int8_float16` (int8 weights, float16 compute, about 1.6 GB VRAM)
  keeps large-v3's accuracy within a hair of `float16` (about 3 GB) at half the VRAM, so it shares
  the card with a large LLM. On the CPU there is no float16, so the server switches to
  `int8_float32` (or `float32` for the float16 setting), for that model only.
- **VRAM budget for a busy card.** `--idle-unload N` releases the model from VRAM N seconds
  after the last transcription, so a long-running chat server gets the card back for
  generation; `0` keeps it resident for instant transcripts.
- **Transcript quality.** Voice activity detection runs by default and automatically falls back
  to a no-VAD pass when it would remove most of a track that is clearly not silent (so lyrics
  are not dropped); the stock Whisper hallucinations ("Thanks for watching", repeated phrases)
  are filtered out of the segments, and every segment keeps its `start` / `end` time, which is
  what `llama-server` turns into the timestamped transcript.

`GET /health` (`{"state": "loading" | "loaded" | ...}`) and the startup log lines are the
source of truth for what actually happened on a given run.

## Startup output

The launcher and server report each step, so a console (or a GUI reading the output) shows
where startup is:

```text
[Setup] venv and requirements OK.
[Start] asr_server.py --port 8178 --model large-v3 --compute-type int8_float16 --preload
... INFO [asr] starting: model=large-v3 device=auto compute=int8_float16 port=8178 models-dir=...
... INFO [asr] downloading large-v3 (~3 GB) from Systran/faster-whisper-large-v3 ...   (first run only, with progress bar)
... INFO [asr] loading large-v3 on cuda (int8_float16) ...
... INFO [asr] loaded large-v3 on cuda (int8_float16) in 4.2s
... INFO [asr] READY - listening on https://127.0.0.1:8178 (model loaded)
```

A port that is already taken or a model that fails to load (no CUDA libraries, for example)
is reported with a hint and exits with code 1 instead of hanging. With too little free VRAM the
log says so and the model loads on the CPU (float32) instead. `GET /health` also reports
`state` (`not loaded`, `downloading`, `loading`, `loaded` or `error: ...`) and the `device` /
`compute_type` the model actually runs on.

Notes from testing (see the wiki): use multilingual `large-v3`; the `distil-*` models are
English-only and `large-v3-turbo` was less accurate. Voice activity detection is automatic
and falls back to a no-VAD pass when it removes most of a track that is clearly not silent,
so lyrics are not dropped; stock Whisper hallucinations ("Thanks for watching") are filtered.

## Endpoints

- `POST /v1/audio/transcriptions` - multipart: `file`, `model`, `language`, `prompt`,
  `response_format` (`json`, `verbose_json`, `text`), `temperature`, `vad_filter`
  (`auto` / `true` / `false`), `word_timestamps`. `verbose_json` returns per-segment
  `start` / `end` times, which `llama-server` uses for the timestamped transcript.
- `GET /health` - `{"status": "ok", "model": ..., "loaded": ...}`
- `GET /v1/models` - the OpenAI model list, so OpenAI clients work against it.

## Using it with llama-server

```bash
llama-server -m model.gguf --mmproj mmproj.gguf --asr-url https://127.0.0.1:8178
```

The ASR server only serves https: it uses `--ssl-cert-file` / `--ssl-key-file` (or the
`LLAMA_ARG_SSL_CERT_FILE` / `LLAMA_ARG_SSL_KEY_FILE` environment variables, which `run.bat` / `run.sh` default to `%AI_ROOT%\ssl\cert.pem` and `key.pem`; AI_ROOT is `C:\AI` on Windows, `$HOME/AI` on Linux) and exits with an error when there is
no certificate. Clients (`llama-server --asr-url`, the speech server) verify it against the
system trust store, so the certificate's CA must be trusted there. `--no-https` serves plain
http instead.
