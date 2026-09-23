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
tools\asr-server\run.bat --port 8178 --model large-v3 --compute-type float16 --preload
```

```bash
# Linux / Git Bash
bash tools/asr-server/run.sh --port 8178 --model large-v3 --compute-type float16 --preload
```

`--setup-only` does the venv/requirements steps and exits without starting the server.

## Options (passed to `asr_server.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--port` | `8178` | listen port |
| `--model` | `large-v3` | faster-whisper model name or a local CTranslate2 model directory |
| `--device` | `auto` | `auto`: the GPU when the model fits with `--vram-margin` GB to spare, otherwise (or when the GPU load fails) the CPU; checked again on every (re)load. `cuda` / `cpu` force one |
| `--compute-type` | `float16` | CTranslate2 compute type; `float16` (~3 GB VRAM) is more accurate than `int8_float16` (~1.6 GB). The CPU has no float16, so there it becomes full-precision `float32` (not lossy int8) |
| `--vram-margin GB` | `2` | VRAM that must stay free besides the model for `--device auto` to pick the GPU, so a llama-server sharing the card is not squeezed |
| `--idle-unload N` | `0` | release the model from VRAM after N idle seconds (0 = keep resident) |
| `--preload` | off | load the model at startup |
| `--models-dir DIR` | `./models` | where models are downloaded to / looked up in (env `ASR_MODELS_DIR`); point it at an existing download to avoid fetching the model again |

## Startup output

The launcher and server report each step, so a console (or a GUI reading the output) shows
where startup is:

```text
[Setup] venv and requirements OK.
[Start] asr_server.py --port 8178 --model large-v3 --compute-type float16 --preload
... INFO [asr] starting: model=large-v3 device=auto compute=float16 port=8178 models-dir=...
... INFO [asr] downloading large-v3 (~3 GB) from Systran/faster-whisper-large-v3 ...   (first run only, with progress bar)
... INFO [asr] loading large-v3 on cuda (float16) ...
... INFO [asr] loaded large-v3 on cuda (float16) in 4.2s
... INFO [asr] READY - listening on http://127.0.0.1:8178 (model loaded)
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
llama-server -m model.gguf --mmproj mmproj.gguf --asr-url http://127.0.0.1:8178
```
