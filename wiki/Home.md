# llama.cpp fork: video, audio and voice

This fork (`Randy420Marsh/llama.cpp`) tracks upstream `ggml-org/llama.cpp`. It adds long-video and audio
input to `llama-server`, and a web UI that reads replies aloud (also in your own, zero-shot cloned voice) and
takes dictation. Those changes stay in the fork; everything else behaves as upstream does.

## Pages

- [Building](Building): Windows (CUDA, OpenSSL) and Ubuntu builds, and what the video/audio features need.
- [Video and audio](Video-and-Audio): user guide covering the request fields, server flags, budgets,
  seeking, transcripts, tested models and errors.
- [Voice](Voice): read aloud, your own voices, dictation; `llama-tts-server`, the speech server, where the
  models go.
- [Media pipeline internals](Media-Pipeline-Internals): how a video part becomes model input, the design
  choices behind it, and the changed files.
- [Model launch configs](Model-Launch-Configs): the exact `llama-server` command per verified model
  (paths as variables), and the ASR server start line.

## What the fork adds

| Area | Addition |
|---|---|
| Requests | `video_url` / `input_video` and `audio_url` / `input_audio` parts take time ranges (`start`, `end`, `duration`, `segments`), `fps`, `detail`, a per-part token budget, an `audio` mode and a `language` hint |
| Sources | HTTP(S) URLs are streamed with range requests, so only the requested parts of a long video are downloaded. `file://` paths under `--media-path` and inline base64 also work |
| Budget | The frame rate drops to fit `--video-max-tokens` / `--video-max-frames`, and resolution stays at the chosen detail level. The 1080p 60 s test video costs 35k tokens instead of 247k |
| Readability | The detail presets (`low` / `standard` / `high` / `max`) are calibrated on real screenshots. `standard` keeps 20 px text on a 1080p source readable |
| Dedup | Frame pairs in which no screen region changed are dropped (`--video-dedup`) |
| Timestamps | Frames and transcript lines carry source time: a range 5 h into a file is described at 18000 s |
| Audio | Native audio for models with an audio encoder (Gemma4-12B, Nemotron-Omni). Other models get timestamped transcripts from an OpenAI-compatible speech-to-text server (`--asr-url`) |
| Safety | Only plain container formats are opened. Playlist, concat and image-sequence inputs, which can reach other files, are refused |
| Speech out | `llama-tts-server` keeps Qwen3-TTS / Pocket-TTS loaded behind `/v1/audio/speech`; any clip in its voices folder is a voice (zero-shot cloning, ~5× realtime) |
| Web UI voice | Settings → Voice lists engines, voices and models from the speech server, records your own voice, downloads models; 🔊 reads a reply sentence by sentence |
| Dictation | the 🎤 button types what you say (or sends the recording, never both); Settings → Microphone |
| Reasoning | `GET /props` reports `reasoning_efforts` (the levels the loaded model offers, for a client picker); `reasoning_effort` is a per-request field; `--reasoning-preserve` keeps the trace across turns |

All media stays in memory. ffmpeg runs over pipes, the prepared clips are in-memory FFV1/Matroska, and
audio goes to the transcription server as an in-memory WAV. No temp files are written.

## Models

The features are verified on the models in [Model-Launch-Configs](Model-Launch-Configs), the exact
`llama-server` command for each (paths as variables): **Huihui-Qwen3.8-27B** (abliterated, NVFP4;
native video, vision, reasoning levels, MTP speculative decoding - 20/20 on the capability matrix),
Gemma4-12B (native audio + vision, MTP), Gemma-4-26B-A4B (vision), and Nemotron-3-Nano-Omni-30B-A3B
(vision; combined audio+vision via a converted mmproj). MTP speculative decoding
(`--spec-type draft-mtp`) and the multi-model router (`--models-preset`) are upstream capabilities
that the launch configs use together with the media flags.

## Quick start

```bash
# 0. one certificate for both servers (https; its CA must be in the system trust store)
export LLAMA_ARG_SSL_CERT_FILE=/etc/ssl/llama/cert.pem LLAMA_ARG_SSL_KEY_FILE=/etc/ssl/llama/key.pem

# 1. start the speech-to-text server (tools/asr-server, faster-whisper; see Video-and-Audio section 7)
bash tools/asr-server/run.sh --port 8178 --model large-v3 --compute-type float16 --preload

# 2. start llama-server
llama-server -m Qwen3.8-27B.gguf --mmproj mmproj.gguf -c 160000 --jinja \
  --media-path /data/videos/ --asr-url https://127.0.0.1:8178
```

```json
{"role": "user", "content": [
  {"type": "video_url", "video_url": {"url": "file://talk.mp4", "start": "01:00:00", "end": "01:05:00"}},
  {"type": "text", "text": "Summarize this part and quote what is said about the budget."}
]}
```

The flag table in [tools/server/README.md](https://github.com/Randy420Marsh/llama.cpp/blob/master/tools/server/README.md)
lists every option with its default. `llama-server --help` shows the same.
