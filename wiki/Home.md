# llama.cpp fork: video and audio in llama-server

This fork (`Randy420Marsh/llama.cpp`) tracks upstream `ggml-org/llama.cpp`. It adds long-video and audio
input to `llama-server`, and those changes stay in the fork. Everything else behaves as upstream does.

## Pages

- [Building](Building): Windows (CUDA, OpenSSL) and Ubuntu builds, and what the video/audio features need.
- [Video and audio](Video-and-Audio): user guide covering the request fields, server flags, budgets,
  seeking, transcripts, tested models and errors.
- [Media pipeline internals](Media-Pipeline-Internals): how a video part becomes model input, the design
  choices behind it, and the changed files.

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

All media stays in memory. ffmpeg runs over pipes, the prepared clips are in-memory FFV1/Matroska, and
audio goes to the transcription server as an in-memory WAV. No temp files are written.

## Quick start

```bash
# 1. start the speech-to-text server (tools/asr-server, faster-whisper; see Video-and-Audio section 7)
bash tools/asr-server/run.sh --port 8178 --model large-v3 --compute-type float16 --preload

# 2. start llama-server
llama-server -m Qwen3.8-27B.gguf --mmproj mmproj.gguf -c 160000 --jinja \
  --media-path /data/videos/ --asr-url http://127.0.0.1:8178
```

```json
{"role": "user", "content": [
  {"type": "video_url", "video_url": {"url": "file://talk.mp4", "start": "01:00:00", "end": "01:05:00"}},
  {"type": "text", "text": "Summarize this part and quote what is said about the budget."}
]}
```

The flag table in [tools/server/README.md](https://github.com/Randy420Marsh/llama.cpp/blob/master/tools/server/README.md)
lists every option with its default. `llama-server --help` shows the same.
