# Video and audio in llama-server

`llama-server` in this fork takes video and audio in chat requests, the same way it takes images: from a
URL, a local file or inline base64. This works through the OpenAI-compatible endpoint and through the
built-in web UI. Before the model sees a video, the server cuts it to the requested time ranges, samples
frames and scales them. Long recordings can then be queried by time without filling the context.

- [1. Starting the server](#1-starting-the-server)
- [2. Using it from the web UI](#2-using-it-from-the-web-ui)
- [3. Using it from the OpenAI endpoint](#3-using-it-from-the-openai-endpoint)
- [4. How to ask](#4-how-to-ask)
- [5. Budget, detail and readability](#5-budget-detail-and-readability)
- [6. Timestamps and dedup](#6-timestamps-and-dedup)
- [7. Audio: native or transcript](#7-audio-native-or-transcript)
- [8. Tested and verified models](#8-tested-and-verified-models)
- [9. Errors and limits](#9-errors-and-limits)

## 1. Starting the server

Requirements:

- an mmproj with a vision encoder (for video) and/or an audio encoder (for native audio);
- `ffmpeg` and `ffprobe` on `PATH`, or `--video-ffmpeg-dir DIR`. Video support (`MTMD_VIDEO`) is on by
  default in the CMake build;
- for transcripts, an OpenAI-compatible speech-to-text server (see [7](#7-audio-native-or-transcript)).

Example (Qwen3.8-27B with its MTP head, 160k context, local files under `C:\media\`, transcripts):

```powershell
llama-server -m Qwen3.8-27B-huihui-NVFP4.gguf --mmproj mmproj-huihui.gguf `
  --spec-type draft-mtp --spec-draft-ngl 99 --spec-draft-n-max 3 `
  -ngl 99 -fa on --jinja -c 160000 --host 127.0.0.1 --port 8080 `
  --media-path C:\media\ --asr-url http://127.0.0.1:8178
```

At startup the server logs what it measured from the mmproj, for example
`video: image tokens 512x288=144 1920x1088=2040, temporal merge 2, audio none, transcripts http://127.0.0.1:8178`.

### Flags

| Flag | Meaning | Default |
|---|---|---|
| `--video-fps F` | sampling rate; also the most a request may ask for | `2.0` |
| `--video-min-fps F` | lowest rate a long range may be thinned to; longer ranges are rejected | `0.05` |
| `--video-detail LEVEL` | default frame detail: `low`, `standard`, `high`, `max` | `standard` |
| `--video-max-tokens N` | token budget per video part (requests may only lower it) | `32768` |
| `--video-max-frames N` | frame budget per video part (requests may only lower it) | `768` |
| `--video-dedup F` | a frame pair is kept only if some screen region changed by at least F luma levels (0–255); `0` = off | `4.0` |
| `--video-timestamp-interval MS` | spacing of clock stamps for models without native video timestamps | `5000` |
| `--video-ffmpeg-dir DIR` | where `ffmpeg`/`ffprobe` live | `PATH` |
| `--media-path DIR` | allow `file://` media under DIR | disabled |
| `--asr-url URL` | OpenAI-compatible `/v1/audio/transcriptions` server for transcripts | unset |
| `--asr-model NAME` | `model` field sent to it | server default |
| `--asr-language LANG` | default transcript language (`en`, `fi`, `zh`, ...) | auto-detect |
| `--audio-native-max-seconds N` | with an audio mmproj, longer audio is transcribed instead (needs `--asr-url`) | `600` |

Every flag also has a `LLAMA_ARG_*` environment variable. The full list is in
[tools/server/README.md](https://github.com/Randy420Marsh/llama.cpp/blob/master/tools/server/README.md).

## 2. Using it from the web UI

1. Start the server as above and open `http://127.0.0.1:8080`.
2. Use the attachment button to add a video (mp4, webm, mkv, ...) or an audio file (wav, mp3, ...). The
   video option appears when the model can see video. The audio option appears when the mmproj has an
   audio encoder **or** the server was started with `--asr-url`, since the server can then transcribe
   audio for any model.
3. Type the question and send. Reasoning on/off and the reasoning effort work the same as for text.

The web UI sends the **whole file** as base64 with the server defaults (`--video-detail`,
`--video-max-tokens`, `--video-fps`). What that means:

- A short clip is sampled at `--video-fps`. A longer one is thinned to fit the budget. At `standard`
  detail with Qwen3.8, a 60 s clip uses about 32k tokens, and a 10 min video gets one frame every ~5 s.
- At `standard`, the longest video that fits is about 38 minutes (the frame rate cannot drop below
  `--video-min-fps 0.05`). Past that the web UI shows the "does not fit" error. Start the server with
  `--video-detail low` to fit about 85 minutes, or use the API with time ranges (below).
- The web UI has no fields for time ranges, `fps` or `detail`. Use the API for those.
- Speech in the video arrives as a transcript with timestamps when `--asr-url` is set.

## 3. Using it from the OpenAI endpoint

`POST /v1/chat/completions`. Media parts go in the `content` array of a user message, next to text parts.

### Part types

| Part | Value (`url` or `data`) | Notes |
|---|---|---|
| `{"type": "video_url", "video_url": {"url": ...}}` | HTTP(S) URL, `file://path`, data URI, bare base64 | OpenAI-style name |
| `{"type": "input_video", "input_video": {"data": ...}}` | same | llama.cpp / web UI name |
| `{"type": "input_audio", "input_audio": {"data": ..., "format": "wav"}}` | same | OpenAI name; `format` is ignored, the container is probed |
| `{"type": "audio_url", "audio_url": {"url": ...}}` | same | |

- HTTP(S) sources are streamed with range requests: only the index and the requested ranges are
  downloaded. In testing, a 10 s range 10.5 h into a 199 MiB file fetched 48 MiB. ffprobe, the frame cut
  and the audio cut each open the URL, and each open reads ahead ~16 MiB before seeking.
- `file://` needs `--media-path DIR`. The path is relative to DIR, for example `file://talks/day1.mp4`.
- Formats: anything ffmpeg reads in a plain container (mp4/mov, mkv/webm, avi, flv, ts, ogg, wav, mp3,
  flac, aac, ...). Playlists (m3u8), concat lists and image sequences are refused.

### Options (inside the `video_url` / `input_video` / `input_audio` / `audio_url` object)

| Field | Values | Default |
|---|---|---|
| `start`, `end` | seconds (`3600`, `"3600s"`) or `"[hh:]mm:ss[.f]"` (`"01:00:00"`) | whole file |
| `duration` | instead of `end`; same formats | |
| `segments` | list of `{start, end}` or `{start, duration}`; several ranges in one part | |
| `fps` | sampling rate, at most `--video-fps` | `--video-fps` |
| `detail` | `low` (256), `standard` (576), `high` (1024), `max` (2048), or a number of tokens per frame pair | `--video-detail` |
| `max_tokens` | token budget for this part; can only lower `--video-max-tokens` | server limit |
| `max_frames` | frame budget for this part; can only lower `--video-max-frames` | server limit |
| `audio` | `auto`, `transcript`, `native`, `both`, `none` (see [7](#7-audio-native-or-transcript)) | `auto` |
| `language` | transcript language hint, e.g. `"fi"` | `--asr-language` |

### Examples

Two ranges of a 12-hour recording, high detail, with the spoken audio:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role": "user", "content": [
    {"type": "video_url", "video_url": {
      "url": "file://recordings/day.mp4",
      "segments": [{"start": "05:00:00", "end": "05:00:12"}, {"start": "10:30:00", "duration": 12}],
      "detail": "high", "audio": "transcript"}},
    {"type": "text", "text": "For each segment: what does the screen show and what is said, with times?"}
  ]}],
  "max_tokens": 2000
}'
```

A remote video, one range, reading small text:

```json
{"type": "video_url", "video_url": {"url": "https://example.com/lecture.mp4",
  "start": "00:42:00", "end": "00:43:30", "fps": 0.5, "detail": "max"}}
```

An audio file, transcribed (or passed natively on audio models):

```json
{"type": "input_audio", "input_audio": {"data": "<base64 wav/mp3/flac>", "format": "wav"}}
```

Python (`openai` package):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(
    model="any",
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "file://talk.mp4", "start": "01:00:00", "end": "01:05:00"}},
        {"type": "text", "text": "Summarize this part. Quote what is said about the budget, with times."},
    ]}],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},  # or leave reasoning on
)
print(r.choices[0].message.content)
```

Reasoning works with media in every mode. Use `chat_template_kwargs.enable_thinking` to turn it on or
off, and `reasoning_effort` for models that support effort levels. With reasoning on, leave room in
`max_tokens` for the thinking.

## 4. How to ask

The model only sees the ranges you send, and the server does not read your question to choose them. So
put the time range in the request fields, and ask the question in text.

- **Media first, question after.** Put the media parts before the text part in `content`.
- **Ask for times.** Frames and transcript lines carry source time. Questions like "when does X happen",
  "quote what is said with times" or "which clock time is shown" get answers in source seconds
  (for example 18000.8 s for 05:00:00.8).
- **Reading on-screen text:** use `detail: "high"` or `"max"` and a low `fps` (0.2–0.5), so the budget goes
  to resolution instead of repeated frames.
- **Following motion:** use `standard` or `low` detail at 1–2 fps.
- **Speech:** set `audio: "transcript"` (default for video when `--asr-url` is set) and `language` for
  non-English audio. Ask the model to quote the transcript, not paraphrase it.
- **Long videos (hours):** one request cannot hold a whole long video at a useful rate, and the server
  says so instead of silently thinning it to nothing. Split the video into windows, ask each window for a
  timestamped summary, then send one text-only request that combines the summaries (map-reduce):

```python
windows = [(t, min(t + 1200, total)) for t in range(0, total, 1200)]   # 20 min each
notes = []
for a, b in windows:
    r = ask([{"type": "video_url", "video_url": {"url": URL, "start": a, "end": b,
                                                 "fps": 0.2, "detail": "standard"}},
             {"type": "text", "text": "List the events in this part with their times (seconds)."}])
    notes.append(f"[{a}-{b} s]\n{r}")
final = ask([{"type": "text", "text": "Combine these notes into one timeline and summary:\n\n" + "\n\n".join(notes)}])
```

## 5. Budget, detail and readability

Frames keep their aspect ratio. Each frame is scaled so its area is `detail × 1024` pixels (multiples of
32, never upscaled). How many tokens a frame costs depends on the model, and the server measures that
from the mmproj at startup. For a 16:9 source on Qwen3.8 (32×32 px per token, two frames merged):

| detail | frame size | tokens / frame | Chinese 20 px body text | fine UI text (26 fields) |
|---|---|---|---|---|
| `low` (256) | 672×384 | ~126 | 93 % of characters | 6/26 |
| `standard` (576) | 1024×576 | ~288 | 100 % | 16/26 |
| `high` (1024) | 1344×768 | ~504 | 100 % | 21/26 |
| `max` (2048) | 1920×1088 | ~1020 | 100 % | 25/26 |

These were measured with a Chinese article at 1080p and a dense ComfyUI screenshot. Text that becomes
unreadable makes a scale unusable, so `standard` is the lowest default that keeps normal text.

With a budget of `--video-max-tokens`, the frame rate is the smallest of: the requested `fps`,
`max_frames / length` and `max_tokens / (tokens per frame × length)`. If that falls below
`--video-min-fps` the request is rejected. The error message says which range length would fit.

Seconds that fit in 32,768 tokens on Qwen3.8:

| detail | at 2 fps | at 0.5 fps | at 0.05 fps (floor) |
|---|---|---|---|
| `low` | 130 s | 8.7 min | 87 min |
| `standard` | 57 s | 3.8 min | 38 min |
| `max` | 16 s | 64 s | 10.7 min |

Each request also has to fit the context (`-c`): video tokens + transcript + text + output.

The server logs its plan for every video part:
`video: file.mp4, 2 segment(s), 24.0 s total, 1920x1080 -> 1024x576, 2.000 fps (requested 2.00), ~288 tokens/frame, est. 13824 tokens`.

## 6. Timestamps and dedup

- **Source time.** Each range is cut with its absolute start time kept, so frames 5 h into a file are
  labelled 18000 s and later, not 0 s.
- **Qwen-VL models** (two frames merged per token) get the model's own format, `<12.5 seconds>`, before
  each frame pair. Other models get a clock stamp such as `[1h02m03.5s]` every
  `--video-timestamp-interval` ms.
- **Ranges** get a header: `Video segment 2 of 3, source time 10:30:00.0 to 10:30:12.0:`.
- **Dedup.** A frame pair is dropped when no region of a 64×36 grid changed by at least `--video-dedup`
  luma levels since the last kept pair. Comparing regions instead of the whole frame means a single
  changing digit on a clock still counts. At least one pair is kept every 10 s even when nothing
  changes, and the first frame after a dropped stretch always gets a timestamp, so gaps stay visible.

## 7. Audio: native or transcript

| `audio` | Video part | Audio part |
|---|---|---|
| `auto` (default) | transcript if `--asr-url` is set; else native if the model has an audio encoder; else none | native if the model has an audio encoder and the clip is under `--audio-native-max-seconds`; else transcript |
| `transcript` | speech-to-text lines with source times | same |
| `native` | audio goes to the model's audio encoder (placed before the frames) | same |
| `both` | native audio and transcript | same |
| `none` | frames only | (rejected) |

`auto` prefers transcripts for video because Gemma4-12B's native audio, placed next to many frames,
made up speech for individual frames. Transcripts were more accurate. For audio alone, native works well.

A transcript reaches the model as text:

```
Audio transcript of this video segment (speech-to-text; times are seconds in the source):
<18000.8 - 18002.7 seconds> Testing, testing, one, two, three.
<18003.6 - 18006.0 seconds> The quick brown fox jumps over the lazy dog.
```

### Speech-to-text server

Any server that implements OpenAI `POST /v1/audio/transcriptions` and returns `verbose_json` segments
works. This fork ships one in [`tools/asr-server/`](../tools/asr-server/): a faster-whisper server
(`asr_server.py`) that keeps everything in memory - uploaded audio is decoded from RAM, and the model
stays resident in VRAM.

Setup (one-time): the launcher creates a `uv` venv (Python 3.11) and installs the requirements
(`faster-whisper`, `fastapi`, `uvicorn`, and the CUDA cuBLAS/cuDNN wheels on Windows and Linux).
Later runs skip the install. The Whisper model (default `large-v3`, ~3 GB) is downloaded from
Hugging Face into `tools/asr-server/models/` on first use, so the first request is slower.

```powershell
# Windows
tools\asr-server\run.bat --port 8178 --model large-v3 --compute-type float16 --preload
# Linux / Git Bash
bash tools/asr-server/run.sh --port 8178 --model large-v3 --compute-type float16 --preload
# --idle-unload 300 frees the VRAM after 5 idle minutes
```

- Use multilingual `large-v3`. The `distil-*` models are English-only, and `turbo` was less accurate here.
- `float16` is more accurate than `int8_float16`: the int8 run heard "quick brown" as "quick-prone".
- Voice activity detection is automatic. When it removes most of the audio (music, singing), the audio is
  transcribed again without it, so lyrics are not dropped. Stock hallucinations ("Thanks for watching")
  are filtered.
- The model stays loaded in VRAM next to the LLM. `--idle-unload` releases it.

## 8. Tested and verified models

All results below are from `llama-server` built from this fork (2026-09-23, 160k context, RTX 5090).
The exact launch commands for each model, with the test date and time, are in
[Model launch configs](Model-Launch-Configs).

**Capability matrix.** Each modality (text, image, video, audio) runs under every reasoning mode the
model offers: off, default, and each effort level. Requests are streamed like the web UI. A run passes
when the answer has the known facts of the test media, reasoning appears only in `reasoning_content`
(and only when on), no think/channel tags leak into the answer, and it finishes normally.

| Model | mmproj | Video | Audio | Matrix |
|---|---|---|---|---|
| Qwen3.8-27B huihui abliterated NVFP4 + MTP (`Qwen3.8-27B-huihui-NVFP4.gguf`) | `mmproj-huihui.gguf` (vision) | native Qwen video: frame pairs + `<t seconds>` | transcript | **20/20** (off, default, low, medium, xhigh) |
| Gemma4-12B QAT Uncensored HauhauCS Balanced Q4_K_M | `mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf` (vision + audio) | frames + clock stamps | native (audio), transcript (video) | **8/8** (off, default) |
| Gemma-4-26B-A4B-it ultra-uncensored heretic i1-Q5_K_M | `...heretic.mmproj-f16.gguf` (vision) | frames + clock stamps | transcript | **8/8** |
| Nemotron-3-Nano-Omni-30B-A3B Q4_K_M | vision-only mmproj | frames + clock stamps | transcript | **8/8** |
| Nemotron-3-Nano-Omni-30B-A3B Q4_K_M | combined audio+vision mmproj, converted with upstream `convert_hf_to_gguf.py --mmproj` from the BF16 checkpoint | frames + clock stamps | native | **7/8** (with reasoning off, one audio answer misheard "fox") |

**Pipeline cases** (Qwen3.8, re-run after the last changes): all 6 pass.

| Case | Result |
|---|---|
| 8 s clip with speech, data URI | direction of motion and both spoken sentences with times; 2.6k tokens |
| 1080p 60 s, data URI | described, all four spoken sentences with times; 35k tokens (was 247k) |
| 12 h video via `file://`, two 12 s ranges at 05:00:00 and 10:30:00 | correct clock readings, SECTION numbers and speech at 18000.8 s / 37800.8 s |
| 12 h video over HTTP, 10 s range at `detail: max` | correct start/end clock; 48 MiB of 199 MiB downloaded |
| `input_audio` wav on a vision-only model | exact transcript with times |
| whole 12 h video in one part | clear 400 error: "43200 s of video does not fit ... at most 2276 s" |

Model notes:

- **Qwen3.8 / Qwen-VL.** The vision-only mmproj handles native video. Use transcripts for speech.
- **Gemma4-12B.** The audio encoder works well for audio on its own. For video, `auto` uses transcripts
  (see [7](#7-audio-native-or-transcript)).
- **Nemotron-Omni.** The audio mmproj from the separate omni fork uses a different parakeet GGUF layout
  and does not load here. Convert the combined mmproj from the Hugging Face BF16 checkpoint with this
  repo's `convert_hf_to_gguf.py --mmproj`. Audio and video then work in one pass.
- The test media (moving shapes with speech, 1080p test pattern, 12 h clock video, speech wav) was
  generated for these tests with known contents, so answers can be checked automatically.

## 9. Errors and limits

| Error (HTTP 400 unless noted) | Meaning |
|---|---|
| `video_url: N s of video does not fit: at detail "standard" (1024x576, ~288 tokens per frame) ... allows at most M s` | the ranges need more than the budget even at `--video-min-fps`: send shorter ranges, lower `detail`, or split into chunks |
| `file:// URLs are not allowed unless --media-path is specified` | start the server with `--media-path` |
| `transcripts need a speech-to-text server; start llama-server with --asr-url` | `audio: "transcript"` without `--asr-url` |
| `this model has no audio encoder ...` | `audio: "native"` on a vision-only mmproj |
| `cannot read media (ffprobe): ...` | unreadable or refused input (e.g. a playlist) |
| `speech-to-text server ... is not reachable` (500) | the `--asr-url` server is down |
| `video input is not supported` | no vision mmproj, or a build without `MTMD_VIDEO` |

Limits:

- The web UI always sends whole files with server defaults. Time ranges need the API.
- URLs are fetched without a private-network filter, the same as upstream `image_url`. Do not expose the
  server to untrusted clients if it can reach internal hosts.
- Each request is prepared in memory. A base64 upload is about 1.33× the file size in the request body.
