# Media pipeline internals (llama-server)

How a `video_url` / `input_audio` part becomes model input, why it is built this way, and where the code is.
For usage see [Video and audio](Video-and-Audio).

## Request flow

```
POST /v1/chat/completions
  oaicompat_chat_params_parse (server-common.cpp)
    input_video / video_url  -> server_media_video_part()   (server-media.cpp)
    input_audio / audio_url  -> server_media_audio_part()
        resolve_source   http(s) URL | file:// under --media-path | data URI / bare base64 (stdin)
        probe_media      ffprobe JSON: duration, size, rotation, audio/video streams
        parse_options + finalize_segments   start/end/duration/segments, fps, detail, budget, audio mode
        plan             frame size from detail, fps lowered to fit max_tokens / max_frames
        per segment:
          ffmpeg cut  -> in-memory FFV1/Matroska clip with source timestamps
          ffmpeg audio -> 16 kHz mono WAV in memory -> native audio part and/or transcript text
        returns content parts: segment header text, media markers, transcript text
  mtmd (mtmd-helper.cpp) decodes each clip:
    container start_time -> absolute frame times
    frames grouped by the model's temporal merge (2 for Qwen-VL), dedup per group
    timestamp text between groups: "<t seconds>" (Qwen) or "[1h02m03.5s]" clock
```

## The ffmpeg commands

Video cut, one per segment (input seek, so hours into a file cost nothing):

```
ffmpeg -ss S -i INPUT -t D -map 0:v:0 -an -sn -dn \
  -vf fps=F,scale=W:H:flags=lanczos,format=yuv420p \
  -c:v ffv1 -level 3 -g 1 -slices 4 -output_ts_offset S -f matroska pipe:1
```

Audio cut:

```
ffmpeg -ss S -i INPUT -t D -map 0:a:0 -vn -sn -dn -ac 1 -ar 16000 -c:a pcm_s16le -f s16le pipe:1
```

The PCM gets a WAV header in memory. Every input also gets
`-format_whitelist mov,mp4,...,wav,mp3,flac -protocol_whitelist <per source>`:
`http,https,tcp,tls` for URLs, `file` for paths, `cache,pipe` for inline data.

## Design choices

| Choice | Reason |
|---|---|
| Cut with ffmpeg in the server, not in mtmd | mtmd samples a whole file at `--video-fps`. Seeking, ranges, budget-driven fps and aspect-preserving scaling have to happen before it, per request |
| FFV1 in Matroska for the clips | lossless and intra-only, so text on screen is not blurred twice. Matroska keeps the `-output_ts_offset` timestamps, which carry source time into mtmd |
| `cache:pipe:0` + `-read_ahead_limit -1` for inline data | MP4 files with the index (`moov`) at the end need seeking. `cache:` makes stdin seekable without a temp file |
| HTTP input handed to ffmpeg as a URL | ffmpeg issues range requests, so only the index and the requested ranges are downloaded |
| Frame area = `detail × 1024` px, multiples of 32 | Qwen-VL uses 32×32 px per merged token. The presets were calibrated for text readability (see the user guide) |
| Token cost measured from the mmproj | `make_media_config` tokenizes 512×288 and 1920×1088 placeholders. That gives tokens per pixel and any per-image cap (Gemma ~920, Nemotron 256), so budgets hold for every architecture |
| Timestamps only between frame groups | mtmd merges adjacent frames for Qwen-VL. Text between the two frames of a pair would break the merge |
| Dedup on a 64×36 grid, max cell change | a whole-frame mean missed a changing clock digit. The max over regions does not |
| Native audio before the frames | after many frames, Gemma4-12B invented speech for individual frames. Before them, it did not |
| `auto` audio = transcript for video | transcripts were more accurate than native audio next to frames (Gemma4-12B) |
| `/props` reports audio when `--asr-url` is set | the web UI then offers audio upload for vision-only models too |

## Windows subprocess fix

The server feeds inline media to ffmpeg's stdin from a thread. If ffmpeg exits without reading everything
(for example after `-t`), a blocked `WriteFile` on Windows never fails: there is no SIGPIPE, and the parent
still holds the pipe's read end. `run_capture` now waits for the process, closes `hStdInput`, and only then
joins the feeder thread. On POSIX the child's exit turns the blocked write into `EPIPE`.

## Changed files

| File | Change |
|---|---|
| `tools/server/server-media.h`, `server-media.cpp` | new: source resolution, probing, planning, ffmpeg cuts, audio extraction, transcription client, content-part assembly |
| `tools/server/server-common.cpp/.h` | `oaicompat_chat_params_parse` routes video/audio parts to server-media; `base64_decode_media`; `server_chat_params.media` |
| `tools/server/server-context.cpp` | `make_media_config` (mmproj token probe, temporal merge, audio support), dedup threshold into mtmd, `/props` audio modality with `--asr-url` |
| `tools/server/CMakeLists.txt` | builds server-media, links cpp-httplib |
| `tools/mtmd/mtmd.h/.cpp` | `mtmd_get_n_temporal_merge()` |
| `tools/mtmd/mtmd-helper.h/.cpp` | container `start_time`, seekable `cache:pipe:0` probe, fps only downsamples, per-group queue with region dedup, Qwen `<t seconds>` and clock timestamps, gap marking |
| `common/common.h`, `common/arg.cpp` | `--video-detail`, `--video-max-tokens`, `--video-max-frames`, `--video-min-fps`, `--video-dedup`, `--asr-url`, `--asr-model`, `--asr-language`, `--audio-native-max-seconds`; `--video-fps` default 2 |
| `tools/server/README.md`, `tools/cli/README.md` | request options; flag tables regenerated with `llama-gen-docs` |

## Test tools

The test tools live outside the repo, in `C:\AI\video-pipeline-tests`:

| Script | Purpose |
|---|---|
| `server_matrix.py URL [--label L] [--json F]` | every modality × every reasoning mode, streamed; checks answers, reasoning placement, tag leaks, finish reason |
| `pipeline_test.py URL LABEL [case ...]` | clip, 1080p, `file://` seek, HTTP seek, audio, over-budget error |
| `range_server.py DIR [PORT]` | static server with HTTP Range support that logs the bytes sent, for seek tests |
| `calib/readability.py URL [--video]` | text-readability ladder over token budgets (Chinese article + UI screenshot) |
| `make_test_videos.sh` | builds the test media with known contents (moving shapes, 1080p pattern, 12 h clock) |
