# Voice: read aloud, your own voices, dictation

The web UI of this fork can read replies aloud, in a built-in voice or in **your own voice** (zero-shot
cloning: record about ten seconds, no training), and can turn the microphone into text. Three programs do the
work:

| Program | What it does | Where |
|---|---|---|
| `llama-tts-server` | keeps a llama.cpp speech model loaded (Qwen3-TTS, Pocket-TTS) and speaks with any voice in a folder of reference clips | `tools/tts/tts-server.cpp`, built with the rest |
| speech server | the web UI's back end: engines, your voices, model downloads, speech-to-text | `tools/speech-server/` (Python, port 8179) |
| ASR server | Whisper large-v3 speech-to-text, also used by `llama-server --asr-url` | `tools/asr-server/` (Python, port 8178) |

```
 browser: llama.cpp web UI (llama-server :8080)
   │  Settings → Voice / Microphone · 🔊 Read aloud · 🎤
   ▼
 speech server :8179 ─┬─► llama-tts-server :8181   Qwen3-TTS + your voices (llama.cpp)
                      ├─► Piper (CPU) · Kokoro · Chatterbox (PyTorch, optional)
                      ├─► llama-server :8180 + SNAC   Orpheus 3B (optional)
                      └─► ASR server :8178 ◄── llama-server --asr-url (audio/video parts)
```

## Quick start

1. **Build** as usual ([Building](Building)). `llama-tts-server` is a normal target; check it with
   `build\bin\Release\llama-tts-server.exe --help`.
2. **Speech-to-text** (recommended; needed for languages other than English):
   `tools\asr-server\run.bat --port 8178 --model large-v3 --compute-type float16 --preload`
   (Linux: `bash tools/asr-server/run.sh ...`).
3. **Speech server**: `tools\speech-server\run.bat --no-https --lan` (Linux: `./run.sh --no-https --lan`).
   The first run creates its Python environment ([uv](https://docs.astral.sh/uv/) must be on `PATH`).
4. **llama-server** as always, e.g. `llama-server -m model.gguf --host 0.0.0.0 --port 8080 --asr-url http://127.0.0.1:8178`.
5. Open `http://<pc>:8080` → **Settings → Voice**:
   * **Models** → *Download* "Qwen3-TTS 1.7B Q8_0" (2.5 GB);
   * **Your voices** → *Record* → read the passage on screen → *Stop* → name → *Save voice*;
   * *Test*, then **Save settings**.
6. Every reply now has a 🔊 button; a second click stops it.

## Settings → Voice

| Setting | |
|---|---|
| Speech server URL | empty = this page's host, port 8179. Works the same from a phone on the LAN |
| Read replies aloud automatically | speaks each reply as soon as it has finished |
| Engine | lists what the speech server has; engines that need missing Python packages are greyed out with the reason. *Automatic* = llama.cpp when it has a model, else Kokoro, else Piper |
| Voice | **Your voices** and the engine's **Built-in** voices; built-in voices that are not downloaded yet are marked |
| Model | the model files of the engine (quantizations) |
| Language | Qwen3-TTS: en, zh, de, it, pt, es, ja, ko, fr, ru. *Automatic* = the voice's language, or the script of the text (Chinese, Japanese, Korean, Russian) |
| Speed | 0.6–1.8×. Engines that cannot change their rate are sped up by the browser (pitch preserved) |
| Your voices | record or upload a clip, play it, edit what is said in it, delete |
| Models | what is on disk and what can be downloaded, per engine, with the folder each one belongs in; *Unload* frees the GPU |

Reading is done sentence by sentence: the first piece is short so sound starts within about a second, and
the next piece is generated while the current one plays.

## Settings → Microphone

| Setting | |
|---|---|
| Voice input becomes | **Auto** (default): text in the message box when the speech server answers, else the recording itself. **Text**: always transcribe. **Audio attachment**: send the recording; a model with an audio encoder hears it, other models get the `--asr-url` transcript |
| Voice input language | e.g. `en`, `fi`. Setting it stops an accent from being transcribed as another language |
| Mic input volume, noise cancellation, auto-stop on silence, auto-send | as named; the bar under the message box shows the live level while recording |

A recording becomes text *or* an attachment, never both: with both, `llama-server --asr-url` transcribed the
attachment again and the message carried the same words twice.

Browsers only allow the microphone on `https://` pages and on `localhost`. To dictate from another device,
serve the web UI over https (`--ssl-key-file` / `--ssl-cert-file`) and run the speech server with https as well
(`run.bat --lan`, its default), then open `https://<pc>:8179` once to accept its certificate.

## Your own voice

A voice is one clip of the speaker plus, optionally, what is said in it. Nothing is trained or fine-tuned: the
cloning engines encode the clip each time they speak (Qwen3-TTS turns it into a speaker embedding with its
speaker encoder).

Tips for a good clip: one speaker, a quiet room, 6–15 seconds of natural speech, no music. The speech server
trims silence at both ends, levels the loudness and converts it to 24 kHz mono.

Measured with two reference voices, one 7–8 s clip each; similarity of the cloned speech to each reference,
by a speaker encoder that took no part in generating it (1.0 = identical):

| Engine | clone of A: to A / to B | clone of B: to A / to B |
|---|---|---|
| llama.cpp Qwen3-TTS 1.7B Q8_0 | **0.925** / 0.628 | 0.558 / **0.946** |
| Orpheus 3B pretrained Q8_0 | 0.551 / 0.681 (missed) | 0.630 / 0.753 |

Whisper transcribed every output with the right words. Qwen3-TTS kept the voice across a 27-second, two-piece
paragraph as well (0.947). Orpheus' pretrained model was never trained to clone; it is offered as
*experimental*.

## Where models and voices go

Everything the speech server uses lives under one folder, `TTS_CACHE` (default: `models\` in
`tools/speech-server`). The Models list downloads into the right place; files copied in by hand are found
the same way (press *Refresh*).

| What | Folder | File names |
|---|---|---|
| llama.cpp speech models | `<cache>/gguf/qwen3-tts/` | `<name>-<quant>.gguf` and `mmproj-<name>-<quant>.gguf` with the same `<name>`. Any bf16/f16 mmproj is preferred over Q8_0 |
| Orpheus | `<cache>/gguf/orpheus/` | `orpheus-3b-0.1-ft-<quant>.gguf`, `orpheus-3b-0.1-pretrained-<quant>.gguf` |
| Piper | `<cache>/piper/` | `<voice>.onnx` + `<voice>.onnx.json` |
| Kokoro, Chatterbox, SNAC | `<cache>/hf/` | Hugging Face cache |
| Your voices | `<cache>/voices/` (or `TTS_VOICES_DIR`) | `<id>/reference.wav`, `<id>/transcript.txt`, `<id>/voice.json`; a bare `<id>.wav` works too |
| Local Whisper | `<cache>/whisper/` | faster-whisper models |

| Model | Download source |
|---|---|
| Qwen3-TTS 1.7B Base: Q8_0 (recommended), Q4_K_M, bf16, with mmproj | `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF` |
| Orpheus 3B finetuned / pretrained GGUF | `unsloth/orpheus-3b-0.1-ft-GGUF` / `HorizonNexusAI/orpheus-3b-0.1-pretrained-GGUF` (community conversions of the `canopylabs` models) |
| Piper voices | `rhasspy/piper-voices` |
| Kokoro, Chatterbox, SNAC | `hexgrad/Kokoro-82M`, `ResembleAI/chatterbox`, `hubertsiuzdak/snac_24khz` |

Command line: `python tools/speech-server/setup/fetch_voices.py --list`, then the same script with no arguments
(the recommended model of each usable engine) or with `engine/id` items.

## llama-tts-server on its own

```bash
llama-tts-server -m Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf -mm mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf \
    -ngl 99 --tts-voices-dir voices --port 8181
llama-tts-server -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF --port 8181      # download and serve
```

| | |
|---|---|
| `--host`, `--port` | default `127.0.0.1:8181` |
| `--tts-voices-dir DIR` | one voice per audio file (`<name>.wav/mp3/flac`) or per sub-folder with `reference.<ext>` (+ `voice.json` `{"language": "en"}`) |
| `--tts-speaker-file FILE` | the voice used when a request names none |
| `--tts-lang` | default language |
| `-c` | context; defaults to 4096 here. Left at 0 (the llama.cpp default) it is the backbone's training length and `--fit` grows it into all free VRAM |
| sampling, `-ngl`, `-b`, ... | as for `llama-tts` |

| Endpoint | |
|---|---|
| `GET /health` | |
| `GET /v1/models` | model, generator (`qwen3tts` / `pockettts`), sample rate, languages |
| `GET /v1/audio/voices` | the voices in `--tts-voices-dir` |
| `POST /v1/audio/speech` | `{"input", "voice", "language", "seed", "response_format": "wav" or "pcm"}` → 16-bit mono audio. Long input is split at sentence ends and spoken piece by piece |

```bash
curl -s localhost:8181/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input": "Hello from llama.cpp.", "voice": "my-voice"}' -o hello.wav
```

One request is generated at a time. On an RTX 5090, next to a 27B chat model: model load 1.4 s, then about
5× realtime (27 s of speech in 4.9 s), 4.5–5.5 GB VRAM with Q8_0 + bf16 mmproj.

Pocket-TTS works the same once converted (see `tools/tts/README.md`); it needs a voice in every request.

## Memory

Only one GPU speech engine is loaded at a time, and it unloads after 10 minutes without speech
(`TTS_IDLE_UNLOAD` seconds in the speech server). **Settings → Voice → Models → Unload** frees it at once, and
stopping the speech server stops `llama-tts-server` and the Orpheus `llama-server` with it (on Windows they are in
a job object that closes with the speech server).

## Ports

| Port | |
|---|---|
| 8080 | `llama-server` (chat, web UI) |
| 8178 | ASR server |
| 8179 | speech server |
| 8180 | Orpheus `llama-server`, started on demand |
| 8181 | `llama-tts-server`, started on demand |

On Windows a second `llama-server` can bind a port that is already taken, and requests then reach either
process; keep every server on its own port.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Settings → Voice: "No speech server" | not started, or not reachable from this device: use `--lan`, or put its URL in the field |
| A voice is read by Piper instead | the chosen engine failed for that sentence; the toast says why (model missing, VRAM full, unknown voice) |
| The 🎤 button is disabled | the page is not on `https://` or `localhost` |
| Dictation in a foreign language comes out wrong | the local fallback Whisper is English-only: start the ASR server, and set *Voice input language* |
| `llama-tts-server` takes all free VRAM | an explicit `-c 0`; leave `-c` unset (4096) |
| Kokoro / Chatterbox / Orpheus "not installed" | run `tools/speech-server/run-tts-setup.bat` (or `.sh`) once |
