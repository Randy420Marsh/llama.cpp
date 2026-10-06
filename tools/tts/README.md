# llama.cpp TTS

This is a tool to demonstrate audio generation capability in llama.cpp via `libmtmd`. It was added via PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Note: this tool used to serve as a demo for OuteTTS, but it was converted to a more model-agnostic tool.

## Common usage

Simple usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF -p "Hello world" --output out.wav
```

Common params:
- Sampling params such as `--top-k`, `--top-p`, `--temp`, etc.
- `-n <number_of_frames>` limits the output length, e.g. `-n 500`. Note that how many milliseconds each frame represents varies by model
- Core inference params such as `-ngl`, `-b`, `-ub`, etc.

## Qwen3-TTS

Available params:
- `--tts-lang` can be `zh`, `en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, `ru` (default: `en`)
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3)

Example usage:

```sh
llama-tts -hf ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF \
    -p "Hello world" \
    --tts-lang english \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

## Pocket TTS

Available params:
- `--tts-speaker-file` should point to a speaker reference audio file (wav, mp3). It is required, the model produces almost no audio without it
- Note: `lang` is not used, the language is a property of the weights

Example usage:

```sh
llama-tts -m pocket-tts.gguf \
    -mm mmproj-pocket-tts.gguf \
    -p "Hello world" \
    --tts-speaker-file speaker.mp3 \
    --output out.wav
```

## llama-tts-server (fork)

`llama-tts` loads the model for every run. `llama-tts-server` keeps it loaded and serves an OpenAI-style
speech endpoint, so a request costs generation time only (about 5x realtime for Qwen3-TTS 1.7B Q8_0 on an
RTX 5090). Every audio file in `--tts-voices-dir` is a voice: zero-shot cloning, no training.

```sh
llama-tts-server -m Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
    -mm mmproj-Qwen3-TTS-12Hz-1.7B-Base-bf16.gguf \
    -ngl 99 --tts-voices-dir voices --port 8181

curl -s localhost:8181/v1/audio/speech -H "Content-Type: application/json" \
    -d '{"input": "Hello world", "voice": "my-voice", "language": "en"}' -o out.wav
```

Extra options: `--host` (default `127.0.0.1`), `--port` (default `8181`), `--tts-voices-dir`. The voices folder
holds `<name>.wav|mp3|flac` files, or `<name>/reference.<ext>` folders with an optional `voice.json`
(`{"language": "en"}`). `-c` defaults to 4096: a request needs about a thousand positions, and the llama.cpp
default (the training length, grown by `--fit`) would take all free VRAM.

Endpoints: `GET /health`, `GET /v1/models`, `GET /v1/audio/voices`, `POST /v1/audio/speech`
(`input`, `voice`, `language`, `seed`, `response_format` = `wav` or `pcm`). Inputs longer than a few sentences
are split at sentence ends and generated piece by piece.

The web UI uses it through the speech server in `tools/speech-server`; see the wiki page `Voice`.

**Note for GGUF conversion:**

The [upstream repository](https://huggingface.co/kyutai/pocket-tts) holds one complete model per language under `languages/`, next to a set of shared files at the root. Convert one of the `languages/<name>` directories, **not** the root directory:

```sh
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --outfile pocket-tts.gguf
python convert_hf_to_gguf.py path/to/pocket-tts/languages/english --mmproj --outfile mmproj-pocket-tts.gguf
```
