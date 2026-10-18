# Model launch configs used in the 2026-09-23 test runs

The exact `llama-server` commands that produced the results in
[Video and audio, section 8](Video-and-Audio#8-tested-and-verified-models).
All runs: `build/bin/Release/llama-server.exe` from this fork (see [Building](Building)), 160k context,
RTX 5090, port 8081, plus the faster-whisper ASR server on port 8178. The 2026-09-23 runs used a second
build directory, `build-nommq` (same code, `GGML_CUDA_FORCE_MMQ=OFF`); the flags below are the same for
either build.

Paths are variables: `%LLAMA_BIN%` = this fork's `build\bin\Release\llama-server.exe`; `%MODEL_PATH%`, `%MMPROJ_PATH%`, `%MTP_PATH%` = your GGUF files; `%MEDIA_DIR%` = a folder of test media; `%AI_ROOT%` = `C:\AI` on Windows, `$HOME/AI` on Linux.

ASR server (started first, same for all models):

```
%AI_ROOT%\whisper-asr-server\run.bat --port 8178 --model large-v3 --compute-type float16 --preload   rem (or the repo's tools\asr-server\run.bat)
```

These are the commands as they ran then, over plain http. The ASR server (`tools\asr-server`) now serves
https only, with the llama-server certificate, so to repeat a run today use `--asr-url https://127.0.0.1:8178`
(see [Voice](Voice#quick-start)).

During the Nemotron AV run the ASR server was `--compute-type int8_float16`;
the final matrix results use `float16`.

## Qwen3.8-27B huihui abliterated NVFP4 + MTP

Tested 2026-09-23 02:07 (matrix 20/20).

```
%LLAMA_BIN% -m "%MODEL_PATH%"   rem a Huihui-Qwen3.8-27B NVFP4 GGUF
  --mmproj "%MMPROJ_PATH%"   rem the matching vision projector
  --spec-type draft-mtp --spec-draft-ngl 99 --spec-draft-n-max 3
  -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -ctkd q8_0 -ctvd q8_0
  --jinja --parallel 1 -b 4096 -ub 2048 --ctx-checkpoints 16
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
  --presence-penalty 0.0 --repeat-penalty 1.0
  --host 127.0.0.1 --port 8081
  --alias Qwen3.8-27B-huihui-NVFP4
  --reasoning-preserve --reasoning on
  -c 160000
  --media-path %MEDIA_DIR%   rem a folder of test media
  --asr-url http://127.0.0.1:8178
```

## Gemma4-12B QAT Uncensored (HauhauCS Balanced Q4_K_M) + MTP

Tested 2026-09-23 02:24 (matrix 8/8).

```
%LLAMA_BIN% -m "%MODEL_PATH%"   rem a Gemma4-12B QAT uncensored Q4_K_M GGUF
  --spec-draft-model "%MTP_PATH%"   rem the separate MTP head
  --mmproj "%MMPROJ_PATH%"   rem the matching vision projector
  --spec-type draft-mtp --spec-draft-n-max 4
  --ctx-size 160000
  --n-gpu-layers 99 --n-gpu-layers-draft 99
  -fa on
  --temp 1.0 --top-p 0.95 --top-k 64
  --host 127.0.0.1 --port 8081
  --alias gemma4
  --ubatch-size 2048
  --mmproj-offload
  --image-max-tokens 1120
  --mtmd-batch-max-tokens 2048
  --media-path %MEDIA_DIR%   rem a folder of test media
  --asr-url http://127.0.0.1:8178
```

## Gemma-4-26B-A4B-it ultra-uncensored heretic i1-Q5_K_M (vision only)

Tested 2026-09-23 02:25 (matrix 8/8).

```
%LLAMA_BIN% -m "%MODEL_PATH%"   rem a gemma-4-26B-A4B ultra-uncensored Q5_K_M GGUF
  --mmproj "%MMPROJ_PATH%"   rem the matching vision projector
  --ctx-size 160000
  --n-gpu-layers 99 -fa on -ctk q8_0 -ctv q8_0
  --temp 1.0 --presence-penalty 1.1 --top-p 0.95 --top-k 64
  --host 127.0.0.1 --port 8081
  --alias gemma4-26b
  --ubatch-size 2048
  --image-max-tokens 1120
  --jinja
  --mtmd-batch-max-tokens 2048
  --media-path %MEDIA_DIR%   rem a folder of test media
  --asr-url http://127.0.0.1:8178
```

## Nemotron-3-Nano-Omni-30B-A3B Q4_K_M

Shared flags for all three mmproj variants:

```
%LLAMA_BIN% -m "%MODEL_PATH%"   rem a Nemotron-Omni-30B-A3B Q4_K_M GGUF
--alias nemotron-omni
--ctx-size 160000
--n-gpu-layers 99 -fa on
--ubatch-size 2048
--mtmd-batch-max-tokens 8192
--temp 0.6 --top-p 0.95 --min-p 0.01
--jinja --reasoning-format deepseek --reasoning-preserve --reasoning 1
--parallel 1
--chat-template-file %CHAT_TEMPLATE%   rem the model's chat template, if separate
--host 127.0.0.1 --port 8081
--media-path %MEDIA_DIR%   rem a folder of test media
--asr-url http://127.0.0.1:8178
```

- **vision-only** (tested 2026-09-23 02:33, matrix 8/8):
  `--mmproj %MMPROJ_PATH%` (the vision-only projector)
- **combined audio+vision** (tested 2026-09-23 02:37, matrix 7/8):
  `--mmproj %MMPROJ_PATH%` (the combined audio+vision projector),
  converted from the BF16 checkpoint:

  ```
  PYTHONPATH=gguf-py python convert_hf_to_gguf.py "%HF_MODEL_DIR%" --mmproj --outtype bf16 --outfile "%OUT_MMPROJ%"   rem %HF_MODEL_DIR%: the Nemotron-Omni HF checkpoint
  ```

- **audio-only** (failed to load, `Key not found: clip.audio.subsampling_factor`):
  `--mmproj %MMPROJ_PATH%` (the audio-only projector)

One extra run used the omni fork binary
(a second, omni fork build: `<other-fork>\build\bin\Release\llama-server.exe`) with the vision mmproj,
`--ctx-size 262144`, and no `--media-path`/`--asr-url` (tested 2026-09-23 02:38).

## Notes

- The Qwen and Gemma4-12B runs use MTP speculative decoding; the Qwen run uses
  `--spec-draft-ngl`/`--spec-draft-n-max` while the Gemma run uses
  `--spec-draft-model` (a separate MTP file).
- The `--alias` values are representative short names; the test runs used the
  full model name for the 26B and Nemotron models.
- All runs were started hidden with stderr redirected to a log
  (e.g. `llama-gemma12.log`) and polled on `/health` before testing.
- Test media: generated files in `%MEDIA_DIR%` (under the shared AI folder, e.g. `%AI_ROOT%\video-pipeline-tests\media\`; AI_ROOT is `C:\AI` on Windows, `$HOME/AI` on Linux)
  (moving shapes with speech, 1080p test pattern, 12 h clock video, speech wav).
