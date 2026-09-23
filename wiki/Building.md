# Building this fork

The build is the same as upstream llama.cpp (see `docs/build.md`); this page lists the settings used for the tested
builds and what the video/audio features need at run time. The default build directory is `build/`.

## What the video and audio features need

| Need | Why | Where |
|---|---|---|
| `ffmpeg` and `ffprobe` | decode and cut video/audio parts | on `PATH`, or `--video-ffmpeg-dir DIR` |
| OpenSSL 3 or newer | HTTPS video URLs and `-hf` downloads in `llama-server` | Windows: build from source (below); Ubuntu: `libssl-dev` |
| a speech-to-text server | transcripts for models without an audio encoder (`--asr-url`) | `tools/asr-server` (faster-whisper) |

Any ffmpeg build works for `llama-server` (the distribution package, or a Windows build from gyan.dev or
BtbN). The tests used a shared FFmpeg build with NVENC/NVDEC and `libvmaf_cuda`, made with the script
`shared_ffmpeg_vmaf_cuda_build_script_final_v14.sh` from
[Randy420Marsh/cuda_vmaf_stuff](https://github.com/Randy420Marsh/cuda_vmaf_stuff) (Ubuntu; the Windows build
follows the same steps under MSYS2). Point `--video-ffmpeg-dir` at its `bin` folder when it is not on `PATH`.

## Windows 11 (tested 2026-09-23: VS 2022, CUDA 13.3, CMake 4.4, OpenSSL 4.0.1, RTX 5090)

Requirements: Visual Studio 2022 with the C++ workload, the CUDA Toolkit, and OpenSSL.

OpenSSL from source (Strawberry Perl and NASM on `PATH`, in an "x64 Native Tools" prompt run as administrator):

```bat
cd C:\openssl-4.0.1
perl Configure VC-WIN64A
nmake
nmake install
```

This installs to `C:\Program Files\OpenSSL`. Without NASM, add `no-asm` to the `Configure` line (slower crypto).

Configure and build llama.cpp:

```bat
cd C:\AI\llama.cpp
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3

cmake -B build -G "Visual Studio 17 2022" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 ^
  -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA_QUANTS=all ^
  -DOPENSSL_ROOT_DIR="C:\Program Files\OpenSSL"
cmake --build build --config Release --parallel
```

The binaries are in `build\bin\Release\` (`llama-server.exe`, `llama-cli.exe`, ...).

Notes:

- `CMAKE_CUDA_ARCHITECTURES=120` is the RTX 50 series; use your GPU's compute capability (89 for RTX 40, 86 for
  RTX 30) or leave it out to build for several.
- `GGML_CUDA_FA_QUANTS=all` compiles FlashAttention for every K/V cache type pair, so mixed caches such as
  `-ctk q8_0 -ctv q5_1` stay on the GPU. The default compiles only a few matching pairs and builds faster.
- `GGML_CUDA_FORCE_MMQ=ON` is the tested setting. `OFF` (cuBLAS for large batches) also works; a build like that
  was kept in a second directory (`build-nommq`) for comparison.
- Older recipes use `GGML_CUDA_FA_ALL_QUANTS=ON` (deprecated, same as `GGML_CUDA_FA_QUANTS=all`) and
  `GGML_CUDA_PEER_MAX_BATCH_SIZE` (removed upstream, ignored).
- To start clean: `rmdir /S /Q build`.

## Ubuntu 24.04

```bash
sudo apt install build-essential cmake git libssl-dev ffmpeg
# CUDA Toolkit from NVIDIA's apt repository (cuda-toolkit-13-x), nvcc on PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_FA_QUANTS=all
cmake --build build --config Release -j
```

The binaries are in `build/bin/`.

## Check the build

```bat
build\bin\Release\llama-server.exe --help | findstr /C:"--asr-url" /C:"--video-max-tokens"
```

Both options are listed in a build of this fork. Then follow [Video and audio](Video-and-Audio) to start the
speech-to-text server and `llama-server`.
