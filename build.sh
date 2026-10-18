#!/usr/bin/env bash
# ===== llama.cpp build with this fork's web UI (Linux, e.g. Ubuntu 24.04) =====
# Builds llama-server and the tools into build/bin, with the web UI built from tools/ui
# (voice, TTS and STT settings), never the upstream UI from the Hugging Face bucket.
#   ./build.sh             incremental build
#   ./build.sh --clean     delete build/ first
#   ./build.sh --cpu       without CUDA
#   ./build.sh -j8         8 parallel jobs (default: all cores)
# CUDA: $CUDA_HOME when it holds nvcc, else the newest /usr/local/cuda-*, else nvcc on PATH.
# GPU architecture: what nvidia-smi reports (12.0 -> 120), else "native".
# Node >= 20.19 builds the UI; with an older or no Node (Ubuntu 24.04 has 18) a pinned
# Node is fetched once into ~/.cache/llama.cpp-build. Needs: cmake, g++, curl, libssl-dev
# (sudo apt install build-essential cmake curl libssl-dev).
set -euo pipefail
cd "$(dirname "$0")"

NODE_VERSION=v24.21.0
CLEAN=0
CUDA=auto
JOBS=$(nproc)
for arg in "$@"; do
    case $arg in
        --clean) CLEAN=1 ;;
        --cpu) CUDA=off ;;
        -j*) JOBS=${arg#-j} ;;
        -h|--help) sed -n 2,13p "$0"; exit 0 ;;
        *) echo "[ERROR] unknown option: $arg"; exit 2 ;;
    esac
done

die() { echo "[ERROR] $*" >&2; exit 1; }
command -v cmake >/dev/null || die "cmake not found (sudo apt install build-essential cmake)"

# ===== Node for the UI build
node_ok() {
    command -v node >/dev/null && node -e '
        const [a, b] = process.versions.node.split(".").map(Number);
        process.exit(a > 22 || (a === 22 && b >= 12) || (a === 20 && b >= 19) ? 0 : 1)'
}
if ! node_ok; then
    case $(uname -m) in x86_64) arch=x64 ;; aarch64) arch=arm64 ;; *) arch=$(uname -m) ;; esac
    dir="${XDG_CACHE_HOME:-$HOME/.cache}/llama.cpp-build/node-$NODE_VERSION-linux-$arch"
    if [ ! -x "$dir/bin/node" ]; then
        echo "[Setup] Node $(node --version 2>/dev/null || echo "is missing and") too old for the UI: fetching Node $NODE_VERSION"
        tmp=$(mktemp -d)
        trap 'rm -rf "$tmp"' EXIT
        file=node-$NODE_VERSION-linux-$arch.tar.xz
        curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/$file" -o "$tmp/$file"
        curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/SHASUMS256.txt" | grep " $file\$" \
            | (cd "$tmp" && sha256sum -c --quiet -) || die "checksum of $file does not match"
        mkdir -p "$dir"
        tar -xJf "$tmp/$file" -C "$dir" --strip-components=1
    fi
    export PATH="$dir/bin:$PATH"
fi
echo "[OK] node $(node --version), npm $(npm --version)"

# ===== CUDA
cuda_args=()
if [ "$CUDA" = auto ]; then
    nvcc=""
    if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
        nvcc=$CUDA_HOME/bin/nvcc
    else
        nvcc=$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)
        [ -n "$nvcc" ] || nvcc=$(command -v nvcc || true)
    fi
    if [ -n "$nvcc" ]; then
        CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$nvcc")")")
        export CUDA_HOME CUDA_PATH=$CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export LIBRARY_PATH="$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
        gpu_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)
        gpu_arch=${gpu_arch:-native}
        echo "[OK] CUDA $("$nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p') at $CUDA_HOME, architecture $gpu_arch"
        cuda_args=(
            -DGGML_CUDA=ON
            "-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc"
            "-DCMAKE_CUDA_ARCHITECTURES=$gpu_arch"
            -DGGML_CUDA_FA_QUANTS=all
            -DGGML_CUDA_FORCE_MMQ=ON
            -DGGML_CUDA_FORCE_CUBLAS=OFF
            "-DCMAKE_INSTALL_RPATH=$CUDA_HOME/lib64;\$ORIGIN"
            -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
        )
    else
        echo "[Info] no CUDA toolkit found: CPU build"
    fi
fi

# ===== Configure and build
[ "$CLEAN" = 1 ] && rm -rf build
# a dist/ in the UI sources wins over the npm build; it is only ever a stale copy
rm -rf tools/ui/dist

cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=ON \
    "${cuda_args[@]}" \
    -DCMAKE_CXX_FLAGS="-march=native -mtune=native -fomit-frame-pointer -flto"
cmake --build build --config Release --parallel "$JOBS"

# ===== Check what was built
grep -rqs ttsVoice build/tools/ui/dist \
    || die "the embedded web UI has no voice settings: the UI build failed (see 'UI:' lines above)"
build/bin/llama-server --help 2>/dev/null | grep -q -- --asr-url \
    || die "build/bin/llama-server has no --asr-url option"
echo "[OK] build/bin/llama-server with this fork's web UI (voice, TTS, STT settings)"
