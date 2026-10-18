#!/usr/bin/env bash
#
# build.sh -- build llama.cpp for the local GPU (RTX 5090, sm_120) with CUDA.
#
# This is the full build the Aegis LLM engine expects: llama-server with
# GGML_CUDA + all-quant flash-attention + forced MMQ, tuned for a Blackwell
# RTX 5090. It can be run manually:
#
#     cd /home/john/AI/llama.cpp && ./build.sh
#
# Tunables (all optional, override on the command line):
#     CUDA_HOME=/path/to/cuda   force a specific CUDA toolkit (bin/nvcc + lib64)
#     JOBS=16                  parallel build jobs (default: all cores)
#     KEEP_BUILD=1             don't wipe the existing build/ before configuring
#
set -euo pipefail
cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# 1) Pick a CUDA toolkit (nvcc + lib64).
#    The RTX 5090 is sm_120; any CUDA 12.8+ / 13.x that targets sm_120 works.
#    The first usable toolkit wins. Set CUDA_HOME to force one.
# -----------------------------------------------------------------------------
find_cuda() {
    local c
    for c in "${CUDA_HOME:-}" \
             /usr/local/cuda \
             /usr/local/cuda-13.4 \
             /usr/local/cuda-13.3 \
             /usr/local/cuda-13 \
             "$HOME/cuda-staging"; do
        [ -n "$c" ] || continue
        if [ -x "$c/bin/nvcc" ] && [ -d "$c/lib64" ]; then
            printf '%s' "$c"
            return 0
        fi
    done
    return 1
}

if ! CUDA_PATH="$(find_cuda)"; then
    echo "No CUDA toolkit found. Point CUDA_HOME at a folder with bin/nvcc and lib64." >&2
    echo "(checked: \$CUDA_HOME, ~/cuda-staging, /usr/local/cuda-13.4, -13.3, -13, /usr/local/cuda)" >&2
    exit 1
fi
NVCC="$CUDA_PATH/bin/nvcc"
echo "Using CUDA toolkit: $CUDA_PATH"
"$NVCC" --version 2>/dev/null | tail -n 2 || true

# Environment the CUDA build needs (mirrors the original build's exports).
export CUDA_HOME="$CUDA_PATH"
export CUDA_PATH
export PATH="${CUDA_PATH}/bin:${PATH}"
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:/usr/share/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig"
export LIBRARY_PATH="/usr/local/lib:${CUDA_PATH}/lib64:${CUDA_PATH}/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

JOBS="${JOBS:-$(nproc)}"

# -----------------------------------------------------------------------------
# 2) Configure + build.
#    sm_120 (RTX 5090), all-quant flash attention, forced MMQ. The install
#    rpath bakes in this toolkit's lib64 so the binaries run without the env.
# -----------------------------------------------------------------------------
if [ "${KEEP_BUILD:-0}" != "1" ]; then
    echo "Wiping build/ ..."
    rm -rf build
fi

echo "Configuring (cmake) ..."
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
    -DCMAKE_CUDA_ARCHITECTURES="120" \
    -DCMAKE_CUDA_COMPILER="$NVCC" \
    -DCMAKE_INSTALL_RPATH="${CUDA_PATH}/lib64;\$ORIGIN" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_CXX_FLAGS="-march=native -mtune=native -fomit-frame-pointer -flto"

echo "Building (cmake, $JOBS jobs) ..."
if ! cmake --build build --config Release --parallel "$JOBS"; then
    # A hiccup in the embedded web UI (npm/HF download of llama-ui) must not block the
    # engines Aegis starts. Rebuild the engine targets alone.
    echo
    echo "Full build failed; retrying the engine targets (llama-server and friends) ..."
    cmake --build build --config Release --parallel "$JOBS" \
        --target llama-server llama-bench llama-quantize llama-cli
fi

# -----------------------------------------------------------------------------
# 3) Verify the pieces Aegis needs are present.
# -----------------------------------------------------------------------------
echo
echo "=== built ==="
ls -la build/bin/llama-server 2>/dev/null || echo "(no llama-server in build/bin)"
# the speech tools only exist on the video-audio-pipeline branch, not plain master
for t in build/bin/llama-server build/bin/llama-bench; do
    [ -x "$t" ] && echo "ok: $t"
done
echo
echo "Done. Test the router (loads a model, needs a GGUF path):"
echo "  ./build/bin/llama-server --models-preset <presets.ini> --host 127.0.0.1 --port 8080"
echo
echo "Optional vision smoke test (fill in a GGUF + mmproj path you have):"
echo "  python3 - <<'PY'"
echo "  from llama_cpp import Llama"
echo "  llm = Llama(model_path='<path/to/model.gguf>', mmproj_path='<path/to/mmproj.gguf>',"
echo "              n_gpu_layers=-1, verbose=False)"
echo "  print('vision OK')"
echo "  PY"
