"""Shared helper: make the nvidia-* wheels' cuBLAS/cuDNN visible to CTranslate2.

faster-whisper (CTranslate2) dlopens libcublas/libcudnn, which the nvidia pip
wheels ship outside the default library path. LD_LIBRARY_PATH is read only at
process start, so preload them with ctypes RTLD_GLOBAL *before* the model loads.
Used by both asr_server.py and transcribe_orpheus_clips.py.
"""
import sys
from pathlib import Path


def add_cuda_dll_dirs() -> None:
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    roots = [Path(p) for p in nvidia.__path__]
    if sys.platform != "win32":
        import ctypes
        for root in roots:
            for lib in sorted(root.glob("*/lib/lib*.so.*")):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        return
    # Windows: the CTranslate2 DLLs also look under the nvidia wheel dirs
    import os
    os.environ["PATH"] = os.pathsep.join(
        [os.path.join(r, m, "bin") for r in roots for m in ("cublas", "cudnn", "nvrtc")]
    ) + os.pathsep + os.environ.get("PATH", "")


add_cuda_dll_dirs()
