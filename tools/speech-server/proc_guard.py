"""Tie every child process (llama-server, Orpheus llama-server, image worker) to this server's life.

Windows: children are adopted into a Job Object with KILL_ON_JOB_CLOSE. The job handle closes when
this process ends for any reason - normal exit, closed console window, crash, Task Manager - and
Windows then kills every process still in the job, so no server is left holding VRAM.
Linux: pass preexec_fn=child_preexec to Popen; the child gets SIGTERM when this process dies.
"""
from __future__ import annotations

import ctypes
import sys

_job = None


def _win_job():
    global _job
    if _job is not None:
        return _job
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in
                    ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # ExtendedLimitInformation
        raise ctypes.WinError(ctypes.get_last_error())
    _job = (k32, job)  # the handle stays open until this process exits
    return _job


def adopt(proc) -> None:
    """Make a freshly started subprocess.Popen die together with this process (Windows)."""
    if sys.platform != "win32" or proc is None:
        return
    try:
        k32, job = _win_job()
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception as e:  # never block a launch on the guard; the shutdown hooks still stop it
        print(f"[proc_guard] could not tie PID {proc.pid} to this process: {e}", flush=True)


def child_preexec() -> None:
    """Linux preexec_fn: SIGTERM the child when the parent dies."""
    try:
        import signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


# preexec_fn is POSIX-only; pass this so the same Popen call works on both
PREEXEC = child_preexec if sys.platform.startswith("linux") else None
