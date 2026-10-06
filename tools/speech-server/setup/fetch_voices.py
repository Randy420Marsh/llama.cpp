"""
Download speech models ahead of time, so offline use (the default) has
everything it needs. The same catalog as Settings -> Voice -> Models.

    python setup/fetch_voices.py --list                 what is on disk / downloadable
    python setup/fetch_voices.py                        the recommended model of every
                                                        usable engine (+ Piper voices)
    python setup/fetch_voices.py --all                  everything in the catalog
    python setup/fetch_voices.py llamacpp/Qwen3-TTS-12Hz-1.7B-Base-Q8_0 piper/de_DE-thorsten-high

Files land where speech_paths.py says (see README "Where the models go").
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["TTS_OFFLINE"] = "0"                 # must be set before tts_engines
for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.pop(k, None)

# the server modules live one level up; this script is a one-off, not the server
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import speech_paths                            # noqa: E402
import tts_engines                             # noqa: E402


def _mb(n: float) -> str:
    return f"{n / 1000:.1f} GB" if n >= 1000 else f"{n:.0f} MB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", nargs="*", help="engine/id items to download")
    ap.add_argument("--list", action="store_true", help="report only, download nothing")
    ap.add_argument("--all", action="store_true", help="every catalog item, not just the recommended")
    args = ap.parse_args()

    cat = tts_engines.catalog()
    print("folders:")
    for k, v in speech_paths.describe().items():
        print(f"  {k:11} {v}")
    print()

    if args.list:
        for e in cat["engines"]:
            state = "ready" if e["installed"] else f"python packages missing ({e['detail']})"
            print(f"{e['label']}: {state}")
            for i in e["items"]:
                mark = "on disk " if i["installed"] else "download"
                rec = " (recommended)" if i["recommended"] else ""
                print(f"    [{mark}] {e['engine']}/{i['id']}  {_mb(i['size_mb'])}{rec}")
        return 0

    wanted: list[tuple[str, str]] = []
    if args.items:
        for it in args.items:
            if "/" not in it:
                print(f"skip '{it}': use engine/id, see --list")
                continue
            eng, iid = it.split("/", 1)
            wanted.append((eng, iid))
    else:
        for e in cat["engines"]:
            # models for an engine whose Python packages are missing would sit unused
            if not e["installed"]:
                continue
            for i in e["items"]:
                if not i["installed"] and (args.all or i["recommended"]) and i["source"] != "local":
                    wanted.append((e["engine"], i["id"]))

    if not wanted:
        print("nothing to download")
        return 0

    failed = 0
    for eng, iid in wanted:
        print(f"downloading {eng}/{iid} ...", flush=True)
        res = tts_engines.start_download(eng, iid, wait=True)
        if res.get("error") or not res.get("ok"):
            failed += 1
            print(f"    FAILED: {res.get('error')}")
        else:
            print("    ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
