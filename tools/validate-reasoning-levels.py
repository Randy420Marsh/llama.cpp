"""Quick validation that reasoning_effort level selection works on the local endpoint.

Queries /props for the server-reported reasoning_efforts, then runs a tiny
prompt at "none" + each reported level, plus one unsupported level to confirm
the template rejects it.
"""
import json
import sys
import time

import requests

BASE = "http://127.0.0.1:8080"
PROMPT = "What is 27 * 43? Answer with just the number."


def fetch_model_name() -> str:
    r = requests.get(f"{BASE}/v1/models", timeout=30)
    r.raise_for_status()
    data = r.json()
    models = data.get("data") or data.get("models") or []
    if not models:
        raise RuntimeError("no models reported by /v1/models")
    return models[0].get("id") or models[0].get("name")


def fetch_reported_efforts() -> list[str]:
    r = requests.get(f"{BASE}/props", timeout=30)
    r.raise_for_status()
    return r.json().get("reasoning_efforts") or []


def run(model: str, level: str | None, max_tokens: int = 120) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if level is not None:
        body["reasoning_effort"] = level
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=600)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return {"level": level, "status": r.status_code, "error": r.text[:300], "time_s": round(dt, 2)}
    data = r.json()
    msg = data["choices"][0]["message"]
    rc = msg.get("reasoning_content", "") or ""
    content = msg.get("content", "") or ""
    return {
        "level": level,
        "status": 200,
        "time_s": round(dt, 2),
        "finish_reason": data["choices"][0].get("finish_reason"),
        "reasoning_chars": len(rc),
        "content_chars": len(content),
        "content_head": content[:120].replace("\n", " "),
        "usage": data.get("usage"),
    }


def main() -> None:
    model = fetch_model_name()
    reported = fetch_reported_efforts()
    print(f"model: {model}")
    print(f"server-reported reasoning_efforts: {reported}")

    # levels to test: none + all reported levels
    levels = ["none"] + reported
    # one level the template must reject (pick the first ladder value not reported)
    ladder = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
    unsupported = next((l for l in ladder if l not in levels), None)

    results = []
    for level in levels:
        res = run(model, level)
        results.append(res)
        print(json.dumps(res, indent=2))
        print("-" * 60)

    if unsupported:
        res = run(model, unsupported)
        results.append(res)
        print(f"unsupported level '{unsupported}':")
        print(json.dumps(res, indent=2))
        print("-" * 60)

    # sanity checks
    problems = []
    for res in results:
        lvl = res["level"]
        if lvl == "none":
            if res["status"] != 200:
                problems.append(f"none: HTTP {res['status']}")
            elif res["reasoning_chars"] > 0:
                problems.append(f"none: expected 0 reasoning chars, got {res['reasoning_chars']}")
        elif lvl in reported:
            if res["status"] != 200:
                problems.append(f"{lvl}: HTTP {res['status']} ({res.get('error')})")
            elif res["reasoning_chars"] == 0:
                problems.append(f"{lvl}: expected reasoning chars > 0, got 0")
        elif lvl == unsupported:
            if res["status"] == 200:
                problems.append(f"unsupported '{lvl}': expected an error, got 200")
    if problems:
        print("VALIDATION FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("VALIDATION PASSED: 'none' has no reasoning, reported levels do, unsupported level rejected.")


if __name__ == "__main__":
    main()
