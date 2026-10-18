"""Reasoning-effort benchmark for local OpenAI-compatible endpoints (llama.cpp).

Works with any model, with or without reasoning support. Per run it:

1. reads the model name from /v1/models
2. reads /props for the server-reported reasoning_efforts
3. runs the prompt once per level:
   - model reports efforts: "none" + each reported effort (in that order)
   - model reports none: a single default run (no reasoning_effort sent)
4. streams each run and logs total time, reasoning time, content time,
   reasoning/content/combined chars, finish reason, token usage and the
   server timings (prompt/decode tokens per second, draft stats)

Results are written per model to <out-dir>/<model-name>.toml (TOML, appended
as new [[runs]] entries) and a human readable <out-dir>/report.html is
regenerated, listing every model and run with speed and stats.

Usage:
    python tools/reasoning-benchmark.py
    python tools/reasoning-benchmark.py --url http://127.0.0.1:8080 --max-tokens 8000
    python tools/reasoning-benchmark.py --levels none,low --repeat 2

Requires: requests (pip install requests)
"""
import argparse
import json
import math
import re
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = SCRIPT_DIR / "32.txt"
DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_OUT_DIR = SCRIPT_DIR / "reasoning-bench"
FALLBACK_LEVELS = ["none", "low", "medium", "xhigh"]


# ---------------------------------------------------------------- endpoint

def fetch_model_name(session: requests.Session, url: str) -> str:
    r = session.get(f"{url}/v1/models", timeout=30)
    r.raise_for_status()
    data = r.json()
    models = data.get("data") or data.get("models") or []
    if not models:
        raise RuntimeError("no models reported by /v1/models")
    return models[0].get("id") or models[0].get("name")


def fetch_props(session: requests.Session, url: str) -> dict:
    try:
        r = session.get(f"{url}/props", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------- running

def run_level(
    session: requests.Session,
    url: str,
    model: str,
    prompt: str,
    level: str | None,
    max_tokens: int,
    timeout: int,
) -> dict:
    """One streaming run. level=None sends no reasoning_effort field."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if level is not None:
        body["reasoning_effort"] = level

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    t_start = time.perf_counter()
    t_first_token = None
    t_reasoning_end = None
    t_content_start = None
    t_end = None
    finish_reason = None
    usage = None
    timings = None
    error = None

    try:
        with session.post(f"{url}/v1/chat/completions", json=body, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)

                if chunk.get("usage"):
                    usage = chunk["usage"]
                if chunk.get("timings"):
                    timings = chunk["timings"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                now = time.perf_counter()

                rc = delta.get("reasoning_content")
                c = delta.get("content")
                if rc:
                    reasoning_parts.append(rc)
                    t_reasoning_end = now
                    if t_first_token is None:
                        t_first_token = now
                if c:
                    content_parts.append(c)
                    if t_content_start is None:
                        t_content_start = now
                    if t_first_token is None:
                        t_first_token = now
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                    t_end = now
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        t_end = time.perf_counter()

    reasoning_content = "".join(reasoning_parts)
    content = "".join(content_parts)
    reasoning_chars = len(reasoning_content)
    content_chars = len(content)

    total_time = (t_end or time.perf_counter()) - t_start
    reasoning_time = (t_reasoning_end - t_start) if t_reasoning_end else 0.0
    content_time = (t_end - t_content_start) if (t_end and t_content_start) else 0.0
    ttft = (t_first_token - t_start) if t_first_token else 0.0

    return {
        "level": level or "default",
        "status": "error" if error else "ok",
        "error": error,
        "total_time_s": round(total_time, 2),
        "ttft_s": round(ttft, 2),
        "reasoning_time_s": round(reasoning_time, 2),
        "content_time_s": round(content_time, 2),
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "combined_chars": reasoning_chars + content_chars,
        "finish_reason": finish_reason,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "cached_tokens": (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens"),
        "prompt_tps": round((timings or {}).get("prompt_per_second", 0.0), 1),
        "gen_tps": round((timings or {}).get("predicted_per_second", 0.0), 1),
        "draft_n": (timings or {}).get("draft_n"),
        "draft_n_accepted": (timings or {}).get("draft_n_accepted"),
        "content_head": content[:200].replace("\n", " "),
    }


# ---------------------------------------------------------------- toml

def toml_escape(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{s}"'


def toml_val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "0.0"
        return repr(v)
    if v is None:
        return "0"
    if isinstance(v, str):
        return toml_escape(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_val(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value: {type(v)}")


def toml_table(header: str, kv: dict) -> list[str]:
    lines = [header]
    for k, v in kv.items():
        if v is None:
            continue
        lines.append(f"{k} = {toml_val(v)}")
    return lines


def emit_toml(model: dict, runs: list[dict]) -> str:
    lines = toml_table("[model]", {
        "name": model["name"],
        "endpoint": model["endpoint"],
        "build_info": model.get("build_info") or "",
        "reasoning_efforts": model.get("reasoning_efforts") or [],
    })
    for run in runs:
        lines.append("")
        lines += toml_table("[[runs]]", {
            "timestamp": run["timestamp"],
            "prompt_file": run["prompt_file"],
            "prompt_chars": run["prompt_chars"],
            "max_tokens": run["max_tokens"],
            "repeat": run["repeat"],
        })
        for lv in run["levels"]:
            lines.append("")
            lines += toml_table("[[runs.levels]]", lv)
    return "\n".join(lines) + "\n"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def load_model_file(path: Path) -> tuple[dict, list[dict]]:
    if not path.exists():
        return {}, []
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("model", {}), data.get("runs", [])


# ---------------------------------------------------------------- html

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Reasoning benchmark</title>
<style>
:root { color-scheme: dark; }
body { font-family: "Segoe UI", system-ui, sans-serif; background: #111418; color: #d8dee6; margin: 24px; }
h1 { font-size: 1.4em; margin-bottom: 4px; }
h2 { font-size: 1.1em; margin: 24px 0 8px; }
.meta { color: #8b98a5; font-size: 0.85em; margin-bottom: 16px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 20px; font-size: 0.9em; }
th, td { border: 1px solid #2a3138; padding: 5px 8px; text-align: right; white-space: nowrap; }
th { background: #1a2027; position: sticky; top: 0; }
td.name, th.name { text-align: left; }
tr:hover td { background: #1a2027; }
.ok { color: #7ec699; } .err { color: #e08585; }
details { margin: 12px 0; border: 1px solid #2a3138; border-radius: 6px; }
summary { cursor: pointer; padding: 10px 14px; font-weight: 600; background: #171b21; }
details[open] summary { border-bottom: 1px solid #2a3138; }
.model-body { padding: 10px 14px; }
.model-meta { color: #8b98a5; font-size: 0.85em; margin-bottom: 8px; }
.head { max-width: 480px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
footer { color: #5c6773; font-size: 0.8em; margin-top: 24px; }
</style>
</head>
<body>
<h1>Reasoning benchmark</h1>
<div class="meta" id="meta"></div>
<h2>All models</h2>
<table id="overview">
<tr><th class="name">model</th><th>level</th><th>total s</th><th>reason s</th><th>content s</th>
<th>reason ch</th><th>content ch</th><th>combined</th><th>gen t/s</th><th>prompt t/s</th><th>finish</th></tr>
</table>
<h2>Models</h2>
<div id="models"></div>
<footer>Data source: <span id="src"></span> (embedded at generation time)</footer>
<script>
const DATA = __DATA__;
const esc = (s) => String(s ?? "")
  .replace(/&/g, "&" + "amp;")
  .replace(/</g, "&" + "lt;")
  .replace(/>/g, "&" + "gt;")
  .replace(/"/g, "&" + "quot;")
  .replace(/'/g, "&" + "#39;");

document.getElementById("meta").textContent =
  `generated ${DATA.generated} | prompt ${esc(DATA.prompt_file)} (${DATA.prompt_chars} chars) | max_tokens ${DATA.max_tokens}`;
document.getElementById("src").textContent = DATA.source;

const ov = document.getElementById("overview");
for (const m of DATA.models) {
  for (const run of m.runs) {
    for (const l of run.levels) {
      ov.insertAdjacentHTML("beforeend",
        `<tr class="${l.status === "ok" ? "ok" : "err"}">
         <td class="name">${esc(m.name)}</td><td>${esc(l.level)}</td>
         <td>${l.total_time_s}</td><td>${l.reasoning_time_s}</td><td>${l.content_time_s}</td>
         <td>${l.reasoning_chars}</td><td>${l.content_chars}</td><td>${l.combined_chars}</td>
         <td>${l.gen_tps}</td><td>${l.prompt_tps}</td><td>${esc(l.finish_reason)}</td></tr>`);
    }
  }
}

const box = document.getElementById("models");
for (const m of DATA.models) {
  const d = document.createElement("details");
  const efforts = (m.reasoning_efforts || []).join(", ") || "none reported";
  d.innerHTML =
    `<summary>${esc(m.name)} <span style="color:#8b98a5;font-weight:400">` +
    `${m.runs.length} run(s) | efforts: ${esc(efforts)}</span></summary>` +
    `<div class="model-body">` +
    `<div class="model-meta">endpoint ${esc(m.endpoint)} | build ${esc(m.build_info || "n/a")}</div>`;
  for (const run of m.runs) {
    d.innerHTML +=
      `<h3 style="font-size:0.95em;margin:14px 0 4px">run ${esc(run.timestamp)} ` +
      `<span style="color:#8b98a5">(${esc(run.prompt_file)}, max_tokens ${run.max_tokens})</span></h3>` +
      `<table><tr><th class="name">level</th><th>status</th><th>total s</th><th>ttft s</th>` +
      `<th>reason s</th><th>content s</th><th>reason ch</th><th>content ch</th><th>combined</th>` +
      `<th>gen t/s</th><th>prompt t/s</th><th>prompt tok</th><th>compl tok</th><th>cached</th>` +
      `<th>draft acc</th><th>finish</th><th>content</th></tr>`;
    for (const l of run.levels) {
      const draft = (l.draft_n != null) ? `${l.draft_n_accepted}/${l.draft_n}` : "";
      d.innerHTML +=
        `<tr class="${l.status === "ok" ? "ok" : "err"}">` +
        `<td class="name">${esc(l.level)}</td><td>${esc(l.status)}${l.error ? " " + esc(l.error) : ""}</td>` +
        `<td>${l.total_time_s}</td><td>${l.ttft_s}</td>` +
        `<td>${l.reasoning_time_s}</td><td>${l.content_time_s}</td>` +
        `<td>${l.reasoning_chars}</td><td>${l.content_chars}</td><td>${l.combined_chars}</td>` +
        `<td>${l.gen_tps}</td><td>${l.prompt_tps}</td>` +
        `<td>${l.prompt_tokens ?? ""}</td><td>${l.completion_tokens ?? ""}</td><td>${l.cached_tokens ?? ""}</td>` +
        `<td>${draft}</td><td>${esc(l.finish_reason)}</td>` +
        `<td class="head" title="${esc(l.content_head)}">${esc(l.content_head)}</td></tr>`;
    }
    d.innerHTML += `</table>`;
  }
  d.innerHTML += `</div>`;
  box.appendChild(d);
}
</script>
</body>
</html>
"""


def build_html(meta: dict, models: list[dict], source: str) -> str:
    data = {
        "generated": meta["generated"],
        "prompt_file": meta["prompt_file"],
        "prompt_chars": meta["prompt_chars"],
        "max_tokens": meta["max_tokens"],
        "source": source,
        "models": models,
    }
    return HTML_TEMPLATE.replace("__DATA__", json.dumps(data, default=str))


# ---------------------------------------------------------------- main

def print_table(rows: list[dict]) -> None:
    header = (
        f"{'level':<10} {'status':<6} {'total_s':>8} {'reason_s':>9} {'content_s':>9} "
        f"{'reason_ch':>9} {'content_ch':>9} {'combined':>9} {'gen_tps':>8} {'finish':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['level']:<10} {r['status']:<6} {r['total_time_s']:>8} "
            f"{r['reasoning_time_s']:>9} {r['content_time_s']:>9} "
            f"{r['reasoning_chars']:>9} {r['content_chars']:>9} "
            f"{r['combined_chars']:>9} {r['gen_tps']:>8} {str(r['finish_reason']):<8}"
        )
        if r["error"]:
            print(f"  error: {r['error']}")
        if r["prompt_tokens"] is not None:
            print(
                f"  tokens: prompt={r['prompt_tokens']} completion={r['completion_tokens']} "
                f"cached={r['cached_tokens']} draft={r.get('draft_n_accepted')}/{r.get('draft_n')}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help=f"server base URL (default {DEFAULT_URL})")
    ap.add_argument("--model", default=None, help="model name (default: first from /v1/models)")
    ap.add_argument("--prompt-file", default=str(DEFAULT_PROMPT), help="prompt file (default tools/32.txt)")
    ap.add_argument("--max-tokens", type=int, default=8000, help="max_tokens per run (default 8000)")
    ap.add_argument("--levels", default=None,
                    help="comma-separated levels in run order (default: none + server-reported reasoning_efforts; "
                         "a model reporting no efforts gets a single default run)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per level (default 1)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-request timeout in s (default 1800)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help=f"results dir (default {DEFAULT_OUT_DIR})")
    args = ap.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    session = requests.Session()

    model = args.model or fetch_model_name(session, args.url)
    props = fetch_props(session, args.url)
    reported = props.get("reasoning_efforts") or []

    if args.levels:
        levels = [x.strip() for x in args.levels.split(",") if x.strip()]
        level_source = "user"
    elif reported:
        levels = ["none"] + list(reported)
        level_source = "server-reported"
    else:
        levels = ["default"]
        level_source = "default (model reports no reasoning_efforts)"

    print(f"endpoint      : {args.url}")
    print(f"model         : {model}")
    print(f"build_info    : {props.get('build_info', 'n/a')}")
    print(f"reasoning_efforts (reported): {reported or '[]'}")
    print(f"levels        : {levels} [{level_source}]")
    print(f"prompt        : {args.prompt_file} ({len(prompt)} chars)")
    print(f"max_tokens    : {args.max_tokens}, repeat: {args.repeat}")
    print()

    rows: list[dict] = []
    for level in levels:
        for i in range(args.repeat):
            tag = level if args.repeat == 1 else f"{level}#{i + 1}"
            print(f"--- running level '{tag}' ...", flush=True)
            # "default" means: send no reasoning_effort field at all
            row = run_level(session, args.url, model, prompt, None if level == "default" else level, args.max_tokens, args.timeout)
            if args.repeat > 1:
                row["level"] = tag
            rows.append(row)
            print(
                f"    total={row['total_time_s']}s reason={row['reasoning_time_s']}s "
                f"content={row['content_time_s']}s reason_ch={row['reasoning_chars']} "
                f"content_ch={row['content_chars']} gen={row['gen_tps']}t/s finish={row['finish_reason']}"
            )
            if row["error"]:
                print(f"    error: {row['error']}")
            print(flush=True)

    print()
    print_table(rows)

    # sanity checks on level selection
    problems = []
    for r in rows:
        if r["status"] != "ok":
            problems.append(f"{r['level']}: request error ({r['error']})")
            continue
        if r["level"] == "none" and r["reasoning_chars"] > 0:
            problems.append(f"none: expected 0 reasoning chars, got {r['reasoning_chars']}")
        if r["level"] not in ("none", "default") and r["reasoning_chars"] == 0:
            problems.append(f"{r['level']}: expected reasoning chars > 0, got 0")
        if r["content_chars"] == 0 and r["finish_reason"] == "stop":
            problems.append(f"{r['level']}: finish=stop but content is empty")
    if problems:
        print("SANITY CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("SANITY CHECK PASSED.")

    # write per-model TOML (append this run) and regenerate the HTML report
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / f"{sanitize_name(model)}.toml"
    model_info, runs = load_model_file(model_file)
    runs.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt_file": args.prompt_file,
        "prompt_chars": len(prompt),
        "max_tokens": args.max_tokens,
        "repeat": args.repeat,
        "levels": rows,
    })
    model_entry = {
        "name": model,
        "endpoint": args.url,
        "build_info": props.get("build_info"),
        "reasoning_efforts": reported,
    }
    model_file.write_text(emit_toml(model_entry, runs), encoding="utf-8")

    # aggregate all model files for the report
    all_models = []
    for f in sorted(out_dir.glob("*.toml")):
        m, r = load_model_file(f)
        if m:
            all_models.append({**m, "runs": r})
    report = out_dir / "report.html"
    report.write_text(
        build_html(
            {"generated": datetime.now().isoformat(timespec="seconds"),
             "prompt_file": args.prompt_file, "prompt_chars": len(prompt),
             "max_tokens": args.max_tokens},
            all_models,
            source=model_file.name,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved: {model_file}\nsaved: {report}")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
