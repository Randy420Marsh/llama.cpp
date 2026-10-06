#!/usr/bin/env python3
"""Reasoning benchmark for OpenAI-compatible servers such as llama-server.

The reasoning levels come from the server: /props reports "reasoning_efforts" (the levels the chat template accepts).

Modes:
  (default)  send the whole prompt file once per level: "none" plus every reported level
  --split    send each numbered question of the prompt file on its own and score the answers with the key file
             (default level: the highest one the server reports)
  --quick    check level selection with a tiny prompt: "none" gives no reasoning, every reported level gives some,
             and a level the template does not accept is rejected

Every run is appended to <out-dir>/<model>.toml, the answers of --split runs go to <out-dir>/answers/, and
<out-dir>/report.html is rebuilt from all model files.

Examples:
  python scripts/reasoning-bench.py --quick
  python scripts/reasoning-bench.py --split
  python scripts/reasoning-bench.py --split --questions 1-4,18 --level medium
  python scripts/reasoning-bench.py --prompt-file my-prompt.txt --levels none,low

Requires Python 3.11+ and requests.
"""
import argparse
import html
import json
import math
import os
import re
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = SCRIPT_DIR / "reasoning-bench-prompt.txt"
DEFAULT_KEY = SCRIPT_DIR / "reasoning-bench-key.json"
DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_OUT_DIR = Path.home() / "reasoning-bench-results"
LADDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
QUICK_PROMPT = "What is 27 * 43? Answer with just the number."
CODE_BLOCK = re.compile(r"```[^\n]*\n(?:[^\n]*\n){4,}?\s*```")


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


def highest_level(reported: list[str]) -> str:
    known = [lv for lv in LADDER if lv in reported]
    if known:
        return known[-1]
    # unknown names: trust the server's order
    return reported[-1] if reported else "default"


# ---------------------------------------------------------------- one request

def run_request(session: requests.Session, url: str, model: str, prompt: str, level: str | None,
                max_tokens: int, timeout: int) -> tuple[dict, str, str]:
    """One streaming request; level None sends no reasoning_effort, max_tokens <= 0 sends no limit.

    Returns (row, content, reasoning).
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if max_tokens > 0:
        body["max_tokens"] = max_tokens
    if level is not None:
        body["reasoning_effort"] = level

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    t_start = time.perf_counter()
    t_first = t_reasoning_end = t_content_start = t_end = None
    finish_reason = usage = timings = error = None
    http_status = None

    try:
        with session.post(f"{url}/v1/chat/completions", json=body, stream=True, timeout=timeout) as r:
            http_status = r.status_code
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
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
                if chunk.get("error"):
                    raise RuntimeError(f"server error: {chunk['error']}")
                usage = chunk.get("usage") or usage
                timings = chunk.get("timings") or timings
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
                    t_first = t_first or now
                if c:
                    content_parts.append(c)
                    t_content_start = t_content_start or now
                    t_first = t_first or now
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                    t_end = now
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    t_end = t_end or time.perf_counter()
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    usage = usage or {}
    timings = timings or {}
    row = {
        "level": level or "default",
        "status": "error" if error else "ok",
        "http_status": http_status,
        "error": error,
        "total_time_s": round(t_end - t_start, 2),
        "ttft_s": round(t_first - t_start, 2) if t_first else None,
        "reasoning_time_s": round(t_reasoning_end - t_start, 2) if t_reasoning_end else 0.0,
        "content_time_s": round(t_end - t_content_start, 2) if t_content_start else 0.0,
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "finish_reason": finish_reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        # prompt_tokens_details may be missing or null
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "prompt_tps": round(timings.get("prompt_per_second") or 0.0, 1),
        "gen_tps": round(timings.get("predicted_per_second") or 0.0, 1),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "content_head": content[:200].replace("\n", " "),
    }
    return row, content, reasoning


# ---------------------------------------------------------------- questions and scoring

def split_questions(text: str) -> dict[int, str]:
    """Numbered paragraphs ("12. ...") of the prompt file; lines before the first one are the instructions."""
    questions: dict[int, str] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
        if m:
            current = int(m.group(1))
            questions[current] = m.group(2).strip()
        elif current is not None and line.strip():
            questions[current] += " " + line.strip()
        elif not line.strip():
            current = None
    return questions


# headings a model uses for "answer 12" in one long reply, tried in this order
HEADING_PATTERNS = [
    r"(?m)^[ \t]{0,3}#{1,6}[ \t]*(?:\*\*)?[ \t]*(?:question[ \t]*|q)?(\d{1,2})\b",
    r"(?m)^[ \t]{0,3}\*\*[ \t]*(?:question[ \t]*|q)?(\d{1,2})[.):]",
    r"(?m)^[ \t]{0,3}(?:question[ \t]*|q)?(\d{1,2})[.)][ \t]",
]


def split_answer(content: str, count: int) -> dict[int, str]:
    """Cut one reply to a numbered prompt into its numbered sections.

    Headings must come in order; a gap of up to two numbers is allowed (a question left out), so numbered
    lists inside an answer are not mistaken for the next question.
    """
    best: list[tuple[int, int]] = []
    for pat in HEADING_PATTERNS:
        found, want = [], 1
        for m in re.finditer(pat, content, re.I):
            n = int(m.group(1))
            if want <= n <= min(count, want + 2):
                found.append((n, m.start()))
                want = n + 1
        if len(found) > len(best):
            best = found
    return {n: content[start:(best[i + 1][1] if i + 1 < len(best) else len(content))]
            for i, (n, start) in enumerate(best)}


def score_full_record(rec: dict, key: dict[str, dict], count: int) -> list[dict]:
    """Per-question score rows for one whole-prompt reply (trap_in_reasoning looks at the whole reasoning)."""
    sections = split_answer(rec.get("content", ""), count)
    rows = []
    for q in range(1, count + 1):
        text = sections.get(q, "")
        req = {"status": "ok" if text else "missing", "finish_reason": rec.get("finish_reason")}
        rows.append({"question": q, "level": rec.get("level"), "status": req["status"],
                     **score_answer(key.get(str(q), {}), req, text, rec.get("reasoning", ""))})
    return rows


def full_summary(scores: list[dict], requests: list[dict]) -> dict:
    """summarize() over the per-question scores, with time and speed taken from the actual requests."""
    s = summarize(scores)
    ok = [r for r in requests if r.get("status") == "ok"]
    gen = [r["gen_tps"] for r in ok if r.get("gen_tps")]
    drafted = sum(r.get("draft_n") or 0 for r in ok)
    accepted = sum(r.get("draft_n_accepted") or 0 for r in ok)
    s.update(total_time_s=round(sum(r.get("total_time_s") or 0 for r in requests), 1),
             completion_tokens=sum(r.get("completion_tokens") or 0 for r in ok),
             reasoning_chars=sum(r.get("reasoning_chars") or 0 for r in ok),
             content_chars=sum(r.get("content_chars") or 0 for r in ok),
             mean_gen_tps=round(sum(gen) / len(gen), 1) if gen else 0.0,
             draft_acceptance=round(accepted / drafted, 3) if drafted else None)
    return s


def parse_selection(spec: str | None, available: list[int]) -> list[int]:
    if not spec:
        return available
    chosen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            chosen.update(range(int(a), int(b) + 1))
        elif part:
            chosen.add(int(part))
    missing = sorted(chosen - set(available))
    if missing:
        raise SystemExit(f"questions not in the prompt file: {missing}")
    return sorted(chosen)


def load_key(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    key = json.loads(path.read_text(encoding="utf-8")).get("questions", {})
    for q, entry in key.items():
        for field in ("detect", "answer"):
            for p in entry.get(field, []):
                re.compile(p)  # fail early on a broken pattern
    return key


def score_answer(entry: dict, row: dict, content: str, reasoning: str = "") -> dict:
    s = {"answered": row.get("status") == "ok" and row.get("finish_reason") == "stop" and bool(content.strip())}
    if "trap" in entry:
        s["trap_caught"] = any(re.search(p, content, re.I | re.S) for p in entry["detect"])
        # not scored: shows whether a model that ran out of tokens had spotted the trap while thinking
        s["trap_in_reasoning"] = any(re.search(p, reasoning, re.I | re.S) for p in entry["detect"])
    elif "code" in entry:
        s["code_ok"] = bool(CODE_BLOCK.search(content))
    if "answer" in entry:
        s["answer_ok"] = all(re.search(p, content, re.I) for p in entry["answer"])
    s["points"] = sum(1 for k in ("answered", "trap_caught", "code_ok", "answer_ok") if s.get(k))
    s["max_points"] = sum(1 for k in ("answered", "trap_caught", "code_ok", "answer_ok") if k in s)
    return s


def summarize(rows: list[dict]) -> dict:
    def count(k):
        return sum(1 for r in rows if r.get(k)), sum(1 for r in rows if k in r)
    answered, n = count("answered")
    traps, n_traps = count("trap_caught")
    traps_reasoning, _ = count("trap_in_reasoning")
    code, n_code = count("code_ok")
    exact, n_exact = count("answer_ok")
    points = sum(r.get("points", 0) for r in rows)
    max_points = sum(r.get("max_points", 0) for r in rows)
    # rows read back from TOML lack the fields that were null
    ok = [r for r in rows if r.get("status") == "ok"]
    gen = [r["gen_tps"] for r in ok if r.get("gen_tps")]
    drafted = sum(r.get("draft_n") or 0 for r in ok)
    accepted = sum(r.get("draft_n_accepted") or 0 for r in ok)
    return {
        "questions": n, "answered": answered, "traps": n_traps, "traps_caught": traps,
        "traps_in_reasoning": traps_reasoning,
        "code_questions": n_code, "code_ok": code, "exact_questions": n_exact, "exact_ok": exact,
        "points": points, "max_points": max_points,
        "score_pct": round(100.0 * points / max_points, 1) if max_points else 0.0,
        "total_time_s": round(sum(r.get("total_time_s") or 0 for r in rows), 1),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok),
        "reasoning_chars": sum(r.get("reasoning_chars") or 0 for r in ok),
        "content_chars": sum(r.get("content_chars") or 0 for r in ok),
        "mean_gen_tps": round(sum(gen) / len(gen), 1) if gen else 0.0,
        "draft_acceptance": round(accepted / drafted, 3) if drafted else None,
    }


# ---------------------------------------------------------------- toml

def toml_val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return "0.0" if (math.isnan(v) or math.isinf(v)) else repr(v)
    if isinstance(v, str):
        s = v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return f'"{s}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_val(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value: {type(v)}")


def toml_table(header: str, kv: dict) -> list[str]:
    # TOML has no null: missing values are left out
    return [header] + [f"{k} = {toml_val(v)}" for k, v in kv.items() if v is not None and not isinstance(v, dict)]


def emit_toml(model: dict, runs: list[dict]) -> str:
    lines = toml_table("[model]", model)
    for run in runs:
        lines += [""] + toml_table("[[runs]]", {k: v for k, v in run.items() if k not in ("rows", "summary", "scores")})
        if run.get("summary"):
            lines += [""] + toml_table("[runs.summary]", run["summary"])
        for row in run["rows"]:
            lines += [""] + toml_table("[[runs.rows]]", row)
        for row in run.get("scores") or []:
            lines += [""] + toml_table("[[runs.scores]]", row)
    return "\n".join(lines) + "\n"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def load_model_file(path: Path) -> tuple[dict, list[dict]]:
    if not path.exists():
        return {}, []
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("model", {}), data.get("runs", [])


# ---------------------------------------------------------------- html report

CSS = """
:root { color-scheme: light dark; --bg: #f6f7f8; --fg: #1d2226; --muted: #5d6870; --line: #d5dade; --head: #eceff1;
  --good: #1e7b45; --bad: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg: #121518; --fg: #dde3e8; --muted: #8b98a3; --line: #2b3238;
  --head: #1a1f24; --good: #6cc38e; --bad: #ef8a80; } }
body { font: 14px/1.45 "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--fg); margin: 0;
  padding-block: 20px; padding-inline: 16px; }
h1 { font-size: 1.35em; margin: 0 0 4px; } h2 { font-size: 1.1em; margin: 26px 0 8px; }
.muted { color: var(--muted); font-size: 0.9em; }
.wrap { overflow-x: auto; } table { border-collapse: collapse; margin: 6px 0 16px; font-size: 0.92em; }
th, td { border: 1px solid var(--line); padding: 4px 8px; text-align: right; white-space: nowrap; }
th { background: var(--head); } td.l, th.l { text-align: left; } td.w { white-space: normal; min-width: 260px; }
.yes { color: var(--good); font-weight: 600; } .no { color: var(--bad); font-weight: 600; }
details { margin: 8px 0; } summary { cursor: pointer; font-weight: 600; }
"""


def cell(v, cls: str = "") -> str:
    if v is True:
        return '<td class="yes">yes</td>'
    if v is False:
        return '<td class="no">no</td>'
    return f'<td class="{cls}">{html.escape("" if v is None else str(v))}</td>'


def build_html(models: list[dict], generated: str) -> str:
    out = [f"<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Reasoning benchmark</title>"
           f"<style>{CSS}</style></head><body><h1>Reasoning benchmark</h1>"
           f"<div class=\"muted\">generated {html.escape(generated)} from {len(models)} model file(s)</div>"]

    split_runs = [(m, r) for m in models for r in m["runs"] if r.get("summary")]
    if split_runs:
        out.append("<h2>Scored runs</h2><div class=\"wrap\"><table><tr><th class=\"l\">model</th>"
                   "<th class=\"l\">mode</th><th class=\"l\">prompt</th>"
                   "<th class=\"l\">level</th><th>score</th><th>answered</th><th>traps caught</th>"
                   "<th>trap seen in reasoning</th><th>code</th>"
                   "<th>exact</th><th>time s</th><th>tokens</th><th>gen t/s</th><th>draft acc</th>"
                   "<th class=\"l\">run</th></tr>")
        for m, r in sorted(split_runs, key=lambda x: -x[1]["summary"]["score_pct"]):
            s = r["summary"]
            level = r.get("level") or ", ".join(str(x.get("level")) for x in r["rows"])
            out.append("<tr>" + cell(m["name"], "l") + cell(r.get("mode"), "l") + cell(r.get("prompt_file"), "l")
                       + cell(level, "l") + cell(f"{s['score_pct']}%")
                       + cell(f"{s['answered']}/{s['questions']}") + cell(f"{s['traps_caught']}/{s['traps']}")
                       + cell(f"{s.get('traps_in_reasoning', '-')}/{s['traps']}")
                       + cell(f"{s['code_ok']}/{s['code_questions']}") + cell(f"{s['exact_ok']}/{s['exact_questions']}")
                       + cell(s["total_time_s"]) + cell(s["completion_tokens"]) + cell(s["mean_gen_tps"])
                       + cell(s.get("draft_acceptance")) + cell(r["timestamp"], "l") + "</tr>")
        out.append("</table></div>")

    for m in models:
        out.append(f"<h2>{html.escape(m['name'])}</h2><div class=\"muted\">endpoint {html.escape(m.get('endpoint', ''))}"
                   f" | build {html.escape(str(m.get('build_info', 'n/a')))} | levels "
                   f"{html.escape(', '.join(m.get('reasoning_efforts') or []) or 'none reported')}</div>")
        for r in reversed(m["runs"]):
            rows = r["rows"]
            out.append(f"<details><summary>{html.escape(r.get('mode', 'full'))} run {html.escape(r['timestamp'])}"
                       f" ({len(rows)} request(s), max_tokens {r.get('max_tokens') or 'unlimited'})</summary>"
                       f"<div class=\"wrap\"><table><tr>")
            cols = ["question", "level", "status", "total_time_s", "ttft_s", "reasoning_chars", "content_chars",
                    "completion_tokens", "gen_tps", "prompt_tps", "finish_reason", "answered", "trap_caught",
                    "trap_in_reasoning", "code_ok", "answer_ok", "content_head"]
            cols = [c for c in cols if any(c in row for row in rows)]
            out.append("".join(f"<th class=\"{'l' if c == 'content_head' else ''}\">{c}</th>" for c in cols) + "</tr>")
            for row in rows:
                out.append("<tr>" + "".join(cell(row.get(c), "w l" if c == "content_head" else "") for c in cols) + "</tr>")
            out.append("</table></div>")
            if r.get("scores"):
                scols = ["question", "level", "answered", "trap_caught", "trap_in_reasoning", "code_ok", "answer_ok"]
                out.append("<div class=\"wrap\"><table><tr>" + "".join(f"<th>{c}</th>" for c in scols) + "</tr>")
                for row in r["scores"]:
                    out.append("<tr>" + "".join(cell(row.get(c)) for c in scols) + "</tr>")
                out.append("</table></div>")
            out.append("</details>")
    out.append("</body></html>")
    return "\n".join(out)


# ---------------------------------------------------------------- modes

def run_quick(session, url, model, reported, max_tokens, timeout) -> tuple[list[dict], list[str]]:
    levels = ["none"] + reported if reported else ["default", "none"]
    unsupported = next((lv for lv in LADDER if lv not in levels), None) if reported else None
    rows, problems = [], []
    for level in levels + ([unsupported] if unsupported else []):
        row, content, _ = run_request(session, url, model, QUICK_PROMPT, None if level == "default" else level,
                                      max_tokens, timeout)
        rows.append(row)
        print(f"  {level:<8} status={row['status']} reasoning_chars={row['reasoning_chars']} "
              f"content={content.strip()[:40]!r} time={row['total_time_s']}s")
        if level == unsupported:
            if row["status"] == "ok":
                problems.append(f"unsupported level '{level}' was accepted")
        elif row["status"] != "ok":
            problems.append(f"{level}: {row['error']}")
        elif level == "none" and reported and row["reasoning_chars"] > 0:
            problems.append(f"none: expected no reasoning, got {row['reasoning_chars']} chars")
        elif level in reported and row["reasoning_chars"] == 0:
            problems.append(f"{level}: expected reasoning, got none")
    return rows, problems


def run_full(session, url, model, prompt, levels, repeat, max_tokens, timeout, answers_file):
    """The whole prompt once per level; returns (rows, problems, records with the full answer and reasoning)."""
    rows, problems, recs = [], [], []
    runs = [(level, level if repeat == 1 else f"{level}#{i + 1}") for level in levels for i in range(repeat)]
    with open(answers_file, "w", encoding="utf-8") as fa:
        for level, tag in runs:
            print(f"--- level {tag} ...", flush=True)
            row, content, reasoning = run_request(session, url, model, prompt, None if level == "default" else level,
                                                  max_tokens, timeout)
            row["level"] = tag
            rows.append(row)
            rec = {"level": tag, "finish_reason": row["finish_reason"], "content": content, "reasoning": reasoning}
            recs.append(rec)
            fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fa.flush()
            print(f"    total={row['total_time_s']}s reasoning_chars={row['reasoning_chars']} "
                  f"content_chars={row['content_chars']} tokens={row['completion_tokens']} gen={row['gen_tps']} t/s "
                  f"finish={row['finish_reason']}")
            if row["status"] != "ok":
                problems.append(f"{tag}: {row['error']}")
            elif level == "none" and row["reasoning_chars"] > 0:
                problems.append(f"none: expected no reasoning, got {row['reasoning_chars']} chars")
            elif level not in ("none", "default") and row["reasoning_chars"] == 0:
                problems.append(f"{tag}: expected reasoning, got none")
    return rows, problems, recs


def run_split(session, url, model, questions, selection, level, key, max_tokens, timeout, answers_file):
    rows, problems = [], []
    with open(answers_file, "w", encoding="utf-8") as fa:
        for q in selection:
            print(f"--- question {q} ({level}) ...", flush=True)
            row, content, reasoning = run_request(session, url, model, questions[q],
                                                  None if level == "default" else level, max_tokens, timeout)
            row = {"question": q, **row, **score_answer(key.get(str(q), {}), row, content, reasoning)}
            rows.append(row)
            fa.write(json.dumps({"question": q, "level": level, "finish_reason": row["finish_reason"],
                                 "score": {k: row[k] for k in ("answered", "trap_caught", "code_ok", "answer_ok")
                                           if k in row},
                                 "content": content, "reasoning": reasoning}, ensure_ascii=False) + "\n")
            fa.flush()
            marks = " ".join(f"{k}={row[k]}" for k in ("answered", "trap_caught", "code_ok", "answer_ok") if k in row)
            print(f"    {row['total_time_s']}s tokens={row['completion_tokens']} gen={row['gen_tps']} t/s "
                  f"finish={row['finish_reason']} {marks}")
            if row["status"] != "ok":
                problems.append(f"question {q}: {row['error']}")
    return rows, problems


def new_answers_file(out_dir: Path, model: str, stamp: str, no_save: bool) -> Path:
    """<out-dir>/answers/<model>-<time>.jsonl with the full answer and reasoning of every request."""
    if no_save:
        return Path(os.devnull)
    answers_dir = out_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    base = f"{sanitize_name(model)}-{stamp.replace(':', '')}"
    path = answers_dir / f"{base}.jsonl"
    n = 2
    while path.exists():  # two runs in the same second
        path = answers_dir / f"{base}-{n}.jsonl"
        n += 1
    return path


def write_report(out_dir: Path, stamp: str) -> None:
    models = []
    for f in sorted(out_dir.glob("*.toml")):
        m, r = load_model_file(f)
        if m:
            models.append({**m, "runs": r})
    (out_dir / "report.html").write_text(build_html(models, stamp), encoding="utf-8")
    print(f"saved: {out_dir / 'report.html'}")


def rescore(answers: Path, key_path: Path, out_dir: Path) -> None:
    """Score a saved --split answers file again with the current key and update its run."""
    if not answers.exists():
        answers = out_dir / "answers" / answers.name
    key = load_key(key_path)
    recs = [json.loads(line) for line in answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = {rec["question"]: rec for rec in recs if "question" in rec}
    fields = ("answered", "trap_caught", "trap_in_reasoning", "code_ok", "answer_ok", "points", "max_points")
    for f in sorted(out_dir.glob("*.toml")):
        model, runs = load_model_file(f)
        for run in runs:
            if run.get("answers_file") != answers.name:
                continue
            if run.get("mode") == "full":
                count = int(run.get("question_count") or 32)
                run["scores"] = [row for rec in recs for row in score_full_record(rec, key, count)]
                run["summary"] = full_summary(run["scores"], run["rows"])
            else:
                for row in run["rows"]:
                    rec = records.get(row.get("question"))
                    if rec is None:
                        continue
                    for k in fields:
                        row.pop(k, None)
                    row.update(score_answer(key.get(str(row["question"]), {}), row, rec.get("content", ""),
                                            rec.get("reasoning", "")))
                run["summary"] = summarize(run["rows"])
            f.write_text(emit_toml(model, runs), encoding="utf-8")
            s = run["summary"]
            print(f"{model.get('name')}: score {s['score_pct']}% ({s['points']}/{s['max_points']}), traps "
                  f"{s['traps_caught']}/{s['traps']} (seen in reasoning {s['traps_in_reasoning']}), code "
                  f"{s['code_ok']}/{s['code_questions']}, exact {s['exact_ok']}/{s['exact_questions']}")
            write_report(out_dir, datetime.now().isoformat(timespec="seconds"))
            return
    raise SystemExit(f"no run in {out_dir} uses {answers.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--split", action="store_true", help="one request per numbered question, scored with --key")
    mode.add_argument("--quick", action="store_true", help="only check reasoning level selection")
    mode.add_argument("--rescore", metavar="ANSWERS", default=None,
                      help="score a saved answers file (.jsonl) again with --key; no server needed")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"server base URL (default {DEFAULT_URL})")
    ap.add_argument("--model", default=None, help="model name (default: first from /v1/models)")
    ap.add_argument("--prompt-file", default=str(DEFAULT_PROMPT), help="prompt file (default %(default)s)")
    ap.add_argument("--key", default=None,
                    help=f"scoring key (default {DEFAULT_KEY.name}; a whole-prompt run with another --prompt-file is "
                         f"scored only when --key is given)")
    ap.add_argument("--questions", default=None, help="--split: question numbers, e.g. 1-4,18 (default all)")
    ap.add_argument("--levels", default=None, help="comma-separated levels (default: none + reported levels; "
                                                    "--split: the highest reported level)")
    ap.add_argument("--level", default=None, help="one level for --split, or 'highest'")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="max_tokens per request; default: no limit (the model stops by itself or at the end of the "
                         "server's context)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per level in the default mode (default 1)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-request timeout in s (default 1800)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="results folder (default %(default)s)")
    ap.add_argument("--no-save", action="store_true", help="print only, write no files")
    args = ap.parse_args()

    if args.rescore:
        rescore(Path(args.rescore), Path(args.key or DEFAULT_KEY), Path(args.out_dir))
        return

    session = requests.Session()
    model = args.model or fetch_model_name(session, args.url)
    props = fetch_props(session, args.url)
    reported = props.get("reasoning_efforts") or []
    max_tokens = args.max_tokens if args.max_tokens is not None else 0  # 0: no limit
    print(f"endpoint {args.url} | model {model} | build {props.get('build_info', 'n/a')}")
    print(f"reported reasoning_efforts: {reported or 'none'}")

    out_dir = Path(args.out_dir)
    stamp = datetime.now().isoformat(timespec="seconds")
    run = {"timestamp": stamp, "max_tokens": max_tokens}

    if args.quick:
        run.update(mode="quick", prompt_file="(built-in)")
        rows, problems = run_quick(session, args.url, model, reported, max_tokens, args.timeout)
    elif args.split:
        text = Path(args.prompt_file).read_text(encoding="utf-8")
        questions = split_questions(text)
        if not questions:
            raise SystemExit(f"no numbered questions in {args.prompt_file}")
        selection = parse_selection(args.questions, sorted(questions))
        wanted = args.level or (args.levels.split(",")[0].strip() if args.levels else "highest")
        level = highest_level(reported) if wanted == "highest" else wanted
        key_path = Path(args.key or DEFAULT_KEY)
        key = load_key(key_path)
        print(f"questions {len(selection)} of {len(questions)} | level {level} | key {key_path if key else '(none)'}")
        answers_file = new_answers_file(out_dir, model, stamp, args.no_save)
        run.update(mode="split", prompt_file=Path(args.prompt_file).name, level=level,
                   answers_file=answers_file.name if not args.no_save else None)
        rows, problems = run_split(session, args.url, model, questions, selection, level, key, max_tokens,
                                   args.timeout, answers_file)
        run["summary"] = summarize(rows)
        s = run["summary"]
        print(f"\nscore {s['score_pct']}% ({s['points']}/{s['max_points']}): answered {s['answered']}/{s['questions']}, "
              f"traps {s['traps_caught']}/{s['traps']} (seen in reasoning {s['traps_in_reasoning']}), "
              f"code {s['code_ok']}/{s['code_questions']}, "
              f"exact {s['exact_ok']}/{s['exact_questions']} | {s['total_time_s']} s, {s['completion_tokens']} tokens, "
              f"{s['mean_gen_tps']} t/s, draft acceptance {s['draft_acceptance']}")
    else:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        if args.level:
            levels = [highest_level(reported) if args.level == "highest" else args.level]
        elif args.levels:
            levels = [x.strip() for x in args.levels.split(",") if x.strip()]
        else:
            levels = ["none"] + reported if reported else ["default"]
        answers_file = new_answers_file(out_dir, model, stamp, args.no_save)
        run.update(mode="full", prompt_file=Path(args.prompt_file).name, prompt_chars=len(prompt), repeat=args.repeat,
                   answers_file=answers_file.name if not args.no_save else None)
        count = len(split_questions(prompt))
        run["question_count"] = count
        print(f"prompt {args.prompt_file} ({len(prompt)} chars, {count} numbered questions) | levels {levels}")
        rows, problems, recs = run_full(session, args.url, model, prompt, levels, args.repeat, max_tokens,
                                        args.timeout, answers_file)
        # the default key belongs to the default prompt; any other prompt is scored only with its own --key
        if count and (args.key or Path(args.prompt_file).resolve() == DEFAULT_PROMPT.resolve()):
            key = load_key(Path(args.key or DEFAULT_KEY))
            run["scores"] = [row for rec in recs for row in score_full_record(rec, key, count)]
            run["summary"] = full_summary(run["scores"], rows)
            s = run["summary"]
            print(f"\nscore {s['score_pct']}% ({s['points']}/{s['max_points']}): answered {s['answered']}/{s['questions']}, "
                  f"traps {s['traps_caught']}/{s['traps']} (seen in reasoning {s['traps_in_reasoning']}), "
                  f"code {s['code_ok']}/{s['code_questions']}, exact {s['exact_ok']}/{s['exact_questions']}")

    run["rows"] = rows
    print("\nCHECK FAILED:\n  - " + "\n  - ".join(problems) if problems else "\nCHECK PASSED.")

    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        model_file = out_dir / f"{sanitize_name(model)}.toml"
        _, runs = load_model_file(model_file)
        runs.append(run)
        model_file.write_text(emit_toml({"name": model, "endpoint": args.url, "build_info": props.get("build_info"),
                                         "reasoning_efforts": reported}, runs), encoding="utf-8")
        print(f"saved: {model_file}")
        write_report(out_dir, stamp)

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
