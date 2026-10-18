
const DATA = {"generated": "2026-09-22T21:39:07", "prompt_file": "tools\\bench-smoke-prompt.txt", "prompt_chars": 71, "max_tokens": 200, "source": "Qwen3.8-27B-huihui-NVFP4.toml", "models": [{"name": "Qwen3.8-27B-huihui-NVFP4", "endpoint": "http://127.0.0.1:8080", "build_info": "b11114-27259a791", "reasoning_efforts": ["low", "medium", "xhigh"], "runs": [{"timestamp": "2026-09-22T21:39:07", "prompt_file": "tools\\bench-smoke-prompt.txt", "prompt_chars": 71, "max_tokens": 200, "repeat": 1, "levels": [{"level": "none", "status": "ok", "error": 0, "total_time_s": 3.27, "ttft_s": 2.09, "reasoning_time_s": 0.0, "content_time_s": 1.18, "reasoning_chars": 0, "content_chars": 343, "combined_chars": 343, "finish_reason": "stop", "prompt_tokens": 33, "completion_tokens": 152, "cached_tokens": 0, "prompt_tps": 147.6, "gen_tps": 127.5, "draft_n": 126, "draft_n_accepted": 112, "content_head": "To calculate $27 \\times 43$, we can break the problem down using the distributive property:  1.  Break down 43 into $40 + 3$. 2.  Calculate $27 \\times 40$:     $$27 \\times 4 = 108$$     $$108 \\times 1"}, {"level": "low", "status": "ok", "error": 0, "total_time_s": 0.96, "ttft_s": 0.25, "reasoning_time_s": 0.72, "content_time_s": 0.25, "reasoning_chars": 80, "content_chars": 46, "combined_chars": 126, "finish_reason": "stop", "prompt_tokens": 61, "completion_tokens": 93, "cached_tokens": 0, "prompt_tps": 326.4, "gen_tps": 129.9, "draft_n": 75, "draft_n_accepted": 67, "content_head": "27 \u00d7 43 = 27 \u00d7 (40 + 3) = 1080 + 81 = **1161**"}, {"level": "medium", "status": "ok", "error": 0, "total_time_s": 1.57, "ttft_s": 0.39, "reasoning_time_s": 1.11, "content_time_s": 0.44, "reasoning_chars": 163, "content_chars": 93, "combined_chars": 256, "finish_reason": "stop", "prompt_tokens": 31, "completion_tokens": 150, "cached_tokens": 0, "prompt_tps": 263.1, "gen_tps": 126.3, "draft_n": 123, "draft_n_accepted": 111, "content_head": "**Short calculation:**  27 \u00d7 43 = 27 \u00d7 (40 + 3) = (27 \u00d7 40) + (27 \u00d7 3) = 1080 + 81 = **1161**"}, {"level": "xhigh", "status": "ok", "error": 0, "total_time_s": 1.42, "ttft_s": 0.26, "reasoning_time_s": 1.1, "content_time_s": 0.3, "reasoning_chars": 259, "content_chars": 49, "combined_chars": 308, "finish_reason": "stop", "prompt_tokens": 73, "completion_tokens": 149, "cached_tokens": 0, "prompt_tps": 392.9, "gen_tps": 126.9, "draft_n": 126, "draft_n_accepted": 108, "content_head": "27 \u00d7 43 = 27 \u00d7 40 + 27 \u00d7 3 = 1080 + 81 = **1161**"}]}]}]};
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
