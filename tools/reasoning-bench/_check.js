// jsdom check for report.html: execute scripts, report DOM state + errors
const path = require("path");
const fs = require("fs");
const { JSDOM } = require(path.join("C:/AI/continue/gui/node_modules", "jsdom"));

const file = process.argv[2] || path.join(__dirname, "report.html");
const html = fs.readFileSync(file, "utf-8");

const errors = [];
const vc = new (require(path.join("C:/AI/continue/gui/node_modules", "jsdom")).VirtualConsole)();
vc.on("jsdomError", (e) => errors.push("jsdomError: " + e.message));
vc.on("error", (msg) => errors.push("console.error: " + msg));

const dom = new JSDOM(html, { runScripts: "dangerously", virtualConsole: vc });
const doc = dom.window.document;

setTimeout(() => {
  const overviewRows = doc.querySelectorAll("#overview tr").length - 1; // minus header
  const details = doc.querySelectorAll("#models details").length;
  const modelTables = doc.querySelectorAll("#models table").length;
  const meta = doc.getElementById("meta") ? doc.getElementById("meta").textContent : "(missing)";
  const firstSummary = doc.querySelector("#models summary") ? doc.querySelector("#models summary").textContent : "(missing)";

  console.log("overview rows:", overviewRows);
  console.log("model sections:", details);
  console.log("model tables:", modelTables);
  console.log("meta:", meta);
  console.log("first section:", firstSummary);
  console.log("errors:", errors.length ? errors.join(" | ") : "none");
  process.exit(errors.length ? 1 : 0);
}, 500);
