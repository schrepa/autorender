// @ts-check
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { initSync, JobWorker } from "../pkg-dev/bare_game.js";

const [path] = process.argv.slice(2);
if (!path) {
  console.error("usage: node web/scripts/pure-wasm.mjs JOBS.json");
  process.exit(1);
}
initSync({ module: readFileSync(new URL("../pkg-dev/bare_game_bg.wasm", import.meta.url)) });
const hash = bytes => createHash("sha256").update(bytes).digest("hex").slice(0, 16);
const entries = JSON.parse(readFileSync(path, "utf8"));
const mismatches = [];
/** @type {JobWorker | undefined} */
let worker;
let [shares, jobs, id] = [0, 0, 0];
for (const entry of entries) {
  if ("list" in entry) {
    worker?.free();
    worker = new JobWorker(Buffer.from(entry.list, "hex"));
    continue;
  }
  if (!worker) {
    console.error(`${path} names no job list before its first entry`);
    process.exit(1);
  }
  if ("share" in entry) {
    worker.share(entry.share, Buffer.from(entry.bytes, "base64"));
    shares++;
    continue;
  }
  id++;
  worker.hold(id, entry.job, entry.source, Buffer.from(entry.input, "base64"), 0);
  /** @type {{ type: "done", id: number, output: Uint8Array } | { type: "failed", id: number, error: string } | undefined} */
  let report;
  while (!report && worker.busy()) report = worker.slice();
  jobs++;
  const found = report?.type === "done" ? hash(report.output) : report?.error ?? "unreported";
  if (found !== entry.output) {
    mismatches.push({ job: entry.job, index: jobs - 1, native: entry.output, wasm: found });
  }
}
console.log(JSON.stringify({ shares, jobs, mismatches: mismatches.slice(0, 8), mismatched: mismatches.length }));
process.exit(mismatches.length ? 3 : 0);
