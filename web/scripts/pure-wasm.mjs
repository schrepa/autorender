// @ts-check
import "./exit-codes.mjs";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { initSync, JobWorker } from "../pkg-dev/engine_player.js";

const [path] = process.argv.slice(2);
if (!path) {
  console.error("usage: node web/scripts/pure-wasm.mjs JOBS.json");
  process.exit(1);
}
initSync({ module: readFileSync(new URL("../pkg-dev/engine_player_bg.wasm", import.meta.url)) });
const worker = new JobWorker();
const hash = bytes => createHash("sha256").update(bytes).digest("hex").slice(0, 16);
const entries = JSON.parse(readFileSync(path, "utf8"));
const mismatches = [];
let [shares, jobs, id] = [0, 0, 0];
for (const entry of entries) {
  if ("share" in entry) {
    worker.share(entry.share, Buffer.from(entry.bytes, "base64"));
    shares++;
    continue;
  }
  id++;
  worker.start(id, entry.job, Buffer.from(entry.input, "base64"));
  const budget = entry.slice ?? Number.MAX_SAFE_INTEGER;
  let output;
  do {
    output = worker.step(id, budget);
  } while (output === undefined || output === null);
  jobs++;
  const found = hash(output);
  if (found !== entry.output) {
    mismatches.push({ job: entry.job, index: jobs - 1, native: entry.output, wasm: found });
  }
}
console.log(JSON.stringify({ shares, jobs, mismatches: mismatches.slice(0, 8), mismatched: mismatches.length }));
process.exit(mismatches.length ? 3 : 0);
