/**
 * Replays the terrain corpus's jobs (`cargo xtask pure --wasm`) in the job
 * worker's Wasm module as browsers run it (web/pkg-dev/engine_worker_bg.wasm),
 * in Node, no browser: every share, then every job run as the corpus ran it
 * natively (whole, a slice at a time, or staged), each output (and staged
 * part) hashed and held to the native one. Usage:
 *   node web/scripts/pure-wasm.mjs JOBS.json
 * Prints one JSON object; a job whose output differs exits 3.
 */
import "./exit-codes.mjs";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { initSync, JobWorker } from "../pkg-dev/engine_worker.js";

const [path] = process.argv.slice(2);
if (!path) {
  console.error("usage: node web/scripts/pure-wasm.mjs JOBS.json");
  process.exit(1);
}
initSync({ module: readFileSync(new URL("../pkg-dev/engine_worker_bg.wasm", import.meta.url)) });
const worker = new JobWorker();
/** The first 16 hexadecimal digits of `bytes`' SHA-256, as xtask names outputs. */
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
  let part = null;
  let output;
  do {
    output = worker.step(id, budget);
    if ("part" in entry && part === null) part = worker.take_part(id) ?? null;
  } while (output === undefined || output === null);
  jobs++;
  const found = { output: hash(output), ...("part" in entry ? { part: part === null ? null : hash(part) } : {}) };
  if (found.output !== entry.output || ("part" in entry && found.part !== entry.part)) {
    mismatches.push({ job: entry.job, index: jobs - 1, native: { output: entry.output, part: entry.part }, wasm: found });
  }
}
console.log(JSON.stringify({ shares, jobs, mismatches: mismatches.slice(0, 8), mismatched: mismatches.length }));
process.exit(mismatches.length ? 3 : 0);
