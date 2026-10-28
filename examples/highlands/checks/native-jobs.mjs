// @ts-check
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { nativeBinary } from "../../../web/scripts/harness.mjs";
import { workspace } from "../../../web/scripts/workspace.mjs";

const program = await nativeBinary("highlands", "highlands-native", { profiling: true });
const highlands = verify => JSON.parse(execFileSync(program, [
  "--size=480x300", "--offscreen=30", "--workers=4", `--jobcache=${verify ? "verify" : "1024"}`,
], { cwd: workspace, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"], timeout: 300_000 }));
const kept = highlands(false);
assert.ok(kept.stats.jobs.completed > 0 && kept.stats.jobs.workers === 4, JSON.stringify(kept.stats.jobs));
const stackLimit = process.env.ENGINE_NATIVE_PROFILE === "dev" ? 1024 * 1024 : 640 * 1024;
assert.ok(kept.main_stack_peak_bytes === null || kept.main_stack_peak_bytes < stackLimit,
  `the main thread's stack peaked at ${kept.main_stack_peak_bytes} bytes`);
const verified = highlands(true);
assert.ok(verified.stats.jobs.cache_hits > 0, `the verifying run is handed kept outputs: ${JSON.stringify(verified.stats.jobs)}`);
assert.equal(verified.stats.jobs.cache_mismatches, 0, `kept outputs match fresh runs: ${JSON.stringify(verified.stats.jobs)}`);
console.log("highlands native jobs check passed");
