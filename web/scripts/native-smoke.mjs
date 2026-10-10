/** The native host: the Stress Lab built natively serves the development
 * protocol over standard input and output, drawing offscreen (status names
 * its GPU, captures write PNGs, the game's controls and benchmarks run),
 * and its job workers' outputs match those its job cache kept.
 * Builds `stress-lab` with `--release --features dev` (the cargo on PATH);
 * needs no development server. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { launchNativeSession, nativeProgram } from "./agent-native.mjs";

const workspace = fileURLToPath(new URL("../../", import.meta.url));
const program = nativeProgram("stress-lab");

const session = await launchNativeSession({ program, args: ["--preset=visual", "--size=640x400", "--workers=2"] });
try {
  const run = async command => {
    const reply = await session.execute(command);
    assert.equal(reply.ok, true, `${command.op}: ${JSON.stringify(reply.error)}`);
    return reply.result;
  };

  // Sessions start paused, name the GPU they draw on and its interface.
  const status = await run({ op: "status" });
  assert.equal(status.paused, true);
  assert.ok(status.adapter.name && status.adapter.backend, `the adapter is named: ${JSON.stringify(status.adapter)}`);
  assert.deepEqual(status.build.features, ["dev"]);

  // The game's own controls and the program's options are discoverable.
  const types = (await run({ op: "describe" })).types;
  const controls = types.find(type => type.name === "GameControls").controls;
  assert.deepEqual(Object.keys(controls).sort(), ["effects", "look", "view", "workload"]);
  assert.ok("preset" in types.find(type => type.name === "PageParameters").parameters);

  // Captures write PNG files of the offscreen frame; the same frame twice
  // compares the same to itself.
  const capture = (await run({ op: "capture" })).artifact;
  assert.equal(capture.mime_type, "image/png");
  assert.ok(capture.bytes > 1000, `a capture holds an image: ${JSON.stringify(capture)}`);
  const same = (await run({ op: "compare", a: capture.path, b: capture.path })).compare;
  assert.deepEqual([same.width, same.height, same.different], [640, 400, 0]);

  // The game's controls change the workload and the look; captures show it.
  const workload = await run({ op: "game", name: "workload", args: { preset: "city" } });
  assert.equal(workload.result.title, "WILD CITY");
  assert.equal(workload.result.config.scene, "city");
  await run({ op: "game", name: "look", args: { name: "night" } });
  const night = (await run({ op: "capture" })).artifact;
  const changed = (await run({ op: "compare", a: capture.path, b: night.path, tolerance: 8 })).compare;
  assert.ok(changed.different > changed.total / 10, `the city at night differs from the field: ${JSON.stringify(changed)}`);
  const unknown = await session.execute({ op: "game", name: "weather", args: {} });
  assert.equal(unknown.error.code, "unsupported_command");

  // Steps advance exact ticks; benchmarks sample frames with the GPU's work.
  const stepped = await run({ op: "step", ticks: 12 });
  assert.equal(stepped.observation.tick, "12");
  const benchmark = (await run({ op: "benchmark", mode: "render", warmup: 2, samples: 8 })).benchmark;
  assert.ok(benchmark.frame_interval_ms.median > 0 && benchmark.cpu_ms.total.median > 0, JSON.stringify(benchmark.cpu_ms));
  assert.equal(benchmark.environment.adapter.name, status.adapter.name);

  // Resuming runs frames in real time until paused again.
  await run({ op: "resume" });
  await new Promise(resolve => setTimeout(resolve, 1500));
  const paused = await run({ op: "pause" });
  assert.equal(paused.paused, true);
  assert.ok(Number(paused.observation.tick) > 12, `ticks ran while resumed: ${JSON.stringify(paused.observation)}`);
} finally {
  await session.close();
}

// Job outputs: a run with workers keeps them; a verifying run reruns every
// one it is handed back and finds them the same bits.
const highlands = verify => JSON.parse(execFileSync(program, [
  "--preset=highlands", "--size=480x300", "--offscreen=30", "--workers=4", `--jobcache=${verify ? "verify" : "1024"}`,
], { cwd: workspace, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"], timeout: 300_000 }));
const kept = highlands(false);
assert.ok(kept.stats.jobs.completed > 0 && kept.stats.jobs.workers === 4, JSON.stringify(kept.stats.jobs));
assert.equal(kept.stats.terrain.unplaced_tiles, 0);
const verified = highlands(true);
assert.ok(verified.stats.jobs.cache_hits > 0, `the verifying run is handed kept outputs: ${JSON.stringify(verified.stats.jobs)}`);
assert.equal(verified.stats.jobs.cache_mismatches, 0, `kept outputs match fresh runs: ${JSON.stringify(verified.stats.jobs)}`);
// How far each run streamed in its frames depends on timing; every tile it
// holds has its objects.
assert.equal(verified.stats.terrain.unplaced_tiles, 0);
console.log("native smoke passed");
