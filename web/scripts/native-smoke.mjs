/** The native host: the Stress Lab built natively serves the development
 * protocol over standard input and output, drawing offscreen (status names
 * its GPU, the engine switches it was given apply, captures write PNGs,
 * the game's controls, benchmarks and
 * profiling captures with per-draw GPU times run, the agent's A/B
 * harness measures in it, and digests of the same ticks of a workload
 * agree), and its job workers' outputs match those its
 * job cache kept. Builds `stress-lab` with `--release --features
 * dev,profiling` (the cargo on PATH); needs no development server. */
// @ts-check
import "./exit-codes.mjs";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { runAb } from "./agent-ab.mjs";
import { launchNativeSession, nativeProgram } from "./agent-native.mjs";
import { holdHeavyRun } from "./heavy-run.mjs";
import { workspace, workspacePath } from "./workspace.mjs";

await holdHeavyRun("smoke native-smoke.mjs");
const program = nativeProgram("stress-lab", { profiling: true });

const session = await launchNativeSession({ program, args: ["--preset=visual", "--size=640x400", "--workers=2", "--taa=on"] });
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
  assert.deepEqual(status.build.features, ["dev", "profiling"]);
  assert.ok(status.adapter.backend, "status names the GPU interface");

  // The game's own controls and the program's options, the engine
  // switches every host takes among them, are discoverable.
  const types = (await run({ op: "describe" })).types;
  const controls = types.find(type => type.name === "GameControls").controls;
  assert.deepEqual(Object.keys(controls).sort(), ["effects", "look", "view", "workload"]);
  const parameters = types.find(type => type.name === "PageParameters").parameters;
  for (const name of ["preset", "taa", "occlusion", "subpixel", "presentation", "pipeline", "terrain-clock"]) assert.ok(name in parameters, name);

  // --taa=on set the world's temporal antialiasing, which ordinary frames
  // then resolve, at the size asked for.
  assert.equal((await run({ op: "inspect", component: "TemporalAntialiasing" })).value.enabled, true);
  const ordinary = (await run({ op: "frames", count: 4, interval_ms: 1000 / 60 })).rendering;
  assert.equal(ordinary.temporal_antialiasing.active, true, "ordinary frames resolve temporally");
  assert.deepEqual([ordinary.view_width, ordinary.view_height, ordinary.dropped_view_pixels], [640, 400, 0]);

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

  // Profiles are deep without setup: the frames' GPU passes (and each
  // opaque draw where the GPU interface times work inside passes), CPU
  // samples of every thread symbolicated to the engine's functions, and a
  // Firefox Profiler file; the summary says what was captured and what
  // this machine could add.
  const profile = await run({ op: "profile", duration_ms: 500 });
  const artifact = path => JSON.parse(readFileSync(workspacePath(path), "utf8"));
  const trace = artifact(profile.artifacts.trace);
  assert.ok(trace.gpu.completed > 0, `GPU spans were read back: ${JSON.stringify({ ...trace.gpu, samples: undefined, draw_samples: undefined })}`);
  assert.ok(trace.gpu.samples.some(sample => sample.name === "opaque_meshes"));
  if (trace.gpu.draws.timestamps) assert.ok(trace.gpu.draw_samples.some(draw => draw.duration_ms !== null), "opaque draws are timed");
  const summary = artifact(profile.artifacts.summary);
  assert.equal(summary.captured.privileges, "none: no admin, no system setting, no installed tool");
  const main = summary.threads[0];
  assert.ok(main.samples > 100, `the main thread was sampled: ${JSON.stringify(summary.captured.cpu_samples)}`);
  assert.ok(main.total.some(entry => entry.name.startsWith("engine")), `samples name the engine's functions: ${JSON.stringify(main.total.slice(0, 5))}`);
  assert.ok(summary.hot_lines.some(line => line.name.includes(".rs:")), "hot samples resolve to source lines");
  assert.ok(Array.isArray(summary.recommended) && summary.recommended.length > 0);
  const firefox = artifact(profile.artifacts.firefox);
  assert.ok(firefox.meta && firefox.threads.length >= 3, "the Firefox Profiler file holds the threads");

  // The A/B harness alternates variants in ABBA windows: here the same
  // look twice against another.
  const ab = await runAb(session, {
    a: JSON.stringify({ op: "game", name: "look", args: { name: "day" } }),
    b: JSON.stringify({ op: "game", name: "look", args: { name: "night" } }),
    rounds: "1", samples: "4", warmup: "1",
  });
  assert.equal(ab.order, "abba");
  assert.ok(ab.compared.cpu_ms.a > 0 && ab.compared.cpu_ms.b > 0, JSON.stringify(ab.compared));

  // Digests: the world's content as hashes of its components and assets,
  // the same after the same ticks of the same workload, and changed as its
  // renderables move on.
  const digestAfter = async ticks => {
    await run({ op: "game", name: "workload", args: { preset: "city" } });
    await run({ op: "step", ticks });
    return run({ op: "digest" });
  };
  const digest = await digestAfter(12);
  assert.ok(digest.entities > 0 && digest.components.Transform.count > 0 && digest.assets.meshes.count > 0,
    `a digest counts the world's entities, components and assets: ${JSON.stringify(digest).slice(0, 300)}`);
  assert.equal((await digestAfter(12)).hash, digest.hash, "the same ticks of the same workload digest the same");
  assert.notEqual((await digestAfter(24)).hash, digest.hash, "moving renderables change the digest");

  // Resuming runs frames in real time until paused again: ticks advance
  // past the stepped ones while it runs.
  await run({ op: "resume" });
  for (let attempt = 0; ; attempt++) {
    const running = await run({ op: "status" });
    if (Number(running.observation.tick) > 12) break;
    assert.ok(attempt < 3000, `ticks run while resumed: ${JSON.stringify(running.observation)}`);
    await new Promise(resolve => setTimeout(resolve, 10));
  }
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
// The main thread's stack, a megabyte on Windows: startup frames that grow
// show here before they overflow it (null where the system cannot tell).
assert.ok(kept.main_stack_peak_bytes === null || kept.main_stack_peak_bytes < 640 * 1024,
  `the main thread's stack peaked at ${kept.main_stack_peak_bytes} bytes`);
assert.equal(kept.stats.terrain.unplaced_tiles, 0);
const verified = highlands(true);
assert.ok(verified.stats.jobs.cache_hits > 0, `the verifying run is handed kept outputs: ${JSON.stringify(verified.stats.jobs)}`);
assert.equal(verified.stats.jobs.cache_mismatches, 0, `kept outputs match fresh runs: ${JSON.stringify(verified.stats.jobs)}`);
// How far each run streamed in its frames depends on timing; every tile it
// holds has its objects.
assert.equal(verified.stats.terrain.unplaced_tiles, 0);
console.log("native smoke passed");
