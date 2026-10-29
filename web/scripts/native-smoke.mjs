// @ts-check
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runAb } from "./agent-ab.mjs";
import { launchNativeSession } from "./agent-native.mjs";
import { gardenScene, playerProgram } from "./harness.mjs";
import { workspacePath } from "./workspace.mjs";

const program = await playerProgram({ profiling: true });
const garden = workspacePath("web/scripts/fixtures/garden.scene.json");
const revision = String(gardenScene().revision);

const session = await launchNativeSession({ program, args: ["--source", garden, "--size=640x400", "--workers=2", "--taa=on"] });
try {
  const run = async command => {
    const reply = await session.execute(command);
    assert.equal(reply.ok, true, `${command.verb} ${command.name ?? ""}: ${JSON.stringify(reply.error)}`);
    return reply.result;
  };

  const status = await run({ verb: "query", name: "status" });
  assert.equal(status.paused, true);
  assert.ok(status.adapter.name && status.adapter.backend, `the adapter is named: ${JSON.stringify(status.adapter)}`);
  assert.deepEqual(status.build.features, ["dev", "profiling"]);
  assert.ok(status.adapter.backend, "status names the GPU interface");

  const listing = await run({ verb: "describe" });
  assert.deepEqual(listing.verbs, ["describe", "inspect", "query", "control", "capture", "profile", "explain", "watch", "check"]);
  assert.ok(listing.bricks.some(brick => brick.name === "culling" && brick.explains.length > 0), JSON.stringify(listing.bricks.map(brick => brick.name)));
  const types = listing.types;
  const controls = types.find(type => type.name === "GameControls").controls;
  assert.deepEqual(Object.keys(controls), []);
  const parameters = types.find(type => type.name === "PageParameters").parameters;
  for (const name of ["source", "without", "taa", "occlusion", "subpixel", "pipeline", "terrain-clock", "dynamic-resolution"]) assert.ok(name in parameters, name);

  assert.equal((await run({ verb: "inspect", component: "TemporalAntialiasing" })).value.enabled, true);
  const ordinary = (await run({ verb: "control", name: "frames", args: { count: 4, interval_ms: 1000 / 60 } })).rendering;
  assert.equal(ordinary.temporal_antialiasing.active, true, "ordinary frames resolve temporally");
  const { resolution } = ordinary;
  assert.deepEqual(
    [resolution.display_width, resolution.display_height, resolution.render_width, resolution.render_height, resolution.dropped_pixels],
    [640, 400, 640, 400, 0],
  );
  assert.equal(ordinary.dynamic_resolution.active, false, "offscreen frames keep their size");
  await run({ verb: "control", name: "environment", args: { resolution: { scale: 0.5 } } });
  const scaled = (await run({ verb: "control", name: "frames", args: { count: 4, interval_ms: 1000 / 60 } })).rendering.resolution;
  assert.deepEqual(
    [scaled.display_width, scaled.display_height, scaled.render_width, scaled.render_height],
    [640, 400, 320, 200],
  );
  await run({ verb: "control", name: "environment", args: { resolution: { scale: 1 } } });

  const capture = (await run({ verb: "capture" })).artifact;
  assert.equal(capture.mime_type, "image/png");
  assert.ok(capture.bytes > 1000, `a capture holds an image: ${JSON.stringify(capture)}`);
  const same = (await run({ verb: "capture", compare: { a: capture.path, b: capture.path } })).compare;
  assert.deepEqual([same.width, same.height, same.different], [640, 400, 0]);

  const atmosphere = preset => ({ verb: "control", name: "environment", args: { merge: true, atmosphere: { preset } } });
  await run(atmosphere("night"));
  const night = (await run({ verb: "capture" })).artifact;
  const changed = (await run({ verb: "capture", compare: { a: capture.path, b: night.path, tolerance: 8 } })).compare;
  assert.ok(changed.different > changed.total / 10, `the garden at night differs from its own light: ${JSON.stringify(changed)}`);
  const unknown = await session.execute({ verb: "control", name: "game", args: { name: "weather", args: {} } });
  assert.equal(unknown.error.code, "unsupported_command");

  const ticked = (await run({ verb: "query", name: "status" })).observation.tick;
  const stepped = await run({ verb: "control", name: "step", args: { ticks: 12 } });
  assert.equal(BigInt(stepped.observation.tick) - BigInt(ticked), 12n);
  const benchmark = (await run({ verb: "control", name: "benchmark", args: { mode: "render", warmup: 2, samples: 8 } })).benchmark;
  assert.ok(benchmark.frame_interval_ms.median > 0 && benchmark.cpu_ms.total.median > 0, JSON.stringify(benchmark.cpu_ms));
  assert.equal(benchmark.environment.adapter.name, status.adapter.name);

  const costs = await run({ verb: "profile", frames: 8 });
  assert.ok(costs.bricks.length > 0 && costs.cpu_ms > 0, `profile measures bricks: ${JSON.stringify(costs).slice(0, 400)}`);
  assert.equal(JSON.parse(readFileSync(workspacePath(costs.artifact.path), "utf8")).frames.length, 8);
  const slow = await run({ verb: "explain", about: "frame", top: 3 });
  assert.ok(slow.records.some(record => record.decided === "cost" && typeof record.cpu_ms === "number"), JSON.stringify(slow.records));
  const look = (target, position) => run({ verb: "control", name: "camera", args: { position, target, vertical_fov_radians: 0.8, near: 0.1, far: 150 } });
  await look([0, 20, 120], [0, 20, 60]);
  const culled = await run({ verb: "explain", about: "entity", entity: "crystal-6" });
  assert.equal(culled.records.find(record => record.brick === "core")?.decided, "culled", JSON.stringify(culled.records));
  assert.ok(culled.records.some(record => record.brick === "culling" && record.decided === "culled"), JSON.stringify(culled.records));
  await look([0, 0, 0], [17, 21, 26]);
  const drawn = await run({ verb: "explain", about: "entity", entity: "crystal-6" });
  assert.ok(drawn.records.some(record => record.brick === "level_of_detail" && Number.isInteger(record.level)), JSON.stringify(drawn.records));
  const scene = await run({ verb: "explain", about: "scene" });
  assert.deepEqual(scene.records, []);
  const watched = [];
  const watch = await session.execute({ verb: "watch", stats: ["tick"], entities: [{ entity: "player", component: "Transform" }], frames: 3 }, record => watched.push(record));
  assert.equal(watch.ok, true, JSON.stringify(watch.error));
  assert.ok(watched.some(record => record.stat === "tick") && watched.some(record => record.component === "Transform"), JSON.stringify(watched));

  const profile = await run({ verb: "profile", deep: { duration_ms: 500 } });
  const artifact = path => JSON.parse(readFileSync(workspacePath(path), "utf8"));
  const trace = artifact(profile.artifacts.trace);
  assert.ok(trace.gpu.completed > 0, `GPU spans were read back: ${JSON.stringify({ ...trace.gpu, samples: undefined, draw_samples: undefined })}`);
  assert.ok(trace.gpu.samples.some(sample => sample.name === "opaque_meshes"));
  if (trace.gpu.draws.timestamps) assert.ok(trace.gpu.draw_samples.some(draw => draw.duration_ms !== null), "opaque draws are timed");
  const summary = artifact(profile.artifacts.summary);
  assert.equal(summary.captured.privileges, "none: no admin, no system setting, no installed tool");
  const main = summary.threads[0];
  if (process.env.ENGINE_NATIVE_PROFILE !== "dev") {
    assert.ok(main.samples > 100, `the main thread was sampled: ${JSON.stringify(summary.captured.cpu_samples)}`);
    assert.ok(main.total.some(entry => entry.name.startsWith("engine")), `samples name the engine's functions: ${JSON.stringify(main.total.slice(0, 5))}`);
    assert.ok(summary.hot_lines.some(line => line.name.includes(".rs:")), "hot samples resolve to source lines");
  }
  assert.ok(Array.isArray(summary.recommended) && summary.recommended.length > 0);
  const firefox = artifact(profile.artifacts.firefox);
  assert.ok(firefox.meta && firefox.threads.length >= 3, "the Firefox Profiler file holds the threads");

  const ab = await runAb(session, {
    a: JSON.stringify(atmosphere("day")),
    b: JSON.stringify(atmosphere("night")),
    rounds: "1", samples: "4", warmup: "1",
  });
  assert.equal(ab.order, "abba");
  assert.ok(ab.compared.cpu_ms.a > 0 && ab.compared.cpu_ms.b > 0, JSON.stringify(ab.compared));

  const digestAfter = async ticks => {
    await run({ verb: "control", name: "reload", args: { expected_revision: revision, revision } });
    await run({ verb: "control", name: "input", args: { forward: 1 } });
    await run({ verb: "control", name: "step", args: { ticks } });
    await run({ verb: "control", name: "input" });
    return run({ verb: "query", name: "digest" });
  };
  const digest = await digestAfter(12);
  assert.ok(digest.entities > 0 && digest.components.Transform.count > 0 && digest.assets.meshes.count > 0,
    `a digest counts the world's entities, components and assets: ${JSON.stringify(digest).slice(0, 300)}`);
  assert.deepEqual((await digestAfter(12)).components, digest.components, "the same ticks of the same scene digest the same");
  assert.notDeepEqual((await digestAfter(24)).components, digest.components, "the walking player changes the digest");

  const resumedAt = Number((await run({ verb: "query", name: "status" })).observation.tick);
  await run({ verb: "control", name: "resume" });
  for (let attempt = 0; ; attempt++) {
    const running = await run({ verb: "query", name: "status" });
    if (Number(running.observation.tick) > resumedAt) break;
    assert.ok(attempt < 3000, `ticks run while resumed: ${JSON.stringify(running.observation)}`);
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  const paused = await run({ verb: "control", name: "pause" });
  assert.equal(paused.paused, true);
  assert.ok(Number(paused.observation.tick) > resumedAt, `ticks ran while resumed: ${JSON.stringify(paused.observation)}`);
} finally {
  await session.close();
}
console.log("native smoke passed");
