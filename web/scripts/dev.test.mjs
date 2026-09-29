import assert from "node:assert/strict";
import test from "node:test";
import { connectDevelopment } from "../src/dev.ts";
import { distribution } from "../src/benchmark.ts";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup(t, options = {}) {
  const previousWindow = globalThis.window;
  const previousFrame = globalThis.requestAnimationFrame;
  const previousCancelFrame = globalThis.cancelAnimationFrame;
  const frames = new Map();
  let nextFrame = 0;
  let timestamp = 0;
  globalThis.window = {};
  globalThis.requestAnimationFrame = callback => {
    const id = ++nextFrame;
    frames.set(id, callback);
    return id;
  };
  globalThis.cancelAnimationFrame = id => { frames.delete(id); };

  const lifecycle = new AbortController();
  const calls = [];
  const busy = [];
  let paused = options.paused ?? true;
  let freed = false;
  let accessesAfterFree = 0;
  let statsFreed = 0;
  let polls = 0;
  let frame = 7;
  let tick = 12;
  let revision = "4";
  function accessed(method) {
    if (freed) {
      accessesAfterFree++;
      throw new Error(`${method} accessed freed runtime`);
    }
  }
  function observation() {
    return { frame: String(frame), tick: String(tick), scene: { id: "garden", revision } };
  }
  function reply(result = {}) {
    return JSON.stringify({ version: 1, ok: true, observation: observation(), result });
  }
  const game = {
    dev_command(source, width, height) {
      accessed("dev_command");
      const command = JSON.parse(source);
      calls.push({ command, width, height });
      if (options.command) return options.command(command, reply);
      if (command.op === "step") { tick += command.ticks; frame++; }
      return reply();
    },
    dev_reload(source, expectedRevision, targetRevision) {
      accessed("dev_reload");
      calls.push({ reload: { source, expectedRevision, targetRevision } });
      revision = targetRevision;
      return reply({ reloaded: true });
    },
    stats() {
      accessed("stats");
      calls.push({ stats: true });
      return {
        instances: 26, draw_calls: 4, resident_meshes: 4, resident_mesh_bytes: 1024,
        resident_textures: 1, resident_texture_bytes: 16, texture_upload_bytes: 0,
        invalid_transforms: 0,
        render_candidates: 26, visible_instances: 26, culled_instances: 0,
        dropped_instances: 0, missing_assets: 0, culling_enabled: true,
        upload_bytes: 2144, fixed_steps: 1, dropped_seconds: 0,
        mesh_upload_bytes: 0, simulation_ms: 0, extraction_ms: 0.1, render_ms: 0.2,
        shadows_enabled: true, shadow_draws: 2, shadow_kept_redraw: 0,
        shadow_near_casters: undefined, shadow_moving_casters: 5, shadow_kept_casters: 0,
        ambient_occlusion: true, bloom: false,
        local_lights: 3, dropped_lights: 0, lights_visible: undefined, light_cluster_overflow: 0,
        sun_shafts: false, reflections: true, refraction: false, visibility_buffer: false, transparent_draws: 1, particles: 40, dropped_particles: 0, decals: 2, dropped_decals: 0, impostors: 0, dropped_impostors: 0, occluded_instances: undefined, subpixel_instances: undefined,
        lod0_instances: undefined, lod1_instances: undefined, lod2_instances: undefined, impostor_instances: undefined, triangles: undefined,
        reflection_probes: 3, reflection_probe_faces: 1, dropped_reflection_probes: 0,
        virtual_clusters: 0, virtual_triangles: 0, virtual_shadow_clusters: 0, virtual_shadow_triangles: 0, virtual_occluded_clusters: 0, dropped_virtual: 0,
        virtual_resident_pages: 0, virtual_loading_pages: 0, virtual_missing_pages: 0, dropped_virtual_pages: 0,
        free() { statsFreed++; },
      };
    },
    frame() { assert.fail("development controls must not call the ordinary frame loop"); },
    suspend() { assert.fail("development adapter must use host pause transitions"); },
    free() { freed = true; },
  };
  const host = {
    game,
    signal: lifecycle.signal,
    paused: () => paused,
    setPaused(value) { paused = value; },
    setBusy(value) { busy.push(value); },
    size: () => [800, 600],
    poll() { polls++; options.poll?.(); },
    source: options.source,
  };
  const dispose = connectDevelopment(host);
  const api = window.__engineDev;
  t.after(() => {
    dispose();
    lifecycle.abort();
    frames.clear();
    globalThis.window = previousWindow;
    globalThis.requestAnimationFrame = previousFrame;
    globalThis.cancelAnimationFrame = previousCancelFrame;
  });
  return {
    api, host, game, lifecycle, dispose, calls, busy,
    get counters() { return { frame, tick, revision, accessesAfterFree, statsFreed, polls }; },
    get pendingFrames() { return frames.size; },
    flushFrame(delta = 16) {
      const ready = [...frames.values()];
      frames.clear();
      timestamp += delta;
      for (const callback of ready) callback(timestamp);
    },
  };
}

test("running sessions reject mutation and capture commands before forwarding them", async t => {
  const session = setup(t, { paused: false });
  for (const op of ["input", "step", "render", "present", "capture", "reload", "camera", "culling", "environment", "benchmark"]) {
    const reply = await session.api.request({ op });
    assert.equal(reply.ok, false);
    assert.equal(reply.error.code, "pause_required");
    assert.equal(reply.observation.tick, "12");
  }
  assert.ok(session.calls.every(call => call.command?.op === "status"));
  assert.deepEqual(session.counters, {
    frame: 7, tick: 12, revision: "4", accessesAfterFree: 0, statsFreed: 0, polls: 0,
  });
  assert.deepEqual(session.busy, Array.from({ length: 10 }, () => [true, false]).flat());
});

test("awaited source reload excludes competing commands and renders the accepted revision", async t => {
  const source = deferred();
  let receivedSignal;
  const session = setup(t, {
    source: {
      url: "/scenes/garden.scene.json",
      load(signal) { receivedSignal = signal; return source.promise; },
    },
    command(command, reply) {
      const value = JSON.parse(reply());
      if (command.op === "render") value.observation.presentation = { mode: "current", alpha: 1, frame: "7", tick: "12" };
      return JSON.stringify(value);
    },
  });
  const loading = session.api.request({ op: "reload", expected_revision: "4", revision: "5" });
  assert.deepEqual(session.busy, [true]);
  assert.equal(receivedSignal, session.lifecycle.signal);
  const competing = await session.api.request({ op: "step", ticks: 1 });
  assert.equal(competing.error.code, "command_busy");
  assert.equal(session.calls.filter(call => call.reload).length, 0);
  source.resolve("saved scene source");
  const reply = await loading;
  assert.equal(reply.ok, true);
  assert.equal(reply.observation.scene.revision, "5");
  assert.deepEqual(reply.observation.presentation, { mode: "current", alpha: 1, frame: "7", tick: "12" });
  assert.equal(reply.result.reloaded, true);
  assert.equal(reply.result.source, "/scenes/garden.scene.json");
  assert.deepEqual(session.calls.filter(call => call.reload), [{
    reload: { source: "saved scene source", expectedRevision: "4", targetRevision: "5" },
  }]);
  assert.deepEqual(session.calls.filter(call => call.command).map(call => call.command.op), ["status", "render"]);
  assert.deepEqual(session.busy, [true, false]);
  assert.equal(session.counters.frame, 7);
  assert.equal(session.counters.tick, 12);
  assert.equal(session.counters.statsFreed, 1);
});

test("disposal during source fetch never invokes the freed runtime", async t => {
  const source = deferred();
  const session = setup(t, { source: { url: "/scene.json", load: () => source.promise } });
  const loading = session.api.request({ op: "reload", expected_revision: "4", revision: "5" });
  session.dispose();
  session.game.free();
  assert.equal(window.__engineDev, undefined);
  source.resolve("late source");
  const reply = await loading;
  assert.equal(reply.ok, false);
  assert.equal(reply.error.code, "session_closed");
  assert.equal(session.counters.accessesAfterFree, 0);
  assert.equal(session.counters.polls, 0);
  assert.deepEqual(session.calls, []);
  assert.equal((await session.api.request({ op: "status" })).error.code, "session_closed");
});

test("capture renders current state then waits for two animation frames without simulation", async t => {
  const session = setup(t);
  let settled = false;
  const capture = session.api.request({ op: "capture" }).then(reply => { settled = true; return reply; });
  assert.deepEqual(session.calls, [{ command: { op: "render", version: 1 }, width: 800, height: 600 }]);
  assert.equal(session.pendingFrames, 1);
  assert.equal(settled, false);
  session.flushFrame();
  await Promise.resolve();
  assert.equal(settled, false);
  assert.equal(session.counters.statsFreed, 0);
  session.flushFrame();
  const reply = await capture;
  assert.equal(reply.ok, true);
  assert.deepEqual(reply.observation, { frame: "7", tick: "12", scene: { id: "garden", revision: "4" } });
  assert.equal(reply.result.paused, true);
  assert.equal(reply.result.rendering.instances, 26);
  assert.equal(reply.result.rendering.resident_textures, 1);
  assert.equal(reply.result.rendering.resident_texture_bytes, 16);
  assert.equal(reply.result.rendering.texture_upload_bytes, 0);
  assert.equal(reply.result.rendering.shadows_enabled, true);
  assert.equal(reply.result.rendering.shadow_draws, 2);
  // Caster counts before their first read-back are explicit nulls.
  assert.equal(reply.result.rendering.shadow_near_casters, null);
  assert.equal(reply.result.rendering.shadow_moving_casters, 5);
  assert.equal(reply.result.rendering.ambient_occlusion, true);
  assert.equal(reply.result.rendering.bloom, false);
  assert.equal(reply.result.rendering.local_lights, 3);
  assert.equal(reply.result.rendering.lights_visible, null);
  assert.equal(reply.result.rendering.reflections, true);
  assert.equal(reply.result.rendering.particles, 40);
  assert.equal(session.counters.statsFreed, 1);
  assert.equal(session.counters.polls, 1);
  assert.deepEqual(session.busy, [true, false]);
});

test("capture disposal cancels its wait and never reads statistics from a freed runtime", async t => {
  const session = setup(t);
  const capture = session.api.request({ op: "capture" });
  session.dispose();
  session.lifecycle.abort();
  session.game.free();
  // Flushing callbacks also exercises a callback already scheduled before abort.
  session.flushFrame();
  session.flushFrame();
  const reply = await capture;
  assert.equal(reply.ok, false);
  assert.equal(reply.error.code, "session_closed");
  assert.equal(session.counters.accessesAfterFree, 0);
  assert.equal(session.counters.statsFreed, 0);
  assert.equal(session.counters.polls, 0);
  assert.equal(session.pendingFrames, 0);
});

test("explicit fractional capture presents the requested pose and holds the busy boundary", async t => {
  const session = setup(t);
  const capture = session.api.request({ op: "capture", alpha: 0.25 });
  assert.deepEqual(session.calls[0].command, { op: "present", alpha: 0.25, version: 1 });
  assert.equal((await session.api.request({ op: "step", ticks: 1 })).error.code, "command_busy");
  session.flushFrame();
  session.flushFrame();
  const reply = await capture;
  assert.equal(reply.ok, true);
  assert.equal(reply.observation.tick, "12");
  assert.equal(session.counters.frame, 7);
  assert.equal(session.calls.filter(call => call.command?.op === "render").length, 0);
});

test("rejected fractional capture does not wait, read statistics, or advance clocks", async t => {
  const session = setup(t, { command(command, reply) {
    return command.op === "present"
      ? JSON.stringify({ ok:false, error:{code:"invalid_presentation_alpha",path:"$.alpha",message:"out of range"} })
      : reply();
  } });
  const result = await session.api.request({ op: "capture", alpha: 2 });
  assert.equal(result.error.code, "invalid_presentation_alpha");
  assert.equal(session.pendingFrames, 0);
  assert.equal(session.counters.statsFreed, 0);
  assert.equal(session.counters.tick, 12);
});

test("aborting capture settles even when animation frames never run", async t => {
  const session = setup(t);
  let result;
  const capture = session.api.request({ op: "capture" }).then(reply => { result = reply; });
  session.lifecycle.abort();
  session.game.free();
  await new Promise(resolve => setImmediate(resolve));
  assert.notEqual(result, undefined, "hidden or disposed pages cannot strand a capture waiting for RAF");
  await capture;
  assert.equal(result.error.code, "session_closed");
  assert.equal(session.pendingFrames, 0);
  assert.equal(session.counters.accessesAfterFree, 0);
});

test("request shape, byte budget, and reload revisions reject before mutation", async t => {
  const session = setup(t);
  for (const command of [null, [], {}, { op: 7 }, { op: "pause", extra: true }]) {
    const reply = await session.api.request(command);
    assert.equal(reply.error.code, "invalid_request");
  }
  const oversized = await session.api.request({ op: "inspect", entity: "😀".repeat(5000) });
  assert.equal(oversized.error.code, "request_too_large", "budget counts UTF-8 bytes");
  for (const revision of [4, "-1", "01", "1.0", "1e3", ""]) {
    const reply = await session.api.request({ op: "reload", expected_revision: "4", revision });
    assert.equal(reply.error.code, "invalid_revision");
  }
  assert.ok(session.calls.every(call => call.command?.op === "status"));
  assert.equal(session.counters.statsFreed, 0);
});

test("unserializable command values produce structured errors", async t => {
  const session = setup(t);
  const cyclic = { op: "inspect" };
  cyclic.cycle = cyclic;
  for (const command of [cyclic, { op: "inspect", entity: 1n }]) {
    const reply = await session.api.request(command);
    assert.equal(reply.ok, false);
    assert.equal(reply.error.code, "invalid_request");
  }
  assert.ok(session.calls.every(call => call.command?.op === "status"));
});

test("runtime failures still return structured errors when status is unavailable", async t => {
  const session = setup(t, { command() { throw new Error("runtime unavailable"); } });
  const reply = await session.api.request({ op: "step", ticks: 1 });
  assert.equal(reply.ok, false);
  assert.equal(reply.error.code, "browser_command_failed");
  assert.match(reply.error.message, /runtime unavailable/);
  assert.equal(reply.observation, undefined);
  assert.deepEqual(session.busy, [true, false]);
});

test("statistics ownership is released when a host poll callback fails", async t => {
  const session = setup(t, { poll() { throw new Error("poll failed"); } });
  const reply = await session.api.request({ op: "status" });
  assert.equal(reply.ok, false);
  assert.equal(reply.error.code, "browser_command_failed");
  assert.equal(session.counters.statsFreed, 1);
  assert.equal(session.counters.polls, 1);
  assert.deepEqual(session.busy, [true, false]);
});

test("disposal of an older connection preserves its replacement", t => {
  const session = setup(t);
  const disposeReplacement = connectDevelopment(session.host);
  const replacement = window.__engineDev;
  assert.notEqual(replacement, session.api);
  session.dispose();
  assert.equal(window.__engineDev, replacement);
  disposeReplacement();
  assert.equal(window.__engineDev, undefined);
});

test("benchmark warms separately, samples exact fixed ticks, and excludes competing commands", async t => {
  const session = setup(t);
  const running = session.api.request({ op: "benchmark", mode: "step", warmup: 2, samples: 3 });
  const competing = await session.api.request({ op: "culling", enabled: false });
  assert.equal(competing.error.code, "command_busy");
  for (let index = 0; index < 6; index++) {
    session.flushFrame();
    await new Promise(resolve => setImmediate(resolve));
  }
  const reply = await running;
  assert.equal(reply.ok, true, JSON.stringify(reply));
  const result = reply.result.benchmark;
  assert.equal(result.samples, 3);
  assert.equal(result.warmup, 2);
  assert.equal(result.observation.start.tick, "12");
  assert.equal(result.observation.end.tick, "17");
  assert.deepEqual(result.frame_interval_ms, { min:16,median:16,p95:16,p99:16,max:16 });
  assert.equal(result.cpu_ms.total.median, 0.1 + 0.2);
  assert.equal(result.rendering.last.render_candidates, 26);
  assert.equal(result.rendering.last.culling_enabled, true);
  assert.deepEqual(result.rendering.ranges.culled_instances, { min:0,max:0 });
  assert.equal(session.calls.filter(call => call.command?.op === "step").length, 5);
  assert.equal(session.calls.filter(call => call.command?.op === "input").length, 1);
  assert.equal(session.counters.statsFreed, 6);
  assert.equal(session.pendingFrames, 0);
});

test("render-only benchmark does not advance simulation and invalid options do not mutate", async t => {
  const session = setup(t);
  for (const fields of [{ samples:0 }, { samples:241 }, { warmup:-1 }, { warmup:121 }, { mode:"frames" }, { samples:1.5 }, { extra:1 }]) {
    const reply = await session.api.request({ op: "benchmark", ...fields });
    assert.equal(reply.error.code, "invalid_benchmark");
  }
  assert.ok(session.calls.every(call => call.command?.op === "status"));
  const running = session.api.request({ op: "benchmark", mode: "render", warmup: 0, samples: 1 });
  session.flushFrame(10);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(session.calls.filter(call => call.command?.op === "render").length, 1);
  session.flushFrame(80);
  const reply = await running;
  assert.equal(reply.ok, true);
  assert.equal(reply.result.benchmark.observation.end.tick, "12");
  assert.equal(reply.result.benchmark.frame_interval_ms.median, 80);
  assert.equal(session.counters.frame, 7);
});

test("benchmark abort settles RAF wait and never touches a freed runtime", async t => {
  const session = setup(t);
  const running = session.api.request({ op: "benchmark", warmup: 0, samples: 2 });
  assert.equal(session.pendingFrames, 1);
  session.lifecycle.abort();
  session.game.free();
  const reply = await running;
  assert.equal(reply.error.code, "session_closed");
  assert.equal(session.pendingFrames, 0);
  assert.equal(session.counters.accessesAfterFree, 0);
  assert.equal(session.counters.tick, 12);
});

test("presentation benchmark samples fractional renders without stepping", async t => {
  const session = setup(t);
  const running = session.api.request({ op: "benchmark", mode: "present", warmup: 1, samples: 2 });
  for (let index = 0; index < 4; index++) {
    session.flushFrame();
    await new Promise(resolve => setImmediate(resolve));
  }
  const reply = await running;
  assert.equal(reply.ok, true, JSON.stringify(reply));
  assert.equal(reply.result.benchmark.observation.end.tick, "12");
  assert.equal(session.counters.frame, 7);
  assert.deepEqual(session.calls.filter(call => call.command?.op === "present").map(call => call.command),
    Array.from({ length:3 }, () => ({ op:"present", alpha:0.5, version:1 })));
  assert.equal(session.calls.filter(call => call.command?.op === "step" || call.command?.op === "render").length, 0);
});

test("benchmark deadline settles when browser animation frames stop", async t => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const session = setup(t);
  const running = session.api.request({ op: "benchmark", warmup: 0, samples: 2 });
  t.mock.timers.tick(10_000);
  const reply = await running;
  assert.equal(reply.error.code, "benchmark_timeout");
  assert.equal(session.pendingFrames, 0);
  assert.equal(session.counters.tick, 12);
  assert.equal(session.busy.at(-1), false);
});

test("benchmark rejects a resized canvas before advancing a sample", async t => {
  const session = setup(t);
  let width = 800;
  session.host.size = () => [width, 600];
  const running = session.api.request({ op: "benchmark", warmup: 0, samples: 2 });
  width = 400;
  session.flushFrame();
  const reply = await running;
  assert.equal(reply.error.code, "benchmark_interrupted");
  assert.equal(session.counters.tick, 12);
  assert.equal(session.pendingFrames, 0);
});

test("benchmark distributions use real values, interpolate percentiles and reject invalid measurements", () => {
  assert.deepEqual(distribution([30,10,20,40,50]), { min:10,median:30,p95:48,p99:49.6,max:50 });
  assert.deepEqual(distribution([0]), { min:0,median:0,p95:0,p99:0,max:0 });
  for (const values of [[], [NaN], [Infinity], [-1]]) assert.throws(() => distribution(values), { code:"invalid_measurement" });
});
