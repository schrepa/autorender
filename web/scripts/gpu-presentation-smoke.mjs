/** GPU presentation against the exact CPU reference in real Chrome, and its
 * occlusion and sub-pixel culling settings: inspected, changed by the
 * paused environment control and observed in GPU-presented frames. */
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, differentPixels, launch, openPaused } from "./harness.mjs";

// Every renderable moves; hierarchy, textures and several meshes exercise
// composition, per-group bounds and compaction across draw groups.
const config = {
  seed: 7, renderables: 6000, moving_percent: 100, hierarchy: true, interpolation: true, colliders: 200,
  sensor_percent: 10, dense_colliders: false, queries_per_tick: 8, mesh_variety: 8, textured: true, culling: true,
};
// Without interpolation, moving nodes compose on the GPU from their current
// locals instead of CPU authority rows; they must match the CPU reference too.
// A still workload settles first: GPU frames then shadow it from the kept map,
// CPU references from the map redrawn every frame.
const workloads = [
  { label: "interpolated", config, fractional: true, culling: true },
  { label: "authoritative", config: { ...config, interpolation: false }, fractional: false },
  { label: "still", config: { ...config, moving_percent: 0 }, fractional: false, settle: true },
];
const output = await artifacts("gpu-presentation-smoke");
const { browser, page, errors } = await launch();
const command = commander(page);

async function capture(name, fields) {
  const reply = await command("capture", fields);
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return { reply, png: bytes.toString("base64") };
}

// Counters partition candidates on both paths; this workload is below budget.
function partition(rendering, label) {
  const { render_candidates, invalid_transforms, missing_assets, culled_instances, visible_instances, dropped_instances = 0, instances } = rendering;
  assert.equal(render_candidates, invalid_transforms + missing_assets + culled_instances + visible_instances, `${label}: counters must partition candidates`);
  assert.equal(dropped_instances, 0, `${label}: workload is below the instance budget`);
  assert.equal(instances, visible_instances, `${label}: submitted equals visible below budget`);
  return rendering;
}

// A GPU-presented frame reports the visible count read back from an earlier
// frame; repeating the identical preview returns that frame's own count.
async function gpuCapture(name, alpha) {
  await command("present", { alpha, gpu: true });
  await delay(100);
  const { reply, png } = await capture(name, { alpha, gpu: true });
  assert.equal(reply.observation.presentation.gpu_pending, true, "GPU previews leave CPU presented poses pending.");
  return { rendering: partition(reply.result.rendering, `${name} gpu`), png };
}

async function compare(label, alpha) {
  const cpu = await capture(`${label}-cpu`, { alpha });
  assert.equal(cpu.reply.observation.presentation.gpu_pending, false);
  const reference = partition(cpu.reply.result.rendering, `${label} cpu`);
  const gpu = await gpuCapture(`${label}-gpu`, alpha);
  // Tolerate edge pixels from f32 rounding differences between WGSL and glam.
  const pixels = await differentPixels(page, [cpu.png, gpu.png], { tolerance: 8 });
  assert.ok(pixels.changed <= pixels.total * 0.001, `${label}: ${pixels.changed} of ${pixels.total} pixels differ`);
  const candidates = reference.render_candidates;
  assert.ok(Math.abs(gpu.rendering.visible_instances - reference.visible_instances) <= Math.max(2, candidates * 0.001),
    `${label}: GPU visible ${gpu.rendering.visible_instances} vs CPU ${reference.visible_instances}`);
  assert.equal(gpu.rendering.render_candidates, candidates);
  return { label, alpha, candidates, cpu_visible: reference.visible_instances, gpu_visible: gpu.rendering.visible_instances, changed_pixels: pixels.changed, cpu: cpu.png };
}

// Occlusion and sub-pixel culling are environment sections: on by default,
// run by GPU-presented frames (two drawing phases around the occlusion test),
// and turned off and on again by the paused environment control without
// changing what the frame shows.
async function cullingSettings(label) {
  const inspect = async component => (await command("inspect", { component })).result.value;
  assert.deepEqual(await inspect("OcclusionCulling"), { enabled: true });
  assert.deepEqual(await inspect("SubpixelCulling"), { enabled: true });
  const preview = async name => (await capture(name, { alpha: 0.5, gpu: true })).reply.result.rendering;
  const on = await preview(`${label}-culling-on`);
  assert.equal(on.occlusion_culling.active, true, "GPU frames run occlusion culling by default");
  assert.equal(on.subpixel_culling.active, true, "GPU frames run sub-pixel culling by default");
  assert.ok(on.render_passes.includes("occlusion_test"), "occlusion culling runs its test pass");
  const off = await command("environment", { occlusion_culling: { enabled: false }, subpixel_culling: { enabled: false } });
  assert.deepEqual(off.result.environment_changed, ["occlusion_culling", "subpixel_culling"]);
  assert.deepEqual(await inspect("OcclusionCulling"), { enabled: false });
  const without = await preview(`${label}-culling-off`);
  assert.equal(without.occlusion_culling.active, false, "turned off, GPU frames skip occlusion culling");
  assert.equal(without.subpixel_culling.active, false, "turned off, GPU frames skip sub-pixel culling");
  assert.equal(without.occlusion_culling.instances, null);
  assert.equal(without.subpixel_culling.instances, null);
  assert.ok(!without.render_passes.includes("occlusion_test"), "no occlusion test runs while it is off");
  const invalid = await command("environment", { occlusion_culling: { enabled: true, phases: 2 } }, false);
  assert.equal(invalid.error.code, "unknown_field", "unknown occlusion culling fields are rejected");
  await command("environment", { occlusion_culling: { enabled: true }, subpixel_culling: { enabled: true } });
  assert.equal((await preview(`${label}-culling-again`)).occlusion_culling.active, true);
}

async function verify(workload) {
  const url = devUrl("stress.html", { config: workload.config });
  await openPaused(page, url);
  if (workload.settle) {
    await command("resume");
    await delay(1500);
    await command("pause");
  }
  await command("render");
  await command("step", { ticks: 3 });
  const cases = [];
  const overview = await command("status");
  for (const [label, pose] of [
    ["default-view", null],
    ["partial-view", { position: [0, 30, 0], target: [60, 20, 40], vertical_fov_radians: 0.9, near: 0.1, far: 400 }],
  ]) {
    if (pose) await command("camera", pose);
    for (const alpha of [0, 0.37, 0.81]) cases.push(await compare(`${workload.label}-${label}-${alpha}`, alpha));
  }
  assert.ok(cases.some(value => value.cpu_visible < value.candidates), "A view must exercise GPU culling.");
  assert.ok(cases.every(value => value.cpu_visible > 0), "Every view must draw instances.");
  if (workload.fractional) {
    const motion = await differentPixels(page, [cases[0].cpu, cases[2].cpu], { tolerance: 8 });
    assert.ok(motion.changed > 0, "Fractions must present visibly different poses.");
  }
  const after = await command("status");
  assert.equal(after.observation.tick, overview.observation.tick, "Previews must not advance simulation.");
  if (workload.culling) await cullingSettings(workload.label);

  // Inspection materializes the pending GPU frame's exact CPU reference.
  const state = await command("inspect", { component: "PresentationState" });
  assert.equal(state.observation.presentation.gpu_pending, true);
  assert.equal(state.result.value.gpu_pending, false);
  assert.equal(state.result.value.mode, "interpolated");

  // Ordinary frames use GPU presentation by default.
  await command("resume");
  await delay(500);
  const resumed = await command("pause");
  assert.equal(resumed.observation.presentation.mode, "interpolated");
  assert.equal(resumed.observation.presentation.gpu_pending, true);
  assert.deepEqual(errors, []);
  return { url: url.href, cases: cases.map(({ cpu, ...value }) => value) };
}

try {
  const results = [];
  for (const workload of workloads) results.push(await verify(workload));
  console.log(JSON.stringify({
    ok: true, urls: results.map(result => result.url), artifact: output,
    cases: results.flatMap(result => result.cases),
    checks: ["gpu_cpu_pixel_equivalence", "authoritative_gpu_composition", "gpu_visible_count", "gpu_culling_exercised", "fractional_motion", "previews_keep_ticks", "culling_settings_inspected", "culling_settings_controlled", "culling_settings_observed", "culling_settings_validated", "inspection_materializes", "ordinary_frames_gpu", "clean_browser_diagnostics"],
  }));
} finally {
  await browser.close();
}
