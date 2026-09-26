/** GPU presentation (decision 014) against the exact CPU reference in real Chrome. */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { chromium } from "@playwright/test";

const base = `${(process.env.DEMO_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "")}/`;
// Every renderable moves; hierarchy, textures and several meshes exercise
// composition, per-group bounds and compaction across draw groups.
const config = {
  seed: 7, renderables: 6000, moving_percent: 100, hierarchy: true, interpolation: true, colliders: 200,
  sensor_percent: 10, dense_colliders: false, queries_per_tick: 8, mesh_variety: 8, textured: true, culling: true,
};
// Without interpolation, moving nodes compose on the GPU from their current
// locals instead of CPU authority rows; they must match the CPU reference too.
const workloads = [
  { label: "interpolated", config, fractional: true },
  { label: "authoritative", config: { ...config, interpolation: false }, fractional: false },
];
const workloadUrl = workload => {
  const url = new URL("stress.html", base);
  url.searchParams.set("engine_dev", "1");
  url.searchParams.set("config", JSON.stringify(workload.config));
  return url;
};
const output = fileURLToPath(new URL(`../../artifacts/gpu-presentation-smoke-${randomUUID()}/`, import.meta.url));
const browser = await chromium.launch({
  ...(process.env.BROWSER_PATH ? { executablePath: resolve(process.env.BROWSER_PATH) } : { channel: process.env.BROWSER_CHANNEL ?? "chrome" }),
  headless: true,
  args: ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])],
});
const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 1 });
page.setDefaultTimeout(30_000);
const errors = [];
const recordError = value => { if (errors.length < 20) errors.push(String(value).slice(0, 2000)); };
page.on("pageerror", recordError);
page.on("console", entry => { if (entry.type() === "error") recordError(entry.text()); });

async function command(op, fields = {}, expectedOk = true) {
  const reply = await page.evaluate(command => window.__engineDev.request(command), { op, ...fields });
  assert.equal(reply.ok, expectedOk, `${op}: ${JSON.stringify(reply).slice(0, 4000)}`);
  return reply;
}

async function capture(name, fields) {
  const reply = await command("capture", fields);
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return { reply, png: bytes.toString("base64") };
}

// Pixels whose largest channel difference exceeds the tolerance.
async function differentPixels(images, tolerance) {
  return page.evaluate(async ({ images, tolerance }) => {
    const pixels = await Promise.all(images.map(async png => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, image.width, image.height);
    }));
    if (pixels[0].width !== pixels[1].width || pixels[0].height !== pixels[1].height) throw new Error("Capture dimensions changed.");
    let changed = 0;
    for (let offset = 0; offset < pixels[0].data.length; offset += 4) {
      if ([0, 1, 2].some(channel => Math.abs(pixels[0].data[offset + channel] - pixels[1].data[offset + channel]) > tolerance)) changed++;
    }
    return { changed, total: pixels[0].width * pixels[0].height };
  }, { images, tolerance });
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
  const pixels = await differentPixels([cpu.png, gpu.png], 8);
  assert.ok(pixels.changed <= pixels.total * 0.001, `${label}: ${pixels.changed} of ${pixels.total} pixels differ`);
  const candidates = reference.render_candidates;
  assert.ok(Math.abs(gpu.rendering.visible_instances - reference.visible_instances) <= Math.max(2, candidates * 0.001),
    `${label}: GPU visible ${gpu.rendering.visible_instances} vs CPU ${reference.visible_instances}`);
  assert.equal(gpu.rendering.render_candidates, candidates);
  return { label, alpha, candidates, cpu_visible: reference.visible_instances, gpu_visible: gpu.rendering.visible_instances, changed_pixels: pixels.changed, cpu: cpu.png };
}

async function verify(workload) {
  const url = workloadUrl(workload);
  await page.goto(url.href, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__engineDev?.version === 1, null, { timeout: 60_000 });
  const deadline = Date.now() + 5000;
  while (true) {
    const reply = await page.evaluate(() => window.__engineDev.request({ op: "pause" }));
    // Only this rejection is retried: it occurs before the command dispatches.
    if (reply.error?.code === "command_busy" && Date.now() < deadline) { await delay(25); continue; }
    assert.equal(reply.ok, true, JSON.stringify(reply).slice(0, 4000));
    break;
  }
  await mkdir(output, { recursive: true });
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
    const motion = await differentPixels([cases[0].cpu, cases[2].cpu], 8);
    assert.ok(motion.changed > 0, "Fractions must present visibly different poses.");
  }
  const after = await command("status");
  assert.equal(after.observation.tick, overview.observation.tick, "Previews must not advance simulation.");

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
    checks: ["gpu_cpu_pixel_equivalence", "authoritative_gpu_composition", "gpu_visible_count", "gpu_culling_exercised", "fractional_motion", "previews_keep_ticks", "inspection_materializes", "ordinary_frames_gpu", "clean_browser_diagnostics"],
  }));
} finally {
  await browser.close();
}
