/** Virtual geometry (decision 044) in the Stress Lab's dense scene, in real Chrome. */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { access, mkdir, readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { chromium } from "@playwright/test";

const base = `${(process.env.DEMO_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "")}/`;
const denseFiles = fileURLToPath(new URL("../public/stress/dense/manifest.json", import.meta.url));
try {
  await access(denseFiles);
} catch {
  throw new Error("The dense scene's files are missing: run `cargo run -p engine_vgbuild --release --bin dense_assets`.");
}
const config = {
  seed: 1337, renderables: 2000, moving_percent: 5, hierarchy: false, interpolation: true, colliders: 0,
  sensor_percent: 0, dense_colliders: false, queries_per_tick: 0, mesh_variety: 8, textured: true, culling: true,
  scene: "dense", compressed_textures: false,
};
// A street-level view across the boulders, and one along the ground where
// nearer boulders hide farther ones.
const cameras = [
  { label: "street", position: [6, 2.2, 14], target: [-4, 1.4, -6] },
  { label: "ground", position: [-20, 1.2, -30], target: [40, 0.5, -80] },
];
const url = new URL("stress.html", base);
url.searchParams.set("engine_dev", "1");
url.searchParams.set("config", JSON.stringify(config));
const output = fileURLToPath(new URL(`../../artifacts/virtual-geometry-smoke-${randomUUID()}/`, import.meta.url));
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  ...(process.env.BROWSER_PATH ? { executablePath: resolve(process.env.BROWSER_PATH) } : { channel: process.env.BROWSER_CHANNEL ?? "chrome" }),
  headless: true,
  args: ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])],
});
const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 1 });
page.setDefaultTimeout(180_000);
const errors = [];
const recordError = value => { if (errors.length < 20) errors.push(String(value).slice(0, 2000)); };
page.on("pageerror", recordError);
page.on("console", entry => { if (entry.type() === "error") recordError(entry.text()); });
// Bytes of virtual mesh files fetched: prefixes at load, pages as views
// stream them.
let fetchedBytes = 0;
const responses = [];
page.on("response", response => {
  if (!response.url().endsWith(".evg")) return;
  responses.push(response.body().then(body => { fetchedBytes += body.length; }, () => {}));
});
const fetched = async () => { await Promise.all(responses); return fetchedBytes; };
const manifest = JSON.parse(await readFile(denseFiles, "utf8"));
let fileBytes = 0;
for (const name of manifest.meshes) fileBytes += (await stat(new URL(name, pathToFileURL(denseFiles)))).size;

async function command(op, fields = {}, expectedOk = true) {
  for (;;) {
    const reply = await page.evaluate(command => window.__engineDev.request(command), { op, ...fields });
    if (reply?.error?.code === "command_busy") { await delay(50); continue; }
    assert.equal(reply.ok, expectedOk, `${op}: ${JSON.stringify(reply).slice(0, 4000)}`);
    return reply;
  }
}

// Frames of `gpu` presentation until streaming settles, and the last one's
// rendering counters. Counts are read back a few frames late, so they must
// hold for several frames running: a selection, nothing loading, and the
// read-back frame lacking no page it wanted; under a budget, which leaves
// pages missing, the resident pages must stay put for longer than unused
// pages take to make room.
async function settle(gpu, budget = false) {
  let rendering, previous, quiet = 0;
  for (let frame = 0; quiet < (budget ? 24 : 8); frame++) {
    assert.ok(frame < 600, `streaming settles: ${JSON.stringify(rendering)}`);
    rendering = (await command("present", { alpha: 0, gpu })).result.rendering;
    const settled = rendering.virtual_clusters > 0 && rendering.virtual_loading_pages === 0 && (budget
      ? rendering.virtual_resident_pages === previous?.virtual_resident_pages
      : rendering.virtual_missing_pages === 0);
    quiet = settled ? quiet + 1 : 0;
    previous = rendering;
    await delay(40);
  }
  return rendering;
}

async function capture(name, gpu, budget = false) {
  await settle(gpu, budget);
  const reply = await command("capture", { alpha: 0, gpu });
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return { rendering: reply.result.rendering, png: bytes.toString("base64") };
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
    let changed = 0;
    for (let offset = 0; offset < pixels[0].data.length; offset += 4) {
      if ([0, 1, 2].some(channel => Math.abs(pixels[0].data[offset + channel] - pixels[1].data[offset + channel]) > tolerance)) changed++;
    }
    return { changed, total: pixels[0].width * pixels[0].height };
  }, { images, tolerance });
}

try {
  await page.goto(url.href, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__engineDev?.version === 1, null, { timeout: 120_000 });
  for (let attempt = 0; ; attempt++) {
    const reply = await page.evaluate(() => window.__engineDev.request({ op: "pause" }));
    if (reply.ok) break;
    assert.ok(attempt < 200, `pause: ${JSON.stringify(reply)}`);
    await delay(100);
  }

  // The page loaded only the files' prefixes; views stream the rest.
  const loadedBytes = await fetched();
  assert.ok(loadedBytes > 0 && loadedBytes < fileBytes / 4,
    `loading fetched ${loadedBytes} bytes of ${fileBytes}`);

  // The settings are discoverable and read as their scene section.
  const listing = await command("describe");
  assert.ok(listing.result.types.some(type => type.name === "VirtualGeometry"));
  const look = await command("inspect", { component: "Environment" });
  assert.deepEqual(look.result.value.virtual_geometry, { enabled: true, pixel_error: 1.5 });

  const cases = [];
  for (const camera of cameras) {
    await command("camera", { position: camera.position, target: camera.target, vertical_fov_radians: 0.96, near: 0.1, far: 5000 });
    // GPU-presented frames cull occluded and sub-pixel clusters; CPU
    // references draw every selected cluster: both show the same pixels.
    const cpu = await capture(`${camera.label}-cpu`, false);
    const gpu = await capture(`${camera.label}-gpu`, true);
    const pixels = await differentPixels([cpu.png, gpu.png], 8);
    assert.ok(pixels.changed <= pixels.total * 0.001, `${camera.label}: ${pixels.changed} of ${pixels.total} pixels differ`);
    // Both resolve through the visibility buffer (decision 045); shading as
    // clusters draw instead shows the same surfaces.
    assert.ok(cpu.rendering.visibility_buffer && gpu.rendering.visibility_buffer, `${camera.label}: visibility buffer`);
    await command("environment", { virtual_geometry: { visibility_buffer: false } });
    const forward = await capture(`${camera.label}-forward`, true);
    await command("environment", { virtual_geometry: { visibility_buffer: true } });
    assert.equal(forward.rendering.visibility_buffer, false);
    const forwardPixels = await differentPixels([gpu.png, forward.png], 8);
    assert.ok(forwardPixels.changed <= forwardPixels.total * 0.001,
      `${camera.label}: ${forwardPixels.changed} of ${forwardPixels.total} pixels differ from forward shading`);
    const rendering = gpu.rendering;
    assert.ok(rendering.virtual_clusters > 0 && rendering.virtual_triangles > 0, `${camera.label}: clusters drawn`);
    assert.ok(rendering.virtual_shadow_clusters > 0 && rendering.virtual_shadow_triangles > 0, `${camera.label}: clusters cast shadows`);
    assert.equal(rendering.dropped_virtual, 0, `${camera.label}: nothing dropped`);
    assert.equal(rendering.virtual_loading_pages, 0, `${camera.label}: streaming settled`);
    assert.equal(rendering.virtual_missing_pages, 0, `${camera.label}: every page the view wants is resident`);
    cases.push({
      label: camera.label, changed_pixels: pixels.changed, forward_changed_pixels: forwardPixels.changed,
      clusters: rendering.virtual_clusters,
      triangles: rendering.virtual_triangles, shadow_triangles: rendering.virtual_shadow_triangles,
      occluded_clusters: rendering.virtual_occluded_clusters, resident_pages: rendering.virtual_resident_pages,
    });
  }
  // Views streamed pages beyond the pinned ones.
  const streamed = { resident_pages: cases.at(-1).resident_pages, fetched_bytes: await fetched() - loadedBytes };
  assert.ok(streamed.resident_pages > 4 && streamed.fetched_bytes > 0, JSON.stringify(streamed));

  // A budget keeps that many pages resident (the pinned ones always) and
  // drops loads beyond it; captures of both paths still match.
  await command("environment", { virtual_geometry: { max_resident_pages: 8 } });
  const budgetCpu = await capture("budget-cpu", false, true);
  const budgetGpu = await capture("budget-gpu", true, true);
  const budgetPixels = await differentPixels([budgetCpu.png, budgetGpu.png], 8);
  assert.ok(budgetPixels.changed <= budgetPixels.total * 0.001, `budget: ${budgetPixels.changed} pixels differ`);
  const budgeted = budgetGpu.rendering;
  assert.ok(budgeted.virtual_resident_pages <= 8 && budgeted.virtual_clusters > 0, JSON.stringify(budgeted));
  assert.ok(budgeted.virtual_missing_pages > 0 && budgeted.dropped_virtual_pages > 0, JSON.stringify(budgeted));
  streamed.budget = {
    resident_pages: budgeted.virtual_resident_pages, missing_pages: budgeted.virtual_missing_pages,
    dropped_pages: budgeted.dropped_virtual_pages, triangles: budgeted.virtual_triangles,
  };
  await command("environment", { virtual_geometry: { enabled: true, pixel_error: 1.5 } });

  // A larger pixel error draws fewer triangles; disabled, the fallback cuts
  // draw instead of clusters. The control changes nothing in source.
  const fine = await settle(true);
  const coarse = await command("environment", { virtual_geometry: { pixel_error: 6 } });
  assert.deepEqual(coarse.result.environment_changed, ["virtual_geometry"]);
  const coarseRendering = await settle(true);
  assert.ok(coarseRendering.virtual_triangles < fine.virtual_triangles,
    `pixel error 6 draws fewer triangles (${coarseRendering.virtual_triangles} against ${fine.virtual_triangles})`);
  await command("environment", { virtual_geometry: { enabled: false } });
  let off;
  for (let frame = 0; frame < 8; frame++) off = (await command("present", { alpha: 0, gpu: true })).result.rendering;
  assert.equal(off.virtual_clusters, 0);
  assert.ok(off.instances > 0, "fallback cuts draw");
  const invalid = await command("environment", { virtual_geometry: { pixel_error: 0.1 } }, false);
  assert.equal(invalid.error.code, "invalid_environment");
  assert.equal(invalid.error.path, "$.virtual_geometry.pixel_error");
  await command("environment", { virtual_geometry: { enabled: true, pixel_error: 1.5 } });
  assert.ok((await settle(true)).virtual_clusters > 0);

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    ok: true, url: url.href, artifact: output, cases, loaded_bytes: loadedBytes, file_bytes: fileBytes, streamed,
    pixel_error: { fine: fine.virtual_triangles, coarse: coarseRendering.virtual_triangles },
    checks: ["prefixes_load", "settings_discoverable", "gpu_cpu_pixel_equivalence", "visibility_buffer_matches_forward", "clusters_drawn", "cluster_shadows", "nothing_dropped", "pages_stream", "budget_bounds_pages", "pixel_error_control", "fallback_when_disabled", "invalid_settings_refused", "clean_browser_diagnostics"],
  }));
} finally {
  await browser.close();
}
