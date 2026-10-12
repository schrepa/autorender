/** Virtual geometry in the Stress Lab's dense scene, in real Chrome. */
import assert from "node:assert/strict";
import { access, readFile, stat } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, differentPixels, launch, openPaused } from "./harness.mjs";
import { workspacePath } from "./workspace.mjs";

const denseFiles = workspacePath("web/public/stress/dense/manifest.json");
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
const url = devUrl("stress.html", { config });
const output = await artifacts("virtual-geometry-smoke");
const { browser, page, errors } = await launch({ timeout: 180_000 });
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

const command = commander(page);

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
    const settled = rendering.virtual_geometry.clusters > 0 && rendering.virtual_geometry.loading_pages === 0 && (budget
      ? rendering.virtual_geometry.resident_pages === previous?.virtual_geometry?.resident_pages
      : rendering.virtual_geometry.missing_pages === 0);
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


try {
  await openPaused(page, url, { timeout: 120_000 });

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
    const pixels = await differentPixels(page, [cpu.png, gpu.png], { tolerance: 8 });
    assert.ok(pixels.changed <= pixels.total * 0.001, `${camera.label}: ${pixels.changed} of ${pixels.total} pixels differ`);
    // Both shade clusters as they draw; resolving them through the
    // visibility buffer instead shows the same surfaces.
    assert.ok(!cpu.rendering.virtual_geometry.visibility_buffer && !gpu.rendering.virtual_geometry.visibility_buffer, `${camera.label}: forward shading`);
    await command("environment", { virtual_geometry: { visibility_buffer: true } });
    const resolved = await capture(`${camera.label}-visibility`, true);
    await command("environment", { virtual_geometry: { visibility_buffer: false } });
    assert.equal(resolved.rendering.virtual_geometry.visibility_buffer, true);
    const forwardPixels = await differentPixels(page, [gpu.png, resolved.png], { tolerance: 8 });
    assert.ok(forwardPixels.changed <= forwardPixels.total * 0.001,
      `${camera.label}: ${forwardPixels.changed} of ${forwardPixels.total} pixels differ through the visibility buffer`);
    const rendering = gpu.rendering;
    assert.ok(rendering.virtual_geometry.clusters > 0 && rendering.virtual_geometry.triangles > 0, `${camera.label}: clusters drawn`);
    assert.ok(rendering.virtual_geometry.shadow_clusters > 0 && rendering.virtual_geometry.shadow_triangles > 0, `${camera.label}: clusters cast shadows`);
    assert.equal(rendering.virtual_geometry.dropped, 0, `${camera.label}: nothing dropped`);
    assert.equal(rendering.virtual_geometry.loading_pages, 0, `${camera.label}: streaming settled`);
    assert.equal(rendering.virtual_geometry.missing_pages, 0, `${camera.label}: every page the view wants is resident`);
    cases.push({
      label: camera.label, changed_pixels: pixels.changed, forward_changed_pixels: forwardPixels.changed,
      clusters: rendering.virtual_geometry.clusters,
      triangles: rendering.virtual_geometry.triangles, shadow_triangles: rendering.virtual_geometry.shadow_triangles,
      occluded_clusters: rendering.virtual_geometry.occluded_clusters, resident_pages: rendering.virtual_geometry.resident_pages,
    });
  }
  // Views streamed pages beyond the pinned ones.
  const streamed = { resident_pages: cases.at(-1).resident_pages, fetched_bytes: await fetched() - loadedBytes };
  assert.ok(streamed.resident_pages > 4 && streamed.fetched_bytes > 0, JSON.stringify(streamed));

  // CPU-prepared frames plan only what they see: looking down at bare
  // ground retires the other meshes and rebuilds the arena, and looking back
  // rebuilds it again. Meshes that stayed keep their streamed pages, the
  // pool never overruns, and the view looks as it did.
  const last = cameras.at(-1);
  const reference = await capture(`${last.label}-reference`, false);
  await command("camera", { position: [0.5, 0.6, 0.5], target: [0.5, -10, 0.52], vertical_fov_radians: 0.3, near: 0.05, far: 5000 });
  let ground;
  for (let frame = 0; frame < 12; frame++) ground = (await command("present", { alpha: 0, gpu: false })).result.rendering;
  assert.ok(ground.instances < reference.rendering.instances, `the plan holds only what it sees: ${JSON.stringify(ground)}`);
  await command("camera", { position: last.position, target: last.target, vertical_fov_radians: 0.96, near: 0.1, far: 5000 });
  const rebuilt = await capture(`${last.label}-rebuilt`, false);
  const rebuiltPixels = await differentPixels(page, [reference.png, rebuilt.png], { tolerance: 8 });
  assert.ok(rebuiltPixels.changed <= rebuiltPixels.total * 0.001, `rebuilt arena: ${rebuiltPixels.changed} pixels differ`);
  streamed.rebuilt = { resident_pages: rebuilt.rendering.virtual_geometry.resident_pages, changed_pixels: rebuiltPixels.changed };

  // A budget keeps that many pages resident (the pinned ones always) and
  // drops loads beyond it; captures of both paths still match.
  await command("environment", { virtual_geometry: { max_resident_pages: 8 } });
  const budgetCpu = await capture("budget-cpu", false, true);
  const budgetGpu = await capture("budget-gpu", true, true);
  const budgetPixels = await differentPixels(page, [budgetCpu.png, budgetGpu.png], { tolerance: 8 });
  assert.ok(budgetPixels.changed <= budgetPixels.total * 0.001, `budget: ${budgetPixels.changed} pixels differ`);
  const budgeted = budgetGpu.rendering;
  assert.ok(budgeted.virtual_geometry.resident_pages <= 8 && budgeted.virtual_geometry.clusters > 0, JSON.stringify(budgeted));
  assert.ok(budgeted.virtual_geometry.missing_pages > 0 && budgeted.virtual_geometry.dropped_pages > 0, JSON.stringify(budgeted));
  streamed.budget = {
    resident_pages: budgeted.virtual_geometry.resident_pages, missing_pages: budgeted.virtual_geometry.missing_pages,
    dropped_pages: budgeted.virtual_geometry.dropped_pages, triangles: budgeted.virtual_geometry.triangles,
  };
  await command("environment", { virtual_geometry: { enabled: true, pixel_error: 1.5 } });

  // A larger pixel error draws fewer triangles; disabled, the fallback cuts
  // draw instead of clusters. The control changes nothing in source.
  const fine = await settle(true);
  const coarse = await command("environment", { virtual_geometry: { pixel_error: 6 } });
  assert.deepEqual(coarse.result.environment_changed, ["virtual_geometry"]);
  const coarseRendering = await settle(true);
  assert.ok(coarseRendering.virtual_geometry.triangles < fine.virtual_geometry.triangles,
    `pixel error 6 draws fewer triangles (${coarseRendering.virtual_geometry.triangles} against ${fine.virtual_geometry.triangles})`);
  await command("environment", { virtual_geometry: { enabled: false } });
  let off;
  for (let frame = 0; frame < 8; frame++) off = (await command("present", { alpha: 0, gpu: true })).result.rendering;
  assert.equal(off.virtual_geometry.clusters, 0);
  assert.ok(off.instances > 0, "fallback cuts draw");
  const invalid = await command("environment", { virtual_geometry: { pixel_error: 0.1 } }, false);
  assert.equal(invalid.error.code, "invalid_environment");
  assert.equal(invalid.error.path, "$.virtual_geometry.pixel_error");
  await command("environment", { virtual_geometry: { enabled: true, pixel_error: 1.5 } });
  assert.ok((await settle(true)).virtual_geometry.clusters > 0);

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    ok: true, url: url.href, artifact: output, cases, loaded_bytes: loadedBytes, file_bytes: fileBytes, streamed,
    pixel_error: { fine: fine.virtual_geometry.triangles, coarse: coarseRendering.virtual_geometry.triangles },
    checks: ["prefixes_load", "settings_discoverable", "gpu_cpu_pixel_equivalence", "visibility_buffer_matches_forward", "arena_rebuilds_keep_pages", "clusters_drawn", "cluster_shadows", "nothing_dropped", "pages_stream", "budget_bounds_pages", "pixel_error_control", "fallback_when_disabled", "invalid_settings_refused", "clean_browser_diagnostics"],
  }));
} finally {
  await browser.close();
}
