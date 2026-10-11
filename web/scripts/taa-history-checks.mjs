/** Shared end-to-end history recovery checks for the uncertainty TAA variants. */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { artifacts, commander, devUrl, launch, openPaused } from "./harness.mjs";

const still = {
  seed: 7, renderables: 6000, moving_percent: 0, hierarchy: true, interpolation: true, colliders: 0,
  sensor_percent: 0, dense_colliders: false, queries_per_tick: 0, mesh_variety: 8, textured: true, culling: true,
};
const moving = { ...still, moving_percent: 100, colliders: 200, sensor_percent: 10, queries_per_tick: 8 };
const replacement = { ...still, seed: 13, renderables: 3000 };
const views = [
  { position: [0, 30, 0], target: [60, 20, 40], vertical_fov_radians: 0.9, near: 0.1, far: 400 },
  { position: [60, 40, 80], target: [0, 10, 0], vertical_fov_radians: 0.9, near: 0.1, far: 400 },
];

// Same image statistics and tolerances as taa-smoke, without inspecting
// private history pixels or introducing a production readback.
async function compare(page, images) {
  return page.evaluate(async images => {
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
    const [a, b] = pixels;
    if (a.width !== b.width || a.height !== b.height) throw new Error("Capture dimensions changed");
    const changed = { 0: 0, 8: 0, 24: 0 };
    const distinct = pixels.map(() => new Set());
    for (let at = 0; at < a.data.length; at += 4) {
      let largest = 0;
      for (let c = 0; c < 3; c++) largest = Math.max(largest, Math.abs(a.data[at + c] - b.data[at + c]));
      for (const tolerance of [0, 8, 24]) if (largest > tolerance) changed[tolerance]++;
      pixels.forEach(({ data }, index) => distinct[index].add(data[at] << 16 | data[at + 1] << 8 | data[at + 2]));
    }
    const variation = ({ data, width, height }) => {
      let sum = 0;
      for (let y = 0; y + 1 < height; y++) for (let x = 0; x + 1 < width; x++) {
        const at = (y * width + x) * 4;
        for (let c = 0; c < 3; c++) sum += Math.abs(data[at + c] - data[at + 4 + c])
          + Math.abs(data[at + c] - data[at + width * 4 + c]);
      }
      return sum / (width * height);
    };
    let blocks = 0, blockSum = 0;
    for (let by = 0; by + 8 <= a.height; by += 8) for (let bx = 0; bx + 8 <= a.width; bx += 8) {
      for (let c = 0; c < 3; c++) {
        let first = 0, second = 0;
        for (let y = by; y < by + 8; y++) for (let x = bx; x < bx + 8; x++) {
          const at = (y * a.width + x) * 4 + c;
          first += a.data[at]; second += b.data[at];
        }
        blockSum += Math.abs(first - second) / 64;
      }
      blocks++;
    }
    return { width: a.width, height: a.height, total: a.width * a.height, changed,
      block_difference: blockSum / (blocks * 3), variation: pixels.map(variation),
      distinct_colors: distinct.map(value => value.size) };
  }, images);
}

export async function checkHistory(name, resources = {}) {
  const output = await artifacts(name);
  const { browser, page, errors } = await launch({ width: 800, height: 600, timeout: 120_000 });
  const context = page.context();
  // The normal ResizeObserver follows a canvas that fills the browser viewport.
  // Only the smoke page's layout changes; the renderer receives real resize events.
  await context.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = ".stress .viewport{position:fixed!important;inset:0!important;width:100vw!important;height:100vh!important;min-height:0!important;border:0!important;border-radius:0!important;z-index:1000}";
    document.head.append(style);
  }));
  await resources.install?.(context);
  const results = { frames: [], comparisons: {}, resources: [], highlands: null };
  const command = commander(page, { deadline: 120_000 });
  const frames = target => target.locator("canvas").evaluate(canvas => Number(canvas.dataset.frames ?? 0));
  const dimensions = target => target.locator("canvas").evaluate(canvas => [canvas.width, canvas.height]);
  const observe = async (target, label, expected) => {
    if (resources.observe) results.resources.push(await resources.observe(target, label, expected));
  };
  async function open(target, config = still, query = { taa: "on" }) {
    const url = devUrl("stress.html", { config, ...query });
    await openPaused(target, url, { timeout: 120_000 });
    const control = commander(target, { deadline: 120_000 });
    await control("render");
    await control("step", { ticks: 3 });
    return url.href;
  }
  async function screenshot(target, label) {
    const bytes = await target.locator("canvas").screenshot({ path: `${output}/${label}.png`,
      style: ".viewport > :not(canvas) { visibility:hidden!important }" });
    return bytes.toString("base64");
  }
  async function ordinary(target, label, active = true) {
    const control = commander(target, { deadline: 120_000 });
    const before = await frames(target);
    await control("resume");
    await target.waitForFunction(before => Number(document.querySelector("canvas").dataset.frames) >= before + 32,
      before, { timeout: 120_000 });
    const after = await frames(target);
    const status = await control("status");
    assert.equal(status.result.rendering.temporal_antialiasing.active, active, `${label}: TAA activity`);
    assert.equal(status.result.rendering.render_passes.includes("temporal_resolve"), active, `${label}: resolve pass`);
    const image = await screenshot(target, label);
    await control("pause");
    results.frames.push({ label, before, after, delta: after - before, size: await dimensions(target),
      rendering: status.result.rendering });
    assert.ok(after - before >= 32, `${label}: at least 32 ordinary frames`);
    await observe(target, label, active ? "active" : "inactive");
    return image;
  }
  async function content(label, candidate, reference, size) {
    const value = await compare(page, [candidate, reference]);
    results.comparisons[label] = value;
    assert.deepEqual([value.width, value.height], size, `${label}: screenshot size`);
    assert.ok(value.distinct_colors.every(count => count > 16), `${label}: textured geometry must remain visible`);
    assert.ok(value.block_difference < 3, `${label}: recovered content differs: ${JSON.stringify(value)}`);
    return value;
  }
  async function fresh(label, config = still, size = [800, 600], camera = null) {
    const reference = await context.newPage();
    reference.setDefaultTimeout(120_000);
    reference.on("pageerror", error => errors.push(String(error)));
    reference.on("console", entry => { if (entry.type() === "error") errors.push(entry.text()); });
    try {
      await reference.setViewportSize({ width: size[0], height: size[1] });
      await open(reference, config);
      if (camera) await commander(reference)("camera", camera);
      return await ordinary(reference, `${label}-fresh`);
    } finally { await reference.close(); }
  }
  try {
    results.url = await open(page, still, { taa: "off" });
    await command("capture");
    const exactOff = await screenshot(page, "exact-off-before");
    await observe(page, "fresh-exact-off", "none");
    await open(page);
    await command("capture");
    const exactOn = await screenshot(page, "fresh-exact-on");
    await observe(page, "fresh-exact-on", "none");
    assert.equal((await compare(page, [exactOff, exactOn])).changed[0], 0, "fresh exact frames ignore TAA");
    const initial = await ordinary(page, "initial");
    const stillDetail = (await compare(page, [initial, initial])).variation[0];
    await command("capture");
    await observe(page, "exact-after-ordinary", "retained");
    await command("preview", { alpha: 0.5, gpu: false });
    await observe(page, "cpu-preview-after-ordinary", "retained");
    for (let cycle = 1; cycle <= 2; cycle++) {
      await command("environment", { temporal_antialiasing: { enabled: false } });
      assert.deepEqual((await command("inspect", { component: "TemporalAntialiasing" })).result.value, { enabled: false });
      await ordinary(page, `off-${cycle}`, false);
      await command("environment", { temporal_antialiasing: { enabled: true } });
      assert.deepEqual((await command("inspect", { component: "TemporalAntialiasing" })).result.value, { enabled: true });
      const image = await ordinary(page, `reenabled-${cycle}`);
      await content(`reenabled-${cycle}`, image, await fresh(`reenabled-${cycle}`), [800, 600]);
    }
    await command("environment", { temporal_antialiasing: { enabled: false } });
    await command("capture");
    const exactAfter = await screenshot(page, "exact-off-after");
    assert.equal((await compare(page, [exactOff, exactAfter])).changed[0], 0, "off exact frames remain identical after recovery");
    await command("environment", { temporal_antialiasing: { enabled: true } });
    await ordinary(page, "before-resize");
    for (const size of [[960, 540], [800, 600]]) {
      const label = `resize-${size.join("x")}`;
      await page.setViewportSize({ width: size[0], height: size[1] });
      const image = await ordinary(page, label);
      assert.deepEqual(await dimensions(page), size, `${label}: actual canvas resized`);
      await content(label, image, await fresh(label, still, size), size);
    }
    await command("game", { name: "workload", args: { config: replacement } });
    // Replacing a workload restores its environment defaults.
    await command("environment", { temporal_antialiasing: { enabled: true } });
    await command("capture");
    const replacedExact = await screenshot(page, "replacement-exact");
    assert.ok((await compare(page, [exactOff, replacedExact])).changed[8] > 0, "workload replacement must change content");
    const replaced = await ordinary(page, "replacement");
    await content("replacement", replaced, await fresh("replacement", replacement), [800, 600]);
    for (const [index, camera] of views.entries()) {
      await command("camera", camera);
      const image = await ordinary(page, `camera-${index}`);
      await content(`camera-${index}`, image, await fresh(`camera-${index}`, replacement, [800, 600], camera), [800, 600]);
    }
    // A new renderer/page recovers without claiming automatic device recovery.
    await open(page);
    const reloaded = await ordinary(page, "reloaded");
    await content("reloaded", reloaded, await fresh("reloaded"), [800, 600]);
    await open(page, still, { taa: "on", presentation: "cpu" });
    await observe(page, "fresh-cpu-exact", "none");
    const cpu = await ordinary(page, "cpu", false);
    await observe(page, "cpu-ordinary", "none");
    const cpuPair = await compare(page, [exactOff, cpu]);
    results.comparisons.cpu = cpuPair;
    assert.ok(cpuPair.changed[8] <= cpuPair.total * 0.001, `CPU presentation stays exact: ${JSON.stringify(cpuPair)}`);
    await open(page, moving);
    const motion = await ordinary(page, "moving");
    const movingPair = await compare(page, [motion, motion]);
    results.comparisons.moving = movingPair;
    assert.ok(movingPair.distinct_colors[0] > 16, "moving geometry remains visible");
    assert.ok(movingPair.variation[0] > stillDetail * 0.8, "moving frames retain the existing detail lower bound");

    await page.setViewportSize({ width: 1400, height: 900 });
    const highlandsUrl = devUrl("stress.html", { preset: "highlands", taa: "on", rocks: "off", jobcache: "off" });
    await openPaused(page, highlandsUrl, { timeout: 120_000 });
    await command("render");
    const initialStatus = await command("status");
    const sequence = [];
    // Ordinary stationary-camera foliage frames, followed by a changing view.
    for (let index = 0; index < 4; index++) {
      await ordinary(page, `highlands-still-${index}`);
      sequence.push(`highlands-still-${index}.png`);
    }
    const viewed = await command("game", { name: "view", args: { name: "overview" } });
    for (let index = 0; index < 4; index++) {
      await ordinary(page, `highlands-overview-${index}`);
      sequence.push(`highlands-overview-${index}.png`);
    }
    results.highlands = { url: highlandsUrl.href, size: await dimensions(page), initial: initialStatus,
      overview: viewed, sequence, interpretation: "Ordinary-frame visual review material; no automatic quality or timing claim." };
    assert.deepEqual(await dimensions(page), [1400, 900], "Highlands uses the default output size");
    assert.deepEqual(errors, [], "clean browser and GPU diagnostics");
    await writeFile(`${output}/results.json`, JSON.stringify({ ok: true, ...results }, null, 2));
    console.log(JSON.stringify({ ok: true, artifact: output, comparisons: results.comparisons,
      frame_counts: results.frames.map(({ label, delta }) => ({ label, delta })) }));
  } catch (error) {
    await writeFile(`${output}/results.json`, JSON.stringify({ ok: false, error: String(error), errors, ...results }, null, 2));
    throw error;
  } finally { await browser.close(); }
}
