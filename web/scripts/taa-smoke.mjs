/** Temporal antialiasing (decision 017) in real Chrome: ordinary GPU-presented
 * frames converge to a stable, smoother image, while captures and CPU-presented
 * frames stay exact. */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { chromium } from "@playwright/test";

const base = `${(process.env.DEMO_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "")}/`;
// Textured meshes of several shapes give many edges; nothing moves, so any
// difference between ordinary frames and captures is the antialiasing.
const still = {
  seed: 7, renderables: 6000, moving_percent: 0, hierarchy: true, interpolation: true, colliders: 0,
  sensor_percent: 0, dense_colliders: false, queries_per_tick: 0, mesh_variety: 8, textured: true, culling: true,
};
const moving = { ...still, moving_percent: 100, colliders: 200, sensor_percent: 10, queries_per_tick: 8 };
const output = fileURLToPath(new URL(`../../artifacts/taa-smoke-${randomUUID()}/`, import.meta.url));
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

async function command(op, fields = {}) {
  for (const deadline = Date.now() + 5000; ;) {
    const reply = await page.evaluate(command => window.__engineDev.request(command), { op, ...fields });
    // Only this rejection is retried: it occurs before the command dispatches.
    if (reply.error?.code === "command_busy" && Date.now() < deadline) { await delay(25); continue; }
    assert.equal(reply.ok, true, `${op}: ${JSON.stringify(reply).slice(0, 4000)}`);
    return reply;
  }
}

async function open(config, query) {
  const url = new URL("stress.html", base);
  url.searchParams.set("engine_dev", "1");
  url.searchParams.set("config", JSON.stringify(config));
  for (const [key, value] of Object.entries(query)) url.searchParams.set(key, value);
  await page.goto(url.href, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__engineDev?.version === 1, null, { timeout: 60_000 });
  await command("pause");
  await command("render");
  await command("step", { ticks: 3 });
  return url.href;
}

async function screenshot(name) {
  // Hide the frame-rate and label overlays, which change while frames run.
  const bytes = await page.locator("canvas").screenshot({
    path: `${output}/${name}.png`, style: ".viewport > :not(canvas) { visibility: hidden !important; }",
  });
  return bytes.toString("base64");
}

async function capture(name) {
  await command("capture");
  return screenshot(name);
}

// Ordinary frames for `ms`, screenshots taken while they run.
async function run(name, ms) {
  await command("resume");
  await delay(ms);
  const first = await screenshot(`${name}-a`);
  await delay(300);
  const second = await screenshot(`${name}-b`);
  await command("pause");
  return [first, second];
}

// Per image pair: pixels whose largest channel difference exceeds each
// tolerance, the mean absolute channel difference, the same between averages
// of 8x8 blocks (content without its sampling), and each image's total
// variation (summed absolute neighbor differences per pixel).
async function compare(images, tolerances = [0, 8, 24]) {
  return page.evaluate(async ({ images, tolerances }) => {
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
    if (a.width !== b.width || a.height !== b.height) throw new Error("Capture dimensions changed.");
    const changed = tolerances.map(() => 0);
    let sum = 0;
    for (let offset = 0; offset < a.data.length; offset += 4) {
      let largest = 0;
      for (const channel of [0, 1, 2]) {
        const difference = Math.abs(a.data[offset + channel] - b.data[offset + channel]);
        sum += difference;
        largest = Math.max(largest, difference);
      }
      tolerances.forEach((tolerance, index) => { if (largest > tolerance) changed[index]++; });
    }
    const variation = ({ data, width, height }) => {
      let total = 0;
      for (let y = 0; y + 1 < height; y++) {
        for (let x = 0; x + 1 < width; x++) {
          const at = (y * width + x) * 4;
          for (const channel of [0, 1, 2]) {
            total += Math.abs(data[at + channel] - data[at + 4 + channel]) + Math.abs(data[at + channel] - data[at + width * 4 + channel]);
          }
        }
      }
      return total / (width * height);
    };
    const block = 8;
    let blocks = 0, blockSum = 0;
    for (let by = 0; by + block <= a.height; by += block) {
      for (let bx = 0; bx + block <= a.width; bx += block) {
        for (const channel of [0, 1, 2]) {
          let first = 0, second = 0;
          for (let y = by; y < by + block; y++) {
            for (let x = bx; x < bx + block; x++) {
              const at = (y * a.width + x) * 4 + channel;
              first += a.data[at]; second += b.data[at];
            }
          }
          blockSum += Math.abs(first - second) / (block * block);
        }
        blocks++;
      }
    }
    const distinct = new Set();
    for (let offset = 0; offset < a.data.length; offset += 4) distinct.add(a.data[offset] << 16 | a.data[offset + 1] << 8 | a.data[offset + 2]);
    return {
      total: a.width * a.height, changed: Object.fromEntries(tolerances.map((tolerance, index) => [tolerance, changed[index]])),
      mean_difference: sum / (a.width * a.height * 3), block_difference: blockSum / (blocks * 3),
      variation: [variation(a), variation(b)], distinct_colors: distinct.size,
    };
  }, { images, tolerances });
}

try {
  await mkdir(output, { recursive: true });
  const results = {};

  // Exact captures do not depend on the setting.
  results.exact_url = await open(still, {});
  const exactOff = await capture("exact-taa-off");

  results.taa_url = await open(still, { taa: "on" });
  const exact = await capture("exact-taa-on");
  const exactPair = await compare([exactOff, exact]);
  assert.equal(exactPair.changed[0], 0, `captures must not depend on the setting: ${JSON.stringify(exactPair)}`);
  assert.ok(exactPair.distinct_colors > 16, "the view must draw textured geometry");

  // Converged ordinary frames: smoother edges, the same content at a coarser
  // scale, stable over time. Thin, distant towers alias across much of the
  // view, so many pixels change.
  const [first, second] = await run("taa-still", 1500);
  const smoothed = await compare([exact, first]);
  results.smoothed = smoothed;
  assert.ok(smoothed.changed[8] > smoothed.total * 0.001, `antialiasing must change edges: ${JSON.stringify(smoothed)}`);
  assert.ok(smoothed.block_difference < 3, `antialiasing must preserve content: ${JSON.stringify(smoothed)}`);
  assert.ok(smoothed.variation[1] < smoothed.variation[0] * 0.95, `antialiasing must smooth edges: ${JSON.stringify(smoothed)}`);
  const stable = await compare([first, second]);
  results.stable = stable;
  assert.ok(stable.changed[24] < stable.total * 0.005, `a still view must not flicker: ${JSON.stringify(stable)}`);

  // Captures after temporal frames are still exact.
  const after = await capture("exact-after-taa");
  assert.equal((await compare([exact, after])).changed[0], 0, "captures must stay exact after temporal frames");

  // CPU-presented ordinary frames are exact with the setting on.
  results.cpu_url = await open(still, { taa: "on", presentation: "cpu" });
  const [cpu] = await run("cpu-still", 800);
  const cpuPair = await compare([exact, cpu]);
  results.cpu = cpuPair;
  assert.ok(cpuPair.changed[8] <= cpuPair.total * 0.001, `CPU presentation must stay exact: ${JSON.stringify(cpuPair)}`);

  // Moving objects: ordinary frames keep drawing the scene.
  results.moving_url = await open(moving, { taa: "on" });
  const movingExact = await capture("moving-exact");
  const [movingFirst] = await run("taa-moving", 1500);
  const movingPair = await compare([movingExact, movingFirst]);
  results.moving = movingPair;
  assert.ok(movingPair.distinct_colors > 16, "the moving view must draw textured geometry");
  // Reprojection follows each object: motion blurs little beyond the still view.
  assert.ok(movingPair.variation[1] > smoothed.variation[1] * 0.8, `moving frames must keep their detail: ${JSON.stringify(movingPair)}`);

  assert.deepEqual(errors, []);
  await writeFile(`${output}/results.json`, JSON.stringify(results, null, 2));
  console.log(JSON.stringify({
    ok: true, artifact: output, ...results,
    checks: ["captures_independent_of_setting", "edges_smoothed", "content_preserved", "still_view_stable", "captures_exact_after_temporal_frames", "cpu_presentation_exact", "moving_frames_draw", "clean_browser_diagnostics"],
  }));
} finally {
  await browser.close();
}
