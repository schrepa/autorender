/** Temporal antialiasing in real Chrome: captures stay exact, while how
 * much ordinary GPU-presented frames change and smooth edges, keep content,
 * flicker when still, keep detail when moving, and how far CPU-presented
 * frames stray from the capture are measured; its settings are inspected,
 * changed by the paused environment control and observed in ordinary
 * frames. */
// @ts-check
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { artifacts, commander, devUrl, launch, openPaused, waitFrames } from "./harness.mjs";

// Textured meshes of several shapes give many edges; nothing moves, so any
// difference between ordinary frames and captures is the antialiasing.
const still = {
  seed: 7, renderables: 6000, moving_percent: 0, hierarchy: true, interpolation: true, colliders: 0,
  sensor_percent: 0, dense_colliders: false, queries_per_tick: 0, mesh_variety: 8, textured: true, culling: true,
};
const moving = { ...still, moving_percent: 100, colliders: 200, sensor_percent: 10, queries_per_tick: 8 };
const output = await artifacts("taa-smoke");
const { browser, page, errors } = await launch();
const command = commander(page);

async function open(config, query) {
  const url = devUrl("stress.html", { config, ...query });
  await openPaused(page, url);
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

// Ordinary frames, screenshots taken while they run: the first once
// `frames` were drawn, the second 18 frames later.
async function run(name, frames) {
  await command("resume");
  await waitFrames(command, frames);
  const first = await screenshot(`${name}-a`);
  await waitFrames(command, 18);
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
      const context = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d"));
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
  const results = {};

  // Exact captures do not depend on the setting, which is off by default.
  results.exact_url = await open(still, {});
  const inspect = async () => (await command("inspect", { component: "TemporalAntialiasing" })).result.value;
  assert.deepEqual(await inspect(), { enabled: false }, "temporal antialiasing is off by default");
  const exactOff = await capture("exact-taa-off");

  results.taa_url = await open(still, { taa: "on" });
  const exact = await capture("exact-taa-on");
  const exactPair = await compare([exactOff, exact]);
  assert.equal(exactPair.changed[0], 0, `captures must not depend on the setting: ${JSON.stringify(exactPair)}`);
  assert.ok(exactPair.distinct_colors > 16, "the view must draw textured geometry");

  // Converged ordinary frames against the exact capture: how many edge
  // pixels change, how much the content at a coarser scale does, how much
  // smoother the image is, and how much a still view then flickers, all
  // measured, not judged. Thin, distant towers alias across much of the view.
  const [first, second] = await run("taa-still", 90);
  const smoothed = await compare([exact, first]);
  results.smoothed = smoothed;
  const stable = await compare([first, second]);
  results.stable = stable;
  const measurements = [{
    name: "edges changed", value: smoothed.changed[8] / smoothed.total * 100, unit: "%",
    note: "pixels whose largest channel differs by more than 8 levels between the exact capture and a converged still frame",
  }, {
    name: "content change", value: smoothed.block_difference, unit: "levels",
    note: "mean channel difference of 8x8 block averages between the exact capture and a converged still frame",
  }, {
    name: "edge smoothing", value: (1 - smoothed.variation[1] / smoothed.variation[0]) * 100, unit: "%",
    note: "how much less total variation (summed neighbor differences) a converged still frame has than the exact capture",
  }, {
    name: "still view flicker", value: stable.changed[24] / stable.total * 100, unit: "%",
    note: "pixels whose largest channel differs by more than 24 levels between two converged still frames 300 ms apart",
  }];

  // The page's ?taa=on set the world's settings; ordinary frames report what
  // they ran, and the paused environment control turns it off and on.
  const observed = async () => (await command("status")).result.rendering;
  assert.deepEqual(await inspect(), { enabled: true });
  const temporalFrame = await observed();
  assert.equal(temporalFrame.temporal_antialiasing.active, true, "ordinary frames resolve temporally");
  assert.ok(temporalFrame.render_passes.includes("temporal_resolve"));
  const off = await command("environment", { temporal_antialiasing: { enabled: false } });
  assert.deepEqual(off.result.environment_changed, ["temporal_antialiasing"]);
  assert.deepEqual(await inspect(), { enabled: false });
  await run("taa-off", 30);
  const plainFrame = await observed();
  assert.equal(plainFrame.temporal_antialiasing.active, false, "turned off, ordinary frames do not resolve temporally");
  assert.ok(!plainFrame.render_passes.includes("temporal_resolve"));
  const invalid = await command("environment", { temporal_antialiasing: { enabled: true, samples: 8 } }, false);
  assert.equal(invalid.error.code, "unknown_field", "unknown temporal antialiasing fields are rejected");
  await command("environment", { temporal_antialiasing: { enabled: true } });
  await run("taa-again", 30);
  assert.equal((await observed()).temporal_antialiasing.active, true, "turned on again, ordinary frames resolve temporally");

  // Captures after temporal frames are still exact.
  const after = await capture("exact-after-taa");
  assert.equal((await compare([exact, after])).changed[0], 0, "captures must stay exact after temporal frames");

  // CPU-presented ordinary frames with the setting on, against the capture.
  results.cpu_url = await open(still, { taa: "on", presentation: "cpu" });
  const [cpu] = await run("cpu-still", 48);
  const cpuPair = await compare([exact, cpu]);
  results.cpu = cpuPair;
  measurements.push({
    name: "CPU presentation difference", value: cpuPair.changed[8] / cpuPair.total * 100, unit: "%",
    note: "pixels whose largest channel differs by more than 8 levels between a CPU-presented ordinary frame with antialiasing on and the exact capture",
  });

  // Moving objects: ordinary frames keep drawing the scene. Reprojection
  // follows each object, so motion should blur little beyond the still view.
  results.moving_url = await open(moving, { taa: "on" });
  const movingExact = await capture("moving-exact");
  const [movingFirst] = await run("taa-moving", 90);
  const movingPair = await compare([movingExact, movingFirst]);
  results.moving = movingPair;
  assert.ok(movingPair.distinct_colors > 16, "the moving view must draw textured geometry");
  measurements.push({
    name: "moving detail", value: movingPair.variation[1] / smoothed.variation[1] * 100, unit: "%",
    note: "total variation of an ordinary frame of moving objects against a converged still frame's",
  });

  assert.deepEqual(errors, []);
  await writeFile(`${output}/results.json`, JSON.stringify(results, null, 2));
  console.log(JSON.stringify({
    ok: true, artifact: output, ...results,
    checks: ["captures_independent_of_setting", "settings_inspected", "settings_controlled", "settings_observed", "settings_validated", "captures_exact_after_temporal_frames", "moving_frames_draw", "clean_browser_diagnostics"],
    measurements,
  }));
} finally {
  await browser.close();
}
