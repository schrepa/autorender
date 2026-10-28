// @ts-check
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { artifacts, commander, gridScene, launch, openPaused, routedScene, sceneUrl, waitFrames } from "./harness.mjs";

const still = gridScene();
const moving = gridScene({ moving: true });
const output = await artifacts("taa-smoke");
const { browser, page, errors } = await launch();
const command = commander(page);

async function open(scene, query) {
  await page.unrouteAll();
  await routedScene(page, scene).route();
  const url = sceneUrl(query);
  await openPaused(page, url);
  await command("render");
  await command("step", { ticks: 3 });
  return url.href;
}

async function screenshot(name) {
  const bytes = await page.locator("canvas").screenshot({
    path: `${output}/${name}.png`, style: ".engine-hud { visibility: hidden !important; }",
  });
  return bytes.toString("base64");
}

async function capture(name) {
  await command("capture");
  return screenshot(name);
}

async function run(name, frames) {
  await command("resume");
  await waitFrames(command, frames);
  const first = await screenshot(`${name}-a`);
  await waitFrames(command, 18);
  const second = await screenshot(`${name}-b`);
  await command("pause");
  return [first, second];
}

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

  results.exact_url = await open(still, {});
  const inspect = async () => (await command("inspect", { component: "TemporalAntialiasing" })).result.value;
  const defaults = { nearest_motion: true, rounded_clamp: false, particle_response: 2 };
  assert.deepEqual(await inspect(), { enabled: false, ...defaults }, "temporal antialiasing is off by default");
  const exactOff = await capture("exact-taa-off");

  results.taa_url = await open(still, { taa: "on" });
  const exact = await capture("exact-taa-on");
  const exactPair = await compare([exactOff, exact]);
  assert.equal(exactPair.changed[0], 0, `captures must not depend on the setting: ${JSON.stringify(exactPair)}`);
  assert.ok(exactPair.distinct_colors > 16, "the view must draw textured geometry");

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

  const observed = async () => (await command("status")).result.rendering;
  assert.deepEqual(await inspect(), { enabled: true, ...defaults });
  const temporalFrame = await observed();
  assert.equal(temporalFrame.temporal_antialiasing.active, true, "ordinary frames resolve temporally");
  assert.ok(temporalFrame.render_passes.includes("temporal_resolve"));
  const off = await command("environment", { temporal_antialiasing: { enabled: false } });
  assert.deepEqual(off.result.environment_changed, ["temporal_antialiasing"]);
  assert.deepEqual(await inspect(), { enabled: false, ...defaults });
  await run("taa-off", 30);
  const plainFrame = await observed();
  assert.equal(plainFrame.temporal_antialiasing.active, false, "turned off, ordinary frames do not resolve temporally");
  assert.ok(!plainFrame.render_passes.includes("temporal_resolve"));
  const invalid = await command("environment", { temporal_antialiasing: { enabled: true, samples: 8 } }, false);
  assert.equal(invalid.error.code, "unknown_field", "unknown temporal antialiasing fields are rejected");
  await command("environment", { temporal_antialiasing: { enabled: true } });
  const [again] = await run("taa-again", 30);
  assert.equal((await observed()).temporal_antialiasing.active, true, "turned on again, ordinary frames resolve temporally");

  const inspectDither = async () => (await command("inspect", { component: "Dither" })).result.value;
  assert.deepEqual(await inspectDither(), { enabled: true }, "dithering is on by default");
  const undithered = await command("environment", { dither: { enabled: false } });
  assert.deepEqual(undithered.result.environment_changed, ["dither"]);
  assert.deepEqual(await inspectDither(), { enabled: false });
  const [unditheredFirst] = await run("taa-undithered", 60);
  await command("environment", { dither: { enabled: true } });
  const [ditherFirst, ditherSecond] = await run("taa-dither", 60);
  results.dither = await compare([unditheredFirst, ditherFirst]);
  results.dither_stable = await compare([ditherFirst, ditherSecond]);
  measurements.push({
    name: "dither difference", value: results.dither.mean_difference, unit: "levels",
    note: "mean channel difference between converged still frames without and with dithered 8-bit writes",
  }, {
    name: "dithered still view flicker", value: results.dither_stable.changed[24] / results.dither_stable.total * 100, unit: "%",
    note: "pixels whose largest channel differs by more than 24 levels between two converged still frames with dithering, 300 ms apart",
  });

  const bare = { nearest_motion: false, rounded_clamp: false, particle_response: 0 };
  await command("environment", { temporal_antialiasing: { enabled: true, ...bare } });
  assert.deepEqual(await inspect(), { enabled: true, ...bare });
  const [bareFirst] = await run("taa-bare", 60);
  const sturdy = { nearest_motion: true, rounded_clamp: true, particle_response: 2 };
  const sturdied = await command("environment", { temporal_antialiasing: { enabled: true, ...sturdy } });
  assert.deepEqual(sturdied.result.environment_changed, ["temporal_antialiasing"]);
  assert.deepEqual(await inspect(), { enabled: true, ...sturdy });
  const [sturdyFirst, sturdySecond] = await run("taa-sturdy", 60);
  assert.equal((await observed()).temporal_antialiasing.active, true, "the sturdier resolve runs");
  results.sturdy = await compare([bareFirst, sturdyFirst]);
  results.sturdy_stable = await compare([sturdyFirst, sturdySecond]);
  measurements.push({
    name: "sturdier resolve difference", value: results.sturdy.mean_difference, unit: "levels",
    note: "mean channel difference between converged still frames of the bare and the sturdiest resolve",
  }, {
    name: "sturdier still view flicker", value: results.sturdy_stable.changed[24] / results.sturdy_stable.total * 100, unit: "%",
    note: "pixels whose largest channel differs by more than 24 levels between two converged still frames of the sturdier resolve, 300 ms apart",
  });
  const beyond = await command("environment", { temporal_antialiasing: { enabled: true, particle_response: 9 } }, false);
  assert.equal(beyond.error.code, "invalid_environment", "a particle response beyond its range is rejected");
  await command("environment", { temporal_antialiasing: { enabled: true } });
  assert.deepEqual(await inspect(), { enabled: true, ...defaults });

  const inspectFoliage = async () => (await command("inspect", { component: "Foliage" })).result.value;
  assert.deepEqual(await inspectFoliage(), { transmission: 1, variation: 1, hashed_cutoff: false, ground_blend: Math.fround(0.4), crown_depth: 2 }, "foliage's defaults");
  const leaves = await command("environment", { foliage: { hashed_cutoff: true } });
  assert.deepEqual(leaves.result.environment_changed, ["foliage"]);
  assert.equal((await inspectFoliage()).hashed_cutoff, true);
  const [leavesFirst] = await run("taa-foliage", 60);
  results.foliage = await compare([again, leavesFirst]);
  measurements.push({
    name: "foliage difference", value: results.foliage.mean_difference, unit: "levels",
    note: "mean channel difference between converged still frames at foliage's defaults and with hashed cutoffs added",
  });
  await command("environment", { foliage: {} });

  const after = await capture("exact-after-taa");
  assert.equal((await compare([exact, after])).changed[0], 0, "captures must stay exact after temporal frames");

  results.cpu_url = await open(still, { taa: "on", presentation: "cpu" });
  const [cpu] = await run("cpu-still", 48);
  const cpuPair = await compare([exact, cpu]);
  results.cpu = cpuPair;
  measurements.push({
    name: "CPU presentation difference", value: cpuPair.changed[8] / cpuPair.total * 100, unit: "%",
    note: "pixels whose largest channel differs by more than 8 levels between a CPU-presented ordinary frame with antialiasing on and the exact capture",
  });

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
    checks: ["captures_independent_of_setting", "settings_inspected", "settings_controlled", "settings_observed", "settings_validated", "dither_controlled", "sturdier_resolve_controlled", "captures_exact_after_temporal_frames", "moving_frames_draw", "clean_browser_diagnostics"],
    measurements,
  }));
} finally {
  await browser.close();
}
