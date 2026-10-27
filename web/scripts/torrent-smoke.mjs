/**
 * The water's debug views over the Highlands' streams, in real Chrome: the
 * scene's torrent, rapids and meadow-river views, each shaded and in every
 * view of the water's state (depth, speed, Froude number, turbulence, white
 * water, reach, carried foam, span, the eddies' turbulent Froude number and
 * roughness). The torrent view holds water; a twentieth to three
 * fifths of it reads white in the white water view, where white water is
 * born at drops and rocks rather than spread as a sheet; its turbulence
 * varies; and the pixel probe reads a torrent pixel's numbers. The water's
 * views show everything but water alike, gray with the sky dark, so water
 * is where they differ.
 */
// @ts-check
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, launch, openPaused } from "./harness.mjs";

const url = devUrl("stress.html", { preset: "highlands" });
const output = await artifacts("torrent-smoke");
const { browser, page, errors } = await launch({ width: 960, height: 540, timeout: 600_000 });
const command = commander(page, { deadline: 600_000 });

/** The scene's named views of its streams. */
const streams = ["torrent", "rapids", "meadow_river"];
/** The water's views, each showing one of its numbers. */
const waterViews = ["water_depth", "water_speed", "water_froude", "water_turbulence", "water_white", "water_reach",
  "water_foam", "water_span", "water_eddies", "water_roughness"];
/** The reaches the probe names. */
const reaches = ["pool", "plane", "steps", "cascade", "falls"];
/** Frame statistics that are not work (bench.rs NOT_WORK): they change
 * from frame to frame however settled the view is. */
const NOT_WORK = new Set(["frame", "tick", "read_back_age", "simulation_ms", "extraction_ms", "render_ms"]);

/** The work counters of frame statistics `stats`: dotted paths to numbers
 * and booleans, lists as their lengths. */
function work(stats, prefix = "", out = {}) {
  for (const [key, value] of Object.entries(stats ?? {})) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (Array.isArray(value)) out[path] = value.length;
    else if (value && typeof value === "object") work(value, path, out);
    else if ((typeof value === "number" || typeof value === "boolean") && !NOT_WORK.has(path)) out[path] = value;
  }
  return out;
}

/** Renders exact frames until nothing loads or compiles and the frames'
 * work counters hold for three frames, then thirty more, so work spread
 * over frames finishes its rounds: what the view draws has settled. */
async function settle(label) {
  const until = Date.now() + 600_000;
  let previous, quiet = 0;
  while (quiet < 3) {
    const rendering = (await command("render")).result.rendering;
    const busy = rendering.loading || rendering.pipelines_compiling;
    const counters = JSON.stringify(work(rendering));
    quiet = !busy && counters === previous ? quiet + 1 : 0;
    previous = counters;
    assert.ok(Date.now() < until, `${label} settles: ${counters.slice(0, 2000)}`);
    if (busy) await delay(20);
  }
  for (let frame = 0; frame < 30; frame++) await command("render");
}

/** The settled view captured into NAME.png: the capture as base64 PNG and
 * its frame's stats. */
async function capture(name) {
  await settle(name);
  const reply = await command("capture");
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return { png: bytes.toString("base64"), rendering: reply.result.rendering };
}

/** What the water's views show of one view, from their captures `shots`
 * (base64 PNGs, in `waterViews`' order): the pixels holding water, those
 * whose colors differ between the views; of those, the ones reading white
 * in the white water view (bright in every channel, as white water shows)
 * and those the turbulence view shows otherwise than its most common
 * color; and the water pixel nearest the view's middle. */
function analyze(shots) {
  return page.evaluate(async ({ shots, white, turbulence }) => {
    const decoded = await Promise.all(shots.map(async png => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width; canvas.height = image.height;
      const context = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d"));
      context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, image.width, image.height);
    }));
    const { width, height } = decoded[0];
    if (decoded.some(image => image.width !== width || image.height !== height)) throw new Error("Capture dimensions changed.");
    const images = decoded.map(image => image.data);
    // Water shows in other colors from view to view; everything else
    // shows alike in all of them.
    const differs = (first, other, at) => [0, 1, 2].some(channel => Math.abs(first[at + channel] - other[at + channel]) > 2);
    const water = [];
    /** @type {number[] | null} */
    let nearest = null;
    let nearestDistance = Infinity;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const at = (y * width + x) * 4;
        if (!images.some(other => differs(images[0], other, at))) continue;
        water.push(at);
        const distance = (x - width / 2) ** 2 + (y - height / 2) ** 2;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = [x, y];
        }
      }
    }
    const whiteView = images[white];
    const whiteWater = water.filter(at => Math.min(whiteView[at], whiteView[at + 1], whiteView[at + 2]) >= 128).length;
    const turbulent = images[turbulence];
    const colors = new Map();
    for (const at of water) {
      const color = (turbulent[at] << 16) | (turbulent[at + 1] << 8) | turbulent[at + 2];
      colors.set(color, (colors.get(color) ?? 0) + 1);
    }
    let common = 0, commonCount = 0;
    for (const [color, count] of colors) {
      if (count > commonCount) {
        common = color;
        commonCount = count;
      }
    }
    const most = [common >> 16, (common >> 8) & 0xff, common & 0xff];
    const varied = water.filter(at => [0, 1, 2].some(channel => Math.abs(turbulent[at + channel] - most[channel]) > 8)).length;
    return { width, height, total: width * height, water: water.length, white: whiteWater, varied, nearest };
  }, { shots, white: waterViews.indexOf("water_white"), turbulence: waterViews.indexOf("water_turbulence") });
}

const report = {};
const measurements = [];
try {
  await openPaused(page, url, { timeout: 600_000 });
  for (const stream of streams) {
    // The scene names the view; the shaded image first, then each of the
    // water's views.
    await command("game", { name: "view", args: { name: stream } });
    await command("environment", { debug_view: { view: "off" } });
    const shaded = await capture(`${stream}-shaded`);
    const shots = [];
    for (const view of waterViews) {
      await command("environment", { debug_view: { view } });
      const shown = await capture(`${stream}-${view}`);
      assert.ok(shown.rendering.render_passes.includes("debug_view"), `${stream} ${view}: ${JSON.stringify(shown.rendering.render_passes)}`);
      shots.push(shown.png);
    }
    const seen = await analyze(shots);
    const shares = {
      water: seen.water / seen.total,
      white: seen.water > 0 ? seen.white / seen.water : 0,
      varied: seen.water > 0 ? seen.varied / seen.water : 0,
    };
    report[stream] = { ...seen, shares, water_leaves: shaded.rendering.water.leaves };
    measurements.push(
      { name: `${stream} water`, value: shares.water * 100, unit: "%", note: "pixels of the view the water's views show water at" },
      { name: `${stream} white water`, value: shares.white * 100, unit: "%", note: "water pixels bright in every channel of the white water view" },
      { name: `${stream} varied turbulence`, value: shares.varied * 100, unit: "%", note: "water pixels the turbulence view shows more than 8 levels off its most common color" },
    );
    if (stream !== "torrent") continue;

    // The torrent: water, white where it is born but not all over, and
    // turbulence that varies.
    assert.ok(seen.water > seen.total / 1000, `the torrent view holds water: ${JSON.stringify(seen)}`);
    assert.ok(shares.white >= 0.05 && shares.white <= 0.6,
      `between 5% and 60% of the torrent's water reads white: ${JSON.stringify(shares)}`);
    assert.ok(shares.varied >= 0.1, `the torrent's turbulence varies: ${JSON.stringify(shares)}`);

    // The probe reads the water nearest the view's middle as the water's
    // views draw it, wherever it lies.
    assert.ok(seen.nearest, "a water pixel to probe");
    await command("environment", { debug_view: { view: "water_depth" } });
    const { display_width: width, display_height: height } = shaded.rendering.resolution;
    const [x, y] = /** @type {number[]} */ (seen.nearest);
    const point = {
      x: Math.min(width - 1, Math.floor((x + 0.5) * width / seen.width)),
      y: Math.min(height - 1, Math.floor((y + 0.5) * height / seen.height)),
    };
    const probed = (await command("probe", point)).result;
    assert.equal(probed.kind, "water", JSON.stringify(probed));
    const water = probed.water;
    assert.ok(water && water.depth > 0 && Number.isFinite(water.depth), JSON.stringify(probed));
    assert.ok(Number.isFinite(water.speed) && water.speed >= 0, JSON.stringify(water));
    assert.ok(Number.isFinite(water.froude) && water.froude >= 0, JSON.stringify(water));
    assert.ok(water.turbulence >= 0 && water.turbulence <= 1 && water.white >= 0 && water.white <= 1, JSON.stringify(water));
    assert.ok(water.reach === null || reaches.includes(water.reach), JSON.stringify(water));
    assert.ok(water.foam >= 0 && water.foam <= 1 && water.roughness >= 0 && water.roughness <= 1, JSON.stringify(water));
    assert.ok(Number.isFinite(water.span) && water.span > 0, JSON.stringify(water));
    assert.ok(Number.isFinite(water.eddies) && water.eddies >= 0, JSON.stringify(water));
    assert.equal(probed.terrain, null, JSON.stringify(probed));
    report.probe = probed;
  }
  await command("environment", { debug_view: { view: "off" } });
  assert.deepEqual(errors, []);
} finally {
  await browser.close();
}

console.log(JSON.stringify({
  ok: true, url: url.href, artifact: output, ...report,
  checks: ["named_views", "water_views_draw", "torrent_holds_water", "white_water_not_a_sheet", "turbulence_varies",
    "water_probe", "clean_browser_diagnostics"],
  measurements,
}));
