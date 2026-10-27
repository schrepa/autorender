/**
 * The terrain fixture's surface draws its materials, cover and tracks.
 *
 * The GPU's surface materials match the CPU's; ground cover grows
 * near the eye; the wind and occupancy sections read back, refuse values
 * out of range and change what draws; tracks press into the ground as deep
 * as its materials let them and refill; the shadow clipmap draws every tile
 * the pixels read, sharper with finer texels. The fixture:
 * terrain-fixture.mjs.
 */
// @ts-check
import assert from "node:assert/strict";
import { differentPixels } from "./harness.mjs";
import { COMPILE_MS, fast, openTerrain, overview } from "./terrain-fixture.mjs";

const { page, errors, command, settled, output, close, settle, capture, heights } = await openTerrain("terrain-surface");
const checks = [];

try {
  // The GPU gives tiles their surface materials, the rules'
  // own but where its rounding crosses a threshold: the view drawn with
  // the CPU's differs in few pixels. Back on the GPU, tiles loading at a
  // new place get theirs from it as they upload.
  const gpuView = await capture("gpu-materials");
  await command("environment", { terrain: { ...fast, gpu_materials: false } });
  await settle();
  const cpuView = await capture("cpu-materials");
  const materials = await differentPixels(page, [gpuView, cpuView], { tolerance: 8 });
  assert.ok(materials.changed < materials.total * 0.01, `GPU materials differ from the CPU's in ${materials.changed} of ${materials.total} pixels`);
  await command("environment", { terrain: fast });
  await settle();
  await command("camera", { ...overview, position: [380, 90, -420], target: [300, 30, -300] });
  let controls = 0;
  for (let frame = 0; frame < 300; frame++) {
    const rendering = (await command("render")).result.rendering;
    controls += rendering.terrain.gpu_controls;
    if (controls > 0 && rendering.terrain.loading_tiles === 0 && rendering.terrain.fallback_tiles === 0) break;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  assert.ok(controls > 0, "the GPU gives new tiles their control values");
  await command("camera", overview);
  await settle();
  checks.push({ check: "gpu materials", differing: materials.changed, controls });

  // Ground cover: the meadow's grass grows near the eye,
  // counted a few frames late; turned off, none grows.
  await command("camera", { position: [-12, 32, -8], target: [10, 30, 6], vertical_fov_radians: 1, near: 0.1, far: 5000 });
  let grown;
  for (let frame = 0; frame < 60; frame++) {
    grown = (await command("render")).result.rendering;
    if (grown.ground_cover.blades > 0) break;
    // The tiles around the eye arrive from workers.
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  assert.ok(grown.ground_cover.blades > 0, `grass grows: ${JSON.stringify(grown)}`);
  assert.equal(grown.ground_cover.dropped, 0);
  const cover = (await command("inspect", { component: "GroundCover" })).result.value;
  await command("environment", { ground_cover: { ...cover, enabled: false } });
  let bare;
  for (let frame = 0; frame < 8; frame++) bare = (await command("render")).result.rendering;
  assert.equal(bare.ground_cover.blades, 0, "no blades once cover stops");
  await command("environment", { ground_cover: cover });
  checks.push({ check: "ground cover", blades: grown.ground_cover.blades });
  // The wind blades and swaying crates bend in: its section
  // reads back as set, and refuses speeds out of range.
  const calm = (await command("inspect", { component: "Wind" })).result.value;
  await command("environment", { wind: { velocity: [8, -2], gusts: 0.25 } });
  assert.deepEqual((await command("inspect", { component: "Wind" })).result.value, { velocity: [8, -2], gusts: 0.25 });
  const gale = await command("environment", { wind: { velocity: [80, 0], gusts: 0.25 } }, false);
  assert.equal(gale.error.code, "invalid_environment", "winds past 40 m/s are refused");
  assert.equal(gale.error.path, "$.wind.velocity");
  await command("environment", { wind: calm });
  checks.push({ check: "wind", calm });
  // Occupancy: the crates' covers are drawn around the camera.
  assert.ok(settled.occupancy.redraws > 0, `the occupancy map draws: ${settled.occupancy.redraws}`);
  // Occupancy reads back as set, and refuses a resolution out of range.
  const occupancy = (await command("inspect", { component: "Occupancy" })).result.value;
  await command("environment", { occupancy: { ...occupancy, strength: 0.5 } });
  assert.equal((await command("inspect", { component: "Occupancy" })).result.value.strength, 0.5);
  const coarse = await command("environment", { occupancy: { ...occupancy, resolution: 16 } }, false);
  assert.equal(coarse.error.path, "$.occupancy.resolution");
  await command("environment", { occupancy });
  checks.push({ check: "occupancy", redraws: settled.occupancy.redraws });
  // Tracks press into the ground as deep as its materials
  // let them, from the deform control and from deformers each tick: the
  // log answers queries and presses them again where the map returns, they
  // refill over the refill time, the budget drops the oldest, and the map
  // shows them.
  {
    const spot = [-12, -6];
    const settings = (await command("inspect", { component: "Deformation" })).result.value;
    assert.equal(settings.enabled, true, "deformation is on by default");
    const [ground] = await heights([spot]);
    const close = { position: [spot[0] + 3, ground.height + 2.5, spot[1] + 3], target: [spot[0] - 1, ground.height, spot[1] - 1], vertical_fov_radians: 0.9, near: 0.05, far: 2000 };
    await command("camera", close);
    await settle();
    const flat = await capture("tracks-before");
    const rut = { from: [spot[0] - 3, spot[1] + 1], to: [spot[0] + 2, spot[1] - 1.5], radius: 0.3, depth: 0.2, flatness: 0.3 };
    const pressed = (await command("deform", { ...rut, clear: true })).result;
    assert.ok(pressed.enabled && pressed.stamps === 1, `a track presses: ${JSON.stringify(pressed)}`);
    const query = async (points, seconds) => (await command("deformation_query", seconds === undefined ? { points } : { points, seconds })).result.samples;
    const middle = [(rut.from[0] + rut.to[0]) / 2, (rut.from[1] + rut.to[1]) / 2];
    const [inside, outside] = await query([middle, [spot[0] + 20, spot[1]]]);
    assert.ok(inside.track > 0.15 && inside.deform > 0 && Math.abs(inside.depth - Math.min(inside.track, inside.deform)) < 1e-3,
      `the log holds the track as deep as the meadow lets it: ${JSON.stringify(inside)}`);
    assert.equal(outside.track, 0);
    const tracked = await capture("tracks");
    const trackPixels = await differentPixels(page, [flat, tracked]);
    assert.ok(trackPixels.changed > 500, `the track shows: ${trackPixels.changed} pixels`);
    // Far beyond the map and back: the log presses the track again.
    await command("camera", { ...close, position: [600, 200, 600], target: [500, 100, 500] });
    await command("camera", close);
    await settle();
    const returned = await capture("tracks-returned");
    const replay = await differentPixels(page, [tracked, returned]);
    assert.ok(replay.changed * 2 < trackPixels.changed, `the track persists: ${replay.changed} pixels differ`);
    // The roller presses where it stands each tick, and inspects as authored.
    const roller = (await command("inspect", { entity: "roller", component: "Deformer" })).result.entity.components.Deformer;
    const authored = { radius: 0.5, depth: 0.2, contact: 0.5, flatness: 0.2 };
    assert.deepEqual(roller, Object.fromEntries(Object.entries(authored).map(([key, value]) => [key, Math.fround(value)])), `the roller inspects as authored: ${JSON.stringify(roller)}`);
    const counted = (await command("inspect", { component: "Tracks" })).result.value;
    await command("step", { ticks: 3 });
    const rolled = (await command("inspect", { component: "Tracks" })).result.value;
    assert.ok(rolled.pressed >= counted.pressed + 3, `the roller presses each tick: ${JSON.stringify([counted, rolled])}`);
    const [dimple] = await query([[10, 10]]);
    assert.ok(dimple.track > 0.1, `the roller's dimple: ${JSON.stringify(dimple)}`);
    // Two refill times after it was pressed, an e-squared of it is left.
    await command("environment", { deformation: { ...settings, refill: 0.5 } });
    const [later] = await query([middle], pressed.track.seconds + 1);
    const left = inside.track * Math.exp(-2);
    assert.ok(Math.abs(later.track - left) < 0.005, `the track refills: ${later.track} against ${left}`);
    // The budget drops the oldest stamps.
    await command("environment", { deformation: { ...settings, max_stamps: 1 } });
    await command("deform", { from: [spot[0] + 5, spot[1]], radius: 0.2, depth: 0.1 });
    const budgeted = (await command("inspect", { component: "Tracks" })).result.value;
    assert.ok(budgeted.stamps === 1 && budgeted.dropped > 0, `the budget drops the oldest: ${JSON.stringify(budgeted)}`);
    const coarse = await command("environment", { deformation: { ...settings, resolution: 100 } }, false);
    assert.equal(coarse.error.path, "$.deformation.resolution");
    const deep = await command("deform", { from: spot, radius: 0.2, depth: 20 }, false);
    assert.equal(deep.error.path, "$.depth");
    await command("environment", { deformation: settings });
    await command("deform", { from: spot, radius: 0.01, depth: 0, clear: true });
    await command("camera", overview);
    checks.push({ check: "deformation", track: inside.track, depth: inside.depth, pixels: trackPixels.changed, replayed: replay.changed, dimple: dimple.track, refilled: later.track });
  }
  // The shadow clipmap draws every tile the pixels read on GPU-presented
  // frames once they settle; coarser finest texels blur what its finest
  // levels held. The section refuses more levels than the engine holds.
  const clipmap = (await command("inspect", { component: "Shadows" })).result.value;
  const gpuCapture = async (name, shadows) => {
    await command("environment", { shadows });
    let rendering;
    // GPU-presented frames need pipelines of their own, which compile for
    // seconds on some GPUs while the browser presents nothing: the capture
    // waits for them as for the tiles.
    const until = Date.now() + COMPILE_MS;
    for (let frame = 0; ; frame++) {
      rendering = (await command("present", { alpha: 1, gpu: true })).result.rendering;
      const drawn = rendering.shadows.tiles_marked > 0 && rendering.shadows.tiles_missing === 0;
      if (frame > 8 && drawn && !rendering.pipelines_compiling) break;
      if (Date.now() > until) assert.fail(`shadow tiles or pipelines still pending after ${COMPILE_MS / 1000} s: ${JSON.stringify(rendering.shadows)}`);
      if (rendering.pipelines_compiling) await new Promise(resolve => setTimeout(resolve, 50));
    }
    await command("capture", { alpha: 1, gpu: true });
    const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
    return { rendering, image: bytes.toString("base64") };
  };
  const sharp = await gpuCapture("shadow-tiles", clipmap);
  assert.equal(sharp.rendering.shadows.tiles_missing, 0, `shadow tiles draw: ${JSON.stringify(sharp.rendering.shadows)}`);
  const blurred = await gpuCapture("coarse-shadow-tiles", { ...clipmap, finest_texel: 0.5 });
  const texelPixels = await differentPixels(page, [sharp.image, blurred.image]);
  assert.ok(texelPixels.changed > 100, `finer texels change ${texelPixels.changed} pixels`);
  const deep = await command("environment", { shadows: { ...clipmap, levels: 64 } }, false);
  assert.equal(deep.error.path, "$.shadows.levels");
  await command("environment", { shadows: clipmap });
  checks.push({ check: "shadow clipmap", tiles: sharp.rendering.shadows.tiles_resident, pixels: texelPixels.changed });
  await command("camera", overview);

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await close();
}
