// @ts-check
import assert from "node:assert/strict";
import { differentPixels } from "./harness.mjs";
import { COMPILE_MS, fast, openTerrain, overview } from "./terrain-fixture.mjs";

const { page, errors, command, settled, output, close, settle, capture } = await openTerrain("terrain-surface");
const checks = [];

try {
  const gpuView = await capture("gpu-materials");
  await command("control", "environment", { terrain: { ...fast, gpu_materials: false } });
  await settle();
  const cpuView = await capture("cpu-materials");
  const materials = await differentPixels(page, [gpuView, cpuView], { tolerance: 8 });
  assert.ok(materials.changed < materials.total * 0.01, `GPU materials differ from the CPU's in ${materials.changed} of ${materials.total} pixels`);
  await command("control", "environment", { terrain: fast });
  await settle();
  await command("control", "camera", { ...overview, position: [380, 90, -420], target: [300, 30, -300] });
  let controls = 0;
  for (let frame = 0; frame < 300; frame++) {
    const rendering = (await command("control", "render")).result.rendering;
    controls += rendering.terrain.gpu_controls;
    if (controls > 0 && rendering.terrain.loading_tiles === 0 && rendering.terrain.fallback_tiles === 0) break;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  assert.ok(controls > 0, "the GPU gives new tiles their control values");
  await command("control", "camera", overview);
  await settle();
  checks.push({ check: "gpu materials", differing: materials.changed, controls });

  await command("control", "camera", { position: [-12, 32, -8], target: [10, 30, 6], vertical_fov_radians: 1, near: 0.1, far: 5000 });
  let grown;
  for (let frame = 0; frame < 60; frame++) {
    grown = (await command("control", "render")).result.rendering;
    if (grown.ground_cover.blades > 0) break;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  assert.ok(grown.ground_cover.blades > 0, `grass grows: ${JSON.stringify(grown)}`);
  assert.equal(grown.ground_cover.dropped, 0);
  const cover = (await command("inspect", { component: "GroundCover" })).result.value;
  await command("control", "environment", { ground_cover: { ...cover, enabled: false } });
  let bare;
  for (let frame = 0; frame < 8; frame++) bare = (await command("control", "render")).result.rendering;
  assert.equal(bare.ground_cover.blades, 0, "no blades once cover stops");
  await command("control", "environment", { ground_cover: cover });
  checks.push({ check: "ground cover", blades: grown.ground_cover.blades });
  const calm = (await command("inspect", { component: "Wind" })).result.value;
  await command("control", "environment", { wind: { velocity: [8, -2], gusts: 0.25 } });
  assert.deepEqual((await command("inspect", { component: "Wind" })).result.value, { velocity: [8, -2], gusts: 0.25 });
  const gale = await command("control", "environment", { wind: { velocity: [80, 0], gusts: 0.25 } }, false);
  assert.equal(gale.error.code, "invalid_environment", "winds past 40 m/s are refused");
  assert.equal(gale.error.path, "$.args.wind.velocity");
  await command("control", "environment", { wind: calm });
  checks.push({ check: "wind", calm });
  assert.ok(settled.occupancy.redraws > 0, `the occupancy map draws: ${settled.occupancy.redraws}`);
  const occupancy = (await command("inspect", { component: "Occupancy" })).result.value;
  await command("control", "environment", { occupancy: { ...occupancy, strength: 0.5 } });
  assert.equal((await command("inspect", { component: "Occupancy" })).result.value.strength, 0.5);
  const coarse = await command("control", "environment", { occupancy: { ...occupancy, resolution: 16 } }, false);
  assert.equal(coarse.error.path, "$.args.occupancy.resolution");
  await command("control", "environment", { occupancy });
  checks.push({ check: "occupancy", redraws: settled.occupancy.redraws });
  const clipmap = (await command("inspect", { component: "Shadows" })).result.value;
  const gpuCapture = async (name, shadows) => {
    await command("control", "environment", { shadows });
    let rendering;
    const until = Date.now() + COMPILE_MS;
    for (let frame = 0; ; frame++) {
      rendering = (await command("control", "present", { alpha: 1 })).result.rendering;
      const drawn = rendering.shadows.tiles_marked > 0 && rendering.shadows.tiles_missing === 0;
      if (frame > 8 && drawn && !rendering.pipelines_compiling) break;
      if (Date.now() > until) assert.fail(`shadow tiles or pipelines still pending after ${COMPILE_MS / 1000} s: ${JSON.stringify(rendering.shadows)}`);
      if (rendering.pipelines_compiling) await new Promise(resolve => setTimeout(resolve, 50));
    }
    await command("capture", { alpha: 1 });
    const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
    return { rendering, image: bytes.toString("base64") };
  };
  const sharp = await gpuCapture("shadow-tiles", clipmap);
  assert.equal(sharp.rendering.shadows.tiles_missing, 0, `shadow tiles draw: ${JSON.stringify(sharp.rendering.shadows)}`);
  const blurred = await gpuCapture("coarse-shadow-tiles", { ...clipmap, finest_texel: 0.5 });
  const texelPixels = await differentPixels(page, [sharp.image, blurred.image]);
  assert.ok(texelPixels.changed > 100, `finer texels change ${texelPixels.changed} pixels`);
  const deep = await command("control", "environment", { shadows: { ...clipmap, levels: 64 } }, false);
  assert.equal(deep.error.path, "$.args.shadows.levels");
  await command("control", "environment", { shadows: clipmap });
  checks.push({ check: "shadow clipmap", tiles: sharp.rendering.shadows.tiles_resident, pixels: texelPixels.changed });
  await command("control", "camera", overview);

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await close();
}
