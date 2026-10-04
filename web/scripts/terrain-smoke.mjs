/**
 * Terrain checks (decision 051): a version 9 playground fixture with a
 * generated terrain streams, draws, casts shadows, collides and answers
 * height queries; the terrain control and environment section change what
 * draws; an `ETR1` file baked from the same layers streams by byte range and
 * matches; a version 8 scene with terrain is refused.
 *
 * Bakes its file with `cargo run -p engine_assetc --release --bin
 * terrain_bake` (the cargo on PATH) into web/public/scenes/terrain-smoke/,
 * which it removes afterwards with the fixture's source.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { artifacts, commander, devUrl, differentPixels, launch, openPaused, routedScene } from "./harness.mjs";

const url = devUrl("playground.html");
const output = await artifacts("terrain-smoke");
const folder = fileURLToPath(new URL("../public/scenes/terrain-smoke/", import.meta.url));
const source = fileURLToPath(new URL("../public/scenes/terrain-smoke.scene.json", import.meta.url));
// Terrain-local plateau at (512, 512), a hole 48 m east of it, and an
// imported dome 200 m high centered at (764, 764).
const terrain = {
  origin: [-512, 0, -512], tiles: [4, 4], spacing: 1,
  layers: [
    { noise: { seed: 3, height: 140, wavelength: 600 } },
    { stamp: { center: [512, 512], radius: 40, falloff: 60, op: { flatten: 30 } } },
    { stamp: { center: [560, 512], radius: 4, op: "hole" } },
    { heightmap: { uri: "terrain-smoke/dome.r16", width: 65, height: 65, min: [700, 700], size: [128, 128], low: 0, high: 200, blend: "max" } },
  ],
};
/** A 65 by 65 dome of little-endian 16-bit samples, highest at its center. */
function dome() {
  const bytes = Buffer.alloc(65 * 65 * 2);
  for (let z = 0; z < 65; z++) {
    for (let x = 0; x < 65; x++) {
      const r2 = ((x - 32) ** 2 + (z - 32) ** 2) / 32 ** 2;
      bytes.writeUInt16LE(Math.round(65535 * Math.max(0, 1 - r2)), (z * 65 + x) * 2);
    }
  }
  return bytes;
}
const fixture = {
  version: 9, id: "terrain-fixture", revision: 1,
  camera: { position: [0, 80, 140], target: [0, 30, 0], fov_degrees: 60, near: 0.5, far: 4000 },
  materials: [{ id: "plain", color: [0.8, 0.3, 0.2, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [
    { id: "player", mesh: "box", material: "plain", tags: ["player"], collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] }, memberships: 2, filter: 1, sensor: false }, transform: { translation: [0, 31, 0] } },
    { id: "ground", terrain },
    // Probes capture terrain on GPU-presented frames (decision 043).
    { id: "probe", reflection_probe: { extent: [60, 40, 60] }, transform: { translation: [0, 45, 0] } },
  ],
};
const overview = { position: [0, 80, 140], target: [0, 30, 0], vertical_fov_radians: Math.PI / 3, near: 0.5, far: 4000 };
const { browser, page, errors } = await launch();
const command = commander(page);
const scene = routedScene(page, fixture);
const checks = [];

/** Renders until every wanted tile is resident; returns the observation. */
async function settle() {
  let rendering;
  for (let frame = 0; frame < 40; frame++) {
    rendering = (await command("render")).result.rendering ?? (await command("status")).result.rendering;
    if (rendering.terrain_tiles > 0 && rendering.terrain_loading_tiles === 0 && rendering.terrain_fallback_patches === 0) return rendering;
  }
  assert.fail(`terrain did not settle: ${JSON.stringify(rendering)}`);
}

async function capture(name) {
  await command("capture");
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return bytes.toString("base64");
}

async function heights(points) {
  return (await command("terrain_height", { points })).result.samples;
}

try {
  await mkdir(folder, { recursive: true });
  await writeFile(`${folder}dome.r16`, dome());
  await scene.route();
  await openPaused(page, url);
  // All tile work in one extraction, so renders settle quickly.
  const fast = await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  assert.deepEqual(fast.result.environment_changed, ["terrain"]);
  await command("camera", overview);
  const settled = await settle();
  assert.ok(settled.terrain_patches > 0 && settled.terrain_triangles > 0, "terrain draws");
  assert.ok(settled.terrain_shadow_patches > 0, "terrain casts shadows");
  assert.equal(settled.dropped_terrain_tiles, 0);
  checks.push({ check: "streams and draws", tiles: settled.terrain_tiles, patches: settled.terrain_patches, triangles: settled.terrain_triangles });

  // Discovery and inspection.
  const listing = await command("describe");
  for (const name of ["Terrain", "Terrains", "TerrainSettings"]) {
    assert.ok(listing.result.types.some(type => type.name === name), `describe lists ${name}`);
  }
  const look = await command("inspect", { component: "Environment" });
  assert.equal(look.result.value.terrain.samples_per_frame, 50_000_000);
  const inspected = await command("inspect", { entity: "ground", component: "Terrain" });
  const view = inspected.result.entity.components.Terrain;
  assert.equal(view.terrain.layers.length, 4);
  assert.equal(view.terrain.layers[3].heightmap.uri, "terrain-smoke/dome.r16");
  assert.deepEqual(view.extent, [1024, 1024]);
  assert.ok(view.resident_tiles_by_level.reduce((a, b) => a + b, 0) > 0);
  checks.push({ check: "inspection", levels: view.levels, resident: view.resident_tiles_by_level });

  // Heights: the plateau is flat at 30 m, the hole is a hole, outside is null.
  const [plateau, hole, outside] = await heights([[0, 0], [48, 0], [5000, 5000]]);
  assert.equal(plateau.terrain, "ground");
  assert.ok(Math.abs(plateau.height - 30) < 1e-3, `plateau height ${plateau.height}`);
  assert.ok(plateau.normal[1] > 0.999);
  assert.equal(plateau.level, 0);
  assert.equal(hole.hole, true);
  assert.equal(outside, null);
  checks.push({ check: "height queries", plateau: plateau.height });

  // The imported heightmap's peak, at one of its samples, is exact.
  const [peak] = await heights([[252, 252]]);
  assert.ok(Math.abs(peak.height - 200) < 1e-3, `heightmap peak ${peak.height}`);
  checks.push({ check: "imported heightmap", peak: peak.height });

  // A tile budget keeps residency within it and reports what it drops.
  const budget = await command("environment", { terrain: { samples_per_frame: 50_000_000, max_tiles: 3 } });
  assert.ok(budget.result.rendering.terrain_tiles <= 3, `${budget.result.rendering.terrain_tiles} tiles under a budget of 3`);
  const budgeted = (await command("render")).result.rendering;
  assert.ok(budgeted.terrain_tiles <= 3 && budgeted.dropped_terrain_tiles > 0, JSON.stringify(budgeted));
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  await settle();
  checks.push({ check: "tile budget", resident: budgeted.terrain_tiles, dropped: budgeted.dropped_terrain_tiles });

  // Collision: rays hit the terrain entity where queries say the ground is.
  const ray = await command("raycast", { origin: [10, 200, -7], direction: [0, -1, 0], max_distance: 400 });
  const [under] = await heights([[10, -7]]);
  assert.equal(ray.result.hit.entity.id, "ground");
  assert.ok(Math.abs(ray.result.hit.point[1] - under.height) < 2e-3, "ray and query agree");
  const through = await command("raycast", { origin: [48, 200, 0], direction: [0, -1, 0], max_distance: 400 });
  assert.equal(through.result.hit, null, "rays fall through holes");
  const sweep = (await command("terrain_sweep", { center: [0, 0], half_extent: 120, samples: 24 })).result;
  assert.ok(sweep.collision.compared > 100, `sweep compared ${sweep.collision.compared}`);
  assert.ok(sweep.collision.worst.max_difference < 5e-3, `collision differs by ${sweep.collision.worst.max_difference}`);
  assert.ok(sweep.layers.compared > 100);
  assert.ok(sweep.layers.worst.max_difference < 5e-3, `layers differ by ${sweep.layers.worst.max_difference}`);
  checks.push({ check: "render versus collision", compared: sweep.collision.compared, worst: sweep.collision.worst.max_difference, layers: sweep.layers.worst.max_difference });

  // The settings section: finer level of detail draws more triangles;
  // invalid values are refused; disabled terrain draws nothing.
  const fine = await command("environment", { terrain: { samples_per_frame: 50_000_000, pixel_error: 2 } });
  const finer = (await settle());
  assert.ok(finer.terrain_triangles > settled.terrain_triangles, "a smaller pixel error draws more triangles");
  assert.deepEqual(fine.result.environment_changed, ["terrain"]);
  const invalid = await command("environment", { terrain: { pixel_error: 0.5 } }, false);
  assert.equal(invalid.error.code, "invalid_environment");
  assert.equal(invalid.error.path, "$.terrain.pixel_error");
  const shaded = await capture("shaded");
  const off = await command("environment", { terrain: { enabled: false } });
  assert.equal(off.result.rendering.terrain_patches, 0);
  const empty = await capture("disabled");
  const shown = await differentPixels(page, [shaded, empty]);
  assert.ok(shown.changed > 10_000, `terrain covers ${shown.changed} pixels`);
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  await settle();
  checks.push({ check: "settings section", more: finer.terrain_triangles, terrainPixels: shown.changed });

  // The terrain control: development views change pixels, freezing keeps
  // the patches while the camera moves away.
  const baseline = await capture("view-shaded");
  const levels = await command("terrain", { view: "levels" });
  assert.equal(levels.result.view, "levels");
  const levelView = await capture("view-levels");
  assert.ok((await differentPixels(page, [baseline, levelView])).changed > 10_000, "the levels view differs");
  await command("terrain", { view: "tiles" });
  await capture("view-tiles");
  await command("terrain", { view: "shaded" });
  const frozen = await command("terrain", { freeze: true });
  assert.ok(Array.isArray(frozen.result.frozen_at));
  await command("render");
  const selected = async () => (await command("inspect", { component: "Terrains" })).result.value.patches;
  const before = await selected();
  await command("camera", { ...overview, position: [0, 900, 1400] });
  const afar = (await command("render")).result.rendering;
  assert.equal(await selected(), before, "a frozen eye keeps its patches");
  await capture("frozen-from-afar");
  await command("terrain", { freeze: false });
  await command("render");
  const moved = await selected();
  assert.notEqual(moved, before, "unfrozen, patches follow the camera");
  await command("camera", overview);
  await settle();
  checks.push({ check: "terrain control", frozenPatches: before, visibleFromAfar: afar.terrain_patches, unfrozenPatches: moved });

  // An ETR1 file of the same layers streams by byte range and matches; the
  // bake reads the heightmap beside the scene, as the page does.
  await writeFile(source, JSON.stringify(fixture));
  execFileSync("cargo", ["run", "-q", "-p", "engine_assetc", "--release", "--bin", "terrain_bake", "--", source, "ground", `${folder}ground.etr`], { stdio: "inherit" });
  const generated = await heights([[10, -7], [-200, 150], [300, -310]]);
  const filed = structuredClone(fixture);
  filed.entities[1].terrain = { origin: terrain.origin, file: "terrain-smoke/ground.etr" };
  await scene.reload(command, filed);
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  await command("camera", overview);
  let streamed;
  for (let attempt = 0; attempt < 60; attempt++) {
    streamed = (await command("render")).result.rendering;
    if (streamed.terrain_tiles > 0 && streamed.terrain_loading_tiles === 0 && streamed.terrain_fallback_patches === 0) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.equal(streamed.terrain_loading_tiles, 0, "file tiles arrive");
  const read = await heights([[10, -7], [-200, 150], [300, -310]]);
  for (const [index, sample] of read.entries()) {
    assert.ok(sample && generated[index], `point ${index} resident from both`);
    if (sample.level === generated[index].level) {
      assert.ok(Math.abs(sample.height - generated[index].height) < 2e-3, `file height ${sample.height} vs ${generated[index].height}`);
    }
  }
  checks.push({ check: "ETR1 file streaming", tiles: streamed.terrain_tiles });

  // Running frames capture the probe's faces with terrain in them.
  await command("resume");
  let probes;
  for (let attempt = 0; attempt < 50 && !(probes?.reflection_probes > 0); attempt++) {
    await new Promise(resolve => setTimeout(resolve, 200));
    probes = (await command("status")).result.rendering;
  }
  await command("pause");
  assert.ok(probes.reflection_probes > 0, `probe captured: ${JSON.stringify(probes)}`);
  checks.push({ check: "probe captures", probes: probes.reflection_probes });

  // Version 8 scenes cannot author terrain.
  const old = structuredClone(fixture);
  old.version = 8;
  old.revision = scene.revision + 1;
  scene.serve(old);
  const refused = await command("reload", { expected_revision: String(scene.revision), revision: String(old.revision) }, false);
  assert.equal(refused.error.code, "version_required");
  assert.equal(refused.error.path, "$.entities[1].terrain");
  checks.push({ check: "version 9 required" });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await browser.close();
  await rm(folder, { recursive: true, force: true });
  await rm(source, { force: true });
}
