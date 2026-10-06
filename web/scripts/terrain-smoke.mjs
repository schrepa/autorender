/**
 * Terrain checks (decision 051): a version 9 playground fixture with a
 * generated terrain streams, draws, casts shadows, collides and answers
 * height queries; the terrain control and environment section change what
 * draws; its horizon map casts long shadows (decision 052) and aerial
 * perspective hazes it (decision 053); an `ETR1` file baked from the same
 * layers streams by byte range and matches; a version 8 scene with terrain
 * or aerial perspective is refused.
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
    // Decision 068: rivers wear valleys into the hills for a few rounds,
    // baked over the two-meter tiles before any tile loads.
    { erosion: { level: 1, rounds: 6, incision: 0.001953125, creep: 2 } },
    // Decision 066: water drains off the hills, carving channels, baked
    // over the same tiles.
    { hydrology: { level: 1, channel_area: 20000, depth: 1, width: 4, max_depth: 3, max_width: 20 } },
    // Decision 069: a road through the south-western hills, leveled to its
    // own smoothed profile, which gravel covers.
    { path: { points: [[100, 300], [250, 260], [400, 150]], width: 6, shoulder: 6, smoothing: 60 } },
    { stamp: { center: [512, 512], radius: 40, falloff: 60, op: { flatten: 30 } } },
    { stamp: { center: [560, 512], radius: 4, op: "hole" } },
    { heightmap: { uri: "terrain-smoke/dome.r16", width: 65, height: 65, min: [700, 700], size: [128, 128], low: 0, high: 200, blend: "max" } },
  ],
  // Decision 054: meadow below, rock on steep ground; a summit biome of
  // snow above 120 m. The meadow grows grass (decision 056).
  surface: {
    materials: [
      { id: "meadow", color: [0.12, 0.2, 0.06], cover: { density: 60 } },
      { id: "rock", color: [0.3, 0.28, 0.26], roughness: 0.8 },
      { id: "snow", color: [0.85, 0.87, 0.9], roughness: 0.5 },
      { id: "gravel", color: [0.36, 0.33, 0.28], roughness: 0.9 },
    ],
    biomes: [
      { id: "lowland", layers: [{ material: "meadow" }, { material: "rock", mask: { slope: [25, 35] } }, { material: "gravel", mask: { path: [1, 0] } }] },
      { id: "summit", fitness: { height: [110, 130] }, layers: [{ material: "snow" }] },
    ],
    // Decision 057: crates stand on the meadow, a coarser box far away,
    // with balls, whose impostors far away cast cards (decision 060).
    assets: [
      // Both collide near the camera (decision 064): the crate as its
      // box, the ball as a sphere around it.
      { id: "crate", mesh: "box", material: "plain", lods: [{ mesh: "box", error: 0.0625 }], sway: 0.5, collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] } } },
      { id: "ball", mesh: "ball", material: "plain", collider: { shape: { kind: "sphere", radius: 1 } } },
    ],
    // A quarter of them also stand on the pinned level-2 tiles, hidden where
    // level-0 tiles hold them (decision 061).
    placements: [{ id: "crates", assets: [{ asset: "crate" }, { asset: "ball" }], density: 100, footprint: 4, materials: ["meadow"], scale: [0.5, 1], far: { level: 2, share: 0.25 } }],
  },
};
/** A unit icosphere subdivided once: 80 triangles, enough for an impostor. */
function ball() {
  const t = (1 + Math.sqrt(5)) / 2;
  const corners = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  const faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
    [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
    [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const unit = ([x, y, z]) => { const l = Math.hypot(x, y, z); return [x / l, y / l, z / l]; };
  const points = corners.map(unit);
  const middles = new Map();
  const middle = (a, b) => {
    const key = a < b ? `${a},${b}` : `${b},${a}`;
    if (!middles.has(key)) {
      middles.set(key, points.length);
      points.push(unit(points[a].map((value, axis) => value + points[b][axis])));
    }
    return middles.get(key);
  };
  const indices = faces.flatMap(([a, b, c]) => {
    const [ab, bc, ca] = [middle(a, b), middle(b, c), middle(c, a)];
    return [a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca];
  });
  return { vertices: points.map(point => ({ position: point, normal: point })), indices };
}
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
  meshes: [{ id: "box", source: { kind: "cube" } }, { id: "ball", source: { kind: "inline", ...ball() } }],
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

/** Renders until every wanted tile has been resident for a frame (bakes,
 * decisions 066 and 068, load them all at once); returns the observation. */
async function settle() {
  let rendering;
  let resident = 0;
  for (let frame = 0; frame < 40; frame++) {
    rendering = (await command("render")).result.rendering ?? (await command("status")).result.rendering;
    const ready = rendering.terrain_tiles > 0 && rendering.terrain_loading_tiles === 0 && rendering.terrain_fallback_patches === 0;
    resident = ready ? resident + 1 : 0;
    if (resident > 1) return rendering;
  }
  assert.fail(`terrain did not settle: ${JSON.stringify(rendering)}`);
}

async function capture(name) {
  // Pipelines go on compiling for seconds after the frames that first
  // need them, and a capture presents only once they have.
  for (let frame = 0; frame < 600; frame++) {
    if (!(await command("render")).result.rendering.pipelines_compiling) break;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
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
  assert.equal(settled.dropped_terrain_materials, 0);
  assert.ok(settled.placed_objects > 0, `crates stand on the meadow: ${settled.placed_objects}`);
  // The pinned tiles stand a share of them everywhere (decision 061).
  assert.ok(settled.placed_far_objects > 0 && settled.placed_far_objects < settled.placed_objects,
    `far crates: ${settled.placed_far_objects} of ${settled.placed_objects}`);
  // Occupancy (decision 059): the crates' covers are drawn around the camera.
  assert.ok(settled.occupancy_redraws > 0, `the occupancy map draws: ${settled.occupancy_redraws}`);
  // GPU-presented frames bake the balls' impostor (decision 060), which far
  // balls draw and cast as cards.
  let presented;
  for (let frame = 0; frame < 60; frame++) {
    presented = (await command("present", { alpha: 0, gpu: true })).result.rendering;
    if (presented.placed_impostors > 0) break;
  }
  assert.equal(presented.placed_impostors, 1, `the balls' impostor bakes: ${JSON.stringify(presented)}`);
  assert.equal(presented.dropped_impostors, 0);
  checks.push({ check: "streams and draws", tiles: settled.terrain_tiles, patches: settled.terrain_patches, triangles: settled.terrain_triangles, placed_impostors: presented.placed_impostors, placed_far: settled.placed_far_objects });

  // Discovery and inspection.
  const listing = await command("describe");
  for (const name of ["Terrain", "Terrains", "TerrainSettings"]) {
    assert.ok(listing.result.types.some(type => type.name === name), `describe lists ${name}`);
  }
  const look = await command("inspect", { component: "Environment" });
  assert.equal(look.result.value.terrain.samples_per_frame, 50_000_000);
  const inspected = await command("inspect", { entity: "ground", component: "Terrain" });
  const view = inspected.result.entity.components.Terrain;
  assert.equal(view.terrain.layers.length, 7);
  assert.deepEqual(view.terrain.layers[1], terrain.layers[1]);
  assert.deepEqual(view.terrain.layers[2], terrain.layers[2]);
  assert.deepEqual(view.terrain.layers[3], terrain.layers[3]);
  assert.equal(view.terrain.layers[6].heightmap.uri, "terrain-smoke/dome.r16");
  assert.deepEqual(view.extent, [1024, 1024]);
  assert.ok(view.resident_tiles_by_level.reduce((a, b) => a + b, 0) > 0);
  // Each level's largest measured difference from the next coarser one,
  // which its patches morph by, and ranges growing level by level.
  assert.ok(view.level_errors.slice(0, -1).every(error => error > 0), `measured ${view.level_errors}`);
  assert.ok(view.level_ranges.every((range, level) => level === 0 || range > view.level_ranges[level - 1]));
  assert.deepEqual(view.terrain.surface.materials.map(material => material.id), ["meadow", "rock", "snow", "gravel"]);
  assert.deepEqual(view.terrain.surface.biomes.map(biome => biome.id), ["lowland", "summit"]);
  assert.deepEqual(view.terrain.surface.assets.map(asset => asset.id), ["crate", "ball"]);
  assert.deepEqual(view.terrain.surface.assets[0], { id: "crate", mesh: "box", material: "plain", lods: [{ mesh: "box", error: 0.0625 }], sway: 0.5, collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] }, memberships: 1, filter: 4294967295 } });
  assert.deepEqual(view.terrain.surface.placements.map(layer => layer.id), ["crates"]);
  assert.deepEqual(view.terrain.surface.placements[0].far, { level: 2, share: 0.25 });
  checks.push({ check: "inspection", levels: view.levels, resident: view.resident_tiles_by_level, errors: view.level_errors, ranges: view.level_ranges });
  // The hydrology bake (decision 066) read every sample of the two-meter
  // tiles, found channels, and queries report what it found where they ask.
  const water = view.hydrology;
  assert.equal(water.baking, false, "the bake is done once tiles load");
  assert.equal(water.samples, 513 * 513, `the bake reads its level: ${JSON.stringify(water)}`);
  assert.ok(water.channel_samples > 0 && water.max_area > 20000, `water gathers into channels: ${JSON.stringify(water)}`);
  const [watered] = (await command("terrain_height", { points: [[-300, -300]] })).result.samples;
  assert.ok(watered.water && watered.water.flow > 0 && Number.isFinite(watered.water.wetness), `queries report water: ${JSON.stringify(watered)}`);
  checks.push({ check: "hydrology", samples: water.samples, channels: water.channel_samples, lakes: water.lake_samples, max_area: water.max_area });
  // The erosion bake (decision 068) ran its rounds over the same samples
  // first, wearing ground away.
  const worn = view.erosion;
  assert.equal(worn.baking, false, "erosion is done once tiles load");
  assert.equal(worn.samples, 513 * 513, `erosion reads its level: ${JSON.stringify(worn)}`);
  assert.ok(worn.rounds === 6 && worn.deepest > 0, `erosion wears ground away: ${JSON.stringify(worn)}`);
  checks.push({ check: "erosion", rounds: worn.rounds, deepest: worn.deepest, highest: worn.highest });
  // The path's bake (decision 069) profiled the ground along its line; the
  // road lies level across, and queries report how far its edge is.
  const [road] = view.paths;
  const line = terrain.layers[3].path.points;
  const length = Math.hypot(150, 40) + Math.hypot(150, 110);
  assert.ok(road && road.layer === 3 && road.baked, `the path bakes: ${JSON.stringify(view.paths)}`);
  assert.ok(Math.abs(road.length - length) < 1e-2 && road.samples === Math.ceil(length / 4) + 1, `the profile reads its line: ${JSON.stringify(road)}`);
  assert.ok(road.steepest > 0 && road.steepest < 0.6, `the road climbs gently: ${JSON.stringify(road)}`);
  const across = [];
  for (const t of [0.25, 0.5, 0.75]) {
    for (const [a, b] of [[line[0], line[1]], [line[1], line[2]]]) {
      const [dx, dz] = [b[0] - a[0], b[1] - a[1]];
      const run = Math.hypot(dx, dz);
      const [nx, nz] = [-dz / run, dx / run];
      const center = [a[0] + dx * t - 512, a[1] + dz * t - 512];
      across.push([center, [center[0] + 2 * nx, center[1] + 2 * nz], [center[0] - 2 * nx, center[1] - 2 * nz]]);
    }
  }
  const sections = await heights(across.flat());
  let tilt = 0;
  for (let i = 0; i < sections.length; i += 3) {
    const [center, left, right] = sections.slice(i, i + 3);
    assert.ok(Math.abs(center.path + 3) < 0.05, `the centerline lies 3 m inside the road's edges: ${JSON.stringify(center)}`);
    tilt = Math.max(tilt, Math.abs(left.height - right.height));
  }
  assert.ok(tilt < 0.15, `the road lies level across: ${tilt} m over 4 m`);
  const [away] = await heights([[400, -400]]);
  assert.equal(away.path, null, "far from roads, queries report none");
  checks.push({ check: "path", length: road.length, samples: road.samples, steepest: road.steepest, cut: road.deepest_cut, fill: road.highest_fill, tilt });
  // Routes (decision 069) keep within their grade between their ends.
  const routed = (await command("terrain_route", { from: [-400, 100], to: [300, -250], max_grade: 0.2 })).result;
  assert.equal(routed.terrain, "ground");
  const route = routed.route;
  assert.ok(route && route.steepest <= 0.2 + 1e-6 && route.points.length >= 2, `a route keeps within its grade: ${JSON.stringify(routed)}`);
  assert.deepEqual(routed.origin, [-512, -512]);
  const world = route.points.map(([x, z]) => [x + routed.origin[0], z + routed.origin[1]]);
  const [first, last] = [world[0], world[world.length - 1]];
  assert.ok(Math.hypot(first[0] + 400, first[1] - 100) < 3 && Math.hypot(last[0] - 300, last[1] + 250) < 3, `the route joins its ends: ${JSON.stringify([first, last])}`);
  assert.ok(route.length >= Math.hypot(700, 350) - 8, `a route is no shorter than its ends' distance: ${route.length}`);
  const unrouted = await command("terrain_route", { from: [-400, 100], to: [300, -250], level: 99 }, false);
  assert.equal(unrouted.error.code, "invalid_request");
  checks.push({ check: "route", points: route.points.length, length: route.length, steepest: route.steepest });

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
  // Placed crates and balls collide (decision 064): some rays meet them
  // above the ground, which still compares beneath them.
  assert.ok(ray.result.snapshot.placed_colliders > 0, `placed colliders: ${ray.result.snapshot.placed_colliders}`);
  assert.ok(sweep.collision.placed_first > 0, `rays meeting placed objects first: ${sweep.collision.placed_first}`);
  assert.ok(sweep.layers.compared > 100);
  assert.ok(sweep.layers.worst.max_difference < 5e-3, `layers differ by ${sweep.layers.worst.max_difference}`);
  checks.push({ check: "render versus collision", compared: sweep.collision.compared, worst: sweep.collision.worst.max_difference, layers: sweep.layers.worst.max_difference, placed_colliders: ray.result.snapshot.placed_colliders, placed_first: sweep.collision.placed_first });

  // The settings section: finer level of detail reaches further and draws
  // more triangles; invalid values are refused; disabled terrain draws
  // nothing.
  const fine = await command("environment", { terrain: { samples_per_frame: 50_000_000, pixel_error: 2 } });
  const finer = (await settle());
  const detailed = (await command("inspect", { entity: "ground", component: "Terrain" })).result.entity.components.Terrain;
  assert.ok(detailed.level_ranges[0] > view.level_ranges[0], `level 0 reaches ${detailed.level_ranges[0]} m`);
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
  // Each surface material in its own color.
  await command("terrain", { view: "materials" });
  const materialView = await capture("view-materials");
  assert.ok((await differentPixels(page, [baseline, materialView])).changed > 10_000, "the materials view differs");
  // Ground cover (decision 056): the meadow's grass grows near the eye,
  // counted a few frames late; turned off, none grows.
  await command("camera", { position: [-12, 32, -8], target: [10, 30, 6], vertical_fov_radians: 1, near: 0.1, far: 5000 });
  let grown;
  for (let frame = 0; frame < 20; frame++) {
    grown = (await command("render")).result.rendering;
    if (grown.ground_cover_blades > 0) break;
  }
  assert.ok(grown.ground_cover_blades > 0, `grass grows: ${JSON.stringify(grown)}`);
  assert.equal(grown.dropped_ground_cover, 0);
  const cover = (await command("inspect", { component: "GroundCover" })).result.value;
  await command("environment", { ground_cover: { ...cover, enabled: false } });
  let bare;
  for (let frame = 0; frame < 8; frame++) bare = (await command("render")).result.rendering;
  assert.equal(bare.ground_cover_blades, 0, "no blades once cover stops");
  await command("environment", { ground_cover: cover });
  checks.push({ check: "ground cover", blades: grown.ground_cover_blades });
  // The wind (decision 058) blades and swaying crates bend in: its section
  // reads back as set, and refuses speeds out of range.
  const calm = (await command("inspect", { component: "Wind" })).result.value;
  await command("environment", { wind: { velocity: [8, -2], gusts: 0.25 } });
  assert.deepEqual((await command("inspect", { component: "Wind" })).result.value, { velocity: [8, -2], gusts: 0.25 });
  const gale = await command("environment", { wind: { velocity: [80, 0], gusts: 0.25 } }, false);
  assert.equal(gale.error.code, "invalid_environment", "winds past 40 m/s are refused");
  assert.equal(gale.error.path, "$.wind.velocity");
  await command("environment", { wind: calm });
  checks.push({ check: "wind", calm });
  // Occupancy reads back as set, and refuses a resolution out of range.
  const occupancy = (await command("inspect", { component: "Occupancy" })).result.value;
  await command("environment", { occupancy: { ...occupancy, strength: 0.5 } });
  assert.equal((await command("inspect", { component: "Occupancy" })).result.value.strength, 0.5);
  const coarse = await command("environment", { occupancy: { ...occupancy, resolution: 16 } }, false);
  assert.equal(coarse.error.path, "$.occupancy.resolution");
  await command("environment", { occupancy });
  checks.push({ check: "occupancy", redraws: settled.occupancy_redraws });
  // Contact shadows (decision 062) trace with the frame's effects: the
  // blades and crates darken the sun's light where they reach; the section
  // reads back as set and refuses rays out of range.
  const contact = (await command("inspect", { component: "ContactShadows" })).result.value;
  const traced = (await command("environment", { contact_shadows: contact })).result.rendering;
  assert.equal(traced.contact_shadows, true, "contact shadows trace");
  const withContact = await capture("contact-shadows");
  await command("environment", { contact_shadows: { ...contact, enabled: false } });
  const withoutContact = await capture("no-contact-shadows");
  const contactPixels = await differentPixels(page, [withContact, withoutContact]);
  assert.ok(contactPixels.changed > 100, `contact shadows shade ${contactPixels.changed} pixels`);
  const reaching = await command("environment", { contact_shadows: { ...contact, length: 10 } }, false);
  assert.equal(reaching.error.path, "$.contact_shadows.length");
  await command("environment", { contact_shadows: contact });
  checks.push({ check: "contact shadows", pixels: contactPixels.changed });
  // Water (decision 067) stands in the bake's lakes: queries find its
  // surface and depth, the overview draws chunks of it, it shades pixels
  // and the section refuses water clearer than its range.
  const grid = [];
  for (let z = -480; z <= 480; z += 40) for (let x = -480; x <= 480; x += 40) grid.push([x, z]);
  const lakes = [];
  for (let i = 0; i < grid.length; i += 100) {
    const samples = (await command("terrain_height", { points: grid.slice(i, i + 100) })).result.samples;
    lakes.push(...samples.filter(sample => sample?.standing_water));
  }
  assert.ok(lakes.length > 0 && lakes.every(sample => sample.standing_water.depth > 0
    && Math.abs(sample.standing_water.surface - sample.standing_water.depth - sample.height) < 1e-2),
    `water stands in lakes: ${lakes.length}`);
  const waterSettings = (await command("inspect", { component: "Water" })).result.value;
  await command("camera", overview);
  const wet = (await command("environment", { water: waterSettings })).result.rendering;
  assert.ok(wet.water_chunks > 0, `water draws: ${wet.water_chunks}`);
  const withWater = await capture("water");
  await command("environment", { water: { ...waterSettings, enabled: false } });
  const withoutWater = await capture("no-water");
  const waterPixels = await differentPixels(page, [withWater, withoutWater]);
  assert.ok(waterPixels.changed > 1000, `water shades ${waterPixels.changed} pixels`);
  const murky = await command("environment", { water: { ...waterSettings, clarity: 0 } }, false);
  assert.equal(murky.error.path, "$.water.clarity");
  await command("environment", { water: waterSettings });
  checks.push({ check: "water", lake_samples: lakes.length, chunks: wet.water_chunks, pixels: waterPixels.changed });
  // Kept levels (decision 065) hold still casters sharper than the kept
  // map around the camera on GPU-presented frames, every tile once built;
  // with cascades out of reach, their shadows differ from the kept map's
  // alone. The section refuses more levels than the engine keeps.
  const kept = (await command("inspect", { component: "Shadows" })).result.value;
  const reachless = { ...kept, near_distance: 1, distance: 400 };
  const gpuCapture = async (name, shadows) => {
    await command("environment", { shadows });
    let rendering;
    for (let frame = 0; frame < 40; frame++) {
      rendering = (await command("present", { alpha: 1, gpu: true })).result.rendering;
      if (frame > 8 && rendering.shadow_levels_pending === 0) break;
    }
    await command("capture", { alpha: 1, gpu: true });
    const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
    return { rendering, image: bytes.toString("base64") };
  };
  const leveled = await gpuCapture("kept-levels", { ...reachless, levels: 2, level_extent: 64 });
  assert.equal(leveled.rendering.shadow_levels_pending, 0, `kept levels build: ${JSON.stringify(leveled.rendering)}`);
  const unleveled = await gpuCapture("kept-map-only", { ...reachless, levels: 0 });
  const levelPixels = await differentPixels(page, [leveled.image, unleveled.image]);
  assert.ok(levelPixels.changed > 100, `kept levels change ${levelPixels.changed} pixels`);
  const deep = await command("environment", { shadows: { ...kept, levels: 4 } }, false);
  assert.equal(deep.error.path, "$.shadows.levels");
  await command("environment", { shadows: kept });
  checks.push({ check: "kept levels", pixels: levelPixels.changed });
  await command("camera", overview);
  // The sky visibility map (decision 055) builds a band of rows a frame
  // once the pinned level is resident; its view shows it unlit.
  let sky;
  for (let frame = 0; frame < 200; frame++) {
    sky = (await command("render")).result.rendering;
    if (sky.terrain_sky_maps === 1 && sky.terrain_sky_rows === 0) break;
  }
  assert.equal(sky.terrain_sky_maps, 1, `the sky map builds: ${JSON.stringify(sky)}`);
  await command("terrain", { view: "sky" });
  const skyView = await capture("view-sky");
  assert.ok((await differentPixels(page, [baseline, skyView])).changed > 10_000, "the sky view differs");
  checks.push({ check: "sky visibility", maps: sky.terrain_sky_maps });
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

  // Long shadows (decision 052): a low sun behind the dome, with shadow maps
  // reaching 60 m, shadows the plateau through the horizon map alone. A new
  // sun rebuilds the map, an unchanged one keeps it.
  const current = (await command("inspect", { component: "Environment" })).result.value;
  const atmosphere = { ...current.atmosphere, sun_direction: [1, 0.22, 1] };
  const shadows = { ...current.shadows, distance: 60, near_distance: 20 };
  // The control renders a frame with the new sun.
  const rebuilt = (await command("environment", { atmosphere, shadows })).result.rendering;
  assert.equal(rebuilt.terrain_horizons, 1);
  assert.ok(rebuilt.terrain_horizon_builds >= 1, `a new sun rebuilds: ${JSON.stringify(rebuilt)}`);
  assert.equal((await command("render")).result.rendering.terrain_horizon_builds, 0, "an unchanged sun keeps the map");
  const above = { ...overview, position: [-60, 320, 220], target: [0, 30, 0] };
  await command("camera", above);
  await settle();
  const long = await capture("horizon-shadows");
  const flat = await command("environment", { terrain: { samples_per_frame: 50_000_000, sun_shadows: false } });
  assert.equal(flat.result.rendering.terrain_horizons, 0);
  await settle();
  const short = await capture("map-shadows-only");
  const lengthened = await differentPixels(page, [long, short]);
  assert.ok(lengthened.changed > 20_000, `the dome's long shadow covers ${lengthened.changed} pixels`);
  checks.push({ check: "horizon shadows", builds: rebuilt.terrain_horizon_builds, pixels: lengthened.changed });

  // Aerial perspective (decision 053): thick air from the eye hazes the
  // terrain; none leaves it as it was.
  await command("environment", { terrain: { samples_per_frame: 50_000_000 }, atmosphere: { ...atmosphere, aerial_start: 0, aerial_density: 0 } });
  await settle();
  const clear = await capture("no-air");
  await command("environment", { atmosphere: { ...atmosphere, aerial_start: 0, aerial_density: 0.004 } });
  await settle();
  const hazed = await capture("thick-air");
  const air = await differentPixels(page, [clear, hazed]);
  assert.ok(air.changed > 100_000, `air hazes ${air.changed} pixels`);
  await command("environment", { atmosphere: current.atmosphere, shadows: current.shadows });
  await command("camera", overview);
  await settle();
  checks.push({ check: "aerial perspective", pixels: air.changed });

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

  // Version 8 scenes cannot author terrain or aerial perspective.
  const old = structuredClone(fixture);
  old.version = 8;
  old.revision = scene.revision + 1;
  scene.serve(old);
  const refused = await command("reload", { expected_revision: String(scene.revision), revision: String(old.revision) }, false);
  assert.equal(refused.error.code, "version_required");
  assert.equal(refused.error.path, "$.entities[1].terrain");
  const hazy = structuredClone(old);
  hazy.entities.splice(1, 1);
  hazy.environment = { atmosphere: { preset: "day", aerial_density: 0.0002 } };
  hazy.revision = old.revision + 1;
  scene.serve(hazy);
  const refusedAir = await command("reload", { expected_revision: String(scene.revision), revision: String(hazy.revision) }, false);
  assert.equal(refusedAir.error.code, "version_required");
  assert.equal(refusedAir.error.path, "$.environment.atmosphere.aerial_density");
  checks.push({ check: "version 9 required" });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await browser.close();
  await rm(folder, { recursive: true, force: true });
  await rm(source, { force: true });
}
