/**
 * Terrain checks (decision 051): a version 9 playground fixture with a
 * generated terrain streams, draws, casts shadows, collides and answers
 * height queries; the terrain control and environment section change what
 * draws; its horizon map casts long shadows (decision 052) and aerial
 * perspective hazes it (decision 053); an `ETR1` file baked from the same
 * layers streams by byte range and matches; a version 8 scene with terrain
 * or aerial perspective is refused; the import tool (decision 070) brings
 * the dome in as a heightmap of its own and the scene it writes loads.
 *
 * Bakes its file with `cargo run -p engine_assetc --release --bin
 * terrain_bake` and imports with `terrain_import` (the cargo on PATH) into
 * web/public/scenes/, removing what they wrote afterwards with the
 * fixture's source.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { artifacts, commander, devUrl, differentPixels, launch, openPaused, routedScene } from "./harness.mjs";

const url = devUrl("playground.html");
const output = await artifacts("terrain-smoke");
const scenes = fileURLToPath(new URL("../public/scenes/", import.meta.url));
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
    // A trench across the road's last stretch, twelve meters deep.
    { stamp: { center: [430, 96], radius: 10, falloff: 10, op: { lower: 12 } } },
    // Decision 069: a road through the south-western hills, leveled to its
    // own smoothed profile, which gravel covers; it bridges the trench
    // (decision 072).
    { path: { points: [[100, 300], [250, 260], [400, 150], [450, 60]], width: 6, shoulder: 6, smoothing: 60, bridge_fill: 4, tunnel_cut: 25 } },
    // A mound forty meters high, and a straight road through it on a long
    // grade, which tunnels (decision 073).
    { stamp: { center: [750, 200], radius: 30, falloff: 40, op: { raise: 40 } } },
    { path: { points: [[600, 200], [900, 200]], width: 6, shoulder: 6, smoothing: 400, bridge_fill: 4, tunnel_cut: 15 } },
    { stamp: { center: [512, 512], radius: 40, falloff: 60, op: { flatten: 30 } } },
    { stamp: { center: [560, 512], radius: 4, op: "hole" } },
    { heightmap: { uri: "terrain-smoke/dome.r16", width: 65, height: 65, min: [700, 700], size: [128, 128], low: 0, high: 200, blend: "max" } },
  ],
  // Decision 054: meadow below, rock on steep ground; a summit biome of
  // snow above 120 m. The meadow grows grass (decision 056). Tracks sink
  // snow deepest and rock never (decision 083).
  surface: {
    materials: [
      { id: "meadow", color: [0.12, 0.2, 0.06], cover: { density: 60 }, deform: 0.05 },
      { id: "rock", color: [0.3, 0.28, 0.26], roughness: 0.8 },
      { id: "snow", color: [0.85, 0.87, 0.9], roughness: 0.5, deform: 0.3 },
      // Gravel's grain is the first map, which sets the maps' size: two
      // pages of the runtime virtual texture (decision 124) along a side.
      { id: "gravel", material: "grain", color: [0.36, 0.33, 0.28], roughness: 0.9, relief: 0.1, deform: 0.04 },
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
    // Decision 085: an author stands a big crate on the plateau, and a ball
    // on the level-1 tiles, which stay resident farther away.
    placed: [{ asset: "crate", position: [530, 500], yaw: 30, scale: 3 }, { asset: "ball", position: [500, 530], scale: 2, level: 1 }],
  },
  // Decision 094: ground generated past its edges to the horizon, its bakes
  // eased out over 128 m.
  surroundings: { reach: 3000, blend: 128 },
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
/** A 288-texel square of gray grain, base64 RGBA8. */
function grain() {
  const size = 288;
  const bytes = Buffer.alloc(size * size * 4);
  for (let texel = 0; texel < size * size; texel++) {
    const hash = Math.imul(texel ^ (texel >>> 7), 0x9e3779b1) >>> 0;
    const value = 150 + (hash >>> 26) * 1.5;
    bytes.set([value, value * 0.95, value * 0.9, 255], texel * 4);
  }
  return bytes.toString("base64");
}
const fixture = {
  version: 9, id: "terrain-fixture", revision: 1,
  camera: { position: [0, 80, 140], target: [0, 30, 0], fov_degrees: 60, near: 0.5, far: 4000 },
  textures: [{ id: "grain", width: 288, height: 288, rgba8: grain() }],
  materials: [{ id: "plain", color: [0.8, 0.3, 0.2, 1] }, { id: "grain", color: [1, 1, 1, 1], texture: "grain" }],
  meshes: [{ id: "box", source: { kind: "cube" } }, { id: "ball", source: { kind: "inline", ...ball() } }],
  entities: [
    { id: "player", mesh: "box", material: "plain", tags: ["player"], collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] }, memberships: 2, filter: 1, sensor: false }, transform: { translation: [0, 31, 0] } },
    { id: "ground", terrain },
    // Probes capture terrain on GPU-presented frames (decision 043).
    { id: "probe", reflection_probe: { extent: [60, 40, 60] }, transform: { translation: [0, 45, 0] } },
    // Rides the sea once one stands (decision 077).
    { id: "buoy", mesh: "box", material: "plain", floating: { draft: 0.25, tilt: 1 }, transform: { translation: [400, 5, -400] } },
    // An interior's box lets in a tenth of the sky and sun (decision 079).
    { id: "hall", reflection_probe: { extent: [20, 200, 20], blend: 0.5, interior: { sky: 0.1, sun: 0.1 } }, transform: { translation: [-300, 40, 300] } },
    // Walks where it is sent (decision 089).
    { id: "hiker", mesh: "box", material: "plain", nav_agent: { speed: 6, offset: 0.5 }, transform: { translation: [20, 31, 20] } },
    // Presses a dimple into the plateau's meadow each tick (decision 083).
    { id: "roller", deformer: { radius: 0.5, depth: 0.2, contact: 0.5, flatness: 0.2 }, transform: { translation: [10, 30.2, 10] } },
  ],
};
const overview = { position: [0, 80, 140], target: [0, 30, 0], vertical_fov_radians: Math.PI / 3, near: 0.5, far: 4000 };
const { browser, page, errors } = await launch();
const command = commander(page);
const scene = routedScene(page, fixture);
const checks = [];

/** Renders until every wanted tile has been resident for a frame (bakes,
 * decisions 066 and 068, load them all at once); returns the observation.
 * Job workers bake and generate tiles beside the page (decision 110), their
 * outputs arriving between commands, so renders wait a little for them. */
async function settle() {
  let rendering;
  let resident = 0;
  for (let frame = 0; frame < 600; frame++) {
    rendering = (await command("render")).result.rendering ?? (await command("status")).result.rendering;
    const ready = rendering.terrain_tiles > 0 && rendering.terrain_loading_tiles === 0 && rendering.terrain_fallback_patches === 0;
    resident = ready ? resident + 1 : 0;
    if (resident > 1) return rendering;
    if (!ready) await new Promise(resolve => setTimeout(resolve, 25));
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

/** The mean color of the top third of a capture (base64 PNG): the sky. */
async function skyColor(png) {
  return page.evaluate(async png => {
    const image = new Image();
    image.src = `data:image/png;base64,${png}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    const { data } = context.getImageData(0, 0, image.width, Math.floor(image.height / 3));
    const sum = [0, 0, 0];
    for (let offset = 0; offset < data.length; offset += 4) {
      sum[0] += data[offset]; sum[1] += data[offset + 1]; sum[2] += data[offset + 2];
    }
    return sum.map(value => value / (data.length / 4));
  }, png);
}

/** Renders until `done` holds for a frame's observation, the virtual
 * texture's read-backs arriving a few frames late; returns it. */
async function rendersUntil(done, what) {
  let rendering;
  for (let frame = 0; frame < 600; frame++) {
    rendering = (await command("render")).result.rendering;
    if (done(rendering)) return rendering;
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  assert.fail(`${what}: ${JSON.stringify(rendering)}`);
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
  assert.equal(view.terrain.layers.length, 10);
  assert.deepEqual(view.terrain.layers[1], terrain.layers[1]);
  assert.deepEqual(view.terrain.layers[2], terrain.layers[2]);
  assert.deepEqual(view.terrain.layers[4], terrain.layers[4]);
  assert.deepEqual(view.terrain.layers[6], terrain.layers[6]);
  assert.equal(view.terrain.layers[9].heightmap.uri, "terrain-smoke/dome.r16");
  assert.deepEqual(view.extent, [1024, 1024]);
  assert.ok(view.resident_tiles_by_level.reduce((a, b) => a + b, 0) > 0);
  // Each level's largest measured difference from the next coarser one,
  // which its patches morph by, and ranges growing level by level.
  assert.ok(view.level_errors.slice(0, -1).every(error => error > 0), `measured ${view.level_errors}`);
  assert.ok(view.level_ranges.every((range, level) => level === 0 || range > view.level_ranges[level - 1]));
  assert.deepEqual(view.terrain.surface.materials.map(material => material.id), ["meadow", "rock", "snow", "gravel"]);
  // Relief (decision 082): what a material sinks between its stones reads
  // back as authored, and its steps take a range.
  assert.ok(Math.abs(view.terrain.surface.materials[3].relief - 0.1) < 1e-6, `relief reads back: ${view.terrain.surface.materials[3].relief}`);
  const stepless = await command("environment", { terrain: { relief_steps: 1 } }, false);
  assert.equal(stepless.error.path, "$.terrain.relief_steps");
  assert.deepEqual(view.terrain.surface.biomes.map(biome => biome.id), ["lowland", "summit"]);
  assert.deepEqual(view.terrain.surface.assets.map(asset => asset.id), ["crate", "ball"]);
  assert.deepEqual(view.terrain.surface.assets[0], { id: "crate", mesh: "box", material: "plain", lods: [{ mesh: "box", error: 0.0625 }], sway: 0.5, collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] }, memberships: 1, filter: 4294967295 } });
  assert.deepEqual(view.terrain.surface.placements.map(layer => layer.id), ["crates"]);
  assert.deepEqual(view.terrain.surface.placements[0].far, { level: 2, share: 0.25 });
  checks.push({ check: "inspection", levels: view.levels, resident: view.resident_tiles_by_level, errors: view.level_errors, ranges: view.level_ranges });
  // Surroundings (decision 094): ground generated past the terrain's edges
  // from its layers, four meters a sample, meets its edges where both read
  // the same ground and fills the view from on high; queries past the
  // edges read it, it places nothing, and settings turn it off.
  {
    const around = view.surroundings;
    assert.ok(around && around.reach.every(reach => reach >= 3000) && around.spacing === 4 && around.resident_tiles > 0,
      `the surroundings stream: ${JSON.stringify(around)}`);
    const status = (await command("inspect", { component: "Terrains" })).result.value;
    assert.ok(status.terrains === 1 && status.surroundings_tiles > 0 && status.surroundings_patches > 0,
      `the surroundings draw: ${JSON.stringify(status)}`);
    const pairs = [];
    for (const along of [-300, 0, 300]) {
      pairs.push([[511.5, along], [512.5, along]], [[-511.5, along], [-512.5, along]],
        [[along, 511.5], [along, 512.5]], [[along, -511.5], [along, -512.5]]);
    }
    const sampled = await heights(pairs.flat());
    for (let index = 0; index < pairs.length; index++) {
      const [inside, outside] = [sampled[2 * index], sampled[2 * index + 1]];
      assert.ok(inside && outside && Math.abs(inside.height - outside.height) < 3,
        `the edge meets at ${JSON.stringify(pairs[index])}: ${JSON.stringify([inside, outside])}`);
    }
    const beyond = await heights([[2500, 2500], [-3000, 1000]]);
    assert.ok(beyond.every(sample => sample && Number.isFinite(sample.height)), `queries past the edges: ${JSON.stringify(beyond)}`);
    await command("camera", { ...overview, position: [0, 1500, 1800], target: [0, 0, -2000], far: 20000 });
    const surroundedFrame = await settle();
    const surrounded = await capture("surroundings");
    const edged = await command("environment", { terrain: { samples_per_frame: 50_000_000, surroundings: false } });
    assert.deepEqual(edged.result.environment_changed, ["terrain"]);
    const alone = await settle();
    assert.equal((await command("inspect", { component: "Terrains" })).result.value.surroundings_patches, 0, "settings turn the surroundings off");
    assert.equal(alone.placed_objects, surroundedFrame.placed_objects, "the surroundings place nothing");
    const bare = await capture("no-surroundings");
    const filled = await differentPixels(page, [surrounded, bare]);
    assert.ok(filled.changed > 0.2 * filled.total, `the surroundings fill ${filled.changed} of ${filled.total} pixels`);
    await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
    await command("camera", overview);
    await settle();
    checks.push({ check: "surroundings", reach: around.reach, tiles: status.surroundings_tiles, patches: status.surroundings_patches, pixels: filled.changed });
  }
  // The hydrology bake (decision 066) read every sample of the two-meter
  // tiles, found channels, and queries report what it found where they ask.
  const water = view.hydrology;
  assert.equal(water.baking, false, "the bake is done once tiles load");
  assert.equal(water.samples, 513 * 513, `the bake reads its level: ${JSON.stringify(water)}`);
  assert.ok(water.channel_samples > 0 && water.max_area > 20000, `water gathers into channels: ${JSON.stringify(water)}`);
  const [watered] = (await command("terrain_height", { points: [[-300, -300]] })).result.samples;
  assert.ok(watered.water && watered.water.flow > 0 && Number.isFinite(watered.water.wetness), `queries report water: ${JSON.stringify(watered)}`);
  // Waterfalls (decision 075) are where its rivers drop steeply; the
  // engine spawns what draws each one.
  assert.equal((await command("inspect", { component: "Terrains" })).result.value.falls, water.falls, `falls spawn as found: ${JSON.stringify(water)}`);
  checks.push({ check: "hydrology", samples: water.samples, channels: water.channel_samples, lakes: water.lake_samples, max_area: water.max_area, falls: water.falls });
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
  const line = terrain.layers[4].path.points;
  const length = Math.hypot(150, 40) + Math.hypot(150, 110) + Math.hypot(50, 90);
  assert.ok(road && road.layer === 4 && road.baked, `the path bakes: ${JSON.stringify(view.paths)}`);
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
  let tilt = 0, level = 0;
  for (let i = 0; i < sections.length; i += 3) {
    const [center, left, right] = sections.slice(i, i + 3);
    // Bridges (decision 072) carry the road over the hills' streams.
    if (center.path === null) continue;
    assert.ok(Math.abs(center.path + 3) < 0.05, `the centerline lies 3 m inside the road's edges: ${JSON.stringify(center)}`);
    tilt = Math.max(tilt, Math.abs(left.height - right.height));
    level += 1;
  }
  assert.ok(level >= 3 && tilt < 0.15, `the road lies level across: ${tilt} m over 4 m at ${level} sections`);
  const [away] = await heights([[400, -400]]);
  assert.equal(away.path, null, "far from roads, queries report none");
  checks.push({ check: "path", length: road.length, samples: road.samples, steepest: road.steepest, cut: road.deepest_cut, fill: road.highest_fill, tilt });
  // Decision 072: where the road would fill over the trench it bridges:
  // the ground under the deck stays a trench, which is no road, and rays
  // land on the deck above it.
  assert.ok(road.bridges >= 1 && road.bridged > 15, `the road bridges the trench: ${JSON.stringify(road)}`);
  assert.equal((await command("inspect", { component: "Terrains" })).result.value.bridges, view.paths.reduce((sum, path) => sum + path.bridges, 0));
  const trench = [430 - 512, 96 - 512];
  const [floor] = await heights([trench]);
  assert.equal(floor.path, null, `under its bridge the ground is no road: ${JSON.stringify(floor)}`);
  const landing = (await command("raycast", { origin: [trench[0], 300, trench[1]], direction: [0, -1, 0], max_distance: 600 })).result.hit;
  assert.ok(landing?.placed?.layer === 65535 && landing.point[1] > floor.height + 5, `rays land on the deck: ${JSON.stringify(landing)} over ${floor.height}`);
  checks.push({ check: "bridge", bridged: road.bridged, deck: landing.point[1], floor: floor.height });
  // Decision 073: the straight road bores through the mound: no road on the
  // ground above it, which rays land on, and the ground opens at its
  // portals.
  const bore = view.paths[1];
  assert.ok(bore && bore.layer === 6 && bore.tunnels === 1 && bore.tunneled > 40, `the road tunnels the mound: ${JSON.stringify(bore)}`);
  assert.equal((await command("inspect", { component: "Terrains" })).result.value.tunnels, 1);
  const [mound] = await heights([[750 - 512, 200 - 512]]);
  assert.ok(mound.path === null && !mound.hole, `above the tunnel the ground is no road: ${JSON.stringify(mound)}`);
  // Rays from above land on the mound where its collision tiles are
  // resident, and on the tunnel's floor, well below it, where they are not.
  const top = (await command("raycast", { origin: [750 - 512, 400, 200 - 512], direction: [0, -1, 0], max_distance: 600 })).result.hit;
  const landed = top?.placed ? top.placed.layer === 65535 && top.point[1] < mound.height - 10 : Math.abs(top?.point[1] - mound.height) < 0.05;
  assert.ok(landed, `rays land on the mound or the floor under it: ${JSON.stringify(top)} under ${mound.height}`);
  const bored = [];
  for (let x = 600; x <= 900; x += 3) bored.push([x - 512, 200 - 512]);
  const portals = (await heights(bored)).filter(sample => sample?.hole).length;
  assert.ok(portals > 0, "the ground opens at its portals");
  checks.push({ check: "tunnel", tunneled: bore.tunneled, portals, mound: mound.height });
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


  // Speed-aware detail (decision 115): while the camera moves fast the
  // finest levels' tiles in view wait, and those out of view until it
  // settles; then they load. Collision's tiles never wait, so collision is
  // off here.
  const moving = { samples_per_frame: 50_000_000, collision_radius: 0, motion_detail: { speed: 1, fastest: 2, levels: 8, settle: 0.25 } };
  await command("environment", { terrain: moving });
  let waited = 0;
  for (let step = 0; step < 12; step++) {
    const along = -300 + step * 50;
    await command("camera", { position: [along, 120, along], target: [along + 100, 30, along + 100], vertical_fov_radians: Math.PI / 3, near: 0.5, far: 4000 });
    await command("step", { ticks: 1 });
    waited = Math.max(waited, (await command("inspect", { component: "Terrains" })).result.value.waiting_tiles);
  }
  assert.ok(waited > 0, `tiles wait while the camera moves fast: ${waited}`);
  for (let step = 0; step < 80; step++) await command("step", { ticks: 1 });
  const rested = (await command("inspect", { component: "Terrains" })).result.value;
  assert.equal(rested.waiting_tiles, 0, `nothing waits once the camera settles: ${JSON.stringify(rested)}`);
  await settle();
  const tooMany = await command("environment", { terrain: { ...moving, motion_detail: { levels: 9 } } }, false);
  assert.equal(tooMany.error.code, "invalid_environment");
  assert.equal(tooMany.error.path, "$.terrain.motion_detail.levels");
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  await command("camera", overview);
  await settle();
  checks.push({ check: "motion detail", waited });

  // Decision 116: tiles the view leaves stay in memory and return at once
  // when it comes back; with nothing missing, workers generate the tiles
  // within the idle reach into memory.
  const inspectTerrains = async () => (await command("inspect", { component: "Terrains" })).result.value;
  const home = await inspectTerrains();
  await command("camera", { ...overview, position: [-420, 90, -420], target: [-300, 30, -300] });
  await settle();
  // Decision 117: the few tiles under the view after the move generated in
  // bands of rows across idle workers, and hold the heights the layers
  // give, as whole tiles do.
  const corner = await inspectTerrains();
  assert.ok(corner.banded_tiles > home.banded_tiles, `tiles under the view generate in bands: ${JSON.stringify(corner)}`);
  const joined = (await command("terrain_sweep", { center: [-380, -380], half_extent: 100, samples: 24 })).result;
  assert.ok(joined.layers.compared > 100, `the sweep compared ${joined.layers.compared}`);
  assert.ok(joined.layers.worst.max_difference < 5e-3, `banded tiles differ from the layers by ${joined.layers.worst.max_difference}`);
  await command("camera", overview);
  await settle();
  const back = await inspectTerrains();
  assert.ok(back.restored_tiles > home.restored_tiles, `tiles return from memory: ${JSON.stringify(back)}`);
  // A budget of a byte empties memory, so what it holds next was made
  // ahead of the view.
  await command("environment", { terrain: { samples_per_frame: 50_000_000, cache_bytes: 1 } });
  await command("render");
  const emptied = await inspectTerrains();
  assert.equal(emptied.cached_tiles, 0, `a byte keeps no tile: ${JSON.stringify(emptied)}`);
  await command("environment", { terrain: { samples_per_frame: 50_000_000, idle_reach: 4 } });
  let ahead = emptied;
  let generating = 0;
  for (let frame = 0; frame < 300 && ahead.cached_tiles === 0; frame++) {
    await command("render");
    await new Promise(resolve => setTimeout(resolve, 25));
    ahead = await inspectTerrains();
    generating = Math.max(generating, ahead.idle_tiles);
  }
  assert.ok(ahead.cached_tiles > 0, `workers generate tiles ahead of the view: ${JSON.stringify(ahead)}`);
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  await settle();
  checks.push({ check: "memory", restored: back.restored_tiles, banded: corner.banded_tiles, cached: ahead.cached_tiles, bytes: ahead.cached_bytes, generating });

  // Decision 119: tiles generated in bands place their objects in bands
  // too, the same objects as whole tiles place: the view of tiles banded
  // draws as many as of the same tiles generated whole.
  const cornerView = { ...overview, position: [-420, 90, -420], target: [-300, 30, -300] };
  await command("camera", cornerView);
  await settle();
  const bandedObjects = (await command("render")).result.rendering.placed_objects;
  await command("environment", { terrain: { samples_per_frame: 50_000_000, max_bands: 1, cache_bytes: 1 } });
  await command("camera", overview);
  await settle();
  await command("camera", cornerView);
  await settle();
  const wholeObjects = (await command("render")).result.rendering.placed_objects;
  assert.ok(bandedObjects > 0, "objects stand on the banded tiles");
  assert.equal(wholeObjects, bandedObjects, "banded and whole tiles place the same objects");
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  await command("camera", overview);
  await settle();
  checks.push({ check: "placing bands", objects: bandedObjects });

  // Decision 118: the GPU gives tiles their surface materials, the rules'
  // own but where its rounding crosses a threshold: the view drawn with
  // the CPU's differs in few pixels. Back on the GPU, tiles loading at a
  // new place get theirs from it as they upload.
  const gpuView = await capture("gpu-materials");
  await command("environment", { terrain: { samples_per_frame: 50_000_000, gpu_materials: false } });
  await settle();
  const cpuView = await capture("cpu-materials");
  const materials = await differentPixels(page, [gpuView, cpuView], { tolerance: 8 });
  assert.ok(materials.changed < materials.total * 0.01, `GPU materials differ from the CPU's in ${materials.changed} of ${materials.total} pixels`);
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  await settle();
  await command("camera", { ...overview, position: [380, 90, -420], target: [300, 30, -300] });
  let controls = 0;
  for (let frame = 0; frame < 300; frame++) {
    const rendering = (await command("render")).result.rendering;
    controls += rendering.terrain_gpu_controls;
    if (controls > 0 && rendering.terrain_loading_tiles === 0 && rendering.terrain_fallback_patches === 0) break;
    await new Promise(resolve => setTimeout(resolve, 25));
  }
  assert.ok(controls > 0, "the GPU gives new tiles their control values");
  await command("camera", overview);
  await settle();
  checks.push({ check: "gpu materials", differing: materials.changed, controls });

  // Decision 124: the runtime virtual texture bakes the ground into the
  // pages the view asks for, its sheets growing to hold them, and the
  // pixels reading them show the ground as computing it does; turned off,
  // it keeps no sheets. Under a budget too small, the pages asked for
  // beyond it drop and the ground computes itself.
  const vt = rendering => ({ slots: rendering.terrain_vt_slots, pages: rendering.terrain_vt_pages, bakes: rendering.terrain_vt_bakes, missing: rendering.terrain_vt_missing, dropped: rendering.dropped_terrain_vt_pages });
  const textured = await rendersUntil(rendering => rendering.terrain_vt_pages > 0 && rendering.terrain_vt_missing === 0 && rendering.terrain_vt_bakes === 0, "the view's pages bake");
  assert.ok(textured.terrain_vt_slots >= textured.terrain_vt_pages && textured.dropped_terrain_vt_pages === 0, `pages bake into the sheets: ${JSON.stringify(vt(textured))}`);
  const bakedView = await capture("virtual-texture");
  await command("environment", { terrain: { samples_per_frame: 50_000_000, virtual_texture: false } });
  const computed = await settle();
  assert.ok(computed.terrain_vt_slots === 0 && computed.terrain_vt_pages === 0, `turned off, it keeps nothing: ${JSON.stringify(vt(computed))}`);
  const computedView = await capture("virtual-texture-off");
  const reading = await differentPixels(page, [bakedView, computedView]);
  const seen = await differentPixels(page, [bakedView, computedView], { tolerance: 8 });
  assert.ok(reading.changed > 0 && seen.changed < seen.total * 0.03, `pixels read the pages (${reading.changed} differ) and show the computed ground (${seen.changed} of ${seen.total} by more than 8)`);
  await command("environment", { terrain: { samples_per_frame: 50_000_000, virtual_texture_pages: 4 } });
  const scarce = await rendersUntil(rendering => rendering.dropped_terrain_vt_pages > 0, "pages beyond the budget drop");
  assert.ok(scarce.terrain_vt_slots === 4 && scarce.terrain_vt_pages <= 4, `the budget bounds the pages: ${JSON.stringify(vt(scarce))}`);
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  await settle();
  checks.push({ check: "virtual texture", pages: textured.terrain_vt_pages, slots: textured.terrain_vt_slots, read: reading.changed, differing: seen.changed, dropped: scarce.dropped_terrain_vt_pages });

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
  for (let frame = 0; frame < 60; frame++) {
    grown = (await command("render")).result.rendering;
    if (grown.ground_cover_blades > 0) break;
    // The tiles around the eye arrive from workers (decision 110).
    await new Promise(resolve => setTimeout(resolve, 25));
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
  // The sea (decision 076) stands at its level wherever the ground lies
  // lower: queries sum the waves frames draw, which move with time and stay
  // within their height, its sectors draw, and a camera just over the
  // surface the CPU finds sees the sea where one just under it sees the
  // ground below.
  {
    const still = (await command("inspect", { component: "Ocean" })).result.value;
    assert.equal(still.enabled, false, "the sea is off by default");
    const sea = { ...still, enabled: true, level: 40 };
    await command("environment", { ocean: sea });
    const shore = [];
    for (let z = -440; z <= 440; z += 80) for (let x = -440; x <= 440; x += 80) shore.push([x, z]);
    const tide = (await command("water_query", { points: shore, seconds: 12.5 })).result;
    assert.ok(tide.level === 40 && tide.amplitude > 0.1 && tide.waves.length === sea.waves, `the sea's waves: ${JSON.stringify(tide).slice(0, 400)}`);
    assert.ok(tide.samples.every(sample => Math.abs(sample.ocean.height - 40) <= tide.amplitude + 1e-2
      && Math.abs(Math.hypot(...sample.ocean.normal) - 1) < 1e-3 && sample.ocean.normal[1] > 0.5),
      `surfaces stay within the waves' height: ${JSON.stringify(tide.samples.slice(0, 3))}`);
    const later = (await command("water_query", { points: shore, seconds: 13.5 })).result;
    assert.ok(later.samples.some((sample, index) => Math.abs(sample.ocean.height - tide.samples[index].ocean.height) > 1e-2), "waves move");
    const [depth, deepest] = tide.samples.map((sample, index) => [sample.depth ?? -1, shore[index]]).sort((a, b) => b[0] - a[0])[0];
    assert.ok(depth > 8, `the sea stands deep somewhere: ${depth}`);
    const surface = (await command("water_query", { points: [deepest] })).result.samples[0].ocean.height;
    const down = offset => ({ position: [deepest[0], surface + offset, deepest[1]], target: [deepest[0], surface - 50, deepest[1] + 0.01], vertical_fov_radians: 0.2, near: 0.05, far: 2000 });
    await command("camera", down(0.3));
    const afloat = (await command("render")).result.rendering;
    assert.ok(afloat.ocean_sectors > 0 && afloat.ocean_triangles > 0, `the sea draws: ${afloat.ocean_sectors} sectors`);
    const above = await capture("sea-above");
    await command("camera", down(-0.3));
    const below = await capture("sea-below");
    const crossing = await differentPixels(page, [above, below], { tolerance: 24 });
    // Rivers (decision 092) carve their mouths deep under the sea, which
    // darkens it from above, and the sky's state (how far its clouds have
    // marched) sets how far the two views differ: between a quarter and half
    // of the view changes across the surface, where a surface drawn off the
    // queried one leaves both cameras on one side and the views alike.
    assert.ok(crossing.changed > 0.15 * crossing.total, `the drawn surface lies where queries find it: ${crossing.changed} of ${crossing.total} pixels change across it`);
    // Floating entities (decision 077) ride the sea each fixed tick at the
    // tick's time, which frames present: the buoy stands on the surface
    // queries find there.
    await command("step", { ticks: 1 });
    const buoy = (await command("inspect", { entity: "buoy", component: "Transform" })).result.entity.components.Transform;
    const swell = (await command("water_query", { points: [[400, -400]] })).result.samples[0].ocean;
    assert.ok(Math.abs(buoy.translation[1] - (swell.height - 0.25)) < 2e-3, `the buoy floats: ${JSON.stringify(buoy)} on ${JSON.stringify(swell)}`);
    // Under the sea its water veils everything (decision 077): a camera a
    // meter under the surface sees nothing like one a meter over it.
    const level = (await command("water_query", { points: [deepest] })).result.samples[0].ocean.height;
    const across = offset => ({ position: [deepest[0], level + offset, deepest[1]], target: [deepest[0] + 40, level + offset, deepest[1] + 40], vertical_fov_radians: 0.9, near: 0.05, far: 2000 });
    await command("camera", across(1));
    const dry = await capture("sea-afloat");
    await command("camera", across(-1));
    const diving = await capture("sea-diving");
    const veiled = await differentPixels(page, [dry, diving], { tolerance: 24 });
    assert.ok(veiled.changed > 0.5 * veiled.total, `under the sea, its water veils the view: ${veiled.changed} of ${veiled.total} pixels change`);
    const sparse = await command("environment", { ocean: { ...sea, waves: 0 } }, false);
    assert.equal(sparse.error.path, "$.ocean.waves");
    await command("environment", { ocean: still });
    await command("camera", overview);
    checks.push({ check: "ocean", sectors: afloat.ocean_sectors, triangles: afloat.ocean_triangles, amplitude: tide.amplitude, depth, crossing: crossing.changed, buoy: buoy.translation[1], veiled: veiled.changed });
  }
  // Inside an interior's box (decision 079) the ground takes a tenth of the
  // sky and sun: looking straight down at the box's edge, the half inside
  // is far darker than the half outside.
  {
    const [edge] = (await command("terrain_height", { points: [[-280, 300]] })).result.samples;
    await command("camera", { position: [-280, edge.height + 60, 300], target: [-280, edge.height, 300.01], vertical_fov_radians: 0.6, near: 0.5, far: 2000 });
    const view = await capture("interior-edge");
    const [first, second] = await page.evaluate(async png => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      const { data, width, height } = context.getImageData(0, 0, image.width, image.height);
      const sums = [0, 0], counts = [0, 0];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          // Leave a band either side of the edge.
          if (Math.abs(x - width / 2) < width * 0.1) continue;
          const offset = (y * width + x) * 4, half = x < width / 2 ? 0 : 1;
          sums[half] += 0.2126 * data[offset] + 0.7152 * data[offset + 1] + 0.0722 * data[offset + 2];
          counts[half]++;
        }
      }
      return [sums[0] / counts[0], sums[1] / counts[1]];
    }, view);
    const [dark, light] = first < second ? [first, second] : [second, first];
    assert.ok(dark * 2 < light, `the interior darkens its ground: ${dark.toFixed(1)} against ${light.toFixed(1)}`);
    await command("camera", overview);
    checks.push({ check: "interior", inside: dark, outside: light });
  }
  // Edits (decision 080) stand over a terrain's layers and bakes: the
  // ground moves where one reaches once its tiles regenerate, its id
  // repeated leaves it alone, undo takes it away and redo puts it back.
  {
    const spot = [-200, 150];
    const ground = async () => (await heights([spot]))[0].height;
    const until = async (wanted, what) => {
      // Workers regenerate the tiles under it (decision 110).
      for (let frame = 0; frame < 400; frame++) {
        const height = await ground();
        if (Math.abs(height - wanted) < 0.05) return height;
        await command("render");
        await new Promise(resolve => setTimeout(resolve, 25));
      }
      assert.fail(`${what}: the ground at ${spot} stays at ${await ground()}, not ${wanted}`);
    };
    const authored = await ground();
    const mound = { id: "smoke-mound", center: spot, radius: 20, falloff: 10, change: { raise: 8 } };
    const raised = (await command("terrain_edit", mound)).result;
    assert.ok(raised.applied && raised.edits === 1 && Math.abs(raised.largest_change - 8) < 1e-3, `the edit applies: ${JSON.stringify(raised)}`);
    const risen = await until(authored + 8, "the edit raises");
    assert.equal((await command("terrain_edit", mound)).result.applied, false, "repeating an edit leaves it alone");
    const view = (await command("inspect", { entity: "ground", component: "Terrain" })).result.entity.components.Terrain;
    assert.deepEqual(view.edits.map(edit => edit.id), ["smoke-mound"], `the inspector lists the edits: ${JSON.stringify(view.edits)}`);
    assert.equal((await command("terrain_undo", {})).result.undone, "smoke-mound");
    await until(authored, "undo restores");
    assert.equal((await command("terrain_redo", {})).result.redone, "smoke-mound");
    await until(authored + 8, "redo raises again");
    const refused = await command("terrain_edit", { ...mound, id: "no spaces" }, false);
    assert.equal(refused.error.path, "$.id");
    assert.equal((await command("terrain_undo", {})).result.undone, "smoke-mound");
    await until(authored, "the fixture is left as authored");
    checks.push({ check: "edits", raised: risen - authored, change: raised.largest_change });
  }
  // Walks (decision 081) cross ground an agent stands on: around a hole
  // cut across the straight line, never into it, and longer than the
  // straight line; options out of range are refused.
  {
    const [start, end] = [[-300, -250], [-100, -250]];
    await command("terrain_edit", { id: "smoke-pit", center: [-200, -250], radius: 18, change: "hole" });
    const walked = (await command("nav_path", { from: start, to: end })).result.walk;
    assert.ok(walked && walked.points.length > 2, `a walk reaches: ${JSON.stringify(walked)?.slice(0, 300)}`);
    const straight = Math.hypot(end[0] - start[0], end[1] - start[1]);
    const clear = walked.points.every(([x, , z]) => Math.hypot(x + 200, z + 250) > 18);
    assert.ok(clear && walked.length > straight, `the walk goes around the hole: ${walked.length} m against ${straight} m`);
    const steep = await command("nav_path", { from: start, to: end, options: { max_slope: 0 } }, false);
    assert.equal(steep.error.path, "$.options.max_slope");
    assert.equal((await command("terrain_undo", {})).result.undone, "smoke-pit");
    checks.push({ check: "navigation", length: walked.length, straight, points: walked.points.length, searched: walked.searched });
  }
  // Navigation agents (decision 089): sent somewhere, an agent's walk is
  // searched over ticks, then it walks there on the ground and arrives; a
  // destination off the terrain is unreachable; settings out of range are
  // refused.
  {
    // North of the channel the hydrology bake found across the plateau.
    const goal = [-25, 12];
    const sent = await command("nav_go", { entity: "hiker", destination: goal });
    assert.deepEqual(sent.result.destination, goal);
    let agent;
    // Its walk is searched on a job worker beside the page (decision 110),
    // in the page's time, which paused steps outrun; or over ticks.
    for (let attempt = 0; attempt < 400; attempt++) {
      await command("step", { ticks: 1 });
      agent = (await command("inspect", { entity: "hiker", component: "NavAgent" })).result.entity.components.NavAgent;
      if (agent.status !== "searching") break;
      await new Promise(resolve => setTimeout(resolve, 25));
    }
    for (let round = 0; round < 30; round++) {
      await command("step", { ticks: 60 });
      agent = (await command("inspect", { entity: "hiker", component: "NavAgent" })).result.entity.components.NavAgent;
      if (agent.status === "arrived") break;
    }
    assert.equal(agent.status, "arrived", `the hiker arrives: ${JSON.stringify(agent)}`);
    assert.ok(agent.searched > 0 && agent.length > 40, `its walk was searched: ${JSON.stringify(agent)}`);
    const at = (await command("inspect", { entity: "hiker", component: "Transform" })).result.entity.components.Transform.translation;
    assert.ok(Math.hypot(at[0] - goal[0], at[2] - goal[1]) < 1.5, `it stands at its goal: ${at}`);
    const [ground] = (await command("terrain_height", { points: [[at[0], at[2]]] })).result.samples;
    assert.ok(Math.abs(at[1] - (ground.height + 0.5)) < 0.25, `on the ground: ${at[1]} over ${ground.height}`);
    await command("nav_go", { entity: "hiker", destination: [5000, 5000] });
    await command("step", { ticks: 2 });
    const lost = (await command("inspect", { entity: "hiker", component: "NavAgent" })).result.entity.components.NavAgent;
    assert.equal(lost.status, "unreachable", `off the terrain nothing reaches: ${JSON.stringify(lost)}`);
    const stats = (await command("inspect", { component: "NavAgents" })).result.value;
    assert.equal(stats.agents, 1, `the tick counted its agents: ${JSON.stringify(stats)}`);
    assert.equal((await command("environment", { navigation: { samples_per_tick: 10 } }, false)).error.path, "$.navigation.samples_per_tick");
    await command("nav_go", { entity: "hiker", destination: null });
    checks.push({ check: "agents", length: agent.length, searched: agent.searched, at });
  }
  // Authored objects (decision 085) stand where authors put them, on the
  // ground drawn there, and collide in their own layer.
  {
    const [ground] = await heights([[18, -12]]);
    const top = (await command("raycast", { origin: [18, ground.height + 50, -12], direction: [0, -1, 0], max_distance: 100 })).result.hit;
    // The box's middle stands on the ground, as the layers' crates' do: its
    // top half the scale above it.
    assert.ok(top?.placed?.layer === 65534 && Math.abs(top.point[1] - (ground.height + 1.5)) < 0.3,
      `the authored crate stands on the plateau: ${JSON.stringify(top)} over ${ground.height}`);
    const view = (await command("inspect", { entity: "ground", component: "Terrain" })).result.entity.components.Terrain;
    assert.equal(view.terrain.surface.placed.length, 2, "the inspector lists the authored objects");
    checks.push({ check: "authored placements", top: top.point[1], ground: ground.height });
  }
  // Tracks (decision 083) press into the ground as deep as its materials
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
    // The roller presses where it stands each tick.
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
  // Clouds (decision 084) are opt-in: the fixture has none until the
  // environment control adds a layer; exact frames march the whole dome,
  // which the sky shows; the view marches them too (decision 095), lying
  // over the ground from above the layer; no coverage marches nothing, and
  // settings out of range are refused.
  {
    const up = { position: [0, 60, 0], target: [300, 400, 200], vertical_fov_radians: 1.0, near: 0.5, far: 4000 };
    await command("camera", up);
    const clear = await capture("sky-clear");
    const layer = { enabled: true, coverage: 0.6, base: 600, thickness: 900, density: 1, wind: [8, 3], scale: 2000, seed: 3, shadows: 0.75, resolution: 1024, slices: 16 };
    const clouded = (await command("environment", { clouds: layer })).result.rendering;
    assert.equal(clouded.cloud_rows, 256, `exact frames march the whole dome: ${clouded.cloud_rows}`);
    const view = (await command("inspect", { component: "Clouds" })).result.value;
    assert.ok(Math.abs(view.coverage - 0.6) < 1e-6, `coverage reads back: ${view.coverage}`);
    const cloudy = await capture("sky-clouds");
    const cloudPixels = await differentPixels(page, [clear, cloudy]);
    assert.ok(cloudPixels.changed > 20000, `clouds show in the sky: ${cloudPixels.changed} pixels`);
    // Their shadows (decision 084) fall on the ground the sun lights.
    await command("camera", { position: [0, 400, 1], target: [0, 30, 0], vertical_fov_radians: 1.0, near: 0.5, far: 4000 });
    const shaded = await capture("cloud-shadows");
    await command("environment", { clouds: { ...layer, shadows: 0 } });
    const unshaded = await capture("no-cloud-shadows");
    const shadowPixels = await differentPixels(page, [shaded, unshaded]);
    assert.ok(shadowPixels.changed > 20000, `clouds shade the ground: ${shadowPixels.changed} pixels`);
    // From above the layer the view's clouds lie over the ground below,
    // which the dome cannot show; off, only the dome shows them.
    await command("camera", { position: [0, 1900, 1], target: [400, 0, 300], vertical_fov_radians: 1.0, near: 0.5, far: 8000 });
    const over = (await command("environment", { clouds: layer })).result.rendering;
    assert.ok(over.cloud_view_blocks > 0, `the view marches its clouds: ${over.cloud_view_blocks}`);
    const below = await capture("clouds-below");
    const domed = (await command("environment", { clouds: { ...layer, in_view: false } })).result.rendering;
    assert.equal(domed.cloud_view_blocks, 0, "off, the view marches none");
    const domeOnly = await capture("clouds-dome-only");
    const viewPixels = await differentPixels(page, [below, domeOnly]);
    assert.ok(viewPixels.changed > 20000, `clouds lie below the camera: ${viewPixels.changed} pixels`);
    const blockless = await command("environment", { clouds: { ...layer, view_block: 0 } }, false);
    assert.equal(blockless.error.path, "$.clouds.view_block");
    const none = (await command("environment", { clouds: { ...layer, coverage: 0 } })).result.rendering;
    assert.equal(none.cloud_rows, 0, "no coverage marches nothing");
    const odd = await command("environment", { clouds: { ...layer, resolution: 1000 } }, false);
    assert.equal(odd.error.path, "$.clouds.resolution");
    await command("environment", { clouds: { ...layer, enabled: false } });
    await command("camera", overview);
    checks.push({ check: "clouds", rows: clouded.cloud_rows, pixels: cloudPixels.changed, shadows: shadowPixels.changed, view_blocks: over.cloud_view_blocks, below: viewPixels.changed });
  }
  // The physically based sky (decision 088) is opt-in: its tables of the
  // air build once, the sky-view again for each step of the sun, which the
  // time of day moves; it replaces the authored sky, a dusk sky is redder
  // than noon's, and settings out of range are refused.
  {
    await command("camera", { position: [0, 60, 0], target: [300, 200, 200], vertical_fov_radians: 1.0, near: 0.5, far: 4000 });
    const authored = await capture("sky-authored");
    const first = (await command("environment", { sky: {}, time_of_day: { hours: 12 } })).result.rendering;
    assert.equal(first.sky_tables, 3, `the air's tables and the sky-view build: ${first.sky_tables}`);
    assert.equal((await command("render")).result.rendering.sky_tables, 0, "unchanged, nothing builds");
    const noon = await capture("sky-noon");
    const replaced = await differentPixels(page, [authored, noon]);
    assert.ok(replaced.changed > 20000, `the physical sky replaces the authored one: ${replaced.changed} pixels`);
    const step = (await command("environment", { time_of_day: { hours: 19.3 } })).result.rendering;
    assert.equal(step.sky_tables, 1, `the sun's step builds the sky-view alone: ${step.sky_tables}`);
    const dusk = await capture("sky-dusk");
    const [noonColor, duskColor] = [await skyColor(noon), await skyColor(dusk)];
    assert.ok(duskColor[0] / Math.max(duskColor[2], 1) > noonColor[0] / Math.max(noonColor[2], 1),
      `dusk is redder than noon: ${noonColor} ${duskColor}`);
    const view = (await command("inspect", { component: "Sky" })).result.value;
    assert.equal(view.enabled, true);
    assert.equal((await command("environment", { sky: { mie_g: 2 } }, false)).error.path, "$.sky.mie_g");
    assert.equal((await command("environment", { time_of_day: { hours: 25 } }, false)).error.path, "$.time_of_day.hours");
    const off = (await command("environment", { sky: { enabled: false }, time_of_day: { enabled: false } })).result.rendering;
    assert.equal(off.sky_tables, 0, "without the sky nothing builds");
    await command("camera", overview);
    checks.push({ check: "sky", noon: noonColor.map(Math.round), dusk: duskColor.map(Math.round), replaced: replaced.changed });
  }
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
    if (sky.terrain_sky_maps === 2 && sky.terrain_sky_rows === 0) break;
    if (sky.terrain_sky_maps < 2) await new Promise(resolve => setTimeout(resolve, 25));
  }
  // One for the terrain and one for its surroundings (decision 094).
  assert.equal(sky.terrain_sky_maps, 2, `the sky maps build: ${JSON.stringify(sky)}`);
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
  // The terrain's and its surroundings' (decision 094).
  assert.equal(rebuilt.terrain_horizons, 2);
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

  // Decision 120: while the view wants nothing, workers generate the
  // terrain a camera at a place a game names wants into memory, with no
  // idle reach around the camera, and the camera sent there finds it.
  const spawnPoint = [380, 90, 380];
  const named = structuredClone(fixture);
  named.entities.push({ id: "spawn-point", prefetch: {}, transform: { translation: spawnPoint } });
  await scene.reload(command, named);
  await command("environment", { terrain: { samples_per_frame: 50_000_000, idle_reach: 0 } });
  await command("camera", overview);
  await settle();
  const prefetch = (await command("inspect", { entity: "spawn-point", component: "TerrainPrefetch" })).result.entity.components.TerrainPrefetch;
  assert.equal(prefetch.reach, 2, `a place reaches as far as a camera loads: ${JSON.stringify(prefetch)}`);
  let stored = await inspectTerrains();
  for (let frame = 0; frame < 600 && !(stored.cached_tiles > 0 && stored.idle_tiles === 0); frame++) {
    await command("render");
    await new Promise(resolve => setTimeout(resolve, 25));
    stored = await inspectTerrains();
  }
  assert.ok(stored.cached_tiles > 0 && stored.idle_tiles === 0, `workers generate the place's tiles into memory: ${JSON.stringify(stored)}`);
  await command("camera", { ...overview, position: spawnPoint, target: [300, 30, 300] });
  await settle();
  const sent = await inspectTerrains();
  assert.ok(sent.restored_tiles > stored.restored_tiles, `the place's tiles return from memory: ${JSON.stringify(sent)}`);
  await command("camera", overview);
  await settle();
  checks.push({ check: "prefetch", cached: stored.cached_tiles, bytes: stored.cached_bytes, restored: sent.restored_tiles - stored.restored_tiles });

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

  // Decision 070: the import tool reads the dome as a RAW heightmap of its
  // own, centered on its middle without detail noise, and the scene it
  // writes loads with the dome's summit, its source and its attribution.
  execFileSync("cargo", ["run", "-q", "-p", "engine_assetc", "--release", "--bin", "terrain_import", "--",
    "--out", scenes, "--id", "terrain-smoke-import", "--size", "1024", "--extent", "128,128", "--range", "0,200",
    "--detail", "0", "--attribution", "smoke fixture", `${folder}dome.r16`], { stdio: "inherit" });
  const report = JSON.parse(await readFile(`${scenes}terrain-smoke-import/import.json`, "utf8"));
  assert.ok(report.samples === 513 && Math.abs(report.highest - 200) < 1e-6 && report.filled_gaps > 0, `the import resamples and fills: ${JSON.stringify(report)}`);
  // Reloads keep the scene's ID.
  const imported = { ...JSON.parse(await readFile(`${scenes}terrain-smoke-import.scene.json`, "utf8")), id: fixture.id };
  await scene.reload(command, imported);
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  await command("camera", overview);
  await settle();
  const [summit] = await heights([[0, 0]]);
  assert.ok(summit && Math.abs(summit.height - 200) < 0.01, `the imported summit stands 200 m high: ${JSON.stringify(summit)}`);
  const importedView = (await command("inspect", { entity: "ground", component: "Terrain" })).result.entity.components.Terrain;
  const importedMap = importedView.terrain.layers[0].heightmap;
  assert.ok(importedMap.attribution === "smoke fixture" && importedMap.source.includes("dome.r16"), `the layer keeps its source and credit: ${JSON.stringify(importedMap)}`);
  assert.equal(importedView.terrain.layers.length, 2, "without detail, the import and its rivers");
  checks.push({ check: "import", samples: report.samples, summit: summit.height, filled: report.filled_gaps });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await browser.close();
  await rm(folder, { recursive: true, force: true });
  await rm(source, { force: true });
  await rm(`${scenes}terrain-smoke-import`, { recursive: true, force: true });
  await rm(`${scenes}terrain-smoke-import.scene.json`, { force: true });
}
