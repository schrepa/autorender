/**
 * The terrain fixture streams, draws, answers queries and collides.
 *
 * Its tiles stream and draw with their placed objects, impostors and
 * shadows; it inspects as authored; its surroundings fill the view past its
 * edges; its erosion bake, roads, bridge, tunnel and routes read back;
 * height queries, a tile budget, collision against rendering, the settings
 * section, speed-aware detail, tile memory and banded placing hold; the
 * terrain control's views, the sky visibility map and freezing work; edits,
 * walks and navigation agents cross it and authored objects stand on it;
 * and a place a game names prefetches its tiles. The fixture and what the
 * terrain smokes share: terrain-fixture.mjs.
 */
import assert from "node:assert/strict";
import { differentPixels } from "./harness.mjs";
import { fast, fixture, openTerrain, overview, terrain } from "./terrain-fixture.mjs";

const { page, errors, command, scene, settled, close, settle, capture, heights, inspectTerrains } = await openTerrain("terrain");
const checks = [];

try {
  assert.ok(settled.terrain.patches > 0 && settled.terrain.triangles > 0, "terrain draws");
  assert.ok(settled.terrain.shadow_patches > 0, "terrain casts shadows");
  assert.equal(settled.terrain.dropped_tiles, 0);
  assert.equal(settled.terrain.dropped_materials, 0);
  assert.ok(settled.terrain.placed_objects > 0, `crates stand on the meadow: ${settled.terrain.placed_objects}`);
  // The pinned tiles stand a share of them everywhere.
  assert.ok(settled.terrain.placed_far_objects > 0 && settled.terrain.placed_far_objects < settled.terrain.placed_objects,
    `far crates: ${settled.terrain.placed_far_objects} of ${settled.terrain.placed_objects}`);
  // GPU-presented frames bake the balls' impostor, which far
  // balls draw and cast as cards.
  let presented;
  for (let frame = 0; frame < 60; frame++) {
    presented = (await command("present", { alpha: 0, gpu: true })).result.rendering;
    if (presented.terrain.placed_impostors > 0) break;
  }
  assert.equal(presented.terrain.placed_impostors, 1, `the balls' impostor bakes: ${JSON.stringify(presented)}`);
  assert.equal(presented.level_of_detail.dropped_impostors, 0);
  checks.push({ check: "streams and draws", tiles: settled.terrain.tiles, patches: settled.terrain.patches, triangles: settled.terrain.triangles, placed_impostors: presented.terrain.placed_impostors, placed_far: settled.terrain.placed_far_objects });

  // Discovery and inspection.
  const listing = await command("describe");
  for (const name of ["Terrain", "Terrains", "TerrainSettings"]) {
    assert.ok(listing.result.types.some(type => type.name === name), `describe lists ${name}`);
  }
  const look = await command("inspect", { component: "Environment" });
  assert.equal(look.result.value.terrain.samples_per_frame, fast.samples_per_frame);
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
  // Relief: what a material sinks between its stones reads
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
  // Surroundings: ground generated past the terrain's edges
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
    const edged = await command("environment", { terrain: { ...fast, surroundings: false } });
    assert.deepEqual(edged.result.environment_changed, ["terrain"]);
    const alone = await settle();
    assert.equal((await command("inspect", { component: "Terrains" })).result.value.surroundings_patches, 0, "settings turn the surroundings off");
    assert.equal(alone.terrain.placed_objects, surroundedFrame.terrain.placed_objects, "the surroundings place nothing");
    const bare = await capture("no-surroundings");
    const filled = await differentPixels(page, [surrounded, bare]);
    assert.ok(filled.changed > 0.2 * filled.total, `the surroundings fill ${filled.changed} of ${filled.total} pixels`);
    await command("environment", { terrain: fast });
    await command("camera", overview);
    await settle();
    checks.push({ check: "surroundings", reach: around.reach, tiles: status.surroundings_tiles, patches: status.surroundings_patches, pixels: filled.changed });
  }
  // The erosion bake ran its rounds over the same samples
  // first, wearing ground away.
  const worn = view.erosion;
  assert.equal(worn.baking, false, "erosion is done once tiles load");
  assert.equal(worn.samples, 513 * 513, `erosion reads its level: ${JSON.stringify(worn)}`);
  assert.ok(worn.rounds === 6 && worn.deepest > 0, `erosion wears ground away: ${JSON.stringify(worn)}`);
  checks.push({ check: "erosion", rounds: worn.rounds, deepest: worn.deepest, highest: worn.highest });
  // The path's bake profiled the ground along its line; the
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
    // Bridges carry the road over the hills' streams.
    if (center.path === null) continue;
    assert.ok(Math.abs(center.path + 3) < 0.05, `the centerline lies 3 m inside the road's edges: ${JSON.stringify(center)}`);
    tilt = Math.max(tilt, Math.abs(left.height - right.height));
    level += 1;
  }
  assert.ok(level >= 3 && tilt < 0.15, `the road lies level across: ${tilt} m over 4 m at ${level} sections`);
  const [away] = await heights([[400, -400]]);
  assert.equal(away.path, null, "far from roads, queries report none");
  checks.push({ check: "path", length: road.length, samples: road.samples, steepest: road.steepest, cut: road.deepest_cut, fill: road.highest_fill, tilt });
  // Where the road would fill over the trench it bridges:
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
  // The straight road bores through the mound: no road on the
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
  // Routes keep within their grade between their ends.
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
  assert.equal(unrouted.error.code, "out_of_range");
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
  const budget = await command("environment", { terrain: { ...fast, max_tiles: 3 } });
  assert.ok(budget.result.rendering.terrain.tiles <= 3, `${budget.result.rendering.terrain.tiles} tiles under a budget of 3`);
  const budgeted = (await command("render")).result.rendering;
  assert.ok(budgeted.terrain.tiles <= 3 && budgeted.terrain.dropped_tiles > 0, JSON.stringify(budgeted));
  await command("environment", { terrain: fast });
  await settle();
  checks.push({ check: "tile budget", resident: budgeted.terrain.tiles, dropped: budgeted.terrain.dropped_tiles });

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
  // Placed crates and balls collide: some rays meet them
  // above the ground, which still compares beneath them.
  assert.ok(ray.result.snapshot.placed_colliders > 0, `placed colliders: ${ray.result.snapshot.placed_colliders}`);
  assert.ok(sweep.collision.placed_first > 0, `rays meeting placed objects first: ${sweep.collision.placed_first}`);
  assert.ok(sweep.layers.compared > 100);
  assert.ok(sweep.layers.worst.max_difference < 5e-3, `layers differ by ${sweep.layers.worst.max_difference}`);
  checks.push({ check: "render versus collision", compared: sweep.collision.compared, worst: sweep.collision.worst.max_difference, layers: sweep.layers.worst.max_difference, placed_colliders: ray.result.snapshot.placed_colliders, placed_first: sweep.collision.placed_first });

  // The settings section: finer level of detail reaches further and draws
  // more triangles; invalid values are refused; disabled terrain draws
  // nothing.
  const fine = await command("environment", { terrain: { ...fast, pixel_error: 2 } });
  const finer = (await settle());
  const detailed = (await command("inspect", { entity: "ground", component: "Terrain" })).result.entity.components.Terrain;
  assert.ok(detailed.level_ranges[0] > view.level_ranges[0], `level 0 reaches ${detailed.level_ranges[0]} m`);
  assert.ok(finer.terrain.triangles > settled.terrain.triangles, "a smaller pixel error draws more triangles");
  assert.deepEqual(fine.result.environment_changed, ["terrain"]);
  const invalid = await command("environment", { terrain: { pixel_error: 0.5 } }, false);
  assert.equal(invalid.error.code, "invalid_environment");
  assert.equal(invalid.error.path, "$.terrain.pixel_error");
  const shaded = await capture("shaded");
  const off = await command("environment", { terrain: { enabled: false } });
  assert.equal(off.result.rendering.terrain.patches, 0);
  const empty = await capture("disabled");
  const shown = await differentPixels(page, [shaded, empty]);
  assert.ok(shown.changed > 10_000, `terrain covers ${shown.changed} pixels`);
  await command("environment", { terrain: fast });
  await settle();
  checks.push({ check: "settings section", more: finer.terrain.triangles, terrainPixels: shown.changed });

  // Speed-aware detail: while the camera moves fast the
  // finest levels' tiles in view wait, and those out of view until it
  // settles; then they load. Collision's tiles never wait, so collision is
  // off here.
  const moving = { ...fast, collision_radius: 0, motion_detail: { speed: 1, fastest: 2, levels: 8, settle: 0.25 } };
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
  await command("environment", { terrain: fast });
  await command("camera", overview);
  await settle();
  checks.push({ check: "motion detail", waited });

  // Tiles the view leaves stay in memory and return at once
  // when it comes back; with nothing missing, workers generate the tiles
  // within the idle reach into memory.
  // Memory starts empty and nothing is made ahead of the view until the
  // camera is back, so the tiles under the view after the move are made
  // then, however far this machine's workers got before (the motion checks'
  // camera passed the corner).
  await command("environment", { terrain: { ...fast, cache_bytes: 1, idle_reach: 0 } });
  await command("render");
  await command("environment", { terrain: { ...fast, idle_reach: 0 } });
  const home = await inspectTerrains();
  await command("camera", { ...overview, position: [-420, 90, -420], target: [-300, 30, -300] });
  await settle();
  // The few tiles under the view after the move generated in
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
  await command("environment", { terrain: { ...fast, cache_bytes: 1 } });
  await command("render");
  const emptied = await inspectTerrains();
  assert.equal(emptied.cached_tiles, 0, `a byte keeps no tile: ${JSON.stringify(emptied)}`);
  await command("environment", { terrain: { ...fast, idle_reach: 4 } });
  let ahead = emptied;
  let generating = 0;
  for (let frame = 0; frame < 300 && ahead.cached_tiles === 0; frame++) {
    await command("render");
    await new Promise(resolve => setTimeout(resolve, 25));
    ahead = await inspectTerrains();
    generating = Math.max(generating, ahead.idle_tiles);
  }
  assert.ok(ahead.cached_tiles > 0, `workers generate tiles ahead of the view: ${JSON.stringify(ahead)}`);
  await command("environment", { terrain: fast });
  await settle();
  checks.push({ check: "memory", restored: back.restored_tiles, banded: corner.banded_tiles, cached: ahead.cached_tiles, bytes: ahead.cached_bytes, generating });

  // Tiles generated in bands place their objects in bands
  // too, the same objects as whole tiles place: the view of tiles banded
  // draws as many as of the same tiles generated whole.
  const cornerView = { ...overview, position: [-420, 90, -420], target: [-300, 30, -300] };
  await command("camera", cornerView);
  await settle();
  const bandedObjects = (await command("render")).result.rendering.terrain.placed_objects;
  await command("environment", { terrain: { ...fast, max_bands: 1, cache_bytes: 1 } });
  await command("camera", overview);
  await settle();
  await command("camera", cornerView);
  await settle();
  const wholeObjects = (await command("render")).result.rendering.terrain.placed_objects;
  assert.ok(bandedObjects > 0, "objects stand on the banded tiles");
  assert.equal(wholeObjects, bandedObjects, "banded and whole tiles place the same objects");
  await command("environment", { terrain: fast });
  await command("camera", overview);
  await settle();
  checks.push({ check: "placing bands", objects: bandedObjects });

  // The terrain control: development views change pixels, freezing keeps
  // the patches while the camera moves away.
  const baseline = await capture("view-shaded");
  const levels = await command("terrain", { view: "levels" });
  assert.equal(levels.result.view, "levels");
  // The control's state inspects.
  const debug = (await command("inspect", { component: "TerrainDebug" })).result.value;
  assert.deepEqual(debug, { freeze: null, view: "levels" }, `the terrain debug state: ${JSON.stringify(debug)}`);
  const levelView = await capture("view-levels");
  assert.ok((await differentPixels(page, [baseline, levelView])).changed > 10_000, "the levels view differs");
  await command("terrain", { view: "tiles" });
  await capture("view-tiles");
  // Each surface material in its own color.
  await command("terrain", { view: "materials" });
  const materialView = await capture("view-materials");
  assert.ok((await differentPixels(page, [baseline, materialView])).changed > 10_000, "the materials view differs");
  // The sky visibility map builds a band of rows a frame
  // once the pinned level is resident; its view shows it unlit.
  let sky;
  for (let frame = 0; frame < 200; frame++) {
    sky = (await command("render")).result.rendering;
    if (sky.terrain.sky_maps === 2 && sky.terrain.sky_rows === 0) break;
    if (sky.terrain.sky_maps < 2) await new Promise(resolve => setTimeout(resolve, 25));
  }
  // One for the terrain and one for its surroundings.
  assert.equal(sky.terrain.sky_maps, 2, `the sky maps build: ${JSON.stringify(sky)}`);
  await command("terrain", { view: "sky" });
  const skyView = await capture("view-sky");
  assert.ok((await differentPixels(page, [baseline, skyView])).changed > 10_000, "the sky view differs");
  checks.push({ check: "sky visibility", maps: sky.terrain.sky_maps });
  await command("terrain", { view: "shaded" });
  const frozen = await command("terrain", { freeze: true });
  assert.ok(Array.isArray(frozen.result.frozen_at));
  assert.deepEqual((await command("inspect", { component: "TerrainDebug" })).result.value.freeze, frozen.result.frozen_at);
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
  checks.push({ check: "terrain control", frozenPatches: before, visibleFromAfar: afar.terrain.patches, unfrozenPatches: moved });

  // Edits stand over a terrain's layers and bakes: the
  // ground moves where one reaches once its tiles regenerate, its id
  // repeated leaves it alone, undo takes it away and redo puts it back.
  {
    const spot = [-200, 150];
    const ground = async () => (await heights([spot]))[0].height;
    const until = async (wanted, what) => {
      // Workers regenerate the tiles under it.
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
  // Walks cross ground an agent stands on: around a hole
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
  // Navigation agents: sent somewhere, an agent's walk is
  // searched over ticks, then it walks there on the ground and arrives; a
  // destination off the terrain is unreachable; settings out of range are
  // refused.
  {
    // North of the channel the hydrology bake found across the plateau.
    const goal = [-25, 12];
    const sent = await command("nav_go", { entity: "hiker", destination: goal });
    assert.deepEqual(sent.result.destination, goal);
    let agent;
    // Its walk is searched on a job worker beside the page,
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
  // Authored objects stand where authors put them, on the
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

  // While the view wants nothing, workers generate the
  // terrain a camera at a place a game names wants into memory, with no
  // idle reach around the camera, and the camera sent there finds it.
  const spawnPoint = [380, 90, 380];
  const named = structuredClone(fixture);
  named.entities.push({ id: "spawn-point", prefetch: {}, transform: { translation: spawnPoint } });
  await scene.reload(command, named);
  await command("environment", { terrain: { ...fast, idle_reach: 0 } });
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

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await close();
}
