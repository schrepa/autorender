/**
 * World streaming checks: a fixture's
 * streamed entities bin into grids by their reach and spawn only while
 * their cells are within range of the camera, with hysteresis, a per-frame
 * spawn budget, prefetch along the camera's motion, and freezing and
 * disabling through the `world_streaming` environment section; removals
 * persist and restore; grounded entities stand on the
 * terrain; loaded streamed entities inspect and collide by their authored
 * IDs; prefab placements, parents and animation players stream within
 * their item; reloads remove the old items; and invalid documents are
 * refused.
 */
// @ts-check
import assert from "node:assert/strict";
import { artifacts, commander, sceneUrl, launch, openPaused, routedScene } from "./harness.mjs";

const url = sceneUrl();
const output = await artifacts("streaming-smoke");
const cube = { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] } };
/** @type {any} */
const fixture = {
  id: "streaming-fixture", revision: 1,
  camera: { position: [0, 60, 40], target: [20, 20, 0], fov_degrees: 60, near: 0.5, far: 20000 },
  materials: [{ id: "plain", color: [0.8, 0.3, 0.2, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  animations: [{ id: "spin", targets: ["spinner-blade"], channels: [
    { target: 0, property: "rotation", times: [0, 1], values: [0, 0, 0, 1, 0, 0.7071068, 0, 0.7071068] },
  ] }],
  prefabs: [{ id: "lamp", entities: [
    { id: "post", mesh: "box", material: "plain", transform: { translation: [0, 1, 0], scale: [0.2, 2, 0.2] } },
    { id: "light", light: { color: [1, 0.7, 0.4], intensity: 10, range: 10 }, transform: { translation: [0, 2.2, 0] } },
  ] }],
  entities: [
    { id: "player", mesh: "box", material: "plain", tags: ["player"], collider: { ...cube, memberships: 2, filter: 1, sensor: false }, transform: { translation: [0, 200, 0] } },
    { id: "ground", terrain: { origin: [-1024, 0, -1024], tiles: [8, 8], spacing: 1, layers: [{ noise: { seed: 5, height: 40, wavelength: 400 } }] } },
  ],
  streamed: [
    // Small: 256 m cells loading within 512 m.
    { id: "crate-near", mesh: "box", material: "plain", collider: cube, grounded: { offset: 0.5 }, transform: { translation: [20, 0, 0] } },
    { id: "crate-mid", mesh: "box", material: "plain", grounded: { offset: 0.5 }, transform: { translation: [900, 0, 0] } },
    { id: "spinner", transform: { translation: [-30, 30, 10] }, animation: { clip: "spin" } },
    { id: "spinner-blade", parent: "spinner", mesh: "box", material: "plain", transform: { scale: [3, 0.2, 0.4] } },
    // Its light reaches ten meters: 1 km cells loading within 2 km.
    { id: "lamp-far", prefab: "lamp", transform: { translation: [1500, 0, 300] } },
    // Sixty meters tall: 4 km cells loading within 8 km.
    { id: "tower", mesh: "box", material: "plain", transform: { translation: [5000, 30, 0], scale: [10, 60, 10] } },
  ],
};
const { browser, page, errors } = await launch();
const command = commander(page);
const scene = routedScene(page, fixture);
const checks = [];

const cells = async () => (await command("inspect", { component: "WorldCells" })).result.value;
const rendered = async () => (await command("render")).result.rendering;
/** Moves the camera above `(x, z)`; returns the observation of the frame
 * the move rendered. */
async function look(x, z, height = 60) {
  return (await command("camera", { position: [x, height, z + 40], target: [x + 20, 20, z], vertical_fov_radians: Math.PI / 3, near: 0.5, far: 20000 })).result.rendering;
}
/** Changes the `world_streaming` section; returns the observation of the
 * frame it rendered. */
async function streaming(section) {
  return (await command("environment", { world_streaming: section })).result.rendering;
}
/** The Transform of streamed entity `id` while it is loaded, else null. */
async function loaded(id) {
  const reply = await page.evaluate(request => window.__engineDev.request(request), { op: "inspect", entity: id, component: "Transform" });
  if (!reply.ok) assert.equal(reply.error.code, "unknown_entity", JSON.stringify(reply.error));
  return reply.ok ? reply.result.entity.components.Transform : null;
}

try {
  await scene.route();
  await openPaused(page, url);
  await command("environment", { terrain: { samples_per_frame: 50_000_000 } });
  let rendering = await look(0, 0);
  // Job workers generate tiles beside the page, their
  // outputs arriving between commands.
  for (let frame = 0; frame < 400 && (rendering.terrain.loading_tiles > 0 || rendering.terrain.fallback_patches > 0); frame++) {
    await new Promise(resolve => setTimeout(resolve, 25));
    rendering = await rendered();
  }

  // Binning and what loads around the start.
  let state = await cells();
  assert.equal(state.items, 5, `items: ${JSON.stringify(state)}`);
  assert.deepEqual(state.grids.map(grid => [grid.cells, grid.loaded]), [[3, 2], [1, 1], [1, 1]], `grids: ${JSON.stringify(state.grids)}`);
  assert.equal(rendering.world_streaming.cells, 5);
  assert.equal(rendering.world_streaming.loaded, 4);
  assert.equal(rendering.world_streaming.pending, 0);
  // The crate, the spinner and its blade, the lamp's root, post and
  // light, and the tower.
  assert.equal(rendering.world_streaming.entities, 7, `streamed entities: ${JSON.stringify(rendering)}`);
  assert.equal(await loaded("crate-mid"), null, "the middle crate's cell is out of range");
  checks.push({ check: "binning", grids: state.grids, entities: rendering.world_streaming.entities });

  // Grounded: the crate's bottom on the terrain under it.
  const [ground] = (await command("terrain_height", { points: [[20, 0]] })).result.samples;
  const crate = await loaded("crate-near");
  assert.ok(Math.abs(crate.translation[1] - (ground.height + 0.5)) < 1e-3, `grounded crate: ${crate.translation[1]} over ${ground.height}`);
  const grounded = (await command("inspect", { entity: "crate-near", component: "Grounded" })).result.entity.components.Grounded;
  assert.equal(grounded.offset, 0.5);
  assert.equal(grounded.level, ground.level ?? 0);
  // It collides by its authored ID.
  const hit = (await command("raycast", { origin: [20, 300, 0], direction: [0, -1, 0], max_distance: 600 })).result.hit;
  assert.equal(hit?.entity?.id, "crate-near", `the ray hits the crate: ${JSON.stringify(hit)}`);
  assert.equal(hit.entity.in_active_scene, true);
  checks.push({ check: "grounded", height: crate.translation[1], ground: ground.height });

  // Prefab placements, parents and players stream within their item.
  const post = (await command("inspect", { entity: "lamp-far.post", component: "Hierarchy" })).result.entity.components.Hierarchy;
  assert.equal(post.parent, "lamp-far");
  const player = (await command("inspect", { entity: "spinner", component: "AnimationPlayer" })).result.entity.components.AnimationPlayer;
  assert.equal(player.clip.id, "spin");
  assert.deepEqual([player.clip.targets, player.clip.channels], [["spinner-blade"], 1]);
  assert.deepEqual(player.targets, ["spinner-blade"]);
  // The animation control sets a streamed player and answers before and after.
  const animated = (await command("animation", { entity: "spinner", time: 0.25, speed: 0 })).result;
  assert.equal(animated.before.clip, "spin");
  assert.deepEqual([animated.after.time, animated.after.speed], [0.25, 0]);
  const held = (await command("inspect", { entity: "spinner", component: "AnimationPlayer" })).result.entity.components.AnimationPlayer;
  assert.deepEqual([held.time, held.speed], [0.25, 0]);
  const unknownClip = await command("animation", { entity: "spinner", clip: "missing" }, false);
  assert.equal(unknownClip.error.code, "unknown_reference");
  checks.push({ check: "items" });

  // Moving east: the middle crate loads, the near cells unload past
  // 1.25 times their range, not before.
  rendering = await look(850, 0);
  assert.notEqual(await loaded("crate-near"), null, "within 1.25 times the range the near crate stays");
  assert.notEqual(await loaded("crate-mid"), null, "the middle crate loads");
  rendering = await look(1000, 0);
  assert.equal(await loaded("crate-near"), null, "past 1.25 times the range the near crate goes");
  assert.ok(rendering.world_streaming.despawned > 0, `the move despawned: ${JSON.stringify(rendering)}`);
  const missed = (await command("raycast", { origin: [20, 300, 0], direction: [0, -1, 0], max_distance: 600 })).result.hit;
  assert.notEqual(missed?.entity?.id, "crate-near", "an unloaded crate no longer collides");
  state = await cells();
  assert.deepEqual(state.grids.map(grid => grid.loaded), [1, 1, 1], `east: ${JSON.stringify(state.grids)}`);
  checks.push({ check: "hysteresis", loaded: state.loaded });

  // A budget of one entity a frame: a cell a frame.
  await streaming({ spawns_per_frame: 1 });
  rendering = await look(0, 0);
  assert.ok(rendering.world_streaming.spawned > 0, `a cell loads: ${JSON.stringify(rendering)}`);
  assert.equal(rendering.world_streaming.pending, 1, `the other waits: ${JSON.stringify(rendering)}`);
  rendering = await rendered();
  assert.equal(rendering.world_streaming.pending, 0);
  assert.equal(rendering.world_streaming.loaded, 4);
  checks.push({ check: "budget" });

  // Frozen, nothing moves in or out; disabled, everything goes.
  await streaming({ frozen: true });
  rendering = await look(3000, 3000);
  assert.equal(rendering.world_streaming.loaded, 4, "frozen cells stay");
  rendering = await streaming({ frozen: false });
  assert.equal(rendering.world_streaming.loaded, 2, `unfrozen, the small cells go: ${JSON.stringify(rendering)}`);
  rendering = await look(0, 0);
  assert.equal(rendering.world_streaming.loaded, 4);
  rendering = await streaming({ enabled: false });
  assert.equal(rendering.world_streaming.loaded, 0);
  assert.equal(rendering.world_streaming.entities, 0);
  rendering = await streaming({ enabled: true });
  assert.equal(rendering.world_streaming.loaded, 4);
  checks.push({ check: "frozen and disabled" });

  // Prefetch: moving east at 600 m/s, the cells 900 m ahead load early.
  await look(0, 0);
  await command("step", { ticks: 1 });
  await command("camera", { position: [10, 60, 40], target: [30, 20, 0], vertical_fov_radians: Math.PI / 3, near: 0.5, far: 20000 });
  await command("step", { ticks: 1 });
  state = await cells();
  assert.ok(state.velocity[0] > 100, `the camera's motion is measured: ${state.velocity}`);
  assert.notEqual(await loaded("crate-mid"), null, "the middle crate loads ahead of the camera");
  checks.push({ check: "prefetch", velocity: state.velocity });

  // Removals persist: a removed item despawns at once and
  // its cell no longer spawns it; restored, it spawns again.
  await look(0, 0);
  const removal = await command("world_changes", { remove: ["streaming-fixture/crate-near"] });
  assert.equal(removal.result.newly_removed, 1);
  assert.equal(await loaded("crate-near"), null, "a removed item despawns at once");
  const changes = (await command("inspect", { component: "WorldChanges" })).result.value;
  assert.deepEqual(changes, { removed: 1, ids: ["streaming-fixture/crate-near"] });
  await look(1000, 0);
  await look(0, 0);
  assert.equal(await loaded("crate-near"), null, "its cell no longer spawns it");
  const marked = (await command("inspect", { entity: "spinner", component: "StreamedId" })).result.entity.components.StreamedId;
  assert.equal(marked.id, "streaming-fixture/spinner", "an item's root carries its ID");
  await command("world_changes", { restore: ["streaming-fixture/crate-near"] });
  assert.notEqual(await loaded("crate-near"), null, "restored, it spawns again");
  const empty = await command("world_changes", { remove: [""] }, false);
  assert.equal(empty.error.path, "$.remove[0]");
  checks.push({ check: "removals" });

  // Settings are validated.
  const refused = await command("environment", { world_streaming: { unload: 0.5 } }, false);
  assert.equal(refused.error.path, "$.world_streaming.unload");

  // A reload removes the old instance's items.
  await scene.reload(command, { ...fixture, streamed: fixture.streamed.slice(0, 1) });
  state = await cells();
  assert.equal(state.items, 1, `after reload: ${JSON.stringify(state)}`);
  checks.push({ check: "reload", items: state.items });

  // Refused documents keep the current scene.
  for (const [name, change, code, path] of /** @type {[string, (document: any) => void, string, string][]} */ ([
    ["terrain", document => { document.streamed.push({ id: "land", terrain: fixture.entities[1].terrain }); }, "invalid_streamed", "$.streamed[6].terrain"],
    ["parent across lists", document => { document.streamed[5].parent = "player"; }, "unknown_parent", "$.streamed[5].parent"],
    ["grounded child", document => { document.streamed[3].grounded = {}; }, "invalid_grounded", "$.streamed[3].grounded"],
    ["duplicate ID", document => { document.streamed[0].id = "player"; }, "duplicate_id", "$.streamed[0].id"],
    ["target outside the item", document => { document.streamed[2].animation = { clip: "spin" }; document.streamed[3].parent = undefined; }, "unknown_reference", "$.streamed[2].animation.clip"],
  ])) {
    const broken = structuredClone(fixture);
    change(broken);
    broken.revision = scene.revision + 1;
    scene.serve(broken);
    const failed = await command("reload", { expected_revision: String(scene.revision), revision: String(broken.revision) }, false);
    assert.equal(failed.error.code, code, `${name}: ${JSON.stringify(failed.error)}`);
    assert.equal(failed.error.path, path, `${name}: ${JSON.stringify(failed.error)}`);
  }
  checks.push({ check: "refusals" });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, output, checks }, null, 2));
} finally {
  await browser.close();
}
