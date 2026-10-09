/** Real Chrome collision contracts; fixtures are routed responses, never source edits. */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { artifacts, commander, devUrl, developmentReady, launch, openPaused, routedScene } from "./harness.mjs";

const url = devUrl("playground.html", process.env.ENGINE_BUILD ? { engine_build: process.env.ENGINE_BUILD } : {});
const output = await artifacts("collision-smoke");
const cube = half_extents => ({ kind: "cuboid", half_extents });
const collider = (shape, memberships = 1, filter = 2, sensor = false) => ({ shape, memberships, filter, sensor });
const filter = { memberships: 2, filter: 1, sensors: "exclude" };
const fixture = {
  version: 4, id: "collision-fixture", revision: 1,
  camera: { position: [12, 10, 16], target: [0, 0, 0] },
  meshes: [{ id: "box", source: { kind: "cube" } }],
  materials: [{ id: "blue", color: [0.1, 0.6, 0.85, 1] }, { id: "gold", color: [0.9, 0.65, 0.1, 1] }],
  entities: [
    { id: "player", tags: ["player"], mesh: "box", material: "blue", transform: { translation: [0, 0.6, 0], scale: [1.2, 1.2, 1.2] }, collider: collider(cube([0.5, 0.5, 0.5]), 2, 5) },
    { id: "floor", transform: { translation: [0, -0.5, 0] }, collider: collider(cube([20, 0.5, 20])) },
    { id: "target", mesh: "box", material: "gold", transform: { translation: [4, 0.5, 0] }, collider: collider(cube([0.5, 0.5, 0.5])) },
    { id: "sensor", transform: { translation: [0, 0.6, -2] }, collider: collider({ kind: "sphere", radius: 0.4 }, 4, 2, true) },
    { id: "parent", transform: { translation: [8, 1, 0], rotation_degrees: [0, 30, 0], scale: [2, 1, 0.5] } },
    { id: "sheared", parent: "parent", mesh: "box", material: "gold", transform: { rotation_degrees: [0, 40, 0] }, collider: collider(cube([0.5, 0.5, 0.5])) },
    { id: "ellipsoid", transform: { translation: [-4, 1, 0], scale: [2, 1, 0.5] }, collider: collider({ kind: "sphere", radius: 0.5 }) },
  ],
};
const { browser, page, errors } = await launch({ timeout: 20_000 });
const checks = [];
const command = commander(page, { deadline: 10_000 });
const scene = routedScene(page, fixture);
async function inspect(entity, component) {
  return (await command("inspect", { entity, component })).result.entity.components[component];
}
function near(value, expected, epsilon = 0.001) {
  assert.ok(Number.isFinite(value) && Math.abs(value - expected) < epsilon, `${value} != ${expected}`);
}
async function state() {
  const observation = (await command("status")).observation;
  return { frame: observation.frame, tick: observation.tick, presentation: observation.presentation,
    transform: await inspect("player", "Transform"), presented: await inspect("player", "PresentedTransform") };
}
async function sensorEvents() {
  const { observation, rendering, paused, source, ...snapshot } = (await command("collision_events", { entity: "sensor" })).result;
  return snapshot;
}

try {
  await scene.route();
  await openPaused(page, url, { timeout: 20_000 });
  await command("render");
  await command("describe", { name: "Collider" });
  await command("describe", { name: "CollisionWorld" });
  const volume = await inspect("player", "Collider");
  assert.ok(volume);
  const before = await state();
  const ray = { origin: [4, 4, 0], direction: [0, -7, 0], max_distance: 5, filter };
  const hit = (await command("raycast", ray)).result.hit;
  assert.equal(hit.entity.id, "target"); near(hit.distance, 3);
  near(hit.normal[1], 1);
  assert.equal((await command("raycast", { ...ray, filter: { memberships: 8, filter: 1 } })).result.hit, null);
  const sensorRay = { origin: [0, 0.6, 0], direction: [0, 0, -1], max_distance: 3,
    filter: { memberships: 2, filter: 4, sensors: "only" } };
  assert.equal((await command("raycast", sensorRay)).result.hit.entity.id, "sensor");
  assert.equal((await command("raycast", { ...sensorRay, filter: { ...sensorRay.filter, sensors: "exclude" } })).result.hit, null);
  assert.equal((await command("raycast", { ...sensorRay, filter: { ...sensorRay.filter, exclude: "sensor" } })).result.hit, null);
  const inside = (await command("raycast", { ...ray, origin: [4, 0.5, 0] })).result.hit;
  assert.equal(inside.started_inside, true); near(inside.distance, 0);
  const overlap = (await command("overlap", { position: [4, 0.5, 0], shape: { kind: "sphere", radius: 0.2 }, filter })).result;
  assert.equal(overlap.hits.length, 1); assert.equal(overlap.hits[0].id, "target");
  const cast = (await command("shape_cast", { position: [4, 4, 0], shape: cube([0.25, 0.25, 0.25]), direction: [0, -1, 0], max_distance: 5, filter })).result.hit;
  assert.equal(cast.entity.id, "target"); near(cast.distance, 2.75);
  for (const [id, x] of [["sheared", 8], ["ellipsoid", -4]]) {
    const result = (await command("raycast", { ...ray, origin: [x, 5, 0], max_distance: 6 })).result.hit;
    assert.equal(result.entity.id, id); near(result.distance, 3.5);
  }
  const shearedCast = (await command("shape_cast", { position: [8, 5, 0], shape: cube([0.25, 0.25, 0.25]), direction: [0, -1, 0], max_distance: 5, filter })).result.hit;
  assert.equal(shearedCast.entity.id, "sheared"); near(shearedCast.distance, 3.25);
  await command("raycast", { ...ray, direction: [0, 0, 0] }, false);
  await command("overlap", { position: [0, 0, 0], shape: { kind: "sphere", radius: 20 }, filter, limit: 1 }, false);
  assert.deepEqual(await state(), before);
  checks.push("closest_raycast", "inside_hit", "symmetric_layers", "sensor_filter_and_exclusion", "overlap", "shape_cast", "shear_and_ellipsoid", "invalid_and_overflow_nonmutating");

  await command("input", { forward: 1 });
  const transitions = [];
  for (let tick = 0; tick < 36; tick++) {
    await command("step", { ticks: 1 });
    const events = await sensorEvents();
    for (const event of events.events) transitions.push({ tick, event });
    if (events.events.length) {
      const exact = await state();
      await command("raycast", ray);
      assert.deepEqual(await sensorEvents(), events);
      assert.deepEqual(await state(), exact);
      await command("present", { alpha: 0.5 });
      const preview = await state();
      await command("raycast", ray);
      assert.deepEqual(await state(), preview);
      assert.deepEqual(await sensorEvents(), events);
    }
  }
  assert.deepEqual(transitions.map(value => value.event.kind), ["enter", "exit"]);
  const moved = await inspect("player", "Transform");
  near(moved.translation[1], 0.6); near(moved.translation[2], -3.9);
  checks.push("sensor_enter_exit", "events_nonconsuming", "queries_preserve_preview_and_clock", "ground_tangent_movement");

  const blocked = structuredClone(fixture);
  blocked.entities.push({ id: "thin-wall", transform: { translation: [0.66, 0.6, 0], rotation_degrees: [0, 90, 0] }, collider: collider(cube([3, 2, 0.01])) });
  await scene.reload(command, blocked);
  await command("input", { right: 1 });
  await command("step", { ticks: 30 });
  near((await inspect("player", "Transform")).translation[0], 0.05);
  const progress = (await command("inspect", { component: "GameProgress" })).result;
  assert.ok(JSON.stringify(progress).includes('"collision_errors":0'));
  checks.push("swept_thin_wall", "no_game_query_errors");

  const invalidBefore = await state();
  const flat = structuredClone(fixture); flat.revision = scene.revision + 1;
  flat.entities.find(entity => entity.id === "target").collider.shape.half_extents[0] = 0;
  scene.serve(flat);
  await command("reload", { expected_revision: String(scene.revision), revision: String(flat.revision) }, false);
  assert.deepEqual(await state(), invalidBefore);
  const removed = structuredClone(fixture);
  removed.entities = removed.entities.filter(entity => entity.id !== "target");
  await scene.reload(command, removed);
  assert.equal((await command("raycast", { ...ray, max_distance: 3.5 })).result.hit, null);
  checks.push("invalid_reload_preserves_world", "removed_colliders_retire");
  await command("capture");
  await page.locator("canvas").screenshot({ path: `${output}/collision-fixture.png` });
  await command("resume");
  await command("raycast", ray);
  await command("pause");
  checks.push("queries_allowed_while_running");
  const orbit = new URL(url); orbit.pathname = "/";
  await page.goto(orbit.href);
  await developmentReady(page);
  const missing = await command("raycast", ray, false);
  assert.match(missing.error.code, /missing_plugin/);
  checks.push("plugin_is_opt_in");
  assert.deepEqual(errors, []);
  const report = { ok: true, browser: browser.version(), url: url.href, artifact: output, checks, transitions };
  await writeFile(`${output}/report.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await browser.close(); }
