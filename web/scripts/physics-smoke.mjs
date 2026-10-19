/** Rigid bodies in real Chrome: a fixture's crate falls onto a floor, its raft floats and its stone sinks in the sea. Fixtures are routed responses. */
// @ts-check
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { artifacts, commander, devUrl, developmentReady, launch, openPaused, routedScene, sceneUrl } from "./harness.mjs";

const url = sceneUrl(process.env.ENGINE_BUILD ? { engine_build: process.env.ENGINE_BUILD } : {});
const output = await artifacts("physics-smoke");
const box = { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] };
/** @type {any} */
const fixture = {
  id: "physics-fixture", revision: 1,
  camera: { position: [26, 9, 22], target: [20, 2, 0] },
  meshes: [{ id: "box", source: { kind: "cube" } }],
  materials: [{ id: "wood", color: [0.55, 0.38, 0.2, 1] }, { id: "stone", color: [0.5, 0.5, 0.52, 1] }],
  environment: { ocean: { enabled: true, level: 0 }, physics: { water_density: 1000 } },
  entities: [
    { id: "floor", transform: { translation: [0, 9.5, 0] }, collider: { shape: { kind: "cuboid", half_extents: [10, 0.5, 10] } } },
    { id: "crate", mesh: "box", material: "wood", transform: { translation: [0, 14, 0] }, collider: { shape: box }, rigid_body: {} },
    { id: "seabed", transform: { translation: [30, -12.5, 0] }, collider: { shape: { kind: "cuboid", half_extents: [20, 0.5, 20] } } },
    { id: "raft", mesh: "box", material: "wood", transform: { translation: [26, 2, 0], scale: [2, 0.5, 2] }, collider: { shape: box }, rigid_body: { density: 500 } },
    { id: "stone", mesh: "box", material: "stone", transform: { translation: [32, 2, 0] }, collider: { shape: { kind: "sphere", radius: 0.5 } }, rigid_body: { density: 2600 } },
  ],
};
const { browser, page, errors } = await launch({ timeout: 20_000 });
const checks = [];
const command = commander(page, { deadline: 10_000 });
const scene = routedScene(page, fixture);
async function inspect(entity, component) {
  return (await command("inspect", { entity, component })).result.entity.components[component];
}
async function seconds(count) {
  let reply;
  for (let tick = 0; tick < count * 60; tick += 120) reply = await command("step", { ticks: 120 });
  return reply;
}

try {
  await scene.route();
  await openPaused(page, url, { timeout: 20_000 });
  await command("render");
  for (const name of ["RigidBody", "PhysicsWorld", "Physics"]) await command("describe", { name });
  // The settings start at their defaults: the bakes' gravity.
  const settings = (await command("inspect", { component: "Physics" })).result.value;
  assert.ok(settings.enabled && Math.abs(settings.gravity[1] + 9.81) < 1e-4, JSON.stringify(settings));
  checks.push("settings_inspect");
  // Three seconds: the crate lands on the floor and rests there, the raft
  // floats half under (half water's density), the stone sinks.
  const landed = await seconds(3);
  const crate = await inspect("crate", "Transform");
  assert.ok(Math.abs(crate.translation[1] - 10.5) < 0.05, `the crate rests on the floor: ${JSON.stringify(crate)}`);
  const body = await inspect("crate", "RigidBody");
  assert.ok(body.simulated && body.problem === null && Math.abs(body.mass - 600) < 1, `the crate's body: ${JSON.stringify(body)}`);
  const raft = await inspect("raft", "RigidBody");
  assert.equal(raft.water?.kind, "sea", JSON.stringify(raft));
  assert.ok(raft.water.submerged > 0.3 && raft.water.submerged < 0.75, `the raft floats half under: ${JSON.stringify(raft.water)}`);
  const stone = await inspect("stone", "Transform");
  assert.ok(stone.translation[1] < -2, `the stone sinks: ${JSON.stringify(stone)}`);
  const world = (await command("inspect", { component: "PhysicsWorld" })).result.value;
  assert.ok(world.bodies === 3 && world.wet >= 2 && world.held === 0 && world.solids >= 2 && world.water_samples > 0, JSON.stringify(world));
  assert.equal(landed.result.rendering.physics.bodies, 3, "frames observe physics");
  checks.push("falls_and_rests", "floats_by_density", "sinks", "world_inspects", "frames_observe");
  // Off, bodies hold where they stand; on again, they move on.
  await command("environment", { merge: true, physics: { enabled: false } });
  const held = await inspect("stone", "Transform");
  await seconds(1);
  assert.deepEqual(await inspect("stone", "Transform"), held, "physics off holds the stone");
  await command("environment", { merge: true, physics: { enabled: true } });
  await seconds(1);
  assert.notDeepEqual(await inspect("stone", "Transform"), held, "physics on moves it again");
  checks.push("environment_control");
  await command("capture");
  await page.locator("canvas").screenshot({ path: `${output}/physics-fixture.png` });
  // A body under a parent is refused, and the world stands as it was.
  const before = await inspect("crate", "Transform");
  const parented = structuredClone(fixture); parented.revision = scene.revision + 1;
  parented.entities.find(entity => entity.id === "raft").parent = "floor";
  scene.serve(parented);
  const refused = await command("reload", { expected_revision: String(scene.revision), revision: String(parented.revision) }, false);
  assert.equal(refused.error?.code, "invalid_rigid_body", JSON.stringify(refused.error));
  assert.deepEqual(await inspect("crate", "Transform"), before);
  checks.push("parented_body_refused");
  // A workload without collision leaves physics out.
  const workload = devUrl("stress.html", { config: { renderables: 100, colliders: 0, queries_per_tick: 0, mesh_variety: 1 } });
  await page.goto(workload.href);
  await developmentReady(page);
  const missing = await command("inspect", { component: "PhysicsWorld" }, false);
  assert.match(missing.error.code, /missing_plugin/);
  checks.push("left_out_without_collision");
  assert.deepEqual(errors, []);
  const report = { ok: true, browser: browser.version(), url: url.href, artifact: output, checks };
  await writeFile(`${output}/report.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
} finally { await browser.close(); }
