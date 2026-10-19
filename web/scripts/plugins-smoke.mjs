/**
 * Features compose: for each engine feature, an engine with every other
 * feature's plugin but its own (`?without=`) loads and draws a scene,
 * leaves it out of status, answers `missing_plugin` to requests about it
 * and refuses scenes authoring it at their path; an engine with that
 * feature alone (and what it needs) loads a scene authoring it and answers
 * about it; one with every feature loads them all; and a feature without
 * one it needs (navigation without terrain) refuses to start. The features,
 * their plugins and needs come from the engine's one list of them
 * (protocol.json's `features`).
 */
// @ts-check
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { commander, developmentReady, launch, protocol, routedScene, sceneUrl } from "./harness.mjs";

const features = protocol.features.map(feature => feature.name);
// What a feature needs, and so what leaving it out leaves out too.
const needs = Object.fromEntries(protocol.features.map(feature => [feature.name, feature.needs]));
const plugin = name => protocol.features.find(feature => feature.name === name).plugin;
const core = () => ({
  id: "plugins-fixture", revision: 1,
  camera: { position: [0, 6, 12], target: [0, 0, 0], fov_degrees: 55, near: 0.1, far: 1000 },
  materials: [{ id: "plain", color: [0.7, 0.6, 0.5, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [{ id: "floor", mesh: "box", material: "plain", transform: { translation: [0, -0.5, 0], scale: [20, 1, 20] } }],
});
const ground = { id: "ground", terrain: { origin: [-128, 0, -128], tiles: [2, 2], spacing: 1, layers: [{ noise: { seed: 3, height: 8, wavelength: 120 } }] } };
// Each feature authored on the core scene, and the path a scene without it fails at.
const authoring = {
  terrain: [scene => scene.entities.push(ground), "$.entities[1].terrain"],
  world_streaming: [scene => { scene.streamed = [{ id: "crate", mesh: "box", material: "plain", transform: { translation: [4, 0.5, 0] } }]; }, "$.streamed"],
  navigation: [scene => {
    if (!scene.entities.some(entity => entity.id === "ground")) scene.entities.push(ground);
    scene.entities.push({ id: "walker", nav_agent: {} });
  }, "$.entities[2].nav_agent"],
  ocean: [scene => scene.entities.push({ id: "buoy", mesh: "box", material: "plain", floating: {} }), "$.entities[1].floating"],
  animation: [scene => {
    scene.animations = [{ id: "spin", targets: ["floor"], channels: [{ target: 0, property: "rotation", times: [0, 1], values: [0, 0, 0, 1, 0, 0.7071068, 0, 0.7071068] }] }];
    scene.entities[0].animation = { clip: "spin" };
  }, "$.entities[0].animation"],
  time_of_day: [scene => { scene.environment = { time_of_day: {} }; }, "$.environment.time_of_day"],
  deformation: [scene => scene.entities.push({ id: "roller", deformer: {} }), "$.entities[1].deformer"],
  collision: [scene => { scene.entities[0].collider = { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] } }; }, "$.entities[0].collider"],
  physics: [scene => scene.entities.push({ id: "body", mesh: "box", material: "plain", transform: { translation: [0, 2, 0] },
    collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] } }, rigid_body: {} }), "$.entities[1].rigid_body"],
};
const authored = names => {
  const scene = core();
  for (const name of names) authoring[name][0](scene);
  return scene;
};
const ray = { origin: [0, 5, 0], direction: [0, -1, 0], max_distance: 20 };
// A request about each feature: [op, fields] or an inspection, which its
// alone engine answers and the engine without it refuses.
const requests = {
  terrain: ["terrain_height", { points: [[0, 0]] }],
  world_streaming: ["inspect", { component: "WorldCells" }],
  navigation: ["inspect", { component: "NavAgents" }],
  ocean: ["inspect", { entity: "buoy", component: "Floating" }],
  animation: ["animation", { entity: "floor", speed: 0 }],
  time_of_day: ["inspect", { component: "TimeOfDay" }],
  deformation: ["inspect", { component: "Tracks" }],
  collision: ["raycast", ray],
  physics: ["inspect", { component: "PhysicsWorld" }],
};
// What the engine without it answers about it, scenes aside.
const refusals = {
  terrain: ["terrain_height", { points: [[0, 0]] }],
  world_streaming: ["world_changes", { clear: true }],
  navigation: ["inspect", { component: "NavAgents" }],
  ocean: ["environment", { ocean: { enabled: true } }],
  animation: ["animation", { entity: "floor", speed: 0 }],
  time_of_day: ["environment", { time_of_day: {} }],
  deformation: ["inspect", { component: "Tracks" }],
  collision: ["raycast", ray],
  physics: ["environment", { physics: { enabled: true } }],
};
const leftOut = name => features.filter(other => other === name || (needs[other] ?? []).includes(name));
// Every feature has its authored scene, a request it answers and one its
// absence refuses.
for (const name of features) {
  assert.ok(authoring[name] && requests[name] && refusals[name], `plugins-smoke needs ${name}'s authoring, request and refusal`);
}

const { browser, page, errors } = await launch({ width: 640, height: 400 });
const command = commander(page);

/** Opens `scene` on an engine without `without`, paused. */
async function open(scene, without) {
  const routed = routedScene(page, scene);
  await page.unrouteAll();
  await routed.route();
  await page.goto(String(sceneUrl({ without: without.join(",") })), { waitUntil: "domcontentloaded" });
  await developmentReady(page);
  await command("pause");
  return routed;
}

try {
  const checks = [];
  for (const name of features) {
    // Without it: the core scene loads and draws, status leaves it out,
    // requests about it and scenes authoring it fail.
    const without = leftOut(name);
    const scene = await open(core(), without);
    const rendering = (await command("render")).result.rendering;
    assert.ok(rendering.frame >= 0);
    const present = (await command("status")).result.features;
    assert.deepEqual(present, features.filter(other => !without.includes(other)), `${name}: status names the features added`);
    const [op, fields] = refusals[name];
    const refused = await command(op, fields, false);
    assert.equal(refused.error.code, "missing_plugin", `${name}: ${op} answers missing_plugin: ${JSON.stringify(refused.error)}`);
    const next = authored([name]);
    next.revision = scene.revision + 1;
    scene.serve(next);
    const reload = await command("reload", { expected_revision: String(scene.revision), revision: String(next.revision) }, false);
    assert.equal(reload.error.code, "missing_plugin", `${name}: ${JSON.stringify(reload.error)}`);
    assert.equal(reload.error.path, authoring[name][1]);

    // Alone, with what it needs: a scene authoring it loads, and requests
    // about it are answered.
    const alone = [name, ...(needs[name] ?? [])];
    await open(authored([name]), features.filter(other => !alone.includes(other)));
    await command("render");
    assert.deepEqual((await command("status")).result.features, features.filter(other => alone.includes(other)));
    const [askOp, askFields] = requests[name];
    await command(askOp, askFields);
    checks.push({ feature: name, without, refused: refused.error.message, path: reload.error.path });
  }

  // Every feature: a scene authoring them all loads.
  await open(authored(features), []);
  await command("render");
  assert.deepEqual((await command("status")).result.features, features);

  // A feature needing another: an engine with it and not the other does
  // not start.
  for (const name of features) {
    for (const need of needs[name]) {
      const message = `${plugin(name)} needs ${need}: add ${plugin(need)} too, or leave ${name} out`;
      const before = errors.length;
      await page.unrouteAll();
      await routedScene(page, core()).route();
      // Other features needing it go too, so this one is refused.
      const without = features.filter(other => other === need || (other !== name && needs[other].includes(need)));
      await page.goto(String(sceneUrl({ without: without.join(",") })), { waitUntil: "domcontentloaded" });
      for (let attempt = 0; attempt < 100 && !errors.slice(before).some(error => error.includes(message)); attempt++) await delay(100);
      assert.ok(errors.slice(before).some(error => error.includes(message)), `${name} without ${need} refuses: ${errors.slice(before).join(" | ")}`);
      errors.splice(before);
    }
  }

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await browser.close();
}
