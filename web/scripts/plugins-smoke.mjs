// @ts-check
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { commander, developmentReady, launch, protocol, routedScene, sceneUrl } from "./harness.mjs";

const records = JSON.parse(readFileSync(new URL("../../bricks.json", import.meta.url), "utf8")).bricks;
const bricks = protocol.bricks;
const core = () => ({
  id: "plugins-fixture", revision: 1,
  camera: { position: [0, 6, 12], target: [0, 0, 0], fov_degrees: 55, near: 0.1, far: 1000 },
  materials: [{ id: "plain", color: [0.7, 0.6, 0.5, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [{ id: "floor", mesh: "box", material: "plain", transform: { translation: [0, -0.5, 0], scale: [20, 1, 20] } }],
});
const ground = { id: "ground", terrain: { origin: [-128, 0, -128], tiles: [2, 2], spacing: 1, layers: [{ noise: { seed: 3, height: 8, wavelength: 120 } }] } };
const authoring = {
  terrain: [scene => scene.entities.push(ground), "$.entities[1].terrain"],
  world_streaming: [scene => { scene.streamed = [{ id: "crate", mesh: "box", material: "plain", transform: { translation: [4, 0.5, 0] } }]; }, "$.streamed"],
  navigation: [scene => {
    if (!scene.entities.some(entity => entity.id === "ground")) scene.entities.push(ground);
    scene.entities.push({ id: "walker", nav_agent: {} });
  }, "$.entities[2].nav_agent"],
  time_of_day: [scene => { scene.environment = { time_of_day: {} }; }, "$.environment.time_of_day"],
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
const requests = {
  terrain: ["terrain_height", { points: [[0, 0]] }],
  world_streaming: ["inspect", { component: "WorldCells" }],
  navigation: ["inspect", { component: "NavAgents" }],
  time_of_day: ["inspect", { component: "TimeOfDay" }],
  collision: ["raycast", ray],
  physics: ["inspect", { component: "PhysicsWorld" }],
};
const refusals = {
  terrain: ["terrain_height", { points: [[0, 0]] }],
  world_streaming: ["world_changes", { clear: true }],
  navigation: ["inspect", { component: "NavAgents" }],
  time_of_day: ["environment", { time_of_day: {} }],
  collision: ["raycast", ray],
  physics: ["environment", { physics: { enabled: true } }],
};
const features = bricks.filter(name => name in authoring);
const reads = Object.fromEntries(records.map(record => [record.name, record.reads]));
const needs = name => {
  const found = new Set();
  const pending = [...(reads[name] ?? [])];
  while (pending.length) {
    const next = pending.pop();
    if (!found.has(next)) {
      found.add(next);
      pending.push(...(reads[next] ?? []));
    }
  }
  return [...found];
};
const leftOut = name => bricks.filter(other => other === name || needs(other).includes(name));
for (const name of Object.keys(authoring)) {
  assert.ok(features.includes(name) && requests[name] && refusals[name], `plugins-smoke needs ${name}'s brick, authoring, request and refusal`);
}

const { browser, page, errors } = await launch({ width: 640, height: 400 });
const command = commander(page);

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
    const without = leftOut(name);
    const scene = await open(core(), without);
    const rendering = (await command("render")).result.rendering;
    assert.ok(rendering.frame >= 0);
    const present = (await command("status")).result.features;
    assert.deepEqual(present, bricks.filter(other => !without.includes(other)), `${name}: status names the bricks added`);
    const [op, fields] = refusals[name];
    const refused = await command(op, fields, false);
    assert.equal(refused.error.code, "missing_plugin", `${name}: ${op} answers missing_plugin: ${JSON.stringify(refused.error)}`);
    const next = authored([name]);
    next.revision = scene.revision + 1;
    scene.serve(next);
    const reload = await command("reload", { expected_revision: String(scene.revision), revision: String(next.revision) }, false);
    assert.equal(reload.error.code, "missing_plugin", `${name}: ${JSON.stringify(reload.error)}`);
    assert.equal(reload.error.path, authoring[name][1]);

    const alone = [name, ...needs(name)];
    const others = features.filter(other => !alone.includes(other));
    await open(authored([name]), others);
    await command("render");
    assert.deepEqual((await command("status")).result.features, bricks.filter(other => !others.includes(other)));
    const [askOp, askFields] = requests[name];
    await command(askOp, askFields);
    checks.push({ feature: name, without, refused: refused.error.message, path: reload.error.path });
  }

  await open(authored(features), []);
  await command("render");
  assert.deepEqual((await command("status")).result.features, bricks);

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await browser.close();
}
