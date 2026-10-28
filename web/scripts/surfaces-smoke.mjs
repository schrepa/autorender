// @ts-check
import assert from "node:assert/strict";
import { artifacts, commander, sceneUrl, launch, openPaused, routedScene } from "./harness.mjs";

const url = sceneUrl();
const output = await artifacts("surfaces-smoke");
const fixture = {
  id: "surfaces-fixture", revision: 1,
  camera: { position: [0, 9, 16], target: [0, 0, 0], fov_degrees: 55, near: 0.1, far: 500 },
  materials: [
    { id: "ground", color: [0.6, 0.62, 0.58, 1] },
    { id: "plain", color: [0.8, 0.3, 0.2, 1] },
  ],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [
    { id: "floor", mesh: "box", material: "ground", transform: { translation: [0, -0.5, 0], scale: [30, 1, 30] } },
    { id: "behind", mesh: "box", material: "plain", transform: { translation: [0, 1, -2], scale: [2, 2, 2] } },
    { id: "tinted", mesh: "box", material: "plain", tint: [0.2, 0.9, 0.3, 1], transform: { translation: [5, 1, 0] } },
    { id: "probe", reflection_probe: { extent: [12, 6, 12] }, transform: { translation: [0, 3, 0] } },
  ],
};
const { browser, page, errors } = await launch({ width: 1280, height: 720 });
const command = commander(page);
const scene = routedScene(page, fixture);

async function inspect(entity, component) {
  return (await command("inspect", { entity, component })).result.entity.components[component];
}

try {
  await scene.route();
  await openPaused(page, url);
  const checks = [];

  const rendering = (await command("render")).result.rendering;
  assert.ok(Number.isInteger(rendering.frame) && Number.isInteger(rendering.tick), "stats name their frame and tick");
  checks.push({ check: "draws", frame: rendering.frame, tick: rendering.tick });

  assert.deepEqual(await inspect("tinted", "Tint"), [0.2, 0.9, 0.3, 1].map(Math.fround));
  const probe = await inspect("probe", "ReflectionProbe");
  assert.deepEqual(probe.extent, [12, 6, 12]);
  const camera = (await command("inspect", { component: "Camera" })).result.value;
  assert.ok(Array.isArray(camera.position), `the camera inspects: ${JSON.stringify(camera)}`);
  checks.push({ check: "inspects", probe });

  const merged = (await command("environment", { merge: true, particles: { max_particles: 1 } })).result;
  assert.deepEqual(merged.environment_changed, ["particles"]);
  assert.equal(merged.before.particles.enabled, true);
  assert.equal(merged.after.particles.enabled, true, "merging keeps the omitted field");
  assert.equal(merged.after.particles.max_particles, 1);
  const off = (await command("environment", { merge: true, particles: { enabled: false } })).result;
  assert.equal(off.after.particles.max_particles, 1, "merging keeps the budget");
  await command("environment", { particles: { enabled: true } });
  checks.push({ check: "merged environment" });

  const benchmark = (await command("benchmark", { mode: "render", warmup: 1, samples: 3 })).result.benchmark;
  assert.equal(benchmark.samples, 3, `benchmark: ${JSON.stringify(benchmark).slice(0, 300)}`);

  const unknown = await command("environment", { particles: { max_particle: 1 } }, false);
  assert.equal(unknown.error.code, "unknown_field", JSON.stringify(unknown.error));
  assert.equal(unknown.error.path, "$.particles.max_particle");
  const mistyped = await command("camera", { position: [0, 1, "x"], target: [0, 0, 0], vertical_fov_radians: 1, near: 0.1, far: 100 }, false);
  assert.equal(mistyped.error.code, "invalid_type", JSON.stringify(mistyped.error));
  const game = await command("game", { name: "workload" }, false);
  assert.equal(game.error.code, "unsupported_command", "scene mode registers no game controls");
  const catalog = (await command("describe", { name: "ErrorCodes" })).result.codes;
  for (const code of [unknown.error.code, mistyped.error.code, game.error.code]) assert.ok(catalog[code], `the catalog describes ${code}`);
  checks.push({ check: "errors", unknown: unknown.error, mistyped: mistyped.error });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, output, checks }));
} finally {
  await browser.close();
}
