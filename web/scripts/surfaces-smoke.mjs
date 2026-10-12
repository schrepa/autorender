/**
 * Surfaces: a fixture with a decal, a refracting glass box, a
 * tinted box and a reflection probe. Frames draw the decal and refract;
 * the components inspect as authored; a merged environment control keeps
 * the fields it omits and answers with each section before and after; a
 * decal budget drops the farther decal; benchmarks sample; requests that
 * break their types answer with the field at fault, and the catalog of
 * error codes describes every code.
 */
// @ts-check
import assert from "node:assert/strict";
import { artifacts, commander, sceneUrl, differentPixels, launch, openPaused, routedScene } from "./harness.mjs";

const url = sceneUrl();
const output = await artifacts("surfaces-smoke");
const fixture = {
  id: "surfaces-fixture", revision: 1,
  camera: { position: [0, 9, 16], target: [0, 0, 0], fov_degrees: 55, near: 0.1, far: 500 },
  materials: [
    { id: "ground", color: [0.6, 0.62, 0.58, 1] },
    { id: "plain", color: [0.8, 0.3, 0.2, 1] },
    { id: "mark", color: [0.1, 0.2, 0.9, 1] },
    { id: "glass", color: [0.9, 0.95, 1, 1], transmission: 1, ior: 1.5, thickness: 0.5 },
  ],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  environment: { refraction: {}, decals: { enabled: true } },
  entities: [
    { id: "floor", mesh: "box", material: "ground", transform: { translation: [0, -0.5, 0], scale: [30, 1, 30] } },
    { id: "mark", decal: { material: "mark", opacity: 0.9 }, transform: { translation: [-4, 0, 2], scale: [4, 2, 4] } },
    { id: "far_mark", decal: { material: "mark" }, transform: { translation: [6, 0, -10], scale: [3, 2, 3] } },
    { id: "glass", mesh: "box", material: "glass", transform: { translation: [0, 1.5, 4], scale: [3, 3, 0.5] } },
    { id: "behind", mesh: "box", material: "plain", transform: { translation: [0, 1, -2], scale: [2, 2, 2] } },
    { id: "tinted", mesh: "box", material: "plain", tint: [0.2, 0.9, 0.3, 1], transform: { translation: [5, 1, 0] } },
    { id: "probe", reflection_probe: { extent: [12, 6, 12] }, transform: { translation: [0, 3, 0] } },
  ],
};
const { browser, page, errors } = await launch({ width: 1280, height: 720 });
const command = commander(page);
const scene = routedScene(page, fixture);

async function capture(name) {
  await command("render");
  await command("capture");
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return bytes.toString("base64");
}

/** Component `component` of authored entity `entity`. */
async function inspect(entity, component) {
  return (await command("inspect", { entity, component })).result.entity.components[component];
}

try {
  await scene.route();
  await openPaused(page, url);
  const checks = [];

  // Frames draw the decal and refract through the glass.
  const rendering = (await command("render")).result.rendering;
  assert.equal(rendering.decals.drawn, 2, `the decals draw: ${JSON.stringify(rendering.decals)}`);
  assert.equal(rendering.decals.invalid, 0);
  assert.equal(rendering.refraction.active, true, "the glass refracts");
  assert.ok(Number.isInteger(rendering.frame) && Number.isInteger(rendering.tick), "stats name their frame and tick");
  checks.push({ check: "draws", decals: rendering.decals, refraction: rendering.refraction });

  // The components inspect as authored.
  const decal = await inspect("mark", "Decal");
  assert.equal(decal.material, "mark");
  assert.equal(decal.material_alive, true);
  assert.ok(Math.abs(decal.opacity - 0.9) < 1e-6);
  assert.deepEqual(await inspect("tinted", "Tint"), [0.2, 0.9, 0.3, 1].map(Math.fround));
  const probe = await inspect("probe", "ReflectionProbe");
  assert.deepEqual(probe.extent, [12, 6, 12]);
  const camera = (await command("inspect", { component: "Camera" })).result.value;
  assert.ok(Array.isArray(camera.position), `the camera inspects: ${JSON.stringify(camera)}`);
  checks.push({ check: "inspects", decal, probe });

  // A merged control keeps what it omits and answers before and after.
  const withDecal = await capture("decal");
  const merged = (await command("environment", { merge: true, decals: { max_decals: 1 } })).result;
  assert.deepEqual(merged.environment_changed, ["decals"]);
  assert.equal(merged.before.decals.enabled, true);
  assert.equal(merged.after.decals.enabled, true, "merging keeps the omitted field");
  assert.equal(merged.after.decals.max_decals, 1);
  const budgeted = (await command("render")).result.rendering;
  assert.equal(budgeted.decals.drawn, 1);
  assert.equal(budgeted.decals.dropped, 1, "the budget drops the farther decal");
  const off = (await command("environment", { merge: true, decals: { enabled: false } })).result;
  assert.equal(off.after.decals.max_decals, 1, "merging keeps the budget");
  const withoutDecal = await capture("no-decal");
  const { changed } = await differentPixels(page, [withDecal, withoutDecal]);
  assert.ok(changed > 500, `the decals cover pixels: ${changed}`);
  await command("environment", { decals: { enabled: true } });
  checks.push({ check: "merged environment", pixels: changed });

  // Benchmarks sample frames.
  const benchmark = (await command("benchmark", { mode: "render", warmup: 1, samples: 3 })).result.benchmark;
  assert.equal(benchmark.samples, 3, `benchmark: ${JSON.stringify(benchmark).slice(0, 300)}`);

  // Requests breaking their types name the field; the catalog has every code.
  const unknown = await command("environment", { decals: { max_decal: 1 } }, false);
  assert.equal(unknown.error.code, "unknown_field", JSON.stringify(unknown.error));
  assert.equal(unknown.error.path, "$.decals.max_decal");
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
