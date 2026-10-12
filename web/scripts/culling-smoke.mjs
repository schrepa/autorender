/** Culling and GPU residency checks; authored fixtures exist only in routed responses. */
// @ts-check
import assert from "node:assert/strict";
import { artifacts, commander, sceneUrl, differentPixels, launch, openPaused, routedScene } from "./harness.mjs";

const url = sceneUrl();
const output = await artifacts("culling-smoke");
const camera = { position: [0, 0.6, 8], target: [0, 0.6, 0], vertical_fov_radians: Math.PI / 3, near: 1, far: 50 };
// Shadows start off, so residency first follows visibility alone: shadow maps
// draw off-screen casters too.
/** @type {any} */
const fixture = {
  id: "culling-fixture", revision: 1,
  camera: { position: camera.position, target: camera.target, fov_degrees: 60, near: 1, far: 50 },
  environment: { shadows: { enabled: false } },
  textures: [{ id: "pixel", width: 1, height: 1, rgba8: Buffer.from([255, 90, 25, 255]).toString("base64") }],
  materials: [{ id: "plain", color: [0.2, 0.8, 0.7, 1] }, { id: "textured", color: [1, 1, 1, 1], texture: "pixel" }],
  meshes: [{ id: "box", source: { kind: "cube" } }, { id: "gem", source: { kind: "octahedron" } }],
  entities: [
    { id: "player", mesh: "box", material: "plain", tags: ["player"], collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] }, memberships: 2, filter: 1, sensor: false }, transform: { translation: [0, 0.6, 0] } },
    { id: "offscreen", mesh: "gem", material: "textured", transform: { translation: [100, 0.6, 0] } },
    { id: "floor", transform: { translation: [0, -0.5, 0] }, collider: { shape: { kind: "cuboid", half_extents: [500, 0.5, 500] }, memberships: 1, filter: 2, sensor: false } },
  ],
};
const { browser, page, errors } = await launch();
const command = commander(page);
const scene = routedScene(page, fixture);

function stats(reply) {
  const value = reply.result.rendering;
  for (const field of ["render_candidates", "visible_instances", "culled_instances", "instances", "invalid_transforms"]) {
    assert.ok(Number.isInteger(value[field]) && value[field] >= 0, `${field} must be a nonnegative integer`);
  }
  assert.equal(value.invalid_transforms, 0);
  assert.equal(value.render_candidates, value.visible_instances + value.culled_instances);
  assert.equal(value.visible_instances, value.instances, "The fixture is below the instance budget.");
  return value;
}

async function capture(name) {
  await command("capture");
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  assert.ok(bytes.length > 1000, "The fixture should produce a nonempty canvas.");
  return bytes.toString("base64");
}

async function equivalent(name, pose) {
  const before = await command("status");
  await command("camera", pose);
  const enabled = await command("culling", { enabled: true });
  const on = stats(enabled);
  const first = await capture(`${name}-on`);
  const disabled = await command("culling", { enabled: false });
  const off = stats(disabled);
  const second = await capture(`${name}-off`);
  const pixels = await differentPixels(page, [first, second], { alpha: true });
  assert.equal(pixels.changed, 0, `${name}: culling changed visible pixels`);
  assert.equal(off.culled_instances, 0);
  assert.equal(off.render_candidates, on.render_candidates);
  assert.ok(on.culled_instances > 0, `${name}: fixture must exercise culling`);
  assert.equal(disabled.observation.tick, before.observation.tick, "Camera/culling controls must not advance simulation.");
  assert.equal(disabled.observation.frame, before.observation.frame);
  await command("culling", { enabled: true });
  return { name, candidates: on.render_candidates, visible: on.visible_instances, culled: on.culled_instances, changedPixels: pixels.changed };
}

try {
  await scene.route();
  await openPaused(page, url);
  const initial = stats(await command("render"));
  assert.equal(initial.render_candidates, 2);
  assert.equal(initial.instances, 1);
  assert.equal(initial.culled_instances, 1);
  assert.equal(initial.resident_meshes, 1);
  assert.equal(initial.resident_textures, 0, "First offscreen appearance must not upload texture pixels.");
  assert.equal((await command("assets")).result.texture_assets[0].gpu_resident, false);
  assert.equal(initial.shadows.active, false);

  // An off-screen caster needs its mesh for the shadow maps, never its texture.
  const lit = await command("environment", { shadows: { enabled: true } });
  assert.deepEqual(lit.result.environment_changed, ["shadows"]);
  const shadowed = stats(lit);
  assert.equal(shadowed.shadows.active, true);
  assert.ok(shadowed.shadows.draws > 0);
  assert.equal(shadowed.instances, 1);
  assert.equal(shadowed.resident_meshes, 2, "An off-screen shadow caster uploads its mesh.");
  assert.equal(shadowed.resident_textures, 0, "Shadow maps must not upload caster textures.");

  const revealed = stats(await command("camera", { ...camera, position: [100, 0.6, 8], target: [100, 0.6, 0] }));
  assert.equal(revealed.instances, 1);
  assert.equal(revealed.resident_meshes, 2, "The first mesh remains resident while culled.");
  assert.equal(revealed.resident_textures, 1);
  assert.equal(revealed.mesh_upload_bytes, 0, "The caster's mesh was already resident.");
  assert.equal(revealed.texture_upload_bytes, 4);
  const away = stats(await command("camera", { ...camera, target: [0, 0.6, 16] }));
  assert.equal(away.instances, 0);
  assert.equal(away.culled_instances, 2);
  assert.equal(away.resident_meshes, 2);
  assert.equal(away.resident_textures, 1);
  const returned = stats(await command("camera", { ...camera, position: [100, 0.6, 8], target: [100, 0.6, 0] }));
  assert.equal(returned.mesh_upload_bytes, 0);
  assert.equal(returned.texture_upload_bytes, 0, "Returning to view must reuse the same GPU allocation.");
  const cases = [await equivalent("separated-objects", camera)];
  const inside = await equivalent("camera-inside-bounds", { ...camera, position: [0, 0.6, 0], target: [0, 0.6, -8], near: 0.01 });
  assert.equal(inside.visible, 1, "A bound containing the camera remains visible even if back-face culling hides its surfaces.");
  cases.push(inside);

  const aspect = await page.locator("canvas").evaluate(/** @param {HTMLCanvasElement} canvas */ canvas => canvas.width / canvas.height);
  const borderX = 8 * Math.tan(camera.vertical_fov_radians / 2) * aspect;
  const complex = structuredClone(fixture);
  complex.prefabs = [{ id: "tilted", entities: [{ id: "child", mesh: "box", material: "textured", transform: { rotation_degrees: [17, 35, 26], scale: [0.8, 1.1, 0.6] } }] }];
  complex.entities.push(
    { id: "sheared", prefab: "tilted", transform: { translation: [-2, 0.6, 0], rotation_degrees: [0, 21, 0], scale: [2, 0.7, 1.3] } },
    { id: "right-border", mesh: "box", material: "plain", transform: { translation: [borderX, 0.6, 0] } },
    { id: "left-border", mesh: "box", material: "plain", transform: { translation: [-borderX, 0.6, 0] } },
    { id: "near-intersection", mesh: "gem", material: "plain", transform: { translation: [0.5, 0.6, 7.05], scale: [0.8, 0.8, 0.8] } },
    { id: "far-intersection", mesh: "box", material: "textured", transform: { translation: [8, 0.6, -42], scale: [2, 2, 2] } },
    { id: "behind-camera", mesh: "box", material: "plain", transform: { translation: [0, 0.6, 12] } },
    { id: "beyond-far", mesh: "box", material: "plain", transform: { translation: [0, 0.6, -80] } },
  );
  // Without an environment a reload keeps the current look: shadows stay on,
  // and culled casters must still shadow the view.
  delete complex.environment;
  const reloaded = await scene.reload(command, complex);
  assert.equal(reloaded.result.rendering.shadows.active, true);
  cases.push(await equivalent("hierarchy-shear-borders", camera));
  cases.push(await equivalent("angled-camera", { ...camera, position: [8, 6, 12], target: [0, 0.6, 0] }));

  const removed = structuredClone(fixture);
  removed.entities = removed.entities.filter(entity => entity.id === "player");
  await scene.reload(command, removed);
  const retired = stats(await command("camera", camera));
  assert.equal(retired.resident_meshes, 1);
  assert.equal(retired.resident_textures, 0, "An unreferenced CPU texture must not retain its retired GPU resource.");
  assert.equal(retired.resident_texture_bytes, 0);
  const texture = (await command("assets")).result.texture_assets[0];
  assert.equal(texture.state, "cpu_ready");
  assert.equal(texture.gpu_resident, false);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, url: url.href, artifact: output, cases, checks: ["offscreen_first_use_no_upload", "offscreen_caster_mesh_only", "out_of_view_retains_gpu", "return_reuses_gpu", "source_removal_retires_gpu", "culling_pixel_equivalence", "shadowed_culling_pixel_equivalence", "environmentless_reload_keeps_look", "paused_camera_controls", "clean_browser_diagnostics"] }));
} finally {
  await browser.close();
}
