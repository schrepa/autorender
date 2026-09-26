/** Culling and GPU residency checks; authored fixtures exist only in routed responses. */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { chromium } from "@playwright/test";

const base = `${(process.env.DEMO_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "")}/`;
const url = new URL("playground.html", base);
url.searchParams.set("engine_dev", "1");
const output = fileURLToPath(new URL(`../../artifacts/culling-smoke-${randomUUID()}/`, import.meta.url));
const camera = { position: [0, 0.6, 8], target: [0, 0.6, 0], vertical_fov_radians: Math.PI / 3, near: 1, far: 50 };
const fixture = {
  version: 4, id: "culling-fixture", revision: 1,
  camera: { position: camera.position, target: camera.target, fov_degrees: 60, near: 1, far: 50 },
  textures: [{ id: "pixel", width: 1, height: 1, rgba8: Buffer.from([255, 90, 25, 255]).toString("base64") }],
  materials: [{ id: "plain", color: [0.2, 0.8, 0.7, 1] }, { id: "textured", color: [1, 1, 1, 1], texture: "pixel" }],
  meshes: [{ id: "box", source: { kind: "cube" } }, { id: "gem", source: { kind: "octahedron" } }],
  entities: [
    { id: "player", mesh: "box", material: "plain", tags: ["player"], collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] }, memberships: 2, filter: 1, sensor: false }, transform: { translation: [0, 0.6, 0] } },
    { id: "offscreen", mesh: "gem", material: "textured", transform: { translation: [100, 0.6, 0] } },
    { id: "floor", transform: { translation: [0, -0.5, 0] }, collider: { shape: { kind: "cuboid", half_extents: [500, 0.5, 500] }, memberships: 1, filter: 2, sensor: false } },
  ],
};
let served = structuredClone(fixture);
let revision = fixture.revision;
const browser = await chromium.launch({
  ...(process.env.BROWSER_PATH ? { executablePath: resolve(process.env.BROWSER_PATH) } : { channel: process.env.BROWSER_CHANNEL ?? "chrome" }),
  headless: true,
  args: ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])],
});
const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 1 });
page.setDefaultTimeout(30_000);
const errors = [];
const recordError = value => { if (errors.length < 20) errors.push(String(value).slice(0, 2000)); };
page.on("pageerror", recordError);
page.on("console", entry => { if (entry.type() === "error") recordError(entry.text()); });

async function command(op, fields = {}, expectedOk = true) {
  const reply = await page.evaluate(command => window.__engineDev.request(command), { op, ...fields });
  assert.equal(reply.ok, expectedOk, `${op}: ${JSON.stringify(reply).slice(0, 4000)}`);
  return reply;
}

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

async function reload(document) {
  served = structuredClone(document);
  served.revision = revision + 1;
  const reply = await command("reload", { expected_revision: String(revision), revision: String(served.revision) });
  revision = served.revision;
  assert.equal(reply.observation.scene.revision, String(revision));
  return reply;
}

async function capture(name) {
  await command("capture");
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  assert.ok(bytes.length > 1000, "The fixture should produce a nonempty canvas.");
  return bytes.toString("base64");
}

async function differentPixels(images) {
  return page.evaluate(async images => {
    const pixels = await Promise.all(images.map(async png => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, image.width, image.height);
    }));
    if (pixels[0].width !== pixels[1].width || pixels[0].height !== pixels[1].height) throw new Error("Capture dimensions changed.");
    let changed = 0;
    for (let offset = 0; offset < pixels[0].data.length; offset += 4) {
      if ([0, 1, 2, 3].some(channel => pixels[0].data[offset + channel] !== pixels[1].data[offset + channel])) changed++;
    }
    return { changed, total: pixels[0].width * pixels[0].height };
  }, images);
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
  const pixels = await differentPixels([first, second]);
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
  await page.route("**/scenes/playground.scene.json", route => route.fulfill({ json: served }));
  await page.goto(url.href, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__engineDev?.version === 1, null, { timeout: 60_000 });
  const deadline = Date.now() + 5000;
  while (true) {
    const reply = await page.evaluate(() => window.__engineDev.request({ op: "pause" }));
    // Only this rejection is retried: it occurs before the command dispatches.
    if (reply.error?.code === "command_busy" && Date.now() < deadline) { await delay(25); continue; }
    assert.equal(reply.ok, true, JSON.stringify(reply).slice(0, 4000));
    break;
  }
  await mkdir(output, { recursive: true });
  const initial = stats(await command("render"));
  assert.equal(initial.render_candidates, 2);
  assert.equal(initial.instances, 1);
  assert.equal(initial.culled_instances, 1);
  assert.equal(initial.resident_meshes, 1);
  assert.equal(initial.resident_textures, 0, "First offscreen appearance must not upload texture pixels.");
  assert.equal((await command("assets")).result.texture_assets[0].gpu_resident, false);

  const revealed = stats(await command("camera", { ...camera, position: [100, 0.6, 8], target: [100, 0.6, 0] }));
  assert.equal(revealed.instances, 1);
  assert.equal(revealed.resident_meshes, 2, "The first mesh remains resident while culled.");
  assert.equal(revealed.resident_textures, 1);
  assert.ok(revealed.mesh_upload_bytes > 0);
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

  const aspect = await page.locator("canvas").evaluate(canvas => canvas.width / canvas.height);
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
  await reload(complex);
  cases.push(await equivalent("hierarchy-shear-borders", camera));
  cases.push(await equivalent("angled-camera", { ...camera, position: [8, 6, 12], target: [0, 0.6, 0] }));

  const removed = structuredClone(fixture);
  removed.entities = removed.entities.filter(entity => entity.id === "player");
  await reload(removed);
  const retired = stats(await command("camera", camera));
  assert.equal(retired.resident_meshes, 1);
  assert.equal(retired.resident_textures, 0, "An unreferenced CPU texture must not retain its retired GPU resource.");
  assert.equal(retired.resident_texture_bytes, 0);
  const texture = (await command("assets")).result.texture_assets[0];
  assert.equal(texture.state, "cpu_ready");
  assert.equal(texture.gpu_resident, false);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, url: url.href, artifact: output, cases, checks: ["offscreen_first_use_no_upload", "out_of_view_retains_gpu", "return_reuses_gpu", "source_removal_retires_gpu", "culling_pixel_equivalence", "paused_camera_controls", "clean_browser_diagnostics"] }));
} finally {
  await browser.close();
}
