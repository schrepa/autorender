/** Texture lifecycle integration against a development host; fixtures stay in routed responses. */
import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { chromium } from "@playwright/test";

const base = `${(process.env.DEMO_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "")}/`;
const url = new URL("playground.html", base);
url.searchParams.set("engine_dev", "1");
const sourceUrl = new URL("scenes/playground.scene.json", base);
const output = fileURLToPath(new URL(`../../artifacts/texture-smoke-${randomUUID()}/`, import.meta.url));
const browser = await chromium.launch({
  ...(process.env.BROWSER_PATH
    ? { executablePath: resolve(process.env.BROWSER_PATH) }
    : { channel: process.env.BROWSER_CHANNEL ?? "chrome" }),
  headless: true,
  args: ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])],
});
const errors = [];
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
page.setDefaultTimeout(30_000);
page.on("pageerror", error => errors.push(String(error)));
page.on("console", entry => { if (entry.type() === "error") errors.push(entry.text()); });

async function command(op, fields = {}, expectedOk = true) {
  const reply = await page.evaluate(command => window.__engineDev.request(command), { op, ...fields });
  assert.equal(reply.ok, expectedOk, `${op}: ${JSON.stringify(reply)}`);
  return reply;
}

function residency(reply) {
  const { result } = reply;
  return {
    meshes: result.meshes,
    materials: result.materials,
    textures: result.textures,
    cpu: result.resident_cpu_bytes,
    textureCpu: result.texture_cpu_bytes,
    gpuMeshes: result.rendering.resident_meshes,
    meshGpu: result.rendering.resident_mesh_bytes,
    gpuTextures: result.rendering.resident_textures,
    textureGpu: result.rendering.resident_texture_bytes,
  };
}

function readyTextures(reply, document) {
  const rows = reply.result.texture_assets;
  assert.equal(reply.result.textures, document.textures.length);
  assert.deepEqual(rows.map(row => row.id).sort(), document.textures.map(texture => texture.id).sort());
  for (const row of rows) {
    const texture = document.textures.find(texture => texture.id === row.id);
    assert.equal(row.state, "cpu_ready", row.id);
    assert.equal(row.gpu_resident, true, row.id);
    assert.equal(row.width, texture.width);
    assert.equal(row.height, texture.height);
    assert.equal(row.bytes, texture.width * texture.height * 4);
  }
}

async function capture(name) {
  await command("capture");
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  assert.ok(bytes.length > 1000, "The rendered canvas should contain a full scene.");
  return { hash: createHash("sha256").update(bytes).digest("hex"), png: bytes.toString("base64") };
}

async function changedPixels(first, second) {
  return page.evaluate(async images => {
    const decoded = await Promise.all(images.map(async png => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width;
      canvas.height = image.height;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, image.width, image.height);
    }));
    if (decoded[0].width !== decoded[1].width || decoded[0].height !== decoded[1].height) {
      throw new Error("Canvas dimensions changed during texture replacement.");
    }
    let changed = 0;
    for (let offset = 0; offset < decoded[0].data.length; offset += 4) {
      for (let channel = 0; channel < 4; channel++) {
        if (decoded[0].data[offset + channel] !== decoded[1].data[offset + channel]) {
          changed++;
          break;
        }
      }
    }
    return { changed, total: decoded[0].width * decoded[0].height };
  }, [first.png, second.png]);
}

try {
  const sourceResponse = await page.request.get(sourceUrl.href);
  assert.equal(sourceResponse.ok(), true, `Scene fetch failed: ${sourceResponse.status()}`);
  const source = await sourceResponse.json();
  assert.ok(Number.isSafeInteger(source.revision) && source.revision <= Number.MAX_SAFE_INTEGER - 10);
  assert.ok(source.textures?.length > 0, "Crystal Garden must include imported textured scenery.");
  const player = source.entities.find(entity => entity.tags?.includes("player"))?.id;
  assert.ok(player, "Crystal Garden must include an authored player.");
  let served = source;
  await page.route("**/scenes/playground.scene.json", route => route.fulfill({ json: served }));
  await page.goto(url.href, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__engineDev?.version === 1, null, { timeout: 60_000 });
  // A busy reply rejects before dispatch; only this initialization rejection is retried.
  const deadline = Date.now() + 5000;
  while (true) {
    const reply = await page.evaluate(() => window.__engineDev.request({ op: "pause" }));
    if (reply.error?.code === "command_busy" && Date.now() < deadline) { await delay(25); continue; }
    assert.equal(reply.ok, true, JSON.stringify(reply));
    assert.equal(reply.result.paused, true);
    break;
  }
  await mkdir(output, { recursive: true });
  await command("render");
  const initial = await command("assets");
  readyTextures(initial, source);
  const initialResidency = residency(initial);
  assert.ok(initialResidency.textureCpu > 0);
  assert.equal(initialResidency.gpuTextures, initialResidency.textures);
  assert.equal(initialResidency.textureGpu, initialResidency.textureCpu);
  const meshViews = await command("inspect", { component: "Mesh3d", limit: 100 });
  const texturedEntity = meshViews.result.entities.find(entity => entity.components.Mesh3d.texture_alive);
  assert.ok(texturedEntity, "A textured material must be used by scenery.");
  const meshView = await command("inspect", { entity: texturedEntity.id, component: "Mesh3d" });
  assert.equal(meshView.result.entity.components.Mesh3d.texture_alive, true);
  for (let frame = 0; frame < 3; frame++) {
    const warmed = await command("render");
    assert.equal(warmed.result.rendering.texture_upload_bytes, 0, "Warm frames must not upload texture pixels again.");
  }

  // Keep recognizable gameplay state so a rejected replacement cannot hide a reset.
  await command("input", { forward: 1 });
  await command("step", { ticks: 6 });
  await command("input", { forward: 0 });
  const playerBefore = await command("inspect", { entity: player, component: "Transform" });
  const progressBefore = await page.locator("canvas").evaluate(canvas => JSON.parse(canvas.dataset.progress));
  const beforeFailure = await command("assets");
  const invalidResults = [];
  for (const kind of ["base64", "alpha"]) {
    served = structuredClone(source);
    served.revision++;
    if (kind === "base64") served.textures[0].rgba8 = "!invalid-base64!";
    else {
      const pixels = Buffer.from(served.textures[0].rgba8, "base64");
      pixels[3] = 254;
      served.textures[0].rgba8 = pixels.toString("base64");
    }
    const failed = await command("reload", {
      expected_revision: String(source.revision), revision: String(served.revision),
    }, false);
    assert.equal(failed.observation.scene.revision, String(source.revision));
    assert.equal(failed.observation.tick, playerBefore.observation.tick);
    assert.match(failed.error.path, /^\$\.textures\[0\]/);
    invalidResults.push({ kind, code: failed.error.code, path: failed.error.path });
    const unchanged = await command("inspect", { entity: player, component: "Transform" });
    assert.deepEqual(unchanged.result.entity, playerBefore.result.entity);
    assert.deepEqual(residency(await command("assets")), residency(beforeFailure));
    assert.deepEqual(await page.locator("canvas").evaluate(canvas => JSON.parse(canvas.dataset.progress)), progressBefore);
  }

  let activeRevision = source.revision;
  async function reload(document) {
    served = structuredClone(document);
    served.revision = activeRevision + 1;
    const loaded = await command("reload", { expected_revision: String(activeRevision), revision: String(served.revision) });
    activeRevision = served.revision;
    assert.equal(loaded.observation.scene.revision, String(activeRevision));
    return loaded;
  }

  // Establish a reset baseline so screenshot differences come from edited pixels.
  await reload(source);
  const baseline = await capture("original");
  const changed = structuredClone(source);
  for (const texture of changed.textures) {
    const pixels = Buffer.from(texture.rgba8, "base64");
    for (let offset = 0; offset < pixels.length; offset += 4) {
      pixels[offset] = 255 - pixels[offset];
      pixels[offset + 1] = 255 - pixels[offset + 1];
      pixels[offset + 2] = 255 - pixels[offset + 2];
    }
    texture.rgba8 = pixels.toString("base64");
  }
  const replaced = await reload(changed);
  assert.equal(replaced.result.rendering.texture_upload_bytes, initialResidency.textureCpu);
  const changedAssets = await command("assets");
  readyTextures(changedAssets, changed);
  assert.deepEqual(residency(changedAssets), initialResidency);
  const changedCapture = await capture("replaced");
  const replacementPixels = await changedPixels(baseline, changedCapture);
  assert.ok(replacementPixels.changed > 100, "Changing only texture pixels must visibly change the rendered canvas.");
  for (let replacement = 0; replacement < 3; replacement++) {
    await reload(changed);
    const next = await command("assets");
    readyTextures(next, changed);
    assert.deepEqual(residency(next), initialResidency, "Repeated replacement must keep live CPU/GPU residency bounded.");
    const warmed = await command("render");
    assert.equal(warmed.result.rendering.texture_upload_bytes, 0);
  }
  const repeatedPixels = await changedPixels(changedCapture, await capture("repeated"));
  // Reallocated mesh handles may reorder ties at shared/coplanar edges. Permit
  // at most one changed pixel per 10,000 while requiring stable scene coverage.
  assert.ok(repeatedPixels.changed <= Math.ceil(repeatedPixels.total / 10_000),
    `Equivalent replacement changed ${repeatedPixels.changed} of ${repeatedPixels.total} pixels.`);

  const untextured = structuredClone(changed);
  untextured.textures = [];
  for (const material of untextured.materials) delete material.texture;
  await reload(untextured);
  const retired = await command("assets");
  assert.equal(retired.result.textures, 0);
  assert.equal(retired.result.texture_cpu_bytes, 0);
  assert.deepEqual(retired.result.texture_assets, []);
  assert.equal(retired.result.rendering.resident_textures, 0);
  assert.equal(retired.result.rendering.resident_texture_bytes, 0);
  assert.equal(retired.result.resident_cpu_bytes, initialResidency.cpu - initialResidency.textureCpu);
  await capture("retired");
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    ok: true, url: url.href, artifact: output, initial: initialResidency,
    invalid: invalidResults, captures: { original: baseline.hash, replaced: changedCapture.hash },
    changedPixels: replacementPixels.changed, repeatedEdgePixels: repeatedPixels.changed,
    checks: ["authored_texture_readiness", "per_id_gpu_residency", "mesh_inspection", "warm_upload_reuse", "invalid_reload_preservation", "changed_pixels_render", "repeated_reload_residency", "texture_retirement", "clean_browser_diagnostics"],
  }));
} finally {
  await browser.close();
}
