/** Texture lifecycle integration against a development host; fixtures stay in routed responses. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { artifacts, commander, gardenScene, sceneUrl, differentPixels, launch, openPaused, residency, routedScene } from "./harness.mjs";

const url = sceneUrl();
const output = await artifacts("texture-smoke");
const { browser, page, errors } = await launch({ width: 1440, height: 1000 });
const command = commander(page);

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

/** Pixels of two captures that differ at all, alpha included. */
const changedPixels = (first, second) => differentPixels(page, [first.png, second.png], { alpha: true });

try {
  const source = gardenScene();
  assert.ok(Number.isSafeInteger(source.revision) && source.revision <= Number.MAX_SAFE_INTEGER - 10);
  assert.ok(source.textures?.length > 0, "Crystal Garden must include imported textured scenery.");
  const player = source.entities.find(entity => entity.tags?.includes("player"))?.id;
  assert.ok(player, "Crystal Garden must include an authored player.");
  const scene = routedScene(page, source);
  await scene.route();
  assert.equal((await openPaused(page, url)).result.paused, true);
  await command("render");
  const initial = await command("assets");
  readyTextures(initial, source);
  const initialResidency = residency(initial);
  assert.ok(initialResidency.textureCpu > 0);
  assert.equal(initialResidency.gpuTextures, initialResidency.textures);
  assert.equal(initialResidency.textureGpu, initialResidency.textureCpu);
  const meshViews = await command("inspect", { component: "Mesh3d", limit: 100 });
  const texturedEntity = meshViews.result.entities.find(entity => entity.components.Mesh3d.textures.some(texture => texture.alive));
  assert.ok(texturedEntity, "A textured material must be used by scenery.");
  const meshView = (await command("inspect", { entity: texturedEntity.id, component: "Mesh3d" })).result.entity.components.Mesh3d;
  // The material reads back as authored, its texture by ID, resident.
  const [texture] = meshView.textures;
  assert.equal(texture.alive, true);
  assert.equal(texture.gpu_resident, true);
  assert.ok(source.textures.some(authored => authored.id === texture.id), `the texture's authored ID: ${texture.id}`);
  assert.equal(meshView.material.texture.id, texture.id);
  assert.ok(source.materials.some(material => material.id === meshView.material.id));
  assert.ok(source.meshes.some(mesh => mesh.id === meshView.mesh.id && mesh.source.kind === meshView.mesh.source.kind), JSON.stringify(meshView.mesh));
  for (let frame = 0; frame < 3; frame++) {
    const warmed = await command("render");
    assert.equal(warmed.result.rendering.texture_upload_bytes, 0, "Warm frames must not upload texture pixels again.");
  }

  // Keep recognizable gameplay state so a rejected replacement cannot hide a reset.
  await command("input", { forward: 1, held: 1 });
  await command("step", { ticks: 6 });
  // What the step brought inspects.
  const input = (await command("inspect", { component: "Input" })).result.value;
  assert.equal(input.move_forward, 1, JSON.stringify(input));
  assert.deepEqual(input.held, [0], JSON.stringify(input));
  await command("input", { forward: 0 });
  const playerBefore = await command("inspect", { entity: player, component: "Transform" });
  const progressBefore = await page.locator("canvas").evaluate(canvas => JSON.parse(canvas.dataset.progress));
  const beforeFailure = await command("assets");
  const invalidResults = [];
  for (const kind of ["base64"]) {
    const invalid = structuredClone(source);
    invalid.revision++;
    invalid.textures[0].rgba8 = "!invalid-base64!";
    scene.serve(invalid);
    const failed = await command("reload", {
      expected_revision: String(source.revision), revision: String(invalid.revision),
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

  // Establish a reset baseline so screenshot differences come from edited pixels.
  await scene.reload(command, source);
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
  const replaced = await scene.reload(command, changed);
  assert.equal(replaced.result.rendering.texture_upload_bytes, initialResidency.textureCpu);
  const changedAssets = await command("assets");
  readyTextures(changedAssets, changed);
  assert.deepEqual(residency(changedAssets), initialResidency);
  const changedCapture = await capture("replaced");
  const replacementPixels = await changedPixels(baseline, changedCapture);
  assert.ok(replacementPixels.changed > 100, "Changing only texture pixels must visibly change the rendered canvas.");
  for (let replacement = 0; replacement < 3; replacement++) {
    await scene.reload(command, changed);
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
  await scene.reload(command, untextured);
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
