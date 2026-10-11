/** Shared rough-glass filtering: invariants, subresources, references and motion. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, sceneUrl, launch, openPaused, routedScene } from "./harness.mjs";
import { observeGlass } from "./rough-glass-observer.mjs";
import { glassReferences } from "./rough-glass-reference.mjs";
import { glassFixture, geometryFixture, mixedFixture, movingFixture } from "./rough-glass-fixtures.mjs";

const output = await artifacts("rough-glass-smoke");
const head = execFileSync("git", ["rev-parse", "HEAD"], { cwd: new URL("../..", import.meta.url), encoding: "utf8" }).trim();
const dirty = execFileSync("git", ["status", "--porcelain"], { cwd: new URL("../..", import.meta.url), encoding: "utf8" }).trim();
const { browser, page, errors } = await launch({ width: 1500, height: 1000, timeout: 60_000 });
await page.addInitScript(observeGlass);
const command = commander(page, { deadline: 60_000 });
let fixture = glassFixture({ constant: true, blur: 0 });
let width = 1400, height = 900;
const scene = routedScene(page, fixture);
const checks = [], captures = [], ordinary = [];
const json = value => JSON.parse(JSON.stringify(value));
const style = ".viewport > :not(canvas) { visibility: hidden !important; }";

function expectedLevels(w, h, blur, active = true) {
  let variance = 0, level = 0, texels = 0;
  const dimensions = [], available = Math.floor(Math.log2(Math.max(w, h)));
  const wanted = active && blur > 0.5 ? 5 / 18 * (blur * blur - 0.25) : 0;
  while (variance < wanted && level < available) {
    variance = 4 * variance + 1.25; level++;
    const dimensionsAtLevel = [Math.max(1, w >> level), Math.max(1, h >> level)];
    dimensions.push(dimensionsAtLevel); texels += dimensionsAtLevel[0] * dimensionsAtLevel[1];
  }
  return { prefilter_levels: level, prefilter_texels: texels, prefilter_bytes: texels * 4, dimensions };
}

async function size(w, h) {
  width = w; height = h;
  await page.evaluate(([w, h]) => {
    const canvas = document.querySelector("canvas");
    canvas.parentElement.style.cssText = `position:fixed;left:0;top:0;width:${w}px;height:${h}px;overflow:visible;border:0;border-radius:0;z-index:1000`;
    canvas.style.cssText = `position:fixed;left:0;top:0;width:${w}px!important;height:${h}px!important;z-index:1000`;
    window.dispatchEvent(new Event("resize"));
  }, [w, h]);
  // The paused host records CSS bounds immediately, but only a render
  // command applies those dimensions to the engine's canvas and targets.
  await command("render");
  await page.waitForFunction(([w, h]) => {
    const canvas = document.querySelector("canvas"); return canvas.width === w && canvas.height === h;
  }, [w, h]);
}

async function replace(next) {
  fixture = structuredClone(next);
  await scene.reload(command, fixture);
  await command("render");
}

async function blur(value, strength = 1) {
  fixture.environment.refraction = { strength, blur: value };
  await command("environment", { refraction: fixture.environment.refraction });
}

function verifyTrace(record, wanted, active = true) {
  const { rendering, trace } = record;
  assert.equal(rendering.refraction.active, active, `${record.name}: refraction eligibility`);
  for (const key of ["prefilter_levels", "prefilter_texels", "prefilter_bytes"]) {
    assert.equal(rendering.refraction[key], wanted[key], `${record.name}: ${key}`);
  }
  const reductions = trace.passes.filter(pass => pass.label === "engine/refraction-prefilter");
  assert.equal(reductions.length, wanted.prefilter_levels, `${record.name}: one pass per lower mip`);
  const live = trace.textures.filter(texture => !texture.destroyed && texture.label === "engine/post-scene");
  assert.ok(live.length <= 1, `${record.name}: no old Scene texture retained`);
  const current = live[0];
  if (active) {
    assert.ok(current, `${record.name}: Scene exists`);
    assert.equal(current.levels, wanted.prefilter_levels + 1);
    assert.deepEqual([current.width, current.height], record.resolution);
    assert.equal(current.format, "rgba8unorm-srgb");
  }
  for (let i = 0; i < reductions.length; i++) {
    const pass = reductions[i], destination = pass.attachments[0], source = pass.groups[0][4];
    assert.deepEqual(destination, { texture: current.id, base: i + 1, levels: 1 });
    assert.deepEqual(source, { texture: current.id, base: i, levels: 1 });
  }
  for (const pass of trace.passes) {
    for (const attachment of pass.attachments) if (attachment && trace.textures[attachment.texture].label === "engine/post-scene") {
      assert.equal(attachment.texture, current.id, `${record.name}: current attachment generation`);
      assert.equal(attachment.levels, 1, `${record.name}: attachments expose a single mip`);
      if (pass.label !== "engine/refraction-prefilter") assert.equal(attachment.base, 0);
    }
    for (const binding of pass.bindings) for (const view of Object.values(binding.views)) {
      const texture = trace.textures[view.texture];
      assert.equal(texture.destroyed, false, `${record.name}: no retired texture is read`);
      if (texture.label === "engine/post-scene") {
        assert.equal(view.texture, current.id, `${record.name}: all readers use current generation`);
        if (pass.label !== "engine/transparent-accumulation") assert.equal(view.levels, 1);
      }
    }
  }
  if (active) {
    const transparent = trace.passes.find(pass => pass.label === "engine/transparent-accumulation");
    assert.ok(transparent?.bindings.some(binding => binding.views[8]?.texture === current.id
      && binding.views[8].base === 0 && binding.views[8].levels === current.levels), `${record.name}: glass binds the full current chain`);
    assert.equal(trace.sampler.mipmapFilter, "linear");
  }
  record.expected = wanted;
  // Keep the pre-existing approximate total-accounting formula separate
  // from this change's exact logical Scene delta and from driver heap use.
  const pixels = record.resolution[0] * record.resolution[1];
  record.target_accounting = { exact_extra_scene_bytes: wanted.prefilter_bytes,
    existing_post_formula_plus_extra: current ? pixels * 10 + Math.floor(pixels / 4) * 24 + Math.floor(pixels / 16) * 4
      + Math.floor(Math.floor(pixels / 4) * 8 * 4 / 3) * 2 + wanted.prefilter_bytes : 0,
    note: "Formula excludes the separately retained reflection-depth pyramid; actual renderer memory.gpu_post_bytes is recorded by native profiling, not inferred as a heap allocation here." };
}

async function capture(name, { active = true, check = true } = {}) {
  await command("render");
  await page.evaluate(() => window.__roughGlass.reset());
  const rendering = json((await command("capture")).result.rendering);
  const trace = await page.evaluate(() => window.__roughGlass.snapshot());
  const shaderHashes = await page.evaluate(async () => {
    const state = window.__roughGlass, hashes = {};
    if (!state.post || !state.mesh) throw new Error("rough-glass assertion: served shaders do not contain the assigned implementation");
    for (const key of ["post", "mesh"]) {
      const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(state[key]));
      hashes[key] = [...new Uint8Array(hash)].map(value => value.toString(16).padStart(2, "0")).join("");
    }
    return hashes;
  });
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png`, style });
  const resolution = await page.locator("canvas").evaluate(canvas => [canvas.width, canvas.height]);
  assert.deepEqual(resolution, [width, height]);
  assert.ok(Number.isInteger(rendering.frame) && Number.isInteger(rendering.tick));
  const record = { name, head, dirty, shader_hashes: shaderHashes, adapter: trace.adapter, resolution, frame: rendering.frame, tick: rendering.tick,
    fixture: structuredClone(fixture), scene_revision: scene.revision, rendering, trace };
  if (check) verifyTrace(record, expectedLevels(width, height, fixture.environment.refraction?.blur ?? 0, active), active);
  captures.push(record);
  await writeFile(`${output}/${name}.json`, JSON.stringify(record, null, 2));
  return { record, png: bytes.toString("base64") };
}

async function compare(a, b, crop = [0, 0, 1, 1]) {
  return page.evaluate(async ({ images, crop }) => {
    const pixels = await Promise.all(images.map(async png => {
      const image = new Image(); image.src = `data:image/png;base64,${png}`; await image.decode();
      const canvas = document.createElement("canvas"); canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext("2d"); context.drawImage(image, 0, 0); return context.getImageData(0, 0, image.width, image.height);
    }));
    const [a, b] = pixels;
    if (a.width !== b.width || a.height !== b.height) throw new Error("rough-glass assertion: capture dimensions changed");
    const x0 = Math.floor(a.width * crop[0]), y0 = Math.floor(a.height * crop[1]);
    const x1 = Math.min(a.width, Math.ceil(a.width * (crop[0] + crop[2]))), y1 = Math.min(a.height, Math.ceil(a.height * (crop[1] + crop[3])));
    let maximum = 0, sum = 0, square = 0, samples = 0, changed = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      let difference = false;
      for (let c = 0; c < 3; c++) {
        const index = (y * a.width + x) * 4 + c, delta = Math.abs(a.data[index] - b.data[index]);
        maximum = Math.max(maximum, delta); sum += delta; square += (delta / 255) ** 2; samples++;
        difference ||= delta > 2;
      }
      changed += Number(difference);
    }
    return { maximum_u8: maximum, mean_absolute_u8: sum / samples, encoded_rgb_mse: square / samples,
      pixels_over_2: changed, pixels: samples / 3, crop };
  }, { images: [a.png, b.png], crop });
}

try {
  await scene.route(); await openPaused(page, sceneUrl()); await size(width, height);
  const sharp = await capture("constant-sharp");
  await blur(0.49); const subpixel = await capture("constant-subpixel");
  assert.equal((await compare(sharp, subpixel)).maximum_u8, 0, "blur zero and .49 are identical");
  await blur(12); const constant12 = await capture("constant-12");
  await blur(64); const constant64 = await capture("constant-64");
  for (const capture of [constant12, constant64]) {
    const delta = await compare(sharp, capture, [0.35, 0.35, 0.3, 0.3]);
    assert.ok(delta.maximum_u8 <= 2, `constant color is preserved: ${JSON.stringify(delta)}`);
    checks.push({ check: "constant energy", name: capture.record.name, delta });
  }
  assert.deepEqual(constant12.record.expected, { prefilter_levels: 4, prefilter_texels: 418222, prefilter_bytes: 1672888,
    dimensions: [[700, 450], [350, 225], [175, 112], [87, 56]] });
  assert.equal(constant64.record.expected.prefilter_texels, 419720);
  assert.equal(constant64.record.expected.prefilter_bytes, 1678880);
  for (const [w, h] of [[701, 451], [1, 1], [1, 257]]) {
    await size(w, h); await blur(0); const baseline = await capture(`constant-${w}x${h}-sharp`);
    await blur(64); const filtered = await capture(`constant-${w}x${h}-filtered`);
    const delta = await compare(baseline, filtered);
    assert.ok(delta.maximum_u8 <= 2, `odd/thin targets have no unwritten band: ${JSON.stringify(delta)}`);
    checks.push({ check: "dimensions", resolution: [w, h], expected: filtered.record.expected, delta });
  }
  await size(701, 451);
  await replace(glassFixture({ blur: 0 }));
  const patternedSharp = await capture("patterned-sharp");
  await blur(0.49); const patternedSubpixel = await capture("patterned-subpixel");
  assert.equal((await compare(patternedSharp, patternedSubpixel)).maximum_u8, 0);
  const boundaries = [0.5];
  for (let variance = 1.25; variance < 1138; variance = 4 * variance + 1.25) boundaries.push(Math.sqrt(variance * 18 / 5 + 0.25));
  for (const boundary of boundaries) {
    await blur(boundary - 0.0001); const below = await capture(`boundary-${boundary.toFixed(5)}-below`);
    await blur(boundary + 0.0001); const above = await capture(`boundary-${boundary.toFixed(5)}-above`);
    const delta = await compare(below, above);
    assert.ok(delta.maximum_u8 <= 2, `continuous blur at ${boundary}: ${JSON.stringify(delta)}`);
    checks.push({ check: "scale boundary", boundary, delta });
  }
  for (const radius of [2, 4.2, 8, 12, 64]) { await blur(radius); await capture(`patterned-radius-${radius}`); }

  const activePipelineCount = captures.at(-1).trace.pipelines;
  await blur(12, 0); const disabled = await capture("strength-zero", { active: false });
  assert.equal(disabled.record.trace.textures.filter(texture => !texture.destroyed && texture.label === "engine/post-scene").length, 0,
    "refraction and other effects off retire Scene and every feature-owned mip");
  const omittedFixture = glassFixture(); delete omittedFixture.environment.refraction;
  await replace(omittedFixture); const omitted = await capture("resource-omitted", { active: false });
  assert.equal((await compare(disabled, omitted)).maximum_u8, 0, "omission and strength zero have unchanged pixels");
  await replace(glassFixture({ glass: false })); await capture("no-transmitting-batch", { active: false });
  const changed = geometryFixture("foreground-edge");
  changed.materials.find(material => material.id === "background").emissive = [0.6, 0.05, 0.12];
  changed.camera.position = [0.5, 0.25, 8];
  await size(733, 477); await replace(changed);
  const restored = await capture("reenabled-changed-scene");
  assert.ok(restored.record.trace.pipelines > activePipelineCount, "off releases the reduction pipeline, so re-enable creates it");
  // Reload the page for a fresh renderer with exactly the final authored
  // scene/camera, preserving the viewport size and paused simulation time.
  scene.serve({ ...changed, revision: scene.revision });
  await openPaused(page, sceneUrl()); await size(733, 477);
  const fresh = await capture("fresh-changed-scene");
  const lifecycle = await compare(restored, fresh);
  assert.ok(lifecycle.maximum_u8 <= 2, `fresh and reused renderer agree: ${JSON.stringify(lifecycle)}`);
  checks.push({ check: "lifecycle", delta: lifecycle });

  for (const kind of ["tilted-thick", "screen-edge", "foreground-edge"]) {
    await replace(geometryFixture(kind)); await capture(kind);
  }
  const references = await page.evaluate(glassReferences);
  const referenceMetadata = { head, dirty, shader_hashes: captures.at(-1).shader_hashes,
    adapter: fresh.record.adapter, frame: captures.at(-1).frame, tick: captures.at(-1).tick,
    fixture: "production reduction and lookup in a fixed single-pane composition", source_domain: "decoded Rgba8UnormSrgb graded Scene" };
  for (const image of references.images) for (const arm of ["pyramid", "gaussian", "old_ring"]) {
    await writeFile(`${output}/reference-${image.name}-${arm}.png`, Buffer.from(image[arm], "base64"));
  }
  delete references.images;
  references.metadata = referenceMetadata;
  references.pyramid_worse_rows = references.rows.filter(row => row.pyramid_mse > row.old_ring_mse);
  await writeFile(`${output}/references.json`, JSON.stringify(references, null, 2));
  checks.push({ check: "spatial references", rows: references.rows.length, worse_rows: references.pyramid_worse_rows.length,
    note: references.note, artifact: "references.json" });

  await replace(mixedFixture());
  let settled;
  for (let i = 0; i < 600; i++) {
    settled = (await command("render")).result.rendering;
    if (settled.terrain.tiles > 0 && settled.terrain.loading_tiles === 0 && settled.terrain.fallback_patches === 0) break;
    await delay(25);
  }
  assert.ok(settled.terrain.tiles > 0 && settled.terrain.loading_tiles === 0 && settled.terrain.fallback_patches === 0, "mixed fixture terrain settles");
  await command("step", { ticks: 60 });
  const mixed = await capture("mixed-surfaces");
  for (const [kind, count] of [["water", mixed.record.rendering.water.chunks], ["clouds", mixed.record.rendering.clouds.view_blocks],
    ["decals", mixed.record.rendering.decals.drawn], ["particles", mixed.record.rendering.particles.drawn]]) assert.ok(count > 0, `${kind} draws in mixed fixture`);
  const labels = mixed.record.trace.passes.map(pass => pass.label), firstFilter = labels.indexOf("engine/refraction-prefilter");
  const transparent = labels.indexOf("engine/transparent-accumulation"), composite = labels.indexOf("engine/transparent-composite");
  for (const producer of ["engine/water", "engine/decals", "engine/cloud-composite"]) {
    assert.ok(labels.indexOf(producer) >= 0 && labels.indexOf(producer) < firstFilter, `${producer} precedes filtering`);
  }
  assert.ok(firstFilter < transparent && transparent < composite);
  assert.ok(labels.indexOf("engine/post-scene-levels") > composite, "reflections rebuild their separate chain after transparency");
  const reflectionPass = mixed.record.trace.passes.find(pass => pass.label === "engine/post-scene-levels");
  const reflectionTexture = mixed.record.trace.textures[reflectionPass.attachments[0].texture];
  assert.equal(reflectionTexture.label, "engine/post-scene-levels");
  checks.push({ check: "mixed pass ordering", labels, rendering: mixed.record.rendering });
  for (const section of ["water", "decals", "clouds", "particles"]) {
    const enabled = structuredClone(fixture.environment[section]);
    fixture.environment[section] = { ...enabled, enabled: false };
    await command("environment", { [section]: fixture.environment[section] });
    const off = await capture(`mixed-no-${section}`), delta = await compare(mixed, off);
    assert.ok(delta.mean_absolute_u8 > 0, `${section} contributes visible pixels in the mixed fixture`);
    checks.push({ check: "mixed visible contribution", section, delta });
    fixture.environment[section] = enabled;
    await command("environment", { [section]: enabled });
  }

  await size(701, 451); await replace(movingFixture());
  let previousOrdinary;
  for (const radius of [4.2, boundaries[2] - 0.0001, boundaries[2] + 0.0001, 12]) {
    await blur(radius);
    const start = (await command("status")).result.rendering.frame;
    await command("resume");
    const until = Date.now() + 60_000;
    let status;
    do { await delay(50); status = await command("status"); }
    while (status.result.rendering.frame - start < 32 && Date.now() < until);
    assert.ok(status.result.rendering.frame - start >= 32, "ordinary presentation advances at least 32 frames");
    assert.equal(status.result.rendering.temporal_antialiasing.active, true, "ordinary frames use TAA");
    const first = { png: (await page.locator("canvas").screenshot({ path: `${output}/moving-${radius}-a.png`, style })).toString("base64") };
    const firstStats = json((await command("status")).result.rendering);
    await delay(150);
    const second = { png: (await page.locator("canvas").screenshot({ path: `${output}/moving-${radius}-b.png`, style })).toString("base64") };
    const secondStats = json((await command("status")).result.rendering);
    await command("pause");
    const row = { head, dirty, adapter: referenceMetadata.adapter, resolution: [width, height], fixture: structuredClone(fixture),
      radius, started_frame: start, first: firstStats, second: secondStats, delta: await compare(first, second),
      scale_transition: previousOrdinary ? await compare(previousOrdinary, first) : null,
      resources: await page.evaluate(() => { const state = window.__roughGlass.snapshot(); return { textures: state.textures, pipelines: state.pipelines }; }),
      note: "Ordinary moving frames, not exact captures. Each screenshot is bracketed by ordinary-frame stats; acquisition can span several frames." };
    assert.ok(firstStats.frame - start >= 32 && secondStats.tick > firstStats.tick);
    assert.ok(row.delta.mean_absolute_u8 > 0, "ordinary images include actual animated motion");
    previousOrdinary = second;
    ordinary.push(row);
  }
  await writeFile(`${output}/ordinary-frames.json`, JSON.stringify(ordinary, null, 2));
  assert.deepEqual(errors, [], "no browser or WebGPU errors");
  await writeFile(`${output}/report.json`, JSON.stringify({ ok: true, head, dirty, checks, captures, ordinary }, null, 2));
  console.log(JSON.stringify({ ok: true, head, output, captures: captures.length, checks, ordinary_frames: ordinary.length }));
} catch (error) {
  await writeFile(`${output}/failure.json`, JSON.stringify({ head, dirty, error: String(error), stack: error.stack, errors, checks, captures, ordinary }, null, 2));
  console.error(error);
  process.exitCode = error instanceof assert.AssertionError || String(error).includes("rough-glass assertion:") ? 3 : 1;
} finally {
  await browser.close();
}
