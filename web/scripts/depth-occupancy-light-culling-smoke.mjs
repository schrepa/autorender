/** Depth occupancy: exact GPU interval probes and requested production masks,
 * selected lists, phase identities, switching, budgets and matched captures. */
// @ts-check
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, sceneUrl, differentPixels, launch, openPaused, routedScene } from "./harness.mjs";
import { value, checkPrimitives } from "./depth-occupancy-oracle.mjs";
import { checkShadingGuards } from "./depth-occupancy-guard.mjs";
import { exerciseVariants, exerciseTransitions, exerciseOcclusion } from "./depth-occupancy-fixtures.mjs";
import { auditSources } from "./depth-occupancy-audit.mjs";

const output = await artifacts(`depth-occupancy-light-culling-smoke-${Date.now()}`);
const { browser, page, errors } = await launch({ width: 1280, height: 720 });
const command = commander(page);
const cluster = 12 * 144 + 4 * 16 + 8;
const fixture = {
  id: "depth-occupancy-fixture", revision: 1,
  camera: { position: [0, 0, 0], target: [0, 0, -1], fov_degrees: 55, near: 0.1, far: 100 },
  materials: [{ id: "plain", color: [0.7, 0.5, 0.3, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  environment: { local_lights: { depth_occupancy: false, distance: 100 }, occlusion_culling: { enabled: false } },
  entities: [
    ...[10.2, 11.9].map((depth, i) => ({ id: `patch-${i}`, mesh: "box", material: "plain",
      transform: { translation: [0.2, 0, -depth], scale: [0.35, 0.6, 0.02] } })),
    ...Array.from({ length: 76 }, (_, i) => ({ id: `light-${i}`,
      light: { range: i % 17 === 0 ? 0.12 : 0.01, intensity: 0.015, color: [1, 0.6, 0.3] },
      transform: { translation: [0.2, 0, i % 17 === 0 ? -10.15 : -11.05] } })),
    { id: "far-light", light: { range: 0.12, intensity: 0.015, spot: { direction: [0, 0, -1] } }, transform: { translation: [0.2, 0, -11.85] } },
    { id: "small-light", light: { range: 2 ** -22, intensity: 0.01 }, transform: { translation: [0.2, 0, -11.05] } },
    { id: "boundary-light", light: { range: 0.012, intensity: 0.001 }, transform: { translation: [0.2, 0, -10.195] } },
  ],
};
const scene = routedScene(page, fixture);
const checks = [], measurements = [];

async function render(gpu = false) {
  const reply = gpu ? await command("present", { alpha: 0.5, gpu: true }) : await command("render");
  return JSON.parse(JSON.stringify(reply.result.rendering));
}

async function productionPage(phase, selected = cluster, offset = 0, allowed_bytes, gpu = false) {
  let pending;
  for (let attempt = 0; attempt < 50; attempt++) {
    const reply = await command("light_depth", { phase, cluster: selected, offset, ...(allowed_bytes === undefined ? {} : { allowed_bytes }) });
    pending = reply.result.rendering.local_lights;
    if (reply.result.page) return reply.result.page;
    await render(gpu);
    await delay(100);
  }
  assert.fail(`phase ${phase}, cluster ${selected}, offset ${offset}: production readback did not arrive: ${JSON.stringify(pending)}`);
}

async function selectedLists(phase = 0, gpu = false) {
  const head = await productionPage(phase, cluster, 0, undefined, gpu);
  const original = [], filtered = [];
  for (let offset = 0; offset < Math.max(head.original_length, head.filtered_length); offset += 128) {
    const part = offset === 0 ? head : await productionPage(phase, cluster, offset, undefined, gpu);
    assert.equal(part.mask, head.mask, "unchanged content has the same occupied bins across pages");
    assert.equal(part.original_length, head.original_length);
    original.push(...part.original); filtered.push(...part.filtered);
  }
  assert.equal(original.length, head.original_length);
  assert.equal(filtered.length, head.filtered_length);
  let previous = -1;
  for (const id of filtered) {
    const at = original.findIndex(light => light.id === id);
    assert.ok(at > previous, "filtered indices are a stable subsequence of the selected full list");
    previous = at;
  }
  return { ...head, original, filtered };
}

async function capture(name) {
  await command("capture");
  return (await page.locator("canvas").screenshot({ path: `${output}/${name}.png` })).toString("base64");
}

async function compareImages(name, left, right) {
  const difference = await differentPixels(page, [left, right], { tolerance: 0 });
  measurements.push({ name: `${name}:changed_pixels`, value: difference.changed, unit: "pixels" });
  measurements.push({ name: `${name}:changed_pixel_share`, value: difference.changed / difference.total, unit: "fraction" });
  const error = await page.evaluate(async images => {
    const pixels = await Promise.all(images.map(async data => {
      const image = new Image(); image.src = `data:image/png;base64,${data}`; await image.decode();
      const canvas = document.createElement("canvas"); canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext("2d"); if (!context) throw new Error("No 2D context");
      context.drawImage(image, 0, 0); return context.getImageData(0, 0, canvas.width, canvas.height).data;
    }));
    let sum = 0, maximum = 0;
    for (let i = 0; i < pixels[0].length; i++) { if (i % 4 === 3) continue; const delta = Math.abs(pixels[0][i] - pixels[1][i]); sum += delta; maximum = Math.max(maximum, delta); }
    return { mean: sum / (pixels[0].length * 3 / 4), maximum };
  }, [left, right]);
  measurements.push({ name: `${name}:mean_absolute_error`, value: error.mean, unit: "channel_levels" },
    { name: `${name}:max_absolute_error`, value: error.maximum, unit: "channel_levels" });
}

try {
  await scene.route();
  await openPaused(page, sceneUrl());
  await writeFile(`${output}/fixture.scene.json`, JSON.stringify(fixture));
  await writeFile(`${output}/source-audit.json`, JSON.stringify(await auditSources()));
  await writeFile(`${output}/browser-build.json`, JSON.stringify({ browser: browser.version(), status: (await command("status")).result }));
  const primitive = await checkPrimitives(page);
  await writeFile(`${output}/primitive-gpu.json`, JSON.stringify(primitive));
  checks.push({ check: "exact_gpu_intervals", cases: primitive.cases, contributors: primitive.contributors, hash: primitive.source_sha256 });
  const offStats = await render();
  assert.equal(offStats.local_lights.depth_occupancy, undefined, "default-off observations omit the feature");
  assert.ok(!offStats.render_passes.some(name => name.startsWith("light_depth")));
  const off = await capture("gap-off");
  await command("environment", { merge: true, local_lights: { depth_occupancy: true } });
  const onStats = await render();
  assert.equal(onStats.local_lights.depth_occupancy.active, true);
  const lists = await selectedLists();
  await writeFile(`${output}/production-gap.json`, JSON.stringify(lists));
  const guards = await checkShadingGuards(page, lists.full_routes);
  await writeFile(`${output}/fragment-guards.json`, JSON.stringify(guards));
  checks.push({ check: "production_fragment_guards", cases: guards.cases.length, source: guards.source_sha256 });
  assert.ok(lists.mask !== 0 && lists.original_length > 64, "occupied cluster crosses a 64-entry compaction chunk");
  assert.ok(lists.filtered_length < lists.original_length, "gap lights are rejected inside an occupied original cluster");
  assert.equal(lists.full_generation, 0, "full-list consumers use the zero-tag guard");
  assert.equal(lists.arena_bytes, Math.ceil((5 * 3456 + 31 * 24 + 8 + lists.index_capacity) * 4 / 16) * 16);
  const nearBin = lists.threshold_bits.filter(word => value(word) <= Math.fround(10.19)).length;
  const farBin = lists.threshold_bits.filter(word => value(word) <= Math.fround(11.89)).length;
  const gapBin = lists.threshold_bits.filter(word => value(word) <= 11.05).length;
  assert.notEqual(nearBin, farBin);
  assert.ok((lists.mask >>> nearBin) & 1, "front patch collected");
  assert.ok((lists.mask >>> farBin) & 1, "hidden patch collected with depth Always");
  assert.equal((lists.mask >>> gapBin) & 1, 0, "the interior gap stays empty");
  for (const light of lists.original) {
    const radius = value(light.range_bits);
    if (radius > 0.1 || radius < 2 ** -20 || Math.abs(value(light.position_bits[2]) + 11.05) > 0.02) assert.ok(lists.filtered.includes(light.id), "contributors and out-of-domain lights remain");
    else assert.ok(!lists.filtered.includes(light.id), "in-domain gap light removed");
  }
  await writeFile(`${output}/production-gap.json`, JSON.stringify(lists));
  const on = await capture("gap-on");
  await compareImages("gap-off-on", off, on);
  const smallBudget = await productionPage(0, cluster, 0, lists.arena_bytes - 1);
  const exactBudget = await productionPage(0, cluster, 0, lists.arena_bytes);
  assert.equal(smallBudget.allocation_fits, false);
  assert.equal(exactBudget.allocation_fits, true);
  const empty = await productionPage(0, 0);
  assert.equal(empty.mask, 0);
  assert.equal(empty.filtered_length, 0);
  checks.push({ check: "production_gap", original: lists.original_length, kept: lists.filtered_length, mask: lists.mask, arena: lists.arena_bytes });

  for (const budget of [1, 65]) {
    await command("environment", { merge: true, local_lights: { lights_per_cluster: budget } });
    const limited = await selectedLists();
    assert.ok(limited.original_length <= budget, "filtering preserves finite selected budgets");
    checks.push({ check: "budget", budget, selected: limited.original_length, kept: limited.filtered_length });
  }
  await command("environment", { merge: true, local_lights: { lights_per_cluster: null } });
  await command("environment", { merge: true, occlusion_culling: { enabled: true } });
  for (let i = 0; i < 3; i++) { await render(true); await delay(100); }
  const phaseZero = await productionPage(0, cluster, 0, undefined, true);
  const phaseOne = await productionPage(1, cluster, 0, undefined, true);
  assert.notEqual(phaseZero.generation, phaseOne.generation);
  assert.equal(phaseZero.phase, 0); assert.equal(phaseOne.phase, 1);
  checks.push({ check: "opaque_phases", first: phaseZero, second: phaseOne });

  await command("environment", { merge: true, local_lights: { depth_occupancy: false } });
  const disabled = await render();
  assert.equal(disabled.local_lights.depth_occupancy, undefined);
  assert.ok(!disabled.render_passes.some(name => name.startsWith("light_depth")));
  assert.equal((await command("light_depth", { phase: 0, cluster })).result.page, null);
  const offAgain = await capture("gap-off-again");
  await compareImages("gap-off-off", off, offAgain);
  await command("environment", { local_lights: { depth_occupancy: true } });
  await command("environment", { local_lights: {} });
  assert.equal((await command("inspect", { component: "LocalLightSettings" })).result.value.depth_occupancy, false, "replacement restores the default");
  const badType = await command("environment", { local_lights: { depth_occupancy: "on" } }, false);
  assert.equal(badType.error.path, "$.local_lights.depth_occupancy");
  const badField = await command("environment", { local_lights: { depth_occupancies: true } }, false);
  assert.equal(badField.error.code, "unknown_field");
  const context = { page, command, scene, fixture, cluster, render, productionPage, capture, compareImages, checks };
  await exerciseOcclusion(context);
  await exerciseVariants(context);
  await exerciseTransitions(context);
  const omitted = structuredClone(fixture);
  Reflect.deleteProperty(omitted.environment.local_lights, "depth_occupancy");
  scene.serve(omitted);
  for (const [query, expected] of [[{}, false], [{ "depth-occupancy": "on" }, true], [{ "depth-occupancy": "off" }, false]]) {
    await openPaused(page, sceneUrl(query));
    assert.equal((await command("inspect", { component: "LocalLightSettings" })).result.value.depth_occupancy, expected);
  }

  assert.deepEqual(errors, []);
  await writeFile(`${output}/results.json`, JSON.stringify({ checks, measurements }));
  console.log(JSON.stringify({ ok: true, output, checks, measurements }));
} finally {
  await browser.close();
}
