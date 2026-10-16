/** Additional authored coverage and resource-lifetime cases for depth masks. */
// @ts-check
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { waitFrames } from "./harness.mjs";
import { value } from "./depth-occupancy-oracle.mjs";

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const vertices = [[-0.2, 0, 0], [0.2, 0, 0], [0.2, 0.5, 0], [-0.2, 0.5, 0]].map(position => ({ position, normal: [0, 0, 1], uv: [0, 0] }));

export async function exerciseVariants({ page, command, scene, fixture, cluster, render, productionPage, capture, compareImages, checks }) {
  const extra = structuredClone(fixture);
  extra.environment.local_lights.depth_occupancy = true;
  Object.assign(extra.environment, { refraction: {}, decals: { enabled: true } });
  extra.materials.push({ id: "mask", color: [1, 1, 1, 1], opacity: 0.1, alpha_mode: "mask", alpha_cutoff: 0.5, double_sided: true },
    { id: "two-sided", color: [0.2, 0.6, 0.8, 1], double_sided: true },
    { id: "glass", color: [0.9, 0.95, 1, 1], transmission: 1, thickness: 0.1, ior: 1.4 });
  extra.meshes.push(
    { id: "extended", source: { kind: "inline", vertices: vertices.map(vertex => ({ ...vertex, color: [0.8, 0.7, 0.6, 1], uv1: [0.1, 0.2] })), indices: [2, 1, 0, 3, 2, 0] } },
    { id: "skinned", source: { kind: "inline", vertices: vertices.map(vertex => ({ ...vertex, joints: [0, 0, 0, 0], weights: [1, 0, 0, 0] })), indices: [0, 1, 2, 0, 2, 3] } });
  extra.skins = [{ id: "rig", joints: ["bone"], inverse_binds: [identity] }];
  extra.animations = [{ id: "move", targets: ["bone", "moving-light"], channels: [
    { target: 0, property: "translation", times: [0, 1], values: [0.2, -0.2, -10.5, 0.6, -0.2, -10.3] },
    { target: 1, property: "translation", times: [0, 1], values: [0.2, 0.2, -10.3, 0.6, 0.2, -10.1] },
  ] }];
  extra.entities.push(
    { id: "masked-gap", mesh: "box", material: "mask", transform: { translation: [0.2, 0, -11.05], scale: [0.3, 0.4, 0.01] } },
    { id: "extended-patch", mesh: "extended", material: "two-sided", transform: { translation: [-0.2, -0.2, -10.8], scale: [1.3, 0.7, 1] } },
    { id: "bone", animation: { clip: "move", speed: 0, loop: false }, transform: { translation: [0.2, -0.2, -10.5] } },
    { id: "skin", mesh: "skinned", material: "two-sided", skin: "rig" },
    { id: "moving-light", mesh: "box", material: "plain", light: { range: 0.4, intensity: 0.01 }, transform: { translation: [0.2, 0.2, -10.3], scale: [0.03, 0.03, 0.03] } },
    { id: "glass", mesh: "box", material: "glass", transform: { translation: [-0.4, 0, -10], scale: [0.2, 0.4, 0.1] } },
    { id: "decal", decal: { material: "plain" }, transform: { translation: [0.2, 0, -10.19], scale: [0.2, 0.2, 0.1] } },
    { id: "probe", reflection_probe: { extent: [3, 3, 3] }, transform: { translation: [0, 0, -10] } });
  await scene.reload(command, extra);
  const drawn = await render();
  assert.equal(drawn.local_lights.depth_occupancy.active, true);
  assert.ok(drawn.transparent_draws > 0 && drawn.decals.drawn > 0, "glass and decals use their normal paths");
  const skin = (await command("inspect", { entity: "skin", component: "Mesh3d" })).result.entity.components.Mesh3d;
  assert.equal(skin.mesh.rendered.skinned, true);
  const alpha = await productionPage(0);
  assert.ok(alpha.mask !== 0);
  const alphaBin = alpha.threshold_bits.filter(word => value(word) <= 11.05).length;
  assert.ok((alpha.mask >>> alphaBin) & 1, "omitting alpha discards conservatively collects the masked gap patch");
  await command("environment", { merge: true, local_lights: { depth_occupancy: false } });
  const off = await capture("variants-off");
  await command("environment", { merge: true, local_lights: { depth_occupancy: true } });
  const on = await capture("variants-on");
  await compareImages("variants-off-on", off, on);
  await command("animation", { entity: "bone", time: 1, speed: 0 });
  await command("step", { ticks: 1 });
  await command("present", { alpha: 0.5, gpu: true });
  await delay(100);
  const moved = await productionPage(0, cluster, 0, undefined, true);
  assert.ok(moved.generation > alpha.generation, "animated skin and displayed light poses get fresh phase tags");
  const beforeSize = await render();
  await page.setViewportSize({ width: 1096, height: 768 });
  await command("resume");
  const resizeStatus = await waitFrames(command, 2, { done: status => status.result.rendering.view_width !== beforeSize.view_width });
  await command("pause");
  const resized = resizeStatus.result.rendering;
  assert.notEqual(resized.view_width, beforeSize.view_width);
  const resizedPage = await productionPage(0);
  assert.ok(resizedPage.generation > moved.generation);
  await command("environment", { temporal_antialiasing: { enabled: true } });
  await command("resume");
  const temporal = await waitFrames(command, 5, { done: status => status.result.rendering.temporal_antialiasing.active });
  await command("pause");
  assert.equal(temporal.result.rendering.local_lights.depth_occupancy.active, true);
  checks.push({ check: "coverage_variants", extended: true, skinned: true, masked: true, two_sided: true, glass: drawn.transparent_draws,
    decals: drawn.decals.drawn, probe_faces: drawn.reflection_probes.faces, before: alpha.generation, moved: moved.generation, resized: resizedPage.generation, temporal: temporal.result.rendering.local_lights.depth_occupancy });
}

export async function exerciseOcclusion({ command, scene, fixture, cluster, render, productionPage, checks }) {
  const hidden = structuredClone(fixture);
  hidden.environment.local_lights.depth_occupancy = true;
  hidden.environment.occlusion_culling.enabled = true;
  hidden.entities[0].transform.scale = [0.8, 1.2, 0.02];
  hidden.entities[1].transform.scale = [0.18, 0.3, 0.02];
  hidden.entities[0].animation = { clip: "uncover", speed: 0, loop: false };
  hidden.animations = [{ id: "uncover", targets: ["patch-0"], channels: [
    { target: 0, property: "translation", times: [0, 1], values: [0.2, 0, -10.2, 2, 0, -10.2] },
  ] }];
  await scene.reload(command, hidden);
  for (let i = 0; i < 4; i++) { await render(true); await delay(100); }
  const initial = await productionPage(1, cluster, 0, undefined, true);
  await command("animation", { entity: "patch-0", time: 1, speed: 0 });
  await command("step", { ticks: 1 });
  const revealed = await render(true);
  let current;
  for (let i = 0; i < 50; i++) {
    await delay(100);
    current = (await command("light_depth", { phase: 1, cluster })).result.page;
    if (current && current.generation > initial.generation) break;
  }
  assert.ok(current && current.generation > initial.generation, "the revealed phase has a fresh submitted generation");
  assert.notEqual(current.mask, 0, "newly exposed geometry is collected in phase two");
  assert.ok(current.filtered_length > 0, "phase two preserves its contributing selected lights");
  assert.ok(revealed.render_passes.indexOf("occlusion_test") < revealed.render_passes.indexOf("light_depth_collect_phase_two"));
  assert.ok(revealed.render_passes.indexOf("light_depth_filter_phase_two") < revealed.render_passes.indexOf("opaque_phase_two"));
  checks.push({ check: "newly_revealed_phase", before: initial.generation, after: current.generation, mask: current.mask, kept: current.filtered_length });
}

export async function exerciseTransitions({ command, scene, fixture, cluster, render, productionPage, checks }) {
  const initial = structuredClone(fixture);
  initial.environment.local_lights.depth_occupancy = true;
  await scene.reload(command, initial);
  const before = await productionPage(0);
  await command("environment", { merge: true, local_lights: { max_lights: 37 } });
  for (let i = 0; i < 3; i++) { await render(); await delay(100); }
  const limited = await render();
  assert.equal(limited.local_lights.shaded, 37);
  assert.equal(limited.local_lights.dropped, initial.entities.filter(entity => entity.light).length - 37);
  await command("environment", { merge: true, local_lights: { depth_occupancy: false } });
  const limitedOff = await render();
  assert.equal(limitedOff.local_lights.dropped, limited.local_lights.dropped);
  assert.equal(limitedOff.local_lights.shaded, limited.local_lights.shaded);
  const growth = structuredClone(initial);
  growth.entities = growth.entities.filter(entity => !entity.light);
  growth.entities.push(...Array.from({ length: 260 }, (_, i) => ({ id: `broad-${i}`, light: { range: 8, intensity: 0.001 }, transform: { translation: [0.2, 0, -11] } })));
  await scene.reload(command, growth);
  let grown;
  for (let attempt = 0; attempt < 50; attempt++) {
    grown = await productionPage(0);
    if (grown.index_capacity > before.index_capacity && grown.original_length === 260) break;
    await render(); await delay(100);
  }
  assert.ok(grown.index_capacity > before.index_capacity, "full index capacity grows without shrinking to fit the optional arena");
  assert.equal(grown.original_length, 260);
  assert.equal(grown.filtered_length, 260);
  assert.deepEqual(grown.filtered, grown.original.map(light => light.id), "all broad selected lights retained in order in this snapshot");
  const end = await productionPage(0, cluster, 256);
  assert.equal(end.original.length, 4);
  assert.deepEqual(end.filtered, end.original.map(light => light.id));
  const noLights = structuredClone(initial);
  noLights.entities = noLights.entities.filter(entity => !entity.light);
  await scene.reload(command, noLights);
  const none = await render();
  assert.equal(none.local_lights.depth_occupancy.active, false);
  assert.equal(none.local_lights.depth_occupancy.allocated_bytes, 0);
  assert.ok(!none.render_passes.some(name => name.startsWith("light_depth")));
  assert.equal((await command("light_depth", { phase: 0, cluster })).result.page, null);
  const noMeshes = structuredClone(initial);
  noMeshes.entities = noMeshes.entities.filter(entity => entity.light);
  await scene.reload(command, noMeshes);
  const unsupported = await render();
  assert.equal(unsupported.local_lights.depth_occupancy.active, false);
  assert.equal(unsupported.local_lights.depth_occupancy.allocated_bytes, 0);
  await scene.reload(command, initial);
  const replaced = await productionPage(0);
  assert.ok(replaced.generation > before.generation, "scene replacement cannot return an old phase snapshot");
  await command("environment", { merge: true, local_lights: { enabled: false } });
  const disabled = await render();
  assert.equal(disabled.local_lights.depth_occupancy.active, false);
  assert.equal(disabled.local_lights.depth_occupancy.allocated_bytes, 0);
  checks.push({ check: "lifetime_and_capacity", initial: before.index_capacity, grown: grown.index_capacity, bytes: grown.arena_bytes,
    initial_view: before.view, replacement_view: replaced.view, finite_budget: limited.local_lights });
}
