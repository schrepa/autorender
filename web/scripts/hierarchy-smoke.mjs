/** Real browser hierarchy/prefab checks. All scene changes stay in routed responses. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  artifacts, commander, gardenScene, sceneUrl, launch, near as nearly, openPaused, residency, routedScene,
} from "./harness.mjs";

const url = sceneUrl();
const output = await artifacts("hierarchy-smoke");
const { browser, page, errors } = await launch({ width: 1440, height: 1000 });
const command = commander(page);

async function inspect(id, component) {
  const reply = await command("inspect", { entity: id, component });
  return reply.result.entity.components[component];
}

// Matrices compose here in JavaScript doubles: allow a little more.
const near = (actual, expected, message) => nearly(actual, expected, message, 0.0002);

function localMatrix({ translation: [tx, ty, tz], rotation: [x, y, z, w], scale: [sx, sy, sz] }) {
  // Independent column-major TRS composition from the public quaternion view.
  return [
    (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx, 2 * (x * z - y * w) * sx, 0,
    2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z + x * w) * sy, 0,
    2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

function multiply(parent, local) {
  return Array.from({ length: 16 }, (_, index) => {
    const row = index % 4, column = Math.floor(index / 4);
    let value = 0;
    for (let k = 0; k < 4; k++) value += parent[k * 4 + row] * local[column * 4 + k];
    return value;
  });
}

async function capture(name) {
  await command("capture");
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  assert.ok(bytes.length > 1000, "The canvas should contain the full rendered game.");
  return createHash("sha256").update(bytes).digest("hex");
}

try {
  const source = gardenScene();
  assert.equal(source.version, 4, "This smoke requires the current hierarchy/collider sample.");
  assert.ok(Number.isSafeInteger(source.revision) && source.revision < Number.MAX_SAFE_INTEGER - 20);
  const fixture = structuredClone(source);
  const prefab = fixture.prefabs.find(prefab => prefab.id === "garden-lantern");
  assert.ok(prefab, "The sample must define garden-lantern.");
  assert.deepEqual(prefab.entities.map(entity => entity.id).sort(), ["lantern", "planter"]);
  assert.ok(prefab.entities.every(entity => entity.mesh && !entity.parent && !entity.prefab));
  for (const id of ["west-lantern", "east-lantern"]) {
    assert.equal(fixture.entities.find(entity => entity.id === id)?.prefab, prefab.id);
  }
  const player = fixture.entities.find(entity => entity.tags?.includes("player"))?.id;
  assert.ok(player);
  assert.equal(fixture.entities.find(entity => entity.id === "player-accessory")?.parent, player);
  // Rotate a template child only in this isolated browser fixture. Combining it
  // with a nonuniform placement scale must retain shear in the complete matrix.
  const tilted = prefab.entities.find(entity => entity.id === "lantern");
  tilted.transform = { ...tilted.transform, rotation_degrees: [17, 31, -9] };
  const scene = routedScene(page, fixture);
  await scene.route();
  assert.equal((await openPaused(page, url)).result.paused, true);
  async function reload(document) {
    const reply = await scene.reload(command, document);
    assert.equal(reply.result.rendering.invalid_transforms, 0);
    return reply;
  }
  await reload(fixture);

  const discovery = await command("describe");
  for (const name of ["Hierarchy", "Transform", "GlobalTransform"]) {
    assert.ok(discovery.result.types.some(type => type.name === name));
    assert.ok((await command("describe", { name })).result.descriptor.schema);
  }
  const hierarchySchema = (await command("describe", { name: "Hierarchy" })).result.descriptor.schema;
  for (const field of ["parent", "parent_scene", "parent_alive", "parent_in_active_scene", "authored_parent", "prefab_id", "prefab_instance", "prefab_local_id"]) {
    assert.ok(hierarchySchema.properties[field], `Hierarchy schema is missing ${field}.`);
  }
  const first = await command("inspect", { parent: "west-lantern", component: "Hierarchy", limit: 1 });
  assert.equal(first.result.total, 2);
  assert.equal(first.result.entities.length, 1);
  assert.equal(first.result.next_offset, 1);
  const second = await command("inspect", { parent: "west-lantern", component: "Hierarchy", limit: 1, offset: first.result.next_offset });
  assert.equal(second.result.total, 2);
  assert.equal(second.result.next_offset, null);
  const children = [...first.result.entities, ...second.result.entities];
  assert.deepEqual(children.map(entity => entity.id).sort(), ["west-lantern.lantern", "west-lantern.planter"]);
  for (const child of children) {
    const hierarchy = child.components.Hierarchy;
    assert.equal(hierarchy.parent, "west-lantern");
    assert.equal(hierarchy.parent_scene, fixture.id);
    assert.equal(hierarchy.parent_alive, true);
    assert.equal(hierarchy.parent_in_active_scene, true);
    assert.equal(hierarchy.authored_parent, "west-lantern");
    assert.equal(hierarchy.prefab_id, prefab.id);
    assert.equal(hierarchy.prefab_instance, "west-lantern");
    assert.equal(child.id, `${hierarchy.prefab_instance}.${hierarchy.prefab_local_id}`);
  }
  const badSelector = await command("inspect", { entity: player, parent: player }, false);
  assert.equal(badSelector.error.code, "invalid_selector");
  const playerChildren = await command("inspect", { parent: player, limit: 10 });
  assert.ok(playerChildren.result.entities.some(entity => entity.id === "player-accessory"));
  const rootHierarchy = await inspect("west-lantern", "Hierarchy");
  assert.equal(rootHierarchy.parent, null);
  assert.equal(rootHierarchy.parent_alive, null);
  assert.equal(rootHierarchy.parent_in_active_scene, null);
  const rootComponents = (await command("inspect", { entity: "west-lantern" })).result.entity.components;
  assert.ok(!rootComponents.includes("Mesh3d"), "A prefab placement root must not add a renderable mesh.");
  const initialAssets = await command("assets");
  const initialResidency = residency(initialAssets);
  const initialRendering = initialAssets.result.rendering;
  const initialEntities = (await command("inspect", { limit: 1 })).result.total;
  assert.equal(initialRendering.invalid_transforms, 0);

  const accessoryLocal = await inspect("player-accessory", "Transform");
  const playerBefore = await inspect(player, "GlobalTransform");
  const accessoryBefore = await inspect("player-accessory", "GlobalTransform");
  const beforeTick = (await command("status")).observation.tick;
  await command("input", { forward: 1 });
  const stepped = await command("step", { ticks: 6 });
  await command("input", { forward: 0 });
  assert.equal(BigInt(stepped.observation.tick) - BigInt(beforeTick), 6n);
  const playerAfter = await inspect(player, "GlobalTransform");
  const accessoryAfter = await inspect("player-accessory", "GlobalTransform");
  const playerDelta = playerAfter.translation.map((value, index) => value - playerBefore.translation[index]);
  const accessoryDelta = accessoryAfter.translation.map((value, index) => value - accessoryBefore.translation[index]);
  assert.ok(Math.hypot(...playerDelta) > 0.1);
  near(accessoryDelta, playerDelta, "Accessory must inherit the exact player displacement");
  assert.deepEqual(await inspect("player-accessory", "Transform"), accessoryLocal);
  const playerState = await command("inspect", { entity: player, component: "Transform" });
  const progress = (await command("inspect", { component: "PlayerWalk" })).result.value;
  const invalid = [];
  for (const kind of ["cycle", "missing_parent", "qualified_id_collision"]) {
    const broken = structuredClone(fixture);
    broken.revision = scene.revision + 1;
    if (kind === "cycle") broken.entities.find(entity => entity.id === "west-lantern").parent = "west-lantern.lantern";
    if (kind === "missing_parent") broken.entities.find(entity => entity.id === "player-accessory").parent = "absent-parent";
    if (kind === "qualified_id_collision") broken.entities.push({ id: "west-lantern.lantern" });
    scene.serve(broken);
    const failed = await command("reload", { expected_revision: String(scene.revision), revision: String(broken.revision) }, false);
    assert.equal(failed.observation.scene.revision, String(scene.revision));
    assert.equal(failed.observation.tick, playerState.observation.tick);
    assert.equal(failed.error.code, { cycle: "hierarchy_cycle", missing_parent: "unknown_parent", qualified_id_collision: "duplicate_id" }[kind]);
    assert.ok(failed.error.path.startsWith("$"));
    invalid.push({ kind, code: failed.error.code, path: failed.error.path });
    assert.deepEqual((await command("inspect", { entity: player, component: "Transform" })).result.entity, playerState.result.entity);
    assert.deepEqual(await inspect("player-accessory", "GlobalTransform"), accessoryAfter);
    assert.deepEqual((await command("inspect", { component: "PlayerWalk" })).result.value, progress);
    assert.deepEqual(residency(await command("assets")), initialResidency);
  }

  await reload(fixture);
  const originalCapture = await capture("original");
  const eastIds = ["east-lantern", ...prefab.entities.map(entity => `east-lantern.${entity.id}`)];
  const eastBefore = [];
  for (const id of eastIds) eastBefore.push(await inspect(id, "GlobalTransform"));
  const transformed = structuredClone(fixture);
  transformed.entities.find(entity => entity.id === "west-lantern").transform = {
    translation: [-6, 0.25, -4], rotation_degrees: [0, 37, 0], scale: [1.6, 0.8, 0.65],
  };
  await reload(transformed);
  const rootLocal = await inspect("west-lantern", "Transform");
  const rootMatrix = localMatrix(rootLocal);
  for (const template of prefab.entities) {
    const id = `west-lantern.${template.id}`;
    const childLocal = await inspect(id, "Transform");
    const global = await inspect(id, "GlobalTransform");
    assert.equal(global.valid, true);
    near(global.matrix, multiply(rootMatrix, localMatrix(childLocal)), `${id} complete world matrix`);
    near(global.translation, global.matrix.slice(12, 15), `${id} world translation`);
  }
  const shear = (await inspect("west-lantern.lantern", "GlobalTransform")).matrix;
  const dot = shear[0] * shear[4] + shear[1] * shear[5] + shear[2] * shear[6];
  assert.ok(Math.abs(dot) > 0.01, "The fixture must exercise shear from rotated children under nonuniform scale.");
  for (let index = 0; index < eastIds.length; index++) assert.deepEqual(await inspect(eastIds[index], "GlobalTransform"), eastBefore[index]);
  assert.deepEqual(residency(await command("assets")), initialResidency);
  const transformedCapture = await capture("transformed");
  assert.notEqual(transformedCapture, originalCapture);

  const triple = structuredClone(transformed);
  triple.entities.push({ id: "third-lantern", prefab: prefab.id, transform: { translation: [0, 0, -7], scale: [0.7, 0.7, 0.7] } });
  await reload(triple);
  const third = await command("inspect", { parent: "third-lantern", component: "Hierarchy", limit: 10 });
  assert.equal(third.result.total, prefab.entities.length);
  assert.equal((await command("inspect", { limit: 1 })).result.total, initialEntities + prefab.entities.length + 1);
  const tripledAssets = await command("assets");
  assert.deepEqual(residency(tripledAssets), initialResidency, "Adding a placement must share CPU and GPU assets.");
  assert.equal(tripledAssets.result.rendering.draw_calls, initialRendering.draw_calls);
  assert.equal(tripledAssets.result.rendering.instances, initialRendering.instances + prefab.entities.length);
  for (let replacement = 0; replacement < 3; replacement++) {
    await reload(triple);
    const assets = await command("assets");
    assert.deepEqual(residency(assets), initialResidency);
    assert.equal(assets.result.rendering.draw_calls, initialRendering.draw_calls);
    assert.equal(assets.result.rendering.instances, initialRendering.instances + prefab.entities.length);
    const warm = (await command("render")).result.rendering;
    assert.equal(warm.mesh_upload_bytes, 0);
    assert.equal(warm.texture_upload_bytes, 0);
    assert.equal(warm.invalid_transforms, 0);
  }
  const tripleCapture = await capture("three-instances");

  await reload(transformed);
  assert.deepEqual(residency(await command("assets")), initialResidency);
  assert.equal((await command("inspect", { limit: 1 })).result.total, initialEntities);
  assert.equal((await command("inspect", { entity: "third-lantern" }, false)).error.code, "unknown_entity");
  const removed = structuredClone(transformed);
  removed.entities = removed.entities.filter(entity => !["west-lantern", "east-lantern"].includes(entity.id));
  await reload(removed);
  const retired = await command("assets");
  assert.equal(retired.result.rendering.instances, initialRendering.instances - prefab.entities.length * 2);
  assert.equal((await command("inspect", { limit: 1 })).result.total, initialEntities - (prefab.entities.length + 1) * 2);
  for (const id of ["west-lantern", "west-lantern.lantern", "east-lantern.planter"]) {
    assert.equal((await command("inspect", { entity: id }, false)).error.code, "unknown_entity");
  }
  // Definitions and declared global assets remain available for later placement,
  // while the renderer retires meshes/textures with no remaining instances.
  assert.equal(retired.result.resident_cpu_bytes, initialResidency.cpu);
  assert.ok(retired.result.rendering.resident_meshes < initialResidency.gpuMeshes);
  assert.ok(retired.result.rendering.resident_mesh_bytes < initialResidency.meshGpu);
  assert.equal(retired.result.rendering.resident_textures, 0);
  assert.equal(retired.result.rendering.resident_texture_bytes, 0);
  assert.equal(retired.result.rendering.invalid_transforms, 0);
  await capture("removed");
  await reload(fixture);
  assert.deepEqual(residency(await command("assets")), initialResidency);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    ok: true, url: url.href, artifact: output, initial: initialResidency,
    rendering: { drawCalls: initialRendering.draw_calls, initialInstances: initialRendering.instances,
      thirdPlacementInstances: tripledAssets.result.rendering.instances, initialEntities,
      retiredGpuMeshes: retired.result.rendering.resident_meshes, retiredGpuTextures: retired.result.rendering.resident_textures },
    playerDelta, accessoryDelta, shearColumnDot: dot, invalid,
    captures: { original: originalCapture, transformed: transformedCapture, triple: tripleCapture },
    checks: ["descriptor_discovery", "direct_children_pagination", "prefab_metadata", "exact_step_parent_motion", "unchanged_local_transform", "invalid_reload_preservation", "full_world_matrix_with_shear", "independent_instances", "shared_assets_and_draws", "warm_upload_reuse", "repeated_reload_residency", "instance_removal", "unused_gpu_retirement", "clean_browser_diagnostics"],
  }));
} finally {
  await browser.close();
}
