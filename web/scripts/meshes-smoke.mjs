/**
 * Meshes as inspected: a version 9 fixture's inline two-joint ribbon with
 * its skin and a clip bending its upper joint, and a dense ball near and
 * far. The skin, mesh and player inspect as authored (IDs, the inline
 * source, joints, inverse binds, the clip's targets and channels), the
 * renderer reports the ribbon skinned and the ball's levels of detail, the
 * running page draws the far ball coarser than the near one, and the clip at
 * its end draws the ribbon bent where its start draws it straight.
 */
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, sceneUrl, differentPixels, launch, openPaused, routedScene } from "./harness.mjs";

const url = sceneUrl();
const output = await artifacts("meshes-smoke");
/** A unit ball of `rings` by `segments` quads. */
function ball(rings, segments) {
  const vertices = [], indices = [];
  for (let ring = 0; ring <= rings; ring++) {
    const phi = Math.PI * ring / rings;
    for (let segment = 0; segment <= segments; segment++) {
      const theta = 2 * Math.PI * segment / segments;
      const normal = [Math.sin(phi) * Math.cos(theta), Math.cos(phi), Math.sin(phi) * Math.sin(theta)];
      vertices.push({ position: normal, normal, uv: [segment / segments, ring / rings] });
    }
  }
  for (let ring = 0; ring < rings; ring++) {
    for (let segment = 0; segment < segments; segment++) {
      const a = ring * (segments + 1) + segment, b = a + segments + 1;
      indices.push(a, b, a + 1, a + 1, b, b + 1);
    }
  }
  return { kind: "inline", vertices, indices };
}
// Rows of two vertices at heights 0, 1 and 2: the lowest moves with the
// root joint, the highest with the tip joint, the middle with both.
const rows = [[0, [1, 0, 0, 0]], [1, [0.5, 0.5, 0, 0]], [2, [0, 1, 0, 0]]];
const vertices = rows.flatMap(([y, weights]) => [-0.3, 0.3].map(x => ({
  position: [x, y, 0], normal: [0, 0, 1], uv: [x + 0.3, y / 2], joints: [0, 1, 0, 0], weights,
})));
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
// The tip joint stands one unit up: its inverse bind moves one unit down.
const tipBind = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1];
const fixture = {
  version: 9, id: "skin-fixture", revision: 1,
  camera: { position: [0, 1, 4], target: [0, 1, 0], fov_degrees: 60, near: 0.1, far: 2000 },
  materials: [{ id: "plain", color: [0.8, 0.4, 0.2, 1], double_sided: true }, { id: "stone", color: [0.5, 0.55, 0.6, 1], double_sided: true }],
  meshes: [
    { id: "ribbon", source: { kind: "inline", vertices, indices: [0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4] } },
    { id: "ball", source: ball(24, 32) },
  ],
  skins: [{ id: "rig", joints: ["root-bone", "tip-bone"], inverse_binds: [identity, tipBind] }],
  animations: [{ id: "bend", targets: ["tip-bone"], channels: [
    { target: 0, property: "rotation", times: [0, 1], values: [0, 0, 0, 1, 0, 0, 0.7071068, 0.7071068] },
  ] }],
  entities: [
    { id: "root-bone", animation: { clip: "bend", speed: 0, loop: false } },
    { id: "tip-bone", parent: "root-bone", transform: { translation: [0, 1, 0] } },
    { id: "limb", mesh: "ribbon", material: "plain", skin: "rig" },
    { id: "near-ball", mesh: "ball", material: "stone", transform: { translation: [2.5, 1, -2] } },
    { id: "far-ball", mesh: "ball", material: "stone", transform: { translation: [-40, 1, -300] } },
  ],
};
const { browser, page, errors } = await launch({ width: 960, height: 720 });
const command = commander(page);
const scene = routedScene(page, fixture);

/** Component `component` of authored entity `entity`. */
async function inspect(entity, component) {
  return (await command("inspect", { entity, component })).result.entity.components[component];
}

async function capture(name) {
  await command("render");
  await command("capture");
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return bytes.toString("base64");
}

try {
  await scene.route();
  await openPaused(page, url);
  const checks = [];

  // The skin, its mesh and the player inspect as authored.
  const skin = await inspect("limb", "Skin");
  assert.deepEqual(skin, { skin: "rig", joints: ["root-bone", "tip-bone"], inverse_binds: 2 }, JSON.stringify(skin));
  await command("render");
  const drawn = await inspect("limb", "Mesh3d");
  assert.equal(drawn.mesh.id, "ribbon");
  assert.deepEqual(drawn.mesh.source, { kind: "inline", vertices: 6, indices: 12 });
  assert.equal(drawn.mesh.joints, 2, "the weights address both joints");
  assert.equal(drawn.mesh.rendered?.skinned, true, `the renderer uploads it skinned: ${JSON.stringify(drawn.mesh)}`);
  assert.equal(drawn.material.id, "plain");
  assert.equal(drawn.material.double_sided, true);
  const player = await inspect("root-bone", "AnimationPlayer");
  assert.deepEqual(player.clip, { id: "bend", name: "", targets: ["tip-bone"], channels: 1, uri: null, duration: 1 }, JSON.stringify(player.clip));
  checks.push({ check: "inspects", skin, mesh: drawn.mesh });

  // The ball's levels of detail, and the level the running page draws each
  // ball at: inspecting watches it, displayed frames read it back.
  const near = await inspect("near-ball", "Mesh3d");
  assert.ok(near.mesh.rendered?.levels.length > 0, `the ball has coarser levels: ${JSON.stringify(near.mesh.rendered)}`);
  assert.equal(near.mesh.rendered.skinned, false);
  async function drawnLevel(entity) {
    await inspect(entity, "Mesh3d");
    await command("resume");
    try {
      // Inspection, a query, answers while frames run.
      for (let attempt = 0; attempt < 100; attempt++) {
        await delay(100);
        const level = (await inspect(entity, "Mesh3d")).drawn_level;
        if (level !== null) return level;
      }
      assert.fail(`${entity}'s drawn level reads back`);
    } finally {
      await command("pause");
    }
  }
  const nearLevel = await drawnLevel("near-ball");
  const farLevel = await drawnLevel("far-ball");
  assert.ok(nearLevel < 3, `the near ball draws as a mesh: ${nearLevel}`);
  assert.ok(farLevel > nearLevel, `the far ball draws coarser: ${farLevel} than ${nearLevel}`);
  checks.push({ check: "levels", levels: near.mesh.rendered.levels, near: nearLevel, far: farLevel });

  // The clip's end bends the ribbon's upper half.
  const straight = await capture("straight");
  await command("animation", { entity: "root-bone", time: 1, speed: 0 });
  await command("step", { ticks: 1 });
  const bent = await capture("bent");
  const { changed } = await differentPixels(page, [straight, bent]);
  assert.ok(changed > 500, `the skin bends with its joint: ${changed} pixels`);
  checks.push({ check: "bends", pixels: changed });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, output, checks }));
} finally {
  await browser.close();
}
