// @ts-check
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, sceneUrl, launch, openPaused, routedScene } from "./harness.mjs";

const url = sceneUrl();
const output = await artifacts("meshes-smoke");
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
const vertices = [0, 1, 2].flatMap(y => [-0.3, 0.3].map(x => ({
  position: [x, y, 0], normal: [0, 0, 1], uv: [x + 0.3, y / 2],
})));
const fixture = {
  id: "mesh-fixture", revision: 1,
  camera: { position: [0, 1, 4], target: [0, 1, 0], fov_degrees: 60, near: 0.1, far: 2000 },
  materials: [{ id: "plain", color: [0.8, 0.4, 0.2, 1], double_sided: true }, { id: "stone", color: [0.5, 0.55, 0.6, 1], double_sided: true }],
  meshes: [
    { id: "ribbon", source: { kind: "inline", vertices, indices: [0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4] } },
    { id: "ball", source: ball(24, 32) },
  ],
  entities: [
    { id: "limb", mesh: "ribbon", material: "plain" },
    { id: "near-ball", mesh: "ball", material: "stone", transform: { translation: [2.5, 1, -2] } },
    { id: "far-ball", mesh: "ball", material: "stone", transform: { translation: [-40, 1, -300] } },
  ],
};
const { browser, page, errors } = await launch({ width: 960, height: 720 });
const command = commander(page);
const scene = routedScene(page, fixture);

async function inspect(entity, component) {
  return (await command("inspect", { entity, component })).result.entity.components[component];
}

try {
  await scene.route();
  await openPaused(page, url);
  const checks = [];

  await command("control", "render");
  const drawn = await inspect("limb", "Mesh3d");
  assert.equal(drawn.mesh.id, "ribbon");
  assert.deepEqual(drawn.mesh.source, { kind: "inline", vertices: 6, indices: 12 });
  assert.equal(drawn.material.id, "plain");
  assert.equal(drawn.material.double_sided, true);
  checks.push({ check: "inspects", mesh: drawn.mesh });

  const near = await inspect("near-ball", "Mesh3d");
  assert.ok(near.mesh.rendered?.levels.length > 0, `the ball has coarser levels: ${JSON.stringify(near.mesh.rendered)}`);
  async function drawnLevel(entity) {
    await inspect(entity, "Mesh3d");
    await command("control", "resume");
    try {
      for (let attempt = 0; attempt < 100; attempt++) {
        await delay(100);
        const level = (await inspect(entity, "Mesh3d")).drawn_level;
        if (level !== null) return level;
      }
      assert.fail(`${entity}'s drawn level reads back`);
    } finally {
      await command("control", "pause");
    }
  }
  const nearLevel = await drawnLevel("near-ball");
  const farLevel = await drawnLevel("far-ball");
  assert.ok(nearLevel < 3, `the near ball draws as a mesh: ${nearLevel}`);
  assert.ok(farLevel > nearLevel, `the far ball draws coarser: ${farLevel} than ${nearLevel}`);
  checks.push({ check: "levels", levels: near.mesh.rendered.levels, near: nearLevel, far: farLevel });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, output, checks }));
} finally {
  await browser.close();
}
