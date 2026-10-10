/**
 * Spline meshes: a version 9 fixture bends a box
 * along a curve; the inspector reports what it built and its chunks draw;
 * moving a point rebuilds them, an invalid curve is refused with the scene
 * kept, and removing the field removes the chunks.
 */
import assert from "node:assert/strict";
import { artifacts, commander, sceneUrl, differentPixels, launch, openPaused, routedScene } from "./harness.mjs";

const url = sceneUrl();
const output = await artifacts("spline-smoke");
const curve = [[-12, 0.5, 4], [-4, 0.5, -2], [4, 0.5, 3], [12, 0.5, -3]];
/** Meters along the straight lines between `points`, which the curve through them exceeds a little. */
const polyline = points => points.slice(1).reduce((sum, point, index) => sum + Math.hypot(...point.map((value, axis) => value - points[index][axis])), 0);
const fixture = {
  version: 9, id: "spline-fixture", revision: 1,
  camera: { position: [0, 14, 22], target: [0, 0, 0], fov_degrees: 55, near: 0.1, far: 500 },
  materials: [{ id: "plain", color: [0.8, 0.3, 0.2, 1] }, { id: "ground", color: [0.3, 0.32, 0.28, 1] }, { id: "rail", color: [0.85, 0.8, 0.7, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [
    { id: "player", mesh: "box", material: "plain", tags: ["player"], collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] }, memberships: 2, filter: 1, sensor: false }, transform: { translation: [0, 0.5, 8] } },
    { id: "floor", mesh: "box", material: "ground", transform: { translation: [0, -0.5, 0], scale: [40, 1, 30] } },
    // A rail of the unit box bent along the curve, a meter of box a copy.
    { id: "rail", spline: { points: curve, mesh: "box", material: "rail" }, transform: { scale: [1, 1, 1] } },
  ],
};
const { browser, page, errors } = await launch({ width: 1280, height: 720 });
const command = commander(page);
const scene = routedScene(page, fixture);

async function capture(name) {
  await command("render");
  await command("capture");
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return bytes.toString("base64");
}

async function spline() {
  const inspected = await command("inspect", { entity: "rail", component: "SplineMesh" });
  return inspected.result.entity.components.SplineMesh;
}

try {
  await scene.route();
  await openPaused(page, url);
  const checks = [];

  // The inspector describes it and reports what it built: the curve's
  // length, a copy of the unit box per meter, chunks of at most 256 m.
  const listing = await command("describe");
  assert.ok(listing.result.types.some(type => type.name === "SplineMesh"), "describe lists SplineMesh");
  const first = await command("render");
  const built = await spline();
  assert.deepEqual(built.points, curve);
  assert.equal(built.mesh, "box");
  assert.equal(built.material, "rail");
  assert.equal(built.error, null);
  assert.ok(built.length >= polyline(curve) && built.length < 1.1 * polyline(curve), `the curve's length: ${built.length} over ${polyline(curve)}`);
  assert.equal(built.copies, Math.round(built.length));
  assert.equal(built.chunks, 1);
  assert.equal(built.vertices, built.copies * 24);
  // The player, the floor and the rail's chunk.
  assert.equal(first.result.rendering.render_candidates, 3);
  checks.push({ check: "built", length: built.length, copies: built.copies, chunks: built.chunks });

  // It draws: without it the view differs.
  const withRail = await capture("rail");
  const bare = structuredClone(fixture);
  delete bare.entities[2].spline;
  await scene.reload(command, bare);
  const without = await capture("bare");
  const { changed } = await differentPixels(page, [withRail, without]);
  assert.ok(changed > 2000, `the rail covers pixels: ${changed}`);
  const removed = await command("render");
  assert.equal(removed.result.rendering.render_candidates, 2, "removing the field removes its chunks");
  checks.push({ check: "draws and goes", pixels: changed });

  // A longer curve rebuilds into more copies and a second chunk.
  const longer = structuredClone(fixture);
  longer.entities[2].spline.points = [...curve, [40, 0.5, 2], [70, 0.5, -2], [160, 0.5, 6], [290, 0.5, -4]];
  await scene.reload(command, longer);
  await command("render");
  const rebuilt = await spline();
  const longerPoints = longer.entities[2].spline.points;
  assert.ok(rebuilt.length >= polyline(longerPoints) && rebuilt.length < 1.1 * polyline(longerPoints), `the longer curve: ${rebuilt.length}`);
  assert.equal(rebuilt.copies, Math.round(rebuilt.length));
  assert.equal(rebuilt.chunks, Math.ceil(rebuilt.copies / 256));
  checks.push({ check: "rebuilds", length: rebuilt.length, chunks: rebuilt.chunks });

  // A curve of one point is refused and the scene kept.
  const invalid = structuredClone(fixture);
  invalid.entities[2].spline.points = [[0, 0, 0]];
  invalid.revision = scene.revision + 1;
  scene.serve(invalid);
  const refused = await command("reload", { expected_revision: String(scene.revision), revision: String(invalid.revision) }, false);
  assert.equal(refused.error.code, "invalid_spline");
  assert.equal(refused.error.path, "$.entities[2].spline.points");
  assert.equal((await spline()).chunks, rebuilt.chunks, "the scene stays");
  checks.push({ check: "refused" });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, checks }));
} finally {
  await browser.close();
}
