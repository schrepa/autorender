/** Material coverage modes, binding order, inspected fallbacks and replacement of shadow/impostor sources. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, launch, openPaused, residency, routedScene, sceneUrl } from "./harness.mjs";

const output = await artifacts("alpha-coverage-mode-smoke");
// Runs the same order regression against the pre-control revision too.
const sharingOnly = process.argv.includes("--sharing-only");
const results = { orders: [], switching: [], fallbacks: [], placed: [] };
const size = [900, 740];
const boxes = [{ x: 264, y: 354, width: 32, height: 32 }, { x: 604, y: 354, width: 32, height: 32 }];
function texture(width = 256, height = 256, fill) {
  const bytes = Buffer.alloc(width * height * 4, 255);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const c = Math.floor(x / 32), r = Math.floor(y / 32);
    const rx = 4 + (c + 3 * r) % 8, ry = 3 + (3 * c + r) % 6;
    bytes[(y * width + x) * 4 + 3] = fill ?? (((x % 32 + 0.5 - 16) / rx) ** 2 + ((y % 32 + 0.5 - 16) / ry) ** 2 <= 1 ? 255 : 0);
  }
  return { id: "mask", width, height, rgba8: bytes.toString("base64") };
}
function quad(grid = 1, opacity = 1) {
  const vertices = [], indices = [];
  for (let y = 0; y <= grid; y++) for (let x = 0; x <= grid; x++) {
    vertices.push({ position: [2 * x / grid - 1, 1 - 2 * y / grid, 0], normal: [0, 0, 1], uv: [x / grid, y / grid],
      ...(opacity === 1 ? {} : { color: [1, 1, 1, opacity === "mixed" ? (x === 0 ? 1 : 0.5) : opacity] }) });
  }
  for (let y = 0; y < grid; y++) for (let x = 0; x < grid; x++) {
    const a = y * (grid + 1) + x, b = a + grid + 1;
    indices.push(a, b, a + 1, a + 1, b, b + 1);
  }
  return { kind: "inline", vertices, indices };
}
const material = (id, opacity = 1, extra = {}) => ({ id, color: [1, 1, 1, 1], texture: "mask", emissive: [8, 8, 8],
  alpha_mode: "mask", alpha_cutoff: 0.5, opacity, double_sided: true, ...extra });
function fixture(order = ["eligible", "reduced"], modes = {}) {
  return {
    id: "coverage-mode-fixture", revision: 1,
    camera: { position: [0, 0, 4], target: [0, 0, 0], fov_degrees: 90, near: 0.1, far: 100 },
    textures: [texture()],
    materials: order.map(id => material(id, id === "reduced" ? 0.5 : 1, modes[id] ? { alpha_coverage: modes[id] } : {})),
    meshes: [{ id: "quad", source: quad() }],
    environment: {
      atmosphere: { sun_color: [0, 0, 0], zenith: [0, 0, 0], horizon: [0, 0, 0], ground: [0, 0, 0],
        ambient: 0, fog_density: 0, fog_max: 0, aerial_density: 0, rim: 0, specular: 0, emission: 1,
        exposure: 1, saturation: 1, contrast: 1 },
      shadows: { enabled: false }, level_of_detail: { enabled: false }, temporal_antialiasing: { enabled: false },
      bloom: { intensity: 0 }, ambient_occlusion: { intensity: 0 }, contact_shadows: { enabled: false },
    },
    entities: order.map(id => {
      const box = boxes[id === "reduced" ? 1 : 0], unit = 8 / size[1];
      return { id, mesh: "quad", material: id, transform: {
        translation: [(box.x + box.width / 2 - size[0] / 2) * unit, 0, 0], scale: [16 * unit, 16 * unit, 1],
      } };
    }),
  };
}
const { browser, page, errors } = await launch({ width: size[0], height: size[1], timeout: 120_000 });
const command = commander(page, { deadline: 120_000 });
const scene = routedScene(page, fixture());
await page.addInitScript(() => {
  addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = "html,body{margin:0!important;overflow:hidden!important}.viewport{position:fixed!important;inset:0!important;width:100vw!important;height:100vh!important;border:0!important;border-radius:0!important;z-index:1000}canvas{display:block!important;width:100%!important;height:100%!important}.viewport>:not(canvas){visibility:hidden!important}";
    document.head.append(style);
  });
});
async function settled() {
  let stats;
  for (let frame = 0; frame < 600; frame++) {
    stats = (await command("present", { alpha: 0, gpu: true })).result.rendering;
    if (!stats.pipelines_compiling && stats.terrain.loading_tiles === 0) return stats;
    await delay(20);
  }
  assert.fail(`renderer did not settle: ${JSON.stringify(stats)}`);
}
async function capture(name, box = { x: 0, y: 0, width: size[0], height: size[1] }) {
  await settled();
  await command("capture");
  const png = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  const crop = await page.evaluate(async ({ png, box }) => {
    const image = new Image(); image.src = `data:image/png;base64,${png}`; await image.decode();
    const canvas = document.createElement("canvas"); canvas.width = box.width; canvas.height = box.height;
    const ctx = canvas.getContext("2d"); ctx.drawImage(image, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
    const { data } = ctx.getImageData(0, 0, box.width, box.height);
    let covered = 0;
    for (let i = 0; i < data.length; i += 4) if (Math.min(data[i], data[i + 1], data[i + 2]) > 128) covered++;
    return { png: canvas.toDataURL("image/png").split(",")[1], covered, pixels: box.width * box.height };
  }, { png: png.toString("base64"), box });
  const bytes = Buffer.from(crop.png, "base64");
  await writeFile(`${output}/${name}-crop.png`, bytes);
  return { hash: createHash("sha256").update(bytes).digest("hex"), covered: crop.covered, pixels: crop.pixels };
}
async function inspect(id = "eligible") {
  const view = (await command("inspect", { entity: id, component: "Mesh3d" })).result.entity.components.Mesh3d;
  view.alpha_coverage = (await command("inspect", { entity: id, component: "AlphaCoverage" })).result.entity.components.AlphaCoverage;
  return view;
}
function coverage(view, effective, reason = null, requested = "auto") {
  assert.equal(view.material.alpha_coverage, requested, "material round-trips the requested mode");
  assert.equal(view.alpha_coverage.requested, requested);
  assert.equal(view.alpha_coverage.effective, effective, JSON.stringify(view.alpha_coverage));
  assert.equal(view.alpha_coverage.fallback_reason, reason);
}
async function warmed() {
  const stats = await settled();
  assert.equal(stats.texture_upload_bytes, 0, "warm material switches do not reupload texture pixels");
  assert.equal(stats.resident_textures, 1, "all modes share one GPU color texture");
  return residency(await command("assets"));
}
try {
  await scene.route();
  await openPaused(page, sceneUrl());
  assert.deepEqual(await page.locator("canvas").evaluate(canvas => [canvas.width, canvas.height]), size);
  const reference = {};
  for (const id of ["eligible", "reduced"]) {
    await scene.reload(command, fixture([id]));
    reference[id] = await capture(`isolated-${id}`, boxes[id === "eligible" ? 0 : 1]);
  }
  // Keep the authored cutoff identical: opacity is the sole binding-relevant
  // difference. Swap both asset order and the entity extraction order.
  for (const order of [["eligible", "reduced"], ["reduced", "eligible"]]) {
    await scene.reload(command, fixture(order));
    const row = { order, images: {} };
    results.orders.push(row);
    for (const id of order) {
      const image = await capture(`order-${order[0]}-${id}`, boxes[id === "eligible" ? 0 : 1]);
      row.images[id] = image;
      assert.equal(image.hash, reference[id].hash, `${order.join(",")}: ${id} keeps its isolated coverage`);
      if (!sharingOnly) coverage(await inspect(id), id === "eligible" ? "corrected" : "legacy", id === "eligible" ? null : "material_opacity");
    }
    row.residency = await warmed();
  }
  if (!sharingOnly) {
    const schema = await command("describe", { name: "SceneDocument", definition: "SceneMaterial" });
    assert.ok(JSON.stringify(schema).includes("alpha_coverage"), "agent authoring schema documents the control");
    const resident = await warmed();
    let legacyReference;
    for (let cycle = 0; cycle < 3; cycle++) for (const mode of ["legacy", "auto"]) {
      await scene.reload(command, fixture(["reduced", "eligible"], { eligible: mode }));
      const image = await capture(`switch-${cycle}-${mode}`, boxes[0]);
      const other = await capture(`switch-${cycle}-${mode}-reduced`, boxes[1]);
      const view = await inspect();
      coverage(view, mode === "auto" ? "corrected" : "legacy", mode === "auto" ? null : "requested_legacy", mode);
      assert.equal(other.hash, reference.reduced.hash, "switching one material leaves its shared user unchanged");
      if (mode === "auto") assert.equal(image.hash, reference.eligible.hash, "auto restores the prepared curve");
      else {
        legacyReference ??= image;
        assert.equal(image.hash, legacyReference.hash, "legacy is stable across replacements");
        assert.notEqual(image.hash, reference.eligible.hash, "explicit legacy actually changes this minified mask");
      }
      assert.deepEqual(await warmed(), resident, "repeated mode replacement has stable residency");
      results.switching.push({ cycle, mode, image, coverage: view.alpha_coverage });
    }
    // Auto and legacy also coexist at unit opacity and identical cutoff.
    for (const order of [["eligible", "reduced"], ["reduced", "eligible"]]) {
      const doc = fixture(order, { reduced: "legacy" });
      doc.materials.find(x => x.id === "reduced").opacity = 1;
      await scene.reload(command, doc);
      assert.equal((await capture(`mode-order-${order[0]}-auto`, boxes[0])).hash, reference.eligible.hash);
      assert.equal((await capture(`mode-order-${order[0]}-legacy`, boxes[1])).hash, legacyReference.hash);
      coverage(await inspect("reduced"), "legacy", "requested_legacy", "legacy");
    }
    const cases = [
      ["opacity", doc => { doc.materials[0].opacity = 0.5; }, "material_opacity"],
      ["world", doc => { doc.materials[0].world_tiling = 1; }, "world_mapping"],
      ["nearest", doc => { doc.materials[0].texture = { id: "mask", mipmap_filter: "nearest" }; }, "sampler"],
      ["no-mips", doc => { doc.materials[0].texture = { id: "mask", mipmap_filter: "none" }; }, "sampler"],
      ["odd", doc => { doc.textures[0] = texture(63, 35); }, "texture_dimensions"],
      ["one-pixel", doc => { doc.textures[0] = texture(1, 1, 128); }, "no_mips"],
      ["full", doc => { doc.textures[0] = texture(32, 32, 255); }, "empty_or_full_coverage"],
      ["empty", doc => { doc.textures[0] = texture(32, 32, 0); }, "empty_or_full_coverage"],
      ["cutoff", doc => { doc.materials[0].alpha_cutoff = 1; }, "cutoff"],
      ["missing", doc => { doc.materials[0].texture = null; }, "missing_base_texture"],
      ["opaque", doc => { doc.materials[0].alpha_mode = "opaque"; delete doc.materials[0].alpha_cutoff; }, "not_masked"],
      ["tint", doc => { doc.entities[0].tint = [1, 1, 1, 0.5]; }, "instance_opacity"],
      ["vertex", doc => { doc.meshes[0].source = quad(1, 0.5); }, "vertex_opacity"],
    ];
    for (const [name, edit, reason] of cases) {
      const doc = fixture(["eligible"]); edit(doc);
      await scene.reload(command, doc);
      const image = await capture(`fallback-${name}-auto`, boxes[0]);
      const view = await inspect(); coverage(view, "legacy", reason);
      doc.materials[0].alpha_coverage = "legacy";
      await scene.reload(command, doc);
      const legacy = await capture(`fallback-${name}-legacy`, boxes[0]);
      assert.equal(image.hash, legacy.hash, `${name}: requested auto falls back to legacy pixels`);
      results.fallbacks.push({ name, image, coverage: view.alpha_coverage });
    }
    const mixed = fixture(["eligible"]); mixed.meshes[0].source = quad(1, "mixed");
    await scene.reload(command, mixed); await settled();
    coverage(await inspect(), "conditional", "vertex_opacity");

    // A material edit must rebuild source-material bakes and shadow state.
    // The two placed meshes exercise ordinary and extended vertex streams.
    const placed = {
      id: "coverage-mode-fixture", revision: 1,
      camera: { position: [10, 12, 40], target: [10, 3, 0], fov_degrees: 55, near: 0.1, far: 5000 },
      textures: [texture()], materials: [material("eligible", 1, { emissive: [0, 0, 0], color: [0.5, 0.8, 0.2, 1] }),
        { id: "receiver", color: [0.7, 0.7, 0.7, 1] }],
      meshes: [{ id: "card", source: quad(8) }, { id: "extended", source: quad(8, 0.99) }, { id: "box", source: { kind: "cube" } }],
      environment: { shadows: { enabled: true }, level_of_detail: { enabled: true }, terrain: { samples_per_frame: 50_000_000 },
        temporal_antialiasing: { enabled: false }, atmosphere: { fog_density: 0, fog_max: 0, aerial_density: 0 }, bloom: { intensity: 0 } },
      entities: [
        { id: "eligible", mesh: "card", material: "eligible", transform: { translation: [10, 5, 0], scale: [4, 4, 4] } },
        { id: "receiver", mesh: "box", material: "receiver", transform: { translation: [10, -1, 0], scale: [70, 1, 70] } },
        { id: "terrain", terrain: { origin: [-128, -0.25, -128], tiles: [1, 1], spacing: 1,
          layers: [{ noise: { seed: 3, height: 0, wavelength: 256 } }], surface: {
            materials: [{ id: "soil", color: [0.2, 0.25, 0.15] }], biomes: [{ id: "plain", layers: [{ material: "soil" }] }],
            assets: [{ id: "basic", mesh: "card", material: "eligible" }, { id: "extended", mesh: "extended", material: "eligible" }],
            placed: [{ asset: "basic", position: [128, 128], scale: 4, sink: -1 }, { asset: "extended", position: [148, 128], scale: 4, sink: -1 }],
          } } },
      ],
    };
    for (const [cycle, mode] of ["auto", "legacy", "auto"].entries()) {
      placed.materials[0].alpha_coverage = mode;
      await scene.reload(command, placed);
      let stats;
      for (let frame = 0; frame < 600; frame++) {
        stats = await settled();
        if (stats.terrain.placed_impostors >= 2) break;
        await delay(20);
      }
      assert.ok(stats.terrain.placed_impostors >= 2, "both source material bakes are ready after replacement");
      assert.ok(stats.shadows.active && stats.shadows.draws > 0, "masked shadow casters actually draw after replacement");
      coverage(await inspect(), mode === "auto" ? "corrected" : "legacy", mode === "auto" ? null : "requested_legacy", mode);
      const row = { cycle, mode, views: [] }; results.placed.push(row);
      for (const [label, distance] of [["near", 40], ["transition", 200], ["far", 600]]) {
        await command("camera", { position: [10, 12, distance], target: [10, 3, 0], vertical_fov_radians: 55 * Math.PI / 180, near: 0.1, far: 5000 });
        for (let frame = 0; frame < 30; frame++) await command("present", { alpha: 0, gpu: true });
        const image = await capture(`placed-${cycle}-${mode}-${label}`);
        const repeat = await capture(`placed-${cycle}-${mode}-${label}-repeat`);
        assert.equal(image.hash, repeat.hash, "static source/bake captures are repeatable");
        if (cycle === 2) assert.equal(image.hash, results.placed[0].views.find(view => view.label === label).image.hash,
          "restoring auto restores foliage, shadows and final impostor pixels");
        row.views.push({ label, image, rendering: (await command("render")).result.rendering });
      }
      assert.ok(row.views.some(view => view.rendering.level_of_detail.impostor_instances > 0), "far view draws baked impostors");
    }
  }
  assert.deepEqual(errors, [], "clean renderer and browser diagnostics");
  await writeFile(`${output}/results.json`, JSON.stringify({ ok: true, ...results }, null, 2));
  console.log(JSON.stringify({ ok: true, output, orders: results.orders.length, switches: results.switching.length,
    fallbacks: results.fallbacks.length, placed: results.placed.length, measurements: [] }));
} catch (error) {
  await writeFile(`${output}/failure.json`, JSON.stringify({ ok: false, error: String(error), stack: error.stack, errors, ...results }, null, 2));
  console.error(error);
  process.exitCode = error.code === "ERR_ASSERTION" ? 3 : 1;
} finally {
  await browser.close();
}
