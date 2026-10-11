/** Material-aware alpha coverage: independent source-area targets, old-path
 * controls, shared-texture lifetime, source bakes and ordinary Highlands frames. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, launch, openPaused, residency, routedScene, sceneUrl } from "./harness.mjs";

const output = await artifacts("alpha-coverage-smoke");
const results = { coverage: [], fallbacks: [], distances: [], shared: [], draws: [], demos: [] };
const mask = (width = 256, height = 256, fill) => {
  const pixels = Buffer.alloc(width * height * 4, 255);
  let opaque = 0;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const c = Math.floor(x / 32), r = Math.floor(y / 32);
    const rx = 4 + (c + 3 * r) % 8, ry = 3 + (3 * c + r) % 6;
    const inside = ((x % 32 + 0.5 - 16) / rx) ** 2 + ((y % 32 + 0.5 - 16) / ry) ** 2 <= 1;
    const alpha = fill ?? (inside ? 255 : 0);
    pixels[(y * width + x) * 4 + 3] = alpha;
    opaque += alpha === 255 ? 1 : 0;
  }
  return { texture: { id: "mask", width, height, rgba8: pixels.toString("base64") }, fraction: opaque / (width * height) };
};
const sourceMask = mask();
const blackEnvironment = {
  atmosphere: { sun_color: [0, 0, 0], zenith: [0, 0, 0], horizon: [0, 0, 0], ground: [0, 0, 0],
    ambient: 0, fog_density: 0, fog_max: 0, aerial_density: 0, rim: 0, specular: 0, emission: 1,
    exposure: 1, saturation: 1, contrast: 1 },
  shadows: { enabled: false }, level_of_detail: { enabled: false }, temporal_antialiasing: { enabled: false },
  bloom: { intensity: 0 }, ambient_occlusion: { intensity: 0 }, contact_shadows: { enabled: false },
};
function quad(grid = 1, vertexOpacity) {
  const vertices = [], indices = [];
  for (let y = 0; y <= grid; y++) for (let x = 0; x <= grid; x++) {
    vertices.push({ position: [2 * x / grid - 1, 1 - 2 * y / grid, 0], normal: [0, 0, 1], uv: [x / grid, y / grid],
      ...(vertexOpacity === undefined ? {} : { color: [1, 1, 1, vertexOpacity] }) });
  }
  for (let y = 0; y < grid; y++) for (let x = 0; x < grid; x++) {
    const a = y * (grid + 1) + x, b = a + grid + 1;
    indices.push(a, b, a + 1, a + 1, b, b + 1);
  }
  return { kind: "inline", vertices, indices };
}
const material = (cutoff, opacity = 1, extra = {}) => ({ id: "cutout", color: [1, 1, 1, 1], texture: "mask",
  emissive: [8, 8, 8], alpha_mode: "mask", alpha_cutoff: cutoff, opacity, double_sided: true, ...extra });
let dimensions = [512, 512];
function fixture(size, cutoff, { opacity = 1, texture = sourceMask.texture, map = {}, vertexOpacity, tint,
  worldTiling, rectangle, extraMaterial = {} } = {}) {
  const [width, height] = dimensions;
  const roi = rectangle ?? { x: Math.floor((width - size) / 2), y: Math.floor((height - size) / 2), width: size, height: size };
  // A 90-degree vertical camera at z=4 maps these bounds to integer pixel
  // edges. Pixel centers then address source/mip texel centers at powers of two.
  const unit = 8 / height;
  return { roi, document: {
    version: 9, id: "alpha-coverage-fixture", revision: 1,
    camera: { position: [0, 0, 4], target: [0, 0, 0], fov_degrees: 90, near: 0.1, far: 100 },
    textures: [texture],
    materials: [material(cutoff, opacity, { texture: { id: "mask", ...map },
      ...(worldTiling === undefined ? {} : { world_tiling: worldTiling }), ...extraMaterial })],
    meshes: [{ id: "quad", source: quad(1, vertexOpacity) }],
    environment: structuredClone(blackEnvironment),
    entities: [{ id: "card", mesh: "quad", material: "cutout", ...(tint ? { tint } : {}), transform: {
      translation: [(roi.x + size / 2 - width / 2) * unit, (height / 2 - roi.y - size / 2) * unit, 0],
      scale: [size * unit / 2, size * unit / 2, 1],
    } }],
  } };
}

const { browser, page, errors } = await launch({ width: 900, height: 740, timeout: 120_000 });
const command = commander(page, { deadline: 120_000 });
const heapSession = await page.context().newCDPSession(page);
const scene = routedScene(page, fixture(256, 0.5).document);
// Real resize events size the targets. This changes only the smoke page's
// layout, so the later demo has the specified 1400x900 output, not a sidebar crop.
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
    stats = (await command("render")).result.rendering;
    if (!stats.pipelines_compiling) return stats;
    await delay(20);
  }
  assert.fail(`pipelines did not settle: ${JSON.stringify(stats)}`);
}
async function capture(name, roi) {
  await settled();
  await command("capture");
  const png = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  const observed = await page.evaluate(async ({ png, roi }) => {
    const image = new Image(); image.src = `data:image/png;base64,${png}`; await image.decode();
    const canvas = document.createElement("canvas"); canvas.width = roi.width; canvas.height = roi.height;
    const context = canvas.getContext("2d"); context.drawImage(image, roi.x, roi.y, roi.width, roi.height, 0, 0, roi.width, roi.height);
    const { data } = context.getImageData(0, 0, roi.width, roi.height);
    const mask = new Uint8Array(data.length / 4);
    let count = 0, ambiguous = 0;
    for (let i = 0; i < mask.length; i++) {
      const bright = Math.min(data[4 * i], data[4 * i + 1], data[4 * i + 2]);
      mask[i] = bright > 128 ? 1 : 0; count += mask[i];
      if (bright > 8 && bright < 224) ambiguous++;
    }
    return { size: [image.width, image.height], count, pixels: mask.length, fraction: count / mask.length,
      ambiguous, mask: btoa(String.fromCharCode(...mask)), pixelsBase64: btoa(Array.from(data, x => String.fromCharCode(x)).join("")),
      crop: canvas.toDataURL("image/png").split(",")[1] };
  }, { png: png.toString("base64"), roi });
  assert.deepEqual(observed.size, dimensions, "captures keep the requested real target dimensions");
  await writeFile(`${output}/${name}-crop.png`, Buffer.from(observed.crop, "base64"));
  delete observed.crop;
  observed.hash = createHash("sha256").update(Buffer.from(observed.mask, "base64")).digest("hex");
  return observed;
}
async function sample(name, size, cutoff, options = {}) {
  const { document, roi } = fixture(size, cutoff, options);
  await scene.reload(command, document);
  const image = await capture(name, roi);
  assert.equal(image.ambiguous, 0, `${name}: emissive foreground and black background classify cleanly`);
  return image;
}
const brief = ({ count, pixels, fraction, hash }) => ({ count, pixels, fraction, hash });
const rawDifference = (a, b) => {
  const x = Buffer.from(a.pixelsBase64, "base64"), y = Buffer.from(b.pixelsBase64, "base64");
  assert.equal(x.length, y.length);
  let changed = 0;
  for (let i = 0; i < x.length; i += 4) if (x[i] !== y[i] || x[i + 1] !== y[i + 1] || x[i + 2] !== y[i + 2]) changed++;
  return changed;
};
async function sameFallback(name, size, cutoff, options = {}, controlOptions = {}) {
  const first = await sample(`${name}-a`, size, cutoff, options);
  const second = await sample(`${name}-legacy`, size, cutoff / 2, { ...options, opacity: 0.5, ...controlOptions });
  assert.equal(first.hash, second.hash, `${name}: unsupported case retains legacy binary acceptance`);
  results.fallbacks.push({ name, first: brief(first), legacy: brief(second) });
}

try {
  await scene.route();
  await openPaused(page, sceneUrl());
  dimensions = await page.locator("canvas").evaluate(canvas => [canvas.width, canvas.height]);
  assert.deepEqual(dimensions, [900, 740]);
  results.target_fraction = sourceMask.fraction;
  // The target is counted directly from the authored binary mask. This check
  // contains no histogram curve or mip-generation implementation.
  for (const cutoff of [0.2, 0.5, 0.8]) for (let mip = 0; mip <= 3; mip++) {
    const size = 256 >> mip;
    const eligible = await sample(`T${cutoff}-mip${mip}`, size, cutoff);
    const legacy = await sample(`T${cutoff}-mip${mip}-legacy`, size, cutoff / 2, { opacity: 0.5 });
    const error = Math.abs(eligible.fraction - sourceMask.fraction);
    const oldError = Math.abs(legacy.fraction - sourceMask.fraction);
    results.coverage.push({ cutoff, mip, eligible: brief(eligible), legacy: brief(legacy), error, old_error: oldError });
    if (mip === 0) {
      assert.equal(eligible.hash, legacy.hash, "base acceptance is unchanged");
      assert.equal(eligible.fraction, sourceMask.fraction, "base pixels hit the authored texel centers");
    } else {
      assert.ok(error <= 0.02, `T=${cutoff}, mip=${mip}: coverage error ${error} exceeds 0.02`);
      assert.ok(error <= oldError + 0.01, `T=${cutoff}, mip=${mip}: error ${error} vs legacy ${oldError}`);
    }
    if (cutoff === 0.5 && mip === 3) assert.ok(oldError - error >= 0.02, `mip 3 improvement ${oldError - error} is below 0.02`);
  }

  for (const size of [181, 91, 45, 23, 16, 8, 4, 2, 1]) {
    const image = await sample(`distance-${size}`, size, 0.5);
    results.distances.push({ size, ...brief(image) });
  }
  // At the uniform last mip the source area is near 0.125. A repeated map
  // lets a full four-pixel card test extreme footprints without losing geometry.
  for (const [scale, cutoff, full] of [[1, 0.2, true], [1, 0.5, false], [256, 0.5, true]]) {
    const size = scale === 1 ? 1 : 4;
    const image = await sample(`tail-${scale}-${cutoff}`, size, cutoff, { map: { scale: [scale, scale] } });
    assert.equal(image.count, full ? image.pixels : 0, "uniform and unbounded legacy tail endpoint");
    results.distances.push({ size, scale, cutoff, full, ...brief(image) });
  }
  for (const cutoff of [0, 1, 1.5]) await sameFallback(`cutoff-${cutoff}`, 32, cutoff);
  for (const [name, texture] of [["zero", mask(32, 32, 0).texture], ["one", mask(32, 32, 255).texture],
    ["one-pixel", mask(1, 1, 128).texture], ["odd", mask(63, 35).texture]]) {
    await sameFallback(name, 32, 0.5, { texture });
  }
  for (const mipmap_filter of ["none", "nearest"]) await sameFallback(`mipmap-${mipmap_filter}`, 32, 0.5, { map: { mipmap_filter } });
  await sameFallback("nearest-min", 32, 0.5, { map: { min_filter: "nearest" } });
  await sameFallback("nearest-mag", 32, 0.5, { map: { mag_filter: "nearest" } });
  await sameFallback("absent-map", 32, 0.5, { extraMaterial: { texture: null } });
  await sameFallback("world", 32, 0.5, { worldTiling: 1 });
  const tinted = await sample("tint-opacity", 32, 0.1, { tint: [1, 1, 1, 0.2] });
  const vertex = await sample("vertex-opacity", 32, 0.1, { vertexOpacity: 0.2 });
  const opacity = await sample("material-opacity", 32, 0.1, { opacity: 0.2 });
  assert.equal(tinted.hash, opacity.hash, "tint opacity uses the legacy effective cutoff");
  assert.equal(vertex.hash, opacity.hash, "vertex opacity uses the legacy effective cutoff");
  results.fallbacks.push({ name: "combined-opacity", tint: brief(tinted), vertex: brief(vertex), material: brief(opacity) });

  // Two cutoff materials and three other users share one color texture.
  // Removing/replacing one mask leaves every other user's pixels untouched.
  const boxes = [0, 1, 2, 3].map(index => ({ x: 60 + index * 190, y: 290, width: 128, height: 128 }));
  boxes.push({ x: 60, y: 470, width: 128, height: 128 });
  function sharedDocument() {
    const doc = fixture(128, 0.5, { rectangle: boxes[0] }).document;
    for (let index = 1; index < boxes.length; index++) {
      const one = fixture(128, 0.5, { rectangle: boxes[index] }).document;
      doc.entities.push({ ...one.entities[0], id: `user-${index}`, material: `user-${index}` });
      doc.materials.push(index === 1 ? { id: `user-${index}`, color: [0.8, 0.3, 0.2, 1], texture: "mask", emissive: [2, 0.3, 0.1] }
        : index === 2 ? { id: `user-${index}`, color: [0.2, 0.7, 0.3, 1], texture: "mask", opacity: 0.5, emissive: [0.1, 2, 0.3] }
          : index === 3 ? { id: `user-${index}`, color: [0.3, 0.4, 0.9, 1], texture: "mask", surface: { emission: 8 } }
            : { ...material(0.8), id: `user-${index}` });
    }
    return doc;
  }
  const shared = sharedDocument();
  results.heap_context = { before_shared: await heapSession.send("Runtime.getHeapUsage"),
    interpretation: "Whole-page JavaScript and backing storage context, including other resources; not an isolated histogram allocation measurement." };
  await scene.reload(command, shared);
  const initial = [];
  for (let i = 0; i < boxes.length; i++) initial.push(await capture(`shared-initial-${i}`, boxes[i]));
  const repeat = [];
  for (let i = 0; i < boxes.length; i++) repeat.push(await capture(`shared-repeat-${i}`, boxes[i]));
  const noise = initial.map((image, index) => rawDifference(image, repeat[index]));
  const beforeAssets = await command("assets");
  const resident = residency(beforeAssets);
  assert.equal(resident.gpuTextures, 1, "shared users upload one color texture");
  for (const mode of ["removed", "cutoff-zero", "restored"]) {
    const next = structuredClone(shared);
    if (mode === "removed") next.entities.shift();
    if (mode === "cutoff-zero") next.materials[0].alpha_cutoff = 0;
    await scene.reload(command, next);
    for (let i = 1; i < boxes.length; i++) {
      const image = await capture(`shared-${mode}-${i}`, boxes[i]);
      assert.ok(rawDifference(initial[i], image) <= noise[i], `${mode}: shared user ${i} remains unchanged`);
    }
    const own = await capture(`shared-${mode}-0`, boxes[0]);
    if (mode === "cutoff-zero") assert.equal(own.count, own.pixels, "replacement cutoff applies now");
    if (mode === "restored") assert.equal(own.hash, initial[0].hash, "restored cutoff uses its curve");
    const stats = (await command("render")).result.rendering;
    assert.equal(stats.texture_upload_bytes, 0, "warm material frames do not reupload pixels");
    assert.equal(stats.resident_textures, 1);
    results.shared.push({ mode, residency: residency(await command("assets")), noise });
  }
  const inverted = structuredClone(shared);
  const pixels = Buffer.from(inverted.textures[0].rgba8, "base64");
  for (let i = 3; i < pixels.length; i += 4) pixels[i] = 255 - pixels[i];
  inverted.textures[0].rgba8 = pixels.toString("base64");
  for (let cycle = 0; cycle < 3; cycle++) for (const [name, document] of [["inverted", inverted], ["restored", shared]]) {
    await scene.reload(command, document);
    const image = await capture(`reload-${cycle}-${name}`, boxes[0]);
    if (name === "inverted") assert.notEqual(image.hash, initial[0].hash, "replacement source alpha changes coverage");
    else assert.equal(image.hash, initial[0].hash, "restoring source pixels restores the curve");
    assert.deepEqual(residency(await command("assets")), resident, "repeated source replacement keeps live residency bounded");
    assert.equal((await command("render")).result.rendering.texture_upload_bytes, 0);
    results.shared.push({ cycle, name, image: brief(image), residency: resident });
  }
  results.heap_context.after_shared_cycles = await heapSession.send("Runtime.getHeapUsage");

  // Masked source meshes large enough to receive impostor bakes, both with
  // and without a vertex-color stream, are also terrain-placed casters.
  const placed = {
    version: 9, id: "alpha-coverage-fixture", revision: 1,
    camera: { position: [10, 12, 40], target: [10, 3, 0], fov_degrees: 55, near: 0.1, far: 5000 },
    textures: [sourceMask.texture], materials: [material(0.5, 1, { emissive: [0, 0, 0], color: [0.5, 0.8, 0.2, 1] }),
      { id: "receiver", color: [0.7, 0.7, 0.7, 1] }],
    meshes: [{ id: "card", source: quad(8) }, { id: "extended-card", source: quad(8, 1) }, { id: "box", source: { kind: "cube" } }],
    environment: { shadows: { enabled: true }, level_of_detail: { enabled: true }, terrain: { samples_per_frame: 50_000_000 },
      atmosphere: { fog_density: 0, fog_max: 0, aerial_density: 0 }, bloom: { intensity: 0 } },
    entities: [
      { id: "caster", mesh: "card", material: "cutout", transform: { translation: [10, 5, 0], scale: [4, 4, 4] } },
      { id: "receiver", mesh: "box", material: "receiver", transform: { translation: [10, -1, 0], scale: [70, 1, 70] } },
      { id: "terrain", terrain: { origin: [-128, -0.25, -128], tiles: [1, 1], spacing: 1,
        layers: [{ noise: { seed: 3, height: 0, wavelength: 256 } }], surface: {
        materials: [{ id: "soil", color: [0.2, 0.25, 0.15] }], biomes: [{ id: "plain", layers: [{ material: "soil" }] }],
        assets: [{ id: "basic", mesh: "card", material: "cutout" }, { id: "extended", mesh: "extended-card", material: "cutout" }],
        placed: [{ asset: "basic", position: [128, 128], scale: 4, sink: -1 },
          { asset: "extended", position: [148, 128], scale: 4, sink: -1 }],
      } } },
    ],
  };
  await scene.reload(command, placed);
  let draw;
  for (let frame = 0; frame < 600; frame++) {
    draw = (await command("present", { alpha: 0, gpu: true })).result.rendering;
    if (draw.terrain.loading_tiles === 0 && draw.terrain.placed_impostors >= 2 && !draw.pipelines_compiling) break;
    await delay(20);
  }
  assert.ok(draw.terrain.placed_impostors >= 2, "both basic and extended placed cutouts receive source bakes");
  assert.ok(draw.shadows.active && draw.shadows.draws > 0, "masked casters draw into active shadow maps");
  assert.equal(draw.level_of_detail.dropped_impostors, 0);
  for (const [label, distance] of [["near", 40], ["transition", 200], ["far", 600]]) {
    await command("camera", { position: [10, 12, distance], target: [10, 3, 0], vertical_fov_radians: 55 * Math.PI / 180, near: 0.1, far: 5000 });
    for (let frame = 0; frame < 30; frame++) draw = (await command("present", { alpha: 0, gpu: true })).result.rendering;
    await page.locator("canvas").screenshot({ path: `${output}/placed-${label}.png` });
    results.draws.push({ label, rendering: draw, camera: (await command("inspect", { component: "Camera" })).result.value });
  }
  assert.ok(results.draws.some(({ rendering }) => rendering.level_of_detail.impostor_instances > 0), "a far view actually selects baked impostors");

  // Default compressed Metro is a control; unchanged Highlands is visual
  // review material. Timing comparisons run separately without this fixture.
  const generated = spawnSync("cargo", ["xtask", "bench", "assets"], { cwd: fileURLToPath(new URL("../../", import.meta.url)),
    encoding: "utf8", windowsHide: true });
  await writeFile(`${output}/assets.log`, `${generated.stdout ?? ""}${generated.stderr ?? ""}`);
  assert.equal(generated.status, 0, `default assets prepare: ${generated.error ?? generated.status}`);
  await page.setViewportSize({ width: 1400, height: 900 }); dimensions = [1400, 900];
  await page.route("**/stress/rocks/*", route => route.fulfill({ status: 204, body: "" }));
  for (const preset of ["metro", "highlands"]) {
    const url = devUrl("stress.html", { preset, view: "overview", look: "scene", effects: "all", visibility: "off",
      lod: "on", impostors: "on", rocks: "off", jobcache: "off" });
    await openPaused(page, url, { timeout: 120_000 });
    await settled();
    await command("capture");
    await page.locator("canvas").screenshot({ path: `${output}/${preset}-exact.png` });
    const observations = [];
    const demo = { preset, url: url.href, observations };
    results.demos.push(demo);
    for (const view of preset === "highlands" ? ["overview", "close"] : ["overview"]) {
      if (view === "close") await command("game", { name: "view", args: { name: "close" } });
      const camera = (await command("inspect", { component: "Camera" })).result.value;
      const before = await page.locator("canvas").evaluate(canvas => Number(canvas.dataset.frames ?? 0));
      assert.ok(Number.isFinite(before), "ordinary-frame baseline is numeric");
      await command("resume");
      await page.waitForFunction(start => Number(document.querySelector("canvas").dataset.frames) >= start + 32, before, { timeout: 120_000 });
      for (let index = 0; index < 4; index++) {
        await page.locator("canvas").screenshot({ path: `${output}/${preset}-${view}-ordinary-${index}.png` });
        observations.push({ view, camera, before, frame: await page.locator("canvas").evaluate(canvas => Number(canvas.dataset.frames)) });
        await delay(80);
      }
      await command("pause");
    }
    const status = await command("status");
    if (preset === "highlands") assert.equal(status.result.rendering.temporal_antialiasing.active, true, "ordinary Highlands frames use actual TAA");
    Object.assign(demo, { status, assets: await command("assets"),
      heap_context: await heapSession.send("Runtime.getHeapUsage"),
      optional_maps: "Empty successful optional scanned-rock responses select the existing terrain-map fallback.",
      interpretation: "Captures and costs are local review context, never a factory image/performance verdict." });
  }
  assert.deepEqual(errors, [], "clean browser and GPU diagnostics");
  await writeFile(`${output}/results.json`, JSON.stringify({ ok: true, ...results }, null, 2));
  console.log(JSON.stringify({ ok: true, output, target: sourceMask.fraction, coverage: results.coverage,
    fallback_cases: results.fallbacks.length, shared_cycles: results.shared.length, placed_views: results.draws.length }));
} catch (error) {
  await writeFile(`${output}/failure.json`, JSON.stringify({ ok: false, error: String(error), stack: error.stack, errors, ...results }, null, 2));
  console.error(error);
  process.exitCode = error.code === "ERR_ASSERTION" ? 3 : 1;
} finally {
  await browser.close();
}
