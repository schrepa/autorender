/** Main-view radiance is composed before one display grade. Constant-emissive
 * routed scenes isolate the same weighted blend, with explicit storage
 * quantization, from lights, fog, temporal filtering and sorted OVER. */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, launch, openPaused, routedScene, sceneUrl } from "./harness.mjs";
import { compose, encode, grade, inverseGrey, neutral, unorm } from "./linear-light-reference.mjs";

const output = await artifacts("linear-light-transparency-smoke");
const { browser, page, errors } = await launch({ width: 960, height: 720, timeout: 60_000 });
const command = commander(page, { deadline: 60_000 });
const B = [0.02, 0.04, 0.08], C = [8, 0.1, 0.01];
const checks = [];
const measurements = [];
const imageComparisons = [];
let lastDiagnostic = null;
const atmosphere = {
  sun_direction: [0, 1, 0], sun_color: [0, 0, 0], zenith: [0, 0, 0],
  horizon: [0, 0, 0], ground: [0, 0, 0], ambient: 0, rim: 0, specular: 0,
  fog_density: 0, fog_max: 0, aerial_density: 0, emission: 1, exposure: 1, ...neutral,
};
const environment = {
  atmosphere, shadows: { enabled: false }, ambient_occlusion: { intensity: 0 },
  bloom: { intensity: 0 }, reflections: { intensity: 0 }, sun_shafts: { intensity: 0 },
  refraction: { strength: 0 }, contact_shadows: { enabled: false },
  clouds: { enabled: false }, sky: { enabled: false }, temporal_antialiasing: { enabled: false },
  reflection_probes: { enabled: false }, virtual_geometry: { enabled: false },
  level_of_detail: { enabled: false },
};
const quad = { id: "panel", source: { kind: "inline", vertices: [
  { position: [-12, -12, 0], normal: [0, 0, 1], uv: [0, 0] },
  { position: [12, -12, 0], normal: [0, 0, 1], uv: [1, 0] },
  { position: [12, 12, 0], normal: [0, 0, 1], uv: [1, 1] },
  { position: [-12, 12, 0], normal: [0, 0, 1], uv: [0, 1] },
], indices: [0, 1, 2, 0, 2, 3] } };
function fixture(layers = [], background = B) {
  return {
    id: "linear-light-fixture", revision: 1,
    camera: { position: [0, 0, 6], target: [0, 0, 0], fov_degrees: 55, near: 0.01, far: 100 },
    environment: structuredClone(environment), meshes: [structuredClone(quad)],
    materials: [{ id: "back", color: [0, 0, 0, 1], emissive: background }, ...layers.map((layer, i) => ({
      id: `layer-${i}`, color: [0, 0, 0, 1], emissive: layer.color,
      opacity: layer.alpha || 1, alpha_mode: "blend", ...(layer.alpha === 0 ? { texture: "clear" } : {}),
    }))],
    textures: [{ id: "clear", width: 1, height: 1, rgba8: Buffer.from([255, 255, 255, 0]).toString("base64") }],
    entities: [{ id: "back", mesh: "panel", material: "back" }, ...layers.map((layer, i) => ({
      id: `layer-${i}`, mesh: "panel", material: `layer-${i}`,
      transform: { translation: [0, 0, 6 - layer.distance] },
    }))],
  };
}
const scene = routedScene(page, fixture());

/** Observe real WebGPU texture retirement without a production test control. */
await page.addInitScript(() => {
  const records = { created: 0, destroyed: 0, live: 0, sizes: [] };
  window.__radianceLifecycle = records;
  const watched = new WeakSet();
  const create = GPUDevice.prototype.createTexture, destroy = GPUTexture.prototype.destroy;
  GPUDevice.prototype.createTexture = function (descriptor) {
    const texture = create.call(this, descriptor);
    if (descriptor.label === "engine/radiance") {
      watched.add(texture); records.created++; records.live++; records.sizes.push(descriptor.size);
    }
    return texture;
  };
  GPUTexture.prototype.destroy = function () {
    if (watched.delete(this)) { records.destroyed++; records.live--; }
    return destroy.call(this);
  };
});

function closeRGB(actual, expected, label, tolerance = 4) {
  const error = actual.slice(0, 3).map((value, i) => Math.abs(value - expected[i]));
  const maximum = Math.max(...error);
  imageComparisons.push({ label, actual, expected, error, maximum, referenceTolerance: tolerance });
  measurements.push({
    name: `${label}: maximum channel error`, value: maximum, unit: "byte-code",
    note: `Accepted-design reference tolerance ${tolerance}; actual ${actual}; reference ${expected}; channel errors ${error}. Image measurements do not decide the check.`,
  });
}
async function center(name) {
  const png = await page.locator("canvas").screenshot({ path: `${output}/${name}.png`,
    style: ".viewport > :not(canvas) { visibility: hidden !important; }" });
  return page.evaluate(async base64 => {
    const image = new Image(); image.src = `data:image/png;base64,${base64}`; await image.decode();
    const canvas = document.createElement("canvas"); canvas.width = image.width; canvas.height = image.height;
    const context = canvas.getContext("2d"); context.drawImage(image, 0, 0);
    return [...context.getImageData(Math.floor(image.width / 2), Math.floor(image.height / 2), 1, 1).data];
  }, png.toString("base64"));
}
function observe(diagnostic) {
  const { source_frame: frame, limit_mask: mask } = diagnostic;
  assert.equal(frame === null, mask === null, "the diagnostic is an atomic frame/mask pair");
  if (frame !== null) {
    assert.ok(Number.isSafeInteger(frame) && frame > 0 && Number.isInteger(mask) && mask >= 0 && mask <= 3);
    if (lastDiagnostic) {
      assert.ok(frame >= lastDiagnostic.source_frame, "completed source frames never go backward");
      if (frame === lastDiagnostic.source_frame) assert.equal(mask, lastDiagnostic.limit_mask, "pending readback retains the pair");
    }
    lastDiagnostic = { source_frame: frame, limit_mask: mask };
  } else assert.equal(lastDiagnostic, null, "completed evidence never becomes an invented null/zero pair");
}
async function limits(expected) {
  // Start from the latest completed pair after this case has submitted,
  // rather than a pair retained from the previous scene or exposure.
  let rendering = (await command("render")).result.rendering;
  assert.ok(Object.hasOwn(rendering, "radiance"), "range observations are present");
  observe(rendering.radiance);
  const start = lastDiagnostic?.source_frame ?? 0;
  for (let i = 0; i < 30; i++) {
    rendering = (await command("render")).result.rendering;
    assert.ok(Object.hasOwn(rendering, "radiance"), "range observations are present");
    observe(rendering.radiance);
    if (rendering.radiance.source_frame >= start + 5) {
      assert.equal(rendering.radiance.limit_mask, expected, `completed range mask: ${JSON.stringify(rendering.radiance)}`);
      return rendering;
    }
    await delay(15);
  }
  assert.fail(`range readback must complete: ${JSON.stringify(rendering?.radiance)}`);
}
async function sample(name, reference, tolerance = 4) {
  await captureReady();
  const actual = await center(name);
  closeRGB(actual, reference.rgb, name, tolerance);
  const rendering = await limits(reference.mask);
  assert.ok(rendering.render_passes.includes("grade"), "every main frame grades");
  checks.push({ name, actual, expected: reference.rgb, mask: rendering.radiance });
  return { actual, rendering };
}
async function captureReady() {
  for (let attempt = 0; attempt < 60; attempt++) {
    const rendering = (await command("capture")).result.rendering;
    if (!rendering.loading && !rendering.pipelines_compiling && rendering.missing_assets === 0
        && rendering.render_passes.length > 0) return;
    await delay(25);
  }
  assert.fail("the analytic fixture must submit a complete frame before sampling");
}
async function look(exposure, settings = neutral, extra = {}) {
  await command("environment", { atmosphere: { ...atmosphere, exposure, ...settings, ...extra } });
}
async function ordinary(name, reference) {
  const previous = await page.locator("canvas").evaluate(canvas => canvas.dataset.frames ?? null);
  await command("resume");
  // The host publishes this counter only on ordinary frames, four times a
  // second. It can be absent initially or lag the previous pause. Start
  // from a fresh publication after resuming this exposure, then count.
  await page.waitForFunction(old => {
    const frames = document.querySelector("canvas").dataset.frames;
    return frames !== undefined && frames !== old && Number.isSafeInteger(Number(frames));
  }, previous, { timeout: 60_000 });
  const before = await page.locator("canvas").evaluate(canvas => Number(canvas.dataset.frames));
  await page.waitForFunction(first => Number(document.querySelector("canvas").dataset.frames) >= first + 40, before,
    { timeout: 60_000 });
  const actual = await center(name);
  await command("pause");
  const rendering = (await command("status")).result.rendering;
  closeRGB(actual, reference.rgb, name);
  assert.equal(rendering.temporal_antialiasing.active, true, "actual ordinary frames use TAA");
  assert.ok(rendering.render_passes.includes("temporal_resolve"));
  observe(rendering.radiance);
  checks.push({ name, actual, expected: reference.rgb, ordinaryFrames: 40, diagnostic: rendering.radiance });
}

async function resize(width, height) {
  await page.addStyleTag({ content: `canvas { width: ${width}px !important; height: ${height}px !important; }` });
  await page.waitForFunction(([w, h]) => {
    const bounds = document.querySelector("canvas").getBoundingClientRect();
    return bounds.width === w && bounds.height === h;
  }, [width, height]);
  // The first paused render may prepare a resized target/pipeline without
  // presenting. Require the physical image to catch up before sampling it.
  for (let attempt = 0; attempt < 20; attempt++) {
    await command("capture");
    const size = await page.locator("canvas").evaluate(canvas => [canvas.width, canvas.height]);
    if (size[0] === width && size[1] === height) return;
    await delay(25);
  }
  assert.fail(`physical canvas did not resize to ${width}x${height}`);
}

try {
  await scene.route();
  await openPaused(page, sceneUrl());
  await resize(641, 481);
  const size = await page.locator("canvas").evaluate(canvas => [canvas.width, canvas.height]);
  assert.deepEqual(size, [641, 481], "an odd physical canvas centers the analytical ray");
  // Preserve the rendered grade-order measurement before asking for any
  // newly added observation or pass name.
  const discriminating = [{ color: C, alpha: 0.25, distance: 4 }];
  await scene.reload(command, fixture(discriminating));
  await captureReady();
  closeRGB(await center("grade-order-first"), compose(B, discriminating, 1).rgb, "grade follows composition");
  await scene.reload(command, fixture());
  await sample("opaque", compose(B, [], 1), 2);

  const grades = [[1, 1, 0], [0.5, 0.7, 0.6], [1.5, 1.4, 0.6]];
  const tints = [[neutral.shadow_tint, neutral.highlight_tint], [[0.25, 0.6, 1], [1, 0.6, 0.25]]];
  for (const alpha of [0.25, 0.5, 0.75, 0, 1]) {
    const layers = [{ color: C, alpha, distance: 4 }];
    await scene.reload(command, fixture(layers));
    for (const exposure of [0, 0.25, 1, 4]) {
      for (let g = 0; g < grades.length; g++) for (let t = 0; t < tints.length; t++) {
        const [contrast, saturation, vibrance] = grades[g], [shadow_tint, highlight_tint] = tints[t];
        const settings = { contrast, saturation, vibrance, shadow_tint, highlight_tint };
        await look(exposure, settings);
        await sample(`single-${alpha}-${exposure}-${g}-${t}`, compose(B, layers, exposure, settings), alpha === 0 ? 2 : 4);
      }
    }
  }
  const reveal = unorm(1 - unorm(0.25, 65535), 255);
  const old = encode(grade(C).map((x, i) => x * (1 - reveal) + grade(B)[i] * reveal));
  const accepted = compose(B, [{ color: C, alpha: 0.25, distance: 4 }], 1).rgb;
  assert.ok(old.some((x, i) => Math.abs(x - accepted[i]) >= 8), `the fixture distinguishes grade order: ${old} vs ${accepted}`);
  checks.push({ name: "discriminates-grade-order", old, accepted });

  const bright = [{ color: [1e6, 1e5, 1e4], alpha: 0.5, distance: 4 }];
  await scene.reload(command, fixture(bright)); await look(1e-6);
  await sample("large-light-small-exposure", compose(B, bright, 1e-6));

  const mixed = [[C, 0.25], [[0.1, 1, 0.2], 0.5], [[0.05, 0.4, 4], 0.75], [[1, 0.3, 0.1], 0.5]]
    .map(([color, alpha], i) => ({ color, alpha, distance: 4 + i * 0.5 }));
  await scene.reload(command, fixture(mixed));
  const forward = await sample("mixed", compose(B, mixed, 1));
  await scene.reload(command, fixture([...mixed].reverse()));
  const reversed = await sample("mixed-reversed", compose(B, [...mixed].reverse(), 1));
  closeRGB(forward.actual, reversed.actual, "submission order", 4);
  for (const exposure of [0.25, 1, 4]) {
    const layers = Array.from({ length: 16 }, (_, i) => ({ color: [64, 32, 1], alpha: 0.5, distance: 0.1 + i * 0.001 }));
    await scene.reload(command, fixture(layers)); await look(exposure);
    await sample(`near-overlap-${exposure}`, compose(B, layers, exposure));
  }

  const particleColor = [0.8, 0.2, 0.05, 0.5];
  for (const blend of ["alpha", "additive"]) for (const emission of [1, 8, 64]) {
    const document = fixture();
    document.environment.atmosphere.sun_color = [2, 2, 2];
    document.entities.push({ id: "particle", transform: { translation: [0, 0, 2] }, particles: {
      rate: 1, lifetime: 1, speed: 0, radius: 0, acceleration: [0, 0, 0], direction: [0, 1, 0],
      spread_degrees: 0, size: [1, 1], color: [particleColor, particleColor], emission, blend, stretch: 0,
    } });
    await scene.reload(command, document);
    for (const exposure of [0.25, 1, 4]) {
      await look(exposure, neutral, { sun_color: [2, 2, 2] });
      const light = particleColor.slice(0, 3).map(x => x * emission);
      const reference = blend === "alpha"
        ? compose(B, [{ color: light, alpha: particleColor[3], distance: 4, packed: false }], exposure)
        : compose(B, [], exposure, neutral, [light.map(x => x * particleColor[3])]);
      const { rendering } = await sample(`particle-${blend}-${emission}-${exposure}`, reference);
      assert.equal(rendering.particles.drawn, 1, "exactly one particle must draw");
      if (emission === 8 && exposure === 1) {
        await command("environment", { bloom: { intensity: 1, threshold: 1e6 } });
        const postParticle = await sample(`particle-post-${blend}`, reference);
        assert.equal(postParticle.rendering.particles.drawn, 1);
        await command("environment", { bloom: { intensity: 0 } });
      }
    }
  }

  const single = [{ color: C, alpha: 0.5, distance: 4 }];
  await scene.reload(command, fixture(single));
  const basic = await sample("basic", compose(B, single, 1));
  await command("environment", { bloom: { intensity: 1, threshold: 1e6 } });
  const post = await sample("post", compose(B, single, 1));
  closeRGB(post.actual, basic.actual, "Post/no-effects encoding", 2);
  assert.ok(post.rendering.render_passes.includes("composite"));
  await command("environment", { bloom: { intensity: 0 }, temporal_antialiasing: { enabled: true } });
  for (const exposure of [1, 0.25, 4]) {
    await look(exposure); await ordinary(`ordinary-taa-${exposure}`, compose(B, single, exposure));
  }
  await command("environment", { temporal_antialiasing: { enabled: false } });
  await command("environment", { temporal_antialiasing: { enabled: true } });
  await ordinary("ordinary-taa-reset", compose(B, single, 4));

  // Normal incidence on constant background makes the unchanged glass ray
  // and blur irrelevant. The Karis split-sum fit determines its through share.
  const glassColor = [0.5, 0.75, 1], roughness = unorm(0.5, 255), f0 = 0.04;
  const r = [-roughness + 1, -0.0275 * roughness + 0.0425, -0.572 * roughness + 1.04, 0.022 * roughness - 0.04];
  const a004 = Math.min(r[0] ** 2, 2 ** -9.28) * r[0] + r[1];
  const through = 1 - (f0 * (-1.04 * a004 + r[2]) + 1.04 * a004 + r[3]);
  const localGlass = [2, 0.1, 0.01];
  for (const refraction of [false, true]) {
    const document = fixture([{ color: localGlass, alpha: 1, distance: 4 }]);
    Object.assign(document.materials[1], { color: [...glassColor, 1], transmission: 1, ior: 1.5,
      thickness: 0.5, surface: { roughness: 0.5, metalness: 0 } });
    document.environment.refraction = { strength: refraction ? 1 : 0, blur: 12 };
    await scene.reload(command, document);
    const alpha = refraction ? 1 : Math.max(1 - through, 1e-3);
    const color = refraction ? B.map((x, i) => localGlass[i] + x * unorm(glassColor[i], 65535) * through)
      : localGlass.map(x => x / alpha);
    await sample(`glass-${refraction}`, compose(B, [{ color, alpha, distance: 4, packed: false }], 1));
  }

  // Every fresh probe observes an emissive grey room, with a black fallback
  // sky. A mirror sees raw probe light; recover it from the unclipped grade.
  const recovered = [];
  for (const exposure of [0.25, 1]) {
    const document = fixture([], [0, 0, 0]);
    document.materials[0] = { id: "back", color: [1, 1, 1, 1], surface: { roughness: 0.05, metalness: 1 } };
    document.meshes.push({ id: "room", source: { kind: "cube" } });
    document.materials.push({ id: "room", color: [0, 0, 0, 1], emissive: [0.2, 0.2, 0.2], double_sided: true });
    document.entities.push({ id: "room", mesh: "room", material: "room", transform: { scale: [40, 40, 40] } },
      { id: exposure === 0.25 ? "probe-low" : "probe-unit", reflection_probe: { extent: [12, 12, 12] },
        transform: { translation: [0, 0, 2] } });
    document.environment.atmosphere.exposure = exposure;
    document.environment.atmosphere.specular = 1;
    document.environment.reflection_probes = { enabled: true, resolution: 32, faces_per_frame: 6 };
    // Scene reload instantiates a new runtime probe. An exposure control
    // alone need not invalidate raw captures, so require all six new faces.
    let ready = (await scene.reload(command, document)).result.rendering;
    let faces = ready.reflection_probes.faces, probeFrames = 0;
    // Probe captures follow GPU presentation after its stillness gate;
    // CPU exact renders alone never run that pass.
    while (probeFrames < 96 && (ready.reflection_probes.ready !== 1 || faces < 6)) {
      ready = (await command("capture", { alpha: 0, gpu: true })).result.rendering;
      faces += ready.reflection_probes.faces;
      probeFrames++;
    }
    assert.equal(ready.reflection_probes.ready, 1, "a fresh local probe must finish");
    assert.ok(faces >= 6, `each exposure captures a fresh cube: ${faces} faces`);
    await captureReady();
    const actual = await center(`probe-${exposure}`);
    measurements.push({
      name: `probe-${exposure}: mirror red`, value: actual[0], unit: "byte-code",
      note: "The inverse-grade reference is interpreted in the unclipped interval 20 < red < 240; values outside it remain visible and do not fail the check.",
    });
    closeRGB(actual, [actual[0], actual[0], actual[0]], `grey probe-${exposure}`, 1);
    recovered.push(inverseGrey(actual[0]) / exposure);
    await limits(0);
    checks.push({ name: `probe-${exposure}`, actual, recoveredRaw: recovered.at(-1), faces, probeFrames });
  }
  measurements.push({
    name: "probe exposure: recovered raw-light relative difference",
    value: Math.abs(recovered[0] / recovered[1] - 1), unit: "fraction",
    note: `Accepted-design reference tolerance 0.05; recovered raw light ${recovered}. This screenshot-derived ratio does not decide the check.`,
  });

  const skyOnly = fixture();
  skyOnly.entities = [];
  Object.assign(skyOnly.environment.atmosphere, { zenith: B, horizon: B, ground: B });
  await scene.reload(command, skyOnly);
  await sample("sky-basic", compose(B, [], 1), 2);
  await command("environment", { bloom: { intensity: 1, threshold: 1e6 } });
  await sample("sky-post", compose(B, [], 1), 2);

  await scene.reload(command, fixture([], [3841024, 0.25, 0.1]));
  await sample("guard-grade", compose([3841024, 0.25, 0.1], [], 1));
  assert.equal(lastDiagnostic.limit_mask, 2);
  const guardLayers = Array.from({ length: 21 }, (_, i) => ({ color: [32, 2, 0.125], alpha: 1, distance: 0.1 + i * 0.001 }));
  await scene.reload(command, fixture(guardLayers));
  await sample("guard-weight-sum", compose(B, guardLayers, 1));
  assert.equal(lastDiagnostic.limit_mask, 1);
  await scene.reload(command, fixture());
  await sample("guard-clears-next-frames", compose(B, [], 1), 2);

  const lifecycleBefore = await page.evaluate(() => window.__radianceLifecycle);
  await resize(503, 379);
  await sample("resized", compose(B, [], 1), 2);
  const lifecycle = await page.evaluate(() => window.__radianceLifecycle);
  assert.equal(lifecycle.live, 1, "only the current main-view radiance texture remains live");
  assert.ok(lifecycle.created > lifecycleBefore.created && lifecycle.destroyed > lifecycleBefore.destroyed,
    `resize retires the old target: ${JSON.stringify({ lifecycleBefore, lifecycle })}`);
  checks.push({ name: "resource-retirement", lifecycle });
  assert.deepEqual(errors, []);
  await writeFile(`${output}/results.json`, JSON.stringify({ ok: true, checks, imageComparisons, measurements }, null, 2));
  console.log(JSON.stringify({ ok: true, output, cases: checks.length, measurements }));
} catch (error) {
  await writeFile(`${output}/results.json`, JSON.stringify({ ok: false, error: String(error), checks, imageComparisons, measurements, gpuErrors: errors }, null, 2));
  console.log(JSON.stringify({ ok: false, output, error: String(error), cases: checks.length, measurements }));
  if (error instanceof assert.AssertionError) { console.error(error); process.exitCode = 3; }
  else throw error;
} finally {
  await browser.close();
}
