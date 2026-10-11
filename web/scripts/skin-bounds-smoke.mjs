/** Joint-influence bounds contain independently skinned vertices on the GPU,
 * and skinned views/shadows match the CPU draw-all reference across poses,
 * sparse influences, camera boundaries, rigid use and scene replacement. */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, sceneUrl, differentPixels, launch, openPaused, routedScene } from "./harness.mjs";

const output = await artifacts("skin-bounds-smoke");
const { browser, page, errors } = await launch({ width: 960, height: 720 });
const command = commander(page);
const f = Math.fround, maximum = f(3.4028234663852886e38);
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const camera = { position: [0, 1, 8], target: [0, 1, 0], fov_degrees: 60, near: 0.1, far: 100 };
const rows = [[0, [1, 0, 0, 0]], [0.5, [0.25, 0.25, 0.25, 0.25]],
  [1, [0.5, 0.5, 0, 0]], [1.5, [1, 1e-9, 0, 0]], [2, [0, 1, 0, 0]]];
const ribbon = rows.flatMap(([y, weights]) => [-0.3, 0.3].map(x => ({
  position: [x, y, 0], normal: [0, 0, 1], uv: [x + 0.3, y / 2],
  joints: [0, 2, 4, 6], weights,
})));
const indices = rows.slice(1).flatMap((_, row) => {
  const a = 2 * row;
  return [a, a + 1, a + 3, a, a + 3, a + 2];
});
// Each triangle moves to x=12. The old whole-mesh box transformed by both
// joints straddles the camera, while the two influence boxes stay offscreen.
const separated = [-100, 100].flatMap((x, joint) => [[-0.2, 0], [0.2, 0], [0, 1]].map(([dx, y]) => ({
  position: [x + dx, y, 0], normal: [0, 0, 1], uv: [0, 0],
  joints: [joint, 0, 0, 0], weights: [1, 0, 0, 0],
})));
const fixture = {
  version: 9, id: "skin-bounds-fixture", revision: 1, camera,
  environment: { shadows: { enabled: false }, occlusion_culling: { enabled: false }, subpixel_culling: { enabled: false } },
  materials: [{ id: "plain", color: [0.8, 0.4, 0.2, 1], double_sided: true }],
  meshes: [
    { id: "ribbon", source: { kind: "inline", vertices: ribbon, indices } },
    { id: "separated", source: { kind: "inline", vertices: separated, indices: [0, 1, 2, 3, 4, 5] } },
    { id: "floor", source: { kind: "cube" } },
  ],
  skins: [
    { id: "rig", joints: Array.from({ length: 7 }, (_, i) => "bone-" + i), inverse_binds: Array(7).fill(identity) },
    { id: "separated-rig", joints: ["separated-left", "separated-right"], inverse_binds: [identity, identity] },
  ],
  animations: [{ id: "bend", targets: ["bone-2"], channels: [
    { target: 0, property: "rotation", times: [0, 1], values: [0, 0, 0, 1, 0, 0, 0.7071068, 0.7071068] },
  ] }],
  entities: [
    { id: "root", animation: { clip: "bend", speed: 0, loop: false } },
    ...Array.from({ length: 7 }, (_, i) => ({
      id: "bone-" + i, parent: "root",
      transform: { translation: [i === 4 ? 0.2 : 0, 0, 0], scale: i === 6 ? [1.3, 0.7, 1.3] : [1, 1, 1] },
    })),
    { id: "limb", mesh: "ribbon", material: "plain", skin: "rig" },
    { id: "rigid", mesh: "ribbon", material: "plain", transform: { translation: [2, 0, 0] } },
    { id: "separated-left", transform: { translation: [112, 0, 0] } },
    { id: "separated-right", transform: { translation: [-88, 0, 0] } },
    { id: "offscreen", mesh: "separated", material: "plain", skin: "separated-rig" },
  ],
};
const scene = routedScene(page, fixture);

/** Read the actual production WGSL function, not a copy of the bound math. */
function declaration(source, prefix) {
  const start = source.indexOf(prefix);
  assert.ok(start >= 0, "production declaration exists: " + prefix);
  const open = source.indexOf("{", start);
  let depth = 1, end = open + 1;
  for (; depth && end < source.length; end++) {
    if (source[end] === "{") depth++;
    if (source[end] === "}") depth--;
  }
  assert.equal(depth, 0);
  return source.slice(start, end);
}

/** Four packed influences, evaluated independently as f32 row blending. */
function skinPoint(point, matrices, packed) {
  const weights = packed.map(value => f(value / 65535));
  const sum = weights.reduce((sum, value) => f(sum + value), 0);
  const w = weights.map(value => f(value / sum));
  const result = [];
  for (let axis = 0; axis < 3; axis++) {
    const row = Array.from({ length: 4 }, (_, column) => matrices.reduce((sum, matrix, joint) =>
      f(sum + f(matrix[4 * axis + column] * w[joint])), 0));
    result.push(row.reduce((sum, value, column) => f(sum + f(value * (point[column] ?? 1))), 0));
  }
  return result;
}

async function arithmetic() {
  const source = await readFile(new URL("../../crates/engine_render/src/presentation.wgsl", import.meta.url), "utf8");
  const shader = [
    source.match(/const F32_MAX: f32 = [^;]+;/)[0],
    declaration(source, "struct Rows {"),
    declaration(source, "struct SkinEnvelope {"),
    declaration(source, "fn transform_skin_envelope("),
    "struct Case { rows: array<Rows, 4>, envelopes: array<SkinEnvelope, 4>, }",
    "@group(0) @binding(0) var<storage, read> cases: array<Case>;",
    "@group(0) @binding(1) var<storage, read_write> results: array<SkinEnvelope>;",
    "@compute @workgroup_size(64) fn check(@builtin(global_invocation_id) id: vec3<u32>) {",
    "if id.x >= arrayLength(&cases) { return; }",
    "var low = vec3<f32>(F32_MAX); var high = vec3<f32>(-F32_MAX);",
    "for (var joint = 0u; joint < 4u; joint++) {",
    "let value = transform_skin_envelope(cases[id.x].rows[joint], cases[id.x].envelopes[joint].low.xyz, cases[id.x].envelopes[joint].high.xyz);",
    "low = min(low, value.low.xyz); high = max(high, value.high.xyz); }",
    "results[id.x] = SkinEnvelope(vec4<f32>(low, 0), vec4<f32>(high, 0)); }",
  ].join("\n");
  let seed = 173;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const cases = [];
  for (let pose = 0; pose < 64; pose++) {
    const scale = 2 ** (pose % 31 - 15);
    const offset = 2 ** (pose % 21);
    const points = Array.from({ length: 12 }, (_, i) => {
      const point = Array.from({ length: 3 }, () => f(offset + (random() * 4 - 2) * scale));
      const packed = i % 3 === 0 ? [65535, 0, 0, 0] : i % 3 === 1 ? [32768, 32767, 0, 0]
        : [1, 16384, 24575, 24575];
      return { point, packed };
    });
    const matrices = Array.from({ length: 4 }, () => Array.from({ length: 3 }, () => {
      const row = Array.from({ length: 3 }, () => f((random() * 4 - 2) / scale));
      // Large opposite translations exercise cancellation near the origin.
      return [...row, f(-row.reduce((sum, value) => sum + value * offset, 0))];
    }).flat());
    const envelopes = matrices.map((_, joint) => {
      const used = points.filter(value => value.packed[joint] > 0);
      return [...[0, 1, 2].map(axis => Math.min(maximum, ...used.map(value => value.point[axis]))), 0,
        ...[0, 1, 2].map(axis => Math.max(-maximum, ...used.map(value => value.point[axis]))), 0];
    });
    for (const value of points) cases.push({ ...value, matrices, envelopes });
  }
  // Subnormal row/position inputs, huge cancellation and finite overflow.
  for (const scale of [f(1e-40), f(1e30), maximum]) {
    const matrices = Array.from({ length: 4 }, () => [scale, 0, 0, -scale, 0, scale, 0, 0, 0, 0, scale, 0]);
    const point = [1, f(1e-40), 0], packed = [65535, 0, 0, 0];
    const envelopes = matrices.map((_, joint) => joint ? [maximum, maximum, maximum, 0, -maximum, -maximum, -maximum, 0]
      : [...point, 0, ...point, 0]);
    cases.push({ point, packed, matrices, envelopes });
  }
  for (const invalid of [NaN, Infinity, -Infinity]) {
    const matrices = Array.from({ length: 4 }, () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
    matrices[0][0] = invalid;
    const envelopes = matrices.map(() => [-1, -1, -1, 0, 1, 1, 1, 0]);
    cases.push({ point: [0, 0, 0], packed: [65535, 0, 0, 0], matrices, envelopes, unbounded: true });
  }
  const flat = cases.flatMap(value => [...value.matrices.flat(), ...value.envelopes.flat()]);
  const result = await page.evaluate(async ({ shader, flat, count }) => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter for skin-bound arithmetic");
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    device.pushErrorScope("validation");
    const input = device.createBuffer({ size: flat.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: count * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: count * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(input, 0, new Float32Array(flat));
    const module = device.createShaderModule({ code: shader });
    const compilation = await module.getCompilationInfo();
    const diagnostics = compilation.messages.filter(value => value.type === "error").map(value => value.message);
    if (diagnostics.length) throw new Error(diagnostics.join("\n"));
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "check" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
    ] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, count * 32);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const values = Array.from(new Float32Array(readback.getMappedRange()));
    readback.unmap();
    const validation = await device.popErrorScope();
    if (validation) errors.push(validation.message);
    input.destroy(); output.destroy(); readback.destroy(); device.destroy();
    return { values, errors };
  }, { shader, flat, count: cases.length });
  assert.deepEqual(result.errors, []);
  for (const [index, value] of cases.entries()) {
    const position = skinPoint(value.point, value.matrices, value.packed);
    for (let axis = 0; axis < 3; axis++) {
      const low = result.values[index * 8 + axis], high = result.values[index * 8 + 4 + axis];
      assert.ok(Number.isFinite(low) && Number.isFinite(high) && low <= high, "finite conservative bound " + index);
      if (value.unbounded) {
        assert.equal(low, -maximum, "nonfinite palette keeps the slot");
        assert.equal(high, maximum, "nonfinite palette keeps every shadow map");
      }
      if (Number.isFinite(position[axis])) assert.ok(position[axis] >= low && position[axis] <= high,
        "fully skinned vertex " + index + " axis " + axis + ": " + position[axis] + " outside [" + low + ", " + high + "]");
    }
  }
  return cases.length;
}

async function capture(name, gpu, alpha = 0) {
  if (gpu) {
    await command("present", { alpha, gpu: true });
    await delay(100);
    // Replies report an earlier frame's asynchronous count. A completed
    // capture lets that readback finish before the identical measured one.
    await command("capture", { alpha, gpu: true });
    await page.locator("canvas").screenshot();
  }
  const reply = await command("capture", { alpha, gpu });
  const png = await page.locator("canvas").screenshot({ path: output + "/" + name + ".png" });
  return { rendering: reply.result.rendering, png: png.toString("base64") };
}

async function compare(name, alpha = 0) {
  const cpu = await capture(name + "-reference", false, alpha);
  const gpu = await capture(name + "-bounded", true, alpha);
  const pixels = await differentPixels(page, [cpu.png, gpu.png], { tolerance: 8 });
  assert.ok(pixels.changed <= pixels.total * 0.001, name + ": bounded image differs at " + pixels.changed + " pixels");
  assert.equal(gpu.rendering.invalid_transforms, 0);
  assert.equal(gpu.rendering.missing_assets, 0);
  assert.equal(gpu.rendering.dropped_instances, 0);
  return { name, changed_pixels: pixels.changed, cpu: cpu.rendering.visible_instances, gpu: gpu.rendering.visible_instances };
}

try {
  await scene.route();
  await openPaused(page, sceneUrl());
  const arithmeticCases = await arithmetic();
  const cases = [await compare("sparse-rigid-and-offscreen")];
  assert.ok(cases[0].gpu < cases[0].cpu, "joint-specific bounds reject the separated offscreen actor the whole-mesh union keeps: " + JSON.stringify(cases[0]));
  for (const time of [0.37, 1]) {
    await command("animation", { entity: "root", time, speed: 0 });
    await command("step", { ticks: 1 });
    for (const alpha of [0, 0.37, 0.81]) cases.push(await compare("bend-" + time + "-" + alpha, alpha));
  }
  await command("environment", { shadows: { enabled: true }, subpixel_culling: { enabled: true } });
  cases.push(await compare("shadow-and-subpixel"));
  for (const [name, position, target, near] of [
    ["near-plane", [0, 1, 0.15], [0, 1, 0], 0.1],
    ["side-boundary", [3, 1, 4], [3, 1, 0], 0.1],
    ["turned-away", [0, 1, 8], [0, 1, 16], 0.1],
  ]) {
    await command("camera", { position, target, vertical_fov_radians: Math.PI / 3, near, far: 100 });
    cases.push(await compare(name));
  }
  const changed = structuredClone(fixture);
  changed.entities.find(value => value.id === "root").transform = { rotation_degrees: [12, 17, 23], scale: [1.7, 0.6, 1.2] };
  changed.entities.find(value => value.id === "bone-2").transform.rotation_degrees = [23, 37, 11];
  changed.entities.push({ id: "receiver", mesh: "floor", material: "plain", transform: { translation: [0, -0.5, 0], scale: [20, 0.5, 20] } });
  changed.entities.find(value => value.id === "separated-left").transform.translation = [112, 10, 0];
  changed.entities.find(value => value.id === "separated-right").transform.translation = [-88, 10, 0];
  changed.environment.shadows.enabled = true;
  changed.environment.atmosphere = { sun_direction: [1, 1, 0.7] };
  await scene.reload(command, changed);
  cases.push(await compare("replacement-shear-and-shadows"));
  assert.ok(cases.at(-1).gpu < cases.at(-1).cpu, "shadow caster is culled from the camera: " + JSON.stringify(cases.at(-1)));
  const caster = await capture("offscreen-caster-shadow", true);
  const without = structuredClone(changed);
  without.entities = without.entities.filter(value => value.id !== "offscreen");
  await scene.reload(command, without);
  const absent = await capture("offscreen-caster-removed", true);
  const shadow = await differentPixels(page, [caster.png, absent.png], { tolerance: 8 });
  assert.ok(shadow.changed > 0, "the offscreen skinned actor casts a visible shadow");
  changed.entities.find(value => value.id === "offscreen").skin = "rig";
  await scene.reload(command, changed);
  cases.push(await compare("rebound-skin"));
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, artifact: output, arithmetic_cases: arithmeticCases, offscreen_shadow_pixels: shadow.changed, cases,
    checks: ["gpu_bound_contains_skinned_vertices", "sparse_and_quantized_weights", "rigid_fallback", "tighter_offscreen_cull",
      "interpolated_pose_equivalence", "view_and_shadow_equivalence", "scene_replacement_and_skin_rebinding", "clean_browser_diagnostics"] }));
} catch (error) {
  if (error?.code !== "ERR_ASSERTION") throw error;
  console.error(error);
  process.exitCode = 3;
} finally {
  await browser.close();
}
