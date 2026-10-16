/** Exact binary32 rational oracle for the production depth-occupancy WGSL. */
// @ts-check
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { workspacePath } from "./workspace.mjs";

const bytes = new ArrayBuffer(4), floats = new Float32Array(bytes), words = new Uint32Array(bytes);
export function bits(value) { floats[0] = value; return words[0]; }
export function value(word) { words[0] = word; return floats[0]; }

// Every finite binary32 is an integer multiple of 2^-149. Products and
// affine sums therefore share the exact denominator 2^298, even when the
// operands' exponents are too far apart for a binary64 sum to be exact.
const scale = 1n << 149n;
export function exact(word) {
  const magnitude = word & 0x7fffffff, exponent = magnitude >>> 23;
  assert.notEqual(exponent, 255, "the rational oracle only accepts finite values");
  const significand = BigInt(magnitude & 0x7fffff) + (exponent ? 1n << 23n : 0n);
  const integer = significand << BigInt(exponent ? exponent - 1 : 0);
  return word >>> 31 ? -integer : integer;
}

function projection(point, plane) {
  return point.slice(0, 3).reduce((sum, word, i) => sum + exact(word) * exact(plane[i]), exact(plane[3]) * scale);
}

function box(light, plane) {
  let low = exact(plane[3]) * scale, high = low;
  const diameter = 2n * exact(light[3]);
  for (let axis = 0; axis < 3; axis++) {
    const center = exact(light[axis]), coefficient = exact(plane[axis]);
    const ends = [center - diameter, center + diameter];
    low += coefficient * ends[coefficient < 0n ? 1 : 0];
    high += coefficient * ends[coefficient < 0n ? 0 : 1];
  }
  return [low, high];
}

function domain(word, bound) {
  const magnitude = word & 0x7fffffff;
  return magnitude <= bound && (magnitude === 0 || magnitude >= 0x800000);
}

const positionValid = point => point.slice(0, 3).every(word => domain(word, 0x49800000));
const planeValid = plane => plane.slice(0, 3).every(word => domain(word, 0x40000000)) && domain(plane[3], 0x4a000000);
const lightValid = light => positionValid(light) && light[3] >= 0x35800000 && light[3] <= 0x49800000;

function enclosed(low, high, exactLow, exactHigh, label) {
  assert.ok(exact(low) * scale <= exactLow && exact(high) * scale >= exactHigh,
    `${label}: [${value(low)}, ${value(high)}] misses an exact endpoint`);
  for (const endpoint of [low, high]) assert.ok(domain(endpoint, 0x4c800000), `${label}: endpoints stay normal or zero and below 2^26`);
}

function bin(numerator, thresholds) {
  return thresholds.filter(threshold => exact(threshold) * scale <= numerator).length;
}

function cases() {
  const rows = [];
  const add = (point, plane = [0, 0, 1, 0], light = [0, 0, 0, 0.01], slice = 0) => {
    rows.push([...point, 0, ...plane, ...light, slice, 0, 0, 0].map((x, i) => i < 12 ? bits(x) : x));
  };
  for (let i = -16; i <= 16; i++) {
    const word = bits(i);
    for (const nearby of [i, value(word + 1), value(word ? word - 1 : 1)]) add([0, 0, nearby]);
  }
  for (const sign of [-1, 1]) {
    for (const x of [0, 2 ** -126, value(0x800001), 2 ** -125, 2 ** -20, 1, 2, 2 ** 20]) {
      for (const coefficient of [0, 2 ** -126, 0.5, 1, 2]) {
        add([sign * x, x, -x], [coefficient, -coefficient, 0, 0]);
        add([sign * x, 0, 0], [coefficient, 0, 0, -sign * Math.fround(x * coefficient)]);
      }
    }
  }
  for (const word of [1, 0x7fffff, 0x49800001, 0x7f800000, 0x7fc00000, 0xff800000]) {
    add([value(word), 0, 0]);
  }
  for (const word of [1, 0x7fffff, 0x40000001, 0x7f800000, 0x7fc00000]) add([1, 2, 3], [value(word), 1, 1, 0]);
  for (const word of [0, 1, 0x357fffff, 0x35800000, 0x49800000, 0x49800001, 0x7f800000, 0x7fc00000]) {
    add([0, 0, 1], [0.5, -1, 2, 1], [0, 0, 1, value(word)]);
  }
  for (const radius of [2 ** -20, 0.01, 1, 512, 2 ** 18]) {
    for (const multiple of [0, 0.9999999, 1, 1.0000002, 1.9999999, 2, 2.0000002]) {
      add([Math.fround(radius * multiple), 0, 0], [1, -0.5, 0.25, 0], [0, 0, 0, radius]);
    }
  }
  add([0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 16]);
  for (const z of [0, value(bits(1) - 1), 1, value(bits(1) + 1), 2]) add([0, 0, z], [0, 0, 1, 0], [0, 0, 1, 0.01], 1);
  let seed = 523;
  const random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return seed >>> 0; };
  const normal = maxExponent => value(((random() & 0x80000000) | ((1 + random() % maxExponent) << 23) | (random() & 0x7fffff)) >>> 0);
  for (let i = 0; i < 512; i++) {
    const point = [normal(146), normal(146), normal(146)];
    add(point, [normal(127), normal(127), normal(127), normal(147)],
      [normal(146), normal(146), normal(146), Math.abs(normal(146))]);
  }
  return rows;
}

const driver = `
@group(0) @binding(0) var<uniform> clusters: Clusters;
@group(0) @binding(1) var<storage, read_write> light_depth_arena: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read> cases: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64)
fn oracle(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i * 4u >= arrayLength(&cases) { return; }
    let point = bitcast<vec4<f32>>(cases[i * 4u]);
    let plane = bitcast<vec4<f32>>(cases[i * 4u + 1u]);
    let light = bitcast<vec4<f32>>(cases[i * 4u + 2u]);
    let slice = cases[i * 4u + 3u].x;
    let base = i * 16u;
    let valid = depth_position_valid(point.xyz) && depth_plane_valid(plane);
    output[base] = select(0u, 1u, valid);
    if valid {
        let interval = depth_point_interval(point.xyz, plane);
        output[base + 1u] = bitcast<u32>(interval.x);
        output[base + 2u] = bitcast<u32>(interval.y);
        output[base + 3u] = depth_atomic_mask(interval, slice);
        let added = depth_outward(point.x + plane.x);
        output[base + 8u] = bitcast<u32>(added.x);
        output[base + 9u] = bitcast<u32>(added.y);
        let multiplied = depth_outward(point.x * plane.x);
        output[base + 10u] = bitcast<u32>(multiplied.x);
        output[base + 11u] = bitcast<u32>(multiplied.y);
    }
    let light_valid = depth_light_valid(light.xyz, light.w) && depth_plane_valid(plane);
    output[base + 4u] = select(0u, 1u, light_valid);
    if light_valid {
        let interval = depth_light_interval(light.xyz, light.w, plane);
        output[base + 5u] = bitcast<u32>(interval.x);
        output[base + 6u] = bitcast<u32>(interval.y);
        output[base + 7u] = depth_atomic_mask(interval, slice);
        if valid {
            let to_light = light.xyz - point.xyz;
            let d2 = dot(to_light, to_light);
            let r2 = light.w * light.w;
            output[base + 12u] = select(0u, 1u, d2 < r2);
        }
    }
    output[base + 13u] = select(0u, 1u, depth_guard(3u, 1u, 7u, 7u));
    output[base + 14u] = select(0u, 1u, depth_guard(1u, 3u, 7u, 7u));
    output[base + 15u] = select(0u, 1u, depth_guard(3u, 1u, 7u, 8u));
}`;

export async function checkPrimitives(page) {
  const files = ["light_types.wgsl", "light_depth_interval.wgsl", "light_depth_layout.wgsl", "light_depth_atomic.wgsl"];
  const sources = await Promise.all(files.map(file => readFile(workspacePath(`crates/engine_render/src/${file}`), "utf8")));
  const source = sources.join("\n") + driver;
  const inputs = cases();
  const tables = [Array.from({ length: 31 }, (_, i) => bits(i - 15)), Array(31).fill(bits(1))];
  const result = await page.evaluate(async ({ source, inputs, tables }) => {
    const { GPUBufferUsage, GPUMapMode } = /** @type {any} */ (globalThis);
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No adapter for the production primitive probe");
    const device = await adapter.requestDevice();
    device.pushErrorScope("validation");
    const buffer = (data, usage) => {
      const created = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(created, 0, data);
      return created;
    };
    const uniform = new Uint32Array(36);
    uniform.set([16, 9, 24, 0xffffffff], 28);
    const arena = new Uint32Array(5 * 3456 + 31 * 24 + 8);
    tables.forEach((table, slice) => arena.set(table, 5 * 3456 + slice * 31));
    const buffers = [buffer(uniform, GPUBufferUsage.UNIFORM), buffer(arena, GPUBufferUsage.STORAGE),
      buffer(new Uint32Array(inputs.flat()), GPUBufferUsage.STORAGE),
      device.createBuffer({ size: inputs.length * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })];
    const readback = device.createBuffer({ size: inputs.length * 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code: source }), entryPoint: "oracle" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(inputs.length / 64)); pass.end();
    encoder.copyBufferToBuffer(buffers[3], 0, readback, 0, inputs.length * 64);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const words = Array.from(new Uint32Array(readback.getMappedRange()));
    readback.unmap();
    const error = await device.popErrorScope();
    if (error) throw new Error(error.message);
    const gpu = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description };
    for (const resource of [...buffers, readback]) resource.destroy();
    device.destroy();
    return { words, gpu };
  }, { source, inputs, tables });
  let validPoints = 0, validLights = 0, contributors = 0, allBits = 0;
  inputs.forEach((row, i) => {
    const point = row.slice(0, 4), plane = row.slice(4, 8), light = row.slice(8, 12), table = tables[row[12]];
    const got = result.words.slice(i * 16, (i + 1) * 16);
    const valid = positionValid(point) && planeValid(plane), validLight = lightValid(light) && planeValid(plane);
    assert.equal(Boolean(got[0]), valid, `point domain ${i}`);
    assert.equal(Boolean(got[4]), validLight, `light domain ${i}`);
    if (valid) {
      validPoints++;
      const projected = projection(point, plane);
      enclosed(got[1], got[2], projected, projected, `point ${i}`);
      const member = (1 << bin(projected, table)) >>> 0;
      assert.equal((got[3] & member) >>> 0, member, `point mask ${i}`);
      allBits = (allBits | got[3]) >>> 0;
      const added = (exact(point[0]) + exact(plane[0])) * scale;
      enclosed(got[8], got[9], added, added, `add ${i}`);
      const multiplied = exact(point[0]) * exact(plane[0]);
      enclosed(got[10], got[11], multiplied, multiplied, `multiply ${i}`);
    }
    if (validLight) {
      validLights++;
      const [low, high] = box(light, plane);
      enclosed(got[5], got[6], low, high, `light box ${i}`);
      for (let b = bin(low, table); b <= bin(high, table); b++) assert.notEqual((got[7] >>> b) & 1, 0, `light mask ${i}, bit ${b}`);
      if (valid && got[12]) {
        contributors++;
        for (let axis = 0; axis < 3; axis++) {
          const distance = exact(light[axis]) - exact(point[axis]);
          assert.ok((distance < 0n ? -distance : distance) <= 2n * exact(light[3]), `range implication ${i}`);
        }
        assert.notEqual((got[7] & got[3]) >>> 0, 0, `contributor mask ${i}`);
      }
    }
    assert.deepEqual(got.slice(13), [1, 0, 0], `guard requires complete occupancy and the same tag: ${i}`);
  });
  assert.equal(allBits, 0xffffffff, "all 32 occupancy bits exercised");
  const denominator = 1n << 23n;
  const proofLeft = 4n * (denominator - 1n) ** 5n;
  const proofRight = (denominator + 1n) * denominator ** 4n;
  assert.ok(proofLeft > proofRight, "exact factor-two range inequality");
  return { cases: inputs.length, validPoints, validLights, contributors,
    source_sha256: createHash("sha256").update(source).digest("hex"),
    proof: { left: String(proofLeft), right: String(proofRight) }, inputs, tables, ...result };
}
