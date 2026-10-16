/** Actual production fragment guard and index-source selection on the GPU. */
// @ts-check
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { workspacePath } from "./workspace.mjs";
import { bits } from "./depth-occupancy-oracle.mjs";

export async function checkShadingGuards(page, routes) {
  const names = ["light_types.wgsl", "light_depth_interval.wgsl", "light_depth_layout.wgsl", "light_depth_atomic.wgsl", "meshes/light_depth.wgsl"];
  const sources = await Promise.all(names.map(name => readFile(workspacePath(`crates/engine_render/src/${name}`), "utf8")));
  const source = sources.join("\n") + `
@group(3) @binding(0) var<uniform> clusters: Clusters;
@group(3) @binding(2) var<storage, read> cluster_ranges: array<vec2<u32>>;
@group(3) @binding(3) var<storage, read> cluster_lights: array<u32>;
@group(0) @binding(0) var<storage, read> probe_point: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> probe_output: array<u32>;
@compute @workgroup_size(1)
fn probe_guard() {
    let point = bitcast<vec3<f32>>(probe_point.xyz);
    var selected = false;
    var fragment = 0u;
    let valid = depth_position_valid(point) && depth_plane_valid(clusters.depth);
    if DEPTH_OCCUPANCY {
        selected = depth_shading_guard(point, 0u);
        if valid && arrayLength(&cluster_ranges) >= (depth_indices() + 1u) / 2u {
            let interval = depth_point_interval(point, clusters.depth);
            fragment = depth_interval_bits(depth_shading_bin(interval.x, 0u), depth_shading_bin(interval.y, 0u));
        }
    }
    probe_output[0] = select(0u, 1u, selected);
    probe_output[1] = fragment;
    probe_output[2] = select(0u, 1u, valid);
    probe_output[3] = depth_selected_light(0u, selected);
}`;
  const covered = { label: "covered", point: [0.2, 0, -10.19].map(bits), mask: 8, tag: 7, phase: 1, override: true, filtered: true };
  const cases = [covered,
    { ...covered, label: "uncollected-but-covered", point: [0.7, 0.2, -10.19].map(bits) },
    { ...covered, label: "moved", point: [0.2, 0, -11].map(bits), filtered: false },
    { ...covered, label: "partial-interval", point: [0.2, 0, -10.1875].map(bits), mask: 4, filtered: false },
    { ...covered, label: "complete-interval", point: [0.2, 0, -10.1875].map(bits), mask: 12 },
    { ...covered, label: "empty", mask: 0, filtered: false },
    { ...covered, label: "stale-view-generation", tag: 8, filtered: false },
    { ...covered, label: "stale-phase", phase: 2, filtered: false },
    { ...covered, label: "full-tag", tag: 0, filtered: false },
    { ...covered, label: "subnormal-position", point: [1, 0, bits(-10.19)], filtered: false },
    { ...covered, label: "nan-position", point: [0x7fc00000, 0, bits(-10.19)], filtered: false },
    { ...covered, label: "outside-domain", point: [0x49800001, 0, bits(-10.19)], filtered: false },
    { ...covered, label: "unsupported-view", invalidPlane: true, filtered: false },
    { ...covered, label: "full-allocation", short: true, filtered: false },
    ...routes.map(route => ({ ...covered, label: `full-route:${route.consumer}`, tag: route.generation, override: route.occupancy_override, filtered: false })),
  ];
  const results = await page.evaluate(async ({ source, cases }) => {
    const { GPUBufferUsage: U, GPUMapMode } = /** @type {any} */ (globalThis);
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No guard probe adapter");
    const device = await adapter.requestDevice();
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code: source });
    const compute = 4;
    const requestLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: compute, buffer: { type: "read-only-storage" } }, { binding: 1, visibility: compute, buffer: { type: "storage" } }] });
    const emptyLayout = device.createBindGroupLayout({ entries: [] });
    const lightLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: compute, buffer: { type: "uniform" } }, ...[2, 3].map(binding => ({ binding, visibility: compute, buffer: { type: /** @type {const} */ ("read-only-storage") } }))] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [requestLayout, emptyLayout, emptyLayout, lightLayout] });
    const pipelines = await Promise.all([false, true].map(enabled => device.createComputePipelineAsync({ layout, compute: { module, entryPoint: "probe_guard", constants: { DEPTH_OCCUPANCY: Number(enabled) } } })));
    const outputs = [];
    for (const item of cases) {
      const floats = new Float32Array(36), uniform = new Uint32Array(floats.buffer);
      floats.set([0, 0, item.invalidPlane ? 2.000001 : -1, 0], 16);
      uniform.set([16, 9, 24, 0xffffffff], 28); uniform.set([1, item.tag, 1, item.phase], 32);
      const arena = new Uint32Array(item.short ? 3456 * 2 : 18034);
      if (!item.short) {
        arena[4 * 3456] = item.mask;
        new Float32Array(arena.buffer).set(Array.from({ length: 31 }, (_, i) => 10 + (i + 1) / 16), 5 * 3456);
        arena[18024] = 7; arena[18025] = 0; arena[18032] = 202;
      }
      const make = (data, usage) => { const buffer = device.createBuffer({ size: data.byteLength, usage: usage | U.COPY_DST }); device.queue.writeBuffer(buffer, 0, data); return buffer; };
      const point = make(new Uint32Array([...item.point, 0]), U.STORAGE);
      const result = device.createBuffer({ size: 16, usage: U.STORAGE | U.COPY_SRC });
      const readback = device.createBuffer({ size: 16, usage: U.COPY_DST | U.MAP_READ });
      const buffers = [make(uniform, U.UNIFORM), make(arena, U.STORAGE), make(new Uint32Array([101]), U.STORAGE)];
      const pipeline = pipelines[Number(item.override)];
      const request = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: point } }, { binding: 1, resource: { buffer: result } }] });
      const entries = buffers.map((buffer, i) => ({ binding: [0, 2, 3][i], resource: { buffer } }));
      const lights = device.createBindGroup({ layout: pipeline.getBindGroupLayout(3), entries });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, request);
      for (const group of [1, 2]) pass.setBindGroup(group, device.createBindGroup({ layout: pipeline.getBindGroupLayout(group), entries: [] }));
      pass.setBindGroup(3, lights); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(result, 0, readback, 0, 16); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ); outputs.push(Array.from(new Uint32Array(readback.getMappedRange()))); readback.unmap();
      for (const buffer of [point, result, readback, ...buffers]) buffer.destroy();
    }
    const error = await device.popErrorScope(); if (error) throw new Error(error.message);
    device.destroy(); return outputs;
  }, { source, cases });
  cases.forEach((item, index) => {
    assert.equal(Boolean(results[index][0]), item.filtered, item.label);
    assert.equal(results[index][3], item.filtered ? 202 : 101, `${item.label}: actual production index source`);
  });
  const partial = results[cases.findIndex(item => item.label === "partial-interval")][1];
  assert.ok((partial & 4) !== 0 && (partial & 4) !== partial, "the partial case overlaps but does not contain F");
  return { source_sha256: createHash("sha256").update(source).digest("hex"), cases, results };
}
