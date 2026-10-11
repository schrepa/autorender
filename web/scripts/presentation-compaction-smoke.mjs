/** Exact production compaction kernels against a stable CPU oracle, plus
 * live storage growth, occlusion and reflection-probe regeneration. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, differentPixels, launch, openPaused, routedScene, sceneUrl } from "./harness.mjs";

const root = fileURLToPath(new URL("../../", import.meta.url));
const sentinel = 0xdeadbeef;
const flags = { VISIBLE: 1, MOVING: 2, VALID: 4, PHASE1: 8, PHASE2: 16, OCCLUDED: 32, SUBPIXEL: 64, PROBE: 128 };
const moved = 0x80000000;
const grow = n => 2 ** Math.ceil(Math.log2(Math.max(1024, n)));
const batchGrow = n => 2 ** Math.ceil(Math.log2(Math.max(64, n)));
const filled = (n, value = sentinel) => new Array(n).fill(value);

function declarations(source, name) {
  const body = source.match(new RegExp(`struct ${name} \\{([\\s\\S]*?)\\n\\}`))?.[1];
  assert.ok(body, `${name} declaration exists`);
  return body.replace(/\/\/[^\n]*/g, "").split("\n").filter(line => line.trim()).map(line => {
    const match = line.match(/^\s*(\w+)\s*:\s*(.+),\s*$/);
    assert.ok(match, `${name}: unsupported fixture ABI declaration ${line}`);
    return [match[1], match[2].replace(/\s/g, "")];
  });
}

function verifyAbi(source, bindings, shaders) {
  const expected = {
    Frame: [
      ["alpha", "f32"], ["candidates", "u32"], ["budget", "u32"], ["culling", "u32"],
      ["groups", "u32"], ["batches", "u32"], ["node_capacity", "u32"], ["blend", "u32"],
      ["planes", "array<vec4<f32>,6>"], ["absolute_view_projection", "mat4x4<f32>"],
      ["motion", "u32"], ["joints", "u32"], ["mask", "u32"], ["occlusion", "u32"],
      ["view_projection", "mat4x4<f32>"], ["viewport", "vec4<f32>"], ["skinned", "u32"],
      ["subpixel", "u32"], ["plane", "u32"], ["lod_error", "f32"], ["eye", "vec4<f32>"], ["fade", "vec4<f32>"],
    ],
    Candidate: [["slot", "u32"], ["node", "u32"], ["batch", "u32"], ["position", "u32"]],
    Batch: [
      ["start", "u32"], ["count", "u32"], ["index_count", "u32"], ["group", "u32"],
      ["bounds_min", "vec4<f32>"], ["bounds_max", "vec4<f32>"], ["first_index", "u32"],
      ["base_vertex", "i32"], ["first_instance", "u32"], ["first_plane", "u32"],
      ["lod_ranges", "vec4<u32>"], ["lod_errors", "vec4<f32>"],
    ],
    DrawArgs: [["index_count", "u32"], ["instance_count", "u32"], ["first_index", "u32"], ["base_vertex", "i32"], ["first_instance", "u32"]],
  };
  for (const [name, fields] of Object.entries(expected)) assert.deepEqual(declarations(source, name), fields, `${name} ABI drift`);
  const frameOffsets = {};
  let bytes = 0;
  for (const [name, type] of expected.Frame) {
    const align = type.startsWith("vec4") || type.startsWith("mat4") || type.startsWith("array") ? 16 : 4;
    const size = type.startsWith("array") ? 96 : type.startsWith("mat4") ? 64 : type.startsWith("vec4") ? 16 : 4;
    bytes = Math.ceil(bytes / align) * align;
    frameOffsets[name] = bytes;
    bytes += size;
  }
  assert.equal(bytes, 336);
  assert.deepEqual(["candidates", "budget", "groups", "batches", "mask", "plane", "lod_error"].map(key => frameOffsets[key]),
    [4, 8, 16, 20, 200, 296, 300]);
  for (const [name, value] of Object.entries({ ...flags, MOVED_SLOT: moved, GROUP: 64, LOD_SHIFT: 8, LOD_LEVELS: 3, IMPOSTOR: 3, IMPOSTOR_VERTICES: 6 })) {
    const spelling = source.match(new RegExp(`const ${name}: u32 = (0x[0-9a-f]+|[0-9]+)u;`))?.[1];
    assert.equal(Number(spelling), value, `${name} ABI drift`);
  }
  const compact = bindings.match(/const PRESENTATION_COMPACT: Table = Table \{([\s\S]*?)\n    \};/)?.[1];
  assert.ok(compact, "production compact binding table exists");
  const entries = [...compact.matchAll(/(uniform|read|write)\((\d+), "(\w+)", C\)/g)].map(match => [Number(match[2]), match[1], match[3]]);
  assert.deepEqual(entries, [[0, "uniform", "frame"], [6, "read", "candidates"], [7, "read", "batches"],
    [9, "write", "scan"], [10, "write", "batch_bases"], [11, "write", "visible_ids"], [12, "write", "draws"],
    [19, "write", "occlusion_counts"], [24, "write", "impostor_ids"]]);
  assert.match(shaders, /PRESENTATION_SHADER\s*=\s*\[\s*"presentation\.wgsl",?\s*\]/);
  return entries;
}

function ranges(n, everyEnd = false) {
  const result = [];
  const sizes = everyEnd ? [1] : [7, 33, 1, 65, 31, 2, 64];
  for (let start = 0; start < n;) {
    const count = Math.min(n - start, sizes[result.length % sizes.length]);
    result.push({ start, count });
    start += count;
  }
  return result;
}

function fixture(options, capacity, batchCapacity) {
  const { n, name, mask = flags.VISIBLE, lod = true, budget = n, occupancy, firstInstance = true, everyEnd = false,
    repeated = false, triangles = false, allVisible = false, phase = 0 } = options;
  const batches = ranges(n, everyEnd).map((range, index) => ({
    ...range, indexCount: triangles ? 0xffffffff : index % 5 === 0 ? 0 : 36 + index * 3,
    firstIndex: index * 71, baseVertex: index - 17, virtual: !triangles && index % 7 === 2,
    firstInstance: firstInstance ? range.start : 0, firstPlane: firstInstance ? capacity : 0,
    levels: lod ? 3 : 1, lodRanges: [index * 83 + 1, 18 + index * 3, index * 89 + 2, 9 + index * 3],
  }));
  const slots = Array.from({ length: n }, (_, index) => index + 101);
  let seed = (0x193b + phase) >>> 0;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; };
  for (let index = n - 1; index > 0; index--) {
    const other = random() % (index + 1);
    [slots[index], slots[other]] = [slots[other], slots[index]];
  }
  const membership = Array.from({ length: n }, (_, index) => {
    const local = index % 64;
    const eligible = occupancy != null ? local < occupancy : allVisible || (index + phase) % 5 !== 0;
    const bucket = occupancy != null || !lod ? Math.floor(index / 64) % 4 : (index * 7 + phase) % 4;
    let value = flags.VALID | (bucket << 8);
    if (index % 3 !== 0) value |= flags.MOVING;
    if (index % 7 === 0) value |= flags.SUBPIXEL;
    if (index % 11 === 0) value |= flags.OCCLUDED;
    if (eligible) value |= mask;
    // Unrelated phase flags must survive and must not imply membership.
    if (mask === flags.VISIBLE && index % 2 === 0) value |= flags.PHASE2 | flags.PROBE;
    if (mask === flags.PHASE1 && index % 2 === 0) value |= flags.PHASE2;
    return value >>> 0;
  });
  const candidates = filled(capacity * 4);
  const batchOf = new Array(n);
  for (const [batch, range] of batches.entries()) {
    for (let index = range.start; index < range.start + range.count; index++) {
      batchOf[index] = batch;
      candidates.splice(index * 4, 4, repeated ? Math.floor(index / 2) + 101 : slots[index], index + 3, batch, index);
    }
  }
  const batchBytes = new ArrayBuffer(Math.max(256, batchCapacity * 96));
  const view = new DataView(batchBytes);
  for (const [index, batch] of batches.entries()) {
    const at = index * 96;
    for (const [offset, value] of [[0, batch.start], [4, batch.count], [8, batch.indexCount], [12, index],
      [48, batch.firstIndex], [52, batch.baseVertex >>> 0], [56, batch.firstInstance], [60, batch.firstPlane]]) view.setUint32(at + offset, value, true);
    view.setFloat32(at + 44, batch.virtual ? 2 : 0, true);
    for (let lane = 0; lane < 4; lane++) view.setUint32(at + 64 + lane * 4, batch.lodRanges[lane], true);
    view.setFloat32(at + 88, batch.levels, true);
  }
  const selected = [[], [], [], []];
  for (let index = 0; index < n; index++) {
    if ((membership[index] & mask) !== 0) selected[lod ? (membership[index] >>> 8) & 3 : 0].push(index);
  }
  const ids = filled(3 * capacity + 64), impostors = filled(2 * capacity + 64);
  const bases = filled(4 * batchCapacity + 64), draws = filled(5 * (3 * batchCapacity + 1) + 64);
  const counters = filled(64);
  counters[1] = 7;
  counters[2] = 13;
  if ((mask & (flags.PHASE2 | flags.PROBE)) === 0) counters[1] += membership.filter(value => (value & flags.SUBPIXEL) !== 0).length;
  const slot = index => (candidates[4 * index] | ((membership[index] & flags.MOVING) !== 0 ? moved : 0)) >>> 0;
  const saturate = value => Number(value > 0xffffffffn ? 0xffffffffn : value);
  const product = (a, b) => saturate(BigInt(a) * BigInt(b));
  for (const [batchIndex, batch] of batches.entries()) {
    const local = selected.map(list => list.filter(index => batchOf[index] === batchIndex));
    const before = selected.map(list => list.filter(index => index < batch.start).length);
    const counts = local.map(list => list.length);
    const count0 = Math.min(counts[0], Math.max(0, budget - before[0]));
    bases.splice(4 * batchIndex, 4, ...before);
    draws.splice(5 * batchIndex, 5, batch.indexCount, count0, batch.firstIndex, batch.baseVertex >>> 0, batch.firstInstance);
    if (batch.levels > 1) draws.splice(5 * (batches.length + batchIndex), 5, batch.lodRanges[1], counts[1], batch.lodRanges[0], batch.baseVertex >>> 0, batch.firstInstance + batch.firstPlane);
    if (batch.levels > 2) draws.splice(5 * (2 * batches.length + batchIndex), 5, batch.lodRanges[3], counts[2], batch.lodRanges[2], batch.baseVertex >>> 0, batch.firstInstance + 2 * batch.firstPlane);
    for (let level = 0; level < 3; level++) {
      const survivors = level === 0 ? local[level].slice(0, count0) : local[level];
      survivors.forEach((index, order) => { ids[level * capacity + batch.start + order] = slot(index); });
    }
    if ((mask & flags.PROBE) === 0 && !batch.virtual) {
      const triangles0 = product(Math.floor(batch.indexCount / 3), count0);
      const triangles1 = product(Math.floor(batch.lodRanges[1] / 3), counts[1]);
      const triangles2 = product(Math.floor(batch.lodRanges[3] / 3), counts[2]);
      counters[2] = (counters[2] + saturate(BigInt(triangles0) + BigInt(triangles1) + BigInt(triangles2))) >>> 0;
    }
  }
  if (batches.length) {
    draws.splice(5 * 3 * batches.length, 5, 6, selected[3].length, 0, 0, 0);
    counters[2] = (counters[2] + product(2, selected[3].length)) >>> 0;
  }
  selected[3].forEach((index, order) => { impostors[2 * order] = slot(index); impostors[2 * order + 1] = batchOf[index]; });
  return { name, n, groups: Math.ceil(n / 64), mask, lod, budget, capacity, batchCapacity, batchCount: batches.length,
    flags: membership, candidates, batches: Array.from(new Uint32Array(batchBytes)),
    expected: { ids, impostors, bases, draws, counters, totals: selected.map(list => list.length) } };
}

function fixtures() {
  const options = [];
  for (const n of [0, 1, 31, 32, 33, 63, 64, 65, 127, 128, 129]) options.push({ name: `boundary-${n}`, n });
  for (const occupancy of [0, 1, 7, 8, 9, 31, 32, 33, 63, 64]) options.push({ name: `occupancy-${occupancy}`, n: 256, occupancy });
  options.push({ name: "every-batch-end", n: 129, everyEnd: true }, { name: "permitted-repeated-slots", n: 129, repeated: true });
  for (const mask of [flags.VISIBLE, flags.PHASE1, flags.PHASE2, flags.PROBE]) {
    for (const lod of mask === flags.PROBE ? [false] : [false, true]) options.push({ name: `mask-${mask}-lod-${lod}`, n: 129, mask, lod, phase: mask });
  }
  for (const budget of [0, 1, 7, 31, 128, 129, 130]) {
    options.push({ name: `budget-${budget}`, n: 129, budget, lod: false, allVisible: true, firstInstance: budget % 2 === 0 });
  }
  options.push({ name: "triangle-saturation", n: 129, triangles: true, lod: false, allVisible: true });
  for (const n of [1023, 1025, 31, 8193, 129, 0]) options.push({ name: `growth-reuse-${n}`, n, phase: 3, firstInstance: false });
  let capacity = 0, batchCapacity = 0;
  return options.map(option => {
    capacity = Math.max(capacity, grow(option.n));
    batchCapacity = Math.max(batchCapacity, batchGrow(ranges(option.n, option.everyEnd).length));
    return fixture(option, capacity, batchCapacity);
  });
}

async function kernelChecks(page, source, entries, cases) {
  return page.evaluate(async ({ source, entries, cases, sentinel }) => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return { failure: "WebGPU adapter unavailable" };
    const device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ code: source });
    const compilation = await module.getCompilationInfo();
    const messages = compilation.messages.filter(message => message.type === "error");
    if (messages.length) return { failure: messages.map(message => message.message).join("\n") };
    const layout = device.createBindGroupLayout({ entries: entries.map(([binding, kind]) => ({ binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: kind === "uniform" ? "uniform" : kind === "read" ? "read-only-storage" : "storage" } })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = {};
    for (const entryPoint of ["sum_groups", "scan_groups", "batch_args", "scatter"]) {
      pipelines[entryPoint] = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } });
    }
    const create = (words, uniform = false) => device.createBuffer({ size: Math.max(256, words * 4),
      usage: (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const upload = (buffer, words) => device.queue.writeBuffer(buffer, 0, new Uint32Array(words));
    const read = async (buffers) => {
      const staging = Object.fromEntries(Object.entries(buffers).map(([name, buffer]) => [name,
        device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })]));
      const encoder = device.createCommandEncoder();
      for (const [name, buffer] of Object.entries(buffers)) encoder.copyBufferToBuffer(buffer, 0, staging[name], 0, buffer.size);
      device.queue.submit([encoder.finish()]);
      const result = {};
      for (const [name, buffer] of Object.entries(staging)) {
        await buffer.mapAsync(GPUMapMode.READ);
        result[name] = Array.from(new Uint32Array(buffer.getMappedRange()));
        buffer.unmap(); buffer.destroy();
      }
      return result;
    };
    const mismatch = (actual, expected, name) => {
      for (let index = 0; index < expected.length; index++) {
        if (actual[index] !== expected[index]) return `${name}[${index}]: ${actual[index]} != ${expected[index]}`;
      }
      return null;
    };
    let pool, priorScan;
    const results = [];
    try {
      for (const test of cases) {
        const { capacity, batchCapacity } = test;
        if (!pool || pool.capacity !== capacity || pool.batchCapacity !== batchCapacity) {
          if (pool) for (const buffer of Object.values(pool.buffers)) buffer.destroy();
          const buffers = { frame: create(84, true), candidates: create(capacity * 4), batches: create(batchCapacity * 24),
            scan: create(capacity + 16 * Math.ceil(capacity / 64) + 5 + 64), bases: create(4 * batchCapacity + 64),
            ids: create(3 * capacity + 64), draws: create(5 * (3 * batchCapacity + 1) + 64),
            counters: create(64), impostors: create(2 * capacity + 64) };
          const names = { frame: "frame", candidates: "candidates", batches: "batches", scan: "scan", batch_bases: "bases",
            visible_ids: "ids", draws: "draws", occlusion_counts: "counters", impostor_ids: "impostors" };
          pool = { capacity, batchCapacity, buffers, group: device.createBindGroup({ layout,
            entries: entries.map(([binding, , name]) => ({ binding, resource: { buffer: buffers[names[name]] } })) }) };
          priorScan = new Array(buffers.scan.size / 4).fill(sentinel);
          upload(buffers.scan, priorScan);
        }
        const b = pool.buffers;
        const frame = new ArrayBuffer(336), view = new DataView(frame);
        for (const [at, value] of [[4, test.n], [8, test.budget], [16, test.groups], [20, test.batchCount], [200, test.mask], [296, test.capacity]]) view.setUint32(at, value, true);
        view.setFloat32(300, test.lod ? 1 : 0, true);
        device.queue.writeBuffer(b.frame, 0, frame);
        upload(b.candidates, test.candidates); upload(b.batches, test.batches);
        if (test.n) upload(b.scan, test.flags);
        for (const key of ["ids", "impostors", "bases", "draws", "counters"]) {
          const initial = new Array(b[key].size / 4).fill(sentinel);
          if (key === "counters") { initial[1] = 7; initial[2] = 13; }
          upload(b[key], initial);
        }
        const encoder = device.createCommandEncoder();
        // Separate dispatches preserve production storage visibility.
        for (const [name, count] of [["sum_groups", test.groups], ["scan_groups", 1],
          ["batch_args", Math.ceil(test.batchCount / 64)], ["scatter", test.groups]]) {
          if (!count) continue;
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipelines[name]); pass.setBindGroup(0, pool.group); pass.dispatchWorkgroups(count); pass.end();
        }
        device.queue.submit([encoder.finish()]);
        const actual = await read({ scan: b.scan, bases: b.bases, ids: b.ids, draws: b.draws, counters: b.counters, impostors: b.impostors });
        let failure = mismatch(actual.scan, test.flags, "flags");
        for (const key of ["ids", "impostors", "bases", "draws", "counters"]) failure ??= mismatch(actual[key], test.expected[key], key);
        const totalAt = test.n + 4 * test.groups;
        failure ??= mismatch(actual.scan.slice(totalAt, totalAt + 4), test.expected.totals, "totals");
        failure ??= mismatch(actual.scan.slice(totalAt + 4, totalAt + 5), [test.expected.totals.reduce((a, b) => a + b, 0)], "total");
        const end = test.n + 16 * test.groups + 5;
        failure ??= mismatch(actual.scan.slice(end), priorScan.slice(end), "scan-tail");
        if (failure) return { failure: `${test.name}: ${failure}`, completed: results };
        priorScan = actual.scan;
        results.push({ name: test.name, candidates: test.n, capacity, totals: test.expected.totals });
      }
      const error = await device.popErrorScope();
      if (error || errors.length) return { failure: error?.message ?? errors.join("\n"), completed: results };
      return { adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description }, cases: results };
    } finally {
      if (pool) for (const buffer of Object.values(pool.buffers)) buffer.destroy();
      device.destroy();
    }
  }, { source, entries, cases, sentinel });
}

const liveConfig = {
  seed: 7, renderables: 513, moving_percent: 0, hierarchy: true, interpolation: true, colliders: 0,
  sensor_percent: 0, dense_colliders: false, queries_per_tick: 0, mesh_variety: 8, textured: true, culling: true,
};
const camera = { position: [0, 30, 0], target: [60, 20, 40], vertical_fov_radians: 0.9, near: 0.1, far: 400 };

function work(rendering) {
  return {
    candidates: rendering.render_candidates, visible: rendering.visible_instances, culled: rendering.culled_instances,
    dropped: rendering.dropped_instances, instances: rendering.instances, triangles: rendering.triangles,
    levels: [rendering.level_of_detail.lod0_instances, rendering.level_of_detail.lod1_instances,
      rendering.level_of_detail.lod2_instances, rendering.level_of_detail.impostor_instances],
    occluded: rendering.occlusion_culling.instances, subpixel: rendering.subpixel_culling.instances,
  };
}

async function settled(command, label) {
  let previous, stable = 0, reply;
  for (let frame = 0; frame < 600; frame++) {
    reply = await command("present", { alpha: 0, gpu: true });
    const rendering = reply.result.rendering;
    const current = JSON.stringify(work(rendering));
    stable = current === previous && !rendering.pipelines_compiling && rendering.triangles != null ? stable + 1 : 0;
    previous = current;
    if (frame >= 8 && stable >= 6) return reply;
    await delay(20);
  }
  assert.fail(`${label}: counts/pipelines did not settle: ${JSON.stringify(reply?.result.rendering)}`);
}

async function liveCapture(page, command, output, label) {
  const before = (await command("status")).observation.tick;
  const reply = await settled(command, label);
  await command("capture", { alpha: 0, gpu: true });
  const png = (await page.locator("canvas").screenshot({ path: `${output}/${label}.png` })).toString("base64");
  const after = (await command("status")).observation.tick;
  assert.equal(after, before, `${label}: GPU previews preserve gameplay ticks`);
  const rendering = reply.result.rendering;
  assert.equal(rendering.render_candidates, rendering.invalid_transforms + rendering.missing_assets
    + rendering.culled_instances + rendering.visible_instances, `${label}: candidate partition`);
  assert.equal(rendering.dropped_instances, 0, `${label}: fixture is below budget`);
  return { png, work: work(rendering) };
}

async function sameLive(page, left, right, label) {
  assert.deepEqual(left.work, right.work, `${label}: counters match a fresh renderer`);
  const pixels = await differentPixels(page, [left.png, right.png], { tolerance: 8 });
  assert.ok(pixels.changed <= pixels.total * 0.001, `${label}: ${pixels.changed}/${pixels.total} changed pixels`);
  return { label, work: left.work, pixels };
}

async function lifecycleChecks(page, output) {
  const command = commander(page, { deadline: 120_000 });
  const checks = [];
  await openPaused(page, devUrl("stress.html", { config: liveConfig }));
  const kept = [];
  for (const [index, renderables] of [513, 4097, 513].entries()) {
    if (index) await command("game", { name: "workload", args: { config: { renderables } } });
    await command("environment", { reflection_probes: { enabled: false },
      occlusion_culling: { enabled: index !== 1 }, subpixel_culling: { enabled: index !== 1 } });
    await command("camera", camera);
    kept.push({ renderables, culling: index !== 1, capture: await liveCapture(page, command, output, `reuse-${index}`) });
  }
  assert.ok(kept[0].capture.work.candidates <= 1024 && kept[1].capture.work.candidates > 4096,
    "live candidates cross the power-of-two DrawGroups capacity boundary");
  for (const [index, state] of kept.entries()) {
    await openPaused(page, devUrl("stress.html", { config: { ...liveConfig, renderables: state.renderables } }));
    await command("environment", { reflection_probes: { enabled: false },
      occlusion_culling: { enabled: state.culling }, subpixel_culling: { enabled: state.culling } });
    await command("camera", camera);
    const fresh = await liveCapture(page, command, output, `fresh-${index}`);
    checks.push(await sameLive(page, state.capture, fresh, `growth-and-reuse-${index}`));
  }
  // Ordinary temporal frames retain MOVED_SLOT history; exact fixtures above
  // already compare every selected moving and stationary ID bit for bit.
  await command("game", { name: "workload", args: { config: { moving_percent: 100 } } });
  await command("environment", { temporal_antialiasing: { enabled: true }, reflection_probes: { enabled: false } });
  await command("step", { ticks: 3 });
  const before = (await command("status")).observation.tick;
  await command("resume");
  await delay(600);
  const moving = await command("pause");
  assert.ok(moving.observation.tick > before, "ordinary moving frames advance ticks");
  assert.equal(moving.observation.presentation.gpu_pending, true);
  assert.equal(moving.result.rendering.temporal_antialiasing.active, true);
  assert.ok(moving.result.rendering.instances > 0, "ordinary moving frames retain their draws");
  checks.push({ label: "ordinary-moving-history", work: work(moving.result.rendering) });
  return checks;
}

function probeFixture(shift) {
  return {
    version: 9, id: "compaction-probe-fixture", revision: 1,
    camera: { position: [15, 12, 23], target: [0, 1, 0], fov_degrees: 55, near: 0.1, far: 500 },
    materials: [
      { id: "floor", color: [0.6, 0.62, 0.58, 1] },
      ...[0, 1, 2].map(index => ({ id: `material-${index}`, color: [0.3 + index * 0.2, 0.7 - index * 0.2, 0.2 + index * 0.2, 1], surface: { roughness: 0.25 } })),
    ],
    meshes: [{ id: "box", source: { kind: "cube" } }],
    environment: { reflection_probes: { enabled: false, faces_per_frame: 6 } },
    entities: [
      { id: "floor", mesh: "box", material: "floor", transform: { translation: [0, -0.5, 0], scale: [30, 1, 30] } },
      ...Array.from({ length: 129 }, (_, index) => ({ id: `box-${index}`, mesh: "box", material: `material-${index % 3}`,
        transform: { translation: [(index % 13) * 1.4 - 8.4 + (index % 5 === 0 ? shift : 0), 0.5 + (index % 3) * 0.3,
          Math.floor(index / 13) * 1.4 - 6.3], scale: [0.7, 1 + (index % 3) * 0.6, 0.7] } })),
      { id: "probe", reflection_probe: { extent: [16, 10, 16] }, transform: { translation: [0, 4, 0] } },
    ],
  };
}

async function captureProbe(page, command, output, label) {
  await command("environment", { reflection_probes: { enabled: true, faces_per_frame: 6 } });
  let faces = 0, ready = false;
  const before = (await command("status")).observation.tick;
  for (let frame = 0; frame < 600; frame++) {
    const rendering = (await command("present", { alpha: 0, gpu: true })).result.rendering;
    faces += rendering.reflection_probes.faces;
    ready = rendering.reflection_probes.ready > 0;
    if (ready && faces >= 6 && !rendering.pipelines_compiling) break;
    await delay(20);
  }
  assert.ok(ready && faces >= 6, `${label}: all six stationary-probe faces captured (${faces}, ready=${ready})`);
  assert.equal((await command("status")).observation.tick, before, `${label}: probe previews keep ticks`);
  return { ...(await liveCapture(page, command, output, label)), faces };
}

async function probeChecks(page, output) {
  const command = commander(page, { deadline: 120_000 });
  const scene = routedScene(page, probeFixture(0));
  await scene.route();
  await openPaused(page, sceneUrl());
  const original = await captureProbe(page, command, output, "probe-original");
  await command("environment", { reflection_probes: { enabled: false } });
  // Reload while disabled so every new arrangement requests a fresh cube.
  // The probe stays at the same position in both stationary arrangements.
  await scene.reload(command, probeFixture(0.8));
  const changed = await captureProbe(page, command, output, "probe-changed");
  scene.serve(probeFixture(0));
  await openPaused(page, sceneUrl());
  const freshOriginal = await captureProbe(page, command, output, "probe-fresh-original");
  scene.serve(probeFixture(0.8));
  await openPaused(page, sceneUrl());
  const freshChanged = await captureProbe(page, command, output, "probe-fresh-changed");
  const changedPixels = await differentPixels(page, [original.png, changed.png], { tolerance: 8 });
  assert.ok(changedPixels.changed > 0, "changed static arrangement affects the rendered fixture");
  return [await sameLive(page, original, freshOriginal, "probe-initial"), await sameLive(page, changed, freshChanged, "probe-regenerated"),
    { label: "six-face-capture", faces: [original.faces, changed.faces, freshOriginal.faces, freshChanged.faces] }];
}

async function main() {
  const [source, bindings, shaders] = await Promise.all([
    readFile(new URL("../../crates/engine_render/src/presentation.wgsl", import.meta.url), "utf8"),
    readFile(new URL("../../crates/engine_render/src/bindings.rs", import.meta.url), "utf8"),
    readFile(new URL("../../crates/engine_render/src/shaders.rs", import.meta.url), "utf8"),
  ]);
  const entries = verifyAbi(source, bindings, shaders);
  const output = await artifacts("presentation-compaction-smoke");
  const { browser, page, errors } = await launch({ width: 1200, height: 900, timeout: 120_000 });
  try {
    await openPaused(page, devUrl("stress.html", { config: { renderables: 129, moving_percent: 0, colliders: 0, queries_per_tick: 0 } }));
    const kernels = await kernelChecks(page, source, entries, fixtures());
    assert.equal(kernels.failure, undefined, kernels.failure);
    const lifecycle = await lifecycleChecks(page, output);
    const probes = await probeChecks(page, output);
    assert.deepEqual(errors, []);
    const result = { ok: true, source_sha: createHash("sha256").update(source).digest("hex"),
      commit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(), kernels, lifecycle, probes, artifact: output };
    await writeFile(`${output}/result.json`, JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally {
    await browser.close();
  }
}

await main().catch(error => {
  if (error?.code !== "ERR_ASSERTION") throw error;
  console.error(error);
  process.exitCode = 3;
});
