/** Pass-equivalent impostor depths, using the engine's registered GPU objects. */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, differentPixels, launch, openPaused, routedScene, sceneUrl } from "./harness.mjs";

// This observer forwards the original API calls. Its optional reference mode
// restores only an opaque extended depth entry immediately before its draw,
// using the actual color pipeline's extra-stream layout. It never changes the
// production shader module, material, draw range, attachment or uniform.
function observe({ originalDepth = false } = {}) {
  const ids = new WeakMap(), pipelines = new WeakMap(), groups = new WeakMap(), passes = new WeakMap();
  const modules = new WeakMap(), layouts = new WeakMap();
  const raw = Object.fromEntries(["createRenderPipeline", "createRenderPipelineAsync", "createBindGroup", "createShaderModule",
    "createBindGroupLayout"].map(name => [name, GPUDevice.prototype[name]]));
  const setPipeline = GPURenderPassEncoder.prototype.setPipeline;
  let next = 1, extraLayout;
  const id = value => { if (!ids.has(value)) ids.set(value, next++); return ids.get(value); };
  const state = window.__impostorVariants = { total: 0, descriptions: [], draws: [], fixtures: false, restored: 0 };
  const remember = (device, value, descriptor) => {
    if (state.fixtures) return value;
    state.total++;
    const fragment = descriptor.fragment?.entryPoint;
    if (!fragment?.startsWith("fs_impostor_")) return value;
    const data = { id: id(value), device, value, descriptor };
    pipelines.set(value, data);
    if (descriptor.vertex.entryPoint === "vs_impostor_bake_extended") extraLayout = descriptor.vertex.buffers[1];
    state.descriptions.push({ id: data.id, vertex: descriptor.vertex.entryPoint, fragment,
      constants: descriptor.fragment.constants, cull: descriptor.primitive?.cullMode,
      buffers: descriptor.vertex.buffers, samples: descriptor.multisample?.count,
      targets: descriptor.fragment.targets, depth: descriptor.depthStencil });
    if (fragment === "fs_impostor_shade" && !descriptor.fragment.constants.ALPHA_MASK && !state.seed) state.seed = data;
    return value;
  };
  for (const name of ["createRenderPipeline", "createRenderPipelineAsync"]) {
    GPUDevice.prototype[name] = function (descriptor) {
      const value = raw[name].call(this, descriptor);
      return name.endsWith("Async") ? value.then(result => remember(this, result, descriptor)) : remember(this, value, descriptor);
    };
  }
  GPUDevice.prototype.createShaderModule = function (descriptor) {
    const value = raw.createShaderModule.call(this, descriptor); modules.set(value, descriptor); return value;
  };
  GPUDevice.prototype.createBindGroupLayout = function (descriptor) {
    const value = raw.createBindGroupLayout.call(this, descriptor); layouts.set(value, descriptor); return value;
  };
  GPUDevice.prototype.createBindGroup = function (descriptor) {
    const value = raw.createBindGroup.call(this, descriptor); groups.set(value, descriptor); return value;
  };
  const begin = GPUCommandEncoder.prototype.beginRenderPass;
  GPUCommandEncoder.prototype.beginRenderPass = function (descriptor) {
    const pass = begin.call(this, descriptor);
    if (!state.fixtures && ["engine/impostor-shade", "engine/impostor-bake"].includes(descriptor.label)) {
      passes.set(pass, { label: descriptor.label, buffers: {}, groups: {}, bindings: {} });
    }
    return pass;
  };
  for (const name of ["setPipeline", "setVertexBuffer", "setIndexBuffer", "setBindGroup", "drawIndexed"]) {
    const original = GPURenderPassEncoder.prototype[name];
    GPURenderPassEncoder.prototype[name] = function (...args) {
      const pass = passes.get(this);
      if (pass) {
        if (name === "setPipeline") pass.pipeline = pipelines.get(args[0]);
        if (name === "setVertexBuffer") pass.buffers[args[0]] = { id: id(args[1]), offset: args[2] ?? 0, size: args[3] };
        if (name === "setIndexBuffer") pass.index = { id: id(args[0]), format: args[1], offset: args[2] ?? 0, size: args[3] };
        if (name === "setBindGroup" && args[1]) { pass.groups[args[0]] = id(args[1]); pass.bindings[args[0]] = args[1]; }
        if (name === "drawIndexed") {
          const pipeline = pass.pipeline;
          let selected = pipeline;
          if (originalDepth && pipeline?.descriptor.fragment.entryPoint === "fs_impostor_shade"
              && !pipeline.descriptor.fragment.constants.ALPHA_MASK && pass.buffers[1]) {
            if (!pipeline.original) {
              if (!extraLayout) throw new Error("The original extended color layout must exist before its depth draw");
              const descriptor = pipeline.descriptor;
              const originalDescriptor = { ...descriptor,
                label: "impostor-variant-smoke/original-depth",
                vertex: { ...descriptor.vertex, entryPoint: "vs_impostor_bake_extended", buffers: [...descriptor.vertex.buffers, extraLayout] },
                fragment: { ...descriptor.fragment, entryPoint: "fs_impostor_shade_extended" } };
              pipeline.original = remember(pipeline.device, raw.createRenderPipeline.call(pipeline.device, originalDescriptor), originalDescriptor);
            }
            setPipeline.call(this, pipeline.original);
            selected = pipelines.get(pipeline.original);
            state.restored++;
          }
          const record = { label: pass.label, pipeline: selected?.id, buffers: structuredClone(pass.buffers),
            groups: { ...pass.groups }, index: pass.index, args };
          state.draws.push(record);
          if (state.seed && !state.bindings && pass.label === "engine/impostor-shade" && pass.pipeline === state.seed) {
            state.bindings = { material: groups.get(pass.bindings[1]), bake: groups.get(pass.bindings[2]) };
            state.layoutDescriptions = { material: layouts.get(state.bindings.material.layout), bake: layouts.get(state.bindings.bake.layout) };
            state.shader = modules.get(state.seed.descriptor.vertex.module).code;
            state.frames = args[1];
          }
          if (pass.label === "engine/impostor-bake" && pipeline?.descriptor.vertex.entryPoint === "vs_impostor_bake_extended") {
            state.colorSeed ??= pipeline;
          }
        }
      }
      return original.apply(this, args);
    };
  }
  state.snapshot = () => ({ total: state.total, descriptions: state.descriptions, draws: state.draws, restored: state.restored });
}

async function settle(command, count = 12) {
  let rendering;
  for (let frame = 0; frame < 600; frame++) {
    rendering = (await command("present", { alpha: 0, gpu: true })).result.rendering;
    if (frame >= count && !rendering.pipelines_compiling) return rendering;
    await delay(25);
  }
  assert.fail("Impostor pipelines did not finish compiling");
}

function checkTrace(trace, originalDepth = false) {
  const descriptions = new Map(trace.descriptions.map(value => [value.id, value]));
  const depths = trace.descriptions.filter(value => value.fragment.startsWith("fs_impostor_shade"));
  assert.ok(depths.length > 0, "real depth bake pipelines were created");
  if (!originalDepth) assert.ok(depths.every(value => value.constants.ALPHA_MASK || value.vertex === "vs_impostor_bake"),
    "no opaque extended depth pipeline is created");
  const associations = [];
  for (let index = 0; index < trace.draws.length; index += 2) {
    const shade = trace.draws[index], color = trace.draws[index + 1];
    assert.equal(shade.label, "engine/impostor-shade");
    assert.equal(color?.label, "engine/impostor-bake");
    assert.deepEqual(shade.args, color.args, "depth/color index range and instances agree");
    assert.deepEqual(shade.index, color.index, "the index buffer and offset agree");
    assert.deepEqual(shade.buffers[0], color.buffers[0], "the vertex arena and offset agree");
    assert.equal(shade.groups[1], color.groups[1], "the material is shared");
    const a = descriptions.get(shade.pipeline), b = descriptions.get(color.pipeline);
    assert.deepEqual(a.constants, b.constants, "every material override is preserved");
    assert.equal(a.cull, b.cull, "culling is preserved");
    assert.equal(a.samples, 1); assert.equal(b.samples, 4);
    assert.equal(a.targets.length, 0); assert.equal(b.targets.length, 4);
    const extended = b.vertex === "vs_impostor_bake_extended";
    if (a.constants.ALPHA_MASK || originalDepth) assert.equal(a.vertex.endsWith("_extended"), extended, "original/masked streams remain intact");
    associations.push({ depth: a.id, color: b.id, extended, masked: !!a.constants.ALPHA_MASK, cull: a.cull,
      vertex: shade.buffers[0], index: shade.index, range: shade.args });
  }
  const shared = depths.filter(depth => !depth.constants.ALPHA_MASK && [true, false].every(extended =>
    associations.some(value => value.depth === depth.id && value.extended === extended)));
  if (!originalDepth) assert.ok(shared.length >= 1, "unchanged Metro uses one actual depth pipeline for both stream classes");
  const originalKeys = new Set(associations.map(value => JSON.stringify([value.masked, value.cull, value.extended])));
  const actualKeys = new Set(associations.map(value => value.depth));
  if (originalDepth) {
    assert.equal(originalKeys.size, actualKeys.size, "the reference creates and uses each original occupied key");
    assert.ok(trace.restored > 0, "the Metro reference actually uses restored extended depth pipelines");
  } else assert.ok(originalKeys.size > actualKeys.size, "an occupied original depth key was eliminated");
  assert.ok(associations.some(value => value.masked && value.extended), "Metro retains a masked extended control");
  return { total_pipeline_creations: trace.total, original_occupied_depth_keys: originalKeys.size,
    actual_depth_pipelines: actualKeys.size, removed: originalKeys.size - actualKeys.size,
    color_keys: trace.descriptions.filter(value => value.fragment.startsWith("fs_impostor_bake"))
      .map(value => JSON.stringify([value.vertex, value.fragment, value.constants, value.cull, value.buffers, value.targets])).sort(), associations };
}

// All pipeline modules and bind-group layouts below are handles captured from
// the running engine. The explicit vertex/uniform ABI checks fail if that ABI
// changes; this fixture does not substitute a hand-written depth shader.
async function depthFixtures() {
  const state = window.__impostorVariants;
  const require = (condition, message) => { if (!condition) throw new Error(`assertion: ${message}`); };
  require(state.seed && state.bindings && state.colorSeed, "the real engine registered basic and extended bake pipelines");
  const { device, descriptor } = state.seed;
  state.fixtures = true;
  device.pushErrorScope("validation");
  const baseLayout = descriptor.vertex.buffers[0], extraLayout = state.colorSeed.descriptor.vertex.buffers[1];
  require(baseLayout.arrayStride === 24 && extraLayout.arrayStride === 16, "current vertex strides");
  const fields = layout => layout.attributes.map(value => [value.shaderLocation, value.format, value.offset]);
  require(JSON.stringify(fields(baseLayout)) === JSON.stringify([[0, "float32x3", 0], [1, "snorm16x2", 12], [7, "float32x2", 16]]), "current primary stream ABI");
  require(JSON.stringify(fields(extraLayout)) === JSON.stringify([[9, "unorm16x4", 0], [10, "float32x2", 8]]), "current extra stream ABI");
  const materialSize = state.layoutDescriptions.material.entries.find(value => value.binding === 10).buffer.minBindingSize;
  require(materialSize === 240, "current material uniform ABI");
  require(state.layoutDescriptions.bake.entries.find(value => value.binding === 20).buffer.minBindingSize === 32, "current bake uniform ABI");
  const frameSide = Math.sqrt(state.frames);
  require(frameSide === 8 && state.frames === 64, "all production impostor frames");
  const hemisphere = Number(state.shader.match(/const IMPOSTOR_HEMISPHERE(?:\s*:\s*u32)?\s*=\s*(0x[\da-f]+|\d+)u?\s*;/i)?.[1]);
  require(Number.isInteger(hemisphere) && hemisphere > 64,
    `registered hemisphere flag: ${state.shader.match(/[^\n]*IMPOSTOR_HEMISPHERE[^\n]*/g)?.slice(0, 2)}`);
  const createBuffer = (bytes, usage) => {
    const buffer = device.createBuffer({ size: bytes.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, bytes); return buffer;
  };
  const formats = state.colorSeed.descriptor.fragment.targets.map(value => value.format);
  require(JSON.stringify(formats) === JSON.stringify(["rgba8unorm-srgb", "rgba8unorm-srgb", "rgba8unorm", "rgba8unorm"]), "production bake targets");
  const pipelineCache = new Map();
  function pipeline(extended, masked, cull, color = false) {
    const key = JSON.stringify([extended, masked, cull, color]);
    if (pipelineCache.has(key)) return pipelineCache.get(key);
    const template = color ? state.colorSeed.descriptor : descriptor;
    const result = device.createRenderPipeline({ ...template, label: `impostor-variant-smoke/${key}`,
      vertex: { ...template.vertex, entryPoint: `vs_impostor_bake${extended ? "_extended" : ""}`, buffers: extended ? [baseLayout, extraLayout] : [baseLayout] },
      fragment: { ...template.fragment, entryPoint: `fs_impostor_${color ? "bake" : "shade"}${extended ? "_extended" : ""}`,
        constants: { ...template.fragment.constants, ALPHA_MASK: Number(masked), DOUBLE_SIDED: Number(cull === "none") } },
      primitive: { ...template.primitive, cullMode: cull } });
    pipelineCache.set(key, result); return result;
  }
  const materialBytes = new ArrayBuffer(materialSize), materialFloats = new Float32Array(materialBytes), materialWords = new Uint32Array(materialBytes);
  for (let map = 0; map < 5; map++) { materialFloats[map * 8] = 1; materialFloats[map * 8 + 5] = 1; }
  materialFloats.set([1, 1, 0.5, 1], 44); // scale, occlusion, cutoff, base map reads UV1.
  materialWords[51] = 1; // VARIANT_MAPS in the unchanged production uniform.
  const materialUniform = createBuffer(materialBytes, GPUBufferUsage.UNIFORM);
  const texture = device.createTexture({ size: [8, 8], format: "rgba8unorm-srgb", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const texels = new Uint8Array(8 * 8 * 4);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) texels.set([80 + x * 20, 100 + y * 15, 190, x < 4 ? 0 : 255], (y * 8 + x) * 4);
  device.queue.writeTexture({ texture }, texels, { bytesPerRow: 32 }, [8, 8]);
  const sampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
  const materialGroup = device.createBindGroup({ ...state.bindings.material, label: "impostor-variant-smoke/material",
    entries: state.bindings.material.entries.map(entry => ({ binding: entry.binding,
      resource: entry.binding === 10 ? { buffer: materialUniform } : entry.binding % 2 ? sampler : texture.createView() })) });
  const emptyDepth = device.createTexture({ size: [1, 1], format: "depth32float", usage: GPUTextureUsage.TEXTURE_BINDING });
  const corners = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
  const indices = [0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,3,7,6,3,6,2,0,4,7,0,7,3,1,2,6,1,6,5];
  const indexBytes = new Uint32Array(9 + indices.length); indexBytes.set(indices, 9);
  const indexBuffer = createBuffer(indexBytes, GPUBufferUsage.INDEX);
  const extraBytes = new ArrayBuffer(15 * 16), extras = new DataView(extraBytes);
  function writeExtras(alpha) {
    for (let i = 0; i < 8; i++) {
      const offset = (7 + i) * 16;
      [12000 + i * 5000, 60000 - i * 3500, 40000, alpha].forEach((value, j) => extras.setUint16(offset + 2 * j, value, true));
      extras.setFloat32(offset + 8, 0.75, true); extras.setFloat32(offset + 12, 0.5, true);
    }
  }
  writeExtras(0);
  const extraBuffer = createBuffer(extraBytes, GPUBufferUsage.VERTEX);
  const coordinates = [
    { name: "ordinary", center: [0, 0, 0], radius: 1, edge: 0.55 },
    { name: "large-boundary", center: [262144, -131072, 131072], radius: 8192, edge: 0.57730 },
    { name: "small-boundary", center: [0, 0, 0], radius: 2 ** -14, edge: 0.57730 },
  ];
  const shapes = [{ name: "sphere", texels: 16, flags: 16 }, { name: "hemisphere", texels: 64, flags: 64 | hemisphere }];
  const results = state.fixtureResults = [];
  const digest = async bytes => [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(value => value.toString(16).padStart(2, "0")).join("");
  function equal(a, b, label) {
    require(a.length === b.length, `${label}: sizes agree`);
    let changed = 0; for (let i = 0; i < a.length; i++) changed += a[i] !== b[i];
    require(changed === 0, `${label}: ${changed} bytes changed`);
  }
  async function pixels(target, format, size) {
    const buffer = device.createBuffer({ size: size * size * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture: target, aspect: format === "depth32float" ? "depth-only" : "all" },
      { buffer, bytesPerRow: size * 4 }, [size, size]);
    device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
    const data = new Uint8Array(buffer.getMappedRange()).slice(); buffer.unmap(); buffer.destroy(); return data;
  }
  for (const shape of shapes) for (const cull of ["back", "none"]) for (const coordinate of coordinates) {
    state.currentFixture = { shape: shape.name, cull, coordinates: coordinate.name };
    const vertexBytes = new ArrayBuffer(15 * 24), vertices = new DataView(vertexBytes);
    corners.forEach((point, i) => {
      const offset = (7 + i) * 24;
      point.forEach((value, j) => vertices.setFloat32(offset + j * 4, coordinate.center[j] + value * coordinate.radius * coordinate.edge, true));
      vertices.setFloat32(offset + 16, 0.25, true); vertices.setFloat32(offset + 20, 0.5, true);
    });
    const vertexBuffer = createBuffer(vertexBytes, GPUBufferUsage.VERTEX);
    const uniformBytes = new ArrayBuffer(32);
    new Float32Array(uniformBytes).set([...coordinate.center, coordinate.radius]); new Uint32Array(uniformBytes)[4] = shape.flags;
    const uniform = createBuffer(uniformBytes, GPUBufferUsage.UNIFORM);
    const group = depth => device.createBindGroup({ ...state.bindings.bake, label: "impostor-variant-smoke/bake",
      entries: [{ binding: 20, resource: { buffer: uniform } }, { binding: 21, resource: depth.createView() }] });
    const depthSize = shape.texels * frameSide * 4, colorSize = shape.texels * frameSide;
    function draw(pass, selected, bakeGroup, extended) {
      pass.setPipeline(selected); pass.setBindGroup(1, materialGroup); pass.setBindGroup(2, bakeGroup);
      pass.setVertexBuffer(0, vertexBuffer, 4 * 24); if (extended) pass.setVertexBuffer(1, extraBuffer, 4 * 16);
      pass.setIndexBuffer(indexBuffer, "uint32", 4 * 4); pass.drawIndexed(indices.length, state.frames, 5, 3, 0); pass.end();
    }
    async function depth(extended, masked) {
      const target = device.createTexture({ size: [depthSize, depthSize], format: "depth32float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({ colorAttachments: [], depthStencilAttachment: {
        view: target.createView(), depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "store" } });
      draw(pass, pipeline(extended, masked, cull), group(emptyDepth), extended);
      device.queue.submit([encoder.finish()]);
      return { target, bytes: await pixels(target, "depth32float", depthSize) };
    }
    async function colors(depthTarget, extended) {
      const targets = formats.map(format => device.createTexture({ size: [colorSize, colorSize], format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }));
      const multisampled = formats.map(format => device.createTexture({ size: [colorSize, colorSize], format, sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT }));
      const z = device.createTexture({ size: [colorSize, colorSize], format: "depth32float", sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({ colorAttachments: targets.map((target, i) => ({ view: multisampled[i].createView(),
        resolveTarget: target.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "discard" })),
        depthStencilAttachment: { view: z.createView(), depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "discard" } });
      draw(pass, pipeline(extended, false, cull, true), group(depthTarget), extended);
      device.queue.submit([encoder.finish()]);
      const bytes = await Promise.all(targets.map((target, i) => pixels(target, formats[i], colorSize)));
      [...targets, ...multisampled, z].forEach(value => value.destroy()); return bytes;
    }
    const basic = await depth(false, false), extended = await depth(true, false);
    equal(basic.bytes, extended.bytes, `${shape.name}/${cull}/${coordinate.name}: opaque depth and coverage`);
    const words = new Uint32Array(basic.bytes.buffer), counts = new Array(state.frames).fill(0);
    for (let y = 0; y < depthSize; y++) for (let x = 0; x < depthSize; x++) {
      if (words[y * depthSize + x]) counts[Math.floor(y / (depthSize / frameSide)) * frameSide + Math.floor(x / (depthSize / frameSide))]++;
    }
    require(counts.every(value => value > 0), "every production frame contains real coverage");
    let colorHashes;
    if (coordinate.name === "ordinary") {
      const a = await colors(extended.target, true), b = await colors(basic.target, true);
      a.forEach((bytes, i) => equal(bytes, b[i], `original/canonical depth feeds color target ${i}`));
      const plain = await colors(basic.target, false);
      require(a[0].some((value, i) => value !== plain[0][i]), "opaque vertex colors and UV1 still affect the separate color bake");
      colorHashes = await Promise.all(a.map(digest));
    }
    const maskedBasic = await depth(false, true), alphaZero = await depth(true, true);
    require(maskedBasic.bytes.every(value => value === 0), "masked basic uses transparent UV0");
    require(alphaZero.bytes.every(value => value === 0), "masked extended honors zero vertex alpha");
    writeExtras(65535); device.queue.writeBuffer(extraBuffer, 0, extraBytes);
    const maskedExtended = await depth(true, true);
    equal(maskedExtended.bytes, basic.bytes, "opaque UV1 and vertex alpha retain every covered masked fragment");
    materialFloats[47] = 0; device.queue.writeBuffer(materialUniform, 0, materialBytes);
    const maskedUv0 = await depth(true, true);
    require(maskedUv0.bytes.every(value => value === 0), "masked extended honors material UV selection");
    materialFloats[47] = 1; device.queue.writeBuffer(materialUniform, 0, materialBytes);
    writeExtras(0); device.queue.writeBuffer(extraBuffer, 0, extraBytes);
    results.push({ shape: shape.name, cull, coordinates: coordinate.name, depth_size: depthSize,
      frame_coverage: counts, depth_sha256: await digest(basic.bytes), color_sha256: colorHashes,
      exact_depth: true, masked_vertex_alpha: true, masked_uv1: true, nonzero_buffer_index_offsets: true });
    [basic, extended, maskedBasic, alphaZero, maskedExtended, maskedUv0].forEach(value => value.target.destroy());
    vertexBuffer.destroy(); uniform.destroy();
  }
  [materialUniform, extraBuffer, indexBuffer].forEach(value => value.destroy()); texture.destroy(); emptyDepth.destroy();
  await device.queue.onSubmittedWorkDone();
  const error = await device.popErrorScope(); require(!error, `clean GPU validation: ${error?.message}`);
  return { cases: results, depth_pairs: results.length, all_frames: state.frames, shader_characters: state.shader.length };
}

function fixture() {
  const vertices = [], indices = [], segments = 32;
  for (const y of [-1.5, 1.5]) vertices.push({ position: [0, y, 0], normal: [0, Math.sign(y), 0], uv: [0.5, 0.5] });
  for (let i = 0; i < segments; i++) {
    const angle = i * 2 * Math.PI / segments, x = Math.cos(angle), z = Math.sin(angle);
    vertices.push({ position: [x, 0, z], normal: [x, 0, z], uv: [i / segments, 0.5] });
    const current = i + 2, next = 2 + (i + 1) % segments;
    indices.push(0, current, next, 1, next, current);
  }
  const extended = vertices.map((vertex, i) => ({ ...vertex, color: [0.3 + (i % 3) * 0.3, 0.8, 0.4, 1], uv1: [0.75, 0.5] }));
  return { version: 9, id: "impostor-variant-fixture", revision: 1,
    camera: { position: [0, 8, 50], target: [0, 0, -50], fov_degrees: 55, near: 0.1, far: 1000 },
    environment: { level_of_detail: { enabled: true, impostors: true, pixel_error: 4, fade_seconds: 0 },
      temporal_antialiasing: { enabled: false }, reflection_probes: { enabled: false } },
    materials: [{ id: "basic", color: [0.8, 0.2, 0.1, 1] }, { id: "extended", color: [0.7, 0.7, 0.9, 1] }],
    meshes: [{ id: "basic", source: { kind: "inline", vertices, indices } },
      { id: "extended", source: { kind: "inline", vertices: extended, indices } }],
    entities: Array.from({ length: 80 }, (_, index) => ({ id: `object-${index}`, mesh: index % 2 ? "extended" : "basic",
      material: index % 2 ? "extended" : "basic", transform: { translation: [(index % 10 - 4.5) * 7, 0, -30 - Math.floor(index / 10) * 12] } })) };
}

async function fixtureImage(output, originalDepth) {
  const { browser, page, errors } = await launch({ width: 1000, height: 700, timeout: 120000 });
  try {
    await page.addInitScript(observe, { originalDepth });
    const document = fixture(), scene = routedScene(page, document), command = commander(page, { deadline: 120000 });
    await scene.route(); await openPaused(page, sceneUrl(), { timeout: 120000 });
    let rendering = await settle(command, 30);
    assert.ok(rendering.level_of_detail.impostor_instances > 0, "fixture actually draws impostors");
    await command("capture", { alpha: 0, gpu: true });
    const bytes = await page.locator("canvas").screenshot({ path: `${output}/${originalDepth ? "original" : "canonical"}.png` });
    const trace = await page.evaluate(() => window.__impostorVariants.snapshot());
    if (originalDepth) assert.ok(trace.restored > 0, "the full-image reference actually restored the original extended depth entry");
    const checks = { impostor_instances: rendering.level_of_detail.impostor_instances, original_depth_draws: trace.restored };
    if (!originalDepth) {
      await command("environment", { level_of_detail: { impostors: false } });
      rendering = await settle(command);
      assert.equal(rendering.level_of_detail.impostor_instances, 0, "impostors-off keeps the original mesh path");
      await command("environment", { level_of_detail: { impostors: true } });
      await settle(command);
      const updated = structuredClone(document); updated.materials[1].color = [0.1, 0.1, 0.9, 1];
      const before = (await page.evaluate(() => window.__impostorVariants.snapshot())).draws.length;
      await scene.reload(command, updated); await settle(command, 30); await command("capture", { alpha: 0, gpu: true });
      const replaced = await page.locator("canvas").screenshot({ path: `${output}/replaced-material.png` });
      const after = (await page.evaluate(() => window.__impostorVariants.snapshot())).draws.length;
      assert.ok(after > before, "material replacement rebakes the production atlas");
      const changed = await differentPixels(page, [bytes.toString("base64"), replaced.toString("base64")]);
      assert.ok(changed.changed > 20, `the replacement reaches the full image: ${JSON.stringify(changed)}`);
      await command("environment", { temporal_antialiasing: { enabled: true } });
      await command("resume"); await delay(500); await command("pause");
      rendering = (await command("status")).result.rendering;
      assert.ok(rendering.temporal_antialiasing.active, "temporal presentation still runs after replacement");
      checks.off_path = true; checks.material_rebaked = true; checks.replacement_pixels = changed.changed; checks.temporal_after_reload = true;
    }
    assert.deepEqual(errors, []);
    return { png: bytes.toString("base64"), checks };
  } finally { await browser.close(); }
}

async function main() {
  const output = await artifacts("impostor-variant-smoke");
  const metro = JSON.parse(await readFile(new URL("../../bench/scenes/metro.json", import.meta.url), "utf8"));
  const settings = JSON.parse(await readFile(new URL("../../bench/settings/default.json", import.meta.url), "utf8"));
  const { browser, page, errors } = await launch({ width: 1400, height: 900, timeout: 180000 });
  let mapping, depths;
  try {
    await page.addInitScript(observe);
    const url = devUrl("stress.html", { ...metro.options, ...settings.options });
    await openPaused(page, url, { timeout: 180000 });
    const command = commander(page, { deadline: 180000 }); await settle(command);
    const trace = await page.evaluate(() => window.__impostorVariants.snapshot());
    mapping = checkTrace(trace);
    await writeFile(`${output}/metro-pipelines.json`, JSON.stringify(trace, null, 2));
    depths = await page.evaluate(depthFixtures);
    await writeFile(`${output}/depth-fixtures.json`, JSON.stringify(depths, null, 2));
    assert.deepEqual(errors, []);
  } catch (error) {
    const partial = await page.evaluate(() => ({ current: window.__impostorVariants?.currentFixture,
      completed: window.__impostorVariants?.fixtureResults })).catch(() => null);
    await writeFile(`${output}/failure.json`, JSON.stringify({ error: String(error), partial, errors }, null, 2));
    throw error;
  } finally { await browser.close(); }
  const reference = await launch({ width: 1400, height: 900, timeout: 180000 });
  let originalMapping;
  try {
    await reference.page.addInitScript(observe, { originalDepth: true });
    await openPaused(reference.page, devUrl("stress.html", { ...metro.options, ...settings.options }), { timeout: 180000 });
    await settle(commander(reference.page, { deadline: 180000 }));
    const trace = await reference.page.evaluate(() => window.__impostorVariants.snapshot());
    originalMapping = checkTrace(trace, true);
    assert.equal(originalMapping.actual_depth_pipelines, mapping.actual_depth_pipelines + mapping.removed,
      "a separate unchanged Metro run actually creates the original depth pipelines");
    assert.deepEqual(originalMapping.color_keys, mapping.color_keys, "the real color pipeline keys are unchanged");
    assert.deepEqual(reference.errors, []);
    await writeFile(`${output}/metro-original-pipelines.json`, JSON.stringify(trace, null, 2));
  } finally { await reference.browser.close(); }
  const original = await fixtureImage(output, true), canonical = await fixtureImage(output, false);
  const comparison = await launch({ width: 1000, height: 700 });
  let images;
  try {
    images = await differentPixels(comparison.page, [original.png, canonical.png]);
    assert.equal(images.changed, 0, `the full impostor image matches the original depth path: ${JSON.stringify(images)}`);
  } finally { await comparison.browser.close(); }
  const report = { ok: true, mapping, original_mapping: originalMapping, depths, full_image: images, original: original.checks, canonical: canonical.checks };
  await writeFile(`${output}/report.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ ok: true, output, removed_pipelines: mapping.removed, depth_cases: depths.depth_pairs, full_image: images }));
}

main().catch(error => { console.error(error); process.exitCode = error.code === "ERR_ASSERTION" || String(error).includes("assertion:") ? 3 : 1; });
