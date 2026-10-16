/** Removable browser diagnostics over the real renderer, never used by timing runs. */
export function installReflectionProbe() {
  const state = { device: null, live: new Set(), created: 0, destroyed: 0, frames: 0,
    armed: 0, pending: [], captures: [], inputs: null, work: null, traces: 0, fallback: 0 };
  window.__reflectionProbe = state;
  const devices = new WeakMap(), views = new WeakMap(), textures = new WeakMap(), encoders = new WeakMap(), commands = new WeakMap();
  const createBuffer = GPUDevice.prototype.createBuffer, createTexture = GPUDevice.prototype.createTexture;
  const createGroup = GPUDevice.prototype.createBindGroup, createView = GPUTexture.prototype.createView;
  const createEncoder = GPUDevice.prototype.createCommandEncoder, beginCompute = GPUCommandEncoder.prototype.beginComputePass;
  const beginRender = GPUCommandEncoder.prototype.beginRenderPass, finish = GPUCommandEncoder.prototype.finish;
  const submit = GPUQueue.prototype.submit, destroyBuffer = GPUBuffer.prototype.destroy;
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (descriptor = {}) {
    const requiredFeatures = [...descriptor.requiredFeatures ?? []];
    if (this.features.has("timestamp-query") && !requiredFeatures.includes("timestamp-query")) requiredFeatures.push("timestamp-query");
    const result = await requestDevice.call(this, { ...descriptor, requiredFeatures });
    state.device = result; state.adapter = { vendor: this.info.vendor, architecture: this.info.architecture, device: this.info.device, description: this.info.description };
    devices.set(result.queue, result); return result;
  };
  GPUDevice.prototype.createBuffer = function (descriptor) {
    const diagnostic = ["engine/reflection-scratch", "engine/post-uniform"].includes(descriptor.label);
    const buffer = createBuffer.call(this, { ...descriptor, usage: descriptor.usage | (diagnostic ? GPUBufferUsage.COPY_SRC : 0) });
    if (descriptor.label === "engine/reflection-scratch") { state.live.add(buffer); state.created++; }
    return buffer;
  };
  GPUBuffer.prototype.destroy = function () {
    if (state.live.delete(this)) state.destroyed++;
    return destroyBuffer.call(this);
  };
  GPUDevice.prototype.createTexture = function (descriptor) {
    const copyable = (descriptor.sampleCount ?? 1) === 1;
    const actual = { ...descriptor, usage: descriptor.usage | (copyable ? GPUTextureUsage.COPY_SRC : 0) };
    const texture = createTexture.call(this, actual); textures.set(texture, actual); return texture;
  };
  GPUTexture.prototype.createView = function (descriptor = {}) {
    const view = createView.call(this, descriptor); views.set(view, { texture: this, descriptor }); return view;
  };
  GPUDevice.prototype.createBindGroup = function (descriptor) {
    if (descriptor.label === "engine/reflection-inputs") state.inputs = descriptor;
    if (descriptor.label === "engine/reflection-work") state.work = descriptor;
    return createGroup.call(this, descriptor);
  };
  GPUDevice.prototype.createCommandEncoder = function (descriptor) {
    const result = createEncoder.call(this, descriptor); encoders.set(result, { reflections: false }); return result;
  };
  GPUCommandEncoder.prototype.beginComputePass = function (descriptor = {}) {
    if (descriptor.label === "engine/reflection-classify-scan-scatter") { encoders.get(this).reflections = true; state.traces++; }
    return beginCompute.call(this, descriptor);
  };
  GPUCommandEncoder.prototype.beginRenderPass = function (descriptor) {
    if (descriptor.label === "engine/post-reflections") state.fallback++;
    return beginRender.call(this, descriptor);
  };
  GPUCommandEncoder.prototype.finish = function (descriptor) {
    const result = finish.call(this, descriptor); commands.set(result, encoders.get(this)); return result;
  };
  function capture(device) {
    const encoder = createEncoder.call(device);
    const owned = [], readbacks = [];
    const copyBuffer = (source, usage, offset = 0, size = source.size - offset) => {
      const copy = createBuffer.call(device, { size, usage: usage | GPUBufferUsage.COPY_DST });
      encoder.copyBufferToBuffer(source, offset, copy, 0, size); owned.push(copy); return copy;
    };
    const entries = state.inputs.entries.map(entry => {
      if (entry.resource.buffer) return { binding: entry.binding, resource: { buffer: copyBuffer(entry.resource.buffer, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC) } };
      const view = views.get(entry.resource);
      if (!view) return entry; // The immutable linear sampler.
      const descriptor = textures.get(view.texture);
      const copy = createTexture.call(device, { ...descriptor, usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING });
      owned.push(copy);
      for (let mipLevel = 0; mipLevel < (descriptor.mipLevelCount ?? 1); mipLevel++) {
        encoder.copyTextureToTexture({ texture: view.texture, mipLevel }, { texture: copy, mipLevel },
          [Math.max(1, view.texture.width >> mipLevel), Math.max(1, view.texture.height >> mipLevel), view.texture.depthOrArrayLayers]);
      }
      return { binding: entry.binding, resource: copy.createView(view.descriptor), texture: copy };
    });
    for (const entry of state.work.entries) {
      const source = entry.resource.buffer;
      readbacks.push({ binding: entry.binding, buffer: copyBuffer(source, GPUBufferUsage.MAP_READ, entry.resource.offset ?? 0, entry.resource.size ?? source.size) });
    }
    const post = entries.find(e => e.binding === 0).resource.buffer;
    readbacks.push({ binding: 7, buffer: copyBuffer(post, GPUBufferUsage.MAP_READ) });
    return { command: encoder.finish(), entries, owned, readbacks, capacity: [...state.live].reduce((sum, b) => sum + b.size, 0) };
  }
  GPUQueue.prototype.submit = function (submitted) {
    const list = [...submitted];
    const reflections = list.some(command => commands.get(command)?.reflections);
    let saved;
    if (reflections) state.frames++;
    if (reflections && state.armed > 0) { state.armed--; saved = capture(devices.get(this)); list.push(saved.command); }
    const result = submit.call(this, list);
    if (saved) {
      const pending = Promise.all(saved.readbacks.map(async ({ binding, buffer }) => {
        await buffer.mapAsync(GPUMapMode.READ); const bytes = buffer.getMappedRange().slice(0); buffer.unmap(); buffer.destroy(); return [binding, bytes];
      })).then(read => { saved.data = Object.fromEntries(read); state.captures.push(saved); });
      state.pending.push(pending);
    }
    return result;
  };
}

/** Verify captured live work from stored inputs, then compare both orderings on
 * those immutable textures. Durations are local diagnostics, not factory evidence. */
export async function inspectLiveReflections({ sources }) {
  const state = window.__reflectionProbe, device = state.device, U = GPUBufferUsage, S = GPUShaderStage;
  const need = (value, message) => { if (!value) throw new Error(`REFLECTION_ASSERTION: ${message}`); };
  await Promise.all(state.pending);
  const read = async buffer => {
    const copy = device.createBuffer({ size: buffer.size, usage: U.COPY_DST | U.MAP_READ });
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, copy, 0, copy.size); device.queue.submit([encoder.finish()]);
    await copy.mapAsync(GPUMapMode.READ); const result = copy.getMappedRange().slice(0); copy.unmap(); copy.destroy(); return result;
  };
  const module = code => device.createShaderModule({ code });
  const inputLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: S.COMPUTE, buffer: { type: "uniform" } },
    ...[1, 2, 3, 10, 13, 14].map(binding => ({ binding, visibility: S.COMPUTE, texture: { sampleType: binding === 1 ? "depth" : [10, 13].includes(binding) ? "unfilterable-float" : "float" } })),
    { binding: 7, visibility: S.COMPUTE, sampler: { type: "filtering" } },
  ] });
  const probeLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: S.COMPUTE, buffer: { type: "storage" } }] });
  const probeModule = module(sources.trace.slice(0, sources.trace.indexOf("@group(1) @binding(0)")) + `
    @group(1) @binding(0) var<storage, read_write> samples: array<vec4<f32>>;
    @compute @workgroup_size(8,8) fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
      let size = max(vec2<u32>(post.size.xy) / 2u, vec2<u32>(1u));
      if any(id.xy >= size) { return; }
      let p = traced_pixel(vec2<i32>(id.xy));
      samples[id.y * size.x + id.x] = vec4<f32>(textureLoad(glow,p,0).g,
        textureLoad(depth_texture,p,0), textureLoad(surface_texture,p,0).z, 0.0);
    }`);
  const probePipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [inputLayout, probeLayout] }), compute: { module: probeModule, entryPoint: "probe" } });
  const output = [];
  for (const capture of state.captures) {
    const params = new Uint32Array(capture.data[0]), post = new Float32Array(capture.data[7]);
    const [width, height, tx, ty] = params, pixels = width * height;
    const samples = device.createBuffer({ size: pixels * 16, usage: U.STORAGE | U.COPY_SRC });
    const inputs = device.createBindGroup({ layout: inputLayout, entries: capture.entries.map(({ binding, resource }) => ({ binding, resource })) });
    const sampleGroup = device.createBindGroup({ layout: probeLayout, entries: [{ binding: 0, resource: { buffer: samples } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(probePipeline); pass.setBindGroup(0, inputs); pass.setBindGroup(1, sampleGroup); pass.dispatchWorkgroups(tx, ty); pass.end(); device.queue.submit([encoder.finish()]);
    const actual = new Float32Array(await read(samples)); samples.destroy();
    const expected = [[], [], [], []];
    for (let tile = 0; tile < tx * ty; tile++) for (let lane = 0; lane < 64; lane++) {
      const x = tile % tx * 8 + lane % 8, y = Math.floor(tile / tx) * 8 + Math.floor(lane / 8);
      if (x >= width || y >= height) continue;
      const i = (y * width + x) * 4;
      if (actual[i] < 0.01 || actual[i + 1] <= 0) continue;
      const roughness = actual[i + 2], bin = Number(roughness >= 0.125) + Number(roughness >= 0.25) + Number(roughness >= 0.375);
      expected[bin].push(x, y);
    }
    const counts = expected.map(b => b.length / 2), metadata = new Uint32Array(capture.data[5]), rays = new Uint32Array(capture.data[3]);
    need(counts.every((count, bin) => count === metadata[bin]), "live bin totals equal scalar reference over actual stored inputs");
    need(expected.flat().every((value, i) => value === rays[i]), "live worklist contains every original ray exactly once in bin/tile/lane order");
    output.push({ phase: post[51], active: counts.reduce((a, b) => a + b, 0), bins: counts, pixels, capacity: capture.capacity, full: [post[36], post[37]] });
  }
  return output;
}
