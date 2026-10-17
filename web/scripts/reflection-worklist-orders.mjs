/** Replay identical captured Metro inputs with four bins and one spatial queue.
 * Only diagnostic shader copies change; production and workloads stay untouched. */
export async function compareReflectionOrders({ sources }) {
  const state = window.__reflectionProbe, device = state.device;
  const need = (ok, message) => { if (!ok) throw new Error(`REFLECTION_ASSERTION: ${message}`); };
  await Promise.all(state.pending);
  const capture = state.captures[0];
  need(capture, "a real unchanged frame must be captured before the ablation");
  const U = GPUBufferUsage, T = GPUTextureUsage, C = GPUShaderStage.COMPUTE, F = GPUShaderStage.FRAGMENT;
  const owned = [], keep = object => { owned.push(object); return object; };
  const buffer = (size, usage) => keep(device.createBuffer({ size, usage }));
  const module = code => device.createShaderModule({ code });
  const params = new Uint32Array(capture.data[0]), [width, height, tx, ty] = params;
  const uniform = new Float32Array(capture.data[7]), fullWidth = uniform[36], fullHeight = uniform[37];
  const pixels = width * height, levels = []; let length = tx * ty, offset = 0;
  for (;;) { levels.push({ offset, length }); offset += length; if (length <= 64) break; length = Math.ceil(length / 64); }
  const stride = Math.ceil(48 / device.limits.minUniformBufferOffsetAlignment) * device.limits.minUniformBufferOffsetAlignment;
  const bytes = [pixels * 4, offset * 4, pixels * 8, pixels * 16, 16, 12, (levels.length + 1) * stride];
  need(bytes.reduce((a, b) => a + b, 0) === capture.capacity, "live Post allocation matches derived full capacity including padding");
  const read = async source => {
    const copy = device.createBuffer({ size: source.size, usage: U.COPY_DST | U.MAP_READ });
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(source, 0, copy, 0, source.size); device.queue.submit([encoder.finish()]);
    await copy.mapAsync(GPUMapMode.READ); const data = copy.getMappedRange().slice(0); copy.unmap(); copy.destroy(); return data;
  };
  const textureBytes = async (texture, bytesPerPixel, aspect = "all") => {
    const pitch = Math.ceil(texture.width * bytesPerPixel / 256) * 256, result = buffer(pitch * texture.height, U.COPY_DST | U.COPY_SRC);
    const encoder = device.createCommandEncoder(); encoder.copyTextureToBuffer({ texture, aspect }, { buffer: result, bytesPerRow: pitch }, [texture.width, texture.height]); device.queue.submit([encoder.finish()]);
    return { bytes: await read(result), pitch };
  };
  const depth = await textureBytes(capture.entries.find(e => e.binding === 1).texture, 4, "depth-only");
  const glow = await textureBytes(capture.entries.find(e => e.binding === 3).texture, 2);
  const surface = await textureBytes(capture.entries.find(e => e.binding === 13).texture, 4);
  const depthValues = new Float32Array(depth.bytes), glowValues = new Uint8Array(glow.bytes), surfaces = new Uint32Array(surface.bytes);
  const uniformEntry = (binding, visibility, dynamic = false) => ({ binding, visibility, buffer: { type: "uniform", hasDynamicOffset: dynamic, minBindingSize: dynamic ? 48 : 0 } });
  const storageEntry = (binding, visibility, type = "storage") => ({ binding, visibility, buffer: { type } });
  const inputLayout = device.createBindGroupLayout({ entries: [uniformEntry(0, C | F),
    ...[1, 2, 3, 10, 13, 14].map(binding => ({ binding, visibility: C | F, texture: { sampleType: binding === 1 ? "depth" : [10, 13].includes(binding) ? "unfilterable-float" : "float" } })),
    { binding: 7, visibility: C | F, sampler: { type: "filtering" } }] });
  const workLayout = device.createBindGroupLayout({ entries: [uniformEntry(0, C, true), ...[1, 2, 3, 4, 5, 6].map(i => storageEntry(i, C))] });
  const traceLayout = device.createBindGroupLayout({ entries: [uniformEntry(0, C, true), storageEntry(1, C, "read-only-storage"), storageEntry(2, C, "read-only-storage"), storageEntry(3, C)] });
  const exportLayout = device.createBindGroupLayout({ entries: [uniformEntry(0, F, true), storageEntry(1, F, "read-only-storage")] });
  const group = (layout, entries) => device.createBindGroup({ layout, entries: entries.map(([binding, resource]) => ({ binding, resource })) });
  const inputs = group(inputLayout, capture.entries.map(({ binding, resource }) => [binding, resource]));
  const workPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [inputLayout, workLayout] });
  const tracePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [inputLayout, traceLayout] });
  const compute = (shader, entryPoint, layout = workPipelineLayout) => device.createComputePipeline({ layout, compute: { module: shader, entryPoint } });
  const variants = [sources, sources.sibling].map((source, variant) => {
    const bins = variant === 0 ? 1 : 4;
    const sizes = variant === 0 ? bytes : [pixels * 4, offset * 16, pixels * 8, pixels * 16, 48, 48, (levels.length + 5) * stride];
    const buffers = sizes.map((size, i) => buffer(size, U.COPY_SRC | U.COPY_DST | (i === 6 ? U.UNIFORM : U.STORAGE) | (i === 5 ? U.INDIRECT : 0)));
    const rowLog2 = params[9];
    const record = (i, scan, bin = 0) => device.queue.writeBuffer(buffers[6], i * stride,
      new Uint32Array([...params.slice(0, 4), ...scan, ...(bins === 1 ? [pixels, rowLog2, 0, 0] : [bin, pixels, rowLog2, 0])]));
    record(0, [0, 0, 0, 0]);
    const row = 2 ** rowLog2;
    levels.forEach((level, i) => record(i + 1, [level.offset, level.length, levels[i + 1]?.offset ?? 0xffffffff, Math.min(row, Math.ceil(level.length / 64))]));
    if (bins === 4) for (let bin = 0; bin < 4; bin++) record(levels.length + bin + 1, [0, 0, 0, 0], bin);
    const parameter = { buffer: buffers[6], size: 48 };
    const workGroup = group(workLayout, [[0, parameter], ...[0, 1, 2, 3, 4, 5].map((b, i) => [i + 1, { buffer: buffers[b] }])]);
    const traceGroup = group(traceLayout, [[0, parameter], [1, { buffer: buffers[2] }], [2, { buffer: buffers[4] }], [3, { buffer: buffers[3] }]]);
    const exportGroup = group(exportLayout, [[0, parameter], [1, { buffer: buffers[3] }]]);
    const shader = module(source.work);
    const work = Object.fromEntries(["classify", "scan_up", "scan_down", "finalize", "scatter"].map(entry => [entry, compute(shader, entry)]));
    return { bins, buffers, workGroup, traceGroup, exportGroup, work, capacity: sizes.reduce((a, b) => a + b, 0),
      trace: compute(module(source.trace), "trace", tracePipelineLayout) };
  });
  const render = (shader, entryPoint, layouts) => device.createRenderPipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
    vertex: { module: shader, entryPoint: "vs_fullscreen" }, fragment: { module: shader, entryPoint, targets: [{ format: "rgba8unorm-srgb" }] } });
  const exportPipeline = render(module(sources.export), "export_reflections", [exportLayout]);
  const basePipeline = render(module(sources.base), "fs_reflections", [inputLayout]);
  const outputs = Array.from({ length: 3 }, () => keep(device.createTexture({ size: [width, height], format: "rgba8unorm-srgb", usage: T.RENDER_ATTACHMENT | T.COPY_SRC })));
  const query = device.features.has("timestamp-query") ? keep(device.createQuerySet({ type: "timestamp", count: 6 })) : null;
  need(query, "GPU timestamps are required for the local partition ablation");
  const queryBuffer = buffer(48, U.QUERY_RESOLVE | U.COPY_SRC);
  const stamps = i => ({ querySet: query, beginningOfPassWriteIndex: i, endOfPassWriteIndex: i + 1 });
  function draw(encoder, pipeline, bindings, target, offsets = [], timestampWrites) {
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }], timestampWrites });
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindings, offsets); pass.draw(3); pass.end();
  }
  async function run(variant) {
    const { work, buffers, workGroup, traceGroup, exportGroup, trace, bins } = variants[variant], encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({ timestampWrites: stamps(0) });
    pass.setBindGroup(0, inputs); pass.setBindGroup(1, workGroup, [0]); pass.setPipeline(work.classify); pass.dispatchWorkgroups(tx, ty);
    pass.setPipeline(work.scan_up);
    levels.forEach((level, i) => { pass.setBindGroup(1, workGroup, [(i + 1) * stride]); pass.dispatchWorkgroups(Math.ceil(level.length / 64)); });
    pass.setPipeline(work.scan_down);
    for (let i = levels.length - 2; i >= 0; i--) { pass.setBindGroup(1, workGroup, [(i + 1) * stride]); pass.dispatchWorkgroups(Math.ceil(levels[i].length / 64)); }
    pass.setBindGroup(1, workGroup, [0]); pass.setPipeline(work.finalize); pass.dispatchWorkgroups(1); pass.setPipeline(work.scatter); pass.dispatchWorkgroups(tx, ty); pass.end();
    const tracePass = encoder.beginComputePass({ timestampWrites: stamps(2) }); tracePass.setPipeline(trace); tracePass.setBindGroup(0, inputs);
    for (let bin = 0; bin < bins; bin++) { tracePass.setBindGroup(1, traceGroup, [bins === 1 ? 0 : (levels.length + 1 + bin) * stride]); tracePass.dispatchWorkgroupsIndirect(buffers[5], bin * 12); }
    tracePass.end(); draw(encoder, exportPipeline, exportGroup, outputs[variant], [0], stamps(4));
    encoder.resolveQuerySet(query, 0, 6, queryBuffer, 0); device.queue.submit([encoder.finish()]);
    const times = new BigUint64Array(await read(queryBuffer));
    return [Number(times[1] - times[0]), Number(times[3] - times[2]), Number(times[5] - times[4])];
  }
  const reports = [];
  for (let phase = 0; phase < 4; phase++) {
    uniform[51] = phase; device.queue.writeBuffer(capture.entries.find(e => e.binding === 0).resource.buffer, 0, uniform);
    const [ox, oy] = [[0, 0], [1, 1], [1, 0], [0, 1]][phase], expected = [[], [], [], []], spatial = [];
    for (let tile = 0; tile < tx * ty; tile++) for (let lane = 0; lane < 64; lane++) {
      const x = tile % tx * 8 + lane % 8, y = Math.floor(tile / tx) * 8 + Math.floor(lane / 8);
      if (x >= width || y >= height) continue;
      const px = Math.min(fullWidth - 1, 2 * x + ox), py = Math.min(fullHeight - 1, 2 * y + oy);
      if (Math.fround(glowValues[py * glow.pitch + px * 2 + 1] / 255) < 0.01 || depthValues[py * depth.pitch / 4 + px] <= 0) continue;
      const roughness = Math.fround(((surfaces[py * surface.pitch / 4 + px] >>> 20) & 1023) / 1023), bin = Number(roughness >= 0.125) + Number(roughness >= 0.25) + Number(roughness >= 0.375);
      expected[bin].push(x, y); spatial.push(x, y);
    }
    const encoder = device.createCommandEncoder(); draw(encoder, basePipeline, inputs, outputs[2]); device.queue.submit([encoder.finish()]);
    for (let variant = 0; variant < 2; variant++) {
      await run(variant);
      const rays = new Uint32Array(await read(variants[variant].buffers[2])), metadata = new Uint32Array(await read(variants[variant].buffers[4]));
      const ordered = variant === 1 ? expected.flat() : spatial, counts = variant === 1 ? expected.map(bin => bin.length / 2) : [spatial.length / 2];
      need(ordered.every((value, i) => value === rays[i]) && counts.every((value, i) => value === metadata[i]), `captured Metro phase ${phase} variant ${variant}: exact original rays`);
    }
    const images = await Promise.all(outputs.map(target => textureBytes(target, 4)));
    const reference = new Uint8Array(images[2].bytes);
    for (let variant = 0; variant < 2; variant++) need(new Uint8Array(images[variant].bytes).every((v, i) => v === reference[i]), `captured Metro phase ${phase} variant ${variant}: stored bytes equal unchanged main`);
    const times = [[], []];
    for (let pair = 0; pair < 8; pair++) for (const variant of pair % 2 ? [1, 0] : [0, 1]) times[variant].push(await run(variant));
    reports.push({ phase, active: spatial.length / 2, bins: expected.map(bin => bin.length / 2), capacity: capture.capacity, adapter: state.adapter, full: [fullWidth, fullHeight],
      spatial_queue_ns: times[0], four_bins_ns: times[1], variant_capacity: variants.map(v => v.capacity), sibling: sources.sibling.commit, columns: ["classify_scan_scatter_clear", "indirect_trace", "f32_export"],
      note: "Local context on identical captured inputs: production scalar spatial queue versus the pinned four-bin implementation, with separate exact allocations. No sibling branch is modified." });
  }
  for (const object of owned) object.destroy();
  return reports;
}
