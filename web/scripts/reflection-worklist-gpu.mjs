/** Real GPU queue and trace checks, using authored shader modules supplied by the smoke. */
export async function checkReflectionGpu({ sources, plans, layouts }) {
  const need = (ok, message) => { if (!ok) throw new Error(message); };
  let device;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    need(adapter, "a WebGPU adapter is required");
    device = await adapter.requestDevice();
    const errors = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    const U = GPUBufferUsage, T = GPUTextureUsage, C = GPUShaderStage.COMPUTE, F = GPUShaderStage.FRAGMENT;
    const buffer = (size, usage, data) => {
      const result = device.createBuffer({ size, usage, mappedAtCreation: !!data });
      if (data) { new Uint8Array(result.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); result.unmap(); }
      return result;
    };
    const module = async (label, code) => {
      const result = device.createShaderModule({ label, code });
      const info = await result.getCompilationInfo();
      need(!info.messages.some(m => m.type === "error"), `${label}: ${JSON.stringify(info.messages.map(m => [m.type, m.lineNum, m.message]))}`);
      return result;
    };
    const workModule = await module("work", sources.work);
    const traceModule = await module("trace", sources.trace);
    const exportModule = await module("export", sources.export);
    const baseModule = await module("unchanged main", sources.base);
    const fallbackModule = await module("extracted fallback", sources.post);
    const depthModule = await module("nearest pyramid", sources.depth);
    const baseStateModule = await module("unchanged-main state diagnostic", sources.baseState);
    const fallbackStateModule = await module("extracted-fallback state diagnostic", sources.fallbackState);
    const traceStateModule = await module("compute state diagnostic", sources.traceState);
    const uniformEntry = (binding, visibility, dynamic = false, minBindingSize = 0) => ({ binding, visibility, buffer: { type: "uniform", hasDynamicOffset: dynamic, minBindingSize } });
    const storageEntry = (binding, visibility, type = "storage") => ({ binding, visibility, buffer: { type } });
    const textureEntry = (binding, visibility, sampleType = "float") => ({ binding, visibility, texture: { sampleType } });
    const inputLayout = device.createBindGroupLayout({ entries: layouts.REFLECTION_INPUTS });
    const fragmentLayout = device.createBindGroupLayout({ entries: layouts.POST.filter(entry => [0,1,2,3,7,10,13,14].includes(entry.binding)) });
    const workLayout = device.createBindGroupLayout({ entries: layouts.REFLECTION_WORK });
    const traceLayout = device.createBindGroupLayout({ entries: layouts.REFLECTION_TRACE });
    const exportLayout = device.createBindGroupLayout({ entries: layouts.REFLECTION_EXPORT });
    const workPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [inputLayout, workLayout] });
    const tracePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [inputLayout, traceLayout] });
    const compute = (module, entryPoint, layout = "auto") => device.createComputePipeline({ layout, compute: { module, entryPoint } });
    const work = Object.fromEntries(["classify", "scan_up", "scan_down", "finalize", "scatter"].map(entry => [entry, compute(workModule, entry, workPipelineLayout)]));
    const trace = compute(traceModule, "trace", tracePipelineLayout);
    const traceStateLayout = device.createBindGroupLayout({ entries: [uniformEntry(0, C, true, 48), storageEntry(1, C, "read-only-storage"), storageEntry(2, C, "read-only-storage"), storageEntry(3, C), storageEntry(4, C)] });
    const traceState = compute(traceStateModule, "trace", device.createPipelineLayout({ bindGroupLayouts: [inputLayout, traceStateLayout] }));
    const render = (module, entryPoint, layout, format = "rgba8unorm-srgb") => device.createRenderPipeline({
      layout, vertex: { module, entryPoint: "vs_fullscreen" }, fragment: { module, entryPoint, targets: [{ format }] }, primitive: { topology: "triangle-list" },
    });
    const exportPipeline = render(exportModule, "export_reflections", device.createPipelineLayout({ bindGroupLayouts: [exportLayout] }));
    const fragmentPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [fragmentLayout] });
    const basePipeline = render(baseModule, "fs_reflections", fragmentPipelineLayout);
    const fallbackPipeline = render(fallbackModule, "fs_reflections", fragmentPipelineLayout);
    const baseRawPipeline = render(baseModule, "fs_reflections", fragmentPipelineLayout, "rgba32float");
    const baseStatePipeline = device.createRenderPipeline({ layout: fragmentPipelineLayout,
      vertex: { module: baseStateModule, entryPoint: "vs_fullscreen" },
      fragment: { module: baseStateModule, entryPoint: "diagnostic_fragment", targets: [{ format: "rgba32float" }, { format: "rgba32float" }] } });
    const fallbackStatePipeline = device.createRenderPipeline({ layout: fragmentPipelineLayout,
      vertex: { module: fallbackStateModule, entryPoint: "vs_fullscreen" },
      fragment: { module: fallbackStateModule, entryPoint: "diagnostic_fragment", targets: [{ format: "rgba32float" }, { format: "rgba32float" }] } });
    const pyramidFirst = compute(depthModule, "pyramid_first_nearest");
    const pyramidDown = compute(depthModule, "pyramid_down_nearest");
    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear" });
    const group = (layout, entries) => device.createBindGroup({ layout, entries: entries.map(([binding, resource]) => ({ binding, resource })) });
    const readBuffers = async (encoder, buffers) => {
      const copies = buffers.map(source => { const copy = buffer(source.size, U.COPY_DST | U.MAP_READ); encoder.copyBufferToBuffer(source, 0, copy, 0, source.size); return copy; });
      device.queue.submit([encoder.finish()]);
      await Promise.all(copies.map(copy => copy.mapAsync(GPUMapMode.READ)));
      return copies.map(copy => { const bytes = copy.getMappedRange().slice(0); copy.unmap(); copy.destroy(); return bytes; });
    };
    const draw = (encoder, pipeline, bindings, target, offsets = []) => {
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: target.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }] });
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindings, offsets); pass.draw(3); pass.end();
    };
    const texture = (width, height, format, usage = T.TEXTURE_BINDING | T.COPY_DST, mipLevelCount = 1) => device.createTexture({ size: [width, height], format, usage, mipLevelCount });
    const upload = (target, data, width, height, bytes, mipLevel = 0) => device.queue.writeTexture({ texture: target, mipLevel }, data, { bytesPerRow: width * bytes }, [width, height]);
    // WebGPU does not allow writeTexture into depth32float. Populate it through
    // a real fragment depth output from the exact uploaded f32 values.
    const depthUploadModule = await module("depth upload", sources.fullscreen + `
      @group(0) @binding(0) var values: texture_2d<f32>;
      @fragment fn depth_value(@builtin(position) p: vec4<f32>) -> @builtin(frag_depth) f32 {
        return textureLoad(values, vec2<i32>(p.xy), 0).r;
      }`);
    const depthUploadPipeline = device.createRenderPipeline({ layout: "auto", vertex: { module: depthUploadModule, entryPoint: "vs_fullscreen" },
      fragment: { module: depthUploadModule, entryPoint: "depth_value", targets: [] }, depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "always" } });
    const measured = { cases: 0, active: 0, bytes: [], phases: [0, 0, 0, 0], hierarchy: [],
      adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description },
      raw_differences: 0, raw_max_error: 0, state_rays: 0, indeterminate: 0, steps: {}, ray_signs: [0, 0, 0], export_boundary_samples: 0 };
    for (const plan of plans) {
      const [fw, fh] = plan.full, [width, height, tx, ty] = plan.size;
      const owned = [];
      const keep = object => { owned.push(object); return object; };
      const buffers = plan.bytes.map((bytes, i) => keep(buffer(bytes, U.COPY_SRC | U.COPY_DST | (i === 6 ? U.UNIFORM : U.STORAGE) | (i === 5 ? U.INDIRECT : 0))));
      measured.bytes.push({ size: plan.full, bytes: buffers.reduce((sum, b) => sum + b.size, 0) });
      need(buffers.reduce((sum, b) => sum + b.size, 0) === plan.capacity, "actual allocations equal host capacity");
      measured.hierarchy.push(plan.levels.map(l => l.length));
      const record = (index, scan) => device.queue.writeBuffer(buffers[6], index * plan.stride, new Uint32Array([...plan.size, ...scan, plan.pixels, plan.row_log2, 0, 0]));
      record(0, [0, 0, 0, 0]);
      plan.levels.forEach((level, i) => record(i + 1, [level.offset, level.length, plan.levels[i + 1]?.offset ?? 0xffffffff, level.grid[0]]));
      const params = { buffer: buffers[6], size: 48 };
      const workGroup = group(workLayout, [[0, params], ...[0, 1, 2, 3, 4, 5].map((b, i) => [i + 1, { buffer: buffers[b] }])]);
      const traceGroup = group(traceLayout, [[0, params], [1, { buffer: buffers[2] }], [2, { buffer: buffers[4] }], [3, { buffer: buffers[3] }]]);
      const diagnosticBuffer = keep(buffer(plan.pixels * 32, U.STORAGE | U.COPY_SRC));
      const traceStateGroup = group(traceStateLayout, [[0, params], [1, { buffer: buffers[2] }], [2, { buffer: buffers[4] }], [3, { buffer: buffers[3] }], [4, { buffer: diagnosticBuffer }]]);
      const exportGroup = group(exportLayout, [[0, params], [1, { buffer: buffers[3] }]]);
      const post = keep(buffer(272, U.UNIFORM | U.COPY_DST));
      const depth = keep(texture(fw, fh, "depth32float", T.TEXTURE_BINDING | T.RENDER_ATTACHMENT));
      const depthDataTexture = keep(texture(fw, fh, "r32float"));
      const depthUploadGroup = group(depthUploadPipeline.getBindGroupLayout(0), [[0, depthDataTexture.createView()]]);
      const glow = keep(texture(fw, fh, "rg8unorm"));
      const surface = keep(texture(fw, fh, "rgb10a2unorm"));
      const scene = keep(texture(fw, fh, "rgba8unorm-srgb"));
      const levels = Math.floor(Math.log2(Math.max(width, height))) + 1;
      const sceneLevels = keep(texture(width, height, "rgba8unorm-srgb", T.TEXTURE_BINDING | T.COPY_DST, levels));
      const pyramidWidth = 2 ** Math.ceil(Math.log2(Math.max(1, Math.ceil(fw / 2))));
      const pyramidHeight = 2 ** Math.ceil(Math.log2(Math.max(1, Math.ceil(fh / 2))));
      const pyramidLevels = Math.floor(Math.log2(Math.max(pyramidWidth, pyramidHeight))) + 1;
      const pyramid = keep(texture(pyramidWidth, pyramidHeight, "r32float", T.TEXTURE_BINDING | T.STORAGE_BINDING, pyramidLevels));
      const pyramidPasses = [];
      let written = [fw, fh];
      for (let i = 0; i < pyramidLevels; i++) {
        const size = written.map(v => Math.max(1, Math.ceil(v / 2)));
        const extent = keep(buffer(16, U.UNIFORM, new Uint32Array([...size, ...written.map(v => v - 1)])));
        const pipeline = i === 0 ? pyramidFirst : pyramidDown;
        const source = i === 0 ? depth.createView() : pyramid.createView({ baseMipLevel: i - 1, mipLevelCount: 1 });
        const bindings = group(pipeline.getBindGroupLayout(0), [[i === 0 ? 0 : 1, source], [2, pyramid.createView({ baseMipLevel: i, mipLevelCount: 1 })], [3, { buffer: extent }]]);
        pyramidPasses.push({ pipeline, bindings, size }); written = size;
      }
      const inputs = [[0, { buffer: post }], [1, depth.createView()], [2, scene.createView()], [3, glow.createView()], [7, sampler], [10, pyramid.createView()], [13, surface.createView()], [14, sceneLevels.createView()]];
      const inputGroup = group(inputLayout, inputs), fragmentGroup = group(fragmentLayout, inputs);
      const outputs = Array.from({ length: 3 }, () => keep(texture(width, height, "rgba8unorm-srgb", T.RENDER_ATTACHMENT | T.COPY_SRC)));
      const pitch = Math.ceil(width * 4 / 256) * 256;
      const outputBuffers = outputs.map(() => keep(buffer(pitch * height, U.COPY_DST | U.COPY_SRC)));
      const stateTargets = Array.from({ length: 5 }, () => keep(texture(width, height, "rgba32float", T.RENDER_ATTACHMENT | T.COPY_SRC)));
      const statePitch = Math.ceil(width * 16 / 256) * 256;
      const stateCopies = stateTargets.map(() => keep(buffer(statePitch * height, U.COPY_DST | U.COPY_SRC)));
      const depthData = new Float32Array(fw * fh), glowData = new Uint8Array(fw * fh * 2), surfaceData = new Uint32Array(fw * fh), sceneData = new Uint8Array(fw * fh * 4);
      for (let y = 0; y < fh; y++) for (let x = 0; x < fw; x++) {
        const index = y * fw + x;
        depthData[index] = Math.fround(x % 13 === 0 ? 0 : (0.2 + (y % 11) * 0.002 + (x % 9 === 0 ? 0.12 : 0)));
        sceneData.set([(x * 17 + y * 5) % 256, (x * 3 + y * 11) % 256, (x + y * 7) % 256, 255], index * 4);
      }
      upload(depthDataTexture, depthData, fw, fh, 4);
      upload(scene, sceneData, fw, fh, 4);
      for (let i = 0; i < levels; i++) {
        const w = Math.max(1, width >> i), h = Math.max(1, height >> i), data = new Uint8Array(w * h * 4);
        for (let p = 0; p < w * h; p++) data.set([63 + i * 9, 123 - i * 5, 197 - i * 3, 255], p * 4);
        upload(sceneLevels, data, w, h, 4, i);
      }
      {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: depth.createView(), depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "store" } });
        pass.setPipeline(depthUploadPipeline); pass.setBindGroup(0, depthUploadGroup); pass.draw(3); pass.end();
        for (const { pipeline, bindings, size } of pyramidPasses) {
          const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(Math.ceil(size[0] / 8), Math.ceil(size[1] / 8)); pass.end();
        }
        device.queue.submit([encoder.finish()]);
      }
      const phases = [[0, 0], [1, 1], [1, 0], [0, 1]];
      const patterns = ["all", "rough0", "rough1", "rough2", "rough3", "zero", "checker", "tiles", "sparse", "threshold", "exhaustion", "stationary", "tiny", "zero"];
      for (let phase = 0; phase < 4; phase++) for (const pattern of patterns) {
        const label = `${fw}x${fh} phase ${phase} ${pattern}`;
        for (let y = 0; y < fh; y++) for (let x = 0; x < fw; x++) {
          const i = y * fw + x;
          depthData[i] = pattern === "exhaustion" ? (y % 2 ? 0.9 : 0.2)
            : Math.fround(pattern === "sparse" && x % 13 === 0 ? 0 : (0.2 + (y % 11) * 0.002 + (x % 9 === 0 ? 0.12 : 0)));
          const bin = pattern.startsWith("rough") ? Number(pattern[5]) : (Math.floor(x / 2) + Math.floor(y / 2)) % 4;
          const roughness = pattern === "threshold" ? [127, 128, 255, 256, 383, 384][i % 6] : [64, 192, 320, 800][bin];
          const strength = pattern === "zero" || (pattern === "checker" && (Math.floor(x / 2) + Math.floor(y / 2)) % 2) || (pattern === "tiles" && (Math.floor(x / 16) + Math.floor(y / 16)) % 2) ? 0 : pattern === "sparse" && i % 5 ? 0 : pattern === "threshold" ? [2, 3][i % 2] : 191;
          glowData[i * 2 + 1] = strength;
          surfaceData[i] = (650 + (x % 5) * 20) | ((460 + (y % 7) * 20) << 10) | (roughness << 20) | (3 << 30);
          if (pattern === "exhaustion") surfaceData[i] = 710 | (512 << 10) | (roughness << 20) | (3 << 30);
        }
        upload(glow, glowData, fw, fh, 2); upload(surface, surfaceData, fw, fh, 4);
        upload(depthDataTexture, depthData, fw, fh, 4);
        const uniform = new Float32Array(68);
        [0, 5, 10, 15, 16, 21, 26, 31].forEach(i => uniform[i] = 1);
        uniform.set([0, 0, 3, fh * 0.5], 32); uniform.set([fw, fh, 1 / fw, 1 / fh], 36);
        uniform.set([pattern === "stationary" ? 1e-12 : pattern === "tiny" ? 1e-6 : 4, 1, 1, phase], 48); device.queue.writeBuffer(post, 0, uniform);
        const tileCounts = new Uint32Array(tx * ty), expected = [];
        const tags = new Uint32Array(plan.pixels).fill(0xffffffff);
        for (let tile = 0; tile < tx * ty; tile++) for (let lane = 0; lane < 64; lane++) {
          const x = (tile % tx) * 8 + lane % 8, y = Math.floor(tile / tx) * 8 + Math.floor(lane / 8);
          if (x >= width || y >= height) continue;
          const px = Math.min(x * 2 + phases[phase][0], fw - 1), py = Math.min(y * 2 + phases[phase][1], fh - 1), i = py * fw + px;
          if (Math.fround(glowData[i * 2 + 1] / 255) < 0.01 || depthData[i] <= 0) continue;
          tags[y * width + x] = tileCounts[tile]++; expected.push(x, y);
        }
        const total = expected.length / 2;
        const encoder = device.createCommandEncoder();
        {
          const pass = encoder.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: depth.createView(), depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "store" } });
          pass.setPipeline(depthUploadPipeline); pass.setBindGroup(0, depthUploadGroup); pass.draw(3); pass.end();
          for (const { pipeline, bindings, size } of pyramidPasses) {
            const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(Math.ceil(size[0] / 8), Math.ceil(size[1] / 8)); pass.end();
          }
        }
        const pass = encoder.beginComputePass(); pass.setBindGroup(0, inputGroup); pass.setBindGroup(1, workGroup, [0]);
        pass.setPipeline(work.classify); pass.dispatchWorkgroups(tx, ty);
        pass.setPipeline(work.scan_up);
        plan.levels.forEach((level, i) => { pass.setBindGroup(1, workGroup, [(i + 1) * plan.stride]); pass.dispatchWorkgroups(...level.grid); });
        pass.setPipeline(work.scan_down);
        for (let i = plan.levels.length - 2; i >= 0; i--) { pass.setBindGroup(1, workGroup, [(i + 1) * plan.stride]); pass.dispatchWorkgroups(...plan.levels[i].grid); }
        pass.setBindGroup(1, workGroup, [0]); pass.setPipeline(work.finalize); pass.dispatchWorkgroups(1);
        pass.setPipeline(work.scatter); pass.dispatchWorkgroups(tx, ty); pass.end();
        const tracing = encoder.beginComputePass(); tracing.setPipeline(trace); tracing.setBindGroup(0, inputGroup);
        tracing.setBindGroup(1, traceGroup, [0]); tracing.dispatchWorkgroupsIndirect(buffers[5], 0);
        tracing.end();
        draw(encoder, exportPipeline, exportGroup, outputs[0], [0]);
        draw(encoder, basePipeline, fragmentGroup, outputs[1]);
        draw(encoder, fallbackPipeline, fragmentGroup, outputs[2]);
        outputs.forEach((output, i) => encoder.copyTextureToBuffer({ texture: output }, { buffer: outputBuffers[i], bytesPerRow: pitch }, [width, height]));
        const read = await readBuffers(encoder, [...buffers.slice(0, 6), ...outputBuffers]);
        const actualTags = new Uint32Array(read[0]), prefixes = new Uint32Array(read[1]), rays = new Uint32Array(read[2]), raw = new Float32Array(read[3]), metadata = new Uint32Array(read[4]), indirect = new Uint32Array(read[5]);
        need(actualTags.every((v, i) => v === tags[i]), `${label}: tags/local ranks`);
        const expectedRays = expected;
        need(expectedRays.every((v, i) => v === rays[i]), `${label}: exact stable queue, no lost or duplicate coordinates`);
        need(metadata[0] === total && metadata[2] === 0 && metadata[3] === 0, `${label}: exact count and reserved metadata`);
        let counts = tileCounts;
        for (const level of plan.levels) {
          let running = 0;
          const parent = [];
          for (let i = 0; i < counts.length; i++) {
            need(prefixes[level.offset + i] === running, `${label}: global prefix level ${level.offset} index ${i}`);
            running += counts[i];
            if (i % 64 === 0) parent.push(0);
            parent[parent.length - 1] += counts[i];
          }
          counts = parent;
        }
        const groups = Math.ceil(total / 64), row = 2 ** plan.row_log2, dispatchWidth = Math.min(groups, row), dispatchHeight = Math.max(1, Math.ceil(groups / row));
        need(indirect[0] === dispatchWidth && indirect[1] === dispatchHeight && indirect[2] === 1 && metadata[1] === dispatchWidth, `${label}: tightly packed indirect triple`);
        for (let i = 0; i < tags.length; i++) if (tags[i] === 0xffffffff) need(raw.slice(i * 4, i * 4 + 4).every(v => v === 0), `${label}: inactive raw must be zero after preceding frame`);
        const images = read.slice(6).map(bytes => new Uint8Array(bytes));
        for (let i = 0; i < images[0].length; i++) {
          need(images[1][i] === images[2][i], `${label}: extracted fallback byte ${i}: ${images[2][i]} vs unchanged-main ${images[1][i]}`);
          need(images[0][i] === images[1][i], `${label}: compute/export byte ${i}: ${images[0][i]} vs unchanged-main ${images[1][i]}`);
        }
        if (["all", "threshold", "exhaustion", "stationary", "tiny"].includes(pattern)) {
          const encoder = device.createCommandEncoder();
          draw(encoder, baseRawPipeline, fragmentGroup, stateTargets[0]);
          const statePass = encoder.beginRenderPass({ colorAttachments: stateTargets.slice(1, 3).map(target => ({ view: target.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" })) });
          statePass.setPipeline(baseStatePipeline); statePass.setBindGroup(0, fragmentGroup); statePass.draw(3); statePass.end();
          const fallbackPass = encoder.beginRenderPass({ colorAttachments: stateTargets.slice(3).map(target => ({ view: target.createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" })) });
          fallbackPass.setPipeline(fallbackStatePipeline); fallbackPass.setBindGroup(0, fragmentGroup); fallbackPass.draw(3); fallbackPass.end();
          const tracePass = encoder.beginComputePass(); tracePass.setPipeline(traceState); tracePass.setBindGroup(0, inputGroup);
          tracePass.setBindGroup(1, traceStateGroup, [0]); tracePass.dispatchWorkgroupsIndirect(buffers[5], 0);
          tracePass.end();
          stateTargets.forEach((target, i) => encoder.copyTextureToBuffer({ texture: target }, { buffer: stateCopies[i], bytesPerRow: statePitch }, [width, height]));
          const diagnostic = await readBuffers(encoder, [diagnosticBuffer, ...stateCopies]);
          const state = new Float32Array(diagnostic[0]), reference = diagnostic.slice(1).map(bytes => new Float32Array(bytes));
          for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
            const i = y * width + x, ref = y * statePitch / 4 + x * 4;
            if (tags[i] === 0xffffffff) continue;
            const values = [...reference[1].slice(ref, ref + 4), ...reference[2].slice(ref, ref + 4)];
            if (!values.every(Number.isFinite)) { measured.indeterminate++; continue; }
            for (let k = 0; k < 8; k++) need(Object.is(state[i * 8 + k], values[k]), `${label}: trace state at ${x},${y} field ${k}: ${state[i * 8 + k]} vs ${values[k]}`);
            const fallback = [...reference[3].slice(ref, ref + 4), ...reference[4].slice(ref, ref + 4)];
            for (let k = 0; k < 8; k++) need(Object.is(fallback[k], values[k]), `${label}: extracted fallback state at ${x},${y} field ${k}`);
            measured.state_rays++;
            measured.steps[values[1]] = (measured.steps[values[1]] ?? 0) + 1;
            measured.ray_signs[Math.sign(values[7]) + 1]++;
            for (let k = 0; k < 4; k++) if (!Object.is(raw[i * 4 + k], reference[0][ref + k])) {
              measured.raw_differences++;
              measured.raw_max_error = Math.max(measured.raw_max_error, Math.abs(raw[i * 4 + k] - reference[0][ref + k]));
            }
          }
        }
        need(errors.length === 0, `${label}: GPU errors ${errors.join("; ")}`);
        measured.cases++; measured.active += total; measured.phases[phase]++;
      }
      if (fw === 128 && fh === 128) {
        // Independent constant-output fragment reference around sRGB and alpha
        // quantization boundaries. A half-float intermediate cannot pass this.
        const colors = Array.from({ length: 32 }, (_, i) => {
          const encoded = (32 + i * 5 + 0.5) / 255;
          const linear = ((encoded + 0.055) / 1.055) ** 2.4;
          const d = 2 ** -24;
          return [Math.fround(linear - d), Math.fround(linear), Math.fround(linear + d), Math.fround((i * 7 + 0.5) / 255)];
        });
        const values = new Float32Array(plan.pixels * 4);
        for (let i = 0; i < plan.pixels; i++) values.set(colors[i & 31], i * 4);
        device.queue.writeBuffer(buffers[3], 0, values);
        const referenceModule = await module("sRGB quantization boundary reference", sources.fullscreen + `
          const colors = array<vec4<f32>,32>(${colors.map(color => `vec4<f32>(${color.map(v => `${v}f`).join(",")})`).join(",")});
          @fragment fn boundary(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
            return colors[(u32(p.y) * ${width}u + u32(p.x)) & 31u];
          }`);
        const referencePipeline = render(referenceModule, "boundary", "auto");
        const encoder = device.createCommandEncoder();
        draw(encoder, exportPipeline, exportGroup, outputs[0], [0]);
        const pass = encoder.beginRenderPass({ colorAttachments: [{ view: outputs[1].createView(), clearValue: [0, 0, 0, 0], loadOp: "clear", storeOp: "store" }] });
        pass.setPipeline(referencePipeline); pass.draw(3); pass.end();
        for (let i = 0; i < 2; i++) encoder.copyTextureToBuffer({ texture: outputs[i] }, { buffer: outputBuffers[i], bytesPerRow: pitch }, [width, height]);
        const images = (await readBuffers(encoder, outputBuffers.slice(0, 2))).map(bytes => new Uint8Array(bytes));
        need(images[0].every((v, i) => v === images[1][i]), "full-f32 export preserves hardware sRGB and alpha quantization boundaries");
        measured.export_boundary_samples += plan.pixels;
      }
      for (const object of owned) object.destroy();
    }
    await device.queue.onSubmittedWorkDone();
    need(errors.length === 0, `GPU errors: ${errors.join("; ")}`);
    need(measured.steps[95] > 0 && measured.steps[96] > 0, "95/96-step exhaustion was actually exercised");
    need(measured.ray_signs[0] > 0 && measured.ray_signs[2] > 0, "both ray-depth signs were actually exercised");
    return { ok: true, ...measured };
  } catch (error) { return { ok: false, error: error.stack ?? String(error) }; }
  finally { device?.destroy(); }
}
