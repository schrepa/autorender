/** Browser-side spatial diagnostics. Gaussian comparisons are artistic references. */
export async function glassReferences() {
  const observed = window.__roughGlass, device = observed.device;
  const check = (ok, message) => { if (!ok) throw new Error(`rough-glass assertion: ${message}`); };
  check(device && observed.post && observed.mesh, "production post and glass WGSL registered");
  device.pushErrorScope("validation");
  const width = 128, height = 128, lower = 6;
  const sampler = device.createSampler({ minFilter: "linear", magFilter: "linear", mipmapFilter: "linear" });
  const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
  const texture = device.createTexture({ size: [width, height], mipLevelCount: lower + 1, format: "rgba8unorm-srgb", usage });
  const output = device.createTexture({ size: [width, height], format: "rgba8unorm-srgb", usage });
  const views = Array.from({ length: lower + 1 }, (_, baseMipLevel) => texture.createView({ baseMipLevel, mipLevelCount: 1 }));
  const module = device.createShaderModule({ code: observed.post, label: "rough-glass/reference-reduction" });
  const reduction = device.createRenderPipeline({ layout: "auto",
    vertex: { module, entryPoint: "vs_fullscreen" }, fragment: { module, entryPoint: "fs_refraction_prefilter", targets: [{ format: "rgba8unorm-srgb" }] } });
  const reductions = views.slice(0, -1).map(view => device.createBindGroup({ layout: reduction.getBindGroupLayout(0),
    entries: [{ binding: 4, resource: view }, { binding: 7, resource: sampler }] }));
  // Copy the actual production lookup function, with a test-only fullscreen
  // entry. The renderer's full pane/composition path is checked separately.
  const start = observed.mesh.indexOf("fn refraction_level("), open = observed.mesh.indexOf("{", start);
  let end = open + 1, nesting = 1;
  for (; nesting && end < observed.mesh.length; end++) nesting += (observed.mesh[end] === "{") - (observed.mesh[end] === "}");
  check(start >= 0 && nesting === 0, "extract the complete production lookup");
  const lookup = device.createShaderModule({ label: "rough-glass/reference-lookup", code: `
@group(0) @binding(0) var refraction_source: texture_2d<f32>;
@group(0) @binding(1) var refraction_sampler: sampler;
@group(0) @binding(2) var<uniform> controls: vec4<f32>;
${observed.mesh.slice(start, end)}
@vertex fn fixture_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(f32((index << 1u) & 2u) * 2.0 - 1.0, f32(index & 2u) * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fixture_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = (position.xy + controls.yy) / vec2<f32>(textureDimensions(refraction_source));
  let sharp = textureSampleLevel(refraction_source, refraction_sampler, uv, 0.0).rgb;
  let blurred = textureSampleLevel(refraction_source, refraction_sampler, uv, refraction_level(controls.x)).rgb;
  // One fixed pane: the engine's single-pane weighted composite, with the
  // same fixed shaded color, albedo, through share and coverage in all arms.
  let color = 0.75 * (vec3<f32>(0.03, 0.02, 0.01) + blurred * vec3<f32>(0.9, 0.95, 1.0) * 0.65) + 0.25 * sharp;
  return vec4<f32>(color, 1.0);
}` });
  const pipeline = device.createRenderPipeline({ layout: "auto", vertex: { module: lookup, entryPoint: "fixture_vertex" },
    fragment: { module: lookup, entryPoint: "fixture_fragment", targets: [{ format: "rgba8unorm-srgb" }] } });
  const controls = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: texture.createView() }, { binding: 1, resource: sampler }, { binding: 2, resource: { buffer: controls } },
  ] });
  const readback = device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const decode = value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  const encode = value => Math.round(255 * (value <= 0.0031308 ? 12.92 * value : 1.055 * Math.max(value, 0) ** (1 / 2.4) - 0.055));
  const clamp = (value, limit) => Math.max(0, Math.min(limit - 1, value));
  const sample = (pixels, x, y, channel) => {
    const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
    const at = (a, b) => pixels[(clamp(b, height) * width + clamp(a, width)) * 3 + channel];
    return (at(ix, iy) * (1 - fx) + at(ix + 1, iy) * fx) * (1 - fy)
      + (at(ix, iy + 1) * (1 - fx) + at(ix + 1, iy + 1) * fx) * fy;
  };
  const taps = [[1, 0], [0.3535534, 0.3535534], [0, 1], [-0.3535534, 0.3535534],
    [-1, 0], [-0.3535534, -0.3535534], [0, -1], [0.3535534, -0.3535534]];
  const compose = (value, sharp, c) => 0.75 * ([0.03, 0.02, 0.01][c] + value * [0.9, 0.95, 1][c] * 0.65) + 0.25 * sharp;
  const png = bytes => {
    const canvas = document.createElement("canvas"); canvas.width = width; canvas.height = height;
    canvas.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(bytes), width, height), 0, 0);
    return canvas.toDataURL("image/png").split(",")[1];
  };
  const rows = [], images = [];
  try {
    for (const pattern of ["impulse", "horizontal", "diagonal", "checkerboard", "depth-edge"]) {
      const bytes = new Uint8Array(width * height * 4), pixels = new Float64Array(width * height * 3);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const bright = pattern === "impulse" ? (x === 64 && y === 64)
          : pattern === "horizontal" ? y % 8 < 4 : pattern === "diagonal" ? (x + y) % 8 < 4
          : pattern === "depth-edge" ? x < 64 : (Math.floor(x / 4) + Math.floor(y / 4)) % 2;
        for (let c = 0; c < 3; c++) {
          const at = y * width + x, value = encode((bright ? 0.9 : 0.02) * [1, 0.7, 0.4][c]);
          bytes[at * 4 + c] = value; pixels[at * 3 + c] = decode(value / 255);
        }
        bytes[(y * width + x) * 4 + 3] = 255;
      }
      device.queue.writeTexture({ texture }, bytes, { bytesPerRow: width * 4 }, [width, height]);
      const down = device.createCommandEncoder();
      for (let level = 1; level <= lower; level++) {
        const pass = down.beginRenderPass({ colorAttachments: [{ view: views[level], loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
        pass.setPipeline(reduction); pass.setBindGroup(0, reductions[level - 1]); pass.draw(3); pass.end();
      }
      device.queue.submit([down.finish()]);
      for (const radius of [2, 4.2, 8, 12, 64]) {
        const variance = (5 / 18) * Math.max(radius * radius - 0.25, 0), sigma = Math.sqrt(variance), reach = Math.ceil(4 * sigma);
        const kernel = Array.from({ length: reach * 2 + 1 }, (_, index) => Math.exp(-((index - reach) ** 2) / (2 * variance)));
        const weight = kernel.reduce((a, b) => a + b); kernel.forEach((value, index) => { kernel[index] = value / weight; });
        const horizontal = new Float64Array(pixels.length), gaussian = new Float64Array(pixels.length);
        for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < 3; c++) {
          let sum = 0; for (let k = -reach; k <= reach; k++) sum += kernel[k + reach] * pixels[(y * width + clamp(x + k, width)) * 3 + c];
          horizontal[(y * width + x) * 3 + c] = sum;
        }
        for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < 3; c++) {
          let sum = 0; for (let k = -reach; k <= reach; k++) sum += kernel[k + reach] * horizontal[(clamp(y + k, height) * width + x) * 3 + c];
          gaussian[(y * width + x) * 3 + c] = sum;
        }
        let integerPhase;
        for (const phase of [0, 0.25]) {
          device.queue.writeBuffer(controls, 0, new Float32Array([radius, phase, 0, 0]));
          const encoder = device.createCommandEncoder(), pass = encoder.beginRenderPass({ colorAttachments: [
            { view: output.createView(), loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] },
          ] });
          pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
          encoder.copyTextureToBuffer({ texture: output }, { buffer: readback, bytesPerRow: width * 4 }, [width, height]);
          device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
          const actual = new Uint8Array(readback.getMappedRange()).slice(); readback.unmap();
          let denseError = 0;
          // Independent direct 2D evaluation at interior and clamped-edge
          // points; the whole-image separable evaluation is its numerical peer.
          for (const [x, y] of [[0, 0], [1, 64], [63, 63], [64, 64], [126, 126]]) for (let c = 0; c < 3; c++) {
            let dense = 0;
            for (let ky = -reach; ky <= reach; ky++) for (let kx = -reach; kx <= reach; kx++) {
              dense += kernel[ky + reach] * kernel[kx + reach] * sample(pixels, x + phase + kx, y + phase + ky, c);
            }
            denseError = Math.max(denseError, Math.abs(dense - sample(gaussian, x + phase, y + phase, c)));
          }
          check(denseError < 1e-10, "dense and separable normalized Gaussian agree including edges");
          const reference = new Uint8Array(actual.length), ringImage = new Uint8Array(actual.length);
          const sums = { pyramid: 0, ring: 0, edge_pyramid: 0, edge_ring: 0,
            leakage_pyramid: 0, leakage_ring: 0, leakage_gaussian: 0, phase_pyramid: 0, phase_ring: 0, phase_gaussian: 0 };
          let edgeSamples = 0, leakageSamples = 0;
          const linear = new Float64Array(pixels.length), oldLinear = new Float64Array(pixels.length), expectedLinear = new Float64Array(pixels.length);
          for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) for (let c = 0; c < 3; c++) {
            const index = (y * width + x) * 4 + c, sharp = sample(pixels, x + phase, y + phase, c);
            const expected = compose(sample(gaussian, x + phase, y + phase, c), sharp, c);
            const old = compose((sharp + taps.reduce((sum, tap) => sum + sample(pixels, x + phase + tap[0] * radius, y + phase + tap[1] * radius, c), 0)) / 9, sharp, c);
            const candidate = decode(actual[index] / 255), square = (candidate - expected) ** 2, oldSquare = (old - expected) ** 2;
            const linearIndex = (y * width + x) * 3 + c;
            linear[linearIndex] = candidate; oldLinear[linearIndex] = old; expectedLinear[linearIndex] = expected;
            check(Number.isFinite(candidate + expected + old), "finite reference and output");
            sums.pyramid += square; sums.ring += oldSquare;
            if (x < 8 || y < 8 || x >= width - 8 || y >= height - 8) { sums.edge_pyramid += square; sums.edge_ring += oldSquare; edgeSamples++; }
            if (pattern === "depth-edge" && x >= 64 && x < 96 && y >= 16 && y < 112) {
              const unblurred = compose(sharp, sharp, c);
              sums.leakage_pyramid += candidate - unblurred; sums.leakage_ring += old - unblurred;
              sums.leakage_gaussian += expected - unblurred; leakageSamples++;
            }
            if (integerPhase) {
              sums.phase_pyramid += (candidate - integerPhase.linear[linearIndex]) ** 2;
              sums.phase_ring += (old - integerPhase.oldLinear[linearIndex]) ** 2;
              sums.phase_gaussian += (expected - integerPhase.expectedLinear[linearIndex]) ** 2;
            }
            reference[index] = encode(expected); ringImage[index] = encode(old);
            reference[index - c + 3] = 255; ringImage[index - c + 3] = 255;
          }
          rows.push({ pattern, radius, phase, width, height, sigma, gaussian_variance: variance,
            old_ring_variance: 5 * radius * radius / 18, dense_separable_max_error: denseError,
            pyramid_mse: sums.pyramid / pixels.length, old_ring_mse: sums.ring / pixels.length,
            edge_pyramid_mse: sums.edge_pyramid / edgeSamples, edge_old_ring_mse: sums.edge_ring / edgeSamples,
            dark_side_leakage: leakageSamples ? { pyramid: sums.leakage_pyramid / leakageSamples,
              old_ring: sums.leakage_ring / leakageSamples, gaussian: sums.leakage_gaussian / leakageSamples } : null,
            quarter_phase_mse: integerPhase ? { pyramid: sums.phase_pyramid / pixels.length,
              old_ring: sums.phase_ring / pixels.length, gaussian: sums.phase_gaussian / pixels.length } : null });
          if (!integerPhase) integerPhase = { linear, oldLinear, expectedLinear };
          const name = `${pattern}-r${radius}-phase${phase}`;
          images.push({ name, pyramid: png(actual), gaussian: png(reference), old_ring: png(ringImage) });
        }
      }
    }
    check(await device.popErrorScope() === null, "spatial diagnostic GPU validation");
    return { width, height, lower_levels: lower,
      composition: { opacity: 0.75, through: 0.65, albedo: [0.9, 0.95, 1], shaded: [0.03, 0.02, 0.01] },
      note: "Normalized discrete Gaussian truncated at 4 sigma, decoded source, clamp-to-edge. Direct dense evaluation checks five interior/edge points against the whole-image separable result. Dark-side leakage is mean added decoded RGB in x=64..95,y=16..111 of the depth-edge pattern. Phase MSE compares integer and quarter-pixel phases. This is an artistic screen-space reference, not physical refraction; all rows, including worse pyramid results, are retained.", rows, images };
  } finally { texture.destroy(); output.destroy(); controls.destroy(); readback.destroy(); }
}
