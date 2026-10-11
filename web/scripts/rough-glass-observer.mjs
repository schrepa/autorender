/** Test-local WebGPU observation. Every engine call is forwarded unchanged. */
export function observeGlass() {
  const textures = new WeakMap(), views = new WeakMap(), groups = new WeakMap(), passes = new WeakMap();
  const state = window.__roughGlass = { textures: [], passes: [], pipelines: 0, device: null, post: null, mesh: null, adapter: null, sampler: null };
  const rawDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = function (descriptor) {
    const info = this.info;
    state.adapter = { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description };
    return rawDevice.call(this, descriptor);
  };
  const rawSampler = GPUDevice.prototype.createSampler;
  GPUDevice.prototype.createSampler = function (descriptor = {}) {
    if (descriptor.label === "engine/refraction-sampler") state.sampler = structuredClone(descriptor);
    return rawSampler.call(this, descriptor);
  };
  const rawTexture = GPUDevice.prototype.createTexture;
  GPUDevice.prototype.createTexture = function (descriptor) {
    const texture = rawTexture.call(this, descriptor);
    if (descriptor.label === "engine/post-scene" || descriptor.label === "engine/post-scene-levels") {
      const record = { id: state.textures.length, width: texture.width, height: texture.height,
        levels: texture.mipLevelCount, format: texture.format, label: descriptor.label, destroyed: false };
      textures.set(texture, record); state.textures.push(record);
    }
    return texture;
  };
  const rawDestroy = GPUTexture.prototype.destroy;
  GPUTexture.prototype.destroy = function () {
    const record = textures.get(this); if (record) record.destroyed = true;
    return rawDestroy.call(this);
  };
  const rawView = GPUTexture.prototype.createView;
  GPUTexture.prototype.createView = function (descriptor = {}) {
    const view = rawView.call(this, descriptor), texture = textures.get(this);
    if (texture) views.set(view, { texture: texture.id, base: descriptor.baseMipLevel ?? 0,
      levels: descriptor.mipLevelCount ?? texture.levels - (descriptor.baseMipLevel ?? 0) });
    return view;
  };
  const rawShader = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function (descriptor) {
    if (descriptor.label?.startsWith("engine/")) {
      if (descriptor.code.includes("fn fs_refraction_prefilter(")) state.post = descriptor.code;
      if (descriptor.code.includes("fn refraction_level(")) state.mesh = descriptor.code;
      state.device = this;
    }
    return rawShader.call(this, descriptor);
  };
  for (const name of ["createRenderPipeline", "createRenderPipelineAsync"]) {
    const original = GPUDevice.prototype[name];
    GPUDevice.prototype[name] = function (descriptor) {
      if (descriptor.label === "engine/refraction-prefilter") state.pipelines++;
      return original.call(this, descriptor);
    };
  }
  const rawGroup = GPUDevice.prototype.createBindGroup;
  GPUDevice.prototype.createBindGroup = function (descriptor) {
    const group = rawGroup.call(this, descriptor), bindings = {};
    for (const entry of descriptor.entries) {
      const view = views.get(entry.resource); if (view) bindings[entry.binding] = view;
    }
    groups.set(group, bindings); return group;
  };
  const rawPass = GPUCommandEncoder.prototype.beginRenderPass;
  GPUCommandEncoder.prototype.beginRenderPass = function (descriptor) {
    const pass = rawPass.call(this, descriptor);
    if (descriptor.label?.startsWith("engine/")) {
      const record = { label: descriptor.label,
        attachments: [...descriptor.colorAttachments].filter(Boolean).map(entry => views.get(entry.view) ?? null), groups: {}, bindings: [] };
      passes.set(pass, record); state.passes.push(record);
    }
    return pass;
  };
  const rawBind = GPURenderPassEncoder.prototype.setBindGroup;
  GPURenderPassEncoder.prototype.setBindGroup = function (index, group, ...rest) {
    const pass = passes.get(this);
    if (pass && group) {
      pass.groups[index] = groups.get(group) ?? {};
      pass.bindings.push({ index, views: groups.get(group) ?? {} });
    }
    return rawBind.call(this, index, group, ...rest);
  };
  state.reset = () => { state.passes = []; };
  state.snapshot = () => ({ textures: structuredClone(state.textures), passes: structuredClone(state.passes),
    pipelines: state.pipelines, adapter: state.adapter, sampler: state.sampler });
}
