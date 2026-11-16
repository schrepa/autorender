# Vendored wgpu 30.0.1

Upstream: wgpu 30.0.1 from crates.io (MIT OR Apache-2.0, license files
kept), with its `src/`, `build.rs` and `README.md`. Development
dependencies are omitted; the workspace patches crates.io's `wgpu` with
this copy (`[patch.crates-io]` in the root `Cargo.toml`).

One change: on the WebGPU backend, `Device::create_render_pipeline_async`
and `Device::create_compute_pipeline_async` call the browser's
`createRenderPipelineAsync()` / `createComputePipelineAsync()`, so Chrome
compiles pipelines on its thread pool, in parallel, instead of one after
another on the page's thread. They take the same descriptors as the
synchronous calls and return a future of `Result<_, wgpu::Error>`: a
rejection (`TypeError`, or `GPUPipelineError` with the `"validation"`
reason) is `Error::Validation`, any other is `Error::Internal`, and nothing
reaches the error scope. They exist only on WebGPU (`cfg(webgpu)`) and panic
for a device of another backend; native builds are unchanged. Upstream pull
request gfx-rs/wgpu#10438 adds the same methods for every backend.

- `src/api/device.rs`: the two methods.
- `src/backend/webgpu.rs`: the descriptor mapping moves out of
  `create_render_pipeline` / `create_compute_pipeline` into
  `map_render_pipeline_descriptor` / `map_compute_pipeline_descriptor`
  (unchanged), shared with `WebDevice::create_*_pipeline_async`;
  `Error::from_js_pipeline` maps a rejection.

To upgrade: copy the new release's `src`, `build.rs`, license files,
`README.md` and `Cargo.toml` (dropping development dependencies) and
reapply the change above, or drop this copy once upstream ships the
methods.
