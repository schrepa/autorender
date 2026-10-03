//! Pipeline creation (decision 049). Every render and compute pipeline the
//! renderer builds goes through these functions, which count creations so
//! frames report when compiling ends: a frame that creates none compiles
//! nothing, and hosts reveal the canvas once frames stop creating them.

use std::cell::Cell;

thread_local! {
    static CREATED: Cell<u32> = const { Cell::new(0) };
}

fn count() {
    CREATED.with(|created| created.set(created.get().saturating_add(1)));
}

/// Creates a render pipeline, counting it.
pub(crate) fn render(
    device: &wgpu::Device,
    descriptor: &wgpu::RenderPipelineDescriptor<'_>,
) -> wgpu::RenderPipeline {
    count();
    device.create_render_pipeline(descriptor)
}

/// Creates a compute pipeline, counting it.
pub(crate) fn compute(
    device: &wgpu::Device,
    descriptor: &wgpu::ComputePipelineDescriptor<'_>,
) -> wgpu::ComputePipeline {
    count();
    device.create_compute_pipeline(descriptor)
}

/// The pipelines created since the last call.
pub(crate) fn take_created() -> u32 {
    CREATED.with(|created| created.replace(0))
}
