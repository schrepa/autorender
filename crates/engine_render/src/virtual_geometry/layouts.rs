//! The selection's compute pipelines over their shared layouts
//! (`crate::bindings`), each listing every binding its passes read within the
//! eight storage buffers per stage WebGPU guarantees. And the depth-only
//! pipeline drawing shadow views' bins.

use super::VISIBLE_BYTES;
use crate::bindings::{
    VIRTUAL_BIN, VIRTUAL_SEED, VIRTUAL_SELECT, VIRTUAL_SHADOW, VIRTUAL_TRAVERSE,
};
use crate::pipelines::{Family, Fillers, layout};
use crate::shadows::SHADOW_FORMAT;

pub(super) struct Pipelines {
    pub(super) seed: wgpu::ComputePipeline,
    pub(super) level_args: [wgpu::ComputePipeline; 2],
    pub(super) traverse: [wgpu::ComputePipeline; 2],
    pub(super) work_args: wgpu::ComputePipeline,
    pub(super) select: wgpu::ComputePipeline,
    pub(super) bin_args: wgpu::ComputePipeline,
    pub(super) retest_args: wgpu::ComputePipeline,
    pub(super) retest: wgpu::ComputePipeline,
    pub(super) seed_family: Family,
    pub(super) traverse_family: Family,
    pub(super) select_family: Family,
    pub(super) bin_family: Family,
    fillers: Fillers,
}

impl Pipelines {
    pub(super) fn new(device: &wgpu::Device, module: &wgpu::ShaderModule) -> Self {
        let seed_family = Family::new(device, &VIRTUAL_SEED, &[]);
        let traverse_family = Family::new(device, &VIRTUAL_TRAVERSE, &[]);
        let select_family = Family::new(device, &VIRTUAL_SELECT, &[]);
        let bin_family = Family::new(device, &VIRTUAL_BIN, &[]);
        let pipeline = |entry: &str, family: &Family| {
            crate::pipelines::compute_entry(device, &family.pipeline, module, entry)
        };
        Self {
            seed: pipeline("seed", &seed_family),
            level_args: [
                pipeline("level_args_even", &seed_family),
                pipeline("level_args_odd", &seed_family),
            ],
            traverse: [
                pipeline("traverse_even", &traverse_family),
                pipeline("traverse_odd", &traverse_family),
            ],
            work_args: pipeline("work_args", &seed_family),
            select: pipeline("select_clusters", &select_family),
            bin_args: pipeline("bin_args", &bin_family),
            retest_args: pipeline("retest_args", &seed_family),
            retest: pipeline("retest_clusters", &select_family),
            seed_family,
            traverse_family,
            select_family,
            bin_family,
            fillers: Fillers::new(device),
        }
    }

    /// A group of `family`'s layout binding `resources` by name, filled
    /// where they leave bindings out.
    pub(super) fn group(
        &self,
        device: &wgpu::Device,
        family: &Family,
        resources: &[(&str, wgpu::BindingResource<'_>)],
    ) -> wgpu::BindGroup {
        family.group(device, "engine/virtual-group", resources, &self.fillers)
    }
}

/// Depth-only drawing of shadow views' bins (`vs_shadow`) and its layout:
/// the arena, the frame's poses and the shadow maps' views.
pub(super) struct ShadowDrawing {
    pub(super) layout: wgpu::BindGroupLayout,
    pub(super) pipeline: wgpu::RenderPipeline,
}

impl ShadowDrawing {
    pub(super) fn new(device: &wgpu::Device, module: &wgpu::ShaderModule) -> Self {
        let layout = layout(device, &VIRTUAL_SHADOW);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/virtual-shadow-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![0 => Uint32, 1 => Uint32];
        let pipeline = crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/virtual-shadow",
                layout: &pipeline_layout,
                module,
                vertex_constants: &[],
                vertex: "vs_shadow",
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: VISIBLE_BYTES,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &attributes,
                })],
                // Both faces cast, as the shadow maps' other casters.
                primitive: wgpu::PrimitiveState::default(),
                // The shadow maps' constant unit of bias.
                depth: Some(wgpu::DepthStencilState {
                    format: SHADOW_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 1,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    },
                }),
                fragment: None,
            },
        );
        Self { layout, pipeline }
    }
}
