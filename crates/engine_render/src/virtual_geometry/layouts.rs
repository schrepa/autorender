//! The selection's compute pipelines over their shared layouts (decision
//! 049), each listing every binding its passes read, by
//! virtual_geometry.wgsl's numbers, within the eight storage buffers per
//! stage WebGPU guarantees; passes dispatching from the dispatch arguments
//! (15) never bind them. And the depth-only pipeline drawing shadow views'
//! bins.

use super::VISIBLE_BYTES;
use crate::pipelines::{Fillers, Slot, filled_group, group_layout};
use crate::shadows::SHADOW_FORMAT;

use Slot::{Read, Texture, Uniform, Write};

// Seeding, and the dispatch arguments of each level, the work and the
// retest.
const SEED_BINDINGS: &[(u32, Slot)] = &[
    (0, Uniform),
    (1, Read),
    (2, Read),
    (7, Read),
    (8, Read),
    (9, Write),
    (10, Write),
    (12, Write),
    (15, Write),
];
const TRAVERSE_BINDINGS: &[(u32, Slot)] = &[
    (0, Uniform),
    (3, Read),
    (6, Read),
    (9, Write),
    (10, Write),
    (11, Write),
    (12, Write),
];
// Selection, and the retest of the clusters it left waiting.
const SELECT_BINDINGS: &[(u32, Slot)] = &[
    (0, Uniform),
    (4, Read),
    (5, Read),
    (6, Read),
    (11, Write),
    (12, Write),
    (13, Write),
    (14, Read),
    (19, Texture),
    (20, Write),
];
const BIN_BINDINGS: &[(u32, Slot)] = &[(0, Uniform), (12, Write), (14, Read), (16, Write)];

/// A shared layout: its bindings, group layout and pipeline layout.
pub(super) struct Family {
    bindings: &'static [(u32, Slot)],
    group: wgpu::BindGroupLayout,
    pipeline: wgpu::PipelineLayout,
}

impl Family {
    fn new(device: &wgpu::Device, label: &str, bindings: &'static [(u32, Slot)]) -> Self {
        let group = group_layout(device, label, wgpu::ShaderStages::COMPUTE, bindings);
        let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&group)],
            immediate_size: 0,
        });
        Self {
            bindings,
            group,
            pipeline,
        }
    }
}

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
        let seed_family = Family::new(device, "engine/virtual-seed-layout", SEED_BINDINGS);
        let traverse_family =
            Family::new(device, "engine/virtual-traverse-layout", TRAVERSE_BINDINGS);
        let select_family = Family::new(device, "engine/virtual-select-layout", SELECT_BINDINGS);
        let bin_family = Family::new(device, "engine/virtual-bin-layout", BIN_BINDINGS);
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

    /// A group of `family`'s layout holding `entries`, filled where they
    /// leave bindings out.
    pub(super) fn group(
        &self,
        device: &wgpu::Device,
        family: &Family,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> wgpu::BindGroup {
        filled_group(
            device,
            "engine/virtual-group",
            &family.group,
            family.bindings,
            entries,
            &self.fillers,
        )
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
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/virtual-shadow-layout"),
            entries: &[
                storage(2),
                storage(5),
                storage(6),
                storage(17),
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/virtual-shadow-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![0 => Uint32, 1 => Uint32];
        let pipeline = crate::pipelines::render(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("engine/virtual-shadow"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_shadow"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: VISIBLE_BYTES,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &attributes,
                    })],
                },
                // Both faces cast, as the shadow maps' other casters.
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: SHADOW_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    // The shadow maps' constant unit of bias (decision 028).
                    bias: wgpu::DepthBiasState {
                        constant: 1,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: None,
                multiview_mask: None,
                cache: None,
            },
        );
        Self { layout, pipeline }
    }
}
