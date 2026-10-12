//! The depth pipelines of shadow maps: caster
//! selection's compute passes, slot casters plain, masked and skinned,
//! placed objects' impostor cards and the clear of kept-map
//! tiles, with the view matrices and selection uniform they read.

use super::{LEVEL_VIEW, SHADOW_FORMAT, VIEW_STRIDE};
use crate::bindings::{SHADOW_POSES, SHADOW_SELECT, SHADOW_SKINNED, SHADOW_VIEW};
use crate::gpu_data::Vertex;
use crate::gpu_data::{SHADOW_SHADER, ShadowCullUniform};
use crate::pipelines::{group, layout, vertex_attributes};
use crate::shaders::Entry;
use crate::shaders::entry::shadow;

/// Shadow pipelines and their fixed bindings, created with the first frame
/// that has shadows: games without them never compile them.
pub(super) struct ShadowPipelines {
    pub(super) select: wgpu::ComputePipeline,
    pub(super) count: wgpu::ComputePipeline,
    pub(super) select_layout: wgpu::BindGroupLayout,
    // Skinned casters of GPU-presented frames.
    pub(super) select_skinned: wgpu::ComputePipeline,
    pub(super) skinned_layout: wgpu::BindGroupLayout,
    // Slot casters by `masked + 2 * skinned`: plain; alpha-masked,
    // with a material group; and of meshes with joints,
    // which read a skin stream and the joint palettes.
    pub(super) casters: [wgpu::RenderPipeline; 4],
    // Placed objects' impostor cards, created once they
    // have impostors, and what creating them takes.
    pub(super) impostor_casters: Option<wgpu::RenderPipeline>,
    module: wgpu::ShaderModule,
    view_layout: wgpu::BindGroupLayout,
    pub(super) clear: wgpu::RenderPipeline,
    pub(super) view_buffer: wgpu::Buffer,
    pub(super) view_group: wgpu::BindGroup,
    pub(super) pose_layout: wgpu::BindGroupLayout,
    pub(super) cull: wgpu::Buffer,
}

/// A shadow caster pipeline of `module`'s `entry_point` over `layouts` and
/// vertex `buffers`; `clear` passes write the far depth everywhere, and
/// casters with a `fragment` stage test texels.
#[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
fn caster_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    label: &str,
    layouts: &[Option<&wgpu::BindGroupLayout>],
    entry_point: Entry,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    clear: bool,
    fragment: Option<Entry>,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: layouts,
        immediate_size: 0,
    });
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
            label,
            layout: &layout,
            module,
            vertex_constants: &[],
            vertex: entry_point,
            buffers,
            // Both faces cast: open and single-sided meshes shadow too.
            primitive: wgpu::PrimitiveState::default(),
            // Receiver planes keep lit surfaces off
            // their own depth; a unit of bias covers the sun shafts'
            // plain comparisons.
            depth: Some(wgpu::DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(if clear {
                    wgpu::CompareFunction::Always
                } else {
                    wgpu::CompareFunction::Less
                }),
                stencil: Default::default(),
                bias: if clear {
                    Default::default()
                } else {
                    wgpu::DepthBiasState {
                        constant: 1,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    }
                },
            }),
            // Only masked casters and cards test texels; the rest write
            // depth alone.
            fragment: fragment.map(|entry| crate::pipelines::Fragment {
                entry,
                constants: &[],
                targets: &[],
            }),
        },
    )
}

impl ShadowPipelines {
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(super) fn new(device: &wgpu::Device, material_layout: &wgpu::BindGroupLayout) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/shadows.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        });
        // Selection and counting share one explicit layout, and so one group.
        let select_layout = layout(device, &SHADOW_SELECT);
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/shadow-select-pipeline-layout"),
            bind_group_layouts: &[Some(&select_layout)],
            immediate_size: 0,
        });
        let compute = |entry: crate::shaders::Entry| {
            crate::pipelines::compute_entry(device, &compute_layout, &module, entry)
        };
        let select = compute(shadow::select_casters);
        let count = compute(shadow::count_casters);
        // Skinned casters bind their palette bounds instead of
        // candidates, poses and stillness: within eight storage buffers.
        let skinned_layout = layout(device, &SHADOW_SKINNED);
        let skinned_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/shadow-skinned-pipeline-layout"),
                bind_group_layouts: &[Some(&skinned_layout)],
                immediate_size: 0,
            });
        let select_skinned = crate::pipelines::compute_entry(
            device,
            &skinned_pipeline_layout,
            &module,
            shadow::select_skinned,
        );
        let view_layout = layout(device, &SHADOW_VIEW);
        // Slot poses, and slot colors and palettes for masked and skinned
        // casters.
        let pose_layout = layout(device, &SHADOW_POSES);
        let vertex_attributes = vertex_attributes!(Vertex { 0 => position as Float32x3 });
        let masked_vertex_attributes = vertex_attributes!(Vertex {
            0 => position as Float32x3,
            7 => uv as Float32x2,
        });
        let slot_attributes = wgpu::vertex_attr_array![2 => Uint32];
        let skin_layout = crate::renderer::skin_layout();
        let mesh_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attributes,
        };
        let masked_mesh_layout = wgpu::VertexBufferLayout {
            array_stride: size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &masked_vertex_attributes,
        };
        let pipeline = |label: &str,
                        layouts: &[Option<&wgpu::BindGroupLayout>],
                        entry_point: Entry,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        clear: bool,
                        masked: bool| {
            caster_pipeline(
                device,
                &module,
                label,
                layouts,
                entry_point,
                buffers,
                clear,
                masked.then_some(shadow::fs_masked),
            )
        };
        let plain = pipeline(
            "engine/shadow-affine",
            &[Some(&view_layout), Some(&pose_layout)],
            shadow::vs_affine,
            &[
                Some(mesh_layout.clone()),
                Some(wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &slot_attributes,
                }),
            ],
            false,
            false,
        );
        let masked = pipeline(
            "engine/shadow-affine-masked",
            &[
                Some(&view_layout),
                Some(&pose_layout),
                Some(material_layout),
            ],
            shadow::vs_affine_masked,
            &[
                Some(masked_mesh_layout.clone()),
                Some(wgpu::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &slot_attributes,
                }),
            ],
            false,
            true,
        );
        let slot_layout = wgpu::VertexBufferLayout {
            array_stride: 4,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &slot_attributes,
        };
        let skinned = pipeline(
            "engine/shadow-affine-skinned",
            &[Some(&view_layout), Some(&pose_layout)],
            shadow::vs_affine_skinned,
            &[
                Some(mesh_layout.clone()),
                Some(slot_layout.clone()),
                skin_layout.clone(),
            ],
            false,
            false,
        );
        let masked_skinned = pipeline(
            "engine/shadow-affine-masked-skinned",
            &[
                Some(&view_layout),
                Some(&pose_layout),
                Some(material_layout),
            ],
            shadow::vs_affine_masked_skinned,
            &[Some(masked_mesh_layout), Some(slot_layout), skin_layout],
            false,
            true,
        );
        // The clear binds nothing but shares the plain casters' layout: each
        // distinct layout costs Firefox a root signature.
        let clear = pipeline(
            "engine/shadow-clear",
            &[Some(&view_layout), Some(&pose_layout)],
            shadow::vs_clear,
            &[],
            true,
            false,
        );
        let view_buffer =
            crate::buffers::uniform(device, "engine/shadow-views", LEVEL_VIEW + VIEW_STRIDE);
        let view_group = group(
            device,
            "engine/shadow-view-group",
            &SHADOW_VIEW,
            &view_layout,
            &[(
                "view",
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &view_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(80),
                }),
            )],
        );
        let cull = crate::buffers::uniform(
            device,
            "engine/shadow-select-uniform",
            size_of::<ShadowCullUniform>() as u64,
        );
        Self {
            select,
            count,
            select_layout,
            select_skinned,
            skinned_layout,
            casters: [plain, masked, skinned, masked_skinned],
            impostor_casters: None,
            module,
            view_layout,
            clear,
            view_buffer,
            view_group,
            pose_layout,
            cull,
        }
    }

    /// The pipeline of placed objects' impostor cards: the
    /// view, the objects' poses and `impostor_layout`'s impostors, each
    /// card's slot and record as instance attributes.
    pub(super) fn impostor_caster(
        &self,
        device: &wgpu::Device,
        impostor_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        caster_pipeline(
            device,
            &self.module,
            "engine/shadow-impostor-cards",
            &[
                Some(&self.view_layout),
                Some(&self.pose_layout),
                Some(impostor_layout),
            ],
            shadow::vs_impostor_card,
            &[Some(wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![2 => Uint32, 3 => Uint32],
            })],
            false,
            Some(shadow::fs_impostor_card),
        )
    }
}
