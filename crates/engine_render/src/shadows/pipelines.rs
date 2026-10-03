//! The depth pipelines of shadow maps (decisions 020 and 031): caster
//! selection's compute passes, retained casters plain, masked and skinned,
//! public snapshots' casters from full instance matrices, and the clear of
//! kept-map tiles, with the view matrices and selection uniform they read.

use super::{FAR_VIEW, SHADOW_FORMAT, VIEW_STRIDE};
use crate::bindings::{SHADOW_POSES, SHADOW_SELECT, SHADOW_SKINNED, SHADOW_VIEW};
use crate::gpu_data::{SHADOW_SHADER, ShadowCullUniform};
use crate::pipelines::{group, layout};

/// Shadow pipelines and their fixed bindings, created with the first frame
/// that has shadows: games without them never compile them.
pub(super) struct ShadowPipelines {
    pub(super) select: wgpu::ComputePipeline,
    pub(super) count: wgpu::ComputePipeline,
    pub(super) select_layout: wgpu::BindGroupLayout,
    // Skinned casters of GPU-presented frames (decision 036).
    pub(super) select_skinned: wgpu::ComputePipeline,
    pub(super) skinned_layout: wgpu::BindGroupLayout,
    pub(super) affine: wgpu::RenderPipeline,
    // Alpha-masked casters (decision 031), with a material group.
    pub(super) affine_masked: wgpu::RenderPipeline,
    // Retained casters of meshes with joints (decision 036), plain and
    // masked, which read a skin stream and the joint palettes.
    pub(super) affine_skinned: wgpu::RenderPipeline,
    pub(super) affine_masked_skinned: wgpu::RenderPipeline,
    // Public snapshots' casters from full instance matrices, plain and
    // masked, created with the first frame drawing a snapshot (decision
    // 049), from the module and view layout kept for them.
    pub(super) full: Option<[wgpu::RenderPipeline; 2]>,
    pub(super) module: wgpu::ShaderModule,
    pub(super) view_layout: wgpu::BindGroupLayout,
    pub(super) clear: wgpu::RenderPipeline,
    pub(super) view_buffer: wgpu::Buffer,
    pub(super) view_group: wgpu::BindGroup,
    pub(super) pose_layout: wgpu::BindGroupLayout,
    pub(super) cull: wgpu::Buffer,
}

/// A shadow caster pipeline of `module`'s `entry_point` over `layouts` and
/// vertex `buffers`; `clear` passes write the far depth everywhere, and
/// `masked` casters test their material's texels.
#[allow(clippy::too_many_arguments)]
fn caster_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    label: &str,
    layouts: &[Option<&wgpu::BindGroupLayout>],
    entry_point: &str,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    clear: bool,
    masked: bool,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: layouts,
        immediate_size: 0,
    });
    crate::pipelines::render(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                buffers,
            },
            // Both faces cast: open and single-sided meshes shadow too.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(if clear {
                    wgpu::CompareFunction::Always
                } else {
                    wgpu::CompareFunction::Less
                }),
                stencil: Default::default(),
                // Receiver planes (decision 028) keep lit surfaces off
                // their own depth; a unit of bias covers the sun shafts'
                // plain comparisons.
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
            multisample: wgpu::MultisampleState::default(),
            // Only masked casters test texels; the rest write depth alone.
            fragment: masked.then_some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_masked"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        },
    )
}

impl ShadowPipelines {
    /// `vertex_stride` and `instance_stride` are the renderer's mesh vertex
    /// and public instance record sizes.
    /// `layout`: the vertex and instance strides and the texture coordinate
    /// offset in a vertex.
    pub(super) fn new(
        device: &wgpu::Device,
        (vertex_stride, _, uv_offset): (u64, u64, u64),
        material_layout: &wgpu::BindGroupLayout,
    ) -> Self {
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
        let compute =
            |entry: &str| crate::pipelines::compute_entry(device, &compute_layout, &module, entry);
        let select = compute("select_casters");
        let count = compute("count_casters");
        // Skinned casters (decision 036) bind their palette bounds instead of
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
            "select_skinned",
        );
        let view_layout = layout(device, &SHADOW_VIEW);
        // Slot poses, and slot colors and palettes for masked and skinned
        // casters.
        let pose_layout = layout(device, &SHADOW_POSES);
        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x3];
        let masked_vertex_attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: uv_offset,
                shader_location: 7,
            },
        ];
        let slot_attributes = wgpu::vertex_attr_array![2 => Uint32];
        let skin_layout = crate::browser::skin_layout();
        let mesh_layout = wgpu::VertexBufferLayout {
            array_stride: vertex_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attributes,
        };
        let masked_mesh_layout = wgpu::VertexBufferLayout {
            array_stride: vertex_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &masked_vertex_attributes,
        };
        let pipeline = |label: &str,
                        layouts: &[Option<&wgpu::BindGroupLayout>],
                        entry_point: &str,
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
                masked,
            )
        };
        let affine = pipeline(
            "engine/shadow-affine",
            &[Some(&view_layout), Some(&pose_layout)],
            "vs_affine",
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
        let affine_masked = pipeline(
            "engine/shadow-affine-masked",
            &[
                Some(&view_layout),
                Some(&pose_layout),
                Some(material_layout),
            ],
            "vs_affine_masked",
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
        let affine_skinned = pipeline(
            "engine/shadow-affine-skinned",
            &[Some(&view_layout), Some(&pose_layout)],
            "vs_affine_skinned",
            &[
                Some(mesh_layout.clone()),
                Some(slot_layout.clone()),
                skin_layout.clone(),
            ],
            false,
            false,
        );
        let affine_masked_skinned = pipeline(
            "engine/shadow-affine-masked-skinned",
            &[
                Some(&view_layout),
                Some(&pose_layout),
                Some(material_layout),
            ],
            "vs_affine_masked_skinned",
            &[Some(masked_mesh_layout), Some(slot_layout), skin_layout],
            false,
            true,
        );
        // The clear binds nothing but shares the plain casters' layout: each
        // distinct layout costs Firefox a root signature (decision 049).
        let clear = pipeline(
            "engine/shadow-clear",
            &[Some(&view_layout), Some(&pose_layout)],
            "vs_clear",
            &[],
            true,
            false,
        );
        let view_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/shadow-views"),
            size: FAR_VIEW + VIEW_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
                    size: wgpu::BufferSize::new(64),
                }),
            )],
        );
        let cull = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/shadow-select-uniform"),
            size: size_of::<ShadowCullUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            select,
            count,
            select_layout,
            select_skinned,
            skinned_layout,
            affine,
            affine_masked,
            affine_skinned,
            affine_masked_skinned,
            full: None,
            module,
            view_layout,
            clear,
            view_buffer,
            view_group,
            pose_layout,
            cull,
        }
    }
}

impl ShadowPipelines {
    /// Creates the public snapshots' casters, plain and masked, unless
    /// created. `layout` and `material_layout` are as for [`Self::new`].
    pub(super) fn ensure_full(
        &mut self,
        device: &wgpu::Device,
        (vertex_stride, instance_stride, uv_offset): (u64, u64, u64),
        material_layout: &wgpu::BindGroupLayout,
    ) {
        if self.full.is_some() {
            return;
        }
        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x3];
        let masked_vertex_attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: uv_offset,
                shader_location: 7,
            },
        ];
        let model_attributes = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4
        ];
        // Columns and the packed color of public instance records.
        let masked_model_attributes = wgpu::vertex_attr_array![
            2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Uint32x2
        ];
        let vertices = |attributes| wgpu::VertexBufferLayout {
            array_stride: vertex_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes,
        };
        let instances = |attributes| wgpu::VertexBufferLayout {
            array_stride: instance_stride,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes,
        };
        let full = caster_pipeline(
            device,
            &self.module,
            "engine/shadow-full",
            &[Some(&self.view_layout)],
            "vs_full",
            &[
                Some(vertices(&vertex_attributes)),
                Some(instances(&model_attributes)),
            ],
            false,
            false,
        );
        let full_masked = caster_pipeline(
            device,
            &self.module,
            "engine/shadow-full-masked",
            &[Some(&self.view_layout), None, Some(material_layout)],
            "vs_full_masked",
            &[
                Some(vertices(&masked_vertex_attributes)),
                Some(instances(&masked_model_attributes)),
            ],
            false,
            true,
        );
        self.full = Some([full, full_masked]);
    }
}
