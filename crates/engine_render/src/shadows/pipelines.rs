use super::{SHADOW_FORMAT, VIEWS_BYTES};
use crate::bindings::{SHADOW_MARKS, SHADOW_POSES, SHADOW_SELECT, SHADOW_SKINNED, SHADOW_VIEW};
use crate::gpu_data::Vertex;
use crate::gpu_data::{
    ComposeData, RouteData, SHADOW_MARKS_SHADER, SHADOW_SHADER, ShadowCullUniform,
};
use crate::pipeline_layouts::{
    CASTER_PIPELINES, IMPOSTOR_CASTER_PIPELINES, MASKED_CASTER_PIPELINES, PipelineLayout,
};
use crate::pipelines::{group, layout, vertex_attributes};
use crate::shaders::Entry;
use crate::shaders::entry::{shadow, shadow_marks};

pub(super) struct ShadowPipelines {
    pub(super) select: wgpu::ComputePipeline,
    pub(super) count: wgpu::ComputePipeline,
    pub(super) select_layout: crate::pipelines::GroupLayout,
    pub(super) select_skinned: wgpu::ComputePipeline,
    pub(super) skinned_layout: crate::pipelines::GroupLayout,
    pub(super) marks: wgpu::ComputePipeline,
    pub(super) marks_layout: crate::pipelines::GroupLayout,
    pub(super) casters: [wgpu::RenderPipeline; 4],
    pub(super) moving: [wgpu::RenderPipeline; 4],
    pub(super) impostor_casters: Option<wgpu::RenderPipeline>,
    module: wgpu::ShaderModule,
    view_layout: crate::pipelines::GroupLayout,
    pub(super) clear_moving: wgpu::RenderPipeline,
    pub(super) compose: wgpu::RenderPipeline,
    pub(super) view_buffer: wgpu::Buffer,
    pub(super) routes: wgpu::Buffer,
    pub(super) view_group: wgpu::BindGroup,
    pub(super) pose_layout: crate::pipelines::GroupLayout,
    pub(super) cull: wgpu::Buffer,
}

#[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
fn caster_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    label: &str,
    (declared, layouts): (&PipelineLayout, &[&crate::pipelines::GroupLayout]),
    entry_point: Entry,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    overwrite: bool,
    fragment: Option<Entry>,
) -> wgpu::RenderPipeline {
    let layout = crate::pipelines::pipeline_layout(device, declared, layouts);
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
            label,
            layout: &layout,
            module,
            vertex_constants: &[],
            vertex: entry_point,
            buffers,
            primitive: wgpu::PrimitiveState::default(),
            depth: Some(wgpu::DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(if overwrite {
                    wgpu::CompareFunction::Always
                } else {
                    wgpu::CompareFunction::Less
                }),
                stencil: Default::default(),
                bias: if overwrite {
                    Default::default()
                } else {
                    wgpu::DepthBiasState {
                        constant: 1,
                        slope_scale: 0.0,
                        clamp: 0.0,
                    }
                },
            }),
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
    pub(super) fn new(
        device: &wgpu::Device,
        material_layout: &crate::pipelines::GroupLayout,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/shadows.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        });
        let select_layout = layout(device, &SHADOW_SELECT);
        let compute_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::SHADOW_SELECT_PIPELINES,
            &[&select_layout],
        );
        let compute = |entry: crate::shaders::Entry| {
            crate::pipelines::compute_entry(device, &compute_layout, &module, entry)
        };
        let select = compute(shadow::select_casters);
        let count = compute(shadow::count_casters);
        let skinned_layout = layout(device, &SHADOW_SKINNED);
        let skinned_pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::SHADOW_SKINNED_PIPELINES,
            &[&skinned_layout],
        );
        let select_skinned = crate::pipelines::compute_entry(
            device,
            &skinned_pipeline_layout,
            &module,
            shadow::select_skinned,
        );
        let marks_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/shadow_marks.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_MARKS_SHADER.into()),
        });
        let marks_layout = layout(device, &SHADOW_MARKS);
        let marks_pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::SHADOW_MARKS_PIPELINES,
            &[&marks_layout],
        );
        let marks = crate::pipelines::compute_entry(
            device,
            &marks_pipeline_layout,
            &marks_module,
            shadow_marks::mark_shadow_tiles,
        );
        let view_layout = layout(device, &SHADOW_VIEW);
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
        let slot_layout = wgpu::VertexBufferLayout {
            array_stride: 4,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &slot_attributes,
        };
        let plain_layouts: (&PipelineLayout, &[&crate::pipelines::GroupLayout]) =
            (&CASTER_PIPELINES, &[&view_layout, &pose_layout]);
        let masked_layouts: (&PipelineLayout, &[&crate::pipelines::GroupLayout]) = (
            &MASKED_CASTER_PIPELINES,
            &[&view_layout, &pose_layout, material_layout],
        );
        let kinds: [(&str, _, Vec<Option<wgpu::VertexBufferLayout<'_>>>, _, _); 4] = [
            (
                "engine/shadow-affine",
                plain_layouts,
                vec![Some(mesh_layout.clone()), Some(slot_layout.clone())],
                (shadow::vs_affine, None),
                (shadow::vs_moving, shadow::fs_moving),
            ),
            (
                "engine/shadow-affine-masked",
                masked_layouts,
                vec![Some(masked_mesh_layout.clone()), Some(slot_layout.clone())],
                (shadow::vs_affine_masked, Some(shadow::fs_masked)),
                (shadow::vs_moving_masked, shadow::fs_moving_masked),
            ),
            (
                "engine/shadow-affine-skinned",
                plain_layouts,
                vec![
                    Some(mesh_layout.clone()),
                    Some(slot_layout.clone()),
                    skin_layout.clone(),
                ],
                (shadow::vs_affine_skinned, None),
                (shadow::vs_moving_skinned, shadow::fs_moving),
            ),
            (
                "engine/shadow-affine-masked-skinned",
                masked_layouts,
                vec![
                    Some(masked_mesh_layout),
                    Some(slot_layout.clone()),
                    skin_layout,
                ],
                (shadow::vs_affine_masked_skinned, Some(shadow::fs_masked)),
                (shadow::vs_moving_masked_skinned, shadow::fs_moving_masked),
            ),
        ];
        let casters = kinds
            .each_ref()
            .map(|(label, layouts, buffers, (vertex, fragment), _)| {
                caster_pipeline(
                    device, &module, label, *layouts, *vertex, buffers, false, *fragment,
                )
            });
        let moving = kinds
            .each_ref()
            .map(|(label, layouts, buffers, _, (vertex, fragment))| {
                caster_pipeline(
                    device,
                    &module,
                    &format!("{label}-moving"),
                    *layouts,
                    *vertex,
                    buffers,
                    false,
                    Some(*fragment),
                )
            });
        let clear_moving = caster_pipeline(
            device,
            &module,
            "engine/shadow-clear-moving",
            plain_layouts,
            shadow::vs_clear_moving,
            &[],
            true,
            None,
        );
        let compose_attributes = wgpu::vertex_attr_array![3 => Float32x4, 4 => Sint32x4];
        let compose = caster_pipeline(
            device,
            &module,
            "engine/shadow-compose",
            plain_layouts,
            shadow::vs_compose,
            &[Some(wgpu::VertexBufferLayout {
                array_stride: size_of::<ComposeData>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &compose_attributes,
            })],
            true,
            Some(shadow::fs_compose),
        );
        let view_buffer = crate::buffers::uniform(device, "engine/shadow-views", VIEWS_BYTES);
        let routes = crate::buffers::storage(
            device,
            "engine/shadow-routes",
            size_of::<RouteData>() as u64,
            wgpu::BufferUsages::empty(),
        );
        let view_group = group(
            device,
            "engine/shadow-view-group",
            &SHADOW_VIEW,
            &view_layout,
            &[
                (
                    "view",
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &view_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(96),
                    }),
                ),
                ("routes", routes.as_entire_binding()),
            ],
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
            marks,
            marks_layout,
            casters,
            moving,
            impostor_casters: None,
            module,
            view_layout,
            clear_moving,
            compose,
            view_buffer,
            routes,
            view_group,
            pose_layout,
            cull,
        }
    }

    pub(super) fn impostor_caster(
        &self,
        device: &wgpu::Device,
        impostor_layout: &crate::pipelines::GroupLayout,
    ) -> wgpu::RenderPipeline {
        caster_pipeline(
            device,
            &self.module,
            "engine/shadow-impostor-cards",
            (
                &IMPOSTOR_CASTER_PIPELINES,
                &[&self.view_layout, &self.pose_layout, impostor_layout],
            ),
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
