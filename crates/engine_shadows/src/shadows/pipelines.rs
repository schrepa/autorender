use engine_faces::shadows::{SHADOW_FORMAT, SHADOW_POSES, SHADOW_VIEW};
use engine_render::gpu_data::Vertex;
use engine_render::pipeline_layouts::PipelineLayout;
use engine_render::pipelines::{group, layout};
use engine_render::shaders::Entry;
use engine_render::vertex_attributes;

use super::VIEWS_BYTES;
use crate::bindings::{SHADOW_MARKS, SHADOW_SELECT};
use crate::gpu_layouts::{ComposeData, RouteData, ShadowCullUniform};
use crate::pipeline_layouts::{
    CASTER_PIPELINES, IMPOSTOR_CASTER_PIPELINES, MASKED_CASTER_PIPELINES, SHADOW_MARKS_PIPELINES,
    SHADOW_SELECT_PIPELINES,
};
use crate::shaders::entry::{shadow, shadow_marks};

pub(super) struct ShadowPipelines {
    pub(super) select: wgpu::ComputePipeline,
    pub(super) count: wgpu::ComputePipeline,
    pub(super) select_layout: engine_render::pipelines::GroupLayout,
    pub(super) marks: wgpu::ComputePipeline,
    pub(super) marks_layout: engine_render::pipelines::GroupLayout,
    pub(super) casters: [wgpu::RenderPipeline; 2],
    pub(super) moving: [wgpu::RenderPipeline; 2],
    pub(super) impostor_casters: Option<wgpu::RenderPipeline>,
    pub(super) clear_moving: wgpu::RenderPipeline,
    pub(super) compose: wgpu::RenderPipeline,
    pub(super) view_buffer: wgpu::Buffer,
    pub(super) routes: wgpu::Buffer,
    pub(super) view_group: wgpu::BindGroup,
    pub(super) pose_layout: engine_render::pipelines::GroupLayout,
    pub(super) cull: wgpu::Buffer,
}

fn caster_pipeline(
    device: &wgpu::Device,
    label: &str,
    declared: &'static PipelineLayout,
    entry_point: Entry,
    buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
    overwrite: bool,
    fragment: Option<Entry>,
) -> wgpu::RenderPipeline {
    let layout = engine_render::pipelines::layout_of(device, declared);
    engine_render::pipelines::geometry(
        device,
        &engine_render::pipelines::Geometry {
            label,
            layout: &layout,
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
            fragment: fragment.map(|entry| engine_render::pipelines::Fragment {
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
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let select_layout = layout(device, &SHADOW_SELECT);
        let compute_layout = engine_render::pipelines::layout_of(device, &SHADOW_SELECT_PIPELINES);
        let compute = |entry: engine_render::shaders::Entry| {
            engine_render::pipelines::compute_entry(device, &compute_layout, entry)
        };
        let select = compute(shadow::select_casters);
        let count = compute(shadow::count_casters);
        let marks_layout = layout(device, &SHADOW_MARKS);
        let marks_pipeline_layout =
            engine_render::pipelines::layout_of(device, &SHADOW_MARKS_PIPELINES);
        let marks = engine_render::pipelines::compute_entry(
            device,
            &marks_pipeline_layout,
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
        let plain_layouts: &'static PipelineLayout = &CASTER_PIPELINES;
        let masked_layouts: &'static PipelineLayout = &MASKED_CASTER_PIPELINES;
        let kinds: [(&str, _, Vec<Option<wgpu::VertexBufferLayout<'_>>>, _, _); 2] = [
            (
                "engine/shadow-affine",
                plain_layouts,
                vec![Some(mesh_layout), Some(slot_layout.clone())],
                (shadow::vs_affine, None),
                (shadow::vs_moving, shadow::fs_moving),
            ),
            (
                "engine/shadow-affine-masked",
                masked_layouts,
                vec![Some(masked_mesh_layout), Some(slot_layout)],
                (shadow::vs_affine_masked, Some(shadow::fs_masked)),
                (shadow::vs_moving_masked, shadow::fs_moving_masked),
            ),
        ];
        let casters = kinds
            .each_ref()
            .map(|(label, layouts, buffers, (vertex, fragment), _)| {
                caster_pipeline(device, label, layouts, *vertex, buffers, false, *fragment)
            });
        let moving = kinds
            .each_ref()
            .map(|(label, layouts, buffers, _, (vertex, fragment))| {
                caster_pipeline(
                    device,
                    &format!("{label}-moving"),
                    layouts,
                    *vertex,
                    buffers,
                    false,
                    Some(*fragment),
                )
            });
        let clear_moving = caster_pipeline(
            device,
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
        let view_buffer =
            engine_render::buffers::uniform(device, "engine/shadow-views", VIEWS_BYTES);
        let routes = engine_render::buffers::storage(
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
        let cull = engine_render::buffers::uniform(
            device,
            "engine/shadow-select-uniform",
            size_of::<ShadowCullUniform>() as u64,
        );
        Self {
            select,
            count,
            select_layout,
            marks,
            marks_layout,
            casters,
            moving,
            impostor_casters: None,
            clear_moving,
            compose,
            view_buffer,
            routes,
            view_group,
            pose_layout,
            cull,
        }
    }

    pub(super) fn impostor_caster(&self, device: &wgpu::Device) -> wgpu::RenderPipeline {
        caster_pipeline(
            device,
            "engine/shadow-impostor-cards",
            &IMPOSTOR_CASTER_PIPELINES,
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
