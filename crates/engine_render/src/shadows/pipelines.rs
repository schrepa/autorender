//! The pipelines of the shadow clipmap: caster selection's compute passes,
//! the marks of the tiles the depth buffer's pixels read, slot casters
//! plain, masked and skinned (into a region's layer, and through moving
//! tiles' routes into the atlas), placed objects' impostor cards, the clear
//! of moving tiles' slots and the tiles composed from the regions into
//! their slots; with the views, routes and selection uniform they read.

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

/// Shadow pipelines and their fixed bindings, created with the first frame
/// that has shadows: games without them never compile them.
pub(super) struct ShadowPipelines {
    pub(super) select: wgpu::ComputePipeline,
    pub(super) count: wgpu::ComputePipeline,
    pub(super) select_layout: crate::pipelines::GroupLayout,
    // Skinned casters of GPU-presented frames.
    pub(super) select_skinned: wgpu::ComputePipeline,
    pub(super) skinned_layout: crate::pipelines::GroupLayout,
    // The marks of the tiles pixels read, after the opaque passes.
    pub(super) marks: wgpu::ComputePipeline,
    pub(super) marks_layout: crate::pipelines::GroupLayout,
    // Slot casters by `masked + 2 * skinned`: plain; alpha-masked,
    // with a material group; and of meshes with joints,
    // which read a skin stream and the joint palettes. Into a region's
    // layer, then drawn over moving tiles through their routes.
    pub(super) casters: [wgpu::RenderPipeline; 4],
    pub(super) moving: [wgpu::RenderPipeline; 4],
    // Placed objects' impostor cards, created once they
    // have impostors, and what creating them takes.
    pub(super) impostor_casters: Option<wgpu::RenderPipeline>,
    module: wgpu::ShaderModule,
    view_layout: crate::pipelines::GroupLayout,
    // The moving tiles' slots cleared, and the tiles composed from the
    // regions into their slots, quads over the slots.
    pub(super) clear_moving: wgpu::RenderPipeline,
    pub(super) compose: wgpu::RenderPipeline,
    pub(super) view_buffer: wgpu::Buffer,
    pub(super) routes: wgpu::Buffer,
    pub(super) view_group: wgpu::BindGroup,
    pub(super) pose_layout: crate::pipelines::GroupLayout,
    pub(super) cull: wgpu::Buffer,
}

/// A shadow caster pipeline of `module`'s `entry_point` over `layouts` and
/// vertex `buffers`; `overwrite` passes write their depth whatever lies
/// there, and casters with a `fragment` stage test texels.
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
            // Both faces cast: open and single-sided meshes shadow too.
            primitive: wgpu::PrimitiveState::default(),
            // Receiver planes keep lit surfaces off
            // their own depth; a unit of bias covers the haze's march's
            // plain comparisons.
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
            // Masked casters, cards and moving casters test texels; the rest
            // write depth alone.
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
        // Selection and counting share one explicit layout, and so one group.
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
        // Skinned casters bind their palette bounds instead of
        // candidates, poses and stillness: within eight storage buffers.
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
        // The marks read the camera and the lookups' uniform, after the
        // scene's declarations.
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
        // Each kind of caster: its label, layouts, vertex buffers and its
        // entry points into a region's layer and over moving tiles, with
        // the fragment stages of each.
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
        // The clear binds nothing but shares the plain casters' layout: each
        // distinct layout costs Firefox a root signature.
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
        // Composing writes every texel's depth as it reads it, through the
        // plain casters' layout, whose pose group binds the regions' layers.
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
        // Too large for a uniform binding: storage.
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

    /// The pipeline of placed objects' impostor cards: the
    /// view, the objects' poses and `impostor_layout`'s impostors, each
    /// card's slot and record as instance attributes.
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
