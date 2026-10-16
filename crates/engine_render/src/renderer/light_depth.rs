//! Optional mesh pipelines: the usual displayed vertex paths with a coverage
//! fragment, and forward opaque fragments with the guarded filtered lists.

use super::*;
use crate::entries::Slots;

#[derive(Clone, Copy)]
pub(super) enum DepthMeshKind {
    Collect,
    Filtered,
}

#[derive(Default)]
pub(super) struct DepthMeshPipelines {
    collection: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    shading: HashMap<(PassKind, Variant), wgpu::RenderPipeline>,
    layouts: [Option<wgpu::PipelineLayout>; 2],
}

impl DepthMeshPipelines {
    pub(super) fn pipeline(
        &self,
        depth: DepthMeshKind,
        kind: PassKind,
        variant: Variant,
    ) -> &wgpu::RenderPipeline {
        match depth {
            DepthMeshKind::Collect => &self.collection[&(kind, variant)],
            DepthMeshKind::Filtered => &self.shading[&(kind, variant)],
        }
    }
}

impl Renderer {
    /// Requests a bounded production light-depth page; returns it after GPU mapping.
    #[cfg(feature = "dev")]
    pub fn light_depth_page(
        &self,
        phase: u32,
        cluster: u32,
        offset: u32,
        allowed_bytes: Option<u64>,
    ) -> Option<engine_core::DepthOccupancyPage> {
        self.features.lights.get()?.depth()?.page(
            &self.device,
            &self.queue,
            [phase, cluster, offset],
            allowed_bytes,
        )
    }
    pub(super) fn ensure_depth_variants(&mut self, kind: PassKind) {
        let Some(depth) = self.features.lights.get().and_then(LightPass::depth) else {
            self.mesh_pipelines.depth = None;
            return;
        };
        let mut pipelines = self.mesh_pipelines.depth.take().unwrap_or_default();
        let temporal = kind.slots() == Slots::Temporal;
        let index = usize::from(temporal);
        if pipelines.layouts[index].is_none() {
            let (declaration, poses) = if temporal {
                let Some(pass) = self.features.temporal.get() else {
                    return;
                };
                (
                    &crate::pipeline_layouts::LIGHT_DEPTH_TEMPORAL_PIPELINES,
                    pass.pose_layout(),
                )
            } else {
                (
                    &crate::pipeline_layouts::LIGHT_DEPTH_MESH_PIPELINES,
                    &self.instances.pose_layout,
                )
            };
            pipelines.layouts[index] = Some(crate::pipelines::pipeline_layout(
                &self.device,
                declaration,
                &[
                    &self.camera.layout,
                    self.materials.layout(),
                    poses,
                    &depth.collection_layout,
                ],
            ));
        }
        let Some(layout) = &pipelines.layouts[index] else {
            return;
        };
        for &batch_index in &self.draw_order.order {
            let batch_index = batch_index as usize;
            if self.virtual_draws.drawn.get(batch_index) == Some(&true) {
                continue;
            }
            let batch = &self.plan.batches[batch_index];
            let (_, variant) = self.batch_binding(batch, self.meshes.get(&batch.mesh));
            pipelines
                .collection
                .entry((kind, variant))
                .or_insert_with(|| self.create_depth_collection(kind, variant, layout));
            pipelines
                .shading
                .entry((kind, variant))
                .or_insert_with(|| self.create_shading_variant(kind, variant, true));
        }
        self.mesh_pipelines.depth = Some(pipelines);
    }

    fn create_depth_collection(
        &self,
        kind: PassKind,
        variant: Variant,
        layout: &wgpu::PipelineLayout,
    ) -> wgpu::RenderPipeline {
        let streams = if variant.skinned() {
            Streams::Skinned
        } else if variant.extended() {
            Streams::Extended
        } else {
            Streams::Basic
        };
        let [vertices, ids] = meshes::slot_layouts();
        let buffers = [vertices, ids, meshes::extra_layout(), meshes::skin_layout()];
        let length = match streams {
            Streams::Basic => 2,
            Streams::Extended => 3,
            Streams::Skinned => 4,
        };
        crate::pipelines::geometry(
            &self.device,
            &crate::pipelines::Geometry {
                label: "engine/light-depth-collection",
                layout,
                module: &self.mesh_shader,
                vertex: kind.vertex_entry(streams),
                vertex_constants: &variant.constants(),
                buffers: &buffers[..length],
                primitive: wgpu::PrimitiveState {
                    cull_mode: (!variant.double_sided()).then_some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(crate::pipelines::Fragment {
                    entry: crate::shaders::entry::meshes::fs_light_depth,
                    constants: &[],
                    targets: &[],
                }),
            },
        )
    }
}
