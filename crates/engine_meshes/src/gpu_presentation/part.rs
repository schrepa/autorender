use crate::batches::DrawBatch;
use engine_render::bindings::{Contribution, Stages, uniform, uniform_of};
use engine_render::depth_pyramid::{DepthPyramid, Reduction};
use engine_render::gpu_data::CameraUniform;
use engine_render::graph::{
    FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
};
use engine_render::impostors::ImpostorRecord;
use engine_render::parts::{
    CORE_BRICK, Create, Encode, GpuPart, PartDecl, PartEntry, Prelude, PreludeStage, Step, View,
};
use engine_render::pieces::PIECES_BINDING;
use engine_core::{LevelOfDetail, OcclusionCulling, SubpixelCulling};

use super::{BatchMeta, CullingView, GpuPrepared, PresentationInputs, RecordsKey, Staging};

const FRAME_VIEW: Contribution = Contribution {
    group: "frame",
    set: "view",
    bindings: &[uniform_of(
        "camera",
        size_of::<CameraUniform>(),
        Stages::VERTEX_FRAGMENT_COMPUTE,
    )],
};

const FRAME_PIECES: Contribution = Contribution {
    group: "frame",
    set: "pieces",
    bindings: &[uniform(PIECES_BINDING, Stages::VERTEX_FRAGMENT_COMPUTE)],
};

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "presentation",
        brick: CORE_BRICK,
        passes: &[PassDecl {
            name: "presentation",
            stage: Stage::Prepare,
        }],
        resources: &[
            ResourceDecl {
                name: "poses",
                lifetime: Lifetime::Kept,
            },
            ResourceDecl {
                name: "visible",
                lifetime: Lifetime::Kept,
            },
            ResourceDecl {
                name: "previous_poses",
                lifetime: Lifetime::Kept,
            },
        ],
        hooks: &[],
        contributes: &[FRAME_VIEW, FRAME_PIECES],
        resolve: None,
        labels: &[],
        lazy: false,
        needs: &[],
        piece: None,
    },
    create: create_presentation,
};

fn create_presentation(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(PresentationPart::default())
}

#[derive(Default)]
pub(crate) struct PresentationPart {
    pass: Option<PassId>,
    poses: Option<ResourceId>,
    visible: Option<ResourceId>,
    prepared: Option<GpuPrepared>,
}

impl PresentationPart {
    pub(crate) fn prepared(&self) -> Option<&GpuPrepared> {
        self.prepared.as_ref()
    }
}

impl GpuPart for PresentationPart {
    fn install(&mut self, table: &GraphTable) {
        self.pass = table.pass_named("presentation");
        self.poses = table.resource("poses");
        self.visible = table.resource("visible");
    }

    fn prepare(&mut self, cx: &mut Step<'_>, prelude: &mut Prelude) {
        let mut copies = None;
        self.prepared = prepare_presentation(cx, &mut copies);
        if let Some(encoder) = copies {
            prelude.push_encoder(PreludeStage::Inputs, "pre_inputs", encoder);
        }
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        if !cx.frame.shape.is_some_and(|shape| shape.gpu) {
            return;
        }
        if let Some(pass) = self.pass {
            let _presentation = graph
                .add(cx.table, pass)
                .write(self.poses)
                .write(self.visible);
        }
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        let core = &mut *cx.core;
        core.gpu_presentation
            .encode(&core.device, encoder, &core.profiler);
        0
    }

    fn after_submit(&mut self) {
        self.prepared = None;
    }
}

fn prepare_presentation(
    cx: &mut Step<'_>,
    input_copies: &mut Option<wgpu::CommandEncoder>,
) -> Option<GpuPrepared> {
    let gpu_mode = cx.frame.kind.gpu_mode;
    let frame = cx.frame.frame;
    let snapshot = cx.frame.snapshot;
    let temporal_frame = &cx.frame.temporal;
    let parts = cx.parts;
    let culls = parts.culling().is_some();
    let core = &mut *cx.core;

    let occlusion_enabled = snapshot.section::<OcclusionCulling>().enabled && culls;
    if !occlusion_enabled {
        core.culling.pyramid = None;
        core.gpu_presentation.release_occlusion();
    }
    let region = core.display.region;
    let occlusion = if gpu_mode && occlusion_enabled {
        let device = &core.device;
        let pyramid = core
            .culling
            .pyramid
            .get_or_insert_with(|| DepthPyramid::new(device, Reduction::Farthest));
        Some(pyramid.prepare(device, &core.queue, &core.depth.view, region))
    } else {
        None
    };
    let gpu = frame.gpu()?;
    let meshes = &core.meshes;
    let arena = &core.mesh_arena;
    let impostor_records = &core.impostors.records;
    let batch_meta = |index, batch: &DrawBatch| {
        let mesh = &snapshot.meshes[batch.source_index].mesh;
        let gpu_mesh = &meshes[&batch.mesh];
        let draw = gpu_mesh.draw(arena);
        let levels = engine_render::renderer::lod_levels(gpu_mesh);
        BatchMeta {
            index_count: draw.index_count,
            first_index: draw.first_index,
            base_vertex: draw.base_vertex,
            bounds: mesh.bounds(),
            group: parts.batch_group(index).unwrap_or(0),
            lods: gpu_mesh.plan[..levels - 1]
                .iter()
                .map(|&(offset, count, error)| (draw.first_index + offset, count, error))
                .collect(),
            impostor: impostor_records
                .get(index)
                .map_or(0.0, ImpostorRecord::error),
        }
    };
    let level_of_detail = snapshot.section::<LevelOfDetail>();
    let records = RecordsKey {
        groups: batch_groups(parts, core.plan.batches.len()),
        meshes: core.mesh_layout,
        impostors: if level_of_detail.impostors {
            core.impostors.atlas.generation() + 1
        } else {
            0
        },
    };
    let view = CullingView {
        view_projection: temporal_frame
            .as_ref()
            .map_or(snapshot.view_projection, |temporal| temporal.rasterized),
        eye: snapshot.detail_eye,
        viewport: region.render.pixels(),
        motion: temporal_frame
            .as_ref()
            .is_some_and(|temporal| temporal.motion_valid),
        occlusion,
        subpixel: snapshot.section::<SubpixelCulling>().enabled,
        lod_error: if level_of_detail.enabled {
            level_of_detail.pixel_error * region.share()[1]
        } else {
            0.0
        },
        impostors: level_of_detail.impostors,
        stillness: parts.views().map(|views| views.still),
        fade: core.switch_fades.share,
    };
    let parity = temporal_frame.as_ref().map_or(0, |frame| frame.parity);
    let (ring, poses) = core.instances.presentation_inputs(parity);
    Some(core.gpu_presentation.prepare(
        &core.device,
        &core.queue,
        Staging {
            ring,
            copies: input_copies,
        },
        &PresentationInputs {
            frame: &frame,
            gpu: &gpu,
            plan: &core.plan,
            batch_meta: &batch_meta,
            records,
            poses,
        },
        &view,
        &core.profiler,
    ))
}

fn batch_groups(parts: &engine_render::parts::Parts, batches: usize) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    for index in 0..batches {
        parts.batch_group(index).hash(&mut hasher);
    }
    hasher.finish()
}
