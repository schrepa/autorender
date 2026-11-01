use bevy_ecs::prelude::*;
use engine_core::{
    Detail, DeviceLimits, Error, FrameTime, GlobalTransforms, ImpostorRecords,
    MAX_HIERARCHY_DEPTH, PresentedTransforms, View,
};
use engine_places::probes::{ProbeCaptures, ProbeViews};
use engine_places::shadows::{ShadowCasters, ShadowMaps};
use engine_render::{
    Bind, Buffer, ComputePipeline, Copied, DepthUse, DrawInput, Encode, GpuPart, Install,
    Lifetime, MainView, Pass, Pipeline, Prepare, RenderPipeline, Resolve, Texture, VertexLayout,
    ViewTarget, wgpu,
};

use crate::assets::MeshAssets;
use crate::batches::BatchPlan;
use crate::casters::CasterPlan;
use crate::drawn::WatchedSlot;
use crate::order::{BatchLook, DrawOrder, MeshPlan};
use crate::presentation::{FrameState, PresentationTotals};
use crate::retained::RetainedRender;
use crate::settings::Impostors;
use crate::debug::RECORD_STREAM;
use crate::shaders::meshes::{cards, casters, debug, forward, presentation, pyramid, vertex};

pub(crate) const FACES: usize = 6;
pub(crate) const PYRAMID_LEVELS: usize = 16;
pub(crate) const NODE_LEVELS: usize = MAX_HIERARCHY_DEPTH + 1;
pub(crate) const PYRAMID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

const SLOT_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Uint32];
const SLOT_STREAM: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 4,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &SLOT_ATTRIBUTES,
};
const MESH_STREAMS: &[VertexLayout] = &[
    VertexLayout::Mesh,
    VertexLayout::Buffer(SLOT_STREAM),
    VertexLayout::Buffer(RECORD_STREAM),
];
const EXTENDED_STREAMS: &[VertexLayout] = &[
    VertexLayout::Mesh,
    VertexLayout::Mesh,
    VertexLayout::Buffer(SLOT_STREAM),
    VertexLayout::Buffer(RECORD_STREAM),
];
const VIEW_STREAMS: &[VertexLayout] = &[VertexLayout::Mesh, VertexLayout::Buffer(SLOT_STREAM)];
const EXTENDED_VIEW_STREAMS: &[VertexLayout] = &[
    VertexLayout::Mesh,
    VertexLayout::Mesh,
    VertexLayout::Buffer(SLOT_STREAM),
];
const CASTER_STREAMS: &[VertexLayout] = &[VertexLayout::Mesh, VertexLayout::Buffer(SLOT_STREAM)];
const CARD_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![2 => Uint32, 3 => Uint32];
const CARD_STREAM: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 8,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &CARD_ATTRIBUTES,
};
const CARD_STREAMS: &[VertexLayout] = &[
    VertexLayout::Buffer(CARD_STREAM),
    VertexLayout::Buffer(RECORD_STREAM),
];
const CASTER_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: 1,
    slope_scale: 0.0,
    clamp: 0.0,
};

pub(crate) struct Compaction {
    pub sum: ComputePipeline<MeshesGpu>,
    pub scan: ComputePipeline<MeshesGpu>,
    pub args: ComputePipeline<MeshesGpu>,
    pub scatter: ComputePipeline<MeshesGpu>,
}

pub(crate) struct Passes {
    pub presentation: Pass<MeshesGpu>,
    pub keep_poses: Pass<MeshesGpu>,
    pub occlusion: Pass<MeshesGpu>,
    pub caster_args: Pass<MeshesGpu>,
    pub casters: Pass<MeshesGpu>,
    pub probes: Pass<MeshesGpu>,
}

pub(crate) struct Instances {
    pub records: Buffer<MeshesGpu>,
    pub poses: Buffer<MeshesGpu>,
    pub previous_poses: Buffer<MeshesGpu>,
    pub colors: Buffer<MeshesGpu>,
    pub still_frames: Buffer<MeshesGpu>,
    pub slots: Buffer<MeshesGpu>,
}

pub(crate) struct Presented {
    pub frame: Buffer<MeshesGpu>,
    pub frame_two: Buffer<MeshesGpu>,
    pub faces: Vec<Buffer<MeshesGpu>>,
    pub locals: Buffer<MeshesGpu>,
    pub info: Buffer<MeshesGpu>,
    pub authority: Buffer<MeshesGpu>,
    pub composed: Buffer<MeshesGpu>,
    pub parents: Buffer<MeshesGpu>,
    pub levels: Vec<Buffer<MeshesGpu>>,
    pub node_staging: Buffer<MeshesGpu>,
    pub node_ranges: Buffer<MeshesGpu>,
    pub candidates: Buffer<MeshesGpu>,
    pub slot_candidates: Buffer<MeshesGpu>,
    pub scan: Buffer<MeshesGpu>,
    pub batches: Buffer<MeshesGpu>,
    pub batch_bases: Buffer<MeshesGpu>,
    pub draws: [Buffer<MeshesGpu>; 2],
    pub visible: [Buffer<MeshesGpu>; 2],
    pub face_draws: Vec<Buffer<MeshesGpu>>,
    pub face_visible: Vec<Buffer<MeshesGpu>>,
    pub impostor_ids: [Buffer<MeshesGpu>; 2],
    pub lod_states: Buffer<MeshesGpu>,
    pub fade_cards: Buffer<MeshesGpu>,
    pub counts: Buffer<MeshesGpu>,
    pub totals: Buffer<MeshesGpu>,
    pub pyramid: Texture<MeshesGpu>,
    pub extents: Vec<Buffer<MeshesGpu>>,
}

pub(crate) struct CasterBuffers {
    pub select: Buffer<MeshesGpu>,
    pub stillness: Buffer<MeshesGpu>,
    pub ids: Buffer<MeshesGpu>,
    pub args: Buffer<MeshesGpu>,
    pub template: Buffer<MeshesGpu>,
    pub totals: Buffer<MeshesGpu>,
    pub listed: Buffer<MeshesGpu>,
    pub listed_groups: Buffer<MeshesGpu>,
    pub no_flags: Buffer<MeshesGpu>,
}

pub(crate) struct Computes {
    pub shift: ComputePipeline<MeshesGpu>,
    pub apply: ComputePipeline<MeshesGpu>,
    pub compose: Vec<ComputePipeline<MeshesGpu>>,
    pub cull: ComputePipeline<MeshesGpu>,
    pub count_still: ComputePipeline<MeshesGpu>,
    pub fades: ComputePipeline<MeshesGpu>,
    pub one: Compaction,
    pub two: Compaction,
    pub faces: Vec<Compaction>,
    pub gather_one: ComputePipeline<MeshesGpu>,
    pub gather_two: ComputePipeline<MeshesGpu>,
    pub occlusion: ComputePipeline<MeshesGpu>,
    pub probe: Vec<ComputePipeline<MeshesGpu>>,
    pub pyramid_first: ComputePipeline<MeshesGpu>,
    pub pyramid_down: Vec<ComputePipeline<MeshesGpu>>,
    pub select: [ComputePipeline<MeshesGpu>; 2],
    pub count_casters: ComputePipeline<MeshesGpu>,
}

pub(crate) struct Looked<P> {
    pub looks: Vec<P>,
}

impl<P> Looked<P> {
    pub(crate) fn of(&self, look: BatchLook) -> &P {
        let index = usize::from(look.masked) << 2
            | usize::from(look.double_sided) << 1
            | usize::from(look.extended);
        &self.looks[index]
    }
}

pub(crate) struct Pipelines {
    pub prepass: Looked<Pipeline<MeshesGpu, MainView>>,
    pub opaque: Looked<Pipeline<MeshesGpu, MainView>>,
    pub card: Pipeline<MeshesGpu, MainView>,
    pub probe: Looked<Pipeline<MeshesGpu, ProbeViews>>,
    pub casters: [Pipeline<MeshesGpu, ShadowCasters>; 2],
    pub moving: [Pipeline<MeshesGpu, ShadowCasters>; 2],
}

pub(crate) struct Inputs {
    pub records: DrawInput<MeshesGpu, MainView>,
    pub card_ids: [DrawInput<MeshesGpu, MainView>; 2],
    pub fade_cards: DrawInput<MeshesGpu, MainView>,
    pub slots: DrawInput<MeshesGpu, MainView>,
    pub ids: [DrawInput<MeshesGpu, MainView>; 2],
    pub args: [DrawInput<MeshesGpu, MainView>; 2],
    pub face_ids: Vec<DrawInput<MeshesGpu, ProbeViews>>,
    pub face_args: Vec<DrawInput<MeshesGpu, ProbeViews>>,
    pub caster_ids: DrawInput<MeshesGpu, ShadowCasters>,
    pub caster_args: DrawInput<MeshesGpu, ShadowCasters>,
}

pub(crate) struct MeshesGpu {
    pub passes: Passes,
    pub instances: Instances,
    pub presented: Presented,
    pub casters: CasterBuffers,
    pub computes: Computes,
    pub pipelines: Pipelines,
    pub inputs: Inputs,
    pub state: FrameState,
}

pub(crate) type MeshReads = (
    FrameReads,
    (
        Query<'static, 'static, &'static ImpostorRecords, With<MeshAssets>>,
        Res<'static, WatchedSlot>,
        Res<'static, PresentationTotals>,
    ),
);

pub(crate) type FrameReads = (
    Res<'static, RetainedRender>,
    Res<'static, BatchPlan>,
    Res<'static, DrawOrder>,
    Res<'static, MeshPlan>,
    Res<'static, CasterPlan>,
    Res<'static, PresentedTransforms>,
    Res<'static, GlobalTransforms>,
    Res<'static, View>,
    Res<'static, Detail>,
    Res<'static, Impostors>,
    Res<'static, Resolve>,
    Res<'static, FrameTime>,
    Res<'static, ShadowMaps>,
    Res<'static, ProbeCaptures>,
    Res<'static, DeviceLimits>,
);

impl GpuPart for MeshesGpu {
    type Reads = MeshReads;

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let instances = install_instances(cx)?;
        let presented = install_presented(cx);
        let casters = install_casters(cx);
        let passes = install_passes(cx, &instances, &casters);
        let computes = install_computes(cx, &passes, &instances, &presented, &casters)?;
        let pipelines = install_pipelines(cx)?;
        let inputs = Inputs {
            records: cx.draw_input(MainView, instances.records),
            card_ids: presented.impostor_ids.map(|ids| cx.draw_input(MainView, ids)),
            fade_cards: cx.draw_input(MainView, presented.fade_cards),
            slots: cx.draw_input(MainView, instances.slots),
            ids: presented.visible.map(|ids| cx.draw_input(MainView, ids)),
            args: presented.draws.map(|args| cx.draw_input(MainView, args)),
            face_ids: presented
                .face_visible
                .iter()
                .map(|&ids| cx.draw_input(ProbeViews, ids))
                .collect(),
            face_args: presented
                .face_draws
                .iter()
                .map(|&args| cx.draw_input(ProbeViews, args))
                .collect(),
            caster_ids: cx.draw_input(ShadowCasters, casters.ids),
            caster_args: cx.draw_input(ShadowCasters, casters.args),
        };
        Ok(Self {
            passes,
            instances,
            presented,
            casters,
            computes,
            pipelines,
            inputs,
            state: FrameState::default(),
        })
    }

    fn prepare(
        &mut self,
        reads: bevy_ecs::system::SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.prepare_frame(reads, cx);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        self.encode_pass(pass, cx)
    }
}

fn install_instances(cx: &mut Install<'_, MeshesGpu>) -> Result<Instances, Error> {
    use crate::shaders::meshes::slots;
    let instances = Instances {
        records: cx.buffer("debug_view_records", Lifetime::Kept),
        poses: cx.buffer("poses", Lifetime::Kept),
        previous_poses: cx.buffer("previous_poses", Lifetime::Previous),
        colors: cx.buffer("colors", Lifetime::Kept),
        still_frames: cx.buffer("still_frames", Lifetime::Kept),
        slots: cx.buffer("slots", Lifetime::Kept),
    };
    cx.frame(slots::INSTANCE_POSES, Bind::Buffer(instances.poses))?;
    cx.frame(slots::INSTANCE_PREVIOUS_POSES, Bind::Buffer(instances.previous_poses))?;
    cx.frame(slots::INSTANCE_COLORS, Bind::Buffer(instances.colors))?;
    cx.frame(slots::INSTANCE_STILL, Bind::Buffer(instances.still_frames))?;
    Ok(instances)
}

fn install_presented(cx: &mut Install<'_, MeshesGpu>) -> Presented {
    let mut buffer = |label: &'static str| cx.buffer(label, Lifetime::Kept);
    let frame = buffer("presentation_frame");
    let frame_two = buffer("presentation_frame_two");
    let locals = buffer("presentation_locals");
    let info = buffer("presentation_info");
    let authority = buffer("presentation_authority");
    let composed = buffer("presentation_composed");
    let parents = buffer("presentation_parents");
    let node_staging = buffer("presentation_node_staging");
    let node_ranges = buffer("presentation_node_ranges");
    let candidates = buffer("presentation_candidates");
    let slot_candidates = buffer("presentation_slot_candidates");
    let scan = buffer("presentation_scan");
    let batches = buffer("presentation_batches");
    let batch_bases = buffer("presentation_batch_bases");
    let draws = [buffer("presentation_draws"), buffer("presentation_draws_two")];
    let visible = [buffer("presentation_visible_ids"), buffer("presentation_visible_ids_two")];
    let impostor_ids = [buffer("presentation_impostor_ids"), buffer("presentation_impostor_ids_two")];
    let lod_states = buffer("presentation_lod_states");
    let fade_cards = buffer("presentation_fading_cards");
    let counts = buffer("presentation_occlusion_counts");
    let totals = buffer("presentation_totals");
    let faces = (0..FACES).map(|_| cx.buffer("presentation_frame_probe", Lifetime::Kept)).collect();
    let face_draws = (0..FACES).map(|_| cx.buffer("presentation_draws_probe", Lifetime::Kept)).collect();
    let face_visible = (0..FACES)
        .map(|_| cx.buffer("presentation_visible_ids_probe", Lifetime::Kept))
        .collect();
    let levels = (0..NODE_LEVELS)
        .map(|_| cx.buffer("presentation_level", Lifetime::Kept))
        .collect();
    let extents = (0..PYRAMID_LEVELS)
        .map(|_| cx.buffer("depth_pyramid_extent", Lifetime::Kept))
        .collect();
    let pyramid = cx.texture("depth_pyramid", PYRAMID_FORMAT, Lifetime::Kept);
    Presented {
        frame,
        frame_two,
        faces,
        locals,
        info,
        authority,
        composed,
        parents,
        levels,
        node_staging,
        node_ranges,
        candidates,
        slot_candidates,
        scan,
        batches,
        batch_bases,
        draws,
        visible,
        face_draws,
        face_visible,
        impostor_ids,
        lod_states,
        fade_cards,
        counts,
        totals,
        pyramid,
        extents,
    }
}

fn install_casters(cx: &mut Install<'_, MeshesGpu>) -> CasterBuffers {
    let mut buffer = |label: &'static str| cx.buffer(label, Lifetime::Kept);
    CasterBuffers {
        select: buffer("shadow_select"),
        stillness: buffer("shadow_stillness"),
        ids: buffer("shadow_ids"),
        args: buffer("shadow_args"),
        template: buffer("shadow_args_template"),
        totals: buffer("shadow_counters"),
        listed: buffer("shadow_casters"),
        listed_groups: buffer("shadow_caster_groups"),
        no_flags: buffer("shadow_no_flags"),
    }
}

fn install_passes(
    cx: &mut Install<'_, MeshesGpu>,
    instances: &Instances,
    casters: &CasterBuffers,
) -> Passes {
    let presentation = cx.pass("presentation").id();
    let keep_poses = cx.copy(
        "presentation_keep_poses",
        Copied::Buffer(instances.poses),
        Copied::Buffer(instances.previous_poses),
    );
    let caster_args = cx.copy(
        "shadow_args_template",
        Copied::Buffer(casters.template),
        Copied::Buffer(casters.args),
    );
    let casters_pass = cx
        .pass("shadow_select")
        .after(presentation)
        .after(caster_args)
        .id();
    cx.group(caster_args, "shadow_select");
    cx.group(casters_pass, "shadow_select");
    let occlusion = cx
        .pass("occlusion_test")
        .at(engine_render::Layer::Late)
        .after(presentation)
        .after(casters_pass)
        .id();
    let probes = cx
        .pass("probe_cull")
        .after(occlusion)
        .after(casters_pass)
        .id();
    Passes {
        presentation,
        keep_poses,
        occlusion,
        caster_args,
        casters: casters_pass,
        probes,
    }
}

fn compaction(
    cx: &mut Install<'_, MeshesGpu>,
    pass: Pass<MeshesGpu>,
    presented: &Presented,
    (frame, visible, draws, impostors): (Buffer<MeshesGpu>, Buffer<MeshesGpu>, Buffer<MeshesGpu>, Buffer<MeshesGpu>),
) -> Result<Compaction, Error> {
    use presentation as p;
    Ok(Compaction {
        sum: cx.compute_pipeline(
            pass,
            p::SUM_GROUPS,
            &[
                (p::FRAME, Bind::Buffer(frame)),
                (p::SCAN, Bind::Buffer(presented.scan)),
                (p::OCCLUSION_COUNTS, Bind::Buffer(presented.counts)),
            ],
        )?,
        scan: cx.compute_pipeline(
            pass,
            p::SCAN_GROUPS,
            &[(p::FRAME, Bind::Buffer(frame)), (p::SCAN, Bind::Buffer(presented.scan))],
        )?,
        args: cx.compute_pipeline(
            pass,
            p::BATCH_ARGS,
            &[
                (p::FRAME, Bind::Buffer(frame)),
                (p::BATCHES, Bind::Buffer(presented.batches)),
                (p::SCAN, Bind::Buffer(presented.scan)),
                (p::BATCH_BASES, Bind::Buffer(presented.batch_bases)),
                (p::DRAWS, Bind::Buffer(draws)),
                (p::OCCLUSION_COUNTS, Bind::Buffer(presented.counts)),
            ],
        )?,
        scatter: cx.compute_pipeline(
            pass,
            p::SCATTER,
            &[
                (p::FRAME, Bind::Buffer(frame)),
                (p::SCAN, Bind::Buffer(presented.scan)),
                (p::CANDIDATES, Bind::Buffer(presented.candidates)),
                (p::BATCHES, Bind::Buffer(presented.batches)),
                (p::BATCH_BASES, Bind::Buffer(presented.batch_bases)),
                (p::VISIBLE_IDS, Bind::Buffer(visible)),
                (p::IMPOSTOR_IDS, Bind::Buffer(impostors)),
            ],
        )?,
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "every compute pipeline of the presentation, the pyramid and the caster cull is made here"
)]
fn install_computes(
    cx: &mut Install<'_, MeshesGpu>,
    passes: &Passes,
    instances: &Instances,
    presented: &Presented,
    casters: &CasterBuffers,
) -> Result<Computes, Error> {
    use presentation as p;
    let at = passes.presentation;
    let shift = cx.compute_pipeline(
        at,
        p::SHIFT_PREVIOUS,
        &[
            (p::INFO, Bind::Buffer(presented.info)),
            (p::LOCALS_SHIFT, Bind::Buffer(presented.locals)),
        ],
    )?;
    let apply = cx.compute_pipeline(
        at,
        p::APPLY_LOCALS,
        &[
            (p::NODE_RANGES, Bind::Buffer(presented.node_ranges)),
            (p::NODE_STAGING, Bind::Buffer(presented.node_staging)),
            (p::LOCALS_SHIFT, Bind::Buffer(presented.locals)),
        ],
    )?;
    let mut compose = Vec::with_capacity(NODE_LEVELS);
    for &level in &presented.levels {
        compose.push(cx.compute_pipeline(
            at,
            p::COMPOSE_PARENTS,
            &[
                (p::FRAME, Bind::Buffer(presented.frame)),
                (p::LOCALS, Bind::Buffer(presented.locals)),
                (p::INFO, Bind::Buffer(presented.info)),
                (p::AUTHORITY, Bind::Buffer(presented.authority)),
                (p::COMPOSED, Bind::Buffer(presented.composed)),
                (p::PARENTS, Bind::Buffer(presented.parents)),
                (p::LEVEL, Bind::Buffer(level)),
            ],
        )?);
    }
    let cull = cx.compute_pipeline(
        at,
        p::CULL,
        &[
            (p::FRAME, Bind::Buffer(presented.frame)),
            (p::LOCALS, Bind::Buffer(presented.locals)),
            (p::INFO, Bind::Buffer(presented.info)),
            (p::AUTHORITY, Bind::Buffer(presented.authority)),
            (p::COMPOSED, Bind::Buffer(presented.composed)),
            (p::BATCHES, Bind::Buffer(presented.batches)),
            (p::POSES, Bind::Buffer(instances.poses)),
            (p::SCAN, Bind::Buffer(presented.scan)),
            (p::SLOT_CANDIDATES, Bind::Buffer(presented.slot_candidates)),
        ],
    )?;
    let count_still = cx.compute_pipeline(
        at,
        p::COUNT_STILL,
        &[
            (p::FRAME, Bind::Buffer(presented.frame)),
            (p::SCAN, Bind::Buffer(presented.scan)),
            (p::SLOT_CANDIDATES, Bind::Buffer(presented.slot_candidates)),
            (p::STILL_FRAMES, Bind::Buffer(instances.still_frames)),
        ],
    )?;
    let fades = cx.compute_pipeline(
        at,
        p::FADE_LEVELS,
        &[
            (p::FRAME, Bind::Buffer(presented.frame)),
            (p::BATCHES, Bind::Buffer(presented.batches)),
            (p::SCAN, Bind::Buffer(presented.scan)),
            (p::SLOT_CANDIDATES, Bind::Buffer(presented.slot_candidates)),
            (p::LOD_STATES, Bind::Buffer(presented.lod_states)),
            (p::FADE_CARDS, Bind::Buffer(presented.fade_cards)),
        ],
    )?;
    let one = compaction(
        cx,
        at,
        presented,
        (presented.frame, presented.visible[0], presented.draws[0], presented.impostor_ids[0]),
    )?;
    let gather = |cx: &mut Install<'_, MeshesGpu>, pass, entry, frame| {
        cx.compute_pipeline(
            pass,
            entry,
            &[
                (p::FRAME, Bind::Buffer(frame)),
                (p::SCAN, Bind::Buffer(presented.scan)),
                (p::OCCLUSION_COUNTS, Bind::Buffer(presented.counts)),
                (p::LOD_STATES, Bind::Buffer(presented.lod_states)),
                (p::TOTALS, Bind::Buffer(presented.totals)),
            ],
        )
    };
    let gather_one = gather(cx, at, p::GATHER_PHASE_ONE, presented.frame)?;
    let late = passes.occlusion;
    let two = compaction(
        cx,
        late,
        presented,
        (presented.frame_two, presented.visible[1], presented.draws[1], presented.impostor_ids[1]),
    )?;
    let gather_two = gather(cx, late, p::GATHER_PHASE_TWO, presented.frame_two)?;
    let occlusion = cx.compute_pipeline(
        late,
        p::OCCLUSION_TEST,
        &[
            (p::FRAME, Bind::Buffer(presented.frame)),
            (p::SLOT_CANDIDATES, Bind::Buffer(presented.slot_candidates)),
            (p::SCAN, Bind::Buffer(presented.scan)),
            (p::BATCHES, Bind::Buffer(presented.batches)),
            (p::DEPTH_PYRAMID, Bind::Texture(presented.pyramid)),
            (p::OCCLUSION_COUNTS, Bind::Buffer(presented.counts)),
        ],
    )?;
    let pyramid_first = cx.compute_pipeline(
        late,
        pyramid::PYRAMID_FIRST,
        &[
            (pyramid::SOURCE_DEPTH, Bind::Target(ViewTarget::Depth)),
            (
                pyramid::TARGET_LEVEL,
                Bind::Slice {
                    texture: presented.pyramid,
                    mips: 0..1,
                    layers: 0..1,
                },
            ),
            (pyramid::EXTENT, Bind::Buffer(presented.extents[0])),
        ],
    )?;
    let mut pyramid_down = Vec::with_capacity(PYRAMID_LEVELS - 1);
    for level in 1..PYRAMID_LEVELS as u32 {
        pyramid_down.push(cx.compute_pipeline(
            late,
            pyramid::PYRAMID_DOWN,
            &[
                (
                    pyramid::SOURCE_LEVEL,
                    Bind::Slice {
                        texture: presented.pyramid,
                        mips: level - 1..level,
                        layers: 0..1,
                    },
                ),
                (
                    pyramid::TARGET_LEVEL,
                    Bind::Slice {
                        texture: presented.pyramid,
                        mips: level..level + 1,
                        layers: 0..1,
                    },
                ),
                (pyramid::EXTENT, Bind::Buffer(presented.extents[level as usize])),
            ],
        )?);
    }
    let mut faces = Vec::with_capacity(FACES);
    let mut probe = Vec::with_capacity(FACES);
    for face in 0..FACES {
        let frame = presented.faces[face];
        probe.push(cx.compute_pipeline(
            passes.probes,
            p::PROBE_CULL,
            &[
                (p::FRAME, Bind::Buffer(frame)),
                (p::SLOT_CANDIDATES, Bind::Buffer(presented.slot_candidates)),
                (p::SCAN, Bind::Buffer(presented.scan)),
                (p::BATCHES, Bind::Buffer(presented.batches)),
            ],
        )?);
        faces.push(compaction(
            cx,
            passes.probes,
            presented,
            (frame, presented.face_visible[face], presented.face_draws[face], presented.impostor_ids[0]),
        )?);
    }
    let select_of = |cx: &mut Install<'_, MeshesGpu>, (candidates, groups, flags)| {
        cx.compute_pipeline(
            passes.casters,
            casters::SELECT_CASTERS,
            &[
                (casters::SELECTION, Bind::Buffer(casters.select)),
                (casters::CANDIDATES, Bind::Buffer(candidates)),
                (casters::BATCHES, Bind::Buffer(groups)),
                (casters::POSES, Bind::Buffer(instances.poses)),
                (casters::FLAGS, Bind::Buffer(flags)),
                (casters::STILLNESS, Bind::Buffer(casters.stillness)),
                (casters::IDS, Bind::Buffer(casters.ids)),
                (casters::ARGS, Bind::Buffer(casters.args)),
            ],
        )
    };
    let select = [
        select_of(cx, (presented.candidates, presented.batches, presented.scan))?,
        select_of(cx, (casters.listed, casters.listed_groups, casters.no_flags))?,
    ];
    let count_casters = cx.compute_pipeline(
        passes.casters,
        casters::COUNT_CASTERS,
        &[
            (casters::SELECTION, Bind::Buffer(casters.select)),
            (casters::ARGS, Bind::Buffer(casters.args)),
            (casters::TOTALS, Bind::Buffer(casters.totals)),
        ],
    )?;
    Ok(Computes {
        shift,
        apply,
        compose,
        cull,
        count_still,
        fades,
        one,
        two,
        faces,
        gather_one,
        gather_two,
        occlusion,
        probe,
        pyramid_first,
        pyramid_down,
        select,
        count_casters,
    })
}

fn mesh_pipeline(
    (vertex, fragment, debug): (engine_core::Entry, engine_core::Entry, Option<engine_core::Entry>),
    (vertices, double_sided, mirrored): (&'static [VertexLayout], bool, bool),
    depth: DepthUse,
    writes: wgpu::ColorWrites,
) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex,
        fragment: Some(fragment),
        debug,
        vertices,
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: if mirrored {
                wgpu::FrontFace::Cw
            } else {
                wgpu::FrontFace::Ccw
            },
            cull_mode: (!double_sided).then_some(wgpu::Face::Back),
            ..Default::default()
        },
        depth,
        blend: None,
        writes,
    }
}

fn install_pipelines(cx: &mut Install<'_, MeshesGpu>) -> Result<Pipelines, Error> {
    let write = DepthUse::Write {
        bias: wgpu::DepthBiasState::default(),
    };
    let mut prepass = Vec::with_capacity(8);
    let mut opaque = Vec::with_capacity(8);
    let mut probe = Vec::with_capacity(8);
    for index in 0..8_usize {
        let (masked, double_sided, extended) = (index & 4 != 0, index & 2 != 0, index & 1 != 0);
        let (vertex, view_vertex, streams, view_streams) = if extended {
            (
                vertex::VS_MESH_EXTENDED,
                vertex::VS_VIEW_EXTENDED,
                EXTENDED_STREAMS,
                EXTENDED_VIEW_STREAMS,
            )
        } else {
            (vertex::VS_MESH, vertex::VS_VIEW, MESH_STREAMS, VIEW_STREAMS)
        };
        let masked_depth = if extended {
            forward::FS_MASKED_DEPTH_EXTENDED
        } else {
            forward::FS_MASKED_DEPTH
        };
        prepass.push(cx.render_pipeline(
            MainView,
            &mesh_pipeline(
                (vertex, masked_depth, Some(masked_depth)),
                (streams, double_sided, false),
                write,
                wgpu::ColorWrites::empty(),
            ),
            &[],
        )?);
        let (lit, shown) = match (extended, masked) {
            (false, false) => (forward::FS_MESH, debug::FS_DEBUG_MESH),
            (false, true) => (forward::FS_MESH_PRETESTED, debug::FS_DEBUG_MESH_PRETESTED),
            (true, false) => (forward::FS_MESH_EXTENDED, debug::FS_DEBUG_MESH_EXTENDED),
            (true, true) => (
                forward::FS_MESH_EXTENDED_PRETESTED,
                debug::FS_DEBUG_MESH_EXTENDED_PRETESTED,
            ),
        };
        let depth = if masked {
            DepthUse::Compare {
                compare: wgpu::CompareFunction::Equal,
                write: false,
                bias: wgpu::DepthBiasState::default(),
            }
        } else {
            write
        };
        opaque.push(cx.render_pipeline(
            MainView,
            &mesh_pipeline(
                (vertex, lit, Some(shown)),
                (streams, double_sided, false),
                depth,
                wgpu::ColorWrites::ALL,
            ),
            &[],
        )?);
        let view = if extended {
            forward::FS_PROBE_EXTENDED
        } else {
            forward::FS_PROBE
        };
        probe.push(cx.render_pipeline(
            ProbeViews,
            &mesh_pipeline(
                (view_vertex, view, None),
                (view_streams, double_sided, engine_places::probes::MIRRORED),
                write,
                wgpu::ColorWrites::ALL,
            ),
            &[],
        )?);
    }
    let card = cx.render_pipeline(
        MainView,
        &RenderPipeline {
            vertex: cards::VS_CARD,
            fragment: Some(cards::FS_CARD),
            debug: Some(cards::FS_DEBUG_CARD),
            vertices: CARD_STREAMS,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth: write,
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        },
        &[],
    )?;
    let caster = |cx: &mut Install<'_, MeshesGpu>, vertex, fragment| {
        cx.render_pipeline(
            ShadowCasters,
            &RenderPipeline {
                vertex,
                fragment,
                debug: None,
                vertices: CASTER_STREAMS,
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::Write { bias: CASTER_BIAS },
                blend: None,
                writes: wgpu::ColorWrites::empty(),
            },
            &[],
        )
    };
    let casters_pipelines = [
        caster(cx, casters::VS_CASTER, None)?,
        caster(cx, casters::VS_CASTER_MASKED, Some(casters::FS_CASTER_MASKED))?,
    ];
    let moving = [
        caster(cx, casters::VS_CASTER_MOVING, Some(casters::FS_CASTER_MOVING))?,
        caster(
            cx,
            casters::VS_CASTER_MOVING_MASKED,
            Some(casters::FS_CASTER_MOVING_MASKED),
        )?,
    ];
    Ok(Pipelines {
        prepass: Looked { looks: prepass },
        opaque: Looked { looks: opaque },
        card,
        probe: Looked { looks: probe },
        casters: casters_pipelines,
        moving,
    })
}
