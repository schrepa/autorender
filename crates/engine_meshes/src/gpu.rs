use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, AssetResidency, BlendedPass, ColorPass, DepthPass, Detail, DeviceLimits,
    DrawnPoses, DrawnTable, Error, FrameTime, GlobalTransforms, ImpostorRecords,
    MAX_HIERARCHY_DEPTH, PassDepth, PassName, PassViews, PrimaryView, ScenePass, Views,
};
use engine_render::{
    AttachPoint, Bind, Buffer, ComputePipeline, Copied, DepthUse, DrawInput, Encode, GpuPart,
    Install, Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, Scene, Scope,
    TargetName, Texture, VertexLayout, wgpu,
};

use crate::MESH;
use crate::assets::MeshAssets;
use crate::debug::RECORD_STREAM;
use crate::drawn::WatchedSlot;
use crate::layouts::PresentationFrame;
use crate::order::{BatchLook, MeshPlan};
use crate::presentation::{FrameState, PresentationTotals};
use crate::settings::Impostors;
use crate::shaders::meshes::{
    blended, cards, debug, depth_views, forward, presentation, pyramid, vertex,
};

pub(crate) const PYRAMID_LEVELS: usize = 16;
pub(crate) const NODE_LEVELS: usize = MAX_HIERARCHY_DEPTH + 1;
pub(crate) const PYRAMID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
pub(crate) const VIEW_FRAME_BYTES: u64 = 512;

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
    pub occluded: Pass<MeshesGpu>,
    pub views: Pass<MeshesGpu>,
    pub blended: Pass<MeshesGpu>,
}

pub(crate) struct Sorted {
    pub rows: Buffer<MeshesGpu>,
    pub view: Buffer<MeshesGpu>,
    pub keys: Buffer<MeshesGpu>,
    pub ids: Buffer<MeshesGpu>,
    pub args: Buffer<MeshesGpu>,
    pub keyed: ComputePipeline<MeshesGpu>,
    pub ranked: ComputePipeline<MeshesGpu>,
}

pub(crate) struct Instances {
    pub records: Buffer<MeshesGpu>,
    pub impostors: Buffer<MeshesGpu>,
    pub poses: Buffer<MeshesGpu>,
    pub previous_poses: Buffer<MeshesGpu>,
    pub colors: Buffer<MeshesGpu>,
}

pub(crate) struct Presented {
    pub frame: Buffer<MeshesGpu>,
    pub frame_two: Buffer<MeshesGpu>,
    pub view_frames: Buffer<MeshesGpu>,
    pub locals: Buffer<MeshesGpu>,
    pub info: Buffer<MeshesGpu>,
    pub authority: Buffer<MeshesGpu>,
    pub composed: Buffer<MeshesGpu>,
    pub parents: Buffer<MeshesGpu>,
    pub levels: Vec<Buffer<MeshesGpu>>,
    pub candidates: Buffer<MeshesGpu>,
    pub slot_candidates: Buffer<MeshesGpu>,
    pub scan: Buffer<MeshesGpu>,
    pub batches: Buffer<MeshesGpu>,
    pub batch_bases: Buffer<MeshesGpu>,
    pub draws: [Buffer<MeshesGpu>; 2],
    pub visible: [Buffer<MeshesGpu>; 2],
    pub view_draws: Buffer<MeshesGpu>,
    pub view_visible: Buffer<MeshesGpu>,
    pub impostor_ids: [Buffer<MeshesGpu>; 2],
    pub lod_states: Buffer<MeshesGpu>,
    pub fade_cards: Buffer<MeshesGpu>,
    pub counts: Buffer<MeshesGpu>,
    pub totals: Buffer<MeshesGpu>,
    pub pyramid: Texture<MeshesGpu>,
    pub extents: Vec<Buffer<MeshesGpu>>,
}

pub(crate) struct Computes {
    pub compose: Vec<ComputePipeline<MeshesGpu>>,
    pub cull: ComputePipeline<MeshesGpu>,
    pub fades: ComputePipeline<MeshesGpu>,
    pub one: Compaction,
    pub two: Compaction,
    pub views: Compaction,
    pub gather_one: ComputePipeline<MeshesGpu>,
    pub gather_two: ComputePipeline<MeshesGpu>,
    pub occlusion: ComputePipeline<MeshesGpu>,
    pub view_cull: ComputePipeline<MeshesGpu>,
    pub pyramid_first: ComputePipeline<MeshesGpu>,
    pub pyramid_down: Vec<ComputePipeline<MeshesGpu>>,
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

pub(crate) struct Streams<S> {
    pub ids: DrawInput<MeshesGpu, S>,
    pub args: DrawInput<MeshesGpu, S>,
    pub records: DrawInput<MeshesGpu, S>,
    pub cards: DrawInput<MeshesGpu, S>,
    pub fading: DrawInput<MeshesGpu, S>,
}

pub(crate) struct ViewStreams {
    pub ids: DrawInput<MeshesGpu, Scene>,
    pub args: DrawInput<MeshesGpu, Scene>,
}

pub(crate) struct BlendedStreams {
    pub ids: DrawInput<MeshesGpu, Scene>,
    pub args: DrawInput<MeshesGpu, Scene>,
    pub records: DrawInput<MeshesGpu, Scene>,
}

pub(crate) enum Drawing {
    Lit {
        looks: Looked<Pipeline<MeshesGpu, Scene>>,
        card: Pipeline<MeshesGpu, Scene>,
        streams: Streams<Scene>,
    },
    Depth {
        looks: Looked<Pipeline<MeshesGpu, Scene>>,
        streams: Streams<Scene>,
    },
    DepthViews {
        pipelines: [Pipeline<MeshesGpu, Scene>; 2],
        streams: ViewStreams,
    },
    Views {
        looks: [Looked<Pipeline<MeshesGpu, Scene>>; 2],
        streams: ViewStreams,
    },
    Blended {
        looks: Looked<Pipeline<MeshesGpu, Scene>>,
        streams: BlendedStreams,
    },
}

pub(crate) struct ScenePipelines {
    pub name: &'static str,
    pub drawing: Drawing,
}

pub(crate) struct Occluded {
    pub depth: Looked<Pipeline<MeshesGpu, Own>>,
    pub lit: Looked<Pipeline<MeshesGpu, Own>>,
    pub card: Pipeline<MeshesGpu, Own>,
    pub streams: Streams<Own>,
}

pub(crate) struct MeshesGpu {
    pub passes: Passes,
    pub instances: Instances,
    pub presented: Presented,
    pub computes: Computes,
    pub sorted: Sorted,
    pub scenes: Vec<ScenePipelines>,
    pub occluded: Occluded,
    pub state: FrameState,
}

pub(crate) type MeshReads = (
    FrameReads,
    (
        Query<'static, 'static, &'static ImpostorRecords, With<MeshAssets>>,
        Res<'static, Views>,
        Res<'static, WatchedSlot>,
        Res<'static, PresentationTotals>,
        Res<'static, AssetResidency>,
    ),
);

pub(crate) type FrameReads = (
    Res<'static, DrawnTable>,
    Res<'static, AssetLibrary>,
    Res<'static, MeshPlan>,
    Res<'static, DrawnPoses>,
    Res<'static, GlobalTransforms>,
    Res<'static, PrimaryView>,
    Res<'static, Detail>,
    Res<'static, Impostors>,
    Res<'static, FrameTime>,
    Res<'static, DeviceLimits>,
);

impl GpuPart for MeshesGpu {
    type Reads = MeshReads;

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let instances = install_instances(cx)?;
        let presented = install_presented(cx);
        let passes = install_passes(cx, &instances);
        let computes = install_computes(cx, &passes, &instances, &presented)?;
        let sorted = install_sorted(cx, passes.blended, &presented)?;
        let scenes = install_scenes(cx, &instances, &presented, &sorted)?;
        let occluded = install_occluded(cx, passes.occluded, &instances, &presented)?;
        Ok(Self {
            passes,
            instances,
            presented,
            computes,
            sorted,
            scenes,
            occluded,
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

    fn view(&mut self, _view: &engine_core::ViewFacts, cx: &mut Prepare<'_, Self>) {
        self.prepare_blended_view(cx);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        self.encode_pass(pass, cx)
    }
}

fn install_instances(cx: &mut Install<'_, MeshesGpu>) -> Result<Instances, Error> {
    use crate::shaders::meshes::slots;
    let instances = Instances {
        records: cx.buffer("debug_view_records", Lifetime::Kept),
        impostors: cx.buffer("impostor_records", Lifetime::Kept),
        poses: cx.buffer("poses", Lifetime::Kept),
        previous_poses: cx.buffer("previous_poses", Lifetime::Previous),
        colors: cx.buffer("colors", Lifetime::Kept),
    };
    cx.frame(slots::INSTANCE_POSES, Bind::Buffer(instances.poses))?;
    cx.frame(
        slots::INSTANCE_PREVIOUS_POSES,
        Bind::Buffer(instances.previous_poses),
    )?;
    cx.frame(slots::INSTANCE_COLORS, Bind::Buffer(instances.colors))?;
    cx.frame(cards::IMPOSTOR_RECORDS, Bind::Buffer(instances.impostors))?;
    Ok(instances)
}

fn install_presented(cx: &mut Install<'_, MeshesGpu>) -> Presented {
    let mut buffer = |label: &'static str| cx.buffer(label, Lifetime::Kept);
    let frame = buffer("presentation_frame");
    let frame_two = buffer("presentation_frame_two");
    let view_frames = buffer("presentation_view_frames");
    let locals = buffer("presentation_locals");
    let info = buffer("presentation_info");
    let authority = buffer("presentation_authority");
    let composed = buffer("presentation_composed");
    let parents = buffer("presentation_parents");
    let candidates = buffer("presentation_candidates");
    let slot_candidates = buffer("presentation_slot_candidates");
    let scan = buffer("presentation_scan");
    let batches = buffer("presentation_batches");
    let batch_bases = buffer("presentation_batch_bases");
    let draws = [
        buffer("presentation_draws"),
        buffer("presentation_draws_two"),
    ];
    let visible = [
        buffer("presentation_visible_ids"),
        buffer("presentation_visible_ids_two"),
    ];
    let view_draws = buffer("presentation_view_draws");
    let view_visible = buffer("presentation_view_ids");
    let impostor_ids = [
        buffer("presentation_impostor_ids"),
        buffer("presentation_impostor_ids_two"),
    ];
    let lod_states = buffer("presentation_lod_states");
    let fade_cards = buffer("presentation_fading_cards");
    let counts = buffer("presentation_occlusion_counts");
    let totals = buffer("presentation_totals");
    let levels = (0..NODE_LEVELS)
        .map(|_| cx.buffer("presentation_level", Lifetime::Kept))
        .collect();
    let extents = (0..PYRAMID_LEVELS)
        .map(|_| cx.buffer("depth_pyramid_extent", Lifetime::Kept))
        .collect();
    let pyramid = cx.texture("depth_pyramid", PYRAMID_FORMAT, Lifetime::Kept);
    cx.window(view_frames, size_of::<PresentationFrame>() as u64);
    Presented {
        frame,
        frame_two,
        view_frames,
        locals,
        info,
        authority,
        composed,
        parents,
        levels,
        candidates,
        slot_candidates,
        scan,
        batches,
        batch_bases,
        draws,
        visible,
        view_draws,
        view_visible,
        impostor_ids,
        lod_states,
        fade_cards,
        counts,
        totals,
        pyramid,
        extents,
    }
}

fn install_passes(cx: &mut Install<'_, MeshesGpu>, instances: &Instances) -> Passes {
    let presentation = cx.pass("presentation").id();
    let keep_poses = cx.copy(
        "presentation_keep_poses",
        Copied::Buffer(instances.poses),
        Copied::Buffer(instances.previous_poses),
    );
    let occlusion = cx
        .pass("occlusion_test")
        .at(AttachPoint::Opaque)
        .after(presentation)
        .id();
    let occluded = cx
        .pass("occluded")
        .at(AttachPoint::Opaque)
        .after(occlusion)
        .id();
    let views = cx.pass("view_cull").after(presentation).id();
    let blended = cx
        .pass("blended_sort")
        .at(AttachPoint::Transparent)
        .after(presentation)
        .id();
    Passes {
        presentation,
        keep_poses,
        occlusion,
        occluded,
        views,
        blended,
    }
}

fn install_sorted(
    cx: &mut Install<'_, MeshesGpu>,
    pass: Pass<MeshesGpu>,
    presented: &Presented,
) -> Result<Sorted, Error> {
    let rows = cx.buffer("blended_rows", Lifetime::Kept);
    let view = cx.view_buffer("blended_view", Lifetime::Kept);
    let keys = cx.view_buffer("blended_keys", Lifetime::Kept);
    let ids = cx.view_buffer("blended_ids", Lifetime::Kept);
    let args = cx.view_buffer("blended_args", Lifetime::Kept);
    let keyed = cx.compute_pipeline(
        pass,
        blended::BLENDED_KEYS,
        &[
            (blended::SORT_VIEW, Bind::Buffer(view)),
            (blended::SORT_ROWS, Bind::Buffer(rows)),
            (blended::SORT_BATCHES, Bind::Buffer(presented.batches)),
            (blended::SORT_KEYS, Bind::Buffer(keys)),
        ],
    )?;
    let ranked = cx.compute_pipeline(
        pass,
        blended::BLENDED_RANK,
        &[
            (blended::SORT_VIEW, Bind::Buffer(view)),
            (blended::SORT_ROWS, Bind::Buffer(rows)),
            (blended::SORT_BATCHES, Bind::Buffer(presented.batches)),
            (blended::SORT_KEYS, Bind::Buffer(keys)),
            (blended::SORT_IDS, Bind::Buffer(ids)),
            (blended::SORT_ARGS, Bind::Buffer(args)),
        ],
    )?;
    Ok(Sorted {
        rows,
        view,
        keys,
        ids,
        args,
        keyed,
        ranked,
    })
}

fn compaction(
    cx: &mut Install<'_, MeshesGpu>,
    pass: Pass<MeshesGpu>,
    presented: &Presented,
    (frame, visible, draws, impostors): (
        Buffer<MeshesGpu>,
        Buffer<MeshesGpu>,
        Buffer<MeshesGpu>,
        Buffer<MeshesGpu>,
    ),
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
            &[
                (p::FRAME, Bind::Buffer(frame)),
                (p::SCAN, Bind::Buffer(presented.scan)),
            ],
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
    reason = "every compute pipeline of the presentation, the pyramid and the view cull is made here"
)]
fn install_computes(
    cx: &mut Install<'_, MeshesGpu>,
    passes: &Passes,
    instances: &Instances,
    presented: &Presented,
) -> Result<Computes, Error> {
    use presentation as p;
    let at = passes.presentation;
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
        (
            presented.frame,
            presented.visible[0],
            presented.draws[0],
            presented.impostor_ids[0],
        ),
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
        (
            presented.frame_two,
            presented.visible[1],
            presented.draws[1],
            presented.impostor_ids[1],
        ),
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
            (pyramid::SOURCE_DEPTH, Bind::Target(TargetName::DEPTH)),
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
                (
                    pyramid::EXTENT,
                    Bind::Buffer(presented.extents[level as usize]),
                ),
            ],
        )?);
    }
    let view_cull = cx.compute_pipeline(
        passes.views,
        p::VIEW_CULL,
        &[
            (p::FRAME, Bind::Buffer(presented.view_frames)),
            (p::SLOT_CANDIDATES, Bind::Buffer(presented.slot_candidates)),
            (p::SCAN, Bind::Buffer(presented.scan)),
            (p::BATCHES, Bind::Buffer(presented.batches)),
        ],
    )?;
    let views = compaction(
        cx,
        passes.views,
        presented,
        (
            presented.view_frames,
            presented.view_visible,
            presented.view_draws,
            presented.impostor_ids[0],
        ),
    )?;
    Ok(Computes {
        compose,
        cull,
        fades,
        one,
        two,
        views,
        gather_one,
        gather_two,
        occlusion,
        view_cull,
        pyramid_first,
        pyramid_down,
    })
}

fn mesh_pipeline(
    (vertex, fragment, debug): (
        engine_core::Entry,
        engine_core::Entry,
        Option<engine_core::Entry>,
    ),
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

const WRITE: DepthUse = DepthUse::Write {
    bias: wgpu::DepthBiasState {
        constant: 0,
        slope_scale: 0.0,
        clamp: 0.0,
    },
};

fn look_of(index: usize) -> (bool, bool, bool) {
    (index & 4 != 0, index & 2 != 0, index & 1 != 0)
}

fn depth_looks<S: Scope<MeshesGpu> + Copy>(
    cx: &mut Install<'_, MeshesGpu>,
    scope: S,
) -> Result<Looked<Pipeline<MeshesGpu, S::Tag>>, Error> {
    let mut looks = Vec::with_capacity(8);
    for index in 0..8_usize {
        let (_, double_sided, extended) = look_of(index);
        let (vertex, streams, masked_depth) = if extended {
            (
                vertex::VS_MESH_EXTENDED,
                EXTENDED_STREAMS,
                forward::FS_MASKED_DEPTH_EXTENDED,
            )
        } else {
            (vertex::VS_MESH, MESH_STREAMS, forward::FS_MASKED_DEPTH)
        };
        looks.push(cx.render_pipeline(
            scope,
            &mesh_pipeline(
                (vertex, masked_depth, Some(masked_depth)),
                (streams, double_sided, false),
                WRITE,
                wgpu::ColorWrites::empty(),
            ),
            &[],
        )?);
    }
    Ok(Looked { looks })
}

fn lit_looks<S: Scope<MeshesGpu> + Copy>(
    cx: &mut Install<'_, MeshesGpu>,
    scope: S,
) -> Result<Looked<Pipeline<MeshesGpu, S::Tag>>, Error> {
    let mut looks = Vec::with_capacity(8);
    for index in 0..8_usize {
        let (masked, double_sided, extended) = look_of(index);
        let (vertex, streams) = if extended {
            (vertex::VS_MESH_EXTENDED, EXTENDED_STREAMS)
        } else {
            (vertex::VS_MESH, MESH_STREAMS)
        };
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
            WRITE
        };
        looks.push(cx.render_pipeline(
            scope,
            &mesh_pipeline(
                (vertex, lit, Some(shown)),
                (streams, double_sided, false),
                depth,
                wgpu::ColorWrites::ALL,
            ),
            &[],
        )?);
    }
    Ok(Looked { looks })
}

fn blended_looks(
    cx: &mut Install<'_, MeshesGpu>,
    scene: Scene,
) -> Result<Looked<Pipeline<MeshesGpu, Scene>>, Error> {
    let mut looks = Vec::with_capacity(8);
    for index in 0..8_usize {
        let (_, double_sided, extended) = look_of(index);
        let (vertex, streams, shown) = if extended {
            (
                vertex::VS_MESH_EXTENDED,
                EXTENDED_STREAMS,
                forward::FS_MESH_EXTENDED_BLENDED,
            )
        } else {
            (vertex::VS_MESH, MESH_STREAMS, forward::FS_MESH_BLENDED)
        };
        looks.push(cx.render_pipeline(
            scene,
            &mesh_pipeline(
                (vertex, shown, None),
                (streams, double_sided, false),
                WRITE,
                wgpu::ColorWrites::ALL,
            ),
            &[],
        )?);
    }
    Ok(Looked { looks })
}

fn view_looks(
    cx: &mut Install<'_, MeshesGpu>,
    scene: Scene,
    mirrored: bool,
) -> Result<Looked<Pipeline<MeshesGpu, Scene>>, Error> {
    let mut looks = Vec::with_capacity(8);
    for index in 0..8_usize {
        let (_, double_sided, extended) = look_of(index);
        let (vertex, streams, shown) = if extended {
            (
                vertex::VS_VIEW_EXTENDED,
                EXTENDED_VIEW_STREAMS,
                forward::FS_VIEW_EXTENDED,
            )
        } else {
            (vertex::VS_VIEW, VIEW_STREAMS, forward::FS_VIEW)
        };
        looks.push(cx.render_pipeline(
            scene,
            &mesh_pipeline(
                (vertex, shown, None),
                (streams, double_sided, mirrored),
                WRITE,
                wgpu::ColorWrites::ALL,
            ),
            &[],
        )?);
    }
    Ok(Looked { looks })
}

fn card_pipeline<S: Scope<MeshesGpu>>(
    cx: &mut Install<'_, MeshesGpu>,
    scope: S,
) -> Result<Pipeline<MeshesGpu, S::Tag>, Error> {
    cx.render_pipeline(
        scope,
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
            depth: WRITE,
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        },
        &[],
    )
}

fn depth_view_pipelines(
    cx: &mut Install<'_, MeshesGpu>,
    scene: Scene,
    depth: PassDepth,
) -> Result<[Pipeline<MeshesGpu, Scene>; 2], Error> {
    let (plain, masked) = if depth.clamped {
        (
            depth_views::VS_DEPTH_VIEW_CLAMPED,
            depth_views::VS_DEPTH_VIEW_MASKED_CLAMPED,
        )
    } else {
        (depth_views::VS_DEPTH_VIEW, depth_views::VS_DEPTH_VIEW_MASKED)
    };
    let bias = wgpu::DepthBiasState {
        constant: depth.bias,
        slope_scale: depth.slope,
        clamp: 0.0,
    };
    let pipeline = |cx: &mut Install<'_, MeshesGpu>, vertex, fragment| {
        cx.render_pipeline(
            scene,
            &RenderPipeline {
                vertex,
                fragment,
                debug: None,
                vertices: VIEW_STREAMS,
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::Write { bias },
                blend: None,
                writes: wgpu::ColorWrites::empty(),
            },
            &[],
        )
    };
    Ok([
        pipeline(cx, plain, None)?,
        pipeline(cx, masked, Some(depth_views::FS_DEPTH_VIEW_MASKED))?,
    ])
}

fn streams<S: Scope<MeshesGpu> + Copy>(
    cx: &mut Install<'_, MeshesGpu>,
    scope: S,
    (instances, presented): (&Instances, &Presented),
    phase: usize,
) -> Streams<S::Tag> {
    Streams {
        ids: cx.draw_input(scope, presented.visible[phase]),
        args: cx.draw_input(scope, presented.draws[phase]),
        records: cx.draw_input(scope, instances.records),
        cards: cx.draw_input(scope, presented.impostor_ids[phase]),
        fading: cx.draw_input(scope, presented.fade_cards),
    }
}

fn install_scenes(
    cx: &mut Install<'_, MeshesGpu>,
    instances: &Instances,
    presented: &Presented,
    sorted: &Sorted,
) -> Result<Vec<ScenePipelines>, Error> {
    let listed: Vec<ScenePass> = cx
        .passes()
        .listed()
        .iter()
        .filter(|listed| listed.bit.is_some() && (listed.pass.takes)(MESH))
        .copied()
        .collect();
    let mut scenes = Vec::with_capacity(listed.len());
    for listed in listed {
        let scene = Scene::named(listed.name);
        let owned = listed.pass.views == PassViews::Owned;
        let drawing = if listed.name == ColorPass::NAME {
            Drawing::Lit {
                looks: lit_looks(cx, scene)?,
                card: card_pipeline(cx, scene)?,
                streams: streams(cx, scene, (instances, presented), 0),
            }
        } else if listed.name == DepthPass::NAME {
            Drawing::Depth {
                looks: depth_looks(cx, scene)?,
                streams: streams(cx, scene, (instances, presented), 0),
            }
        } else if listed.name == BlendedPass::NAME {
            Drawing::Blended {
                looks: blended_looks(cx, scene)?,
                streams: BlendedStreams {
                    ids: cx.draw_input(scene, sorted.ids),
                    args: cx.draw_input(scene, sorted.args),
                    records: cx.draw_input(scene, instances.records),
                },
            }
        } else if owned && !listed.pass.reads.base_color {
            Drawing::DepthViews {
                pipelines: depth_view_pipelines(cx, scene, listed.depth)?,
                streams: ViewStreams {
                    ids: cx.draw_input(scene, presented.view_visible),
                    args: cx.draw_input(scene, presented.view_draws),
                },
            }
        } else if owned {
            Drawing::Views {
                looks: [view_looks(cx, scene, false)?, view_looks(cx, scene, true)?],
                streams: ViewStreams {
                    ids: cx.draw_input(scene, presented.view_visible),
                    args: cx.draw_input(scene, presented.view_draws),
                },
            }
        } else {
            continue;
        };
        scenes.push(ScenePipelines {
            name: listed.name,
            drawing,
        });
    }
    Ok(scenes)
}

fn install_occluded(
    cx: &mut Install<'_, MeshesGpu>,
    pass: Pass<MeshesGpu>,
    instances: &Instances,
    presented: &Presented,
) -> Result<Occluded, Error> {
    Ok(Occluded {
        depth: depth_looks(cx, pass)?,
        lit: lit_looks(cx, pass)?,
        card: card_pipeline(cx, pass)?,
        streams: streams(cx, pass, (instances, presented), 1),
    })
}
