use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, AssetResidency, CullPlan, Detail, DeviceLimits, DrawnTable, Error, FrameTime,
    GeometryStream, GlobalTransforms, InstanceInput, PassRole, Views,
};
use engine_places::meshes::LastLevels;
use engine_places::skin::{JOINTS, WEIGHTS};
use engine_render::{
    AttachPoint, Bind, Buffer, ComputePipeline, DrawInput, Encode, Form, FormFragment, FormSide,
    FragmentPass, FragmentUse, GpuPart, Install, KindForm, Lifetime, Own, Pass, PassFragment,
    PerPass, Prepare, Scene, Scope, Varyings, VertexLayout, wgpu,
};

use crate::assets::MeshAssets;
use crate::debug::RECORD_STREAM;
use crate::drawn::WatchedSlot;
use crate::layouts::{JOINTS_AT, PresentationFrame, WEIGHTS_AT};
use crate::order::BatchLook;
use crate::presentation::{FrameState, PresentationTotals};
use crate::shaders::meshes::{presentation, vertex};
use crate::{MESH, SKIN};

pub(crate) const VIEW_FRAME_BYTES: u64 = 512;

const fn instance_input(input: InstanceInput, offset: u64) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset,
        shader_location: input.location(),
    }
}

const VERTEX: VertexLayout = VertexLayout::Geometry(GeometryStream::Vertex);
const EXTRA: VertexLayout = VertexLayout::Geometry(GeometryStream::Extra);
const JOINTS_STREAM: VertexLayout = VertexLayout::Geometry(GeometryStream::Named {
    shape: JOINTS,
    location: JOINTS_AT,
});
const WEIGHTS_STREAM: VertexLayout = VertexLayout::Geometry(GeometryStream::Named {
    shape: WEIGHTS,
    location: WEIGHTS_AT,
});
const SLOT_ATTRIBUTES: [wgpu::VertexAttribute; 1] = [instance_input(InstanceInput::Slot, 0)];
const SLOT_STREAM: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 4,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &SLOT_ATTRIBUTES,
};
const MESH_STREAMS: &[VertexLayout] = &[
    VERTEX,
    VertexLayout::Buffer(SLOT_STREAM),
    VertexLayout::Buffer(RECORD_STREAM),
];
const EXTENDED_STREAMS: &[VertexLayout] = &[
    VERTEX,
    EXTRA,
    VertexLayout::Buffer(SLOT_STREAM),
    VertexLayout::Buffer(RECORD_STREAM),
];
const SKIN_STREAMS: &[VertexLayout] = &[
    VERTEX,
    EXTRA,
    JOINTS_STREAM,
    WEIGHTS_STREAM,
    VertexLayout::Buffer(SLOT_STREAM),
    VertexLayout::Buffer(RECORD_STREAM),
];
const VIEW_STREAMS: &[VertexLayout] = &[VERTEX, VertexLayout::Buffer(SLOT_STREAM)];
const EXTENDED_VIEW_STREAMS: &[VertexLayout] = &[VERTEX, EXTRA, VertexLayout::Buffer(SLOT_STREAM)];
const SKIN_VIEW_STREAMS: &[VertexLayout] = &[
    VERTEX,
    EXTRA,
    JOINTS_STREAM,
    WEIGHTS_STREAM,
    VertexLayout::Buffer(SLOT_STREAM),
];
pub(crate) struct Compaction {
    pub sum: ComputePipeline<MeshesGpu>,
    pub scan: ComputePipeline<MeshesGpu>,
    pub args: ComputePipeline<MeshesGpu>,
    pub scatter: ComputePipeline<MeshesGpu>,
}

pub(crate) struct Passes {
    pub presentation: Pass<MeshesGpu>,
    pub occlusion: Pass<MeshesGpu>,
    pub occluded: Pass<MeshesGpu>,
    pub views: Pass<MeshesGpu>,
}

pub(crate) struct Instances {
    pub records: Buffer<MeshesGpu>,
    pub colors: Buffer<MeshesGpu>,
}

pub(crate) struct Presented {
    pub frame: Buffer<MeshesGpu>,
    pub frame_two: Buffer<MeshesGpu>,
    pub view_frames: Buffer<MeshesGpu>,
    pub candidates: Buffer<MeshesGpu>,
    pub slot_candidates: Buffer<MeshesGpu>,
    pub scan: Buffer<MeshesGpu>,
    pub batches: Buffer<MeshesGpu>,
    pub batch_bases: Buffer<MeshesGpu>,
    pub draws: [Buffer<MeshesGpu>; 2],
    pub visible: [Buffer<MeshesGpu>; 2],
    pub view_draws: Buffer<MeshesGpu>,
    pub view_visible: Buffer<MeshesGpu>,
    pub last_ids: [Buffer<MeshesGpu>; 2],
    pub lod_states: Buffer<MeshesGpu>,
    pub fading_ids: Buffer<MeshesGpu>,
    pub counts: Buffer<MeshesGpu>,
    pub totals: Buffer<MeshesGpu>,
}

pub(crate) struct Computes {
    pub cull: ComputePipeline<MeshesGpu>,
    pub fades: ComputePipeline<MeshesGpu>,
    pub one: Compaction,
    pub two: Compaction,
    pub views: Compaction,
    pub gather_one: ComputePipeline<MeshesGpu>,
    pub gather_two: ComputePipeline<MeshesGpu>,
    pub occlusion: ComputePipeline<MeshesGpu>,
    pub view_cull: ComputePipeline<MeshesGpu>,
}

pub(crate) struct Forms<P> {
    pub plain: P,
    pub extended: P,
    pub skin: P,
}

impl<P> Forms<P> {
    pub(crate) fn of(&self, look: BatchLook) -> &P {
        match (look.skinned, look.extended) {
            (true, _) => &self.skin,
            (false, true) => &self.extended,
            (false, false) => &self.plain,
        }
    }
}

const PLAIN_FORM: KindForm = KindForm {
    kind: MESH,
    placed: FormSide {
        vertex: vertex::VS_MESH,
        streams: MESH_STREAMS,
        depth_bias: FormSide::UNBIASED,
    },
    owned: Some(FormSide {
        vertex: vertex::VS_VIEW,
        streams: VIEW_STREAMS,
        depth_bias: FormSide::UNBIASED,
    }),
    varyings: Varyings::Surface(FragmentUse::Masked),
};

const EXTENDED_FORM: KindForm = KindForm {
    kind: MESH,
    placed: FormSide {
        vertex: vertex::VS_MESH_EXTENDED,
        streams: EXTENDED_STREAMS,
        depth_bias: FormSide::UNBIASED,
    },
    owned: Some(FormSide {
        vertex: vertex::VS_VIEW_EXTENDED,
        streams: EXTENDED_VIEW_STREAMS,
        depth_bias: FormSide::UNBIASED,
    }),
    varyings: Varyings::SurfaceExtra(FragmentUse::Masked),
};

const SKIN_FORM: KindForm = KindForm {
    kind: SKIN,
    placed: FormSide {
        vertex: vertex::VS_SKIN,
        streams: SKIN_STREAMS,
        depth_bias: FormSide::UNBIASED,
    },
    owned: Some(FormSide {
        vertex: vertex::VS_SKIN_VIEW,
        streams: SKIN_VIEW_STREAMS,
        depth_bias: FormSide::UNBIASED,
    }),
    varyings: Varyings::SurfaceExtra(FragmentUse::Masked),
};

const OCCLUDED_PASS: &str = "occluded";

const OCCLUDED: PassFragment = PassFragment {
    pass: FragmentPass::Own(OCCLUDED_PASS),
    side: None,
    fragment: FormFragment::Role(PassRole::Lit),
};

pub(crate) struct Streams<S> {
    pub ids: DrawInput<MeshesGpu, S>,
    pub args: DrawInput<MeshesGpu, S>,
    pub records: DrawInput<MeshesGpu, S>,
}

pub(crate) struct ViewStreams {
    pub ids: DrawInput<MeshesGpu, Scene>,
    pub args: DrawInput<MeshesGpu, Scene>,
}

pub(crate) struct BlendedStreams {
    pub one: Streams<Scene>,
    pub two: Streams<Scene>,
}

pub(crate) enum Drawing {
    Placed(Streams<Scene>),
    Views(ViewStreams),
    Blended(BlendedStreams),
}

pub(crate) struct MeshesGpu {
    pub passes: Passes,
    pub instances: Instances,
    pub presented: Presented,
    pub computes: Computes,
    pub forms: Forms<Form<MeshesGpu>>,
    pub scenes: PerPass<Drawing>,
    pub occluded: Streams<Own>,
    pub state: FrameState,
}

pub(crate) type MeshReads = (
    FrameReads,
    (
        Query<'static, 'static, &'static LastLevels, With<MeshAssets>>,
        Res<'static, Views>,
        Res<'static, WatchedSlot>,
        Res<'static, PresentationTotals>,
        Res<'static, AssetResidency>,
    ),
);

pub(crate) type FrameReads = (
    Res<'static, DrawnTable>,
    Res<'static, AssetLibrary>,
    Res<'static, CullPlan>,
    Res<'static, GlobalTransforms>,
    Res<'static, Detail>,
    Res<'static, FrameTime>,
    Res<'static, DeviceLimits>,
);

pub(crate) const INSTANCE_STAGING_ROW: &str = "memory.instance_staging_capacity_bytes";
pub(crate) const INSTANCE_BUFFER_ROW: &str = "memory.gpu_instance_buffer_bytes";

impl GpuPart for MeshesGpu {
    type Reads = MeshReads;
    const MEMORY: &'static [&'static str] = &[INSTANCE_STAGING_ROW, INSTANCE_BUFFER_ROW];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let instances = install_instances(cx)?;
        let presented = install_presented(cx)?;
        let passes = install_passes(cx);
        let computes = install_computes(cx, &passes, &presented)?;
        let forms = Forms {
            plain: cx.form_owning(&PLAIN_FORM, &[OCCLUDED], &[])?,
            extended: cx.form_owning(&EXTENDED_FORM, &[OCCLUDED], &[])?,
            skin: cx.form_owning(&SKIN_FORM, &[OCCLUDED], &[])?,
        };
        let scenes = install_scenes(cx, &forms, (&instances, &presented));
        let occluded = streams(cx, passes.occluded, (&instances, &presented), 1);
        Ok(Self {
            passes,
            instances,
            presented,
            computes,
            forms,
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

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        self.encode_pass(pass, cx)
    }
}

fn install_instances(cx: &mut Install<'_, MeshesGpu>) -> Result<Instances, Error> {
    use crate::shaders::meshes::slots;
    let instances = Instances {
        records: cx.buffer("debug_view_records", Lifetime::Kept),
        colors: cx.buffer("colors", Lifetime::Kept),
    };
    cx.frame(slots::INSTANCE_COLORS, Bind::Buffer(instances.colors))?;
    Ok(instances)
}

fn install_presented(cx: &mut Install<'_, MeshesGpu>) -> Result<Presented, Error> {
    use crate::shaders::meshes::faces::instances as face;
    let mut buffer = |label: &'static str| cx.buffer(label, Lifetime::Kept);
    let frame = buffer("presentation_frame");
    let frame_two = buffer("presentation_frame_two");
    let view_frames = buffer("presentation_view_frames");
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
    let last_ids = [
        buffer("presentation_last_ids"),
        buffer("presentation_last_ids_two"),
    ];
    let lod_states = buffer("presentation_lod_states");
    let fading_ids = buffer("presentation_fading_ids");
    let counts = buffer("presentation_occlusion_counts");
    let totals = buffer("presentation_totals");
    cx.window(view_frames, size_of::<PresentationFrame>() as u64);
    cx.frame(face::LAST_IDS, Bind::Buffer(last_ids[0]))?;
    cx.frame(face::LAST_LATE_IDS, Bind::Buffer(last_ids[1]))?;
    cx.frame(face::LAST_FADING_IDS, Bind::Buffer(fading_ids))?;
    Ok(Presented {
        frame,
        frame_two,
        view_frames,
        candidates,
        slot_candidates,
        scan,
        batches,
        batch_bases,
        draws,
        visible,
        view_draws,
        view_visible,
        last_ids,
        lod_states,
        fading_ids,
        counts,
        totals,
    })
}

fn install_passes(cx: &mut Install<'_, MeshesGpu>) -> Passes {
    let presentation = cx.pass("presentation").id();
    let views = cx.pass("view_cull").after(presentation).id();
    let occlusion = cx
        .pass("occlusion_lists")
        .at(AttachPoint::LateOpaque)
        .after(presentation)
        .after(views)
        .id();
    let occluded = cx
        .pass(OCCLUDED_PASS)
        .at(AttachPoint::LateOpaque)
        .after(occlusion)
        .id();
    Passes {
        presentation,
        occlusion,
        occluded,
        views,
    }
}

fn compaction(
    cx: &mut Install<'_, MeshesGpu>,
    pass: Pass<MeshesGpu>,
    presented: &Presented,
    (frame, visible, draws, last): (
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
                (p::LAST_IDS, Bind::Buffer(last)),
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
                (p::LAST_IDS, Bind::Buffer(last)),
            ],
        )?,
    })
}

fn install_computes(
    cx: &mut Install<'_, MeshesGpu>,
    passes: &Passes,
    presented: &Presented,
) -> Result<Computes, Error> {
    use presentation as p;
    let at = passes.presentation;
    let cull = cx.compute_pipeline(
        at,
        p::CULL,
        &[
            (p::FRAME, Bind::Buffer(presented.frame)),
            (p::BATCHES, Bind::Buffer(presented.batches)),
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
            (p::FADING_IDS, Bind::Buffer(presented.fading_ids)),
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
            presented.last_ids[0],
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
            presented.last_ids[1],
        ),
    )?;
    let gather_two = gather(cx, late, p::GATHER_PHASE_TWO, presented.frame_two)?;
    let occlusion = cx.compute_pipeline(
        late,
        p::OCCLUDED_LISTS,
        &[
            (p::FRAME, Bind::Buffer(presented.frame)),
            (p::SLOT_CANDIDATES, Bind::Buffer(presented.slot_candidates)),
            (p::SCAN, Bind::Buffer(presented.scan)),
            (p::OCCLUSION_COUNTS, Bind::Buffer(presented.counts)),
        ],
    )?;
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
            presented.last_ids[0],
        ),
    )?;
    Ok(Computes {
        cull,
        fades,
        one,
        two,
        views,
        gather_one,
        gather_two,
        occlusion,
        view_cull,
    })
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
    }
}

fn install_scenes(
    cx: &mut Install<'_, MeshesGpu>,
    forms: &Forms<Form<MeshesGpu>>,
    (instances, presented): (&Instances, &Presented),
) -> PerPass<Drawing> {
    let passes = forms.plain.passes().chain(forms.skin.passes());
    PerPass::of(passes, |pass| {
        let scene = pass.scene;
        match (pass.role, pass.owned) {
            (PassRole::Blended, true | false) => Drawing::Blended(BlendedStreams {
                one: streams(cx, scene, (instances, presented), 0),
                two: streams(cx, scene, (instances, presented), 1),
            }),
            (PassRole::Lit | PassRole::Depth | PassRole::Shown, true) => {
                Drawing::Views(ViewStreams {
                    ids: cx.draw_input(scene, presented.view_visible),
                    args: cx.draw_input(scene, presented.view_draws),
                })
            }
            (PassRole::Lit | PassRole::Depth | PassRole::Shown, false) => {
                Drawing::Placed(streams(cx, scene, (instances, presented), 0))
            }
        }
    })
}
