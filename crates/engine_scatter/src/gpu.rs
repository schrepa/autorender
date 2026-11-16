use bevy_ecs::prelude::*;
use engine_core::{
    AssetResidency, AttachPoint, DrawnTable, DrawnView, Entry, Error, GeometryHandle,
    GeometryStream, InstanceInput, MaterialHandle, Mobility, PassBit, PassRole, PassViews,
    TargetName, Var, ViewFacts, Views,
};
use engine_open_world_kit::ground::{GroundTiles, TerrainTiles};
use engine_places_rendering::impostors::IMPOSTOR_VERTICES;
use engine_render::{
    AfterRead, Allocation, Bind, Buffer, ComputePipeline, Count, Draw, DrawInput, Encode, FaceCull,
    Form, FormColor, FormDepth, FormFragment, FormPass, FormSide, FragmentPass, FragmentUse,
    GpuPart, Install, KindForm, Lifetime, OwnFragments, Pass, PassFragment, PerPass, Pipeline,
    Prepare, RoleFragments, Scene, SceneDraw, Shape, Texture, Varyings, VertexLayout, wgpu,
};
use glam::Vec3;

use crate::far_cover::{CanopyGpu, CanopyMaps};
use crate::hidden::Hidden;
use crate::layouts::{
    CullFrame, CullGroup, CullView, DrawArgs, DrawKind, DrawRecord, LAID_GROUP, LEVELS,
    PlacedGroup, PlacedRecord, PlacedValues, PoseData, SCATTER_GROUP, SlotColor, Snap,
    placed_group, record_words,
};
use crate::pool::{SlotWrite, Spread, dead};
use crate::shaders::scatter::{
    cards as card_shaders, casters as caster_shaders, draw as object_shaders, laid as laid_shaders,
    matched as matched_shaders,
};
use crate::snapping::{SnapHeld, SnapInputs};
use crate::systems::{GroupData, ScatterFeedback, ScatterState};

const IDS: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 4,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &[wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset: 0,
        shader_location: InstanceInput::Slot.location(),
    }],
};

const RECORD: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 0,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &[wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 0,
        shader_location: InstanceInput::Draw.location(),
    }],
};

const CARDS: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 8,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 0,
            shader_location: InstanceInput::Slot.location(),
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 4,
            shader_location: InstanceInput::Batch.location(),
        },
    ],
};

const PLAIN_STREAMS: &[VertexLayout] = &[
    VertexLayout::Geometry(GeometryStream::Vertex),
    VertexLayout::Buffer(IDS),
    VertexLayout::Buffer(RECORD),
];

const EXTENDED_STREAMS: &[VertexLayout] = &[
    VertexLayout::Geometry(GeometryStream::Vertex),
    VertexLayout::Geometry(GeometryStream::Extra),
    VertexLayout::Buffer(IDS),
    VertexLayout::Buffer(RECORD),
];

const CARD_STREAMS: &[VertexLayout] = &[VertexLayout::Buffer(CARDS), VertexLayout::Buffer(RECORD)];

const SHADOW_CARD_STREAMS: &[VertexLayout] = &[VertexLayout::Buffer(CARDS)];

const PLACED_MESHES: &str = "the placed meshes";
const PLACED_MESH: &str = "$.placed.mesh";

pub(crate) const PLACED: engine_core::DrawKind = engine_core::DrawKind("placed");

const fn object_form(
    vertex: engine_core::Entry,
    streams: &'static [VertexLayout],
    varyings: Varyings,
) -> KindForm {
    KindForm {
        kind: PLACED,
        placed: FormSide {
            vertex,
            streams,
            depth_bias: FormSide::UNBIASED,
        },
        owned: None,
        varyings,
    }
}

const PLAIN_FORM: KindForm = object_form(
    object_shaders::VS_OBJECT,
    PLAIN_STREAMS,
    Varyings::Object(FragmentUse::Masked),
);

const EXTENDED_FORM: KindForm = object_form(
    object_shaders::VS_OBJECT_EXTENDED,
    EXTENDED_STREAMS,
    Varyings::ObjectExtra(FragmentUse::Masked),
);

const FADING_FORM: KindForm = object_form(
    object_shaders::VS_OBJECT_FADING,
    PLAIN_STREAMS,
    Varyings::Object(FragmentUse::Always),
);

const EXTENDED_FADING_FORM: KindForm = object_form(
    object_shaders::VS_OBJECT_EXTENDED_FADING,
    EXTENDED_STREAMS,
    Varyings::ObjectExtra(FragmentUse::Always),
);

const MATCHED_FORMS: [(KindForm, Entry); 4] = [
    (
        object_form(
            object_shaders::VS_OBJECT,
            PLAIN_STREAMS,
            Varyings::Own(&OwnFragments::NONE),
        ),
        matched_shaders::FS_OBJECT_MATCHED,
    ),
    (
        object_form(
            object_shaders::VS_OBJECT_EXTENDED,
            EXTENDED_STREAMS,
            Varyings::Own(&OwnFragments::NONE),
        ),
        matched_shaders::FS_OBJECT_MATCHED_EXTENDED,
    ),
    (
        object_form(
            object_shaders::VS_OBJECT_FADING,
            PLAIN_STREAMS,
            Varyings::Own(&OwnFragments::NONE),
        ),
        matched_shaders::FS_OBJECT_MATCHED,
    ),
    (
        object_form(
            object_shaders::VS_OBJECT_EXTENDED_FADING,
            EXTENDED_STREAMS,
            Varyings::Own(&OwnFragments::NONE),
        ),
        matched_shaders::FS_OBJECT_MATCHED_EXTENDED,
    ),
];

const CARD_FRAGMENTS: OwnFragments = OwnFragments {
    placed: RoleFragments {
        lit: Some(card_shaders::FS_PLACED_CARD),
        depth: Some(card_shaders::FS_PLACED_CARD_DEPTH),
        blended: None,
        shown: Some(card_shaders::FS_DEBUG_PLACED_CARD),
    },
    owned: RoleFragments::NONE,
    cull: FaceCull::None,
};

const CARD_FORM: KindForm = object_form(
    card_shaders::VS_PLACED_CARD,
    CARD_STREAMS,
    Varyings::Own(&CARD_FRAGMENTS),
);

const CASTER_FRAGMENTS: OwnFragments = OwnFragments {
    placed: RoleFragments::NONE,
    owned: RoleFragments {
        depth: Some(caster_shaders::FS_IMPOSTOR_CARD),
        ..RoleFragments::NONE
    },
    cull: FaceCull::None,
};

const CASTER_FORM: KindForm = object_form(
    caster_shaders::VS_IMPOSTOR_CARD,
    SHADOW_CARD_STREAMS,
    Varyings::Own(&CASTER_FRAGMENTS),
);

const LEFT_OUT: u32 = 0;

struct Level {
    mesh: GeometryHandle,
    material: Option<MaterialHandle>,
    group: u32,
    level: u32,
    extended: bool,
    masked: bool,
    fades: bool,
}

#[derive(Clone, Copy)]
struct Listed {
    drawn: u32,
    casts: bool,
    laid: bool,
}

#[derive(Clone, Copy, PartialEq)]
struct Sizes {
    views: u64,
    ids: u64,
    draws: u64,
    cards: u64,
    card_args: u64,
}

impl Held {
    fn laid_at(&self, index: u32) -> bool {
        self.views
            .iter()
            .any(|listed| listed.drawn == index && listed.laid)
    }
}

impl Default for Sizes {
    fn default() -> Self {
        Self {
            views: size_of::<CullView>() as u64,
            ids: 16,
            draws: 20,
            cards: 8,
            card_args: 16,
        }
    }
}

#[derive(Default)]
struct Held {
    lit: Vec<Level>,
    cast: Vec<Level>,
    lit_waiting: bool,
    cast_waiting: bool,
    refused: Option<Error>,
    lists: Vec<(u32, u32)>,
    ids_per_view: u32,
    group_count: u32,
    views: Vec<Listed>,
    sizes: Sizes,
    left_out: bool,
    fading_view: usize,
    objects: u32,
    ids_room: u64,
    cards_room: u64,
    uploaded: Option<u64>,
    slot_sizes: [u64; 4],
    snap: SnapHeld,
    prepared: bool,
    cards: bool,
    atlas: bool,
    pyramid: bool,
    placed_groups: Vec<PlacedGroup>,
    culling: Vec<(Listed, CullView)>,
    culled: Vec<CullView>,
    reached: Vec<u32>,
    card_args: Vec<[u32; 4]>,
    args: Vec<DrawArgs>,
    first_instance: bool,
}

fn lay_args(args: &mut Vec<DrawArgs>, data: &GroupData, lists: usize) {
    args.clear();
    for view in 0..lists {
        let at = args.len();
        args.extend_from_slice(&data.template);
        if !data.first_instance {
            continue;
        }
        let levels = data.lists.iter().flat_map(|lists| {
            let lists = (lists.ids, lists.count);
            (0..LEVELS).map(move |level| ids_start(data.ids_per_view, lists, view, level))
        });
        for (drawn, first) in args[at..].iter_mut().zip(levels) {
            drawn[4] = u32::try_from(first).unwrap_or(u32::MAX);
        }
    }
}

impl Held {
    fn reach(&mut self, state: &ScatterState) {
        let lists = self.culled.len();
        let main = self.culled[0];
        let nearest = state.pool.nearest(Vec3::from_slice(&main.eye[..3]));
        self.reached.clear();
        for at in 0..=lists {
            let (list, view) = match at {
                0 => (List::Main, main),
                _ if at == lists => (List::Fading, main),
                _ => (List::Own, self.culled[at]),
            };
            self.reached.extend(state.groups.groups.iter().map(|drawn| {
                let owner = drawn.objects as usize;
                let near = match list {
                    List::Main | List::Fading => nearest.get(owner).copied().unwrap_or(0.0),
                    List::Own => 0.0,
                };
                match (
                    state.group_data.data.get(owner),
                    state.pool.spreads.get(owner),
                ) {
                    (Some(group), Some(&spread)) => {
                        levels_reached(&view, (group, spread, near), list, self.cards)
                    }
                    _ => u32::MAX,
                }
            }));
        }
    }
}

type DrawBuffers = (
    Buffer<ScatterGpu>,
    Buffer<ScatterGpu>,
    Buffer<ScatterGpu>,
    Buffer<ScatterGpu>,
    Buffer<ScatterGpu>,
);

struct Forms {
    plain: Form<ScatterGpu>,
    extended: Form<ScatterGpu>,
    fading: Form<ScatterGpu>,
    extended_fading: Form<ScatterGpu>,
    matched: [Form<ScatterGpu>; 4],
    cards: Form<ScatterGpu>,
    tested_cards: Form<ScatterGpu>,
    crossing_cards: Form<ScatterGpu>,
    casters: Form<ScatterGpu>,
}

type Pair = [Option<Pipeline<ScatterGpu, Scene>>; 2];

#[derive(Clone, Copy)]
struct Streams {
    ids: DrawInput<ScatterGpu, Scene>,
    args: DrawInput<ScatterGpu, Scene>,
    records: DrawInput<ScatterGpu, Scene>,
    cards: DrawInput<ScatterGpu, Scene>,
    card_args: DrawInput<ScatterGpu, Scene>,
}

struct Joined {
    bit: PassBit,
    role: PassRole,
    owned: bool,
    lays: bool,
    follows: bool,
    cards: bool,
    streams: Streams,
}

impl Joined {
    fn takes(&self, view: &DrawnView) -> bool {
        let opaque = match self.role {
            PassRole::Lit | PassRole::Depth | PassRole::Shown => true,
            PassRole::Blended => false,
        };
        opaque && self.owned == view.owner.is_some() && view.passes.has(self.bit)
    }
}

fn depth_at(install: &Install<'_, ScatterGpu>, pass: FormPass) -> Option<AttachPoint> {
    install
        .passes()
        .listed()
        .iter()
        .find(|listed| listed.name == pass.scene.name())
        .filter(|listed| listed.pass.writes.contains(TargetName::DEPTH))
        .map(|listed| listed.pass.at)
}

fn lays(install: &Install<'_, ScatterGpu>, pass: FormPass) -> bool {
    pass.role == PassRole::Depth
        && !pass.owned
        && depth_at(install, pass) == Some(AttachPoint::Depth)
}

fn follows(install: &Install<'_, ScatterGpu>, pass: FormPass) -> bool {
    pass.role == PassRole::Lit
        && !pass.owned
        && depth_at(install, pass).is_some_and(|at| at > AttachPoint::Depth)
}

fn laid(joined: &PerPass<Option<Joined>>, view: &DrawnView) -> bool {
    joined
        .rows()
        .flatten()
        .any(|scene| scene.lays && scene.takes(view))
}

fn listed_view(
    joined: &PerPass<Option<Joined>>,
    (index, view): (usize, &DrawnView),
) -> Option<(Listed, CullView)> {
    let mut taking = joined
        .rows()
        .flatten()
        .filter(|scene| scene.takes(view))
        .peekable();
    taking.peek()?;
    let (casts, cards) = taking.fold((true, true), |(casts, cards), scene| {
        (casts && scene.role == PassRole::Depth, cards && scene.cards)
    });
    let projection = view.view_projection;
    let pixels = if view.detail > 0.0 {
        projection.row(1).truncate().length() * view.size.y as f32 * 0.5 / view.detail
    } else {
        0.0
    };
    let perspective = projection.row(3).truncate() != Vec3::ZERO;
    Some((
        Listed {
            drawn: index as u32,
            casts,
            laid: laid(joined, view),
        },
        CullView {
            eye: view.eye.extend(pixels).to_array(),
            shape: [
                u32::from(perspective),
                u32::from(cards),
                u32::from(casts),
                index as u32,
            ],
        },
    ))
}

const SLACK: f32 = 1.0 / 1024.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum List {
    Main,
    Fading,
    Own,
}

fn levels_reached(
    view: &CullView,
    (group, spread, near): (&CullGroup, Spread, f32),
    list: List,
    cards: bool,
) -> u32 {
    let camera = group.camera.clamp(1, LEVELS) as usize;
    let every = (1 << camera) - 1;
    let pixels = view.eye[3];
    let errors = &group.errors[..camera - 1];
    let known = pixels.is_finite()
        && near >= 0.0
        && spread.low > 0.0
        && spread.low <= spread.high
        && spread.high.is_finite()
        && errors
            .iter()
            .all(|error| *error >= 0.0 && error.is_finite());
    if !known {
        return every;
    }
    if pixels <= 0.0 {
        return u32::from(list != List::Fading);
    }
    let impostor = group.impostor;
    let cards = cards && view.shape[1] != 0 && impostor > 0.0;
    if list == List::Fading && !cards {
        return 0;
    }
    let mut reached = every;
    if cards {
        for (at, error) in errors.iter().enumerate() {
            if *error > impostor * (1.0 + SLACK) {
                reached &= !(2 << at);
            }
        }
        if list == List::Fading {
            let fits = impostor * std::f32::consts::FRAC_1_SQRT_2 * (1.0 - SLACK);
            let finest = errors
                .iter()
                .rposition(|error| *error <= fits)
                .map_or(0, |at| at + 1);
            reached &= !((1 << finest) - 1);
        }
    }
    let flat = view.shape[0] == 0;
    if flat && list == List::Own && spread.radius * pixels < 1.0 - SLACK {
        return 0;
    }
    let step = 1.0 / pixels;
    let least = if flat { step } else { near * step } * (1.0 - SLACK);
    if cards && least >= impostor * spread.high * (1.0 + SLACK) {
        return 0;
    }
    let low = if flat && cards {
        engine_math::max(spread.low, step / impostor * (1.0 - SLACK))
    } else {
        spread.low
    };
    for level in 0..camera {
        let reaches = !flat || level == 0 || errors[level - 1] * low <= step * (1.0 + SLACK);
        let stays = errors[level..]
            .iter()
            .all(|error| *error * spread.high > least);
        if !reaches || !stays {
            reached &= !(1 << level);
        }
    }
    reached
}

pub(crate) struct ScatterGpu {
    scatter: Pass<Self>,
    snap_list: Buffer<Self>,
    frame: Buffer<Self>,
    views: Buffer<Self>,
    records: Buffer<Self>,
    groups: Buffer<Self>,
    ids: Buffer<Self>,
    draws: Buffer<Self>,
    poses: Buffer<Self>,
    colors: Buffer<Self>,
    values: Buffer<Self>,
    cards: Buffer<Self>,
    card_args: Buffer<Self>,
    placed_groups: Buffer<Self>,
    debug_records: Buffer<Self>,
    canopy: CanopyGpu,
    snapping: ComputePipeline<Self>,
    culling: ComputePipeline<Self>,
    lay: Pass<Self>,
    laid: Texture<Self>,
    laying: ComputePipeline<Self>,
    laid_size: (u32, u32),
    laid_copied: Option<u32>,
    forms: Forms,
    joined: PerPass<Option<Joined>>,
    records_written: bool,
    held: Held,
    hidden: Hidden,
}

fn ids_start(ids_per_view: u32, (ids, count): (u32, u32), view: usize, level: u32) -> u64 {
    view as u64 * u64::from(ids_per_view) + u64::from(ids) + u64::from(level) * u64::from(count)
}

fn grow(held: &mut u64, bytes: u64, room: u64) {
    let size = bytes.next_power_of_two().min(room);
    if bytes > *held && size > *held {
        *held = size;
    }
}

impl ScatterGpu {
    fn forms(
        cx: &mut Install<'_, Self>,
        (poses, colors, laid): (Buffer<Self>, Buffer<Self>, Texture<Self>),
    ) -> Result<Forms, Error> {
        let placed = [
            (object_shaders::POSES, Bind::Buffer(poses)),
            (object_shaders::COLORS, Bind::Buffer(colors)),
        ];
        let plain = cx.form(&PLAIN_FORM, &placed)?;
        let following: Vec<FormPass> = plain.passes().filter(|pass| follows(cx, *pass)).collect();
        let owning =
            |side: Option<FormSide>, entry: Entry, depth: FormDepth| -> Vec<PassFragment> {
                following
                    .iter()
                    .map(|pass| PassFragment {
                        pass: FragmentPass::Scene(pass.scene),
                        side,
                        fragment: FormFragment::Entry {
                            entry,
                            cull: FaceCull::None,
                            depth,
                            color: FormColor::Written,
                        },
                    })
                    .collect()
            };
        let mut matched = |index: usize| {
            let (form, entry) = &MATCHED_FORMS[index];
            cx.form_owning(form, &owning(None, *entry, FormDepth::Matched), &placed)
        };
        let matched = [matched(0)?, matched(1)?, matched(2)?, matched(3)?];
        let card_bound: [(Var, Bind<ScatterGpu>); 3] = [
            (card_shaders::POSES, Bind::Buffer(poses)),
            (card_shaders::COLORS, Bind::Buffer(colors)),
            (card_shaders::LAID_DEPTH, Bind::Texture(laid)),
        ];
        let card_side = |vertex: Entry| FormSide {
            vertex,
            ..CARD_FORM.placed
        };
        let tested = owning(
            Some(card_side(card_shaders::VS_PLACED_CARD_TESTED)),
            card_shaders::FS_PLACED_CARD_TESTED,
            FormDepth::Tested,
        );
        let crossing = owning(
            Some(card_side(card_shaders::VS_PLACED_CARD_CROSSING)),
            card_shaders::FS_PLACED_CARD_LAID,
            FormDepth::Written,
        );
        Ok(Forms {
            plain,
            matched,
            extended: cx.form(&EXTENDED_FORM, &placed)?,
            fading: cx.form(&FADING_FORM, &placed)?,
            extended_fading: cx.form(&EXTENDED_FADING_FORM, &placed)?,
            cards: cx.form(&CARD_FORM, &card_bound[..2])?,
            tested_cards: cx.form_owning(&CARD_FORM, &tested, &card_bound)?,
            crossing_cards: cx.form_owning(&CARD_FORM, &crossing, &card_bound)?,
            casters: cx.form(
                &CASTER_FORM,
                &[(caster_shaders::POSES, Bind::Buffer(poses))],
            )?,
        })
    }

    fn scenes(
        cx: &mut Install<'_, Self>,
        forms: &Forms,
        (ids, draws, records, cards, card_args): DrawBuffers,
    ) -> PerPass<Option<Joined>> {
        PerPass::of(forms.plain.passes(), |pass| {
            let scene = pass.scene;
            let name = scene.name();
            let bit = cx.passes().bit(name)?;
            let streams = Streams {
                ids: cx.draw_input(scene, ids),
                args: cx.args_input(scene, draws),
                records: cx.draw_input(scene, records),
                cards: cx.draw_input(scene, cards),
                card_args: cx.args_input(scene, card_args),
            };
            Some(Joined {
                bit,
                role: pass.role,
                owned: pass.owned,
                lays: lays(cx, pass),
                follows: follows(cx, pass),
                cards: forms.cards.draws_in(name) || forms.casters.draws_in(name),
                streams,
            })
        })
    }

    fn list_draws(&mut self, state: &ScatterState, residency: &AssetResidency) {
        let held = &mut self.held;
        let data = &state.group_data;
        held.lists.clear();
        held.lists
            .extend(data.lists.iter().map(|lists| (lists.ids, lists.count)));
        held.group_count = state.groups.layouts.len() as u32;
        held.ids_per_view = data.ids_per_view;
        held.lit.clear();
        held.cast.clear();
        held.lit_waiting = false;
        held.cast_waiting = false;
        held.refused = None;
        for (group, drawn) in state.groups.groups.iter().enumerate() {
            if drawn.count == 0 {
                continue;
            }
            let resident = residency.material_resident(drawn.material);
            let uncut = drawn.masked && !resident;
            let fades = data
                .data
                .get(drawn.objects as usize)
                .is_some_and(|main| main.impostor > 0.0);
            held.cast_waiting |= uncut;
            for (level, &mesh) in drawn.meshes.iter().enumerate() {
                if !residency.resident(mesh) {
                    held.lit_waiting = true;
                    held.cast_waiting = true;
                    if held.refused.is_none() {
                        held.refused = residency.refusal(mesh, PLACED_MESH);
                    }
                    continue;
                }
                let lit = Level {
                    mesh,
                    material: Some(drawn.material),
                    group: group as u32,
                    level: level as u32,
                    extended: drawn.extended,
                    masked: drawn.masked,
                    fades,
                };
                if !uncut {
                    held.cast.push(Level {
                        material: resident.then_some(drawn.material),
                        ..lit
                    });
                }
                held.lit.push(lit);
            }
        }
    }

    fn write_slot(
        &self,
        cx: &mut Prepare<'_, Self>,
        write: &SlotWrite,
        objects: u32,
    ) -> Option<std::ops::Range<u32>> {
        match write {
            SlotWrite::Dead(slots) => {
                let slots = slots.start.min(objects)..slots.end.min(objects);
                if !slots.is_empty() {
                    cx.write(
                        self.records,
                        u64::from(slots.start) * size_of::<PlacedRecord>() as u64,
                        &vec![dead(); (slots.end - slots.start) as usize],
                    );
                }
                None
            }
            SlotWrite::Objects(first, (poses, colors, records, values)) => {
                let count = objects.saturating_sub(*first).min(records.len() as u32) as usize;
                if count == 0 {
                    return None;
                }
                let start = u64::from(*first);
                cx.write(
                    self.poses,
                    start * size_of::<PoseData>() as u64,
                    &poses[..count],
                );
                cx.write(
                    self.colors,
                    start * size_of::<SlotColor>() as u64,
                    &colors[..count],
                );
                cx.write(
                    self.records,
                    start * size_of::<PlacedRecord>() as u64,
                    &records[..count],
                );
                cx.write(
                    self.values,
                    start * size_of::<PlacedValues>() as u64,
                    &values[..count],
                );
                Some(*first..*first + count as u32)
            }
        }
    }

    fn write_slots(&mut self, cx: &mut Prepare<'_, Self>, state: &ScatterState) {
        let buffers = [self.poses, self.colors, self.records, self.values];
        let strides = [
            size_of::<PoseData>(),
            size_of::<SlotColor>(),
            size_of::<PlacedRecord>(),
            size_of::<PlacedValues>(),
        ];
        let mut fresh = false;
        let sizes = self.held.slot_sizes.iter_mut();
        for (((buffer, bytes), stride), size) in
            buffers.into_iter().zip(state.slot_bytes).zip(strides).zip(sizes)
        {
            let bytes = bytes.max(stride as u64);
            let copied = *size > 0 && bytes > *size && bytes.is_power_of_two();
            let allocation = if copied {
                cx.grow_buffer(buffer, bytes)
            } else {
                cx.allocate_buffer(buffer, bytes)
            };
            fresh |= allocation == Allocation::Fresh;
            *size = bytes;
        }
        let objects = state.objects;
        if fresh || self.held.uploaded != Some(state.slot_sequence) {
            let missed = self.held.uploaded.map(|seen| seen + 1) != Some(state.slot_sequence);
            if fresh || missed {
                for write in state.pool.whole(&state.groups) {
                    self.write_slot(cx, &write, objects);
                }
                self.held.snap.wrote_whole();
            }
            for write in &state.pool.writes {
                if let Some(slots) = self.write_slot(cx, write, objects) {
                    self.held.snap.wrote(slots);
                }
            }
        }
        self.held.uploaded = Some(state.slot_sequence);
        self.held.objects = objects;
    }

    fn write_placed_groups(&mut self, cx: &mut Prepare<'_, Self>, state: &ScatterState) {
        let groups = state
            .impostors
            .iter()
            .zip(&state.groups.colors)
            .map(|(&impostor, &color)| placed_group(impostor, color));
        let bytes = (groups.len().max(1) * size_of::<PlacedGroup>()) as u64;
        let fresh = cx.allocate_buffer(self.placed_groups, bytes) == Allocation::Fresh;
        let held = &mut self.held.placed_groups;
        let same = held.len() == groups.len()
            && held
                .iter()
                .zip(groups.clone())
                .all(|(held, group)| bytemuck::bytes_of(held) == bytemuck::bytes_of(&group));
        if same && !fresh {
            return;
        }
        held.clear();
        held.extend(groups);
        if !held.is_empty() {
            cx.write(self.placed_groups, 0, held);
        }
    }

    fn prepare_snap(
        &mut self,
        cx: &mut Prepare<'_, Self>,
        (state, residency): (&ScatterState, &AssetResidency),
        (tiles, ground, baking): (
            &Res<'_, TerrainTiles>,
            &Res<'_, GroundTiles>,
            &Res<'_, engine_open_world_kit::ground::GroundBaking>,
        ),
    ) {
        let objects = self.held.objects;
        let snap = &mut self.held.snap;
        snap.snapping = false;
        let terrains = tiles.terrains.len() as u32;
        if cx.grow_buffer(self.snap_list, 2 * size_of::<Snap>() as u64) == Allocation::Fresh {
            snap.snapped = None;
        }
        if ground.layers == 0 || terrains == 0 || state.resident == 0 {
            return;
        }
        let inputs = SnapInputs {
            tiles,
            ground,
            baking,
            assets: (state.library, residency.revision()),
            changed: tiles.is_changed() || ground.is_changed() || baking.is_changed(),
        };
        let planned = snap.plan(&inputs, (&state.pool, state.generation, objects));
        if planned.is_empty() {
            return;
        }
        let words = snap.words(&planned, terrains);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.grow_buffer(self.snap_list, std::mem::size_of_val(words.as_slice()) as u64);
        cx.write(self.snap_list, 0, &words);
        snap.snapping = true;
    }

    fn prepare_cull(
        &mut self,
        cx: &mut Prepare<'_, Self>,
        state: &ScatterState,
        (views, table): (&Views, &DrawnTable),
    ) {
        let plan = &state.plan;
        let drawn = table.rows(PLACED).next().map(|(slot, _, _)| slot);
        let main = plan
            .main
            .filter(|_| plan.prepared && drawn.is_some())
            .filter(|_| {
                views
                    .drawn()
                    .first()
                    .is_some_and(|view| view.owner.is_none())
            });
        let held = &mut self.held;
        let listed = &mut held.culling;
        listed.clear();
        if let Some(main) = main {
            let first = Listed {
                drawn: 0,
                casts: false,
                laid: views
                    .drawn()
                    .first()
                    .is_some_and(|view| laid(&self.joined, view)),
            };
            listed.push((first, main));
            listed.extend(
                views
                    .drawn()
                    .iter()
                    .enumerate()
                    .skip(1)
                    .filter(|(_, view)| view.draws && view.takes != Some(Mobility::Moving))
                    .filter_map(|view| listed_view(&self.joined, view)),
            );
        }
        held.prepared = false;
        held.cards = plan.cards;
        held.pyramid = views.main().is_some();
        held.atlas = plan.atlas;
        held.views.clear();
        held.views.extend(listed.iter().map(|(view, _)| *view));
        if state.objects == 0 {
            held.sizes = Sizes::default();
        }
        let view_count = listed.len() as u32;
        let lists = u64::from(view_count) + 3;
        let ids = u64::from(state.group_data.ids_per_view) * lists * 4;
        let draws = u64::from(held.group_count * LEVELS) * lists * size_of::<DrawArgs>() as u64;
        let cards = u64::from(state.objects) * (u64::from(view_count) + 1) * 8;
        let sizes = &mut held.sizes;
        grow(
            &mut sizes.views,
            u64::from(view_count) * size_of::<CullView>() as u64,
            state.room,
        );
        grow(&mut sizes.ids, ids, state.room);
        grow(&mut sizes.draws, draws, state.room);
        grow(&mut sizes.cards, cards, state.room);
        grow(&mut sizes.card_args, (lists - 1) * 16, state.room);
        let sizes = *sizes;
        let rebuilt = [
            (
                self.groups,
                state.group_bytes.max(size_of::<CullGroup>() as u64),
            ),
            (self.views, sizes.views),
            (self.ids, sizes.ids),
            (self.draws, sizes.draws),
            (self.cards, sizes.cards),
            (self.card_args, sizes.card_args),
            (self.frame, size_of::<CullFrame>() as u64),
            (self.hidden.picks, u64::from(state.objects).max(4) * 4),
        ];
        for (buffer, bytes) in rebuilt {
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, bytes);
        }
        held.ids_room = sizes.ids;
        held.cards_room = sizes.cards;
        if !state.group_data.data.is_empty() {
            cx.write(self.groups, 0, &state.group_data.data);
        }
        let (Some(mut frame), Some(drawn)) = (plan.frame, drawn) else {
            return;
        };
        if listed.is_empty() {
            return;
        }
        frame.counts[3] = view_count;
        let render = views.drawn().first().map_or(glam::UVec2::ZERO, |view| view.size);
        frame.drawn = [drawn, render.x, render.y, 0];
        held.fading_view = listed.len();
        held.first_instance = state.group_data.first_instance;
        lay_args(&mut held.args, &state.group_data, listed.len() + 3);
        let card_args = &mut held.card_args;
        card_args.clear();
        card_args.resize(listed.len() + 1, [IMPOSTOR_VERTICES, 0, 0, 0]);
        card_args.push([0; 4]);
        let culled = &mut held.culled;
        culled.clear();
        culled.extend(listed.drain(..).map(|(_, view)| view));
        cx.write(self.frame, 0, &[frame]);
        cx.write(self.card_args, 0, card_args);
        cx.write(self.views, 0, culled);
        if !held.args.is_empty() {
            cx.write(self.draws, 0, &held.args);
        }
        let left = ids > state.room || cards > state.room;
        if (left || held.left_out) && held.objects > 0 {
            let at = (u64::from(view_count) + 1) * 16;
            cx.read_back::<ScatterFeedback>(LEFT_OUT, self.card_args, at..at + 16, AfterRead::Keep);
        }
        held.left_out = left;
        held.reach(state);
        held.prepared = true;
    }

    fn draw_source(&self, view: usize, group: usize, level: usize) -> Option<(u64, u64)> {
        let held = &self.held;
        let &lists = held.lists.get(group)?;
        let start = ids_start(held.ids_per_view, lists, view, level as u32) * 4;
        let draw = (view * held.group_count as usize + group) * LEVELS as usize + level;
        (start < held.ids_room).then(|| (start, (draw * size_of::<DrawArgs>()) as u64))
    }

    fn card_source(&self, view: usize) -> Option<(u64, u64)> {
        let start = view as u64 * u64::from(self.held.objects) * 8;
        (start < self.held.cards_room).then_some((start, view as u64 * 16))
    }

    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let Some(joined) = draw.per_pass(&self.joined).and_then(Option::as_ref) else {
            return draw.skip("pass has no placed pipelines");
        };
        if joined.role == PassRole::Blended {
            return draw.skip("placed objects draw opaque");
        }
        if !self.held.prepared {
            return draw.skip("nothing is placed this frame");
        }
        let Some(view) = scene
            .index
            .and_then(|index| self.held.views.iter().position(|view| view.drawn == index))
        else {
            return draw.skip("view has no visible list");
        };
        let streams = &joined.streams;
        let (drawn, empty) = if joined.owned {
            self.draw_views((scene, view), draw, streams)
        } else {
            let matched = joined.follows && self.held.views[view].laid;
            self.draw_objects((scene, view, matched), draw, streams)
        };
        let waiting = if joined.owned && self.held.views[view].casts {
            self.held.cast_waiting
        } else {
            self.held.lit_waiting
        };
        match (drawn, empty, &self.held.refused, waiting) {
            (0, 0, Some(error), _) => draw.refuse(PLACED_MESHES, error),
            (0, 0, None, true) => draw.wait(PLACED_MESHES),
            (0, _, _, _) => draw.skip("nothing placed reaches this view"),
            (drawn, ..) => drawn,
        }
    }

    fn draw_levels(
        &self,
        (view, items, fading): (usize, &[Level], bool),
        (scene, matched): (&SceneDraw, bool),
        draw: &mut Draw<'_, Self, Scene>,
        inputs: &Streams,
        pipelines: [&Pair; 2],
    ) -> (u32, u32) {
        let mut draws = 0;
        let mut empty = 0;
        let fading_view = self.held.fading_view;
        let source = match view.checked_sub(fading_view + 1) {
            Some(0) => 0,
            Some(_) => fading_view,
            None => view,
        };
        let reached = self.held.group_count as usize * source;
        let turns: &[Option<bool>] = match scene.role {
            PassRole::Depth => &[Some(false), Some(true)],
            PassRole::Lit | PassRole::Blended | PassRole::Shown => &[None],
        };
        let mut listed: Vec<&Level> = turns
            .iter()
            .flat_map(|turn| {
                items.iter().filter(move |item| {
                    (!fading || item.fades) && turn.is_none_or(|masked| item.masked == masked)
                })
            })
            .collect();
        let depth = scene.role == PassRole::Depth && !fading;
        if depth {
            let finest = |item: &Level| {
                let bits = self.held.reached.get(reached + item.group as usize);
                bits.map_or(0, |bits| bits.trailing_zeros())
            };
            listed.sort_by_key(|item| (item.masked, finest(item)));
        }
        for item in listed {
            let pair = pipelines[usize::from(item.masked)];
            let extended = item.extended && (item.masked || !depth);
            let Some(pipeline) = &pair[usize::from(extended)] else {
                continue;
            };
            let level = item.level as usize;
            let Some((ids, offset)) = self.draw_source(view, item.group as usize, level) else {
                continue;
            };
            let at = reached + item.group as usize;
            if self
                .held
                .reached
                .get(at)
                .is_some_and(|bits| bits >> level & 1 == 0)
            {
                empty += 1;
                continue;
            }
            let shown = if scene.role == PassRole::Shown { level } else { 0 };
            let record = DrawRecord::of(DrawKind::Placed, shown);
            draw.draw_timed(
                pipeline,
                Shape::MeshIndirect {
                    mesh: item.mesh,
                    material: item.material,
                    args: inputs.args,
                    offset,
                    streams: &[
                        (inputs.ids, if self.held.first_instance { 0 } else { ids }),
                        (inputs.records, record.offset()),
                    ],
                },
                || level_label(scene, item, (matched, fading)),
            );
            draws += 1;
        }
        (draws, empty)
    }

    fn draw_objects(
        &self,
        (scene, view, matched): (&SceneDraw, usize, bool),
        draw: &mut Draw<'_, Self, Scene>,
        inputs: &Streams,
    ) -> (u32, u32) {
        let held = &self.held;
        let forms = &self.forms;
        let plain = [draw.form(&forms.plain), draw.form(&forms.extended)];
        let fading = [draw.form(&forms.fading), draw.form(&forms.extended_fading)];
        let [cut, cut_fading] = if matched {
            let cut = &forms.matched;
            [
                [draw.form(&cut[0]), draw.form(&cut[1])],
                [draw.form(&cut[2]), draw.form(&cut[3])],
            ]
        } else {
            [plain, fading]
        };
        let coloured = matched && view == 0 && self.hidden.coloured == scene.index;
        let colour = held.fading_view + 1;
        let lists: &[(usize, bool, [&Pair; 2])] = match (view, held.cards, coloured) {
            (0, true, true) => &[
                (colour, false, [&plain, &cut]),
                (colour + 1, true, [&fading, &cut_fading]),
            ],
            (0, false, true) => &[(colour, false, [&plain, &cut])],
            (0, true, false) => &[
                (0, false, [&plain, &cut]),
                (held.fading_view, true, [&fading, &cut_fading]),
            ],
            _ => &[(view, false, [&plain, &cut])],
        };
        let mut draws = 0;
        let mut empty = 0;
        for &(list, fades, pipelines) in lists {
            let (drawn, skipped) = self.draw_levels(
                (list, &held.lit, fades),
                (scene, matched),
                draw,
                inputs,
                pipelines,
            );
            draws += drawn;
            empty += skipped;
        }
        if !held.cards {
            return (draws, empty);
        }
        let laid = matched && self.laid_copied.is_some() && self.laid_copied == scene.index;
        let (first, second) = if laid {
            match (
                draw.form(&self.forms.tested_cards),
                draw.form(&self.forms.crossing_cards),
            ) {
                (Some(tested), Some(crossing)) => (
                    Some((tested, "cards tested")),
                    Some((crossing, "cards crossing")),
                ),
                _ => (None, None),
            }
        } else {
            (
                draw.form(&self.forms.cards).map(|card| (card, "cards")),
                None,
            )
        };
        let listed = if coloured { held.fading_view } else { view };
        let Some((cards, offset)) = self.card_source(listed) else {
            return (draws, empty);
        };
        let record = DrawRecord::of(DrawKind::PlacedCard, 0);
        for (card, what) in [first, second].into_iter().flatten() {
            draw.draw_timed(
                &card,
                Shape::IndirectAt {
                    vertices: &[(inputs.cards, cards), (inputs.records, record.offset())],
                    indices: None,
                    args: inputs.card_args,
                    offset,
                },
                || scene_label(scene, what),
            );
            draws += 1;
        }
        (draws, empty)
    }

    fn draw_views(
        &self,
        (scene, view): (&SceneDraw, usize),
        draw: &mut Draw<'_, Self, Scene>,
        inputs: &Streams,
    ) -> (u32, u32) {
        let held = &self.held;
        let items = if held.views[view].casts {
            &held.cast
        } else {
            &held.lit
        };
        let chosen = [
            draw.form(&self.forms.plain),
            draw.form(&self.forms.extended),
        ];
        let (mut draws, empty) = self.draw_levels(
            (view, items, false),
            (scene, false),
            draw,
            inputs,
            [&chosen; 2],
        );
        let card = draw.form(&self.forms.casters).filter(|_| held.atlas);
        if let (Some(card), Some((cards, offset))) = (card, self.card_source(view)) {
            draw.draw_timed(
                &card,
                Shape::IndirectAt {
                    vertices: &[(inputs.cards, cards)],
                    indices: None,
                    args: inputs.card_args,
                    offset,
                },
                || scene_label(scene, "casters"),
            );
            draws += 1;
        }
        (draws, empty)
    }
}

fn scene_label(scene: &SceneDraw, what: &str) -> String {
    let pass = scene.pass.name();
    match scene.view {
        Some(view) => format!("scatter {pass} {what} view {view}"),
        None => format!("scatter {pass} {what}"),
    }
}

fn level_label(scene: &SceneDraw, item: &Level, (matched, fading): (bool, bool)) -> String {
    let material = item
        .material
        .map_or_else(|| "none".to_owned(), |material| material.slot().to_string());
    let cut = match (item.masked, matched) {
        (false, _) => "opaque",
        (true, false) => "masked",
        (true, true) => "masked matched",
    };
    let fading = if fading { " fading" } else { "" };
    let what = format!(
        "group {} level {} mesh {} material {material} {cut}{fading}",
        item.group,
        item.level,
        item.mesh.slot()
    );
    scene_label(scene, &what)
}

impl GpuPart for ScatterGpu {
    type Reads = (
        Res<'static, ScatterState>,
        Res<'static, CanopyMaps>,
        Res<'static, TerrainTiles>,
        Res<'static, GroundTiles>,
        Res<'static, engine_open_world_kit::ground::GroundBaking>,
        Res<'static, AssetResidency>,
        Res<'static, Views>,
        Res<'static, DrawnTable>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::scatter::{cull, faces, snap, values as placed};
        let snap_list = cx.buffer("snap", Lifetime::Kept);
        let frame = cx.buffer("frame", Lifetime::Kept);
        let views = cx.buffer("views", Lifetime::Kept);
        let records = cx.buffer("records", Lifetime::Kept);
        let groups = cx.buffer("groups", Lifetime::Kept);
        let ids = cx.buffer("ids", Lifetime::Kept);
        let draws = cx.buffer("draws", Lifetime::Kept);
        let poses = cx.buffer("poses", Lifetime::Kept);
        let colors = cx.buffer("colors", Lifetime::Kept);
        let values = cx.buffer("values", Lifetime::Kept);
        let cards = cx.buffer("cards", Lifetime::Kept);
        let card_args = cx.buffer("card_args", Lifetime::Kept);
        let placed_groups = cx.buffer("placed_groups", Lifetime::Kept);
        let debug_records = cx.buffer("debug_records", Lifetime::Kept);
        let canopy = CanopyGpu::install(cx)?;
        cx.frame(faces::placed_tiles::PLACED_RECORDS, Bind::Buffer(records))?;
        cx.frame(faces::placed_tiles::PLACED_POSES, Bind::Buffer(poses))?;
        cx.frame(
            faces::placed_tiles::PLACED_GROUPS,
            Bind::Buffer(placed_groups),
        )?;
        cx.frame(placed::PLACED_VALUES, Bind::Buffer(values))?;
        let scatter = cx.pass("scatter").id();
        let snapping = cx.compute_pipeline(
            scatter,
            snap::SNAP_SCATTER,
            &[
                (snap::SNAP, Bind::Buffer(snap_list)),
                (snap::RECORDS, Bind::Buffer(records)),
                (snap::POSES, Bind::Buffer(poses)),
                (snap::VALUES, Bind::Buffer(values)),
            ],
        )?;
        let cull_bound = || {
            vec![
                (cull::FRAME, Bind::Buffer(frame)),
                (cull::RECORDS, Bind::Buffer(records)),
                (cull::GROUPS, Bind::Buffer(groups)),
                (cull::IDS, Bind::Buffer(ids)),
                (cull::DRAWS, Bind::Buffer(draws)),
                (cull::VIEWS, Bind::Buffer(views)),
                (cull::CARDS, Bind::Buffer(cards)),
                (cull::CARD_ARGS, Bind::Buffer(card_args)),
            ]
        };
        let laid = cx.view_texture("laid_depth", wgpu::TextureFormat::R32Float, Lifetime::Kept);
        let lay = cx
            .pass("scatter_laid")
            .at(AttachPoint::Inputs)
            .views(PassViews::Main)
            .label("engine/scatter-laid")
            .span("scatter_laid")
            .id();
        let laying = cx.compute_pipeline(
            lay,
            laid_shaders::COPY_LAID,
            &[
                (laid_shaders::SCENE_DEPTH, Bind::Target(TargetName::DEPTH)),
                (laid_shaders::LAID_OUT, Bind::Texture(laid)),
            ],
        )?;
        let hidden = Hidden::install(cx, lay, cull_bound())?;
        let mut bound = cull_bound();
        bound.push((cull::PICKS, Bind::Buffer(hidden.picks)));
        let culling = cx.compute_pipeline(scatter, cull::CULL_SCATTER, &bound)?;
        let forms = Self::forms(cx, (poses, colors, laid))?;
        let joined = Self::scenes(cx, &forms, (ids, draws, debug_records, cards, card_args));
        Ok(Self {
            scatter,
            snap_list,
            frame,
            views,
            records,
            groups,
            ids,
            draws,
            poses,
            colors,
            values,
            cards,
            card_args,
            placed_groups,
            debug_records,
            canopy,
            snapping,
            culling,
            lay,
            laid,
            laying,
            laid_size: (0, 0),
            laid_copied: None,
            forms,
            joined,
            records_written: false,
            held: Held::default(),
            hidden,
        })
    }

    fn prepare(
        &mut self,
        (state, canopy, tiles, ground, baking, residency, views, table): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        if state.is_added() {
            self.held = Held::default();
        }
        self.laid_copied = None;
        self.hidden.begin();
        let words = record_words();
        let records = cx.allocate_buffer(
            self.debug_records,
            std::mem::size_of_val(words.as_slice()) as u64,
        );
        if records == Allocation::Fresh {
            self.records_written = false;
        }
        if !self.records_written {
            cx.write(self.debug_records, 0, &words);
            self.records_written = true;
        }
        self.write_slots(cx, &state);
        self.list_draws(&state, &residency);
        self.write_placed_groups(cx, &state);
        self.canopy.prepare(cx, &canopy);
        self.prepare_snap(cx, (&state, &residency), (&tiles, &ground, &baking));
        self.prepare_cull(cx, &state, (&*views, &*table));
        if self.held.prepared && self.held.objects > 0 {
            cx.run(self.scatter);
        } else if self.held.snap.snapping {
            self.held.snap.snapped = None;
        }
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        let index = cx.view_index();
        let held = &self.held;
        let laid = held.prepared && held.laid_at(index);
        let main = laid && held.views.first().is_some_and(|first| first.drawn == index);
        if !(laid && (held.cards || main)) {
            return;
        }
        let size = view.size();
        let extent = wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        };
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(self.laid, extent, 1);
        self.laid_size = (view.render.x, view.render.y);
        if main && held.pyramid {
            self.hidden.view = Some(index);
        }
        cx.run(self.lay);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.lay {
            if self.held.cards {
                let (width, height) = self.laid_size;
                let groups = [width.div_ceil(LAID_GROUP), height.div_ceil(LAID_GROUP), 1];
                let copied = cx.dispatch(&self.laying, Count::Groups(groups)) != 0;
                self.laid_copied = copied.then(|| cx.view_index());
            }
            self.hidden.encode(cx, self.held.objects);
            return 0;
        }
        if pass == self.scatter {
            let snap = &mut self.held.snap;
            let snapped = [snap.threads.div_ceil(SCATTER_GROUP), 1, 1];
            if snap.snapping && cx.dispatch(&self.snapping, Count::Groups(snapped)) == 0 {
                snap.snapped = None;
            }
            let groups = [self.held.objects.div_ceil(SCATTER_GROUP), 1, 1];
            self.hidden.culled = cx.dispatch(&self.culling, Count::Groups(groups)) != 0;
        }
        0
    }
}
