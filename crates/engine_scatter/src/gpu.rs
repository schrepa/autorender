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
use crate::layouts::{
    CullFrame, CullGroup, CullView, DrawArgs, DrawKind, DrawRecord, LAID_GROUP, LEVELS,
    PlacedGroup, PlacedRecord, PlacedValues, PoseData, SCATTER_GROUP, SlotColor, Snap,
    placed_group, record_words,
};
use crate::pool::{SlotWrite, dead};
use crate::shaders::scatter::{
    cards as card_shaders, casters as caster_shaders, draw as object_shaders, laid as laid_shaders,
    matched as matched_shaders,
};
use crate::systems::{ScatterFeedback, ScatterState};

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
    snapped: Option<u64>,
    snap_again: bool,
    slots_written: bool,
    snapping: bool,
    prepared: bool,
    cards: bool,
    atlas: bool,
    placed_groups: Vec<PlacedGroup>,
    culling: Vec<(Listed, CullView)>,
    culled: Vec<CullView>,
    card_args: Vec<[u32; 4]>,
    args: Vec<DrawArgs>,
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
    laid_cards: Form<ScatterGpu>,
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

pub(crate) struct ScatterGpu {
    scatter: Pass<Self>,
    snap_uniform: Buffer<Self>,
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
        let owning = |entry: Entry, depth: FormDepth| -> Vec<PassFragment> {
            following
                .iter()
                .map(|pass| PassFragment {
                    pass: FragmentPass::Scene(pass.scene),
                    side: None,
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
            cx.form_owning(form, &owning(*entry, FormDepth::Matched), &placed)
        };
        let matched = [matched(0)?, matched(1)?, matched(2)?, matched(3)?];
        let card_bound: [(Var, Bind<ScatterGpu>); 3] = [
            (card_shaders::POSES, Bind::Buffer(poses)),
            (card_shaders::COLORS, Bind::Buffer(colors)),
            (card_shaders::LAID_DEPTH, Bind::Texture(laid)),
        ];
        let laid = owning(card_shaders::FS_PLACED_CARD_LAID, FormDepth::Written);
        Ok(Forms {
            plain,
            matched,
            extended: cx.form(&EXTENDED_FORM, &placed)?,
            fading: cx.form(&FADING_FORM, &placed)?,
            extended_fading: cx.form(&EXTENDED_FADING_FORM, &placed)?,
            cards: cx.form(&CARD_FORM, &card_bound[..2])?,
            laid_cards: cx.form_owning(&CARD_FORM, &laid, &card_bound)?,
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
                args: cx.draw_input(scene, draws),
                records: cx.draw_input(scene, records),
                cards: cx.draw_input(scene, cards),
                card_args: cx.draw_input(scene, card_args),
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

    fn write_slot(&self, cx: &mut Prepare<'_, Self>, write: &SlotWrite, objects: u32) -> bool {
        match write {
            SlotWrite::Dead(slots) => {
                let slots = slots.start.min(objects)..slots.end.min(objects);
                if slots.is_empty() {
                    return false;
                }
                cx.write(
                    self.records,
                    u64::from(slots.start) * size_of::<PlacedRecord>() as u64,
                    &vec![dead(); (slots.end - slots.start) as usize],
                );
                true
            }
            SlotWrite::Objects(start, (poses, colors, records, values)) => {
                let count = objects.saturating_sub(*start).min(records.len() as u32) as usize;
                if count == 0 {
                    return false;
                }
                let start = u64::from(*start);
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
                true
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
        for ((buffer, bytes), stride) in buffers.into_iter().zip(state.slot_bytes).zip(strides) {
            fresh |= cx.allocate_buffer(buffer, bytes.max(stride as u64)) == Allocation::Fresh;
        }
        let objects = state.objects;
        if fresh || self.held.uploaded != Some(state.slot_sequence) {
            let mut wrote = false;
            let missed = self.held.uploaded.map(|seen| seen + 1) != Some(state.slot_sequence);
            if fresh || missed || state.whole {
                for write in state.pool.whole(&state.groups) {
                    wrote |= self.write_slot(cx, &write, objects);
                }
            }
            for write in &state.pool.writes {
                wrote |= self.write_slot(cx, write, objects);
            }
            self.held.slots_written |= wrote;
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
        state: &ScatterState,
        (tiles, ground, baking): (
            &Res<'_, TerrainTiles>,
            &Res<'_, GroundTiles>,
            &Res<'_, engine_open_world_kit::ground::GroundBaking>,
        ),
    ) {
        let held = &mut self.held;
        held.snapping = false;
        let terrains = tiles.terrains.len() as u32;
        if cx.allocate_buffer(self.snap_uniform, size_of::<Snap>() as u64) == Allocation::Fresh {
            held.snapped = None;
        }
        if ground.layers == 0 || terrains == 0 || state.resident == 0 {
            return;
        }
        let changed = tiles.is_changed()
            || ground.is_changed()
            || baking.is_changed()
            || held.slots_written
            || held.snapped != Some(state.generation);
        if !changed && !held.snap_again {
            return;
        }
        held.snap_again = changed;
        held.snapped = Some(state.generation);
        held.slots_written = false;
        let snap = Snap {
            counts: [held.objects, terrains, 0, 0],
        };
        cx.write(self.snap_uniform, 0, &[snap]);
        held.snapping = true;
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
        held.atlas = plan.atlas;
        held.views.clear();
        held.views.extend(listed.iter().map(|(view, _)| *view));
        if state.objects == 0 {
            held.sizes = Sizes::default();
        }
        let view_count = listed.len() as u32;
        let lists = u64::from(view_count) + 1;
        let ids = u64::from(state.group_data.ids_per_view) * lists * 4;
        let draws = u64::from(held.group_count * LEVELS) * lists * size_of::<DrawArgs>() as u64;
        let cards = u64::from(state.objects) * u64::from(view_count) * 8;
        let sizes = &mut held.sizes;
        grow(
            &mut sizes.views,
            u64::from(view_count) * size_of::<CullView>() as u64,
            state.room,
        );
        grow(&mut sizes.ids, ids, state.room);
        grow(&mut sizes.draws, draws, state.room);
        grow(&mut sizes.cards, cards, state.room);
        grow(&mut sizes.card_args, lists * 16, state.room);
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
        frame.drawn[0] = drawn;
        held.fading_view = listed.len();
        let card_args = &mut held.card_args;
        card_args.clear();
        card_args.resize(listed.len(), [IMPOSTOR_VERTICES, 0, 0, 0]);
        card_args.push([0; 4]);
        let args = &mut held.args;
        args.clear();
        for _ in 0..=listed.len() {
            args.extend_from_slice(&state.group_data.template);
        }
        let culled = &mut held.culled;
        culled.clear();
        culled.extend(listed.drain(..).map(|(_, view)| view));
        cx.write(self.frame, 0, &[frame]);
        cx.write(self.card_args, 0, card_args);
        cx.write(self.views, 0, culled);
        if !args.is_empty() {
            cx.write(self.draws, 0, args);
        }
        let left = ids > state.room || cards > state.room;
        if (left || held.left_out) && held.objects > 0 {
            let at = u64::from(view_count) * 16;
            cx.read_back::<ScatterFeedback>(LEFT_OUT, self.card_args, at..at + 16, AfterRead::Keep);
        }
        held.left_out = left;
        held.prepared = true;
    }

    fn draw_source(&self, view: usize, group: usize, level: usize) -> Option<(u64, u64)> {
        let held = &self.held;
        let &(ids, count) = held.lists.get(group)?;
        let start = (view as u64 * u64::from(held.ids_per_view)
            + u64::from(ids + level as u32 * count))
            * 4;
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
        let drawn = if joined.owned {
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
        match (drawn, &self.held.refused, waiting) {
            (0, Some(error), _) => draw.refuse(PLACED_MESHES, error),
            (0, None, true) => draw.wait(PLACED_MESHES),
            (0, None, false) => draw.skip("nothing placed reaches this view"),
            (drawn, _, _) => drawn,
        }
    }

    fn draw_levels(
        &self,
        (view, items, fading): (usize, &[Level], bool),
        (scene, matched): (&SceneDraw, bool),
        draw: &mut Draw<'_, Self, Scene>,
        inputs: &Streams,
        pipelines: [&Pair; 2],
    ) -> u32 {
        let mut draws = 0;
        let turns: &[Option<bool>] = match scene.role {
            PassRole::Depth => &[Some(false), Some(true)],
            PassRole::Lit | PassRole::Blended | PassRole::Shown => &[None],
        };
        let listed = turns.iter().flat_map(|turn| {
            items.iter().filter(move |item| {
                (!fading || item.fades) && turn.is_none_or(|masked| item.masked == masked)
            })
        });
        for item in listed {
            let pair = pipelines[usize::from(item.masked)];
            let Some(pipeline) = &pair[usize::from(item.extended)] else {
                continue;
            };
            let level = item.level as usize;
            let Some((ids, offset)) = self.draw_source(view, item.group as usize, level) else {
                continue;
            };
            let record = DrawRecord::of(DrawKind::Placed, level);
            draw.draw_timed(
                pipeline,
                Shape::MeshIndirect {
                    mesh: item.mesh,
                    material: item.material,
                    args: inputs.args,
                    offset,
                    streams: &[(inputs.ids, ids), (inputs.records, record.offset())],
                },
                || level_label(scene, item, (matched, fading)),
            );
            draws += 1;
        }
        draws
    }

    fn draw_objects(
        &self,
        (scene, view, matched): (&SceneDraw, usize, bool),
        draw: &mut Draw<'_, Self, Scene>,
        inputs: &Streams,
    ) -> u32 {
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
        let lists: &[(usize, bool, [&Pair; 2])] = match (view, held.cards) {
            (0, true) => &[
                (0, false, [&plain, &cut]),
                (held.fading_view, true, [&fading, &cut_fading]),
            ],
            _ => &[(view, false, [&plain, &cut])],
        };
        let mut draws = 0;
        for &(list, fades, pipelines) in lists {
            draws += self.draw_levels(
                (list, &held.lit, fades),
                (scene, matched),
                draw,
                inputs,
                pipelines,
            );
        }
        if !held.cards {
            return draws;
        }
        let laid = matched && self.laid_copied.is_some() && self.laid_copied == scene.index;
        let card = draw.form(if laid {
            &self.forms.laid_cards
        } else {
            &self.forms.cards
        });
        if let (Some(card), Some((cards, offset))) = (card, self.card_source(view)) {
            let record = DrawRecord::of(DrawKind::PlacedCard, 0);
            draw.draw_timed(
                &card,
                Shape::IndirectAt {
                    vertices: &[(inputs.cards, cards), (inputs.records, record.offset())],
                    indices: None,
                    args: inputs.card_args,
                    offset,
                },
                || scene_label(scene, if laid { "cards laid" } else { "cards" }),
            );
            draws += 1;
        }
        draws
    }

    fn draw_views(
        &self,
        (scene, view): (&SceneDraw, usize),
        draw: &mut Draw<'_, Self, Scene>,
        inputs: &Streams,
    ) -> u32 {
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
        let mut draws = self.draw_levels(
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
        draws
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
        let snap_uniform = cx.buffer("snap", Lifetime::Kept);
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
                (snap::SNAP, Bind::Buffer(snap_uniform)),
                (snap::RECORDS, Bind::Buffer(records)),
                (snap::POSES, Bind::Buffer(poses)),
                (snap::VALUES, Bind::Buffer(values)),
            ],
        )?;
        let culling = cx.compute_pipeline(
            scatter,
            cull::CULL_SCATTER,
            &[
                (cull::FRAME, Bind::Buffer(frame)),
                (cull::RECORDS, Bind::Buffer(records)),
                (cull::GROUPS, Bind::Buffer(groups)),
                (cull::IDS, Bind::Buffer(ids)),
                (cull::DRAWS, Bind::Buffer(draws)),
                (cull::VIEWS, Bind::Buffer(views)),
                (cull::CARDS, Bind::Buffer(cards)),
                (cull::CARD_ARGS, Bind::Buffer(card_args)),
            ],
        )?;
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
        let forms = Self::forms(cx, (poses, colors, laid))?;
        let joined = Self::scenes(cx, &forms, (ids, draws, debug_records, cards, card_args));
        Ok(Self {
            scatter,
            snap_uniform,
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
        self.prepare_snap(cx, &state, (&tiles, &ground, &baking));
        self.prepare_cull(cx, &state, (&*views, &*table));
        if self.held.prepared && self.held.objects > 0 {
            cx.run(self.scatter);
        } else if self.held.snapping {
            self.held.snapped = None;
        }
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        let index = cx.view_index();
        let held = &self.held;
        let laid = held
            .views
            .iter()
            .any(|listed| listed.drawn == index && listed.laid);
        if !(held.prepared && held.cards && laid) {
            return;
        }
        let size = view.size();
        let extent = wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        };
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(self.laid, extent, 1);
        self.laid_size = (size.x, size.y);
        cx.run(self.lay);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.lay {
            let (width, height) = self.laid_size;
            let groups = [width.div_ceil(LAID_GROUP), height.div_ceil(LAID_GROUP), 1];
            let copied = cx.dispatch(&self.laying, Count::Groups(groups)) != 0;
            self.laid_copied = copied.then(|| cx.view_index());
            return 0;
        }
        if pass == self.scatter {
            let groups = [self.held.objects.div_ceil(SCATTER_GROUP), 1, 1];
            if self.held.snapping && cx.dispatch(&self.snapping, Count::Groups(groups)) == 0 {
                self.held.snapped = None;
            }
            cx.dispatch(&self.culling, Count::Groups(groups));
        }
        0
    }
}
