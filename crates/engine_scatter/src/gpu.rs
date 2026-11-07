use bevy_ecs::prelude::*;
use engine_core::{
    AssetResidency, ColorPass, Cull, Cursor, DepthPass, Error, MaterialHandle, MeshHandle,
    PassName, PassViews, ScenePass, Since, Var,
};
use engine_places_world::ground::{GroundTiles, TerrainTiles};
use engine_render::{
    Access, AfterRead, Allocation, Attach, Bind, Buffer, ComputePipeline, Count, DepthUse, Draw,
    DrawInput, Encode, GpuPart, Install, Lifetime, MeshPass, Own, Pass, Pipeline, Prepare,
    RenderPipeline, SamplerDesc, Scene, SceneDraw, Shape, Texture, VertexLayout, wgpu,
};

use crate::far_cover::FarCoverTable;
use crate::layouts::{
    CELLS, Cover, DrawArgs, DrawKind, DrawRecord, GroundData, LEVELS, ListedDraw,
    MAX_OCCUPANCY_LEVELS, PlacedGroup, PlacedRecord, PoseData, QUAD, SCATTER_GROUP, SlotColor,
    Snap, Splat, placed_group, record_words,
};
use crate::occupancy::{ALL_CELLS, LISTED_RESET, OccupancyWindows};
use crate::pool::{SlotWrite, dead};
use crate::systems::{ScatterFeedback, ScatterState};

const CASTER_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: 1,
    slope_scale: 0.0,
    clamp: 0.0,
};

const IDS: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 4,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &[wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32,
        offset: 0,
        shader_location: 2,
    }],
};

const RECORD: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 0,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &[wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 0,
        shader_location: 14,
    }],
};

const CARDS: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 8,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 0,
            shader_location: 2,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 4,
            shader_location: 3,
        },
    ],
};

const PLAIN_STREAMS: &[VertexLayout] = &[
    VertexLayout::Mesh,
    VertexLayout::Buffer(IDS),
    VertexLayout::Buffer(RECORD),
];

const EXTENDED_STREAMS: &[VertexLayout] = &[
    VertexLayout::Mesh,
    VertexLayout::Mesh,
    VertexLayout::Buffer(IDS),
    VertexLayout::Buffer(RECORD),
];

const CASTER_STREAMS: &[VertexLayout] = &[VertexLayout::Mesh, VertexLayout::Buffer(IDS)];

const CARD_STREAMS: &[VertexLayout] = &[VertexLayout::Buffer(CARDS), VertexLayout::Buffer(RECORD)];

const SHADOW_CARD_STREAMS: &[VertexLayout] = &[VertexLayout::Buffer(CARDS)];

const OCCUPANCY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub(crate) const PLACED: engine_core::DrawKind = engine_core::DrawKind("placed");

const LEFT_OUT: u32 = 0;

struct MapPipelines {
    select: ComputePipeline<ScatterGpu>,
    clear: Pipeline<ScatterGpu, Own>,
    splat: Pipeline<ScatterGpu, Own>,
    means: Pipeline<ScatterGpu, Own>,
    listed: DrawInput<ScatterGpu, Own>,
}

struct LitLevel {
    mesh: MeshHandle,
    material: MaterialHandle,
    group: u32,
    level: u32,
    look: usize,
    masked: bool,
}

struct CastLevel {
    mesh: MeshHandle,
    material: Option<MaterialHandle>,
    cuts: bool,
    group: u32,
    level: u32,
}

#[derive(Default)]
struct Held {
    lit: Vec<LitLevel>,
    cast: Vec<CastLevel>,
    lists: Vec<(u32, u32)>,
    ids_per_view: u32,
    group_count: u32,
    maps: Vec<u32>,
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
    occupancy: [(bool, bool); MAX_OCCUPANCY_LEVELS],
    occupancy_seen: Cursor,
    occupancy_asked: bool,
    workgroups: u32,
    covers: u64,
    listed: u64,
}

type CellWords = [[u32; 4]; 2];

fn joined(held: &mut CellWords, more: CellWords) {
    for (row, more) in held.iter_mut().zip(more) {
        for (word, more) in row.iter_mut().zip(more) {
            *word |= more;
        }
    }
}

type Looks = Vec<[Pipeline<ScatterGpu, Scene>; 2]>;

type DrawBuffers = (
    Buffer<ScatterGpu>,
    Buffer<ScatterGpu>,
    Buffer<ScatterGpu>,
    Buffer<ScatterGpu>,
    Buffer<ScatterGpu>,
);

enum Joined {
    Lit {
        looks: Looks,
        card: Pipeline<ScatterGpu, Scene>,
    },
    Depth {
        looks: Looks,
    },
    Casters {
        casting: [Pipeline<ScatterGpu, Scene>; 2],
        card: Pipeline<ScatterGpu, Scene>,
    },
}

#[derive(Clone, Copy)]
struct Streams {
    ids: DrawInput<ScatterGpu, Scene>,
    args: DrawInput<ScatterGpu, Scene>,
    records: DrawInput<ScatterGpu, Scene>,
    cards: DrawInput<ScatterGpu, Scene>,
    card_args: DrawInput<ScatterGpu, Scene>,
}

struct ScenePipelines {
    name: &'static str,
    joined: Joined,
    streams: Streams,
}

pub(crate) struct ScatterGpu {
    scatter: Pass<Self>,
    occupancy: Pass<Self>,
    snap_uniform: Buffer<Self>,
    frame: Buffer<Self>,
    views: Buffer<Self>,
    records: Buffer<Self>,
    groups: Buffer<Self>,
    ids: Buffer<Self>,
    draws: Buffer<Self>,
    poses: Buffer<Self>,
    colors: Buffer<Self>,
    grounds: Buffer<Self>,
    cards: Buffer<Self>,
    card_args: Buffer<Self>,
    placed_groups: Buffer<Self>,
    debug_records: Buffer<Self>,
    covers: Buffer<Self>,
    listed: Buffer<Self>,
    splats: Vec<Buffer<Self>>,
    listed_draws: Vec<Buffer<Self>>,
    placed_far: Buffer<Self>,
    occupancy_map: Texture<Self>,
    unfiner: Texture<Self>,
    snapping: ComputePipeline<Self>,
    culling: ComputePipeline<Self>,
    maps: Vec<MapPipelines>,
    joined: Vec<ScenePipelines>,
    records_written: bool,
    held: Held,
}

fn grow(held: &mut u64, bytes: u64, room: u64) {
    let size = bytes.next_power_of_two().min(room);
    if bytes > *held && size > *held {
        *held = size;
    }
}

fn component(
    dst_factor: wgpu::BlendFactor,
    operation: wgpu::BlendOperation,
) -> wgpu::BlendComponent {
    wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor,
        operation,
    }
}

fn layer(texels: u32, layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: texels,
        height: texels,
        depth_or_array_layers: layers,
    }
}

fn look(masked: bool, double_sided: bool, extended: bool) -> usize {
    usize::from(masked) << 2 | usize::from(double_sided) << 1 | usize::from(extended)
}

fn primitive(double_sided: bool) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: (!double_sided).then_some(wgpu::Face::Back),
        ..Default::default()
    }
}

fn objects(
    (poses, colors, grounds): (Var, Var, Var),
    (pose_buffer, color_buffer, ground_buffer): (
        Buffer<ScatterGpu>,
        Buffer<ScatterGpu>,
        Buffer<ScatterGpu>,
    ),
) -> [(Var, Bind<ScatterGpu>); 3] {
    [
        (poses, Bind::Buffer(pose_buffer)),
        (colors, Bind::Buffer(color_buffer)),
        (grounds, Bind::Buffer(ground_buffer)),
    ]
}

impl ScatterGpu {
    fn map_pipelines(
        cx: &mut Install<'_, Self>,
        map: usize,
        pass: Pass<Self>,
        (splat, listed_draw): (Buffer<Self>, Buffer<Self>),
        (records, poses, covers, listed): (Buffer<Self>, Buffer<Self>, Buffer<Self>, Buffer<Self>),
        (occupancy, unfiner): (Texture<Self>, Texture<Self>),
    ) -> Result<MapPipelines, Error> {
        use crate::shaders::scatter::occupancy_maps as maps;
        let target = |vertex, fragment| RenderPipeline {
            vertex,
            fragment: Some(fragment),
            debug: None,
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        };
        let finer = |texture: Texture<Self>, held: Texture<Self>| match map.checked_sub(1) {
            Some(finer) => Bind::Slice {
                texture,
                mips: 0..1,
                layers: finer as u32..map as u32,
            },
            None => Bind::Texture(held),
        };
        let written = wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN | wgpu::ColorWrites::ALPHA;
        let splatted = Some(wgpu::BlendState {
            color: component(wgpu::BlendFactor::One, wgpu::BlendOperation::Max),
            alpha: component(wgpu::BlendFactor::OneMinusSrc, wgpu::BlendOperation::Add),
        });
        let select = cx.compute_pipeline(
            pass,
            maps::SELECT_COVERS,
            &[
                (maps::SPLAT, Bind::Buffer(splat)),
                (maps::RECORDS, Bind::Buffer(records)),
                (maps::POSES, Bind::Buffer(poses)),
                (maps::COVERS, Bind::Buffer(covers)),
                (maps::LISTING, Bind::Buffer(listed)),
                (maps::LISTED_DRAW, Bind::Buffer(listed_draw)),
            ],
        )?;
        let clear = cx.render_pipeline_targets(
            pass,
            &target(maps::VS_CLEAR, maps::FS_CLEAR),
            &[(None, written)],
            &[(maps::SPLAT, Bind::Buffer(splat))],
        )?;
        let splat_pipeline = cx.render_pipeline_targets(
            pass,
            &target(maps::VS_SPLAT, maps::FS_SPLAT),
            &[(splatted, written)],
            &[
                (maps::SPLAT, Bind::Buffer(splat)),
                (maps::RECORDS, Bind::Buffer(records)),
                (maps::POSES, Bind::Buffer(poses)),
                (maps::COVERS, Bind::Buffer(covers)),
                (maps::LISTED, Bind::Buffer(listed)),
            ],
        )?;
        let means = cx.render_pipeline_targets(
            pass,
            &target(maps::VS_MEANS, maps::FS_MEANS),
            &[(None, written)],
            &[
                (maps::SPLAT, Bind::Buffer(splat)),
                (maps::FINER_MAP, finer(occupancy, unfiner)),
            ],
        )?;
        let listed = cx.draw_input(pass, listed_draw);
        Ok(MapPipelines {
            select,
            clear,
            splat: splat_pipeline,
            means,
            listed,
        })
    }

    fn look_pipelines(
        cx: &mut Install<'_, Self>,
        scene: Scene,
        (poses, colors, grounds): (Buffer<Self>, Buffer<Self>, Buffer<Self>),
        depth_only: bool,
    ) -> Result<Looks, Error> {
        use crate::shaders::scatter::draw;
        let vars = (draw::POSES, draw::COLORS, draw::GROUNDS);
        let buffers = (poses, colors, grounds);
        let mut looks = Vec::with_capacity(8);
        for index in 0..8_usize {
            let (masked, double_sided, extended) = (index & 4 != 0, index & 2 != 0, index & 1 != 0);
            let (vertices, vertex, fading) = if extended {
                (
                    EXTENDED_STREAMS,
                    draw::VS_OBJECT_EXTENDED,
                    draw::VS_OBJECT_EXTENDED_FADING,
                )
            } else {
                (PLAIN_STREAMS, draw::VS_OBJECT, draw::VS_OBJECT_FADING)
            };
            let (fragment, debug, masked_depth) = match (extended, masked) {
                (false, false) => (
                    draw::FS_OBJECT,
                    draw::FS_DEBUG_OBJECT,
                    draw::FS_MASKED_DEPTH,
                ),
                (false, true) => (
                    draw::FS_OBJECT_PRETESTED,
                    draw::FS_DEBUG_OBJECT,
                    draw::FS_MASKED_DEPTH,
                ),
                (true, false) => (
                    draw::FS_OBJECT_EXTENDED,
                    draw::FS_DEBUG_OBJECT_EXTENDED,
                    draw::FS_MASKED_DEPTH_EXTENDED,
                ),
                (true, true) => (
                    draw::FS_OBJECT_EXTENDED_PRETESTED,
                    draw::FS_DEBUG_OBJECT_EXTENDED,
                    draw::FS_MASKED_DEPTH_EXTENDED,
                ),
            };
            let written = DepthUse::Write {
                bias: wgpu::DepthBiasState::default(),
            };
            let (shaded, shown, depth, writes) = match (depth_only, masked) {
                (true, false) => continue,
                (true, true) => (
                    masked_depth,
                    masked_depth,
                    written,
                    wgpu::ColorWrites::empty(),
                ),
                (false, true) => (
                    fragment,
                    debug,
                    DepthUse::Compare {
                        compare: wgpu::CompareFunction::Equal,
                        write: false,
                        bias: wgpu::DepthBiasState::default(),
                    },
                    wgpu::ColorWrites::ALL,
                ),
                (false, false) => (fragment, debug, written, wgpu::ColorWrites::ALL),
            };
            let desc = |vertex| RenderPipeline {
                vertex,
                fragment: Some(shaded),
                debug: Some(shown),
                vertices,
                primitive: primitive(double_sided),
                depth,
                blend: None,
                writes,
            };
            looks.push([
                cx.render_pipeline(scene, &desc(vertex), &objects(vars, buffers))?,
                cx.render_pipeline(scene, &desc(fading), &objects(vars, buffers))?,
            ]);
        }
        Ok(looks)
    }

    fn card_pipeline(
        cx: &mut Install<'_, Self>,
        scene: Scene,
        (poses, colors): (Buffer<Self>, Buffer<Self>),
    ) -> Result<Pipeline<Self, Scene>, Error> {
        use crate::shaders::scatter::cards;
        cx.render_pipeline(
            scene,
            &RenderPipeline {
                vertex: cards::VS_PLACED_CARD,
                fragment: Some(cards::FS_PLACED_CARD),
                debug: Some(cards::FS_DEBUG_PLACED_CARD),
                vertices: CARD_STREAMS,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth: DepthUse::Write {
                    bias: wgpu::DepthBiasState::default(),
                },
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &[
                (cards::POSES, Bind::Buffer(poses)),
                (cards::COLORS, Bind::Buffer(colors)),
            ],
        )
    }

    fn caster_pipelines(
        cx: &mut Install<'_, Self>,
        scene: Scene,
        (poses, colors): (Buffer<Self>, Buffer<Self>),
    ) -> Result<Joined, Error> {
        use crate::shaders::scatter::casters;
        let caster_desc = |vertex, fragment| RenderPipeline {
            vertex,
            fragment,
            debug: None,
            vertices: CASTER_STREAMS,
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::Write { bias: CASTER_BIAS },
            blend: None,
            writes: wgpu::ColorWrites::empty(),
        };
        let caster_bound = || {
            [
                (casters::POSES, Bind::Buffer(poses)),
                (casters::COLORS, Bind::Buffer(colors)),
            ]
        };
        let casting = [
            cx.render_pipeline(
                scene,
                &caster_desc(casters::VS_CASTER, None),
                &caster_bound(),
            )?,
            cx.render_pipeline(
                scene,
                &caster_desc(casters::VS_CASTER_MASKED, Some(casters::FS_CASTER_MASKED)),
                &caster_bound(),
            )?,
        ];
        let card = cx.render_pipeline(
            scene,
            &RenderPipeline {
                vertex: casters::VS_IMPOSTOR_CARD,
                fragment: Some(casters::FS_IMPOSTOR_CARD),
                debug: None,
                vertices: SHADOW_CARD_STREAMS,
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::Write { bias: CASTER_BIAS },
                blend: None,
                writes: wgpu::ColorWrites::empty(),
            },
            &[(casters::POSES, Bind::Buffer(poses))],
        )?;
        Ok(Joined::Casters { casting, card })
    }

    fn scene_pipelines(
        cx: &mut Install<'_, Self>,
        (poses, colors, grounds): (Buffer<Self>, Buffer<Self>, Buffer<Self>),
        (ids, draws, records, cards, card_args): DrawBuffers,
    ) -> Result<Vec<ScenePipelines>, Error> {
        let listed: Vec<ScenePass> = cx
            .passes()
            .listed()
            .iter()
            .filter(|listed| listed.bit.is_some() && (listed.pass.takes)(PLACED))
            .copied()
            .collect();
        let objects = (poses, colors, grounds);
        let mut joined = Vec::with_capacity(listed.len());
        for listed in listed {
            let scene = Scene::named(listed.name);
            let made = if listed.name == ColorPass::NAME {
                Joined::Lit {
                    looks: Self::look_pipelines(cx, scene, objects, false)?,
                    card: Self::card_pipeline(cx, scene, (poses, colors))?,
                }
            } else if listed.name == DepthPass::NAME {
                Joined::Depth {
                    looks: Self::look_pipelines(cx, scene, objects, true)?,
                }
            } else if listed.pass.views == PassViews::Owned && !listed.pass.reads.base_color {
                Self::caster_pipelines(cx, scene, (poses, colors))?
            } else {
                continue;
            };
            let streams = Streams {
                ids: cx.draw_input(scene, ids),
                args: cx.draw_input(scene, draws),
                records: cx.draw_input(scene, records),
                cards: cx.draw_input(scene, cards),
                card_args: cx.draw_input(scene, card_args),
            };
            joined.push(ScenePipelines {
                name: listed.name,
                joined: made,
                streams,
            });
        }
        Ok(joined)
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
        for (group, drawn) in state.groups.groups.iter().enumerate() {
            if drawn.count == 0 {
                continue;
            }
            let masked = drawn.masked();
            let shape = look(masked, drawn.source.cull() == Cull::None, drawn.extended);
            for (level, &mesh) in drawn.meshes.iter().enumerate() {
                if residency.resident(mesh) {
                    held.lit.push(LitLevel {
                        mesh,
                        material: drawn.material,
                        group: group as u32,
                        level: level as u32,
                        look: shape,
                        masked,
                    });
                }
            }
            let Some(cuts) = drawn.casts else {
                continue;
            };
            let resident = residency.material_resident(drawn.material);
            if cuts && !resident {
                continue;
            }
            for (level, &mesh) in drawn.casters.iter().enumerate() {
                if residency.resident(mesh) {
                    held.cast.push(CastLevel {
                        mesh,
                        material: resident.then_some(drawn.material),
                        cuts,
                        group: group as u32,
                        level: level as u32,
                    });
                }
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
            SlotWrite::Objects(start, (poses, colors, records, grounds)) => {
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
                    self.grounds,
                    start * size_of::<GroundData>() as u64,
                    &grounds[..count],
                );
                true
            }
        }
    }

    fn write_slots(&mut self, cx: &mut Prepare<'_, Self>, state: &ScatterState) {
        let buffers = [self.poses, self.colors, self.records, self.grounds];
        let strides = [
            size_of::<PoseData>(),
            size_of::<SlotColor>(),
            size_of::<PlacedRecord>(),
            size_of::<GroundData>(),
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
        let groups: Vec<PlacedGroup> = state
            .impostors
            .iter()
            .zip(&state.groups.colors)
            .map(|(&impostor, &color)| placed_group(impostor, color))
            .collect();
        let bytes = (groups.len().max(1) * size_of::<PlacedGroup>()) as u64;
        cx.allocate_buffer(self.placed_groups, bytes);
        if !groups.is_empty() {
            cx.write(self.placed_groups, 0, &groups);
        }
    }

    fn prepare_snap(
        &mut self,
        cx: &mut Prepare<'_, Self>,
        state: &ScatterState,
        (tiles, ground, baking): (
            &Res<'_, TerrainTiles>,
            &Res<'_, GroundTiles>,
            &Res<'_, engine_places_world::ground::GroundBaking>,
        ),
    ) {
        let held = &mut self.held;
        held.snapping = false;
        let terrains = tiles.terrains.len() as u32;
        cx.allocate_buffer(self.snap_uniform, size_of::<Snap>() as u64);
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

    fn prepare_cull(&mut self, cx: &mut Prepare<'_, Self>, state: &ScatterState) {
        let held = &mut self.held;
        let plan = &state.plan;
        held.maps.clear();
        held.prepared = false;
        held.cards = plan.cards;
        held.atlas = plan.atlas;
        let sizes = state.sizes;
        cx.allocate_buffer(
            self.groups,
            sizes
                .groups
                .max(size_of::<crate::layouts::CullGroup>() as u64),
        );
        cx.allocate_buffer(self.views, sizes.views);
        cx.allocate_buffer(self.ids, sizes.ids);
        cx.allocate_buffer(self.draws, sizes.draws);
        cx.allocate_buffer(self.cards, sizes.cards);
        cx.allocate_buffer(self.card_args, sizes.card_args);
        cx.allocate_buffer(self.frame, size_of::<crate::layouts::CullFrame>() as u64);
        held.ids_room = sizes.ids;
        held.cards_room = sizes.cards;
        if !state.group_data.data.is_empty() {
            cx.write(self.groups, 0, &state.group_data.data);
        }
        let Some(frame) = plan.frame.filter(|_| plan.prepared) else {
            return;
        };
        held.maps.extend_from_slice(&plan.maps);
        held.fading_view = plan.fading_view;
        cx.write(self.frame, 0, &[frame]);
        cx.write(self.card_args, 0, &plan.card_args);
        cx.write(self.views, 0, &plan.views);
        if !plan.args.is_empty() {
            cx.write(self.draws, 0, &plan.args);
        }
        if plan.lists_left && held.objects > 0 {
            let at = plan.fading_view as u64 * 16;
            cx.read_back::<ScatterFeedback>(LEFT_OUT, self.card_args, at..at + 16, AfterRead::Keep);
        }
        held.prepared = true;
    }

    fn prepare_occupancy(
        &mut self,
        cx: &mut Prepare<'_, Self>,
        windows: &OccupancyWindows,
        covers: &[Cover],
        room: u64,
    ) {
        let held = &mut self.held;
        let (texels, layers) = if windows.runs && windows.maps > 0 {
            (windows.resolution.max(1), windows.maps as u32)
        } else {
            (1, 1)
        };
        let fresh = cx.allocate(self.occupancy_map, layer(texels, layers), 1) == Allocation::Fresh;
        cx.allocate(self.unfiner, layer(1, 1), 1);
        let cover = size_of::<Cover>() as u64;
        let word = size_of::<u32>() as u64;
        if !windows.runs {
            held.covers = cover;
            held.listed = word;
        }
        held.covers = held.covers.max(cover);
        held.listed = held.listed.max(word);
        let covers = if windows.runs { covers } else { &[] };
        grow(&mut held.covers, std::mem::size_of_val(covers) as u64, room);
        if windows.runs && windows.slots > 0 && !covers.is_empty() {
            grow(&mut held.listed, 2 * u64::from(windows.slots) * word, room);
        }
        cx.allocate_buffer(self.covers, held.covers);
        cx.allocate_buffer(self.listed, held.listed);
        if !covers.is_empty() {
            cx.write(self.covers, 0, covers);
        }
        held.workgroups = windows.workgroups;
        held.occupancy = [(false, false); MAX_OCCUPANCY_LEVELS];
        for (&splat, &listed) in self.splats.iter().zip(&self.listed_draws) {
            cx.allocate_buffer(splat, size_of::<Splat>() as u64);
            cx.allocate_buffer(listed, size_of::<ListedDraw>() as u64);
        }
        let missed = std::mem::take(&mut held.occupancy_asked) && !cx.ran(self.occupancy);
        let mut wanted = [(CellWords::default(), CellWords::default()); MAX_OCCUPANCY_LEVELS];
        let whole = match windows.changes.since(&mut held.occupancy_seen) {
            Since::Same => false,
            Since::Listed(changes) => {
                for change in changes {
                    if let Some((dirty, means)) = wanted.get_mut(change.map) {
                        joined(dirty, change.dirty);
                        joined(means, change.means);
                    }
                }
                false
            }
            Since::All => true,
        } || fresh
            || missed;
        for (map, frame) in windows.frames.iter().enumerate() {
            let Some((dirty, means)) = wanted.get_mut(map).filter(|_| frame.held) else {
                continue;
            };
            if whole {
                *dirty = ALL_CELLS;
                if frame.splat.finer[2] != 0 {
                    *means = ALL_CELLS;
                }
            }
            let redraw = *dirty != CellWords::default();
            let average = *means != CellWords::default();
            if !redraw && !average {
                continue;
            }
            let splat = Splat {
                dirty: *dirty,
                means: *means,
                ..frame.splat
            };
            cx.write(self.splats[map], 0, &[splat]);
            if redraw {
                cx.write(self.listed_draws[map], 0, &[LISTED_RESET]);
            }
            held.occupancy[map] = (redraw, average);
        }
        if held
            .occupancy
            .iter()
            .any(|&(redraw, average)| redraw || average)
        {
            held.occupancy_asked = true;
            cx.run(self.occupancy);
        }
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

    fn map_view(&self, region: u32) -> Option<usize> {
        self.held
            .maps
            .iter()
            .position(|&drawn| drawn == region)
            .map(|index| index + 1)
    }

    pub(crate) fn draw(&self, scene: &SceneDraw, draw: &mut Draw<'_, Self, Scene>) -> u32 {
        let Some(joined) = self
            .joined
            .iter()
            .find(|joined| joined.name == scene.pass.name())
        else {
            return 0;
        };
        let streams = &joined.streams;
        match &joined.joined {
            Joined::Lit { looks, card } => self.draw_objects(draw, streams, looks, Some(card)),
            Joined::Depth { looks } => self.draw_objects(draw, streams, looks, None),
            Joined::Casters { casting, card } => {
                self.draw_casters(scene.view, draw, streams, (casting, card))
            }
        }
    }

    fn draw_objects(
        &self,
        draw: &mut Draw<'_, Self, Scene>,
        inputs: &Streams,
        looks: &Looks,
        card: Option<&Pipeline<Self, Scene>>,
    ) -> u32 {
        let held = &self.held;
        if !held.prepared {
            return 0;
        }
        let prepass = card.is_none();
        let mut draws = 0;
        for (view, fades) in [(0, false), (held.fading_view, true)] {
            for item in &held.lit {
                let pipeline = if prepass {
                    if !item.masked {
                        continue;
                    }
                    &looks[item.look & 3][usize::from(fades)]
                } else {
                    &looks[item.look][usize::from(fades)]
                };
                let level = item.level as usize;
                let Some((ids, offset)) = self.draw_source(view, item.group as usize, level) else {
                    continue;
                };
                let record = DrawRecord::of(DrawKind::Placed, level);
                draw.draw_with(
                    pipeline,
                    Shape::MeshIndirect {
                        mesh: item.mesh,
                        material: Some(item.material),
                        args: inputs.args,
                        offset,
                        streams: &[(inputs.ids, ids), (inputs.records, record.offset())],
                    },
                    MeshPass { fading: fades },
                );
                draws += 1;
            }
        }
        let Some(card) = card.filter(|_| held.cards) else {
            return draws;
        };
        if let Some((cards, offset)) = self.card_source(0) {
            let record = DrawRecord::of(DrawKind::PlacedCard, 0);
            draw.draw(
                card,
                Shape::IndirectAt {
                    vertices: &[(inputs.cards, cards), (inputs.records, record.offset())],
                    indices: None,
                    args: inputs.card_args,
                    offset,
                },
            );
            draws += 1;
        }
        draws
    }

    fn draw_casters(
        &self,
        region: Option<u32>,
        draw: &mut Draw<'_, Self, Scene>,
        inputs: &Streams,
        (casting, card): (&[Pipeline<Self, Scene>; 2], &Pipeline<Self, Scene>),
    ) -> u32 {
        if !self.held.prepared {
            return 0;
        }
        let Some(view) = region.and_then(|region| self.map_view(region)) else {
            return 0;
        };
        let mut draws = 0;
        for item in &self.held.cast {
            let Some((ids, offset)) =
                self.draw_source(view, item.group as usize, item.level as usize)
            else {
                continue;
            };
            draw.draw(
                &casting[usize::from(item.cuts)],
                Shape::MeshIndirect {
                    mesh: item.mesh,
                    material: item.material,
                    args: inputs.args,
                    offset,
                    streams: &[(inputs.ids, ids)],
                },
            );
            draws += 1;
        }
        if !self.held.atlas {
            return draws;
        }
        if let Some((cards, offset)) = self.card_source(view) {
            draw.draw(
                card,
                Shape::IndirectAt {
                    vertices: &[(inputs.cards, cards)],
                    indices: None,
                    args: inputs.card_args,
                    offset,
                },
            );
            draws += 1;
        }
        draws
    }

    fn encode_maps(&self, cx: &mut Encode<'_, Self>) {
        let cells = 0..CELLS * CELLS;
        for (map, pipelines) in self.maps.iter().enumerate() {
            let (redraw, average) = self.held.occupancy[map];
            if !redraw && !average {
                continue;
            }
            if redraw {
                cx.dispatch(
                    &pipelines.select,
                    Count::Groups([self.held.workgroups, 1, 1]),
                );
            }
            let layer = map as u32;
            let mut draw = cx.render(Attach::Colors(vec![(self.occupancy_map, layer)]), None);
            if redraw {
                draw.draw(
                    &pipelines.clear,
                    Shape::Pulled {
                        vertices: 0..QUAD,
                        instances: cells.clone(),
                    },
                );
                draw.draw(
                    &pipelines.splat,
                    Shape::Indirect {
                        vertices: &[],
                        indices: None,
                        args: pipelines.listed,
                        offset: 0,
                    },
                );
            }
            if average {
                draw.draw(
                    &pipelines.means,
                    Shape::Pulled {
                        vertices: 0..QUAD,
                        instances: cells.clone(),
                    },
                );
            }
        }
    }
}

impl GpuPart for ScatterGpu {
    type Reads = (
        Res<'static, ScatterState>,
        Res<'static, OccupancyWindows>,
        Res<'static, FarCoverTable>,
        Res<'static, TerrainTiles>,
        Res<'static, GroundTiles>,
        Res<'static, engine_places_world::ground::GroundBaking>,
        Res<'static, AssetResidency>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::scatter::{ambient, cull, faces, snap};
        let snap_uniform = cx.buffer("snap", Lifetime::Kept);
        let frame = cx.buffer("frame", Lifetime::Kept);
        let views = cx.buffer("views", Lifetime::Kept);
        let records = cx.buffer("records", Lifetime::Kept);
        let groups = cx.buffer("groups", Lifetime::Kept);
        let ids = cx.buffer("ids", Lifetime::Kept);
        let draws = cx.buffer("draws", Lifetime::Kept);
        let poses = cx.buffer("poses", Lifetime::Kept);
        let colors = cx.buffer("colors", Lifetime::Kept);
        let grounds = cx.buffer("grounds", Lifetime::Kept);
        let cards = cx.buffer("cards", Lifetime::Kept);
        let card_args = cx.buffer("card_args", Lifetime::Kept);
        let placed_groups = cx.buffer("placed_groups", Lifetime::Kept);
        let debug_records = cx.buffer("debug_records", Lifetime::Kept);
        let covers = cx.buffer("occupancy_covers", Lifetime::Kept);
        let listed = cx.buffer("occupancy_listed", Lifetime::Kept);
        let placed_far = cx.buffer("placed_far", Lifetime::Kept);
        let occupancy_map = cx.texture("occupancy", OCCUPANCY_FORMAT, Lifetime::Kept);
        let unfiner = cx.texture("occupancy_unfiner", OCCUPANCY_FORMAT, Lifetime::Kept);
        let linear = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(faces::placed_tiles::PLACED_RECORDS, Bind::Buffer(records))?;
        cx.frame(faces::placed_tiles::PLACED_POSES, Bind::Buffer(poses))?;
        cx.frame(
            faces::placed_tiles::PLACED_GROUPS,
            Bind::Buffer(placed_groups),
        )?;
        cx.frame(
            faces::placed_tiles::PLACED_FRAMES_SAMPLER,
            Bind::Sampler(linear),
        )?;
        cx.frame(ambient::OCCUPANCY, Bind::Texture(occupancy_map))?;
        cx.frame(ambient::OCCUPANCY_SAMPLER, Bind::Sampler(linear))?;
        cx.frame(faces::far_cover::PLACED_FAR, Bind::Buffer(placed_far))?;
        let scatter = cx.pass("scatter").id();
        let occupancy = cx
            .pass("occupancy")
            .after(scatter)
            .color(occupancy_map, Access::Load)
            .id();
        let snapping = cx.compute_pipeline(
            scatter,
            snap::SNAP_SCATTER,
            &[
                (snap::SNAP, Bind::Buffer(snap_uniform)),
                (snap::RECORDS, Bind::Buffer(records)),
                (snap::POSES, Bind::Buffer(poses)),
                (snap::GROUNDS, Bind::Buffer(grounds)),
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
        let mut splats = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        let mut listed_draws = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        let mut maps = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        for map in 0..MAX_OCCUPANCY_LEVELS {
            let splat = cx.buffer("occupancy_splat", Lifetime::Kept);
            let listed_draw = cx.buffer("occupancy_draw", Lifetime::Kept);
            maps.push(Self::map_pipelines(
                cx,
                map,
                occupancy,
                (splat, listed_draw),
                (records, poses, covers, listed),
                (occupancy_map, unfiner),
            )?);
            splats.push(splat);
            listed_draws.push(listed_draw);
        }
        let joined = Self::scene_pipelines(
            cx,
            (poses, colors, grounds),
            (ids, draws, debug_records, cards, card_args),
        )?;
        Ok(Self {
            scatter,
            occupancy,
            snap_uniform,
            frame,
            views,
            records,
            groups,
            ids,
            draws,
            poses,
            colors,
            grounds,
            cards,
            card_args,
            placed_groups,
            debug_records,
            covers,
            listed,
            splats,
            listed_draws,
            placed_far,
            occupancy_map,
            unfiner,
            snapping,
            culling,
            maps,
            joined,
            records_written: false,
            held: Held::default(),
        })
    }

    fn prepare(
        &mut self,
        (state, windows, far, tiles, ground, baking, residency): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        if state.is_added() {
            self.held = Held::default();
        }
        let words = record_words();
        cx.allocate_buffer(
            self.debug_records,
            std::mem::size_of_val(words.as_slice()) as u64,
        );
        if !self.records_written {
            cx.write(self.debug_records, 0, &words);
            self.records_written = true;
        }
        self.write_slots(cx, &state);
        self.list_draws(&state, &residency);
        self.write_placed_groups(cx, &state);
        if far.is_changed() {
            let words = far.words();
            cx.allocate_buffer(
                self.placed_far,
                std::mem::size_of_val(words.as_slice()) as u64,
            );
            cx.write(self.placed_far, 0, &words);
        }
        self.prepare_snap(cx, &state, (&tiles, &ground, &baking));
        self.prepare_cull(cx, &state);
        if self.held.prepared && self.held.objects > 0 {
            cx.run(self.scatter);
        }
        self.prepare_occupancy(cx, &windows, &state.groups.covers, state.room);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.scatter {
            let groups = [self.held.objects.div_ceil(SCATTER_GROUP), 1, 1];
            if self.held.snapping {
                cx.dispatch(&self.snapping, Count::Groups(groups));
            }
            cx.dispatch(&self.culling, Count::Groups(groups));
            return 0;
        }
        if pass == self.occupancy {
            self.encode_maps(cx);
        }
        0
    }
}
