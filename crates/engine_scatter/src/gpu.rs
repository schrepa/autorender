use bevy_ecs::prelude::*;
use engine_core::{AssetResidency, Error, MaterialHandle, MeshHandle, Var, View};
use engine_places::ground::{GroundTiles, TerrainTiles};
use engine_places::shadows::{CasterMap, CasterView, ShadowCasters};
use engine_render::{
    Access, AfterRead, Attach, Bind, Buffer, ComputePipeline, Count, DepthUse, Draw, DrawInput,
    Encode, GpuPart, Install, Lifetime, MainView, Own, Pass, Pipeline, Prepare, RenderPipeline,
    SamplerDesc, Shape, Texture, VertexLayout, wgpu,
};

use crate::far_cover::FarCoverTable;
use crate::layouts::{
    CELLS, DrawArgs, DrawKind, DrawRecord, LEVELS, ListedDraw, MAX_OCCUPANCY_LEVELS, PlacedGroup,
    QUAD, SCATTER_GROUP, Snap, Splat, placed_group, record_words,
};
use crate::occupancy::{LISTED_RESET, OccupancyWindows};
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

const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

const HEIGHTS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

const LEFT_OUT: u32 = 0;

struct MapPipelines {
    select: ComputePipeline<ScatterGpu>,
    clear: Pipeline<ScatterGpu, Own>,
    splat: Pipeline<ScatterGpu, Own>,
    means: Pipeline<ScatterGpu, Own>,
    listed: DrawInput<ScatterGpu, Own>,
}

struct HeldGroup {
    meshes: Vec<(MeshHandle, bool)>,
    casters: Vec<(MeshHandle, bool)>,
    material: MaterialHandle,
    material_resident: bool,
    count: u32,
    casts: Option<bool>,
    look: usize,
    masked: bool,
}

#[derive(Default)]
struct Snapped {
    generation: u64,
    eye: [f32; 4],
}

#[derive(Default)]
struct Held {
    groups: Vec<HeldGroup>,
    lists: Vec<(u32, u32)>,
    ids_per_view: u32,
    group_count: u32,
    maps: Vec<u32>,
    fading_view: usize,
    objects: u32,
    ids_room: u64,
    cards_room: u64,
    uploaded: Option<u64>,
    groups_revision: Option<u64>,
    impostors_revision: Option<u64>,
    covers_revision: Option<u64>,
    snapped: Option<Snapped>,
    slots_written: bool,
    snapping: bool,
    prepared: bool,
    casts: bool,
    cards: bool,
    atlas: bool,
    occupancy: [(bool, bool); MAX_OCCUPANCY_LEVELS],
    workgroups: u32,
    covers: u64,
    listed: u64,
}

struct Lit {
    lit: Vec<[Pipeline<ScatterGpu, MainView>; 2]>,
    prepass: Vec<[Pipeline<ScatterGpu, MainView>; 2]>,
    card: Pipeline<ScatterGpu, MainView>,
}

struct Inputs {
    ids: DrawInput<ScatterGpu, MainView>,
    args: DrawInput<ScatterGpu, MainView>,
    records: DrawInput<ScatterGpu, MainView>,
    cards: DrawInput<ScatterGpu, MainView>,
    card_args: DrawInput<ScatterGpu, MainView>,
    caster_ids: DrawInput<ScatterGpu, ShadowCasters>,
    caster_args: DrawInput<ScatterGpu, ShadowCasters>,
    caster_cards: DrawInput<ScatterGpu, ShadowCasters>,
    caster_card_args: DrawInput<ScatterGpu, ShadowCasters>,
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
    far_bases: Buffer<Self>,
    far_entries: Buffer<Self>,
    coverage: Texture<Self>,
    heights: Texture<Self>,
    unfiner: [Texture<Self>; 2],
    snapping: ComputePipeline<Self>,
    culling: ComputePipeline<Self>,
    maps: Vec<MapPipelines>,
    lit: Lit,
    casting: [Pipeline<Self, ShadowCasters>; 2],
    shadow_card: Pipeline<Self, ShadowCasters>,
    inputs: Inputs,
    records_written: bool,
    held: Held,
}

fn grow(held: &mut u64, bytes: u64, room: u64) {
    let size = bytes.next_power_of_two().min(room);
    if bytes > *held && size > *held {
        *held = size;
    }
}

fn blend(dst_factor: wgpu::BlendFactor, operation: wgpu::BlendOperation) -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor,
        operation,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
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
        (coverage, heights, unfiner): (Texture<Self>, Texture<Self>, [Texture<Self>; 2]),
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
        let red = wgpu::ColorWrites::RED;
        let red_green = wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN;
        let over = Some(blend(wgpu::BlendFactor::OneMinusSrc, wgpu::BlendOperation::Add));
        let max = Some(blend(wgpu::BlendFactor::One, wgpu::BlendOperation::Max));
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
            &[(None, red), (None, red_green)],
            &[(maps::SPLAT, Bind::Buffer(splat))],
        )?;
        let splat_pipeline = cx.render_pipeline_targets(
            pass,
            &target(maps::VS_SPLAT, maps::FS_SPLAT),
            &[(over, red), (max, red_green)],
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
            &[(None, red), (None, red_green)],
            &[
                (maps::SPLAT, Bind::Buffer(splat)),
                (maps::FINER_COVERAGE, finer(coverage, unfiner[0])),
                (maps::FINER_HEIGHTS, finer(heights, unfiner[1])),
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

    fn lit_pipelines(
        cx: &mut Install<'_, Self>,
        (poses, colors, grounds): (Buffer<Self>, Buffer<Self>, Buffer<Self>),
    ) -> Result<Lit, Error> {
        use crate::shaders::scatter::{cards, draw};
        let vars = (draw::POSES, draw::COLORS, draw::GROUNDS);
        let buffers = (poses, colors, grounds);
        let mut lit = Vec::with_capacity(8);
        let mut prepass = Vec::with_capacity(4);
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
                (false, false) => (draw::FS_OBJECT, draw::FS_DEBUG_OBJECT, draw::FS_MASKED_DEPTH),
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
            let depth = if masked {
                DepthUse::Compare {
                    compare: wgpu::CompareFunction::Equal,
                    write: false,
                    bias: wgpu::DepthBiasState::default(),
                }
            } else {
                DepthUse::Write {
                    bias: wgpu::DepthBiasState::default(),
                }
            };
            let lit_desc = |vertex| RenderPipeline {
                vertex,
                fragment: Some(fragment),
                debug: Some(debug),
                vertices,
                primitive: primitive(double_sided),
                depth,
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            };
            lit.push([
                cx.render_pipeline(MainView, &lit_desc(vertex), &objects(vars, buffers))?,
                cx.render_pipeline(MainView, &lit_desc(fading), &objects(vars, buffers))?,
            ]);
            if !masked {
                continue;
            }
            let depth_desc = |vertex| RenderPipeline {
                vertex,
                fragment: Some(masked_depth),
                debug: Some(masked_depth),
                vertices,
                primitive: primitive(double_sided),
                depth: DepthUse::Write {
                    bias: wgpu::DepthBiasState::default(),
                },
                blend: None,
                writes: wgpu::ColorWrites::empty(),
            };
            prepass.push([
                cx.render_pipeline(MainView, &depth_desc(vertex), &objects(vars, buffers))?,
                cx.render_pipeline(MainView, &depth_desc(fading), &objects(vars, buffers))?,
            ]);
        }
        let card = cx.render_pipeline(
            MainView,
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
        )?;
        Ok(Lit { lit, prepass, card })
    }

    fn set_groups(&mut self, state: &ScatterState) {
        let held = &mut self.held;
        let data = &state.group_data;
        held.lists.clear();
        held.lists
            .extend(data.lists.iter().map(|lists| (lists.ids, lists.count)));
        held.group_count = state.groups.layouts.len() as u32;
        held.ids_per_view = data.ids_per_view;
        held.groups.clear();
        held.groups.extend(state.groups.groups.iter().map(|group| HeldGroup {
            meshes: group.meshes.iter().map(|&mesh| (mesh, false)).collect(),
            casters: group.casters.iter().map(|&mesh| (mesh, false)).collect(),
            material: group.material,
            material_resident: false,
            count: group.count,
            casts: group.casts,
            look: look(
                group.masked(),
                group.source.is_double_sided(),
                group.extended,
            ),
            masked: group.masked(),
        }));
    }

    fn write_slots(&mut self, cx: &mut Prepare<'_, Self>, state: &ScatterState) {
        let held = &mut self.held;
        let buffers = [self.poses, self.colors, self.records, self.grounds];
        let strides = [
            size_of::<crate::layouts::PoseData>(),
            size_of::<crate::layouts::SlotColor>(),
            size_of::<crate::layouts::PlacedRecord>(),
            size_of::<crate::layouts::GroundData>(),
        ];
        for ((buffer, bytes), stride) in buffers.into_iter().zip(state.slot_bytes).zip(strides) {
            cx.allocate_buffer(buffer, bytes.max(stride as u64));
        }
        let pool = &state.pool;
        let objects = state.objects;
        let range = match held.uploaded {
            Some(seen) if seen == state.slots_revision => None,
            Some(seen) if seen + 1 == state.slots_revision => state.dirty.clone(),
            Some(_) | None => Some(0..objects),
        }
        .filter(|range| !range.is_empty());
        if let Some(range) = range {
            held.slots_written = true;
            let slots = range.start as usize..range.end as usize;
            cx.write(
                self.poses,
                u64::from(range.start) * size_of::<crate::layouts::PoseData>() as u64,
                &pool.poses[slots.clone()],
            );
            cx.write(
                self.colors,
                u64::from(range.start) * size_of::<crate::layouts::SlotColor>() as u64,
                &pool.colors[slots.clone()],
            );
            cx.write(
                self.records,
                u64::from(range.start) * size_of::<crate::layouts::PlacedRecord>() as u64,
                &pool.records[slots.clone()],
            );
            cx.write(
                self.grounds,
                u64::from(range.start) * size_of::<crate::layouts::GroundData>() as u64,
                &pool.grounds[slots],
            );
        }
        held.uploaded = Some(state.slots_revision);
        held.objects = objects;
    }

    fn write_placed_groups(&mut self, cx: &mut Prepare<'_, Self>, state: &ScatterState) {
        if self.held.impostors_revision == Some(state.impostors_revision) {
            return;
        }
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
        self.held.impostors_revision = Some(state.impostors_revision);
    }

    fn prepare_snap(
        &mut self,
        cx: &mut Prepare<'_, Self>,
        (state, view): (&ScatterState, &View),
        (tiles, ground): (&Res<'_, TerrainTiles>, &Res<'_, GroundTiles>),
    ) {
        let held = &mut self.held;
        held.snapping = false;
        let terrains = tiles.terrains.len() as u32;
        cx.allocate_buffer(self.snap_uniform, size_of::<Snap>() as u64);
        if ground.layers == 0 || terrains == 0 || state.resident == 0 {
            return;
        }
        let eye = view.detail_eye().extend(0.0).to_array();
        let snap = Snap {
            counts: [held.objects, terrains, 0, 0],
            eye,
        };
        cx.write(self.snap_uniform, 0, &[snap]);
        let unchanged = !tiles.is_changed()
            && !ground.is_changed()
            && held
                .snapped
                .as_ref()
                .is_some_and(|snapped| snapped.generation == state.generation);
        if unchanged && !held.slots_written {
            match &mut held.snapped {
                Some(snapped) if snapped.eye != eye => snapped.eye = eye,
                Some(_) | None => return,
            }
        } else {
            held.snapped = Some(Snapped {
                generation: state.generation,
                eye,
            });
            held.slots_written = false;
        }
        held.snapping = true;
    }

    fn prepare_cull(&mut self, cx: &mut Prepare<'_, Self>, state: &ScatterState) {
        let held = &mut self.held;
        let plan = &state.plan;
        held.maps.clear();
        held.prepared = false;
        held.casts = plan.casts;
        held.cards = plan.cards;
        held.atlas = plan.atlas;
        let sizes = state.sizes;
        cx.allocate_buffer(
            self.groups,
            sizes.groups.max(size_of::<crate::layouts::CullGroup>() as u64),
        );
        cx.allocate_buffer(self.views, sizes.views);
        cx.allocate_buffer(self.ids, sizes.ids);
        cx.allocate_buffer(self.draws, sizes.draws);
        cx.allocate_buffer(self.cards, sizes.cards);
        cx.allocate_buffer(self.card_args, sizes.card_args);
        cx.allocate_buffer(self.frame, size_of::<crate::layouts::CullFrame>() as u64);
        held.ids_room = sizes.ids;
        held.cards_room = sizes.cards;
        if held.groups_revision != Some(state.groups_revision) {
            if !state.group_data.data.is_empty() {
                cx.write(self.groups, 0, &state.group_data.data);
            }
            held.groups_revision = Some(state.groups_revision);
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
        if plan.lists_left {
            let at = plan.fading_view as u64 * 16;
            cx.read_back::<ScatterFeedback>(LEFT_OUT, self.card_args, at..at + 16, AfterRead::Keep);
        }
        held.prepared = true;
    }

    fn prepare_occupancy(
        &mut self,
        cx: &mut Prepare<'_, Self>,
        windows: &OccupancyWindows,
        room: u64,
    ) {
        let held = &mut self.held;
        let (texels, layers) = if windows.runs && windows.maps > 0 {
            (windows.resolution.max(1), windows.maps as u32)
        } else {
            (1, 1)
        };
        cx.allocate(self.coverage, layer(texels, layers), 1);
        cx.allocate(self.heights, layer(texels, layers), 1);
        for unfiner in self.unfiner {
            cx.allocate(unfiner, layer(1, 1), 1);
        }
        let cover = size_of::<crate::layouts::Cover>() as u64;
        let word = size_of::<u32>() as u64;
        if !windows.runs {
            held.covers = cover;
            held.listed = word;
        }
        held.covers = held.covers.max(cover);
        held.listed = held.listed.max(word);
        if held.covers_revision != Some(windows.covers_revision) {
            let bytes = std::mem::size_of_val(windows.covers.as_slice()) as u64;
            grow(&mut held.covers, bytes, room);
            cx.allocate_buffer(self.covers, held.covers);
            if !windows.covers.is_empty() {
                cx.write(self.covers, 0, &windows.covers);
            }
            held.covers_revision = Some(windows.covers_revision);
        }
        if windows.runs && windows.slots > 0 && !windows.covers.is_empty() {
            grow(&mut held.listed, 2 * u64::from(windows.slots) * word, room);
        }
        cx.allocate_buffer(self.covers, held.covers);
        cx.allocate_buffer(self.listed, held.listed);
        held.workgroups = windows.workgroups;
        held.occupancy = [(false, false); MAX_OCCUPANCY_LEVELS];
        for (&splat, &listed) in self.splats.iter().zip(&self.listed_draws) {
            cx.allocate_buffer(splat, size_of::<Splat>() as u64);
            cx.allocate_buffer(listed, size_of::<ListedDraw>() as u64);
        }
        if !windows.drawn() {
            return;
        }
        for (map, frame) in windows.frames.iter().enumerate() {
            if !frame.redraw && !frame.average {
                continue;
            }
            cx.write(self.splats[map], 0, &[frame.splat]);
            if frame.redraw {
                cx.write(self.listed_draws[map], 0, &[LISTED_RESET]);
            }
            held.occupancy[map] = (frame.redraw, frame.average);
        }
        cx.run(self.occupancy);
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

    fn draw_objects(&self, draw: &mut Draw<'_, Self, MainView>, prepass: bool) -> u32 {
        let held = &self.held;
        if !held.prepared {
            return 0;
        }
        let inputs = &self.inputs;
        let mut draws = 0;
        for (view, fades) in [(0, false), (held.fading_view, true)] {
            for (index, group) in held.groups.iter().enumerate() {
                if group.count == 0 {
                    continue;
                }
                let pipeline = if prepass {
                    if !group.masked {
                        continue;
                    }
                    &self.lit.prepass[group.look & 3][usize::from(fades)]
                } else {
                    &self.lit.lit[group.look][usize::from(fades)]
                };
                let casts = held.casts && group.casts.is_some();
                for (level, &(mesh, resident)) in group.meshes.iter().enumerate() {
                    if !resident {
                        continue;
                    }
                    let Some((ids, offset)) = self.draw_source(view, index, level) else {
                        continue;
                    };
                    let record = DrawRecord::of(DrawKind::Placed, level, casts);
                    draw.draw(
                        pipeline,
                        Shape::MeshIndirect {
                            mesh,
                            material: Some(group.material),
                            args: inputs.args,
                            offset,
                            streams: &[(inputs.ids, ids), (inputs.records, record.offset())],
                        },
                    );
                    draws += 1;
                }
            }
        }
        if prepass || !held.cards {
            return draws;
        }
        if let Some((cards, offset)) = self.card_source(0) {
            let record = DrawRecord::of(DrawKind::PlacedCard, 0, held.casts);
            draw.draw(
                &self.lit.card,
                Shape::IndirectStreams {
                    streams: &[(inputs.cards, cards), (inputs.records, record.offset())],
                    args: inputs.card_args,
                    offset,
                },
            );
            draws += 1;
        }
        draws
    }

    pub(crate) fn draw_occluders(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        self.draw_objects(draw, true)
    }

    pub(crate) fn draw_late(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        self.draw_objects(draw, false)
    }

    pub(crate) fn draw_casters(
        &self,
        at: &CasterView,
        draw: &mut Draw<'_, Self, ShadowCasters>,
    ) -> u32 {
        if !self.held.prepared || at.map != CasterMap::Region {
            return 0;
        }
        let Some(view) = self.map_view(at.region) else {
            return 0;
        };
        let inputs = &self.inputs;
        let mut draws = 0;
        for (index, group) in self.held.groups.iter().enumerate() {
            let Some(masked) = group.casts.filter(|_| group.count > 0) else {
                continue;
            };
            if masked && !group.material_resident {
                continue;
            }
            let pipeline = &self.casting[usize::from(masked)];
            let material = masked.then_some(group.material);
            for (level, &(mesh, resident)) in group.casters.iter().enumerate() {
                if !resident {
                    continue;
                }
                let Some((ids, offset)) = self.draw_source(view, index, level) else {
                    continue;
                };
                draw.draw(
                    pipeline,
                    Shape::MeshIndirect {
                        mesh,
                        material,
                        args: inputs.caster_args,
                        offset,
                        streams: &[(inputs.caster_ids, ids)],
                    },
                );
                draws += 1;
            }
        }
        if !self.held.atlas {
            return draws;
        }
        if let Some((cards, offset)) = self.card_source(view) {
            draw.draw(
                &self.shadow_card,
                Shape::IndirectStreams {
                    streams: &[(inputs.caster_cards, cards)],
                    args: inputs.caster_card_args,
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
            let mut draw = cx.render(
                Attach::Colors(vec![(self.coverage, layer), (self.heights, layer)]),
                None,
            );
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
        Res<'static, View>,
        Res<'static, TerrainTiles>,
        Res<'static, GroundTiles>,
        Res<'static, AssetResidency>,
    );

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::scatter::{casters, cull, faces, snap};
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
        let far_bases = cx.buffer("placed_far_bases", Lifetime::Kept);
        let far_entries = cx.buffer("placed_far", Lifetime::Kept);
        let coverage = cx.texture("occupancy_coverage", COVERAGE_FORMAT, Lifetime::Kept);
        let heights = cx.texture("occupancy_heights", HEIGHTS_FORMAT, Lifetime::Kept);
        let unfiner = [
            cx.texture("occupancy_unfiner_coverage", COVERAGE_FORMAT, Lifetime::Kept),
            cx.texture("occupancy_unfiner_heights", HEIGHTS_FORMAT, Lifetime::Kept),
        ];
        let linear = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(faces::placed_tiles::PLACED_RECORDS, Bind::Buffer(records))?;
        cx.frame(faces::placed_tiles::PLACED_POSES, Bind::Buffer(poses))?;
        cx.frame(faces::placed_tiles::PLACED_GROUPS, Bind::Buffer(placed_groups))?;
        cx.frame(faces::placed_tiles::PLACED_FRAMES_SAMPLER, Bind::Sampler(linear))?;
        cx.frame(faces::occupancy::OCCUPANCY_COVERAGE, Bind::Texture(coverage))?;
        cx.frame(faces::occupancy::OCCUPANCY_HEIGHTS, Bind::Texture(heights))?;
        cx.frame(faces::occupancy::OCCUPANCY_SAMPLER, Bind::Sampler(linear))?;
        cx.frame(faces::far_cover::PLACED_FAR_BASES, Bind::Buffer(far_bases))?;
        cx.frame(faces::far_cover::PLACED_FAR, Bind::Buffer(far_entries))?;
        let scatter = cx.pass("scatter").id();
        let occupancy = cx
            .pass("occupancy")
            .after(scatter)
            .color(coverage, Access::Load)
            .color(heights, Access::Load)
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
                (coverage, heights, unfiner),
            )?);
            splats.push(splat);
            listed_draws.push(listed_draw);
        }
        let lit = Self::lit_pipelines(cx, (poses, colors, grounds))?;
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
                ShadowCasters,
                &caster_desc(casters::VS_CASTER, None),
                &caster_bound(),
            )?,
            cx.render_pipeline(
                ShadowCasters,
                &caster_desc(casters::VS_CASTER_MASKED, Some(casters::FS_CASTER_MASKED)),
                &caster_bound(),
            )?,
        ];
        let shadow_card = cx.render_pipeline(
            ShadowCasters,
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
        let inputs = Inputs {
            ids: cx.draw_input(MainView, ids),
            args: cx.draw_input(MainView, draws),
            records: cx.draw_input(MainView, debug_records),
            cards: cx.draw_input(MainView, cards),
            card_args: cx.draw_input(MainView, card_args),
            caster_ids: cx.draw_input(ShadowCasters, ids),
            caster_args: cx.draw_input(ShadowCasters, draws),
            caster_cards: cx.draw_input(ShadowCasters, cards),
            caster_card_args: cx.draw_input(ShadowCasters, card_args),
        };
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
            far_bases,
            far_entries,
            coverage,
            heights,
            unfiner,
            snapping,
            culling,
            maps,
            lit,
            casting,
            shadow_card,
            inputs,
            records_written: false,
            held: Held::default(),
        })
    }

    fn prepare(
        &mut self,
        (state, windows, far, view, tiles, ground, residency): SystemParamItem<
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
        if self.held.groups_revision != Some(state.groups_revision) {
            self.set_groups(&state);
        }
        for group in &mut self.held.groups {
            for (mesh, resident) in group.meshes.iter_mut().chain(&mut group.casters) {
                *resident = residency.resident(*mesh);
            }
            group.material_resident = residency.material_resident(group.material);
        }
        self.write_placed_groups(cx, &state);
        if far.is_changed() {
            let bases = std::mem::size_of_val(far.bases.as_slice()) as u64;
            let entries = std::mem::size_of_val(far.entries.as_slice()) as u64;
            cx.allocate_buffer(self.far_bases, bases.max(16));
            cx.allocate_buffer(self.far_entries, entries.max(32));
            if !far.bases.is_empty() {
                cx.write(self.far_bases, 0, &far.bases);
            }
            if !far.entries.is_empty() {
                cx.write(self.far_entries, 0, &far.entries);
            }
        }
        self.prepare_snap(cx, (&*state, &*view), (&tiles, &ground));
        self.prepare_cull(cx, &state);
        if self.held.prepared && self.held.objects > 0 {
            cx.run(self.scatter);
        }
        self.prepare_occupancy(cx, &windows, state.room);
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
