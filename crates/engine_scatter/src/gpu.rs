use bevy_ecs::prelude::*;
use engine_core::{AssetResidency, Detail, Error, MaterialHandle, MeshHandle, Var, View};
use engine_places::ground::{GroundTiles, TerrainTiles};
use engine_places::shadows::{CasterMap, CasterView, Refresh, ShadowCasters, ShadowMaps};
use engine_render::{
    Access, Attach, Bind, Buffer, ComputePipeline, Count, DepthUse, Draw, DrawInput, Encode,
    GpuPart, Install, Lifetime, MainView, Own, Pass, Pipeline, Prepare, RenderPipeline,
    SamplerDesc, Shape, Texture, VertexLayout, wgpu,
};

use crate::far_cover::FarCoverTable;
use crate::layouts::{
    CELLS, CullFrame, CullGroup, CullView, DrawArgs, LEVELS, MAX_OCCUPANCY_LEVELS, QUAD,
    SCATTER_GROUP, Snap, draw_args, planes,
};
use crate::occupancy::{LISTED_RESET, OccupancyWindows};
use crate::systems::ScatterState;

const FADE_MARGIN: f32 = 0.1;

const CARD_VERTICES: u32 = 6;

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

const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

const HEIGHTS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

const SELECTS: [&str; MAX_OCCUPANCY_LEVELS] = [
    "occupancy_select_0",
    "occupancy_select_1",
    "occupancy_select_2",
    "occupancy_select_3",
    "occupancy_select_4",
    "occupancy_select_5",
    "occupancy_select_6",
    "occupancy_select_7",
];

const MAPS: [&str; MAX_OCCUPANCY_LEVELS] = [
    "occupancy_0",
    "occupancy_1",
    "occupancy_2",
    "occupancy_3",
    "occupancy_4",
    "occupancy_5",
    "occupancy_6",
    "occupancy_7",
];

struct MapPipelines {
    clear: [Pipeline<ScatterGpu, Own>; 2],
    splat: [Pipeline<ScatterGpu, Own>; 2],
    means: [Pipeline<ScatterGpu, Own>; 2],
}

struct HeldGroup {
    meshes: Vec<(MeshHandle, bool)>,
    casters: Vec<(MeshHandle, bool)>,
    material: MaterialHandle,
    count: u32,
    casts: Option<bool>,
    masked: bool,
}

#[derive(Clone, Copy, Default)]
struct GroupLists {
    ids: u32,
    count: u32,
}

#[derive(Default)]
struct Sizes {
    groups: u64,
    views: u64,
    ids: u64,
    draws: u64,
    cards: u64,
    card_args: u64,
    listed: u64,
}

#[derive(Default)]
struct Held {
    groups: Vec<HeldGroup>,
    lists: Vec<GroupLists>,
    ids_per_view: u32,
    fading_view: usize,
    group_count: u32,
    template: Vec<DrawArgs>,
    casters: Vec<DrawArgs>,
    args: Vec<DrawArgs>,
    views: Vec<CullView>,
    maps: Vec<u32>,
    card_args: Vec<[u32; 4]>,
    objects: u32,
    resident: u32,
    sizes: Sizes,
    uploaded: Option<u64>,
    groups_revision: Option<u64>,
    covers_revision: Option<u64>,
    slots_written: bool,
    prepared: bool,
    occupancy: [(bool, bool); MAX_OCCUPANCY_LEVELS],
    workgroups: u32,
}

pub(crate) struct ScatterGpu {
    snap: Pass<Self>,
    cull: Pass<Self>,
    selects: Vec<Pass<Self>>,
    maps: Vec<Pass<Self>>,
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
    selecting: Vec<ComputePipeline<Self>>,
    occupancy: Vec<MapPipelines>,
    listed_inputs: Vec<DrawInput<Self, Own>>,
    lit: [Pipeline<Self, MainView>; 2],
    prepass: [[Pipeline<Self, MainView>; 2]; 2],
    casting: [Pipeline<Self, ShadowCasters>; 2],
    main_ids: DrawInput<Self, MainView>,
    main_args: DrawInput<Self, MainView>,
    caster_ids: DrawInput<Self, ShadowCasters>,
    caster_args: DrawInput<Self, ShadowCasters>,
    held: Held,
}

fn grow(held: &mut u64, bytes: u64) {
    let size = bytes.next_power_of_two();
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

fn objects(
    (poses, colors, grounds): (Var, Var, Var),
    (pose_buffer, color_buffer, ground_buffer): (Buffer<ScatterGpu>, Buffer<ScatterGpu>, Buffer<ScatterGpu>),
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
        (drawn, splat): (Pass<Self>, Buffer<Self>),
        (records, poses, covers, listed): (Buffer<Self>, Buffer<Self>, Buffer<Self>, Buffer<Self>),
        (coverage, heights, unfiner): (Texture<Self>, Texture<Self>, [Texture<Self>; 2]),
    ) -> Result<MapPipelines, Error> {
        use crate::shaders::scatter::occupancy_maps as maps;
        let target = |vertex, fragment, blend, writes| RenderPipeline {
            vertex,
            fragment: Some(fragment),
            debug: None,
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
            blend,
            writes,
        };
        let finer = |texture: Texture<Self>, held: Texture<Self>| match map.checked_sub(1) {
            Some(finer) => Bind::Slice {
                texture,
                mips: 0..1,
                layers: finer as u32..map as u32,
            },
            None => Bind::Texture(held),
        };
        let splatted = || {
            [
                (maps::SPLAT, Bind::Buffer(splat)),
                (maps::RECORDS, Bind::Buffer(records)),
                (maps::POSES, Bind::Buffer(poses)),
                (maps::COVERS, Bind::Buffer(covers)),
                (maps::LISTED, Bind::Buffer(listed)),
            ]
        };
        let red = wgpu::ColorWrites::RED;
        let red_green = wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN;
        let over = Some(blend(wgpu::BlendFactor::OneMinusSrc, wgpu::BlendOperation::Add));
        let max = Some(blend(wgpu::BlendFactor::One, wgpu::BlendOperation::Max));
        Ok(MapPipelines {
            clear: [
                cx.render_pipeline(
                    drawn,
                    &target(maps::VS_CLEAR, maps::FS_CLEAR_COVERAGE, None, red),
                    &[(maps::SPLAT, Bind::Buffer(splat))],
                )?,
                cx.render_pipeline(
                    drawn,
                    &target(maps::VS_CLEAR, maps::FS_CLEAR_HEIGHTS, None, red_green),
                    &[(maps::SPLAT, Bind::Buffer(splat))],
                )?,
            ],
            splat: [
                cx.render_pipeline(
                    drawn,
                    &target(maps::VS_SPLAT, maps::FS_SPLAT_COVERAGE, over, red),
                    &splatted(),
                )?,
                cx.render_pipeline(
                    drawn,
                    &target(maps::VS_SPLAT, maps::FS_SPLAT_HEIGHTS, max, red_green),
                    &splatted(),
                )?,
            ],
            means: [
                cx.render_pipeline(
                    drawn,
                    &target(maps::VS_MEANS, maps::FS_MEANS_COVERAGE, None, red),
                    &[
                        (maps::SPLAT, Bind::Buffer(splat)),
                        (maps::FINER_COVERAGE, finer(coverage, unfiner[0])),
                    ],
                )?,
                cx.render_pipeline(
                    drawn,
                    &target(maps::VS_MEANS, maps::FS_MEANS_HEIGHTS, None, red_green),
                    &[
                        (maps::SPLAT, Bind::Buffer(splat)),
                        (maps::FINER_HEIGHTS, finer(heights, unfiner[1])),
                    ],
                )?,
            ],
        })
    }

    fn set_groups(&mut self, cx: &mut Prepare<'_, Self>, state: &ScatterState) {
        let held = &mut self.held;
        let layouts = &state.groups.layouts;
        let mut data = Vec::with_capacity(layouts.len());
        held.template.clear();
        held.casters.clear();
        held.lists.clear();
        let mut ids = 0;
        for (index, (layout, group)) in layouts.iter().zip(&state.groups.groups).enumerate() {
            let count = group.count;
            data.push(CullGroup {
                ids,
                count,
                levels: layout.levels.len() as u32,
                camera: layout.camera,
                errors: layout.errors,
                impostor: 0.0,
                record: index as u32,
                casters: layout.casters.len() as u32,
                parts: layout.parts,
            });
            held.lists.push(GroupLists { ids, count });
            for level in 0..LEVELS as usize {
                held.template.push(draw_args(
                    layout.levels.get(level).copied().unwrap_or((0, 0, 0)),
                ));
                held.casters.push(draw_args(
                    layout.casters.get(level).copied().unwrap_or((0, 0, 0)),
                ));
            }
            ids += count * LEVELS;
        }
        grow(&mut held.sizes.groups, (data.len() * size_of::<CullGroup>()) as u64);
        cx.allocate_buffer(self.groups, held.sizes.groups.max(32));
        if !data.is_empty() {
            cx.write(self.groups, 0, &data);
        }
        held.group_count = layouts.len() as u32;
        held.ids_per_view = ids;
        held.resident = state.resident;
        held.groups.clear();
        held.groups.extend(state.groups.groups.iter().map(|group| HeldGroup {
            meshes: group.meshes.iter().map(|&mesh| (mesh, false)).collect(),
            casters: group.casters.iter().map(|&mesh| (mesh, false)).collect(),
            material: group.material,
            count: group.count,
            casts: group.casts,
            masked: group.masked(),
        }));
    }

    fn write_slots(&mut self, cx: &mut Prepare<'_, Self>, state: &ScatterState) {
        let held = &mut self.held;
        let buffers = [self.poses, self.colors, self.records, self.grounds];
        for (buffer, bytes) in buffers.into_iter().zip(state.slot_bytes) {
            cx.allocate_buffer(buffer, bytes);
        }
        let pool = &state.pool;
        let records = pool.records.len() as u32;
        let range = match held.uploaded {
            Some(seen) if seen == state.slots_revision => None,
            Some(seen) if seen + 1 == state.slots_revision => state.dirty.clone(),
            Some(_) | None => Some(0..records),
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
        held.objects = records;
    }

    fn prepare_cull(
        &mut self,
        cx: &mut Prepare<'_, Self>,
        (view, detail, maps): (&View, &Detail, &ShadowMaps),
        fade: f32,
    ) {
        let held = &mut self.held;
        held.maps.clear();
        held.prepared = false;
        if held.resident == 0 {
            return;
        }
        let view_projection = view.view_projection();
        let eye = view.detail_eye();
        let height = view.render_size().y as f32;
        let pixel_error = detail.enabled.then_some(detail.pixel_error);
        let impostors: Option<f32> = None;
        let y_row = view_projection.row(1).truncate().length();
        let spanned = y_row * height * 0.5;
        let pixels = pixel_error.map_or(0.0, |error| spanned / error);
        let cards = impostors.map_or(0.0, |size| y_row * height / size);
        held.views.clear();
        held.views.push(CullView {
            planes: planes(view_projection),
            limits: [0.0; 4],
        });
        for (index, region) in maps.regions.iter().enumerate() {
            if region.map != CasterMap::Region {
                continue;
            }
            let (region_planes, texel) = match region.refresh {
                Refresh::Kept(kept) => (kept.planes, kept.texel),
                Refresh::Redraw => (planes(region.view), 0.0),
            };
            held.views.push(CullView {
                planes: region_planes,
                limits: [0.5 * texel, texel, 0.0, 0.0],
            });
            held.maps.push(index as u32);
        }
        let view_count = held.views.len() as u32;
        let draws_per_view = held.group_count * LEVELS;
        held.fading_view = held.views.len();
        let lists = view_count + 1;
        let list_count = u64::from(lists);
        grow(
            &mut held.sizes.views,
            u64::from(view_count) * size_of::<CullView>() as u64,
        );
        grow(
            &mut held.sizes.ids,
            u64::from(held.ids_per_view) * list_count * 4,
        );
        grow(
            &mut held.sizes.draws,
            u64::from(draws_per_view) * list_count * size_of::<DrawArgs>() as u64,
        );
        grow(
            &mut held.sizes.cards,
            u64::from(held.objects) * u64::from(view_count) * 8,
        );
        grow(&mut held.sizes.card_args, list_count * 16);
        cx.allocate_buffer(self.views, held.sizes.views.max(size_of::<CullView>() as u64));
        cx.allocate_buffer(self.ids, held.sizes.ids.max(16));
        cx.allocate_buffer(self.draws, held.sizes.draws.max(20));
        cx.allocate_buffer(self.cards, held.sizes.cards.max(8));
        cx.allocate_buffer(self.card_args, held.sizes.card_args.max(16));
        cx.allocate_buffer(self.frame, size_of::<CullFrame>() as u64);
        let frame = CullFrame {
            eye: eye.extend(pixels).to_array(),
            counts: [held.objects, held.group_count, LEVELS, view_count],
            strides: [
                held.ids_per_view,
                draws_per_view,
                u32::from(impostors.is_some()),
                held.objects,
            ],
            fade: [fade, FADE_MARGIN, cards, spanned],
        };
        cx.write(self.frame, 0, &[frame]);
        held.card_args.clear();
        held.card_args
            .resize(view_count as usize, [CARD_VERTICES, 0, 0, 0]);
        held.card_args.push([0; 4]);
        cx.write(self.card_args, 0, &held.card_args);
        cx.write(self.views, 0, &held.views);
        held.args.clear();
        for list in 0..lists {
            let shadow = list > 0 && list < view_count;
            held.args.extend_from_slice(if shadow {
                &held.casters
            } else {
                &held.template
            });
        }
        if !held.args.is_empty() {
            cx.write(self.draws, 0, &held.args);
        }
        held.prepared = true;
    }

    fn prepare_occupancy(&mut self, cx: &mut Prepare<'_, Self>, windows: &OccupancyWindows) {
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
        if held.covers_revision != Some(windows.covers_revision) {
            let bytes = std::mem::size_of_val(windows.covers.as_slice()) as u64;
            cx.allocate_buffer(self.covers, bytes.max(16));
            if !windows.covers.is_empty() {
                cx.write(self.covers, 0, &windows.covers);
            }
            held.covers_revision = Some(windows.covers_revision);
        }
        grow(
            &mut held.sizes.listed,
            2 * u64::from(windows.slots) * size_of::<u32>() as u64,
        );
        cx.allocate_buffer(self.listed, held.sizes.listed.max(4));
        held.workgroups = windows.workgroups;
        held.occupancy = [(false, false); MAX_OCCUPANCY_LEVELS];
        if !windows.drawn() {
            return;
        }
        for (map, frame) in windows.frames.iter().enumerate() {
            if !frame.redraw && !frame.average {
                continue;
            }
            cx.allocate_buffer(self.splats[map], size_of::<crate::layouts::Splat>() as u64);
            cx.write(self.splats[map], 0, &[frame.splat]);
            if frame.redraw {
                cx.allocate_buffer(
                    self.listed_draws[map],
                    size_of::<crate::layouts::ListedDraw>() as u64,
                );
                cx.write(self.listed_draws[map], 0, &[LISTED_RESET]);
                cx.run(self.selects[map]);
            }
            cx.run(self.maps[map]);
            held.occupancy[map] = (frame.redraw, frame.average);
        }
    }

    fn draw_source(&self, view: usize, group: usize, level: usize) -> Option<(u64, u64)> {
        let held = &self.held;
        let lists = held.lists.get(group)?;
        let start = (view as u64 * u64::from(held.ids_per_view)
            + u64::from(lists.ids + level as u32 * lists.count))
            * 4;
        let draw = (view * held.group_count as usize + group) * LEVELS as usize + level;
        (start < held.sizes.ids.max(16))
            .then(|| (start, (draw * size_of::<DrawArgs>()) as u64))
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
        let mut draws = 0;
        for (view, fades) in [(0, false), (held.fading_view, true)] {
            for (index, group) in held.groups.iter().enumerate() {
                if group.count == 0 {
                    continue;
                }
                let pipeline = if prepass {
                    &self.prepass[usize::from(group.masked)][usize::from(fades)]
                } else {
                    &self.lit[usize::from(fades)]
                };
                for (level, &(mesh, resident)) in group.meshes.iter().enumerate() {
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
                            material: Some(group.material),
                            args: self.main_args,
                            offset,
                            instances: Some((self.main_ids, ids)),
                        },
                    );
                    draws += 1;
                }
            }
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
        if !self.held.prepared {
            return 0;
        }
        let Some(view) = self.map_view(at.region) else {
            return 0;
        };
        let mut draws = 0;
        for (index, group) in self.held.groups.iter().enumerate() {
            let Some(masked) = group.casts.filter(|_| group.count > 0) else {
                continue;
            };
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
                        args: self.caster_args,
                        offset,
                        instances: Some((self.caster_ids, ids)),
                    },
                );
                draws += 1;
            }
        }
        draws
    }

    fn encode_map(&self, map: usize, cx: &mut Encode<'_, Self>) -> u32 {
        let (redraw, average) = self.held.occupancy[map];
        let pipelines = &self.occupancy[map];
        let cells = 0..CELLS * CELLS;
        let mut drawn = 0;
        for (index, target) in [self.coverage, self.heights].into_iter().enumerate() {
            let mut draw = cx.render(Attach::Color(target, Some(map as u32)), None);
            if redraw {
                draw.draw(
                    &pipelines.clear[index],
                    Shape::Pulled {
                        vertices: 0..QUAD,
                        instances: cells.clone(),
                    },
                );
                draw.draw(
                    &pipelines.splat[index],
                    Shape::Indirect {
                        vertices: &[],
                        indices: None,
                        args: self.listed_inputs[map],
                        offset: 0,
                    },
                );
                drawn += 2;
            }
            if average {
                draw.draw(
                    &pipelines.means[index],
                    Shape::Pulled {
                        vertices: 0..QUAD,
                        instances: cells.clone(),
                    },
                );
                drawn += 1;
            }
        }
        drawn
    }
}

impl GpuPart for ScatterGpu {
    type Reads = (
        Res<'static, ScatterState>,
        Res<'static, OccupancyWindows>,
        Res<'static, FarCoverTable>,
        Res<'static, View>,
        Res<'static, Detail>,
        Res<'static, ShadowMaps>,
        Res<'static, TerrainTiles>,
        Res<'static, GroundTiles>,
        Res<'static, AssetResidency>,
    );

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::scatter::{casters, cull, draw, faces, occupancy_maps, snap};
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
        cx.frame(faces::occupancy::OCCUPANCY_COVERAGE, Bind::Texture(coverage))?;
        cx.frame(faces::occupancy::OCCUPANCY_HEIGHTS, Bind::Texture(heights))?;
        cx.frame(faces::occupancy::OCCUPANCY_SAMPLER, Bind::Sampler(linear))?;
        cx.frame(faces::far_cover::PLACED_FAR_BASES, Bind::Buffer(far_bases))?;
        cx.frame(faces::far_cover::PLACED_FAR, Bind::Buffer(far_entries))?;
        let snap_pass = cx.pass("snap").id();
        let cull_pass = cx.pass("cull").after(snap_pass).id();
        let snapping = cx.compute_pipeline(
            snap_pass,
            snap::SNAP_SCATTER,
            &[
                (snap::SNAP, Bind::Buffer(snap_uniform)),
                (snap::RECORDS, Bind::Buffer(records)),
                (snap::POSES, Bind::Buffer(poses)),
                (snap::GROUNDS, Bind::Buffer(grounds)),
            ],
        )?;
        let culling = cx.compute_pipeline(
            cull_pass,
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
        let mut selects = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        let mut maps = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        let mut splats = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        let mut listed_draws = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        let mut selecting = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        let mut occupancy = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        let mut listed_inputs = Vec::with_capacity(MAX_OCCUPANCY_LEVELS);
        let mut previous = cull_pass;
        for map in 0..MAX_OCCUPANCY_LEVELS {
            let splat = cx.buffer("occupancy_splat", Lifetime::Kept);
            let listed_draw = cx.buffer("occupancy_draw", Lifetime::Kept);
            let select = cx.pass(SELECTS[map]).after(previous).id();
            let drawn = cx
                .pass(MAPS[map])
                .after(select)
                .color(coverage, Access::Load)
                .color(heights, Access::Load)
                .id();
            previous = drawn;
            selecting.push(cx.compute_pipeline(
                select,
                occupancy_maps::SELECT_COVERS,
                &[
                    (occupancy_maps::SPLAT, Bind::Buffer(splat)),
                    (occupancy_maps::RECORDS, Bind::Buffer(records)),
                    (occupancy_maps::POSES, Bind::Buffer(poses)),
                    (occupancy_maps::COVERS, Bind::Buffer(covers)),
                    (occupancy_maps::LISTING, Bind::Buffer(listed)),
                    (occupancy_maps::LISTED_DRAW, Bind::Buffer(listed_draw)),
                ],
            )?);
            occupancy.push(Self::map_pipelines(
                cx,
                map,
                (drawn, splat),
                (records, poses, covers, listed),
                (coverage, heights, unfiner),
            )?);
            listed_inputs.push(cx.draw_input(drawn, listed_draw));
            selects.push(select);
            maps.push(drawn);
            splats.push(splat);
            listed_draws.push(listed_draw);
        }
        let with_ids = [VertexLayout::Mesh, VertexLayout::Buffer(IDS)];
        let object_vars = (draw::POSES, draw::COLORS, draw::GROUNDS);
        let object_buffers = (poses, colors, grounds);
        let lit_desc = |vertex| RenderPipeline {
            vertex,
            fragment: Some(draw::FS_OBJECT),
            debug: None,
            vertices: &with_ids,
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::Write {
                bias: wgpu::DepthBiasState::default(),
            },
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        };
        let depth_desc = |vertex, fragment| RenderPipeline {
            vertex,
            fragment,
            debug: None,
            vertices: &with_ids,
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::Write {
                bias: wgpu::DepthBiasState::default(),
            },
            blend: None,
            writes: wgpu::ColorWrites::empty(),
        };
        let lit = [
            cx.render_pipeline(
                MainView,
                &lit_desc(draw::VS_OBJECT),
                &objects(object_vars, object_buffers),
            )?,
            cx.render_pipeline(
                MainView,
                &lit_desc(draw::VS_OBJECT_FADING),
                &objects(object_vars, object_buffers),
            )?,
        ];
        let prepass = [
            [
                cx.render_pipeline(
                    MainView,
                    &depth_desc(draw::VS_OBJECT, None),
                    &objects(object_vars, object_buffers),
                )?,
                cx.render_pipeline(
                    MainView,
                    &depth_desc(draw::VS_OBJECT_FADING, None),
                    &objects(object_vars, object_buffers),
                )?,
            ],
            [
                cx.render_pipeline(
                    MainView,
                    &depth_desc(draw::VS_OBJECT, Some(draw::FS_MASKED_DEPTH)),
                    &objects(object_vars, object_buffers),
                )?,
                cx.render_pipeline(
                    MainView,
                    &depth_desc(draw::VS_OBJECT_FADING, Some(draw::FS_MASKED_DEPTH)),
                    &objects(object_vars, object_buffers),
                )?,
            ],
        ];
        let caster_desc = |vertex, fragment| RenderPipeline {
            vertex,
            fragment,
            debug: None,
            vertices: &with_ids,
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
        let main_ids = cx.draw_input(MainView, ids);
        let main_args = cx.draw_input(MainView, draws);
        let caster_ids = cx.draw_input(ShadowCasters, ids);
        let caster_args = cx.draw_input(ShadowCasters, draws);
        Ok(Self {
            snap: snap_pass,
            cull: cull_pass,
            selects,
            maps,
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
            selecting,
            occupancy,
            listed_inputs,
            lit,
            prepass,
            casting,
            main_ids,
            main_args,
            caster_ids,
            caster_args,
            held: Held::default(),
        })
    }

    fn prepare(
        &mut self,
        (state, windows, far, view, detail, maps, tiles, ground, residency): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        if state.is_added() {
            self.held.uploaded = None;
            self.held.groups_revision = None;
            self.held.covers_revision = None;
        }
        self.write_slots(cx, &state);
        if self.held.groups_revision != Some(state.groups_revision) {
            self.set_groups(cx, &state);
            self.held.groups_revision = Some(state.groups_revision);
        }
        for group in &mut self.held.groups {
            for (mesh, resident) in group.meshes.iter_mut().chain(&mut group.casters) {
                *resident = residency.resident(*mesh);
            }
        }
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
        let terrains = tiles.terrains.len() as u32;
        let snapping = terrains > 0
            && self.held.resident > 0
            && (self.held.slots_written || tiles.is_changed() || ground.is_changed());
        cx.allocate_buffer(self.snap_uniform, size_of::<Snap>() as u64);
        if snapping {
            let snap = Snap {
                counts: [self.held.objects, terrains, 0, 0],
                eye: view.detail_eye().extend(0.0).to_array(),
            };
            cx.write(self.snap_uniform, 0, &[snap]);
            self.held.slots_written = false;
        }
        self.prepare_cull(cx, (&*view, &*detail, &*maps), state.fade);
        if self.held.prepared && self.held.objects > 0 {
            if snapping {
                cx.run(self.snap);
            }
            cx.run(self.cull);
        }
        self.prepare_occupancy(cx, &windows);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let groups = [self.held.objects.div_ceil(SCATTER_GROUP), 1, 1];
        if pass == self.snap {
            return cx.dispatch(&self.snapping, Count::Groups(groups));
        }
        if pass == self.cull {
            return cx.dispatch(&self.culling, Count::Groups(groups));
        }
        for map in 0..MAX_OCCUPANCY_LEVELS {
            if pass == self.selects[map] {
                return cx.dispatch(
                    &self.selecting[map],
                    Count::Groups([self.held.workgroups, 1, 1]),
                );
            }
            if pass == self.maps[map] {
                return self.encode_map(map, cx);
            }
        }
        0
    }
}
