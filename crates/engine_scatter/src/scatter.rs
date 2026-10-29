use std::cell::Cell;
use std::collections::BTreeSet;
use std::rc::Rc;

use crate::shaders::entry::{scatter_cull, scatter_snap};
use engine_core::glam::{Mat4, Vec3};
use engine_core::{MeshHandle, RecordKey, RenderSnapshot, RenderStats};
use engine_faces::placed::{PlacedFace, Placements};
use engine_faces::shadows::KeptRefresh;

use crate::bindings::{CASTER_POSES, SCATTER_CULL, SCATTER_SNAP};
use crate::gpu_layouts::Cover;
use crate::scatter_draws::{PlacedImpostors, ScatterDraws};
use engine_render::bindings::{RETAINED_POSES, TEMPORAL_POSES};
use engine_render::buffers::{Growable, ReadbackRing};
use engine_render::cached::Cached;
use engine_render::gpu_data::{ColorData, GroundData, PoseData};
use engine_render::gpu_layouts::impostors::ImpostorRecord;
use engine_render::graph::{
    FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
};
use engine_render::parts::{
    Begin, BeginOut, Create, DrawAt, Encode, GpuPart, Hook, OpaqueSlot, PartAssets, PartDecl,
    PartEntry, Prelude, Step, View,
};

pub(crate) use crate::gpu_layouts::DEAD;
use crate::gpu_layouts::FADED;
use crate::gpu_layouts::FrameUniform;
pub(crate) use crate::gpu_layouts::LAYER_SHIFT;
pub(crate) use crate::gpu_layouts::NEAR_LEVEL_SHIFT;
pub(crate) use crate::gpu_layouts::NO_LAYER;
pub(crate) use crate::gpu_layouts::RANK_SHIFT;
use crate::gpu_layouts::SCATTER_GROUP;
pub(crate) use crate::gpu_layouts::SCATTER_LEVELS as LEVELS;
use crate::gpu_layouts::SHOWN_NONE;
pub(crate) use crate::gpu_layouts::STATE_ADOPT;
pub(crate) use crate::gpu_layouts::TERRAIN_INDEX;
use engine_render::gpu_layouts::constants::CARD_VERTICES;

pub(crate) const STATE_FRESH: u32 = SHOWN_NONE | SHOWN_NONE << 3 | FADED << 6;

const FADE_MARGIN: f32 = 0.1;

const REGIONS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;

use crate::gpu_layouts::ViewData;

pub(crate) const NEAR_LEVEL_MASK: u32 = 0x1f;
pub(crate) const RANK_MAX: f32 = 1023.0;

pub(crate) use crate::gpu_layouts::Record;

pub(crate) use crate::gpu_layouts::TerrainInfo;

pub(crate) struct SnapSources<'a> {
    pub heights: &'a wgpu::TextureView,
    pub control: &'a wgpu::TextureView,
    pub directory: &'a wgpu::Buffer,
    pub palette: &'a wgpu::Buffer,
    pub macro_maps: &'a wgpu::TextureView,
    pub generation: (u64, u64, u64),
    pub revision: u64,
    pub terrains: &'a [TerrainInfo],
}

use crate::gpu_layouts::GroupData;

type SnapKey = (u64, u64, (u64, u64, u64));

use crate::gpu_layouts::DrawArgs;

pub(crate) struct GroupLayout {
    pub levels: Vec<(u32, u32, i32)>,
    pub casters: Vec<(u32, u32, i32)>,
    pub camera: u32,
    pub errors: [f32; LEVELS as usize - 1],
    pub impostor: (f32, u32),
    pub parts: u32,
    pub part: bool,
}

#[derive(Clone, Copy, Default)]
struct GroupLists {
    ids: u32,
    count: u32,
}

pub(crate) trait PlacedCovers {
    fn covers(&self) -> &[Cover];

    fn objects(&self) -> u32;
}

pub(crate) struct ScatterPass {
    layout: engine_render::pipelines::GroupLayout,
    pose_layouts: [engine_render::pipelines::GroupLayout; 2],
    shadow_layout: engine_render::pipelines::GroupLayout,
    pipeline: Option<wgpu::ComputePipeline>,
    snap_layout: engine_render::pipelines::GroupLayout,
    snap_pipeline: Option<wgpu::ComputePipeline>,
    snap: wgpu::Buffer,
    terrains: Growable,
    snap_group: Cached<SnapKey, wgpu::BindGroup>,
    snapping: bool,
    snapped: Option<(SnapKey, u64, Vec<TerrainInfo>, [f32; 4])>,
    slots_written: bool,
    frame: wgpu::Buffer,
    views: Growable,
    records: Growable,
    groups: Growable,
    ids: Growable,
    draws: Growable,
    poses: Growable,
    colors: Growable,
    grounds: Growable,
    cards: Growable,
    card_args: Growable,
    generation: u64,
    cull_group: Cached<u64, wgpu::BindGroup>,
    pose_groups: Cached<u64, [wgpu::BindGroup; 2]>,
    shadow_group: Cached<u64, wgpu::BindGroup>,
    regions_placeholder: RegionsPlaceholder,
    args: DrawArgsLists,
    objects: u32,
    group_count: u32,
    lists: ListLayout,
    frame_views: FrameViews,
    terrains_generation: u64,
    left_out: LeftOut,
    pub(crate) resident: u32,
    pub(crate) placed: PlacedImpostors,
    held: Held,
}

#[derive(Default)]
struct Held {
    draws: ScatterDraws,
    meshes: (BTreeSet<MeshHandle>, u64),
    prepared: bool,
    moved: Vec<(Vec3, Vec3)>,
    restyled: bool,
    pass: Option<PassId>,
    placed: Option<ResourceId>,
}

struct RegionsPlaceholder {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Default)]
struct ListLayout {
    groups: Vec<GroupLists>,
    ids_per_view: u32,
    fading_view: usize,
}

#[derive(Default)]
struct FrameViews {
    maps: Vec<usize>,
    data: Vec<ViewData>,
    card_args: Vec<[u32; 4]>,
}

#[derive(Default)]
struct DrawArgsLists {
    template: Vec<DrawArgs>,
    casters: Vec<DrawArgs>,
    lists: Vec<DrawArgs>,
}

struct LeftOut {
    objects: u32,
    lists: bool,
    entries: Rc<Cell<u32>>,
    readback: ReadbackRing<()>,
}

impl ScatterPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let _owner = engine_render::memory::owned(PART.decl.brick);
        let growable =
            |label, bytes| Growable::new(device, label, bytes, wgpu::BufferUsages::empty());
        let regions_placeholder = engine_render::targets::texture_2d(
            device,
            "engine/scatter-regions-placeholder",
            (1, 1),
            1,
            1,
            REGIONS_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        Self {
            layout: engine_render::pipelines::layout(device, &SCATTER_CULL),
            pose_layouts: [
                engine_render::pipelines::layout(device, &RETAINED_POSES),
                engine_render::pipelines::layout(device, &TEMPORAL_POSES),
            ],
            shadow_layout: engine_render::pipelines::layout(device, &CASTER_POSES),
            pipeline: None,
            snap_layout: engine_render::pipelines::layout(device, &SCATTER_SNAP),
            snap_pipeline: None,
            snap: engine_render::buffers::uniform(
                device,
                "engine/scatter-snap",
                size_of::<crate::gpu_layouts::SnapUniform>() as u64,
            ),
            terrains: growable("engine/scatter-terrains", size_of::<TerrainInfo>() as u64),
            snap_group: Cached::default(),
            snapping: false,
            snapped: None,
            slots_written: false,
            terrains_generation: 0,
            frame: engine_render::buffers::uniform(
                device,
                "engine/scatter-frame",
                size_of::<FrameUniform>() as u64,
            ),
            views: growable("engine/scatter-views", size_of::<ViewData>() as u64),
            records: growable("engine/scatter-records", 32),
            groups: growable("engine/scatter-groups", 32),
            ids: Growable::new(device, "engine/scatter-ids", 16, wgpu::BufferUsages::VERTEX),
            draws: Growable::new(
                device,
                "engine/scatter-draws",
                20,
                wgpu::BufferUsages::INDIRECT,
            ),
            poses: growable("engine/scatter-poses", size_of::<PoseData>() as u64),
            colors: growable("engine/scatter-colors", size_of::<ColorData>() as u64),
            grounds: growable("engine/scatter-grounds", size_of::<GroundData>() as u64),
            cards: Growable::new(
                device,
                "engine/scatter-cards",
                8,
                wgpu::BufferUsages::VERTEX,
            ),
            card_args: Growable::new(
                device,
                "engine/scatter-card-args",
                16,
                wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
            ),
            generation: 0,
            cull_group: Cached::default(),
            pose_groups: Cached::default(),
            shadow_group: Cached::default(),
            regions_placeholder: RegionsPlaceholder {
                view: regions_placeholder.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("engine/scatter-regions-placeholder"),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    ..Default::default()
                }),
                _texture: regions_placeholder,
            },
            args: DrawArgsLists::default(),
            objects: 0,
            group_count: 0,
            lists: ListLayout::default(),
            frame_views: FrameViews::default(),
            left_out: LeftOut {
                objects: 0,
                lists: false,
                entries: Rc::new(Cell::new(0)),
                readback: ReadbackRing::new("engine/scatter-readback"),
            },
            resident: 0,
            placed: PlacedImpostors::new(device),
            held: Held::default(),
        }
    }

    pub(crate) fn left_out(&self) -> u32 {
        let entries = if self.left_out.lists {
            self.left_out.entries.get()
        } else {
            0
        };
        self.left_out.objects.saturating_add(entries)
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        [
            &self.views,
            &self.records,
            &self.groups,
            &self.ids,
            &self.draws,
            &self.poses,
            &self.colors,
            &self.grounds,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }

    pub(crate) fn set_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        groups: &[GroupLayout],
        counts: &[u32],
    ) {
        let mut data = Vec::with_capacity(groups.len());
        self.args.template.clear();
        self.args.casters.clear();
        self.lists.groups.clear();
        let mut ids = 0;
        for (group, &count) in groups.iter().zip(counts) {
            data.push(GroupData {
                ids,
                count,
                levels: group.levels.len() as u32,
                camera: group.camera,
                errors: group.errors,
                impostor: group.impostor.0,
                record: group.impostor.1,
                casters: group.casters.len() as u32,
                parts: group.parts,
            });
            self.lists.groups.push(GroupLists { ids, count });
            for level in 0..LEVELS as usize {
                for (template, ranges) in [
                    (&mut self.args.template, &group.levels),
                    (&mut self.args.casters, &group.casters),
                ] {
                    let (index_count, first_index, base_vertex) =
                        ranges.get(level).copied().unwrap_or((0, 0, 0));
                    template.push(DrawArgs {
                        index_count,
                        instance_count: 0,
                        first_index,
                        base_vertex,
                        first_instance: 0,
                    });
                }
            }
            ids += count * LEVELS;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        if self.groups.grow(device, bytes.len() as u64) {
            self.generation += 1;
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.groups, 0, bytes);
        }
        self.group_count = groups.len() as u32;
        self.lists.ids_per_view = ids;
        self.resident = counts
            .iter()
            .zip(groups)
            .filter(|(_, group)| !group.part)
            .map(|(&count, _)| count)
            .sum();
    }

    pub(crate) fn write_slots(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        (poses, colors, records, grounds): (&[PoseData], &[ColorData], &[Record], &[GroundData]),
        dirty: Option<std::ops::Range<u32>>,
    ) -> Option<std::ops::Range<u32>> {
        let room = [
            (&self.poses, size_of::<PoseData>()),
            (&self.colors, size_of::<ColorData>()),
            (&self.records, size_of::<Record>()),
            (&self.grounds, size_of::<GroundData>()),
        ]
        .into_iter()
        .map(|(buffer, stride)| buffer.room() / stride as u64)
        .min()
        .unwrap_or(0);
        let held = [poses.len(), colors.len(), records.len(), grounds.len()]
            .into_iter()
            .min()
            .unwrap_or(0)
            .min(usize::try_from(room).unwrap_or(usize::MAX))
            .min(u32::MAX as usize);
        self.left_out.objects = (records.len() - held).min(u32::MAX as usize) as u32;
        let (poses, colors, records, grounds) = (
            &poses[..held],
            &colors[..held],
            &records[..held],
            &grounds[..held],
        );
        let mut grown = false;
        for (buffer, bytes) in [
            (&mut self.poses, size_of_val(poses)),
            (&mut self.colors, size_of_val(colors)),
            (&mut self.records, size_of_val(records)),
            (&mut self.grounds, size_of_val(grounds)),
        ] {
            grown |= buffer.grow(device, bytes as u64);
        }
        if grown {
            self.generation += 1;
        }
        let range = if grown {
            Some(0..records.len())
        } else {
            dirty.map(|range| (range.start as usize).min(held)..(range.end as usize).min(held))
        };
        let written = range
            .clone()
            .filter(|range| !range.is_empty())
            .map(|range| range.start as u32..range.end as u32);
        if let Some(range) = range.filter(|range| !range.is_empty()) {
            self.slots_written = true;
            let write = |buffer: &wgpu::Buffer, stride: usize, bytes: &[u8]| {
                queue.write_buffer(buffer, (range.start * stride) as u64, bytes);
            };
            write(
                &self.poses,
                size_of::<PoseData>(),
                bytemuck::cast_slice(&poses[range.clone()]),
            );
            write(
                &self.colors,
                size_of::<ColorData>(),
                bytemuck::cast_slice(&colors[range.clone()]),
            );
            write(
                &self.records,
                size_of::<Record>(),
                bytemuck::cast_slice(&records[range.clone()]),
            );
            write(
                &self.grounds,
                size_of::<GroundData>(),
                bytemuck::cast_slice(&grounds[range.clone()]),
            );
        }
        self.objects = records.len() as u32;
        written
    }

    pub(crate) fn warm(&mut self, device: &wgpu::Device) {
        self.pipeline.get_or_insert_with(|| pipeline(device));
        self.snap_pipeline
            .get_or_insert_with(|| snap_pipeline(device));
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        (view_projection, eye): (Mat4, Vec3),
        (pixel_error, height, impostors, fade): (Option<f32>, f32, Option<f32>, f32),
        regions: &[(usize, KeptRefresh)],
    ) -> bool {
        self.frame_views.maps.clear();
        if self.resident == 0 {
            return false;
        }
        let y_row = view_projection.row(1).truncate().length();
        let spanned = y_row * height * 0.5;
        let pixels = pixel_error.map_or(0.0, |error| spanned / error);
        let cards = impostors.map_or(0.0, |size| y_row * height / size);
        let mut views = std::mem::take(&mut self.frame_views.data);
        views.clear();
        views.push(ViewData {
            planes: engine_render::gpu_data::planes(view_projection),
            limits: [0.0; 4],
        });
        for &(map, region) in regions {
            views.push(ViewData {
                planes: region.planes,
                limits: [0.5 * region.texel, region.texel, 0.0, 0.0],
            });
            self.frame_views.maps.push(map);
        }
        let view_count = views.len() as u32;
        let draws_per_view = self.group_count * LEVELS;
        self.lists.fading_view = views.len();
        let lists = view_count + 1;
        self.grow_views(device, view_count);
        let frame = FrameUniform {
            eye: eye.extend(pixels).to_array(),
            counts: [self.objects, self.group_count, LEVELS, view_count],
            strides: [
                self.lists.ids_per_view,
                draws_per_view,
                u32::from(impostors.is_some()),
                self.objects,
            ],
            fade: [fade, FADE_MARGIN, cards, spanned],
        };
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&frame));
        let card_args = &mut self.frame_views.card_args;
        card_args.clear();
        card_args.resize(view_count as usize, [CARD_VERTICES, 0, 0, 0]);
        card_args.push([0; 4]);
        queue.write_buffer(
            &self.card_args,
            0,
            bytemuck::cast_slice(card_args.as_slice()),
        );
        queue.write_buffer(&self.views, 0, bytemuck::cast_slice(&views));
        self.frame_views.data = views;
        self.args.lists.clear();
        for list in 0..lists {
            let shadow = list > 0 && list < view_count;
            self.args.lists.extend_from_slice(if shadow {
                &self.args.casters
            } else {
                &self.args.template
            });
        }
        queue.write_buffer(&self.draws, 0, bytemuck::cast_slice(&self.args.lists));
        self.pipeline.get_or_insert_with(|| pipeline(device));
        if self.cull_group.stale(&self.generation) {
            let group = engine_render::pipelines::group(
                device,
                "engine/scatter-cull",
                &SCATTER_CULL,
                &self.layout,
                &[
                    engine_render::buffers::entry("frame", &self.frame),
                    engine_render::buffers::entry("records", &self.records),
                    engine_render::buffers::entry("groups", &self.groups),
                    engine_render::buffers::entry("ids", &self.ids),
                    engine_render::buffers::entry("draws", &self.draws),
                    engine_render::buffers::entry("views", &self.views),
                    engine_render::buffers::entry("cards", &self.cards),
                    engine_render::buffers::entry("card_args", &self.card_args),
                ],
            );
            self.cull_group.set(self.generation, group);
        }
        self.build_pose_groups(device);
        true
    }

    fn build_pose_groups(&mut self, device: &wgpu::Device) {
        let [retained, temporal] = &self.pose_layouts;
        self.pose_groups.get_or_insert_with(self.generation, || {
            [
                engine_render::pipelines::group(
                    device,
                    "engine/scatter-poses",
                    &RETAINED_POSES,
                    retained,
                    &[
                        engine_render::buffers::entry("poses", &self.poses),
                        engine_render::buffers::entry("colors", &self.colors),
                        engine_render::buffers::entry("grounds", &self.grounds),
                    ],
                ),
                engine_render::pipelines::group(
                    device,
                    "engine/scatter-temporal-poses",
                    &TEMPORAL_POSES,
                    temporal,
                    &[
                        engine_render::buffers::entry("poses", &self.poses),
                        engine_render::buffers::entry("colors", &self.colors),
                        engine_render::buffers::entry("previous_poses", &self.poses),
                        engine_render::buffers::entry("grounds", &self.grounds),
                    ],
                ),
            ]
        });
        self.shadow_group.get_or_insert_with(self.generation, || {
            engine_render::pipelines::group(
                device,
                "engine/scatter-shadow-poses",
                &CASTER_POSES,
                &self.shadow_layout,
                &[
                    engine_render::buffers::entry("slot_poses", &self.poses),
                    engine_render::buffers::entry("slot_colors", &self.colors),
                    (
                        "shadow_regions",
                        engine_render::pipelines::view(&self.regions_placeholder.view),
                    ),
                ],
            )
        });
    }

    pub(crate) fn prepare_snap(
        &mut self,
        device: &wgpu::Device,
        queue: &engine_render::gpu_calls::Queue,
        sources: Option<SnapSources<'_>>,
        eye: Vec3,
    ) {
        self.snapping = false;
        let Some(sources) = sources.filter(|sources| !sources.terrains.is_empty()) else {
            return;
        };
        if self.resident == 0 {
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(sources.terrains);
        if self.terrains.grow(device, bytes.len() as u64) {
            self.terrains_generation += 1;
        }
        queue.write_buffer(&self.terrains, 0, bytes);
        let view = eye.extend(0.0).to_array();
        let uniform = crate::gpu_layouts::SnapUniform {
            counts: [self.objects, sources.terrains.len() as u32, 0, 0],
            eye: view,
        };
        queue.write_buffer(&self.snap, 0, bytemuck::bytes_of(&uniform));
        self.snap_pipeline
            .get_or_insert_with(|| snap_pipeline(device));
        let key = (
            self.generation,
            self.terrains_generation,
            sources.generation,
        );
        let unchanged = self
            .snapped
            .as_ref()
            .is_some_and(|(snapped, revision, terrains, _)| {
                *snapped == key && *revision == sources.revision && terrains == sources.terrains
            });
        if unchanged && !self.slots_written {
            match &mut self.snapped {
                Some((.., seen)) if *seen != view => *seen = view,
                _ => return,
            }
        } else {
            self.snapped = Some((key, sources.revision, sources.terrains.to_vec(), view));
            self.slots_written = false;
        }
        if self.snap_group.stale(&key) {
            let group = engine_render::pipelines::group(
                device,
                "engine/scatter-snap",
                &SCATTER_SNAP,
                &self.snap_layout,
                &[
                    engine_render::buffers::entry("snap", &self.snap),
                    engine_render::buffers::entry("records", &self.records),
                    engine_render::buffers::entry("poses", &self.poses),
                    engine_render::buffers::entry("terrains", &self.terrains),
                    engine_render::buffers::entry("directory", sources.directory),
                    ("heights", engine_render::pipelines::view(sources.heights)),
                    engine_render::buffers::entry("grounds", &self.grounds),
                    ("control", engine_render::pipelines::view(sources.control)),
                    engine_render::buffers::entry("palette", sources.palette),
                    (
                        "macro_maps",
                        engine_render::pipelines::view(sources.macro_maps),
                    ),
                ],
            );
            self.snap_group.set(key, group);
        }
        self.snapping = true;
    }

    pub(crate) fn pose_groups(&self) -> Option<&[wgpu::BindGroup; 2]> {
        self.pose_groups.get()
    }

    pub(crate) fn shadow_group(&self) -> Option<&wgpu::BindGroup> {
        self.shadow_group.get()
    }

    pub(crate) fn fading_view(&self) -> usize {
        self.lists.fading_view
    }

    pub(crate) fn map_view(&self, map: usize) -> Option<usize> {
        self.frame_views
            .maps
            .iter()
            .position(|&drawn| drawn == map)
            .map(|index| index + 1)
    }

    fn grow_views(&mut self, device: &wgpu::Device, view_count: u32) {
        let lists = u64::from(view_count) + 1;
        let ids = u64::from(self.lists.ids_per_view) * lists * 4;
        let draws = u64::from(self.group_count * LEVELS) * lists * size_of::<DrawArgs>() as u64;
        let cards = u64::from(self.objects) * u64::from(view_count) * 8;
        let mut replaced = self
            .views
            .grow(device, u64::from(view_count) * size_of::<ViewData>() as u64);
        replaced |= self.ids.grow(device, ids);
        replaced |= self.draws.grow(device, draws);
        replaced |= self.cards.grow(device, cards);
        replaced |= self
            .card_args
            .grow(device, (u64::from(view_count) + 1) * 16);
        if replaced {
            self.generation += 1;
        }
        self.left_out.lists = ids > self.ids.room() || cards > self.cards.room();
        if !self.left_out.lists {
            self.left_out.entries.set(0);
        }
    }

    fn encode_culling(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &engine_render::profiling::GpuProfiler,
    ) {
        let (Some(pipeline), Some(group)) = (&self.pipeline, self.cull_group.get()) else {
            return;
        };
        if self.objects == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/scatter-cull"),
            timestamp_writes: profiler.span("scatter").map(|span| span.compute()),
        });
        if let (true, Some(snap), Some(group)) =
            (self.snapping, &self.snap_pipeline, self.snap_group.get())
        {
            pass.set_pipeline(snap);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(self.objects.div_ceil(SCATTER_GROUP), 1, 1);
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(self.objects.div_ceil(SCATTER_GROUP), 1, 1);
        drop(pass);
        if self.left_out.lists {
            self.left_out.readback.record_at(
                device,
                encoder,
                (&self.card_args, self.lists.fading_view as u64 * 16),
                16,
                (),
            );
        }
    }

    pub(crate) fn card_source(
        &self,
        view: usize,
    ) -> Option<(wgpu::BufferSlice<'_>, &wgpu::Buffer, u64)> {
        let start = view as u64 * u64::from(self.objects) * 8;
        let args: &wgpu::Buffer = &self.card_args;
        (start < self.cards.size()).then(|| (self.cards.slice(start..), args, view as u64 * 16))
    }

    pub(crate) fn draw_source(
        &self,
        view: usize,
        group: usize,
        level: usize,
    ) -> Option<(wgpu::BufferSlice<'_>, &wgpu::Buffer, u64)> {
        let lists = self.lists.groups.get(group)?;
        let start = (view as u64 * u64::from(self.lists.ids_per_view)
            + u64::from(lists.ids + level as u32 * lists.count))
            * 4;
        let draw = (view * self.group_count as usize + group) * LEVELS as usize + level;
        let args: &wgpu::Buffer = &self.draws;
        (start < self.ids.size()).then(|| {
            (
                self.ids.slice(start..),
                args,
                (draw * size_of::<DrawArgs>()) as u64,
            )
        })
    }
}

fn snap_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let pipeline_layout = engine_render::pipelines::layout_of(
        device,
        &crate::pipeline_layouts::SCATTER_SNAP_PIPELINES,
    );
    engine_render::pipelines::compute_entry(device, &pipeline_layout, scatter_snap::snap_scatter)
}

fn pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let pipeline_layout = engine_render::pipelines::layout_of(
        device,
        &crate::pipeline_layouts::SCATTER_CULL_PIPELINES,
    );
    engine_render::pipelines::compute_entry(device, &pipeline_layout, scatter_cull::cull_scatter)
}

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "scatter",
        brick: "scatter",
        passes: &[PassDecl {
            name: "scatter",
            stage: Stage::Prepare,
        }],
        resources: &[ResourceDecl {
            name: "placed",
            lifetime: Lifetime::Frame,
        }],
        hooks: &[
            Hook::Opaque(OpaqueSlot::Prepass),
            Hook::Opaque(OpaqueSlot::Objects),
            Hook::ShadowCasters,
        ],
        contributes: &[],
        effects: &[],
        labels: &[],
        lazy: false,
        faces: &[
            engine_render::offer_face!(ScatterPass => dyn PlacedFace),
            engine_render::offer_face!(ScatterPass => dyn PlacedCovers),
        ],
        needs: &["meshes"],
        piece: None,
    },
    create,
};

fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(ScatterPass::new(&cx.core.device))
}

impl PlacedFace for ScatterPass {
    fn moved(&self) -> &[(Vec3, Vec3)] {
        &self.held.moved
    }

    fn restyled(&self) -> bool {
        self.held.restyled
    }

    fn impostors(&self) -> &[ImpostorRecord] {
        self.held.draws.impostors()
    }

    fn placements(&self) -> Placements<'_> {
        self.held.draws.placements()
    }

    fn slot_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, u64) {
        (&self.records, &self.poses, self.generation)
    }

    fn frames(&self) -> Option<(&wgpu::TextureView, u64)> {
        self.placed.atlas.frames()
    }
}

impl PlacedCovers for ScatterPass {
    fn covers(&self) -> &[Cover] {
        self.held.draws.covers()
    }

    fn objects(&self) -> u32 {
        self.objects
    }
}

impl GpuPart for ScatterPass {
    fn install(&mut self, table: &GraphTable) {
        self.held.pass = table.pass_named("scatter");
        self.held.placed = table.resource("placed");
    }

    fn begin(&mut self, cx: &mut Begin<'_>, _out: &mut BeginOut) {
        let placed: BTreeSet<MeshHandle> = engine_terrain::RenderTerrains::of(cx.snapshot)
            .terrains
            .iter()
            .flat_map(|terrain| terrain.assets.iter().flatten())
            .flat_map(|asset| asset.meshes().map(|mesh| mesh.handle))
            .collect();
        if placed != self.held.meshes.0 {
            self.held.meshes = (placed, self.held.meshes.1 + 1);
        }
    }

    fn asset_generation(&self) -> u64 {
        self.held.meshes.1
    }

    fn assets<'s>(&self, snapshot: &'s RenderSnapshot, assets: &mut PartAssets<'s>) {
        let placed = engine_terrain::RenderTerrains::of(snapshot)
            .terrains
            .iter()
            .flat_map(|terrain| terrain.assets.iter().flatten());
        for asset in placed {
            assets
                .meshes
                .extend(asset.meshes().map(|mesh| (mesh.handle, mesh)));
            assets.materials.extend(asset.materials());
            assets.textures.extend(asset.textures.iter());
        }
    }

    fn prepare(&mut self, cx: &mut Step<'_>, prelude: &mut Prelude) {
        self.held.restyled = self
            .held
            .draws
            .prepare_impostors(cx, prelude, &mut self.placed);
    }

    fn prepare_draws(&mut self, cx: &mut Step<'_>, _encoder: &mut wgpu::CommandEncoder) {
        let mut draws = std::mem::take(&mut self.held.draws);
        let mut moved = std::mem::take(&mut self.held.moved);
        moved.clear();
        self.held.prepared = draws.prepare(self, cx, &mut moved);
        self.held.draws = draws;
        self.held.moved = moved;
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        if let (true, Some(pass)) = (self.held.prepared, self.held.pass) {
            let _ = graph.add(cx.table, pass).write(self.held.placed);
        }
    }

    fn encode(
        &mut self,
        _pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        self.encode_culling(&cx.core.device, encoder, &cx.core.profiler);
        0
    }

    fn draw(&self, at: DrawAt<'_>, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        if !self.held.prepared {
            return 0;
        }
        let draws = &self.held.draws;
        match at {
            DrawAt::Opaque {
                slot: OpaqueSlot::Prepass,
                phase: 0,
            } => draws.draw(self, cx, pass, true),
            DrawAt::Opaque {
                slot: OpaqueSlot::Objects,
                ..
            } => draws.draw(self, cx, pass, false),
            DrawAt::ShadowCasters {
                map,
                casters,
                cards,
            } => draws.draw_shadow(self, cx, pass, (map, casters, cards)),
            DrawAt::Opaque { .. } | DrawAt::View { .. } | DrawAt::Transparent { .. } => 0,
        }
    }

    fn after_submit(&mut self) {
        self.held.prepared = false;
        self.placed.atlas.after_submit();
        let entries = Rc::clone(&self.left_out.entries);
        self.left_out
            .readback
            .after_submit(move |words, ()| entries.set(words.get(1)));
    }

    fn capacity_bytes(&self) -> u64 {
        ScatterPass::capacity_bytes(self)
    }

    fn stats(&self, _cx: &View<'_>, stats: &mut RenderStats) {
        let placed = self.placed.atlas.live();
        stats.terrain.placed_objects = self.resident;
        stats.terrain.placed_far_objects = self.held.draws.far_objects();
        stats.terrain.placed_impostors = placed;
        stats.terrain.dropped_placed_objects = self.left_out();
        stats.level_of_detail.impostors += placed;
        stats.level_of_detail.dropped_impostors += self.placed.dropped;
    }

    fn set_cooking(&mut self, cooking: bool) {
        self.placed.atlas.set_cooking(cooking);
    }

    fn records_left(&self) -> Option<(&'static str, u64)> {
        Some(("placed impostors", self.placed.atlas.left()))
    }

    fn record_wants(&mut self, wants: &mut Vec<RecordKey>) {
        self.placed.atlas.take_wants(wants);
    }

    fn asks_record(&self, record: RecordKey) -> bool {
        self.placed.atlas.asks(record)
    }

    fn answer_record(&mut self, record: RecordKey, bytes: Option<Vec<u8>>) {
        self.placed.atlas.answer(record, bytes);
    }

    fn made_records(&mut self, made: &mut Vec<(RecordKey, Vec<u8>)>) {
        self.placed.atlas.take_made(made);
    }
}
