//! Placed objects: what terrains' placement layers stand on
//! their tiles, drawn through the mesh pipelines as instances of their own.
//! Objects hold slots of a pool whose poses and colors are laid out as
//! slots' are, so the mesh and depth shaders read them unchanged; each
//! tile's objects upload when it arrives. Each frame a compute pass first
//! sets every object on its terrain's finest resident heights
//! (`scatter_snap.wgsl`), then culls it against each view (the camera,
//! then the shadow maps drawn every frame) and picks its level of detail
//! from the camera (`scatter_cull.wgsl`), appending it to its draw
//! group's list for that view and level, or to the view's impostor cards;
//! each group and level then draws indirectly, and
//! each view's cards. Objects draw in the first opaque phase and cast into
//! every shadow map: the kept ones as seen from afar, the near cascades as
//! the camera sees them, cards cast facing the sun, and the
//! moving map without the camera's cards. They are never occlusion culled.

use std::cell::Cell;
use std::rc::Rc;

use crate::shaders::entry::{scatter_cull, scatter_snap};
use engine_core::glam::{Mat4, Vec3};

use crate::bindings::{RETAINED_POSES, SCATTER_CULL, SCATTER_SNAP, SHADOW_POSES, TEMPORAL_POSES};
use crate::buffers::{Growable, ReadbackRing};
use crate::cached::Cached;
use crate::gpu_data::{ColorData, GroundData, PoseData};
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{FrameShape, Node, Pass, RenderGraph, Resource};
use crate::shadows::KeptRefresh;

use crate::gpu_layouts::constants::CARD_VERTICES;
pub(crate) use crate::gpu_layouts::constants::DEAD;
use crate::gpu_layouts::constants::FADED;
pub(crate) use crate::gpu_layouts::constants::LAYER_SHIFT;
pub(crate) use crate::gpu_layouts::constants::NEAR_LEVEL_SHIFT;
pub(crate) use crate::gpu_layouts::constants::NO_LAYER;
pub(crate) use crate::gpu_layouts::constants::RANK_SHIFT;
use crate::gpu_layouts::constants::SCATTER_GROUP;
pub(crate) use crate::gpu_layouts::constants::SCATTER_LEVELS as LEVELS;
use crate::gpu_layouts::constants::SHOWN_NONE;
pub(crate) use crate::gpu_layouts::constants::STATE_ADOPT;
pub(crate) use crate::gpu_layouts::constants::TERRAIN_INDEX;
pub(crate) use crate::gpu_layouts::constants::THIN_LAYERS;
use crate::gpu_layouts::scatter::FrameUniform;

pub(crate) const STATE_FRESH: u32 = SHOWN_NONE | SHOWN_NONE << 3 | FADED << 6;

/// The share of its distance past a switch the eye goes before the camera
/// switches an object back, so objects near a switch's
/// distance do not flicker between two ways as the eye wavers.
const FADE_MARGIN: f32 = 0.1;

use crate::gpu_layouts::scatter::ViewData;

/// The shadow maps placed objects cast into: those drawn every frame
/// (their world-to-clip matrices), this frame's redraws of the kept map and
/// of a kept level, and the far maps' texels a side.
pub(crate) type ShadowViews<'a> = (
    &'a [(usize, Mat4)],
    Option<KeptRefresh>,
    Option<KeptRefresh>,
    u32,
);

pub(crate) const NEAR_LEVEL_MASK: u32 = 0x1f;
pub(crate) const RANK_MAX: f32 = 1023.0;

pub(crate) use crate::gpu_layouts::scatter::Record;

pub(crate) use crate::gpu_layouts::scatter::TerrainInfo;

/// What setting objects on the terrain reads: the terrain pass's resident
/// heights and directory, their generations, a revision advancing whenever
/// resident tiles change, and each terrain's place.
pub(crate) struct SnapSources<'a> {
    pub heights: &'a wgpu::TextureView,
    pub directory: &'a wgpu::Buffer,
    pub generation: (u64, u64),
    pub revision: u64,
    pub terrains: &'a [TerrainInfo],
}

use crate::gpu_layouts::scatter::GroupData;

/// What setting objects on the terrain binds: the generations of the
/// objects' buffers, the terrains' places and the terrain pass's heights
/// and directory.
type SnapKey = (u64, u64, (u64, u64));

use crate::gpu_layouts::scatter::DrawArgs;

/// One draw group as the renderer lays it out: each level's index count,
/// first index and base vertex, the levels the camera draws, the coarser
/// levels' errors, and its impostor's error (mesh units; 0 without one)
/// and record.
pub(crate) struct GroupLayout {
    pub levels: Vec<(u32, u32, i32)>,
    pub camera: u32,
    pub errors: [f32; LEVELS as usize - 1],
    pub impostor: (f32, u32),
}

pub(crate) struct ScatterPass {
    layout: crate::pipelines::GroupLayout,
    // The mesh pipelines' pose layouts, plain and temporal, and the depth
    // passes', which equal descriptors make interchangeable with theirs.
    pose_layouts: [crate::pipelines::GroupLayout; 2],
    shadow_layout: crate::pipelines::GroupLayout,
    pipeline: Option<wgpu::ComputePipeline>,
    // Setting objects on the terrain: its layout and pipeline, its uniform,
    // the terrains' places, its group (by the buffers' and terrain
    // sources' generations), and whether this frame runs it.
    snap_layout: crate::pipelines::GroupLayout,
    snap_pipeline: Option<wgpu::ComputePipeline>,
    snap: wgpu::Buffer,
    terrains: Growable,
    snap_group: Cached<SnapKey, wgpu::BindGroup>,
    snapping: bool,
    // What objects were last set on: the group's key, the resident tiles'
    // revision and the terrains' places, and whether slots were written
    // since, undoing it.
    snapped: Option<(SnapKey, u64, Vec<TerrainInfo>)>,
    slots_written: bool,
    // Advances when objects' poses or records change.
    version: u64,
    frame: wgpu::Buffer,
    views: Growable,
    records: Growable,
    groups: Growable,
    ids: Growable,
    draws: Growable,
    poses: Growable,
    colors: Growable,
    palette: wgpu::Buffer,
    // What lies under each object.
    grounds: Growable,
    // Each view's impostor cards (each one's slot and impostor record, as
    // many as objects a view) and their draws.
    cards: Growable,
    card_args: Growable,
    // Advances when buffers are replaced.
    generation: u64,
    // How each terrain's placement layers thin with distance,
    // as last written.
    thins: Growable,
    thins_written: Vec<[f32; 4]>,
    cull_group: Cached<u64, wgpu::BindGroup>,
    // The pose groups of plain and temporal mesh pipelines, and of depth
    // passes.
    pose_groups: Cached<u64, [wgpu::BindGroup; 2]>,
    shadow_group: Cached<u64, wgpu::BindGroup>,
    // Each draw's arguments with no instances, and every view's, rewritten
    // every frame.
    args: DrawArgsLists,
    objects: u32,
    group_count: u32,
    // The IDs each view's lists take, and the view whose lists hold the
    // camera's fading objects: the one after the last.
    ids_per_view: u32,
    fading_view: usize,
    // The frame's views: the shadow map each one after the camera's draws
    // into, what culling reads of each, and their card arguments.
    frame_views: FrameViews,
    // Advances when the terrains' buffer is replaced.
    terrains_generation: u64,
    // What the device's storage bindings leave out of the objects.
    left_out: LeftOut,
    /// Objects resident.
    pub(crate) resident: u32,
}

/// A frame's views: the shadow map each one after the camera's draws into,
/// what culling reads of each, and their card arguments, then the count of
/// entries lists left out; kept from frame to frame.
#[derive(Default)]
struct FrameViews {
    maps: Vec<usize>,
    data: Vec<ViewData>,
    card_args: Vec<[u32; 4]>,
}

/// Each draw's arguments with no instances, and every list's, rewritten
/// every frame.
#[derive(Default)]
struct DrawArgsLists {
    template: Vec<DrawArgs>,
    lists: Vec<DrawArgs>,
}

/// What the device's storage bindings leave out of placed objects: those
/// past what the objects' buffers hold, and, while the views' lists may
/// reach past what theirs hold, the entries culling left out, read back a
/// few frames late.
struct LeftOut {
    objects: u32,
    lists: bool,
    entries: Rc<Cell<u32>>,
    readback: ReadbackRing<()>,
}

impl ScatterPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let storage = |label, bytes| {
            crate::buffers::storage(device, label, bytes, wgpu::BufferUsages::empty())
        };
        let growable =
            |label, bytes| Growable::new(device, label, bytes, wgpu::BufferUsages::empty());
        Self {
            layout: crate::pipelines::layout(device, &SCATTER_CULL),
            pose_layouts: [
                crate::pipelines::layout(device, &RETAINED_POSES),
                crate::pipelines::layout(device, &TEMPORAL_POSES),
            ],
            shadow_layout: crate::pipelines::layout(device, &SHADOW_POSES),
            pipeline: None,
            snap_layout: crate::pipelines::layout(device, &SCATTER_SNAP),
            snap_pipeline: None,
            snap: crate::buffers::uniform(device, "engine/scatter-snap", 16),
            terrains: growable("engine/scatter-terrains", size_of::<TerrainInfo>() as u64),
            snap_group: Cached::default(),
            snapping: false,
            snapped: None,
            slots_written: false,
            version: 0,
            terrains_generation: 0,
            frame: crate::buffers::uniform(
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
            palette: storage("engine/scatter-palette", size_of::<PoseData>() as u64),
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
            thins: growable("engine/scatter-thins", 16),
            thins_written: Vec::new(),
            cull_group: Cached::default(),
            pose_groups: Cached::default(),
            shadow_group: Cached::default(),
            args: DrawArgsLists::default(),
            objects: 0,
            group_count: 0,
            ids_per_view: 0,
            fading_view: 0,
            frame_views: FrameViews::default(),
            left_out: LeftOut {
                objects: 0,
                lists: false,
                entries: Rc::new(Cell::new(0)),
                readback: ReadbackRing::new("engine/scatter-readback"),
            },
            resident: 0,
        }
    }

    /// Placed objects the device's storage bindings left out: those past
    /// what the objects' buffers hold, and the entries the views' lists left
    /// out, each once a view and level, read back a few frames late.
    pub(crate) fn left_out(&self) -> u32 {
        let entries = if self.left_out.lists {
            self.left_out.entries.get()
        } else {
            0
        };
        self.left_out.objects.saturating_add(entries)
    }

    /// Bytes of the objects' buffers.
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

    /// Lays out the draw groups `groups` holding `counts` live objects:
    /// each group's lists and the draws' template.
    pub(crate) fn set_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        groups: &[GroupLayout],
        counts: &[u32],
    ) {
        let mut data = Vec::with_capacity(groups.len());
        self.args.template.clear();
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
                pad: [0; 2],
            });
            for level in 0..LEVELS as usize {
                let (index_count, first_index, base_vertex) =
                    group.levels.get(level).copied().unwrap_or((0, 0, 0));
                self.args.template.push(DrawArgs {
                    index_count,
                    instance_count: 0,
                    first_index,
                    base_vertex,
                    first_instance: 0,
                });
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
        self.ids_per_view = ids;
        self.resident = counts.iter().sum();
    }

    /// Writes slots `dirty` of `poses`, `colors`, `records` and `grounds`,
    /// every slot when the buffers grow to hold them all; returns the slots
    /// written.
    pub(crate) fn write_slots(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (poses, colors, records, grounds): (&[PoseData], &[ColorData], &[Record], &[GroundData]),
        dirty: Option<std::ops::Range<u32>>,
    ) -> Option<std::ops::Range<u32>> {
        // Objects past what one storage binding of any of their buffers
        // holds are left out, counted.
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
            self.version += 1;
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

    /// Creates the pipelines objects are set on the ground and culled with
    /// unless they exist.
    pub(crate) fn warm(&mut self, device: &wgpu::Device) {
        self.pipeline
            .get_or_insert_with(|| pipeline(device, &self.layout));
        self.snap_pipeline
            .get_or_insert_with(|| snap_pipeline(device, &self.snap_layout));
    }

    /// Writes how each terrain's placement layers thin with distance
    /// when it changed: `THIN_LAYERS` entries a terrain.
    pub(crate) fn set_thinning(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        thins: &[[f32; 4]],
    ) {
        if thins == self.thins_written.as_slice() {
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(thins);
        if self.thins.grow(device, (bytes.len() as u64).max(16)) {
            self.generation += 1;
        }
        queue.write_buffer(&self.thins, 0, bytes);
        self.thins_written = thins.to_vec();
    }

    /// Plans this frame's culling: the camera's view, from its
    /// `view_projection` (world) and eye, those of the shadow `maps`
    /// drawn every frame (their world-to-clip matrices, `resolution` texels
    /// square), where objects smaller than half a texel cast nothing, and
    /// the regions of the kept map and of a kept level
    /// redrawn this frame, if any. Levels are picked within `pixel_error`
    /// pixels of a viewport `height` pixels tall (no error: finest levels
    /// only); maps draw a coarser level where a texel hides its error, and
    /// the camera draws impostors within the error when `impostors`, which
    /// kept maps draw wherever their texels are no finer than an impostor's.
    /// The camera's switches fade `fade` of their way this
    /// frame, or stand still where it is negative. Returns
    /// whether any object draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (view_projection, eye): (Mat4, Vec3),
        (pixel_error, height, impostors, fade): (Option<f32>, f32, bool, f32),
        (maps, kept, level, resolution): ShadowViews<'_>,
    ) -> bool {
        self.frame_views.maps.clear();
        if self.resident == 0 {
            return false;
        }
        let y_row = view_projection.row(1).truncate().length();
        let pixels = pixel_error.map_or(0.0, |error| y_row * height * 0.5 / error);
        let mut views = std::mem::take(&mut self.frame_views.data);
        views.clear();
        views.push(ViewData {
            planes: crate::ground_cover::planes(view_projection),
            limits: [0.0; 4],
        });
        let half_texel = |clip: Mat4| {
            let texels = clip.row(0).truncate().length() * resolution as f32;
            1.0 / engine_core::math::max(texels, 1.0e-12)
        };
        for &(map, clip) in maps {
            // An orthographic map: its sides alone bound what casts into it.
            let mut planes = crate::ground_cover::planes(clip);
            planes[4] = [0.0, 0.0, 0.0, 1.0];
            planes[5] = [0.0, 0.0, 0.0, 1.0];
            let half_texel = half_texel(clip);
            // Near cascades cast the camera's cards.
            let cascade = if map < crate::shadows::MOVING_MAP {
                1.0
            } else {
                0.0
            };
            views.push(ViewData {
                planes,
                limits: [half_texel, 2.0 * half_texel, 0.0, cascade],
            });
            self.frame_views.maps.push(map);
        }
        for (map, refresh) in [
            (crate::shadows::KEPT_MAP, kept),
            (crate::shadows::KEPT_LEVELS, level),
        ] {
            if let Some(refresh) = refresh {
                views.push(ViewData {
                    planes: refresh.planes,
                    limits: [0.5 * refresh.texel, refresh.texel, 1.0, 0.0],
                });
                self.frame_views.maps.push(map);
            }
        }
        let view_count = views.len() as u32;
        let draws_per_view = self.group_count * LEVELS;
        // The camera's fading objects list after the last view's.
        self.fading_view = views.len();
        let lists = view_count + 1;
        self.grow_views(device, view_count);
        let frame = FrameUniform {
            eye: eye.extend(pixels).to_array(),
            counts: [self.objects, self.group_count, LEVELS, view_count],
            strides: [
                self.ids_per_view,
                draws_per_view,
                u32::from(impostors),
                self.objects,
            ],
            fade: [fade, FADE_MARGIN, 0.0, 0.0],
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
        for _ in 0..lists {
            self.args.lists.extend_from_slice(&self.args.template);
        }
        queue.write_buffer(&self.draws, 0, bytemuck::cast_slice(&self.args.lists));
        self.pipeline
            .get_or_insert_with(|| pipeline(device, &self.layout));
        if self.cull_group.stale(&self.generation) {
            let group = crate::pipelines::group(
                device,
                "engine/scatter-cull",
                &SCATTER_CULL,
                &self.layout,
                &[
                    crate::buffers::entry("frame", &self.frame),
                    crate::buffers::entry("records", &self.records),
                    crate::buffers::entry("groups", &self.groups),
                    crate::buffers::entry("ids", &self.ids),
                    crate::buffers::entry("draws", &self.draws),
                    crate::buffers::entry("views", &self.views),
                    crate::buffers::entry("cards", &self.cards),
                    crate::buffers::entry("card_args", &self.card_args),
                    crate::buffers::entry("thins", &self.thins),
                ],
            );
            self.cull_group.set(self.generation, group);
        }
        self.build_pose_groups(device);
        true
    }

    /// Builds the pose groups of plain and temporal mesh pipelines and of
    /// depth passes over the objects' poses (their previous poses being the
    /// same: objects stand still) when buffers changed.
    fn build_pose_groups(&mut self, device: &wgpu::Device) {
        let [retained, temporal] = &self.pose_layouts;
        self.pose_groups.get_or_insert_with(self.generation, || {
            [
                crate::pipelines::group(
                    device,
                    "engine/scatter-poses",
                    &RETAINED_POSES,
                    retained,
                    &[
                        crate::buffers::entry("poses", &self.poses),
                        crate::buffers::entry("colors", &self.colors),
                        crate::buffers::entry("palette", &self.palette),
                        crate::buffers::entry("grounds", &self.grounds),
                    ],
                ),
                crate::pipelines::group(
                    device,
                    "engine/scatter-temporal-poses",
                    &TEMPORAL_POSES,
                    temporal,
                    &[
                        crate::buffers::entry("poses", &self.poses),
                        crate::buffers::entry("colors", &self.colors),
                        crate::buffers::entry("previous_poses", &self.poses),
                        crate::buffers::entry("palette", &self.palette),
                        crate::buffers::entry("previous_palette", &self.palette),
                        crate::buffers::entry("grounds", &self.grounds),
                    ],
                ),
            ]
        });
        self.shadow_group.get_or_insert_with(self.generation, || {
            crate::pipelines::group(
                device,
                "engine/scatter-shadow-poses",
                &SHADOW_POSES,
                &self.shadow_layout,
                &[
                    crate::buffers::entry("slot_poses", &self.poses),
                    crate::buffers::entry("slot_colors", &self.colors),
                    crate::buffers::entry("slot_palette", &self.palette),
                ],
            )
        });
    }

    /// Plans setting objects on the terrain this frame from `sources`: when
    /// slots were written or the resident heights or terrains changed since
    /// they last were; never while the terrain pass holds no heights.
    pub(crate) fn prepare_snap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sources: Option<SnapSources<'_>>,
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
        let counts = [self.objects, sources.terrains.len() as u32, 0, 0];
        queue.write_buffer(&self.snap, 0, bytemuck::cast_slice(&counts));
        self.snap_pipeline
            .get_or_insert_with(|| snap_pipeline(device, &self.snap_layout));
        let key = (
            self.generation,
            self.terrains_generation,
            sources.generation,
        );
        let unchanged = self
            .snapped
            .as_ref()
            .is_some_and(|(snapped, revision, terrains)| {
                *snapped == key && *revision == sources.revision && terrains == sources.terrains
            });
        if unchanged && !self.slots_written {
            return;
        }
        self.snapped = Some((key, sources.revision, sources.terrains.to_vec()));
        self.slots_written = false;
        self.version += 1;
        if self.snap_group.stale(&key) {
            let group = crate::pipelines::group(
                device,
                "engine/scatter-snap",
                &SCATTER_SNAP,
                &self.snap_layout,
                &[
                    crate::buffers::entry("snap", &self.snap),
                    crate::buffers::entry("records", &self.records),
                    crate::buffers::entry("poses", &self.poses),
                    crate::buffers::entry("terrains", &self.terrains),
                    crate::buffers::entry("directory", sources.directory),
                    ("heights", crate::pipelines::view(sources.heights)),
                ],
            );
            self.snap_group.set(key, group);
        }
        self.snapping = true;
    }

    /// Slots the culling spans, live or dead.
    pub(crate) fn objects(&self) -> u32 {
        self.objects
    }

    /// A version advancing whenever objects' poses or records change:
    /// written, or set on the terrain.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// The objects' records and poses, and a generation advancing when the
    /// buffers are replaced.
    pub(crate) fn slot_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u64) {
        (&self.records, &self.poses, &self.thins, self.generation)
    }

    /// The pose groups of plain and temporal mesh pipelines.
    pub(crate) fn pose_groups(&self) -> Option<&[wgpu::BindGroup; 2]> {
        self.pose_groups.get()
    }

    /// The pose group of depth passes.
    pub(crate) fn shadow_group(&self) -> Option<&wgpu::BindGroup> {
        self.shadow_group.get()
    }

    /// The view whose lists hold the camera's fading objects this frame.
    pub(crate) fn fading_view(&self) -> usize {
        self.fading_view
    }

    /// The view shadow map `map` reads its lists from this frame, if it
    /// draws objects.
    pub(crate) fn map_view(&self, map: usize) -> Option<usize> {
        self.frame_views
            .maps
            .iter()
            .position(|&drawn| drawn == map)
            .map(|index| index + 1)
    }

    /// Grows the buffers culling writes to hold `view_count` views' lists
    /// (the camera's fading objects' after them), cards and card arguments
    /// (then the count of entries lists left out), and notes whether the
    /// lists may reach past what their bindings hold, which leaves entries
    /// out that culling counts and frames read back.
    fn grow_views(&mut self, device: &wgpu::Device, view_count: u32) {
        let lists = u64::from(view_count) + 1;
        let ids = u64::from(self.ids_per_view) * lists * 4;
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

    /// Records this frame's culling on `device`, timed by `profiler`
    /// (`scatter`), and the read-back of the entries its lists left out
    /// while they may.
    fn encode_culling(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
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
                (&self.card_args, self.fading_view as u64 * 16),
                16,
                (),
            );
        }
    }

    /// `view`'s impostor cards (0 the camera's), each one's slot and
    /// impostor record as instance attributes: the slice to bind and the
    /// indirect buffer and offset of their draw; none where they start past
    /// what the cards' binding holds, which culling left them out of.
    pub(crate) fn card_source(
        &self,
        view: usize,
    ) -> Option<(wgpu::BufferSlice<'_>, &wgpu::Buffer, u64)> {
        let start = view as u64 * u64::from(self.objects) * 8;
        let args: &wgpu::Buffer = &self.card_args;
        (start < self.cards.size()).then(|| (self.cards.slice(start..), args, view as u64 * 16))
    }

    /// Group `group`'s list of level `level` in `view` (0 the camera's) and
    /// its draw's arguments: the visible-ID slice to bind and the indirect
    /// buffer and offset; its lists start at `ids_base`. None where the list
    /// starts past what the IDs' binding holds, which culling left it out
    /// of (a list reaching past it draws no instances).
    pub(crate) fn draw_source(
        &self,
        view: usize,
        (group, ids_base): (usize, u32),
        level: usize,
    ) -> Option<(wgpu::BufferSlice<'_>, &wgpu::Buffer, u64)> {
        let start = (view as u64 * u64::from(self.ids_per_view) + u64::from(ids_base)) * 4;
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

fn snap_pipeline(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/scatter-snap.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::SCATTER_SNAP_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::SCATTER_SNAP_PIPELINES,
        &[layout],
    );
    crate::pipelines::compute_entry(
        device,
        &pipeline_layout,
        &module,
        scatter_snap::snap_scatter,
    )
}

fn pipeline(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/scatter-cull.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::SCATTER_CULL_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::SCATTER_CULL_PIPELINES,
        &[layout],
    );
    crate::pipelines::compute_entry(
        device,
        &pipeline_layout,
        &module,
        scatter_cull::cull_scatter,
    )
}

/// A frame that draws placed objects.
pub(crate) struct ScatterFrame;

impl RenderFeature for ScatterPass {
    type Frame = ScatterFrame;
    type Stats = ();

    /// Culling writes the placed objects' lists and draws.
    fn declare(_frame: &ScatterFrame, _shape: &FrameShape, graph: &mut RenderGraph) {
        graph
            .add(Pass::Feature(FeaturePass::Scatter))
            .write(Resource::Placed);
    }

    /// Placed objects cull before the shadow maps and the first opaque
    /// phase draw them.
    fn encode(
        &self,
        _frame: &ScatterFrame,
        _node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        self.encode_culling(encode.device, encoder, encode.profiler);
        0
    }

    /// Requests the map of the entries this frame's lists left out, where
    /// it read them back.
    fn after_submit(&mut self, _frame: Option<&ScatterFrame>) {
        let entries = Rc::clone(&self.left_out.entries);
        self.left_out
            .readback
            .after_submit(move |words, ()| entries.set(words.get(1)));
    }

    fn capacity_bytes(&self) -> u64 {
        ScatterPass::capacity_bytes(self)
    }

    fn stats(
        _this: Option<&Self>,
        _frame: Option<&ScatterFrame>,
        _snapshot: &engine_core::RenderSnapshot,
    ) {
    }
}
