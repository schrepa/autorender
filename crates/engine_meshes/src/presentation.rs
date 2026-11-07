use bevy_ecs::prelude::*;
use engine_core::{
    AlphaMode, AssetLibrary, AssetResidency, Cull, Cursor, DrawnPoses, FrameId,
    MaterialHandle, MeshHandle, MeshResidence, NodeInfo, NodePose, NodeStreams, Readable, Readback,
    ResidentMesh, Since,
};
use glam::{Mat4, Vec3};

use crate::layouts::{BatchData, CandidateData, LOD_LEVELS};
use crate::order::{BatchLook, DrawBatch, MeshBatches};
use crate::uploads::{ColorData, PoseData, UploadRange, listed_ranges};

pub(crate) const MIN_STORAGE_BYTES: u64 = 256;
pub(crate) const TOTALS_BYTES: u64 = 56;
pub(crate) const WATCHED: u32 = 2;
pub(crate) const POSE: u32 = 3;
pub(crate) const VIEWS: u32 = 4;
const STREAM_STRIDE: usize = size_of::<engine_core::PoseRows>();
const NODE_BYTES: usize =
    2 * size_of::<NodePose>() + size_of::<NodeInfo>() + size_of::<engine_core::PoseRows>();
pub(crate) const FADE_MARGIN: f32 = 0.1;
pub(crate) const PHASE_ONE: usize = 0;
pub(crate) const PHASE_TWO: usize = 1;

pub(crate) fn storage_bytes(bytes: u64) -> u64 {
    bytes.max(MIN_STORAGE_BYTES)
}

pub(crate) fn grow(needed: usize) -> usize {
    needed.max(1_024).next_power_of_two()
}

pub(crate) fn frustum_uniform(view_projection: Mat4) -> ([[f32; 4]; 6], [[f32; 4]; 4]) {
    let rows = view_projection.transpose();
    let planes = [
        rows.w_axis + rows.x_axis,
        rows.w_axis - rows.x_axis,
        rows.w_axis + rows.y_axis,
        rows.w_axis - rows.y_axis,
        rows.z_axis,
        rows.w_axis - rows.z_axis,
    ]
    .map(|plane| plane.to_array());
    let absolute = Mat4::from_cols(
        view_projection.x_axis.abs(),
        view_projection.y_axis.abs(),
        view_projection.z_axis.abs(),
        view_projection.w_axis.abs(),
    );
    (planes, absolute.to_cols_array_2d())
}

fn jittered(view_projection: Mat4, jitter: [f32; 2]) -> Mat4 {
    Mat4::from_cols(
        glam::Vec4::X,
        glam::Vec4::Y,
        glam::Vec4::Z,
        glam::Vec4::new(jitter[0], jitter[1], 0.0, 1.0),
    ) * view_projection
}

pub(crate) fn rasterized(view: &engine_core::PrimaryView) -> Mat4 {
    let view_projection = view.view_projection();
    let Some(offset) = view.jitter() else {
        return view_projection;
    };
    let size = view.render_size();
    jittered(
        view_projection,
        engine_render::jitter_clip(offset.to_array(), size.x, size.y),
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameTotals {
    pub visible: u32,
    pub occluded: u32,
    pub subpixel: u32,
    pub levels: [u32; 4],
    pub triangles: u32,
    pub watched: Option<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ViewTotals {
    pub drawn: u32,
    pub refused: u32,
}

#[derive(Resource, Default)]
pub(crate) struct PresentationTotals {
    last: Option<Readback<FrameTotals>>,
    watching: Vec<(FrameId, u32)>,
    pose: Option<Readback<PoseData>>,
    views: Option<Readback<ViewTotals>>,
}

impl PresentationTotals {
    pub(crate) fn last(&self) -> Option<FrameTotals> {
        self.last.map(|read| read.value)
    }

    pub(crate) fn pose(&self) -> Option<PoseData> {
        let pose = self.pose?;
        (self.last?.written == pose.written).then_some(pose.value)
    }

    pub(crate) fn age(&self, now: u64) -> Option<u64> {
        self.last.map(|read| now.saturating_sub(read.written.0))
    }

    pub(crate) fn views(&self) -> Option<ViewTotals> {
        self.views.map(|read| read.value)
    }
}

fn word(bytes: &[u8], index: usize) -> u32 {
    bytes.get(index * 4..index * 4 + 4).map_or(0, |word| {
        u32::from_le_bytes([word[0], word[1], word[2], word[3]])
    })
}

impl Readable for PresentationTotals {
    fn arrived(&mut self, tag: u32, read: Readback<&[u8]>) {
        let (written, bytes) = (read.written, read.value);
        if tag == WATCHED {
            self.watching.push((written, word(bytes, 0)));
            return;
        }
        if tag == POSE {
            self.pose = bytes
                .get(..size_of::<PoseData>())
                .and_then(|bytes| bytemuck::try_pod_read_unaligned::<PoseData>(bytes).ok())
                .map(|value| Readback { value, written });
            return;
        }
        if tag == VIEWS {
            let value = ViewTotals {
                drawn: word(bytes, 0),
                refused: word(bytes, 1),
            };
            self.views = Some(Readback { value, written });
            return;
        }
        let watched = self
            .watching
            .iter()
            .find(|&&(asked, _)| asked == written)
            .map(|&(_, slot)| (slot, word(bytes, 13)));
        self.watching.retain(|&(asked, _)| asked > written);
        let (hidden, tiny) = (word(bytes, 2), word(bytes, 3));
        let value = FrameTotals {
            visible: word(bytes, 0)
                .saturating_add(word(bytes, 1))
                .saturating_add(hidden)
                .saturating_add(tiny),
            occluded: hidden,
            subpixel: tiny,
            levels: std::array::from_fn(|level| {
                word(bytes, 4 + level).saturating_add(word(bytes, 8 + level))
            }),
            triangles: word(bytes, 12),
            watched,
        };
        self.last = Some(Readback { value, written });
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Kept {
    pub phase_two: bool,
    pub impostors: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewSlot {
    pub pass: &'static str,
    pub index: u32,
    pub frame_at: u32,
    pub ids_base: u64,
    pub args_base: u64,
    pub detailed: bool,
    pub mirrored: bool,
}

impl ViewSlot {
    pub(crate) fn levels(&self, drawn: u8) -> impl Iterator<Item = usize> {
        let detailed = self.detailed;
        (0..LOD_LEVELS as usize)
            .filter(move |&level| drawn & (1 << level) != 0 && (detailed || level == 0))
    }
}

pub(crate) struct UploadSources<'a> {
    pub assets: &'a AssetLibrary,
    pub residency: &'a AssetResidency,
    pub hierarchy: &'a engine_core::GlobalTransforms,
    pub impostors: &'a [engine_core::ImpostorRecord],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DrawItem {
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub start: u32,
    pub batch: u32,
    pub look: BatchLook,
    pub levels: u8,
    pub blended: bool,
    flat: bool,
    variant: u32,
    buffers: u8,
}

#[derive(Default)]
pub(crate) struct Draws {
    pub items: Vec<DrawItem>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BlendedBatch {
    pub batch: u32,
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub look: BatchLook,
    pub args: u64,
    pub ids: u64,
}

#[derive(Default)]
pub(crate) struct Blended {
    pub batches: Vec<BlendedBatch>,
    pub rows: Vec<[u32; 4]>,
    pub generation: Option<u64>,
    pub uploaded: bool,
}

impl Blended {
    pub(crate) fn plan(&mut self, items: &[DrawItem], batches: &MeshBatches) -> bool {
        let mut listed: Vec<BlendedBatch> = items
            .iter()
            .filter(|item| item.blended)
            .map(|item| BlendedBatch {
                batch: item.batch,
                mesh: item.mesh,
                material: item.material,
                look: item.look,
                args: 0,
                ids: 0,
            })
            .collect();
        listed.sort_unstable_by_key(|batch| batch.batch);
        let mut base = 0_u32;
        for (entry, listed) in listed.iter_mut().enumerate() {
            listed.args = entry as u64 * crate::layouts::DRAW_ARGS_BYTES;
            listed.ids = u64::from(base) * 4;
            base += batches
                .batches
                .get(listed.batch as usize)
                .map_or(0, |batch| batch.instances.end - batch.instances.start);
        }
        let generation = Some(batches.generation());
        if self.generation == generation && self.batches == listed {
            return false;
        }
        self.generation = generation;
        self.batches = listed;
        self.rows.clear();
        for (entry, listed) in self.batches.iter().enumerate() {
            let Some(batch) = batches.batches.get(listed.batch as usize) else {
                continue;
            };
            let base = (listed.ids / 4) as u32;
            let range = batch.instances.start as usize..batch.instances.end as usize;
            self.rows.extend(
                batches.ordered[range]
                    .iter()
                    .map(|&slot| [slot, listed.batch, base, entry as u32]),
            );
        }
        self.uploaded = false;
        true
    }
}

fn drawn_levels(mesh: Option<&ResidentMesh>) -> u8 {
    let plan = mesh.map_or(&[][..], ResidentMesh::plan);
    let lods = plan.iter().take(LOD_LEVELS as usize - 1);
    lods.enumerate()
        .filter(|(_, lod)| lod.1 != 0)
        .fold(1, |levels, (lod, _)| levels | (1 << (lod + 1)))
}

impl Draws {
    pub(crate) fn plan(
        &mut self,
        batches: &[DrawBatch],
        assets: &AssetLibrary,
        residency: &AssetResidency,
    ) {
        self.items.clear();
        self.items
            .extend(batches.iter().zip(0_u32..).map(|(batch, index)| {
                let material = batch.material.and_then(|handle| assets.material(handle));
                let mesh = assets.mesh(batch.mesh);
                let resident = residency.mesh(batch.mesh);
                let extended =
                    mesh.is_some_and(|mesh| mesh.colors().is_some() || mesh.uv1().is_some());
                let flat = mesh.is_some_and(|mesh| {
                    let bounds = mesh.bounds();
                    let size = bounds.max() - bounds.min();
                    size.y < 0.1 * engine_core::math::max(size.x, size.z)
                });
                DrawItem {
                    mesh: batch.mesh,
                    material: batch.material,
                    start: batch.instances.start,
                    batch: index,
                    look: BatchLook {
                        masked: material.is_some_and(|material| {
                            matches!(material.alpha_mode(), AlphaMode::Mask { .. })
                        }),
                        double_sided: material.is_some_and(|material| material.cull() == Cull::None),
                        extended,
                    },
                    levels: drawn_levels(resident),
                    blended: material.is_some_and(|material| material.alpha_mode().blended()),
                    flat,
                    variant: engine_render::material_state(material),
                    buffers: resident.map_or(u8::MAX, |resident| match resident.placement() {
                        MeshResidence::Arena(placement) => placement.streams as u8,
                        MeshResidence::Own => u8::MAX,
                    }),
                }
            }));
        self.items.sort_unstable_by_key(|item| {
            (
                item.flat,
                item.variant,
                item.material,
                item.buffers,
                item.batch,
            )
        });
    }
}

#[derive(Default)]
pub(crate) struct NodeInputs {
    pub capacity: usize,
    pub composed_capacity: usize,
    pub parent_capacity: usize,
    pub level_ranges: Vec<(u32, u32)>,
    pub seen: Cursor,
    pub whole: bool,
    pub ranges: Vec<UploadRange>,
    pub upload_bits: Vec<u64>,
}

impl NodeInputs {
    pub(crate) fn plan(&mut self, poses: &DrawnPoses, nodes: usize) -> Option<bool> {
        let changes = poses.changes_since(&mut self.seen);
        let whole = std::mem::take(&mut self.whole);
        self.ranges.clear();
        match changes {
            Since::Same if !whole => return None,
            Since::Listed(listed)
                if !whole && listed.iter().all(|&node| (node as usize) < nodes) =>
            {
                listed_ranges(
                    listed,
                    nodes,
                    STREAM_STRIDE,
                    &mut self.upload_bits,
                    &mut self.ranges,
                );
                return Some(false);
            }
            Since::Same | Since::Listed(_) | Since::All => {}
        }
        if nodes != 0 {
            self.ranges.push(UploadRange {
                start: 0,
                end: nodes,
            });
        }
        Some(true)
    }

    pub(crate) fn planned_bytes(&self) -> u64 {
        self.ranges
            .iter()
            .map(|range| (range.end - range.start) * NODE_BYTES)
            .sum::<usize>() as u64
    }

    pub(crate) fn levels(&mut self, streams: &NodeStreams<'_>) {
        self.level_ranges.clear();
        let mut start = 0;
        for &end in streams.levels {
            self.level_ranges.push((start, end));
            start = end;
        }
    }
}

#[derive(Default)]
pub(crate) struct DrawGroups {
    pub capacity: usize,
    pub batch_capacity: usize,
    pub count: u32,
    pub batch_count: u32,
    pub kept: Kept,
    pub key: Option<(u64, u64)>,
    pub first_instance: bool,
    pub candidate_staging: Vec<CandidateData>,
    pub slot_staging: Vec<CandidateData>,
    pub slot_positions: Vec<u32>,
    pub batch_staging: Vec<BatchData>,
}

impl DrawGroups {
    pub(crate) fn stage_batches(
        &mut self,
        batches: &[DrawBatch],
        (assets, residency): (&AssetLibrary, &AssetResidency),
        impostor: impl Fn(usize) -> f32,
        reach: impl Fn(Option<MaterialHandle>) -> f32,
    ) {
        let planes = (self.first_instance, self.capacity);
        self.batch_staging.clear();
        self.batch_staging
            .extend(batches.iter().enumerate().map(|(index, batch)| {
                let grown = Vec3::splat(reach(batch.material));
                let bounds = assets
                    .mesh(batch.mesh)
                    .map_or((Vec3::ZERO, Vec3::ZERO), |mesh| {
                        let bounds = mesh.bounds();
                        (bounds.min() - grown, bounds.max() + grown)
                    });
                batch_record(
                    batch,
                    (residency.mesh(batch.mesh), bounds),
                    impostor(index),
                    planes,
                )
            }));
        self.batch_count = batches.len() as u32;
    }

    pub(crate) fn stage_candidates(
        &mut self,
        batches: &[DrawBatch],
        ordered: &[u32],
        node: impl Fn(usize) -> u32,
    ) {
        let candidates = ordered.len();
        self.candidate_staging.clear();
        self.candidate_staging.reserve_exact(candidates);
        for (index, batch) in batches.iter().enumerate() {
            let range = batch.instances.start as usize..batch.instances.end as usize;
            let base = self.candidate_staging.len() as u32;
            self.candidate_staging
                .extend(
                    ordered[range]
                        .iter()
                        .zip(base..)
                        .map(|(&slot, position)| CandidateData {
                            slot,
                            node: node(slot as usize),
                            batch: index as u32,
                            position,
                        }),
                );
        }
        let slots = self
            .candidate_staging
            .iter()
            .map(|candidate| candidate.slot as usize + 1)
            .max()
            .unwrap_or(0);
        if self.slot_positions.len() < slots {
            self.slot_positions.resize(slots, u32::MAX);
        }
        for candidate in &self.candidate_staging {
            self.slot_positions[candidate.slot as usize] = candidate.position;
        }
        self.slot_staging.clear();
        self.slot_staging.reserve_exact(candidates);
        for position in &mut self.slot_positions[..slots] {
            if *position != u32::MAX {
                self.slot_staging
                    .push(self.candidate_staging[*position as usize]);
                *position = u32::MAX;
            }
        }
        self.count = candidates as u32;
    }

    pub(crate) fn draw_source(&self, start: u32, index: usize, level: usize) -> (u64, u64) {
        let args = (level * self.batch_count as usize + index) as u64;
        let ids = if self.first_instance {
            0
        } else {
            ((level * self.capacity) as u64 + u64::from(start)) * 4
        };
        (ids, args * crate::layouts::DRAW_ARGS_BYTES)
    }

    pub(crate) fn view_source(&self, slot: &ViewSlot, item: &DrawItem, level: usize) -> (u64, u64) {
        let args = slot.args_base + (level * self.batch_count as usize + item.batch as usize) as u64;
        let ids = if self.first_instance {
            0
        } else {
            (slot.ids_base + (level * self.capacity) as u64 + u64::from(item.start)) * 4
        };
        (ids, args * crate::layouts::DRAW_ARGS_BYTES)
    }
}

fn batch_record(
    batch: &DrawBatch,
    (mesh, bounds): (Option<&ResidentMesh>, (Vec3, Vec3)),
    impostor: f32,
    (first_instance, capacity): (bool, usize),
) -> BatchData {
    let first_index = mesh.map_or(0, ResidentMesh::first_index);
    let plan = mesh.map_or(&[][..], ResidentMesh::plan);
    let lods = plan.len().min(LOD_LEVELS as usize - 1);
    let lod = |level: usize| {
        plan[..lods]
            .get(level)
            .map_or((0, 0, 0.0), |&(offset, count, error)| {
                (first_index + offset, count, error)
            })
    };
    BatchData {
        start: batch.instances.start,
        count: batch.instances.end - batch.instances.start,
        index_count: mesh.map_or(0, ResidentMesh::index_count),
        first_index,
        base_vertex: mesh.map_or(0, ResidentMesh::base_vertex),
        first_instance: if first_instance {
            batch.instances.start
        } else {
            0
        },
        first_plane: if first_instance { capacity as u32 } else { 0 },
        lod_ranges: [lod(0).0, lod(0).1, lod(1).0, lod(1).1],
        lod_errors: [lod(0).2, lod(1).2, (1 + lods) as f32, impostor],
        pad: 0,
        bounds_min: bounds.0.extend(0.0).to_array(),
        bounds_max: bounds.1.extend(0.0).to_array(),
    }
}

#[derive(Default)]
pub(crate) struct Pyramid {
    pub count: u32,
    pub written: Vec<(u32, u32)>,
}

#[derive(Default)]
pub(crate) struct FrameState {
    pub capacity: usize,
    pub poses_capacity: usize,
    pub batches: MeshBatches,
    pub color_staging: Vec<ColorData>,
    pub color_ranges: Vec<UploadRange>,
    pub color_bits: Vec<u64>,
    pub colors_seen: Cursor,
    pub colors_library: Option<(u64, u64)>,
    pub motion_generation: Option<u64>,
    pub prepared_at: Option<f32>,
    pub faded_at: Option<f32>,
    pub cards: bool,
    pub draws: Draws,
    pub blended: Blended,
    pub nodes: NodeInputs,
    pub groups: DrawGroups,
    pub fade_slots: u64,
    pub card_capacity: usize,
    pub frame: Option<crate::layouts::PresentationFrame>,
    pub candidates: u32,
    pub occlusion: bool,
    pub lod: bool,
    pub pyramid: Pyramid,
    pub views: Vec<ViewSlot>,
    pub instance_count: usize,
    pub watched: Option<u32>,
}

impl FrameState {
    pub(crate) fn view_slot(&self, pass: &str, view: Option<u32>) -> Option<&ViewSlot> {
        let view = view?;
        self.views
            .iter()
            .find(|slot| slot.pass == pass && slot.index == view)
    }
}
