use bevy_ecs::prelude::*;
use engine_core::{
    AssetLibrary, AssetResidency, Cursor, FrameId, GeometryHandle, MaterialHandle, MeshResidence,
    Readable, Readback, ResidentMesh,
};
use engine_places::meshes::LastLevel;
use engine_render::UploadRange;
use glam::Vec3;

use crate::layouts::{BatchData, CandidateData, LOD_LEVELS};
use crate::order::{BatchLook, DrawBatch, MeshBatches};
use crate::uploads::{ColorData, PoseData};

pub(crate) const MIN_STORAGE_BYTES: u64 = 256;
pub(crate) const TOTALS_BYTES: u64 = 104;
pub(crate) const WATCHED: u32 = 2;
pub(crate) const VIEWS: u32 = 4;
const POSE_WORD: usize = 14;
pub(crate) const PHASE_ONE: usize = 0;
pub(crate) const PHASE_TWO: usize = 1;

pub(crate) fn storage_bytes(bytes: u64) -> u64 {
    bytes.max(MIN_STORAGE_BYTES)
}

pub(crate) fn grow(needed: usize) -> usize {
    needed.max(1_024).next_power_of_two()
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
        self.pose = watched
            .and_then(|_| bytes.get(POSE_WORD * 4..POSE_WORD * 4 + size_of::<PoseData>()))
            .and_then(|bytes| bytemuck::try_pod_read_unaligned::<PoseData>(bytes).ok())
            .map(|value| Readback { value, written });
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
    pub last: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewSlot {
    pub pass: &'static str,
    pub index: u32,
    pub frame_at: u32,
    pub ids_base: u64,
    pub args_base: u64,
    pub detailed: bool,
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
    pub last: &'a [LastLevel],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DrawItem {
    pub mesh: GeometryHandle,
    pub material: Option<MaterialHandle>,
    pub start: u32,
    pub batch: u32,
    pub look: BatchLook,
    pub levels: u8,
    pub blended: bool,
    flat: bool,
    variant: u32,
    buffers: u32,
}

#[derive(Default)]
pub(crate) struct Draws {
    pub items: Vec<DrawItem>,
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
                let extended = batch.skinned
                    || mesh.is_some_and(|mesh| mesh.colors().is_some() || mesh.uv1().is_some());
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
                        extended,
                        skinned: batch.skinned,
                    },
                    levels: drawn_levels(resident),
                    blended: material.is_some_and(|material| material.alpha_mode().blended()),
                    flat,
                    variant: engine_render::material_state(material),
                    buffers: resident.map_or(u32::MAX, |resident| match resident.placement() {
                        MeshResidence::Arena(placement) => placement.class,
                        MeshResidence::Own => u32::MAX,
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
        last: impl Fn(usize) -> LastLevel,
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
                    last(index),
                    planes,
                )
            }));
        self.batch_count = batches.len() as u32;
    }

    pub(crate) fn stage_candidates(&mut self, batches: &[DrawBatch], ordered: &[u32]) {
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
                            batch: index as u32,
                            position,
                            pad: 0,
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
        let args =
            slot.args_base + (level * self.batch_count as usize + item.batch as usize) as u64;
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
    last: LastLevel,
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
        last_triangles: last.triangles,
        first_index,
        base_vertex: mesh.map_or(0, ResidentMesh::base_vertex),
        first_instance: if first_instance {
            batch.instances.start
        } else {
            0
        },
        first_plane: if first_instance { capacity as u32 } else { 0 },
        lod_ranges: [lod(0).0, lod(0).1, lod(1).0, lod(1).1],
        lod_errors: [lod(0).2, lod(1).2, (1 + lods) as f32, last.error],
        bounds_min: bounds.0.extend(0.0).to_array(),
        bounds_max: bounds.1.extend(0.0).to_array(),
    }
}

#[derive(Default)]
pub(crate) struct FrameState {
    pub capacity: usize,
    pub colors_capacity: usize,
    pub batches: MeshBatches,
    pub color_staging: Vec<ColorData>,
    pub color_ranges: Vec<UploadRange>,
    pub color_bits: Vec<u64>,
    pub colors_seen: Cursor,
    pub colors_library: Option<(u64, u64)>,
    pub prepared_at: Option<f32>,
    pub faded_at: Option<f32>,
    pub draws: Draws,
    pub groups: DrawGroups,
    pub fade_slots: u64,
    pub fading_capacity: usize,
    pub frame: Option<crate::layouts::PresentationFrame>,
    pub candidates: u32,
    pub occlusion: bool,
    pub lod: bool,
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
