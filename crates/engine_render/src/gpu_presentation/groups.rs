//! Draw groups of GPU presentation: the plan's candidates in
//! draw-group order and in slot order, each batch's record, skinned slots,
//! and where each compaction writes its draws. They upload
//! only when the candidate structure, the node space or what batch records
//! carry changed.

use crate::batches::DrawBatch;
use crate::gpu_data::{BatchData, CandidateData, DRAW_ARGS_BYTES, LOD_LEVELS, SkinnedData};
use crate::mesh_arena::Ids;

use super::{BatchMeta, PresentationInputs, RecordsKey, SCAN_GROUP, grow, storage};

/// The compactions writing draws: drawing phase one (the only phase
/// without occlusion culling), phase two, and reflection
/// probe faces.
pub(super) const PHASE_ONE: usize = 0;
pub(super) const PHASE_TWO: usize = 1;
pub(super) const PROBE: usize = 2;

// The frame identity, structure and source revisions and gather identity
// the groups were built for, and what batch records carry.
type GroupsKey = (u64, u64, u64, u64, RecordsKey);

pub(super) struct DrawGroups {
    capacity: usize,
    pub(super) candidates: wgpu::Buffer,
    // The same candidates in slot order, for culling.
    pub(super) slot_candidates: wgpu::Buffer,
    // Candidate flags, then the compactions' group sums and totals.
    pub(super) scan: wgpu::Buffer,
    batch_capacity: usize,
    pub(super) batches: wgpu::Buffer,
    // Each batch's visible count before it, per level of detail.
    pub(super) batch_bases: wgpu::Buffer,
    // Each compaction's draw arguments (per level of detail and batch, then
    // the impostors') and visible IDs (a plane per level of detail).
    pub(super) draws: [wgpu::Buffer; 3],
    pub(super) visible_ids: [wgpu::Buffer; 3],
    // Each drawing phase's impostor instances; probe faces
    // write phase one's once the frame drew them.
    pub(super) impostor_ids: [wgpu::Buffer; 2],
    // Skinned slots among the candidates and their world
    // bounds from this frame's palettes, with each batch's joint count.
    skinned_capacity: usize,
    pub(super) skinned: wgpu::Buffer,
    pub(super) skin_bounds: wgpu::Buffer,
    skinned_count: u32,
    batch_joints: Vec<u32>,
    count: u32,
    batch_count: u32,
    key: Option<GroupsKey>,
    // Whether draws' arguments name their visible-ID regions as first
    // instances (`indirect-first-instance`).
    first_instance: bool,
    candidate_staging: Vec<CandidateData>,
    slot_staging: Vec<CandidateData>,
    // Per slot: its candidate's draw-group position (`u32::MAX`: none).
    slot_positions: Vec<u32>,
    batch_staging: Vec<BatchData>,
    skinned_staging: Vec<SkinnedData>,
    // Advances when any of this storage is replaced.
    generation: u64,
}

impl DrawGroups {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let empty = |label, usage| storage(device, label, 0, usage);
        let none = wgpu::BufferUsages::empty();
        let (indirect, vertex) = (wgpu::BufferUsages::INDIRECT, wgpu::BufferUsages::VERTEX);
        Self {
            capacity: 0,
            candidates: empty("engine/presentation-candidates", none),
            slot_candidates: empty("engine/presentation-slot-candidates", none),
            scan: empty("engine/presentation-scan", none),
            batch_capacity: 0,
            batches: empty("engine/presentation-batches", none),
            batch_bases: empty("engine/presentation-batch-bases", none),
            draws: [
                empty("engine/presentation-draws", indirect),
                empty("engine/presentation-draws2", indirect),
                empty("engine/presentation-draws-probe", indirect),
            ],
            visible_ids: [
                empty("engine/presentation-visible-ids", vertex),
                empty("engine/presentation-visible-ids2", vertex),
                empty("engine/presentation-visible-ids-probe", vertex),
            ],
            impostor_ids: [
                empty("engine/presentation-impostor-ids", vertex),
                empty("engine/presentation-impostor-ids2", vertex),
            ],
            skinned_capacity: 0,
            skinned: empty("engine/presentation-skinned", none),
            skin_bounds: empty("engine/presentation-skin-bounds", none),
            skinned_count: 0,
            batch_joints: Vec::new(),
            count: 0,
            batch_count: 0,
            key: None,
            first_instance: device
                .features()
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
            candidate_staging: Vec::new(),
            slot_staging: Vec::new(),
            slot_positions: Vec::new(),
            batch_staging: Vec::new(),
            skinned_staging: Vec::new(),
            generation: 0,
        }
    }

    /// Candidates of the groups last built.
    pub(super) fn count(&self) -> u32 {
        self.count
    }

    /// Plan batches of the groups last built.
    pub(super) fn batch_count(&self) -> u32 {
        self.batch_count
    }

    /// Skinned slots among the candidates.
    pub(super) fn skinned_count(&self) -> u32 {
        self.skinned_count
    }

    /// Candidates the storage holds: the size of each visible-ID plane.
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Advances when any of this storage is replaced.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        [
            &self.candidates,
            &self.slot_candidates,
            &self.scan,
            &self.batches,
            &self.batch_bases,
        ]
        .into_iter()
        .chain(&self.draws)
        .chain(&self.visible_ids)
        .chain(&self.impostor_ids)
        .map(wgpu::Buffer::size)
        .sum()
    }

    /// Sizes storage for `candidates` candidates in `batches` batches.
    pub(super) fn ensure(&mut self, device: &wgpu::Device, candidates: usize, batches: usize) {
        if candidates > self.capacity {
            self.capacity = grow(candidates);
            let capacity = self.capacity as u64;
            let groups = capacity.div_ceil(u64::from(SCAN_GROUP));
            self.candidates = storage(
                device,
                "engine/presentation-candidates",
                capacity * size_of::<CandidateData>() as u64,
                wgpu::BufferUsages::empty(),
            );
            self.slot_candidates = storage(
                device,
                "engine/presentation-slot-candidates",
                capacity * size_of::<CandidateData>() as u64,
                wgpu::BufferUsages::empty(),
            );
            // Flags, four sums per group (one per level of detail), four totals and their sum.
            self.scan = storage(
                device,
                "engine/presentation-scan",
                (capacity + 4 * (groups + 1) + 1) * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            // A plane of visible IDs per level of detail.
            let planes = LOD_LEVELS as u64;
            self.visible_ids = [
                "engine/presentation-visible-ids",
                "engine/presentation-visible-ids2",
                "engine/presentation-visible-ids-probe",
            ]
            .map(|label| {
                storage(
                    device,
                    label,
                    planes * capacity * 4,
                    wgpu::BufferUsages::VERTEX,
                )
            });
            // Impostor instances: two words each.
            self.impostor_ids = [
                "engine/presentation-impostor-ids",
                "engine/presentation-impostor-ids2",
            ]
            .map(|label| storage(device, label, capacity * 8, wgpu::BufferUsages::VERTEX));
            self.key = None;
            self.generation += 1;
        }
        if batches > self.batch_capacity {
            self.batch_capacity = batches.max(64).next_power_of_two();
            let capacity = self.batch_capacity as u64;
            self.batches = storage(
                device,
                "engine/presentation-batches",
                capacity * size_of::<BatchData>() as u64,
                wgpu::BufferUsages::empty(),
            );
            self.batch_bases = storage(
                device,
                "engine/presentation-batch-bases",
                capacity * 16,
                wgpu::BufferUsages::empty(),
            );
            // Draw arguments per level of detail and batch, then the
            // impostors'.
            let args = (LOD_LEVELS as u64 * capacity + 1) * DRAW_ARGS_BYTES;
            self.draws = [
                "engine/presentation-draws",
                "engine/presentation-draws2",
                "engine/presentation-draws-probe",
            ]
            .map(|label| storage(device, label, args, wgpu::BufferUsages::INDIRECT));
            self.key = None;
            self.generation += 1;
        }
    }

    /// Rebuilds the groups of `inputs`' plan when its candidate structure,
    /// node space or record choices changed; returns the bytes uploaded.
    pub(super) fn rebuild(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &PresentationInputs<'_>,
    ) -> Option<u64> {
        let PresentationInputs {
            frame,
            gpu,
            plan,
            batch_meta,
            records,
            ..
        } = inputs;
        // Listed materials replaced in place change the structure revision,
        // not the plan's draws.
        let key = (
            frame.identity(),
            plan.generation(),
            frame.source_revision(),
            gpu.identity,
            *records,
        );
        if self.key == Some(key) {
            return None;
        }
        let candidates = plan.ordered_instances.len();
        let slot_nodes = frame.slot_nodes();
        self.candidate_staging.clear();
        self.candidate_staging.reserve_exact(candidates);
        self.batch_staging.clear();
        self.batch_staging.reserve_exact(plan.batches.len());
        self.batch_joints.clear();
        for (index, batch) in plan.batches.iter().enumerate() {
            let meta = batch_meta(index, batch);
            self.batch_joints.push(meta.joints);
            self.batch_staging.push(batch_record(
                batch,
                &meta,
                self.first_instance,
                self.capacity,
            ));
            let range = batch.instances.start as usize..batch.instances.end as usize;
            let base = self.candidate_staging.len() as u32;
            self.candidate_staging
                .extend(plan.ordered_instances[range].iter().zip(base..).map(
                    |(&slot, position)| CandidateData {
                        slot: slot as u32,
                        node: slot_nodes[slot],
                        batch: index as u32,
                        position,
                    },
                ));
        }
        // The same candidates in slot order, placed by slot in linear time.
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
        // Skinned slots, in slot order.
        self.skinned_staging.clear();
        self.skinned_staging
            .extend(self.slot_staging.iter().filter_map(|candidate| {
                let joints = self.batch_joints[candidate.batch as usize];
                (joints != 0).then_some(SkinnedData {
                    slot: candidate.slot,
                    position: candidate.position,
                    batch: candidate.batch,
                    joints,
                })
            }));
        if self.skinned_staging.len() > self.skinned_capacity {
            self.skinned_capacity = grow(self.skinned_staging.len());
            let capacity = self.skinned_capacity as u64;
            self.skinned = storage(
                device,
                "engine/presentation-skinned",
                capacity * size_of::<SkinnedData>() as u64,
                wgpu::BufferUsages::empty(),
            );
            // A minimum and a maximum per skinned slot.
            self.skin_bounds = storage(
                device,
                "engine/presentation-skin-bounds",
                capacity * 32,
                wgpu::BufferUsages::empty(),
            );
            self.generation += 1;
        }
        if !self.skinned_staging.is_empty() {
            queue.write_buffer(
                &self.skinned,
                0,
                bytemuck::cast_slice(&self.skinned_staging),
            );
        }
        self.skinned_count = self.skinned_staging.len() as u32;
        if !self.candidate_staging.is_empty() {
            queue.write_buffer(
                &self.candidates,
                0,
                bytemuck::cast_slice(&self.candidate_staging),
            );
            queue.write_buffer(
                &self.slot_candidates,
                0,
                bytemuck::cast_slice(&self.slot_staging),
            );
        }
        if !self.batch_staging.is_empty() {
            queue.write_buffer(&self.batches, 0, bytemuck::cast_slice(&self.batch_staging));
        }
        self.count = candidates as u32;
        self.batch_count = plan.batches.len() as u32;
        self.key = Some(key);
        Some(
            (2 * self.candidate_staging.len() * size_of::<CandidateData>()
                + self.batch_staging.len() * size_of::<BatchData>()
                + self.skinned_staging.len() * size_of::<SkinnedData>()) as u64,
        )
    }

    /// Byte offset of the compaction's visible total over every level of
    /// detail in the scan buffer, after the per-level totals.
    pub(super) fn total_offset(&self) -> u64 {
        self.level_offset() + 16
    }

    /// Byte offset of the compaction's totals per level of detail in the
    /// scan buffer, after the flags and per-level group sums.
    pub(super) fn level_offset(&self) -> u64 {
        u64::from(self.count + 4 * self.count.div_ceil(SCAN_GROUP)) * 4
    }

    /// Visible IDs and indirect arguments of plan batch `index` in
    /// compaction `compaction` at level of detail `level`.
    pub(super) fn draw_source<'a>(
        &'a self,
        batch: &DrawBatch,
        index: usize,
        compaction: usize,
        level: usize,
    ) -> (Ids<'a>, &'a wgpu::Buffer, u64) {
        let args = (level * self.batch_count as usize + index) as u64;
        (
            self.ids(&self.visible_ids[compaction], batch, level),
            &self.draws[compaction],
            args * DRAW_ARGS_BYTES,
        )
    }

    /// The impostor instances compaction `compaction` writes: probe faces
    /// write phase one's, which the frame drew already.
    pub(super) fn impostor_ids_of(&self, compaction: usize) -> &wgpu::Buffer {
        &self.impostor_ids[if compaction == PHASE_TWO {
            PHASE_TWO
        } else {
            PHASE_ONE
        }]
    }

    /// Drawing phase `phase`'s impostor instances and the
    /// arguments of the non-indexed draw of their cards.
    pub(super) fn impostor_source(&self, phase: usize) -> (&wgpu::Buffer, &wgpu::Buffer, u64) {
        let args = LOD_LEVELS as u64 * u64::from(self.batch_count);
        (
            &self.impostor_ids[phase],
            &self.draws[phase],
            args * DRAW_ARGS_BYTES,
        )
    }

    /// A batch's visible IDs at level of detail `level` in `ids`: the whole
    /// buffer where its draw arguments' first instance names its region,
    /// otherwise the region in the level's plane.
    fn ids<'a>(&self, ids: &'a wgpu::Buffer, batch: &DrawBatch, level: usize) -> Ids<'a> {
        if self.first_instance {
            Ids::Whole(ids)
        } else {
            let plane = (level * self.capacity) as u64;
            let start = (plane + u64::from(batch.instances.start)) * 4;
            let end = (plane + u64::from(batch.instances.end)) * 4;
            Ids::Region(ids.slice(start..end))
        }
    }
}

/// The record of `batch`, described by `meta`, whose draw arguments name
/// its visible-ID regions in planes of `capacity` IDs as first instances
/// when `first_instance`.
fn batch_record(
    batch: &DrawBatch,
    meta: &BatchMeta,
    first_instance: bool,
    capacity: usize,
) -> BatchData {
    let lod = |level: usize| meta.lods.get(level).copied().unwrap_or((0, 0, 0.0));
    let skinned = meta.joints != 0;
    BatchData {
        start: batch.instances.start,
        count: batch.instances.end - batch.instances.start,
        index_count: meta.index_count,
        first_index: meta.first_index,
        base_vertex: meta.base_vertex,
        first_instance: if first_instance {
            batch.instances.start
        } else {
            0
        },
        first_plane: if first_instance { capacity as u32 } else { 0 },
        lod_ranges: [lod(0).0, lod(0).1, lod(1).0, lod(1).1],
        lod_errors: [
            lod(0).2,
            lod(1).2,
            (1 + meta.lods.len()) as f32,
            meta.impostor,
        ],
        pad: meta.group,
        bounds_min: meta
            .bounds
            .min()
            .extend(if skinned { 1.0 } else { 0.0 })
            .to_array(),
        // Transparent groups are never occluded, nor are
        // instances drawing virtual geometry (2): their
        // clusters draw in phase one.
        bounds_max: meta
            .bounds
            .max()
            .extend(if batch.transparent {
                1.0
            } else if meta.virtual_geometry {
                2.0
            } else {
                0.0
            })
            .to_array(),
    }
}
