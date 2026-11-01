use bevy_ecs::prelude::*;
use engine_core::{GpuPresentation, MaterialHandle, MeshHandle, NodeInfo, NodePose, Readback, Stamp};
use glam::Mat4;

use crate::batches::DrawBatch;
use crate::layouts::{BatchData, CandidateData};
use crate::order::{BatchLook, BatchMeta};
use crate::uploads::{
    ColorData, RetainedUploads, UploadRange, changed_ranges, listed_ranges, whole_stream,
};

pub(crate) const MIN_STORAGE_BYTES: u64 = 256;
pub(crate) const TOTALS_BYTES: u64 = 56;
pub(crate) const CASTER_TOTALS_BYTES: u64 = 16;
pub(crate) const WATCHED: u32 = 2;
const STREAM_STRIDES: [usize; 4] = [
    size_of::<NodePose>(),
    size_of::<NodePose>(),
    size_of::<NodeInfo>(),
    size_of::<engine_core::PoseRows>(),
];
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

fn jitter_clip(offset: [f32; 2], width: u32, height: u32) -> [f32; 2] {
    [
        2.0 * offset[0] / width.max(1) as f32,
        -2.0 * offset[1] / height.max(1) as f32,
    ]
}

fn jittered(view_projection: Mat4, jitter: [f32; 2]) -> Mat4 {
    Mat4::from_cols(
        glam::Vec4::X,
        glam::Vec4::Y,
        glam::Vec4::Z,
        glam::Vec4::new(jitter[0], jitter[1], 0.0, 1.0),
    ) * view_projection
}

pub(crate) fn rasterized(view: &engine_core::View, offset: glam::Vec2, temporal: bool) -> Mat4 {
    let view_projection = view.view_projection();
    if !temporal {
        return view_projection;
    }
    let size = view.render_size();
    jittered(view_projection, jitter_clip(offset.to_array(), size.x, size.y))
}

pub(crate) fn mip_levels(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

pub(crate) fn pyramid_extents(drawn: (u32, u32), count: u32) -> Vec<[u32; 4]> {
    let mut source = (drawn.0.max(1), drawn.1.max(1));
    (0..count)
        .map(|_| {
            let written = (source.0.div_ceil(2), source.1.div_ceil(2));
            let extent = [written.0, written.1, source.0 - 1, source.1 - 1];
            source = written;
            extent
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameTotals {
    pub visible: u32,
    pub occluded: u32,
    pub subpixel: u32,
    pub levels: [u32; 4],
    pub triangles: u32,
    pub watched: Option<(u32, u32)>,
    pub frame: u64,
}

#[derive(Resource, Default)]
pub(crate) struct PresentationTotals {
    last: Option<FrameTotals>,
    watching: Vec<(u64, u32)>,
}

impl PresentationTotals {
    pub(crate) fn last(&self) -> Option<FrameTotals> {
        self.last
    }

    pub(crate) fn age(&self, now: u64) -> Option<u64> {
        self.last.map(|totals| now.saturating_sub(totals.frame))
    }
}

fn word(bytes: &[u8], index: usize) -> u32 {
    bytes
        .get(index * 4..index * 4 + 4)
        .map_or(0, |word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
}

impl Readback for PresentationTotals {
    fn arrived(&mut self, frame: u64, tag: u32, bytes: &[u8]) {
        if tag == WATCHED {
            self.watching.push((frame, word(bytes, 0)));
            return;
        }
        let watched = self
            .watching
            .iter()
            .find(|&&(asked, _)| asked == frame)
            .map(|&(_, slot)| (slot, word(bytes, 13)));
        self.watching.retain(|&(asked, _)| asked > frame);
        let (hidden, tiny) = (word(bytes, 2), word(bytes, 3));
        self.last = Some(FrameTotals {
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
            frame,
        });
    }
}

#[derive(Resource, Default)]
pub(crate) struct CasterTotals {
    last: Option<[u32; 2]>,
}

impl CasterTotals {
    pub(crate) fn last(&self) -> Option<[u32; 2]> {
        self.last
    }
}

impl Readback for CasterTotals {
    fn arrived(&mut self, _frame: u64, _tag: u32, bytes: &[u8]) {
        self.last = Some([word(bytes, 0), word(bytes, 1)]);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Kept {
    pub phase_two: bool,
    pub probe_faces: bool,
    pub impostors: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DrawnBatch {
    pub mesh: MeshHandle,
    pub material: Option<MaterialHandle>,
    pub instances: std::ops::Range<u32>,
}

impl DrawnBatch {
    pub(crate) fn of(batch: &DrawBatch) -> Self {
        Self {
            mesh: batch.mesh,
            material: batch.material,
            instances: batch.instances.clone(),
        }
    }
}

#[derive(Default)]
pub(crate) struct Draws {
    pub generation: Option<(u64, u64)>,
    pub batches: Vec<DrawnBatch>,
    pub order: Vec<u32>,
    pub masked: Vec<u32>,
    pub looks: Vec<BatchLook>,
    pub metas: Vec<BatchMeta>,
}

#[derive(Default)]
pub(crate) struct NodeInputs {
    pub capacity: usize,
    pub composed_capacity: usize,
    pub parent_capacity: usize,
    pub level_ranges: Vec<(u32, u32)>,
    pub blend: bool,
    pub uploaded: Option<(u64, u64)>,
    pub ranges: [Vec<UploadRange>; 4],
    pub upload_bits: Vec<u64>,
    pub staged: Vec<f32>,
    pub rows: Vec<[u32; 4]>,
    pub shift: bool,
}

impl NodeInputs {
    pub(crate) fn holds(&self, gpu: &GpuPresentation<'_>) -> bool {
        gpu.hold
            && self
                .uploaded
                .is_some_and(|(identity, _)| identity == gpu.identity)
    }

    pub(crate) fn gather(&mut self, gpu: &GpuPresentation<'_>, held: bool) -> Option<(Option<u64>, bool)> {
        let since = match self.uploaded {
            Some((identity, sequence)) if identity == gpu.identity => Some(sequence),
            _ => None,
        };
        if held || since == Some(gpu.sequence) {
            return None;
        }
        let mut shift = false;
        match since {
            None => self.blend = true,
            Some(since) if gpu.shift_sequence > since => {
                shift = gpu.shift && since + 1 == gpu.sequence;
                self.blend = shift;
            }
            Some(_) => {}
        }
        Some((since, shift))
    }

    pub(crate) fn planned_bytes(&self) -> u64 {
        self.ranges
            .iter()
            .zip(STREAM_STRIDES)
            .map(|(ranges, stride)| {
                ranges
                    .iter()
                    .map(|range| (range.end - range.start) * stride)
                    .sum::<usize>()
            })
            .sum::<usize>() as u64
    }

    pub(crate) fn plan_ranges(&mut self, gpu: &GpuPresentation<'_>, since: Option<u64>) {
        let strides = STREAM_STRIDES;
        let stamps = [
            gpu.current_stamps,
            gpu.previous_stamps,
            gpu.info_stamps,
            gpu.authority_stamps,
        ];
        let consecutive = since.is_some_and(|since| since + 1 == gpu.sequence);
        let bits = &mut self.upload_bits;
        for (stream, ranges) in self.ranges.iter_mut().enumerate() {
            let (stride, stamps) = (strides[stream], stamps[stream]);
            let stamped = gpu.stamped[stream];
            let listed = gpu.stamped_nodes[stream];
            match since {
                Some(_) if consecutive && stamped == 0 => ranges.clear(),
                Some(since) if !consecutive || !whole_stream(stamped, stamps.len()) => {
                    match listed.filter(|_| consecutive) {
                        Some(nodes) => {
                            listed_ranges(nodes, stamps.len(), stride, bits, ranges);
                        }
                        None => {
                            changed_ranges(stamps, since, stride, ranges);
                        }
                    }
                }
                _ => {
                    ranges.clear();
                    if !stamps.is_empty() {
                        ranges.push(UploadRange {
                            start: 0,
                            end: stamps.len(),
                        });
                    }
                }
            }
        }
    }

    pub(crate) fn stage_locals(&mut self, gpu: &GpuPresentation<'_>) {
        self.staged.clear();
        self.rows.clear();
        let floats = size_of::<NodePose>() / size_of::<f32>();
        let streams: [(&[NodePose], usize); 2] = [(gpu.current, 0), (gpu.previous, self.capacity)];
        for ((poses, base), ranges) in streams.into_iter().zip(&self.ranges[..2]) {
            for range in ranges {
                let staging = self.staged.len() as u32;
                for pose in &poses[range.start..range.end] {
                    self.staged.extend_from_slice(pose);
                }
                self.rows.push([
                    ((base + range.start) * floats) as u32,
                    ((range.end - range.start) * floats) as u32,
                    staging,
                    0,
                ]);
            }
        }
    }

    pub(crate) fn levels(&mut self, gpu: &GpuPresentation<'_>) {
        self.level_ranges.clear();
        let mut start = 0;
        for &end in gpu.levels {
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
    pub key: Option<(Stamp, u64, u64, u64, u64, u64, u64)>,
    pub first_instance: bool,
    pub candidate_staging: Vec<CandidateData>,
    pub slot_staging: Vec<CandidateData>,
    pub slot_positions: Vec<u32>,
    pub batch_staging: Vec<BatchData>,
}

impl DrawGroups {
    pub(crate) fn stage(
        &mut self,
        batches: &[DrawBatch],
        ordered: &[usize],
        slot_nodes: &[u32],
        metas: &[BatchMeta],
        (groups, impostor): (impl Fn(usize) -> u32, impl Fn(usize) -> f32),
    ) {
        let candidates = ordered.len();
        self.candidate_staging.clear();
        self.candidate_staging.reserve_exact(candidates);
        self.batch_staging.clear();
        self.batch_staging.reserve_exact(batches.len());
        for (index, batch) in batches.iter().enumerate() {
            let meta = &metas[index];
            self.batch_staging.push(batch_record(
                batch,
                meta,
                (groups(index), impostor(index)),
                (self.first_instance, self.capacity),
            ));
            let range = batch.instances.start as usize..batch.instances.end as usize;
            let base = self.candidate_staging.len() as u32;
            self.candidate_staging
                .extend(ordered[range].iter().zip(base..).map(|(&slot, position)| {
                    CandidateData {
                        slot: slot as u32,
                        node: slot_nodes[slot],
                        batch: index as u32,
                        position,
                    }
                }));
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
        self.batch_count = batches.len() as u32;
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
}

fn batch_record(
    batch: &DrawBatch,
    meta: &BatchMeta,
    (group, impostor): (u32, f32),
    (first_instance, capacity): (bool, usize),
) -> BatchData {
    let lod = |level: usize| meta.lods.get(level).copied().unwrap_or((0, 0, 0.0));
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
            impostor,
        ],
        group,
        bounds_min: meta.bounds.0.extend(0.0).to_array(),
        bounds_max: meta.bounds.1.extend(0.0).to_array(),
    }
}

#[derive(Default)]
pub(crate) struct Stillness {
    pub slots: u64,
    pub counting: bool,
    pub counted: u32,
}

impl Stillness {
    pub(crate) fn kept(&self) -> bool {
        self.slots != 0
    }
}

#[derive(Default)]
pub(crate) struct Pyramid {
    pub base: (u32, u32),
    pub count: u32,
    pub drawn: (u32, u32),
    pub written: Vec<(u32, u32)>,
}

#[derive(Default)]
pub(crate) struct CasterDraws {
    pub groups: Vec<(MeshHandle, Option<MaterialHandle>, u32)>,
    pub masked: Vec<bool>,
    pub maps: Vec<Option<usize>>,
    pub regions_in_args: bool,
    pub id_capacity: usize,
    pub generation: Option<u64>,
    pub candidates: u32,
    pub listed: bool,
    pub runs: bool,
}

#[derive(Default)]
pub(crate) struct FrameState {
    pub capacity: usize,
    pub capacity_grown: Option<u64>,
    pub capacity_generation: u64,
    pub slot_capacity: usize,
    pub slot_staging: Vec<u32>,
    pub slot_revision: Option<(Stamp, u64)>,
    pub color_staging: Vec<ColorData>,
    pub uploads: RetainedUploads,
    pub gpu_frame: bool,
    pub temporal: bool,
    pub motion_key: Option<(Stamp, u64, u64, glam::UVec2)>,
    pub presented_seconds: f32,
    pub impostor_errors: Vec<f32>,
    pub impostor_generation: u64,
    pub cards: bool,
    pub casts: bool,
    pub records_written: bool,
    pub draws: Draws,
    pub nodes: NodeInputs,
    pub groups: DrawGroups,
    pub stillness: Stillness,
    pub fade_slots: u64,
    pub card_capacity: usize,
    pub frame: Option<crate::layouts::PresentationFrame>,
    pub candidates: u32,
    pub occlusion: bool,
    pub lod: bool,
    pub pyramid: Pyramid,
    pub faces: Vec<u32>,
    pub casters: CasterDraws,
    pub instance_count: usize,
    pub watched: Option<u32>,
}

impl FrameState {
    pub(crate) fn levels_of(&self, index: usize) -> usize {
        self.draws.metas[index].levels.levels()
    }

    pub(crate) fn level_empty(&self, index: usize, level: usize) -> bool {
        level > 0 && self.draws.metas[index].levels.plan[level - 1].1 == 0
    }
}
