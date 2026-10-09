//! Node inputs of GPU presentation: current and previous
//! local poses, node records, authoritative world rows, composed parents
//! and each depth level's parents. They upload when core's gather sequence
//! advanced, as the changed ranges of each stream: staged together when
//! large, and current poses copied from a buffer staged ahead of the frame
//! when they changed whole.

use engine_core::{GpuPresentation, MAX_HIERARCHY_DEPTH, NodeInfo, NodePose, PoseRows};

use super::{grow, storage};
use crate::bindings::PRESENTATION_LEVEL;
use crate::gpu_data::COMPOSED_BYTES;
use crate::gpu_uploads::{changed_ranges, listed_ranges, whole_stream};
use crate::staging::{self, Prestaged, Segment, Staged, StagingRing};
use crate::uploads::UploadRange;

const LEVELS: usize = MAX_HIERARCHY_DEPTH + 1;
// Uniform offset alignment between the per-depth `Level` records.
const LEVEL_STRIDE: u64 = 256;

/// How a frame applies a new gather: it uploads the records stamped after
/// gather `since` (every record without one), after previous poses take
/// the last current ones on the GPU when `shift`.
pub(super) struct Gather {
    pub(super) since: Option<u64>,
    pub(super) shift: bool,
}

/// What an upload wrote: bytes of node records, whether one staging
/// buffer carried them, and whether current poses were copied from a
/// buffer staged ahead of the frame.
pub(super) struct NodeUpload {
    pub(super) bytes: u64,
    pub(super) staged: bool,
    pub(super) prestaged: bool,
}

/// Where uploads too large to queue as writes go: a ring of staging
/// buffers, and the encoder of copies submitted before the frame.
pub(crate) struct Staging<'a> {
    pub ring: &'a mut StagingRing,
    pub copies: &'a mut Option<wgpu::CommandEncoder>,
}

pub(super) struct NodeInputs {
    capacity: usize,
    // Current poses in slots [0, capacity), previous poses after them.
    pub(super) locals: wgpu::Buffer,
    pub(super) info: wgpu::Buffer,
    pub(super) authority: wgpu::Buffer,
    pub(super) composed: wgpu::Buffer,
    composed_capacity: usize,
    pub(super) parents: wgpu::Buffer,
    parent_capacity: usize,
    // Each depth level's range of parents as a `Level` record, with a group
    // per level for parent composition.
    levels: wgpu::Buffer,
    level_groups: Vec<wgpu::BindGroup>,
    level_ranges: Vec<(u32, u32)>,
    // Previous poses are consistent with the gathers this pass applied.
    blend: bool,
    // The gather identity and sequence uploaded last.
    uploaded: Option<(u64, u64)>,
    // Changed ranges of the current, previous, info and authority streams,
    // and a node bitmap for planning them from listed records.
    ranges: [Vec<UploadRange>; 4],
    upload_bits: Vec<u64>,
    // Current poses staged between frames for the next gather, with the
    // `GpuPresentation::current_version` they were copied at.
    prestaged: Option<(Prestaged, (u64, u64))>,
    // Advances when node storage is replaced.
    generation: u64,
}

impl NodeInputs {
    pub(super) fn new(device: &wgpu::Device, level_layout: &wgpu::BindGroupLayout) -> Self {
        let levels = crate::buffers::uniform(
            device,
            "engine/presentation-levels",
            LEVELS as u64 * LEVEL_STRIDE,
        );
        let level_groups = (0..LEVELS as u64)
            .map(|level| {
                crate::pipelines::group(
                    device,
                    "engine/presentation-level-group",
                    &PRESENTATION_LEVEL,
                    level_layout,
                    &[(
                        "level",
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &levels,
                            offset: level * LEVEL_STRIDE,
                            size: wgpu::BufferSize::new(16),
                        }),
                    )],
                )
            })
            .collect();
        let empty = |label| storage(device, label, 0, wgpu::BufferUsages::empty());
        Self {
            capacity: 0,
            locals: empty("engine/presentation-locals"),
            info: empty("engine/presentation-info"),
            authority: empty("engine/presentation-authority"),
            composed: empty("engine/presentation-composed"),
            composed_capacity: 0,
            parents: empty("engine/presentation-parents"),
            parent_capacity: 0,
            levels,
            level_groups,
            level_ranges: Vec::new(),
            blend: false,
            uploaded: None,
            ranges: Default::default(),
            upload_bits: Vec::new(),
            prestaged: None,
            generation: 0,
        }
    }

    /// Nodes the storage holds.
    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Advances when node storage is replaced.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether previous poses are consistent with the applied gathers, so
    /// culling blends them with current ones.
    pub(super) fn blend(&self) -> bool {
        self.blend
    }

    /// Each depth level's range of parents, as last uploaded.
    pub(super) fn level_ranges(&self) -> &[(u32, u32)] {
        &self.level_ranges
    }

    /// The group of depth level `level`'s `Level` record.
    pub(super) fn level_group(&self, level: usize) -> &wgpu::BindGroup {
        &self.level_groups[level]
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        [
            &self.locals,
            &self.info,
            &self.authority,
            &self.composed,
            &self.parents,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }

    /// Whether `gpu` was gathered ahead of its due frame: frames keep
    /// presenting the inputs applied last, in their node space and parent
    /// buffers.
    pub(super) fn holds(&self, gpu: &GpuPresentation<'_>) -> bool {
        gpu.hold
            && self
                .uploaded
                .is_some_and(|(identity, _)| identity == gpu.identity)
    }

    /// Sizes storage for `nodes` nodes and `parents` parent entries.
    pub(super) fn ensure(&mut self, device: &wgpu::Device, nodes: usize, parents: usize) {
        if nodes > self.capacity {
            self.capacity = grow(nodes);
            let capacity = self.capacity as u64;
            self.locals = storage(
                device,
                "engine/presentation-locals",
                2 * capacity * size_of::<NodePose>() as u64,
                wgpu::BufferUsages::empty(),
            );
            self.info = storage(
                device,
                "engine/presentation-info",
                capacity * size_of::<NodeInfo>() as u64,
                wgpu::BufferUsages::empty(),
            );
            self.authority = storage(
                device,
                "engine/presentation-authority",
                capacity * size_of::<PoseRows>() as u64,
                wgpu::BufferUsages::empty(),
            );
            self.uploaded = None;
            self.generation += 1;
            // Composed parents are indexed by node; resize with the node space.
            self.composed_capacity = 0;
        }
        if parents != 0 && self.composed_capacity < self.capacity {
            self.composed_capacity = self.capacity;
            self.composed = storage(
                device,
                "engine/presentation-composed",
                self.composed_capacity as u64 * COMPOSED_BYTES,
                wgpu::BufferUsages::empty(),
            );
            self.generation += 1;
        }
        if parents > self.parent_capacity {
            self.parent_capacity = grow(parents);
            self.parents = storage(
                device,
                "engine/presentation-parents",
                self.parent_capacity as u64 * 4,
                wgpu::BufferUsages::empty(),
            );
            self.generation += 1;
        }
    }

    /// How this frame applies `gpu`: nothing while `held` or when it was
    /// uploaded already. A complete upload carries every previous pose;
    /// after a quiet gather, previous poses take the last current ones on
    /// the GPU first, unless a copy was missed, which suspends blending
    /// until a copy applies in order.
    pub(super) fn gather(&mut self, gpu: &GpuPresentation<'_>, held: bool) -> Option<Gather> {
        let since = match self.uploaded {
            Some((identity, sequence)) if identity == gpu.identity => Some(sequence),
            _ => None,
        };
        if held || since == Some(gpu.sequence) {
            return None;
        }
        let mut shift = false;
        match since {
            // A complete upload carries every previous pose explicitly.
            None => self.blend = true,
            Some(since) if gpu.shift_sequence > since => {
                // A missed copy leaves previous poses inconsistent until a
                // copy this pass applies in order.
                shift = gpu.shift && since + 1 == gpu.sequence;
                self.blend = shift;
            }
            Some(_) => {}
        }
        Some(Gather { since, shift })
    }

    /// Uploads `gpu`'s records stamped after gather `since` (every record
    /// without one), its parents and its depth levels.
    pub(super) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        staging: Staging<'_>,
        gpu: &GpuPresentation<'_>,
        since: Option<u64>,
    ) -> NodeUpload {
        let mut upload = NodeUpload {
            bytes: 0,
            staged: false,
            prestaged: false,
        };
        let pose_stride = size_of::<NodePose>();
        let streams = [
            (
                &self.locals,
                0,
                bytemuck::cast_slice::<NodePose, u8>(gpu.current),
                pose_stride,
                gpu.current_stamps,
            ),
            (
                &self.locals,
                (self.capacity * pose_stride) as u64,
                bytemuck::cast_slice::<NodePose, u8>(gpu.previous),
                pose_stride,
                gpu.previous_stamps,
            ),
            (
                &self.info,
                0,
                bytemuck::cast_slice::<NodeInfo, u8>(gpu.info),
                size_of::<NodeInfo>(),
                gpu.info_stamps,
            ),
            (
                &self.authority,
                0,
                bytemuck::cast_slice::<PoseRows, u8>(gpu.authority),
                size_of::<PoseRows>(),
                gpu.authority_stamps,
            ),
        ];
        // One gather behind, the gather's per-stream counts decide clean
        // and dense streams without scanning their stamps.
        let consecutive = since.is_some_and(|since| since + 1 == gpu.sequence);
        let mut total = 0;
        let bits = &mut self.upload_bits;
        for (((ranges, (_, _, _, stride, stamps)), &stamped), listed) in self
            .ranges
            .iter_mut()
            .zip(&streams)
            .zip(&gpu.stamped)
            .zip(gpu.stamped_nodes)
        {
            match since {
                Some(_) if consecutive && stamped == 0 => ranges.clear(),
                Some(since) if !consecutive || !whole_stream(stamped, stamps.len()) => {
                    // One gather behind, its list names the records
                    // stamped since: no scan of every stamp.
                    match listed.filter(|_| consecutive) {
                        Some(nodes) => {
                            listed_ranges(nodes, stamps.len(), *stride, bits, ranges);
                        }
                        None => {
                            changed_ranges(stamps, since, *stride, ranges);
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
        // Current poses staged ahead of this frame replace their whole-stream
        // copy when unchanged since, after any shift submitted before.
        let current_bytes = size_of_val(gpu.current) as u64;
        if let Some((prestaged, _)) = self.prestaged.take().filter(|(prestaged, version)| {
            *version == gpu.current_version
                && prestaged.bytes() == current_bytes
                && matches!(self.ranges[0].as_slice(),
                    [range] if range.start == 0 && range.end == gpu.current.len())
        }) {
            let encoder = staging.copies.get_or_insert_with(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/presentation-input-copies"),
                })
            });
            prestaged.copy_to(encoder, &self.locals, 0);
            self.ranges[0].clear();
            upload.prestaged = true;
        }
        for (ranges, (_, _, _, stride, _)) in self.ranges.iter().zip(&streams) {
            total += ranges
                .iter()
                .map(|range| (range.end - range.start) * stride)
                .sum::<usize>();
        }
        upload.bytes = total as u64 + if upload.prestaged { current_bytes } else { 0 };
        // Every stream shares one staging buffer when the total is large.
        if total >= staging::MIN_STAGED_BYTES {
            let segments: Vec<Segment<'_>> = streams
                .iter()
                .zip(&self.ranges)
                .map(
                    |(&(destination, base, records, stride, _), ranges)| Segment {
                        destination,
                        base,
                        records,
                        stride,
                        ranges,
                    },
                )
                .collect();
            let encoder = staging.copies.get_or_insert_with(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/presentation-input-copies"),
                })
            });
            upload.staged =
                staging.ring.record_segments(device, encoder, &segments) == Staged::Recorded;
        }
        if !upload.staged {
            for (ranges, &(buffer, base, bytes, stride, _)) in self.ranges.iter().zip(&streams) {
                for range in ranges {
                    queue.write_buffer(
                        buffer,
                        base + (range.start * stride) as u64,
                        &bytes[range.start * stride..range.end * stride],
                    );
                }
            }
        }
        if !gpu.parents.is_empty() {
            queue.write_buffer(&self.parents, 0, bytemuck::cast_slice(gpu.parents));
        }
        self.level_ranges.clear();
        let mut start = 0;
        for &end in gpu.levels {
            self.level_ranges.push((start, end));
            start = end;
        }
        for (level, &(start, end)) in self.level_ranges.iter().enumerate() {
            queue.write_buffer(
                &self.levels,
                level as u64 * LEVEL_STRIDE,
                bytemuck::cast_slice(&[start, end, 0, 0]),
            );
        }
        self.uploaded = Some((gpu.identity, gpu.sequence));
        upload
    }

    /// Holds current poses staged between frames for the next gather to copy.
    pub(super) fn prestage(&mut self, prestaged: Prestaged, version: (u64, u64)) {
        self.prestaged = Some((prestaged, version));
    }
}
