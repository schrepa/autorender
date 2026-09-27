//! GPU presentation (decision 014): uploads gathered interpolation inputs and
//! candidate draw groups, then records compute passes that blend, compose,
//! validate and cull displayed poses and compact visible slots per group for
//! indirect draws. Per-frame CPU work is a uniform write and a few dispatches;
//! inputs upload only when core's gather sequence advanced (normally once per
//! fixed tick) and groups only when the candidate structure changed. After a
//! quiet tick only current poses upload: a small pass first copies the last
//! current poses into the previous half of the pose buffer.

use std::{cell::Cell, rc::Rc};

use engine_core::{
    Aabb, GpuPresentation, MAX_HIERARCHY_DEPTH, NodeInfo, NodePose, PoseRows, RenderFrame,
    RenderJoint,
};

use crate::batches::{BatchPlan, DrawBatch};
use crate::gpu_data::{
    BatchData, COMPOSED_BYTES, CandidateData, DRAW_ARGS_BYTES, PRESENTATION_SHADER,
    PresentationFrame, frustum_uniform,
};
use crate::gpu_uploads::{changed_ranges, listed_ranges, whole_stream};
use crate::staging::{self, Prestaged, Segment, Staged, StagingRing};
use crate::uploads::UploadRange;

/// Presentation stages a profiling capture times as separate passes: the
/// previous-pose shift, parent composition, culling, scanning (group sums
/// and draw arguments) and scattering of visible slots.
#[cfg_attr(not(feature = "profiling"), allow(dead_code))]
pub(crate) const STAGE_COUNT: usize = 5;
const SHIFT: usize = 0;
const COMPOSE: usize = 1;
const CULL: usize = 2;
const SCAN: usize = 3;
const SCATTER: usize = 4;

/// Timestamp queries of one frame's presentation stages: stage `s` writes
/// query `2 + 2s` when its pass begins and `3 + 2s` when it ends.
#[derive(Clone, Copy)]
pub(crate) struct StageStamps<'a> {
    query_set: &'a wgpu::QuerySet,
}

impl<'a> StageStamps<'a> {
    #[cfg_attr(not(feature = "profiling"), allow(dead_code))]
    pub(crate) fn new(query_set: &'a wgpu::QuerySet) -> Self {
        Self { query_set }
    }

    fn writes(self, stage: usize) -> wgpu::ComputePassTimestampWrites<'a> {
        let begin = 2 + 2 * stage as u32;
        wgpu::ComputePassTimestampWrites {
            query_set: self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(begin + 1),
        }
    }
}

const GROUP: u32 = 256;
const PARENT_GROUP: u32 = 64;
const JOINT_GROUP: u32 = 64;
const ARGS_GROUP: u32 = 64;
const LEVELS: usize = MAX_HIERARCHY_DEPTH + 1;
const READBACKS: usize = 3;
// Uniform offset alignment between the per-depth `Level` records.
const LEVEL_STRIDE: u64 = 256;
// Beyond this bound the CPU frustum also fails open (see culling.rs).
const MAX_COEFFICIENT: f32 = 1.0e15;
// Bindings need at least one element; exceeds every record stride.
const MIN_STORAGE_BYTES: u64 = 256;

struct Pipelines {
    shift_previous: wgpu::ComputePipeline,
    compose_parents: wgpu::ComputePipeline,
    cull: wgpu::ComputePipeline,
    skin_palette: wgpu::ComputePipeline,
    sum_groups: wgpu::ComputePipeline,
    scan_groups: wgpu::ComputePipeline,
    batch_args: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
}

struct BindGroups {
    shift_previous: wgpu::BindGroup,
    compose_parents: wgpu::BindGroup,
    // One per pose buffer: temporally antialiased frames alternate two.
    cull: [Option<wgpu::BindGroup>; 2],
    // One per palette buffer, which alternate with the pose buffers.
    skin: [Option<wgpu::BindGroup>; 2],
    sum_groups: wgpu::BindGroup,
    scan_groups: wgpu::BindGroup,
    batch_args: wgpu::BindGroup,
    scatter: wgpu::BindGroup,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Readback {
    Idle,
    Recorded,
    Pending,
}

struct VisibleReadback {
    buffer: wgpu::Buffer,
    state: Rc<Cell<Readback>>,
    value: Rc<Cell<Option<u32>>>,
}

/// Per-frame outcome for profiling counters and stats.
#[derive(Default)]
pub(crate) struct GpuPrepared {
    pub candidates: u32,
    pub node_upload_bytes: u64,
    pub group_upload_bytes: u64,
    pub full_node_upload: bool,
    pub staged: bool,
    pub parent_levels: usize,
    /// This frame copied current poses into previous slots on the GPU.
    pub shifted: bool,
    /// Blending is suspended after a missed copy until the next applied one.
    pub blend: bool,
    /// Current poses were copied from a buffer staged ahead of this frame.
    pub prestaged: bool,
    /// Inputs gathered ahead of their due frame were held, not applied.
    pub held: bool,
    /// Stages timed in their own passes (bit per stage), for profiling.
    pub timed: u8,
}

pub(crate) struct GpuPresentationPass {
    pipelines: Pipelines,
    frame: wgpu::Buffer,
    levels: wgpu::Buffer,
    // One group per depth level: auto layouts have no dynamic offsets.
    level_groups: Vec<wgpu::BindGroup>,
    node_capacity: usize,
    // Current poses in slots [0, node_capacity), previous poses after them.
    locals: wgpu::Buffer,
    // Previous poses are consistent with the gathers this pass applied.
    blend: bool,
    info: wgpu::Buffer,
    authority: wgpu::Buffer,
    composed: wgpu::Buffer,
    composed_capacity: usize,
    parents: wgpu::Buffer,
    parent_capacity: usize,
    candidate_capacity: usize,
    candidates: wgpu::Buffer,
    // The same candidates in slot order, for culling.
    slot_candidates: wgpu::Buffer,
    scan: wgpu::Buffer,
    visible_ids: wgpu::Buffer,
    batch_capacity: usize,
    batches: wgpu::Buffer,
    batch_bases: wgpu::Buffer,
    draws: wgpu::Buffer,
    bind_groups: Option<BindGroups>,
    // Advances when candidate, batch or scan storage is replaced.
    buffers_generation: u64,
    // Identity of the pose storage bound for writing; rebuild groups on change.
    poses_generation: Option<u64>,
    // The pose buffer this frame's culling writes.
    poses_index: usize,
    uploaded: Option<(u64, u64)>,
    groups_key: Option<(u64, u64, u64, u64, bool)>,
    candidate_count: u32,
    batch_count: u32,
    level_ranges: Vec<(u32, u32)>,
    // Changed ranges of the current, previous, info and authority streams,
    // and a node bitmap for planning them from listed records.
    ranges: [Vec<UploadRange>; 4],
    upload_bits: Vec<u64>,
    candidate_staging: Vec<CandidateData>,
    slot_staging: Vec<CandidateData>,
    // Per slot: its candidate's draw-group position (`u32::MAX`: none).
    slot_positions: Vec<u32>,
    batch_staging: Vec<BatchData>,
    // Skinned slots' joints (decision 036): nodes and inverse bind matrices
    // as last uploaded, and how many this frame's palette pass writes.
    joint_capacity: usize,
    joint_nodes: wgpu::Buffer,
    inverse_binds: wgpu::Buffer,
    uploaded_nodes: Vec<u32>,
    uploaded_binds: Vec<PoseRows>,
    joint_count: u32,
    readbacks: Vec<VisibleReadback>,
    // Current poses staged between frames for the next gather, with the
    // `GpuPresentation::current_version` they were copied at.
    prestaged: Option<(Prestaged, (u64, u64))>,
    last_visible: Rc<Cell<Option<u32>>>,
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(MIN_STORAGE_BYTES).next_multiple_of(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn grow(needed: usize) -> usize {
    needed.max(1_024).next_power_of_two()
}

impl GpuPresentationPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/presentation.wgsl"),
            source: wgpu::ShaderSource::Wgsl(PRESENTATION_SHADER.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            shift_previous: pipeline("shift_previous"),
            compose_parents: pipeline("compose_parents"),
            cull: pipeline("cull"),
            skin_palette: pipeline("skin_palette"),
            sum_groups: pipeline("sum_groups"),
            scan_groups: pipeline("scan_groups"),
            batch_args: pipeline("batch_args"),
            scatter: pipeline("scatter"),
        };
        let frame = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/presentation-frame"),
            size: size_of::<PresentationFrame>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let levels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/presentation-levels"),
            size: LEVELS as u64 * LEVEL_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let level_layout = pipelines.compose_parents.get_bind_group_layout(1);
        let level_groups = (0..LEVELS as u64)
            .map(|level| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine/presentation-level-group"),
                    layout: &level_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &levels,
                            offset: level * LEVEL_STRIDE,
                            size: wgpu::BufferSize::new(16),
                        }),
                    }],
                })
            })
            .collect();
        let empty = |label| storage(device, label, 0, wgpu::BufferUsages::empty());
        Self {
            pipelines,
            frame,
            levels,
            level_groups,
            node_capacity: 0,
            locals: empty("engine/presentation-locals"),
            blend: false,
            info: empty("engine/presentation-info"),
            authority: empty("engine/presentation-authority"),
            composed: empty("engine/presentation-composed"),
            composed_capacity: 0,
            parents: empty("engine/presentation-parents"),
            parent_capacity: 0,
            candidate_capacity: 0,
            candidates: empty("engine/presentation-candidates"),
            slot_candidates: empty("engine/presentation-slot-candidates"),
            scan: empty("engine/presentation-scan"),
            visible_ids: storage(
                device,
                "engine/presentation-visible-ids",
                0,
                wgpu::BufferUsages::VERTEX,
            ),
            batch_capacity: 0,
            batches: empty("engine/presentation-batches"),
            batch_bases: empty("engine/presentation-batch-bases"),
            draws: storage(
                device,
                "engine/presentation-draws",
                0,
                wgpu::BufferUsages::INDIRECT,
            ),
            bind_groups: None,
            buffers_generation: 0,
            poses_generation: None,
            poses_index: 0,
            uploaded: None,
            groups_key: None,
            candidate_count: 0,
            batch_count: 0,
            level_ranges: Vec::new(),
            ranges: Default::default(),
            upload_bits: Vec::new(),
            candidate_staging: Vec::new(),
            slot_staging: Vec::new(),
            slot_positions: Vec::new(),
            batch_staging: Vec::new(),
            joint_capacity: 0,
            joint_nodes: empty("engine/presentation-joint-nodes"),
            inverse_binds: empty("engine/presentation-inverse-binds"),
            uploaded_nodes: Vec::new(),
            uploaded_binds: Vec::new(),
            joint_count: 0,
            readbacks: Vec::new(),
            prestaged: None,
            last_visible: Rc::new(Cell::new(None)),
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        [
            &self.locals,
            &self.info,
            &self.authority,
            &self.composed,
            &self.parents,
            &self.candidates,
            &self.slot_candidates,
            &self.scan,
            &self.visible_ids,
            &self.batches,
            &self.batch_bases,
            &self.draws,
            &self.joint_nodes,
            &self.inverse_binds,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }

    /// Latest visible instance count read back from the GPU, a few frames old.
    pub(crate) fn visible(&self) -> Option<u32> {
        self.last_visible.get()
    }

    fn ensure_nodes(&mut self, device: &wgpu::Device, nodes: usize, parents: usize) {
        if nodes > self.node_capacity {
            self.node_capacity = grow(nodes);
            let capacity = self.node_capacity as u64;
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
            self.bind_groups = None;
            // Composed parents are indexed by node; resize with the node space.
            self.composed_capacity = 0;
        }
        if parents != 0 && self.composed_capacity < self.node_capacity {
            self.composed_capacity = self.node_capacity;
            self.composed = storage(
                device,
                "engine/presentation-composed",
                self.composed_capacity as u64 * COMPOSED_BYTES,
                wgpu::BufferUsages::empty(),
            );
            self.bind_groups = None;
        }
        if parents > self.parent_capacity {
            self.parent_capacity = grow(parents);
            self.parents = storage(
                device,
                "engine/presentation-parents",
                self.parent_capacity as u64 * 4,
                wgpu::BufferUsages::empty(),
            );
            self.bind_groups = None;
        }
    }

    fn ensure_groups(&mut self, device: &wgpu::Device, candidates: usize, batches: usize) {
        if candidates > self.candidate_capacity {
            self.candidate_capacity = grow(candidates);
            let capacity = self.candidate_capacity as u64;
            let groups = capacity.div_ceil(u64::from(GROUP));
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
            // Flags, one sum per group, and the total.
            self.scan = storage(
                device,
                "engine/presentation-scan",
                (capacity + groups + 1) * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            self.visible_ids = storage(
                device,
                "engine/presentation-visible-ids",
                capacity * 4,
                wgpu::BufferUsages::VERTEX,
            );
            self.groups_key = None;
            self.bind_groups = None;
            self.buffers_generation += 1;
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
                capacity * 4,
                wgpu::BufferUsages::empty(),
            );
            self.draws = storage(
                device,
                "engine/presentation-draws",
                capacity * DRAW_ARGS_BYTES,
                wgpu::BufferUsages::INDIRECT,
            );
            self.groups_key = None;
            self.bind_groups = None;
            self.buffers_generation += 1;
        }
    }

    /// This frame's candidates in slot order, draw groups and candidate
    /// flags, for shadow caster selection (decision 020) after culling.
    pub(crate) fn shadow_inputs(&self) -> crate::shadows::Presented<'_> {
        crate::shadows::Presented {
            candidates: &self.slot_candidates,
            batches: &self.batches,
            flags: &self.scan,
            count: self.candidate_count,
            generation: self.buffers_generation,
        }
    }

    fn bind_groups(
        &mut self,
        device: &wgpu::Device,
        poses: &wgpu::Buffer,
        palette: &wgpu::Buffer,
        index: usize,
    ) {
        if let Some(groups) = &self.bind_groups {
            if groups.cull[index].is_none() {
                let cull = self.cull_group(device, poses);
                self.bind_groups.as_mut().expect("checked above").cull[index] = Some(cull);
            }
            if self.bind_groups.as_ref().expect("checked above").skin[index].is_none() {
                let skin = self.skin_group(device, palette);
                self.bind_groups.as_mut().expect("checked above").skin[index] = Some(skin);
            }
            return;
        }
        let group = |pipeline: &wgpu::ComputePipeline, label, entries: &[wgpu::BindGroupEntry]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            })
        };
        let pipelines = &self.pipelines;
        let mut cull = [None, None];
        cull[index] = Some(self.cull_group(device, poses));
        let mut skin = [None, None];
        skin[index] = Some(self.skin_group(device, palette));
        self.bind_groups = Some(BindGroups {
            shift_previous: group(
                &pipelines.shift_previous,
                "engine/presentation-shift-group",
                &[entry(2, &self.info), entry(13, &self.locals)],
            ),
            compose_parents: group(
                &pipelines.compose_parents,
                "engine/presentation-compose-group",
                &[
                    entry(0, &self.frame),
                    entry(1, &self.locals),
                    entry(2, &self.info),
                    entry(3, &self.authority),
                    entry(4, &self.composed),
                    entry(5, &self.parents),
                ],
            ),
            cull,
            skin,
            sum_groups: group(
                &pipelines.sum_groups,
                "engine/presentation-sum-group",
                &[entry(0, &self.frame), entry(9, &self.scan)],
            ),
            scan_groups: group(
                &pipelines.scan_groups,
                "engine/presentation-scan-group",
                &[entry(0, &self.frame), entry(9, &self.scan)],
            ),
            batch_args: group(
                &pipelines.batch_args,
                "engine/presentation-args-group",
                &[
                    entry(0, &self.frame),
                    entry(7, &self.batches),
                    entry(9, &self.scan),
                    entry(10, &self.batch_bases),
                    entry(12, &self.draws),
                ],
            ),
            scatter: group(
                &pipelines.scatter,
                "engine/presentation-scatter-group",
                &[
                    entry(0, &self.frame),
                    entry(6, &self.candidates),
                    entry(7, &self.batches),
                    entry(9, &self.scan),
                    entry(10, &self.batch_bases),
                    entry(11, &self.visible_ids),
                ],
            ),
        });
    }

    fn cull_group(&self, device: &wgpu::Device, poses: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/presentation-cull-group"),
            layout: &self.pipelines.cull.get_bind_group_layout(0),
            entries: &[
                entry(0, &self.frame),
                entry(1, &self.locals),
                entry(2, &self.info),
                entry(3, &self.authority),
                entry(4, &self.composed),
                entry(7, &self.batches),
                entry(8, poses),
                entry(9, &self.scan),
                entry(14, &self.slot_candidates),
            ],
        })
    }

    fn skin_group(&self, device: &wgpu::Device, palette: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/presentation-skin-group"),
            layout: &self.pipelines.skin_palette.get_bind_group_layout(0),
            entries: &[
                entry(0, &self.frame),
                entry(1, &self.locals),
                entry(2, &self.info),
                entry(3, &self.authority),
                entry(4, &self.composed),
                entry(15, &self.joint_nodes),
                entry(16, &self.inverse_binds),
                entry(17, palette),
            ],
        })
    }

    /// Uploads skinned slots' joints (decision 036) when they changed.
    fn prepare_joints(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        joints: &[RenderJoint],
    ) {
        self.joint_count = joints.len() as u32;
        if joints.len() > self.joint_capacity {
            self.joint_capacity = grow(joints.len());
            let capacity = self.joint_capacity as u64;
            self.joint_nodes = storage(
                device,
                "engine/presentation-joint-nodes",
                capacity * 4,
                wgpu::BufferUsages::empty(),
            );
            self.inverse_binds = storage(
                device,
                "engine/presentation-inverse-binds",
                capacity * size_of::<PoseRows>() as u64,
                wgpu::BufferUsages::empty(),
            );
            self.uploaded_nodes.clear();
            self.uploaded_binds.clear();
            self.bind_groups = None;
        }
        let unchanged = self.uploaded_nodes.len() == joints.len()
            && joints
                .iter()
                .zip(&self.uploaded_nodes)
                .zip(&self.uploaded_binds)
                .all(|((joint, node), bind)| joint.node == *node && joint.inverse_bind == *bind);
        if unchanged || joints.is_empty() {
            return;
        }
        self.uploaded_nodes.clear();
        self.uploaded_nodes
            .extend(joints.iter().map(|joint| joint.node));
        self.uploaded_binds.clear();
        self.uploaded_binds
            .extend(joints.iter().map(|joint| joint.inverse_bind));
        queue.write_buffer(
            &self.joint_nodes,
            0,
            bytemuck::cast_slice(&self.uploaded_nodes),
        );
        queue.write_buffer(
            &self.inverse_binds,
            0,
            bytemuck::cast_slice(&self.uploaded_binds),
        );
    }

    /// Holds current poses staged between frames for the next gather to copy.
    pub(crate) fn prestage(&mut self, prestaged: Prestaged, version: (u64, u64)) {
        self.prestaged = Some((prestaged, version));
    }

    /// Uploads changed inputs, groups and the frame uniform. Large node
    /// uploads record staged copies into `copies`, submitted before the frame.
    /// Culling writes displayed poses into `poses`, pose buffer `poses_index`
    /// (0 or 1) of the pair `poses_generation` identifies, and tests them
    /// against `view_projection`, the projection the frame rasterizes.
    /// With `motion`, visible IDs of moving slots carry `MOVED_SLOT`.
    /// `batch_meta` gives each plan batch's index count, mesh bounds, shadow
    /// draw group and whether its mesh has joints; `shadow_groups`
    /// distinguishes plans whose batch records carry shadow groups. Skinned
    /// slots' palettes (decision 036) are written into `palette`, of the
    /// pose storage generation, from the frame's joints.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        staging_ring: &mut StagingRing,
        copies: &mut Option<wgpu::CommandEncoder>,
        frame: &RenderFrame<'_>,
        gpu: &GpuPresentation<'_>,
        plan: &BatchPlan,
        batch_meta: impl Fn(usize, &DrawBatch) -> (u32, Aabb, u32, bool),
        shadow_groups: bool,
        poses: &wgpu::Buffer,
        palette: &wgpu::Buffer,
        poses_index: usize,
        poses_generation: u64,
        view_projection: glam::Mat4,
        motion: bool,
        stamps: Option<StageStamps<'_>>,
    ) -> GpuPrepared {
        let mut prepared = GpuPrepared::default();
        let nodes = gpu.info.len();
        // A held gather waits for its due frame: keep presenting the inputs
        // applied last, in their node space and parent buffers.
        let held = gpu.hold
            && self
                .uploaded
                .is_some_and(|(identity, _)| identity == gpu.identity);
        prepared.held = held;
        self.ensure_nodes(device, nodes, if held { 0 } else { gpu.parents.len() });
        let candidates = plan.ordered_instances.len();
        self.ensure_groups(device, candidates, plan.batches.len());
        self.prepare_joints(device, queue, &frame.snapshot().joints);
        if self.poses_generation != Some(poses_generation) {
            self.poses_generation = Some(poses_generation);
            self.bind_groups = None;
        }
        self.bind_groups(device, poses, palette, poses_index);
        self.poses_index = poses_index;

        // Node inputs: records newer than the last uploaded gather.
        let since = match self.uploaded {
            Some((identity, sequence)) if identity == gpu.identity => Some(sequence),
            _ => None,
        };
        if !held && since != Some(gpu.sequence) {
            prepared.full_node_upload = since.is_none();
            match since {
                // A complete upload carries every previous pose explicitly.
                None => self.blend = true,
                Some(since) if gpu.shift_sequence > since => {
                    if gpu.shift && since + 1 == gpu.sequence {
                        // Submitted before this gather's writes: queued writes and
                        // staged copies take effect at the next submission.
                        let groups = self.bind_groups.as_ref().expect("bound above");
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("engine/presentation-shift"),
                            });
                        {
                            let mut pass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("engine/presentation-shift"),
                                    timestamp_writes: stamps.map(|stamps| stamps.writes(SHIFT)),
                                });
                            if stamps.is_some() {
                                prepared.timed |= 1 << SHIFT;
                            }
                            pass.set_pipeline(&self.pipelines.shift_previous);
                            pass.set_bind_group(0, &groups.shift_previous, &[]);
                            // One invocation per node; the pass copies only
                            // nodes the last applied gather made active.
                            pass.dispatch_workgroups(
                                (self.node_capacity as u32).div_ceil(GROUP),
                                1,
                                1,
                            );
                        }
                        queue.submit([encoder.finish()]);
                        prepared.shifted = true;
                        self.blend = true;
                    } else {
                        // A copy was missed: previous poses stay inconsistent
                        // until a copy this pass applies in order.
                        self.blend = false;
                    }
                }
                Some(_) => {}
            }
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
                    (self.node_capacity * pose_stride) as u64,
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
            // copy when unchanged since, after any shift submitted above.
            let current_bytes = size_of_val(gpu.current) as u64;
            if let Some((prestaged, _)) = self.prestaged.take().filter(|(prestaged, version)| {
                *version == gpu.current_version
                    && prestaged.bytes() == current_bytes
                    && matches!(self.ranges[0].as_slice(),
                        [range] if range.start == 0 && range.end == gpu.current.len())
            }) {
                let encoder = copies.get_or_insert_with(|| {
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("engine/presentation-input-copies"),
                    })
                });
                prestaged.copy_to(encoder, &self.locals, 0);
                self.ranges[0].clear();
                prepared.prestaged = true;
            }
            for (ranges, (_, _, _, stride, _)) in self.ranges.iter().zip(&streams) {
                total += ranges
                    .iter()
                    .map(|range| (range.end - range.start) * stride)
                    .sum::<usize>();
            }
            prepared.node_upload_bytes =
                total as u64 + if prepared.prestaged { current_bytes } else { 0 };
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
                let encoder = copies.get_or_insert_with(|| {
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("engine/presentation-input-copies"),
                    })
                });
                prepared.staged =
                    staging_ring.record_segments(device, encoder, &segments) == Staged::Recorded;
            }
            if !prepared.staged {
                for (ranges, &(buffer, base, bytes, stride, _)) in self.ranges.iter().zip(&streams)
                {
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
        }
        prepared.parent_levels = self.level_ranges.len();
        prepared.blend = self.blend;

        // Candidate groups: rebuilt with the candidate structure or node space.
        let key = (
            frame.identity(),
            frame.structure_revision(),
            frame.source_revision(),
            gpu.identity,
            shadow_groups,
        );
        if self.groups_key != Some(key) {
            let slot_nodes = frame.slot_nodes();
            self.candidate_staging.clear();
            self.candidate_staging.reserve_exact(candidates);
            self.batch_staging.clear();
            self.batch_staging.reserve_exact(plan.batches.len());
            for (index, batch) in plan.batches.iter().enumerate() {
                let (index_count, bounds, group, skinned) = batch_meta(index, batch);
                let range = batch.instances.start as usize..batch.instances.end as usize;
                self.batch_staging.push(BatchData {
                    start: batch.instances.start,
                    count: batch.instances.end - batch.instances.start,
                    index_count,
                    pad: group,
                    bounds_min: bounds
                        .min()
                        .extend(if skinned { 1.0 } else { 0.0 })
                        .to_array(),
                    bounds_max: bounds.max().extend(0.0).to_array(),
                });
                let base = self.candidate_staging.len() as u32;
                self.candidate_staging.extend(
                    plan.ordered_instances[range]
                        .iter()
                        .zip(base..)
                        .map(|(&slot, position)| CandidateData {
                            slot: slot as u32,
                            node: slot_nodes[slot],
                            batch: index as u32,
                            position,
                        }),
                );
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
            prepared.group_upload_bytes =
                (2 * self.candidate_staging.len() * size_of::<CandidateData>()
                    + self.batch_staging.len() * size_of::<BatchData>()) as u64;
            self.candidate_count = candidates as u32;
            self.batch_count = plan.batches.len() as u32;
            self.groups_key = Some(key);
        }
        prepared.candidates = self.candidate_count;

        // A view-projection the CPU frustum rejects disables culling there too.
        let (planes, absolute) = frustum_uniform(view_projection);
        let bounded = absolute
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value <= MAX_COEFFICIENT)
            && planes.iter().flatten().all(|value| value.is_finite());
        let uniform = PresentationFrame {
            alpha: gpu.alpha,
            candidates: self.candidate_count,
            budget: frame.instance_budget().min(u32::MAX as usize) as u32,
            culling: u32::from(frame.culling() && bounded),
            groups: self.candidate_count.div_ceil(GROUP),
            batches: self.batch_count,
            node_capacity: self.node_capacity as u32,
            blend: u32::from(self.blend),
            planes,
            absolute_view_projection: absolute,
            motion: u32::from(motion),
            joints: self.joint_count,
            pad: [0; 2],
        };
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&uniform));
        prepared
    }

    /// Records blending, composition, culling and compaction in one compute
    /// pass. With `stamps` (profiling captures) each stage gets its own timed
    /// pass instead; returns the stages timed, one bit each.
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        stamps: Option<StageStamps<'_>>,
    ) -> u8 {
        let Some(groups) = self.bind_groups.as_ref() else {
            return 0;
        };
        let compose = !self.level_ranges.is_empty();
        let candidates = self.candidate_count != 0;
        // Dispatches of each stage, in order.
        let record = |pass: &mut wgpu::ComputePass<'_>, stage: usize| match stage {
            COMPOSE => {
                pass.set_pipeline(&self.pipelines.compose_parents);
                pass.set_bind_group(0, &groups.compose_parents, &[]);
                for (level, &(start, end)) in self.level_ranges.iter().enumerate() {
                    if end > start {
                        pass.set_bind_group(1, &self.level_groups[level], &[]);
                        pass.dispatch_workgroups((end - start).div_ceil(PARENT_GROUP), 1, 1);
                    }
                }
            }
            CULL => {
                // Skinned slots' palettes, from the composed parents.
                if self.joint_count != 0 {
                    pass.set_pipeline(&self.pipelines.skin_palette);
                    pass.set_bind_group(
                        0,
                        groups.skin[self.poses_index]
                            .as_ref()
                            .expect("bound in prepare"),
                        &[],
                    );
                    pass.dispatch_workgroups(self.joint_count.div_ceil(JOINT_GROUP), 1, 1);
                }
                // Flags in slot order, then their sums in draw-group order.
                let workgroups = self.candidate_count.div_ceil(GROUP);
                pass.set_pipeline(&self.pipelines.cull);
                pass.set_bind_group(
                    0,
                    groups.cull[self.poses_index]
                        .as_ref()
                        .expect("bound in prepare"),
                    &[],
                );
                pass.dispatch_workgroups(workgroups, 1, 1);
                pass.set_pipeline(&self.pipelines.sum_groups);
                pass.set_bind_group(0, &groups.sum_groups, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            SCAN => {
                pass.set_pipeline(&self.pipelines.scan_groups);
                pass.set_bind_group(0, &groups.scan_groups, &[]);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&self.pipelines.batch_args);
                pass.set_bind_group(0, &groups.batch_args, &[]);
                pass.dispatch_workgroups(self.batch_count.div_ceil(ARGS_GROUP), 1, 1);
            }
            _ => {
                pass.set_pipeline(&self.pipelines.scatter);
                pass.set_bind_group(0, &groups.scatter, &[]);
                pass.dispatch_workgroups(self.candidate_count.div_ceil(GROUP), 1, 1);
            }
        };
        let stages = [
            (COMPOSE, compose),
            (CULL, candidates || self.joint_count != 0),
            (SCAN, candidates),
            (SCATTER, candidates),
        ];
        let mut timed = 0;
        if let Some(stamps) = stamps {
            for (stage, runs) in stages {
                if runs {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("engine/presentation-stage"),
                        timestamp_writes: Some(stamps.writes(stage)),
                    });
                    record(&mut pass, stage);
                    timed |= 1 << stage;
                }
            }
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/presentation"),
                timestamp_writes: None,
            });
            for (stage, runs) in stages {
                if runs {
                    record(&mut pass, stage);
                }
            }
        }
        if self.candidate_count != 0 {
            // Read back the total visible count without waiting for it.
            let total = u64::from(self.candidate_count + self.candidate_count.div_ceil(GROUP)) * 4;
            let scan = self.scan.clone();
            if let Some(readback) = self.idle_readback(device) {
                encoder.copy_buffer_to_buffer(&scan, total, &readback.buffer, 0, 4);
                readback.state.set(Readback::Recorded);
            }
        }
        timed
    }

    fn idle_readback(&mut self, device: &wgpu::Device) -> Option<&VisibleReadback> {
        if let Some(index) = self
            .readbacks
            .iter()
            .position(|readback| readback.state.get() == Readback::Idle)
        {
            return Some(&self.readbacks[index]);
        }
        if self.readbacks.len() == READBACKS {
            return None;
        }
        self.readbacks.push(VisibleReadback {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/presentation-visible-readback"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Rc::new(Cell::new(Readback::Idle)),
            value: Rc::clone(&self.last_visible),
        });
        self.readbacks.last()
    }

    /// Requests maps of read-backs recorded in the frame just submitted.
    pub(crate) fn after_submit(&mut self) {
        for readback in &self.readbacks {
            if readback.state.get() != Readback::Recorded {
                continue;
            }
            readback.state.set(Readback::Pending);
            let state = Rc::clone(&readback.state);
            let value = Rc::clone(&readback.value);
            let buffer = readback.buffer.clone();
            readback
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        if let Ok(view) = buffer.slice(..).get_mapped_range() {
                            let bytes: [u8; 4] = view[..4].try_into().unwrap_or([0; 4]);
                            value.set(Some(u32::from_le_bytes(bytes)));
                        }
                        buffer.unmap();
                    }
                    state.set(Readback::Idle);
                });
        }
    }

    /// Visible-ID region and indirect arguments of plan batch `index`.
    pub(crate) fn draw_source<'a>(
        &'a self,
        batch: &DrawBatch,
        index: usize,
    ) -> (wgpu::BufferSlice<'a>, &'a wgpu::Buffer, u64) {
        let start = u64::from(batch.instances.start) * 4;
        let end = u64::from(batch.instances.end) * 4;
        (
            self.visible_ids.slice(start..end),
            &self.draws,
            index as u64 * DRAW_ARGS_BYTES,
        )
    }
}
