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
    BatchData, COMPOSED_BYTES, CandidateData, DRAW_ARGS_BYTES, FLAG_PHASE1, FLAG_PHASE2,
    FLAG_PROBE, FLAG_VISIBLE, LOD_LEVELS, PRESENTATION_SHADER, PresentationFrame, STILL_FRAMES,
    SkinnedData, frustum_uniform,
};
use crate::gpu_uploads::{changed_ranges, listed_ranges, whole_stream};
use crate::mesh_arena::Ids;
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

// Candidates per workgroup of the per-candidate passes, and of the
// compaction passes (`GROUP` in presentation.wgsl), whose workgroup arrays
// are kept small for Firefox's compile times (decision 049).
const GROUP: u32 = 256;
const SCAN_GROUP: u32 = 64;
const PARENT_GROUP: u32 = 64;
const JOINT_GROUP: u32 = 64;
const SKINNED_GROUP: u32 = 64;
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
    skin_bounds: wgpu::ComputePipeline,
    sum_groups: wgpu::ComputePipeline,
    scan_groups: wgpu::ComputePipeline,
    batch_args: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    occlusion_test: wgpu::ComputePipeline,
    count_still: wgpu::ComputePipeline,
    probe_cull: wgpu::ComputePipeline,
}

struct BindGroups {
    shift_previous: wgpu::BindGroup,
    compose_parents: wgpu::BindGroup,
    // One per pose buffer: temporally antialiased frames alternate two.
    cull: [Option<wgpu::BindGroup>; 2],
    // One per palette buffer, which alternate with the pose buffers.
    skin: [Option<wgpu::BindGroup>; 2],
    // Skinned slots' bounds (decision 036), per pose and palette buffer.
    skin_bounds: [Option<wgpu::BindGroup>; 2],
    sum_groups: wgpu::BindGroup,
    scan_groups: wgpu::BindGroup,
    batch_args: wgpu::BindGroup,
    scatter: wgpu::BindGroup,
    // Phase two of occlusion culling (decision 040): the same compaction
    // over its own uniform, draw arguments and visible IDs.
    sum_groups2: wgpu::BindGroup,
    scan_groups2: wgpu::BindGroup,
    batch_args2: wgpu::BindGroup,
    scatter2: wgpu::BindGroup,
    // Slots' still frames for reflection probes (decision 043) while they
    // are counted, per pose buffer.
    count_still: [Option<wgpu::BindGroup>; 2],
}

// Candidate, batch and scan storage generation, pose storage generation and
// pose buffer index that reflection probe faces' groups bind (decision 043).
type ProbeGroupsKey = (u64, Option<u64>, usize);

// One reflection probe cube face's cull and compaction (decision 043).
struct ProbeGroups {
    cull: wgpu::BindGroup,
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
    occluded: Rc<Cell<Option<u32>>>,
    subpixel: Rc<Cell<Option<u32>>>,
    levels: Rc<Cell<Option<[u32; 4]>>>,
    triangles: Rc<Cell<Option<u32>>>,
}

// The occlusion test group's pose buffer index, pyramid generation, buffer
// generation and pose storage generation.
type OcclusionKey = (usize, u64, u64, Option<u64>);

// Totals read back: phase one's (or the only) visible count, phase two's,
// the candidates phase two found hidden, and those covering no pixel center
// (decision 041); then phase one's and phase two's drawn counts per level of
// detail (decision 047); then the triangles both phases' draws submit.
const TOTALS_BYTES: u64 = 52;

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

/// What GPU presentation needs of a plan batch: its mesh's index count,
/// first index and base vertex in the mesh arena, bounds and joint count,
/// its shadow group, and whether it draws virtual geometry (decision 044).
pub(crate) struct BatchMeta {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub bounds: Aabb,
    pub group: u32,
    pub joints: u32,
    pub virtual_geometry: bool,
    /// Coarser levels of detail the camera may draw instead (decision
    /// 047): each one's first index, index count and geometric error in
    /// mesh units, finest first.
    pub lods: Vec<(u32, u32, f32)>,
    /// The error of its impostor in mesh units (decision 048), 0 without
    /// one.
    pub impostor: f32,
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
    // Occlusion culling (decision 040): phase two's uniform, draw arguments
    // and visible IDs, the hidden count, the totals read back, and the test's
    // group with the pose buffer index, pyramid generation, buffer generation
    // and pose storage generation it binds.
    frame_phase2: wgpu::Buffer,
    draws2: wgpu::Buffer,
    visible_ids2: wgpu::Buffer,
    // Impostors (decision 048): each phase's impostor instances, a visible
    // ID and draw group each, drawn by the argument after every level's.
    impostor_ids: wgpu::Buffer,
    impostor_ids2: wgpu::Buffer,
    occlusion_counts: wgpu::Buffer,
    totals: wgpu::Buffer,
    occlusion_group: Option<(OcclusionKey, wgpu::BindGroup)>,
    // Reflection probe captures (decision 043): a uniform per cube face, the
    // draw arguments and visible IDs faces draw from in turn, each face's
    // cull and compaction groups over the buffer and pose storage
    // generations and pose buffer index they bind, and this frame's uniform
    // they start from.
    frame_probes: [wgpu::Buffer; 6],
    draws_probe: wgpu::Buffer,
    visible_ids_probe: wgpu::Buffer,
    probe_groups: Vec<(ProbeGroupsKey, Vec<ProbeGroups>)>,
    probe_current: usize,
    frame_uniform: Option<PresentationFrame>,
    // Each slot's still frames while probes need them, the slots they
    // cover (0 without probes), whether this frame counts them (only while
    // captures wait), and the frames counted in a row, up to STILL_FRAMES.
    still_frames: wgpu::Buffer,
    still_slots: u64,
    still_counting: bool,
    still_counted: u32,
    // Advances when `still_frames` is replaced.
    still_generation: u64,
    // This frame draws in two phases.
    occlusion: bool,
    // This frame leaves candidates covering no pixel center undrawn
    // (decision 041).
    subpixel: bool,
    bind_groups: Option<BindGroups>,
    // Advances when candidate, batch or scan storage is replaced.
    buffers_generation: u64,
    // Identity of the pose storage bound for writing; rebuild groups on change.
    poses_generation: Option<u64>,
    // The pose buffer this frame's culling writes.
    poses_index: usize,
    uploaded: Option<(u64, u64)>,
    groups_key: Option<(u64, u64, u64, u64, bool, u64, u64)>,
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
    // Skinned slots among the candidates (decision 036), with each draw
    // group's joint count, and their world bounds from this frame's
    // palettes.
    skinned_capacity: usize,
    skinned: wgpu::Buffer,
    skin_bounds: wgpu::Buffer,
    skinned_count: u32,
    skinned_staging: Vec<SkinnedData>,
    batch_joints: Vec<u32>,
    // The retained stream and joints revision the joint buffers hold.
    joints_revision: Option<(u64, u64)>,
    // Whether draws' arguments name their visible-ID regions as first
    // instances (`indirect-first-instance`).
    first_instance: bool,
    readbacks: Vec<VisibleReadback>,
    // Current poses staged between frames for the next gather, with the
    // `GpuPresentation::current_version` they were copied at.
    prestaged: Option<(Prestaged, (u64, u64))>,
    last_visible: Rc<Cell<Option<u32>>>,
    last_occluded: Rc<Cell<Option<u32>>>,
    last_subpixel: Rc<Cell<Option<u32>>>,
    last_levels: Rc<Cell<Option<[u32; 4]>>>,
    last_triangles: Rc<Cell<Option<u32>>>,
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
            crate::pipelines::compute(
                device,
                &wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        let pipelines = Pipelines {
            shift_previous: pipeline("shift_previous"),
            compose_parents: pipeline("compose_parents"),
            cull: pipeline("cull"),
            skin_palette: pipeline("skin_palette"),
            skin_bounds: pipeline("skin_bounds_pass"),
            sum_groups: pipeline("sum_groups"),
            scan_groups: pipeline("scan_groups"),
            batch_args: pipeline("batch_args"),
            scatter: pipeline("scatter"),
            occlusion_test: pipeline("occlusion_test"),
            count_still: pipeline("count_still"),
            probe_cull: pipeline("probe_cull"),
        };
        let uniform = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size_of::<PresentationFrame>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let frame = uniform("engine/presentation-frame");
        let frame_phase2 = uniform("engine/presentation-frame-phase2");
        let frame_probes = std::array::from_fn(|_| uniform("engine/presentation-frame-probe"));
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
            frame_phase2,
            draws2: storage(
                device,
                "engine/presentation-draws2",
                0,
                wgpu::BufferUsages::INDIRECT,
            ),
            visible_ids2: storage(
                device,
                "engine/presentation-visible-ids2",
                0,
                wgpu::BufferUsages::VERTEX,
            ),
            impostor_ids: storage(
                device,
                "engine/presentation-impostor-ids",
                0,
                wgpu::BufferUsages::VERTEX,
            ),
            impostor_ids2: storage(
                device,
                "engine/presentation-impostor-ids2",
                0,
                wgpu::BufferUsages::VERTEX,
            ),
            occlusion_counts: storage(
                device,
                "engine/presentation-occlusion-counts",
                16,
                wgpu::BufferUsages::COPY_SRC,
            ),
            totals: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/presentation-totals"),
                size: TOTALS_BYTES,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            occlusion_group: None,
            frame_probes,
            draws_probe: storage(
                device,
                "engine/presentation-draws-probe",
                0,
                wgpu::BufferUsages::INDIRECT,
            ),
            visible_ids_probe: storage(
                device,
                "engine/presentation-visible-ids-probe",
                0,
                wgpu::BufferUsages::VERTEX,
            ),
            probe_groups: Vec::new(),
            probe_current: 0,
            frame_uniform: None,
            still_frames: storage(
                device,
                "engine/presentation-still-frames",
                0,
                wgpu::BufferUsages::empty(),
            ),
            still_slots: 0,
            still_counting: false,
            still_counted: 0,
            still_generation: 0,
            occlusion: false,
            subpixel: false,
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
            skinned_capacity: 0,
            skinned: empty("engine/presentation-skinned"),
            skin_bounds: empty("engine/presentation-skin-bounds"),
            skinned_count: 0,
            skinned_staging: Vec::new(),
            batch_joints: Vec::new(),
            joints_revision: None,
            first_instance: device
                .features()
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
            readbacks: Vec::new(),
            prestaged: None,
            last_visible: Rc::new(Cell::new(None)),
            last_occluded: Rc::new(Cell::new(None)),
            last_subpixel: Rc::new(Cell::new(None)),
            last_levels: Rc::new(Cell::new(None)),
            last_triangles: Rc::new(Cell::new(None)),
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
            &self.draws2,
            &self.visible_ids2,
            &self.draws_probe,
            &self.visible_ids_probe,
            &self.impostor_ids,
            &self.impostor_ids2,
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

    /// Latest count of visible instances occlusion culling left undrawn
    /// (decision 040), read back with [`Self::visible`].
    pub(crate) fn occluded(&self) -> Option<u32> {
        self.last_occluded.get()
    }

    /// Visible instances left undrawn because their bounds cover no pixel
    /// center (decision 041), read back with [`Self::visible`].
    pub(crate) fn subpixel(&self) -> Option<u32> {
        self.last_subpixel.get()
    }

    /// Instances drawn at each level of detail (decision 047), read back
    /// with [`Self::visible`]: the mesh, its two simplifications, then
    /// impostors.
    /// Triangles the camera's mesh draws submitted, levels of detail and
    /// impostors included, read back with `visible`.
    pub(crate) fn triangles(&self) -> Option<u32> {
        self.last_triangles.get()
    }

    pub(crate) fn lod_instances(&self) -> Option<[u32; 4]> {
        self.last_levels.get()
    }

    /// Whether this frame draws in two phases around the occlusion test.
    pub(crate) fn occlusion(&self) -> bool {
        self.occlusion
    }

    /// Whether this frame leaves candidates covering no pixel center
    /// undrawn (decision 041).
    pub(crate) fn culls_subpixel(&self) -> bool {
        self.subpixel
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
            // Flags, four sums per group (one per level of detail, decision
            // 047), four totals and their sum.
            self.scan = storage(
                device,
                "engine/presentation-scan",
                (capacity + 4 * (groups + 1) + 1) * 4,
                wgpu::BufferUsages::COPY_SRC,
            );
            // A plane of visible IDs per level of detail.
            let planes = LOD_LEVELS as u64;
            self.visible_ids = storage(
                device,
                "engine/presentation-visible-ids",
                planes * capacity * 4,
                wgpu::BufferUsages::VERTEX,
            );
            self.visible_ids2 = storage(
                device,
                "engine/presentation-visible-ids2",
                planes * capacity * 4,
                wgpu::BufferUsages::VERTEX,
            );
            self.visible_ids_probe = storage(
                device,
                "engine/presentation-visible-ids-probe",
                planes * capacity * 4,
                wgpu::BufferUsages::VERTEX,
            );
            // Impostor instances (decision 048): two words each.
            self.impostor_ids = storage(
                device,
                "engine/presentation-impostor-ids",
                capacity * 8,
                wgpu::BufferUsages::VERTEX,
            );
            self.impostor_ids2 = storage(
                device,
                "engine/presentation-impostor-ids2",
                capacity * 8,
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
            // Each batch's visible count before it, per level of detail.
            self.batch_bases = storage(
                device,
                "engine/presentation-batch-bases",
                capacity * 16,
                wgpu::BufferUsages::empty(),
            );
            // Draw arguments per level of detail and batch, then the
            // impostors' (decision 048).
            let args = (LOD_LEVELS as u64 * capacity + 1) * DRAW_ARGS_BYTES;
            self.draws = storage(
                device,
                "engine/presentation-draws",
                args,
                wgpu::BufferUsages::INDIRECT,
            );
            self.draws2 = storage(
                device,
                "engine/presentation-draws2",
                args,
                wgpu::BufferUsages::INDIRECT,
            );
            self.draws_probe = storage(
                device,
                "engine/presentation-draws-probe",
                args,
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
            skinned: &self.skinned,
            skin_bounds: &self.skin_bounds,
            skinned_count: self.skinned_count,
            generation: self.buffers_generation,
        }
    }

    fn bind_groups(
        &mut self,
        device: &wgpu::Device,
        poses: &wgpu::Buffer,
        palette: &wgpu::Buffer,
        colors: &wgpu::Buffer,
        index: usize,
    ) {
        if let Some(groups) = &self.bind_groups {
            if groups.cull[index].is_none() {
                let cull = self.cull_group(device, poses);
                self.bind_groups.as_mut().expect("checked above").cull[index] = Some(cull);
            }
            if self
                .bind_groups
                .as_ref()
                .expect("checked above")
                .count_still[index]
                .is_none()
            {
                let group = self.count_still_group(device, poses);
                self.bind_groups
                    .as_mut()
                    .expect("checked above")
                    .count_still[index] = group;
            }
            if self.bind_groups.as_ref().expect("checked above").skin[index].is_none() {
                let skin = self.skin_group(device, palette);
                self.bind_groups.as_mut().expect("checked above").skin[index] = Some(skin);
            }
            if self
                .bind_groups
                .as_ref()
                .expect("checked above")
                .skin_bounds[index]
                .is_none()
            {
                let bounds = self.skin_bounds_group(device, poses, palette, colors);
                self.bind_groups
                    .as_mut()
                    .expect("checked above")
                    .skin_bounds[index] = Some(bounds);
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
        let mut skin_bounds = [None, None];
        skin_bounds[index] = Some(self.skin_bounds_group(device, poses, palette, colors));
        let mut count_still = [None, None];
        count_still[index] = self.count_still_group(device, poses);
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
            skin_bounds,
            sum_groups: group(
                &pipelines.sum_groups,
                "engine/presentation-sum-group",
                &[
                    entry(0, &self.frame),
                    entry(9, &self.scan),
                    entry(19, &self.occlusion_counts),
                ],
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
                    entry(19, &self.occlusion_counts),
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
                    entry(24, &self.impostor_ids),
                ],
            ),
            sum_groups2: group(
                &pipelines.sum_groups,
                "engine/presentation-sum-group2",
                &[
                    entry(0, &self.frame_phase2),
                    entry(9, &self.scan),
                    entry(19, &self.occlusion_counts),
                ],
            ),
            scan_groups2: group(
                &pipelines.scan_groups,
                "engine/presentation-scan-group2",
                &[entry(0, &self.frame_phase2), entry(9, &self.scan)],
            ),
            batch_args2: group(
                &pipelines.batch_args,
                "engine/presentation-args-group2",
                &[
                    entry(0, &self.frame_phase2),
                    entry(7, &self.batches),
                    entry(9, &self.scan),
                    entry(10, &self.batch_bases),
                    entry(12, &self.draws2),
                    entry(19, &self.occlusion_counts),
                ],
            ),
            scatter2: group(
                &pipelines.scatter,
                "engine/presentation-scatter-group2",
                &[
                    entry(0, &self.frame_phase2),
                    entry(6, &self.candidates),
                    entry(7, &self.batches),
                    entry(9, &self.scan),
                    entry(10, &self.batch_bases),
                    entry(11, &self.visible_ids2),
                    entry(24, &self.impostor_ids2),
                ],
            ),
            count_still,
        });
    }

    // Counting slots' still frames over `poses` while probes need them.
    fn count_still_group(
        &self,
        device: &wgpu::Device,
        poses: &wgpu::Buffer,
    ) -> Option<wgpu::BindGroup> {
        (self.still_slots != 0).then(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/presentation-count-still-group"),
                layout: &self.pipelines.count_still.get_bind_group_layout(0),
                entries: &[
                    entry(0, &self.frame),
                    entry(8, poses),
                    entry(9, &self.scan),
                    entry(14, &self.slot_candidates),
                    entry(23, &self.still_frames),
                ],
            })
        })
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

    fn skin_bounds_group(
        &self,
        device: &wgpu::Device,
        poses: &wgpu::Buffer,
        palette: &wgpu::Buffer,
        colors: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/presentation-skin-bounds-group"),
            layout: &self.pipelines.skin_bounds.get_bind_group_layout(0),
            entries: &[
                entry(0, &self.frame),
                entry(7, &self.batches),
                entry(8, poses),
                entry(9, &self.scan),
                entry(17, palette),
                entry(20, colors),
                entry(21, &self.skinned),
                entry(22, &self.skin_bounds),
            ],
        })
    }

    /// Uploads skinned slots' joints (decision 036) when they changed.
    fn prepare_joints(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        joints: &[RenderJoint],
        revision: (u64, u64),
    ) {
        self.joint_count = joints.len() as u32;
        // The frame's retained stream and joints revision name its joints.
        if self.joints_revision == Some(revision) && joints.len() <= self.joint_capacity {
            return;
        }
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
        self.joints_revision = Some(revision);
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
    /// draw group and its mesh's joint count; `shadow_groups` distinguishes
    /// plans whose batch records carry shadow groups. Skinned slots'
    /// palettes (decision 036) are written into `palette`, of the pose
    /// storage generation, from the frame's joints, and bound with their
    /// offsets in the slot `colors`. With `occlusion`
    /// (the viewport's width and height and the depth pyramid's levels),
    /// frames that cull without an instance budget draw in two phases around
    /// the occlusion test (decision 040), and with `subpixel` they leave
    /// candidates covering no pixel center of the `viewport` (its width and
    /// height) undrawn (decision 041).
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
        batch_meta: impl Fn(usize, &DrawBatch) -> BatchMeta,
        shadow_groups: bool,
        poses: &wgpu::Buffer,
        palette: &wgpu::Buffer,
        colors: &wgpu::Buffer,
        poses_index: usize,
        poses_generation: u64,
        view_projection: glam::Mat4,
        eye: glam::Vec3,
        motion: bool,
        occlusion: Option<(u32, u32, u32)>,
        subpixel: bool,
        lod_error: f32,
        impostors: bool,
        stillness: Option<bool>,
        virtual_generation: u64,
        impostor_generation: u64,
        viewport: (u32, u32),
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
        self.prepare_joints(
            device,
            queue,
            &frame.snapshot().joints,
            (frame.identity(), frame.joints_revision()),
        );
        if self.poses_generation != Some(poses_generation) {
            self.poses_generation = Some(poses_generation);
            self.bind_groups = None;
        }
        self.ensure_still_frames(device, stillness, poses);
        self.bind_groups(device, poses, palette, colors, poses_index);
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
        // Which batches draw virtual geometry (decision 044) and which have
        // impostors (decision 048), by the generations of those choices.
        let key = (
            frame.identity(),
            frame.structure_revision(),
            frame.source_revision(),
            gpu.identity,
            shadow_groups,
            virtual_generation,
            impostor_generation,
        );
        if self.groups_key != Some(key) {
            let slot_nodes = frame.slot_nodes();
            self.candidate_staging.clear();
            self.candidate_staging.reserve_exact(candidates);
            self.batch_staging.clear();
            self.batch_staging.reserve_exact(plan.batches.len());
            self.batch_joints.clear();
            for (index, batch) in plan.batches.iter().enumerate() {
                let BatchMeta {
                    index_count,
                    first_index,
                    base_vertex,
                    bounds,
                    group,
                    joints,
                    virtual_geometry,
                    lods,
                    impostor,
                } = batch_meta(index, batch);
                let lod = |level: usize| lods.get(level).copied().unwrap_or((0, 0, 0.0));
                let skinned = joints != 0;
                self.batch_joints.push(joints);
                let range = batch.instances.start as usize..batch.instances.end as usize;
                self.batch_staging.push(BatchData {
                    start: batch.instances.start,
                    count: batch.instances.end - batch.instances.start,
                    index_count,
                    first_index,
                    base_vertex,
                    first_instance: if self.first_instance {
                        batch.instances.start
                    } else {
                        0
                    },
                    first_plane: if self.first_instance {
                        self.candidate_capacity as u32
                    } else {
                        0
                    },
                    lod_ranges: [lod(0).0, lod(0).1, lod(1).0, lod(1).1],
                    lod_errors: [lod(0).2, lod(1).2, (1 + lods.len()) as f32, impostor],
                    pad: group,
                    bounds_min: bounds
                        .min()
                        .extend(if skinned { 1.0 } else { 0.0 })
                        .to_array(),
                    // Transparent groups are never occluded (decision 040),
                    // nor are instances drawing virtual geometry (decision
                    // 044, 2): their clusters draw in phase one.
                    bounds_max: bounds
                        .max()
                        .extend(if batch.transparent {
                            1.0
                        } else if virtual_geometry {
                            2.0
                        } else {
                            0.0
                        })
                        .to_array(),
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
            // Skinned slots (decision 036), in slot order.
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
                self.bind_groups = None;
                self.buffers_generation += 1;
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
            prepared.group_upload_bytes =
                (2 * self.candidate_staging.len() * size_of::<CandidateData>()
                    + self.batch_staging.len() * size_of::<BatchData>()
                    + self.skinned_staging.len() * size_of::<SkinnedData>()) as u64;
            self.candidate_count = candidates as u32;
            self.batch_count = plan.batches.len() as u32;
            self.groups_key = Some(key);
        }
        prepared.candidates = self.candidate_count;
        // Groups replaced with grown skinned storage.
        self.bind_groups(device, poses, palette, colors, poses_index);

        // A view-projection the CPU frustum rejects disables culling there too.
        let (planes, absolute) = frustum_uniform(view_projection);
        let bounded = absolute
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value <= MAX_COEFFICIENT)
            && planes.iter().flatten().all(|value| value.is_finite());
        let culling = frame.culling() && bounded;
        // Occlusion culling needs culling and no instance budget: a budget
        // keeps the first visible instances of one submission order.
        self.occlusion = occlusion.is_some()
            && culling
            && self.candidate_count != 0
            && frame.instance_budget() >= self.candidate_count as usize;
        // Sub-pixel culling needs culling and no instance budget, and it
        // measures pixels of the viewport occlusion culling sizes.
        self.subpixel = subpixel
            && culling
            && self.candidate_count != 0
            && frame.instance_budget() >= self.candidate_count as usize;
        let (width, height, levels) = occlusion.unwrap_or((viewport.0, viewport.1, 0));
        let mut uniform = PresentationFrame {
            alpha: gpu.alpha,
            candidates: self.candidate_count,
            budget: frame.instance_budget().min(u32::MAX as usize) as u32,
            culling: u32::from(culling),
            groups: self.candidate_count.div_ceil(SCAN_GROUP),
            batches: self.batch_count,
            node_capacity: self.node_capacity as u32,
            blend: u32::from(self.blend),
            planes,
            absolute_view_projection: absolute,
            motion: u32::from(motion),
            joints: self.joint_count,
            mask: if self.occlusion {
                FLAG_PHASE1
            } else {
                FLAG_VISIBLE
            },
            occlusion: u32::from(self.occlusion),
            view_projection: view_projection.to_cols_array_2d(),
            viewport: [width as f32, height as f32, levels as f32, 0.0],
            skinned: self.skinned_count,
            subpixel: u32::from(self.subpixel),
            plane: self.candidate_capacity as u32,
            // Levels of detail need culling and no instance budget, which
            // keeps the first visible instances of one level.
            lod_error: if culling && frame.instance_budget() >= self.candidate_count as usize {
                lod_error
            } else {
                0.0
            },
            eye: eye.extend(if impostors { 1.0 } else { 0.0 }).to_array(),
        };
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&uniform));
        self.frame_uniform = Some(uniform);
        if self.occlusion {
            uniform.mask = FLAG_PHASE2;
            queue.write_buffer(&self.frame_phase2, 0, bytemuck::bytes_of(&uniform));
        }
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
        // Hidden and sub-pixel counts start at zero every frame.
        if candidates {
            encoder.clear_buffer(&self.occlusion_counts, 0, None);
        }
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
                // Still frames for reflection probes (decision 043).
                if let Some(group) = groups.count_still[self.poses_index]
                    .as_ref()
                    .filter(|_| self.still_counting)
                {
                    pass.set_pipeline(&self.pipelines.count_still);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                // Skinned slots' bounds from their palettes, culled.
                if self.skinned_count != 0 {
                    pass.set_pipeline(&self.pipelines.skin_bounds);
                    pass.set_bind_group(
                        0,
                        groups.skin_bounds[self.poses_index]
                            .as_ref()
                            .expect("bound in prepare"),
                        &[],
                    );
                    pass.dispatch_workgroups(self.skinned_count.div_ceil(SKINNED_GROUP), 1, 1);
                }
                pass.set_pipeline(&self.pipelines.sum_groups);
                pass.set_bind_group(0, &groups.sum_groups, &[]);
                pass.dispatch_workgroups(self.candidate_count.div_ceil(SCAN_GROUP), 1, 1);
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
                pass.dispatch_workgroups(self.candidate_count.div_ceil(SCAN_GROUP), 1, 1);
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
            // The (phase one) visible total; frames without occlusion
            // culling read it back now, without waiting for it.
            encoder.clear_buffer(&self.totals, 0, None);
            encoder.copy_buffer_to_buffer(&self.scan, self.total_offset(), &self.totals, 0, 4);
            encoder.copy_buffer_to_buffer(&self.scan, self.level_offset(), &self.totals, 16, 16);
            // Phase one's group sums counted sub-pixel candidates.
            encoder.copy_buffer_to_buffer(&self.occlusion_counts, 4, &self.totals, 12, 4);
            if !self.occlusion {
                encoder.copy_buffer_to_buffer(&self.occlusion_counts, 8, &self.totals, 48, 4);
                self.read_back_totals(device, encoder);
            }
        }
        timed
    }

    // Byte offset of the compaction's total in the scan buffer.
    // The visible total over every level of detail, after the per-level
    // group sums and totals.
    fn total_offset(&self) -> u64 {
        self.level_offset() + 16
    }

    // The compaction's totals per level of detail, after the per-level
    // group sums.
    fn level_offset(&self) -> u64 {
        u64::from(self.candidate_count + 4 * self.candidate_count.div_ceil(SCAN_GROUP)) * 4
    }

    fn read_back_totals(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let totals = self.totals.clone();
        if let Some(readback) = self.idle_readback(device) {
            encoder.copy_buffer_to_buffer(&totals, 0, &readback.buffer, 0, TOTALS_BYTES);
            readback.state.set(Readback::Recorded);
        }
    }

    /// Occlusion culling (decision 040) after phase one drew and `pyramid`
    /// (its view and generation) holds its depths: tests every
    /// frustum-visible candidate, then compacts phase two's draws, and reads
    /// back both phases' totals and the hidden count. With `timestamp` (a
    /// query set and index), the pass writes it when it ends.
    pub(crate) fn encode_occlusion(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pyramid: (&wgpu::TextureView, u64),
        poses: &wgpu::Buffer,
        timestamp: Option<(&wgpu::QuerySet, u32)>,
    ) {
        if !self.occlusion {
            return;
        }
        let Some(groups) = self.bind_groups.as_ref() else {
            return;
        };
        let key = (
            self.poses_index,
            pyramid.1,
            self.buffers_generation,
            self.poses_generation,
        );
        if self
            .occlusion_group
            .as_ref()
            .is_none_or(|(built, _)| *built != key)
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/presentation-occlusion-group"),
                layout: &self.pipelines.occlusion_test.get_bind_group_layout(0),
                entries: &[
                    entry(0, &self.frame),
                    entry(7, &self.batches),
                    entry(8, poses),
                    entry(9, &self.scan),
                    entry(14, &self.slot_candidates),
                    wgpu::BindGroupEntry {
                        binding: 18,
                        resource: wgpu::BindingResource::TextureView(pyramid.0),
                    },
                    entry(19, &self.occlusion_counts),
                ],
            });
            self.occlusion_group = Some((key, group));
        }
        let workgroups = self.candidate_count.div_ceil(GROUP);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/presentation-occlusion"),
                timestamp_writes: timestamp.map(|(query_set, index)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: None,
                        end_of_pass_write_index: Some(index),
                    }
                }),
            });
            pass.set_pipeline(&self.pipelines.occlusion_test);
            pass.set_bind_group(
                0,
                &self.occlusion_group.as_ref().expect("built above").1,
                &[],
            );
            pass.dispatch_workgroups(workgroups, 1, 1);
            // Phase two's compaction over its own mask and outputs.
            let scan_workgroups = self.candidate_count.div_ceil(SCAN_GROUP);
            pass.set_pipeline(&self.pipelines.sum_groups);
            pass.set_bind_group(0, &groups.sum_groups2, &[]);
            pass.dispatch_workgroups(scan_workgroups, 1, 1);
            pass.set_pipeline(&self.pipelines.scan_groups);
            pass.set_bind_group(0, &groups.scan_groups2, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.pipelines.batch_args);
            pass.set_bind_group(0, &groups.batch_args2, &[]);
            pass.dispatch_workgroups(self.batch_count.div_ceil(ARGS_GROUP), 1, 1);
            pass.set_pipeline(&self.pipelines.scatter);
            pass.set_bind_group(0, &groups.scatter2, &[]);
            pass.dispatch_workgroups(scan_workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.scan, self.total_offset(), &self.totals, 4, 4);
        encoder.copy_buffer_to_buffer(&self.scan, self.level_offset(), &self.totals, 32, 16);
        encoder.copy_buffer_to_buffer(&self.occlusion_counts, 0, &self.totals, 8, 4);
        encoder.copy_buffer_to_buffer(&self.occlusion_counts, 8, &self.totals, 48, 4);
        self.read_back_totals(device, encoder);
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
                size: TOTALS_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Rc::new(Cell::new(Readback::Idle)),
            value: Rc::clone(&self.last_visible),
            occluded: Rc::clone(&self.last_occluded),
            subpixel: Rc::clone(&self.last_subpixel),
            levels: Rc::clone(&self.last_levels),
            triangles: Rc::clone(&self.last_triangles),
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
            let occluded = Rc::clone(&readback.occluded);
            let subpixel = Rc::clone(&readback.subpixel);
            let levels = Rc::clone(&readback.levels);
            let triangles = Rc::clone(&readback.triangles);
            let buffer = readback.buffer.clone();
            readback
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        if let Ok(view) = buffer.slice(..).get_mapped_range() {
                            let word = |index: usize| {
                                let bytes: [u8; 4] =
                                    view[index * 4..index * 4 + 4].try_into().unwrap_or([0; 4]);
                                u32::from_le_bytes(bytes)
                            };
                            // Visible: drawn in either phase, hidden, or
                            // covering no pixel center.
                            let hidden = word(2);
                            let tiny = word(3);
                            value.set(Some(
                                word(0)
                                    .saturating_add(word(1))
                                    .saturating_add(hidden)
                                    .saturating_add(tiny),
                            ));
                            occluded.set(Some(hidden));
                            subpixel.set(Some(tiny));
                            levels.set(Some(std::array::from_fn(|level| {
                                word(4 + level).saturating_add(word(8 + level))
                            })));
                            triangles.set(Some(word(12)));
                        }
                        buffer.unmap();
                    }
                    state.set(Readback::Idle);
                });
        }
    }

    /// Keeps slots' still frames for reflection probes (decision 043) over
    /// every slot of `poses` while `stillness` is some, counting them this
    /// frame when it holds true (captures wait): storage replaced or
    /// released, and counting resumed, start the count over.
    fn ensure_still_frames(
        &mut self,
        device: &wgpu::Device,
        stillness: Option<bool>,
        poses: &wgpu::Buffer,
    ) {
        let slots = if stillness.is_some() {
            poses.size() / size_of::<PoseRows>() as u64
        } else {
            0
        };
        if slots > self.still_slots || (slots == 0 && self.still_slots != 0) {
            self.still_frames = storage(
                device,
                "engine/presentation-still-frames",
                slots * 8,
                wgpu::BufferUsages::empty(),
            );
            self.still_slots = slots;
            self.still_counted = 0;
            self.still_generation += 1;
            self.bind_groups = None;
            self.probe_groups.clear();
        }
        let counting = stillness == Some(true);
        if counting && !self.still_counting {
            self.still_counted = 0;
        }
        self.still_counting = counting;
        if counting {
            self.still_counted = (self.still_counted + 1).min(STILL_FRAMES);
        }
    }

    /// Each slot's pose fingerprint and still frames (decision 043), and the
    /// generation of their storage.
    pub(crate) fn still_frames(&self) -> (&wgpu::Buffer, u64) {
        (&self.still_frames, self.still_generation)
    }

    /// Whether culling has counted still frames long enough for reflection
    /// probe captures to find every still slot (decision 043).
    pub(crate) fn stillness_settled(&self) -> bool {
        self.still_counting && self.still_counted >= STILL_FRAMES
    }

    /// Records the cull and compaction of reflection probe cube face `face`
    /// (decision 043), `size` texels across, seen through `view_projection`,
    /// over `poses` (this frame's culling wrote them), after the frame's own
    /// drawing; returns whether [`Self::probe_draw_source`] can draw it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_probe_face(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        face: usize,
        size: u32,
        view_projection: glam::Mat4,
        poses: &wgpu::Buffer,
    ) -> bool {
        let Some(mut uniform) = self.frame_uniform else {
            return false;
        };
        if self.bind_groups.is_none() || self.candidate_count == 0 {
            return false;
        }
        let (planes, absolute) = frustum_uniform(view_projection);
        uniform.planes = planes;
        uniform.absolute_view_projection = absolute;
        uniform.view_projection = view_projection.to_cols_array_2d();
        uniform.culling = 1;
        uniform.mask = FLAG_PROBE;
        uniform.occlusion = 0;
        // Candidates covering no texel center draw nothing (decision 041).
        uniform.subpixel = 1;
        uniform.viewport = [size as f32, size as f32, 0.0, 0.0];
        uniform.motion = 0;
        uniform.budget = u32::MAX;
        // Probe faces draw every candidate at full detail.
        uniform.lod_error = 0.0;
        queue.write_buffer(&self.frame_probes[face], 0, bytemuck::bytes_of(&uniform));
        let key = (
            self.buffers_generation,
            self.poses_generation,
            self.poses_index,
        );
        // Groups per pose buffer: temporally antialiased frames alternate
        // two.
        if let Some(index) = self
            .probe_groups
            .iter()
            .position(|(built, _)| *built == key)
        {
            self.probe_current = index;
        } else {
            // Stale generations never come back.
            self.probe_groups
                .retain(|((buffers, poses, _), _)| (*buffers, *poses) == (key.0, key.1));
            let groups = (0..6)
                .map(|face| self.probe_face_groups(device, face, poses))
                .collect();
            self.probe_groups.push((key, groups));
            self.probe_current = self.probe_groups.len() - 1;
        }
        let groups = &self.probe_groups[self.probe_current].1[face];
        let workgroups = self.candidate_count.div_ceil(GROUP);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/presentation-probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.probe_cull);
        pass.set_bind_group(0, &groups.cull, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        let scan_workgroups = self.candidate_count.div_ceil(SCAN_GROUP);
        pass.set_pipeline(&self.pipelines.sum_groups);
        pass.set_bind_group(0, &groups.sum_groups, &[]);
        pass.dispatch_workgroups(scan_workgroups, 1, 1);
        pass.set_pipeline(&self.pipelines.scan_groups);
        pass.set_bind_group(0, &groups.scan_groups, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.pipelines.batch_args);
        pass.set_bind_group(0, &groups.batch_args, &[]);
        pass.dispatch_workgroups(self.batch_count.div_ceil(ARGS_GROUP), 1, 1);
        pass.set_pipeline(&self.pipelines.scatter);
        pass.set_bind_group(0, &groups.scatter, &[]);
        pass.dispatch_workgroups(scan_workgroups, 1, 1);
        true
    }

    fn probe_face_groups(
        &self,
        device: &wgpu::Device,
        face: usize,
        poses: &wgpu::Buffer,
    ) -> ProbeGroups {
        let frame = &self.frame_probes[face];
        let group = |pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/presentation-probe-group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            })
        };
        let pipelines = &self.pipelines;
        ProbeGroups {
            cull: group(
                &pipelines.probe_cull,
                &[
                    entry(0, frame),
                    entry(7, &self.batches),
                    entry(8, poses),
                    entry(9, &self.scan),
                    entry(14, &self.slot_candidates),
                    entry(23, &self.still_frames),
                ],
            ),
            sum_groups: group(
                &pipelines.sum_groups,
                &[
                    entry(0, frame),
                    entry(9, &self.scan),
                    entry(19, &self.occlusion_counts),
                ],
            ),
            scan_groups: group(
                &pipelines.scan_groups,
                &[entry(0, frame), entry(9, &self.scan)],
            ),
            batch_args: group(
                &pipelines.batch_args,
                &[
                    entry(0, frame),
                    entry(7, &self.batches),
                    entry(9, &self.scan),
                    entry(10, &self.batch_bases),
                    entry(12, &self.draws_probe),
                    // Probe faces count no triangles.
                    entry(19, &self.occlusion_counts),
                ],
            ),
            scatter: group(
                &pipelines.scatter,
                &[
                    entry(0, frame),
                    entry(6, &self.candidates),
                    entry(7, &self.batches),
                    entry(9, &self.scan),
                    entry(10, &self.batch_bases),
                    entry(11, &self.visible_ids_probe),
                    // Probe faces draw no impostors.
                    entry(24, &self.impostor_ids),
                ],
            ),
        }
    }

    /// The visible IDs and indirect arguments of phase one's compaction and
    /// the generation of their storage, which virtual geometry selection
    /// (decision 044) reads.
    pub(crate) fn visible_sources(&self) -> (&wgpu::Buffer, &wgpu::Buffer, u64) {
        (&self.visible_ids, &self.draws, self.buffers_generation)
    }

    /// Visible IDs and indirect arguments of plan batch `index` in the
    /// reflection probe face last recorded (decision 043).
    pub(crate) fn probe_draw_source<'a>(
        &'a self,
        batch: &DrawBatch,
        index: usize,
    ) -> (Ids<'a>, &'a wgpu::Buffer, u64) {
        (
            self.ids(&self.visible_ids_probe, batch, 0),
            &self.draws_probe,
            index as u64 * DRAW_ARGS_BYTES,
        )
    }

    /// Visible IDs and indirect arguments of plan batch `index` in drawing
    /// phase `phase` (1 only after the occlusion test, decision 040).
    pub(crate) fn draw_source<'a>(
        &'a self,
        batch: &DrawBatch,
        index: usize,
        phase: usize,
        level: usize,
    ) -> (Ids<'a>, &'a wgpu::Buffer, u64) {
        let (ids, draws) = if phase == 0 {
            (&self.visible_ids, &self.draws)
        } else {
            (&self.visible_ids2, &self.draws2)
        };
        let args = (level * self.batch_count as usize + index) as u64;
        (self.ids(ids, batch, level), draws, args * DRAW_ARGS_BYTES)
    }

    /// Phase `phase`'s impostor instances (decision 048) and the arguments
    /// of the non-indexed draw of their cards.
    pub(crate) fn impostor_source(&self, phase: usize) -> (&wgpu::Buffer, &wgpu::Buffer, u64) {
        let (ids, draws) = if phase == 0 {
            (&self.impostor_ids, &self.draws)
        } else {
            (&self.impostor_ids2, &self.draws2)
        };
        let args = LOD_LEVELS as u64 * u64::from(self.batch_count);
        (ids, draws, args * DRAW_ARGS_BYTES)
    }

    /// A batch's visible IDs at level of detail `level` in `ids`: the whole
    /// buffer where its draw arguments' first instance names its region,
    /// otherwise the region in the level's plane.
    fn ids<'a>(&self, ids: &'a wgpu::Buffer, batch: &DrawBatch, level: usize) -> Ids<'a> {
        if self.first_instance {
            Ids::Whole(ids)
        } else {
            let plane = (level * self.candidate_capacity) as u64;
            let start = (plane + u64::from(batch.instances.start)) * 4;
            let end = (plane + u64::from(batch.instances.end)) * 4;
            Ids::Region(ids.slice(start..end))
        }
    }
}
