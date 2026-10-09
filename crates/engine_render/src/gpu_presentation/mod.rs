//! GPU presentation: uploads gathered interpolation inputs and
//! candidate draw groups, then records compute passes that blend, compose,
//! validate and cull displayed poses and compact visible slots per group for
//! indirect draws. Per-frame CPU work is a uniform write and a few dispatches;
//! inputs upload only when core's gather sequence advanced (normally once per
//! fixed tick) and groups only when the candidate structure changed. After a
//! quiet tick only current poses upload: a small pass first copies the last
//! current poses into the previous half of the pose buffer.
//!
//! Its state lives in parts that own their storage and advance a generation
//! whenever they replace it: node inputs (`inputs`), draw groups and where
//! compactions write (`groups`), skinned slots' joints (`joints`), still
//! frames and reflection probe faces (`probes`), and the totals read back
//! (`totals`). Bind groups are kept by the generations of what they bind.

mod fades;
mod groups;
mod inputs;
mod joints;
mod layouts;
mod probes;
mod totals;

use engine_core::{Aabb, GpuPresentation, RenderFrame};

use crate::batches::{BatchPlan, DrawBatch};
use crate::browser::FramePoses;
use crate::buffers::entry;
use crate::cached::Cached;
use crate::gpu_data::{FLAG_PHASE1, FLAG_PHASE2, FLAG_VISIBLE, PresentationFrame, frustum_uniform};
use crate::mesh_arena::Ids;
use crate::profiling::{GpuProfiler, Span};
use crate::staging::Prestaged;

use fades::Fades;
use groups::{DrawGroups, PHASE_ONE, PHASE_TWO};
use inputs::NodeInputs;
pub(crate) use inputs::Staging;
use joints::Joints;
use layouts::Shaders;
use probes::{ProbeFaces, Stillness};
use totals::Totals;

/// The stages of a frame's compute pass, which profiling captures time as
/// separate passes: parent composition, culling, scanning (group sums and
/// draw arguments) and the scatter of visible slots.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Compose,
    Cull,
    Scan,
    Scatter,
}

impl Stage {
    /// The profiler's span of the stage.
    fn span(self) -> &'static str {
        match self {
            Self::Compose => "presentation_compose",
            Self::Cull => "presentation_cull",
            Self::Scan => "presentation_scan",
            Self::Scatter => "presentation_scatter",
        }
    }
}

// Candidates per workgroup of the per-candidate passes, and of the
// compaction passes (`GROUP` in presentation.wgsl), whose workgroup arrays
// are kept small for Firefox's compile times.
const GROUP: u32 = 256;
const SCAN_GROUP: u32 = 64;
const PARENT_GROUP: u32 = 64;
const JOINT_GROUP: u32 = 64;
const SKINNED_GROUP: u32 = 64;
const ARGS_GROUP: u32 = 64;
// Beyond this bound the CPU frustum also fails open (see culling.rs).
const MAX_COEFFICIENT: f32 = 1.0e15;
// Bindings need at least one element; exceeds every record stride.
const MIN_STORAGE_BYTES: u64 = 256;

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
}

/// What GPU presentation needs of a plan batch: its mesh's index count,
/// first index and base vertex in the mesh arena, bounds and joint count,
/// its shadow group, and whether it draws virtual geometry.
pub(crate) struct BatchMeta {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub bounds: Aabb,
    pub group: u32,
    pub joints: u32,
    pub virtual_geometry: bool,
    /// Coarser levels of detail the camera may draw instead:
    /// each one's first index, index count and geometric error in
    /// mesh units, finest first.
    pub lods: Vec<(u32, u32, f32)>,
    /// The error of its impostor in mesh units, 0 without
    /// one.
    pub impostor: f32,
}

/// What batch records carry beyond the plan, by the generations of those
/// choices: draw groups rebuild when it changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RecordsKey {
    /// Whether records carry shadow groups.
    pub shadow_groups: bool,
    /// The mesh arena layout and which batches draw virtual geometry.
    pub meshes: u64,
    /// The impostors records name, 0 without them.
    pub impostors: u64,
}

/// What a frame presents: the retained frame and its gather, the plan and
/// what each of its batches' records holds, and the pose storage culling
/// writes displayed poses and skinned slots' palettes into.
pub(crate) struct PresentationInputs<'a> {
    pub frame: &'a RenderFrame<'a>,
    pub gpu: &'a GpuPresentation<'a>,
    pub plan: &'a BatchPlan,
    pub batch_meta: &'a dyn Fn(usize, &DrawBatch) -> BatchMeta,
    pub records: RecordsKey,
    pub poses: FramePoses<'a>,
}

/// How a frame culls: the projection it rasterizes, the eye and the
/// viewport's width and height, and what culling does where culling runs
/// without an instance budget.
pub(crate) struct CullingView {
    pub view_projection: glam::Mat4,
    pub eye: glam::Vec3,
    pub viewport: (u32, u32),
    /// Visible IDs of moving slots carry `MOVED_SLOT`.
    pub motion: bool,
    /// The depth pyramid's levels when frames may draw in two phases around
    /// the occlusion test.
    pub occlusion: Option<u32>,
    /// Candidates covering no pixel center stay undrawn.
    pub subpixel: bool,
    /// The pixel error levels of detail keep within, 0 for
    /// the meshes themselves.
    pub lod_error: f32,
    /// Coarse instances draw as impostors.
    pub impostors: bool,
    /// Whether reflection probes need slots' still frames,
    /// and whether this frame counts them (while captures wait).
    pub stillness: Option<bool>,
    /// The share of a switch between a mesh and its card this frame fades:
    /// 1 or more switches at once.
    pub fade: f32,
}

// Groups over node inputs alone: the previous-pose shift and parent
// composition.
struct NodeGroups {
    shift_previous: wgpu::BindGroup,
    compose_parents: wgpu::BindGroup,
}

// Groups binding one pose buffer: culling, skinned slots' palettes and
// bounds, and the still frame count of reflection probes
// while still frames are kept.
struct PoseGroups {
    cull: wgpu::BindGroup,
    fades: wgpu::BindGroup,
    skin: wgpu::BindGroup,
    skin_bounds: Option<wgpu::BindGroup>,
    count_still: Option<wgpu::BindGroup>,
}

// Node, draw group, joint, still frame and fade storage generations, the
// pose storage generation that one pose buffer's groups bind, and whether
// candidates include skinned slots.
type PoseKey = (u64, u64, u64, u64, u64, u64, bool);

// The occlusion test group's pose buffer index, depth pyramid generation,
// draw group generation and pose storage generation.
type OcclusionKey = (usize, u64, u64, u64);

pub(crate) struct GpuPresentationPass {
    shaders: Shaders,
    // This frame's uniform, phase two's, and the uniform last
    // written, which reflection probe faces start from.
    frame: wgpu::Buffer,
    frame_phase2: wgpu::Buffer,
    frame_uniform: Option<PresentationFrame>,
    nodes: NodeInputs,
    groups: DrawGroups,
    joints: Joints,
    stillness: Stillness,
    fades: Fades,
    probes: ProbeFaces,
    totals: Totals,
    // Each frame's hidden and sub-pixel counts.
    occlusion_counts: wgpu::Buffer,
    // This frame draws in two phases around the occlusion test, and leaves
    // candidates covering no pixel center undrawn.
    occlusion: bool,
    subpixel: bool,
    // The pose buffer this frame's culling writes and its storage
    // generation.
    poses: (usize, u64),
    // Groups over node inputs, the two drawing phases' compactions (by the
    // draw group generation), the occlusion test's, and those of each pose
    // buffer (temporally antialiased frames alternate two).
    node_groups: Cached<u64, NodeGroups>,
    compact_groups: Cached<u64, [wgpu::BindGroup; 2]>,
    // Per pose buffer (temporal frames alternate them).
    occlusion_groups: [Cached<OcclusionKey, wgpu::BindGroup>; 2],
    pose_groups: [Cached<PoseKey, PoseGroups>; 2],
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    crate::buffers::storage(device, label, bytes.max(MIN_STORAGE_BYTES), extra)
}

fn grow(needed: usize) -> usize {
    needed.max(1_024).next_power_of_two()
}

impl GpuPresentationPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shaders = Shaders::new(device);
        let uniform =
            |label| crate::buffers::uniform(device, label, size_of::<PresentationFrame>() as u64);
        Self {
            nodes: NodeInputs::new(device, &shaders.level_layout),
            shaders,
            frame: uniform("engine/presentation-frame"),
            frame_phase2: uniform("engine/presentation-frame-phase2"),
            frame_uniform: None,
            groups: DrawGroups::new(device),
            joints: Joints::new(device),
            stillness: Stillness::new(device),
            fades: Fades::new(device),
            probes: ProbeFaces::new(std::array::from_fn(|_| {
                uniform("engine/presentation-frame-probe")
            })),
            totals: Totals::new(device),
            occlusion_counts: storage(
                device,
                "engine/presentation-occlusion-counts",
                16,
                wgpu::BufferUsages::COPY_SRC,
            ),
            occlusion: false,
            subpixel: false,
            poses: (0, 0),
            node_groups: Cached::default(),
            compact_groups: Cached::default(),
            occlusion_groups: Default::default(),
            pose_groups: Default::default(),
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.nodes.capacity_bytes() + self.groups.capacity_bytes() + self.joints.capacity_bytes()
    }

    /// Latest visible instance count read back from the GPU, a few frames old.
    pub(crate) fn visible(&self) -> Option<u32> {
        self.totals.last().map(|totals| totals.visible)
    }

    /// Latest count of visible instances occlusion culling left undrawn,
    /// read back with [`Self::visible`].
    pub(crate) fn occluded(&self) -> Option<u32> {
        self.totals.last().map(|totals| totals.occluded)
    }

    /// Visible instances left undrawn because their bounds cover no pixel
    /// center, read back with [`Self::visible`].
    pub(crate) fn subpixel(&self) -> Option<u32> {
        self.totals.last().map(|totals| totals.subpixel)
    }

    /// Triangles the camera's mesh draws submitted, levels of detail and
    /// impostors included, read back with [`Self::visible`].
    pub(crate) fn triangles(&self) -> Option<u32> {
        self.totals.last().map(|totals| totals.triangles)
    }

    /// Instances drawn at each level of detail, read back
    /// with [`Self::visible`]: the mesh, its two simplifications, then
    /// impostors.
    pub(crate) fn lod_instances(&self) -> Option<[u32; 4]> {
        self.totals.last().map(|totals| totals.levels)
    }

    /// Whether this frame draws in two phases around the occlusion test.
    pub(crate) fn occlusion(&self) -> bool {
        self.occlusion
    }

    /// Whether this frame leaves candidates covering no pixel center
    /// undrawn.
    pub(crate) fn culls_subpixel(&self) -> bool {
        self.subpixel
    }

    /// This frame's candidates in slot order, draw groups and candidate
    /// flags, for shadow caster selection after culling.
    pub(crate) fn shadow_inputs(&self) -> crate::shadows::Presented<'_> {
        crate::shadows::Presented {
            candidates: &self.groups.slot_candidates,
            batches: &self.groups.batches,
            flags: &self.groups.scan,
            count: self.groups.count(),
            skinned: &self.groups.skinned,
            skin_bounds: &self.groups.skin_bounds,
            skinned_count: self.groups.skinned_count(),
            generation: self.groups.generation(),
        }
    }

    /// Holds current poses staged between frames for the next gather to copy.
    pub(crate) fn prestage(&mut self, prestaged: Prestaged, version: (u64, u64)) {
        self.nodes.prestage(prestaged, version);
    }

    /// Uploads changed inputs and groups, binds `inputs`' pose storage and
    /// writes the frame uniform. Large node uploads record copies into
    /// `staging`'s encoder, submitted before the frame. Culling writes
    /// displayed poses into the frame's pose buffer and tests them against
    /// `view`. Profiling captures time a shift of previous poses
    /// (`presentation_shift`).
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        staging: Staging<'_>,
        inputs: &PresentationInputs<'_>,
        view: &CullingView,
        profiler: &GpuProfiler,
    ) -> GpuPrepared {
        let PresentationInputs {
            frame,
            gpu,
            plan,
            poses,
            ..
        } = inputs;
        let mut prepared = GpuPrepared::default();
        let held = self.nodes.holds(gpu);
        prepared.held = held;
        self.nodes.ensure(
            device,
            gpu.info.len(),
            if held { 0 } else { gpu.parents.len() },
        );
        self.groups
            .ensure(device, plan.ordered_instances.len(), plan.batches.len());
        self.joints.prepare(
            device,
            queue,
            &frame.snapshot().joints,
            (frame.identity(), frame.joints_revision()),
        );
        self.stillness
            .ensure(device, view.stillness, poses.current());
        self.fades
            .ensure(device, poses.current(), self.groups.capacity());
        self.poses = (poses.parity, poses.generation);

        // Node inputs: records newer than the last uploaded gather.
        if let Some(gather) = self.nodes.gather(gpu, held) {
            prepared.full_node_upload = gather.since.is_none();
            if gather.shift {
                self.submit_shift(device, queue, profiler);
                prepared.shifted = true;
            }
            let upload = self.nodes.upload(device, queue, staging, gpu, gather.since);
            prepared.node_upload_bytes = upload.bytes;
            prepared.staged = upload.staged;
            prepared.prestaged = upload.prestaged;
        }
        prepared.parent_levels = self.nodes.level_ranges().len();
        prepared.blend = self.nodes.blend();

        // Candidate groups: rebuilt with the candidate structure or node space.
        if let Some(bytes) = self.groups.rebuild(device, queue, inputs) {
            prepared.group_upload_bytes = bytes;
        }
        prepared.candidates = self.groups.count();
        self.bind(device, poses);
        self.write_frame_uniform(queue, frame, gpu.alpha, view);
        prepared
    }

    /// Copies the last current poses into previous slots, submitted at once:
    /// queued writes and staged copies of this gather take effect at the
    /// next submission, after it.
    fn submit_shift(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, profiler: &GpuProfiler) {
        let groups = self
            .node_groups
            .get_or_insert_with(self.nodes.generation(), || {
                node_groups(device, &self.shaders, &self.frame, &self.nodes)
            });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/presentation-shift"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/presentation-shift"),
                timestamp_writes: profiler.span("presentation_shift").map(Span::compute),
            });
            pass.set_pipeline(&self.shaders.pipelines.shift_previous);
            pass.set_bind_group(0, &groups.shift_previous, &[]);
            // One invocation per node; the pass copies only nodes the last
            // applied gather made active.
            pass.dispatch_workgroups((self.nodes.capacity() as u32).div_ceil(GROUP), 1, 1);
        }
        queue.submit([encoder.finish()]);
    }

    /// Builds the groups this frame binds that storage replaced since, or
    /// that `poses` has not bound yet.
    fn bind(&mut self, device: &wgpu::Device, poses: &FramePoses<'_>) {
        let Self {
            shaders,
            frame,
            frame_phase2,
            nodes,
            groups,
            joints,
            stillness,
            fades,
            occlusion_counts,
            node_groups: node_cache,
            compact_groups,
            pose_groups,
            ..
        } = self;
        node_cache.get_or_insert_with(nodes.generation(), || {
            node_groups(device, shaders, frame, nodes)
        });
        compact_groups.get_or_insert_with(groups.generation(), || {
            [
                ("engine/presentation-compact-group", &*frame, PHASE_ONE),
                (
                    "engine/presentation-compact-group2",
                    &*frame_phase2,
                    PHASE_TWO,
                ),
            ]
            .map(|(label, uniform, phase)| {
                compact_group(
                    device,
                    shaders,
                    groups,
                    occlusion_counts,
                    label,
                    uniform,
                    phase,
                )
            })
        });
        let key = (
            nodes.generation(),
            groups.generation(),
            joints.generation(),
            stillness.generation(),
            fades.generation(),
            poses.generation,
            groups.skinned_count() != 0,
        );
        pose_groups[poses.parity].get_or_insert_with(key, || {
            let current = poses.current();
            let palette = poses.palette();
            let group = |family, label, entries: &[(&str, wgpu::BindingResource<'_>)]| {
                shaders.group(device, family, label, entries)
            };
            PoseGroups {
                cull: group(
                    &shaders.families.cull,
                    "engine/presentation-cull-group",
                    &[
                        entry("frame", frame),
                        entry("locals", &nodes.locals),
                        entry("info", &nodes.info),
                        entry("authority", &nodes.authority),
                        entry("composed", &nodes.composed),
                        entry("batches", &groups.batches),
                        entry("poses", current),
                        entry("scan", &groups.scan),
                        entry("slot_candidates", &groups.slot_candidates),
                    ],
                ),
                fades: group(
                    &shaders.families.fades,
                    "engine/presentation-fades-group",
                    &[
                        entry("frame", frame),
                        entry("batches", &groups.batches),
                        entry("poses", current),
                        entry("scan", &groups.scan),
                        entry("slot_candidates", &groups.slot_candidates),
                        entry("lod_states", &fades.states),
                        entry("fade_cards", &fades.cards),
                    ],
                ),
                skin: group(
                    &shaders.families.compose,
                    "engine/presentation-skin-group",
                    &[
                        entry("frame", frame),
                        entry("locals", &nodes.locals),
                        entry("info", &nodes.info),
                        entry("authority", &nodes.authority),
                        entry("composed", &nodes.composed),
                        entry("joint_nodes", &joints.nodes),
                        entry("inverse_binds", &joints.inverse_binds),
                        entry("palette", palette),
                    ],
                ),
                skin_bounds: (groups.skinned_count() != 0).then(|| {
                    group(
                        shaders.skin_bounds_family(device),
                        "engine/presentation-skin-bounds-group",
                        &[
                            entry("frame", frame),
                            entry("batches", &groups.batches),
                            entry("poses", current),
                            entry("scan", &groups.scan),
                            entry("palette", palette),
                            entry("slot_colors", poses.colors),
                            entry("skinned", &groups.skinned),
                            entry("skin_bounds", &groups.skin_bounds),
                        ],
                    )
                }),
                count_still: stillness.kept().then(|| {
                    group(
                        &shaders.families.tests,
                        "engine/presentation-count-still-group",
                        &[
                            entry("frame", frame),
                            entry("poses", current),
                            entry("scan", &groups.scan),
                            entry("slot_candidates", &groups.slot_candidates),
                            entry("still_frames", &stillness.frames),
                        ],
                    )
                }),
            }
        });
    }

    /// Writes this frame's uniform, and phase two's when it draws two
    /// phases: occlusion culling, sub-pixel culling and levels of detail run
    /// as `view` asks only where culling runs without an instance budget,
    /// which keeps the first visible instances of one submission order.
    fn write_frame_uniform(
        &mut self,
        queue: &wgpu::Queue,
        frame: &RenderFrame<'_>,
        alpha: f32,
        view: &CullingView,
    ) {
        // A view-projection the CPU frustum rejects disables culling there too.
        let (planes, absolute) = frustum_uniform(view.view_projection);
        let bounded = absolute
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value <= MAX_COEFFICIENT)
            && planes.iter().flatten().all(|value| value.is_finite());
        let culling = frame.culling() && bounded;
        let candidates = self.groups.count();
        let unbudgeted = culling && frame.instance_budget() >= candidates as usize;
        self.occlusion = view.occlusion.is_some() && unbudgeted && candidates != 0;
        // Sub-pixel culling measures pixels of the viewport occlusion culling
        // sizes.
        self.subpixel = view.subpixel && unbudgeted && candidates != 0;
        let (width, height) = view.viewport;
        let mut uniform = PresentationFrame {
            alpha,
            candidates,
            budget: frame.instance_budget().min(u32::MAX as usize) as u32,
            culling: u32::from(culling),
            groups: candidates.div_ceil(SCAN_GROUP),
            batches: self.groups.batch_count(),
            node_capacity: self.nodes.capacity() as u32,
            blend: u32::from(self.nodes.blend()),
            planes,
            absolute_view_projection: absolute,
            motion: u32::from(view.motion),
            joints: self.joints.count(),
            mask: if self.occlusion {
                FLAG_PHASE1
            } else {
                FLAG_VISIBLE
            },
            occlusion: u32::from(self.occlusion),
            view_projection: view.view_projection.to_cols_array_2d(),
            viewport: [
                width as f32,
                height as f32,
                view.occlusion.unwrap_or(0) as f32,
                0.0,
            ],
            skinned: self.groups.skinned_count(),
            subpixel: u32::from(self.subpixel),
            plane: self.groups.capacity() as u32,
            lod_error: if unbudgeted { view.lod_error } else { 0.0 },
            eye: view
                .eye
                .extend(if view.impostors { 1.0 } else { 0.0 })
                .to_array(),
            fade: [view.fade, fades::FADE_MARGIN, 0.0, 0.0],
        };
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&uniform));
        self.fades.begin_frame(queue);
        self.frame_uniform = Some(uniform);
        if self.occlusion {
            uniform.mask = FLAG_PHASE2;
            queue.write_buffer(&self.frame_phase2, 0, bytemuck::bytes_of(&uniform));
        }
    }

    /// Records blending, composition, culling and compaction in one compute
    /// pass; profiling captures time each stage in its own pass instead.
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &GpuProfiler,
    ) {
        let (Some(nodes), Some(compact), Some(posed)) = (
            self.node_groups.get(),
            self.compact_groups.get(),
            self.pose_groups[self.poses.0].get(),
        ) else {
            return;
        };
        let shaders = &self.shaders;
        let pipelines = &shaders.pipelines;
        let candidates = self.groups.count();
        let joints = self.joints.count();
        let skinned = self.groups.skinned_count();
        let lod = self
            .frame_uniform
            .is_some_and(|uniform| uniform.lod_error > 0.0);
        // Hidden and sub-pixel counts start at zero every frame.
        if candidates != 0 {
            encoder.clear_buffer(&self.occlusion_counts, 0, None);
        }
        // Dispatches of each stage, in order.
        let record = |pass: &mut wgpu::ComputePass<'_>, stage: Stage| match stage {
            Stage::Compose => {
                pass.set_pipeline(&pipelines.compose_parents);
                pass.set_bind_group(0, &nodes.compose_parents, &[]);
                for (level, &(start, end)) in self.nodes.level_ranges().iter().enumerate() {
                    if end > start {
                        pass.set_bind_group(1, self.nodes.level_group(level), &[]);
                        pass.dispatch_workgroups((end - start).div_ceil(PARENT_GROUP), 1, 1);
                    }
                }
            }
            Stage::Cull => {
                // Skinned slots' palettes, from the composed parents.
                if joints != 0 {
                    pass.set_pipeline(shaders.skin_palette(device));
                    pass.set_bind_group(0, &posed.skin, &[]);
                    // The shared layout's level group, which palettes ignore.
                    pass.set_bind_group(1, self.nodes.level_group(0), &[]);
                    pass.dispatch_workgroups(joints.div_ceil(JOINT_GROUP), 1, 1);
                }
                // Flags in slot order, then their sums in draw-group order.
                let workgroups = candidates.div_ceil(GROUP);
                pass.set_pipeline(&pipelines.cull);
                pass.set_bind_group(0, &posed.cull, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                // Still frames for reflection probes.
                if let Some(group) = posed
                    .count_still
                    .as_ref()
                    .filter(|_| self.stillness.counting())
                {
                    pass.set_pipeline(shaders.count_still(device));
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                // Skinned slots' bounds from their palettes, culled.
                if let Some(group) = posed.skin_bounds.as_ref().filter(|_| skinned != 0) {
                    pass.set_pipeline(shaders.skin_bounds(device));
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(skinned.div_ceil(SKINNED_GROUP), 1, 1);
                }
                // Levels kept and faded, on frames with them.
                if lod {
                    pass.set_pipeline(&pipelines.fade_levels);
                    pass.set_bind_group(0, &posed.fades, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
                pass.set_pipeline(&pipelines.sum_groups);
                pass.set_bind_group(0, &compact[PHASE_ONE], &[]);
                pass.dispatch_workgroups(candidates.div_ceil(SCAN_GROUP), 1, 1);
            }
            Stage::Scan => {
                pass.set_pipeline(&pipelines.scan_groups);
                pass.set_bind_group(0, &compact[PHASE_ONE], &[]);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&pipelines.batch_args);
                pass.dispatch_workgroups(self.groups.batch_count().div_ceil(ARGS_GROUP), 1, 1);
            }
            Stage::Scatter => {
                pass.set_pipeline(&pipelines.scatter);
                pass.set_bind_group(0, &compact[PHASE_ONE], &[]);
                pass.dispatch_workgroups(candidates.div_ceil(SCAN_GROUP), 1, 1);
            }
        };
        let stages = [
            (Stage::Compose, !self.nodes.level_ranges().is_empty()),
            (Stage::Cull, candidates != 0 || joints != 0),
            (Stage::Scan, candidates != 0),
            (Stage::Scatter, candidates != 0),
        ];
        if profiler.capturing() {
            for (stage, runs) in stages {
                if runs {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("engine/presentation-stage"),
                        timestamp_writes: profiler
                            .span_in("presentation_compute", stage.span())
                            .map(Span::compute),
                    });
                    record(&mut pass, stage);
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
        if candidates != 0 {
            // The (phase one) visible total; frames without occlusion
            // culling read it back now, without waiting for it.
            let (scan, totals) = (&self.groups.scan, &self.totals.buffer);
            encoder.clear_buffer(totals, 0, None);
            encoder.copy_buffer_to_buffer(scan, self.groups.total_offset(), totals, 0, 4);
            encoder.copy_buffer_to_buffer(scan, self.groups.level_offset(), totals, 16, 16);
            // Phase one's group sums counted sub-pixel candidates.
            encoder.copy_buffer_to_buffer(&self.occlusion_counts, 4, totals, 12, 4);
            if !self.occlusion {
                encoder.copy_buffer_to_buffer(&self.occlusion_counts, 8, totals, 48, 4);
                self.totals.read_back(device, encoder);
            }
        }
    }

    /// Occlusion culling after phase one drew and `pyramid`
    /// (its view and generation) holds its depths: tests every
    /// frustum-visible candidate, then compacts phase two's draws, and reads
    /// back both phases' totals and the hidden count. With `span`, the pass
    /// ends it.
    pub(crate) fn encode_occlusion(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pyramid: (&wgpu::TextureView, u64),
        poses: &wgpu::Buffer,
        span: Option<Span<'_>>,
    ) {
        if !self.occlusion {
            return;
        }
        let Some(compact) = self.compact_groups.get() else {
            return;
        };
        let (index, generation) = self.poses;
        let key = (index, pyramid.1, self.groups.generation(), generation);
        let groups = &self.groups;
        let group = self.occlusion_groups[index].get_or_insert_with(key, || {
            self.shaders.group(
                device,
                &self.shaders.families.tests,
                "engine/presentation-occlusion-group",
                &[
                    entry("frame", &self.frame),
                    entry("batches", &groups.batches),
                    entry("poses", poses),
                    entry("scan", &groups.scan),
                    entry("slot_candidates", &groups.slot_candidates),
                    ("depth_pyramid", crate::pipelines::view(pyramid.0)),
                    entry("occlusion_counts", &self.occlusion_counts),
                ],
            )
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/presentation-occlusion"),
                timestamp_writes: span.map(|span| span.compute_part(false, true)),
            });
            pass.set_pipeline(&self.shaders.pipelines.occlusion_test);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups.count().div_ceil(GROUP), 1, 1);
            // Phase two's compaction over its own mask and outputs.
            self.dispatch_compaction(&mut pass, &compact[PHASE_TWO]);
        }
        let (scan, totals) = (&self.groups.scan, &self.totals.buffer);
        encoder.copy_buffer_to_buffer(scan, self.groups.total_offset(), totals, 4, 4);
        encoder.copy_buffer_to_buffer(scan, self.groups.level_offset(), totals, 32, 16);
        encoder.copy_buffer_to_buffer(&self.occlusion_counts, 0, totals, 8, 4);
        encoder.copy_buffer_to_buffer(&self.occlusion_counts, 8, totals, 48, 4);
        self.totals.read_back(device, encoder);
    }

    /// Records a compaction over `group`: group sums, their scan, draw
    /// arguments and the scatter of visible slots.
    fn dispatch_compaction(&self, pass: &mut wgpu::ComputePass<'_>, group: &wgpu::BindGroup) {
        let pipelines = &self.shaders.pipelines;
        let scan_workgroups = self.groups.count().div_ceil(SCAN_GROUP);
        pass.set_pipeline(&pipelines.sum_groups);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(scan_workgroups, 1, 1);
        pass.set_pipeline(&pipelines.scan_groups);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&pipelines.batch_args);
        pass.dispatch_workgroups(self.groups.batch_count().div_ceil(ARGS_GROUP), 1, 1);
        pass.set_pipeline(&pipelines.scatter);
        pass.dispatch_workgroups(scan_workgroups, 1, 1);
    }

    /// Requests maps of read-backs recorded in the frame just submitted.
    pub(crate) fn after_submit(&mut self) {
        self.totals.after_submit();
    }

    /// The visible IDs and indirect arguments of phase one's compaction and
    /// the generation of their storage, which virtual geometry selection
    /// reads.
    pub(crate) fn visible_sources(&self) -> (&wgpu::Buffer, &wgpu::Buffer, u64) {
        (
            &self.groups.visible_ids[PHASE_ONE],
            &self.groups.draws[PHASE_ONE],
            self.groups.generation(),
        )
    }

    /// Visible IDs and indirect arguments of plan batch `index` in drawing
    /// phase `phase` (1 only after the occlusion test).
    pub(crate) fn draw_source<'a>(
        &'a self,
        batch: &DrawBatch,
        index: usize,
        phase: usize,
        level: usize,
    ) -> (Ids<'a>, &'a wgpu::Buffer, u64) {
        self.groups.draw_source(batch, index, phase, level)
    }

    /// Phase `phase`'s impostor instances and the arguments
    /// of the non-indexed draw of their cards.
    pub(crate) fn impostor_source(&self, phase: usize) -> (&wgpu::Buffer, &wgpu::Buffer, u64) {
        self.groups.impostor_source(phase)
    }

    /// The cards fading over their meshes this frame: their
    /// instances and the arguments of their non-indexed draw.
    pub(crate) fn fading_cards(&self) -> (wgpu::BufferSlice<'_>, &wgpu::Buffer, u64) {
        self.fades.source()
    }
}

fn node_groups(
    device: &wgpu::Device,
    shaders: &Shaders,
    frame: &wgpu::Buffer,
    nodes: &NodeInputs,
) -> NodeGroups {
    NodeGroups {
        shift_previous: shaders.group(
            device,
            &shaders.families.tests,
            "engine/presentation-shift-group",
            &[
                entry("frame", frame),
                entry("info", &nodes.info),
                entry("locals_shift", &nodes.locals),
            ],
        ),
        compose_parents: shaders.group(
            device,
            &shaders.families.compose,
            "engine/presentation-compose-group",
            &[
                entry("frame", frame),
                entry("locals", &nodes.locals),
                entry("info", &nodes.info),
                entry("authority", &nodes.authority),
                entry("composed", &nodes.composed),
                entry("parents", &nodes.parents),
            ],
        ),
    }
}

/// A compaction's group: its uniform `frame`, the draw groups, the hidden
/// and sub-pixel `counts`, and compaction `compaction`'s draw arguments,
/// visible IDs and impostor instances.
#[allow(clippy::too_many_arguments)]
fn compact_group(
    device: &wgpu::Device,
    shaders: &Shaders,
    groups: &DrawGroups,
    counts: &wgpu::Buffer,
    label: &str,
    frame: &wgpu::Buffer,
    compaction: usize,
) -> wgpu::BindGroup {
    shaders.group(
        device,
        &shaders.families.compact,
        label,
        &[
            entry("frame", frame),
            entry("candidates", &groups.candidates),
            entry("batches", &groups.batches),
            entry("scan", &groups.scan),
            entry("batch_bases", &groups.batch_bases),
            entry("visible_ids", &groups.visible_ids[compaction]),
            entry("draws", &groups.draws[compaction]),
            entry("occlusion_counts", counts),
            entry("impostor_ids", groups.impostor_ids_of(compaction)),
        ],
    )
}
