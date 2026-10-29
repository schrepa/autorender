mod fades;
mod groups;
mod inputs;
mod layouts;
mod probes;
mod totals;

use engine_core::{Aabb, GpuPresentation, RenderFrame};

use crate::batches::{BatchPlan, DrawBatch};
use crate::buffers::entry;
use crate::cached::Cached;
use crate::gpu_calls::Queue;
use crate::gpu_data::{FLAG_PHASE1, FLAG_PHASE2, FLAG_VISIBLE, PresentationFrame, frustum_uniform};
use crate::mesh_arena::Ids;
use crate::profiling::{GpuProfiler, Span};
use crate::renderer::FramePoses;
use crate::staging::Prestaged;

use crate::gpu_layouts::constants::ARGS_GROUP;
use crate::gpu_layouts::constants::PARENT_GROUP;
use crate::gpu_layouts::constants::PRESENTATION_GROUP as GROUP;
use crate::gpu_layouts::constants::SCAN_GROUP;
use fades::Fades;
use groups::{DrawGroups, PHASE_ONE, PHASE_TWO};
use inputs::NodeInputs;
pub(crate) use inputs::Staging;
use layouts::Shaders;
use probes::{ProbeFaces, Stillness};
use totals::Totals;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Compose,
    Cull,
    Scan,
    Scatter,
}

impl Stage {
    fn span(self) -> &'static str {
        match self {
            Self::Compose => "presentation_compose",
            Self::Cull => "presentation_cull",
            Self::Scan => "presentation_scan",
            Self::Scatter => "presentation_scatter",
        }
    }
}

const MIN_STORAGE_BYTES: u64 = 256;

#[derive(Default)]
pub(crate) struct GpuPrepared {
    pub candidates: u32,
    pub node_upload_bytes: u64,
    pub group_upload_bytes: u64,
    pub full_node_upload: bool,
    pub staged: bool,
    pub parent_levels: usize,
    pub shifted: bool,
    pub blend: bool,
    pub prestaged: bool,
    pub held: bool,
}

pub(crate) struct BatchMeta {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub bounds: Aabb,
    pub group: u32,
    pub lods: Vec<(u32, u32, f32)>,
    pub impostor: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RecordsKey {
    pub shadow_groups: bool,
    pub meshes: u64,
    pub impostors: u64,
}

pub(crate) struct PresentationInputs<'a> {
    pub frame: &'a RenderFrame<'a>,
    pub gpu: &'a GpuPresentation<'a>,
    pub plan: &'a BatchPlan,
    pub batch_meta: &'a dyn Fn(usize, &DrawBatch) -> BatchMeta,
    pub records: RecordsKey,
    pub poses: FramePoses<'a>,
}

pub(crate) struct CullingView {
    pub view_projection: glam::Mat4,
    pub eye: glam::Vec3,
    pub viewport: (u32, u32),
    pub motion: bool,
    pub occlusion: Option<u32>,
    pub subpixel: bool,
    pub lod_error: f32,
    pub impostors: bool,
    pub stillness: Option<bool>,
    pub fade: f32,
}

struct NodeGroups {
    shift_previous: wgpu::BindGroup,
    compose_parents: wgpu::BindGroup,
}

struct PoseGroups {
    cull: wgpu::BindGroup,
    fades: wgpu::BindGroup,
    count_still: Option<wgpu::BindGroup>,
}

type PoseKey = (u64, u64, u64, u64, u64);

type OcclusionKey = (usize, u64, u64, u64);

pub(crate) struct GpuPresentationPass {
    shaders: Shaders,
    frame: wgpu::Buffer,
    frame_phase2: wgpu::Buffer,
    frame_uniform: Option<PresentationFrame>,
    nodes: NodeInputs,
    groups: DrawGroups,
    stillness: Stillness,
    fades: Fades,
    probes: ProbeFaces,
    totals: Totals,
    watched: std::cell::Cell<Option<u32>>,
    occlusion_counts: wgpu::Buffer,
    occlusion: bool,
    subpixel: bool,
    poses: (usize, u64),
    node_groups: Cached<u64, NodeGroups>,
    compact_groups: Cached<u64, [wgpu::BindGroup; 2]>,
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
            stillness: Stillness::new(device),
            fades: Fades::new(device),
            probes: ProbeFaces::new(std::array::from_fn(|_| {
                uniform("engine/presentation-frame-probe")
            })),
            totals: Totals::new(device),
            watched: std::cell::Cell::new(None),
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
        self.nodes.capacity_bytes() + self.groups.capacity_bytes()
    }

    pub(crate) fn visible(&self) -> Option<u32> {
        self.totals.last().map(|totals| totals.visible)
    }

    pub(crate) fn occluded(&self) -> Option<u32> {
        self.totals.last().map(|totals| totals.occluded)
    }

    pub(crate) fn subpixel(&self) -> Option<u32> {
        self.totals.last().map(|totals| totals.subpixel)
    }

    pub(crate) fn triangles(&self) -> Option<u32> {
        self.totals.last().map(|totals| totals.triangles)
    }

    pub(crate) fn lod_instances(&self) -> Option<[u32; 4]> {
        self.totals.last().map(|totals| totals.levels)
    }

    pub(crate) fn read_back_age(&self) -> Option<u64> {
        self.totals.age()
    }

    pub(crate) fn watch(&self, slot: u32) {
        self.watched.set(Some(slot));
    }

    pub(crate) fn watched(&self) -> Option<(u32, u32)> {
        self.totals.last().and_then(|totals| totals.watched)
    }

    fn copy_watched(&self, encoder: &mut wgpu::CommandEncoder) -> Option<u32> {
        let slot = self.watched.get()?;
        let offset = u64::from(slot) * 4;
        if offset + 4 > self.fades.states.size() {
            return None;
        }
        encoder.copy_buffer_to_buffer(
            &self.fades.states,
            offset,
            &self.totals.buffer,
            totals::WATCHED_OFFSET,
            4,
        );
        Some(slot)
    }

    pub(crate) fn occlusion(&self) -> bool {
        self.occlusion
    }

    pub(crate) fn culls_subpixel(&self) -> bool {
        self.subpixel
    }

    pub(crate) fn shadow_inputs(&self) -> crate::shadows::Presented<'_> {
        crate::shadows::Presented {
            candidates: &self.groups.slot_candidates,
            batches: &self.groups.batches,
            flags: &self.groups.scan,
            count: self.groups.count(),
            generation: self.groups.generation(),
        }
    }

    pub(crate) fn prestage(&mut self, prestaged: Prestaged, version: (u64, u64)) {
        self.nodes.prestage(prestaged, version);
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
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
        self.stillness
            .ensure(device, view.stillness, poses.current());
        self.fades
            .ensure(device, poses.current(), self.groups.capacity());
        self.poses = (poses.parity, poses.generation());

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

        if let Some(bytes) = self.groups.rebuild(queue, inputs) {
            prepared.group_upload_bytes = bytes;
        }
        prepared.candidates = self.groups.count();
        self.bind(device, poses);
        self.write_frame_uniform(queue, frame, gpu.alpha, view);
        prepared
    }

    fn submit_shift(&mut self, device: &wgpu::Device, queue: &Queue, profiler: &GpuProfiler) {
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
            pass.dispatch_workgroups((self.nodes.capacity() as u32).div_ceil(GROUP), 1, 1);
        }
        queue.submit_now([encoder.finish()]);
    }

    fn bind(&mut self, device: &wgpu::Device, poses: &FramePoses<'_>) {
        let Self {
            shaders,
            frame,
            frame_phase2,
            nodes,
            groups,
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
            stillness.generation(),
            fades.generation(),
            poses.generation(),
        );
        pose_groups[poses.parity].get_or_insert_with(key, || {
            let current = poses.current();
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

    fn write_frame_uniform(
        &mut self,
        queue: &crate::gpu_calls::Queue,
        frame: &RenderFrame<'_>,
        alpha: f32,
        view: &CullingView,
    ) {
        let (planes, absolute) = frustum_uniform(view.view_projection);
        let bounded = absolute.iter().flatten().all(|value| {
            value.is_finite() && *value <= engine_core::presentation_bounds::MAX_COEFFICIENT
        }) && planes.iter().flatten().all(|value| value.is_finite());
        let culling = frame.culling() && bounded;
        let candidates = self.groups.count();
        let unbudgeted = culling && frame.instance_budget() >= candidates as usize;
        self.occlusion = view.occlusion.is_some() && unbudgeted && candidates != 0;
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
            mask: if self.occlusion {
                FLAG_PHASE1
            } else {
                FLAG_VISIBLE
            },
            occlusion: u32::from(self.occlusion),
            subpixel: u32::from(self.subpixel),
            view_projection: view.view_projection.to_cols_array_2d(),
            viewport: [
                width as f32,
                height as f32,
                view.occlusion.unwrap_or(0) as f32,
                0.0,
            ],
            plane: self.groups.capacity() as u32,
            lod_error: if unbudgeted { view.lod_error } else { 0.0 },
            eye: view
                .eye
                .extend(if view.impostors { 1.0 } else { 0.0 })
                .to_array(),
            fade: [view.fade, fades::FADE_MARGIN, 0.0, 0.0],
            pad: [0; 2],
        };
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&uniform));
        self.fades.begin_frame(queue);
        self.frame_uniform = Some(uniform);
        if self.occlusion {
            uniform.mask = FLAG_PHASE2;
            queue.write_buffer(&self.frame_phase2, 0, bytemuck::bytes_of(&uniform));
        }
    }

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
        let lod = self
            .frame_uniform
            .is_some_and(|uniform| uniform.lod_error > 0.0);
        if candidates != 0 {
            encoder.clear_buffer(&self.occlusion_counts, 0, None);
        }
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
                let workgroups = candidates.div_ceil(GROUP);
                pass.set_pipeline(&pipelines.cull);
                pass.set_bind_group(0, &posed.cull, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
                if let Some(group) = posed
                    .count_still
                    .as_ref()
                    .filter(|_| self.stillness.counting())
                {
                    pass.set_pipeline(shaders.count_still(device));
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
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
            (Stage::Cull, candidates != 0),
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
            let (scan, totals) = (&self.groups.scan, &self.totals.buffer);
            encoder.clear_buffer(totals, 0, None);
            encoder.copy_buffer_to_buffer(scan, self.groups.total_offset(), totals, 0, 4);
            encoder.copy_buffer_to_buffer(scan, self.groups.level_offset(), totals, 16, 16);
            encoder.copy_buffer_to_buffer(&self.occlusion_counts, 4, totals, 12, 4);
            if !self.occlusion {
                encoder.copy_buffer_to_buffer(&self.occlusion_counts, 8, totals, 48, 4);
                let watched = self.copy_watched(encoder);
                self.totals.read_back(device, encoder, watched);
            }
        }
    }

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
            self.dispatch_compaction(&mut pass, &compact[PHASE_TWO]);
        }
        let (scan, totals) = (&self.groups.scan, &self.totals.buffer);
        encoder.copy_buffer_to_buffer(scan, self.groups.total_offset(), totals, 4, 4);
        encoder.copy_buffer_to_buffer(scan, self.groups.level_offset(), totals, 32, 16);
        encoder.copy_buffer_to_buffer(&self.occlusion_counts, 0, totals, 8, 4);
        encoder.copy_buffer_to_buffer(&self.occlusion_counts, 8, totals, 48, 4);
        let watched = self.copy_watched(encoder);
        self.totals.read_back(device, encoder, watched);
    }

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

    pub(crate) fn after_submit(&mut self) {
        self.totals.after_submit();
    }

    pub(crate) fn draw_source<'a>(
        &'a self,
        batch: &DrawBatch,
        index: usize,
        phase: usize,
        level: usize,
    ) -> (Ids<'a>, &'a wgpu::Buffer, u64) {
        self.groups.draw_source(batch, index, phase, level)
    }

    pub(crate) fn impostor_source(&self, phase: usize) -> (&wgpu::Buffer, &wgpu::Buffer, u64) {
        self.groups.impostor_source(phase)
    }

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
