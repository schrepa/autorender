use crate::bindings::{BindingDecl, Contribution, Stages, Table, read, uniform};
use crate::gpu_layouts::constants::GROUP_LIGHTS as LIGHTS;
pub(crate) use crate::gpu_layouts::constants::LIGHT_BYTES;
use crate::gpu_layouts::constants::{
    KEPT_REGIONS, LIGHT_EXTERIOR, LIGHT_INTERIOR, NO_SLOT, SHAPED_LIGHTS,
};
use crate::graph::{Lifetime, PassDecl, ResourceDecl, Stage};
use crate::parts::{PartDecl, PartEntry};
use crate::shaders::entry::lights;
use engine_core::{LightRegion, RenderLight, glam::Mat4};

pub(crate) const TILES_X: u32 = 16;
pub(crate) const TILES_Y: u32 = 9;
pub(crate) const SLICES: u32 = 24;
const CLUSTERS: u64 = (TILES_X * TILES_Y * SLICES) as u64;
const TOTALS_BYTES: u64 = 16;
pub(crate) const NO_BUDGET: u32 = u32::MAX;

const SHADING: &[BindingDecl] = &[
    uniform("clusters", Stages::FRAGMENT),
    read("lights", Stages::FRAGMENT),
    read("cluster_ranges", Stages::FRAGMENT),
    read("cluster_lights", Stages::FRAGMENT),
];

pub(crate) const LIGHTS_LIGHTS: Contribution = Contribution {
    group: "lights",
    set: "lights",
    bindings: SHADING,
};

const PROBE_SHADING: Table = Table {
    label: "engine/light-shading-layout",
    bindings: SHADING,
};

const LABELS: &[&str] = &["engine/light-", "engine/lights"];

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "lights",
        brick: "lights",
        passes: &[PassDecl {
            name: "lights",
            stage: Stage::Prepare,
        }],
        resources: &[ResourceDecl {
            name: "lights",
            lifetime: Lifetime::Frame,
        }],
        hooks: &[],
        contributes: &[LIGHTS_LIGHTS],
        effects: &[],
        labels: LABELS,
        lazy: true,
    },
    create: gpu::create,
};

pub(crate) use crate::gpu_layouts::lights::ClusterUniform;

impl ClusterUniform {
    pub(crate) fn new(
        view_projection: Mat4,
        width: u32,
        height: u32,
        distance: f32,
        lights: u32,
        budget: Option<u32>,
        capacity: u32,
    ) -> Self {
        let depth = view_projection.row(3);
        let scale_x = view_projection.row(0).truncate().length();
        let scale_y = view_projection.row(1).truncate().length();
        let perspective = (depth.truncate().length_squared() - 1.0).abs() < 1.0e-3;
        let p22 = -view_projection.row(2).truncate().dot(depth.truncate());
        let p23 = view_projection.row(2).w + p22 * depth.w;
        let (mut near, mut far) = (p23 / (p22 + 1.0), p23 / p22);
        if !(perspective
            && p22 > 0.0
            && near.is_finite()
            && far.is_finite()
            && near > 0.0
            && far > near)
        {
            (near, far) = (0.1, 1000.0);
        }
        let far = engine_core::math::max(engine_core::math::min(far, distance), near * 2.0);
        let start = engine_core::math::max(near, engine_core::math::min(1.0, far * 0.01));
        let slice_scale = SLICES as f32 / engine_core::math::max((far / start).log2(), 1.0e-3);
        let tile = |size: u32, tiles: u32| size.max(1).div_ceil(tiles) as f32;
        Self {
            view_projection: view_projection.to_cols_array_2d(),
            depth: depth.to_array(),
            projection: [scale_x, scale_y, start, far],
            grid: [
                tile(width, TILES_X),
                tile(height, TILES_Y),
                slice_scale,
                -start.log2() * slice_scale,
            ],
            counts: [TILES_X, TILES_Y, SLICES, budget.unwrap_or(NO_BUDGET)],
            lights: [lights, 0, capacity, 0],
        }
    }
}

pub(crate) use crate::gpu_layouts::lights::LightInput;

fn shading_flags(snapshot: &engine_core::RenderSnapshot, regions: bool) -> u32 {
    let mut flags = 0;
    if snapshot.environment.highlights.enabled {
        flags |= SHAPED_LIGHTS;
    }
    if regions {
        flags |= KEPT_REGIONS;
    }
    flags
}

impl From<&RenderLight> for LightInput {
    fn from(light: &RenderLight) -> Self {
        Self {
            position: light.position.to_array(),
            range: light.range,
            color: light.color.to_array(),
            slot: light.slot.unwrap_or(NO_SLOT),
            direction: light.direction.to_array(),
            cos_inner: light.cone[0],
            cos_outer: light.cone[1],
            radius: light.shape[0],
            length: light.shape[1],
            region: match light.region {
                LightRegion::Both => 0,
                LightRegion::Interior => LIGHT_INTERIOR,
                LightRegion::Exterior => LIGHT_EXTERIOR,
            },
        }
    }
}

mod gpu {
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
    };

    use engine_core::{LightStats, RenderSnapshot, RenderStats};

    use super::*;
    use crate::bindings::LIGHT_BINNING;
    use crate::buffers::{Growable, ReadbackRing, entry, storage};
    use crate::cached::Bound;
    use crate::graph::{FrameGraph, GraphTable, PassId, ResourceId};
    use crate::lazy::Lazy;
    use crate::parts::{Create, Encode, GpuPart, Prelude, Step, View};
    use crate::profiling::Span;
    use crate::renderer::FramePoses;

    struct LightFrame {
        count: u32,
        group: usize,
        flags: u32,
    }

    type ProbeLightGroups = (
        (u64, usize, u64, u64),
        Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
    );

    #[derive(Default)]
    struct ProbeLights {
        uniforms: Vec<wgpu::Buffer>,
        groups: Vec<ProbeLightGroups>,
        current: usize,
    }

    struct LightBins {
        count: wgpu::ComputePipeline,
        scan: wgpu::ComputePipeline,
        fill: wgpu::ComputePipeline,
        compute_layout: crate::pipelines::GroupLayout,
        shading_layout: crate::pipelines::GroupLayout,
        uniform: wgpu::Buffer,
        written: Option<ClusterUniform>,
        inputs: wgpu::Buffer,
        lights: wgpu::Buffer,
        capacity: usize,
        counts: wgpu::Buffer,
        ranges: wgpu::Buffer,
        indices: Growable,
        totals: wgpu::Buffer,
        staged: Vec<LightInput>,
        uploaded_revision: Option<(u64, u64)>,
        regions: bool,
        compute_groups: Vec<((u64, usize, u64), wgpu::BindGroup)>,
        probes: RefCell<ProbeLights>,
        no_still: wgpu::Buffer,
        buffers_generation: u64,
        readbacks: ReadbackRing<()>,
        totals_read: Rc<Cell<Option<[u32; 3]>>>,
    }

    impl LightBins {
        fn new(device: &wgpu::Device) -> Self {
            let compute_layout = crate::pipelines::layout(device, &LIGHT_BINNING);
            let shading_layout = crate::pipelines::layout(device, &PROBE_SHADING);
            let layout = crate::pipelines::layout_of(
                device,
                &crate::pipeline_layouts::LIGHT_BINNING_PIPELINES,
            );
            let pipeline = |entry: crate::shaders::Entry| {
                crate::pipelines::compute_entry(device, &layout, entry)
            };
            let none = wgpu::BufferUsages::empty();
            Self {
                count: pipeline(lights::count_lights),
                scan: pipeline(lights::scan_clusters),
                fill: pipeline(lights::bin_lights),
                compute_layout,
                shading_layout,
                uniform: crate::buffers::uniform(
                    device,
                    "engine/light-clusters-uniform",
                    size_of::<ClusterUniform>() as u64,
                ),
                written: None,
                inputs: storage(device, "engine/light-inputs", 64, none),
                lights: storage(device, "engine/lights", LIGHT_BYTES, none),
                capacity: 1,
                counts: storage(device, "engine/light-cluster-counts", CLUSTERS * 4, none),
                ranges: storage(device, "engine/light-cluster-ranges", CLUSTERS * 8, none),
                indices: Growable::new(device, "engine/light-cluster-indices", CLUSTERS * 4, none),
                totals: storage(
                    device,
                    "engine/light-totals",
                    TOTALS_BYTES,
                    wgpu::BufferUsages::COPY_SRC,
                ),
                staged: Vec::new(),
                uploaded_revision: None,
                regions: false,
                compute_groups: Vec::new(),
                probes: RefCell::default(),
                no_still: storage(device, "engine/light-no-still", 16, none),
                buffers_generation: 0,
                readbacks: ReadbackRing::new("engine/light-totals-readback"),
                totals_read: Rc::new(Cell::new(None)),
            }
        }

        fn totals(&self) -> Option<[u32; 2]> {
            self.totals_read
                .get()
                .map(|[reached, lost, _]| [reached, lost])
        }

        fn capacity_bytes(&self) -> u64 {
            [
                &self.inputs,
                &self.lights,
                &self.counts,
                &self.ranges,
                &self.indices,
            ]
            .iter()
            .map(|buffer| buffer.size())
            .sum()
        }

        fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &crate::gpu_calls::Queue,
            snapshot: &RenderSnapshot,
            (width, height): (u32, u32),
            poses: FramePoses<'_>,
            revision: (u64, u64),
        ) -> Option<LightFrame> {
            if !snapshot.environment.local_lights.active() || snapshot.lights.list.is_empty() {
                return None;
            }
            let count = snapshot.lights.list.len();
            if count > self.capacity {
                self.capacity = count.next_power_of_two();
                crate::memory::release_buffer(&self.inputs);
                self.inputs = storage(
                    device,
                    "engine/light-inputs",
                    self.capacity as u64 * 64,
                    wgpu::BufferUsages::empty(),
                );
                crate::memory::release_buffer(&self.lights);
                self.lights = storage(
                    device,
                    "engine/lights",
                    self.capacity as u64 * LIGHT_BYTES,
                    wgpu::BufferUsages::empty(),
                );
                self.uploaded_revision = None;
                self.buffers_generation += 1;
            }
            if self.uploaded_revision != Some(revision) {
                self.staged.clear();
                self.staged
                    .extend(snapshot.lights.list.iter().map(LightInput::from));
                queue.write_buffer(&self.inputs, 0, bytemuck::cast_slice(&self.staged));
                self.uploaded_revision = Some(revision);
                self.regions = self.staged.iter().any(|light| light.region != 0);
            }
            let needed = self
                .totals_read
                .get()
                .map_or(0, |[_, _, needed]| u64::from(needed))
                .max(count as u64 * 32);
            if self.indices.grow(device, needed * 4) {
                self.buffers_generation += 1;
            }
            let mut uniform = ClusterUniform::new(
                snapshot.view_projection,
                width,
                height,
                snapshot.environment.local_lights.distance,
                count as u32,
                snapshot.environment.local_lights.lights_per_cluster,
                self.index_capacity(),
            );
            uniform.lights[1] = shading_flags(snapshot, self.regions);
            if self.written != Some(uniform) {
                queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
                self.written = Some(uniform);
            }
            let key = (poses.generation(), poses.parity, self.buffers_generation);
            let group = match self
                .compute_groups
                .iter()
                .position(|(built, _)| *built == key)
            {
                Some(index) => index,
                None => {
                    self.compute_groups.retain(|((generation, _, buffers), _)| {
                        *generation == key.0 && *buffers == key.2
                    });
                    let group = crate::pipelines::group(
                        device,
                        "engine/light-binning-group",
                        &LIGHT_BINNING,
                        &self.compute_layout,
                        &[
                            entry("clusters", &self.uniform),
                            entry("inputs", &self.inputs),
                            entry("poses", poses.current()),
                            entry("lights", &self.lights),
                            entry("counts", &self.counts),
                            entry("indices", &self.indices),
                            entry("totals", &self.totals),
                            entry("ranges", &self.ranges),
                            entry("still_frames", &self.no_still),
                        ],
                    );
                    self.compute_groups.push((key, group));
                    self.compute_groups.len() - 1
                }
            };
            Some(LightFrame {
                count: count as u32,
                group,
                flags: uniform.lights[1],
            })
        }

        fn index_capacity(&self) -> u32 {
            (self.indices.size() / 4).min(u64::from(u32::MAX)) as u32
        }

        fn shading(&self) -> [(&'static str, Bound<'_>); 4] {
            [
                ("clusters", Bound::fixed(self.uniform.as_entire_binding())),
                (
                    "lights",
                    Bound::counted(self.buffers_generation, self.lights.as_entire_binding()),
                ),
                (
                    "cluster_ranges",
                    Bound::fixed(self.ranges.as_entire_binding()),
                ),
                ("cluster_lights", self.indices.bound()),
            ]
        }

        fn encode_binning(
            &self,
            device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            frame: &LightFrame,
            span: Option<Span<'_>>,
        ) {
            encoder.clear_buffer(&self.counts, 0, None);
            encoder.clear_buffer(&self.totals, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("engine/light-binning"),
                    timestamp_writes: span.map(Span::compute),
                });
                let workgroups = frame.count.div_ceil(LIGHTS);
                let (columns, rows) = (workgroups.min(65_535), workgroups.div_ceil(65_535));
                pass.set_bind_group(0, &self.compute_groups[frame.group].1, &[]);
                pass.set_pipeline(&self.count);
                pass.dispatch_workgroups(columns, rows, 1);
                pass.set_pipeline(&self.scan);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&self.fill);
                pass.dispatch_workgroups(columns, rows, 1);
            }
            self.readbacks
                .record(device, encoder, &self.totals, TOTALS_BYTES, ());
        }

        #[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
        fn encode_probe_face(
            &self,
            device: &wgpu::Device,
            queue: &crate::gpu_calls::Queue,
            encoder: &mut wgpu::CommandEncoder,
            face: usize,
            view_projection: Mat4,
            size: u32,
            settings: &engine_core::LocalLightSettings,
            frame: &LightFrame,
            poses: FramePoses<'_>,
            still: (&wgpu::Buffer, u64),
        ) {
            let mut probes = self.probes.borrow_mut();
            let probes = &mut *probes;
            if probes.uniforms.is_empty() {
                probes.uniforms = (0..6)
                    .map(|_| {
                        crate::buffers::uniform(
                            device,
                            "engine/light-clusters-probe",
                            size_of::<ClusterUniform>() as u64,
                        )
                    })
                    .collect();
            }
            let mut uniform = ClusterUniform::new(
                view_projection,
                size,
                size,
                settings.distance,
                frame.count,
                settings.lights_per_cluster,
                self.index_capacity(),
            );
            uniform.lights[1] = frame.flags;
            uniform.lights[3] = 1;
            queue.write_buffer(&probes.uniforms[face], 0, bytemuck::bytes_of(&uniform));
            let key = (
                poses.generation(),
                poses.parity,
                self.buffers_generation,
                still.1,
            );
            if let Some(index) = probes.groups.iter().position(|(built, _)| *built == key) {
                probes.current = index;
            } else {
                probes
                    .groups
                    .retain(|((generation, _, buffers, still), _)| {
                        (*generation, *buffers, *still) == (key.0, key.2, key.3)
                    });
                let groups = probes
                    .uniforms
                    .iter()
                    .map(|uniform| {
                        let compute = crate::pipelines::group(
                            device,
                            "engine/light-binning-probe-group",
                            &LIGHT_BINNING,
                            &self.compute_layout,
                            &[
                                entry("clusters", uniform),
                                entry("inputs", &self.inputs),
                                entry("poses", poses.current()),
                                entry("lights", &self.lights),
                                entry("counts", &self.counts),
                                entry("indices", &self.indices),
                                entry("totals", &self.totals),
                                entry("ranges", &self.ranges),
                                entry("still_frames", still.0),
                            ],
                        );
                        let fragment = crate::pipelines::group(
                            device,
                            "engine/light-shading-probe-group",
                            &PROBE_SHADING,
                            &self.shading_layout,
                            &[
                                entry("clusters", uniform),
                                entry("lights", &self.lights),
                                entry("cluster_ranges", &self.ranges),
                                entry("cluster_lights", &self.indices),
                            ],
                        );
                        (compute, fragment)
                    })
                    .collect();
                probes.groups.push((key, groups));
                probes.current = probes.groups.len() - 1;
            }
            let group = &probes.groups[probes.current].1[face].0;
            encoder.clear_buffer(&self.counts, 0, None);
            encoder.clear_buffer(&self.totals, 0, None);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/light-binning-probe"),
                timestamp_writes: None,
            });
            let workgroups = frame.count.div_ceil(LIGHTS);
            let (columns, rows) = (workgroups.min(65_535), workgroups.div_ceil(65_535));
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&self.count);
            pass.dispatch_workgroups(columns, rows, 1);
            pass.set_pipeline(&self.scan);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.fill);
            pass.dispatch_workgroups(columns, rows, 1);
        }

        fn probe_group(&self, face: usize) -> wgpu::BindGroup {
            let probes = self.probes.borrow();
            probes.groups[probes.current].1[face].1.clone()
        }
    }

    pub(super) fn create(_cx: &Create<'_>) -> Box<dyn GpuPart> {
        Box::new(LightPass::default())
    }

    #[derive(Default)]
    pub(crate) struct LightPass {
        bins: Lazy<LightBins>,
        frame: Option<LightFrame>,
        pass: Option<PassId>,
        poses: Option<ResourceId>,
        lights: Option<ResourceId>,
    }

    impl LightPass {
        pub(crate) fn created(&self) -> bool {
            self.bins.get().is_some()
        }

        pub(crate) fn lit(&self) -> bool {
            self.frame.is_some() && self.created()
        }

        pub(crate) fn encode_probe_face(
            &self,
            (device, queue): (&wgpu::Device, &crate::gpu_calls::Queue),
            encoder: &mut wgpu::CommandEncoder,
            (face, view_projection, size): (usize, Mat4, u32),
            settings: &engine_core::LocalLightSettings,
            poses: FramePoses<'_>,
            still: (&wgpu::Buffer, u64),
        ) {
            if let (Some(bins), Some(frame)) = (self.bins.get(), &self.frame) {
                bins.encode_probe_face(
                    device,
                    queue,
                    encoder,
                    face,
                    view_projection,
                    size,
                    settings,
                    frame,
                    poses,
                    still,
                );
            }
        }

        pub(crate) fn probe_group(&self, face: usize) -> Option<wgpu::BindGroup> {
            self.bins.get().map(|bins| bins.probe_group(face))
        }
    }

    impl GpuPart for LightPass {
        fn install(&mut self, table: &GraphTable) {
            self.pass = table.pass_named("lights");
            self.poses = table.resource("poses");
            self.lights = table.resource("lights");
        }

        fn prepare(&mut self, cx: &mut Step<'_>, _prelude: &mut Prelude) {
            self.frame = None;
            if cx.frame.kind.debug {
                return;
            }
            let frame = cx.frame.frame;
            let snapshot = cx.frame.snapshot;
            let needs =
                snapshot.environment.local_lights.active() && !snapshot.lights.list.is_empty();
            if !needs {
                return;
            }
            let core = &*cx.core;
            let device = &core.device;
            self.bins
                .create(device, LABELS[0], || LightBins::new(device));
            let Some(bins) = self.bins.get_mut() else {
                return;
            };
            let parity = cx.frame.temporal.as_ref().map_or(0, |frame| frame.parity);
            self.frame = bins.prepare(
                device,
                &core.queue,
                snapshot,
                core.render_pixels(),
                core.frame_poses(parity),
                (frame.identity(), frame.lights_revision()),
            );
        }

        fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
            if let (Some(pass), Some(_)) = (self.pass, &self.frame) {
                let _ = graph
                    .add(cx.table, pass)
                    .read(self.poses)
                    .write(self.lights);
            }
        }

        fn encode(
            &mut self,
            _pass: usize,
            cx: &mut Encode<'_>,
            encoder: &mut wgpu::CommandEncoder,
        ) -> u32 {
            if let (Some(bins), Some(frame)) = (self.bins.get(), &self.frame) {
                bins.encode_binning(
                    &cx.core.device,
                    encoder,
                    frame,
                    cx.core.profiler.span("light_bins"),
                );
            }
            0
        }

        fn contribute<'s>(&'s self, group: &str, out: &mut Vec<(&'static str, Bound<'s>)>) {
            if group != "lights" {
                return;
            }
            if let Some(bins) = self.bins.get() {
                out.extend(bins.shading());
            }
        }

        fn after_submit(&mut self) {
            if let Some(bins) = self.bins.get_mut() {
                let totals = Rc::clone(&bins.totals_read);
                bins.readbacks.after_submit(move |words, ()| {
                    totals.set(Some([words.get(0), words.get(1), words.get(2)]));
                });
            }
        }

        fn capacity_bytes(&self) -> u64 {
            self.bins.get().map_or(0, LightBins::capacity_bytes)
        }

        fn stats(&self, cx: &View<'_>, stats: &mut RenderStats) {
            let snapshot = cx.frame.snapshot;
            let lit = self.frame.is_some();
            let bins = self.bins.get();
            let totals = self.frame.as_ref().and(bins.and_then(LightBins::totals));
            stats.local_lights = LightStats {
                shaded: if lit {
                    snapshot.lights.list.len() as u32
                } else {
                    0
                },
                dropped: if lit {
                    snapshot.lights.dropped.min(u32::MAX as usize) as u32
                } else {
                    0
                },
                invalid: snapshot.lights.invalid.min(u32::MAX as usize) as u32,
                visible: totals.map(|totals| totals[0]),
                cluster_overflow: totals.map(|totals| totals[1]),
            };
        }

        fn poll(&mut self) -> bool {
            self.bins.poll()
        }

        fn disable(&mut self) -> bool {
            self.bins.disable()
        }

        fn disabled(&self) -> bool {
            self.bins.disabled()
        }

        #[cfg(feature = "dev")]
        fn inject(&mut self) -> bool {
            self.bins.inject()
        }

        #[cfg(feature = "dev")]
        fn encode_fault(&self, encoder: &mut wgpu::CommandEncoder) {
            self.bins.encode_fault(encoder);
        }
    }
}

pub(crate) use gpu::LightPass;
