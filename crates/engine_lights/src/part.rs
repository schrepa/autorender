use engine_core::{LightRegion, RenderLight, glam::Mat4};
use engine_faces::lights::LightListFace;
use engine_render::bindings::{BindingDecl, Contribution, Stages, Table, read, uniform, write};
use engine_render::graph::{Lifetime, PassDecl, ResourceDecl, Stage};
use engine_render::parts::{PartDecl, PartEntry};

use crate::gpu_layouts::GROUP_LIGHTS as LIGHTS;
use crate::gpu_layouts::{
    KEPT_REGIONS, LIGHT_BYTES, LIGHT_EXTERIOR, LIGHT_INTERIOR, NO_SLOT, SHAPED_LIGHTS,
};
use crate::shaders::entry::lights;

pub(crate) const TILES_X: u32 = 16;
pub(crate) const TILES_Y: u32 = 9;
pub(crate) const SLICES: u32 = 24;
const CLUSTERS: u64 = (TILES_X * TILES_Y * SLICES) as u64;
const RANGES_BYTES: u64 = CLUSTERS * 8;
const SECTION_ALIGN: u64 = 256;
const INDEX_START: u64 = RANGES_BYTES.next_multiple_of(SECTION_ALIGN);
const TOTALS_BYTES: u64 = 16;
pub(crate) const NO_BUDGET: u32 = u32::MAX;
const LOCAL_LIGHTS: &str = "LOCAL_LIGHTS";

const SHADING: &[BindingDecl] = &[
    uniform("clusters", Stages::FRAGMENT),
    read("light_data", Stages::FRAGMENT),
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

pub(crate) const LIGHT_BINNING: Table = Table {
    label: "engine/light-binning-layout",
    bindings: &[
        uniform("clusters", Stages::COMPUTE),
        read("inputs", Stages::COMPUTE),
        read("poses", Stages::COMPUTE),
        write("lights", Stages::COMPUTE),
        write("counts", Stages::COMPUTE),
        write("indices", Stages::COMPUTE),
        write("totals", Stages::COMPUTE),
        write("ranges", Stages::COMPUTE),
        read("still_frames", Stages::COMPUTE),
    ],
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
        faces: &[engine_render::offer_face!(gpu::LightPass => dyn LightListFace)],
        needs: &[],
        piece: None,
    },
    create: gpu::create,
};

pub(crate) use crate::gpu_layouts::ClusterUniform;

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
            offsets: [0; 4],
        }
    }
}

pub(crate) use crate::gpu_layouts::LightInput;

fn shading_flags(shaped: bool, regions: bool) -> u32 {
    let mut flags = 0;
    if shaped {
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

    use engine_core::{
        Cut, Highlights, LightStats, LocalLightSettings, RenderSnapshot, RenderStats,
    };
    use engine_render::buffers::{ReadbackRing, entry, storage};
    use engine_render::cached::{Bound, Versioned};
    use engine_render::gpu_calls::Queue;
    use engine_render::graph::{FrameGraph, GraphTable, PassId, ResourceId};
    use engine_render::lazy::Lazy;
    use engine_render::parts::{Create, Encode, GpuPart, Prelude, Shading, Step, View};
    use engine_render::profiling::Span;

    use super::*;

    const DATA_LABEL: &str = "engine/light-data";

    struct LightFrame {
        count: u32,
        group: usize,
        flags: u32,
        settings: (f32, Option<u32>),
        poses: (wgpu::Buffer, u64, usize),
        still: (wgpu::Buffer, u64),
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
        compute_layout: engine_render::pipelines::GroupLayout,
        shading_layout: engine_render::pipelines::GroupLayout,
        uniform: wgpu::Buffer,
        written: Option<ClusterUniform>,
        inputs: wgpu::Buffer,
        data: Versioned<wgpu::Buffer>,
        capacity: usize,
        index_bytes: u64,
        room: u64,
        counts: wgpu::Buffer,
        totals: wgpu::Buffer,
        staged: Vec<LightInput>,
        uploaded_revision: Option<(u64, u64)>,
        regions: bool,
        compute_groups: Vec<((u64, usize, u64), wgpu::BindGroup)>,
        probes: RefCell<ProbeLights>,
        no_still: wgpu::Buffer,
        readbacks: ReadbackRing<u32>,
        totals_read: Rc<Cell<Option<[u32; 4]>>>,
    }

    impl LightBins {
        fn new(device: &wgpu::Device) -> Self {
            let compute_layout = engine_render::pipelines::layout(device, &LIGHT_BINNING);
            let shading_layout = engine_render::pipelines::layout(device, &PROBE_SHADING);
            let layout = engine_render::pipelines::layout_of(
                device,
                &crate::pipeline_layouts::LIGHT_BINNING_PIPELINES,
            );
            let pipeline = |entry: engine_render::shaders::Entry| {
                engine_render::pipelines::compute_entry(device, &layout, entry)
            };
            let none = wgpu::BufferUsages::empty();
            let limits = device.limits();
            let room = limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size)
                / 16
                * 16;
            let index_bytes = (CLUSTERS * 4)
                .next_multiple_of(SECTION_ALIGN)
                .min(index_room(room, 1));
            Self {
                count: pipeline(lights::count_lights),
                scan: pipeline(lights::scan_clusters),
                fill: pipeline(lights::bin_lights),
                compute_layout,
                shading_layout,
                uniform: engine_render::buffers::uniform(
                    device,
                    "engine/light-clusters-uniform",
                    size_of::<ClusterUniform>() as u64,
                ),
                written: None,
                inputs: storage(device, "engine/light-inputs", 64, none),
                data: Versioned::new(storage(
                    device,
                    DATA_LABEL,
                    data_bytes(index_bytes, 1),
                    none,
                )),
                capacity: 1,
                index_bytes,
                room,
                counts: storage(device, "engine/light-cluster-counts", CLUSTERS * 4, none),
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
                readbacks: ReadbackRing::new("engine/light-totals-readback"),
                totals_read: Rc::new(Cell::new(None)),
            }
        }

        fn totals(&self) -> Option<(u32, Cut)> {
            self.totals_read
                .get()
                .map(|[reached, lost, needed, capacity]| {
                    let stored = needed.min(capacity);
                    (reached, Cut::of(stored.saturating_add(lost), stored))
                })
        }

        fn capacity_bytes(&self) -> u64 {
            self.inputs.size() + self.data.size() + self.counts.size()
        }

        fn lights_offset(&self) -> u64 {
            INDEX_START + self.index_bytes
        }

        fn lights_bytes(&self) -> u64 {
            self.capacity as u64 * LIGHT_BYTES
        }

        fn offsets(&self) -> [u32; 4] {
            [
                (self.lights_offset() / 16) as u32,
                0,
                (INDEX_START / 16) as u32,
                0,
            ]
        }

        fn sections(&self) -> [(&'static str, wgpu::BindingResource<'_>); 3] {
            let data: &wgpu::Buffer = &self.data;
            [
                (
                    "lights",
                    section(data, self.lights_offset(), self.lights_bytes()),
                ),
                ("indices", section(data, INDEX_START, self.index_bytes)),
                ("ranges", section(data, 0, RANGES_BYTES)),
            ]
        }

        fn grow_indices(&mut self, bytes: u64) -> bool {
            let size = bytes
                .next_power_of_two()
                .min(index_room(self.room, self.capacity));
            if bytes <= self.index_bytes || size <= self.index_bytes {
                return false;
            }
            self.index_bytes = size;
            true
        }

        fn prepare(
            &mut self,
            (device, queue): (&wgpu::Device, &Queue),
            (snapshot, settings, shaped): (&RenderSnapshot, LocalLightSettings, bool),
            (width, height): (u32, u32),
            (poses, poses_generation, parity): (&wgpu::Buffer, u64, usize),
            still: (&wgpu::Buffer, u64),
            revision: (u64, u64),
        ) -> Option<LightFrame> {
            if !settings.active() || snapshot.lights.list.is_empty() {
                return None;
            }
            let count = snapshot.lights.list.len();
            let mut grown = false;
            if count > self.capacity {
                self.capacity = count.next_power_of_two();
                engine_render::memory::release_buffer(&self.inputs);
                self.inputs = storage(
                    device,
                    "engine/light-inputs",
                    self.capacity as u64 * 64,
                    wgpu::BufferUsages::empty(),
                );
                self.uploaded_revision = None;
                grown = true;
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
                .map_or(0, |[_, _, needed, _]| u64::from(needed))
                .max(count as u64 * 32);
            if self.grow_indices(needed * 4) || grown {
                self.data.replace(storage(
                    device,
                    DATA_LABEL,
                    data_bytes(self.index_bytes, self.capacity),
                    wgpu::BufferUsages::empty(),
                ));
            }
            let mut uniform = ClusterUniform::new(
                snapshot.view_projection,
                width,
                height,
                settings.distance,
                count as u32,
                settings.lights_per_cluster,
                self.index_capacity(),
            );
            uniform.lights[1] = shading_flags(shaped, self.regions);
            uniform.offsets = self.offsets();
            if self.written != Some(uniform) {
                queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
                self.written = Some(uniform);
            }
            let key = (poses_generation, parity, self.data.generation());
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
                    let [lights, indices, ranges] = self.sections();
                    let group = engine_render::pipelines::group(
                        device,
                        "engine/light-binning-group",
                        &LIGHT_BINNING,
                        &self.compute_layout,
                        &[
                            entry("clusters", &self.uniform),
                            entry("inputs", &self.inputs),
                            entry("poses", poses),
                            lights,
                            entry("counts", &self.counts),
                            indices,
                            entry("totals", &self.totals),
                            ranges,
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
                settings: (settings.distance, settings.lights_per_cluster),
                poses: (poses.clone(), poses_generation, parity),
                still: (still.0.clone(), still.1),
            })
        }

        fn index_capacity(&self) -> u32 {
            (self.index_bytes / 4).min(u64::from(u32::MAX)) as u32
        }

        fn shading(&self) -> [(&'static str, Bound<'_>); 2] {
            [
                ("clusters", Bound::fixed(self.uniform.as_entire_binding())),
                ("light_data", self.data.bound()),
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
            let capacity = self.index_capacity();
            self.readbacks
                .record(device, encoder, &self.totals, TOTALS_BYTES, capacity);
        }

        fn encode_probe_face(
            &self,
            (device, queue): (&wgpu::Device, &Queue),
            encoder: &mut wgpu::CommandEncoder,
            (face, view_projection, size): (usize, Mat4, u32),
            frame: &LightFrame,
        ) {
            let mut probes = self.probes.borrow_mut();
            let probes = &mut *probes;
            if probes.uniforms.is_empty() {
                probes.uniforms = (0..6)
                    .map(|_| {
                        engine_render::buffers::uniform(
                            device,
                            "engine/light-clusters-probe",
                            size_of::<ClusterUniform>() as u64,
                        )
                    })
                    .collect();
            }
            let (distance, budget) = frame.settings;
            let mut uniform = ClusterUniform::new(
                view_projection,
                size,
                size,
                distance,
                frame.count,
                budget,
                self.index_capacity(),
            );
            uniform.lights[1] = frame.flags;
            uniform.lights[3] = 1;
            uniform.offsets = self.offsets();
            queue.write_buffer(&probes.uniforms[face], 0, bytemuck::bytes_of(&uniform));
            let (poses, poses_generation, parity) = &frame.poses;
            let (still, still_generation) = &frame.still;
            let key = (
                *poses_generation,
                *parity,
                self.data.generation(),
                *still_generation,
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
                        let [lights, indices, ranges] = self.sections();
                        let compute = engine_render::pipelines::group(
                            device,
                            "engine/light-binning-probe-group",
                            &LIGHT_BINNING,
                            &self.compute_layout,
                            &[
                                entry("clusters", uniform),
                                entry("inputs", &self.inputs),
                                entry("poses", poses),
                                lights,
                                entry("counts", &self.counts),
                                indices,
                                entry("totals", &self.totals),
                                ranges,
                                entry("still_frames", still),
                            ],
                        );
                        let fragment = engine_render::pipelines::group(
                            device,
                            "engine/light-shading-probe-group",
                            &PROBE_SHADING,
                            &self.shading_layout,
                            &[entry("clusters", uniform), entry("light_data", &self.data)],
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

        fn probe_group(&self, face: usize) -> Option<wgpu::BindGroup> {
            let probes = self.probes.borrow();
            Some(probes.groups.get(probes.current)?.1.get(face)?.1.clone())
        }
    }

    fn index_room(room: u64, capacity: usize) -> u64 {
        room.saturating_sub(INDEX_START + capacity as u64 * LIGHT_BYTES) / SECTION_ALIGN
            * SECTION_ALIGN
    }

    fn data_bytes(index_bytes: u64, capacity: usize) -> u64 {
        INDEX_START + index_bytes + capacity as u64 * LIGHT_BYTES
    }

    fn section(buffer: &wgpu::Buffer, offset: u64, bytes: u64) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset,
            size: wgpu::BufferSize::new(bytes),
        })
    }

    pub(super) fn create(cx: &Create<'_>) -> Box<dyn GpuPart> {
        let device = &cx.core.device;
        Box::new(LightPass {
            idle: (
                engine_render::buffers::uniform(
                    device,
                    "engine/light-clusters-idle",
                    size_of::<ClusterUniform>() as u64,
                ),
                Versioned::new(storage(
                    device,
                    "engine/light-data-idle",
                    16,
                    wgpu::BufferUsages::empty(),
                )),
            ),
            bins: Lazy::default(),
            frame: None,
            pass: None,
            poses: None,
            lights: None,
        })
    }

    pub(crate) struct LightPass {
        idle: (wgpu::Buffer, Versioned<wgpu::Buffer>),
        bins: Lazy<LightBins>,
        frame: Option<LightFrame>,
        pass: Option<PassId>,
        poses: Option<ResourceId>,
        lights: Option<ResourceId>,
    }

    impl LightListFace for LightPass {
        fn probe_group(&self, face: usize) -> Option<wgpu::BindGroup> {
            self.frame.as_ref().and(self.bins.get())?.probe_group(face)
        }

        fn encode_probe_face(
            &self,
            gpu: (&wgpu::Device, &Queue),
            encoder: &mut wgpu::CommandEncoder,
            view: (usize, Mat4, u32),
        ) {
            if let (Some(bins), Some(frame)) = (self.bins.get(), &self.frame) {
                bins.encode_probe_face(gpu, encoder, view, frame);
            }
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
            let settings = snapshot.section::<LocalLightSettings>();
            let needs = settings.active() && !snapshot.lights.list.is_empty();
            if !needs {
                return;
            }
            let shaped = snapshot.section::<Highlights>().enabled;
            let core = &*cx.core;
            let device = &core.device;
            self.bins
                .create(device, LABELS[0], || LightBins::new(device));
            let Some(bins) = self.bins.get_mut() else {
                return;
            };
            let parity = cx.frame.temporal.as_ref().map_or(0, |frame| frame.parity);
            let poses = core.frame_poses(parity);
            self.frame = bins.prepare(
                (device, &core.queue),
                (snapshot, settings, shaped),
                core.render_pixels(),
                (poses.current(), poses.generation(), poses.parity),
                core.gpu_presentation.still_frames(),
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
            match self.bins.get() {
                Some(bins) => out.extend(bins.shading()),
                None => out.extend([
                    ("clusters", Bound::fixed(self.idle.0.as_entire_binding())),
                    ("light_data", self.idle.1.bound()),
                ]),
            }
        }

        fn shading(&self, _snapshot: &RenderSnapshot, shading: &mut Shading) {
            let created = self.bins.get().is_some();
            let lit = self.frame.is_some() && created;
            shading.set(LOCAL_LIGHTS, f64::from(u8::from(lit)));
            if created {
                shading.warm(LOCAL_LIGHTS, f64::from(u8::from(!lit)));
            }
        }

        fn after_submit(&mut self) {
            if let Some(bins) = self.bins.get_mut() {
                let totals = Rc::clone(&bins.totals_read);
                bins.readbacks.after_submit(move |words, capacity| {
                    totals.set(Some([words.get(0), words.get(1), words.get(2), capacity]));
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
            let kept = snapshot.lights.list.len().min(u32::MAX as usize) as u32;
            let cut = snapshot.lights.cut.min(u32::MAX as usize) as u32;
            stats.local_lights = LightStats {
                shaded: if lit {
                    snapshot.lights.list.len() as u32
                } else {
                    0
                },
                invalid: snapshot.lights.invalid.min(u32::MAX as usize) as u32,
                visible: totals.map(|(reached, _)| reached),
                light_cut: Cut::of(kept.saturating_add(cut), kept),
                cluster_cut: totals.map_or_else(Cut::default, |(_, cut)| cut),
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
