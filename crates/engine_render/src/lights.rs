#![expect(
    clippy::expect_used,
    reason = "shading reads the group this frame prepared"
)]

use crate::gpu_layouts::constants::GROUP_LIGHTS as LIGHTS;
pub(crate) use crate::gpu_layouts::constants::LIGHT_BYTES;
use crate::gpu_layouts::constants::{
    KEPT_REGIONS, LIGHT_EXTERIOR, LIGHT_INTERIOR, NO_SLOT, SHAPED_LIGHTS,
};
use crate::shaders::entry::lights;
use engine_core::{LightRegion, RenderLight, glam::Mat4};

pub(crate) const TILES_X: u32 = 16;
pub(crate) const TILES_Y: u32 = 9;
pub(crate) const SLICES: u32 = 24;
const CLUSTERS: u64 = (TILES_X * TILES_Y * SLICES) as u64;
const TOTALS_BYTES: u64 = 16;
pub(crate) const NO_BUDGET: u32 = u32::MAX;

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
    use std::{cell::Cell, rc::Rc};

    use engine_core::{LightStats, RenderSnapshot};

    use super::*;
    use crate::bindings::{LIGHT_BINNING, LIGHT_SHADING};
    use crate::buffers::{Growable, ReadbackRing, entry, storage};
    use crate::cached::Cached;
    use crate::gpu_data::LIGHT_SHADER;
    use crate::profiling::Span;
    use crate::render_features::{Encode, FeaturePass, RenderFeature};
    use crate::render_graph::{FrameShape, Node, Pass, RenderGraph, Resource};
    use crate::renderer::FramePoses;

    pub(crate) struct LightFrame {
        count: u32,
        group: usize,
        flags: u32,
    }

    type ProbeLightGroups = (
        (u64, usize, u64, u64),
        Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
    );

    pub(crate) struct LightPass {
        count: wgpu::ComputePipeline,
        scan: wgpu::ComputePipeline,
        fill: wgpu::ComputePipeline,
        compute_layout: crate::pipelines::GroupLayout,
        fragment_layout: crate::pipelines::GroupLayout,
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
        fragment_group: Cached<u64, wgpu::BindGroup>,
        probe_uniforms: Vec<wgpu::Buffer>,
        probe_groups: Vec<ProbeLightGroups>,
        probe_current: usize,
        no_still: wgpu::Buffer,
        buffers_generation: u64,
        readbacks: ReadbackRing<()>,
        totals_read: Rc<Cell<Option<[u32; 3]>>>,
    }

    impl LightPass {
        pub(crate) fn new(device: &wgpu::Device) -> Self {
            let compute_layout = crate::pipelines::layout(device, &LIGHT_BINNING);
            let fragment_layout = crate::pipelines::layout(device, &LIGHT_SHADING);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine/lights.wgsl"),
                source: wgpu::ShaderSource::Wgsl(LIGHT_SHADER.into()),
            });
            let layout = crate::pipelines::pipeline_layout(
                device,
                &crate::pipeline_layouts::LIGHT_BINNING_PIPELINES,
                &[&compute_layout],
            );
            let pipeline = |entry: crate::shaders::Entry| {
                crate::pipelines::compute_entry(device, &layout, &module, entry)
            };
            let none = wgpu::BufferUsages::empty();
            Self {
                count: pipeline(lights::count_lights),
                scan: pipeline(lights::scan_clusters),
                fill: pipeline(lights::bin_lights),
                compute_layout,
                fragment_layout,
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
                fragment_group: Cached::default(),
                probe_uniforms: Vec::new(),
                probe_groups: Vec::new(),
                probe_current: 0,
                no_still: storage(device, "engine/light-no-still", 16, none),
                buffers_generation: 0,
                readbacks: ReadbackRing::new("engine/light-totals-readback"),
                totals_read: Rc::new(Cell::new(None)),
            }
        }

        pub(crate) fn fragment_layout(&self) -> &crate::pipelines::GroupLayout {
            &self.fragment_layout
        }

        pub(crate) fn totals(&self) -> Option<[u32; 2]> {
            self.totals_read
                .get()
                .map(|[reached, lost, _]| [reached, lost])
        }

        pub(crate) fn capacity_bytes(&self) -> u64 {
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

        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &crate::gpu_calls::Queue,
            snapshot: &RenderSnapshot,
            size: crate::targets::FrameSize,
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
                size.width,
                size.height,
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
            if self.fragment_group.stale(&self.buffers_generation) {
                let group = crate::pipelines::group(
                    device,
                    "engine/light-shading-group",
                    &LIGHT_SHADING,
                    &self.fragment_layout,
                    &[
                        entry("clusters", &self.uniform),
                        entry("lights", &self.lights),
                        entry("cluster_ranges", &self.ranges),
                        entry("cluster_lights", &self.indices),
                    ],
                );
                self.fragment_group.set(self.buffers_generation, group);
            }
            Some(LightFrame {
                count: count as u32,
                group,
                flags: uniform.lights[1],
            })
        }

        fn index_capacity(&self) -> u32 {
            (self.indices.size() / 4).min(u64::from(u32::MAX)) as u32
        }

        pub(crate) fn fragment_group(&self) -> &wgpu::BindGroup {
            self.fragment_group.get().expect("prepared this frame")
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
        pub(crate) fn encode_probe_face(
            &mut self,
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
            if self.probe_uniforms.is_empty() {
                self.probe_uniforms = (0..6)
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
            queue.write_buffer(&self.probe_uniforms[face], 0, bytemuck::bytes_of(&uniform));
            let key = (
                poses.generation(),
                poses.parity,
                self.buffers_generation,
                still.1,
            );
            if let Some(index) = self
                .probe_groups
                .iter()
                .position(|(built, _)| *built == key)
            {
                self.probe_current = index;
            } else {
                self.probe_groups
                    .retain(|((generation, _, buffers, still), _)| {
                        (*generation, *buffers, *still) == (key.0, key.2, key.3)
                    });
                let groups = self
                    .probe_uniforms
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
                            &LIGHT_SHADING,
                            &self.fragment_layout,
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
                self.probe_groups.push((key, groups));
                self.probe_current = self.probe_groups.len() - 1;
            }
            let group = &self.probe_groups[self.probe_current].1[face].0;
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

        pub(crate) fn probe_fragment_group(&self, face: usize) -> &wgpu::BindGroup {
            &self.probe_groups[self.probe_current].1[face].1
        }
    }

    impl RenderFeature for LightPass {
        type Frame = LightFrame;
        type Stats = LightStats;

        const LABELS: &'static [&'static str] = &["engine/light-", "engine/lights"];

        fn declare(_frame: &LightFrame, _shape: &FrameShape, graph: &mut RenderGraph) {
            graph
                .add(Pass::Feature(FeaturePass::Lights))
                .read(Resource::Poses)
                .write(Resource::Lights);
        }

        fn encode(
            &self,
            frame: &LightFrame,
            _node: &Node,
            encode: &Encode<'_>,
            encoder: &mut wgpu::CommandEncoder,
        ) -> u32 {
            self.encode_binning(
                encode.device,
                encoder,
                frame,
                encode.profiler.span("light_bins"),
            );
            0
        }

        fn after_submit(&mut self, _frame: Option<&LightFrame>) {
            let totals = Rc::clone(&self.totals_read);
            self.readbacks.after_submit(move |words, ()| {
                totals.set(Some([words.get(0), words.get(1), words.get(2)]));
            });
        }

        fn capacity_bytes(&self) -> u64 {
            LightPass::capacity_bytes(self)
        }

        fn stats(
            this: Option<&Self>,
            frame: Option<&LightFrame>,
            snapshot: &RenderSnapshot,
        ) -> LightStats {
            let lit = frame.is_some();
            let totals = frame.and(this.and_then(LightPass::totals));
            LightStats {
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
            }
        }
    }
}

pub(crate) use gpu::{LightFrame, LightPass};
