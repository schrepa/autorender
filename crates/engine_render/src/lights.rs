//! Local lights: each frame's lights, binned into view
//! clusters on the GPU, and the bind group lit mesh pipelines shade with.
//!
//! Lights upload only when their records change. Lights that follow a slot
//! read that slot's pose from the frame's pose buffer while binning, so they
//! move with interpolated, GPU-presented meshes at no CPU cost. Binning
//! counts each cluster's lights, turns the counts into list ranges and fills
//! the lists, so lists are exactly as long as their lights; the list storage
//! grows to what a frame read back needed. Everything here is created with
//! the first frame that has lights.

#![expect(
    clippy::expect_used,
    reason = "shading reads the group this frame prepared"
)]

use crate::gpu_layouts::constants::GROUP_LIGHTS as LIGHTS;
pub(crate) use crate::gpu_layouts::constants::LIGHT_BYTES;
use crate::gpu_layouts::constants::NO_SLOT;
use crate::shaders::entry::lights;
use engine_core::{RenderLight, glam::Mat4};

/// Screen tiles across and down, and exponential depth slices.
pub(crate) const TILES_X: u32 = 16;
pub(crate) const TILES_Y: u32 = 9;
pub(crate) const SLICES: u32 = 24;
const CLUSTERS: u64 = (TILES_X * TILES_Y * SLICES) as u64;
// Three counters, padded.
const TOTALS_BYTES: u64 = 16;
/// The per-cluster budget of clusters without one.
pub(crate) const NO_BUDGET: u32 = u32::MAX;

pub(crate) use crate::gpu_layouts::lights::ClusterUniform;

impl ClusterUniform {
    /// Clusters of a `width` x `height` frame seen through the unjittered
    /// perspective `view_projection` (reversed WebGPU depth:
    /// near 1, far 0), up to view depth
    /// `distance` where local light fades out, with an optional per-cluster
    /// `budget` and list storage of `capacity` entries. Its fourth row gives
    /// view depth, its first two rows' lengths the projection scales, and
    /// the third row the near and far planes; a matrix without usable planes
    /// clusters over [0.1, 1000].
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
        // Row 2 is P22 times the view's z row plus P23, and row 3 is minus
        // the view's z row (of unit length): with reversed depth P22 =
        // near / (far - near) > 0, depth -P22 + P23 / d is 1 at the near
        // plane and 0 at the far one.
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
        // Light ends at the fade distance.
        let far = engine_core::math::max(engine_core::math::min(far, distance), near * 2.0);
        // The first slice spans everything nearer than the start.
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
            pad: [0.0; 3],
        }
    }
}

/// GPU binning, wasm32 only like the renderer.
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

    /// A frame's binned lights.
    pub(crate) struct LightFrame {
        count: u32,
        group: usize,
    }

    // Binning and shading groups of each reflection probe face, and the
    // (pose generation, pose buffer, storage generation, still-frame
    // generation) they were built for.
    type ProbeLightGroups = (
        (u64, usize, u64, u64),
        Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
    );

    pub(crate) struct LightPass {
        // Counting, scanning and filling the cluster lists.
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
        // The clusters' lists of lights, an entry each.
        indices: Growable,
        totals: wgpu::Buffer,
        staged: Vec<LightInput>,
        // The retained stream and light revision the inputs hold.
        uploaded_revision: Option<(u64, u64)>,
        // Binning groups over (pose generation, pose buffer) and the storage
        // generation of this pass's own buffers they were built with.
        compute_groups: Vec<((u64, usize, u64), wgpu::BindGroup)>,
        fragment_group: Cached<u64, wgpu::BindGroup>,
        // Reflection probe faces: each face's clusters, their
        // binning and shading groups over the pose, still-frame and list
        // storage they were built for (the last used current), and the
        // still frames main binning leaves unread.
        probe_uniforms: Vec<wgpu::Buffer>,
        probe_groups: Vec<ProbeLightGroups>,
        probe_current: usize,
        no_still: wgpu::Buffer,
        buffers_generation: u64,
        readbacks: ReadbackRing<()>,
        // Lights that reached a cluster, cluster entries lost to the budget
        // or the storage, and the entries the lists needed, read back a few
        // frames late.
        totals_read: Rc<Cell<Option<[u32; 3]>>>,
        depth: Option<crate::light_depth::DepthPass>,
        depth_tag: u32,
        depth_fallback: Option<&'static str>,
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
                compute_groups: Vec::new(),
                fragment_group: Cached::default(),
                probe_uniforms: Vec::new(),
                probe_groups: Vec::new(),
                probe_current: 0,
                no_still: storage(device, "engine/light-no-still", 16, none),
                buffers_generation: 0,
                readbacks: ReadbackRing::new("engine/light-totals-readback"),
                totals_read: Rc::new(Cell::new(None)),
                depth: None,
                depth_tag: 0,
                depth_fallback: None,
            }
        }

        /// Group 3 of the lit mesh pipelines.
        pub(crate) fn fragment_layout(&self) -> &crate::pipelines::GroupLayout {
            &self.fragment_layout
        }

        /// Lights that reached a cluster and cluster entries lost to the
        /// per-cluster budget or to list storage still growing, read back a
        /// few frames late.
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
            .sum::<u64>()
                + self
                    .depth
                    .as_ref()
                    .map_or(0, crate::light_depth::DepthPass::extra_bytes)
        }

        pub(crate) fn depth(&self) -> Option<&crate::light_depth::DepthPass> {
            self.depth.as_ref()
        }

        fn release_depth(&mut self, device: &wgpu::Device) {
            if self.depth.take().is_some() {
                self.ranges = storage(
                    device,
                    "engine/light-cluster-ranges",
                    CLUSTERS * 8,
                    wgpu::BufferUsages::empty(),
                );
                self.buffers_generation += 1;
                self.compute_groups.clear();
                self.fragment_group = Cached::default();
                self.probe_groups.clear();
            }
        }

        /// Configures the optional arena only after the frame knows its actual
        /// ordinary forward batches. Call `prepare` again if bindings changed.
        pub(crate) fn configure_depth(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            size: crate::targets::FrameSize,
            (view, eligible): (u64, u32),
            enabled: bool,
        ) -> bool {
            let before = self.buffers_generation;
            self.depth_fallback = None;
            let Some(uniform) = self.written.filter(|_| enabled && eligible != 0) else {
                self.depth_fallback = Some(if enabled {
                    "no_eligible_draws"
                } else {
                    "inactive"
                });
                self.release_depth(device);
                return self.buffers_generation != before;
            };
            let Some(table) = crate::light_depth::thresholds(&uniform) else {
                self.depth_fallback = Some("unsupported_view");
                self.release_depth(device);
                return self.buffers_generation != before;
            };
            let limits = device.limits();
            let limit = limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size);
            let Some(bytes) = crate::light_depth::arena_bytes(self.index_capacity(), limit) else {
                self.depth_fallback = Some("arena_binding_limit");
                self.release_depth(device);
                return self.buffers_generation != before;
            };
            if self.depth.as_ref().is_none_or(|depth| {
                depth.storage_generation != self.buffers_generation || depth.size != size
            }) {
                self.ranges = storage(
                    device,
                    "engine/light-depth-arena",
                    bytes,
                    wgpu::BufferUsages::COPY_SRC,
                );
                self.buffers_generation += 1;
                self.depth = Some(crate::light_depth::DepthPass::new(
                    device,
                    size,
                    self.buffers_generation,
                    crate::light_depth::DepthBuffers {
                        arena: &self.ranges,
                        lights: &self.lights,
                        indices: &self.indices,
                        shading: &self.fragment_layout,
                    },
                ));
                self.compute_groups.clear();
                self.fragment_group = Cached::default();
                self.probe_groups.clear();
            }
            let tags = std::array::from_fn(|_| {
                self.depth_tag = self.depth_tag.wrapping_add(1).max(1);
                self.depth_tag
            });
            if let Some(depth) = &mut self.depth {
                depth.prepare(queue, (view, uniform), table, tags, eligible);
            }
            self.buffers_generation != before
        }

        /// Uploads `snapshot`'s lights when their `revision` (retained stream
        /// and light revision) changed and writes this frame's clusters for a
        /// `size` frame; `None` when it shades no local lights. Slot lights
        /// read `poses`.
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
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
                self.inputs = storage(
                    device,
                    "engine/light-inputs",
                    self.capacity as u64 * 64,
                    wgpu::BufferUsages::empty(),
                );
                self.lights = storage(
                    device,
                    "engine/lights",
                    self.capacity as u64 * LIGHT_BYTES,
                    wgpu::BufferUsages::empty(),
                );
                self.uploaded_revision = None;
                self.buffers_generation += 1;
            }
            // Unchanged lights skip staging.
            if self.uploaded_revision != Some(revision) {
                self.staged.clear();
                self.staged
                    .extend(snapshot.lights.list.iter().map(LightInput::from));
                queue.write_buffer(&self.inputs, 0, bytemuck::cast_slice(&self.staged));
                self.uploaded_revision = Some(revision);
            }
            // List storage grows to what the lists needed when last read
            // back, and to a first guess of a few dozen clusters per light,
            // within what one storage binding holds; until then, entries
            // beyond it are lost and counted.
            let needed = self
                .totals_read
                .get()
                .map_or(0, |[_, _, needed]| u64::from(needed))
                .max(count as u64 * 32);
            if self.indices.grow(device, needed * 4) {
                self.buffers_generation += 1;
            }
            let uniform = ClusterUniform::new(
                snapshot.view_projection,
                size.width,
                size.height,
                snapshot.environment.local_lights.distance,
                count as u32,
                snapshot.environment.local_lights.lights_per_cluster,
                self.index_capacity(),
            );
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
                    // Stale generations never come back.
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
            })
        }

        /// Entries the clusters' lists hold: what binning may write.
        fn index_capacity(&self) -> u32 {
            (self.indices.size() / 4).min(u64::from(u32::MAX)) as u32
        }

        /// The bindings lit mesh pipelines read at group 3.
        pub(crate) fn fragment_group(&self) -> &wgpu::BindGroup {
            self.fragment_group.get().expect("prepared this frame")
        }

        /// Records this frame's binning: clears the clusters, counts every
        /// light's clusters, gives the clusters their list ranges, fills the
        /// lists and copies the totals for read-back.
        /// With `span`, the binning pass is timed.
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
                // LIGHTS lights per workgroup, in rows of at most 65,535.
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

        /// Records the binning of reflection probe cube face `face`,
        /// `size` texels across, seen through `view_projection`: the
        /// lights of `frame` without a slot or on slots `still` (storage and
        /// generation) counts still, into the lists the frame's own shading
        /// read before; [`Self::probe_fragment_group`] shades with them.
        #[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
        pub(crate) fn encode_probe_face(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
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
                // Stale generations never come back.
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

        /// The bindings lit probe pipelines read at group 3 for face `face`,
        /// after [`Self::encode_probe_face`] recorded its binning.
        pub(crate) fn probe_fragment_group(&self, face: usize) -> &wgpu::BindGroup {
            &self.probe_groups[self.probe_current].1[face].1
        }
    }

    impl RenderFeature for LightPass {
        type Frame = LightFrame;
        type Stats = LightStats;

        const LABELS: &'static [&'static str] = &["engine/light-", "engine/lights"];

        /// Binning reads the poses lights ride on and writes their lists.
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

        /// Requests maps of totals recorded in the frame just submitted.
        fn after_submit(&mut self, _frame: Option<&LightFrame>) {
            let totals = Rc::clone(&self.totals_read);
            self.readbacks.after_submit(move |words, ()| {
                totals.set(Some([words.get(0), words.get(1), words.get(2)]));
            });
            if let Some(depth) = &self.depth {
                depth.after_submit();
            }
        }

        fn capacity_bytes(&self) -> u64 {
            LightPass::capacity_bytes(self)
        }

        /// Lights shaded and dropped in frames shading them, invalid ones,
        /// and the binning's totals, read back a few frames late.
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
                depth_occupancy: snapshot.environment.local_lights.depth_occupancy.then(|| {
                    if let Some(depth) = this.and_then(|pass| pass.depth.as_ref()).filter(|_| lit) {
                        depth.stats()
                    } else {
                        engine_core::DepthOccupancyStats {
                            enabled: true,
                            fallback: Some(if lit {
                                this.and_then(|pass| pass.depth_fallback)
                                    .unwrap_or("inactive")
                            } else {
                                "no_local_lights"
                            }),
                            ..Default::default()
                        }
                    }
                }),
            }
        }
    }
}

pub(crate) use gpu::{LightFrame, LightPass};
