//! Local lights (decision 023): each frame's lights, binned into view
//! clusters on the GPU, and the bind group lit mesh pipelines shade with.
//!
//! Lights upload only when their records change. Lights that follow a slot
//! read that slot's pose from the frame's pose buffer while binning, so they
//! move with interpolated, GPU-presented meshes at no CPU cost. Binning
//! counts each cluster's lights, turns the counts into list ranges and fills
//! the lists, so lists are exactly as long as their lights; the list storage
//! grows to what a frame read back needed. Everything here is created with
//! the first frame that has lights.

use bytemuck::{Pod, Zeroable};
use engine_core::{RenderLight, glam::Mat4};

/// Screen tiles across and down, and exponential depth slices.
pub(crate) const TILES_X: u32 = 16;
pub(crate) const TILES_Y: u32 = 9;
pub(crate) const SLICES: u32 = 24;
#[cfg(target_arch = "wasm32")]
const CLUSTERS: u64 = (TILES_X * TILES_Y * SLICES) as u64;
const NO_SLOT: u32 = u32::MAX;
/// Lights each binning workgroup places (`LIGHTS` of lights.wgsl).
#[cfg(target_arch = "wasm32")]
const LIGHTS: u32 = 8;
/// Bytes of a binned light (WGSL `Light`).
pub(crate) const LIGHT_BYTES: u64 = 48;
// Three counters, padded.
#[cfg(target_arch = "wasm32")]
const TOTALS_BYTES: u64 = 16;
/// The per-cluster budget of clusters without one.
pub(crate) const NO_BUDGET: u32 = u32::MAX;

/// WGSL `Clusters` of `lights.wgsl` and `meshes/declarations.wgsl`: 144 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ClusterUniform {
    pub view_projection: [[f32; 4]; 4],
    /// Depth along the view of point p: dot(xyz, p) + w (clip w).
    pub depth: [f32; 4],
    /// x and y projection scales, clustered depth range start and end.
    pub projection: [f32; 4],
    /// Tile width and height in pixels; slice = log2(depth) * z + w.
    pub grid: [f32; 4],
    /// Tiles across and down, depth slices, and the per-cluster budget
    /// ([`NO_BUDGET`] without one).
    pub counts: [u32; 4],
    /// Lights, an unused word, the entries the cluster lists' storage holds,
    /// and 1 when only lights without a slot or on still slots are binned
    /// (reflection probe faces, decision 043).
    pub lights: [u32; 4],
}

impl ClusterUniform {
    /// Clusters of a `width` x `height` frame seen through the unjittered
    /// perspective `view_projection` (reversed WebGPU depth, decision 042:
    /// near 1, far 0), up to view depth
    /// `distance` where local light fades out, with an optional per-cluster
    /// `budget` and list storage of `capacity` entries. Its fourth row gives
    /// view depth, its first two rows' lengths the projection scales, and
    /// the third row the near and far planes; a matrix without usable planes
    /// clusters over [0.1, 1000].
    #[allow(clippy::too_many_arguments)]
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

/// WGSL `Input` of `lights.wgsl`: one extracted light, 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct LightInput {
    pub position: [f32; 3],
    pub range: f32,
    pub color: [f32; 3],
    pub slot: u32,
    pub direction: [f32; 3],
    pub cos_inner: f32,
    pub cos_outer: f32,
    pub pad: [f32; 3],
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
            pad: [0.0; 3],
        }
    }
}

/// GPU binning, wasm32 only like the renderer.
#[cfg(target_arch = "wasm32")]
mod gpu {
    use std::{cell::Cell, rc::Rc};

    use engine_core::RenderSnapshot;

    use super::*;
    use crate::bindings::{LIGHT_BINNING, LIGHT_SHADING};
    use crate::browser::FramePoses;
    use crate::buffers::{ReadbackRing, entry, storage};
    use crate::cached::Cached;
    use crate::gpu_data::LIGHT_SHADER;
    use crate::profiling::Span;

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
        compute_layout: wgpu::BindGroupLayout,
        fragment_layout: wgpu::BindGroupLayout,
        uniform: wgpu::Buffer,
        written: Option<ClusterUniform>,
        inputs: wgpu::Buffer,
        lights: wgpu::Buffer,
        capacity: usize,
        counts: wgpu::Buffer,
        ranges: wgpu::Buffer,
        indices: wgpu::Buffer,
        // Entries `indices` holds.
        index_capacity: u64,
        // What one storage binding holds, read once.
        binding_limit: u64,
        totals: wgpu::Buffer,
        staged: Vec<LightInput>,
        // The retained stream and light revision the inputs hold.
        uploaded_revision: Option<(u64, u64)>,
        // Binning groups over (pose generation, pose buffer) and the storage
        // generation of this pass's own buffers they were built with.
        compute_groups: Vec<((u64, usize, u64), wgpu::BindGroup)>,
        fragment_group: Cached<u64, wgpu::BindGroup>,
        // Reflection probe faces (decision 043): each face's clusters, their
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
    }

    impl LightPass {
        pub(crate) fn new(device: &wgpu::Device) -> Self {
            let compute_layout = crate::pipelines::layout(device, &LIGHT_BINNING);
            let fragment_layout = crate::pipelines::layout(device, &LIGHT_SHADING);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine/lights.wgsl"),
                source: wgpu::ShaderSource::Wgsl(LIGHT_SHADER.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/light-binning-pipeline-layout"),
                bind_group_layouts: &[Some(&compute_layout)],
                immediate_size: 0,
            });
            let pipeline =
                |entry: &str| crate::pipelines::compute_entry(device, &layout, &module, entry);
            let none = wgpu::BufferUsages::empty();
            Self {
                count: pipeline("count_lights"),
                scan: pipeline("scan_clusters"),
                fill: pipeline("bin_lights"),
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
                indices: storage(device, "engine/light-cluster-indices", CLUSTERS * 4, none),
                index_capacity: CLUSTERS,
                binding_limit: device.limits().max_storage_buffer_binding_size,
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
            }
        }

        /// Group 3 of the lit mesh pipelines.
        pub(crate) fn fragment_layout(&self) -> &wgpu::BindGroupLayout {
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
            .sum()
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
            size: (u32, u32),
            poses: FramePoses<'_>,
            revision: (u64, u64),
        ) -> Option<LightFrame> {
            if !snapshot.environment.local_lights.active() || snapshot.lights.is_empty() {
                return None;
            }
            let count = snapshot.lights.len();
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
                    .extend(snapshot.lights.iter().map(LightInput::from));
                queue.write_buffer(&self.inputs, 0, bytemuck::cast_slice(&self.staged));
                self.uploaded_revision = Some(revision);
            }
            // List storage grows to what the lists needed when last read
            // back, and to a first guess of a few dozen clusters per light,
            // within what one storage binding holds; until then, entries
            // beyond it are lost and counted.
            let binding = self.binding_limit / 4;
            let needed = self
                .totals_read
                .get()
                .map_or(0, |[_, _, needed]| u64::from(needed))
                .max(count as u64 * 32);
            if needed > self.index_capacity && self.index_capacity < binding {
                self.index_capacity = needed.next_power_of_two().min(binding);
                self.indices = storage(
                    device,
                    "engine/light-cluster-indices",
                    self.index_capacity * 4,
                    wgpu::BufferUsages::empty(),
                );
                self.buffers_generation += 1;
            }
            let uniform = ClusterUniform::new(
                snapshot.view_projection,
                size.0,
                size.1,
                snapshot.environment.local_lights.distance,
                count as u32,
                snapshot.environment.local_lights.lights_per_cluster,
                self.index_capacity.min(u64::from(u32::MAX)) as u32,
            );
            if self.written != Some(uniform) {
                queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
                self.written = Some(uniform);
            }
            let key = (poses.generation, poses.parity, self.buffers_generation);
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

        /// The bindings lit mesh pipelines read at group 3.
        pub(crate) fn fragment_group(&self) -> &wgpu::BindGroup {
            self.fragment_group.get().expect("prepared this frame")
        }

        /// Records this frame's binning: clears the clusters, counts every
        /// light's clusters, gives the clusters their list ranges, fills the
        /// lists and copies the totals for read-back.
        /// With `span`, the binning pass is timed.
        pub(crate) fn encode(
            &mut self,
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

        /// Records the binning of reflection probe cube face `face` (decision
        /// 043), `size` texels across, seen through `view_projection`: the
        /// lights of `frame` without a slot or on slots `still` (storage and
        /// generation) counts still, into the lists the frame's own shading
        /// read before; [`Self::probe_fragment_group`] shades with them.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn encode_probe_face(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            encoder: &mut wgpu::CommandEncoder,
            face: usize,
            view_projection: Mat4,
            size: u32,
            settings: &engine_core::LocalLights,
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
                self.index_capacity.min(u64::from(u32::MAX)) as u32,
            );
            uniform.lights[3] = 1;
            queue.write_buffer(&self.probe_uniforms[face], 0, bytemuck::bytes_of(&uniform));
            let key = (
                poses.generation,
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

        /// Requests maps of totals recorded in the frame just submitted.
        pub(crate) fn after_submit(&mut self) {
            let totals = Rc::clone(&self.totals_read);
            self.readbacks.after_submit(move |words, ()| {
                totals.set(Some([words.get(0), words.get(1), words.get(2)]));
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use gpu::{LightFrame, LightPass};
