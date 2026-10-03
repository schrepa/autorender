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
#[cfg(target_arch = "wasm32")]
const READBACKS: usize = 3;
// Three counters, padded.
#[cfg(target_arch = "wasm32")]
const TOTALS_BYTES: u64 = 16;
/// The per-cluster budget of clusters without one.
pub(crate) const NO_BUDGET: u32 = u32::MAX;

/// WGSL `Clusters` of `lights.wgsl` and `meshes.wgsl`: 144 bytes.
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
    /// Lights, 1 when slot poses are bound, the entries the cluster lists'
    /// storage holds, and 1 when only lights without a slot or on still
    /// slots are binned (reflection probe faces, decision 043).
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
        poses: bool,
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
            lights: [lights, u32::from(poses), capacity, 0],
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
    use crate::gpu_data::LIGHT_SHADER;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ReadbackState {
        Idle,
        Recorded,
        Pending,
    }

    struct Readback {
        buffer: wgpu::Buffer,
        state: Rc<Cell<ReadbackState>>,
    }

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

    /// Pose storage a frame's slot lights read: a buffer, its storage
    /// generation and which of the frame's pose buffers it is.
    pub(crate) struct LightPoses<'a> {
        pub buffer: &'a wgpu::Buffer,
        pub generation: u64,
        pub index: usize,
    }

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
        totals: wgpu::Buffer,
        no_poses: wgpu::Buffer,
        staged: Vec<LightInput>,
        uploaded: Vec<LightInput>,
        // The retained stream and light revision the inputs hold, when a
        // retained frame uploaded them.
        uploaded_revision: Option<(u64, u64)>,
        // Binning groups over (pose generation, pose buffer) and the storage
        // generation of this pass's own buffers they were built with.
        compute_groups: Vec<((u64, usize, u64), wgpu::BindGroup)>,
        fragment_group: Option<(u64, wgpu::BindGroup)>,
        // Reflection probe faces (decision 043): each face's clusters, their
        // binning and shading groups over the pose, still-frame and list
        // storage they were built for (the last used current), and the
        // still frames main binning leaves unread.
        probe_uniforms: Vec<wgpu::Buffer>,
        probe_groups: Vec<ProbeLightGroups>,
        probe_current: usize,
        no_still: wgpu::Buffer,
        buffers_generation: u64,
        readbacks: Vec<Readback>,
        // Lights that reached a cluster, cluster entries lost to the budget
        // or the storage, and the entries the lists needed, read back a few
        // frames late.
        totals_read: Rc<Cell<Option<[u32; 3]>>>,
    }

    fn storage(
        device: &wgpu::Device,
        label: &str,
        bytes: u64,
        extra: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16).next_multiple_of(16),
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

    fn layout_entry(
        binding: u32,
        visibility: wgpu::ShaderStages,
        ty: wgpu::BufferBindingType,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    impl LightPass {
        pub(crate) fn new(device: &wgpu::Device) -> Self {
            let compute = wgpu::ShaderStages::COMPUTE;
            let read = wgpu::BufferBindingType::Storage { read_only: true };
            let write = wgpu::BufferBindingType::Storage { read_only: false };
            let compute_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("engine/light-binning-layout"),
                    entries: &[
                        layout_entry(0, compute, wgpu::BufferBindingType::Uniform),
                        layout_entry(1, compute, read),
                        layout_entry(2, compute, read),
                        layout_entry(3, compute, write),
                        layout_entry(4, compute, write),
                        layout_entry(5, compute, write),
                        layout_entry(6, compute, write),
                        layout_entry(7, compute, write),
                        layout_entry(8, compute, read),
                    ],
                });
            let fragment = wgpu::ShaderStages::FRAGMENT;
            let fragment_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("engine/light-shading-layout"),
                    entries: &[
                        layout_entry(0, fragment, wgpu::BufferBindingType::Uniform),
                        layout_entry(1, fragment, read),
                        layout_entry(2, fragment, read),
                        layout_entry(3, fragment, read),
                    ],
                });
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
                uniform: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("engine/light-clusters-uniform"),
                    size: size_of::<ClusterUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                written: None,
                inputs: storage(device, "engine/light-inputs", 64, none),
                lights: storage(device, "engine/lights", LIGHT_BYTES, none),
                capacity: 1,
                counts: storage(device, "engine/light-cluster-counts", CLUSTERS * 4, none),
                ranges: storage(device, "engine/light-cluster-ranges", CLUSTERS * 8, none),
                indices: storage(device, "engine/light-cluster-indices", CLUSTERS * 4, none),
                index_capacity: CLUSTERS,
                totals: storage(
                    device,
                    "engine/light-totals",
                    TOTALS_BYTES,
                    wgpu::BufferUsages::COPY_SRC,
                ),
                no_poses: storage(device, "engine/light-no-poses", 48, none),
                staged: Vec::new(),
                uploaded: Vec::new(),
                uploaded_revision: None,
                compute_groups: Vec::new(),
                fragment_group: None,
                probe_uniforms: Vec::new(),
                probe_groups: Vec::new(),
                probe_current: 0,
                no_still: storage(device, "engine/light-no-still", 16, none),
                buffers_generation: 0,
                readbacks: Vec::new(),
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

        /// Uploads `snapshot`'s lights when they changed and writes this frame's
        /// clusters for a `size` frame; `None` when it shades no local lights.
        /// Slot lights read `poses`; without them (public snapshots) lights are
        /// in world space.
        pub(crate) fn prepare(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            snapshot: &RenderSnapshot,
            size: (u32, u32),
            poses: Option<LightPoses<'_>>,
            revision: Option<(u64, u64)>,
        ) -> Option<LightFrame> {
            if !snapshot.local_lights.active() || snapshot.lights.is_empty() {
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
                self.uploaded.clear();
                self.uploaded_revision = None;
                self.buffers_generation += 1;
            }
            // Retained frames name their lights' revision: unchanged lights
            // skip staging. Owned snapshots compare their inputs' bytes.
            if revision.is_none() || revision != self.uploaded_revision {
                self.staged.clear();
                self.staged
                    .extend(snapshot.lights.iter().map(LightInput::from));
                let bytes: &[u8] = bytemuck::cast_slice(&self.staged);
                if revision.is_some() || bytes != bytemuck::cast_slice::<_, u8>(&self.uploaded) {
                    queue.write_buffer(&self.inputs, 0, bytes);
                    if revision.is_some() {
                        self.uploaded.clear();
                    } else {
                        self.uploaded.clone_from(&self.staged);
                    }
                }
                self.uploaded_revision = revision;
            }
            // List storage grows to what the lists needed when last read
            // back, and to a first guess of a few dozen clusters per light,
            // within what one storage binding holds; until then, entries
            // beyond it are lost and counted.
            let binding = device.limits().max_storage_buffer_binding_size / 4;
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
                snapshot.local_lights.distance,
                count as u32,
                poses.is_some(),
                snapshot.local_lights.lights_per_cluster,
                self.index_capacity.min(u64::from(u32::MAX)) as u32,
            );
            if self.written != Some(uniform) {
                queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
                self.written = Some(uniform);
            }
            let key = poses
                .as_ref()
                .map_or((0, usize::MAX, self.buffers_generation), |poses| {
                    (poses.generation, poses.index, self.buffers_generation)
                });
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
                    let pose_buffer = poses.as_ref().map_or(&self.no_poses, |poses| poses.buffer);
                    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("engine/light-binning-group"),
                        layout: &self.compute_layout,
                        entries: &[
                            entry(0, &self.uniform),
                            entry(1, &self.inputs),
                            entry(2, pose_buffer),
                            entry(3, &self.lights),
                            entry(4, &self.counts),
                            entry(5, &self.indices),
                            entry(6, &self.totals),
                            entry(7, &self.ranges),
                            entry(8, &self.no_still),
                        ],
                    });
                    self.compute_groups.push((key, group));
                    self.compute_groups.len() - 1
                }
            };
            if self
                .fragment_group
                .as_ref()
                .is_none_or(|(built, _)| *built != self.buffers_generation)
            {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine/light-shading-group"),
                    layout: &self.fragment_layout,
                    entries: &[
                        entry(0, &self.uniform),
                        entry(1, &self.lights),
                        entry(2, &self.ranges),
                        entry(3, &self.indices),
                    ],
                });
                self.fragment_group = Some((self.buffers_generation, group));
            }
            Some(LightFrame {
                count: count as u32,
                group,
            })
        }

        /// The bindings lit mesh pipelines read at group 3.
        pub(crate) fn fragment_group(&self) -> &wgpu::BindGroup {
            &self.fragment_group.as_ref().expect("prepared this frame").1
        }

        /// Records this frame's binning: clears the clusters, counts every
        /// light's clusters, gives the clusters their list ranges, fills the
        /// lists and copies the totals for read-back.
        /// With `timestamps` (a query set and two indices), the binning pass
        /// writes its begin and end queries.
        pub(crate) fn encode(
            &mut self,
            device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            frame: &LightFrame,
            timestamps: Option<(&wgpu::QuerySet, u32)>,
        ) {
            encoder.clear_buffer(&self.counts, 0, None);
            encoder.clear_buffer(&self.totals, 0, None);
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("engine/light-binning"),
                    timestamp_writes: timestamps.map(|(query_set, first)| {
                        wgpu::ComputePassTimestampWrites {
                            query_set,
                            beginning_of_pass_write_index: Some(first),
                            end_of_pass_write_index: Some(first + 1),
                        }
                    }),
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
            if let Some(index) = self.idle_readback(device) {
                let readback = &self.readbacks[index];
                encoder.copy_buffer_to_buffer(&self.totals, 0, &readback.buffer, 0, TOTALS_BYTES);
                readback.state.set(ReadbackState::Recorded);
            }
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
            poses: LightPoses<'_>,
            still: (&wgpu::Buffer, u64),
        ) {
            if self.probe_uniforms.is_empty() {
                self.probe_uniforms = (0..6)
                    .map(|_| {
                        device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("engine/light-clusters-probe"),
                            size: size_of::<ClusterUniform>() as u64,
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                            mapped_at_creation: false,
                        })
                    })
                    .collect();
            }
            let mut uniform = ClusterUniform::new(
                view_projection,
                size,
                size,
                settings.distance,
                frame.count,
                true,
                settings.lights_per_cluster,
                self.index_capacity.min(u64::from(u32::MAX)) as u32,
            );
            uniform.lights[3] = 1;
            queue.write_buffer(&self.probe_uniforms[face], 0, bytemuck::bytes_of(&uniform));
            let key = (
                poses.generation,
                poses.index,
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
                        let compute = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("engine/light-binning-probe-group"),
                            layout: &self.compute_layout,
                            entries: &[
                                entry(0, uniform),
                                entry(1, &self.inputs),
                                entry(2, poses.buffer),
                                entry(3, &self.lights),
                                entry(4, &self.counts),
                                entry(5, &self.indices),
                                entry(6, &self.totals),
                                entry(7, &self.ranges),
                                entry(8, still.0),
                            ],
                        });
                        let fragment = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("engine/light-shading-probe-group"),
                            layout: &self.fragment_layout,
                            entries: &[
                                entry(0, uniform),
                                entry(1, &self.lights),
                                entry(2, &self.ranges),
                                entry(3, &self.indices),
                            ],
                        });
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

        fn idle_readback(&mut self, device: &wgpu::Device) -> Option<usize> {
            if let Some(index) = self
                .readbacks
                .iter()
                .position(|readback| readback.state.get() == ReadbackState::Idle)
            {
                return Some(index);
            }
            if self.readbacks.len() == READBACKS {
                return None;
            }
            self.readbacks.push(Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("engine/light-totals-readback"),
                    size: TOTALS_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Rc::new(Cell::new(ReadbackState::Idle)),
            });
            Some(self.readbacks.len() - 1)
        }

        /// Requests maps of totals recorded in the frame just submitted.
        pub(crate) fn after_submit(&mut self) {
            for readback in &self.readbacks {
                if readback.state.get() != ReadbackState::Recorded {
                    continue;
                }
                readback.state.set(ReadbackState::Pending);
                let state = Rc::clone(&readback.state);
                let totals = Rc::clone(&self.totals_read);
                let buffer = readback.buffer.clone();
                readback
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        if result.is_ok() {
                            if let Ok(view) = buffer.slice(..).get_mapped_range() {
                                let word = |index: usize| {
                                    view.get(index * 4..index * 4 + 4).map_or(0, |bytes| {
                                        u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4]))
                                    })
                                };
                                totals.set(Some([word(0), word(1), word(2)]));
                            }
                            buffer.unmap();
                        }
                        state.set(ReadbackState::Idle);
                    });
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use gpu::{LightFrame, LightPass, LightPoses};

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::glam::Vec3;

    #[test]
    fn clusters_recover_view_depth_scales_and_planes_from_the_view_projection() {
        let eye = Vec3::new(3.0, 5.0, 12.0);
        let target = Vec3::new(-1.0, 0.5, -4.0);
        let camera = engine_core::Camera {
            position: eye,
            target,
            vertical_fov_radians: 0.9,
            near: 0.5,
            far: 800.0,
        };
        let view_projection = camera.view_projection(1.6);
        let uniform = ClusterUniform::new(view_projection, 1600, 900, 1.0e9, 7, true, None, 4096);
        let forward = (target - eye).normalize();
        let point = Vec3::new(10.0, -3.0, -40.0);
        let depth = Vec3::from_slice(&uniform.depth[..3]).dot(point) + uniform.depth[3];
        assert!((depth - (point - eye).dot(forward)).abs() < 1.0e-3);
        let focal = 1.0 / (0.45_f32).tan();
        assert!((uniform.projection[1] - focal).abs() < 1.0e-4);
        assert!((uniform.projection[0] - focal / 1.6).abs() < 1.0e-4);
        assert!((uniform.projection[3] - 800.0).abs() < 0.5);
        // The first slice ends at max(near, min(1, far / 100)) = 1.
        assert!((uniform.projection[2] - 1.0).abs() < 1.0e-3);
        let slice = |depth: f32| depth.log2() * uniform.grid[2] + uniform.grid[3];
        assert!(slice(1.0).abs() < 1.0e-3);
        assert!((slice(800.0) - SLICES as f32).abs() < 1.0e-2);
        assert_eq!(uniform.grid[..2], [100.0, 100.0]);
        assert_eq!(uniform.counts, [TILES_X, TILES_Y, SLICES, NO_BUDGET]);
        assert_eq!(uniform.lights, [7, 1, 4096, 0]);
        let budgeted =
            ClusterUniform::new(view_projection, 1600, 900, 1.0e9, 7, true, Some(32), 4096);
        assert_eq!(budgeted.counts[3], 32);
        // Without usable planes the clusters span [0.1, 1000].
        let flat = ClusterUniform::new(Mat4::IDENTITY, 10, 10, 1.0e9, 1, false, None, 1);
        assert_eq!(flat.projection[3], 1000.0);
        // Light fades out by its distance, before the far plane.
        let near = ClusterUniform::new(view_projection, 1600, 900, 120.0, 7, true, None, 1);
        assert_eq!(near.projection[3], 120.0);
    }

    #[test]
    fn light_inputs_keep_slots_and_cones() {
        let light = RenderLight {
            position: Vec3::new(1.0, 2.0, 3.0),
            range: 9.0,
            color: Vec3::new(4.0, 5.0, 6.0),
            direction: Vec3::NEG_Y,
            cone: [0.9, 0.8],
            slot: Some(12),
        };
        let input = LightInput::from(&light);
        assert_eq!(input.slot, 12);
        assert_eq!([input.cos_inner, input.cos_outer], [0.9, 0.8]);
        let world = LightInput::from(&RenderLight {
            slot: None,
            cone: RenderLight::POINT,
            ..light
        });
        assert_eq!(world.slot, NO_SLOT);
        assert_eq!(size_of::<LightInput>(), 64);
        assert_eq!(size_of::<ClusterUniform>(), 144);
    }
}

#[cfg(test)]
mod binning_tests {
    use super::*;
    use engine_core::glam::{Vec3, Vec4};

    /// The clusters `bin_lights` covers for a light at `position`.
    fn covered(uniform: &ClusterUniform, position: Vec3, radius: f32) -> u32 {
        use engine_core::math::{clamp, max, min};
        let view_projection = Mat4::from_cols_array_2d(&uniform.view_projection);
        let clip = view_projection * position.extend(1.0);
        let depth = Vec4::from(uniform.depth).truncate().dot(position) + uniform.depth[3];
        assert!(
            (depth - clip.w).abs() < 1.0e-3,
            "depth {depth} clip w {}",
            clip.w
        );
        if depth + radius <= 0.0 || depth - radius > uniform.projection[3] {
            return 0;
        }
        let slice = |depth: f32| {
            clamp(
                (max(depth, 1.0e-6).log2() * uniform.grid[2] + uniform.grid[3]).floor(),
                0.0,
                (SLICES - 1) as f32,
            ) as u32
        };
        let near = depth - radius;
        let far = depth + radius;
        let range = |center: f32, extent: f32| {
            let high = center + extent;
            let low = center - extent;
            (
                low / if low > 0.0 { far } else { near },
                high / if high > 0.0 { near } else { far },
            )
        };
        let (mut tiles_x, mut tiles_y) = (TILES_X, TILES_Y);
        if near > 1.0e-3 {
            let x = range(clip.x, radius * uniform.projection[0]);
            let y = range(clip.y, radius * uniform.projection[1]);
            let first = clamp(((x.0 * 0.5 + 0.5) * TILES_X as f32).floor(), 0.0, 15.0);
            let last = clamp(((x.1 * 0.5 + 0.5) * TILES_X as f32).floor(), 0.0, 15.0);
            let top = clamp(((0.5 - y.1 * 0.5) * TILES_Y as f32).floor(), 0.0, 8.0);
            let bottom = clamp(((0.5 - y.0 * 0.5) * TILES_Y as f32).floor(), 0.0, 8.0);
            tiles_x = (last - first) as u32 + 1;
            tiles_y = (bottom - top) as u32 + 1;
        }
        let slices = slice(far) - slice(near) + 1;
        let boxed = tiles_x * tiles_y * slices;
        // Count as the shader does: clusters whose view-space box the
        // sphere reaches.
        let view = Vec3::new(
            clip.x / uniform.projection[0],
            clip.y / uniform.projection[1],
            depth,
        );
        let unslice = |z: u32| ((z as f32 - uniform.grid[3]) / uniform.grid[2]).exp2();
        let mut touched = 0;
        let tile = (2.0 / TILES_X as f32, 2.0 / TILES_Y as f32);
        for z in slice(near)..=slice(far) {
            let (z0, z1) = (if z == 0 { 0.0 } else { unslice(z) }, unslice(z + 1));
            let dz = max(max(z0 - depth, depth - z1), 0.0);
            for y in 0..TILES_Y {
                let top = 1.0 - y as f32 * tile.1;
                let bottom = top - tile.1;
                for x in 0..TILES_X {
                    let left = -1.0 + x as f32 * tile.0;
                    let right = left + tile.0;
                    let x_low = min(left * z0, left * z1) / uniform.projection[0];
                    let x_high = max(right * z0, right * z1) / uniform.projection[0];
                    let y_low = min(bottom * z0, bottom * z1) / uniform.projection[1];
                    let y_high = max(top * z0, top * z1) / uniform.projection[1];
                    let dx = max(max(x_low - view.x, view.x - x_high), 0.0);
                    let dy = max(max(y_low - view.y, view.y - y_high), 0.0);
                    touched += u32::from(dx * dx + dy * dy + dz * dz <= radius * radius);
                }
            }
        }
        assert!(touched <= boxed, "refinement only removes clusters");
        touched
    }

    #[test]
    fn a_distant_street_lamp_covers_few_clusters() {
        let camera = engine_core::Camera {
            position: Vec3::new(250.0, 30.0, 250.0),
            target: Vec3::new(300.0, 0.0, 300.0),
            vertical_fov_radians: 1.0,
            near: 0.1,
            far: 2000.0,
        };
        let uniform = ClusterUniform::new(
            camera.view_projection(1.6),
            1600,
            1000,
            220.0,
            1,
            true,
            None,
            1,
        );
        // A lamp's 10-unit reach, 60 units away, touches a few dozen of the
        // 3456 clusters; the loose screen box around it holds far more.
        let clusters = covered(&uniform, Vec3::new(300.0, 4.0, 290.0), 10.0);
        assert!(clusters <= 60, "covers {clusters} clusters");
    }
}
