//! Placed objects (decision 057): what terrains' placement layers stand on
//! their tiles, drawn through the mesh pipelines as instances of their own.
//! Objects hold slots of a pool whose poses and colors are laid out as
//! slots' are, so the mesh and depth shaders read them unchanged; each
//! tile's objects upload when it arrives. Each frame a compute pass first
//! sets every object on its terrain's finest resident heights
//! (`scatter_snap.wgsl`), then culls it against each view (the camera,
//! then the shadow maps drawn every frame) and picks its level of detail
//! from the camera (`scatter_cull.wgsl`), appending it to its draw
//! group's list for that view and level; each group and level then draws
//! indirectly. Objects draw in the first opaque phase, and cast into the
//! near cascades and the moving map, never the kept one: the tiles they
//! stand on come and go with the camera. They are never occlusion culled.

use bytemuck::{Pod, Zeroable};
use engine_core::glam::{Mat4, Vec3};

use crate::bindings::{RETAINED_POSES, SCATTER_CULL, SCATTER_SNAP, SHADOW_POSES, TEMPORAL_POSES};
use crate::cached::Cached;
use crate::gpu_data::{ColorData, PoseData};

/// Levels each draw group lists: its mesh and four coarser levels.
pub(crate) const LEVELS: u32 = 5;

/// WGSL `Frame` of `scatter_cull.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct FrameUniform {
    eye: [f32; 4],
    counts: [u32; 4],
    strides: [u32; 4],
}

/// WGSL `View`: a view's frustum planes and the radius objects need in it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct ViewData {
    planes: [[f32; 4]; 6],
    limits: [f32; 4],
}

/// The draw group of slots no object holds, which culling skips.
pub(crate) const DEAD: u32 = u32::MAX;

/// WGSL `Record`: an object's bounding sphere, draw group and scale, and
/// the terrain it stands on (its index this frame; none: `u32::MAX`) and
/// how far its base sinks below the ground.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct Record {
    pub center: [f32; 3],
    pub radius: f32,
    pub group: u32,
    pub scale: f32,
    pub terrain: u32,
    pub sink: f32,
}

/// WGSL `TerrainInfo` of `scatter_snap.wgsl`: a terrain's origin and
/// level-0 spacing, and where its levels start in the directory and how
/// many there are.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct TerrainInfo {
    pub origin: [f32; 4],
    pub levels: [u32; 4],
}

/// What setting objects on the terrain reads: the terrain pass's resident
/// heights and directory, their generations, and each terrain's place.
pub(crate) struct SnapSources<'a> {
    pub heights: &'a wgpu::TextureView,
    pub directory: &'a wgpu::Buffer,
    pub generation: (u64, u64),
    pub terrains: &'a [TerrainInfo],
}

/// WGSL `Group`: where its lists start, its objects, its levels, those
/// the camera draws, and the coarser ones' errors.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct GroupData {
    ids: u32,
    count: u32,
    levels: u32,
    camera: u32,
    errors: [f32; 4],
}

/// An indexed indirect draw's arguments.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

/// One draw group as the renderer lays it out: each level's index count,
/// first index and base vertex, the levels the camera draws, and the
/// coarser levels' errors.
pub(crate) struct GroupLayout {
    pub levels: Vec<(u32, u32, i32)>,
    pub camera: u32,
    pub errors: [f32; LEVELS as usize - 1],
}

pub(crate) struct ScatterPass {
    layout: wgpu::BindGroupLayout,
    // The mesh pipelines' pose layouts, plain and temporal, and the depth
    // passes', which equal descriptors make interchangeable with theirs.
    pose_layouts: [wgpu::BindGroupLayout; 2],
    shadow_layout: wgpu::BindGroupLayout,
    pipeline: Option<wgpu::ComputePipeline>,
    // Setting objects on the terrain: its layout and pipeline, its uniform,
    // the terrains' places, its group (by the buffers' and terrain
    // sources' generations), and whether this frame runs it.
    snap_layout: wgpu::BindGroupLayout,
    snap_pipeline: Option<wgpu::ComputePipeline>,
    snap: wgpu::Buffer,
    terrains: wgpu::Buffer,
    snap_group: Cached<(u64, u64, (u64, u64)), wgpu::BindGroup>,
    snapping: bool,
    frame: wgpu::Buffer,
    views: wgpu::Buffer,
    records: wgpu::Buffer,
    groups: wgpu::Buffer,
    ids: wgpu::Buffer,
    draws: wgpu::Buffer,
    poses: wgpu::Buffer,
    colors: wgpu::Buffer,
    palette: wgpu::Buffer,
    // Advances when buffers are replaced.
    generation: u64,
    cull_group: Cached<u64, wgpu::BindGroup>,
    // The pose groups of plain and temporal mesh pipelines, and of depth
    // passes.
    pose_groups: Cached<u64, [wgpu::BindGroup; 2]>,
    shadow_group: Cached<u64, wgpu::BindGroup>,
    // Each draw's arguments with no instances, and every view's, rewritten
    // every frame.
    template: Vec<DrawArgs>,
    args: Vec<DrawArgs>,
    objects: u32,
    group_count: u32,
    // The IDs each view's lists take.
    ids_per_view: u32,
    // The shadow map each view after the camera's draws into.
    maps: Vec<usize>,
    // Advances when the terrains' buffer is replaced.
    terrains_generation: u64,
    /// Objects resident.
    pub(crate) resident: u32,
}

impl ScatterPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let storage = |label, bytes| {
            crate::buffers::storage(device, label, bytes, wgpu::BufferUsages::empty())
        };
        Self {
            layout: crate::pipelines::layout(device, &SCATTER_CULL),
            pose_layouts: [
                crate::pipelines::layout(device, &RETAINED_POSES),
                crate::pipelines::layout(device, &TEMPORAL_POSES),
            ],
            shadow_layout: crate::pipelines::layout(device, &SHADOW_POSES),
            pipeline: None,
            snap_layout: crate::pipelines::layout(device, &SCATTER_SNAP),
            snap_pipeline: None,
            snap: crate::buffers::uniform(device, "engine/scatter-snap", 16),
            terrains: storage("engine/scatter-terrains", size_of::<TerrainInfo>() as u64),
            snap_group: Cached::default(),
            snapping: false,
            terrains_generation: 0,
            frame: crate::buffers::uniform(
                device,
                "engine/scatter-frame",
                size_of::<FrameUniform>() as u64,
            ),
            views: storage("engine/scatter-views", size_of::<ViewData>() as u64),
            records: storage("engine/scatter-records", 32),
            groups: storage("engine/scatter-groups", 32),
            ids: crate::buffers::storage(
                device,
                "engine/scatter-ids",
                16,
                wgpu::BufferUsages::VERTEX,
            ),
            draws: crate::buffers::storage(
                device,
                "engine/scatter-draws",
                20,
                wgpu::BufferUsages::INDIRECT,
            ),
            poses: storage("engine/scatter-poses", size_of::<PoseData>() as u64),
            colors: storage("engine/scatter-colors", size_of::<ColorData>() as u64),
            palette: storage("engine/scatter-palette", size_of::<PoseData>() as u64),
            generation: 0,
            cull_group: Cached::default(),
            pose_groups: Cached::default(),
            shadow_group: Cached::default(),
            template: Vec::new(),
            args: Vec::new(),
            objects: 0,
            group_count: 0,
            ids_per_view: 0,
            maps: Vec::new(),
            resident: 0,
        }
    }

    /// Bytes of the objects' buffers.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        [
            &self.views,
            &self.records,
            &self.groups,
            &self.ids,
            &self.draws,
            &self.poses,
            &self.colors,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }

    /// Lays out the draw groups `groups` holding `counts` live objects:
    /// each group's lists and the draws' template.
    pub(crate) fn set_groups(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        groups: &[GroupLayout],
        counts: &[u32],
    ) {
        let mut data = Vec::with_capacity(groups.len());
        self.template.clear();
        let mut ids = 0;
        for (group, &count) in groups.iter().zip(counts) {
            data.push(GroupData {
                ids,
                count,
                levels: group.levels.len() as u32,
                camera: group.camera,
                errors: group.errors,
            });
            for level in 0..LEVELS as usize {
                let (index_count, first_index, base_vertex) =
                    group.levels.get(level).copied().unwrap_or((0, 0, 0));
                self.template.push(DrawArgs {
                    index_count,
                    instance_count: 0,
                    first_index,
                    base_vertex,
                    first_instance: 0,
                });
            }
            ids += count * LEVELS;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        if grow(
            device,
            &mut self.groups,
            "engine/scatter-groups",
            bytes.len() as u64,
            wgpu::BufferUsages::empty(),
        ) {
            self.generation += 1;
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.groups, 0, bytes);
        }
        self.group_count = groups.len() as u32;
        self.ids_per_view = ids;
        self.resident = counts.iter().sum();
    }

    /// Writes slots `dirty` of `poses`, `colors` and `records`, every
    /// slot when the buffers grow to hold them all.
    pub(crate) fn write_slots(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (poses, colors, records): (&[PoseData], &[ColorData], &[Record]),
        dirty: Option<std::ops::Range<u32>>,
    ) {
        let mut grown = false;
        for (buffer, label, bytes) in [
            (&mut self.poses, "engine/scatter-poses", size_of_val(poses)),
            (
                &mut self.colors,
                "engine/scatter-colors",
                size_of_val(colors),
            ),
            (
                &mut self.records,
                "engine/scatter-records",
                size_of_val(records),
            ),
        ] {
            grown |= grow(
                device,
                buffer,
                label,
                bytes as u64,
                wgpu::BufferUsages::empty(),
            );
        }
        if grown {
            self.generation += 1;
        }
        let range = if grown {
            Some(0..records.len())
        } else {
            dirty.map(|range| range.start as usize..range.end as usize)
        };
        if let Some(range) = range.filter(|range| !range.is_empty()) {
            let write = |buffer: &wgpu::Buffer, stride: usize, bytes: &[u8]| {
                queue.write_buffer(buffer, (range.start * stride) as u64, bytes);
            };
            write(
                &self.poses,
                size_of::<PoseData>(),
                bytemuck::cast_slice(&poses[range.clone()]),
            );
            write(
                &self.colors,
                size_of::<ColorData>(),
                bytemuck::cast_slice(&colors[range.clone()]),
            );
            write(
                &self.records,
                size_of::<Record>(),
                bytemuck::cast_slice(&records[range.clone()]),
            );
        }
        self.objects = records.len() as u32;
    }

    /// Plans this frame's culling: the camera's view, from its
    /// `view_projection` (world) and eye, and those of the shadow `maps`
    /// drawn every frame (their world-to-clip matrices, `resolution` texels
    /// square), where objects smaller than half a texel cast nothing. Levels
    /// are picked within `pixel_error` pixels of a viewport `height` pixels
    /// tall (no error: finest levels only); maps draw a coarser level where
    /// a texel hides its error. Returns whether any object draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (view_projection, eye): (Mat4, Vec3),
        (pixel_error, height): (Option<f32>, f32),
        (maps, resolution): (&[(usize, Mat4)], u32),
    ) -> bool {
        self.maps.clear();
        if self.resident == 0 {
            return false;
        }
        let y_row = view_projection.row(1).truncate().length();
        let pixels = pixel_error.map_or(0.0, |error| y_row * height * 0.5 / error);
        let mut views = vec![ViewData {
            planes: crate::ground_cover::planes(view_projection),
            limits: [0.0; 4],
        }];
        for &(map, clip) in maps {
            // An orthographic map: its sides alone bound what casts into it.
            let mut planes = crate::ground_cover::planes(clip);
            planes[4] = [0.0, 0.0, 0.0, 1.0];
            planes[5] = [0.0, 0.0, 0.0, 1.0];
            let texels = clip.row(0).truncate().length() * resolution as f32;
            let half_texel = 1.0 / engine_core::math::max(texels, 1.0e-12);
            views.push(ViewData {
                planes,
                limits: [half_texel, 2.0 * half_texel, 0.0, 0.0],
            });
            self.maps.push(map);
        }
        let view_count = views.len() as u32;
        let draws_per_view = self.group_count * LEVELS;
        let mut replaced = false;
        replaced |= grow(
            device,
            &mut self.views,
            "engine/scatter-views",
            (views.len() * size_of::<ViewData>()) as u64,
            wgpu::BufferUsages::empty(),
        );
        replaced |= grow(
            device,
            &mut self.ids,
            "engine/scatter-ids",
            u64::from(self.ids_per_view) * u64::from(view_count) * 4,
            wgpu::BufferUsages::VERTEX,
        );
        replaced |= grow(
            device,
            &mut self.draws,
            "engine/scatter-draws",
            u64::from(draws_per_view) * u64::from(view_count) * size_of::<DrawArgs>() as u64,
            wgpu::BufferUsages::INDIRECT,
        );
        if replaced {
            self.generation += 1;
        }
        let frame = FrameUniform {
            eye: eye.extend(pixels).to_array(),
            counts: [self.objects, self.group_count, LEVELS, view_count],
            strides: [self.ids_per_view, draws_per_view, 0, 0],
        };
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&frame));
        queue.write_buffer(&self.views, 0, bytemuck::cast_slice(&views));
        self.args.clear();
        for _ in 0..view_count {
            self.args.extend_from_slice(&self.template);
        }
        queue.write_buffer(&self.draws, 0, bytemuck::cast_slice(&self.args));
        self.pipeline
            .get_or_insert_with(|| pipeline(device, &self.layout));
        if self.cull_group.stale(&self.generation) {
            let group = crate::pipelines::group(
                device,
                "engine/scatter-cull",
                &SCATTER_CULL,
                &self.layout,
                &[
                    crate::buffers::entry("frame", &self.frame),
                    crate::buffers::entry("records", &self.records),
                    crate::buffers::entry("groups", &self.groups),
                    crate::buffers::entry("ids", &self.ids),
                    crate::buffers::entry("draws", &self.draws),
                    crate::buffers::entry("views", &self.views),
                ],
            );
            self.cull_group.set(self.generation, group);
        }
        self.build_pose_groups(device);
        true
    }

    /// Builds the pose groups of plain and temporal mesh pipelines and of
    /// depth passes over the objects' poses (their previous poses being the
    /// same: objects stand still) when buffers changed.
    fn build_pose_groups(&mut self, device: &wgpu::Device) {
        let [retained, temporal] = &self.pose_layouts;
        self.pose_groups.get_or_insert_with(self.generation, || {
            [
                crate::pipelines::group(
                    device,
                    "engine/scatter-poses",
                    &RETAINED_POSES,
                    retained,
                    &[
                        crate::buffers::entry("poses", &self.poses),
                        crate::buffers::entry("colors", &self.colors),
                        crate::buffers::entry("palette", &self.palette),
                    ],
                ),
                crate::pipelines::group(
                    device,
                    "engine/scatter-temporal-poses",
                    &TEMPORAL_POSES,
                    temporal,
                    &[
                        crate::buffers::entry("poses", &self.poses),
                        crate::buffers::entry("colors", &self.colors),
                        crate::buffers::entry("previous_poses", &self.poses),
                        crate::buffers::entry("palette", &self.palette),
                        crate::buffers::entry("previous_palette", &self.palette),
                    ],
                ),
            ]
        });
        self.shadow_group.get_or_insert_with(self.generation, || {
            crate::pipelines::group(
                device,
                "engine/scatter-shadow-poses",
                &SHADOW_POSES,
                &self.shadow_layout,
                &[
                    crate::buffers::entry("slot_poses", &self.poses),
                    crate::buffers::entry("slot_colors", &self.colors),
                    crate::buffers::entry("slot_palette", &self.palette),
                ],
            )
        });
    }

    /// Plans setting objects on the terrain this frame from `sources`; none
    /// while the terrain pass holds no heights.
    pub(crate) fn prepare_snap(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sources: Option<SnapSources<'_>>,
    ) {
        self.snapping = false;
        let Some(sources) = sources.filter(|sources| !sources.terrains.is_empty()) else {
            return;
        };
        if self.resident == 0 {
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(sources.terrains);
        if grow(
            device,
            &mut self.terrains,
            "engine/scatter-terrains",
            bytes.len() as u64,
            wgpu::BufferUsages::empty(),
        ) {
            self.terrains_generation += 1;
        }
        queue.write_buffer(&self.terrains, 0, bytes);
        let counts = [self.objects, sources.terrains.len() as u32, 0, 0];
        queue.write_buffer(&self.snap, 0, bytemuck::cast_slice(&counts));
        self.snap_pipeline
            .get_or_insert_with(|| snap_pipeline(device, &self.snap_layout));
        let key = (
            self.generation,
            self.terrains_generation,
            sources.generation,
        );
        if self.snap_group.stale(&key) {
            let group = crate::pipelines::group(
                device,
                "engine/scatter-snap",
                &SCATTER_SNAP,
                &self.snap_layout,
                &[
                    crate::buffers::entry("snap", &self.snap),
                    crate::buffers::entry("records", &self.records),
                    crate::buffers::entry("poses", &self.poses),
                    crate::buffers::entry("terrains", &self.terrains),
                    crate::buffers::entry("directory", sources.directory),
                    ("heights", crate::pipelines::view(sources.heights)),
                ],
            );
            self.snap_group.set(key, group);
        }
        self.snapping = true;
    }

    /// The pose groups of plain and temporal mesh pipelines.
    pub(crate) fn pose_groups(&self) -> Option<&[wgpu::BindGroup; 2]> {
        self.pose_groups.get()
    }

    /// The pose group of depth passes.
    pub(crate) fn shadow_group(&self) -> Option<&wgpu::BindGroup> {
        self.shadow_group.get()
    }

    /// The view shadow map `map` reads its lists from this frame, if it
    /// draws objects.
    pub(crate) fn map_view(&self, map: usize) -> Option<usize> {
        self.maps
            .iter()
            .position(|&drawn| drawn == map)
            .map(|index| index + 1)
    }

    /// Records this frame's culling, timed by `profiler` (`scatter`).
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some(pipeline), Some(group)) = (&self.pipeline, self.cull_group.get()) else {
            return;
        };
        if self.objects == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/scatter-cull"),
            timestamp_writes: profiler.span("scatter").map(|span| span.compute()),
        });
        if let (true, Some(snap), Some(group)) =
            (self.snapping, &self.snap_pipeline, self.snap_group.get())
        {
            pass.set_pipeline(snap);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(self.objects.div_ceil(64), 1, 1);
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(self.objects.div_ceil(64), 1 + self.maps.len() as u32, 1);
    }

    /// Group `group`'s list of level `level` in `view` (0 the camera's) and
    /// its draw's arguments: the visible-ID slice to bind and the indirect
    /// buffer and offset; its lists start at `ids_base`.
    pub(crate) fn draw_source(
        &self,
        view: usize,
        (group, ids_base): (usize, u32),
        level: usize,
    ) -> (wgpu::BufferSlice<'_>, &wgpu::Buffer, u64) {
        let start = (view as u64 * u64::from(self.ids_per_view) + u64::from(ids_base)) * 4;
        let draw = (view * self.group_count as usize + group) * LEVELS as usize + level;
        (
            self.ids.slice(start..),
            &self.draws,
            (draw * size_of::<DrawArgs>()) as u64,
        )
    }
}

/// Replaces `buffer` with one of the next power of two at least `bytes`
/// long when it is shorter; returns whether it did.
fn grow(
    device: &wgpu::Device,
    buffer: &mut wgpu::Buffer,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> bool {
    if bytes <= buffer.size() {
        return false;
    }
    *buffer = crate::buffers::storage(device, label, bytes.next_power_of_two(), extra);
    true
}

fn snap_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/scatter-snap.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::SCATTER_SNAP_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine/scatter-snap-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    crate::pipelines::compute_entry(device, &pipeline_layout, &module, "snap_scatter")
}

fn pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/scatter-cull.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::SCATTER_CULL_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine/scatter-cull-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    crate::pipelines::compute_entry(device, &pipeline_layout, &module, "cull_scatter")
}
