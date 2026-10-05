//! Placed objects (decision 057): what terrains' placement layers stand on
//! their level-0 tiles, drawn through the mesh pipelines as instances of
//! their own. Objects keep poses and colors in buffers laid out as slots'
//! are, so the mesh and depth shaders read them unchanged, rewritten only
//! when the terrains' tiles or assets change. Each frame a compute pass
//! culls every object against each view (the camera, then the shadow maps
//! drawn every frame) and picks its level of detail from the camera
//! (`scatter_cull.wgsl`), appending it to its draw group's list for that
//! view and level; each group and level then draws indirectly. Objects draw
//! in the first opaque phase, and cast into the near cascades and the
//! moving map, never the kept one: the tiles they stand on come and go with
//! the camera. They are never occlusion culled.

use bytemuck::{Pod, Zeroable};
use engine_core::glam::{Mat4, Vec3};

use crate::bindings::{RETAINED_POSES, SCATTER_CULL, SHADOW_POSES, TEMPORAL_POSES};
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

/// WGSL `Record`: an object's bounding sphere, draw group and scale.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct Record {
    pub center: [f32; 3],
    pub radius: f32,
    pub group: u32,
    pub scale: f32,
    pub pad: [u32; 2],
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

/// One draw group as the renderer lays it out: its objects (counted from
/// the group's first), each level's index count, first index and base
/// vertex, the levels the camera draws, and the coarser levels' errors.
pub(crate) struct GroupLayout {
    pub count: u32,
    pub levels: Vec<(u32, u32, i32)>,
    pub camera: u32,
    pub errors: [f32; LEVELS as usize - 1],
}

/// What [`ScatterPass::upload`] writes: every object's pose, color and
/// record, grouped, and each group's layout.
pub(crate) struct Objects {
    pub poses: Vec<PoseData>,
    pub colors: Vec<ColorData>,
    pub records: Vec<Record>,
    pub groups: Vec<GroupLayout>,
}

pub(crate) struct ScatterPass {
    layout: wgpu::BindGroupLayout,
    // The mesh pipelines' pose layouts, plain and temporal, and the depth
    // passes', which equal descriptors make interchangeable with theirs.
    pose_layouts: [wgpu::BindGroupLayout; 2],
    shadow_layout: wgpu::BindGroupLayout,
    pipeline: Option<wgpu::ComputePipeline>,
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

    /// Replaces every object and group with `objects`.
    pub(crate) fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, objects: &Objects) {
        let mut groups = Vec::with_capacity(objects.groups.len());
        self.template.clear();
        let mut ids = 0;
        for group in &objects.groups {
            groups.push(GroupData {
                ids,
                count: group.count,
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
            ids += group.count * LEVELS;
        }
        let mut replaced = false;
        for (buffer, label, bytes) in [
            (
                &mut self.poses,
                "engine/scatter-poses",
                bytemuck::cast_slice(&objects.poses),
            ),
            (
                &mut self.colors,
                "engine/scatter-colors",
                bytemuck::cast_slice(&objects.colors),
            ),
            (
                &mut self.records,
                "engine/scatter-records",
                bytemuck::cast_slice(&objects.records),
            ),
            (
                &mut self.groups,
                "engine/scatter-groups",
                bytemuck::cast_slice(&groups),
            ),
        ] {
            replaced |= grow(
                device,
                buffer,
                label,
                bytes.len() as u64,
                wgpu::BufferUsages::empty(),
            );
            if !bytes.is_empty() {
                queue.write_buffer(buffer, 0, bytes);
            }
        }
        if replaced {
            self.generation += 1;
        }
        self.objects = objects.records.len() as u32;
        self.group_count = objects.groups.len() as u32;
        self.ids_per_view = ids;
        self.resident = self.objects;
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
        if self.objects == 0 {
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
