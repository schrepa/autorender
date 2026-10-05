//! Ground cover (decision 056): the blades of every terrain's surface
//! materials' covers within a radius of the eye, grown each frame by a
//! compute pass (`cover_grow.wgsl`) into one buffer and drawn with one
//! indirect draw of strips in the opaque pass (`meshes/cover.wgsl`). Blades
//! past the buffer's capacity (the most the radius and densities can grow,
//! or the settings' budget) drop and count; cells grow in order outward
//! from the eye's corner of the grid, so the farthest tend to drop first.
//! Nothing here exists until a frame has cover to grow.

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use bytemuck::{Pod, Zeroable};
use engine_core::GroundCover;
use engine_core::glam::{Mat4, Vec3, Vec4};

use crate::bindings::{COVER_DRAW, COVER_GROW};
use crate::buffers::ReadbackRing;
use crate::cached::Cached;
use crate::render_graph::Targets;
use crate::terrain::TerrainPass;

/// Bytes of a blade (`CoverBlade`).
const BLADE_BYTES: u64 = 32;
/// Uniform offset alignment between terrains' growth.
const SLOT: u64 = 256;
/// Bytes of the growth's uniform (`Grow`).
pub(crate) const GROW_BYTES: u64 = size_of::<GrowUniform>() as u64;

/// WGSL `Grow` of `cover_grow.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct GrowUniform {
    origin: [f32; 4],
    eye: [f32; 4],
    grid: [i32; 4],
    counts: [u32; 4],
    scale: [f32; 4],
    planes: [[f32; 4]; 6],
}

/// WGSL `CoverFrame` of `meshes/cover.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct FrameUniform {
    wind: [f32; 4],
}

/// One terrain's growth this frame: its uniform's offset and its grid's
/// cells along each side.
struct Growth {
    offset: u32,
    cells: u32,
}

pub(crate) struct GroundCoverPass {
    grow_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
    // The growth's and the finish's pipelines, created with the first
    // frame that has cover.
    compute: Option<(wgpu::ComputePipeline, wgpu::ComputePipeline)>,
    pipelines: HashMap<(Targets, bool), wgpu::RenderPipeline>,
    blades: wgpu::Buffer,
    capacity: u32,
    blades_generation: u64,
    counts: wgpu::Buffer,
    args: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    frame: wgpu::Buffer,
    grow_group: Cached<(u64, u64, u64, u64), wgpu::BindGroup>,
    draw_group: Cached<u64, wgpu::BindGroup>,
    growths: Vec<Growth>,
    // Last frame's presentation time, which blades swayed at then.
    previous_time: Option<f32>,
    readbacks: ReadbackRing<u32>,
    // Blades grown and those past capacity, read back a few frames late.
    totals: Rc<Cell<Option<[u32; 2]>>>,
}

impl GroundCoverPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            grow_layout: crate::pipelines::layout(device, &COVER_GROW),
            draw_layout: crate::pipelines::layout(device, &COVER_DRAW),
            compute: None,
            pipelines: HashMap::new(),
            blades: blades_buffer(device, 1),
            capacity: 0,
            blades_generation: 0,
            counts: crate::buffers::storage(
                device,
                "engine/cover-counts",
                16,
                wgpu::BufferUsages::COPY_SRC,
            ),
            args: crate::buffers::storage(
                device,
                "engine/cover-draw",
                16,
                wgpu::BufferUsages::INDIRECT,
            ),
            uniforms: crate::buffers::uniform(device, "engine/cover-grow", SLOT),
            frame: crate::buffers::uniform(device, "engine/cover-frame", 16),
            grow_group: Cached::default(),
            draw_group: Cached::default(),
            growths: Vec::new(),
            previous_time: None,
            readbacks: ReadbackRing::new("engine/cover-readback"),
            totals: Rc::new(Cell::new(None)),
        }
    }

    /// Blades the last read-back frame drew, and those past capacity.
    pub(crate) fn totals(&self) -> [u32; 2] {
        self.totals.get().unwrap_or([0; 2])
    }

    /// Bytes of the blades' buffer.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.blades.size()
    }

    /// Plans this frame's growth for every terrain whose materials have
    /// cover, with `settings` on, seen through `culling` (world) from
    /// `eye` at presentation time `time`. Returns whether any grows.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &TerrainPass,
        terrains: &engine_core::RenderTerrains,
        (settings, wind, culling, eye, time): (&GroundCover, [f32; 2], Mat4, Vec3, f32),
    ) -> bool {
        self.growths.clear();
        let previous = self.previous_time.replace(time).unwrap_or(time);
        let grows = self.plan(
            device,
            queue,
            terrain,
            terrains,
            (settings, wind),
            (culling, eye, time, previous),
        );
        if !grows {
            // Nothing grows: none drawn, whatever the last read-back said.
            self.totals.set(Some([0; 2]));
        }
        grows
    }

    fn plan(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &TerrainPass,
        terrains: &engine_core::RenderTerrains,
        (settings, wind): (&GroundCover, [f32; 2]),
        (culling, eye, time, previous): (Mat4, Vec3, f32, f32),
    ) -> bool {
        if !settings.enabled || settings.density <= 0.0 || terrain.cover_sources().is_none() {
            return false;
        }
        let planes = planes(culling);
        let radius = settings.radius;
        let mut uniforms = Vec::new();
        let mut most = 0.0;
        for (index, render) in terrains.terrains.iter().enumerate() {
            let (directory, palette, densest) = terrain.cover_terrain(index);
            let density = densest * settings.density;
            if density <= 0.0 || render.patches.is_empty() {
                continue;
            }
            let spacing = render.pyramid.spacing(0);
            let candidates = engine_core::math::max((density * spacing * spacing).ceil(), 1.0);
            let local = eye - render.origin;
            let first = [
                ((local.x - radius) / spacing).floor() as i32,
                ((local.z - radius) / spacing).floor() as i32,
            ];
            let cells = (2.0 * radius / spacing).ceil() as u32 + 1;
            let uniform = GrowUniform {
                origin: render.origin.extend(spacing).to_array(),
                eye: eye.extend(radius).to_array(),
                grid: [first[0], first[1], cells as i32, candidates as i32],
                counts: [directory, render.pyramid.levels(), palette, 0],
                scale: [settings.density, spacing * spacing / candidates, 0.0, 0.0],
                planes,
            };
            self.growths.push(Growth {
                offset: (uniforms.len() as u64 * SLOT) as u32,
                cells,
            });
            uniforms.push(uniform);
            most += disc_area(radius) * density;
        }
        if self.growths.is_empty() {
            return false;
        }
        // Room for the most blades the discs can grow, or the budget.
        let capacity = settings
            .max_blades
            .map_or(most.ceil() as u32, |budget| budget.min(most.ceil() as u32))
            .max(1);
        if u64::from(capacity) * BLADE_BYTES > self.blades.size() {
            self.blades = blades_buffer(device, capacity.next_power_of_two());
            self.blades_generation += 1;
        }
        self.capacity = capacity;
        let mut bytes = vec![0_u8; uniforms.len() * SLOT as usize];
        for (index, mut uniform) in uniforms.into_iter().enumerate() {
            uniform.counts[3] = capacity;
            let at = index * SLOT as usize;
            bytes[at..at + size_of::<GrowUniform>()].copy_from_slice(bytemuck::bytes_of(&uniform));
        }
        if bytes.len() as u64 > self.uniforms.size() {
            self.uniforms = crate::buffers::uniform(
                device,
                "engine/cover-grow",
                (bytes.len() as u64).next_power_of_two(),
            );
            self.blades_generation += 1;
        }
        queue.write_buffer(&self.uniforms, 0, &bytes);
        let frame = FrameUniform {
            wind: [wind[0], wind[1], time, previous],
        };
        queue.write_buffer(&self.frame, 0, bytemuck::bytes_of(&frame));
        self.compute
            .get_or_insert_with(|| compute(device, &self.grow_layout));
        let sources = terrain.cover_sources().expect("checked above");
        let key = (
            sources.generation.0,
            sources.generation.1,
            self.blades_generation,
            0,
        );
        if self.grow_group.stale(&key) {
            let group = crate::pipelines::group(
                device,
                "engine/cover-grow",
                &COVER_GROW,
                &self.grow_layout,
                &[
                    ("heights", crate::pipelines::view(sources.heights)),
                    ("control", crate::pipelines::view(sources.control)),
                    crate::buffers::entry("directory", sources.directory),
                    crate::buffers::entry("palette", sources.palette),
                    crate::buffers::entry("covers", sources.covers),
                    (
                        "grow",
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniforms,
                            offset: 0,
                            size: wgpu::BufferSize::new(GROW_BYTES),
                        }),
                    ),
                    crate::buffers::entry("blades", &self.blades),
                    crate::buffers::entry("counts", &self.counts),
                    crate::buffers::entry("args", &self.args),
                ],
            );
            self.grow_group.set(key, group);
        }
        if self.draw_group.stale(&self.blades_generation) {
            let group = crate::pipelines::group(
                device,
                "engine/cover",
                &COVER_DRAW,
                &self.draw_layout,
                &[
                    crate::buffers::entry("cover_blades", &self.blades),
                    crate::buffers::entry("cover_frame", &self.frame),
                ],
            );
            self.draw_group.set(self.blades_generation, group);
        }
        true
    }

    /// Records this frame's growth and the indirect draw's arguments, and
    /// a copy of the counts to read back, timed by `profiler`
    /// (`ground_cover`).
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some((grow, finish)), Some(group)) = (&self.compute, self.grow_group.get()) else {
            return;
        };
        if self.growths.is_empty() {
            return;
        }
        encoder.clear_buffer(&self.counts, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/cover-grow"),
                timestamp_writes: profiler.span("ground_cover").map(|span| span.compute()),
            });
            pass.set_pipeline(grow);
            for growth in &self.growths {
                pass.set_bind_group(0, group, &[growth.offset]);
                pass.dispatch_workgroups(growth.cells, growth.cells, 1);
            }
            pass.set_pipeline(finish);
            pass.set_bind_group(0, group, &[self.growths[0].offset]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.readbacks
            .record(device, encoder, &self.counts, 16, self.capacity);
    }

    /// Requests the map of this frame's counts once it is submitted.
    pub(crate) fn after_submit(&mut self) {
        let totals = Rc::clone(&self.totals);
        self.readbacks.after_submit(move |words, capacity| {
            let grown = words.get(0);
            totals.set(Some([grown.min(capacity), grown.saturating_sub(capacity)]));
        });
    }

    /// Creates the pipeline drawing blades into `targets` (`lit` or not)
    /// unless it exists.
    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        (camera, empty, lights): (
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            Option<&wgpu::BindGroupLayout>,
        ),
        (targets, lit): (Targets, bool),
        canvas: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.pipelines.contains_key(&(targets, lit)) {
            return;
        }
        let mut layouts = vec![Some(camera), Some(&self.draw_layout)];
        if lit {
            let Some(lights) = lights else {
                return;
            };
            layouts.push(Some(empty));
            layouts.push(Some(lights));
        }
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/cover-pipeline-layout"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        let (vertex, fragment) = crate::entries::cover(targets, lit);
        let color_targets = targets.color_targets(canvas);
        let pipeline = crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/cover",
                layout: &layout,
                module: shader,
                vertex,
                buffers: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    cull_mode: None,
                    ..Default::default()
                },
                depth: Some(wgpu::DepthStencilState {
                    format: depth,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(crate::pipelines::Fragment {
                    entry: fragment,
                    constants: &[],
                    targets: &color_targets,
                }),
            },
        );
        self.pipelines.insert((targets, lit), pipeline);
    }

    /// Draws this frame's blades into `targets` (`lit` or not), the camera
    /// and lights already bound; returns the draws.
    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, targets: Targets, lit: bool) -> u32 {
        let (Some(pipeline), Some(group)) =
            (self.pipelines.get(&(targets, lit)), self.draw_group.get())
        else {
            return 0;
        };
        if self.growths.is_empty() {
            return 0;
        }
        pass.set_pipeline(pipeline);
        pass.set_bind_group(1, group, &[]);
        pass.draw_indirect(&self.args, 0);
        1
    }
}

/// The square meters of densest cover a disc of `radius` grows: its area
/// weighted by `cover_grow.wgsl`'s `cover_share`, summed ring by ring.
fn disc_area(radius: f32) -> f32 {
    const RINGS: u32 = 64;
    let width = radius / RINGS as f32;
    (0..RINGS)
        .map(|ring| {
            let away = (ring as f32 + 0.5) * width;
            let share = engine_core::math::min(1.0, 0.2 * radius / away)
                * (1.0 - smoothstep(0.5 * radius, radius, away));
            std::f32::consts::TAU * away * width * share
        })
        .sum()
}

fn smoothstep(from: f32, to: f32, value: f32) -> f32 {
    let t = engine_core::math::clamp((value - from) / (to - from), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn blades_buffer(device: &wgpu::Device, blades: u32) -> wgpu::Buffer {
    crate::buffers::storage(
        device,
        "engine/cover-blades",
        u64::from(blades) * BLADE_BYTES,
        wgpu::BufferUsages::empty(),
    )
}

/// The view's frustum planes from its view-projection (world), each
/// normalized with its inside positive.
pub(crate) fn planes(view_projection: Mat4) -> [[f32; 4]; 6] {
    let rows = [
        view_projection.row(0),
        view_projection.row(1),
        view_projection.row(2),
        view_projection.row(3),
    ];
    // Reversed depth: the near plane is z = w, the far z = 0.
    [
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
        rows[2],
        rows[3] - rows[2],
    ]
    .map(|plane: Vec4| {
        (plane / engine_core::math::max(plane.truncate().length(), 1.0e-12)).to_array()
    })
}

fn compute(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/cover-grow.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::COVER_GROW_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine/cover-grow-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    (
        crate::pipelines::compute_entry(device, &pipeline_layout, &module, "grow_cover"),
        crate::pipelines::compute_entry(device, &pipeline_layout, &module, "finish_cover"),
    )
}
