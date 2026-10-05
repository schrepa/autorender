//! Occupancy (decision 059): a top-down map around the camera of what
//! placed objects cover, drawn from the scatter pass's slots
//! (`occupancy.wgsl`) and read by the shared shading (`scene.wgsl`'s
//! `occupancy_visibility`), which darkens the sky light of what lies under
//! or within a cover. The map is drawn again only when it no longer holds
//! the camera's surroundings or the objects changed.

use bytemuck::{Pod, Zeroable};
use engine_core::Occupancy;
use engine_core::glam::Vec3;

use crate::bindings::OCCUPANCY;
use crate::cached::Cached;

/// The coverage target's format, and the heights target's.
const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const HEIGHTS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// What heights hold where nothing covers: no top, no bottom.
const NOTHING: f64 = -1.0e4;

/// WGSL `Splat` of `occupancy.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct SplatUniform {
    area: [f32; 4],
    counts: [u32; 4],
}

/// WGSL `Cover`: a draw group's footprint radius, the bottom and top of
/// what it covers above its origin (mesh units), and its density.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct Cover {
    pub footprint: f32,
    pub bottom: f32,
    pub top: f32,
    pub density: f32,
}

impl Cover {
    /// The cover of a mesh with bounds `min` to `max`: its footprint a
    /// little inside its bounds' mean horizontal half extent. Tall meshes,
    /// such as trees, cover from a fifth of their height up; the rest from
    /// their bottom.
    pub(crate) fn of(min: Vec3, max: Vec3) -> Self {
        let half = (max - min) * 0.5;
        let footprint = (half.x + half.z) * 0.5 * 0.85;
        let height = max.y - min.y;
        let bottom = if height > 2.5 * footprint {
            min.y + 0.2 * height
        } else {
            min.y
        };
        Self {
            footprint,
            bottom,
            top: max.y,
            density: 0.9,
        }
    }
}

/// Where the map lies: its center (x and z), meters a side, reference
/// height and texels a side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Area {
    pub center: [f32; 2],
    pub size: f32,
    pub reference: f32,
    pub resolution: u32,
}

pub(crate) struct OccupancyPass {
    layout: wgpu::BindGroupLayout,
    pipeline: Option<wgpu::RenderPipeline>,
    uniform: wgpu::Buffer,
    covers: wgpu::Buffer,
    covers_written: Vec<Cover>,
    // The coverage and heights targets, their views, and a generation
    // advancing when they are replaced.
    targets: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    generation: u64,
    group: Cached<(u64, u64, u64), wgpu::BindGroup>,
    // The area last drawn and what it was drawn from: the slots' version.
    drawn: Option<(Area, u64)>,
    // This frame's draw, if it redraws: the area and the slots to draw.
    pending: Option<(Area, u32)>,
    /// Times the map was drawn, and objects in its last draw's slots.
    pub(crate) redraws: u32,
    pub(crate) slots: u32,
}

impl OccupancyPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        // Zeroed targets cover nothing until the first draw.
        let (targets, views) = targets(device, 1);
        Self {
            layout: crate::pipelines::layout(device, &OCCUPANCY),
            pipeline: None,
            uniform: crate::buffers::uniform(
                device,
                "engine/occupancy-splat",
                size_of::<SplatUniform>() as u64,
            ),
            covers: crate::buffers::storage(
                device,
                "engine/occupancy-covers",
                size_of::<Cover>() as u64,
                wgpu::BufferUsages::empty(),
            ),
            covers_written: Vec::new(),
            targets,
            views,
            generation: 0,
            group: Cached::default(),
            drawn: None,
            pending: None,
            redraws: 0,
            slots: 0,
        }
    }

    /// The coverage and heights views the main passes read, and the
    /// generation that changes when they are replaced.
    pub(crate) fn views(&self) -> (&wgpu::TextureView, &wgpu::TextureView, u64) {
        (&self.views[0], &self.views[1], self.generation)
    }

    /// Bytes of the map's targets.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets
            .iter()
            .map(|target| u64::from(target.width()) * u64::from(target.height()))
            .sum::<u64>()
            * 6
            / 2
    }

    /// Plans this frame: with `settings` on and placed objects whose
    /// slots' version is `version` (`covers` per draw group, `slots` in
    /// all), the map's area around `eye`, drawn again when it moved too far
    /// for the last draw to hold the camera's surroundings, or the slots or
    /// covers changed. Returns the area shading reads, with the strength
    /// covers darken by; `None` without a map.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &Occupancy,
        eye: Vec3,
        (covers, slots, version): (&[Cover], u32, u64),
    ) -> Option<(Area, f32)> {
        self.pending = None;
        if !settings.enabled || settings.strength <= 0.0 || slots == 0 || covers.is_empty() {
            self.drawn = None;
            return None;
        }
        let size = 2.0 * settings.radius;
        let resolution = settings.resolution;
        if self.targets[0].width() != resolution {
            let (targets, views) = targets(device, resolution);
            self.targets = targets;
            self.views = views;
            self.generation += 1;
            self.drawn = None;
        }
        if covers != self.covers_written.as_slice() {
            let bytes: &[u8] = bytemuck::cast_slice(covers);
            if bytes.len() as u64 > self.covers.size() {
                self.covers = crate::buffers::storage(
                    device,
                    "engine/occupancy-covers",
                    (bytes.len() as u64).next_power_of_two(),
                    wgpu::BufferUsages::empty(),
                );
                self.generation += 1;
            }
            queue.write_buffer(&self.covers, 0, bytes);
            self.covers_written = covers.to_vec();
            self.drawn = None;
        }
        // The map follows the eye in steps of an eighth of its side, and
        // is drawn again once the eye strays a quarter of it from its
        // center.
        let step = size / 8.0;
        let snapped = [(eye.x / step).round() * step, (eye.z / step).round() * step];
        let current = self.drawn.filter(|(area, drawn)| {
            *drawn == version
                && area.size == size
                && area.resolution == resolution
                && (eye.x - area.center[0]).abs() <= size / 4.0
                && (eye.z - area.center[1]).abs() <= size / 4.0
        });
        let area = match current {
            Some((area, _)) => area,
            None => {
                let area = Area {
                    center: snapped,
                    size,
                    reference: eye.y,
                    resolution,
                };
                let uniform = SplatUniform {
                    area: [area.center[0], area.center[1], size, area.reference],
                    counts: [slots, covers.len() as u32, 0, 0],
                };
                queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
                self.pending = Some((area, slots));
                self.drawn = Some((area, version));
                area
            }
        };
        Some((area, settings.strength))
    }

    /// Whether this frame draws the map.
    pub(crate) fn redrawing(&self) -> bool {
        self.pending.is_some()
    }

    /// Draws the map from the placed objects' `records` and `poses`, when
    /// this frame redraws it, into `encoder`; timed by `profiler`
    /// (`occupancy`).
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        (records, poses, generation): (&wgpu::Buffer, &wgpu::Buffer, u64),
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let Some((_, slots)) = self.pending.take() else {
            return;
        };
        let pipeline = self
            .pipeline
            .get_or_insert_with(|| pipeline(device, &self.layout));
        let key = (generation, self.generation, 0);
        if self.group.stale(&key) {
            let group = crate::pipelines::group(
                device,
                "engine/occupancy",
                &OCCUPANCY,
                &self.layout,
                &[
                    crate::buffers::entry("splat", &self.uniform),
                    crate::buffers::entry("records", records),
                    crate::buffers::entry("poses", poses),
                    crate::buffers::entry("covers", &self.covers),
                ],
            );
            self.group.set(key, group);
        }
        let Some(group) = self.group.get() else {
            return;
        };
        let attachment = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let mut heights = attachment(&self.views[1]);
        if let Some(heights) = heights.as_mut() {
            heights.ops.load = wgpu::LoadOp::Clear(wgpu::Color {
                r: NOTHING,
                g: NOTHING,
                b: 0.0,
                a: 0.0,
            });
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("engine/occupancy"),
            color_attachments: &[attachment(&self.views[0]), heights],
            depth_stencil_attachment: None,
            timestamp_writes: profiler.span("occupancy").map(|span| span.render()),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..6, 0..slots);
        self.redraws += 1;
        self.slots = slots;
    }
}

/// The coverage and heights targets, `resolution` texels a side, and their
/// views.
fn targets(device: &wgpu::Device, resolution: u32) -> ([wgpu::Texture; 2], [wgpu::TextureView; 2]) {
    let target = |label, format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let targets = [
        target("engine/occupancy-coverage", COVERAGE_FORMAT),
        target("engine/occupancy-heights", HEIGHTS_FORMAT),
    ];
    let views = [
        targets[0].create_view(&wgpu::TextureViewDescriptor::default()),
        targets[1].create_view(&wgpu::TextureViewDescriptor::default()),
    ];
    (targets, views)
}

fn pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/occupancy.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::OCCUPANCY_SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("engine/occupancy-layout"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    // Each target keeps its maximum.
    let maximum = Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        },
    });
    crate::pipelines::render(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some("engine/occupancy"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_splat"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_splat"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: COVERAGE_FORMAT,
                        blend: maximum,
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: HEIGHTS_FORMAT,
                        blend: maximum,
                        write_mask: wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        },
    )
}
