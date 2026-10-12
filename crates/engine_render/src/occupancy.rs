//! Occupancy: a top-down map around the camera of what
//! placed objects cover, drawn from the scatter pass's slots
//! (`occupancy.wgsl`) and read by the shared shading (`scene.wgsl`'s
//! `occupancy_visibility`), which darkens the sky light of what lies under
//! or within a cover, less toward the end of its reach around the eye;
//! and a coarser far map in the targets' second layer,
//! reaching on to the settings' far radius, whose texels
//! keep how much of their ground covers hide. Each map is drawn again only
//! when it no longer holds the camera's surroundings or the objects
//! changed.

use crate::shaders::entry::occupancy;
use engine_core::Occupancy;
use engine_core::glam::Vec3;

use crate::bindings::OCCUPANCY;
use crate::cached::{Bound, GroupCache, Versioned};

/// The coverage target's format, and the heights target's.
const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const HEIGHTS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// What heights hold where nothing covers: no top, no bottom.
const NOTHING: f64 = -1.0e4;

/// The maps: the near one, and the far one, each a layer of
/// the targets.
const NEAR: usize = 0;
const FAR: usize = 1;

use crate::gpu_layouts::occupancy::SplatUniform;

pub(crate) use crate::gpu_layouts::occupancy::Cover;

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

/// Where a map lies: its center (x and z), meters a side, reference
/// height and texels a side, and the meters around the eye covers darken
/// within.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Area {
    pub center: [f32; 2],
    pub size: f32,
    pub reference: f32,
    pub resolution: u32,
    pub reach: f32,
}

pub(crate) struct OccupancyPass {
    layout: crate::pipelines::GroupLayout,
    // Each map's pipeline: the near one keeps the densest cover, the far
    // one adds covers up.
    pipelines: [Option<wgpu::RenderPipeline>; 2],
    uniforms: [wgpu::Buffer; 2],
    covers: Versioned<wgpu::Buffer>,
    covers_written: Vec<Cover>,
    // The coverage and heights targets, the views shading reads (made anew
    // with the targets) and each map's layer of them.
    targets: [wgpu::Texture; 2],
    views: [Versioned<wgpu::TextureView>; 2],
    layers: [[wgpu::TextureView; 2]; 2],
    groups: [GroupCache<5>; 2],
    // Each map's area last drawn and what it was drawn from: the slots'
    // version.
    drawn: [Option<(Area, u64)>; 2],
    // This frame's draws, where they redraw: the area and the slots to draw.
    pending: [Option<(Area, u32)>; 2],
    /// Times a map was drawn, and objects in the last draw's slots.
    pub(crate) redraws: u32,
    pub(crate) slots: u32,
}

impl OccupancyPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        // Zeroed targets cover nothing until the first draw.
        let (targets, views, layers) = targets(device, 1);
        let uniform =
            |label| crate::buffers::uniform(device, label, size_of::<SplatUniform>() as u64);
        Self {
            layout: crate::pipelines::layout(device, &OCCUPANCY),
            pipelines: [None, None],
            uniforms: [
                uniform("engine/occupancy-splat"),
                uniform("engine/occupancy-far-splat"),
            ],
            covers: Versioned::new(crate::buffers::storage(
                device,
                "engine/occupancy-covers",
                size_of::<Cover>() as u64,
                wgpu::BufferUsages::empty(),
            )),
            covers_written: Vec::new(),
            targets,
            views,
            layers,
            groups: Default::default(),
            drawn: [None, None],
            pending: [None, None],
            redraws: 0,
            slots: 0,
        }
    }

    /// The coverage and heights views the main passes read, as a group
    /// binds them.
    pub(crate) fn views(&self) -> (Bound<'_>, Bound<'_>) {
        (self.views[0].bound(), self.views[1].bound())
    }

    /// Bytes of the maps' targets.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.targets
            .iter()
            .map(|target| {
                u64::from(target.width())
                    * u64::from(target.height())
                    * u64::from(target.depth_or_array_layers())
            })
            .sum::<u64>()
            * 6
            / 2
    }

    /// Plans this frame: with `settings` on and placed objects whose
    /// slots' version is `version` (`covers` per draw group, `slots` in
    /// all), each map's area around `eye`, drawn again when it moved too
    /// far for the last draw to hold the camera's surroundings, or the
    /// slots or covers changed. Returns the near map's area and the far
    /// map's, if any, with the strength covers darken by;
    /// `None` without maps.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &Occupancy,
        eye: Vec3,
        (covers, slots, version): (&[Cover], u32, u64),
    ) -> Option<(Area, Option<Area>, f32)> {
        self.pending = [None, None];
        if !settings.enabled || settings.strength <= 0.0 || slots == 0 || covers.is_empty() {
            self.drawn = [None, None];
            return None;
        }
        let resolution = settings.resolution;
        if self.targets[0].width() != resolution {
            let (targets, views, layers) = targets(device, resolution);
            self.targets = targets;
            self.views = views;
            self.layers = layers;
            self.drawn = [None, None];
        }
        if covers != self.covers_written.as_slice() {
            let bytes: &[u8] = bytemuck::cast_slice(covers);
            crate::buffers::ensure_storage(
                device,
                &mut self.covers,
                "engine/occupancy-covers",
                bytes.len() as u64,
                wgpu::BufferUsages::empty(),
            );
            queue.write_buffer(&self.covers, 0, bytes);
            self.covers_written = covers.to_vec();
            self.drawn = [None, None];
        }
        let near = self.plan(queue, NEAR, settings.radius, eye, (covers, slots, version));
        let far = (settings.far_radius > settings.radius).then(|| {
            self.plan(
                queue,
                FAR,
                settings.far_radius,
                eye,
                (covers, slots, version),
            )
        });
        if far.is_none() {
            self.drawn[FAR] = None;
        }
        Some((near, far, settings.strength))
    }

    /// Plans map `map` reaching `reach` meters around `eye`: covers darken
    /// within it, and the eye strays at most an eighth of the map's side
    /// from its center before it is drawn again, so the map
    /// reaches a third beyond. The map follows the eye in steps of a
    /// sixteenth of its side.
    fn plan(
        &mut self,
        queue: &wgpu::Queue,
        map: usize,
        reach: f32,
        eye: Vec3,
        (covers, slots, version): (&[Cover], u32, u64),
    ) -> Area {
        let size = 8.0 / 3.0 * reach;
        let resolution = self.targets[0].width();
        let step = size / 16.0;
        let snapped = [(eye.x / step).round() * step, (eye.z / step).round() * step];
        let current = self.drawn[map].filter(|(area, drawn)| {
            *drawn == version
                && area.size == size
                && area.resolution == resolution
                && (eye.x - area.center[0]).abs() <= size / 8.0
                && (eye.z - area.center[1]).abs() <= size / 8.0
        });
        if let Some((area, _)) = current {
            return area;
        }
        let area = Area {
            center: snapped,
            size,
            reference: eye.y,
            resolution,
            reach,
        };
        let uniform = SplatUniform {
            area: [area.center[0], area.center[1], size, area.reference],
            counts: [
                slots,
                covers.len() as u32,
                u32::from(map == FAR),
                resolution,
            ],
        };
        queue.write_buffer(&self.uniforms[map], 0, bytemuck::bytes_of(&uniform));
        self.pending[map] = Some((area, slots));
        self.drawn[map] = Some((area, version));
        area
    }

    /// Creates the maps' pipelines unless they exist.
    pub(crate) fn warm(&mut self, device: &wgpu::Device) {
        for map in [NEAR, FAR] {
            self.pipelines[map].get_or_insert_with(|| pipeline(device, &self.layout, map));
        }
    }

    /// Whether this frame draws a map.
    pub(crate) fn redrawing(&self) -> bool {
        self.pending.iter().any(Option::is_some)
    }

    /// Draws the maps this frame redraws from the placed objects' `records`
    /// and `poses`, and how their layers thin (`thins`), into
    /// `encoder`; timed by `profiler` (`occupancy`).
    pub(crate) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        (records, poses, thins, generation): (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u64),
        profiler: &crate::profiling::GpuProfiler,
    ) {
        for map in [NEAR, FAR] {
            let Some((_, slots)) = self.pending[map].take() else {
                continue;
            };
            self.pipelines[map].get_or_insert_with(|| pipeline(device, &self.layout, map));
            // The slots' buffers are replaced together, counted by the
            // scatter pass.
            let group = self.groups[map].bind(
                device,
                "engine/occupancy",
                (&OCCUPANCY, &self.layout),
                [
                    (
                        "splat",
                        Bound::fixed(self.uniforms[map].as_entire_binding()),
                    ),
                    (
                        "records",
                        Bound::counted(generation, records.as_entire_binding()),
                    ),
                    (
                        "poses",
                        Bound::counted(generation, poses.as_entire_binding()),
                    ),
                    ("covers", self.covers.bound()),
                    (
                        "thins",
                        Bound::counted(generation, thins.as_entire_binding()),
                    ),
                ],
            );
            let Some(pipeline) = &self.pipelines[map] else {
                continue;
            };
            let attachment = |view, clear: f64| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear,
                            g: clear,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/occupancy"),
                color_attachments: &[
                    attachment(&self.layers[0][map], 0.0),
                    attachment(&self.layers[1][map], NOTHING),
                ],
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
}

/// The coverage and heights targets, `resolution` texels a side with a
/// layer for each map, the views shading reads and each map's layer of
/// each target.
fn targets(
    device: &wgpu::Device,
    resolution: u32,
) -> (
    [wgpu::Texture; 2],
    [Versioned<wgpu::TextureView>; 2],
    [[wgpu::TextureView; 2]; 2],
) {
    let target = |label, format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 2,
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
    let whole = |target: &wgpu::Texture| {
        target.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
    };
    let layer = |target: &wgpu::Texture, map: u32| {
        target.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: map,
            array_layer_count: Some(1),
            ..Default::default()
        })
    };
    let views = [
        Versioned::new(whole(&targets[0])),
        Versioned::new(whole(&targets[1])),
    ];
    let layers = [
        [layer(&targets[0], 0), layer(&targets[0], 1)],
        [layer(&targets[1], 0), layer(&targets[1], 1)],
    ];
    (targets, views, layers)
}

/// Map `map`'s pipeline: the near map keeps each target's maximum, the
/// far one adds covers up and keeps the heights' maximum.
fn pipeline(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
    map: usize,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("engine/occupancy.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu_data::OCCUPANCY_SHADER.into()),
    });
    let pipeline_layout = crate::pipelines::pipeline_layout(
        device,
        &crate::pipeline_layouts::OCCUPANCY_PIPELINES,
        &[layout],
    );
    let blend = |operation| {
        Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation,
            },
        })
    };
    let coverage = if map == FAR {
        wgpu::BlendOperation::Add
    } else {
        wgpu::BlendOperation::Max
    };
    crate::pipelines::render(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some("engine/occupancy"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(occupancy::vs_splat.name),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(occupancy::fs_splat.name),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: COVERAGE_FORMAT,
                        blend: blend(coverage),
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: HEIGHTS_FORMAT,
                        blend: blend(wgpu::BlendOperation::Max),
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
