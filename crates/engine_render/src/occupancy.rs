//! Occupancy: a top-down map around the camera of what
//! placed objects cover, drawn from the scatter pass's slots
//! (`occupancy.wgsl`) and read by the shared shading (`scene.wgsl`'s
//! `occupancy_visibility`), which darkens the sky light of what lies under
//! or within a cover, less toward the end of its reach around the eye;
//! and a coarser far map in the targets' second layer,
//! reaching on to the settings' far radius, whose texels
//! keep how much of their ground covers hide.
//!
//! Each map wraps, as the kept shadow levels do: a world point falls on
//! the texel its place in its map-wide page gives, and a window of `CELLS`
//! by `CELLS` cells follows the camera over the map in whole cells. A frame
//! redraws only cells: those the window uncovers, and those under objects
//! that came or went. Its selection lists the objects whose footprints
//! reach them, the cells are cleared, and the listed footprints are drawn
//! into them alone.

use crate::shaders::entry::occupancy;
use engine_core::Occupancy;
use engine_core::glam::Vec3;

use crate::bindings::OCCUPANCY;
use crate::buffers::Growable;
use crate::cached::{Bound, GroupCache, Versioned};
use crate::gpu_layouts::constants::{OCCUPANCY_CELLS as CELLS, OCCUPANCY_GROUP, OCCUPANCY_QUAD};
use crate::gpu_layouts::occupancy::{ListedDraw, SplatUniform};
use crate::pipelines::{Fillers, GroupLayout};
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{FrameShape, Node, Pass, RenderGraph, Resource};
use crate::shaders::Entry;

/// The coverage target's format, and the heights target's.
const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const HEIGHTS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// The maps: the near one, and the far one, each a layer of
/// the targets.
const NEAR: usize = 0;
const FAR: usize = 1;

/// Cells across a window, as a signed count.
const SIDE: i32 = CELLS as i32;

/// Words of a map's cells, a bit each.
const CELL_WORDS: usize = (CELLS * CELLS / 32) as usize;

const _: () = assert!(CELLS == 16, "a word holds two rows of cells");

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

/// Where a map lies: its window's center (x and z), meters a side,
/// reference height and texels a side, and the meters around the eye
/// covers darken within.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Area {
    pub center: [f32; 2],
    pub size: f32,
    pub reference: f32,
    pub resolution: u32,
    pub reach: f32,
}

/// A map's cells to redraw: bit `16 y + x` for the cell at column `x` and
/// row `y` of its texels.
#[derive(Clone, Copy, Debug, Default)]
struct Cells([u32; CELL_WORDS]);

impl Cells {
    const ALL: Self = Self([u32::MAX; CELL_WORDS]);

    /// Marks the cell at column `column` and row `row`.
    fn set(&mut self, column: u32, row: u32) {
        let bit = row * CELLS + column;
        self.0[(bit / 32) as usize] |= 1 << (bit % 32);
    }

    /// Marks every cell of column `column`.
    fn set_column(&mut self, column: u32) {
        for row in 0..CELLS {
            self.set(column, row);
        }
    }

    /// Marks every cell of row `row`.
    fn set_row(&mut self, row: u32) {
        for column in 0..CELLS {
            self.set(column, row);
        }
    }

    fn any(&self) -> bool {
        self.0.iter().any(|&word| word != 0)
    }
}

/// A map's window: `CELLS` by `CELLS` cells `cell` meters wide following
/// the camera, the absolute cell of its first corner (x and z), and the
/// height its heights are above. The map holds the absolute cell `(a, b)`,
/// whose first corner is `(a, b) * cell`, in its cell `(a mod CELLS, b mod
/// CELLS)`: the window moves in whole cells, and only the cells it uncovers
/// redraw.
#[derive(Clone, Copy, Debug)]
struct Window {
    cell: f32,
    origin: [i32; 2],
    reference: f32,
}

impl Window {
    /// Moves the window's first corner to `target` once the eye is more
    /// than a cell from its middle cells, marking the cells it uncovers in
    /// `dirty`.
    fn scroll(&mut self, target: [i32; 2], dirty: &mut Cells) {
        let shift = [target[0] - self.origin[0], target[1] - self.origin[1]];
        if shift.iter().all(|step| step.abs() <= 1) {
            return;
        }
        let entering = |origin: i32, shift: i32| {
            if shift >= 0 {
                origin + SIDE..origin + SIDE + shift.min(SIDE)
            } else {
                origin + shift.max(-SIDE)..origin
            }
        };
        for column in entering(self.origin[0], shift[0]) {
            dirty.set_column(column.rem_euclid(SIDE) as u32);
        }
        for row in entering(self.origin[1], shift[1]) {
            dirty.set_row(row.rem_euclid(SIDE) as u32);
        }
        self.origin = target;
    }

    /// The window's first corner (x and z, world).
    fn corner(&self) -> [f32; 2] {
        self.origin.map(|cell| cell as f32 * self.cell)
    }
}

/// The maps' pipelines: each map's (the near one keeps each target's
/// maximum, the far one adds covers up and keeps the heights' maximum),
/// the one clearing the cells redrawn, and the selection's.
struct Pipelines {
    splats: [wgpu::RenderPipeline; 2],
    clear: wgpu::RenderPipeline,
    select: wgpu::ComputePipeline,
}

impl Pipelines {
    fn new(device: &wgpu::Device, layout: &GroupLayout) -> Self {
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
        // Into both targets, the coverage blended by `coverage` and the
        // heights by `heights` (written over without either).
        let draw = |label: &str,
                    (vertex, fragment): (Entry, Entry),
                    (coverage, heights): (Option<wgpu::BlendState>, Option<wgpu::BlendState>)| {
            crate::pipelines::geometry(
                device,
                &crate::pipelines::Geometry {
                    label,
                    layout: &pipeline_layout,
                    module: &module,
                    vertex,
                    vertex_constants: &[],
                    buffers: &[],
                    primitive: wgpu::PrimitiveState::default(),
                    depth: None,
                    fragment: Some(crate::pipelines::Fragment {
                        entry: fragment,
                        constants: &[],
                        targets: &[
                            Some(wgpu::ColorTargetState {
                                format: COVERAGE_FORMAT,
                                blend: coverage,
                                write_mask: wgpu::ColorWrites::RED,
                            }),
                            Some(wgpu::ColorTargetState {
                                format: HEIGHTS_FORMAT,
                                blend: heights,
                                write_mask: wgpu::ColorWrites::RED | wgpu::ColorWrites::GREEN,
                            }),
                        ],
                    }),
                },
            )
        };
        let splat = (occupancy::vs_splat, occupancy::fs_splat);
        let (max, add) = (
            blend(wgpu::BlendOperation::Max),
            blend(wgpu::BlendOperation::Add),
        );
        Self {
            splats: [
                draw("engine/occupancy", splat, (max, max)),
                draw("engine/occupancy-far", splat, (add, max)),
            ],
            clear: draw(
                "engine/occupancy-clear",
                (occupancy::vs_clear, occupancy::fs_clear),
                (None, None),
            ),
            select: crate::pipelines::compute_entry(
                device,
                &pipeline_layout,
                &module,
                occupancy::select_covers,
            ),
        }
    }
}

pub(crate) struct OccupancyPass {
    layout: GroupLayout,
    // Bindings each pass leaves alone of the table the selection and the
    // draws share.
    fillers: Fillers,
    pipelines: Option<Pipelines>,
    // Each map's uniform, and the arguments of the draw of its listed
    // footprints.
    uniforms: [wgpu::Buffer; 2],
    draws: [wgpu::Buffer; 2],
    covers: Growable,
    covers_written: Vec<Cover>,
    // The footprints a map's selection lists, which its draw reads.
    listed: Growable,
    // The coverage and heights targets, the views shading reads (made anew
    // with the targets) and each map's layer of them.
    targets: [wgpu::Texture; 2],
    views: [Versioned<wgpu::TextureView>; 2],
    layers: [[wgpu::TextureView; 2]; 2],
    // Each map's groups of its draws and of its selection.
    groups: [GroupCache<6>; 2],
    select_groups: [GroupCache<6>; 2],
    // Each map's window, none before its first draw or once its texels no
    // longer hold it; the cells to redraw with its next uniform; and
    // whether this frame redraws cells of it.
    windows: [Option<Window>; 2],
    dirty: [Cells; 2],
    pending: [bool; 2],
    // The selection's workgroups this frame, over every slot.
    workgroups: u32,
    /// Times a map redrew cells.
    pub(crate) redraws: u32,
}

impl OccupancyPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        // Zeroed targets cover nothing until the first draw.
        let (targets, views, layers) = targets(device, 1);
        let uniform =
            |label| crate::buffers::uniform(device, label, size_of::<SplatUniform>() as u64);
        let draw = |label| {
            crate::buffers::storage(
                device,
                label,
                size_of::<ListedDraw>() as u64,
                wgpu::BufferUsages::INDIRECT,
            )
        };
        Self {
            layout: crate::pipelines::layout(device, &OCCUPANCY),
            fillers: Fillers::new(device),
            pipelines: None,
            uniforms: [
                uniform("engine/occupancy-splat"),
                uniform("engine/occupancy-far-splat"),
            ],
            draws: [
                draw("engine/occupancy-draw"),
                draw("engine/occupancy-far-draw"),
            ],
            covers: Growable::new(
                device,
                "engine/occupancy-covers",
                size_of::<Cover>() as u64,
                wgpu::BufferUsages::empty(),
            ),
            covers_written: Vec::new(),
            listed: Growable::new(
                device,
                "engine/occupancy-listed",
                size_of::<u32>() as u64,
                wgpu::BufferUsages::empty(),
            ),
            targets,
            views,
            layers,
            groups: Default::default(),
            select_groups: Default::default(),
            windows: [None, None],
            dirty: [Cells::default(); 2],
            pending: [false; 2],
            workgroups: 0,
            redraws: 0,
        }
    }

    /// The coverage and heights views the main passes read, as a group
    /// binds them.
    pub(crate) fn views(&self) -> (Bound<'_>, Bound<'_>) {
        (self.views[0].bound(), self.views[1].bound())
    }

    /// Bytes of the maps' targets and of the footprints listed.
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
            + self.listed.size()
    }

    /// Plans this frame: with `settings` on and placed objects
    /// (`covers` per draw group, `slots` in all), each map's window around
    /// `eye`, and the cells it redraws: those its window uncovered and
    /// those under objects that came or went since the last frame's. Every
    /// cell redraws once the targets, the covers or the settings changed.
    /// Returns the near map's area and the far map's, if any, with the
    /// strength covers darken by; `None` without maps.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &Occupancy,
        eye: Vec3,
        (covers, slots): (&[Cover], u32),
    ) -> Option<(Area, Option<Area>, f32)> {
        self.pending = [false; 2];
        if !settings.enabled || settings.strength <= 0.0 || slots == 0 || covers.is_empty() {
            self.windows = [None, None];
            return None;
        }
        // Cells of whole texels.
        let resolution = settings.resolution.next_multiple_of(CELLS);
        if self.targets[0].width() != resolution {
            (self.targets, self.views, self.layers) = targets(device, resolution);
            self.windows = [None, None];
        }
        if covers != self.covers_written.as_slice() {
            let bytes: &[u8] = bytemuck::cast_slice(covers);
            self.covers.grow(device, bytes.len() as u64);
            queue.write_buffer(&self.covers, 0, bytes);
            self.covers_written = covers.to_vec();
            self.windows = [None, None];
        }
        // Room for every object's footprint and as many again listed past
        // the map's edges.
        self.listed
            .grow(device, 2 * u64::from(slots) * size_of::<u32>() as u64);
        self.workgroups = slots.div_ceil(OCCUPANCY_GROUP);
        let counts = (covers.len() as u32, slots);
        let near = self.plan(queue, NEAR, settings.radius, eye, counts);
        let far = (settings.far_radius > settings.radius)
            .then(|| self.plan(queue, FAR, settings.far_radius, eye, counts));
        if far.is_none() {
            self.windows[FAR] = None;
        }
        Some((near, far, settings.strength))
    }

    /// Plans map `map` reaching `reach` meters around `eye`: covers darken
    /// within it, and its window, a third wider each way, follows the eye
    /// once it strays more than a cell from the window's middle cells,
    /// redrawing the cells it uncovers. A window first placed, of another
    /// size or whose reference height the eye left by more than the reach
    /// starts over at the eye, every cell redrawn. The cells to redraw go to
    /// the GPU with the map's uniform, which `counts` (draw groups and
    /// slots) complete.
    fn plan(
        &mut self,
        queue: &wgpu::Queue,
        map: usize,
        reach: f32,
        eye: Vec3,
        (groups, slots): (u32, u32),
    ) -> Area {
        let size = 8.0 / 3.0 * reach;
        let cell = size / CELLS as f32;
        let resolution = self.targets[0].width();
        let target = [
            (eye.x / cell).floor() as i32 - SIDE / 2,
            (eye.z / cell).floor() as i32 - SIDE / 2,
        ];
        let dirty = &mut self.dirty[map];
        let window = match &mut self.windows[map] {
            Some(window) if window.cell == cell && (eye.y - window.reference).abs() <= reach => {
                window.scroll(target, dirty);
                *window
            }
            window => {
                *dirty = Cells::ALL;
                *window.insert(Window {
                    cell,
                    origin: target,
                    reference: eye.y,
                })
            }
        };
        let corner = window.corner();
        let cells = std::mem::take(dirty);
        if cells.any() {
            let texels = resolution / CELLS;
            let uniform = SplatUniform {
                area: [corner[0], corner[1], size, window.reference],
                counts: [slots, groups, u32::from(map == FAR), resolution],
                place: [
                    window.origin[0].rem_euclid(SIDE) as u32 * texels,
                    window.origin[1].rem_euclid(SIDE) as u32 * texels,
                    texels,
                    0,
                ],
                dirty: bytemuck::cast(cells.0),
            };
            queue.write_buffer(&self.uniforms[map], 0, bytemuck::bytes_of(&uniform));
            let draw = ListedDraw {
                vertices: OCCUPANCY_QUAD,
                instances: 0,
                first_vertex: 0,
                first_instance: 0,
            };
            queue.write_buffer(&self.draws[map], 0, bytemuck::bytes_of(&draw));
            self.pending[map] = true;
        }
        Area {
            center: [corner[0] + size * 0.5, corner[1] + size * 0.5],
            size,
            reference: window.reference,
            resolution,
            reach,
        }
    }

    /// Marks the cells of each map's window under the world box `min` to
    /// `max`, where objects came or went, a texel wider each way as far
    /// footprints widen to most of one: the next frame's maps redraw them.
    pub(crate) fn dirty_box(&mut self, min: Vec3, max: Vec3) {
        let resolution = self.targets[0].width() as f32;
        for (window, dirty) in self.windows.iter().zip(&mut self.dirty) {
            let Some(window) = window else {
                continue;
            };
            let margin = window.cell * CELLS as f32 / resolution;
            let span = |low: f32, high: f32, origin: i32| {
                let cell = |value: f32| (value / window.cell).floor() as i32;
                cell(low - margin).max(origin)..=cell(high + margin).min(origin + SIDE - 1)
            };
            let rows = span(min.z, max.z, window.origin[1]);
            for column in span(min.x, max.x, window.origin[0]) {
                for row in rows.clone() {
                    dirty.set(column.rem_euclid(SIDE) as u32, row.rem_euclid(SIDE) as u32);
                }
            }
        }
    }

    /// Creates the maps' pipelines unless they exist.
    pub(crate) fn warm(&mut self, device: &wgpu::Device) {
        if self.pipelines.is_none() {
            self.pipelines = Some(Pipelines::new(device, &self.layout));
        }
    }

    /// Readies the maps this frame redraws cells of, from the placed
    /// objects' `records` and `poses`, and how their layers thin (`thins`),
    /// whose buffers `generation` counts; `None` when none redraws.
    pub(crate) fn prepare_draws(
        &mut self,
        device: &wgpu::Device,
        (records, poses, thins, generation): (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, u64),
    ) -> Option<OccupancyFrame> {
        if !self.pending.contains(&true) {
            return None;
        }
        self.warm(device);
        // The slots' buffers are replaced together, counted by the
        // scatter pass.
        let records = Bound::counted(generation, records.as_entire_binding());
        let poses = Bound::counted(generation, poses.as_entire_binding());
        let thins = Bound::counted(generation, thins.as_entire_binding());
        for map in [NEAR, FAR] {
            if !self.pending[map] {
                continue;
            }
            let splat = Bound::fixed(self.uniforms[map].as_entire_binding());
            self.groups[map].bind_filled(
                device,
                "engine/occupancy",
                (&OCCUPANCY, &self.layout),
                [
                    ("splat", splat.clone()),
                    ("records", records.clone()),
                    ("poses", poses.clone()),
                    ("covers", self.covers.bound()),
                    ("thins", thins.clone()),
                    ("listed", self.listed.bound()),
                ],
                &self.fillers,
            );
            self.select_groups[map].bind_filled(
                device,
                "engine/occupancy-select",
                (&OCCUPANCY, &self.layout),
                [
                    ("splat", splat),
                    ("records", records.clone()),
                    ("poses", poses.clone()),
                    ("covers", self.covers.bound()),
                    ("listing", self.listed.bound()),
                    (
                        "listed_draw",
                        Bound::fixed(self.draws[map].as_entire_binding()),
                    ),
                ],
                &self.fillers,
            );
            self.redraws += 1;
        }
        Some(OccupancyFrame {
            draws: self.pending,
            workgroups: self.workgroups,
        })
    }

    /// Redraws the cells of the maps `frame` redraws into `encoder`, one
    /// map after the other: its selection lists the footprints reaching
    /// them, then its cells clear and the listed footprints draw. Timed by
    /// `profiler` (`occupancy`).
    fn encode_maps(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &OccupancyFrame,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let Some(pipelines) = &self.pipelines else {
            return;
        };
        let span = profiler.span("occupancy");
        if let Some(span) = span {
            span.mark(encoder, false);
        }
        for map in [NEAR, FAR] {
            let (true, Some(select), Some(group)) = (
                frame.draws[map],
                self.select_groups[map].get(),
                self.groups[map].get(),
            ) else {
                continue;
            };
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("engine/occupancy-select"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.select);
                pass.set_bind_group(0, select, &[]);
                pass.dispatch_workgroups(frame.workgroups, 1, 1);
            }
            // The cells not redrawn keep what they hold.
            let attachment = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/occupancy"),
                color_attachments: &[
                    attachment(&self.layers[0][map]),
                    attachment(&self.layers[1][map]),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, group, &[]);
            pass.set_pipeline(&pipelines.clear);
            pass.draw(0..OCCUPANCY_QUAD, 0..CELLS * CELLS);
            pass.set_pipeline(&pipelines.splats[map]);
            pass.draw_indirect(&self.draws[map], 0);
        }
        if let Some(span) = span {
            span.mark(encoder, true);
        }
    }
}

/// The maps a frame redraws cells of, and its selection's workgroups.
pub(crate) struct OccupancyFrame {
    draws: [bool; 2],
    workgroups: u32,
}

impl RenderFeature for OccupancyPass {
    type Frame = OccupancyFrame;
    type Stats = engine_core::OccupancyStats;

    /// The maps draw the placed objects' covers, where they draw.
    fn declare(_frame: &OccupancyFrame, shape: &FrameShape, graph: &mut RenderGraph) {
        graph
            .add(Pass::Feature(FeaturePass::Occupancy))
            .read_if(shape.scatter, Resource::Placed)
            .write(Resource::Occupancy);
    }

    fn encode(
        &self,
        frame: &OccupancyFrame,
        _node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        self.encode_maps(encoder, frame, encode.profiler);
        0
    }

    fn capacity_bytes(&self) -> u64 {
        OccupancyPass::capacity_bytes(self)
    }

    /// Times a map redrew cells.
    fn stats(
        this: Option<&Self>,
        _frame: Option<&OccupancyFrame>,
        _snapshot: &engine_core::RenderSnapshot,
    ) -> engine_core::OccupancyStats {
        engine_core::OccupancyStats {
            redraws: this.map_or(0, |occupancy| occupancy.redraws),
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
