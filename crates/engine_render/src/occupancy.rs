//! Occupancy: top-down maps around the detail eye of what placed objects
//! cover, drawn from the scatter pass's slots (`occupancy.wgsl`) and read
//! by the shared shading (`scene.wgsl`'s `occupancy_visibility`), which
//! darkens the sky light of what lies under or within a cover. The maps
//! nest, a layer of the targets each, the finest first: each reaches twice
//! as far as the one inside it with texels twice as wide, and where it lies
//! over the finer one its texels are the finer one's means, so it shows what
//! the finer one does there; shading reads the map a pixel's size picks and
//! blends it with the next. Every map's texels keep how much of their
//! ground covers hide, each footprint at least most of a texel wide
//! carrying its share of it.
//!
//! Each map wraps, as the kept shadow levels do: a world point falls on
//! the texel its place in its map-wide page gives, and a window of `CELLS`
//! by `CELLS` cells follows the detail eye over the map in whole cells. A
//! frame redraws only cells: those a window uncovers, those under objects
//! that came or went, and a coarser map's cells under those its finer
//! window left. Its selection lists the objects whose footprints reach
//! them, the cells are cleared, and the listed footprints are drawn into
//! them alone; then a coarser map takes the finer one's means over the
//! finer cells that changed and those under its own cells redrawn.

use crate::shaders::entry::occupancy;
use engine_core::Occupancy;
use engine_core::glam::Vec3;

use crate::bindings::OCCUPANCY;
use crate::buffers::Growable;
use crate::cached::{Bound, GroupCache, Versioned};
use crate::gpu_layouts::constants::{OCCUPANCY_CELLS as CELLS, OCCUPANCY_GROUP, OCCUPANCY_QUAD};
use crate::gpu_layouts::occupancy::{ListedDraw, SplatUniform};
use crate::gpu_layouts::scene::MAX_OCCUPANCY_LEVELS;
use crate::pipelines::{Fillers, GroupLayout};
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{FrameShape, Node, Pass, RenderGraph, Resource};
use crate::shaders::Entry;

/// The coverage target's format, and the heights target's.
const COVERAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
const HEIGHTS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Cells across a window, as a signed count.
const SIDE: i32 = CELLS as i32;

/// Words of a map's cells, a bit each.
const CELL_WORDS: usize = (CELLS * CELLS / 32) as usize;

const _: () = assert!(CELLS == 16, "a word holds two rows of cells");

/// The share of a map's side its window holds every point within either
/// way of the eye: six of its sixteen cells, as the window follows the eye
/// once it strays more than a cell from its middle cells.
const REACH_SHARE: f32 = 3.0 / 8.0;

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

/// What shading reads of the maps (`scene.wgsl`'s `Lighting`): one over the
/// finest map's side, its texels a side, the meters around the detail eye
/// within which it holds every point and the strength covers darken by;
/// the maps' count and the reference height their heights are above; and
/// each map's window's center (x and z), two maps a vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Lit {
    pub(crate) occupancy: [f32; 4],
    pub(crate) maps: [f32; 4],
    pub(crate) centers: [[f32; 4]; MAX_OCCUPANCY_LEVELS / 2],
}

/// The finest map's texel over the meters around the detail eye within
/// which it holds every point, as `settings` size it, which ambient
/// occlusion hands over by (`post.wgsl`'s `occlusion_share`).
pub(crate) fn texel_share(settings: &Occupancy) -> f32 {
    let (texel, reach) = finest(settings);
    texel / reach
}

/// The finest map's texel as `settings` size it, and the meters around
/// the eye within which it holds every point: its window's six cells either
/// way, less the half texel a lookup reads past the point.
fn finest(settings: &Occupancy) -> (f32, f32) {
    let texel = settings.radius / (REACH_SHARE * side_texels(settings) as f32);
    (texel, settings.radius - 0.5 * texel)
}

/// Each map's texels a side under `settings`: cells of whole texels, and
/// each finer cell whole texels of the next map.
fn side_texels(settings: &Occupancy) -> u32 {
    settings.resolution.next_multiple_of(2 * CELLS)
}

/// A map's cells: bit `16 y + x` for the cell at column `x` and row `y`,
/// of its texels or of its window.
#[derive(Clone, Copy, Debug, Default)]
struct Cells([u32; CELL_WORDS]);

impl Cells {
    const ALL: Self = Self([u32::MAX; CELL_WORDS]);

    /// Marks the cell at column `column` and row `row`.
    fn set(&mut self, column: u32, row: u32) {
        let bit = row * CELLS + column;
        self.0[(bit / 32) as usize] |= 1 << (bit % 32);
    }

    /// Whether the cell at column `column` and row `row` is marked.
    fn get(&self, column: u32, row: u32) -> bool {
        let bit = row * CELLS + column;
        self.0[(bit / 32) as usize] & (1 << (bit % 32)) != 0
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
/// the detail eye, and the absolute cell of its first corner (x and z). The
/// map holds the absolute cell `(a, b)`, whose first corner is
/// `(a, b) * cell`, in its cell `(a mod CELLS, b mod CELLS)`: the window
/// moves in whole cells, and only the cells it uncovers redraw.
#[derive(Clone, Copy, Debug)]
struct Window {
    cell: f32,
    origin: [i32; 2],
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

    /// Whether the window holds the absolute cell `cell`.
    fn holds(&self, cell: [i32; 2]) -> bool {
        (0..2).all(|axis| (self.origin[axis]..self.origin[axis] + SIDE).contains(&cell[axis]))
    }

    /// Whether the window holds the world box `min` to `max` (x and z).
    fn holds_box(&self, min: Vec3, max: Vec3) -> bool {
        let corner = self.corner();
        let side = self.cell * CELLS as f32;
        min.x >= corner[0]
            && min.z >= corner[1]
            && max.x < corner[0] + side
            && max.z < corner[1] + side
    }

    /// The window's first corner (x and z, world).
    fn corner(&self) -> [f32; 2] {
        self.origin.map(|cell| cell as f32 * self.cell)
    }
}

/// Where absolute cell `cell` falls among a map's cells.
fn wrapped(cell: [i32; 2]) -> [u32; 2] {
    cell.map(|axis| axis.rem_euclid(SIDE) as u32)
}

/// The maps' pipelines: the footprints' (covers composed as what each lets
/// through, heights keeping their extremes), the one clearing the cells
/// redrawn, the one taking a finer map's means, and the selection's.
struct Pipelines {
    splat: wgpu::RenderPipeline,
    clear: wgpu::RenderPipeline,
    means: wgpu::RenderPipeline,
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
        let blend = |dst_factor, operation| {
            let component = wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor,
                operation,
            };
            Some(wgpu::BlendState {
                color: component,
                alpha: component,
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
        // A footprint lets through what it does not cover of what the
        // others let through.
        let over = blend(wgpu::BlendFactor::OneMinusSrc, wgpu::BlendOperation::Add);
        let max = blend(wgpu::BlendFactor::One, wgpu::BlendOperation::Max);
        Self {
            splat: draw(
                "engine/occupancy",
                (occupancy::vs_splat, occupancy::fs_splat),
                (over, max),
            ),
            clear: draw(
                "engine/occupancy-clear",
                (occupancy::vs_clear, occupancy::fs_clear),
                (None, None),
            ),
            means: draw(
                "engine/occupancy-means",
                (occupancy::vs_means, occupancy::fs_means),
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
    uniforms: Vec<wgpu::Buffer>,
    draws: Vec<wgpu::Buffer>,
    covers: Growable,
    covers_written: Vec<Cover>,
    // The footprints a map's selection lists, which its draw reads.
    listed: Growable,
    // The coverage and heights targets, the views shading reads (made anew
    // with the targets), each map's layer of them, and what the finest map
    // reads in place of a finer one.
    targets: [wgpu::Texture; 2],
    views: [Versioned<wgpu::TextureView>; 2],
    layers: [Vec<wgpu::TextureView>; 2],
    unfiner: [wgpu::TextureView; 2],
    // Each map's groups of its draws and of its selection.
    groups: Vec<GroupCache<7>>,
    select_groups: Vec<GroupCache<6>>,
    // Each map's window, none before its first draw or once its texels no
    // longer hold it; the height every map's heights are above; the cells
    // to redraw with each map's next uniform; and whether this frame
    // redraws cells of each, and takes the finer one's means into each.
    windows: Vec<Option<Window>>,
    reference: f32,
    dirty: Vec<Cells>,
    pending: Vec<bool>,
    averaging: Vec<bool>,
    // The selection's workgroups this frame, over every slot.
    workgroups: u32,
    /// Times a map redrew cells, and took the finer one's means.
    pub(crate) redraws: u32,
    pub(crate) averages: u32,
}

impl OccupancyPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        // Zeroed targets cover nothing until the first draw.
        let (targets, views, layers) = targets(device, 1, 1);
        let unfiner = |label, format| {
            crate::targets::texture_2d(
                device,
                label,
                (1, 1),
                1,
                1,
                format,
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .create_view(&wgpu::TextureViewDescriptor::default())
        };
        Self {
            layout: crate::pipelines::layout(device, &OCCUPANCY),
            fillers: Fillers::new(device),
            pipelines: None,
            uniforms: Vec::new(),
            draws: Vec::new(),
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
            unfiner: [
                unfiner("engine/occupancy-unfiner-coverage", COVERAGE_FORMAT),
                unfiner("engine/occupancy-unfiner-heights", HEIGHTS_FORMAT),
            ],
            groups: Vec::new(),
            select_groups: Vec::new(),
            windows: Vec::new(),
            reference: 0.0,
            dirty: Vec::new(),
            pending: Vec::new(),
            averaging: Vec::new(),
            workgroups: 0,
            redraws: 0,
            averages: 0,
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
    /// the detail eye `eye`, the cells it redraws (those its window
    /// uncovered, those under objects that came or went since the last
    /// frame's, and those under the cells its finer window left) and the
    /// finer map's cells whose means it takes (those that changed, and
    /// those under its own cells redrawn). Every cell redraws once the
    /// targets, the covers or the settings changed, or the eye left the
    /// maps' reference height by more than the finest map's reach. Returns
    /// what shading reads of the maps; `None` without them.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: &Occupancy,
        eye: Vec3,
        (covers, slots): (&[Cover], u32),
    ) -> Option<Lit> {
        self.pending.fill(false);
        self.averaging.fill(false);
        if !settings.enabled || settings.strength <= 0.0 || slots == 0 || covers.is_empty() {
            self.windows.clear();
            return None;
        }
        let maps = settings.levels.clamp(1, MAX_OCCUPANCY_LEVELS as u32) as usize;
        let resolution = side_texels(settings);
        if self.targets[0].width() != resolution || self.layers[0].len() != maps {
            (self.targets, self.views, self.layers) = targets(device, resolution, maps);
            self.windows.clear();
        }
        self.size(device, maps);
        if covers != self.covers_written.as_slice() {
            let bytes: &[u8] = bytemuck::cast_slice(covers);
            self.covers.grow(device, bytes.len() as u64);
            queue.write_buffer(&self.covers, 0, bytes);
            self.covers_written = covers.to_vec();
            self.windows.clear();
        }
        // Room for every object's footprint and as many again listed past
        // the map's edges.
        self.listed
            .grow(device, 2 * u64::from(slots) * size_of::<u32>() as u64);
        self.workgroups = slots.div_ceil(OCCUPANCY_GROUP);
        if self.windows.first().copied().flatten().is_none()
            || (eye.y - self.reference).abs() > settings.radius
        {
            self.windows.clear();
            self.reference = eye.y;
        }
        self.windows.resize(maps, None);
        // Each map's window around the eye, its cells uncovered redrawing.
        let before = self.windows.clone();
        for (map, (window, dirty)) in self.windows.iter_mut().zip(&mut self.dirty).enumerate() {
            let cell = settings.radius * (1_u32 << map) as f32 / REACH_SHARE / CELLS as f32;
            let target = [
                (eye.x / cell).floor() as i32 - SIDE / 2,
                (eye.z / cell).floor() as i32 - SIDE / 2,
            ];
            match window {
                Some(window) if window.cell == cell => window.scroll(target, dirty),
                window => {
                    *dirty = Cells::ALL;
                    *window = Some(Window {
                        cell,
                        origin: target,
                    });
                }
            }
        }
        // A coarser map redraws the cells under those its finer window
        // left, which it held the finer one's means over.
        for map in 1..maps {
            let (Some(left), Some(finer), Some(window)) =
                (before[map - 1], self.windows[map - 1], self.windows[map])
            else {
                continue;
            };
            for cell in cells(left).filter(|&cell| !finer.holds(cell)) {
                let under = cell.map(|axis| axis.div_euclid(2));
                if window.holds(under) {
                    let [column, row] = wrapped(under);
                    self.dirty[map].set(column, row);
                }
            }
        }
        let texels = resolution / CELLS;
        let mut lit = Lit {
            occupancy: [0.0; 4],
            maps: [maps as f32, self.reference, 0.0, 0.0],
            centers: [[0.0; 4]; MAX_OCCUPANCY_LEVELS / 2],
        };
        // The finer map's cells that changed this frame, which the next
        // coarser one takes the means of.
        let mut changed = Cells::default();
        for map in 0..maps {
            let Some(window) = self.windows[map] else {
                continue;
            };
            let dirty = std::mem::take(&mut self.dirty[map]);
            let finer = map.checked_sub(1).and_then(|finer| self.windows[finer]);
            let mut means = Cells::default();
            let mut now = dirty;
            if let Some(finer) = finer {
                for (index, cell) in cells(finer).enumerate() {
                    let [x, y] = wrapped(cell);
                    let [under_x, under_y] = wrapped(cell.map(|axis| axis.div_euclid(2)));
                    if changed.get(x, y) || dirty.get(under_x, under_y) {
                        means.set(index as u32 % CELLS, index as u32 / CELLS);
                        now.set(under_x, under_y);
                    }
                }
            }
            changed = now;
            let size = window.cell * CELLS as f32;
            let corner = window.corner();
            let center = [corner[0] + size * 0.5, corner[1] + size * 0.5];
            lit.centers[map / 2][2 * (map % 2)] = center[0];
            lit.centers[map / 2][2 * (map % 2) + 1] = center[1];
            let (redraw, average) = (dirty.any(), means.any());
            if !redraw && !average {
                continue;
            }
            let uniform = SplatUniform {
                area: [corner[0], corner[1], size, self.reference],
                counts: [slots, covers.len() as u32, 0, resolution],
                place: [
                    window.origin[0].rem_euclid(SIDE) as u32 * texels,
                    window.origin[1].rem_euclid(SIDE) as u32 * texels,
                    texels,
                    0,
                ],
                dirty: bytemuck::cast(dirty.0),
                finer: finer.map_or([0; 4], |finer| [finer.origin[0], finer.origin[1], 1, 0]),
                means: bytemuck::cast(means.0),
            };
            queue.write_buffer(&self.uniforms[map], 0, bytemuck::bytes_of(&uniform));
            if redraw {
                let draw = ListedDraw {
                    vertices: OCCUPANCY_QUAD,
                    instances: 0,
                    first_vertex: 0,
                    first_instance: 0,
                };
                queue.write_buffer(&self.draws[map], 0, bytemuck::bytes_of(&draw));
            }
            self.pending[map] = redraw;
            self.averaging[map] = average;
        }
        let (_, reach) = finest(settings);
        let finest_side = self.windows[0].map_or(1.0, |window| window.cell * CELLS as f32);
        lit.occupancy = [
            1.0 / finest_side,
            resolution as f32,
            reach,
            settings.strength,
        ];
        Some(lit)
    }

    /// Sizes each map's uniform, draw, groups and plan to `maps` maps.
    fn size(&mut self, device: &wgpu::Device, maps: usize) {
        while self.uniforms.len() < maps {
            self.uniforms.push(crate::buffers::uniform(
                device,
                "engine/occupancy-splat",
                size_of::<SplatUniform>() as u64,
            ));
            self.draws.push(crate::buffers::storage(
                device,
                "engine/occupancy-draw",
                size_of::<ListedDraw>() as u64,
                wgpu::BufferUsages::INDIRECT,
            ));
        }
        self.groups.resize_with(maps, GroupCache::default);
        self.select_groups.resize_with(maps, GroupCache::default);
        self.dirty.resize(maps, Cells::default());
        self.pending.resize(maps, false);
        self.averaging.resize(maps, false);
    }

    /// Marks the cells of each map's window under the world box `min` to
    /// `max`, where objects came or went, a texel wider each way as
    /// footprints widen to most of one: the next frame's maps redraw them.
    /// A coarser map whose finer window holds the box takes the finer one's
    /// means over it instead.
    pub(crate) fn dirty_box(&mut self, min: Vec3, max: Vec3) {
        let resolution = self.targets[0].width() as f32;
        for map in 0..self.windows.len() {
            let Some(window) = self.windows[map] else {
                continue;
            };
            let margin = window.cell * CELLS as f32 / resolution;
            let finer = map.checked_sub(1).and_then(|finer| self.windows[finer]);
            if finer.is_some_and(|finer| {
                let margin = Vec3::splat(finer.cell * CELLS as f32 / resolution);
                finer.holds_box(min - margin, max + margin)
            }) {
                continue;
            }
            let span = |low: f32, high: f32, origin: i32| {
                let cell = |value: f32| (value / window.cell).floor() as i32;
                cell(low - margin).max(origin)..=cell(high + margin).min(origin + SIDE - 1)
            };
            let rows = span(min.z, max.z, window.origin[1]);
            for column in span(min.x, max.x, window.origin[0]) {
                for row in rows.clone() {
                    let [column, row] = wrapped([column, row]);
                    self.dirty[map].set(column, row);
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

    /// Readies the maps this frame redraws cells of or takes means into,
    /// from the placed objects' `records` and `poses`, whose buffers
    /// `generation` counts; `None` when none changes.
    pub(crate) fn prepare_draws(
        &mut self,
        device: &wgpu::Device,
        (records, poses, generation): (&wgpu::Buffer, &wgpu::Buffer, u64),
    ) -> Option<OccupancyFrame> {
        if !self.pending.contains(&true) && !self.averaging.contains(&true) {
            return None;
        }
        self.warm(device);
        // The slots' buffers are replaced together, counted by the
        // scatter pass.
        let records = Bound::counted(generation, records.as_entire_binding());
        let poses = Bound::counted(generation, poses.as_entire_binding());
        for map in 0..self.pending.len() {
            let (redraw, average) = (self.pending[map], self.averaging[map]);
            if !redraw && !average {
                continue;
            }
            let splat = Bound::fixed(self.uniforms[map].as_entire_binding());
            // The finer map's layers, made anew with the targets.
            let finer = |target: usize| match map.checked_sub(1) {
                Some(finer) => Bound::of(
                    &self.views[target],
                    crate::pipelines::view(&self.layers[target][finer]),
                ),
                None => Bound::fixed(crate::pipelines::view(&self.unfiner[target])),
            };
            self.groups[map].bind_filled(
                device,
                "engine/occupancy",
                (&OCCUPANCY, &self.layout),
                [
                    ("splat", splat.clone()),
                    ("records", records.clone()),
                    ("poses", poses.clone()),
                    ("covers", self.covers.bound()),
                    ("listed", self.listed.bound()),
                    ("finer_coverage", finer(0)),
                    ("finer_heights", finer(1)),
                ],
                &self.fillers,
            );
            if redraw {
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
            self.averages += u32::from(average);
        }
        Some(OccupancyFrame {
            draws: self.pending.clone(),
            means: self.averaging.clone(),
            workgroups: self.workgroups,
        })
    }

    /// Redraws the cells of the maps `frame` redraws into `encoder`, one
    /// map after the other, the finest first: its selection lists the
    /// footprints reaching them, then its cells clear and the listed
    /// footprints draw; then it takes the finer map's means where `frame`
    /// asks. Timed by `profiler` (`occupancy`).
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
        let changes = frame.draws.iter().zip(&frame.means).enumerate();
        for (map, (&redraw, &average)) in changes {
            let Some(group) = self.groups[map].get().filter(|_| redraw || average) else {
                continue;
            };
            let select = self.select_groups[map].get().filter(|_| redraw);
            if let Some(select) = select {
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
            if select.is_some() {
                pass.set_pipeline(&pipelines.clear);
                pass.draw(0..OCCUPANCY_QUAD, 0..CELLS * CELLS);
                pass.set_pipeline(&pipelines.splat);
                pass.draw_indirect(&self.draws[map], 0);
            }
            if average {
                pass.set_pipeline(&pipelines.means);
                pass.draw(0..OCCUPANCY_QUAD, 0..CELLS * CELLS);
            }
        }
        if let Some(span) = span {
            span.mark(encoder, true);
        }
    }
}

/// The absolute cells of `window`, rows along z.
fn cells(window: Window) -> impl Iterator<Item = [i32; 2]> {
    (0..SIDE).flat_map(move |row| {
        (0..SIDE).map(move |column| [window.origin[0] + column, window.origin[1] + row])
    })
}

/// The maps a frame redraws cells of and takes the finer one's means into,
/// and its selection's workgroups.
pub(crate) struct OccupancyFrame {
    draws: Vec<bool>,
    means: Vec<bool>,
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

    /// Times a map redrew cells, and took the finer one's means.
    fn stats(
        this: Option<&Self>,
        _frame: Option<&OccupancyFrame>,
        _snapshot: &engine_core::RenderSnapshot,
    ) -> engine_core::OccupancyStats {
        engine_core::OccupancyStats {
            redraws: this.map_or(0, |occupancy| occupancy.redraws),
            averages: this.map_or(0, |occupancy| occupancy.averages),
        }
    }
}

/// The coverage and heights targets, `resolution` texels a side with a
/// layer for each of `maps` maps, the views shading reads and each map's
/// layer of each target.
fn targets(
    device: &wgpu::Device,
    resolution: u32,
    maps: usize,
) -> (
    [wgpu::Texture; 2],
    [Versioned<wgpu::TextureView>; 2],
    [Vec<wgpu::TextureView>; 2],
) {
    let target = |label, format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: maps as u32,
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
    let layers = |target: &wgpu::Texture| {
        (0..maps as u32)
            .map(|map| {
                target.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: map,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect()
    };
    let views = [
        Versioned::new(whole(&targets[0])),
        Versioned::new(whole(&targets[1])),
    ];
    let layers = [layers(&targets[0]), layers(&targets[1])];
    (targets, views, layers)
}
