//! Foam's field (`meshes/water_foam_field.wgsl`): what turbulence makes on
//! terrains' water as it runs, simulated live on a grid of its own finer
//! than the water's cells, which water's shading reads as coverage and as
//! the slopes of its surface (`meshes/water_foam.wgsl`'s
//! `water_foam_held`). Each cell holds the white water covering it, the
//! foam the water carries there, the turbulence reaching its surface and
//! the surface's own disturbance from it, its height and how fast that
//! rises.
//!
//! Turbulence reaches the surface in bursts (`water_bursts.wgsl`, the rule
//! the live solve's turbulent pushes share): below its grid the simulation
//! knows how fast the water's eddies turn, about as large as it is deep, at
//! least a cell, and no larger than two of the water's cells, past which its
//! own flow carries them; each eddy reaches the surface over its size
//! squared of it once as it turns over (surface renewal), so bursts are
//! born at random in space and time about that rate, each drawn from a hash
//! of the world point it is born at and the simulation's step index, the
//! same on every host. A burst brings the surface turbulence over a patch
//! as wide as its eddy, dying as its eddies turn over, and raises a boil
//! there as high as its eddies' velocity head lets it. The turbulence
//! reaching the surface lays fresh white water where the water breaks, by
//! the cook's rule (`engine_terrain`'s `breaking`), which drains as its
//! bubbles rise out over its depth, and churns foam where the water churns,
//! by the cook's rule for it (`water_sim.wgsl`'s `sim_churn`: where the
//! water loses energy, plunges or jumps, along its eddy lines and where it
//! converges), which fades over the cook's foam life; on average as the
//! cook lays them, so the field's mean is the settled foam past its reach,
//! but in patches, open water between them. The boils spread as gravity
//! waves on the field's grid and die by the water's friction and viscosity.
//! All of it rides the current the water's surface runs at, the water's
//! velocity turned outward in bends and dragged by the wind, the velocity
//! read as water's shading reads it (`water_pixel`: the live solve's where
//! it runs, the settled flow elsewhere), by semi-Lagrangian transport, each
//! cell taking the state at the point the current carried it from, by
//! Catmull and Rom's spline kept within its nearest cells; the white water,
//! foam and turbulence by how much the surface there shrank or grew, so
//! they gather where the current converges and pile against what stops it,
//! dry ground and what stands up through the water. Patches stretch into
//! streaks, merge, drain and break up by that transport and their decay
//! alone, and reflections and glints travel downstream with the boils.
//!
//! Its cells split each of the water's cells ([`Water::simulation_cell`])
//! [`FOAM_SPLIT`] ways along each side, 25 cm for the cook's meter cells,
//! what two pixels resolve where it runs farthest: the water's velocity,
//! smooth over its own cells, draws foam into lines a few of the field's
//! cells across. How many cells it holds follows from the screen alone,
//! whatever their size: it shows out to where its cells span
//! [`FOAM_PIXELS`] pixels seen from the detail eye, past which water shows
//! its own foam, the field's mean, on a surface it draws nothing on, handing
//! over as a pyramid's level does, so the cells along a side are the pixels
//! a radian holds over FOAM_PIXELS, twice. Seen through a view 60° tall, a
//! pixel at 1920×1080 spans 1.07 mrad, so the field shows out to 117 m and
//! holds 960 cells along a side; at 2560×1440 a pixel spans 0.80 mrad, out
//! to 156 m over 1280 cells. A view narrower or a display larger asks for
//! more cells; it holds no more than the display has pixels, nor than one
//! of the device's storage bindings holds, and where that cuts it, its cells
//! span more pixels at its reach.
//!
//! Its grid is anchored in the world: a world cell's texel is its remainder
//! over the cells along a side, so its window, centered on the detail eye,
//! shifts by whole cells as the eye moves and every cell it keeps keeps its
//! state. Cells it newly covers, and what is carried in across its rim,
//! start as the settled foam there on a still surface.
//!
//! Each frame runs a fixed number of dispatches, none read back: the water
//! read at every other corner of its cells (`foam_gather`), what carries and
//! churns its foam there (`foam_currents`), the bursts born at the sites
//! those corners stand for (`foam_bursts`), the step of every cell
//! (`foam_step`), its textures' finest level (`foam_look`) and their coarser
//! levels (`foam_down`, one dispatch each), eleven in all. It costs 57 MB at
//! 1920×1080 (its state, twice over, 29 MB; its lattice 8 MB; its two
//! textures and their levels 20 MB) and 101 MB at 2560×1440 (52, 13 and 35
//! MB); its step reads some 750 bytes a cell, each from the few around it,
//! its other passes under a hundred, and dry cells exit early.

use std::cell::Cell;

use engine_core::glam::{Vec2, Vec3};
use engine_core::math::{clamp, max};
use engine_core::terrain_data::shallow_water::step_seconds;
use engine_core::{RenderTerrain, Water};

use crate::bindings::{EMPTY, WATER_FOAM};
use crate::gpu_layouts::constants::{FOAM_GROUP, MATERIAL_ANISOTROPY};
use crate::gpu_layouts::water::FoamFrame;
use crate::pipelines::GroupLayout;
use crate::shaders::entry::meshes;
use crate::targets::FrameSize;
use crate::terrain::TerrainWaterDraws;
use crate::water_sim::LONGEST_FRAME;

/// The field's cells along a side of one of the water's: four, so the
/// lines the water's velocity draws foam into, narrower than its cells, are
/// a few of the field's across.
const FOAM_SPLIT: u32 = 4;
/// The field's cells across the largest eddies its bursts stand for: two of
/// the water's cells, the shortest wave its cells resolve (Nyquist); its
/// own flow carries larger eddies.
const EDDY_CELLS: u32 = 2 * FOAM_SPLIT;
/// The levels of the bursts' sites (`foam_level`): level L's sites span
/// 2^(L+1) cells and hold eddies up to twice that, so as many levels as
/// hold eddies EDDY_CELLS across.
const BURST_LEVELS: u32 = EDDY_CELLS.div_ceil(4).next_power_of_two().trailing_zeros() + 1;
/// The pixels the field's cells span, seen from the detail eye, where it
/// hands over to the water's own foam (`water_foam_held`): two, the finest
/// a line drawn on screen keeps without aliasing (Nyquist).
const FOAM_PIXELS: f32 = 2.0;
/// The levels its textures hold coarser than its cells: down to texels 64
/// cells across, past the most its texels' anisotropic filtering takes
/// along the view at its reach; its cells along a side are a whole number
/// of the coarsest's, so every level wraps as its cells do.
const FOAM_LEVELS: u32 = 6;
/// The cells its window keeps past its reach on each side: as many as a
/// pixel's footprint at the reach spans along the view, at most
/// MATERIAL_ANISOTROPY times its width, a cell over FOAM_PIXELS, which its
/// texels' anisotropic filtering reads, and one more, as the window shifts
/// by whole cells.
const FOAM_MARGIN: u32 = MATERIAL_ANISOTROPY as u32 / FOAM_PIXELS as u32 + 1;
/// The lattice points its lattice keeps past its window's cells on each
/// side: the frame's current carries foam from at most the fastest water
/// over the longest frame (`shallow_water::FASTEST` over 30 frames a
/// second, two thirds of a meter), and the spline reading it there takes
/// the two cells past that point.
const LATTICE_MARGIN: u32 = 3;
/// Bytes a cell takes of each of its state's two buffers: two floats, or
/// two words of half floats, twice over.
const STATE_BYTES: u64 = 16;
/// Bytes a texel of each of its textures takes: four half floats.
const TEXEL_BYTES: u64 = 8;

/// Foam's field: its passes' group layout and pipelines, the empty group
/// standing first and the sampler water's shading reads its textures
/// through; the grid it runs over, when the last frame was and the
/// simulation's step it ended in, and a count advancing whenever its
/// textures change.
pub(crate) struct FoamField {
    layout: GroupLayout,
    empty_layout: GroupLayout,
    empty: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    pipelines: Option<FoamPipelines>,
    grid: Option<FoamGrid>,
    last: Option<f64>,
    step: Option<u32>,
    generation: u64,
}

/// The pipelines of the field's passes.
struct FoamPipelines {
    gather: wgpu::ComputePipeline,
    currents: wgpu::ComputePipeline,
    bursts: wgpu::ComputePipeline,
    step: wgpu::ComputePipeline,
    look: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
}

/// The grid the field runs over: its cells along a side and the meters
/// across each, its lattice's points along a side and the pixels its cells
/// span where it hands over; its frame's uniform, its lattice, its state
/// twice over in two buffers, and the textures water's shading reads (its
/// white water, foam and disturbance's slopes; and those slopes' squares),
/// with their whole views, and its groups (the first for their finest
/// level, which the lattice's, the bursts', the step's and the look's
/// passes take, then each coarser level's); the window its state holds
/// (none until a step ran) and the half holding it; this frame's window and
/// the uniform offset of the terrain whose water it runs over; and whether
/// the frame's encoding recorded its step.
struct FoamGrid {
    cells: u32,
    cell: f32,
    points: u32,
    pixels: f32,
    frame: wgpu::Buffer,
    lattice: wgpu::Buffer,
    state: wgpu::Buffer,
    waves: wgpu::Buffer,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    spread: wgpu::Texture,
    spread_view: wgpu::TextureView,
    groups: Vec<wgpu::BindGroup>,
    held: Option<[i64; 2]>,
    read: u32,
    window: [i64; 2],
    terrain: u32,
    stepped: Cell<bool>,
}

/// How the frame's camera sees terrains' water, which sizes the field: the
/// detail eye every handover measures from, the angle a display pixel
/// spans at the view's center (`crate::gpu_data::pixel_angle`) and the
/// display's size.
#[derive(Clone, Copy)]
pub(crate) struct FoamView {
    pub(crate) detail: Vec3,
    pub(crate) pixel_angle: f32,
    pub(crate) display: FrameSize,
}

impl FoamField {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let empty_layout = crate::pipelines::layout(device, &EMPTY);
        let empty = crate::pipelines::group(
            device,
            "engine/water-foam-empty",
            &EMPTY,
            &empty_layout,
            &[],
        );
        Self {
            layout: crate::pipelines::layout(device, &WATER_FOAM),
            empty_layout,
            empty,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/water-foam-sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                // Water seen at grazing angles keeps its foam sharp across
                // the view.
                anisotropy_clamp: MATERIAL_ANISOTROPY,
                ..Default::default()
            }),
            pipelines: None,
            grid: None,
            last: None,
            step: None,
            generation: 0,
        }
    }

    /// Prepares the field for this frame over the water of the terrain
    /// whose uniform stands at `terrain`, seen as `seen` says, the water's
    /// cells `settings` gives, in the scene's wind `wind` at simulation
    /// time `seconds`: what the frame before stepped taken as its state; a
    /// grid anew where it holds too few cells or twice as many as it needs,
    /// or cells of another size; its window around the detail eye; and its
    /// frame, with the simulation's steps the frame begins and ends in, each
    /// as long as a step of the water's cells (`step_seconds`), which its
    /// bursts are drawn by. Drops it where no terrain's water draws.
    pub(crate) fn prepare(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        terrain: Option<u32>,
        seen: FoamView,
        (settings, wind, seconds): (&Water, [f32; 2], f64),
    ) {
        if let Some(grid) = self.grid.as_mut()
            && grid.stepped.replace(false)
        {
            grid.held = Some(grid.window);
            grid.read ^= 1;
        }
        let Some(terrain) = terrain else {
            if self.grid.take().is_some() {
                self.generation += 1;
            }
            self.last = None;
            self.step = None;
            return;
        };
        let cell = settings.simulation_cell / FOAM_SPLIT as f32;
        let cells = wanted(device, seen);
        let keeps = self
            .grid
            .as_ref()
            .is_some_and(|grid| grid.cell == cell && grid.cells >= cells && grid.cells < 2 * cells);
        if !keeps {
            self.grid = Some(FoamGrid::new(device, &self.layout, (cells, cell)));
            self.generation += 1;
        }
        let Some(grid) = self.grid.as_mut() else {
            return;
        };
        // Its cells span FOAM_PIXELS where it hands over, or more where it
        // holds fewer than its reach asks.
        let reach = (grid.cells / 2 - FOAM_MARGIN) as f32;
        grid.pixels = max(FOAM_PIXELS, 1.0 / (reach * max(seen.pixel_angle, 1.0e-6)));
        let half = i64::from(grid.cells / 2);
        let window = [seen.detail.x, seen.detail.z]
            .map(|at| (f64::from(at) / f64::from(cell)).floor() as i64 - half);
        grid.window = window;
        grid.terrain = terrain;
        let elapsed = self.last.map_or(0.0, |last| {
            clamp((seconds - last) as f32, 0.0, LONGEST_FRAME as f32)
        });
        self.last = Some(seconds);
        // The simulation's steps, the frame's last and the one the frame
        // before ended in, as an index wrapping past u32::MAX's; and the
        // fewest a slot of bursts spans, the power of two holding the
        // longest frame's.
        let tick = step_seconds(settings.simulation_cell);
        let step = (seconds / f64::from(tick)).floor() as i64 as u32;
        let begun = self.step.unwrap_or(step);
        self.step = Some(step);
        let least = ((LONGEST_FRAME / f64::from(tick)).ceil() as u32).next_power_of_two();
        let held = grid.held.unwrap_or(window);
        let lattice = window.map(|first| first.div_euclid(2) - i64::from(LATTICE_MARGIN));
        let frame = FoamFrame {
            window: [
                window[0] as i32,
                window[1] as i32,
                held[0] as i32,
                held[1] as i32,
            ],
            lattice: [
                lattice[0] as i32,
                lattice[1] as i32,
                grid.points as i32,
                grid.cells as i32,
            ],
            step: [cell, elapsed, wind[0], wind[1]],
            flags: [grid.read, u32::from(grid.held.is_none()), begun, step],
            sites: [least, BURST_LEVELS, 0, 0],
            eddies: [tick, grid.largest(), 0.0, 0.0],
        };
        queue.write_buffer(&grid.frame, 0, bytemuck::bytes_of(&frame));
    }

    /// Creates its passes' pipelines from the mesh module `shader`, after
    /// the water's group layout `water`, unless they exist; none until a
    /// grid runs.
    pub(crate) fn ensure_pipelines(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        water: &GroupLayout,
    ) {
        if self.grid.is_none() || self.pipelines.is_some() {
            return;
        }
        let layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::WATER_FOAM_PIPELINES,
            &[&self.empty_layout, water, &self.layout],
        );
        let entry = |entry| crate::pipelines::compute_entry(device, &layout, shader, entry);
        self.pipelines = Some(FoamPipelines {
            gather: entry(meshes::foam_gather),
            currents: entry(meshes::foam_currents),
            bursts: entry(meshes::foam_bursts),
            step: entry(meshes::foam_step),
            look: entry(meshes::foam_look),
            down: entry(meshes::foam_down),
        });
    }

    /// Records the field's step over `water`, the water's group, timed by
    /// `profiler` (`water_foam`): its lattice's water, what carries its
    /// foam and the bursts born there, every cell's step, and its textures'
    /// finest level and their coarser levels.
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        water: &wgpu::BindGroup,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let (Some(pipelines), Some(grid)) = (&self.pipelines, &self.grid) else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/water-foam"),
            timestamp_writes: profiler.span("water_foam").map(|span| span.compute()),
        });
        pass.set_bind_group(0, &self.empty, &[]);
        // The terrain's uniform at its offset, the views' and water trees'
        // at their first, which the passes read none of.
        pass.set_bind_group(1, water, &[0, grid.terrain, 0]);
        pass.set_bind_group(2, &grid.groups[0], &[]);
        let points = grid.points.div_ceil(FOAM_GROUP);
        for pipeline in [&pipelines.gather, &pipelines.currents, &pipelines.bursts] {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(points, points, 1);
        }
        let cells = grid.cells.div_ceil(FOAM_GROUP);
        for pipeline in [&pipelines.step, &pipelines.look] {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(cells, cells, 1);
        }
        pass.set_pipeline(&pipelines.down);
        for (level, group) in grid.groups.iter().enumerate().skip(1) {
            pass.set_bind_group(2, group, &[]);
            let side = (grid.cells >> level).div_ceil(FOAM_GROUP);
            pass.dispatch_workgroups(side, side, 1);
        }
        grid.stepped.set(true);
    }

    /// The field's texture of its white water, foam and disturbance's
    /// slopes as water's shading reads it, with its levels; none while none
    /// runs.
    pub(crate) fn view(&self) -> Option<&wgpu::TextureView> {
        self.grid.as_ref().map(|grid| &grid.view)
    }

    /// The field's texture of its disturbance's slopes' squares as water's
    /// shading reads it, with its levels; none while none runs.
    pub(crate) fn spread_view(&self) -> Option<&wgpu::TextureView> {
        self.grid.as_ref().map(|grid| &grid.spread_view)
    }

    /// The sampler water's shading reads the field's textures through: it
    /// repeats them, as their texels hold their world cells' remainders.
    pub(crate) fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// What water's uniform holds of the field for terrains' water
    /// (`water.foam`): the meters across its cells (none while none runs),
    /// the meters its textures span, the pixels its cells span where it
    /// hands over and the meters across the largest eddies its bursts stand
    /// for.
    pub(crate) fn shading(&self) -> [f32; 4] {
        self.grid.as_ref().map_or([0.0; 4], |grid| {
            [
                grid.cell,
                grid.cell * grid.cells as f32,
                grid.pixels,
                grid.largest(),
            ]
        })
    }

    /// A count advancing whenever the field's textures change.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Cells along a side of its grid; none while none runs.
    pub(crate) fn cells(&self) -> u32 {
        self.grid.as_ref().map_or(0, |grid| grid.cells)
    }

    /// Bytes of its grid's buffers and textures.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let texels = |texture: &wgpu::Texture| -> u64 {
            (0..texture.mip_level_count())
                .map(|level| u64::from(texture.width() >> level).pow(2))
                .sum()
        };
        self.grid.as_ref().map_or(0, |grid| {
            grid.frame.size()
                + grid.lattice.size()
                + grid.state.size()
                + grid.waves.size()
                + (texels(&grid.texture) + texels(&grid.spread)) * TEXEL_BYTES
        })
    }
}

impl FoamGrid {
    /// A grid of `cells` cells along a side, `cell` meters across each,
    /// with its passes' groups of `layout`; its state holds none yet.
    fn new(device: &wgpu::Device, layout: &GroupLayout, (cells, cell): (u32, f32)) -> Self {
        let points = cells / 2 + 2 * LATTICE_MARGIN + 1;
        let frame = crate::buffers::uniform(
            device,
            "engine/water-foam-frame",
            size_of::<FoamFrame>() as u64,
        );
        let lattice = crate::buffers::storage(
            device,
            "engine/water-foam-lattice",
            2 * u64::from(points) * u64::from(points) * 16,
            wgpu::BufferUsages::empty(),
        );
        let state_buffer = |label| {
            crate::buffers::storage(
                device,
                label,
                u64::from(cells) * u64::from(cells) * STATE_BYTES,
                wgpu::BufferUsages::empty(),
            )
        };
        let state = state_buffer("engine/water-foam-state");
        let waves = state_buffer("engine/water-foam-waves");
        let textures = |label| {
            let texture = crate::targets::texture_2d(
                device,
                label,
                (cells, cells),
                1,
                FOAM_LEVELS + 1,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let levels: Vec<wgpu::TextureView> = (0..=FOAM_LEVELS)
                .map(|level| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("engine/water-foam-level"),
                        base_mip_level: level,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            (texture, levels)
        };
        let (texture, levels) = textures("engine/water-foam-field");
        let (spread, spread_levels) = textures("engine/water-foam-spread");
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let spread_view = spread.create_view(&wgpu::TextureViewDescriptor::default());
        // The finest level's group, which writes it, reads the next coarser
        // in the finer one's place; each coarser level's reads the level
        // finer than it.
        let groups = (0..=FOAM_LEVELS as usize)
            .map(|level| {
                let finer = if level == 0 { 1 } else { level - 1 };
                crate::pipelines::group(
                    device,
                    "engine/water-foam-group",
                    &WATER_FOAM,
                    layout,
                    &[
                        ("foam_frame", frame.as_entire_binding()),
                        crate::buffers::entry("foam_lattice", &lattice),
                        crate::buffers::entry("foam_state", &state),
                        crate::buffers::entry("foam_waves", &waves),
                        ("foam_source", crate::pipelines::view(&levels[finer])),
                        ("foam_target", crate::pipelines::view(&levels[level])),
                        (
                            "foam_spread_source",
                            crate::pipelines::view(&spread_levels[finer]),
                        ),
                        (
                            "foam_spread_target",
                            crate::pipelines::view(&spread_levels[level]),
                        ),
                    ],
                )
            })
            .collect();
        Self {
            cells,
            cell,
            points,
            pixels: FOAM_PIXELS,
            frame,
            lattice,
            state,
            waves,
            texture,
            view,
            spread,
            spread_view,
            groups,
            held: None,
            read: 0,
            window: [0; 2],
            terrain: 0,
            stepped: Cell::new(false),
        }
    }

    /// The meters across the largest eddies its bursts stand for:
    /// EDDY_CELLS of its cells.
    fn largest(&self) -> f32 {
        EDDY_CELLS as f32 * self.cell
    }
}

/// The cells along a side the field's grid takes as `seen` sees it: as many
/// as reach, from the detail eye, out to where its cells span FOAM_PIXELS
/// pixels, with FOAM_MARGIN more each side, whatever their size, in whole
/// texels of its textures' coarsest level; no more than the display has
/// pixels, nor than one of the device's storage bindings holds of its
/// state, nor than a side of its textures holds.
fn wanted(device: &wgpu::Device, seen: FoamView) -> u32 {
    let block = 1 << FOAM_LEVELS;
    let reach = 1.0 / (FOAM_PIXELS * max(seen.pixel_angle, 1.0e-6));
    let asked = (2.0 * reach).ceil() as u32 + 2 * FOAM_MARGIN;
    let limits = device.limits();
    let binding = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let held = (binding as f64 / STATE_BYTES as f64).sqrt() as u32;
    let shown = (f64::from(seen.display.width) * f64::from(seen.display.height)).sqrt() as u32;
    let most = held.min(shown).min(limits.max_texture_dimension_2d) / block * block;
    asked.next_multiple_of(block).min(most).max(block)
}

/// The uniform offset of the terrain whose water the field runs over: of
/// those drawing tiles' water (`tiles`, `terrains` the frame's), the one
/// whose square holds the detail eye `eye`, a terrain's own before its
/// surroundings; none where none draws.
pub(crate) fn foam_terrain(
    tiles: &TerrainWaterDraws<'_>,
    terrains: &[RenderTerrain],
    eye: Vec3,
) -> Option<u32> {
    let rank = |id: u64| {
        terrains
            .iter()
            .find(|terrain| terrain.id == id)
            .map_or((false, false), |terrain| {
                let local = Vec2::new(eye.x - terrain.origin.x, eye.z - terrain.origin.z);
                let extent = terrain.pyramid.extent();
                let holds = local.cmpge(Vec2::ZERO).all() && local.cmple(extent).all();
                (holds, !terrain.surroundings)
            })
    };
    tiles
        .terrains
        .iter()
        .max_by_key(|terrain| rank(terrain.0))
        .map(|terrain| terrain.1)
}
