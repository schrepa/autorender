//! Water's particles (`meshes/water_particles.wgsl`): what turbulence makes
//! on terrains' water as it runs, its foam and the small waves on its
//! surface, as particles the live flow carries around the detail eye,
//! splatted each frame into textures water's shading reads as foam's
//! coverage and as the slopes of its surface (`meshes/water_foam.wgsl`'s
//! `water_splatted`).
//!
//! Foam particles hold the white water and the foam the water carries:
//! born where the water breaks and churns, by the cook's rules
//! (`engine_terrain`'s `breaking`, `water_sim.wgsl`'s `sim_churn`), in
//! turbulence's bursts (`water_bursts.wgsl`, the rule the live solve's
//! turbulent pushes share), each drawn from the world point it is born at
//! and the simulation's step index, the same on every host; carried by the
//! current the water's surface runs at, the water's velocity turned outward
//! in bends and dragged by the wind, read as water's shading reads it
//! (`water_pixel`: the live solve's where it runs, the settled flow
//! elsewhere), smooth across the lattice it is read at and integrated by
//! the midpoint rule, so its shear draws them out into streaks and they
//! gather where it converges and against what stops it; and draining as
//! white water's bubbles rise out over its depth and as foam fades over the
//! cook's foam life. On average they lay what the cook lays, so their mean
//! is the settled foam past their reach. Wave particles are segments of
//! wavefronts (Yuksel, House and Keyser 2007), raised by the bursts' boils
//! and where the water runs against what stands in it, each running at its
//! waves' group speed plus the current, spreading and splitting in three as
//! it spreads, dying by the water's viscosity and eddy viscosity; they stand
//! for the eddies finer than the cells of the water they ride, whose own
//! flow carries larger ones.
//!
//! Each frame the particles splat into a window of texels anchored in the
//! world around the detail eye, by fixed-point atomic sums, which any order
//! of the particles adds alike: foam particles by a quadratic B-spline over
//! the 3 x 3 texels around them, wave particles by their raised cosines'
//! slopes. Its texels split each of the water's settled cells
//! ([`Water::simulation_cell`]) [`TEXEL_SPLIT`] ways along each side, 6.25 cm
//! for the cook's meter cells. How many it holds follows from the screen: it
//! shows out to where its texels span [`HANDOVER_PIXELS`] seen from the
//! detail eye, past which water shows its own foam, the particles' mean, on
//! a surface they draw nothing on, handing over as a pyramid's level does,
//! so its texels along a side are the pixels a radian holds over
//! HANDOVER_PIXELS, twice, with [`WINDOW_MARGIN`] more each side; no more than
//! the display has pixels, nor than one of the device's storage bindings
//! holds, and where that cuts it, its texels span more pixels at its reach.
//! Seen through a view 60° tall, a pixel at 1920×1080 spans 1.07 mrad, which
//! asks for 1,905 texels along a side, and the display's 1,440 hold 1,408:
//! a window 88 m across, the particles showing out to 43 m, their texels
//! spanning 1.36 pixels there; at 2560×1440 a pixel spans 0.80 mrad, which
//! asks for 2,529, and the display holds 1,920: 120 m across, out to 59 m,
//! 1.32 pixels. The pools hold a foam particle for every [`FOAM_TEXELS`]
//! texels of the window and a wave particle for every [`WAVE_TEXELS`]:
//! 495,616 and 123,904 at 1920×1080, 921,600 and 230,400 at 2560×1440.
//!
//! Its window shifts by whole texels as the detail eye moves, a world
//! texel's texel its remainder over the texels along a side, and every
//! particle in it runs on; those it leaves go. Sites it newly covers, and
//! the foam the current carries in across its rim, take particles laying
//! the settled foam there.
//!
//! Each frame runs a fixed number of dispatches, in three passes with the
//! dispatches over the particles, which the GPU counts, copied between them
//! (`particles_begin`, then the lattice's water `particles_gather` and what
//! carries it `particles_currents`; each particle's step `particles_age`,
//! the sites' births `particles_births` and the scan over both
//! `particles_sum` and `particles_scan`; their laying `particles_place`,
//! the splat `particles_splat`, the textures' finest level `particles_look`
//! and their coarser levels `particles_down`), sixteen in all; the pools'
//! tallies alone are read back, a few frames late. It holds 98 MB at
//! 1920×1080 (its splat's sums 32 MB, its textures and their levels 42 MB,
//! its pools twice over 20 MB, its counts and lattice 4 MB) and 181 MB at
//! 2560×1440 (59, 79, 37 and 7 MB). Its passes over the window's texels and
//! the lattice's points cost as many as the window holds, about 2.0 and 3.7
//! million texels; those over the particles as many as live.

use std::cell::Cell;
use std::rc::Rc;

use engine_core::glam::{Vec2, Vec3};
use engine_core::math::{clamp, max};
use engine_core::terrain_data::shallow_water::step_seconds;
use engine_core::{RenderTerrain, Water};

use crate::bindings::{EMPTY, WATER_PARTICLES};
use crate::buffers::ReadbackRing;
use crate::gpu_layouts::constants::{
    MATERIAL_ANISOTROPY, PARTICLE_ARGS, PARTICLE_FOAM_BEFORE, PARTICLE_FOAM_BORN,
    PARTICLE_FOAM_DROPPED, PARTICLE_FOAM_LIVE, PARTICLE_GROUP, PARTICLE_ITEMS, PARTICLE_LATTICE,
    PARTICLE_SIDE, PARTICLE_WAVES_BORN, PARTICLE_WAVES_DROPPED, PARTICLE_WAVES_LIVE,
};
use crate::gpu_layouts::water::ParticleFrame;
use crate::pipelines::GroupLayout;
use crate::profiling::GpuProfiler;
use crate::shaders::entry::meshes;
use crate::targets::FrameSize;
use crate::terrain::TerrainWaterDraws;
use crate::water_sim::LONGEST_FRAME;

/// The window's texels along a side of one of the water's settled cells:
/// sixteen, 6.25 cm for the cook's meter cells, eight along each of the
/// half-meter cells those run live in, so foam particles' kernels, three
/// texels across, draw foam's flecks and lace and the lines the live
/// water's velocity draws it into, narrower than its cells.
const TEXEL_SPLIT: u32 = 16;
/// The pixels the window's texels span, seen from the detail eye, where
/// water hands over from its particles to its own foam (`water_splatted`):
/// one, so no texel the particles draw is finer than a pixel spans.
const HANDOVER_PIXELS: f32 = 1.0;
/// The levels its textures hold coarser than its texels: down to texels 64
/// across, past the most anisotropic filtering takes along the view at its
/// reach; its texels along a side are a whole number of the coarsest's, so
/// every level wraps as its texels do.
const TEXTURE_LEVELS: u32 = 6;
/// The texels the window keeps past its reach on each side: as many as a
/// pixel's footprint at the reach spans along the view, at most
/// MATERIAL_ANISOTROPY times its width, a texel over HANDOVER_PIXELS, which
/// anisotropic filtering reads, and one more, as the window shifts by whole
/// texels.
const WINDOW_MARGIN: u32 = MATERIAL_ANISOTROPY as u32 / HANDOVER_PIXELS as u32 + 1;
/// The points the lattice keeps past the window's on each side: the
/// quadratic B-spline reading the current at a particle takes the point
/// past its nearest, and the frame's current carries one past the window by
/// no more than the fastest water over half the longest frame
/// (`shallow_water::FASTEST` over 60 frames a second, a third of a meter),
/// under a point more.
const LATTICE_MARGIN: u32 = 2;
/// Texels of the window each particle the pools hold stands for: four of
/// the foam pool's, so it lays nearly half a layer over all of the window,
/// more than rapids cover of it; and sixteen of the wave pool's, its
/// particles living a second or so where foam lives tens.
const FOAM_TEXELS: u32 = 4;
const WAVE_TEXELS: u32 = 16;
/// Bytes a texel takes of the splat's sums: four words, the layers of white
/// water and of foam laid over it and how fast the waves' surface falls
/// along x and z.
const SPLAT_BYTES: u64 = 16;
/// Bytes a particle takes of each half of the pools, a lattice point of the
/// lattice (its water, then what carries the foam there) and a texel of each
/// of the textures (four half floats).
const PARTICLE_BYTES: u64 = 16;
const POINT_BYTES: u64 = 32;
const TEXEL_BYTES: u64 = 8;
/// The words of the pools' counts read back: those before the counts of
/// the particles the pools held before the frame.
const TALLIES: usize = PARTICLE_FOAM_BEFORE as usize;
/// Bytes of the dispatches over the particles and over the scan's items,
/// three words each.
const ARGS_BYTES: u64 = 24;

/// Water's particles: their passes' group layout and pipelines, the empty
/// group standing first and the sampler water's shading reads their textures
/// through; the window they run over, when the last frame was and the
/// simulation's step it ended in, and a count advancing whenever their
/// textures change; and the pools' tallies' read-backs and what they last
/// read (`PARTICLE_FOAM_LIVE` on).
pub(crate) struct WaterParticles {
    layout: GroupLayout,
    empty_layout: GroupLayout,
    empty: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    pipelines: Option<ParticlePipelines>,
    grid: Option<ParticleGrid>,
    last: Option<f64>,
    step: Option<u32>,
    generation: u64,
    readbacks: ReadbackRing<()>,
    tallies: Rc<Cell<[u32; TALLIES]>>,
}

/// The pipelines of the particles' passes.
struct ParticlePipelines {
    begin: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    currents: wgpu::ComputePipeline,
    age: wgpu::ComputePipeline,
    births: wgpu::ComputePipeline,
    sum: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    place: wgpu::ComputePipeline,
    splat: wgpu::ComputePipeline,
    look: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
}

/// The window the particles run over: its texels along a side and the
/// meters across each, its lattice's points along a side, the most foam and
/// wave particles its pools hold, the pixels its texels span where they hand
/// over and the meters across the cells of the water they ride; its frame's
/// uniform, its lattice, its pools twice over, its counts and the
/// dispatches copied from them, its splat's sums, and the textures water's
/// shading reads (its white water's and foam's coverage and the waves'
/// slopes; and those slopes' squares), with their whole views, and its
/// groups (the first for their finest level, which every pass but the
/// coarser levels' takes, then each coarser level's); the window its pools'
/// particles ran in (none until a frame ran) and the half holding them;
/// this frame's window and the uniform offset of the terrain whose water it
/// runs over; and whether the frame's encoding recorded its passes.
struct ParticleGrid {
    texels: u32,
    texel: f32,
    points: u32,
    pools: [u32; 2],
    pixels: f32,
    ridden: f32,
    frame: wgpu::Buffer,
    lattice: wgpu::Buffer,
    pool: wgpu::Buffer,
    counts: wgpu::Buffer,
    args: wgpu::Buffer,
    splat: wgpu::Buffer,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    spread: wgpu::Texture,
    spread_view: wgpu::TextureView,
    groups: Vec<wgpu::BindGroup>,
    held: Option<[i64; 2]>,
    read: u32,
    window: [i64; 2],
    terrain: u32,
    encoded: Cell<bool>,
}

/// How the frame's camera sees terrains' water, which sizes the window: the
/// detail eye every handover measures from, the angle a display pixel
/// spans at the view's center (`crate::gpu_data::pixel_angle`) and the
/// display's size.
#[derive(Clone, Copy)]
pub(crate) struct ParticleView {
    pub(crate) detail: Vec3,
    pub(crate) pixel_angle: f32,
    pub(crate) display: FrameSize,
}

/// What water's particles report of a frame: the window's texels along a
/// side (none while none run); the foam and the wave particles the pools
/// hold, those born in a frame and those the pools had no room for, foam
/// and waves together, as last read back.
#[derive(Clone, Copy, Default)]
pub(crate) struct ParticleStats {
    pub(crate) texels: u32,
    pub(crate) foam: u32,
    pub(crate) waves: u32,
    pub(crate) born: u32,
    pub(crate) dropped: u32,
}

impl WaterParticles {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let empty_layout = crate::pipelines::layout(device, &EMPTY);
        let empty = crate::pipelines::group(
            device,
            "engine/water-particles-empty",
            &EMPTY,
            &empty_layout,
            &[],
        );
        Self {
            layout: crate::pipelines::layout(device, &WATER_PARTICLES),
            empty_layout,
            empty,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/water-particles-sampler"),
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
            readbacks: ReadbackRing::new("engine/water-particles-tallies"),
            tallies: Rc::new(Cell::new([0; TALLIES])),
        }
    }

    /// Prepares the particles for this frame over the water of the terrain
    /// whose uniform stands at `terrain`, seen as `seen` says, the water's
    /// settled cells `settings` gives, riding water whose cells are `ridden`
    /// meters across (the live water's where it runs), in the scene's wind
    /// `wind` at simulation time `seconds`: the half the frame before laid
    /// its particles in taken as the one they run from, and its window as
    /// theirs; a window anew where it holds too few texels or twice as many
    /// as it needs, or texels of another size; its window around the detail
    /// eye; and its frame, with the simulation's steps the frame begins and
    /// ends in, each as long as a step of the water's settled cells
    /// (`step_seconds`), which its events are drawn by. Drops it where no
    /// terrain's water draws.
    pub(crate) fn prepare(
        &mut self,
        (device, queue): (&wgpu::Device, &wgpu::Queue),
        terrain: Option<u32>,
        seen: ParticleView,
        (settings, ridden, wind, seconds): (&Water, f32, [f32; 2], f64),
    ) {
        if let Some(grid) = self.grid.as_mut()
            && grid.encoded.replace(false)
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
        let texel = settings.simulation_cell / TEXEL_SPLIT as f32;
        let asked = wanted(device, seen);
        let keeps = self.grid.as_ref().is_some_and(|grid| {
            grid.texel == texel && grid.texels >= asked && grid.texels < 2 * asked
        });
        if !keeps {
            self.grid = Some(ParticleGrid::new(device, &self.layout, (asked, texel)));
            self.generation += 1;
        }
        let Some(grid) = self.grid.as_mut() else {
            return;
        };
        // Its texels span HANDOVER_PIXELS where it hands over, or more where
        // it holds fewer than its reach asks.
        let reach = (grid.texels / 2 - WINDOW_MARGIN) as f32;
        grid.pixels = max(
            HANDOVER_PIXELS,
            1.0 / (reach * max(seen.pixel_angle, 1.0e-6)),
        );
        grid.ridden = ridden;
        let half = i64::from(grid.texels / 2);
        let window = [seen.detail.x, seen.detail.z]
            .map(|at| (f64::from(at) / f64::from(texel)).floor() as i64 - half);
        grid.window = window;
        grid.terrain = terrain;
        let elapsed = self.last.map_or(0.0, |last| {
            clamp((seconds - last) as f32, 0.0, LONGEST_FRAME as f32)
        });
        self.last = Some(seconds);
        // The simulation's steps, the frame's last and the one the frame
        // before ended in, as an index wrapping past u32::MAX's; and the
        // fewest a slot of events spans, the power of two holding the
        // longest frame's.
        let tick = step_seconds(settings.simulation_cell);
        let step = (seconds / f64::from(tick)).floor() as i64 as u32;
        let begun = self.step.unwrap_or(step);
        self.step = Some(step);
        let least = ((LONGEST_FRAME / f64::from(tick)).ceil() as u32).next_power_of_two();
        let held = grid.held.unwrap_or(window);
        let lattice = window
            .map(|first| first.div_euclid(i64::from(PARTICLE_LATTICE)) - i64::from(LATTICE_MARGIN));
        let frame = ParticleFrame {
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
                grid.texels as i32,
            ],
            step: [texel, elapsed, wind[0], wind[1]],
            flags: [u32::from(grid.held.is_none()), begun, step, least],
            pool: [
                grid.pools[0],
                grid.pools[1],
                grid.read,
                grid.points * grid.points,
            ],
            eddies: [tick, ridden, 0.0, 0.0],
        };
        queue.write_buffer(&grid.frame, 0, bytemuck::bytes_of(&frame));
    }

    /// Creates their passes' pipelines from the mesh module `shader`, after
    /// the water's group layout `water`, unless they exist; none until a
    /// window runs.
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
            &crate::pipeline_layouts::WATER_PARTICLE_PIPELINES,
            &[&self.empty_layout, water, &self.layout],
        );
        let entry = |entry| crate::pipelines::compute_entry(device, &layout, shader, entry);
        self.pipelines = Some(ParticlePipelines {
            begin: entry(meshes::particles_begin),
            gather: entry(meshes::particles_gather),
            currents: entry(meshes::particles_currents),
            age: entry(meshes::particles_age),
            births: entry(meshes::particles_births),
            sum: entry(meshes::particles_sum),
            scan: entry(meshes::particles_scan),
            place: entry(meshes::particles_place),
            splat: entry(meshes::particles_splat),
            look: entry(meshes::particles_look),
            down: entry(meshes::particles_down),
        });
    }

    /// Records the particles' passes over `water`, the water's group, timed
    /// by `profiler` (`water_particles`): the particles the pools held and
    /// the lattice's water and what carries it; each particle's step, the
    /// sites' births and the scan over them, after the dispatch over the
    /// particles and over the scan's items are copied from their counts;
    /// and, after the dispatch over the particles the scan left is, their
    /// laying into the other half of the pools, their splat and the
    /// textures' levels; then a copy of the pools' tallies to read back.
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        water: &wgpu::BindGroup,
        profiler: &GpuProfiler,
    ) {
        let (Some(pipelines), Some(grid)) = (&self.pipelines, &self.grid) else {
            return;
        };
        let span = profiler.span("water_particles");
        let points = grid.points.div_ceil(PARTICLE_SIDE);
        let args = u64::from(PARTICLE_ARGS) * 4;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/water-particles-lattice"),
                timestamp_writes: span.map(|span| span.compute_part(true, false)),
            });
            self.bind(&mut pass, water, grid, &grid.groups[0]);
            pass.set_pipeline(&pipelines.begin);
            pass.dispatch_workgroups(1, 1, 1);
            for pipeline in [&pipelines.gather, &pipelines.currents] {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(points, points, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&grid.counts, args, &grid.args, 0, ARGS_BYTES);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/water-particles-scan"),
                timestamp_writes: None,
            });
            self.bind(&mut pass, water, grid, &grid.groups[0]);
            pass.set_pipeline(&pipelines.age);
            pass.dispatch_workgroups_indirect(&grid.args, 0);
            pass.set_pipeline(&pipelines.births);
            pass.dispatch_workgroups(points, points, 1);
            pass.set_pipeline(&pipelines.sum);
            pass.dispatch_workgroups_indirect(&grid.args, ARGS_BYTES / 2);
            pass.set_pipeline(&pipelines.scan);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&grid.counts, args, &grid.args, 0, ARGS_BYTES / 2);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/water-particles-splat"),
                timestamp_writes: span.map(|span| span.compute_part(false, true)),
            });
            self.bind(&mut pass, water, grid, &grid.groups[0]);
            pass.set_pipeline(&pipelines.place);
            pass.dispatch_workgroups_indirect(&grid.args, ARGS_BYTES / 2);
            pass.set_pipeline(&pipelines.splat);
            pass.dispatch_workgroups_indirect(&grid.args, 0);
            pass.set_pipeline(&pipelines.look);
            let texels = grid.texels.div_ceil(PARTICLE_SIDE);
            pass.dispatch_workgroups(texels, texels, 1);
            pass.set_pipeline(&pipelines.down);
            for (level, group) in grid.groups.iter().enumerate().skip(1) {
                pass.set_bind_group(2, group, &[]);
                let side = (grid.texels >> level).div_ceil(PARTICLE_SIDE);
                pass.dispatch_workgroups(side, side, 1);
            }
        }
        self.readbacks.record(
            device,
            encoder,
            &grid.counts,
            (TALLIES * size_of::<u32>()) as u64,
            (),
        );
        grid.encoded.set(true);
    }

    /// Binds the particles' groups into `pass`: the empty one, the water's
    /// `water` at the terrain's uniform offset (the views' and water trees'
    /// at their first, which the passes read none of), and `group` of
    /// `grid`'s.
    fn bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        water: &wgpu::BindGroup,
        grid: &ParticleGrid,
        group: &wgpu::BindGroup,
    ) {
        pass.set_bind_group(0, &self.empty, &[]);
        pass.set_bind_group(1, water, &[0, grid.terrain, 0]);
        pass.set_bind_group(2, group, &[]);
    }

    /// Requests the map of the tallies the frame just submitted copied.
    pub(crate) fn after_submit(&self) {
        let tallies = Rc::clone(&self.tallies);
        self.readbacks.after_submit(move |words, ()| {
            tallies.set(std::array::from_fn(|index| words.get(index)));
        });
    }

    /// The particles' textures of their white water's and foam's coverage
    /// and the waves' slopes as water's shading reads them, with their
    /// levels; none while none run.
    pub(crate) fn view(&self) -> Option<&wgpu::TextureView> {
        self.grid.as_ref().map(|grid| &grid.view)
    }

    /// The particles' texture of the waves' slopes' squares as water's
    /// shading reads it, with its levels; none while none run.
    pub(crate) fn spread_view(&self) -> Option<&wgpu::TextureView> {
        self.grid.as_ref().map(|grid| &grid.spread_view)
    }

    /// The sampler water's shading reads the particles' textures through: it
    /// repeats them, as their texels hold their world texels' remainders.
    pub(crate) fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

    /// What water's uniform holds of the particles for terrains' water
    /// (`water.foam`): the meters across their texels (none while none run),
    /// the meters their textures span, the pixels their texels span where
    /// they hand over and the meters across the cells of the water they
    /// ride.
    pub(crate) fn shading(&self) -> [f32; 4] {
        self.grid.as_ref().map_or([0.0; 4], |grid| {
            [
                grid.texel,
                grid.texel * grid.texels as f32,
                grid.pixels,
                grid.ridden,
            ]
        })
    }

    /// A count advancing whenever the particles' textures change.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// What the particles report: none while none run.
    pub(crate) fn stats(&self) -> ParticleStats {
        let Some(grid) = &self.grid else {
            return ParticleStats::default();
        };
        let tallies = self.tallies.get();
        let tally = |word: u32| tallies[word as usize];
        ParticleStats {
            texels: grid.texels,
            foam: tally(PARTICLE_FOAM_LIVE),
            waves: tally(PARTICLE_WAVES_LIVE),
            born: tally(PARTICLE_FOAM_BORN) + tally(PARTICLE_WAVES_BORN),
            dropped: tally(PARTICLE_FOAM_DROPPED) + tally(PARTICLE_WAVES_DROPPED),
        }
    }

    /// Bytes of the window's buffers and textures.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let texels = |texture: &wgpu::Texture| -> u64 {
            (0..texture.mip_level_count())
                .map(|level| u64::from(texture.width() >> level).pow(2))
                .sum()
        };
        self.grid.as_ref().map_or(0, |grid| {
            [
                &grid.frame,
                &grid.lattice,
                &grid.pool,
                &grid.counts,
                &grid.args,
                &grid.splat,
            ]
            .iter()
            .map(|buffer| buffer.size())
            .sum::<u64>()
                + (texels(&grid.texture) + texels(&grid.spread)) * TEXEL_BYTES
        })
    }
}

impl ParticleGrid {
    /// A window of `texels` texels along a side, `texel` meters across each,
    /// with its passes' groups of `layout`; its pools hold none yet.
    fn new(device: &wgpu::Device, layout: &GroupLayout, (texels, texel): (u32, f32)) -> Self {
        let points = texels / PARTICLE_LATTICE + 2 * LATTICE_MARGIN + 1;
        let area = u64::from(texels) * u64::from(texels);
        let pools = [
            (area / u64::from(FOAM_TEXELS)) as u32,
            (area / u64::from(WAVE_TEXELS)) as u32,
        ];
        let pooled = u64::from(pools[0]) + u64::from(pools[1]);
        let sites = u64::from(points) * u64::from(points);
        let scanned = (pooled + sites).div_ceil(u64::from(PARTICLE_GROUP));
        let frame = crate::buffers::uniform(
            device,
            "engine/water-particles-frame",
            size_of::<ParticleFrame>() as u64,
        );
        let storage = |label, bytes, extra| crate::buffers::storage(device, label, bytes, extra);
        let lattice = storage(
            "engine/water-particles-lattice",
            sites * POINT_BYTES,
            wgpu::BufferUsages::empty(),
        );
        let pool = storage(
            "engine/water-particles-pools",
            2 * pooled * PARTICLE_BYTES,
            wgpu::BufferUsages::empty(),
        );
        let counts = storage(
            "engine/water-particles-counts",
            (u64::from(PARTICLE_ITEMS) + pooled + sites + 2 * scanned) * 4,
            wgpu::BufferUsages::COPY_SRC,
        );
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/water-particles-args"),
            size: ARGS_BYTES.next_multiple_of(16),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let splat = storage(
            "engine/water-particles-splat",
            area * SPLAT_BYTES,
            wgpu::BufferUsages::empty(),
        );
        let textures = |label| {
            let texture = crate::targets::texture_2d(
                device,
                label,
                (texels, texels),
                1,
                TEXTURE_LEVELS + 1,
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            let levels: Vec<wgpu::TextureView> = (0..=TEXTURE_LEVELS)
                .map(|level| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("engine/water-particles-level"),
                        base_mip_level: level,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            (texture, levels)
        };
        let (texture, levels) = textures("engine/water-particles-splatted");
        let (spread, spread_levels) = textures("engine/water-particles-spread");
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let spread_view = spread.create_view(&wgpu::TextureViewDescriptor::default());
        // The finest level's group, which writes it, reads the next coarser
        // in the finer one's place; each coarser level's reads the level
        // finer than it.
        let groups = (0..=TEXTURE_LEVELS as usize)
            .map(|level| {
                let finer = if level == 0 { 1 } else { level - 1 };
                crate::pipelines::group(
                    device,
                    "engine/water-particles-group",
                    &WATER_PARTICLES,
                    layout,
                    &[
                        ("particle_frame", frame.as_entire_binding()),
                        crate::buffers::entry("particle_lattice", &lattice),
                        crate::buffers::entry("particle_pool", &pool),
                        crate::buffers::entry("particle_counts", &counts),
                        crate::buffers::entry("particle_splat", &splat),
                        ("particle_source", crate::pipelines::view(&levels[finer])),
                        ("particle_target", crate::pipelines::view(&levels[level])),
                        (
                            "particle_spread_source",
                            crate::pipelines::view(&spread_levels[finer]),
                        ),
                        (
                            "particle_spread_target",
                            crate::pipelines::view(&spread_levels[level]),
                        ),
                    ],
                )
            })
            .collect();
        Self {
            texels,
            texel,
            points,
            pools,
            pixels: HANDOVER_PIXELS,
            ridden: 0.0,
            frame,
            lattice,
            pool,
            counts,
            args,
            splat,
            texture,
            view,
            spread,
            spread_view,
            groups,
            held: None,
            read: 0,
            window: [0; 2],
            terrain: 0,
            encoded: Cell::new(false),
        }
    }
}

/// The texels along a side the particles' window takes as `seen` sees it:
/// as many as reach, from the detail eye, out to where its texels span
/// HANDOVER_PIXELS, with WINDOW_MARGIN more each side, whatever their size,
/// in whole texels of its textures' coarsest level; no more than the display
/// has pixels, nor than one of the device's storage bindings holds of its
/// splat's sums, the largest of its buffers, nor than a side of its textures
/// holds.
fn wanted(device: &wgpu::Device, seen: ParticleView) -> u32 {
    let block = 1 << TEXTURE_LEVELS;
    let reach = 1.0 / (HANDOVER_PIXELS * max(seen.pixel_angle, 1.0e-6));
    let asked = (2.0 * reach).ceil() as u32 + 2 * WINDOW_MARGIN;
    let limits = device.limits();
    let binding = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let held = (binding as f64 / SPLAT_BYTES as f64).sqrt() as u32;
    let shown = (f64::from(seen.display.width) * f64::from(seen.display.height)).sqrt() as u32;
    let most = held.min(shown).min(limits.max_texture_dimension_2d) / block * block;
    asked.next_multiple_of(block).min(most).max(block)
}

/// The uniform offset of the terrain whose water the particles run over:
/// of those drawing tiles' water (`tiles`, `terrains` the frame's), the one
/// whose square holds the detail eye `eye`, a terrain's own before its
/// surroundings; none where none draws.
pub(crate) fn particle_terrain(
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
