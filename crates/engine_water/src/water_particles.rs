use bevy_ecs::prelude::*;
use engine_core::math::{clamp, max};
use engine_core::{Cut, DeviceLimits, Error, FrameTime, Readback, View};
use engine_places::ground::{GroundTiles, TerrainTiles};
use engine_places::wind::Wind;
use engine_render::{
    AfterRead, Bind, Buffer, ComputePipeline, Copied, CopyRegion, Count, Draw, DrawInput, Encode,
    Install, Lifetime, Own, Pass, Pipeline, Prepare, Sampler, SamplerDesc, Shape, Texture, wgpu,
};
use glam::{UVec2, Vec2, Vec3};

use crate::Water;
use crate::gpu::{
    PARTICLE_ARGS, PARTICLE_DRAW, PARTICLE_FOAM_BEFORE, PARTICLE_GROUP, PARTICLE_ITEMS,
    PARTICLE_LATTICE, PARTICLE_MARGIN, PARTICLE_SIDE, ParticleFrame,
};
use crate::shallow_water::step_seconds;
use crate::state::WaterState;
use crate::water::WaterGpu;
use crate::water_settle::TALLIES;
use crate::windows::LONGEST_FRAME;

const TEXEL_SPLIT: u32 = 16;
const HANDOVER_PIXELS: f32 = 1.0;
const TEXTURE_LEVELS: u32 = 6;
const TEXTURE_LAYERS: u32 = 2;
const WINDOW_MARGIN: u32 = engine_render::MATERIAL_ANISOTROPY as u32 / HANDOVER_PIXELS as u32 + 1;
const FOAM_PIXELS: u32 = 4;
const WAVE_TEXELS: u32 = 16;
const SPLAT_BYTES: u64 = 16;
const PARTICLE_BYTES: u64 = 32;
const POINT_BYTES: u64 = 32;
const TALLY_WORDS: usize = PARTICLE_FOAM_BEFORE as usize;
const ARGS_BYTES: u64 = 24;
const DRAW_ARGS: u64 = 32;
const DRAW_BYTES: u64 = 16;
const PARTICLES_SPAN: &str = "water_particles";

#[derive(Resource, Default)]
pub(crate) struct ParticleTallies {
    pub(crate) counts: [u32; TALLY_WORDS],
}

impl Readback for ParticleTallies {
    fn arrived(&mut self, _: u64, _: u32, bytes: &[u8]) {
        for (count, word) in self.counts.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *count = u32::from_le_bytes(*word);
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ParticleView {
    pub(crate) detail: Vec3,
    pub(crate) pixel_angle: f32,
    pub(crate) display: UVec2,
}

impl ParticleView {
    pub(crate) fn of(view: &View) -> Self {
        let display = view.display_size();
        Self {
            detail: view.detail_eye(),
            pixel_angle: engine_render::pixel_angle(view.view_projection(), display.y),
            display,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParticleSizes {
    pub(crate) texels: u32,
    pub(crate) square: u32,
    pub(crate) pools: [u32; 2],
}

pub(crate) struct Grid {
    pub(crate) sizes: ParticleSizes,
    pub(crate) texel: f32,
    pub(crate) cut: Cut,
    pub(crate) points: u32,
    pub(crate) pixels: f32,
    pub(crate) ridden: f32,
    held: Option<[i64; 2]>,
    read: u32,
    window: [i64; 2],
    pub(crate) generation: u64,
    encoded: bool,
}

#[derive(Default)]
pub(crate) struct Particles {
    pub(crate) grid: Option<Grid>,
    last: Option<f64>,
    step: Option<u32>,
    kept: Option<(Option<f64>, Option<u32>)>,
    generation: u64,
    pub(crate) frame: Option<ParticleFrame>,
}

impl Particles {
    fn rest(&mut self) {
        self.frame = None;
        if self.grid.take().is_some() {
            self.generation += 1;
        }
        self.last = None;
        self.step = None;
        self.kept = None;
    }

    fn prepare(
        &mut self,
        (terrain, seen, limits): (u32, ParticleView, &DeviceLimits),
        (settings, ridden, wind, seconds): (&Water, f32, [f32; 2], f64),
        submitted: bool,
    ) {
        if let Some((last, step)) = self.kept.take().filter(|_| !submitted) {
            (self.last, self.step) = (last, step);
        }
        self.kept = Some((self.last, self.step));
        if let Some(grid) = self.grid.as_mut()
            && std::mem::take(&mut grid.encoded)
            && submitted
        {
            grid.held = Some(grid.window);
            grid.read ^= 1;
        }
        let texel = settings.simulation_cell / TEXEL_SPLIT as f32;
        let asked = wanted(limits, seen);
        let kept = self.grid.as_ref().filter(|grid| {
            grid.texel == texel && grid.sizes.texels >= asked && grid.sizes.texels < 2 * asked
        });
        let texels = kept.map_or(asked, |grid| grid.sizes.texels);
        let (sizes, cut) = sized(limits, (texels, seen.display), settings.foam_points);
        let keeps = kept.is_some_and(|grid| {
            grid.sizes.square == sizes.square
                && grid.sizes.pools[0] >= sizes.pools[0]
                && grid.sizes.pools[0] <= 2 * sizes.pools[0]
        });
        if !keeps {
            self.generation += 1;
            self.grid = Some(Grid {
                sizes,
                texel,
                cut: Cut::default(),
                points: lattice_points(sizes.square),
                pixels: HANDOVER_PIXELS,
                ridden: 0.0,
                held: None,
                read: 0,
                window: [0; 2],
                generation: self.generation,
                encoded: false,
            });
        }
        let Some(grid) = self.grid.as_mut() else {
            return;
        };
        grid.cut = cut;
        let reach = (grid.sizes.texels / 2 - WINDOW_MARGIN) as f32;
        grid.pixels = max(
            HANDOVER_PIXELS,
            1.0 / (reach * max(seen.pixel_angle, 1.0e-6)),
        );
        grid.ridden = ridden;
        let half = i64::from(grid.sizes.texels / 2);
        let window = [seen.detail.x, seen.detail.z]
            .map(|at| (f64::from(at) / f64::from(texel)).floor() as i64 - half);
        grid.window = window;
        let elapsed = self.last.map_or(0.0, |last| {
            clamp((seconds - last) as f32, 0.0, LONGEST_FRAME as f32)
        });
        self.last = Some(seconds);
        let tick = step_seconds(settings.simulation_cell);
        let step = (seconds / f64::from(tick)).floor() as i64 as u32;
        let begun = self.step.unwrap_or(step);
        self.step = Some(step);
        let least = ((LONGEST_FRAME / f64::from(tick)).ceil() as u32).next_power_of_two();
        let held = grid.held.unwrap_or(window);
        let spacing = f64::from(texel) * f64::from(PARTICLE_LATTICE);
        let before = i64::from(grid.sizes.square / 2 + PARTICLE_MARGIN);
        let lattice = [seen.detail.x, seen.detail.z]
            .map(|at| (f64::from(at) / spacing).floor() as i64 - before);
        self.frame = Some(ParticleFrame {
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
                grid.sizes.texels as i32,
            ],
            step: [texel, elapsed, wind[0], wind[1]],
            flags: [u32::from(grid.held.is_none()), begun, step, least],
            pool: [
                grid.sizes.pools[0],
                grid.sizes.pools[1],
                grid.read,
                grid.points * grid.points,
            ],
            eddies: [tick, ridden, 0.0, 0.0],
            terrain: [terrain, 0, 0, 0],
        });
        grid.encoded = true;
    }

    pub(crate) fn shading(&self) -> [f32; 4] {
        self.grid.as_ref().map_or([0.0; 4], |grid| {
            [
                grid.texel,
                grid.texel * grid.sizes.texels as f32,
                grid.pixels,
                grid.ridden,
            ]
        })
    }

    pub(crate) fn stats(&self) -> (u32, u32, Cut, u32) {
        let Some(grid) = &self.grid else {
            return (0, 0, Cut::default(), 0);
        };
        let spacing = grid.texel * PARTICLE_LATTICE as f32;
        (
            grid.sizes.texels,
            grid.sizes.square,
            grid.cut,
            (0.5 * grid.sizes.square as f32 * spacing).round() as u32,
        )
    }
}

fn wanted(limits: &DeviceLimits, seen: ParticleView) -> u32 {
    let block = 1 << TEXTURE_LEVELS;
    let reach = 1.0 / (HANDOVER_PIXELS * max(seen.pixel_angle, 1.0e-6));
    let asked = (2.0 * reach).ceil() as u32 + 2 * WINDOW_MARGIN;
    let binding = limits.storage_room;
    let held = (binding as f64 / SPLAT_BYTES as f64).sqrt() as u32;
    let shown = (f64::from(seen.display.x) * f64::from(seen.display.y)).sqrt() as u32;
    let most = held.min(shown).min(limits.max_texture_dimension_2d) / block * block;
    asked.next_multiple_of(block).min(most).max(block)
}

fn sized(
    limits: &DeviceLimits,
    (texels, display): (u32, UVec2),
    asked: u32,
) -> (ParticleSizes, Cut) {
    let binding = limits.storage_room;
    let items = u64::from(limits.max_compute_workgroups_per_dimension) * u64::from(PARTICLE_GROUP);
    let held = (binding as f64 / POINT_BYTES as f64).sqrt() as u32;
    let shared = (items as f64 / 2.0).sqrt() as u32;
    let most = held.min(shared).saturating_sub(lattice_points(0)) / 2 * 2;
    let asked = asked.next_multiple_of(2);
    let square = asked.min(most).max(texels / PARTICLE_LATTICE + 2);
    let sites = u64::from(lattice_points(square)).pow(2);
    let waves = u64::from(texels).pow(2) / u64::from(WAVE_TEXELS);
    let shown = u64::from(display.x) * u64::from(display.y);
    let foam = (shown / u64::from(FOAM_PIXELS))
        .min((binding / (2 * PARTICLE_BYTES)).saturating_sub(waves))
        .min(items.saturating_sub(waves + sites));
    let sizes = ParticleSizes {
        texels,
        square,
        pools: [foam as u32, waves as u32],
    };
    (sizes, Cut::of(asked, square))
}

fn lattice_points(square: u32) -> u32 {
    square + 2 * PARTICLE_MARGIN + 1
}

pub(crate) fn particle_terrain(
    watered: &[Entity],
    (ground, tiles): (&GroundTiles, &TerrainTiles),
    eye: Vec3,
) -> Option<u32> {
    let rank = |entity: Entity| {
        tiles
            .terrains
            .iter()
            .find(|entry| entry.terrain == entity)
            .map_or((false, false), |entry| {
                let local = Vec2::new(eye.x - entry.origin.x, eye.z - entry.origin.z);
                let extent = entry.pyramid.extent();
                let holds = local.cmpge(Vec2::ZERO).all() && local.cmple(extent).all();
                (holds, !entry.surroundings)
            })
    };
    ground
        .tables
        .iter()
        .filter(|(entity, _)| watered.contains(entity))
        .max_by_key(|(entity, _)| rank(*entity))
        .map(|&(_, index)| index)
}

pub(crate) fn plan_particles(
    water: Res<Water>,
    (view, limits, time, wind): (Res<View>, Res<DeviceLimits>, Res<FrameTime>, Res<Wind>),
    (ground, tiles): (Res<GroundTiles>, Res<TerrainTiles>),
    mut state: ResMut<WaterState>,
) {
    let state = &mut *state;
    let seen = ParticleView::of(&view);
    let carried = (water.enabled && !state.watered.is_empty())
        .then(|| particle_terrain(&state.watered, (&*ground, &*tiles), seen.detail))
        .flatten();
    let seconds = time.presented_seconds;
    let Some(terrain) = carried else {
        state.particles.rest();
        return;
    };
    let ridden = state
        .live
        .work
        .drawn
        .as_ref()
        .map_or(water.simulation_cell, |drawn| drawn.window.cell);
    state.particles.prepare(
        (terrain, seen, &limits),
        (&water, ridden, wind.velocity, seconds),
        view.submitted(),
    );
}

pub(crate) struct ParticlesGpu {
    pub(crate) frame: Buffer<WaterGpu>,
    lattice: Buffer<WaterGpu>,
    pub(crate) pool: Buffer<WaterGpu>,
    counts: Buffer<WaterGpu>,
    args: Buffer<WaterGpu>,
    splat: Buffer<WaterGpu>,
    pub(crate) texture: Texture<WaterGpu>,
    pub(crate) sampler: Sampler<WaterGpu>,
    lattice_pass: Pass<WaterGpu>,
    first_copy: Pass<WaterGpu>,
    scan_pass: Pass<WaterGpu>,
    second_copy: Pass<WaterGpu>,
    splat_pass: Pass<WaterGpu>,
    scan_args: DrawInput<WaterGpu, Own>,
    splat_args: DrawInput<WaterGpu, Own>,
    pub(crate) foam_args: Option<DrawInput<WaterGpu, Own>>,
    begin: ComputePipeline<WaterGpu>,
    gather: ComputePipeline<WaterGpu>,
    currents: ComputePipeline<WaterGpu>,
    age: ComputePipeline<WaterGpu>,
    births: ComputePipeline<WaterGpu>,
    sum: ComputePipeline<WaterGpu>,
    scan: ComputePipeline<WaterGpu>,
    place: ComputePipeline<WaterGpu>,
    splat_pipeline: ComputePipeline<WaterGpu>,
    look: ComputePipeline<WaterGpu>,
    down: Vec<ComputePipeline<WaterGpu>>,
    generation: Option<u64>,
    running: Option<(u32, u32)>,
}

pub(crate) fn levels(
    texture: Texture<WaterGpu>,
    level: u32,
) -> [(engine_core::Var, Bind<WaterGpu>); 4] {
    use crate::shaders::water::particles;
    let finer = if level == 0 { 1 } else { level - 1 };
    let slice = |mip: u32, layer: u32| Bind::Slice {
        texture,
        mips: mip..mip + 1,
        layers: layer..layer + 1,
    };
    [
        (particles::PARTICLE_SOURCE, slice(finer, 0)),
        (particles::PARTICLE_TARGET, slice(level, 0)),
        (particles::PARTICLE_SPREAD_SOURCE, slice(finer, 1)),
        (particles::PARTICLE_SPREAD_TARGET, slice(level, 1)),
    ]
}

fn passes(
    cx: &mut Install<'_, WaterGpu>,
    after: Pass<WaterGpu>,
    (counts, args): (Buffer<WaterGpu>, Buffer<WaterGpu>),
) -> [Pass<WaterGpu>; 5] {
    let lattice_pass = cx
        .pass("water_particles")
        .after(after)
        .label("engine/water-particles-lattice")
        .span(PARTICLES_SPAN)
        .id();
    let first_copy = cx.copy(
        "water_particles_args",
        Copied::Buffer(counts),
        Copied::Buffer(args),
    );
    let scan_pass = cx
        .pass("water_particles_scan")
        .after(first_copy)
        .label("engine/water-particles-scan")
        .span(PARTICLES_SPAN)
        .id();
    let second_copy = cx.copy(
        "water_particles_draw_args",
        Copied::Buffer(counts),
        Copied::Buffer(args),
    );
    cx.after(second_copy, scan_pass);
    let splat_pass = cx
        .pass("water_particles_splat")
        .after(second_copy)
        .label("engine/water-particles-splat")
        .span(PARTICLES_SPAN)
        .id();
    [lattice_pass, first_copy, scan_pass, second_copy, splat_pass]
}

impl ParticlesGpu {
    pub(crate) fn install(
        cx: &mut Install<'_, WaterGpu>,
        after: Pass<WaterGpu>,
        shared: &[(engine_core::Var, Bind<WaterGpu>)],
    ) -> Result<Self, Error> {
        use crate::shaders::water::particles;
        let frame = cx.buffer("water_particles_frame", Lifetime::Kept);
        let lattice = cx.buffer("water_particles_lattice", Lifetime::Kept);
        let pool = cx.buffer("water_particles_pools", Lifetime::Kept);
        let counts = cx.buffer("water_particles_counts", Lifetime::Kept);
        let args = cx.buffer("water_particles_args", Lifetime::Kept);
        let splat = cx.buffer("water_particles_splat", Lifetime::Kept);
        let texture = cx.texture(
            "water_particles_splatted",
            wgpu::TextureFormat::Rgba16Float,
            Lifetime::Kept,
        );
        let sampler = cx.sampler(SamplerDesc {
            filter: wgpu::FilterMode::Linear,
            mipmap: wgpu::MipmapFilterMode::Linear,
            address: [
                wgpu::AddressMode::Repeat,
                wgpu::AddressMode::Repeat,
                wgpu::AddressMode::ClampToEdge,
            ],
            compare: None,
            anisotropy: engine_render::MATERIAL_ANISOTROPY,
        });
        let [lattice_pass, first_copy, scan_pass, second_copy, splat_pass] =
            passes(cx, after, (counts, args));
        let own = [
            (particles::PARTICLE_FRAME, Bind::Buffer(frame)),
            (particles::PARTICLE_LATTICE, Bind::Buffer(lattice)),
            (particles::PARTICLE_POOL, Bind::Buffer(pool)),
            (particles::PARTICLE_COUNTS, Bind::Buffer(counts)),
            (particles::PARTICLE_SPLAT, Bind::Buffer(splat)),
        ];
        let bound = |cx: &mut Install<'_, WaterGpu>, entry, level: u32| {
            let mut all: Vec<(engine_core::Var, Bind<WaterGpu>)> = own
                .iter()
                .map(|(var, bind)| (*var, crate::water::copied(bind)))
                .collect();
            all.extend(levels(texture, level));
            all.extend(
                shared
                    .iter()
                    .map(|(var, bind)| (*var, crate::water::copied(bind))),
            );
            crate::water::reached(cx, &[entry], all)
        };
        let pipeline = |cx: &mut Install<'_, WaterGpu>, pass, entry, level| {
            let bound = bound(cx, entry, level);
            cx.compute_pipeline(pass, entry, &bound)
        };
        let begin = pipeline(cx, lattice_pass, particles::PARTICLES_BEGIN, 0)?;
        let gather = pipeline(cx, lattice_pass, particles::PARTICLES_GATHER, 0)?;
        let currents = pipeline(cx, lattice_pass, particles::PARTICLES_CURRENTS, 0)?;
        let age = pipeline(cx, scan_pass, particles::PARTICLES_AGE, 0)?;
        let births = pipeline(cx, scan_pass, particles::PARTICLES_BIRTHS, 0)?;
        let sum = pipeline(cx, scan_pass, particles::PARTICLES_SUM, 0)?;
        let scan = pipeline(cx, scan_pass, particles::PARTICLES_SCAN, 0)?;
        let place = pipeline(cx, splat_pass, particles::PARTICLES_PLACE, 0)?;
        let splat_pipeline = pipeline(cx, splat_pass, particles::PARTICLES_SPLAT, 0)?;
        let look = pipeline(cx, splat_pass, particles::PARTICLES_LOOK, 0)?;
        let down = (1..=TEXTURE_LEVELS)
            .map(|level| pipeline(cx, splat_pass, particles::PARTICLES_DOWN, level))
            .collect::<Result<Vec<_>, Error>>()?;
        let scan_args = cx.draw_input(scan_pass, args);
        let splat_args = cx.draw_input(splat_pass, args);
        Ok(Self {
            frame,
            lattice,
            pool,
            counts,
            args,
            splat,
            texture,
            sampler,
            lattice_pass,
            first_copy,
            scan_pass,
            second_copy,
            splat_pass,
            scan_args,
            splat_args,
            foam_args: None,
            begin,
            gather,
            currents,
            age,
            births,
            sum,
            scan,
            place,
            splat_pipeline,
            look,
            down,
            generation: None,
            running: None,
        })
    }

    pub(crate) fn after(&self) -> Pass<WaterGpu> {
        self.splat_pass
    }

    pub(crate) fn args(&self) -> Buffer<WaterGpu> {
        self.args
    }

    pub(crate) fn bound(&self) -> [(engine_core::Var, Bind<WaterGpu>); 4] {
        use crate::shaders::water::{foam, particles};
        [
            (foam::WATER_PARTICLES, Bind::Texture(self.texture)),
            (foam::WATER_PARTICLES_SAMPLER, Bind::Sampler(self.sampler)),
            (particles::PARTICLE_FRAME, Bind::Buffer(self.frame)),
            (particles::FOAM_POOL, Bind::Buffer(self.pool)),
        ]
    }

    pub(crate) fn prepare(
        &mut self,
        particles: &crate::water_particles::Particles,
        cx: &mut Prepare<'_, WaterGpu>,
    ) {
        self.running = None;
        cx.allocate_buffer(self.frame, size_of::<ParticleFrame>() as u64);
        let (Some(grid), Some(frame)) = (particles.grid.as_ref(), particles.frame.as_ref()) else {
            self.generation = None;
            for buffer in [self.lattice, self.pool, self.counts, self.splat] {
                cx.allocate_buffer(buffer, 256);
            }
            cx.allocate_buffer(self.args, DRAW_ARGS + DRAW_BYTES);
            cx.allocate(
                self.texture,
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: TEXTURE_LAYERS,
                },
                1,
            );
            return;
        };
        let ParticleSizes {
            texels,
            square: _,
            pools,
        } = grid.sizes;
        let points = grid.points;
        if self.generation != Some(grid.generation) {
            self.generation = Some(grid.generation);
            let area = u64::from(texels) * u64::from(texels);
            let pooled = u64::from(pools[0]) + u64::from(pools[1]);
            let sites = u64::from(points) * u64::from(points);
            let scanned = (pooled + sites).div_ceil(u64::from(PARTICLE_GROUP));
            let sizes = [
                (self.lattice, sites * POINT_BYTES),
                (self.pool, 2 * pooled * PARTICLE_BYTES),
                (
                    self.counts,
                    (u64::from(PARTICLE_ITEMS) + pooled + sites + 2 * scanned) * 4,
                ),
                (self.splat, area * SPLAT_BYTES),
            ];
            for (buffer, bytes) in sizes {
                cx.release_buffer(buffer);
                cx.allocate_buffer(buffer, bytes);
            }
            cx.release_buffer(self.args);
            cx.allocate_buffer(self.args, DRAW_ARGS + DRAW_BYTES);
            cx.release(self.texture);
            cx.allocate(
                self.texture,
                wgpu::Extent3d {
                    width: texels,
                    height: texels,
                    depth_or_array_layers: TEXTURE_LAYERS,
                },
                TEXTURE_LEVELS + 1,
            );
        }
        cx.write(self.frame, 0, &[*frame]);
        self.running = Some((points.div_ceil(PARTICLE_SIDE), texels));
        let args = u64::from(PARTICLE_ARGS) * 4;
        cx.run(self.lattice_pass);
        cx.copy(
            self.first_copy,
            &[CopyRegion::Bytes {
                from: args,
                to: 0,
                size: ARGS_BYTES,
            }],
        );
        cx.run(self.scan_pass);
        cx.copy(
            self.second_copy,
            &[
                CopyRegion::Bytes {
                    from: args,
                    to: 0,
                    size: ARGS_BYTES / 2,
                },
                CopyRegion::Bytes {
                    from: u64::from(PARTICLE_DRAW) * 4,
                    to: DRAW_ARGS,
                    size: DRAW_BYTES,
                },
            ],
        );
        cx.run(self.splat_pass);
        cx.read_back::<ParticleTallies>(
            TALLIES,
            self.counts,
            0..(TALLY_WORDS * size_of::<u32>()) as u64,
            AfterRead::Keep,
        );
    }

    pub(crate) fn running(&self) -> bool {
        self.running.is_some()
    }

    pub(crate) fn forget(&mut self) {
        self.generation = None;
    }

    pub(crate) fn encode(&mut self, pass: Pass<WaterGpu>, cx: &mut Encode<'_, WaterGpu>) -> bool {
        let ours = [self.lattice_pass, self.scan_pass, self.splat_pass].contains(&pass);
        let Some((points, texels)) = self.running.filter(|_| ours) else {
            return ours;
        };
        let groups = |cx: &mut Encode<'_, WaterGpu>, pipeline, count| {
            cx.dispatch(pipeline, Count::Groups(count));
        };
        if pass == self.lattice_pass {
            groups(cx, &self.begin, [1, 1, 1]);
            groups(cx, &self.gather, [points, points, 1]);
            groups(cx, &self.currents, [points, points, 1]);
        } else if pass == self.scan_pass {
            cx.dispatch(&self.age, Count::Indirect(self.scan_args, 0));
            groups(cx, &self.births, [points, points, 1]);
            cx.dispatch(&self.sum, Count::Indirect(self.scan_args, ARGS_BYTES / 2));
            groups(cx, &self.scan, [1, 1, 1]);
        } else {
            cx.dispatch(
                &self.place,
                Count::Indirect(self.splat_args, ARGS_BYTES / 2),
            );
            cx.dispatch(&self.splat_pipeline, Count::Indirect(self.splat_args, 0));
            let side = texels.div_ceil(PARTICLE_SIDE);
            groups(cx, &self.look, [side, side, 1]);
            for (level, down) in (1..=TEXTURE_LEVELS).zip(&self.down) {
                let side = (texels >> level).div_ceil(PARTICLE_SIDE);
                groups(cx, down, [side, side, 1]);
            }
        }
        true
    }

    pub(crate) fn draw_foam(
        &self,
        draw: &mut Draw<'_, WaterGpu, Own>,
        pipeline: &Pipeline<WaterGpu, Own>,
    ) {
        let (Some(args), true) = (self.foam_args, self.running()) else {
            return;
        };
        draw.draw(
            pipeline,
            Shape::Indirect {
                vertices: &[],
                indices: None,
                args,
                offset: DRAW_ARGS,
            },
        );
    }
}
