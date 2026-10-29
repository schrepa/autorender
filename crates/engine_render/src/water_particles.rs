use std::cell::Cell;
use std::rc::Rc;

use engine_core::glam::{Vec2, Vec3};
use engine_core::math::{clamp, max};
use engine_terrain::shallow_water::step_seconds;
use engine_terrain::{RenderTerrain, Water};

use crate::bindings::WATER_PARTICLES;
use crate::buffers::ReadbackRing;
use crate::gpu_layouts::constants::{
    MATERIAL_ANISOTROPY, PARTICLE_ARGS, PARTICLE_DRAW, PARTICLE_FOAM_BEFORE, PARTICLE_FOAM_BORN,
    PARTICLE_FOAM_DROPPED, PARTICLE_FOAM_LIVE, PARTICLE_GROUP, PARTICLE_ITEMS, PARTICLE_LATTICE,
    PARTICLE_MARGIN, PARTICLE_SIDE, PARTICLE_WAVES_BORN, PARTICLE_WAVES_DROPPED,
    PARTICLE_WAVES_LIVE,
};
use crate::gpu_layouts::water::ParticleFrame;
use crate::pipelines::GroupLayout;
use crate::profiling::GpuProfiler;
use crate::shaders::entry::meshes;
use crate::targets::FrameSize;
use crate::terrain::TerrainWaterDraws;
use crate::water_sim::LONGEST_FRAME;

const TEXEL_SPLIT: u32 = 16;
const HANDOVER_PIXELS: f32 = 1.0;
const TEXTURE_LEVELS: u32 = 6;
const WINDOW_MARGIN: u32 = MATERIAL_ANISOTROPY as u32 / HANDOVER_PIXELS as u32 + 1;
const FOAM_PIXELS: u32 = 4;
const WAVE_TEXELS: u32 = 16;
const SPLAT_BYTES: u64 = 16;
const PARTICLE_BYTES: u64 = 32;
const POINT_BYTES: u64 = 32;
const TEXEL_BYTES: u64 = 8;
const TALLIES: usize = PARTICLE_FOAM_BEFORE as usize;
const ARGS_BYTES: u64 = 24;
const DRAW_ARGS: u64 = 32;
const DRAW_BYTES: u64 = 16;

pub(crate) struct WaterParticles {
    layout: GroupLayout,
    sampler: wgpu::Sampler,
    pipelines: Option<ParticlePipelines>,
    grid: Option<ParticleGrid>,
    last: Option<f64>,
    step: Option<u32>,
    generation: u64,
    readbacks: ReadbackRing<()>,
    tallies: Rc<Cell<[u32; TALLIES]>>,
}

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

struct ParticleGrid {
    texels: u32,
    texel: f32,
    square: u32,
    cut: u32,
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

#[derive(Clone, Copy)]
pub(crate) struct ParticleView {
    pub(crate) detail: Vec3,
    pub(crate) pixel_angle: f32,
    pub(crate) display: FrameSize,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ParticleStats {
    pub(crate) texels: u32,
    pub(crate) points: u32,
    pub(crate) cut: u32,
    pub(crate) reach: u32,
    pub(crate) foam: u32,
    pub(crate) waves: u32,
    pub(crate) born: u32,
    pub(crate) dropped: u32,
}

impl WaterParticles {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &WATER_PARTICLES),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/water-particles-sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
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

    pub(crate) fn prepare(
        &mut self,
        (device, queue): (&wgpu::Device, &crate::gpu_calls::Queue),
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
        let kept = self
            .grid
            .as_ref()
            .filter(|grid| grid.texel == texel && grid.texels >= asked && grid.texels < 2 * asked);
        let texels = kept.map_or(asked, |grid| grid.texels);
        let (sizes, cut) = sized(device, (texels, seen.display), settings.foam_points);
        let keeps = kept.is_some_and(|grid| {
            grid.square == sizes.square
                && grid.pools[0] >= sizes.pools[0]
                && grid.pools[0] <= 2 * sizes.pools[0]
        });
        if !keeps {
            self.grid = Some(ParticleGrid::new(device, &self.layout, texel, sizes));
            self.generation += 1;
        }
        let Some(grid) = self.grid.as_mut() else {
            return;
        };
        grid.cut = cut;
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
        let tick = step_seconds(settings.simulation_cell);
        let step = (seconds / f64::from(tick)).floor() as i64 as u32;
        let begun = self.step.unwrap_or(step);
        self.step = Some(step);
        let least = ((LONGEST_FRAME / f64::from(tick)).ceil() as u32).next_power_of_two();
        let held = grid.held.unwrap_or(window);
        let spacing = f64::from(texel) * f64::from(PARTICLE_LATTICE);
        let before = i64::from(grid.square / 2 + PARTICLE_MARGIN);
        let lattice = [seen.detail.x, seen.detail.z]
            .map(|at| (f64::from(at) / spacing).floor() as i64 - before);
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

    pub(crate) fn ensure_pipelines(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        (camera, water): (&GroupLayout, &GroupLayout),
    ) {
        if self.grid.is_none() || self.pipelines.is_some() {
            return;
        }
        let layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::WATER_PARTICLE_PIPELINES,
            &[camera, water, &self.layout],
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

    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        (camera, water): (&wgpu::BindGroup, &wgpu::BindGroup),
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
            self.bind(&mut pass, (camera, water), grid, &grid.groups[0]);
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
            self.bind(&mut pass, (camera, water), grid, &grid.groups[0]);
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
        encoder.copy_buffer_to_buffer(
            &grid.counts,
            u64::from(PARTICLE_DRAW) * 4,
            &grid.args,
            DRAW_ARGS,
            DRAW_BYTES,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/water-particles-splat"),
                timestamp_writes: span.map(|span| span.compute_part(false, true)),
            });
            self.bind(&mut pass, (camera, water), grid, &grid.groups[0]);
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

    fn bind(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        (camera, water): (&wgpu::BindGroup, &wgpu::BindGroup),
        grid: &ParticleGrid,
        group: &wgpu::BindGroup,
    ) {
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, water, &[0, grid.terrain, 0]);
        pass.set_bind_group(2, group, &[]);
    }

    pub(crate) fn draw_foam(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        (camera, water, behind): (&wgpu::BindGroup, &wgpu::BindGroup, &wgpu::BindGroup),
    ) {
        let Some(grid) = self.grid.as_ref().filter(|grid| grid.encoded.get()) else {
            return;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, water, &[0, grid.terrain, 0]);
        pass.set_bind_group(2, behind, &[]);
        pass.draw_indirect(&grid.args, DRAW_ARGS);
    }

    pub(crate) fn foam_buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        self.grid.as_ref().map(|grid| (&grid.frame, &grid.pool))
    }

    pub(crate) fn after_submit(&self) {
        let tallies = Rc::clone(&self.tallies);
        self.readbacks.after_submit(move |words, ()| {
            tallies.set(std::array::from_fn(|index| words.get(index)));
        });
    }

    pub(crate) fn view(&self) -> Option<&wgpu::TextureView> {
        self.grid.as_ref().map(|grid| &grid.view)
    }

    pub(crate) fn spread_view(&self) -> Option<&wgpu::TextureView> {
        self.grid.as_ref().map(|grid| &grid.spread_view)
    }

    pub(crate) fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }

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

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn stats(&self) -> ParticleStats {
        let Some(grid) = &self.grid else {
            return ParticleStats::default();
        };
        let tallies = self.tallies.get();
        let tally = |word: u32| tallies[word as usize];
        let spacing = grid.texel * PARTICLE_LATTICE as f32;
        ParticleStats {
            texels: grid.texels,
            points: grid.square,
            cut: grid.cut,
            reach: (0.5 * grid.square as f32 * spacing).round() as u32,
            foam: tally(PARTICLE_FOAM_LIVE),
            waves: tally(PARTICLE_WAVES_LIVE),
            born: tally(PARTICLE_FOAM_BORN) + tally(PARTICLE_WAVES_BORN),
            dropped: tally(PARTICLE_FOAM_DROPPED) + tally(PARTICLE_WAVES_DROPPED),
        }
    }

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
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn new(device: &wgpu::Device, layout: &GroupLayout, texel: f32, sizes: ParticleSizes) -> Self {
        let ParticleSizes {
            texels,
            square,
            pools,
        } = sizes;
        let points = lattice_points(square);
        let area = u64::from(texels) * u64::from(texels);
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
        let args = crate::memory::buffer(
            device,
            &wgpu::BufferDescriptor {
                label: Some("engine/water-particles-args"),
                size: DRAW_ARGS + DRAW_BYTES,
                usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
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
            square,
            cut: 0,
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

#[derive(Clone, Copy)]
struct ParticleSizes {
    texels: u32,
    square: u32,
    pools: [u32; 2],
}

fn sized(
    device: &wgpu::Device,
    (texels, display): (u32, FrameSize),
    asked: u32,
) -> (ParticleSizes, u32) {
    let limits = device.limits();
    let binding = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let items = u64::from(limits.max_compute_workgroups_per_dimension) * u64::from(PARTICLE_GROUP);
    let held = (binding as f64 / POINT_BYTES as f64).sqrt() as u32;
    let shared = (items as f64 / 2.0).sqrt() as u32;
    let most = held.min(shared).saturating_sub(lattice_points(0)) / 2 * 2;
    let asked = asked.next_multiple_of(2);
    let square = asked.min(most).max(texels / PARTICLE_LATTICE + 2);
    let sites = u64::from(lattice_points(square)).pow(2);
    let waves = u64::from(texels).pow(2) / u64::from(WAVE_TEXELS);
    let shown = u64::from(display.width) * u64::from(display.height);
    let foam = (shown / u64::from(FOAM_PIXELS))
        .min((binding / (2 * PARTICLE_BYTES)).saturating_sub(waves))
        .min(items.saturating_sub(waves + sites));
    let sizes = ParticleSizes {
        texels,
        square,
        pools: [foam as u32, waves as u32],
    };
    (sizes, asked.saturating_sub(square))
}

fn lattice_points(square: u32) -> u32 {
    square + 2 * PARTICLE_MARGIN + 1
}

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
