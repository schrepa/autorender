use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::max;
use engine_core::{Entry, Error, Var};
use engine_heightfield::Fall;
use engine_places::ground::{GroundTiles, Heightfield};
use engine_render::{
    Access, Attach, Bind, Buffer, ComputePipeline, Count, DepthUse, DrawInput, Encode, GpuPart,
    Install, Layer, Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, SamplerDesc, Shape,
    Texture, VertexLayout, ViewTarget, wgpu,
};
use glam::Vec3;

use crate::Water;
use crate::gpu::{FallVertex, WATER_LEAF_FIRST, WaterUniform};
use crate::state::WaterState;
use crate::water_blocks::{BlocksGpu, INFO_FORMAT, SURFACE_FORMAT};
use crate::water_falls::{fall_layout, sheets};
use crate::water_particles::ParticlesGpu;
use crate::water_settle::{SettleFeedback, SettleGpu};
use crate::water_sim::SimGpu;

const FOAM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

type BuiltFalls = (Entity, Vec3, Arc<[Fall]>, Option<u32>, Range<u32>);

pub(crate) fn copied<P>(bound: &Bind<P>) -> Bind<P> {
    match bound {
        Bind::Texture(texture) => Bind::Texture(*texture),
        Bind::Slice {
            texture,
            mips,
            layers,
        } => Bind::Slice {
            texture: *texture,
            mips: mips.clone(),
            layers: layers.clone(),
        },
        Bind::Buffer(buffer) => Bind::Buffer(*buffer),
        Bind::Sampler(sampler) => Bind::Sampler(*sampler),
        Bind::Target(view) => Bind::Target(*view),
        Bind::Asset => Bind::Asset,
    }
}

pub(crate) fn reached(
    cx: &mut Install<'_, WaterGpu>,
    entries: &[Entry],
    all: Vec<(Var, Bind<WaterGpu>)>,
) -> Vec<(Var, Bind<WaterGpu>)> {
    all.into_iter()
        .filter(|(var, _)| entries.iter().any(|&entry| cx.reaches(entry, *var)))
        .collect()
}

fn pipelines(
    cx: &mut Install<'_, WaterGpu>,
    (surface_pass, foam_pass): (Pass<WaterGpu>, Pass<WaterGpu>),
    behind: &dyn Fn() -> Vec<(Var, Bind<WaterGpu>)>,
    foam_all: Vec<(Var, Bind<WaterGpu>)>,
) -> Result<[Pipeline<WaterGpu, Own>; 3], Error> {
    use crate::shaders::water::{debug, falls, particles, surface};
    let blended = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    };
    let primitive = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        cull_mode: None,
        ..Default::default()
    };
    let entries = [surface::VS_WATER, surface::FS_WATER, debug::FS_DEBUG_WATER];
    let drawing_bound = reached(cx, &entries, behind());
    let drawing = cx.render_pipeline(
        surface_pass,
        &RenderPipeline {
            vertex: surface::VS_WATER,
            fragment: Some(surface::FS_WATER),
            debug: Some(debug::FS_DEBUG_WATER),
            vertices: &[],
            primitive,
            depth: DepthUse::Write {
                bias: wgpu::DepthBiasState::default(),
            },
            blend: Some(blended),
            writes: wgpu::ColorWrites::ALL,
        },
        &drawing_bound,
    )?;
    let entries = [
        falls::VS_FALL_TEMPORAL,
        falls::FS_FALL,
        debug::FS_DEBUG_FALL,
    ];
    let falling_bound = reached(cx, &entries, behind());
    let falling = cx.render_pipeline(
        surface_pass,
        &RenderPipeline {
            vertex: falls::VS_FALL_TEMPORAL,
            fragment: Some(falls::FS_FALL),
            debug: Some(debug::FS_DEBUG_FALL),
            vertices: &[VertexLayout::Buffer(fall_layout())],
            primitive,
            depth: DepthUse::Write {
                bias: wgpu::DepthBiasState::default(),
            },
            blend: Some(blended),
            writes: wgpu::ColorWrites::ALL,
        },
        &falling_bound,
    )?;
    let added = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    let foam_bound = reached(cx, &[particles::VS_FOAM, particles::FS_FOAM], foam_all);
    let foaming = cx.render_pipeline(
        foam_pass,
        &RenderPipeline {
            vertex: particles::VS_FOAM,
            fragment: Some(particles::FS_FOAM),
            debug: None,
            vertices: &[],
            primitive,
            depth: DepthUse::None,
            blend: Some(wgpu::BlendState {
                color: added,
                alpha: added,
            }),
            writes: wgpu::ColorWrites::ALL,
        },
        &foam_bound,
    )?;
    Ok([drawing, falling, foaming])
}

pub(crate) struct WaterGpu {
    uniform: Buffer<Self>,
    foam: Texture<Self>,
    leaf_args: Buffer<Self>,
    leaf_pattern: Buffer<Self>,
    fall_vertices: Buffer<Self>,
    fall_indices: Buffer<Self>,
    count_leaves: Pass<Self>,
    foam_pass: Pass<Self>,
    surface: Pass<Self>,
    counting: ComputePipeline<Self>,
    drawing: Pipeline<Self, Own>,
    falling: Pipeline<Self, Own>,
    foaming: Pipeline<Self, Own>,
    leaves: DrawInput<Self, Own>,
    pattern: DrawInput<Self, Own>,
    sheet_vertices: DrawInput<Self, Own>,
    sheet_indices: DrawInput<Self, Own>,
    blocks: BlocksGpu,
    settle: SettleGpu,
    sim: SimGpu,
    particles: ParticlesGpu,
    built: Vec<BuiltFalls>,
    vertices: Vec<FallVertex>,
    indices: Vec<u32>,
    pattern_written: Option<Arc<[u32]>>,
    drawn: bool,
}

impl GpuPart for WaterGpu {
    type Reads = (
        Res<'static, Water>,
        Res<'static, WaterState>,
        Res<'static, SettleFeedback>,
        Res<'static, engine_core::View>,
        Res<'static, GroundTiles>,
        Res<'static, Heightfield>,
    );
    const NAME: &'static str = "water";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::water::{faces::water as face, foam, refraction, surface};
        let uniform = cx.buffer("water", Lifetime::Kept);
        let foam_texture = cx.texture("water_foam", FOAM_FORMAT, Lifetime::Frame);
        let leaf_args = cx.buffer("water_leaf_args", Lifetime::Frame);
        let leaf_pattern = cx.buffer("water_leaf_pattern", Lifetime::Kept);
        let fall_vertices = cx.buffer("water_falls", Lifetime::Kept);
        let fall_indices = cx.buffer("water_fall_indices", Lifetime::Kept);
        let refraction_sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(
            refraction::REFRACTION_SAMPLER,
            Bind::Sampler(refraction_sampler),
        )?;
        let blocks = BlocksGpu::new(
            cx.texture("water_surfaces", SURFACE_FORMAT, Lifetime::Kept),
            cx.texture("water_info", INFO_FORMAT, Lifetime::Kept),
            cx.buffer("water_directory", Lifetime::Kept),
        );
        cx.frame(face::WATER_SURFACE, Bind::Texture(blocks.surface))?;
        cx.frame(face::WATER_INFO, Bind::Texture(blocks.info))?;
        cx.frame(face::WATER_DIRECTORY, Bind::Buffer(blocks.directory))?;
        let settle = SettleGpu::install(cx)?;
        let sim = SimGpu::install(cx)?;
        let shared = vec![
            (surface::WATER, Bind::Buffer(uniform)),
            (surface::WATER_LIVE, Bind::Buffer(sim.looks())),
        ];
        let mut particles_gpu = ParticlesGpu::install(cx, sim.after(), &shared)?;
        let count_leaves = cx.pass("water_leaf_args").id();
        let foam_pass = cx
            .pass("water_foam")
            .at(Layer::Media)
            .after(particles_gpu.after())
            .color(foam_texture, Access::Clear)
            .label("engine/water-foam")
            .id();
        let surface_pass = cx
            .pass("water")
            .at(Layer::Media)
            .after(foam_pass)
            .after(count_leaves)
            .label("engine/water")
            .span("water")
            .id();
        let counting_bound = reached(
            cx,
            &[surface::CS_WATER_LEAF_ARGS],
            vec![
                (surface::WATER, Bind::Buffer(uniform)),
                (surface::WATER_LEAF_ARGS, Bind::Buffer(leaf_args)),
            ],
        );
        let counting =
            cx.compute_pipeline(count_leaves, surface::CS_WATER_LEAF_ARGS, &counting_bound)?;
        let behind = || {
            let mut all = vec![
                (surface::WATER, Bind::Buffer(uniform)),
                (surface::WATER_LIVE, Bind::Buffer(sim.looks())),
                (surface::WATER_LEAF_SPANS, Bind::Buffer(leaf_args)),
                (surface::WATER_DEPTH, Bind::Target(ViewTarget::Depth)),
                (surface::WATER_SCENE, Bind::Target(ViewTarget::Scene)),
                (foam::WATER_FOAM_DRAWN, Bind::Texture(foam_texture)),
            ];
            all.extend(particles_gpu.bound());
            all
        };
        let mut foam_all = vec![
            (surface::WATER, Bind::Buffer(uniform)),
            (surface::WATER_LIVE, Bind::Buffer(sim.looks())),
            (surface::WATER_DEPTH, Bind::Target(ViewTarget::Depth)),
        ];
        foam_all.extend(particles_gpu.bound());
        let [drawing, falling, foaming] =
            pipelines(cx, (surface_pass, foam_pass), &behind, foam_all)?;
        particles_gpu.foam_args = Some(cx.draw_input(foam_pass, particles_gpu.args()));
        let leaves = cx.draw_input(surface_pass, leaf_args);
        let pattern = cx.draw_input(surface_pass, leaf_pattern);
        let sheet_vertices = cx.draw_input(surface_pass, fall_vertices);
        let sheet_indices = cx.draw_input(surface_pass, fall_indices);
        Ok(Self {
            uniform,
            foam: foam_texture,
            leaf_args,
            leaf_pattern,
            fall_vertices,
            fall_indices,
            count_leaves,
            foam_pass,
            surface: surface_pass,
            counting,
            drawing,
            falling,
            foaming,
            leaves,
            pattern,
            sheet_vertices,
            sheet_indices,
            blocks,
            settle,
            sim,
            particles: particles_gpu,
            built: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            pattern_written: None,
            drawn: false,
        })
    }

    fn prepare(
        &mut self,
        (water, state, feedback, view, ground, heightfield): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if cx.fresh() {
            self.forget();
        }
        let pattern = &heightfield.leaf_indices;
        let fresh = self
            .pattern_written
            .as_ref()
            .is_none_or(|written| !Arc::ptr_eq(written, pattern));
        if fresh {
            cx.allocate_buffer(
                self.leaf_pattern,
                std::mem::size_of_val(&pattern[..]) as u64,
            );
            if !pattern.is_empty() {
                cx.write(self.leaf_pattern, 0, &pattern[..]);
            }
            self.pattern_written = Some(Arc::clone(pattern));
        }
        cx.allocate_buffer(self.uniform, size_of::<WaterUniform>() as u64);
        self.settle.prepare(&feedback.work, cx);
        self.blocks.prepare(&state.blocks, cx);
        self.sim.prepare(&state.live.work, cx);
        self.particles.prepare(&state.particles, cx);
        let watering = water.enabled && !state.watered.is_empty();
        self.prepare_falls(&state, &ground, water.enabled, cx);
        let terrains = ground.tables.len() as u32;
        cx.allocate_buffer(
            self.leaf_args,
            u64::from(WATER_LEAF_FIRST + 2 * terrains.max(1)) * 4,
        );
        let size = view.display_size();
        cx.allocate(
            self.foam,
            wgpu::Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: 1,
            },
            1,
        );
        self.drawn = watering;
        if !self.drawn {
            return;
        }
        let running = state.live.work.drawn.as_ref().map(|drawn| {
            let window = &drawn.window;
            let [wrap_x, wrap_z] = drawn.wrap;
            (
                [
                    window.origin[0],
                    window.origin[1],
                    window.cell,
                    window.cells as f32,
                ],
                [
                    window.datum,
                    wrap_x as f32,
                    wrap_z as f32,
                    drawn.pool as f32,
                ],
                [window.pixels, 0.0, 0.0, 0.0],
            )
        });
        let [red, green, blue] = water.color;
        let sediment = water.sediment_color;
        let uniform = WaterUniform {
            color: [red, green, blue, water.clarity],
            absorption: absorption(water.color, water.clarity),
            foam: state.particles.shading(),
            sim: running.map_or([0.0; 4], |(square, _, _)| square),
            sim_datum: running.map_or([0.0; 4], |(_, datum, _)| datum),
            sim_pixels: running.map_or([0.0; 4], |(_, _, pixels)| pixels),
            sediment: [sediment[0], sediment[1], sediment[2], water.sediment],
            lift: [
                heightfield.subdivision_error,
                water.subdivision_error,
                0.0,
                0.0,
            ],
            terrains: [terrains, 0, 0, 0],
        };
        cx.write(self.uniform, 0, &[uniform]);
        cx.run(self.count_leaves);
        cx.run(self.foam_pass);
        cx.run(self.surface);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if self.settle.encode(pass, cx)
            || self.sim.encode(pass, cx)
            || self.particles.encode(pass, cx)
        {
            return 0;
        }
        if pass == self.count_leaves {
            cx.dispatch(&self.counting, Count::Groups([1, 1, 1]));
            return 0;
        }
        if pass == self.foam_pass {
            let mut draw = cx.render(Attach::Color(self.foam, None), None);
            self.particles.draw_foam(&mut draw, &self.foaming);
            return 0;
        }
        if pass != self.surface || !self.drawn {
            return 0;
        }
        let mut draw = cx.render(Attach::Declared, None);
        let mut draws = 0;
        draw.draw(
            &self.drawing,
            Shape::Indirect {
                vertices: &[],
                indices: Some(self.pattern),
                args: self.leaves,
                offset: 0,
            },
        );
        draws += 1;
        for (_, _, _, drawn, run) in &self.built {
            if drawn.is_none() || run.is_empty() {
                continue;
            }
            draw.draw(
                &self.falling,
                Shape::Buffers {
                    vertices: &[self.sheet_vertices],
                    indices: Some(self.sheet_indices),
                    elements: run.clone(),
                    instances: 0..1,
                },
            );
            draws += 1;
        }
        draws
    }
}

impl WaterGpu {
    fn forget(&mut self) {
        self.pattern_written = None;
        self.built.clear();
        self.blocks.forget();
        self.settle.forget();
        self.sim.forget();
        self.particles.forget();
    }

    fn prepare_falls(
        &mut self,
        state: &WaterState,
        ground: &GroundTiles,
        enabled: bool,
        cx: &mut Prepare<'_, Self>,
    ) {
        let index_of = |entity: Entity| {
            ground
                .tables
                .iter()
                .find(|(terrain, _)| *terrain == entity)
                .map(|&(_, index)| index)
        };
        let falling = || {
            state
                .terrains
                .iter()
                .filter(move |terrain| enabled && !terrain.falls.is_empty())
                .map(|terrain| {
                    let drawn = index_of(terrain.entity)
                        .filter(|_| state.watered.contains(&terrain.entity));
                    (terrain, drawn)
                })
        };
        let same = self.built.len() == falling().count()
            && self.built.iter().zip(falling()).all(|(a, (b, drawn))| {
                a.0 == b.entity && a.1 == b.origin && Arc::ptr_eq(&a.2, &b.falls) && a.3 == drawn
            });
        if same || (self.built.is_empty() && falling().next().is_none()) {
            return;
        }
        let runs = sheets(
            falling().map(|(terrain, drawn)| (terrain.origin, &*terrain.falls, drawn.unwrap_or(0))),
            &mut self.vertices,
            &mut self.indices,
        );
        self.built = falling()
            .zip(runs)
            .map(|((terrain, drawn), run)| {
                (
                    terrain.entity,
                    terrain.origin,
                    terrain.falls.clone(),
                    drawn,
                    run,
                )
            })
            .collect();
        if self.indices.is_empty() {
            return;
        }
        cx.allocate_buffer(
            self.fall_vertices,
            std::mem::size_of_val(&self.vertices[..]) as u64,
        );
        cx.write(self.fall_vertices, 0, &self.vertices);
        cx.allocate_buffer(
            self.fall_indices,
            std::mem::size_of_val(&self.indices[..]) as u64,
        );
        cx.write(self.fall_indices, 0, &self.indices);
    }
}

fn absorption(color: [f32; 3], clarity: f32) -> [f32; 4] {
    const LUMINANCE: [f32; 3] = [0.2126, 0.7152, 0.0722];
    let luminance = max(
        color[0] * LUMINANCE[0] + color[1] * LUMINANCE[1] + color[2] * LUMINANCE[2],
        1.0e-4,
    );
    let weights = color.map(|channel| luminance / max(channel, luminance * 0.125));
    let mean = weights[0] * LUMINANCE[0] + weights[1] * LUMINANCE[1] + weights[2] * LUMINANCE[2];
    let rate = std::f32::consts::LN_2 / max(clarity, 0.01) / mean;
    [weights[0] * rate, weights[1] * rate, weights[2] * rate, 0.0]
}
