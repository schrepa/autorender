use std::any::TypeId;
use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::max;
use engine_core::{CookMode, Error, FrameTime, View};
use engine_heightfield::Fall;
use engine_places::ground::GroundTiles;
use engine_places::water::WaterObstacles;
use engine_places::wind::Wind;
use engine_render::{
    Access, Attach, Bind, Buffer, DebugShown, DepthUse, DrawInput, Encode, GpuPart, Install, Layer,
    Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, SamplerDesc, Shape, Texture,
    VertexLayout, ViewTarget, wgpu,
};
use glam::Vec3;

use crate::gpu::{FallVertex, WaterUniform};
use crate::state::WaterState;
use crate::water_blocks::WaterBlocks;
use crate::water_falls::{fall_layout, sheets};
use crate::water_particles::{ParticleView, WaterParticles};
use crate::water_settle::WaterSettle;
use crate::water_sim::WaterSim;
use crate::{Water, WaterBrick};

const FOAM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

const LEAF_ARGS_BYTES: u64 = 20;

type BuiltFalls = (Entity, Vec3, Arc<[Fall]>, Range<u32>);

pub(crate) struct WaterGpu {
    uniform: Buffer<Self>,
    foam: Texture<Self>,
    leaf_args: Buffer<Self>,
    leaf_pattern: Buffer<Self>,
    fall_vertices: Buffer<Self>,
    fall_indices: Buffer<Self>,
    clear_args: Pass<Self>,
    count_leaves: Pass<Self>,
    foam_pass: Pass<Self>,
    surface: Pass<Self>,
    counting: engine_render::ComputePipeline<Self>,
    drawing: Pipeline<Self, Own>,
    falling: Pipeline<Self, Own>,
    foaming: Pipeline<Self, Own>,
    leaves: DrawInput<Self, Own>,
    pattern: DrawInput<Self, Own>,
    sheet_vertices: DrawInput<Self, Own>,
    sheet_indices: DrawInput<Self, Own>,
    blocks: WaterBlocks,
    settle: WaterSettle,
    sim: WaterSim,
    particles: WaterParticles,
    built: Vec<BuiltFalls>,
    vertices: Vec<FallVertex>,
    indices: Vec<u32>,
    drawn: bool,
}

impl GpuPart for WaterGpu {
    type Reads = (
        Res<'static, Water>,
        Res<'static, WaterState>,
        Res<'static, WaterObstacles>,
        Res<'static, Wind>,
        Res<'static, FrameTime>,
        Res<'static, DebugShown>,
        Res<'static, CookMode>,
        Res<'static, View>,
        Res<'static, GroundTiles>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::water::{debug, falls, faces::water as face, particles, refraction, surface};
        let uniform = cx.buffer("water", Lifetime::Kept);
        let foam = cx.texture("water_foam", FOAM_FORMAT, Lifetime::Frame);
        let leaf_args = cx.buffer("water_leaf_args", Lifetime::Frame);
        let leaf_pattern = cx.buffer("water_leaf_pattern", Lifetime::Kept);
        let fall_vertices = cx.buffer("water_falls", Lifetime::Kept);
        let fall_indices = cx.buffer("water_fall_indices", Lifetime::Kept);
        let refraction_sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(refraction::REFRACTION_SAMPLER, Bind::Sampler(refraction_sampler))?;
        let blocks = WaterBlocks::install(cx)?;
        cx.frame(face::WATER_SURFACE, Bind::Texture(blocks.surface()))?;
        cx.frame(face::WATER_INFO, Bind::Texture(blocks.info()))?;
        cx.frame(face::WATER_DIRECTORY, Bind::Buffer(blocks.directory()))?;
        let settle = WaterSettle::install(cx)?;
        let sim = WaterSim::install(cx, settle.after())?;
        let particles = WaterParticles::install(cx, sim.after(), uniform)?;
        let clear_args = cx.clear("water_clear_leaf_args", leaf_args);
        let count_leaves = cx.pass("water_leaf_args").after(clear_args).id();
        let foam_pass = cx
            .pass("water_foam")
            .after(particles.after())
            .color(foam, Access::Clear)
            .id();
        let surface_pass = cx.pass("water").at(Layer::Media).after(foam_pass).id();
        let counting = cx.compute_pipeline(
            count_leaves,
            surface::CS_WATER_LEAF_ARGS,
            &[(surface::WATER_LEAF_ARGS, Bind::Buffer(leaf_args))],
        )?;
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
        let behind = [
            (surface::WATER_DEPTH, Bind::Target(ViewTarget::Depth)),
            (surface::WATER_SCENE, Bind::Target(ViewTarget::Scene)),
            (surface::WATER_FOAM_DRAWN, Bind::Texture(foam)),
        ];
        let mut bound = vec![
            (surface::WATER, Bind::Buffer(uniform)),
            (surface::WATER_LIVE, Bind::Buffer(sim.looks())),
        ];
        bound.extend(particles.bound());
        bound.extend(behind);
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
            &bound,
        )?;
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
            &bound,
        )?;
        let added = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let mut foam_bound = vec![(surface::WATER, Bind::Buffer(uniform))];
        foam_bound.extend(particles.bound());
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
        let leaves = cx.draw_input(surface_pass, leaf_args);
        let pattern = cx.draw_input(surface_pass, leaf_pattern);
        let sheet_vertices = cx.draw_input(surface_pass, fall_vertices);
        let sheet_indices = cx.draw_input(surface_pass, fall_indices);
        Ok(Self {
            uniform,
            foam,
            leaf_args,
            leaf_pattern,
            fall_vertices,
            fall_indices,
            clear_args,
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
            particles,
            built: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
            drawn: false,
        })
    }

    fn prepare(
        &mut self,
        (water, state, obstacles, wind, time, shown, cooking, view, held): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        if water.is_changed() {
            let pattern = crate::water_blocks::leaf_pattern();
            cx.allocate_buffer(self.leaf_pattern, std::mem::size_of_val(pattern) as u64);
            cx.write(self.leaf_pattern, 0, pattern);
            cx.allocate_buffer(self.leaf_args, LEAF_ARGS_BYTES);
        }
        self.settle.prepare(&state, cooking.cooking, cx);
        self.prepare_falls(&state, cx);
        self.blocks.prepare(&state, &held, cx);
        self.sim
            .prepare(&state, &obstacles, view.detail_eye(), time.presented_seconds, cx);
        let size = view.render_size();
        let seen = ParticleView::of(&view);
        let ridden = self
            .sim
            .drawn()
            .map_or(water.simulation_cell, |drawn| drawn.window.cell);
        self.particles.prepare(
            &state,
            seen,
            (&water, ridden, wind.velocity, time.presented_seconds),
            cx,
        );
        self.drawn = !state.terrains.is_empty() && !cooking.cooking;
        if !self.drawn {
            return;
        }
        let running = self.sim.drawn().map(|drawn| {
            let window = drawn.window;
            let [wrap_x, wrap_z] = drawn.wrap;
            (
                [
                    window.origin[0],
                    window.origin[1],
                    window.cell,
                    window.cells as f32,
                ],
                [window.datum, wrap_x as f32, wrap_z as f32, drawn.pool as f32],
                [window.pixels, 0.0, 0.0, 0.0],
            )
        });
        let [red, green, blue] = water.color;
        let sediment = water.sediment_color;
        let mine = shown.owner == Some(TypeId::of::<WaterBrick>());
        let uniform = WaterUniform {
            color: [red, green, blue, water.clarity],
            absorption: absorption(water.color, water.clarity),
            foam: self.particles.shading(),
            sim: running.map_or([0.0; 4], |(square, _, _)| square),
            sim_datum: running.map_or([0.0; 4], |(_, datum, _)| datum),
            sim_pixels: running.map_or([0.0; 4], |(_, _, pixels)| pixels),
            sediment: [sediment[0], sediment[1], sediment[2], water.sediment],
            lift: self.blocks.lift(),
            debug: [u32::from(mine), shown.local, 0, 0],
        };
        cx.write(self.uniform, 0, &[uniform]);
        cx.allocate(
            self.foam,
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            1,
        );
        cx.run(self.clear_args);
        cx.run(self.count_leaves);
        cx.run(self.foam_pass);
        cx.run(self.surface);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if let Some(draws) = self.settle.encode(pass, cx) {
            return draws;
        }
        if let Some(draws) = self.sim.encode(pass, cx) {
            return draws;
        }
        if let Some(draws) = self.particles.encode(pass, cx) {
            return draws;
        }
        if pass == self.count_leaves {
            return cx.dispatch(&self.counting, engine_render::Count::Groups([1, 1, 1]));
        }
        if pass == self.foam_pass {
            let mut draw = cx.render(Attach::Color(self.foam, None), None);
            return self.particles.draw_foam(&mut draw, &self.foaming);
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
        for (_, _, _, run) in &self.built {
            if run.is_empty() {
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
    fn prepare_falls(&mut self, state: &WaterState, cx: &mut Prepare<'_, Self>) {
        let falling = || {
            state
                .terrains
                .iter()
                .filter(|terrain| !terrain.falls.is_empty())
        };
        let same = self.built.len() == falling().count()
            && self
                .built
                .iter()
                .zip(falling())
                .all(|(a, b)| a.0 == b.entity && a.1 == b.origin && Arc::ptr_eq(&a.2, &b.falls));
        if same {
            return;
        }
        let runs = sheets(
            falling().map(|terrain| (terrain.origin, &*terrain.falls)),
            &mut self.vertices,
            &mut self.indices,
        );
        self.built = falling()
            .zip(runs)
            .map(|(terrain, run)| (terrain.entity, terrain.origin, terrain.falls.clone(), run))
            .collect();
        if self.indices.is_empty() {
            self.built.clear();
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
