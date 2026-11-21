use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{Entry, Error, Var};
use engine_render::{ComputePipeline, Encode, PerBatch, Scene};

use super::shaders::world::{grass, mesh, particles, sim, sky};
use super::{
    GrassBlade, MAX_BLADES, MAX_PARTICLES, MeshVertex, POINT_LIGHT_COUNT, Particle, PointLight,
    WAKE_SIDE, WakeCell, WorldUniforms,
};
use crate::mesh::{WorldMeshes, build_world_meshes};
use crate::names::COLOR_PASS;
use crate::state::VerdantState;

const INIT_PASS: &str = "verdant_sim_init";
const SIM_PASS: &str = "verdant_sim_step";

pub(super) struct WorldGpu {
    init_pass: engine_render::Pass<Self>,
    sim_pass: engine_render::Pass<Self>,
    init_grass: ComputePipeline<Self>,
    init_particles: ComputePipeline<Self>,
    wake_sim: ComputePipeline<Self>,
    particle_sim: ComputePipeline<Self>,
    mesh_opaque: Pipeline<Self, Scene>,
    mesh_foliage: Pipeline<Self, Scene>,
    grass_pipe: Pipeline<Self, Scene>,
    sky_pipe: Pipeline<Self, Scene>,
    particles_pipe: Pipeline<Self, Scene>,
    uniforms: Buffer<Self>,
    mesh_verts: Buffer<Self>,
    mesh_opaque_idx: Buffer<Self>,
    mesh_foliage_idx: Buffer<Self>,
    blades: Buffer<Self>,
    wake: Buffer<Self>,
    particles: Buffer<Self>,
    lights: Buffer<Self>,
    meshes: Option<WorldMeshes>,
    opaque_count: u32,
    foliage_count: u32,
    active_blades: u32,
    grass_inited: bool,
    particles_inited: bool,
}

fn pick_bound(
    cx: &mut Install<'_, WorldGpu>,
    entries: &[Entry],
    all: &[(Var, Bind<WorldGpu>)],
) -> Vec<(Var, Bind<WorldGpu>)> {
    all.iter()
        .filter(|(var, _)| entries.iter().any(|entry| cx.reaches(*entry, *var)))
        .map(|(var, bind)| (*var, bind.clone()))
        .collect()
}

fn depth_state(write: bool) -> DepthUse {
    DepthUse::Compare {
        compare: wgpu::CompareFunction::GreaterEqual,
        write,
        bias: wgpu::DepthBiasState::default(),
    }
}

fn scene_pipe<'a>(
    vertex: Entry,
    fragment: Entry,
    primitive: wgpu::PrimitiveState,
    depth: DepthUse,
    targets: &'a [(Option<wgpu::BlendState>, wgpu::ColorWrites)],
) -> RenderPipeline<'a> {
    RenderPipeline {
        vertex,
        fragment: Some(fragment),
        vertices: &[],
        primitive,
        depth,
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets,
        constants: &[],
        per_batch: PerBatch::NONE,
    }
}

struct WorldBuffers {
    uniforms: Buffer<WorldGpu>,
    mesh_verts: Buffer<WorldGpu>,
    mesh_opaque_idx: Buffer<WorldGpu>,
    mesh_foliage_idx: Buffer<WorldGpu>,
    blades: Buffer<WorldGpu>,
    wake: Buffer<WorldGpu>,
    particles: Buffer<WorldGpu>,
    lights: Buffer<WorldGpu>,
}

struct ScenePipelines {
    mesh_opaque: Pipeline<WorldGpu, Scene>,
    mesh_foliage: Pipeline<WorldGpu, Scene>,
    grass_pipe: Pipeline<WorldGpu, Scene>,
    sky_pipe: Pipeline<WorldGpu, Scene>,
    particles_pipe: Pipeline<WorldGpu, Scene>,
}

fn install_scene_pipelines(
    cx: &mut Install<'_, WorldGpu>,
    buf: &WorldBuffers,
) -> Result<ScenePipelines, Error> {
    let scene = Scene::named(COLOR_PASS);
    let opaque_target = [(None, wgpu::ColorWrites::ALL)];
    let alpha_blend = Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent::OVER,
    });
    let alpha_target = [(alpha_blend, wgpu::ColorWrites::ALL)];

    let mesh_opaque_binds = [
        (mesh::WORLD, Bind::Buffer(buf.uniforms)),
        (mesh::MESH_VERTS, Bind::Buffer(buf.mesh_verts)),
        (mesh::MESH_INDICES, Bind::Buffer(buf.mesh_opaque_idx)),
        (mesh::LIGHTS, Bind::Buffer(buf.lights)),
    ];
    let mesh_foliage_binds = [
        (mesh::WORLD, Bind::Buffer(buf.uniforms)),
        (mesh::MESH_VERTS, Bind::Buffer(buf.mesh_verts)),
        (mesh::MESH_INDICES, Bind::Buffer(buf.mesh_foliage_idx)),
        (mesh::LIGHTS, Bind::Buffer(buf.lights)),
    ];
    let back_cull = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: Some(wgpu::Face::Back),
        ..wgpu::PrimitiveState::default()
    };
    let no_cull = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        ..wgpu::PrimitiveState::default()
    };
    let strip = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleStrip,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        ..wgpu::PrimitiveState::default()
    };

    let b_mo = pick_bound(cx, &[mesh::VS_MESH, mesh::FS_MESH], &mesh_opaque_binds);
    let mesh_opaque = cx.render_pipeline(
        scene,
        &scene_pipe(mesh::VS_MESH, mesh::FS_MESH, back_cull, depth_state(true), &opaque_target),
        &b_mo,
    )?;

    let b_mf = pick_bound(cx, &[mesh::VS_MESH, mesh::FS_FOLIAGE], &mesh_foliage_binds);
    let mesh_foliage = cx.render_pipeline(
        scene,
        &scene_pipe(mesh::VS_MESH, mesh::FS_FOLIAGE, no_cull, depth_state(true), &opaque_target),
        &b_mf,
    )?;

    let grass_binds = [
        (grass::WORLD, Bind::Buffer(buf.uniforms)),
        (grass::BLADES, Bind::Buffer(buf.blades)),
        (grass::WAKE, Bind::Buffer(buf.wake)),
        (grass::LIGHTS, Bind::Buffer(buf.lights)),
    ];
    let b_gr = pick_bound(cx, &[grass::VS_GRASS, grass::FS_GRASS], &grass_binds);
    let grass_pipe = cx.render_pipeline(
        scene,
        &scene_pipe(grass::VS_GRASS, grass::FS_GRASS, strip, depth_state(true), &opaque_target),
        &b_gr,
    )?;

    let sky_binds = [(sky::WORLD, Bind::Buffer(buf.uniforms))];
    let b_sk = pick_bound(cx, &[sky::VS_SKY, sky::FS_SKY], &sky_binds);
    let sky_pipe = cx.render_pipeline(
        scene,
        &scene_pipe(sky::VS_SKY, sky::FS_SKY, no_cull, depth_state(false), &opaque_target),
        &b_sk,
    )?;

    let part_binds = [
        (particles::WORLD, Bind::Buffer(buf.uniforms)),
        (particles::PARTICLES, Bind::Buffer(buf.particles)),
    ];
    let b_pt = pick_bound(cx, &[particles::VS_PARTICLE, particles::FS_PARTICLE], &part_binds);
    let particles_pipe = cx.render_pipeline(
        scene,
        &scene_pipe(particles::VS_PARTICLE, particles::FS_PARTICLE, no_cull, depth_state(false), &alpha_target),
        &b_pt,
    )?;

    Ok(ScenePipelines {
        mesh_opaque,
        mesh_foliage,
        grass_pipe,
        sky_pipe,
        particles_pipe,
    })
}

fn build_uniforms(state: &VerdantState) -> WorldUniforms {
    let mood = state.mood();
    WorldUniforms {
        time_wind: [state.sim_time, state.dt, state.wind_strength, state.active_blades() as f32],
        camera_pos: [state.camera_eye.x, state.camera_eye.y, state.camera_eye.z, 0.0],
        camera_fwd: [state.camera_forward.x, state.camera_forward.y, state.camera_forward.z, 0.0],
        camera_right: [state.camera_right.x, state.camera_right.y, state.camera_right.z, state.time_phase],
        camera_up: [state.camera_up.x, state.camera_up.y, state.camera_up.z, mood.moon_factor],
        sun_dir: [mood.sun_dir.x, mood.sun_dir.y, mood.sun_dir.z, mood.star_power],
        sun_color: [mood.sun_color.x, mood.sun_color.y, mood.sun_color.z, mood.shaft_strength],
        sky_zenith: [mood.sky_zenith.x, mood.sky_zenith.y, mood.sky_zenith.z, mood.lantern_power],
        sky_horizon: [mood.sky_horizon.x, mood.sky_horizon.y, mood.sky_horizon.z, mood.firefly_power],
        haze_color: [mood.haze_color.x, mood.haze_color.y, mood.haze_color.z, mood.exposure],
        ground_ambient: [mood.ground_ambient.x, mood.ground_ambient.y, mood.ground_ambient.z, mood.bloom_strength],
        cloud_lit: [mood.cloud_lit.x, mood.cloud_lit.y, mood.cloud_lit.z, 0.0],
        cloud_shadow: [mood.cloud_shadow.x, mood.cloud_shadow.y, mood.cloud_shadow.z, 0.0],
        wake_body: [state.wake_body_pos.x, state.wake_body_pos.z, state.wake_body_vel.x, state.wake_body_vel.y],
        wake_cursor: [
            state.wake_cursor_pos.x,
            state.wake_cursor_pos.z,
            state.wake_cursor_vel.x,
            if state.wake_cursor_active { 1.0 } else { 0.0 },
        ],
        view_proj: state.view_projection.to_cols_array_2d(),
        inv_view_proj: state.inv_view_projection.to_cols_array_2d(),
    }
}

impl GpuPart for WorldGpu {
    type Reads = Res<'static, VerdantState>;
    const NAME: &'static str = "verdant_world";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let buf = WorldBuffers {
            uniforms: cx.buffer("verdant_uniforms", Lifetime::Kept),
            mesh_verts: cx.buffer("verdant_mesh_verts", Lifetime::Kept),
            mesh_opaque_idx: cx.buffer("verdant_mesh_opaque_idx", Lifetime::Kept),
            mesh_foliage_idx: cx.buffer("verdant_mesh_foliage_idx", Lifetime::Kept),
            blades: cx.buffer("verdant_grass_blades", Lifetime::Kept),
            wake: cx.buffer("verdant_wake_grid", Lifetime::Kept),
            particles: cx.buffer("verdant_particles", Lifetime::Kept),
            lights: cx.buffer("verdant_point_lights", Lifetime::Kept),
        };

        let sim_binds = [
            (sim::WORLD, Bind::Buffer(buf.uniforms)),
            (sim::BLADES_RW, Bind::Buffer(buf.blades)),
            (sim::WAKE_RW, Bind::Buffer(buf.wake)),
            (sim::PARTICLES_RW, Bind::Buffer(buf.particles)),
            (sim::LIGHTS_RW, Bind::Buffer(buf.lights)),
        ];

        let init_pass = cx.pass(INIT_PASS).label("verdant/init").span("init").id();
        let sim_pass = cx.pass(SIM_PASS).label("verdant/sim").span("sim").id();
        cx.after(sim_pass, init_pass);

        let b_ig = pick_bound(cx, &[sim::CS_INIT_GRASS], &sim_binds);
        let init_grass = cx.compute_pipeline(init_pass, sim::CS_INIT_GRASS, &[], &b_ig)?;
        let b_ip = pick_bound(cx, &[sim::CS_INIT_PARTICLES], &sim_binds);
        let init_particles = cx.compute_pipeline(init_pass, sim::CS_INIT_PARTICLES, &[], &b_ip)?;
        let b_wk = pick_bound(cx, &[sim::CS_WAKE_SIM], &sim_binds);
        let wake_sim = cx.compute_pipeline(sim_pass, sim::CS_WAKE_SIM, &[], &b_wk)?;
        let b_ps = pick_bound(cx, &[sim::CS_SIM_PARTICLES], &sim_binds);
        let particle_sim = cx.compute_pipeline(sim_pass, sim::CS_SIM_PARTICLES, &[], &b_ps)?;

        let pipes = install_scene_pipelines(cx, &buf)?;
        let meshes = build_world_meshes();
        let opaque_count = meshes.opaque_indices.len() as u32;
        let foliage_count = meshes.foliage_indices.len() as u32;

        Ok(Self {
            init_pass,
            sim_pass,
            init_grass,
            init_particles,
            wake_sim,
            particle_sim,
            mesh_opaque: pipes.mesh_opaque,
            mesh_foliage: pipes.mesh_foliage,
            grass_pipe: pipes.grass_pipe,
            sky_pipe: pipes.sky_pipe,
            particles_pipe: pipes.particles_pipe,
            uniforms: buf.uniforms,
            mesh_verts: buf.mesh_verts,
            mesh_opaque_idx: buf.mesh_opaque_idx,
            mesh_foliage_idx: buf.mesh_foliage_idx,
            blades: buf.blades,
            wake: buf.wake,
            particles: buf.particles,
            lights: buf.lights,
            meshes: Some(meshes),
            opaque_count,
            foliage_count,
            active_blades: MAX_BLADES,
            grass_inited: false,
            particles_inited: false,
        })
    }

    fn prepare(&mut self, state: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        let u = build_uniforms(&state);
        let _ = cx.allocate_buffer(self.uniforms, size_of::<WorldUniforms>() as u64);
        cx.write(self.uniforms, 0, &[u]);

        if let Some(meshes) = self.meshes.as_ref() {
            let vb = (meshes.vertices.len().max(1) * size_of::<MeshVertex>()) as u64;
            let oi = (meshes.opaque_indices.len().max(1) * size_of::<u32>()) as u64;
            let fi = (meshes.foliage_indices.len().max(1) * size_of::<u32>()) as u64;
            if cx.allocate_buffer(self.mesh_verts, vb) == Allocation::Fresh {
                cx.write(self.mesh_verts, 0, &meshes.vertices);
            }
            if cx.allocate_buffer(self.mesh_opaque_idx, oi) == Allocation::Fresh {
                cx.write(self.mesh_opaque_idx, 0, &meshes.opaque_indices);
            }
            if cx.allocate_buffer(self.mesh_foliage_idx, fi) == Allocation::Fresh {
                cx.write(self.mesh_foliage_idx, 0, &meshes.foliage_indices);
            }
        }

        let blades_bytes = u64::from(MAX_BLADES) * size_of::<GrassBlade>() as u64;
        let wake_bytes = u64::from(WAKE_SIDE * WAKE_SIDE) * size_of::<WakeCell>() as u64;
        let part_bytes = u64::from(MAX_PARTICLES) * size_of::<Particle>() as u64;
        let light_bytes = u64::from(POINT_LIGHT_COUNT) * size_of::<PointLight>() as u64;

        let fresh_blades = cx.allocate_buffer(self.blades, blades_bytes) == Allocation::Fresh;
        let _ = cx.allocate_buffer(self.wake, wake_bytes);
        let fresh_parts = cx.allocate_buffer(self.particles, part_bytes) == Allocation::Fresh;
        let _ = cx.allocate_buffer(self.lights, light_bytes);

        if fresh_blades {
            self.grass_inited = false;
        }
        if fresh_parts {
            self.particles_inited = false;
        }
        if !self.grass_inited || !self.particles_inited {
            cx.run(self.init_pass);
        }
        cx.run(self.sim_pass);
        self.active_blades = state.active_blades();
    }

    fn encode(&mut self, pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        if pass == self.init_pass {
            if !self.grass_inited {
                let _ = cx.recorded();
                cx.dispatch(&self.init_grass, [MAX_BLADES.div_ceil(256), 1, 1]);
                if cx.recorded() == engine_render::Recorded::Exact {
                    self.grass_inited = true;
                }
            }
            if !self.particles_inited {
                let _ = cx.recorded();
                cx.dispatch(&self.init_particles, [MAX_PARTICLES.div_ceil(256), 1, 1]);
                if cx.recorded() == engine_render::Recorded::Exact {
                    self.particles_inited = true;
                }
            }
            return;
        }
        if pass == self.sim_pass {
            cx.dispatch(&self.wake_sim, [WAKE_SIDE.div_ceil(16), WAKE_SIDE.div_ceil(16), 1]);
            cx.dispatch(&self.particle_sim, [MAX_PARTICLES.div_ceil(256), 1, 1]);
        }
    }
}

impl WorldGpu {
    pub(super) fn draw_world(&self, draw: &mut Draw<'_, Self, Scene>) {
        if self.opaque_count > 0 {
            draw.draw(
                &self.mesh_opaque,
                Shape::Pulled {
                    vertices: 0..self.opaque_count,
                    instances: 0..1,
                    key: None,
                },
            );
        }
        if self.foliage_count > 0 {
            draw.draw(
                &self.mesh_foliage,
                Shape::Pulled {
                    vertices: 0..self.foliage_count,
                    instances: 0..1,
                    key: None,
                },
            );
        }
        if self.active_blades > 0 {
            draw.draw(
                &self.grass_pipe,
                Shape::Pulled {
                    vertices: 0..7,
                    instances: 0..self.active_blades,
                    key: None,
                },
            );
        }
        draw.draw(
            &self.sky_pipe,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                key: None,
            },
        );
        draw.draw(
            &self.particles_pipe,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..MAX_PARTICLES,
                key: None,
            },
        );
    }
}
