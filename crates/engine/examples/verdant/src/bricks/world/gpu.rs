use bevy_ecs::system::SystemParamItem;
use engine::prelude::*;
use engine_core::{Entry, Error, Var};
use engine_render::{ComputePipeline, Encode, PerBatch, Scene};

use super::shaders::world::{cull, grass, mesh, particles, sim, sky};
use super::{
    GrassBlade, MAX_BLADES, MAX_PARTICLES, MAX_VIS_FAR, MAX_VIS_NEAR0, MAX_VIS_NEAR1, MeshVertex,
    NUM_PATCHES, POINT_LIGHT_COUNT, Particle, PatchBounds, PointLight, VisibleBlade,
    VisibleParticle, WAKE_SIDE, WakeCell, WorldUniforms,
};
use crate::mesh::{WorldMeshes, build_world_meshes};
use crate::names::COLOR_PASS;
use crate::state::VerdantState;

const INIT_PASS: &str = "verdant_sim_init";
const SIM_PASS: &str = "verdant_sim_step";
const CULL_PASS: &str = "verdant_cull_grass";

pub(super) struct WorldGpu {
    init_pass: engine_render::Pass<Self>,
    sim_pass: engine_render::Pass<Self>,
    cull_pass: engine_render::Pass<Self>,
    init_grass: ComputePipeline<Self>,
    init_particles: ComputePipeline<Self>,
    wake_sim: ComputePipeline<Self>,
    particle_sim: ComputePipeline<Self>,
    cull_grass: ComputePipeline<Self>,
    mesh_opaque: Pipeline<Self, Scene>,
    mesh_foliage: Pipeline<Self, Scene>,
    grass_near0_pipe: Pipeline<Self, Scene>,
    grass_near1_pipe: Pipeline<Self, Scene>,
    grass_far_pipe: Pipeline<Self, Scene>,
    sky_pipe: Pipeline<Self, Scene>,
    particles_pipe: Pipeline<Self, Scene>,
    uniforms: Buffer<Self>,
    mesh_verts: Buffer<Self>,
    mesh_opaque_idx: Buffer<Self>,
    mesh_foliage_idx: Buffer<Self>,
    blades: Buffer<Self>,
    patch_bounds: Buffer<Self>,
    vis_near0: Buffer<Self>,
    vis_near1: Buffer<Self>,
    vis_far: Buffer<Self>,
    grass_indirect: Buffer<Self>,
    wake: Buffer<Self>,
    particles: Buffer<Self>,
    vis_particles: Buffer<Self>,
    particle_indirect: Buffer<Self>,
    lights: Buffer<Self>,
    meshes: Option<WorldMeshes>,
    terrain_count: u32,
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
    patch_bounds: Buffer<WorldGpu>,
    vis_near0: Buffer<WorldGpu>,
    vis_near1: Buffer<WorldGpu>,
    vis_far: Buffer<WorldGpu>,
    grass_indirect: Buffer<WorldGpu>,
    wake: Buffer<WorldGpu>,
    particles: Buffer<WorldGpu>,
    vis_particles: Buffer<WorldGpu>,
    particle_indirect: Buffer<WorldGpu>,
    lights: Buffer<WorldGpu>,
}

struct ScenePipelines {
    mesh_opaque: Pipeline<WorldGpu, Scene>,
    mesh_foliage: Pipeline<WorldGpu, Scene>,
    grass_near0_pipe: Pipeline<WorldGpu, Scene>,
    grass_near1_pipe: Pipeline<WorldGpu, Scene>,
    grass_far_pipe: Pipeline<WorldGpu, Scene>,
    sky_pipe: Pipeline<WorldGpu, Scene>,
    particles_pipe: Pipeline<WorldGpu, Scene>,
}

fn alloc_buffers(cx: &mut Install<'_, WorldGpu>) -> WorldBuffers {
    WorldBuffers {
        uniforms: cx.buffer("verdant_uniforms", Lifetime::Kept),
        mesh_verts: cx.buffer("verdant_mesh_verts", Lifetime::Kept),
        mesh_opaque_idx: cx.buffer("verdant_mesh_opaque_idx", Lifetime::Kept),
        mesh_foliage_idx: cx.buffer("verdant_mesh_foliage_idx", Lifetime::Kept),
        blades: cx.buffer("verdant_grass_blades", Lifetime::Kept),
        patch_bounds: cx.buffer("verdant_patch_bounds", Lifetime::Kept),
        vis_near0: cx.buffer("verdant_vis_near0", Lifetime::Kept),
        vis_near1: cx.buffer("verdant_vis_near1", Lifetime::Kept),
        vis_far: cx.buffer("verdant_vis_far", Lifetime::Kept),
        grass_indirect: cx.indirect_buffer("verdant_grass_indirect", Lifetime::Kept),
        wake: cx.buffer("verdant_wake_grid", Lifetime::Kept),
        particles: cx.buffer("verdant_particles", Lifetime::Kept),
        vis_particles: cx.buffer("verdant_vis_particles", Lifetime::Kept),
        particle_indirect: cx.indirect_buffer("verdant_particle_indirect", Lifetime::Kept),
        lights: cx.buffer("verdant_point_lights", Lifetime::Kept),
    }
}

struct CoreScenePipelines {
    mesh_opaque: Pipeline<WorldGpu, Scene>,
    mesh_foliage: Pipeline<WorldGpu, Scene>,
    grass_near0_pipe: Pipeline<WorldGpu, Scene>,
    grass_near1_pipe: Pipeline<WorldGpu, Scene>,
    grass_far_pipe: Pipeline<WorldGpu, Scene>,
}

fn install_grass_and_mesh_pipes(
    cx: &mut Install<'_, WorldGpu>,
    buf: &WorldBuffers,
) -> Result<CoreScenePipelines, Error> {
    let scene = Scene::named(COLOR_PASS);
    let opaque_target = [(None, wgpu::ColorWrites::ALL)];
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

    let mesh_opaque_binds = [
        (mesh::WORLD, Bind::Buffer(buf.uniforms)),
        (mesh::MESH_VERTS, Bind::Buffer(buf.mesh_verts)),
        (mesh::MESH_INDICES, Bind::Buffer(buf.mesh_opaque_idx)),
        (mesh::LIGHTS, Bind::Buffer(buf.lights)),
    ];
    let b_mo = pick_bound(cx, &[mesh::VS_MESH, mesh::FS_MESH], &mesh_opaque_binds);
    let mesh_opaque = cx.render_pipeline(
        scene,
        &scene_pipe(mesh::VS_MESH, mesh::FS_MESH, back_cull, depth_state(true), &opaque_target),
        &b_mo,
    )?;

    let mesh_foliage_binds = [
        (mesh::WORLD, Bind::Buffer(buf.uniforms)),
        (mesh::MESH_VERTS, Bind::Buffer(buf.mesh_verts)),
        (mesh::MESH_INDICES, Bind::Buffer(buf.mesh_foliage_idx)),
        (mesh::LIGHTS, Bind::Buffer(buf.lights)),
    ];
    let b_mf = pick_bound(cx, &[mesh::VS_FOLIAGE, mesh::FS_FOLIAGE], &mesh_foliage_binds);
    let mesh_foliage = cx.render_pipeline(
        scene,
        &scene_pipe(mesh::VS_FOLIAGE, mesh::FS_FOLIAGE, no_cull, depth_state(true), &opaque_target),
        &b_mf,
    )?;

    let gn0_binds = [
        (grass::WORLD, Bind::Buffer(buf.uniforms)),
        (grass::VIS_BLADES, Bind::Buffer(buf.vis_near0)),
    ];
    let b_gn0 = pick_bound(cx, &[grass::VS_GRASS_NEAR, grass::FS_GRASS_NEAR], &gn0_binds);
    let grass_near0_pipe = cx.render_pipeline(
        scene,
        &scene_pipe(grass::VS_GRASS_NEAR, grass::FS_GRASS_NEAR, strip, depth_state(true), &opaque_target),
        &b_gn0,
    )?;

    let gn1_binds = [
        (grass::WORLD, Bind::Buffer(buf.uniforms)),
        (grass::VIS_BLADES, Bind::Buffer(buf.vis_near1)),
    ];
    let b_gn1 = pick_bound(cx, &[grass::VS_GRASS_NEAR, grass::FS_GRASS_NEAR], &gn1_binds);
    let grass_near1_pipe = cx.render_pipeline(
        scene,
        &scene_pipe(grass::VS_GRASS_NEAR, grass::FS_GRASS_NEAR, strip, depth_state(true), &opaque_target),
        &b_gn1,
    )?;

    let gf_binds = [
        (grass::WORLD, Bind::Buffer(buf.uniforms)),
        (grass::VIS_BLADES, Bind::Buffer(buf.vis_far)),
    ];
    let b_gf = pick_bound(cx, &[grass::VS_GRASS_FAR, grass::FS_GRASS_FAR], &gf_binds);
    let grass_far_pipe = cx.render_pipeline(
        scene,
        &scene_pipe(grass::VS_GRASS_FAR, grass::FS_GRASS_FAR, no_cull, depth_state(true), &opaque_target),
        &b_gf,
    )?;

    Ok(CoreScenePipelines {
        mesh_opaque,
        mesh_foliage,
        grass_near0_pipe,
        grass_near1_pipe,
        grass_far_pipe,
    })
}

fn install_scene_pipelines(
    cx: &mut Install<'_, WorldGpu>,
    buf: &WorldBuffers,
) -> Result<ScenePipelines, Error> {
    let core = install_grass_and_mesh_pipes(cx, buf)?;
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
    let no_cull = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: None,
        ..wgpu::PrimitiveState::default()
    };

    let sky_binds = [(sky::WORLD, Bind::Buffer(buf.uniforms))];
    let b_sk = pick_bound(cx, &[sky::VS_SKY, sky::FS_SKY], &sky_binds);
    let sky_pipe = cx.render_pipeline(
        scene,
        &scene_pipe(sky::VS_SKY, sky::FS_SKY, no_cull, depth_state(false), &opaque_target),
        &b_sk,
    )?;

    let part_binds = [
        (particles::WORLD, Bind::Buffer(buf.uniforms)),
        (particles::VIS_PARTICLES, Bind::Buffer(buf.vis_particles)),
    ];
    let b_pt = pick_bound(cx, &[particles::VS_PARTICLE, particles::FS_PARTICLE], &part_binds);
    let particles_pipe = cx.render_pipeline(
        scene,
        &scene_pipe(particles::VS_PARTICLE, particles::FS_PARTICLE, no_cull, depth_state(false), &alpha_target),
        &b_pt,
    )?;

    Ok(ScenePipelines {
        mesh_opaque: core.mesh_opaque,
        mesh_foliage: core.mesh_foliage,
        grass_near0_pipe: core.grass_near0_pipe,
        grass_near1_pipe: core.grass_near1_pipe,
        grass_far_pipe: core.grass_far_pipe,
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
        let buf = alloc_buffers(cx);

        let sim_binds = [
            (sim::WORLD, Bind::Buffer(buf.uniforms)),
            (sim::BLADES_RW, Bind::Buffer(buf.blades)),
            (sim::PATCH_BOUNDS_RW, Bind::Buffer(buf.patch_bounds)),
            (sim::WAKE_RW, Bind::Buffer(buf.wake)),
            (sim::PARTICLES_RW, Bind::Buffer(buf.particles)),
            (sim::LIGHTS_RW, Bind::Buffer(buf.lights)),
            (sim::VIS_PARTICLES_RW, Bind::Buffer(buf.vis_particles)),
            (sim::PART_ARGS_RW, Bind::Buffer(buf.particle_indirect)),
        ];
        let cull_binds = [
            (cull::WORLD, Bind::Buffer(buf.uniforms)),
            (cull::BLADES, Bind::Buffer(buf.blades)),
            (cull::PATCH_BOUNDS, Bind::Buffer(buf.patch_bounds)),
            (cull::WAKE, Bind::Buffer(buf.wake)),
            (cull::LIGHTS, Bind::Buffer(buf.lights)),
            (cull::VIS_NEAR0_RW, Bind::Buffer(buf.vis_near0)),
            (cull::VIS_NEAR1_RW, Bind::Buffer(buf.vis_near1)),
            (cull::VIS_FAR_RW, Bind::Buffer(buf.vis_far)),
            (cull::GRASS_ARGS_RW, Bind::Buffer(buf.grass_indirect)),
        ];

        let init_pass = cx.pass(INIT_PASS).label("verdant/init").span("init").id();
        let sim_pass = cx.pass(SIM_PASS).label("verdant/sim").span("sim").id();
        let cull_pass = cx.pass(CULL_PASS).label("verdant/cull").span("cull").id();
        cx.after(sim_pass, init_pass);
        cx.after(cull_pass, sim_pass);

        let b_ig = pick_bound(cx, &[sim::CS_INIT_GRASS], &sim_binds);
        let init_grass = cx.compute_pipeline(init_pass, sim::CS_INIT_GRASS, &[], &b_ig)?;
        let b_ip = pick_bound(cx, &[sim::CS_INIT_PARTICLES], &sim_binds);
        let init_particles = cx.compute_pipeline(init_pass, sim::CS_INIT_PARTICLES, &[], &b_ip)?;
        let b_wk = pick_bound(cx, &[sim::CS_WAKE_SIM], &sim_binds);
        let wake_sim = cx.compute_pipeline(sim_pass, sim::CS_WAKE_SIM, &[], &b_wk)?;
        let b_ps = pick_bound(cx, &[sim::CS_SIM_PARTICLES], &sim_binds);
        let particle_sim = cx.compute_pipeline(sim_pass, sim::CS_SIM_PARTICLES, &[], &b_ps)?;
        let b_cg = pick_bound(cx, &[cull::CS_CULL_GRASS], &cull_binds);
        let cull_grass = cx.compute_pipeline(cull_pass, cull::CS_CULL_GRASS, &[], &b_cg)?;

        let pipes = install_scene_pipelines(cx, &buf)?;
        let meshes = build_world_meshes();
        let terrain_count = meshes.terrain_index_count;
        let opaque_count = meshes.opaque_indices.len() as u32;
        let foliage_count = meshes.foliage_indices.len() as u32;

        Ok(Self {
            init_pass,
            sim_pass,
            cull_pass,
            init_grass,
            init_particles,
            wake_sim,
            particle_sim,
            cull_grass,
            mesh_opaque: pipes.mesh_opaque,
            mesh_foliage: pipes.mesh_foliage,
            grass_near0_pipe: pipes.grass_near0_pipe,
            grass_near1_pipe: pipes.grass_near1_pipe,
            grass_far_pipe: pipes.grass_far_pipe,
            sky_pipe: pipes.sky_pipe,
            particles_pipe: pipes.particles_pipe,
            uniforms: buf.uniforms,
            mesh_verts: buf.mesh_verts,
            mesh_opaque_idx: buf.mesh_opaque_idx,
            mesh_foliage_idx: buf.mesh_foliage_idx,
            blades: buf.blades,
            patch_bounds: buf.patch_bounds,
            vis_near0: buf.vis_near0,
            vis_near1: buf.vis_near1,
            vis_far: buf.vis_far,
            grass_indirect: buf.grass_indirect,
            wake: buf.wake,
            particles: buf.particles,
            vis_particles: buf.vis_particles,
            particle_indirect: buf.particle_indirect,
            lights: buf.lights,
            meshes: Some(meshes),
            terrain_count,
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
        let patches_bytes = u64::from(NUM_PATCHES) * size_of::<PatchBounds>() as u64;
        let near0_bytes = u64::from(MAX_VIS_NEAR0) * size_of::<VisibleBlade>() as u64;
        let near1_bytes = u64::from(MAX_VIS_NEAR1) * size_of::<VisibleBlade>() as u64;
        let far_bytes = u64::from(MAX_VIS_FAR) * size_of::<VisibleBlade>() as u64;
        let wake_bytes = u64::from(WAKE_SIDE * WAKE_SIDE + 256) * size_of::<WakeCell>() as u64;
        let part_bytes = u64::from(MAX_PARTICLES) * size_of::<Particle>() as u64;
        let vis_part_bytes = u64::from(MAX_PARTICLES) * size_of::<VisibleParticle>() as u64;
        let light_bytes = u64::from(POINT_LIGHT_COUNT) * size_of::<PointLight>() as u64;

        let fresh_blades = cx.allocate_buffer(self.blades, blades_bytes) == Allocation::Fresh;
        let fresh_patches =
            cx.allocate_buffer(self.patch_bounds, patches_bytes) == Allocation::Fresh;
        let _ = cx.allocate_buffer(self.vis_near0, near0_bytes);
        let _ = cx.allocate_buffer(self.vis_near1, near1_bytes);
        let _ = cx.allocate_buffer(self.vis_far, far_bytes);
        let _ = cx.allocate_buffer(self.grass_indirect, 48);
        let _ = cx.allocate_buffer(self.wake, wake_bytes);
        let fresh_parts = cx.allocate_buffer(self.particles, part_bytes) == Allocation::Fresh;
        let _ = cx.allocate_buffer(self.vis_particles, vis_part_bytes);
        let _ = cx.allocate_buffer(self.particle_indirect, 16);
        let _ = cx.allocate_buffer(self.lights, light_bytes);

        cx.write(self.grass_indirect, 0, &[5u32, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0]);
        cx.write(self.particle_indirect, 0, &[3u32, 0, 0, 0]);

        if fresh_blades || fresh_patches {
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
        if self.active_blades > 0 {
            cx.run(self.cull_pass);
        }
    }

    fn encode(&mut self, pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        if pass == self.init_pass {
            if !self.grass_inited {
                let _ = cx.recorded();
                cx.dispatch(&self.init_grass, [NUM_PATCHES, 1, 1]);
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
            return;
        }
        if pass == self.cull_pass && self.active_blades > 0 {
            cx.dispatch(&self.cull_grass, [self.active_blades.div_ceil(256), 1, 1]);
        }
    }
}

impl WorldGpu {
    pub(super) fn draw_world(&self, draw: &mut Draw<'_, Self, Scene>) {
        if self.opaque_count > self.terrain_count {
            draw.draw(
                &self.mesh_opaque,
                Shape::Pulled {
                    vertices: self.terrain_count..self.opaque_count,
                    instances: 0..1,
                    key: None,
                },
            );
        }
        if self.active_blades > 0 {
            draw.draw_indirect(&self.grass_near0_pipe, self.grass_indirect, 0, None);
            draw.draw_indirect(&self.grass_near1_pipe, self.grass_indirect, 16, None);
            draw.draw_indirect(&self.grass_far_pipe, self.grass_indirect, 32, None);
        }
        if self.terrain_count > 0 {
            draw.draw(
                &self.mesh_opaque,
                Shape::Pulled {
                    vertices: 0..self.terrain_count,
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
        draw.draw(
            &self.sky_pipe,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                key: None,
            },
        );
        draw.draw_indirect(&self.particles_pipe, self.particle_indirect, 0, None);
    }
}
