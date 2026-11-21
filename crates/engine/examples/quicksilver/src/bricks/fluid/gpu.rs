use bytemuck::{Pod, Zeroable};
use engine::math;
use engine::prelude::*;
use engine_core::{Entry, Error, Filter, PassViews, Sampler, TargetName, Var, ViewFacts, Wrap};
use engine_render::{
    Access, Allocation, Bind, Buffer, ComputePipeline, DepthUse, Encode, FULLSCREEN, GpuPart,
    Install, Lifetime, Own, Pass, PerBatch, Pipeline, Prepare, Recorded, RenderPipeline, Scene,
    Shape, SharedTarget, TargetClear, Texture, wgpu,
};

use super::shaders::fluid::{bilateral, post, sim, splat, studio, ui};
use super::{
    BLOOM_0, BLOOM_1, BLOOM_2, FLUID_AUX, FLUID_DEPTH_0, FLUID_DEPTH_1, FLUID_DEPTH_2,
    FLUID_DEPTH_3, FLUID_DEPTH_4, FLUID_SPLAT_PASS, FLUID_THICK_0, FLUID_THICK_1, FLUID_THICK_2,
    HDR_COLOR, HW_DEPTH,
};
use crate::font::{ATLAS_H, ATLAS_W, build_sdf_atlas};
use crate::state::{
    BASIN_FLOOR_Y, BASIN_HALF_X, BASIN_HALF_Z, CELL_SIZE, GRID_X, GRID_Y, GRID_Z, MAX_PARTICLES,
    QuicksilverState,
};
use crate::ui_layout::{UiQuad, build_ui_quads, compute_layout};

const PARTICLE_BYTES: u64 = 64;
const GRID_CELLS: u32 = GRID_X * GRID_Y * GRID_Z;
const SUBSTEPS: u32 = 2;
const UI_HEADER_BYTES: u64 = 16;

const LINEAR_CLAMP: Sampler = Sampler {
    wrap_u: Wrap::ClampToEdge,
    wrap_v: Wrap::ClampToEdge,
    wrap_w: Wrap::ClampToEdge,
    mag_filter: Filter::Linear,
    min_filter: Filter::Linear,
    mipmap_filter: None,
    compare: None,
    anisotropy: 1,
};

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
struct SimParams {
    dims: [u32; 4],
    physics: [f32; 4],
    fluid: [f32; 4],
    stir_pos: [f32; 4],
    stir_vel: [f32; 4],
    drop_info: [u32; 4],
    bounds: [f32; 4],
    time_info: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Pod, Zeroable)]
struct FrameUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
    eye_time: [f32; 4],
    viewport: [f32; 4],
    splat: [f32; 4],
    style: [f32; 4],
    stir: [f32; 4],
    basin: [f32; 4],
    grid_dims: [f32; 4],
}

struct SimPipelines {
    pass: Pass<FluidGpu>,
    init: ComputePipeline<FluidGpu>,
    drop_burst: ComputePipeline<FluidGpu>,
    clear_grid: ComputePipeline<FluidGpu>,
    p2g_fused: ComputePipeline<FluidGpu>,
    update_grid: ComputePipeline<FluidGpu>,
    g2p: ComputePipeline<FluidGpu>,
    heightfield: ComputePipeline<FluidGpu>,
}

struct BlurPasses {
    passes: [Pass<FluidGpu>; 4],
    pipelines: [Pipeline<FluidGpu, Own>; 4],
    thick_splat_pass: Pass<FluidGpu>,
    thick_splat: Pipeline<FluidGpu, Own>,
    thick_h_pass: Pass<FluidGpu>,
    thick_h: Pipeline<FluidGpu, Own>,
    thick_v_pass: Pass<FluidGpu>,
    thick_v: Pipeline<FluidGpu, Own>,
}

struct PostPasses {
    studio_pass: Pass<FluidGpu>,
    studio: Pipeline<FluidGpu, Own>,
    bloom_ext_pass: Pass<FluidGpu>,
    bloom_ext: Pipeline<FluidGpu, Own>,
    bloom_h_pass: Pass<FluidGpu>,
    bloom_h: Pipeline<FluidGpu, Own>,
    bloom_v_pass: Pass<FluidGpu>,
    bloom_v: Pipeline<FluidGpu, Own>,
    tonemap_pass: Pass<FluidGpu>,
    tonemap: Pipeline<FluidGpu, Own>,
    ui_pass: Pass<FluidGpu>,
    ui_pipe: Pipeline<FluidGpu, Own>,
}

struct GpuBuffers {
    particles: Buffer<FluidGpu>,
    grid_atomic: Buffer<FluidGpu>,
    grid_vel: Buffer<FluidGpu>,
    heightfield: Buffer<FluidGpu>,
    sim_params: Buffer<FluidGpu>,
    frame: Buffer<FluidGpu>,
    ui_quads: Buffer<FluidGpu>,
    font_tex: Texture<FluidGpu>,
}

pub(super) struct FluidGpu {
    buffers: GpuBuffers,
    sim: SimPipelines,
    splat_pipe: Pipeline<FluidGpu, Scene>,
    blur: BlurPasses,
    post: PostPasses,
    particle_count: u32,
    frame_index: u32,
    last_reset: u32,
    needs_init: bool,
    last_drop: u32,
    pending_drop: bool,
    drop_cursor: u32,
    last_ui_rev: u64,
    ui_quad_count: u32,
    ui_scratch: Vec<UiQuad>,
    ui_bytes: Vec<u8>,
}

fn bound_compute(
    cx: &mut Install<'_, FluidGpu>,
    entry: Entry,
    resources: &[(Var, Bind<FluidGpu>)],
) -> Vec<(Var, Bind<FluidGpu>)> {
    resources
        .iter()
        .filter(|(var, _)| cx.reaches(entry, *var))
        .map(|(var, bind)| (*var, bind.clone()))
        .collect()
}

fn fs_desc(
    fragment: Entry,
    blend: Option<wgpu::BlendState>,
    constants: &'static [(&'static str, f64)],
) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: FULLSCREEN,
        fragment: Some(fragment),
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: match blend {
            Some(_) => &[(
                Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                wgpu::ColorWrites::ALL,
            )],
            None => &[(None, wgpu::ColorWrites::ALL)],
        },
        constants,
        per_batch: PerBatch::NONE,
    }
}

fn declare_targets(cx: &mut Install<'_, FluidGpu>) {
    let clear_zero = TargetClear::Color([0.0, 0.0, 0.0, 0.0]);
    for name in [
        FLUID_DEPTH_0,
        FLUID_DEPTH_1,
        FLUID_DEPTH_2,
        FLUID_DEPTH_3,
        FLUID_DEPTH_4,
    ] {
        cx.shared_target(SharedTarget {
            name,
            format: wgpu::TextureFormat::R32Float,
            samples: 1,
            clear: clear_zero,
        });
    }
    cx.shared_target(SharedTarget {
        name: HW_DEPTH,
        format: wgpu::TextureFormat::Depth32Float,
        samples: 1,
        clear: TargetClear::Depth(0.0),
    });
    for name in [
        FLUID_AUX,
        FLUID_THICK_0,
        FLUID_THICK_1,
        FLUID_THICK_2,
        HDR_COLOR,
        BLOOM_0,
        BLOOM_1,
        BLOOM_2,
    ] {
        cx.shared_target(SharedTarget {
            name,
            format: wgpu::TextureFormat::Rgba16Float,
            samples: 1,
            clear: clear_zero,
        });
    }
}

fn install_sim(cx: &mut Install<'_, FluidGpu>, bufs: &GpuBuffers) -> Result<SimPipelines, Error> {
    let pass = cx.pass("mpm_sim").label("quicksilver/mpm_sim").id();
    let res = [
        (sim::SIM_PARAMS, Bind::Buffer(bufs.sim_params)),
        (sim::SIM_PARTICLES, Bind::Buffer(bufs.particles)),
        (sim::GRID_ATOMIC, Bind::Buffer(bufs.grid_atomic)),
        (sim::GRID_VEL, Bind::Buffer(bufs.grid_vel)),
        (sim::HEIGHTFIELD_OUT, Bind::Buffer(bufs.heightfield)),
    ];
    let make = |cx: &mut Install<'_, FluidGpu>, entry: Entry| {
        let b = bound_compute(cx, entry, &res);
        cx.compute_pipeline(pass, entry, &[], &b)
    };
    Ok(SimPipelines {
        pass,
        init: make(cx, sim::INIT_PARTICLES)?,
        drop_burst: make(cx, sim::DROP_BURST)?,
        clear_grid: make(cx, sim::CLEAR_GRID)?,
        p2g_fused: make(cx, sim::P2G_FUSED)?,
        update_grid: make(cx, sim::UPDATE_GRID)?,
        g2p: make(cx, sim::G2P)?,
        heightfield: make(cx, sim::BUILD_HEIGHTFIELD)?,
    })
}

type DepthBlurs = ([Pass<FluidGpu>; 4], [Pipeline<FluidGpu, Own>; 4]);

fn install_depth_blurs(
    cx: &mut Install<'_, FluidGpu>,
    bufs: &GpuBuffers,
) -> Result<DepthBlurs, Error> {
    let stages: [(&'static str, TargetName, TargetName, Entry, f64); 4] = [
        ("blur_h1", FLUID_DEPTH_0, FLUID_DEPTH_1, bilateral::FS_BILATERAL_H, 2.0),
        ("blur_v1", FLUID_DEPTH_1, FLUID_DEPTH_2, bilateral::FS_BILATERAL_V, 2.0),
        ("blur_h2", FLUID_DEPTH_2, FLUID_DEPTH_3, bilateral::FS_BILATERAL_H, 1.0),
        ("blur_v2", FLUID_DEPTH_3, FLUID_DEPTH_4, bilateral::FS_BILATERAL_V, 1.0),
    ];
    let mut pass_vec = Vec::with_capacity(4);
    let mut pipe_vec = Vec::with_capacity(4);
    for (name, src, dst, entry, step) in stages {
        let p = cx
            .pass(name)
            .label(name)
            .views(PassViews::Main)
            .target(dst, Access::Clear)
            .id();
        let desc = RenderPipeline {
            vertex: FULLSCREEN,
            fragment: Some(entry),
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
            stencil: wgpu::StencilState::default(),
            samples: 1,
            targets: &[(None, wgpu::ColorWrites::ALL)],
            constants: &[("FILTER_STEP", step)],
            per_batch: PerBatch::NONE,
        };
        let pipe = cx.render_pipeline(
            p,
            &desc,
            &[
                (bilateral::BLUR_FRAME, Bind::Buffer(bufs.frame)),
                (bilateral::DEPTH_IN, Bind::Target(src)),
            ],
        )?;
        pass_vec.push(p);
        pipe_vec.push(pipe);
    }
    Ok((
        [pass_vec[0], pass_vec[1], pass_vec[2], pass_vec[3]],
        [pipe_vec[0], pipe_vec[1], pipe_vec[2], pipe_vec[3]],
    ))
}

fn install_blurs(cx: &mut Install<'_, FluidGpu>, bufs: &GpuBuffers) -> Result<BlurPasses, Error> {
    let (passes, pipelines) = install_depth_blurs(cx, bufs)?;
    let thick_splat_pass = cx
        .pass("fluid_thick_splat")
        .label("quicksilver/thick_splat")
        .views(PassViews::Main)
        .target(FLUID_THICK_0, Access::Clear)
        .id();
    let additive = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
    };
    let thick_splat = cx.render_pipeline(
        thick_splat_pass,
        &RenderPipeline {
            vertex: splat::VS_THICK,
            fragment: Some(splat::FS_THICKNESS),
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
            stencil: wgpu::StencilState::default(),
            samples: 1,
            targets: &[(Some(additive), wgpu::ColorWrites::ALL)],
            constants: &[],
            per_batch: PerBatch::NONE,
        },
        &[
            (splat::SPLAT_FRAME, Bind::Buffer(bufs.frame)),
            (splat::SPLAT_PARTICLES, Bind::Buffer(bufs.particles)),
        ],
    )?;
    let thick_h_pass = cx
        .pass("thick_h")
        .label("quicksilver/thick_h")
        .views(PassViews::Main)
        .target(FLUID_THICK_1, Access::Clear)
        .id();
    let thick_h = cx.render_pipeline(
        thick_h_pass,
        &fs_desc(bilateral::FS_THICK_H, None, &[]),
        &[(bilateral::THICK_IN, Bind::Target(FLUID_THICK_0))],
    )?;
    let thick_v_pass = cx
        .pass("thick_v")
        .label("quicksilver/thick_v")
        .views(PassViews::Main)
        .target(FLUID_THICK_2, Access::Clear)
        .id();
    let thick_v = cx.render_pipeline(
        thick_v_pass,
        &fs_desc(bilateral::FS_THICK_V, None, &[]),
        &[(bilateral::THICK_IN, Bind::Target(FLUID_THICK_1))],
    )?;
    Ok(BlurPasses {
        passes,
        pipelines,
        thick_splat_pass,
        thick_splat,
        thick_h_pass,
        thick_h,
        thick_v_pass,
        thick_v,
    })
}

fn install_ui_pass(
    cx: &mut Install<'_, FluidGpu>,
    bufs: &GpuBuffers,
    linear: engine_render::Sampler<FluidGpu>,
    tonemap_pass: Pass<FluidGpu>,
) -> Result<(Pass<FluidGpu>, Pipeline<FluidGpu, Own>), Error> {
    let ui_pass = cx
        .pass("ui_dock")
        .label("quicksilver/ui")
        .views(PassViews::Main)
        .target(TargetName::CANVAS, Access::Load)
        .id();
    cx.after(ui_pass, tonemap_pass);
    let ui_pipe = cx.render_pipeline(
        ui_pass,
        &RenderPipeline {
            vertex: ui::VS_UI,
            fragment: Some(ui::FS_UI),
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
            stencil: wgpu::StencilState::default(),
            samples: 1,
            targets: &[(
                Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                wgpu::ColorWrites::ALL,
            )],
            constants: &[],
            per_batch: PerBatch::NONE,
        },
        &[
            (ui::UI_DATA, Bind::Buffer(bufs.ui_quads)),
            (ui::UI_FONT, Bind::Texture(bufs.font_tex)),
            (ui::UI_SAMPLER, Bind::Sampler(linear)),
        ],
    )?;
    Ok((ui_pass, ui_pipe))
}

fn install_post(cx: &mut Install<'_, FluidGpu>, bufs: &GpuBuffers) -> Result<PostPasses, Error> {
    let linear = cx.sampler(LINEAR_CLAMP);
    let studio_pass = cx
        .pass("studio_composite")
        .label("quicksilver/studio")
        .views(PassViews::Main)
        .target(HDR_COLOR, Access::Clear)
        .id();
    let studio = cx.render_pipeline(
        studio_pass,
        &fs_desc(studio::FS_STUDIO, None, &[]),
        &[
            (studio::STUDIO_FRAME, Bind::Buffer(bufs.frame)),
            (studio::STUDIO_DEPTH, Bind::Target(FLUID_DEPTH_4)),
            (studio::STUDIO_AUX, Bind::Target(FLUID_AUX)),
            (studio::STUDIO_THICK, Bind::Target(FLUID_THICK_2)),
            (studio::STUDIO_SAMPLER, Bind::Sampler(linear)),
            (studio::STUDIO_HEIGHTFIELD, Bind::Buffer(bufs.heightfield)),
        ],
    )?;
    let bloom_ext_pass = cx
        .pass("bloom_extract")
        .label("quicksilver/bloom_ext")
        .views(PassViews::Main)
        .target(BLOOM_0, Access::Clear)
        .id();
    let bloom_ext = cx.render_pipeline(
        bloom_ext_pass,
        &fs_desc(post::FS_BLOOM_EXTRACT, None, &[]),
        &[
            (post::POST_HDR, Bind::Target(HDR_COLOR)),
            (post::POST_SAMPLER, Bind::Sampler(linear)),
        ],
    )?;
    let bloom_h_pass = cx
        .pass("bloom_h")
        .label("quicksilver/bloom_h")
        .views(PassViews::Main)
        .target(BLOOM_1, Access::Clear)
        .id();
    let bloom_h = cx.render_pipeline(
        bloom_h_pass,
        &fs_desc(post::FS_BLOOM_H, None, &[]),
        &[
            (post::POST_BLOOM, Bind::Target(BLOOM_0)),
            (post::POST_SAMPLER, Bind::Sampler(linear)),
        ],
    )?;
    let bloom_v_pass = cx
        .pass("bloom_v")
        .label("quicksilver/bloom_v")
        .views(PassViews::Main)
        .target(BLOOM_2, Access::Clear)
        .id();
    let bloom_v = cx.render_pipeline(
        bloom_v_pass,
        &fs_desc(post::FS_BLOOM_V, None, &[]),
        &[
            (post::POST_BLOOM, Bind::Target(BLOOM_1)),
            (post::POST_SAMPLER, Bind::Sampler(linear)),
        ],
    )?;
    let tonemap_pass = cx
        .pass("tonemap")
        .label("quicksilver/tonemap")
        .views(PassViews::Main)
        .target(TargetName::CANVAS, Access::Clear)
        .id();
    let tonemap = cx.render_pipeline(
        tonemap_pass,
        &fs_desc(post::FS_TONEMAP, None, &[]),
        &[
            (post::POST_FRAME, Bind::Buffer(bufs.frame)),
            (post::POST_HDR, Bind::Target(HDR_COLOR)),
            (post::POST_BLOOM, Bind::Target(BLOOM_2)),
            (post::POST_SAMPLER, Bind::Sampler(linear)),
        ],
    )?;
    let (ui_pass, ui_pipe) = install_ui_pass(cx, bufs, linear, tonemap_pass)?;
    Ok(PostPasses {
        studio_pass,
        studio,
        bloom_ext_pass,
        bloom_ext,
        bloom_h_pass,
        bloom_h,
        bloom_v_pass,
        bloom_v,
        tonemap_pass,
        tonemap,
        ui_pass,
        ui_pipe,
    })
}

fn build_sim_params(
    state: &QuicksilverState,
    drop_cursor: u32,
    frame_index: u32,
) -> SimParams {
    let count = state.preset.count();
    let particle_mass = (MAX_PARTICLES as f32) / (count as f32);
    let stir_grid = QuicksilverState::world_to_grid(state.stir.world_pos);
    let stir_vel_grid = state.stir.world_vel * (0.018 / CELL_SIZE);
    let drop_count = (count / 22).max(2048);
    SimParams {
        dims: [GRID_X, GRID_Y, GRID_Z, count],
        physics: [0.135, particle_mass, 6.0, state.material.stiffness()],
        fluid: [
            state.material.viscosity(),
            -0.34,
            if state.vortex { 1.0 } else { 0.0 },
            if state.zero_g { 1.0 } else { 0.0 },
        ],
        stir_pos: [
            stir_grid.x,
            stir_grid.y,
            stir_grid.z,
            if state.stir.active { state.stir.strength } else { 0.0 },
        ],
        stir_vel: [
            stir_vel_grid.x,
            stir_vel_grid.y,
            stir_vel_grid.z,
            state.stir.radius / CELL_SIZE,
        ],
        drop_info: [drop_cursor % count, drop_count, state.drop_seed, frame_index],
        bounds: [CELL_SIZE, BASIN_HALF_X, BASIN_HALF_Z, BASIN_FLOOR_Y],
        time_info: [state.elapsed, state.material.index() as f32, 0.0, 0.0],
    }
}

fn build_frame_uniform(state: &QuicksilverState) -> FrameUniform {
    let w = math::max(state.viewport[0], 1.0);
    let h = math::max(state.viewport[1], 1.0);
    let aspect = w / h;
    let fov_y = state.camera.fov_y;
    let near = 0.05_f32;
    let far = 80.0_f32;
    let view = glam::Mat4::look_to_rh(state.camera.eye, state.camera.forward, state.camera.up);
    let proj = glam::Mat4::perspective_rh(fov_y, aspect, far, near);
    let inv_proj = proj.inverse();
    let inv_view = view.inverse();
    let count = state.preset.count() as f32;
    let scale_factor = math::pow((MAX_PARTICLES as f32) / count, 1.0 / 3.0);
    let radius_world = CELL_SIZE * 1.18 * scale_factor;
    let focal_y = h * 0.5 / (fov_y * 0.5).tan();
    FrameUniform {
        view: view.to_cols_array_2d(),
        proj: proj.to_cols_array_2d(),
        inv_proj: inv_proj.to_cols_array_2d(),
        inv_view: inv_view.to_cols_array_2d(),
        eye_time: [
            state.camera.eye.x,
            state.camera.eye.y,
            state.camera.eye.z,
            state.elapsed,
        ],
        viewport: [w, h, 1.0 / w, 1.0 / h],
        splat: [radius_world, focal_y, near, far],
        style: [
            state.material.index() as f32,
            if state.zero_g { 1.0 } else { 0.0 },
            if state.vortex { 1.0 } else { 0.0 },
            count,
        ],
        stir: [
            state.stir.world_pos.x,
            state.stir.world_pos.y,
            state.stir.world_pos.z,
            if state.stir.active { 1.0 } else { 0.0 },
        ],
        basin: [BASIN_HALF_X, BASIN_HALF_Z, BASIN_FLOOR_Y, CELL_SIZE],
        grid_dims: [GRID_X as f32, GRID_Y as f32, GRID_Z as f32, 0.0],
    }
}

impl FluidGpu {
    pub(super) fn draw_splat(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.draw(
            &self.splat_pipe,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..self.particle_count,
                key: None,
            },
        );
    }

    fn upload_ui(&mut self, state: &QuicksilverState, cx: &mut Prepare<'_, Self>) {
        let font_size = wgpu::Extent3d {
            width: ATLAS_W,
            height: ATLAS_H,
            depth_or_array_layers: 1,
        };
        if cx.allocate(self.buffers.font_tex, font_size) == Allocation::Fresh {
            let atlas = build_sdf_atlas();
            cx.write_texture(
                self.buffers.font_tex,
                wgpu::Origin3d::ZERO,
                &atlas,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ATLAS_W),
                    rows_per_image: Some(ATLAS_H),
                },
                font_size,
            );
        }
        let layout = compute_layout(state);
        build_ui_quads(state, &layout, &mut self.ui_scratch);
        self.ui_quad_count = self.ui_scratch.len() as u32;
        let bytes_needed = UI_HEADER_BYTES + (self.ui_scratch.len().max(1) * size_of::<UiQuad>()) as u64;
        let fresh = cx.allocate_buffer(self.buffers.ui_quads, bytes_needed.next_power_of_two())
            == Allocation::Fresh;
        if fresh || self.last_ui_rev != state.ui_revision {
            self.ui_bytes.clear();
            let header = [state.viewport[0], state.viewport[1], state.pixel_ratio, 0.0];
            self.ui_bytes.extend_from_slice(bytemuck::bytes_of(&header));
            self.ui_bytes
                .extend_from_slice(bytemuck::cast_slice(&self.ui_scratch));
            cx.write(self.buffers.ui_quads, 0, &self.ui_bytes);
            self.last_ui_rev = state.ui_revision;
        }
    }
}

impl GpuPart for FluidGpu {
    type Reads = Res<'static, QuicksilverState>;
    const NAME: &'static str = "quicksilver";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        declare_targets(cx);
        let buffers = GpuBuffers {
            particles: cx.buffer("fluid_particles", Lifetime::Kept),
            grid_atomic: cx.buffer("fluid_grid_atomic", Lifetime::Kept),
            grid_vel: cx.buffer("fluid_grid_vel", Lifetime::Kept),
            heightfield: cx.buffer("fluid_heightfield", Lifetime::Kept),
            sim_params: cx.buffer("fluid_sim_params", Lifetime::Kept),
            frame: cx.buffer("fluid_frame", Lifetime::Kept),
            ui_quads: cx.buffer("fluid_ui_quads", Lifetime::Kept),
            font_tex: cx.texture("fluid_sdf_font", wgpu::TextureFormat::R8Unorm, Lifetime::Kept),
        };
        let sim = install_sim(cx, &buffers)?;
        let splat_pipe = cx.render_pipeline(
            Scene::named(FLUID_SPLAT_PASS),
            &RenderPipeline {
                vertex: splat::VS_SPLAT,
                fragment: Some(splat::FS_DEPTH_AUX),
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::Compare {
                    compare: wgpu::CompareFunction::GreaterEqual,
                    write: true,
                    bias: wgpu::DepthBiasState::default(),
                },
                stencil: wgpu::StencilState::default(),
                samples: 1,
                targets: &[
                    (None, wgpu::ColorWrites::ALL),
                    (None, wgpu::ColorWrites::ALL),
                ],
                constants: &[],
                per_batch: PerBatch::NONE,
            },
            &[
                (splat::SPLAT_FRAME, Bind::Buffer(buffers.frame)),
                (splat::SPLAT_PARTICLES, Bind::Buffer(buffers.particles)),
            ],
        )?;
        let blur = install_blurs(cx, &buffers)?;
        let post = install_post(cx, &buffers)?;
        Ok(Self {
            buffers,
            sim,
            splat_pipe,
            blur,
            post,
            particle_count: MAX_PARTICLES,
            frame_index: 0,
            last_reset: 0,
            needs_init: true,
            last_drop: 0,
            pending_drop: false,
            drop_cursor: 0,
            last_ui_rev: 0,
            ui_quad_count: 0,
            ui_scratch: Vec::with_capacity(256),
            ui_bytes: Vec::with_capacity(16384),
        })
    }

    fn prepare(&mut self, state: Res<'_, QuicksilverState>, cx: &mut Prepare<'_, Self>) {
        self.particle_count = state.preset.count();
        self.frame_index = self.frame_index.wrapping_add(1);

        let p_bytes = u64::from(MAX_PARTICLES) * PARTICLE_BYTES;
        let g_bytes = u64::from(GRID_CELLS) * 16;
        let h_bytes = u64::from(GRID_X * GRID_Z) * 16;
        let fresh_p = cx.allocate_buffer(self.buffers.particles, p_bytes) == Allocation::Fresh;
        let _ = cx.allocate_buffer(self.buffers.grid_atomic, g_bytes);
        let _ = cx.allocate_buffer(self.buffers.grid_vel, g_bytes);
        let _ = cx.allocate_buffer(self.buffers.heightfield, h_bytes);
        let _ = cx.allocate_buffer(self.buffers.sim_params, size_of::<SimParams>() as u64);
        let _ = cx.allocate_buffer(self.buffers.frame, size_of::<FrameUniform>() as u64);

        if fresh_p || self.last_reset != state.reset_counter {
            self.needs_init = true;
            self.last_reset = state.reset_counter;
        }
        if self.last_drop != state.drop_counter {
            self.pending_drop = true;
            self.last_drop = state.drop_counter;
        }

        let sim_params = build_sim_params(&state, self.drop_cursor, self.frame_index);
        cx.write(self.buffers.sim_params, 0, &[sim_params]);
        let frame_u = build_frame_uniform(&state);
        cx.write(self.buffers.frame, 0, &[frame_u]);

        cx.scale_target(FLUID_THICK_0, 0.35);
        cx.scale_target(FLUID_THICK_1, 0.35);
        cx.scale_target(FLUID_THICK_2, 0.35);
        cx.scale_target(BLOOM_0, 0.25);
        cx.scale_target(BLOOM_1, 0.25);
        cx.scale_target(BLOOM_2, 0.25);

        self.upload_ui(&state, cx);
        cx.run(self.sim.pass);
    }

    fn view(&mut self, _view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        cx.run(self.blur.thick_splat_pass);
        cx.run(self.blur.thick_h_pass);
        cx.run(self.blur.thick_v_pass);
        for pass in self.blur.passes {
            cx.run(pass);
        }
        cx.run(self.post.studio_pass);
        cx.run(self.post.bloom_ext_pass);
        cx.run(self.post.bloom_h_pass);
        cx.run(self.post.bloom_v_pass);
        cx.run(self.post.tonemap_pass);
        if self.ui_quad_count > 0 {
            cx.run(self.post.ui_pass);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) {
        if pass == self.sim.pass {
            self.encode_sim(cx);
            return;
        }
        if pass == self.blur.thick_splat_pass {
            cx.render().draw(
                &self.blur.thick_splat,
                Shape::Pulled {
                    vertices: 0..6,
                    instances: 0..(self.particle_count / 4),
                    key: None,
                },
            );
            return;
        }
        if pass == self.post.ui_pass {
            cx.render().draw(
                &self.post.ui_pipe,
                Shape::Pulled {
                    vertices: 0..6,
                    instances: 0..self.ui_quad_count,
                    key: None,
                },
            );
            return;
        }
        let pipe = if let Some(idx) = self.blur.passes.iter().position(|&p| p == pass) {
            &self.blur.pipelines[idx]
        } else if pass == self.blur.thick_h_pass {
            &self.blur.thick_h
        } else if pass == self.blur.thick_v_pass {
            &self.blur.thick_v
        } else if pass == self.post.studio_pass {
            &self.post.studio
        } else if pass == self.post.bloom_ext_pass {
            &self.post.bloom_ext
        } else if pass == self.post.bloom_h_pass {
            &self.post.bloom_h
        } else if pass == self.post.bloom_v_pass {
            &self.post.bloom_v
        } else if pass == self.post.tonemap_pass {
            &self.post.tonemap
        } else {
            return;
        };
        cx.render().draw(
            pipe,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                key: None,
            },
        );
    }
}

impl FluidGpu {
    fn encode_sim(&mut self, cx: &mut Encode<'_, Self>) {
        let p_groups = self.particle_count.div_ceil(128);
        let g_groups = GRID_CELLS.div_ceil(256);
        let h_groups = [GRID_X.div_ceil(8), GRID_Z.div_ceil(8), 1];

        if self.needs_init {
            cx.recorded();
            cx.dispatch(&self.sim.clear_grid, [g_groups, 1, 1]);
            cx.dispatch(&self.sim.init, [p_groups, 1, 1]);
            if cx.recorded() == Recorded::Exact {
                self.needs_init = false;
            }
        }
        if self.pending_drop && !self.needs_init {
            let drop_count = (self.particle_count / 22).max(2048);
            cx.recorded();
            cx.dispatch(&self.sim.drop_burst, [drop_count.div_ceil(128), 1, 1]);
            if cx.recorded() == Recorded::Exact {
                self.pending_drop = false;
                self.drop_cursor = self.drop_cursor.wrapping_add(drop_count);
            }
        }
        for _ in 0..SUBSTEPS {
            cx.dispatch(&self.sim.p2g_fused, [p_groups, 1, 1]);
            cx.dispatch(&self.sim.update_grid, [g_groups, 1, 1]);
            cx.dispatch(&self.sim.g2p, [p_groups, 1, 1]);
        }
        cx.dispatch(&self.sim.heightfield, h_groups);
    }
}
