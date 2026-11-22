engine::brick!(frame);

use bevy_ecs::system::SystemParamItem;
use brick_nanite::{NANITE_VIS_HW, NaniteConfig};
use brick_present::{PRESENTED, Present, needs_intermediate};
use engine::prelude::*;
use engine_core::{
    Compare, Error, Filter, GpuStruct, Sampler, TargetName, ViewFacts, ViewSkip, Views, Wrap,
    gpu_struct,
};
use engine_render::{
    Access, Allocation, ComputePipeline, Encode, FULLSCREEN, Own, PerBatch, SharedTarget,
    TargetClear, Texture, wgpu,
};

use crate::bc_decode::{decode_bc5_level, decode_bc7_level};
use crate::gi::GiUploadBridge;
use crate::names::{
    BLOOM_A, BLOOM_B, BLOOM_C, BLOOM_UP_A, BLOOM_UP_B, HDR_COLOR, SUN_DIR, TAA_HDR,
};
use crate::pbr::{
    GpuMaterialParams, MAX_MATERIALS, PbrChannel, PbrMipUpload, PbrUploadBridge, mip_count_for,
};
use crate::state::MonolithState;

gpu_struct! {
    pub struct FrameUniforms {
        pub curr_unjittered_vp: [[f32; 4]; 4],
        pub prev_unjittered_vp: [[f32; 4]; 4],
        pub sun_dir_illum: [f32; 4],
        pub sun_color_shaft: [f32; 4],
        pub sun_uv_vis: [f32; 4],
        pub grade: [f32; 4],
        pub fog_params: [f32; 4],
        pub fog_color: [f32; 4],
        pub sky_zenith: [f32; 4],
        pub sky_horiz_sun: [f32; 4],
        pub sky_horiz_opp: [f32; 4],
        pub gi_min_pad: [f32; 4],
        pub gi_max_pad: [f32; 4],
        pub taa_params: [f32; 4],
    }
}

const GENERATED: &[&str] = &[
    <FrameUniforms as GpuStruct>::WGSL,
    <GpuMaterialParams as GpuStruct>::WGSL,
];

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

const PBR_ANISO_REPEAT: Sampler = Sampler {
    wrap_u: Wrap::Repeat,
    wrap_v: Wrap::Repeat,
    wrap_w: Wrap::ClampToEdge,
    mag_filter: Filter::Linear,
    min_filter: Filter::Linear,
    mipmap_filter: Some(Filter::Linear),
    compare: None,
    anisotropy: 4,
};

const SHADOW_CMP: Sampler = Sampler {
    wrap_u: Wrap::ClampToEdge,
    wrap_v: Wrap::ClampToEdge,
    wrap_w: Wrap::ClampToEdge,
    mag_filter: Filter::Linear,
    min_filter: Filter::Linear,
    mipmap_filter: None,
    compare: Some(Compare::GreaterEqual),
    anisotropy: 1,
};

pub struct FrameBrick;

impl Brick for FrameBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks.generated(GENERATED).part::<PostProcess>();
    }
}

struct FramePasses {
    resolve: engine_render::Pass<PostProcess>,
    taa: engine_render::Pass<PostProcess>,
    taa_store: engine_render::Pass<PostProcess>,
    bloom_d1: engine_render::Pass<PostProcess>,
    bloom_d2: engine_render::Pass<PostProcess>,
    bloom_d3: engine_render::Pass<PostProcess>,
    bloom_u1: engine_render::Pass<PostProcess>,
    bloom_u2: engine_render::Pass<PostProcess>,
    tone: engine_render::Pass<PostProcess>,
    tone_direct: engine_render::Pass<PostProcess>,
}

struct FramePipes {
    resolve: Pipeline<PostProcess, Own>,
    taa: Pipeline<PostProcess, Own>,
    taa_store: ComputePipeline<PostProcess>,
    bloom_d1: Pipeline<PostProcess, Own>,
    bloom_d2: Pipeline<PostProcess, Own>,
    bloom_d3: Pipeline<PostProcess, Own>,
    bloom_u1: Pipeline<PostProcess, Own>,
    bloom_u2: Pipeline<PostProcess, Own>,
    tone: Pipeline<PostProcess, Own>,
    tone_direct: Pipeline<PostProcess, Own>,
}

struct FrameResources {
    uniforms: Buffer<PostProcess>,
    mat_params: Buffer<PostProcess>,
    mat_albedo: Texture<PostProcess>,
    mat_normal: Texture<PostProcess>,
    mat_orm: Texture<PostProcess>,
    gi_sh_r: Texture<PostProcess>,
    gi_sh_g: Texture<PostProcess>,
    gi_sh_b: Texture<PostProcess>,
    taa_history: Texture<PostProcess>,
}

struct PostProcess {
    passes: FramePasses,
    pipes: FramePipes,
    res: FrameResources,
    gi_min: [f32; 3],
    gi_max: [f32; 3],
    gi_loaded: bool,
    render_size: (u32, u32),
    direct: bool,
    skipped: Vec<Entity>,
}

fn fs_pipe(fragment: engine_core::Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: FULLSCREEN,
        fragment: Some(fragment),
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: &[(None, wgpu::ColorWrites::ALL)],
        constants: &[],
        per_batch: PerBatch::NONE,
    }
}

fn bloom_pipe(fragment: engine_core::Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: shaders::frame::bloom::VS_BLOOM,
        fragment: Some(fragment),
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        stencil: wgpu::StencilState::default(),
        samples: 1,
        targets: &[(None, wgpu::ColorWrites::ALL)],
        constants: &[],
        per_batch: PerBatch::NONE,
    }
}

fn hdr_target(name: TargetName) -> SharedTarget {
    SharedTarget {
        name,
        format: wgpu::TextureFormat::Rgba16Float,
        samples: 1,
        clear: TargetClear::Color([0.0, 0.0, 0.0, 0.0]),
    }
}

fn build_frame_uniforms(
    state: &MonolithState,
    cfg: &NaniteConfig,
    gi_min: [f32; 3],
    gi_max: [f32; 3],
    gi_loaded: bool,
    force_taa_reset: bool,
) -> FrameUniforms {
    let sun_dir = Vec3::from_array(SUN_DIR).normalize();
    let sun_world = state.camera_eye + sun_dir * 600.0;
    let clip = state.view_projection * sun_world.extend(1.0);
    let forward_dot = state.camera_forward.dot(sun_dir);
    let (uv_x, uv_y, vis) = if clip.w > 0.01 && forward_dot > 0.08 {
        let ndc = clip.truncate() / clip.w;
        let ux = ndc.x * 0.5 + 0.5;
        let uy = 0.5 - ndc.y * 0.5;
        let edge = engine::math::clamp(1.95 - (ux - 0.5).abs() - (uy - 0.5).abs(), 0.0, 1.0);
        let align = engine::math::clamp((forward_dot - 0.08) * 1.9, 0.0, 1.0);
        (ux, uy, edge * align)
    } else {
        (0.5, 0.5, 0.0)
    };
    let reset = if state.taa_reset || force_taa_reset {
        1.0
    } else {
        0.0
    };
    FrameUniforms {
        curr_unjittered_vp: state.view_projection.to_cols_array_2d(),
        prev_unjittered_vp: state.prev_unjittered_vp.to_cols_array_2d(),
        sun_dir_illum: [sun_dir.x, sun_dir.y, sun_dir.z, 13.5],
        sun_color_shaft: [1.00, 0.91, 0.76, 0.22],
        sun_uv_vis: [uv_x, uv_y, vis, state.sim_time],
        grade: [1.02, 0.055, 1.06, 0.09],
        fog_params: [0.0009, 0.04, 0.0, cfg.view_mode.index() as f32],
        fog_color: [0.88, 0.90, 0.95, 0.72],
        sky_zenith: [0.16, 0.38, 0.82, 1.0],
        sky_horiz_sun: [0.98, 0.90, 0.76, 1.0],
        sky_horiz_opp: [0.64, 0.76, 0.92, 1.0],
        gi_min_pad: [
            gi_min[0],
            gi_min[1],
            gi_min[2],
            if gi_loaded { 1.0 } else { 0.0 },
        ],
        gi_max_pad: [
            gi_max[0],
            gi_max[1],
            gi_max[2],
            if cfg.shadows_enabled { 1.0 } else { 0.0 },
        ],
        taa_params: [
            state.jitter_px[0],
            state.jitter_px[1],
            reset,
            (state.frame_count & 1023) as f32,
        ],
    }
}

fn is_bc_format(fmt: wgpu::TextureFormat) -> bool {
    matches!(
        fmt,
        wgpu::TextureFormat::Bc7RgbaUnormSrgb
            | wgpu::TextureFormat::Bc7RgbaUnorm
            | wgpu::TextureFormat::Bc5RgUnorm
    )
}

fn upload_pbr_mip(
    cx: &mut Prepare<'_, PostProcess>,
    tex: Texture<PostProcess>,
    mip: &PbrMipUpload,
) {
    let dim = mip.dim.max(1);
    let fmt = cx.texture_format(tex);
    let origin = wgpu::Origin3d {
        x: 0,
        y: 0,
        z: mip.layer_idx,
    };
    if is_bc_format(fmt) {
        let cols = dim.div_ceil(4);
        cx.write_texture_mip(
            tex,
            mip.mip_level,
            origin,
            &mip.bc_bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(cols * 16),
                rows_per_image: Some(cols),
            },
            wgpu::Extent3d {
                width: cols * 4,
                height: cols * 4,
                depth_or_array_layers: 1,
            },
        );
    } else {
        let rgba = match mip.channel {
            PbrChannel::Albedo | PbrChannel::Orm => decode_bc7_level(dim, &mip.bc_bytes),
            PbrChannel::Normal => decode_bc5_level(dim, &mip.bc_bytes),
        };
        cx.write_texture_mip(
            tex,
            mip.mip_level,
            origin,
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dim * 4),
                rows_per_image: Some(dim),
            },
            wgpu::Extent3d {
                width: dim,
                height: dim,
                depth_or_array_layers: 1,
            },
        );
    }
}

struct BloomChain {
    d1_pass: engine_render::Pass<PostProcess>,
    d2_pass: engine_render::Pass<PostProcess>,
    d3_pass: engine_render::Pass<PostProcess>,
    u1_pass: engine_render::Pass<PostProcess>,
    u2_pass: engine_render::Pass<PostProcess>,
    d1_pipe: Pipeline<PostProcess, Own>,
    d2_pipe: Pipeline<PostProcess, Own>,
    d3_pipe: Pipeline<PostProcess, Own>,
    u1_pipe: Pipeline<PostProcess, Own>,
    u2_pipe: Pipeline<PostProcess, Own>,
}

fn install_bloom_chain(
    cx: &mut Install<'_, PostProcess>,
    sampler: engine_render::Sampler<PostProcess>,
    uniforms: Buffer<PostProcess>,
) -> Result<BloomChain, Error> {
    use shaders::frame::bloom;
    let d1_pass = cx
        .pass("monolith_bloom_d1")
        .label("monolith/bloom_d1")
        .span("bloom")
        .target(BLOOM_A, Access::Clear)
        .id();
    let d1_pipe = cx.render_pipeline(
        d1_pass,
        &bloom_pipe(bloom::FS_EXTRACT),
        &[
            (bloom::SRC_TEX, Bind::Target(TAA_HDR)),
            (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
            (bloom::FRAME_PARAMS, Bind::Buffer(uniforms)),
        ],
    )?;

    let d2_pass = cx
        .pass("monolith_bloom_d2")
        .label("monolith/bloom_d2")
        .span("bloom")
        .target(BLOOM_B, Access::Clear)
        .id();
    let d2_pipe = cx.render_pipeline(
        d2_pass,
        &bloom_pipe(bloom::FS_DOWN),
        &[
            (bloom::SRC_TEX, Bind::Target(BLOOM_A)),
            (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
        ],
    )?;

    let d3_pass = cx
        .pass("monolith_bloom_d3")
        .label("monolith/bloom_d3")
        .span("bloom")
        .target(BLOOM_C, Access::Clear)
        .id();
    let d3_pipe = cx.render_pipeline(
        d3_pass,
        &bloom_pipe(bloom::FS_DOWN),
        &[
            (bloom::SRC_TEX, Bind::Target(BLOOM_B)),
            (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
        ],
    )?;

    let u1_pass = cx
        .pass("monolith_bloom_u1")
        .label("monolith/bloom_u1")
        .span("bloom")
        .target(BLOOM_UP_B, Access::Clear)
        .id();
    let u1_pipe = cx.render_pipeline(
        u1_pass,
        &bloom_pipe(bloom::FS_UP),
        &[
            (bloom::SRC_TEX, Bind::Target(BLOOM_C)),
            (bloom::ADD_TEX, Bind::Target(BLOOM_B)),
            (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
        ],
    )?;

    let u2_pass = cx
        .pass("monolith_bloom_u2")
        .label("monolith/godrays")
        .span("godrays")
        .target(BLOOM_UP_A, Access::Clear)
        .id();
    let u2_pipe = cx.render_pipeline(
        u2_pass,
        &bloom_pipe(bloom::FS_UP_FINAL),
        &[
            (bloom::SRC_TEX, Bind::Target(BLOOM_UP_B)),
            (bloom::ADD_TEX, Bind::Target(BLOOM_A)),
            (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
            (bloom::SCENE_TEX, Bind::Target(TAA_HDR)),
            (bloom::FRAME_PARAMS, Bind::Buffer(uniforms)),
        ],
    )?;

    Ok(BloomChain {
        d1_pass,
        d2_pass,
        d3_pass,
        u1_pass,
        u2_pass,
        d1_pipe,
        d2_pipe,
        d3_pipe,
        u1_pipe,
        u2_pipe,
    })
}

fn alloc_frame_resources(cx: &mut Install<'_, PostProcess>) -> FrameResources {
    FrameResources {
        uniforms: cx.buffer("monolith_frame_uniforms", Lifetime::Kept),
        mat_params: cx.buffer("monolith_mat_params", Lifetime::Kept),
        mat_albedo: cx.texture_or(
            "monolith_mat_albedo",
            wgpu::TextureFormat::Bc7RgbaUnormSrgb,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Lifetime::Kept,
        ),
        mat_normal: cx.texture_or(
            "monolith_mat_normal",
            wgpu::TextureFormat::Bc5RgUnorm,
            wgpu::TextureFormat::Rgba8Unorm,
            Lifetime::Kept,
        ),
        mat_orm: cx.texture_or(
            "monolith_mat_orm",
            wgpu::TextureFormat::Bc7RgbaUnorm,
            wgpu::TextureFormat::Rgba8Unorm,
            Lifetime::Kept,
        ),
        gi_sh_r: cx.texture_3d(
            "monolith_gi_sh_r",
            wgpu::TextureFormat::Rgba16Float,
            Lifetime::Kept,
        ),
        gi_sh_g: cx.texture_3d(
            "monolith_gi_sh_g",
            wgpu::TextureFormat::Rgba16Float,
            Lifetime::Kept,
        ),
        gi_sh_b: cx.texture_3d(
            "monolith_gi_sh_b",
            wgpu::TextureFormat::Rgba16Float,
            Lifetime::Kept,
        ),
        taa_history: cx.texture(
            "monolith_taa_history",
            wgpu::TextureFormat::Rgba16Float,
            Lifetime::Kept,
        ),
    }
}

impl PostProcess {
    fn sync_pbr_textures(&mut self, bridge: &PbrUploadBridge, cx: &mut Prepare<'_, Self>) {
        let (albedo_dim, normal_dim, orm_dim, layers, pending, dirty_mats) =
            if let Ok(mut q) = bridge.0.lock() {
                (
                    q.albedo_dim.max(4),
                    q.normal_dim.max(4),
                    q.orm_dim.max(4),
                    q.layer_count.max(2),
                    std::mem::take(&mut q.pending_mips),
                    q.dirty_materials.take(),
                )
            } else {
                (4, 4, 4, 2, Vec::new(), None)
            };

        let _ = cx.allocate_mips(
            self.res.mat_albedo,
            wgpu::Extent3d {
                width: albedo_dim,
                height: albedo_dim,
                depth_or_array_layers: layers,
            },
            mip_count_for(albedo_dim),
        );
        let _ = cx.allocate_mips(
            self.res.mat_normal,
            wgpu::Extent3d {
                width: normal_dim,
                height: normal_dim,
                depth_or_array_layers: layers,
            },
            mip_count_for(normal_dim),
        );
        let _ = cx.allocate_mips(
            self.res.mat_orm,
            wgpu::Extent3d {
                width: orm_dim,
                height: orm_dim,
                depth_or_array_layers: layers,
            },
            mip_count_for(orm_dim),
        );

        for mip in pending {
            if mip.layer_idx < layers {
                let tex = match mip.channel {
                    PbrChannel::Albedo => self.res.mat_albedo,
                    PbrChannel::Normal => self.res.mat_normal,
                    PbrChannel::Orm => self.res.mat_orm,
                };
                upload_pbr_mip(cx, tex, &mip);
            }
        }

        let mat_bytes = (MAX_MATERIALS * size_of::<GpuMaterialParams>()) as u64;
        let fresh_mat = cx.allocate_buffer(self.res.mat_params, mat_bytes) == Allocation::Fresh;
        if let Some(mats) = dirty_mats {
            cx.write(self.res.mat_params, 0, &mats);
        } else if fresh_mat {
            let defaults = vec![GpuMaterialParams::fallback(); MAX_MATERIALS];
            cx.write(self.res.mat_params, 0, &defaults);
        }
    }

    fn sync_gi_volumes(&mut self, bridge: &GiUploadBridge, cx: &mut Prepare<'_, Self>) {
        let pending = bridge
            .0
            .lock()
            .ok()
            .and_then(|mut q| q.pending.take());
        if let Some(vol) = pending {
            let extent = wgpu::Extent3d {
                width: vol.dims[0],
                height: vol.dims[1],
                depth_or_array_layers: vol.dims[2],
            };
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(vol.dims[0] * 8),
                rows_per_image: Some(vol.dims[1]),
            };
            let _ = cx.allocate(self.res.gi_sh_r, extent);
            let _ = cx.allocate(self.res.gi_sh_g, extent);
            let _ = cx.allocate(self.res.gi_sh_b, extent);
            cx.write_texture(self.res.gi_sh_r, wgpu::Origin3d::ZERO, &vol.sh_r, layout, extent);
            cx.write_texture(self.res.gi_sh_g, wgpu::Origin3d::ZERO, &vol.sh_g, layout, extent);
            cx.write_texture(self.res.gi_sh_b, wgpu::Origin3d::ZERO, &vol.sh_b, layout, extent);
            self.gi_min = vol.grid_min;
            self.gi_max = vol.grid_max;
            self.gi_loaded = true;
        } else if !self.gi_loaded {
            let one = wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            };
            let _ = cx.allocate(self.res.gi_sh_r, one);
            let _ = cx.allocate(self.res.gi_sh_g, one);
            let _ = cx.allocate(self.res.gi_sh_b, one);
        }
    }
}

fn declare_frame_targets(cx: &mut Install<'_, PostProcess>) {
    cx.shared_target(hdr_target(HDR_COLOR));
    cx.shared_target(hdr_target(TAA_HDR));
    cx.shared_target(hdr_target(BLOOM_A));
    cx.shared_target(hdr_target(BLOOM_B));
    cx.shared_target(hdr_target(BLOOM_C));
    cx.shared_target(hdr_target(BLOOM_UP_B));
    cx.shared_target(hdr_target(BLOOM_UP_A));
}

impl GpuPart for PostProcess {
    type Reads = (
        Res<'static, MonolithState>,
        Res<'static, NaniteConfig>,
        Res<'static, PbrUploadBridge>,
        Res<'static, GiUploadBridge>,
        Query<'static, 'static, &'static Present>,
        Query<'static, 'static, Entity, With<ViewSkip>>,
        Option<Res<'static, Views>>,
    );
    const NAME: &'static str = "monolith_frame";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::frame::{resolve, taa, tone};
        declare_frame_targets(cx);

        let sampler = cx.sampler(LINEAR_CLAMP);
        let pbr_sampler = cx.sampler(PBR_ANISO_REPEAT);
        let shadow_sampler = cx.sampler(SHADOW_CMP);
        let res = alloc_frame_resources(cx);

        let resolve_pass = cx
            .pass("monolith_resolve")
            .label("monolith/resolve")
            .span("resolve")
            .target(HDR_COLOR, Access::Clear)
            .id();
        let resolve_pipe = cx.render_pipeline(
            resolve_pass,
            &fs_pipe(resolve::FS_RESOLVE),
            &[
                (resolve::VIS_HW_TEX, Bind::Target(NANITE_VIS_HW)),
                (resolve::FRAME_PARAMS, Bind::Buffer(res.uniforms)),
                (resolve::MAT_PARAMS, Bind::Buffer(res.mat_params)),
                (resolve::MAT_ALBEDO, Bind::Texture(res.mat_albedo)),
                (resolve::MAT_NORMAL, Bind::Texture(res.mat_normal)),
                (resolve::MAT_ORM, Bind::Texture(res.mat_orm)),
                (resolve::PBR_SAMPLER, Bind::Sampler(pbr_sampler)),
                (resolve::SHADOW_SAMPLER, Bind::Sampler(shadow_sampler)),
                (resolve::GI_SH_R, Bind::Texture(res.gi_sh_r)),
                (resolve::GI_SH_G, Bind::Texture(res.gi_sh_g)),
                (resolve::GI_SH_B, Bind::Texture(res.gi_sh_b)),
                (resolve::GI_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;

        let taa_pass = cx
            .pass("monolith_taa")
            .label("monolith/taa")
            .span("taa")
            .target(TAA_HDR, Access::Clear)
            .id();
        let taa_pipe = cx.render_pipeline(
            taa_pass,
            &fs_pipe(taa::FS_TAA),
            &[
                (taa::CURR_HDR, Bind::Target(HDR_COLOR)),
                (taa::PREV_HDR, Bind::Texture(res.taa_history)),
                (taa::TAA_SAMPLER, Bind::Sampler(sampler)),
                (taa::FRAME_PARAMS, Bind::Buffer(res.uniforms)),
            ],
        )?;

        let taa_store_pass = cx
            .pass("monolith_taa_store")
            .label("monolith/taa_store")
            .span("taa")
            .id();
        let taa_store_pipe = cx.compute_pipeline(
            taa_store_pass,
            taa::CS_TAA_STORE,
            &[],
            &[
                (taa::TAA_SRC, Bind::Target(TAA_HDR)),
                (taa::TAA_DST, Bind::Texture(res.taa_history)),
            ],
        )?;
        cx.after(taa_store_pass, taa_pass);

        let b = install_bloom_chain(cx, sampler, res.uniforms)?;

        let tone_binds = [
            (tone::SCENE_COLOR, Bind::Target(TAA_HDR)),
            (tone::SCENE_BLOOM, Bind::Target(BLOOM_UP_A)),
            (tone::TONE_SAMPLER, Bind::Sampler(sampler)),
            (tone::FRAME_PARAMS, Bind::Buffer(res.uniforms)),
        ];
        let tone_pass = cx
            .pass("monolith_tone")
            .label("monolith/tone")
            .span("tone")
            .target(PRESENTED, Access::Clear)
            .id();
        let tone_pipe = cx.render_pipeline(tone_pass, &fs_pipe(tone::FS_TONE), &tone_binds)?;

        let tone_direct_pass = cx
            .pass("monolith_tone_direct")
            .label("monolith/tone_direct")
            .span("tone")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        let tone_direct_pipe =
            cx.render_pipeline(tone_direct_pass, &fs_pipe(tone::FS_TONE), &tone_binds)?;

        Ok(Self {
            passes: FramePasses {
                resolve: resolve_pass,
                taa: taa_pass,
                taa_store: taa_store_pass,
                bloom_d1: b.d1_pass,
                bloom_d2: b.d2_pass,
                bloom_d3: b.d3_pass,
                bloom_u1: b.u1_pass,
                bloom_u2: b.u2_pass,
                tone: tone_pass,
                tone_direct: tone_direct_pass,
            },
            pipes: FramePipes {
                resolve: resolve_pipe,
                taa: taa_pipe,
                taa_store: taa_store_pipe,
                bloom_d1: b.d1_pipe,
                bloom_d2: b.d2_pipe,
                bloom_d3: b.d3_pipe,
                bloom_u1: b.u1_pipe,
                bloom_u2: b.u2_pipe,
                tone: tone_pipe,
                tone_direct: tone_direct_pipe,
            },
            res,
            gi_min: [-17.0, 0.35, -8.8],
            gi_max: [19.0, 14.5, 9.6],
            gi_loaded: false,
            render_size: (1, 1),
            direct: false,
            skipped: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (state, cfg, pbr_bridge, gi_bridge, presents, skipped, views): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.skipped.clear();
        self.skipped.extend(&skipped);
        self.sync_pbr_textures(&pbr_bridge, cx);
        self.sync_gi_volumes(&gi_bridge, cx);

        let scale = engine::math::clamp(state.render_scale, 0.25, 1.0);
        let pixel_ratio = views.map_or(1.0, |v| v.frame().pixel_ratio);
        self.direct = !needs_intermediate(presents.iter().next().copied(), scale, pixel_ratio);
        let (cw, ch) = cx.canvas_size();
        let scale_side = |p: u32| ((f64::from(p) * f64::from(scale)).round() as u32).max(1);
        let (rw, rh) = (scale_side(cw), scale_side(ch));
        self.render_size = (rw, rh);

        let hist_fresh = cx.allocate(
            self.res.taa_history,
            wgpu::Extent3d {
                width: rw,
                height: rh,
                depth_or_array_layers: 1,
            },
        ) == Allocation::Fresh;

        let max_bloom_px: u32 = 72_000;
        let bloom_raw =
            (f64::from(max_bloom_px) / (f64::from(cw.max(1)) * f64::from(ch.max(1)))).sqrt() as f32;
        let bloom_frac = engine::math::clamp(bloom_raw, 0.25, 0.5);
        let b_scale = engine::math::min(scale * 0.5, bloom_frac);
        cx.scale_target(HDR_COLOR, scale);
        cx.scale_target(TAA_HDR, scale);
        cx.scale_target_to_pixels(BLOOM_A, max_bloom_px);
        cx.scale_target(BLOOM_B, b_scale * 0.5);
        cx.scale_target(BLOOM_C, b_scale * 0.25);
        cx.scale_target(BLOOM_UP_B, b_scale * 0.5);
        cx.scale_target_to_pixels(BLOOM_UP_A, max_bloom_px);

        let u = build_frame_uniforms(
            &state,
            &cfg,
            self.gi_min,
            self.gi_max,
            self.gi_loaded,
            hist_fresh,
        );
        let _ = cx.allocate_buffer(self.res.uniforms, size_of::<FrameUniforms>() as u64);
        cx.write(self.res.uniforms, 0, &[u]);
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.skipped.contains(&view.view) {
            return;
        }
        cx.run(self.passes.resolve);
        cx.run(self.passes.taa);
        cx.run(self.passes.taa_store);
        cx.run(self.passes.bloom_d1);
        cx.run(self.passes.bloom_d2);
        cx.run(self.passes.bloom_d3);
        cx.run(self.passes.bloom_u1);
        cx.run(self.passes.bloom_u2);
        if self.direct {
            cx.run(self.passes.tone_direct);
        } else {
            cx.run(self.passes.tone);
        }
    }

    fn encode(&mut self, pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        if pass == self.passes.taa_store {
            cx.dispatch(
                &self.pipes.taa_store,
                [
                    self.render_size.0.div_ceil(8),
                    self.render_size.1.div_ceil(8),
                    1,
                ],
            );
            return;
        }
        let pipe = if pass == self.passes.resolve {
            &self.pipes.resolve
        } else if pass == self.passes.taa {
            &self.pipes.taa
        } else if pass == self.passes.bloom_d1 {
            &self.pipes.bloom_d1
        } else if pass == self.passes.bloom_d2 {
            &self.pipes.bloom_d2
        } else if pass == self.passes.bloom_d3 {
            &self.pipes.bloom_d3
        } else if pass == self.passes.bloom_u1 {
            &self.pipes.bloom_u1
        } else if pass == self.passes.bloom_u2 {
            &self.pipes.bloom_u2
        } else if pass == self.passes.tone_direct {
            &self.pipes.tone_direct
        } else {
            &self.pipes.tone
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
