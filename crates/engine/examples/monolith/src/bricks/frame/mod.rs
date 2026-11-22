engine::brick!(frame);

use bevy_ecs::system::SystemParamItem;
use brick_nanite::{NANITE_DEPTH, NANITE_VIS_HW, NaniteConfig};
use brick_present::{PRESENTED, Present, needs_intermediate};
use engine::prelude::*;
use engine_core::{
    Error, Filter, GpuStruct, Sampler, TargetName, ViewFacts, ViewSkip, Views, Wrap, gpu_struct,
};
use engine_render::{Access, Encode, FULLSCREEN, Own, PerBatch, SharedTarget, TargetClear};

use crate::names::{
    BLOOM_A, BLOOM_B, BLOOM_C, BLOOM_UP_A, BLOOM_UP_B, HDR_COLOR, SUN_DIR,
};
use crate::state::MonolithState;

gpu_struct! {
    pub struct FrameUniforms {
        pub sun_dir_illum: [f32; 4],
        pub sun_color_shaft: [f32; 4],
        pub sun_uv_vis: [f32; 4],
        pub grade: [f32; 4],
        pub fog_params: [f32; 4],
        pub fog_color: [f32; 4],
        pub sky_zenith: [f32; 4],
        pub sky_horiz_sun: [f32; 4],
        pub sky_horiz_opp: [f32; 4],
    }
}

const GENERATED: &[&str] = &[<FrameUniforms as GpuStruct>::WGSL];

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

pub struct FrameBrick;

impl Brick for FrameBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks.generated(GENERATED).part::<PostProcess>();
    }
}

struct PostProcess {
    resolve_pass: engine_render::Pass<Self>,
    bloom_d1_pass: engine_render::Pass<Self>,
    bloom_d2_pass: engine_render::Pass<Self>,
    bloom_d3_pass: engine_render::Pass<Self>,
    bloom_u1_pass: engine_render::Pass<Self>,
    bloom_u2_pass: engine_render::Pass<Self>,
    tone_pass: engine_render::Pass<Self>,
    tone_direct_pass: engine_render::Pass<Self>,
    resolve_pipe: Pipeline<Self, Own>,
    bloom_d1_pipe: Pipeline<Self, Own>,
    bloom_d2_pipe: Pipeline<Self, Own>,
    bloom_d3_pipe: Pipeline<Self, Own>,
    bloom_u1_pipe: Pipeline<Self, Own>,
    bloom_u2_pipe: Pipeline<Self, Own>,
    tone_pipe: Pipeline<Self, Own>,
    tone_direct_pipe: Pipeline<Self, Own>,
    uniforms: Buffer<Self>,
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

fn build_frame_uniforms(state: &MonolithState, cfg: &NaniteConfig) -> FrameUniforms {
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
    FrameUniforms {
        sun_dir_illum: [sun_dir.x, sun_dir.y, sun_dir.z, 6.8],
        sun_color_shaft: [1.00, 0.72, 0.44, 0.64],
        sun_uv_vis: [uv_x, uv_y, vis, state.sim_time],
        grade: [0.58, 0.08, 1.08, 0.16],
        fog_params: [0.0020, 0.055, -4.0, cfg.view_mode.index() as f32],
        fog_color: [0.96, 0.66, 0.40, 0.76],
        sky_zenith: [0.13, 0.27, 0.58, 1.0],
        sky_horiz_sun: [1.00, 0.60, 0.30, 1.0],
        sky_horiz_opp: [0.76, 0.59, 0.48, 1.0],
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
            (bloom::SRC_TEX, Bind::Target(HDR_COLOR)),
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
            (bloom::SCENE_TEX, Bind::Target(HDR_COLOR)),
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

impl GpuPart for PostProcess {
    type Reads = (
        Res<'static, MonolithState>,
        Res<'static, NaniteConfig>,
        Query<'static, 'static, &'static Present>,
        Query<'static, 'static, Entity, With<ViewSkip>>,
        Option<Res<'static, Views>>,
    );
    const NAME: &'static str = "monolith_frame";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::frame::{resolve, tone};
        cx.shared_target(hdr_target(HDR_COLOR));
        cx.shared_target(hdr_target(BLOOM_A));
        cx.shared_target(hdr_target(BLOOM_B));
        cx.shared_target(hdr_target(BLOOM_C));
        cx.shared_target(hdr_target(BLOOM_UP_B));
        cx.shared_target(hdr_target(BLOOM_UP_A));

        let sampler = cx.sampler(LINEAR_CLAMP);
        let uniforms = cx.buffer("monolith_frame_uniforms", Lifetime::Kept);

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
                (resolve::HW_DEPTH_TEX, Bind::Target(NANITE_DEPTH)),
                (resolve::FRAME_PARAMS, Bind::Buffer(uniforms)),
            ],
        )?;

        let b = install_bloom_chain(cx, sampler, uniforms)?;

        let tone_binds = [
            (tone::SCENE_COLOR, Bind::Target(HDR_COLOR)),
            (tone::SCENE_BLOOM, Bind::Target(BLOOM_UP_A)),
            (tone::TONE_SAMPLER, Bind::Sampler(sampler)),
            (tone::FRAME_PARAMS, Bind::Buffer(uniforms)),
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
            resolve_pass,
            bloom_d1_pass: b.d1_pass,
            bloom_d2_pass: b.d2_pass,
            bloom_d3_pass: b.d3_pass,
            bloom_u1_pass: b.u1_pass,
            bloom_u2_pass: b.u2_pass,
            tone_pass,
            tone_direct_pass,
            resolve_pipe,
            bloom_d1_pipe: b.d1_pipe,
            bloom_d2_pipe: b.d2_pipe,
            bloom_d3_pipe: b.d3_pipe,
            bloom_u1_pipe: b.u1_pipe,
            bloom_u2_pipe: b.u2_pipe,
            tone_pipe,
            tone_direct_pipe,
            uniforms,
            direct: false,
            skipped: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (state, cfg, presents, skipped, views): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.skipped.clear();
        self.skipped.extend(&skipped);
        let scale = state.render_scale;
        let pixel_ratio = views.map_or(1.0, |v| v.frame().pixel_ratio);
        self.direct = !needs_intermediate(presents.iter().next().copied(), scale, pixel_ratio);
        let (cw, ch) = cx.canvas_size();
        let max_bloom_px: u32 = 144_000;
        let bloom_raw =
            (f64::from(max_bloom_px) / (f64::from(cw.max(1)) * f64::from(ch.max(1)))).sqrt() as f32;
        let bloom_frac = engine::math::clamp(bloom_raw, 0.25, 0.5);
        let b_scale = engine::math::min(scale * 0.5, bloom_frac);
        cx.scale_target(HDR_COLOR, scale);
        cx.scale_target_to_pixels(BLOOM_A, max_bloom_px);
        cx.scale_target(BLOOM_B, b_scale * 0.5);
        cx.scale_target(BLOOM_C, b_scale * 0.25);
        cx.scale_target(BLOOM_UP_B, b_scale * 0.5);
        cx.scale_target_to_pixels(BLOOM_UP_A, max_bloom_px);

        let u = build_frame_uniforms(&state, &cfg);
        let _ = cx.allocate_buffer(self.uniforms, size_of::<FrameUniforms>() as u64);
        cx.write(self.uniforms, 0, &[u]);
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.skipped.contains(&view.view) {
            return;
        }
        cx.run(self.resolve_pass);
        cx.run(self.bloom_d1_pass);
        cx.run(self.bloom_d2_pass);
        cx.run(self.bloom_d3_pass);
        cx.run(self.bloom_u1_pass);
        cx.run(self.bloom_u2_pass);
        if self.direct {
            cx.run(self.tone_direct_pass);
        } else {
            cx.run(self.tone_pass);
        }
    }

    fn encode(&mut self, pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        let pipe = if pass == self.resolve_pass {
            &self.resolve_pipe
        } else if pass == self.bloom_d1_pass {
            &self.bloom_d1_pipe
        } else if pass == self.bloom_d2_pass {
            &self.bloom_d2_pipe
        } else if pass == self.bloom_d3_pass {
            &self.bloom_d3_pipe
        } else if pass == self.bloom_u1_pass {
            &self.bloom_u1_pipe
        } else if pass == self.bloom_u2_pass {
            &self.bloom_u2_pipe
        } else if pass == self.tone_direct_pass {
            &self.tone_direct_pipe
        } else {
            &self.tone_pipe
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
