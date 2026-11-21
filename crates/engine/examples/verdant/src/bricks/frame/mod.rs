engine::brick!(frame);

use bevy_ecs::system::SystemParamItem;
use brick_present::{PRESENTED, Present, needs_intermediate};
use engine::prelude::*;
use engine_core::{Error, Filter, GpuStruct, Sampler, TargetName, ViewFacts, ViewSkip, Views, Wrap, gpu_struct};
use engine_render::{Access, Encode, FULLSCREEN, Own, PerBatch, SharedTarget, TargetClear};

use crate::names::{BLOOM_A, BLOOM_B, BLOOM_C, BLOOM_UP_A, BLOOM_UP_B, COLOR, DEPTH};
use crate::state::VerdantState;

gpu_struct! {
    pub struct ToneUniforms {
        pub sun_uv_vis: [f32; 4],
        pub sun_color_shaft: [f32; 4],
        pub grade: [f32; 4],
        pub tint_moon: [f32; 4],
    }
}

const GENERATED: &[&str] = &[<ToneUniforms as GpuStruct>::WGSL];

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
    bloom_d1_pass: engine_render::Pass<Self>,
    bloom_d2_pass: engine_render::Pass<Self>,
    bloom_d3_pass: engine_render::Pass<Self>,
    bloom_u1_pass: engine_render::Pass<Self>,
    bloom_u2_pass: engine_render::Pass<Self>,
    tone_pass: engine_render::Pass<Self>,
    tone_direct_pass: engine_render::Pass<Self>,
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

fn hdr_target(name: TargetName) -> SharedTarget {
    SharedTarget {
        name,
        format: wgpu::TextureFormat::Rgba16Float,
        samples: 1,
        clear: TargetClear::Color([0.0, 0.0, 0.0, 1.0]),
    }
}

fn build_tone(state: &VerdantState) -> ToneUniforms {
    let mood = state.mood();
    let sun_world = state.camera_eye + mood.sun_dir * 420.0;
    let clip = state.view_projection * sun_world.extend(1.0);
    let forward_dot = state.camera_forward.dot(mood.sun_dir);
    let (uv_x, uv_y, vis) = if clip.w > 0.01 && forward_dot > 0.05 {
        let ndc = clip.truncate() / clip.w;
        let ux = ndc.x * 0.5 + 0.5;
        let uy = 0.5 - ndc.y * 0.5;
        let edge = engine::math::clamp(1.65 - (ux - 0.5).abs() - (uy - 0.5).abs(), 0.0, 1.0);
        let align = engine::math::clamp((forward_dot - 0.05) * 2.2, 0.0, 1.0);
        (ux, uy, edge * align)
    } else {
        (0.5, 0.5, 0.0)
    };
    ToneUniforms {
        sun_uv_vis: [uv_x, uv_y, vis, state.sim_time],
        sun_color_shaft: [
            mood.sun_color.x,
            mood.sun_color.y,
            mood.sun_color.z,
            mood.shaft_strength,
        ],
        grade: [mood.exposure, mood.bloom_strength, 1.08, 0.22],
        tint_moon: [mood.haze_color.x, mood.haze_color.y, mood.haze_color.z, mood.moon_factor],
    }
}

impl GpuPart for PostProcess {
    type Reads = (
        Res<'static, VerdantState>,
        Query<'static, 'static, &'static Present>,
        Query<'static, 'static, Entity, With<ViewSkip>>,
        Option<Res<'static, Views>>,
    );
    const NAME: &'static str = "verdant_frame";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::frame::{bloom, tone};
        cx.shared_target(hdr_target(COLOR));
        cx.shared_target(SharedTarget {
            name: DEPTH,
            format: wgpu::TextureFormat::Depth32Float,
            samples: 1,
            clear: TargetClear::Depth(0.0),
        });
        cx.shared_target(hdr_target(BLOOM_A));
        cx.shared_target(hdr_target(BLOOM_B));
        cx.shared_target(hdr_target(BLOOM_C));
        cx.shared_target(hdr_target(BLOOM_UP_B));
        cx.shared_target(hdr_target(BLOOM_UP_A));

        let sampler = cx.sampler(LINEAR_CLAMP);
        let uniforms = cx.buffer("verdant_tone_uniforms", Lifetime::Kept);

        let bloom_d1_pass = cx.pass("verdant_bloom_d1").target(BLOOM_A, Access::Clear).id();
        let bloom_d1_pipe = cx.render_pipeline(
            bloom_d1_pass,
            &fs_pipe(bloom::FS_EXTRACT),
            &[
                (bloom::SRC_TEX, Bind::Target(COLOR)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;

        let bloom_d2_pass = cx.pass("verdant_bloom_d2").target(BLOOM_B, Access::Clear).id();
        let bloom_d2_pipe = cx.render_pipeline(
            bloom_d2_pass,
            &fs_pipe(bloom::FS_DOWN),
            &[
                (bloom::SRC_TEX, Bind::Target(BLOOM_A)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;

        let bloom_d3_pass = cx.pass("verdant_bloom_d3").target(BLOOM_C, Access::Clear).id();
        let bloom_d3_pipe = cx.render_pipeline(
            bloom_d3_pass,
            &fs_pipe(bloom::FS_DOWN),
            &[
                (bloom::SRC_TEX, Bind::Target(BLOOM_B)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;

        let bloom_u1_pass = cx.pass("verdant_bloom_u1").target(BLOOM_UP_B, Access::Clear).id();
        let bloom_u1_pipe = cx.render_pipeline(
            bloom_u1_pass,
            &fs_pipe(bloom::FS_UP),
            &[
                (bloom::SRC_TEX, Bind::Target(BLOOM_C)),
                (bloom::ADD_TEX, Bind::Target(BLOOM_B)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;

        let bloom_u2_pass = cx.pass("verdant_bloom_u2").target(BLOOM_UP_A, Access::Clear).id();
        let bloom_u2_pipe = cx.render_pipeline(
            bloom_u2_pass,
            &fs_pipe(bloom::FS_UP_FINAL),
            &[
                (bloom::SRC_TEX, Bind::Target(BLOOM_UP_B)),
                (bloom::ADD_TEX, Bind::Target(BLOOM_A)),
                (bloom::BLOOM_SAMPLER, Bind::Sampler(sampler)),
                (bloom::SCENE_DEPTH, Bind::Target(DEPTH)),
                (bloom::TONE_PARAMS, Bind::Buffer(uniforms)),
            ],
        )?;

        let tone_binds = [
            (tone::SCENE_COLOR, Bind::Target(COLOR)),
            (tone::SCENE_BLOOM, Bind::Target(BLOOM_UP_A)),
            (tone::TONE_SAMPLER, Bind::Sampler(sampler)),
            (tone::TONE_PARAMS, Bind::Buffer(uniforms)),
        ];
        let tone_pass = cx.pass("verdant_tone").target(PRESENTED, Access::Clear).id();
        let tone_pipe = cx.render_pipeline(tone_pass, &fs_pipe(tone::FS_TONE), &tone_binds)?;

        let tone_direct_pass = cx
            .pass("verdant_tone_direct")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        let tone_direct_pipe =
            cx.render_pipeline(tone_direct_pass, &fs_pipe(tone::FS_TONE), &tone_binds)?;

        Ok(Self {
            bloom_d1_pass,
            bloom_d2_pass,
            bloom_d3_pass,
            bloom_u1_pass,
            bloom_u2_pass,
            tone_pass,
            tone_direct_pass,
            bloom_d1_pipe,
            bloom_d2_pipe,
            bloom_d3_pipe,
            bloom_u1_pipe,
            bloom_u2_pipe,
            tone_pipe,
            tone_direct_pipe,
            uniforms,
            direct: false,
            skipped: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (state, presents, skipped, views): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.skipped.clear();
        self.skipped.extend(&skipped);
        let scale = state.render_scale;
        let pixel_ratio = views.map_or(1.0, |v| v.frame().pixel_ratio);
        self.direct = !needs_intermediate(presents.iter().next().copied(), scale, pixel_ratio);
        cx.scale_target(COLOR, scale);
        cx.scale_target(DEPTH, scale);
        cx.scale_target(BLOOM_A, scale * 0.5);
        cx.scale_target(BLOOM_B, scale * 0.25);
        cx.scale_target(BLOOM_C, scale * 0.125);
        cx.scale_target(BLOOM_UP_B, scale * 0.25);
        cx.scale_target(BLOOM_UP_A, scale * 0.5);

        let u = build_tone(&state);
        let _ = cx.allocate_buffer(self.uniforms, size_of::<ToneUniforms>() as u64);
        cx.write(self.uniforms, 0, &[u]);
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.skipped.contains(&view.view) {
            return;
        }
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
        let pipe = if pass == self.bloom_d1_pass {
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
