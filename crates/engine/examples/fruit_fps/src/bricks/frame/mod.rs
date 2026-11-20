engine::brick!(frame);

use bevy_ecs::prelude::{Entity, Query, Res, With};
use bevy_ecs::system::SystemParamItem;
use brick_present::{PRESENTED, Present, Scale, needs_intermediate};
use engine_core::{
    Brick, Bricks, Error, GpuStruct, Pass, PassName, PassViews, Source, TargetName, ViewFacts,
    ViewSkip, Views, Writes, gpu_struct,
};
use engine_render::{
    Access, Allocation, Bind, Buffer, DepthUse, Encode, FULLSCREEN, GpuPart, Install, Lifetime,
    Own, PerBatch, Pipeline, Prepare, RenderBricks, RenderPipeline, Shape, SharedTarget,
    TargetClear, wgpu,
};

use crate::names::{COLOR, COLOR_PASS, DEPTH, DEPTH_PASS, REVEAL_PASS};
use crate::play::light::{Grade, Lighting};
use crate::play::view::Controls;

gpu_struct! {
    pub struct FrameGrade {
        pub grade: [f32; 4],
    }
}

const GENERATED: &[&str] = &[<FrameGrade as GpuStruct>::WGSL];

const TONE_PASS: &str = "tone";

const TONE_DIRECT_PASS: &str = "tone_direct";

const SKY: [f64; 4] = [0.36, 0.58, 0.88, 1.0];

pub struct DepthPass;

impl PassName for DepthPass {
    const NAME: &'static str = DEPTH_PASS;
    const PASS: Pass = Pass {
        writes: Writes(&[COLOR, DEPTH]),
        views: PassViews::Every,
    };
}

pub struct ColorPass;

impl PassName for ColorPass {
    const NAME: &'static str = COLOR_PASS;
    const PASS: Pass = Pass {
        writes: Writes(&[COLOR, DEPTH]),
        views: PassViews::Every,
    };
}

pub struct RevealPass;

impl PassName for RevealPass {
    const NAME: &'static str = REVEAL_PASS;
    const PASS: Pass = Pass {
        writes: Writes(&[COLOR, DEPTH]),
        views: PassViews::Every,
    };
}

impl Scale for Controls {
    fn fraction(&self) -> f32 {
        self.render_scale
    }

    fn direct_supported() -> bool {
        true
    }
}

pub struct FrameBrick;

impl Brick for FrameBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .pass::<DepthPass>()
            .pass::<ColorPass>()
            .pass::<RevealPass>()
            .generated(GENERATED)
            .part::<Tone>();
    }
}

struct Tone {
    pass: engine_render::Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    direct_pass: engine_render::Pass<Self>,
    direct_pipeline: Pipeline<Self, Own>,
    grade: Buffer<Self>,
    sent: Option<FrameGrade>,
    direct: bool,
    skipped: Vec<Entity>,
}

fn graded(grade: Grade) -> FrameGrade {
    FrameGrade {
        grade: [grade.exposure, grade.contrast, grade.saturation, 0.0],
    }
}

fn fullscreen(fragment: engine_core::Entry) -> RenderPipeline<'static> {
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

impl GpuPart for Tone {
    type Reads = (
        Option<Res<'static, Lighting>>,
        Option<Res<'static, Controls>>,
        Query<'static, 'static, &'static Present>,
        Query<'static, 'static, Entity, With<ViewSkip>>,
        Option<Res<'static, Views>>,
    );
    const NAME: &'static str = "frame";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::frame::tone;
        cx.shared_target_or(
            SharedTarget {
                name: COLOR,
                format: wgpu::TextureFormat::Rg11b10Ufloat,
                samples: 1,
                clear: TargetClear::Color(SKY),
            },
            wgpu::TextureFormat::Rgba16Float,
        );
        cx.shared_target(SharedTarget {
            name: DEPTH,
            format: wgpu::TextureFormat::Depth32Float,
            samples: 1,
            clear: TargetClear::Depth(0.0),
        });
        cx.frame(tone::SCENE, Bind::Target(COLOR))?;
        let grade = cx.buffer("frame_grade", Lifetime::Kept);
        let bound = [(tone::TONE_GRADE, Bind::Buffer(grade))];
        let pass = cx
            .pass(TONE_PASS)
            .label("fruit_fps/tone")
            .span("tone")
            .target(PRESENTED, Access::Clear)
            .id();
        let pipeline = cx.render_pipeline(pass, &fullscreen(tone::FS_TONE), &bound)?;
        let direct_pass = cx
            .pass(TONE_DIRECT_PASS)
            .label("fruit_fps/tone_direct")
            .span("tone")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        let direct_pipeline =
            cx.render_pipeline(direct_pass, &fullscreen(tone::FS_TONE), &bound)?;
        Ok(Self {
            pass,
            pipeline,
            direct_pass,
            direct_pipeline,
            grade,
            sent: None,
            direct: false,
            skipped: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (lighting, controls, presents, skipped, views): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.skipped.clear();
        self.skipped.extend(&skipped);
        let scale = controls.map_or(1.0, |controls| controls.render_scale);
        let pixel_ratio = views.map_or(1.0, |views| views.frame().pixel_ratio);
        self.direct = !needs_intermediate(presents.iter().next().copied(), scale, pixel_ratio);
        cx.scale_target(COLOR, scale);
        cx.scale_target(DEPTH, scale);
        let shown = lighting.map_or_else(
            || graded(Grade::default()),
            |lighting| graded(lighting.grade),
        );
        let bytes = size_of::<FrameGrade>() as u64;
        if cx.allocate_buffer(self.grade, bytes) == Allocation::Fresh || self.sent != Some(shown) {
            cx.write(self.grade, 0, &[shown]);
            self.sent = Some(shown);
        }
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.skipped.contains(&view.view) {
            return;
        }
        if self.direct {
            cx.run(self.direct_pass);
        } else {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        let pipeline = if pass == self.direct_pass {
            &self.direct_pipeline
        } else {
            &self.pipeline
        };
        cx.render().draw(
            pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                key: None,
            },
        );
    }
}
