engine::brick!(frame);

use bevy_ecs::prelude::Res;
use bevy_ecs::system::SystemParamItem;
use engine_core::{
    Brick, Bricks, Error, GpuStruct, Pass, PassName, PassViews, Sampler, Source, TargetName,
    ViewFacts, Writes, gpu_struct,
};
use engine_render::{
    Access, Allocation, Bind, Buffer, DepthUse, Encode, FULLSCREEN, GpuPart, Install, Lifetime,
    Own, PerBatch, Pipeline, Prepare, RenderBricks, RenderPipeline, Shape, SharedTarget,
    TargetClear, wgpu,
};

use crate::names::{COLOR, COLOR_PASS, COMPOSITE_PASS, DEPTH};
use crate::play::light::{Grade, Lighting};
use crate::play::screen::Covered;
use crate::play::view::Controls;

gpu_struct! {
    pub struct FrameGrade {
        pub grade: [f32; 4],
    }
}

const GENERATED: &[&str] = &[<FrameGrade as GpuStruct>::WGSL];

const TONED: TargetName = TargetName("toned");

const TONE_PASS: &str = "tone";

const COVER_PASS: &str = "cover";

pub struct ColorPass;

impl PassName for ColorPass {
    const NAME: &'static str = COLOR_PASS;
    const PASS: Pass = Pass {
        writes: Writes(&[COLOR, DEPTH]),
        views: PassViews::Every,
    };
}

pub struct FrameBrick;

impl Brick for FrameBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .pass::<ColorPass>()
            .generated(GENERATED)
            .part::<Composite>();
    }
}

struct Composite {
    tone_pass: engine_render::Pass<Self>,
    tone: Pipeline<Self, Own>,
    pass: engine_render::Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    cover: engine_render::Pass<Self>,
    grade: Buffer<Self>,
    sent: Option<FrameGrade>,
    covered: bool,
}

fn graded(grade: Grade, antialias: bool) -> FrameGrade {
    FrameGrade {
        grade: [
            grade.exposure,
            grade.contrast,
            grade.saturation,
            f32::from(u8::from(antialias)),
        ],
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

impl GpuPart for Composite {
    type Reads = (
        Option<Res<'static, Lighting>>,
        Option<Res<'static, Controls>>,
        Option<Res<'static, Covered>>,
    );
    const NAME: &'static str = "frame";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::frame::{composite, tone};
        cx.shared_target(SharedTarget {
            name: COLOR,
            format: wgpu::TextureFormat::Rgba16Float,
            samples: 1,
            clear: TargetClear::Color([0.0, 0.0, 0.0, 1.0]),
        });
        cx.shared_target(SharedTarget {
            name: DEPTH,
            format: wgpu::TextureFormat::Depth32Float,
            samples: 1,
            clear: TargetClear::Depth(0.0),
        });
        cx.shared_target(SharedTarget {
            name: TONED,
            format: wgpu::TextureFormat::Rgba8Unorm,
            samples: 1,
            clear: TargetClear::Color([0.0, 0.0, 0.0, 1.0]),
        });
        cx.frame(tone::SCENE, Bind::Target(COLOR))?;
        cx.frame(composite::TONED, Bind::Target(TONED))?;
        let grade = cx.buffer("frame_grade", Lifetime::Kept);
        let sampler = cx.sampler(Sampler::LINEAR_CLAMP);
        let tone_pass = cx
            .pass(TONE_PASS)
            .label("fruit_fps/tone")
            .span("tone")
            .target(TONED, Access::Clear)
            .id();
        let tone = cx.render_pipeline(
            tone_pass,
            &fullscreen(tone::FS_TONE),
            &[(tone::TONE_GRADE, Bind::Buffer(grade))],
        )?;
        let pass = cx
            .pass(COMPOSITE_PASS)
            .label("fruit_fps/composite")
            .span("composite")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        cx.after(pass, tone_pass);
        let pipeline = cx.render_pipeline(
            pass,
            &fullscreen(composite::FS_COMPOSITE),
            &[
                (composite::FRAME_GRADE, Bind::Buffer(grade)),
                (composite::TONED_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;
        let cover = cx
            .pass(COVER_PASS)
            .label("fruit_fps/cover")
            .span("cover")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        Ok(Self {
            tone_pass,
            tone,
            pass,
            pipeline,
            cover,
            grade,
            sent: None,
            covered: false,
        })
    }

    fn prepare(
        &mut self,
        (lighting, controls, covered): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.covered = covered.is_some_and(|covered| covered.0);
        let scale = controls.map_or(1.0, |controls| controls.render_scale);
        cx.scale_target(COLOR, scale);
        cx.scale_target(DEPTH, scale);
        cx.scale_target(TONED, scale);
        let shown = lighting.map_or_else(
            || graded(Grade::default(), false),
            |lighting| graded(lighting.grade, lighting.antialias),
        );
        let bytes = size_of::<FrameGrade>() as u64;
        if cx.allocate_buffer(self.grade, bytes) == Allocation::Fresh || self.sent != Some(shown) {
            cx.write(self.grade, 0, &[shown]);
            self.sent = Some(shown);
        }
    }

    fn view(&mut self, _view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.covered {
            cx.run(self.cover);
            return;
        }
        cx.run(self.tone_pass);
        cx.run(self.pass);
    }

    fn encode(&mut self, pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        if pass == self.cover {
            let _ = cx.render();
            return;
        }
        let pipeline = if pass == self.tone_pass {
            &self.tone
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
