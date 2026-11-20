engine::brick!(frame);

use bevy_ecs::prelude::Res;
use bevy_ecs::system::SystemParamItem;
use brick_present::{PRESENTED, Scale};
use engine_core::{
    Brick, Bricks, Error, GpuStruct, Pass, PassName, PassViews, Source, ViewFacts, Writes,
    gpu_struct,
};
use engine_render::{
    Access, Allocation, Bind, Buffer, DepthUse, Encode, FULLSCREEN, GpuPart, Install, Lifetime,
    Own, PerBatch, Pipeline, Prepare, RenderBricks, RenderPipeline, Shape, SharedTarget,
    TargetClear, wgpu,
};

use crate::names::{COLOR, COLOR_PASS, DEPTH};
use crate::play::light::{Grade, Lighting};
use crate::play::view::Controls;

gpu_struct! {
    pub struct FrameGrade {
        pub grade: [f32; 4],
    }
}

const GENERATED: &[&str] = &[<FrameGrade as GpuStruct>::WGSL];

const TONE_PASS: &str = "tone";

const SKY: [f64; 4] = [0.36, 0.58, 0.88, 1.0];

pub struct ColorPass;

impl PassName for ColorPass {
    const NAME: &'static str = COLOR_PASS;
    const PASS: Pass = Pass {
        writes: Writes(&[COLOR, DEPTH]),
        views: PassViews::Every,
    };
}

impl Scale for Controls {
    fn fraction(&self) -> f32 {
        self.render_scale
    }
}

pub struct FrameBrick;

impl Brick for FrameBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .pass::<ColorPass>()
            .generated(GENERATED)
            .part::<Tone>();
    }
}

struct Tone {
    pass: engine_render::Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    grade: Buffer<Self>,
    sent: Option<FrameGrade>,
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
    );
    const NAME: &'static str = "frame";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::frame::tone;
        cx.shared_target(SharedTarget {
            name: COLOR,
            format: wgpu::TextureFormat::Rgba16Float,
            samples: 1,
            clear: TargetClear::Color(SKY),
        });
        cx.shared_target(SharedTarget {
            name: DEPTH,
            format: wgpu::TextureFormat::Depth32Float,
            samples: 1,
            clear: TargetClear::Depth(0.0),
        });
        cx.frame(tone::SCENE, Bind::Target(COLOR))?;
        let grade = cx.buffer("frame_grade", Lifetime::Kept);
        let pass = cx
            .pass(TONE_PASS)
            .label("fruit_fps/tone")
            .span("tone")
            .target(PRESENTED, Access::Clear)
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &fullscreen(tone::FS_TONE),
            &[(tone::TONE_GRADE, Bind::Buffer(grade))],
        )?;
        Ok(Self {
            pass,
            pipeline,
            grade,
            sent: None,
        })
    }

    fn prepare(
        &mut self,
        (lighting, controls): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let scale = controls.map_or(1.0, |controls| controls.render_scale);
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

    fn view(&mut self, _view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        cx.run(self.pass);
    }

    fn encode(&mut self, _pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        cx.render().draw(
            &self.pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                key: None,
            },
        );
    }
}
