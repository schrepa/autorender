engine::brick!(frame);

use bevy_ecs::prelude::Res;
use bevy_ecs::system::SystemParamItem;
use engine_core::{
    Brick, Bricks, Error, GpuStruct, Pass, PassName, PassViews, Sampler, Source, TargetName,
    ViewFacts, Writes, gpu_struct,
};
use engine_render::{
    Access, Allocation, Attach, Bind, Buffer, DepthUse, Encode, FULLSCREEN, GpuPart, Install,
    Lifetime, Own, PerBatch, Pipeline, Prepare, RenderBricks, RenderPipeline, Shape, SharedTarget,
    TargetClear, wgpu,
};

use crate::names::{COLOR, COLOR_PASS, COMPOSITE_PASS, DEPTH};
use crate::play::light::{Grade, Lighting};

gpu_struct! {
    pub struct FrameGrade {
        pub grade: [f32; 4],
    }
}

const GENERATED: &[&str] = &[<FrameGrade as GpuStruct>::WGSL];

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
    pass: engine_render::Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    grade: Buffer<Self>,
    sent: Option<FrameGrade>,
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

impl GpuPart for Composite {
    type Reads = Option<Res<'static, Lighting>>;
    const NAME: &'static str = "frame";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::frame::composite;
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
        cx.frame(composite::SCENE, Bind::Target(COLOR))?;
        let grade = cx.buffer("frame_grade", Lifetime::Kept);
        let sampler = cx.sampler(Sampler::LINEAR_CLAMP);
        let pass = cx
            .pass(COMPOSITE_PASS)
            .label("fruit_fps/composite")
            .span("composite")
            .target(TargetName::CANVAS, Access::Clear)
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: FULLSCREEN,
                fragment: Some(composite::FS_COMPOSITE),
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::None,
                stencil: wgpu::StencilState::default(),
                samples: 1,
                targets: &[(None, wgpu::ColorWrites::ALL)],
                constants: &[],
                per_batch: PerBatch::NONE,
            },
            &[
                (composite::FRAME_GRADE, Bind::Buffer(grade)),
                (composite::SCENE_SAMPLER, Bind::Sampler(sampler)),
            ],
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
        lighting: SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
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
        cx.run(self.pass);
    }

    fn encode(&mut self, _pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        cx.render(Attach::Declared).draw(
            &self.pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
                key: None,
            },
        );
    }
}
