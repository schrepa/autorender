engine::brick!(scene_pipeline);

use engine_core::{
    Brick, Bricks, Error, Pass, PassName, PassViews, Source, TargetName, ViewFacts, Writes,
};
use engine_render::{
    Access, Attach, Bind, DepthUse, Encode, FULLSCREEN, GpuPart, Install, Own, PerBatch, Pipeline,
    Prepare, RenderBricks, RenderPipeline, SharedTarget, Shape, TargetClear, wgpu,
};

pub const COLOR: TargetName = TargetName("color");

pub const DEPTH: TargetName = TargetName("depth");

pub const LIT: Writes = Writes(&[COLOR, DEPTH]);

pub struct ColorPass;

impl PassName for ColorPass {
    const NAME: &'static str = "color";
    const PASS: Pass = Pass {
        writes: LIT,
        views: PassViews::Every,
    };
}

pub struct ScenePipelineBrick;

impl Brick for ScenePipelineBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks.pass::<ColorPass>().part::<Composite>();
    }
}

struct Composite {
    pass: engine_render::Pass<Self>,
    pipeline: Pipeline<Self, Own>,
}

impl GpuPart for Composite {
    type Reads = ();
    const NAME: &'static str = "composite";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::scene_pipeline::composite;
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
        cx.frame(composite::DEPTH_TEXTURE, Bind::Target(DEPTH))?;
        let pass = cx
            .pass("composite")
            .label("engine/post-composite")
            .span("post_composite")
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
            &[],
        )?;
        Ok(Self { pass, pipeline })
    }

    fn prepare(&mut self, (): (), _cx: &mut Prepare<'_, Self>) {}

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
