engine::brick!(plates);

use engine_core::{
    Attributes, Brick, Bricks, DrawKind, Error, KindDecl, Pass, PassName, PassViews, Source,
    TargetName, Writes,
};
use engine_render::{
    DepthUse, Draw, GpuPart, Install, PerBatch, Pipeline, Prepare, RenderBricks, RenderPipeline,
    Scene, SharedTarget, TargetClear, wgpu,
};

pub const PLATE: DrawKind = DrawKind("plate");

pub const PLATES_DEPTH: TargetName = TargetName("plates_depth");

pub struct PlatesPass;

impl PassName for PlatesPass {
    const NAME: &'static str = "plates";
    const PASS: Pass = Pass {
        writes: Writes(&[TargetName::CANVAS, PLATES_DEPTH]),
        views: PassViews::Main,
    };
}

pub struct PlatesBrick;

impl Brick for PlatesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .draw_kind(KindDecl {
                kind: PLATE,
                needs: Attributes::NONE,
                streams: &[],
                passes: &[PlatesPass::NAME],
            })
            .pass::<PlatesPass>()
            .draws::<PlatesGpu>(PLATE, PlatesGpu::draw);
    }
}

struct PlatesGpu {
    pipeline: Pipeline<Self, Scene>,
}

impl GpuPart for PlatesGpu {
    type Reads = ();

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::plates::draw;
        cx.shared_target(SharedTarget {
            name: PLATES_DEPTH,
            format: wgpu::TextureFormat::Depth32Float,
            samples: 1,
            clear: TargetClear::Depth(0.0),
        });
        let pipeline = cx.render_pipeline(
            Scene::of::<PlatesPass>(),
            &RenderPipeline {
                vertex: draw::VS_PLATE,
                fragment: Some(draw::FS_PLATE),
                vertices: &[],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..wgpu::PrimitiveState::default()
                },
                depth: DepthUse::Compare {
                    compare: wgpu::CompareFunction::GreaterEqual,
                    write: true,
                    bias: wgpu::DepthBiasState::default(),
                },
                stencil: wgpu::StencilState::default(),
                samples: 1,
                targets: &[(None, wgpu::ColorWrites::ALL)],
                constants: &[],
                per_batch: PerBatch::NONE,
            },
            &[],
        )?;
        Ok(Self { pipeline })
    }

    fn prepare(&mut self, (): (), _: &mut Prepare<'_, Self>) {}
}

impl PlatesGpu {
    fn draw(&self, draw: &mut Draw<'_, Self, Scene>) {
        draw.batches(&self.pipeline, 0..6);
    }
}
