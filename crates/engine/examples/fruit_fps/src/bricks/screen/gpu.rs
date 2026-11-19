use engine::prelude::Res;
use engine_core::{Error, PassViews, Sampler, TargetName, ViewFacts};
use engine_render::{
    Access, Allocation, Bind, Buffer, DepthUse, Encode, GpuPart, Install, Lifetime, Own, PerBatch,
    Pipeline, Prepare, RenderPipeline, Shape, wgpu,
};

use super::shaders::screen::quads;
use crate::play::screen::{Quad, ScreenDraw};

const HEADER_BYTES: u64 = 16;

const SCREEN_PASS: &str = "screen";

pub(super) struct ScreenGpu {
    pass: engine_render::Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    quads: Buffer<Self>,
    filled: Option<Pipeline<Self, Own>>,
    sent: Option<u64>,
    count: u32,
}

impl GpuPart for ScreenGpu {
    type Reads = Res<'static, ScreenDraw>;
    const NAME: &'static str = "screen";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let quads_buffer = cx.buffer("screen_quads", Lifetime::Kept);
        let sampler = cx.sampler(Sampler::LINEAR_CLAMP);
        let pass = cx
            .pass(SCREEN_PASS)
            .label("fruit_fps/screen")
            .span("screen")
            .views(PassViews::Main)
            .target(TargetName::CANVAS, Access::Load)
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: quads::VS_SCREEN,
                fragment: Some(quads::FS_SCREEN),
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
                (quads::SCREEN_QUADS, Bind::Buffer(quads_buffer)),
                (quads::SCREEN_ICONS, Bind::Asset),
                (quads::SCREEN_GLYPHS_0, Bind::Asset),
                (quads::SCREEN_GLYPHS_1, Bind::Asset),
                (quads::SCREEN_PICTURE_0, Bind::Asset),
                (quads::SCREEN_PICTURE_1, Bind::Asset),
                (quads::SCREEN_PICTURE_2, Bind::Asset),
                (quads::SCREEN_PICTURE_3, Bind::Asset),
                (quads::SCREEN_PICTURE_4, Bind::Asset),
                (quads::SCREEN_PICTURE_5, Bind::Asset),
                (quads::SCREEN_PICTURE_6, Bind::Asset),
                (quads::SCREEN_PICTURE_7, Bind::Asset),
                (quads::SCREEN_PICTURE_8, Bind::Asset),
                (quads::SCREEN_PICTURE_9, Bind::Asset),
                (quads::SCREEN_PICTURE_10, Bind::Asset),
                (quads::SCREEN_PICTURE_11, Bind::Asset),
                (quads::SCREEN_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;
        Ok(Self {
            pass,
            pipeline,
            quads: quads_buffer,
            filled: None,
            sent: None,
            count: 0,
        })
    }

    fn prepare(&mut self, draw: Res<'_, ScreenDraw>, cx: &mut Prepare<'_, Self>) {
        self.filled = draw
            .atlases
            .map(|atlases| cx.fill(&self.pipeline, &atlases));
        self.count = draw.quads.len() as u32;
        let wanted = HEADER_BYTES + (draw.quads.len().max(1) * size_of::<Quad>()) as u64;
        let fresh = cx.allocate_buffer(self.quads, wanted.next_power_of_two()) == Allocation::Fresh;
        if fresh || self.sent != Some(draw.revision) {
            cx.write(self.quads, 0, &[draw.size.x, draw.size.y, 0.0, 0.0]);
            if !draw.quads.is_empty() {
                cx.write(self.quads, HEADER_BYTES, &draw.quads);
            }
            self.sent = Some(draw.revision);
        }
    }

    fn view(&mut self, _view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if self.count > 0 && self.filled.is_some() {
            cx.run(self.pass);
        }
    }

    fn encode(&mut self, _pass: engine_render::Pass<Self>, cx: &mut Encode<'_, Self>) {
        let Some(filled) = &self.filled else {
            return;
        };
        cx.render().draw(
            filled,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..self.count,
                key: None,
            },
        );
    }
}
