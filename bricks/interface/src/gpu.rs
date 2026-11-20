use engine::prelude::Res;
use engine_core::{Error, Filter, PassViews, Sampler, TargetName, ViewFacts, Wrap};
use engine_render::{
    Access, Allocation, Bind, Buffer, DepthUse, Encode, GpuPart, Install, Lifetime, Own, PerBatch,
    Pipeline, Prepare, RenderPipeline, Shape, wgpu,
};

use super::draw::{InterfaceDraw, Quad};
use super::shaders::interface::quads;

const HEADER_BYTES: u64 = 16;

const INTERFACE_PASS: &str = "interface";

const LINEAR_CLAMP: Sampler = Sampler {
    wrap_u: Wrap::ClampToEdge,
    wrap_v: Wrap::ClampToEdge,
    wrap_w: Wrap::ClampToEdge,
    mag_filter: Filter::Linear,
    min_filter: Filter::Linear,
    mipmap_filter: Some(Filter::Linear),
    compare: None,
    anisotropy: 1,
};

pub(super) struct InterfaceGpu {
    pass: engine_render::Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    quads: Buffer<Self>,
    filled: Option<Pipeline<Self, Own>>,
    sent: Option<u64>,
    count: u32,
    staged: Vec<u8>,
}

impl GpuPart for InterfaceGpu {
    type Reads = Res<'static, InterfaceDraw>;
    const NAME: &'static str = "interface";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let quads_buffer = cx.buffer("interface_quads", Lifetime::Kept);
        let sampler = cx.sampler(LINEAR_CLAMP);
        let pass = cx
            .pass(INTERFACE_PASS)
            .label("interface")
            .span("interface")
            .views(PassViews::Main)
            .target(TargetName::CANVAS, Access::Load)
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: quads::VS_INTERFACE,
                fragment: Some(quads::FS_INTERFACE),
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
                (quads::INTERFACE_QUADS, Bind::Buffer(quads_buffer)),
                (quads::INTERFACE_ICONS, Bind::Asset),
                (quads::INTERFACE_GLYPHS_0, Bind::Asset),
                (quads::INTERFACE_GLYPHS_1, Bind::Asset),
                (quads::INTERFACE_PICTURE_0, Bind::Asset),
                (quads::INTERFACE_PICTURE_1, Bind::Asset),
                (quads::INTERFACE_PICTURE_2, Bind::Asset),
                (quads::INTERFACE_PICTURE_3, Bind::Asset),
                (quads::INTERFACE_PICTURE_4, Bind::Asset),
                (quads::INTERFACE_PICTURE_5, Bind::Asset),
                (quads::INTERFACE_PICTURE_6, Bind::Asset),
                (quads::INTERFACE_PICTURE_7, Bind::Asset),
                (quads::INTERFACE_PICTURE_8, Bind::Asset),
                (quads::INTERFACE_PICTURE_9, Bind::Asset),
                (quads::INTERFACE_PICTURE_10, Bind::Asset),
                (quads::INTERFACE_PICTURE_11, Bind::Asset),
                (quads::INTERFACE_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;
        Ok(Self {
            pass,
            pipeline,
            quads: quads_buffer,
            filled: None,
            sent: None,
            count: 0,
            staged: Vec::new(),
        })
    }

    fn prepare(&mut self, draw: Res<'_, InterfaceDraw>, cx: &mut Prepare<'_, Self>) {
        self.filled = draw
            .atlases
            .map(|atlases| cx.fill(&self.pipeline, &atlases));
        let parts = draw.parts();
        let count: usize = parts.iter().map(|part| part.len()).sum();
        self.count = count as u32;
        let wanted = HEADER_BYTES + (count.max(1) * size_of::<Quad>()) as u64;
        let fresh = cx.allocate_buffer(self.quads, wanted.next_power_of_two()) == Allocation::Fresh;
        if fresh || self.sent != Some(draw.revision) {
            let total = HEADER_BYTES as usize + count * size_of::<Quad>();
            self.staged.clear();
            self.staged.reserve(total.saturating_sub(self.staged.capacity()));
            let header = [draw.size.x, draw.size.y, draw.sharpen_levels, 0.0];
            self.staged.extend_from_slice(bytemuck::bytes_of(&header));
            for part in parts.into_iter().filter(|part| !part.is_empty()) {
                self.staged.extend_from_slice(bytemuck::cast_slice(part));
            }
            cx.write(self.quads, 0, &self.staged);
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
