use engine_core::Error;
use engine_render::{
    Bind, DepthUse, Draw, Install, Lifetime, MainView, Pipeline, Prepare, RenderPipeline,
    SamplerDesc, Shape, Texture, wgpu,
};

use crate::SpritesGpu;
use crate::shaders::sprites::flat;

pub const LIGHT: [u8; 4] = [224, 64, 32, 255];

pub const DARK: [u8; 4] = [32, 96, 224, 255];

const TEXELS: [[u8; 4]; 4] = [LIGHT, DARK, DARK, LIGHT];

const CHECKER_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: 2,
    height: 2,
    depth_or_array_layers: 1,
};

pub(crate) struct Flat {
    checker: Texture<SpritesGpu>,
    pipeline: Pipeline<SpritesGpu, MainView>,
    written: bool,
}

impl Flat {
    pub(crate) fn install(cx: &mut Install<'_, SpritesGpu>) -> Result<Self, Error> {
        let checker = cx.texture(
            "checker",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Lifetime::Kept,
        );
        let sampler = cx.sampler(SamplerDesc::NEAREST);
        let pipeline = cx.render_pipeline(
            MainView,
            &RenderPipeline {
                vertex: flat::VS_SPRITE,
                fragment: Some(flat::FS_SPRITE),
                debug: None,
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::Write {
                    bias: wgpu::DepthBiasState::default(),
                },
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &[
                (flat::CHECKER, Bind::Texture(checker)),
                (flat::CHECKER_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;
        Ok(Self {
            checker,
            pipeline,
            written: false,
        })
    }

    pub(crate) fn prepare(&mut self, cx: &mut Prepare<'_, SpritesGpu>) {
        cx.allocate(self.checker, CHECKER_SIZE, 1);
        if self.written {
            return;
        }
        cx.write_texture(
            self.checker,
            wgpu::Origin3d::ZERO,
            TEXELS.as_flattened(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(2),
            },
            CHECKER_SIZE,
        );
        self.written = true;
    }

    pub(crate) fn draw(&self, draw: &mut Draw<'_, SpritesGpu, MainView>) -> u32 {
        draw.draw(
            &self.pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..1,
            },
        );
        1
    }
}
