use engine_core::{Error, PrimaryView, ScenePass, TargetName};
use engine_render::{
    Allocation, Bind, DepthUse, Draw, Install, Lifetime, Pipeline, Prepare, RenderPipeline,
    SamplerDesc, Scene, Shape, Texture, wgpu,
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
    pipelines: Vec<(&'static str, Pipeline<SpritesGpu, Scene>)>,
    drawn: bool,
}

impl Flat {
    pub(crate) fn install(
        cx: &mut Install<'_, SpritesGpu>,
        listed: &[ScenePass],
    ) -> Result<Self, Error> {
        let checker = cx.texture(
            "checker",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            Lifetime::Kept,
        );
        let sampler = cx.sampler(SamplerDesc::NEAREST);
        let mut pipelines = Vec::with_capacity(listed.len());
        for listed in listed {
            let lit = listed.pass.writes.contains(TargetName::COLOR);
            let pipeline = cx.render_pipeline(
                Scene::named(listed.name),
                &RenderPipeline {
                    vertex: flat::VS_SPRITE,
                    fragment: lit.then_some(flat::FS_SPRITE),
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
            pipelines.push((listed.name, pipeline));
        }
        Ok(Self {
            checker,
            pipelines,
            drawn: false,
        })
    }

    pub(crate) fn prepare(&mut self, view: &PrimaryView, cx: &mut Prepare<'_, SpritesGpu>) {
        self.drawn = view.jitter().is_none();
        if cx.allocate(self.checker, CHECKER_SIZE, 1) == Allocation::Kept {
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
    }

    pub(crate) fn draw(&self, name: &str, draw: &mut Draw<'_, SpritesGpu, Scene>) -> u32 {
        if !self.drawn {
            return 0;
        }
        let Some((_, pipeline)) = self.pipelines.iter().find(|(pass, _)| *pass == name) else {
            return 0;
        };
        draw.draw(
            pipeline,
            Shape::Pulled {
                vertices: 0..6,
                instances: 0..1,
            },
        );
        1
    }
}
