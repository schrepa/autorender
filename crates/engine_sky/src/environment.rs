use engine_core::Error;
use engine_render::{
    Access, Attach, Bind, DepthUse, Encode, Install, Lifetime, Own, Pass, Pipeline, Prepare,
    RenderPipeline, SamplerDesc, Shape, Texture, wgpu,
};

use crate::sky::SkyGpu;

pub(crate) const ENVIRONMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(crate) const ENVIRONMENT_SIZE: u32 = 128;

const FACES: u32 = 6;
const FACES_PER_PASS: u32 = 3;
const CUBES: u32 = 1;

pub(crate) type SkyLight = [[f32; 4]; 5];

pub(crate) fn mip_levels(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

pub(crate) struct Environment {
    radiance: Texture<SkyGpu>,
    maps: Texture<SkyGpu>,
    pass: Pass<SkyGpu>,
    sky: Pipeline<SkyGpu, Own>,
    down: Vec<Pipeline<SkyGpu, Own>>,
    prefilter: Pipeline<SkyGpu, Own>,
    levels: u32,
    drawn: Option<[[f32; 4]; 7]>,
    runs: bool,
}

impl Environment {
    pub(crate) fn install(cx: &mut Install<'_, SkyGpu>) -> Result<Self, Error> {
        use crate::shaders::sky::{environment, environment_maps as maps_shaders};
        let levels = mip_levels(ENVIRONMENT_SIZE, ENVIRONMENT_SIZE);
        let radiance = cx.texture("sky_radiance", ENVIRONMENT_FORMAT, Lifetime::Kept);
        let maps = cx.texture("sky_environment", ENVIRONMENT_FORMAT, Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(environment::ENVIRONMENT_MAPS, Bind::Texture(maps))?;
        cx.frame(environment::ENVIRONMENT_SAMPLER, Bind::Sampler(sampler))?;
        let pass = cx
            .pass("sky_environment")
            .color(radiance, Access::Clear)
            .color(maps, Access::Clear)
            .id();
        let faces = |fragment| RenderPipeline {
            vertex: maps_shaders::VS_ENVIRONMENT,
            fragment: Some(fragment),
            debug: None,
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        };
        let sky = cx.render_pipeline(pass, &faces(maps_shaders::FS_ENVIRONMENT_SKY), &[])?;
        let down = (1..levels)
            .map(|level| {
                cx.render_pipeline(
                    pass,
                    &faces(maps_shaders::FS_ENVIRONMENT_DOWN),
                    &[
                        (
                            maps_shaders::LEVEL_SOURCE,
                            Bind::Slice {
                                texture: radiance,
                                mips: level - 1..level,
                                layers: 0..FACES,
                            },
                        ),
                        (maps_shaders::CUBE_SAMPLER, Bind::Sampler(sampler)),
                    ],
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let prefilter = cx.render_pipeline(
            pass,
            &faces(maps_shaders::FS_ENVIRONMENT_PREFILTER),
            &[
                (
                    maps_shaders::RADIANCE,
                    Bind::Slice {
                        texture: radiance,
                        mips: 0..levels,
                        layers: 0..FACES,
                    },
                ),
                (maps_shaders::CUBE_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;
        Ok(Self {
            radiance,
            maps,
            pass,
            sky,
            down,
            prefilter,
            levels,
            drawn: None,
            runs: false,
        })
    }

    pub(crate) fn pass(&self) -> Pass<SkyGpu> {
        self.pass
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, SkyGpu>,
        light: &SkyLight,
        (clouds, air): ([f32; 4], [f32; 4]),
    ) {
        let extent = |layers| wgpu::Extent3d {
            width: ENVIRONMENT_SIZE,
            height: ENVIRONMENT_SIZE,
            depth_or_array_layers: layers,
        };
        cx.allocate(self.radiance, extent(FACES), self.levels);
        cx.allocate(self.maps, extent(CUBES * FACES), self.levels);
        let sky = sky(light, clouds, air);
        self.runs = self.drawn != Some(sky);
        if self.runs {
            self.drawn = Some(sky);
            cx.run(self.pass);
        }
    }

    pub(crate) fn encode(&self, cx: &mut Encode<'_, SkyGpu>) -> u32 {
        if !self.runs {
            return 0;
        }
        for first in (0..FACES).step_by(FACES_PER_PASS as usize) {
            faces_pass(cx, (self.radiance, 0, first), &self.sky, first);
        }
        for level in 1..self.levels {
            for first in (0..FACES).step_by(FACES_PER_PASS as usize) {
                let down = &self.down[(level - 1) as usize];
                faces_pass(cx, (self.radiance, level, first), down, first);
            }
        }
        for level in 0..self.levels {
            for first in (0..FACES).step_by(FACES_PER_PASS as usize) {
                let start = level * FACES + first;
                faces_pass(cx, (self.maps, level, first), &self.prefilter, start);
            }
        }
        0
    }
}

fn faces_pass(
    cx: &mut Encode<'_, SkyGpu>,
    (texture, mip, first): (Texture<SkyGpu>, u32, u32),
    pipeline: &Pipeline<SkyGpu, Own>,
    item: u32,
) {
    let mut draw = cx.render(
        Attach::Layers {
            texture,
            mip,
            layers: first..first + FACES_PER_PASS,
        },
        None,
    );
    draw.draw(
        pipeline,
        Shape::Pulled {
            vertices: 0..3,
            instances: item..item + 1,
        },
    );
}

fn sky(light: &SkyLight, clouds: [f32; 4], air: [f32; 4]) -> [[f32; 4]; 7] {
    let [sun, sun_color, zenith, horizon, ground] = *light;
    [sun, sun_color, zenith, horizon, ground, clouds, air]
}
