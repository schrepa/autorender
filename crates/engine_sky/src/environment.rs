use engine_core::Error;
use engine_places::probes::ProbeCaptures;
use engine_places::sky::{EnvironmentFaces, EnvironmentLayers};
use engine_render::{
    Attach, Bind, DepthUse, Draws, Encode, Install, Lifetime, Own, Pass, Pipeline, Prepare,
    RenderPipeline, SamplerDesc, Shape, Texture, mip_levels, wgpu,
};

use crate::sky::SkyGpu;

const ENVIRONMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const ENVIRONMENT_SIZE: u32 = 128;
const MOST_LEVELS: u32 = 11;

const FACES: u32 = 6;
const FACES_PER_PASS: u32 = 3;

pub(crate) type EnvironmentLight = [[f32; 4]; 5];

#[derive(Default)]
pub(crate) struct EnvironmentPlan {
    drawn: Option<[[f32; 4]; 7]>,
    changes: u64,
    beyond: u64,
}

impl EnvironmentPlan {
    pub(crate) fn prepare(
        &mut self,
        light: &EnvironmentLight,
        (clouds, air): ([f32; 4], [f32; 4]),
    ) {
        let sky = sky(light, clouds, air);
        if self.drawn == Some(sky) {
            return;
        }
        let beyond_clouds = self.drawn.is_none_or(|drawn| {
            drawn
                .iter()
                .zip(&sky)
                .enumerate()
                .any(|(index, (old, new))| index != 5 && old != new)
        });
        self.drawn = Some(sky);
        self.changes += 1;
        if beyond_clouds {
            self.beyond += 1;
        }
    }

    pub(crate) fn beyond(&self) -> u64 {
        self.beyond
    }
}

pub(crate) struct Environment {
    radiance: Texture<SkyGpu>,
    maps: Texture<SkyGpu>,
    pass: Pass<SkyGpu>,
    probe_pass: Pass<SkyGpu>,
    sky: Pipeline<SkyGpu, Own>,
    down: Vec<Pipeline<SkyGpu, Own>>,
    prefilter: Pipeline<SkyGpu, Own>,
    layers: Draws<EnvironmentLayers>,
    levels: u32,
    placed: Option<(u32, u32)>,
    drawn: Option<u64>,
    runs: bool,
    finished: Option<u32>,
}

impl Environment {
    pub(crate) fn install(cx: &mut Install<'_, SkyGpu>) -> Result<Self, Error> {
        use crate::shaders::sky::{environment, environment_maps as maps_shaders};
        let radiance = cx.texture("sky_radiance", ENVIRONMENT_FORMAT, Lifetime::Kept);
        let maps = cx.texture("sky_environment", ENVIRONMENT_FORMAT, Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(environment::ENVIRONMENT_MAPS, Bind::Texture(maps))?;
        cx.frame(environment::ENVIRONMENT_SAMPLER, Bind::Sampler(sampler))?;
        let pass = cx
            .pass("sky_environment")
            .label("engine/environment")
            .color_cleared(radiance, wgpu::Color::BLACK)
            .color_cleared(maps, wgpu::Color::BLACK)
            .id();
        let probe_pass = cx
            .pass("probe_environment")
            .label("engine/environment")
            .after(pass)
            .color_cleared(maps, wgpu::Color::BLACK)
            .id();
        let layers = cx.claim::<EnvironmentLayers>(probe_pass)?;
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
        let down = (1..MOST_LEVELS)
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
                        mips: 0..MOST_LEVELS,
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
            probe_pass,
            sky,
            down,
            prefilter,
            layers,
            levels: 1,
            placed: None,
            drawn: None,
            runs: false,
            finished: None,
        })
    }

    pub(crate) fn pass(&self) -> Pass<SkyGpu> {
        self.pass
    }

    pub(crate) fn probe_pass(&self) -> Pass<SkyGpu> {
        self.probe_pass
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, SkyGpu>,
        plan: &EnvironmentPlan,
        captures: &ProbeCaptures,
    ) {
        let size = if captures.size == 0 {
            ENVIRONMENT_SIZE
        } else {
            captures.size
        };
        let cubes = captures.cubes.max(1);
        self.levels = mip_levels(size, size);
        let extent = |layers| wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        };
        let kept = self
            .placed
            .is_some_and(|(known, held)| known == size && cubes > held);
        if self.placed != Some((size, cubes)) {
            if !kept {
                self.drawn = None;
            }
            self.placed = Some((size, cubes));
        }
        cx.allocate(self.radiance, extent(FACES), self.levels);
        if kept {
            cx.grow(self.maps, extent(cubes * FACES), self.levels);
        } else {
            cx.allocate(self.maps, extent(cubes * FACES), self.levels);
        }
        self.runs = self.drawn != Some(plan.changes);
        if self.runs {
            self.drawn = Some(plan.changes);
            cx.run(self.pass);
        }
        self.finished = captures.finished.filter(|&cube| cube < cubes);
        if self.finished.is_some() {
            cx.run(self.probe_pass);
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

    pub(crate) fn encode_probe(&self, cx: &mut Encode<'_, SkyGpu>) -> u32 {
        let Some(cube) = self.finished else {
            return 0;
        };
        let base = cube * FACES;
        for level in 0..self.levels {
            for first in (0..FACES).step_by(FACES_PER_PASS as usize) {
                let mut draw = cx.render(
                    Attach::Layers {
                        texture: self.maps,
                        mip: level,
                        layers: base + first..base + first + FACES_PER_PASS,
                    },
                    None,
                );
                self.layers
                    .draw(&mut draw, (), &EnvironmentFaces { level, first });
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

fn sky(light: &EnvironmentLight, clouds: [f32; 4], air: [f32; 4]) -> [[f32; 4]; 7] {
    let [sun, sun_color, zenith, horizon, ground] = *light;
    [sun, sun_color, zenith, horizon, ground, clouds, air]
}
