use bevy_ecs::prelude::Resource;
use engine_core::{Error, Readable, Readback, Settling, StoreId};
use engine_render::{
    Allocation, Attach, Bind, Copied, CopyRegion, DepthUse, Encode, Install, Lifetime, Own, Pass,
    Pipeline, Prepare, RenderPipeline, Shape, Texture, mip_levels, wgpu,
};

use crate::gpu_layouts::SkyAir;
use crate::sky::SkyGpu;

const ENVIRONMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const ENVIRONMENT_SIZE: u32 = 128;
const MOST_LEVELS: u32 = 11;

const FACES: u32 = 6;
const FACES_PER_PASS: u32 = 3;
const LAST_STEP: u32 = FACES / FACES_PER_PASS;

const LOST: u32 = 0;
const DRAWN: u32 = 1;

#[derive(Clone, Copy)]
pub(crate) struct EnvironmentInputs {
    pub(crate) light: [[f32; 4]; 5],
    pub(crate) clouds: [f32; 4],
    pub(crate) tables: Option<u64>,
    pub(crate) air: SkyAir,
}

impl EnvironmentInputs {
    fn same_beyond_clouds(&self, other: &Self) -> bool {
        self.tables == other.tables
            && bytemuck::bytes_of(&self.light) == bytemuck::bytes_of(&other.light)
            && bytemuck::bytes_of(&self.air) == bytemuck::bytes_of(&other.air)
    }

    fn same(&self, other: &Self) -> bool {
        self.same_beyond_clouds(other)
            && bytemuck::bytes_of(&self.clouds) == bytemuck::bytes_of(&other.clouds)
    }
}

#[derive(Resource, Default)]
pub(crate) struct EnvironmentSteps {
    lost: bool,
    drawn: Option<u64>,
}

impl Readable for EnvironmentSteps {
    fn arrived(&mut self, tag: u32, read: Readback<&[u8]>) {
        if tag == LOST {
            self.lost = true;
        } else if let Ok(image) = <[u8; 8]>::try_from(read.value) {
            self.drawn = Some(u64::from_le_bytes(image));
        }
    }
}

#[derive(Default)]
pub(crate) struct EnvironmentPlan {
    shown: Option<EnvironmentInputs>,
    drawing: Option<EnvironmentInputs>,
    image: u64,
    beyond: Option<StoreId>,
    settling: Settling,
}

impl EnvironmentPlan {
    pub(crate) fn prepare(&mut self, inputs: EnvironmentInputs, steps: &mut EnvironmentSteps) {
        if steps.drawn.take() == Some(self.image)
            && let Some(drawn) = self.drawing.take()
        {
            if self
                .shown
                .is_none_or(|shown| !shown.same_beyond_clouds(&drawn))
            {
                self.beyond = Some(StoreId::next());
            }
            self.shown = Some(drawn);
        }
        if std::mem::take(&mut steps.lost) {
            self.shown = None;
        }
        if self.drawing.is_none() && self.shown.is_none_or(|shown| !shown.same(&inputs)) {
            self.drawing = Some(inputs);
            self.image += 1;
        }
        self.settling.left(u32::from(self.drawing.is_some()));
    }

    pub(crate) fn beyond(&self) -> Option<StoreId> {
        self.beyond
    }
}

pub(crate) struct Environment {
    radiance: Texture<SkyGpu>,
    next: Texture<SkyGpu>,
    maps: Texture<SkyGpu>,
    pass: Pass<SkyGpu>,
    sky: Pipeline<SkyGpu, Own>,
    down: Vec<Pipeline<SkyGpu, Own>>,
    prefilter: Pipeline<SkyGpu, Own>,
    levels: u32,
    copies: Vec<CopyRegion>,
    image: u64,
    step: Option<u32>,
    asked: Option<u32>,
}

impl Environment {
    pub(crate) fn install(cx: &mut Install<'_, SkyGpu>) -> Result<Self, Error> {
        use crate::shaders::sky::{environment, environment_maps as maps_shaders};
        let radiance = cx.texture("sky_radiance", ENVIRONMENT_FORMAT, Lifetime::Kept);
        let next = cx.texture("sky_environment_next", ENVIRONMENT_FORMAT, Lifetime::Kept);
        let maps = cx.texture("sky_environment", ENVIRONMENT_FORMAT, Lifetime::Kept);
        let sampler = cx.sampler(engine_core::Sampler::LINEAR_CLAMP);
        cx.frame(environment::ENVIRONMENT_MAPS, Bind::Texture(maps))?;
        cx.frame(environment::ENVIRONMENT_SAMPLER, Bind::Sampler(sampler))?;
        let pass = cx
            .pass("sky_environment")
            .label("engine/environment")
            .color_cleared(radiance, wgpu::Color::BLACK)
            .color_cleared(next, wgpu::Color::BLACK)
            .id();
        let faces = |fragment| RenderPipeline {
            vertex: maps_shaders::VS_ENVIRONMENT,
            fragment: Some(fragment),
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
        let levels = mip_levels(ENVIRONMENT_SIZE, ENVIRONMENT_SIZE);
        let copies = (0..levels)
            .map(|level| {
                let size = (ENVIRONMENT_SIZE >> level).max(1);
                CopyRegion::Levels {
                    from: (level, wgpu::Origin3d::ZERO),
                    to: (level, wgpu::Origin3d::ZERO),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: FACES,
                    },
                }
            })
            .collect();
        Ok(Self {
            radiance,
            next,
            maps,
            pass,
            sky,
            down,
            prefilter,
            levels,
            copies,
            image: 0,
            step: None,
            asked: None,
        })
    }

    pub(crate) fn pass(&self) -> Pass<SkyGpu> {
        self.pass
    }

    pub(crate) fn prepare(&mut self, cx: &mut Prepare<'_, SkyGpu>, plan: &EnvironmentPlan) {
        let extent = wgpu::Extent3d {
            width: ENVIRONMENT_SIZE,
            height: ENVIRONMENT_SIZE,
            depth_or_array_layers: FACES,
        };
        let radiance = cx.allocate(self.radiance, extent, self.levels);
        let next = cx.allocate(self.next, extent, self.levels);
        if cx.allocate(self.maps, extent, self.levels) == Allocation::Fresh {
            cx.deliver::<EnvironmentSteps>(LOST, Vec::new());
        }
        if let Some(asked) = self.asked.take() {
            self.step = match (cx.ran(self.pass), asked == LAST_STEP) {
                (false, _) => Some(asked),
                (true, false) => Some(asked + 1),
                (true, true) => {
                    cx.deliver::<EnvironmentSteps>(DRAWN, self.image.to_le_bytes().to_vec());
                    None
                }
            };
        }
        if radiance == Allocation::Fresh || next == Allocation::Fresh {
            self.step = self.step.map(|_| 0);
        }
        if plan.image != self.image {
            self.image = plan.image;
            self.step = Some(0);
        }
        let Some(step) = self.step else {
            return;
        };
        cx.run(self.pass);
        if step == LAST_STEP {
            cx.copy_at_end(
                Copied::Texture(self.next),
                Copied::Texture(self.maps),
                &self.copies,
            );
        }
        self.asked = Some(step);
    }

    pub(crate) fn encode(&self, cx: &mut Encode<'_, SkyGpu>) -> u32 {
        match self.asked {
            None => {}
            Some(0) => {
                for first in (0..FACES).step_by(FACES_PER_PASS as usize) {
                    faces_pass(cx, (self.radiance, 0, first), &self.sky, first);
                }
                for level in 1..self.levels {
                    let down = &self.down[(level - 1) as usize];
                    for first in (0..FACES).step_by(FACES_PER_PASS as usize) {
                        faces_pass(cx, (self.radiance, level, first), down, first);
                    }
                }
            }
            Some(step) => {
                let first = (step - 1) * FACES_PER_PASS;
                for level in 0..self.levels {
                    let item = level * FACES + first;
                    faces_pass(cx, (self.next, level, first), &self.prefilter, item);
                }
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
    let mut draw = cx.render(Attach::Layers {
        texture,
        mip,
        layers: first..first + FACES_PER_PASS,
    });
    draw.draw(
        pipeline,
        Shape::Pulled {
            vertices: 0..3,
            instances: item..item + 1,
            material: None,
        },
    );
}
