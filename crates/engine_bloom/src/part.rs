use bevy_ecs::prelude::*;
use engine_core::{Entry, Error, View};
use engine_render::{
    Access, Attach, Bind, DebugShown, DepthUse, Encode, GpuPart, Install, Lifetime, Own, Pass,
    Pipeline, Prepare, RenderPipeline, SamplerDesc, Shape, Texture, wgpu,
};

use crate::shaders::bloom::{apply, chain};

const BLOOM_LEVELS: u32 = 6;

const ADD: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent::REPLACE,
};

pub(crate) struct BloomGpu {
    pass: Pass<Self>,
    levels: Texture<Self>,
    first: Pipeline<Self, Own>,
    down: Vec<Pipeline<Self, Own>>,
    up: Vec<Pipeline<Self, Own>>,
    count: u32,
    region: Region,
}

impl GpuPart for BloomGpu {
    type Reads = (Res<'static, View>, Res<'static, DebugShown>);

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let format = if cx.features().contains(wgpu::Features::RG11B10UFLOAT_RENDERABLE) {
            wgpu::TextureFormat::Rg11b10Ufloat
        } else {
            wgpu::TextureFormat::Rgba16Float
        };
        let levels = cx.texture("bloom", format, Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(apply::BLOOM, mip(levels, 0))?;
        cx.frame(apply::BLOOM_SAMPLER, Bind::Sampler(sampler))?;
        let pass = cx.pass("bloom").color(levels, Access::Load).id();
        let first = cx.render_pipeline(pass, &fullscreen(chain::FS_BLOOM_FIRST, None), &[])?;
        let mut down = Vec::new();
        let mut up = Vec::new();
        for index in 0..BLOOM_LEVELS - 1 {
            down.push(cx.render_pipeline(
                pass,
                &fullscreen(chain::FS_BLOOM_DOWN, None),
                &[(chain::SOURCE, mip(levels, index))],
            )?);
            up.push(cx.render_pipeline(
                pass,
                &fullscreen(chain::FS_BLOOM_UP, Some(ADD)),
                &[(chain::SOURCE, mip(levels, index + 1))],
            )?);
        }
        Ok(Self {
            pass,
            levels,
            first,
            down,
            up,
            count: 0,
            region: Region::default(),
        })
    }

    fn prepare(
        &mut self,
        (view, shown): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if shown.owner.is_some() {
            return;
        }
        let display = view.display_size();
        let render = view.render_size();
        let half = ((display.x / 2).max(1), (display.y / 2).max(1));
        self.count = BLOOM_LEVELS.min(mip_levels(half.0, half.1));
        cx.allocate(
            self.levels,
            wgpu::Extent3d {
                width: half.0,
                height: half.1,
                depth_or_array_layers: 1,
            },
            self.count,
        );
        self.region = Region::new((display.x, display.y), (render.x, render.y));
        cx.run(self.pass);
    }

    fn encode(&mut self, _pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let chained = self.count.saturating_sub(1) as usize;
        self.draw(cx, 0, &self.first);
        for (index, down) in self.down.iter().take(chained).enumerate() {
            self.draw(cx, index as u32 + 1, down);
        }
        for (index, up) in self.up.iter().take(chained).enumerate().rev() {
            self.draw(cx, index as u32, up);
        }
        0
    }
}

impl BloomGpu {
    fn draw(&self, cx: &mut Encode<'_, Self>, level: u32, pipeline: &Pipeline<Self, Own>) {
        let mut draw = cx.render(Attach::ColorMip(self.levels, level, 0), None);
        draw.scissor(self.region.level(level));
        draw.draw(
            pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
            },
        );
    }
}

fn mip(levels: Texture<BloomGpu>, at: u32) -> Bind<BloomGpu> {
    Bind::Slice {
        texture: levels,
        mips: at..at + 1,
        layers: 0..1,
    }
}

fn fullscreen(fragment: Entry, blend: Option<wgpu::BlendState>) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: chain::VS_BLOOM,
        fragment: Some(fragment),
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        blend,
        writes: wgpu::ColorWrites::ALL,
    }
}

fn mip_levels(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).max(1).leading_zeros()
}

#[derive(Clone, Copy, Default)]
struct Region {
    display: (u32, u32),
    render: (u32, u32),
}

impl Region {
    fn new(display: (u32, u32), render: (u32, u32)) -> Self {
        let within = |size: u32, whole: u32| size.clamp(1, whole.max(1));
        Self {
            display,
            render: (within(render.0, display.0), within(render.1, display.1)),
        }
    }

    fn level(self, level: u32) -> [u32; 4] {
        let part = |drawn: u32, whole: u32| {
            let whole = (whole >> (level + 1)).max(1);
            ((drawn >> (level + 1)).max(1) + 1).min(whole)
        };
        [
            0,
            0,
            part(self.render.0, self.display.0),
            part(self.render.1, self.display.1),
        ]
    }
}
