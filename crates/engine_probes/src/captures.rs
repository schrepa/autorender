use bevy_ecs::prelude::*;
use engine_core::Error;
use engine_places::probes::{ProbeView, ProbeViews};
use engine_render::{
    Access, Attach, Bind, Buffer, DepthUse, Draws, Encode, GpuPart, Install, Lifetime, Own, Pass,
    Pipeline, Prepare, RenderPipeline, SamplerDesc, Shape, Texture, wgpu,
};

use crate::probes::{FACES, ProbeState, ProbeWords};
use crate::shaders;

const PROBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MOST_LEVELS: u32 = 11;
const GRID_BYTES: u64 = 32;
const SIDES: u32 = FACES as u32;

pub(crate) struct ProbesGpu {
    captures: Pass<Self>,
    filter: Pass<Self>,
    radiance: Texture<Self>,
    maps: Texture<Self>,
    depth: Texture<Self>,
    words: Buffer<Self>,
    down: Vec<Pipeline<Self, Own>>,
    prefilter: Pipeline<Self, Own>,
    views: Draws<ProbeViews>,
    at: Vec<ProbeView>,
    finished: Option<u32>,
    levels: u32,
}

impl GpuPart for ProbesGpu {
    type Reads = (Res<'static, ProbeState>, Res<'static, ProbeWords>);

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::probes::{filter, sample};
        let radiance = cx.texture("radiance", PROBE_FORMAT, Lifetime::Kept);
        let maps = cx.texture("maps", PROBE_FORMAT, Lifetime::Previous);
        let depth = cx.texture("depth", DEPTH_FORMAT, Lifetime::Kept);
        let words = cx.buffer("words", Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(sample::PROBE_WORDS, Bind::Buffer(words))?;
        cx.frame(sample::PROBE_MAPS, Bind::Texture(maps))?;
        cx.frame(sample::PROBE_SAMPLER, Bind::Sampler(sampler))?;
        let captures = cx
            .pass("probe_captures")
            .color(radiance, Access::Clear)
            .depth(depth, Access::Clear)
            .id();
        let filter_pass = cx
            .pass("probe_filter")
            .after(captures)
            .color(radiance, Access::Clear)
            .color(maps, Access::Clear)
            .id();
        let face_pipeline = |fragment| RenderPipeline {
            vertex: filter::VS_FACE,
            fragment: Some(fragment),
            debug: None,
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::None,
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        };
        let mut down = Vec::new();
        for level in 0..MOST_LEVELS - 1 {
            down.push(cx.render_pipeline(
                filter_pass,
                &face_pipeline(filter::FS_DOWN),
                &[
                    (
                        filter::LEVEL_SOURCE,
                        Bind::Slice {
                            texture: radiance,
                            mips: level..level + 1,
                            layers: 0..SIDES,
                        },
                    ),
                    (filter::CUBE_SAMPLER, Bind::Sampler(sampler)),
                ],
            )?);
        }
        let prefilter = cx.render_pipeline(
            filter_pass,
            &face_pipeline(filter::FS_PREFILTER),
            &[
                (filter::RADIANCE, Bind::Texture(radiance)),
                (filter::CUBE_SAMPLER, Bind::Sampler(sampler)),
            ],
        )?;
        let views = cx.claim::<ProbeViews>(captures)?;
        Ok(Self {
            captures,
            filter: filter_pass,
            radiance,
            maps,
            depth,
            words,
            down,
            prefilter,
            views,
            at: Vec::new(),
            finished: None,
            levels: 1,
        })
    }

    fn prepare(
        &mut self,
        (state, words): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let size = state.size;
        self.levels = mip_levels(size);
        let extent = |layers: u32| wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        };
        cx.allocate(self.radiance, extent(SIDES), self.levels);
        cx.allocate(self.maps, extent(state.probe_cubes() * SIDES), self.levels);
        if words.is_changed() {
            cx.allocate_buffer(
                self.words,
                (std::mem::size_of_val(words.words.as_slice()) as u64).max(GRID_BYTES),
            );
            cx.write(self.words, 0, &words.words);
        }
        self.at.clear();
        self.at.extend_from_slice(&state.views);
        self.finished = state.finished;
        if !self.at.is_empty() {
            cx.allocate(self.depth, extent(1), 1);
            cx.cameras(&state.cameras);
            cx.run(self.captures);
        }
        if self.finished.is_some() {
            cx.run(self.filter);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.filter {
            if let Some(cube) = self.finished {
                self.filter_cube(cube, cx);
            }
            return 0;
        }
        for (view, at) in self.at.iter().enumerate() {
            let mut draw = cx.render(
                Attach::ColorDepth(self.radiance, Some(at.face), self.depth),
                Some(view as u32),
            );
            self.views.draw(&mut draw, (), at);
        }
        0
    }
}

impl ProbesGpu {
    fn filter_cube(&self, cube: u32, cx: &mut Encode<'_, Self>) {
        for level in 1..self.levels {
            for face in 0..SIDES {
                let mut draw = cx.render(Attach::ColorMip(self.radiance, level, face), None);
                draw.draw(
                    &self.down[(level - 1) as usize],
                    Shape::Pulled {
                        vertices: 0..3,
                        instances: face..face + 1,
                    },
                );
            }
        }
        let first = (cube - 1) * SIDES;
        for level in 0..self.levels {
            for face in 0..SIDES {
                let item = level * SIDES + face;
                let mut draw = cx.render(Attach::ColorMip(self.maps, level, first + face), None);
                draw.draw(
                    &self.prefilter,
                    Shape::Pulled {
                        vertices: 0..3,
                        instances: item..item + 1,
                    },
                );
            }
        }
    }
}

fn mip_levels(size: u32) -> u32 {
    u32::BITS - size.max(1).leading_zeros()
}
