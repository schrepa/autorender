use bevy_ecs::prelude::*;
use engine_core::{DeviceLimits, Error};
use engine_places::probes::{ProbeView, ProbeViews};
use engine_render::{
    Access, Attach, Bind, Buffer, DEPTH_FORMAT, DepthUse, Draws, Encode, GpuPart, Install,
    Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, SamplerDesc, Shape, Texture, wgpu,
};

use crate::probes::{FACES, ProbeState, ProbeWords, SETS};
use crate::shaders;

const PROBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MOST_LEVELS: u32 = 11;
const GRID_BYTES: u64 = 32;
const SIDES: u32 = FACES as u32;
const FACES_PER_PASS: u32 = 3;

pub(crate) struct ProbesGpu {
    captures: Pass<Self>,
    filter: Pass<Self>,
    radiance: Texture<Self>,
    maps: Texture<Self>,
    depth: Texture<Self>,
    words: Buffer<Self>,
    down: Vec<Pipeline<Self, Own>>,
    prefilter: Vec<Pipeline<Self, Own>>,
    views: Draws<ProbeViews>,
    drawn: Vec<(ProbeView, u32)>,
    finished: Option<(u32, u32)>,
    levels: u32,
    placed: Option<(u32, u32)>,
    grid: u64,
}

impl GpuPart for ProbesGpu {
    type Reads = (
        Res<'static, ProbeState>,
        Res<'static, ProbeWords>,
        Res<'static, DeviceLimits>,
    );

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
            .color_cleared(radiance, wgpu::Color::BLACK)
            .depth(depth, Access::Clear)
            .id();
        let filter_pass = cx
            .pass("probe_filter")
            .after(captures)
            .color_cleared(radiance, wgpu::Color::BLACK)
            .color_cleared(maps, wgpu::Color::BLACK)
            .id();
        let faces = |fragment| RenderPipeline {
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
        let mut prefilter = Vec::new();
        for set in 0..SETS {
            let layers = set * SIDES..(set + 1) * SIDES;
            for level in 1..MOST_LEVELS {
                down.push(cx.render_pipeline(
                    filter_pass,
                    &faces(filter::FS_DOWN),
                    &[
                        (
                            filter::LEVEL_SOURCE,
                            Bind::Slice {
                                texture: radiance,
                                mips: level - 1..level,
                                layers: layers.clone(),
                            },
                        ),
                        (filter::CUBE_SAMPLER, Bind::Sampler(sampler)),
                    ],
                )?);
            }
            prefilter.push(cx.render_pipeline(
                filter_pass,
                &faces(filter::FS_PREFILTER),
                &[
                    (
                        filter::RADIANCE,
                        Bind::Slice {
                            texture: radiance,
                            mips: 0..MOST_LEVELS,
                            layers,
                        },
                    ),
                    (filter::CUBE_SAMPLER, Bind::Sampler(sampler)),
                ],
            )?);
        }
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
            drawn: Vec::new(),
            finished: None,
            levels: 1,
            placed: None,
            grid: 0,
        })
    }

    fn prepare(
        &mut self,
        (state, words, limits): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let size = state.size;
        self.levels = mip_levels(size);
        let extent = |layers: u32| wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        };
        let placed = (state.size, state.cubes);
        let kept = self
            .placed
            .is_some_and(|(size, cubes)| size == placed.0 && placed.1 > cubes);
        if self.placed != Some(placed) {
            cx.release(self.radiance);
            if !kept {
                cx.release(self.maps);
            }
            self.placed = Some(placed);
        }
        cx.allocate(self.radiance, extent(SETS * SIDES), self.levels);
        let maps = extent(state.probe_cubes() * SIDES);
        if kept {
            cx.grow(self.maps, maps, self.levels);
        } else {
            cx.allocate(self.maps, maps, self.levels);
        }
        if words.is_changed() {
            let room = limits.storage_room / 16 * 16;
            if self.grid == 0 {
                self.grid = GRID_BYTES.min(room);
            }
            let bytes = std::mem::size_of_val(words.words.as_slice()) as u64;
            let grown = bytes.next_power_of_two().min(room);
            if bytes > self.grid && grown > self.grid {
                self.grid = grown;
            }
            cx.allocate_buffer(self.words, self.grid);
            cx.write(self.words, 0, &words.words);
        }
        self.drawn.clear();
        self.drawn.extend_from_slice(&state.drawn);
        self.finished = state.finished;
        if !state.listed {
            cx.release(self.depth);
        }
        if !self.drawn.is_empty() {
            cx.allocate(self.depth, extent(1), 1);
            cx.cameras(&state.cameras);
        }
        if state.timed {
            cx.run(self.captures);
        }
        if self.finished.is_some() {
            cx.run(self.filter);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.filter {
            if let Some((cube, set)) = self.finished {
                self.filter_cube((cube, set), cx);
            }
            return 0;
        }
        for (view, (at, layer)) in self.drawn.iter().enumerate() {
            let mut draw = cx.render(
                Attach::ColorDepth(self.radiance, Some(*layer), self.depth),
                Some(view as u32),
            );
            self.views.draw(&mut draw, (), at);
        }
        0
    }
}

impl ProbesGpu {
    fn filter_cube(&self, (cube, set): (u32, u32), cx: &mut Encode<'_, Self>) {
        let first_layer = set * SIDES;
        for level in 1..self.levels {
            let down = &self.down[(set * (MOST_LEVELS - 1) + level - 1) as usize];
            for first in (0..SIDES).step_by(FACES_PER_PASS as usize) {
                faces_pass(cx, (self.radiance, level, first_layer + first), down, first);
            }
        }
        let base = (cube - 1) * SIDES;
        let prefilter = &self.prefilter[set as usize];
        for level in 0..self.levels {
            for first in (0..SIDES).step_by(FACES_PER_PASS as usize) {
                let start = level * SIDES + first;
                faces_pass(cx, (self.maps, level, base + first), prefilter, start);
            }
        }
    }
}

fn faces_pass(
    cx: &mut Encode<'_, ProbesGpu>,
    (texture, mip, first): (Texture<ProbesGpu>, u32, u32),
    pipeline: &Pipeline<ProbesGpu, Own>,
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

fn mip_levels(size: u32) -> u32 {
    u32::BITS - size.max(1).leading_zeros()
}
