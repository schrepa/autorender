use bevy_ecs::prelude::*;
use engine_core::{DeviceLimits, Error, StoreId};
use engine_render::{
    Access, Allocation, Attach, Bind, Buffer, DEPTH_FORMAT, DepthUse, Encode, GpuPart, Install,
    Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, SamplerDesc, Shape, Texture, wgpu,
};

use crate::ProbePass;
use crate::probes::{
    CAPTURED_ACK, FACES, LOST_ACK, ProbeAcks, ProbeState, ProbeWords, SETS, captured_bytes,
};
use crate::shaders;

const PROBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MOST_LEVELS: u32 = 11;
const GRID_BYTES: u64 = 32;
const SIDES: u32 = FACES as u32;
const FACES_PER_PASS: u32 = 3;

pub(crate) struct ProbesGpu {
    captures: Pass<Self>,
    filter: Pass<Self>,
    prefiltering: Pass<Self>,
    radiance: Texture<Self>,
    maps: Texture<Self>,
    depth: Texture<Self>,
    words: Buffer<Self>,
    backdrop: Pipeline<Self, Own>,
    down: Vec<Pipeline<Self, Own>>,
    prefilter: Vec<Pipeline<Self, Own>>,
    drawn: Vec<u32>,
    finished: Option<(u32, u32)>,
    taken: Option<(bool, Vec<(usize, u64)>)>,
    layout: Option<StoreId>,
    timed: bool,
    levels: u32,
    grid: u64,
}

fn faces(fragment: engine_core::Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: shaders::probes::filter::VS_FACE,
        fragment: Some(fragment),
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}

fn backdrop() -> RenderPipeline<'static> {
    use shaders::probes::backdrop;
    RenderPipeline {
        vertex: backdrop::VS_BACKDROP,
        fragment: Some(backdrop::FS_BACKDROP),
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::Compare {
            compare: wgpu::CompareFunction::GreaterEqual,
            write: false,
            bias: wgpu::DepthBiasState::default(),
        },
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
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
        let maps = cx.texture("probe_maps", PROBE_FORMAT, Lifetime::Kept);
        let depth = cx.texture("depth", DEPTH_FORMAT, Lifetime::Kept);
        let words = cx.buffer("words", Lifetime::Kept);
        let sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(sample::PROBE_WORDS, Bind::Buffer(words))?;
        cx.frame(sample::PROBE_MAPS, Bind::Texture(maps))?;
        cx.frame(sample::PROBE_SAMPLER, Bind::Sampler(sampler))?;
        let captures = cx
            .scene::<ProbePass>()
            .label("engine/probe-face")
            .color_cleared(radiance, wgpu::Color::BLACK)
            .depth(depth, Access::Clear)
            .id();
        let filter_pass = cx
            .pass("probe_filter")
            .label("engine/environment")
            .after(captures)
            .color_cleared(radiance, wgpu::Color::BLACK)
            .id();
        let prefiltering = cx
            .pass("probe_prefilter")
            .label("engine/environment")
            .after(filter_pass)
            .color_cleared(maps, wgpu::Color::BLACK)
            .id();
        cx.group(captures, "probe_captures");
        cx.group(filter_pass, "probe_captures");
        cx.group(prefiltering, "probe_captures");
        let backdrop = cx.render_pipeline(captures, &backdrop(), &[])?;
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
                prefiltering,
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
        Ok(Self {
            captures,
            filter: filter_pass,
            prefiltering,
            radiance,
            maps,
            depth,
            words,
            backdrop,
            down,
            prefilter,
            drawn: Vec::new(),
            finished: None,
            taken: None,
            layout: None,
            timed: false,
            levels: 1,
            grid: 0,
        })
    }

    fn prepare(
        &mut self,
        (state, words, limits): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if let Some((finished, slots)) = self.taken.take() {
            let filtered = cx.ran(self.filter) && cx.ran(self.prefiltering);
            let ran = cx.ran(self.captures) && (!finished || filtered);
            cx.deliver::<ProbeAcks>(CAPTURED_ACK, captured_bytes(ran, &slots));
        }
        self.allocate(&state, cx);
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
        self.drawn.clone_from(&state.drawn);
        self.finished = state
            .finished
            .filter(|&(cube, _)| cube >= 1 && cube < state.cubes);
        self.timed = state.timed;
        if !state.listed {
            cx.release(self.depth);
        }
        if !self.drawn.is_empty() {
            cx.allocate(self.depth, extent(state.size, 1), 1);
            cx.cameras(&state.cameras);
        }
        if state.captures {
            cx.run(self.captures);
        }
        if self.finished.is_some() {
            cx.run(self.filter);
            cx.run(self.prefiltering);
        }
        if state.captures && !self.drawn.is_empty() {
            self.taken = Some((self.finished.is_some(), state.taken.clone()));
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.filter {
            if let Some((_, set)) = self.finished {
                self.down_cube(set, cx);
            }
            return 0;
        }
        if pass == self.prefiltering {
            if let Some((cube, set)) = self.finished {
                self.prefilter_cube((cube, set), cx);
            }
            return 0;
        }
        if self.timed {
            cx.span("probe_captures");
        }
        let mut drawn = 0;
        for (view, &layer) in self.drawn.iter().enumerate() {
            let view = view as u32;
            let mut draw = cx.render(
                Attach::ColorDepth(self.radiance, Some(layer), self.depth),
                Some(view),
            );
            draw.draw(
                &self.backdrop,
                Shape::Pulled {
                    vertices: 0..3,
                    instances: 0..1,
                },
            );
            drawn += draw.scene(self, Some(view));
        }
        drawn
    }
}

fn extent(size: u32, layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: layers,
    }
}

impl ProbesGpu {
    fn allocate(&mut self, state: &ProbeState, cx: &mut Prepare<'_, Self>) {
        let size = state.size;
        self.levels = mip_levels(size);
        cx.allocate(self.radiance, extent(size, SETS * SIDES), self.levels);
        let layers = state.cubes.saturating_sub(1).max(1) * SIDES;
        let kept = self.layout.replace(state.layout) == Some(state.layout);
        let maps = if kept {
            cx.grow(self.maps, extent(size, layers), self.levels)
        } else {
            cx.allocate(self.maps, extent(size, layers), self.levels)
        };
        if maps == Allocation::Fresh {
            cx.deliver::<ProbeAcks>(LOST_ACK, Vec::new());
        }
    }

    fn down_cube(&self, set: u32, cx: &mut Encode<'_, Self>) {
        let first_layer = set * SIDES;
        for level in 1..self.levels {
            let down = &self.down[(set * (MOST_LEVELS - 1) + level - 1) as usize];
            for first in (0..SIDES).step_by(FACES_PER_PASS as usize) {
                faces_pass(cx, (self.radiance, level, first_layer + first), down, first);
            }
        }
    }

    fn prefilter_cube(&self, (cube, set): (u32, u32), cx: &mut Encode<'_, Self>) {
        let prefilter = &self.prefilter[set as usize];
        let base = (cube - 1) * SIDES;
        for level in 0..self.levels {
            for first in (0..SIDES).step_by(FACES_PER_PASS as usize) {
                faces_pass(
                    cx,
                    (self.maps, level, base + first),
                    prefilter,
                    level * SIDES + first,
                );
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
