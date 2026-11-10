use bevy_ecs::prelude::*;
use engine_core::{DeviceLimits, Error, StoreId};
use engine_render::{
    Access, Allocation, Attach, Bind, Buffer, DEPTH_FORMAT, DepthUse, Encode, GpuPart, Install,
    Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, Shape, Texture, wgpu,
};

use crate::probes::{
    CAPTURED_ACK, FACES, FaceJob, LOST_ACK, ProbeAcks, ProbeState, ProbeWords, SETS, captured_bytes,
};
use crate::{ProbeLightPass, ProbePass, SURFACES, shaders};

const PROBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MOST_LEVELS: u32 = 11;
const GRID_BYTES: u64 = 32;
const SIDES: u32 = FACES as u32;
const FACES_PER_PASS: u32 = 3;

struct Taken {
    drew: bool,
    finished: bool,
    slots: Vec<(usize, u64)>,
}

pub(crate) struct ProbesGpu {
    captures: Pass<Self>,
    light: Pass<Self>,
    filter: Pass<Self>,
    prefiltering: Pass<Self>,
    radiance: Texture<Self>,
    maps: Texture<Self>,
    surfaces: [Texture<Self>; SURFACES.len()],
    depths: Texture<Self>,
    words: Buffer<Self>,
    relight: Pipeline<Self, Own>,
    down: Vec<Pipeline<Self, Own>>,
    prefilter: Vec<Pipeline<Self, Own>>,
    jobs: Vec<FaceJob>,
    finished: Option<(u32, u32)>,
    taken: Option<Taken>,
    layout: Option<StoreId>,
    timed: bool,
    levels: u32,
    grid: u64,
}

fn texel_bytes(format: wgpu::TextureFormat) -> u64 {
    u64::from(format.block_copy_size(None).unwrap_or(0))
}

pub(crate) fn stored_texel_bytes() -> u64 {
    SURFACES
        .iter()
        .map(|target| texel_bytes(target.format))
        .sum::<u64>()
        + texel_bytes(DEPTH_FORMAT)
}

pub(crate) fn lit_texel_bytes() -> u64 {
    texel_bytes(PROBE_FORMAT)
}

fn faces(fragment: engine_core::Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: shaders::probes::filter::VS_FACE,
        fragment: Some(fragment),
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
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
        use shaders::probes::{filter, relight, sample};
        let radiance = cx.texture("radiance", PROBE_FORMAT, Lifetime::Kept);
        let maps = cx.texture("probe_maps", PROBE_FORMAT, Lifetime::Kept);
        let surfaces =
            SURFACES.map(|target| cx.texture(target.name.name(), target.format, Lifetime::Kept));
        let depths = cx.texture("probe_depths", DEPTH_FORMAT, Lifetime::Kept);
        let words = cx.buffer("words", Lifetime::Kept);
        let sampler = cx.sampler(engine_core::Sampler::LINEAR_CLAMP);
        cx.frame(sample::PROBE_WORDS, Bind::Buffer(words))?;
        cx.frame(sample::PROBE_MAPS, Bind::Texture(maps))?;
        cx.frame(sample::PROBE_SAMPLER, Bind::Sampler(sampler))?;
        let mut capturing = cx.scene::<ProbePass>().label("engine/probe-face");
        for (target, texture) in SURFACES.iter().zip(surfaces) {
            capturing = capturing.writes(target.name, texture, wgpu::Color::TRANSPARENT);
        }
        let captures = capturing.depth(depths, Access::Clear).id();
        let light = cx
            .scene::<ProbeLightPass>()
            .label("engine/probe-light")
            .after(captures)
            .color_cleared(radiance, wgpu::Color::BLACK)
            .id();
        let filter_pass = cx
            .pass("probe_filter")
            .label("engine/environment")
            .after(light)
            .color_cleared(radiance, wgpu::Color::BLACK)
            .id();
        let prefiltering = cx
            .pass("probe_prefilter")
            .label("engine/environment")
            .after(filter_pass)
            .color_cleared(maps, wgpu::Color::BLACK)
            .id();
        cx.group(captures, "probe_captures");
        cx.group(light, "probe_captures");
        cx.group(filter_pass, "probe_captures");
        cx.group(prefiltering, "probe_captures");
        let [base, surface, normal, emissive, bounce] = surfaces;
        let relight = cx.render_pipeline(
            light,
            &faces(relight::FS_RELIGHT),
            &[
                (relight::STORED_BASE, Bind::Texture(base)),
                (relight::STORED_SURFACE, Bind::Texture(surface)),
                (relight::STORED_NORMAL, Bind::Texture(normal)),
                (relight::STORED_EMISSIVE, Bind::Texture(emissive)),
                (relight::STORED_BOUNCE, Bind::Texture(bounce)),
                (relight::STORED_DEPTH, Bind::Texture(depths)),
            ],
        )?;
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
            light,
            filter: filter_pass,
            prefiltering,
            radiance,
            maps,
            surfaces,
            depths,
            words,
            relight,
            down,
            prefilter,
            jobs: Vec::new(),
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
        if let Some(taken) = self.taken.take() {
            let filtered = cx.ran(self.filter) && cx.ran(self.prefiltering);
            let ran = (!taken.drew || cx.ran(self.captures))
                && cx.ran(self.light)
                && (!taken.finished || filtered);
            cx.deliver::<ProbeAcks>(CAPTURED_ACK, captured_bytes(ran, &taken.slots));
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
            let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.words, self.grid);
            cx.write(self.words, 0, &words.words);
        }
        self.jobs.clone_from(&state.jobs);
        self.finished = state
            .finished
            .filter(|&(cube, _)| cube >= 1 && cube < state.cubes);
        self.timed = state.timed;
        let drew = self.jobs.iter().any(|job| job.draws);
        if drew {
            cx.run(self.captures);
        }
        if self.jobs.is_empty() {
            return;
        }
        cx.run(self.light);
        if self.finished.is_some() {
            cx.run(self.filter);
            cx.run(self.prefiltering);
        }
        self.taken = Some(Taken {
            drew,
            finished: self.finished.is_some(),
            slots: state.taken.clone(),
        });
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
        if pass == self.light {
            for job in &self.jobs {
                let target = Attach::Color(self.radiance, Some(job.lit));
                let Some(mut draw) = cx.view(&target, job.known.face) else {
                    continue;
                };
                draw.draw(
                    &self.relight,
                    Shape::Pulled {
                        vertices: 0..3,
                        instances: job.stored..job.stored + 1,
                        material: None,
                    },
                );
            }
            return 0;
        }
        let mut drawn = 0;
        for job in self.jobs.iter().filter(|job| job.draws) {
            let Some(mut draw) = cx.view(&Attach::Declared, job.known.face) else {
                continue;
            };
            drawn += draw.scene(self);
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

fn held(
    cx: &mut Prepare<'_, ProbesGpu>,
    (texture, levels): (Texture<ProbesGpu>, u32),
    size: wgpu::Extent3d,
    kept: bool,
) -> Allocation {
    if kept {
        cx.grow(texture, size, levels)
    } else {
        cx.allocate(texture, size, levels)
    }
}

impl ProbesGpu {
    fn allocate(&mut self, state: &ProbeState, cx: &mut Prepare<'_, Self>) {
        let size = state.size;
        self.levels = mip_levels(size);
        let radiance = cx.allocate(self.radiance, extent(size, SETS * SIDES), self.levels);
        let layers = extent(size, state.cubes.saturating_sub(1).max(1) * SIDES);
        let kept = self.layout.replace(state.layout) == Some(state.layout);
        let maps = held(cx, (self.maps, self.levels), layers, kept);
        let stored: Vec<Allocation> = self
            .surfaces
            .iter()
            .copied()
            .chain([self.depths])
            .map(|texture| {
                if state.listed {
                    held(cx, (texture, 1), layers, kept)
                } else {
                    cx.release(texture);
                    Allocation::Kept
                }
            })
            .collect();
        if [radiance, maps]
            .iter()
            .chain(&stored)
            .any(|allocation| *allocation == Allocation::Fresh)
        {
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

fn mip_levels(size: u32) -> u32 {
    u32::BITS - size.max(1).leading_zeros()
}
