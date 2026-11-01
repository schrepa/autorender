use bevy_ecs::prelude::*;
use engine_core::{DeviceLimits, Entry, Error, View};
use engine_places::ground::{Cover, GroundTiles, PlacedTiles, TerrainTiles};
use engine_render::{
    Access, Attach, Bind, Buffer, ComputePipeline, Copied, CopyRegion, Count, DepthUse, Encode,
    FULLSCREEN, GpuPart, Install, Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline,
    SCENE_FORMAT, SamplerDesc, Shape, Texture, ViewTarget, level_region, mip_levels, wgpu,
};

use crate::gpu::{PYRAMID_GROUP, PyramidExtent};
use crate::shaders::reflections::{apply, levels, nearest, trace};
use crate::world_trace::{TraceFrame, WorldTrace};

const SCENE_LEVELS: u32 = 6;
const PYRAMID_LEVELS: u32 = 15;
const SPAN: &str = "reflections";
const REFLECTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const NEAREST_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

pub(crate) struct TraceGpu {
    reduce: Pass<Self>,
    copy: Pass<Self>,
    chain: Pass<Self>,
    reflect: Pass<Self>,
    reflected: Texture<Self>,
    levels: Texture<Self>,
    pyramid: Texture<Self>,
    terrains: Buffer<Self>,
    placed: Buffer<Self>,
    extents: Vec<Buffer<Self>>,
    reductions: Vec<ComputePipeline<Self>>,
    downs: Vec<Pipeline<Self, Own>>,
    tracing: Pipeline<Self, Own>,
    world: WorldTrace,
    changed: (bool, bool),
    count: u32,
    render: (u32, u32),
    drawn: [(u32, u32); SCENE_LEVELS as usize],
    reduced: Vec<PyramidExtent>,
}

impl GpuPart for TraceGpu {
    type Reads = (
        Res<'static, View>,
        Res<'static, DeviceLimits>,
        Res<'static, TerrainTiles>,
        Res<'static, GroundTiles>,
        Res<'static, PlacedTiles>,
        Res<'static, Cover>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let reflected = cx.texture("reflections", REFLECTION_FORMAT, Lifetime::Frame);
        let levels_texture = cx.texture("scene_levels", SCENE_FORMAT, Lifetime::Frame);
        let pyramid = cx.texture("nearest_depth", NEAREST_FORMAT, Lifetime::Frame);
        let terrains = cx.buffer("world_terrains", Lifetime::Kept);
        let placed = cx.buffer("world_placed", Lifetime::Kept);
        let extents: Vec<Buffer<Self>> = (0..PYRAMID_LEVELS)
            .map(|_| cx.buffer("nearest_extent", Lifetime::Kept))
            .collect();
        let sampler = cx.sampler(SamplerDesc::LINEAR_CLAMP);
        cx.frame(apply::REFLECTIONS, Bind::Texture(reflected))?;
        cx.frame(apply::REFLECTIONS_SAMPLER, Bind::Sampler(sampler))?;
        let reduce = cx.pass("nearest_depth").span(SPAN).id();
        let copy = cx.copy(
            "scene_levels_copy",
            Copied::Target(ViewTarget::Scene),
            Copied::Texture(levels_texture),
        );
        let chain = cx
            .pass("scene_levels")
            .span(SPAN)
            .after(copy)
            .color(levels_texture, Access::Load)
            .id();
        let reflect = cx
            .pass("reflections")
            .span(SPAN)
            .color(reflected, Access::Clear)
            .id();
        let mut reductions = Vec::new();
        for (level, uniform) in (0..PYRAMID_LEVELS).zip(&extents) {
            let reduction = if level == 0 {
                cx.compute_pipeline(
                    reduce,
                    nearest::CS_NEAREST_FIRST,
                    &[
                        (nearest::SOURCE_DEPTH, Bind::Target(ViewTarget::Depth)),
                        (nearest::TARGET_LEVEL, mip(pyramid, 0)),
                        (nearest::EXTENT, Bind::Buffer(*uniform)),
                    ],
                )?
            } else {
                cx.compute_pipeline(
                    reduce,
                    nearest::CS_NEAREST_DOWN,
                    &[
                        (nearest::SOURCE_LEVEL, mip(pyramid, level - 1)),
                        (nearest::TARGET_LEVEL, mip(pyramid, level)),
                        (nearest::EXTENT, Bind::Buffer(*uniform)),
                    ],
                )?
            };
            reductions.push(reduction);
        }
        let mut downs = Vec::new();
        for level in 1..=SCENE_LEVELS {
            let fragment = if level == 1 {
                levels::FS_LEVELS_FIRST
            } else {
                levels::FS_LEVELS_DOWN
            };
            downs.push(cx.render_pipeline(
                chain,
                &screen(fragment),
                &[(levels::SOURCE, mip(levels_texture, level - 1))],
            )?);
        }
        let tracing = cx.render_pipeline(
            reflect,
            &screen(trace::FS_REFLECTIONS),
            &[
                (trace::NEAREST_DEPTH, Bind::Texture(pyramid)),
                (trace::SCENE_LEVELS, Bind::Texture(levels_texture)),
                (trace::WORLD_TERRAINS, Bind::Buffer(terrains)),
                (trace::WORLD_PLACED, Bind::Buffer(placed)),
            ],
        )?;
        Ok(Self {
            reduce,
            copy,
            chain,
            reflect,
            reflected,
            levels: levels_texture,
            pyramid,
            terrains,
            placed,
            extents,
            reductions,
            downs,
            tracing,
            world: WorldTrace::default(),
            changed: (false, false),
            count: 0,
            render: (0, 0),
            drawn: [(0, 0); SCENE_LEVELS as usize],
            reduced: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (view, limits, tiles, ground, placed, cover): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if cx.fresh() {
            self.world.forget();
            self.reduced.clear();
        }
        self.changed.0 |= ground.is_changed();
        self.changed.1 |= placed.is_changed();
        if view.debug() {
            return;
        }
        let (ground_changed, placed_changed) = std::mem::take(&mut self.changed);
        self.world.prepare(
            cx,
            (self.terrains, self.placed),
            &TraceFrame {
                tiles: &tiles,
                ground: &ground,
                placed: &placed,
                cover: &cover,
                room: limits.storage_room / 16 * 16,
                ground_changed,
                placed_changed,
            },
        );
        let display = view.display_size();
        let render = view.render_size();
        self.render = (render.x, render.y);
        self.count = SCENE_LEVELS.min(mip_levels(display.x, display.y) - 1);
        for (level, drawn) in (0..).zip(&mut self.drawn) {
            *drawn = level_region(&view, level);
        }
        let whole = extent(display.x, display.y);
        cx.allocate(self.reflected, whole, 1);
        cx.allocate(self.levels, whole, 1 + self.count);
        let base = (
            display.x.div_ceil(2).max(1).next_power_of_two(),
            display.y.div_ceil(2).max(1).next_power_of_two(),
        );
        let reductions = mip_levels(base.0, base.1).min(PYRAMID_LEVELS);
        cx.allocate(self.pyramid, extent(base.0, base.1), reductions);
        let reduced = extents(self.render, reductions);
        for buffer in &self.extents {
            cx.allocate_buffer(*buffer, size_of::<PyramidExtent>() as u64);
        }
        if reduced != self.reduced {
            for (buffer, written) in self.extents.iter().zip(&reduced) {
                cx.write(*buffer, 0, &[*written]);
            }
            self.reduced = reduced;
        }
        cx.run(self.reduce);
        cx.copy(
            self.copy,
            &[CopyRegion::Texels {
                from: wgpu::Origin3d::ZERO,
                to: wgpu::Origin3d::ZERO,
                size: whole,
            }],
        );
        if self.count > 0 {
            cx.run(self.chain);
        }
        cx.run(self.reflect);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.reduce {
            for (reduction, written) in self.reductions.iter().zip(&self.reduced) {
                let [width, height] = written.size;
                let groups = [width.div_ceil(PYRAMID_GROUP), height.div_ceil(PYRAMID_GROUP), 1];
                cx.dispatch(reduction, Count::Groups(groups));
            }
            return 0;
        }
        if pass == self.chain {
            for ((level, down), drawn) in (0..self.count).zip(&self.downs).zip(self.drawn) {
                let mut draw = cx.render(Attach::ColorMip(self.levels, level + 1, 0), None);
                draw.within(drawn);
                draw.draw(down, TRIANGLE);
            }
            return 0;
        }
        if pass == self.reflect {
            let mut draw = cx.render(Attach::Declared, None);
            draw.within(self.render);
            draw.draw(&self.tracing, TRIANGLE);
        }
        0
    }
}

const TRIANGLE: Shape<'static, TraceGpu, Own> = Shape::Pulled {
    vertices: 0..3,
    instances: 0..1,
};

fn screen(fragment: Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: FULLSCREEN,
        fragment: Some(fragment),
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}

fn mip(texture: Texture<TraceGpu>, level: u32) -> Bind<TraceGpu> {
    Bind::Slice {
        texture,
        mips: level..level + 1,
        layers: 0..1,
    }
}

fn extent(width: u32, height: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    }
}

fn extents((width, height): (u32, u32), count: u32) -> Vec<PyramidExtent> {
    let mut source = (width.max(1), height.max(1));
    (0..count)
        .map(|_| {
            let written = (source.0.div_ceil(2), source.1.div_ceil(2));
            let extent = PyramidExtent {
                size: [written.0, written.1],
                last: [source.0 - 1, source.1 - 1],
            };
            source = written;
            extent
        })
        .collect()
}
