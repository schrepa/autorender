use bevy_ecs::prelude::*;
use engine_core::{Entry, Error, PassName, ViewFacts};
use engine_render::{
    Access, Allocation, Attach, Bind, Buffer, ComputePipeline, Count, DepthUse, Encode, FULLSCREEN,
    GpuPart, Install, Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, SCENE_FORMAT, Shape,
    TargetName, Texture, level_region, mip_levels, pyramid_extents, wgpu,
};

use crate::gpu::{PYRAMID_GROUP, PyramidExtent};
use crate::shaders::reflections::{blend, far, levels, nearest, trace};
use crate::{Reflections, ReflectionsPass};

const SCENE_LEVELS: u32 = 6;
const PYRAMID_LEVELS: u32 = 15;
const FAR_LEVELS: u32 = 6;
const SPAN: &str = "reflections";
const REFLECTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const NEAREST_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const CELL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;

pub(crate) struct TraceGpu {
    reduce: Pass<Self>,
    chain: Pass<Self>,
    survey: Pass<Self>,
    spread: Pass<Self>,
    own: Pass<Self>,
    reflect: Pass<Self>,
    blend: Pass<Self>,
    reflected: Texture<Self>,
    levels: Texture<Self>,
    pyramid: Texture<Self>,
    cells: Texture<Self>,
    far_light: Texture<Self>,
    far_reach: Texture<Self>,
    traced: Texture<Self>,
    extents: Vec<Buffer<Self>>,
    reductions: Vec<ComputePipeline<Self>>,
    first: Pipeline<Self, Own>,
    downs: Vec<Pipeline<Self, Own>>,
    surveying: Pipeline<Self, Own>,
    spreading: Vec<ComputePipeline<Self>>,
    owning: Pipeline<Self, Own>,
    tracing: Pipeline<Self, Own>,
    blending: Pipeline<Self, Own>,
    count: u32,
    cell: u32,
    render: (u32, u32),
    surveyed: (u32, u32),
    drawn: [(u32, u32); SCENE_LEVELS as usize],
    reduced: Vec<PyramidExtent>,
    spreads: Vec<[u32; 3]>,
}

impl GpuPart for TraceGpu {
    type Reads = Res<'static, Reflections>;
    const NAME: &'static str = "reflections";
    const LAZY: bool = true;
    const LABELS: &'static [&'static str] = &["engine/reflections-"];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let reflected = cx.view_texture("reflections", REFLECTION_FORMAT, Lifetime::Frame);
        let levels_texture = cx.view_texture("scene_levels", SCENE_FORMAT, Lifetime::Frame);
        let pyramid = cx.view_texture("nearest_depth", NEAREST_FORMAT, Lifetime::Frame);
        let cells = cx.view_texture("far_cells", CELL_FORMAT, Lifetime::Frame);
        let far_light = cx.view_texture("far_light", REFLECTION_FORMAT, Lifetime::Frame);
        let far_reach = cx.view_texture("far_reach", REFLECTION_FORMAT, Lifetime::Frame);
        let traced = cx.view_texture("far_traced", REFLECTION_FORMAT, Lifetime::Frame);
        let extents: Vec<Buffer<Self>> = (0..PYRAMID_LEVELS)
            .map(|_| cx.view_buffer("nearest_extent", Lifetime::Kept))
            .collect();
        let (at, views) = (ReflectionsPass::PASS.at, ReflectionsPass::PASS.views);
        let reduce = cx
            .pass("nearest_depth")
            .at(at)
            .views(views)
            .label("engine/depth-pyramid")
            .span(SPAN)
            .id();
        let chain = cx
            .pass("scene_levels")
            .at(at)
            .views(views)
            .label("engine/post-scene-levels")
            .span(SPAN)
            .color(levels_texture, Access::Clear)
            .id();
        let survey = cx
            .pass("reflections_far")
            .at(at)
            .views(views)
            .label("engine/post-reflections-far")
            .span(SPAN)
            .after(reduce)
            .color(cells, Access::Clear)
            .id();
        let spread = cx
            .pass("reflections_far_levels")
            .at(at)
            .views(views)
            .label("engine/post-reflections-far-levels")
            .span(SPAN)
            .after(survey)
            .id();
        let own = cx
            .pass("reflections_far_own")
            .at(at)
            .views(views)
            .label("engine/post-reflections-far-own")
            .span(SPAN)
            .after(survey)
            .color(traced, Access::Clear)
            .id();
        let reflect = cx
            .screen::<ReflectionsPass>()
            .label("engine/post-reflections")
            .span(SPAN)
            .after(chain)
            .after(spread)
            .after(own)
            .color(reflected, Access::Clear)
            .id();
        let blend = cx
            .pass("reflections_blend")
            .at(at)
            .views(views)
            .label("engine/post-reflections-blend")
            .span(SPAN)
            .after(reflect)
            .target(TargetName::COLOR)
            .id();
        let mut reductions = Vec::new();
        for (level, uniform) in (0..PYRAMID_LEVELS).zip(&extents) {
            let reduction = if level == 0 {
                cx.compute_pipeline(
                    reduce,
                    nearest::CS_NEAREST_FIRST,
                    &[
                        (nearest::SOURCE_DEPTH, Bind::Target(TargetName::DEPTH)),
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
        let first = cx.render_pipeline(chain, &screen(levels::FS_LEVELS_SCENE), &[])?;
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
        let surveying = cx.render_pipeline(
            survey,
            &screen(trace::FS_FAR),
            &[
                (trace::NEAREST_DEPTH, Bind::Texture(pyramid)),
                (trace::SURFACE_TARGET, Bind::Target(crate::SURFACE.name)),
                (trace::REFLECTED_TARGET, Bind::Target(crate::REFLECTED.name)),
            ],
        )?;
        let owning = cx.render_pipeline(
            own,
            &screen(trace::FS_FAR_OWN),
            &[
                (trace::SURFACE_TARGET, Bind::Target(crate::SURFACE.name)),
                (trace::REFLECTED_TARGET, Bind::Target(crate::REFLECTED.name)),
                (trace::FAR_CELLS, Bind::Texture(cells)),
            ],
        )?;
        let mut spreading = Vec::new();
        for level in 0..=FAR_LEVELS {
            let widened = if level == 0 {
                cx.compute_pipeline(
                    spread,
                    far::CS_FAR_FIRST,
                    &[
                        (far::FAR_CELLS, Bind::Texture(cells)),
                        (far::LIGHT_TARGET, mip(far_light, 0)),
                        (far::REACH_TARGET, mip(far_reach, 0)),
                    ],
                )?
            } else {
                cx.compute_pipeline(
                    spread,
                    far::CS_FAR_DOWN,
                    &[
                        (far::LIGHT_LEVEL, mip(far_light, level - 1)),
                        (far::REACH_LEVEL, mip(far_reach, level - 1)),
                        (far::LIGHT_TARGET, mip(far_light, level)),
                        (far::REACH_TARGET, mip(far_reach, level)),
                    ],
                )?
            };
            spreading.push(widened);
        }
        let tracing = cx.render_pipeline(
            reflect,
            &screen(trace::FS_REFLECTIONS),
            &[
                (trace::NEAREST_DEPTH, Bind::Texture(pyramid)),
                (trace::SCENE_LEVELS, Bind::Texture(levels_texture)),
                (trace::SURFACE_TARGET, Bind::Target(crate::SURFACE.name)),
                (trace::REFLECTED_TARGET, Bind::Target(crate::REFLECTED.name)),
                (trace::FAR_LIGHT, Bind::Texture(far_light)),
                (trace::FAR_REACH, Bind::Texture(far_reach)),
                (trace::FAR_TRACED, Bind::Texture(traced)),
            ],
        )?;
        let blending = cx.render_pipeline(
            blend,
            &RenderPipeline {
                blend: Some(ADDED_COLOR),
                ..screen(blend::FS_REFLECTIONS_BLEND)
            },
            &[
                (blend::REFLECTIONS, Bind::Texture(reflected)),
                (blend::REFLECTED_TARGET, Bind::Target(crate::REFLECTED.name)),
                (blend::BLENDED_MASK, Bind::Target(TargetName::TRANSPARENCY)),
            ],
        )?;
        Ok(Self {
            reduce,
            chain,
            survey,
            spread,
            own,
            reflect,
            blend,
            reflected,
            levels: levels_texture,
            pyramid,
            cells,
            far_light,
            far_reach,
            traced,
            extents,
            reductions,
            first,
            downs,
            surveying,
            spreading,
            owning,
            tracing,
            blending,
            count: 0,
            cell: 1,
            render: (0, 0),
            surveyed: (0, 0),
            drawn: [(0, 0); SCENE_LEVELS as usize],
            reduced: Vec::new(),
            spreads: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        reflections: SystemParamItem<'_, '_, Self::Reads>,
        _cx: &mut Prepare<'_, Self>,
    ) {
        self.cell = reflections.far_cell;
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        let display = view.size();
        let render = view.render;
        self.render = (render.x, render.y);
        self.surveyed = (render.x.div_ceil(self.cell), render.y.div_ceil(self.cell));
        self.count = SCENE_LEVELS.min(mip_levels(display.x, display.y) - 1);
        for (level, drawn) in (0..).zip(&mut self.drawn) {
            *drawn = level_region(view, level);
        }
        let whole = extent(display.x, display.y);
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(self.reflected, whole, 1);
        cx.run(self.reflect);
        let coarse = (display.x.div_ceil(self.cell), display.y.div_ceil(self.cell));
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate(self.cells, extent(coarse.0, coarse.1), 1);
        cx.run(self.survey);
        let halved = 1 << FAR_LEVELS;
        let padded = (
            coarse.0.next_multiple_of(halved),
            coarse.1.next_multiple_of(halved),
        );
        for texture in [self.far_light, self.far_reach] {
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate(texture, extent(padded.0, padded.1), 1 + FAR_LEVELS);
        }
        self.spreads = (0..=FAR_LEVELS)
            .map(|level| {
                [
                    (padded.0 >> level).div_ceil(PYRAMID_GROUP),
                    (padded.1 >> level).div_ceil(PYRAMID_GROUP),
                    1,
                ]
            })
            .collect();
        cx.run(self.spread);
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(self.traced, whole, 1);
        cx.run(self.own);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate(self.levels, whole, 1 + self.count);
        cx.run(self.chain);
        let base = (
            display.x.div_ceil(2).max(1).next_power_of_two(),
            display.y.div_ceil(2).max(1).next_power_of_two(),
        );
        let reductions = mip_levels(base.0, base.1).min(PYRAMID_LEVELS);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate(self.pyramid, extent(base.0, base.1), reductions);
        cx.run(self.reduce);
        self.reduced = extents(self.render, reductions);
        for (at, buffer) in self.extents.iter().enumerate() {
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate_buffer(*buffer, size_of::<PyramidExtent>() as u64);
            if let Some(written) = self.reduced.get(at) {
                cx.write(*buffer, 0, &[*written]);
            }
        }
        cx.run(self.blend);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.reduce {
            for (reduction, written) in self.reductions.iter().zip(&self.reduced) {
                let [width, height] = written.size;
                let groups = [
                    width.div_ceil(PYRAMID_GROUP),
                    height.div_ceil(PYRAMID_GROUP),
                    1,
                ];
                cx.dispatch(reduction, Count::Groups(groups));
            }
            return 0;
        }
        if pass == self.chain {
            let mut draw = cx.render(Attach::ColorMip(self.levels, 0, 0));
            draw.within(self.render);
            draw.draw(&self.first, TRIANGLE);
            for ((level, down), drawn) in (0..self.count).zip(&self.downs).zip(self.drawn) {
                let mut draw = cx.render(Attach::ColorMip(self.levels, level + 1, 0));
                draw.within(drawn);
                draw.draw(down, TRIANGLE);
            }
            return 0;
        }
        if pass == self.survey {
            let mut draw = cx.render(Attach::Color(self.cells, None));
            draw.within(self.surveyed);
            draw.draw(&self.surveying, TRIANGLE);
            return 0;
        }
        if pass == self.spread {
            for (widened, groups) in self.spreading.iter().zip(&self.spreads) {
                cx.dispatch(widened, Count::Groups(*groups));
            }
            return 0;
        }
        if pass == self.own {
            let mut draw = cx.render(Attach::Color(self.traced, None));
            draw.within(self.render);
            draw.draw(&self.owning, TRIANGLE);
            return 0;
        }
        let pipeline = if pass == self.reflect {
            &self.tracing
        } else {
            &self.blending
        };
        let mut draw = cx.render(Attach::Declared);
        draw.within(self.render);
        draw.draw(pipeline, TRIANGLE);
        0
    }
}

const ADDED_COLOR: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

const TRIANGLE: Shape<'static, TraceGpu, Own> = Shape::Pulled {
    vertices: 0..3,
    instances: 0..1,
    material: None,
};

fn screen(fragment: Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: FULLSCREEN,
        fragment: Some(fragment),
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

fn extents(render: (u32, u32), count: u32) -> Vec<PyramidExtent> {
    pyramid_extents(render, count)
        .into_iter()
        .map(|[width, height, last_x, last_y]| PyramidExtent {
            size: [width, height],
            last: [last_x, last_y],
        })
        .collect()
}
