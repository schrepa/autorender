use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::{Entry, Error, PassViews, Sampler, ViewFacts, Wrap};
use engine_places_rendering::shadows::{CasterMap, CasterView, ShadowMaps};
use engine_render::{
    Access, Allocation, Attach, AttachPoint, Bind, Buffer, DebugShown, DepthUse, DrawInput,
    Encode, FULLSCREEN, GpuPart, Install, Lifetime, Own, Pass, Pipeline, Prepare, Recorded,
    RenderPipeline, Shape, Texture, VertexLayout, wgpu,
};

use crate::ShadowPass;
use crate::constants::SKY_LAYERS;
use crate::gpu_layouts::{ComposeData, ShadowLevel, SkyViews};
use crate::plan::{LOST, SKY_LOST, SKY_STALE, ShadowDrawn, ShadowPlan, WAITED};
use crate::shaders;
use crate::sky::RESOLUTION;

pub(crate) const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
const SKY_OPEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const COMPOSE_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 0,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Sint32x4,
        offset: 16,
        shader_location: 4,
    },
];
const COMPOSE_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: size_of::<ComposeData>() as u64,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: &COMPOSE_ATTRIBUTES,
};
const COMPOSE_VERTICES: &[VertexLayout] = &[VertexLayout::Buffer(COMPOSE_LAYOUT)];

pub(crate) struct ShadowsGpu {
    maps: Pass<Self>,
    composing_pass: Pass<Self>,
    opening: Pass<Self>,
    regions: Texture<Self>,
    levels: Texture<Self>,
    sky: Texture<Self>,
    sky_open: Texture<Self>,
    level_data: Buffer<Self>,
    sky_views: Buffer<Self>,
    composes: Buffer<Self>,
    composing: Pipeline<Self, Own>,
    sky_opening: Pipeline<Self, Own>,
    compose_vertices: DrawInput<Self, Own>,
    at: Vec<CasterView>,
    composed: Vec<(u32, Range<u32>)>,
    drawn: Option<(u32, u32)>,
    sky_live: bool,
    drawing: Option<(u32, Vec<u8>)>,
    compose_bytes: u64,
    kept: Recorded,
    sky_drawn: Recorded,
    stood_in: Vec<(u32, Vec<u8>)>,
    sky_stood_in: bool,
    preparing: bool,
    region_levels: Vec<u32>,
    sky_layers: Vec<u32>,
    waited: Vec<u32>,
    sky_waited: Vec<u32>,
}

fn listed(status: u8, listed: &[u32]) -> Vec<u8> {
    let mut bytes = vec![status];
    for word in listed {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn extent(width: u32, height: u32, layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: layers,
    }
}

fn repeat(sampler: Sampler) -> Sampler {
    Sampler {
        wrap_u: Wrap::Repeat,
        wrap_v: Wrap::Repeat,
        ..sampler
    }
}

fn fullscreen(fragment: Entry) -> RenderPipeline<'static> {
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

impl ShadowsGpu {
    fn upload(&mut self, plan: &ShadowPlan, cx: &mut Prepare<'_, Self>) {
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.sky_views, size_of::<SkyViews>() as u64);
        cx.write(self.sky_views, 0, std::slice::from_ref(&*plan.sky.uniform));
        let levels = (plan.level_data.len().max(1) * size_of::<ShadowLevel>()) as u64;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.level_data, levels);
        if !plan.level_data.is_empty() {
            cx.write(self.level_data, 0, &plan.level_data);
        }
        if !plan.composes.is_empty() {
            let bytes = std::mem::size_of_val(plan.composes.as_slice()) as u64;
            self.compose_bytes = self.compose_bytes.max(bytes);
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate_buffer(self.composes, self.compose_bytes);
            cx.write(self.composes, 0, &plan.composes);
        }
    }

    fn allocate(&mut self, plan: &ShadowPlan, cx: &mut Prepare<'_, Self>) -> (bool, bool) {
        let Some(key) = plan.key else {
            return (false, false);
        };
        let side = key.side;
        let lost =
            cx.allocate(self.levels, extent(side, side, key.layers()), 1) == Allocation::Fresh;
        let (wide, layers) = crate::regions::region_shape(side);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate(self.regions, extent(wide, wide, layers), 1);
        let mut sky = false;
        if plan.sky.live {
            sky = cx.allocate(self.sky, extent(RESOLUTION, RESOLUTION, SKY_LAYERS), 1)
                == Allocation::Fresh;
        } else {
            cx.release(self.sky);
        }
        (lost, sky)
    }
}

impl GpuPart for ShadowsGpu {
    type Reads = (
        Res<'static, ShadowMaps>,
        Res<'static, ShadowPlan>,
        Res<'static, DebugShown>,
    );
    const NAME: &'static str = "shadows";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::shadows::{
            faces::sky_visibility as face_sky, maps as map_shaders, sample, sky as sky_shaders,
        };
        let regions = cx.texture("shadow_regions", SHADOW_FORMAT, Lifetime::Kept);
        let levels = cx.texture("shadow_levels", SHADOW_FORMAT, Lifetime::Kept);
        let sky = cx.texture("sky_maps", SHADOW_FORMAT, Lifetime::Kept);
        let opened = cx.view_texture("sky_open", SKY_OPEN_FORMAT, Lifetime::Kept);
        let level_data = cx.buffer("shadow_level_data", Lifetime::Kept);
        let sky_views = cx.buffer("sky_views", Lifetime::Kept);
        let composes = cx.buffer("shadow_composes", Lifetime::Kept);
        let compare = cx.sampler(Sampler::COMPARE_LINEAR);
        let level_compare = cx.sampler(repeat(Sampler::COMPARE_LINEAR));
        let texels = cx.sampler(repeat(Sampler::NEAREST));
        cx.frame(sample::SHADOW_LEVELS, Bind::Texture(levels))?;
        cx.frame(sample::SHADOW_LEVEL_DATA, Bind::Buffer(level_data))?;
        cx.frame(sample::SHADOW_SAMPLER, Bind::Sampler(level_compare))?;
        cx.frame(sample::SHADOW_TEXELS, Bind::Sampler(texels))?;
        cx.frame(sky_shaders::SKY_MAPS, Bind::Texture(sky))?;
        cx.frame(sky_shaders::SKY_SAMPLER, Bind::Sampler(compare))?;
        cx.frame(sky_shaders::SKY_VIEWS, Bind::Buffer(sky_views))?;
        cx.frame(face_sky::SKY_OPEN, Bind::Texture(opened))?;
        let maps = cx
            .scene::<ShadowPass>()
            .label("engine/shadow-map")
            .depth_cleared(regions, 1.0)
            .depth_cleared(levels, 1.0)
            .depth_cleared(sky, 1.0)
            .id();
        let composing_pass = cx
            .pass("shadow_compose")
            .depth(levels, Access::Load)
            .id();
        let opening = cx
            .pass("sky_occlusion")
            .at(AttachPoint::Inputs)
            .views(PassViews::Main)
            .color(opened, Access::Clear)
            .id();
        let compose_vertices = cx.draw_input(composing_pass, composes);
        let composing = cx.render_pipeline(
            composing_pass,
            &RenderPipeline {
                vertex: map_shaders::VS_COMPOSE,
                fragment: Some(map_shaders::FS_COMPOSE),
                vertices: COMPOSE_VERTICES,
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::Overwrite,
                blend: None,
                writes: wgpu::ColorWrites::empty(),
            },
            &[(map_shaders::SHADOW_REGIONS, Bind::Texture(regions))],
        )?;
        let sky_opening =
            cx.render_pipeline(opening, &fullscreen(sky_shaders::FS_SKY_OCCLUSION), &[])?;
        Ok(Self {
            maps,
            composing_pass,
            opening,
            regions,
            levels,
            sky,
            sky_open: opened,
            level_data,
            sky_views,
            composes,
            composing,
            sky_opening,
            compose_vertices,
            at: Vec::new(),
            composed: Vec::new(),
            drawn: None,
            sky_live: false,
            drawing: None,
            compose_bytes: 0,
            kept: Recorded::Exact,
            sky_drawn: Recorded::Exact,
            stood_in: Vec::new(),
            sky_stood_in: false,
            preparing: false,
            region_levels: Vec::new(),
            sky_layers: Vec::new(),
            waited: Vec::new(),
            sky_waited: Vec::new(),
        })
    }

    fn prepare(
        &mut self,
        (maps, plan, shown): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let kept = std::mem::take(&mut self.kept);
        let waited = std::mem::take(&mut self.waited);
        let sky_waited = std::mem::take(&mut self.sky_waited);
        if let Some((layout, levels)) = self.drawing.take() {
            if !(cx.ran(self.maps) && cx.ran(self.composing_pass)) {
                cx.deliver::<ShadowDrawn>(layout, levels);
            } else if kept == Recorded::Waiting && self.preparing {
                cx.deliver::<ShadowDrawn>(layout, listed(WAITED, &waited));
            } else if kept == Recorded::Waiting {
                cx.deliver::<ShadowDrawn>(layout, levels);
            } else if kept == Recorded::StandIn {
                self.stood_in.push((layout, levels));
            }
        }
        let layout = u32::from(plan.layout);
        let stale = if self.preparing { &sky_waited[..] } else { &[] };
        match std::mem::take(&mut self.sky_drawn) {
            Recorded::Waiting => cx.deliver::<ShadowDrawn>(layout, listed(SKY_STALE, stale)),
            Recorded::StandIn => self.sky_stood_in = true,
            Recorded::Exact | Recorded::Refused => {}
        }
        if cx.warming() == 0 {
            for (layout, levels) in self.stood_in.drain(..) {
                cx.deliver::<ShadowDrawn>(layout, levels);
            }
            if std::mem::take(&mut self.sky_stood_in) {
                cx.deliver::<ShadowDrawn>(layout, vec![SKY_STALE]);
            }
        }
        self.sky_live = plan.sky.live && shown.owner.is_none();
        let (lost, sky_lost) = self.allocate(&plan, cx);
        self.upload(&plan, cx);
        self.at.clear();
        self.at.extend(maps.views());
        self.composed.clone_from(&plan.composed);
        if !self.at.is_empty() {
            cx.run(self.maps);
        }
        if !self.composed.is_empty() {
            cx.run(self.composing_pass);
        }
        let layout = u32::from(plan.layout);
        if lost && !plan.restarted {
            cx.deliver::<ShadowDrawn>(layout, vec![LOST]);
        }
        if sky_lost {
            cx.deliver::<ShadowDrawn>(layout, vec![SKY_LOST]);
        }
        if !self.composed.is_empty() {
            self.drawing = Some((layout, plan.drawn_bytes()));
        }
        self.preparing = plan.preparing;
        let regions = plan.levels().regions.iter();
        self.region_levels.clear();
        self.region_levels.extend(regions.map(|region| region.level as u32));
        self.sky_layers.clear();
        self.sky_layers.extend(plan.sky_draws.iter().map(|draw| draw.layer as u32));
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        self.drawn = None;
        if !self.sky_live || !view.target.surface() || cx.view_index() != 0 {
            cx.release(self.sky_open);
            return;
        }
        let display = view.size();
        let render = view.render;
        let within = |size: u32, whole: u32| size.clamp(1, whole.max(1));
        self.drawn = Some((within(render.x, display.x), within(render.y, display.y)));
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(
            self.sky_open,
            extent(display.x.max(1), display.y.max(1), 1),
            1,
        );
        cx.run(self.opening);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.opening {
            if let Some(drawn) = self.drawn {
                let mut draw = cx.render(Attach::Declared);
                draw.within(drawn);
                draw.draw(
                    &self.sky_opening,
                    Shape::Pulled {
                        vertices: 0..3,
                        instances: 0..1,
                        material: None,
                    },
                );
            }
            return 0;
        }
        if pass == self.composing_pass {
            cx.recorded();
            let vertices = [self.compose_vertices];
            for (level, instances) in &self.composed {
                let mut layer = cx.render(Attach::Depth(self.levels, Some(*level)));
                layer.draw(
                    &self.composing,
                    Shape::Buffers {
                        vertices: &vertices,
                        indices: None,
                        elements: 0..6,
                        instances: instances.clone(),
                    },
                );
            }
            let recorded = cx.recorded();
            if recorded == Recorded::Waiting {
                self.waited.extend(self.composed.iter().map(|&(level, _)| level));
            }
            self.kept = self.kept.max(recorded);
            return 0;
        }
        let count = |map: CasterMap| self.at.iter().filter(|at| at.map == map).count() as u32;
        let regions = count(CasterMap::Region);
        let moving = regions..regions + count(CasterMap::Moving);
        let sky = moving.end..moving.end + count(CasterMap::Sky);
        cx.span("shadow_regions");
        cx.recorded();
        let mut drawn = 0;
        for index in 0..regions {
            drawn += cx.views(self, &Attach::Depth(self.regions, None), index..index + 1);
            let recorded = cx.recorded();
            if recorded == Recorded::Waiting {
                self.waited.extend(self.region_levels.get(index as usize));
            }
            self.kept = self.kept.max(recorded);
        }
        if !moving.is_empty() {
            cx.span("shadow_moving");
            drawn += cx.views(self, &Attach::Depth(self.levels, None), moving);
            cx.recorded();
        }
        if !sky.is_empty() {
            cx.span("shadow_sky");
            for (index, view) in sky.enumerate() {
                drawn += cx.views(self, &Attach::Depth(self.sky, None), view..view + 1);
                let recorded = cx.recorded();
                if recorded == Recorded::Waiting {
                    self.sky_waited.extend(self.sky_layers.get(index));
                }
                self.sky_drawn = self.sky_drawn.max(recorded);
            }
        }
        drawn
    }
}
