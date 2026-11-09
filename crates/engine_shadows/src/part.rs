use bevy_ecs::prelude::*;
use engine_core::{Error, PassViews, ViewFacts, Views};
use engine_places_rendering::shadows::{CasterMap, CasterView, SHADOW_REGIONS, ShadowMaps};
use engine_render::{
    Access, AfterRead, Allocation, Attach, AttachPoint, Bind, Buffer, ComputePipeline, Count,
    DebugShown, DepthUse, DrawInput, Encode, FULLSCREEN, GpuPart, Install, Lifetime, Own, Pass,
    Pipeline, Prepare, RenderPipeline, Shape, TargetName, Texture, VertexLayout, wgpu,
};

use crate::ShadowPass;
use crate::constants::{MARK_GROUP, SKY_LAYERS};
use crate::gpu_layouts::{CasterCull, ComposeData, Route, SkyViews};
use crate::plan::{LOST, MarkKind, SKIPPED, SKY_LOST, ShadowDrawn, ShadowFeedback, ShadowPlan};
use crate::shaders;
use crate::sky::RESOLUTION;

pub(crate) const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
const SKY_OPEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
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
    marks: Pass<Self>,
    opening: Pass<Self>,
    regions: Texture<Self>,
    atlas: Texture<Self>,
    sky: Texture<Self>,
    sky_open: Texture<Self>,
    tiles: Buffer<Self>,
    map_marks: Buffer<Self>,
    depth_marks: Buffer<Self>,
    routes: Buffer<Self>,
    cull: Buffer<Self>,
    sky_views: Buffer<Self>,
    composes: Buffer<Self>,
    composing: Pipeline<Self, Own>,
    clearing: Pipeline<Self, Own>,
    sky_opening: Pipeline<Self, Own>,
    compose_vertices: DrawInput<Self, Own>,
    marking: ComputePipeline<Self>,
    at: Vec<CasterView>,
    composed: u32,
    moving: u32,
    marked: Option<[u32; 3]>,
    opened: Option<(u32, u32)>,
    sky_live: bool,
    drawing: Option<(u32, Vec<u8>)>,
    written_table: Vec<u64>,
    compose_bytes: u64,
}

fn compose_pipeline(
    vertex: engine_core::Entry,
    fragment: Option<engine_core::Entry>,
    vertices: &'static [VertexLayout],
) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex,
        fragment,
        vertices,
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::Overwrite,
        blend: None,
        writes: wgpu::ColorWrites::empty(),
    }
}

fn extent(width: u32, height: u32, layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: layers,
    }
}

impl ShadowsGpu {
    fn upload(&mut self, plan: &ShadowPlan, cx: &mut Prepare<'_, Self>) {
        let routes = (plan.routes.len().max(1) * size_of::<Route>()) as u64;
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(self.routes, routes);
        cx.write(self.routes, 0, &plan.routes);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.cull, size_of::<CasterCull>() as u64);
        cx.write(self.cull, 0, &[plan.cull]);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.sky_views, size_of::<SkyViews>() as u64);
        cx.write(self.sky_views, 0, std::slice::from_ref(&*plan.sky.uniform));
        if !plan.composes.is_empty() {
            let bytes = std::mem::size_of_val(plan.composes.as_slice()) as u64;
            self.compose_bytes = self.compose_bytes.max(bytes);
            let (Allocation::Fresh | Allocation::Kept) =
                cx.allocate_buffer(self.composes, self.compose_bytes);
            cx.write(self.composes, 0, &plan.composes);
        }
        let side = plan.side();
        let cells = (side * side) as usize;
        self.written_table.resize(plan.revisions.len(), u64::MAX);
        for (level, &revision) in plan.revisions.iter().enumerate() {
            if self.written_table[level] == revision {
                continue;
            }
            let Some(block) = plan.table.get(level * cells..(level + 1) * cells) else {
                continue;
            };
            cx.write(self.tiles, (level * cells * 4) as u64, block);
            self.written_table[level] = revision;
        }
        if !plan.levels.is_empty() {
            let start = plan.levels.len() * cells * 4;
            cx.write(self.tiles, start as u64, &plan.levels);
        }
    }

    fn allocate(&mut self, plan: &ShadowPlan, cx: &mut Prepare<'_, Self>) -> (bool, bool) {
        let Some(key) = plan.key else {
            return (false, false);
        };
        let (width, height) = key.atlas.size();
        let side = key.region_side();
        let atlas = cx.allocate(self.atlas, extent(width, height, 1), 1) == Allocation::Fresh;
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate(self.regions, extent(side, side, SHADOW_REGIONS), 1);
        if cx.allocate_buffer(self.tiles, key.table_bytes()) == Allocation::Fresh {
            self.written_table.clear();
        }
        let mut sky = false;
        if plan.sky.live {
            sky = cx.allocate(self.sky, extent(RESOLUTION, RESOLUTION, SKY_LAYERS), 1)
                == Allocation::Fresh;
        } else {
            cx.release(self.sky);
        }
        (atlas, sky)
    }
}

impl GpuPart for ShadowsGpu {
    type Reads = (
        Res<'static, ShadowMaps>,
        Res<'static, ShadowPlan>,
        Res<'static, Views>,
        Res<'static, DebugShown>,
    );
    const NAME: &'static str = "shadows";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::shadows::faces::routes as face_routes;
        use shaders::shadows::{
            faces::sky_visibility as face_sky, maps as map_shaders, marks as mark_shaders, sample,
            sky as sky_shaders,
        };
        let regions = cx.texture("shadow_regions", SHADOW_FORMAT, Lifetime::Kept);
        let atlas = cx.texture("shadow_atlas", SHADOW_FORMAT, Lifetime::Kept);
        let sky = cx.texture("sky_maps", SHADOW_FORMAT, Lifetime::Kept);
        let opened = cx.view_texture("sky_open", SKY_OPEN_FORMAT, Lifetime::Kept);
        let tiles = cx.buffer("shadow_tiles", Lifetime::Kept);
        let map_marks = cx.buffer("caster_marks", Lifetime::Kept);
        let depth_marks = cx.buffer("shadow_marks", Lifetime::Kept);
        let routes = cx.buffer("shadow_routes", Lifetime::Kept);
        let cull = cx.buffer("caster_cull", Lifetime::Kept);
        let sky_views = cx.buffer("sky_views", Lifetime::Kept);
        let composes = cx.buffer("shadow_composes", Lifetime::Kept);
        let compare = cx.sampler(engine_core::Sampler::COMPARE_LINEAR);
        let texels = cx.sampler(engine_core::Sampler::NEAREST);
        cx.frame(sample::SHADOW_ATLAS, Bind::Texture(atlas))?;
        cx.frame(sample::SHADOW_TILES, Bind::Buffer(tiles))?;
        cx.frame(sample::SHADOW_SAMPLER, Bind::Sampler(compare))?;
        cx.frame(sample::SHADOW_TEXELS, Bind::Sampler(texels))?;
        cx.frame(sky_shaders::SKY_MAPS, Bind::Texture(sky))?;
        cx.frame(sky_shaders::SKY_SAMPLER, Bind::Sampler(compare))?;
        cx.frame(sky_shaders::SKY_VIEWS, Bind::Buffer(sky_views))?;
        cx.frame(face_sky::SKY_OPEN, Bind::Texture(opened))?;
        cx.frame(face_routes::CASTER_CULL, Bind::Buffer(cull))?;
        cx.frame(face_routes::CASTER_ROUTES, Bind::Buffer(routes))?;
        cx.frame(face_routes::CASTER_MARKS, Bind::Buffer(map_marks))?;
        let maps = cx
            .scene::<ShadowPass>()
            .label("engine/shadow-map")
            .depth_cleared(regions, 1.0)
            .depth(atlas, Access::Load)
            .depth_cleared(sky, 1.0)
            .id();
        let marks = cx
            .pass("shadow_marks")
            .label("engine/shadow-marks")
            .span("shadow_marks")
            .at(AttachPoint::LateOpaque)
            .id();
        let opening = cx
            .pass("sky_occlusion")
            .at(AttachPoint::Inputs)
            .views(PassViews::Main)
            .color(opened, Access::Clear)
            .id();
        let compose_vertices = cx.draw_input(maps, composes);
        let composing = cx.render_pipeline(
            maps,
            &compose_pipeline(
                map_shaders::VS_COMPOSE,
                Some(map_shaders::FS_COMPOSE),
                COMPOSE_VERTICES,
            ),
            &[(map_shaders::SHADOW_REGIONS, Bind::Texture(regions))],
        )?;
        let clearing = cx.render_pipeline(
            maps,
            &compose_pipeline(map_shaders::VS_CLEAR_MOVING, None, &[]),
            &[],
        )?;
        let sky_opening = cx.render_pipeline(
            opening,
            &RenderPipeline {
                vertex: FULLSCREEN,
                fragment: Some(sky_shaders::FS_SKY_OCCLUSION),
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::None,
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &[],
        )?;
        let marking = cx.compute_pipeline(
            marks,
            mark_shaders::MARK_SHADOW_TILES,
            &[
                (mark_shaders::SHADOW_DEPTH, Bind::Target(TargetName::DEPTH)),
                (mark_shaders::SHADOW_MARKS, Bind::Buffer(depth_marks)),
            ],
        )?;
        Ok(Self {
            maps,
            marks,
            opening,
            regions,
            atlas,
            sky,
            sky_open: opened,
            tiles,
            map_marks,
            depth_marks,
            routes,
            cull,
            sky_views,
            composes,
            composing,
            clearing,
            sky_opening,
            compose_vertices,
            marking,
            at: Vec::new(),
            composed: 0,
            moving: 0,
            marked: None,
            opened: None,
            sky_live: false,
            drawing: None,
            written_table: Vec::new(),
            compose_bytes: 0,
        })
    }

    fn prepare(
        &mut self,
        (maps, plan, views, shown): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if let Some((layout, tiles)) = self.drawing.take()
            && !cx.ran(self.maps)
        {
            cx.deliver::<ShadowDrawn>(layout, tiles);
        }
        self.sky_live = plan.sky.live && shown.owner.is_none();
        let (atlas_lost, sky_lost) = self.allocate(&plan, cx);
        self.upload(&plan, cx);
        self.at.clear();
        self.at.extend(maps.views());
        (self.composed, self.moving) = (plan.composes.len() as u32, plan.moving);
        if !self.at.is_empty() || self.composed > 0 {
            cx.run(self.maps);
        }
        let layout = u32::from(plan.layout);
        if atlas_lost {
            cx.deliver::<ShadowDrawn>(layout, plan.drawn_bytes(LOST));
        }
        if sky_lost {
            cx.deliver::<ShadowDrawn>(layout, vec![SKY_LOST]);
        }
        if self.composed > 0 {
            self.drawing = Some((layout, plan.drawn_bytes(SKIPPED)));
        }
        let size = views.main().map_or(engine_core::glam::UVec2::ZERO, |main| main.render);
        self.marked = (plan.mask_words != 0 && size.x != 0 && size.y != 0).then_some([
            size.x.div_ceil(MARK_GROUP),
            size.y.div_ceil(MARK_GROUP),
            1,
        ]);
        let mask_bytes = plan.mark_bytes().max(4);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.map_marks, 2 * mask_bytes);
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.depth_marks, mask_bytes);
        if plan.mask_words == 0 {
            return;
        }
        let bytes = 0..plan.mark_bytes();
        if self.marked.is_some() {
            cx.run(self.marks);
            cx.read_back::<ShadowFeedback>(
                plan.tag(MarkKind::Depth),
                self.depth_marks,
                bytes.clone(),
                AfterRead::Clear,
            );
        }
        cx.read_back::<ShadowFeedback>(
            plan.tag(MarkKind::Map),
            self.map_marks,
            bytes.clone(),
            AfterRead::Clear,
        );
        cx.read_back::<ShadowFeedback>(
            plan.tag(MarkKind::Masks),
            self.map_marks,
            bytes.end..2 * bytes.end,
            AfterRead::Clear,
        );
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        self.opened = None;
        if !self.sky_live || !view.target.surface() || cx.view_index() != 0 {
            cx.release(self.sky_open);
            return;
        }
        let display = view.size();
        let render = view.render;
        let within = |size: u32, whole: u32| size.clamp(1, whole.max(1));
        self.opened = Some((within(render.x, display.x), within(render.y, display.y)));
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(
            self.sky_open,
            extent(display.x.max(1), display.y.max(1), 1),
            1,
        );
        cx.run(self.opening);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.opening {
            if let Some(drawn) = self.opened {
                let mut draw = cx.render(Attach::Declared);
                draw.within(drawn);
                draw.draw(
                    &self.sky_opening,
                    Shape::Pulled {
                        vertices: 0..3,
                        instances: 0..1,
                    },
                );
            }
            return 0;
        }
        if pass == self.marks {
            if let Some(groups) = self.marked {
                cx.dispatch(&self.marking, Count::Groups(groups));
            }
            return 0;
        }
        let count = |map: CasterMap| self.at.iter().filter(|at| at.map == map).count() as u32;
        let regions = count(CasterMap::Region);
        let tiles = regions..regions + count(CasterMap::Moving);
        let sky = tiles.end..tiles.end + count(CasterMap::Sky);
        cx.span("shadow_regions");
        let mut drawn = cx.views(self, &Attach::Depth(self.regions, None), 0..regions);
        if self.composed > 0 || self.moving > 0 {
            cx.span("shadow_atlas");
            let vertices = [self.compose_vertices];
            let mut atlas = cx.render(Attach::Depth(self.atlas, None));
            if self.composed > 0 {
                atlas.draw(
                    &self.composing,
                    Shape::Buffers {
                        vertices: &vertices,
                        indices: None,
                        elements: 0..6,
                        instances: 0..self.composed,
                    },
                );
            }
            if self.moving > 0 {
                atlas.draw(
                    &self.clearing,
                    Shape::Pulled {
                        vertices: 0..6,
                        instances: 0..self.moving,
                    },
                );
            }
        }
        drawn += cx.views(self, &Attach::Depth(self.atlas, None), tiles);
        if !sky.is_empty() {
            cx.span("shadow_sky");
            drawn += cx.views(self, &Attach::Depth(self.sky, None), sky);
        }
        drawn
    }
}
