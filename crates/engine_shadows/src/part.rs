use bevy_ecs::prelude::*;
use engine_core::{Error, View};
use engine_places::shadows::{CasterMap, CasterView, SHADOW_REGIONS, ShadowCasters, ShadowMaps};
use engine_render::{
    Access, AfterRead, Attach, Bind, Buffer, CameraUniform, ComputePipeline, Count, DepthUse,
    DrawInput, Draws, Encode, GpuPart, Install, Layer, Lifetime, Own, Pass, Pipeline, Prepare,
    RenderPipeline, SamplerDesc, Shape, Texture, VertexLayout, ViewTarget, wgpu,
};

use crate::constants::{MARK_GROUP, SKY_LAYERS};
use crate::gpu_layouts::{CasterCull, ComposeData, Routes, SkyViews};
use crate::plan::{DEPTH_MARKS, MAP_MARKS, MASKS, ShadowFeedback, ShadowPlan};
use crate::regions::{MOVING_VIEW, SKY_VIEW};
use crate::shaders;
use crate::sky::RESOLUTION;

const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
const TABLE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
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

#[derive(Default)]
struct Written {
    cameras: Option<Vec<CameraUniform>>,
    routes: Option<Box<Routes>>,
    cull: Option<CasterCull>,
    sky: Option<Box<SkyViews>>,
    table: Vec<u64>,
    compose_bytes: u64,
}

pub(crate) struct ShadowsGpu {
    maps: Pass<Self>,
    marks: Pass<Self>,
    regions: Texture<Self>,
    atlas: Texture<Self>,
    sky: Texture<Self>,
    tiles: Texture<Self>,
    map_marks: Buffer<Self>,
    depth_marks: Buffer<Self>,
    masks: Buffer<Self>,
    routes: Buffer<Self>,
    cull: Buffer<Self>,
    sky_views: Buffer<Self>,
    composes: Buffer<Self>,
    composing: Pipeline<Self, Own>,
    clearing: Pipeline<Self, Own>,
    compose_vertices: DrawInput<Self, Own>,
    marking: ComputePipeline<Self>,
    casters: Draws<ShadowCasters>,
    at: Vec<CasterView>,
    sky_layer: Option<u32>,
    composed: u32,
    moving: u32,
    marked: Option<[u32; 3]>,
    made: Option<u64>,
    written: Written,
}

fn compose_pipeline(
    vertex: engine_core::Entry,
    fragment: Option<engine_core::Entry>,
    vertices: &'static [VertexLayout],
) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex,
        fragment,
        debug: None,
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
        if self.written.cameras.as_deref() != Some(plan.cameras.as_slice()) {
            cx.cameras(&plan.cameras);
            self.written.cameras = Some(plan.cameras.clone());
        }
        if self.written.routes.as_deref() != Some(&*plan.routes) {
            cx.write(self.routes, 0, std::slice::from_ref(&*plan.routes));
            self.written.routes = Some(plan.routes.clone());
        }
        if self
            .written
            .cull
            .is_none_or(|written| bytemuck::bytes_of(&written) != bytemuck::bytes_of(&plan.cull))
        {
            cx.write(self.cull, 0, &[plan.cull]);
            self.written.cull = Some(plan.cull);
        }
        if self.written.sky.as_deref() != Some(&*plan.sky.uniform) {
            cx.write(self.sky_views, 0, std::slice::from_ref(&*plan.sky.uniform));
            self.written.sky = Some(plan.sky.uniform.clone());
        }
        if !plan.composes.is_empty() {
            let bytes = std::mem::size_of_val(plan.composes.as_slice()) as u64;
            self.written.compose_bytes = self.written.compose_bytes.max(bytes);
            cx.allocate_buffer(self.composes, self.written.compose_bytes);
            cx.write(self.composes, 0, &plan.composes);
        }
        let side = plan.side();
        let cells = (side * side) as usize;
        self.written.table.resize(plan.revisions.len(), u64::MAX);
        for (level, &revision) in plan.revisions.iter().enumerate() {
            if self.written.table[level] == revision {
                continue;
            }
            let Some(block) = plan.table.get(level * cells..(level + 1) * cells) else {
                continue;
            };
            cx.write_texture(
                self.tiles,
                wgpu::Origin3d {
                    x: 0,
                    y: level as u32 * side,
                    z: 0,
                },
                bytemuck::cast_slice(block),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(side * 4),
                    rows_per_image: Some(side),
                },
                extent(side, side, 1),
            );
            self.written.table[level] = revision;
        }
    }

    fn allocate(&mut self, plan: &ShadowPlan, cx: &mut Prepare<'_, Self>) {
        let Some(key) = plan.key else {
            return;
        };
        if self.made != Some(plan.made) {
            self.made = Some(plan.made);
            for texture in [self.atlas, self.regions, self.tiles] {
                cx.release(texture);
            }
        }
        let (width, height) = key.atlas.size();
        let side = key.region_side();
        cx.allocate(self.atlas, extent(width, height, 1), 1);
        cx.allocate(self.regions, extent(side, side, SHADOW_REGIONS), 1);
        cx.allocate(self.tiles, extent(key.side, key.side * key.levels, 1), 1);
        if plan.sky.live {
            cx.allocate(self.sky, extent(RESOLUTION, RESOLUTION, SKY_LAYERS), 1);
        } else {
            cx.release(self.sky);
        }
        let mask_bytes = plan.mark_bytes().max(4);
        for marks in [self.map_marks, self.depth_marks, self.masks] {
            cx.allocate_buffer(marks, mask_bytes);
        }
        cx.allocate_buffer(self.routes, size_of::<Routes>() as u64);
        cx.allocate_buffer(self.cull, size_of::<CasterCull>() as u64);
        cx.allocate_buffer(self.sky_views, size_of::<SkyViews>() as u64);
    }
}

impl GpuPart for ShadowsGpu {
    type Reads = (Res<'static, ShadowMaps>, Res<'static, ShadowPlan>, Res<'static, View>);

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::shadows::faces::routes as face_routes;
        use shaders::shadows::{
            maps as map_shaders, marks as mark_shaders, sample, sky as sky_shaders,
        };
        let regions = cx.texture("shadow_regions", SHADOW_FORMAT, Lifetime::Kept);
        let atlas = cx.texture("shadow_atlas", SHADOW_FORMAT, Lifetime::Kept);
        let sky = cx.texture("sky_maps", SHADOW_FORMAT, Lifetime::Kept);
        let tiles = cx.texture("shadow_tiles", TABLE_FORMAT, Lifetime::Kept);
        let map_marks = cx.buffer("caster_marks", Lifetime::Kept);
        let depth_marks = cx.buffer("shadow_marks", Lifetime::Kept);
        let masks = cx.buffer("caster_masks", Lifetime::Kept);
        let routes = cx.buffer("shadow_routes", Lifetime::Kept);
        let cull = cx.buffer("caster_cull", Lifetime::Kept);
        let sky_views = cx.buffer("sky_views", Lifetime::Kept);
        let composes = cx.buffer("shadow_composes", Lifetime::Kept);
        let compare = cx.sampler(SamplerDesc::COMPARE_LINEAR);
        let texels = cx.sampler(SamplerDesc::NEAREST);
        cx.frame(sample::SHADOW_ATLAS, Bind::Texture(atlas))?;
        cx.frame(sample::SHADOW_TILES, Bind::Texture(tiles))?;
        cx.frame(sample::SHADOW_SAMPLER, Bind::Sampler(compare))?;
        cx.frame(sample::SHADOW_TEXELS, Bind::Sampler(texels))?;
        cx.frame(sky_shaders::SKY_MAPS, Bind::Texture(sky))?;
        cx.frame(sky_shaders::SKY_SAMPLER, Bind::Sampler(compare))?;
        cx.frame(sky_shaders::SKY_VIEWS, Bind::Buffer(sky_views))?;
        cx.frame(face_routes::CASTER_CULL, Bind::Buffer(cull))?;
        cx.frame(face_routes::CASTER_ROUTES, Bind::Buffer(routes))?;
        cx.frame(face_routes::CASTER_MARKS, Bind::Buffer(map_marks))?;
        cx.frame(face_routes::CASTER_MASKS, Bind::Buffer(masks))?;
        let maps = cx
            .pass("shadow_maps")
            .depth_cleared(regions, 1.0)
            .depth(atlas, Access::Load)
            .depth_cleared(sky, 1.0)
            .id();
        let marks = cx.pass("shadow_marks").at(Layer::Backdrop).id();
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
        let marking = cx.compute_pipeline(
            marks,
            mark_shaders::MARK_SHADOW_TILES,
            &[
                (mark_shaders::SHADOW_DEPTH, Bind::Target(ViewTarget::Depth)),
                (mark_shaders::SHADOW_MARKS, Bind::Buffer(depth_marks)),
            ],
        )?;
        let casters = cx.claim::<ShadowCasters>(maps)?;
        Ok(Self {
            maps,
            marks,
            regions,
            atlas,
            sky,
            tiles,
            map_marks,
            depth_marks,
            masks,
            routes,
            cull,
            sky_views,
            composes,
            composing,
            clearing,
            compose_vertices,
            marking,
            casters,
            at: Vec::new(),
            sky_layer: None,
            composed: 0,
            moving: 0,
            marked: None,
            made: None,
            written: Written::default(),
        })
    }

    fn prepare(
        &mut self,
        (maps, plan, view): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.allocate(&plan, cx);
        self.upload(&plan, cx);
        self.at.clear();
        self.at.extend(maps.regions.iter().enumerate().map(|(index, region)| CasterView {
            region: index as u32,
            map: region.map,
            refresh: region.refresh,
        }));
        self.sky_layer = plan.sky_draw.map(|draw| draw.layer as u32);
        (self.composed, self.moving) = (plan.composes.len() as u32, plan.moving);
        if !self.at.is_empty() || self.composed > 0 {
            cx.run(self.maps);
        }
        let size = view.render_size();
        self.marked = (plan.mask_words != 0 && size.x != 0 && size.y != 0).then_some([
            size.x.div_ceil(MARK_GROUP),
            size.y.div_ceil(MARK_GROUP),
            1,
        ]);
        if plan.mask_words == 0 {
            return;
        }
        let bytes = 0..plan.mark_bytes();
        if self.marked.is_some() {
            cx.run(self.marks);
            cx.read_back::<ShadowFeedback>(
                DEPTH_MARKS,
                self.depth_marks,
                bytes.clone(),
                AfterRead::Clear,
            );
        }
        cx.read_back::<ShadowFeedback>(MAP_MARKS, self.map_marks, bytes.clone(), AfterRead::Clear);
        cx.read_back::<ShadowFeedback>(MASKS, self.masks, bytes, AfterRead::Clear);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.marks {
            if let Some(groups) = self.marked {
                cx.dispatch(&self.marking, Count::Groups(groups));
            }
            return 0;
        }
        let mut drawn = 0;
        cx.span("shadow_regions");
        for at in self.at.iter().filter(|at| at.map == CasterMap::Region) {
            let mut draw = cx.render(Attach::Depth(self.regions, Some(at.region)), Some(at.region));
            drawn += self.casters.draw(&mut draw, (), at);
        }
        if self.composed > 0 || self.moving > 0 {
            cx.span("shadow_atlas");
            let vertices = [self.compose_vertices];
            let mut atlas = cx.render(Attach::Depth(self.atlas, None), Some(MOVING_VIEW));
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
                for at in self.at.iter().filter(|at| at.map == CasterMap::Moving) {
                    drawn += self.casters.draw(&mut atlas, (), at);
                }
            }
        }
        if let Some(layer) = self.sky_layer {
            cx.span("shadow_sky");
            for at in self.at.iter().filter(|at| at.map == CasterMap::Sky) {
                let mut draw = cx.render(Attach::Depth(self.sky, Some(layer)), Some(SKY_VIEW));
                drawn += self.casters.draw(&mut draw, (), at);
            }
        }
        drawn
    }
}
