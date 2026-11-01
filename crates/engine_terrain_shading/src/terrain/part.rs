use std::any::TypeId;

use bevy_ecs::prelude::*;
use engine_core::{Camera, CookMode, Error, FrameTime, View};
use engine_places::ground::{Cover, TerrainTiles};
use engine_places::probes::{ProbeCaptures, ProbeView};
use engine_places::shadows::{CasterView, Refresh, ShadowCasters, ShadowMaps};
use engine_places::water::Waterline;
use engine_render::{
    Access, Attach, Bind, Buffer, ComputePipeline, Count, DebugShown, DepthUse, Draw, DrawInput,
    Encode, GpuPart, Install, Layer, Lifetime, MainView, Own, Pass, Pipeline, Prepare,
    RenderPipeline, Resolve, Sampler, SamplerDesc, Shape, Texture, VertexLayout, wgpu,
};
use glam::{Mat4, Vec3};

use super::atlas::TileAtlas;
use super::patches::{PatchDraws, patch_layout};
use super::pipelines::{CASTER_BIAS, surface_primitive};
use super::surface::SurfaceTrees;
use super::tables::TerrainTables;
use super::{IDS_FORMAT, SURFACE_SLOTS, TerrainFrame};
use crate::TerrainShadingBrick;
use crate::gpu::HEIGHT_FORMAT;
use crate::settings::TerrainShading;
use crate::state::{Atlas, Bakes, SurfaceTotals};
use crate::terrain_bounce::Bounce;
use crate::terrain_horizons::Horizons;
use crate::terrain_macro::MacroMaps;
use crate::terrain_maps::TerrainMaps;
use crate::terrain_sky::SkyMaps;

pub(crate) struct TerrainViews<'a> {
    pub(crate) camera: Mat4,
    pub(crate) culling: Mat4,
    pub(crate) eye: Vec3,
    pub(crate) regions: Vec<(usize, engine_places::shadows::KeptRefresh)>,
    pub(crate) settings: &'a TerrainShading,
    pub(crate) water: &'a Waterline,
    pub(crate) fades: bool,
    pub(crate) now: f32,
    pub(crate) cover: Option<(Cover, f32)>,
}

pub(crate) struct Handles {
    pub(crate) heights: Texture<TerrainGpu>,
    pub(crate) normals: Texture<TerrainGpu>,
    pub(crate) control: Texture<TerrainGpu>,
    pub(crate) rivers: Texture<TerrainGpu>,
    pub(crate) errors: Texture<TerrainGpu>,
    pub(crate) water_errors: Texture<TerrainGpu>,
    pub(crate) ids: Texture<TerrainGpu>,
    pub(crate) terrains: Buffer<TerrainGpu>,
    pub(crate) views: Buffer<TerrainGpu>,
    pub(crate) holes: Buffer<TerrainGpu>,
    pub(crate) directory: Buffer<TerrainGpu>,
    pub(crate) palette: Buffer<TerrainGpu>,
    pub(crate) leaves: Buffer<TerrainGpu>,
    pub(crate) nodes: Buffer<TerrainGpu>,
    pub(crate) counters: Buffer<TerrainGpu>,
    pub(crate) args: Buffer<TerrainGpu>,
    pub(crate) meshlet: Buffer<TerrainGpu>,
    pub(crate) patches: Buffer<TerrainGpu>,
    pub(crate) grid: Buffer<TerrainGpu>,
    pub(crate) normal_jobs: Buffer<TerrainGpu>,
    pub(crate) control_jobs: Buffer<TerrainGpu>,
    pub(crate) slots: [(Buffer<TerrainGpu>, Buffer<TerrainGpu>); SURFACE_SLOTS],
    pub(crate) control_sampler: Sampler<TerrainGpu>,
    pub(crate) linear: Sampler<TerrainGpu>,
}

struct Passes {
    normals: Pass<TerrainGpu>,
    control: Pass<TerrainGpu>,
    clear_counters: Pass<TerrainGpu>,
    begin: Pass<TerrainGpu>,
    copy_args: Pass<TerrainGpu>,
    subdivide: Pass<TerrainGpu>,
    bounce: Pass<TerrainGpu>,
    ids: Pass<TerrainGpu>,
    resolve: Pass<TerrainGpu>,
}

struct Pipelines {
    normals: ComputePipeline<TerrainGpu>,
    control: ComputePipeline<TerrainGpu>,
    begin: [ComputePipeline<TerrainGpu>; SURFACE_SLOTS],
    decide: [ComputePipeline<TerrainGpu>; SURFACE_SLOTS],
    apply: [ComputePipeline<TerrainGpu>; SURFACE_SLOTS],
    ids: [Pipeline<TerrainGpu, Own>; 2],
    resolve: [Pipeline<TerrainGpu, Own>; 2],
    debug: [Pipeline<TerrainGpu, MainView>; 2],
    probe: [Pipeline<TerrainGpu, engine_places::probes::ProbeViews>; 2],
    caster: Pipeline<TerrainGpu, ShadowCasters>,
}

struct Inputs {
    args: DrawInput<TerrainGpu, Own>,
    meshlet: DrawInput<TerrainGpu, Own>,
    main_args: DrawInput<TerrainGpu, MainView>,
    main_meshlet: DrawInput<TerrainGpu, MainView>,
    probe_args: DrawInput<TerrainGpu, engine_places::probes::ProbeViews>,
    probe_meshlet: DrawInput<TerrainGpu, engine_places::probes::ProbeViews>,
    patches: DrawInput<TerrainGpu, ShadowCasters>,
    grid: DrawInput<TerrainGpu, ShadowCasters>,
    subdivision_args: DrawInput<TerrainGpu, Own>,
}

pub(crate) struct TerrainGpu {
    handles: Handles,
    passes: Passes,
    pipelines: Pipelines,
    inputs: Inputs,
    atlas: TileAtlas,
    tables: TerrainTables,
    surface: SurfaceTrees,
    patches: PatchDraws,
    horizons: Horizons,
    maps: TerrainMaps,
    sky: SkyMaps,
    bounce: Bounce,
    macro_maps: MacroMaps,
    frame: TerrainFrame,
    drawn: bool,
    deferred: bool,
    debug: bool,
    temporal: bool,
    render: (u32, u32),
}

impl GpuPart for TerrainGpu {
    type Reads = (
        Res<'static, TerrainShading>,
        Res<'static, TerrainTiles>,
        Res<'static, Atlas>,
        Res<'static, Bakes>,
        Res<'static, Waterline>,
        Res<'static, Cover>,
        Res<'static, ShadowMaps>,
        Res<'static, ProbeCaptures>,
        Res<'static, View>,
        Res<'static, FrameTime>,
        Res<'static, DebugShown>,
        Res<'static, Resolve>,
        Res<'static, CookMode>,
        Res<'static, Camera>,
        Res<'static, engine_core::DeviceLimits>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use crate::shaders::terrain_shading::{
            control, debug, draw, faces, field, horizons, normals, shadow, sky, subdivision,
            surface,
        };
        let handles = Handles {
            heights: cx.texture("heights", HEIGHT_FORMAT, Lifetime::Kept),
            normals: cx.texture("normals", super::NORMAL_FORMAT, Lifetime::Kept),
            control: cx.texture("control", super::CONTROL_FORMAT, Lifetime::Kept),
            rivers: cx.texture("rivers", super::RIVER_FORMAT, Lifetime::Kept),
            errors: cx.texture("errors", wgpu::TextureFormat::R16Float, Lifetime::Kept),
            water_errors: cx.texture("water_errors", wgpu::TextureFormat::R16Float, Lifetime::Kept),
            ids: cx.texture("terrain_ids", IDS_FORMAT, Lifetime::Frame),
            terrains: cx.buffer("terrains", Lifetime::Kept),
            views: cx.buffer("terrain_views", Lifetime::Kept),
            holes: cx.buffer("holes", Lifetime::Kept),
            directory: cx.buffer("directory", Lifetime::Kept),
            palette: cx.buffer("palette", Lifetime::Kept),
            leaves: cx.buffer("leaves", Lifetime::Kept),
            nodes: cx.buffer("nodes", Lifetime::Kept),
            counters: cx.buffer("counters", Lifetime::Kept),
            args: cx.buffer("surface_args", Lifetime::Kept),
            meshlet: cx.buffer("meshlet_indices", Lifetime::Kept),
            patches: cx.buffer("patches", Lifetime::Kept),
            grid: cx.buffer("grid_indices", Lifetime::Kept),
            normal_jobs: cx.buffer("normal_jobs", Lifetime::Frame),
            control_jobs: cx.buffer("control_jobs", Lifetime::Frame),
            slots: std::array::from_fn(|_| {
                (
                    cx.buffer("subdivision", Lifetime::Kept),
                    cx.buffer("subdivision_tree", Lifetime::Kept),
                )
            }),
            control_sampler: cx.sampler(SamplerDesc::NEAREST),
            linear: cx.sampler(SamplerDesc::LINEAR_CLAMP),
        };
        let h = &handles;
        let maps = TerrainMaps::install(cx)?;
        let sky_maps = SkyMaps::install(cx)?;
        let macro_maps = MacroMaps::install(cx)?;
        let horizon_maps = Horizons::install(cx)?;
        cx.frame(faces::heightfield::TERRAIN_VIEWS, Bind::Buffer(h.views))?;
        cx.frame(faces::heightfield::TERRAINS, Bind::Buffer(h.terrains))?;
        cx.frame(faces::heightfield::TERRAIN_SURFACE_LEAVES, Bind::Buffer(h.leaves))?;
        cx.frame(faces::heightfield::TERRAIN_SURFACE_ARGS, Bind::Buffer(h.counters))?;
        cx.frame(field::TERRAIN_HEIGHTS, Bind::Texture(h.heights))?;
        cx.frame(field::TERRAIN_DIRECTORY, Bind::Buffer(h.directory))?;
        cx.frame(draw::TERRAIN_HOLES, Bind::Buffer(h.holes))?;
        cx.frame(draw::TERRAIN_NORMALS, Bind::Texture(h.normals))?;
        cx.frame(draw::TERRAIN_CONTROL, Bind::Texture(h.control))?;
        cx.frame(draw::TERRAIN_CONTROL_SAMPLER, Bind::Sampler(h.control_sampler))?;
        cx.frame(draw::TERRAIN_PALETTE, Bind::Buffer(h.palette))?;
        cx.frame(draw::TERRAIN_ALBEDO, Bind::Texture(maps.albedo))?;
        cx.frame(draw::TERRAIN_MATERIAL_DATA, Bind::Texture(maps.data))?;
        cx.frame(draw::TERRAIN_MATERIAL_SAMPLER, Bind::Sampler(maps.sampler))?;
        cx.frame(draw::TERRAIN_MACRO, Bind::Texture(macro_maps.texture))?;
        cx.frame(sky::TERRAIN_SKY, Bind::Texture(sky_maps.texture))?;
        cx.frame(sky::TERRAIN_SKY_SAMPLER, Bind::Sampler(h.linear))?;
        cx.frame(surface::TERRAIN_ERRORS, Bind::Texture(h.errors))?;
        cx.frame(surface::WATER_ERRORS, Bind::Texture(h.water_errors))?;
        cx.frame(horizons::TERRAIN_HORIZONS, Bind::Texture(horizon_maps.texture))?;
        cx.frame(horizons::TERRAIN_HORIZON_SAMPLER, Bind::Sampler(h.linear))?;
        cx.frame(faces::ground_tiles::HEIGHTS, Bind::Texture(h.heights))?;
        cx.frame(faces::ground_tiles::DIRECTORY, Bind::Buffer(h.directory))?;
        cx.frame(faces::ground_tiles::CONTROL, Bind::Texture(h.control))?;
        cx.frame(faces::ground_tiles::NORMALS, Bind::Texture(h.normals))?;
        cx.frame(faces::ground_tiles::MACRO_MAPS, Bind::Texture(macro_maps.texture))?;
        cx.frame(faces::ground_tiles::PALETTE, Bind::Buffer(h.palette))?;

        let normals_pass = cx.pass("tile_normals").id();
        let control_pass = cx.pass("tile_control").id();
        let clear_counters = cx.clear("clear_surface_counters", h.counters);
        let begin = cx.pass("subdivision_begin").after(clear_counters).after(control_pass).after(normals_pass).id();
        let copy_args = cx.copy(
            "subdivision_args",
            engine_render::Copied::Buffer(h.counters),
            engine_render::Copied::Buffer(h.args),
        );
        let subdivide = cx.pass("subdivision").after(copy_args).after(begin).id();
        let bounce_pass = cx.pass("bounce").id();
        let ids_pass = cx.pass("terrain_ids").at(Layer::Opaque).color(h.ids, Access::Clear).id();
        let resolve_pass = cx.pass("terrain_resolve").at(Layer::Opaque).after(ids_pass).id();

        let normals_pipeline = cx.compute_pipeline(
            normals_pass,
            normals::TILE_NORMALS,
            &[
                (normals::NORMAL_JOBS, Bind::Buffer(h.normal_jobs)),
                (normals::NORMALS, Bind::Texture(h.normals)),
            ],
        )?;
        let control_pipeline = cx.compute_pipeline(
            control_pass,
            control::TILE_CONTROL,
            &[
                (control::CONTROL_JOBS, Bind::Buffer(h.control_jobs)),
                (control::CONTROL, Bind::Texture(h.control)),
            ],
        )?;
        let slot_binds = |slot: usize| {
            let (uniform, tree) = h.slots[slot];
            [
                (subdivision::SUBDIVISION, Bind::Buffer(uniform)),
                (subdivision::SUBDIVISION_TREE, Bind::Buffer(tree)),
                (subdivision::SUBDIVISION_NODES, Bind::Buffer(h.nodes)),
                (subdivision::SUBDIVISION_LEAVES, Bind::Buffer(h.leaves)),
                (subdivision::SUBDIVISION_ARGS, Bind::Buffer(h.counters)),
            ]
        };
        let mut begins = Vec::with_capacity(SURFACE_SLOTS);
        let mut decides = Vec::with_capacity(SURFACE_SLOTS);
        let mut applies = Vec::with_capacity(SURFACE_SLOTS);
        for slot in 0..SURFACE_SLOTS {
            begins.push(cx.compute_pipeline(begin, subdivision::SUBDIVIDE_BEGIN, &slot_binds(slot))?);
            decides.push(cx.compute_pipeline(subdivide, subdivision::SUBDIVIDE_DECIDE, &slot_binds(slot))?);
            applies.push(cx.compute_pipeline(subdivide, subdivision::SUBDIVIDE_APPLY, &slot_binds(slot))?);
        }
        let surface = |fragment, debug_fragment| RenderPipeline {
            vertex: draw::VS_TERRAIN,
            fragment: Some(fragment),
            debug: debug_fragment,
            vertices: &[],
            primitive: surface_primitive(wgpu::FrontFace::Ccw),
            depth: DepthUse::Write {
                bias: wgpu::DepthBiasState::default(),
            },
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        };
        let ids = [
            cx.render_pipeline(ids_pass, &surface(draw::FS_TERRAIN_IDS, None), &[])?,
            cx.render_pipeline(ids_pass, &surface(draw::FS_TERRAIN_IDS_HOLES, None), &[])?,
        ];
        let resolve = |fragment| RenderPipeline {
            vertex: draw::VS_TERRAIN_RESOLVE,
            fragment: Some(fragment),
            debug: None,
            vertices: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: DepthUse::Overwrite,
            blend: None,
            writes: wgpu::ColorWrites::ALL,
        };
        let resolve_bound = [(draw::TERRAIN_IDS_TEXTURE, Bind::Texture(h.ids))];
        let resolves = [
            cx.render_pipeline(resolve_pass, &resolve(draw::FS_TERRAIN_RESOLVE_POST), &resolve_bound)?,
            cx.render_pipeline(resolve_pass, &resolve(draw::FS_TERRAIN_RESOLVE_TEMPORAL), &resolve_bound)?,
        ];
        let debugs = [
            cx.render_pipeline(MainView, &surface(debug::FS_DEBUG_TERRAIN, Some(debug::FS_DEBUG_TERRAIN)), &[])?,
            cx.render_pipeline(
                MainView,
                &surface(debug::FS_DEBUG_TERRAIN_HOLES, Some(debug::FS_DEBUG_TERRAIN_HOLES)),
                &[],
            )?,
        ];
        let probe_views = engine_places::probes::ProbeViews;
        let probes = [
            cx.render_pipeline(probe_views, &surface(draw::FS_TERRAIN_PROBE, None), &[])?,
            cx.render_pipeline(
                engine_places::probes::ProbeViews,
                &surface(draw::FS_TERRAIN_PROBE_HOLES, None),
                &[],
            )?,
        ];
        let caster = cx.render_pipeline(
            ShadowCasters,
            &RenderPipeline {
                vertex: shadow::VS_TERRAIN_SHADOW,
                fragment: None,
                debug: None,
                vertices: &[VertexLayout::Buffer(patch_layout())],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::Write { bias: CASTER_BIAS },
                blend: None,
                writes: wgpu::ColorWrites::empty(),
            },
            &[],
        )?;
        let inputs = Inputs {
            args: cx.draw_input(ids_pass, h.args),
            meshlet: cx.draw_input(ids_pass, h.meshlet),
            main_args: cx.draw_input(MainView, h.args),
            main_meshlet: cx.draw_input(MainView, h.meshlet),
            probe_args: cx.draw_input(engine_places::probes::ProbeViews, h.args),
            probe_meshlet: cx.draw_input(engine_places::probes::ProbeViews, h.meshlet),
            patches: cx.draw_input(ShadowCasters, h.patches),
            grid: cx.draw_input(ShadowCasters, h.grid),
            subdivision_args: cx.draw_input(subdivide, h.args),
        };
        let bounce = Bounce::install(cx, (&horizon_maps, &sky_maps, &macro_maps, h.palette), bounce_pass)?;
        let to_array = |list: Vec<ComputePipeline<TerrainGpu>>| -> [ComputePipeline<TerrainGpu>; SURFACE_SLOTS] {
            std::array::from_fn(|slot| list[slot])
        };
        Ok(Self {
            passes: Passes {
                normals: normals_pass,
                control: control_pass,
                clear_counters,
                begin,
                copy_args,
                subdivide,
                bounce: bounce_pass,
                ids: ids_pass,
                resolve: resolve_pass,
            },
            pipelines: Pipelines {
                normals: normals_pipeline,
                control: control_pipeline,
                begin: to_array(begins),
                decide: to_array(decides),
                apply: to_array(applies),
                ids,
                resolve: resolves,
                debug: debugs,
                probe: probes,
                caster,
            },
            inputs,
            atlas: TileAtlas::default(),
            tables: TerrainTables::default(),
            surface: SurfaceTrees::default(),
            patches: PatchDraws::default(),
            horizons: horizon_maps,
            maps,
            sky: sky_maps,
            bounce,
            macro_maps,
            frame: TerrainFrame::default(),
            drawn: false,
            deferred: false,
            debug: false,
            temporal: false,
            render: (0, 0),
            handles,
        })
    }

    fn prepare(
        &mut self,
        (settings, tiles, atlas, bakes, water, cover, shadows, captures, view, time, shown, resolve, cooking, camera, limits): SystemParamItem<
            '_,
            '_,
            Self::Reads,
        >,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.debug = shown.owner.is_some();
        self.temporal = resolve.offset != glam::Vec2::ZERO || resolve.turn != 0.0;
        let size = view.render_size();
        self.render = (size.x, size.y);
        if tiles.terrains.is_empty() && atlas.slots.is_empty() {
            self.drawn = false;
            self.deferred = false;
            return;
        }
        let h = &self.handles;
        let eye = camera.position;
        let culling = view.view_projection() * Mat4::from_translation(eye);
        let jitter = [resolve.offset.x, resolve.offset.y];
        let regions: Vec<(usize, engine_places::shadows::KeptRefresh)> = shadows
            .regions
            .iter()
            .enumerate()
            .filter_map(|(index, region)| match region.refresh {
                Refresh::Kept(kept) => Some((index, kept)),
                Refresh::Redraw => None,
            })
            .collect();
        let canopy = (cover.enabled && cover.density > 0.0).then(|| {
            let height = size.y as f32;
            (*cover, pixels_per_meter(culling, height))
        });
        let views = TerrainViews {
            camera: jittered(culling, jitter),
            culling,
            eye,
            regions,
            settings: &settings,
            water: &water,
            fades: !cooking.cooking,
            now: time.elapsed_seconds as f32,
            cover: canopy,
        };
        self.atlas.upload(cx, h, (&tiles, &atlas, &bakes));
        if self.atlas.normal_jobs() > 0 {
            cx.run(self.passes.normals);
        }
        if self.atlas.control_jobs() > 0 {
            cx.run(self.passes.control);
        }
        self.tables.write(cx, h, (&tiles, &atlas, &self.maps, &self.macro_maps, &self.sky));
        self.maps.prepare(cx, &tiles);
        self.sky.prepare(cx, (&tiles, &bakes, &settings, limits.texture_layers));
        self.macro_maps.prepare(cx, &tiles);
        let (_, slots) = crate::terrain_horizons::room(limits.texture_layers, tiles.terrains.len());
        self.horizons.prepare(cx, (&tiles, &bakes, &settings, shadows.on, limits.texture_layers), slots);
        if self.bounce.prepare(cx, (&tiles, &bakes, &settings, &self.tables), cooking.cooking) {
            cx.run(self.passes.bounce);
        }
        let held = tiles.terrains.iter().any(|entry| atlas.slots.keys().any(|(terrain, _)| *terrain == entry.terrain));
        self.drawn = held;
        if !held {
            self.surface.rest();
            return;
        }
        let drawn: Vec<bool> = tiles
            .terrains
            .iter()
            .map(|entry| atlas.slots.keys().any(|(terrain, _)| *terrain == entry.terrain))
            .collect();
        self.surface.prepare(cx, h, (&tiles, &views, limits.storage_bytes, &drawn));
        self.frame = self.patches.lay_out(cx, h, (&tiles, &atlas, &self.tables, &self.surface, &self.sky, &self.macro_maps), &views);
        self.patches.write_views(cx, h, &views, &captures);
        if self.surface.any() {
            cx.run(self.passes.clear_counters);
            cx.run(self.passes.begin);
            cx.run(self.passes.copy_args);
            cx.run(self.passes.subdivide);
            cx.read_back::<SurfaceTotals>(
                self.surface.water_mask(),
                h.counters,
                self.surface.counter_bytes(),
                engine_render::AfterRead::Keep,
            );
        }
        self.deferred = !self.debug && !self.frame.terrains.is_empty();
        if self.deferred {
            cx.run(self.passes.ids);
            cx.run(self.passes.resolve);
        }
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        if pass == self.passes.normals {
            return cx.dispatch(&self.pipelines.normals, Count::Groups(self.atlas.normal_groups()));
        }
        if pass == self.passes.control {
            return cx.dispatch(&self.pipelines.control, Count::Groups(self.atlas.control_groups()));
        }
        if pass == self.passes.begin {
            let mut dispatched = 0;
            for slot in self.surface.slots() {
                dispatched += cx.dispatch(&self.pipelines.begin[slot], Count::Groups([1, 1, 1]));
            }
            return dispatched;
        }
        if pass == self.passes.subdivide {
            let mut dispatched = 0;
            for slot in self.surface.slots() {
                let offset = self.surface.args_offset(slot);
                dispatched += cx.dispatch(&self.pipelines.decide[slot], Count::Indirect(self.inputs.subdivision_args, offset));
                dispatched += cx.dispatch(&self.pipelines.apply[slot], Count::Indirect(self.inputs.subdivision_args, offset));
            }
            return dispatched;
        }
        if pass == self.passes.bounce {
            return self.bounce.encode(cx);
        }
        if pass == self.passes.ids {
            let mut draw = cx.render(Attach::Color(self.handles.ids, None), None);
            return self.draw_ids(&mut draw);
        }
        if pass == self.passes.resolve {
            let mut draw = cx.render(Attach::Declared, None);
            return self.draw_resolve(&mut draw);
        }
        0
    }
}

impl TerrainGpu {
    fn draw_ids(&self, draw: &mut Draw<'_, Self, Own>) -> u32 {
        let mut draws = 0;
        for holes in [false, true] {
            for terrain in self.frame.terrains.iter().filter(|terrain| terrain.holes == holes) {
                let Some(offset) = self.surface.draw_offset(terrain.terrain) else {
                    continue;
                };
                draw.draw(
                    &self.pipelines.ids[usize::from(holes)],
                    Shape::Indirect {
                        vertices: &[],
                        indices: Some(self.inputs.meshlet),
                        args: self.inputs.args,
                        offset,
                    },
                );
                draws += 1;
            }
        }
        draws
    }

    fn draw_resolve(&self, draw: &mut Draw<'_, Self, Own>) -> u32 {
        let pipeline = &self.pipelines.resolve[usize::from(self.temporal)];
        let mut draws = 0;
        for terrain in &self.frame.terrains {
            let Some(rect) = terrain.scissor(self.render) else {
                continue;
            };
            draw.scissor(rect);
            let index = terrain.terrain as u32;
            draw.draw(
                pipeline,
                Shape::Pulled {
                    vertices: 0..3,
                    instances: index..index + 1,
                },
            );
            draws += 1;
        }
        draws
    }

    pub(crate) fn draw_main(&self, _: &(), draw: &mut Draw<'_, Self, MainView>) -> u32 {
        if !(self.debug && self.drawn) {
            return 0;
        }
        let mut draws = 0;
        for holes in [false, true] {
            for terrain in self.frame.terrains.iter().filter(|terrain| terrain.holes == holes) {
                let Some(offset) = self.surface.draw_offset(terrain.terrain) else {
                    continue;
                };
                draw.draw(
                    &self.pipelines.debug[usize::from(holes)],
                    Shape::Indirect {
                        vertices: &[],
                        indices: Some(self.inputs.main_meshlet),
                        args: self.inputs.main_args,
                        offset,
                    },
                );
                draws += 1;
            }
        }
        draws
    }

    pub(crate) fn draw_casters(&self, at: &CasterView, draw: &mut Draw<'_, Self, ShadowCasters>) -> u32 {
        if !self.drawn {
            return 0;
        }
        let Some(range) = self.frame.map_records(at.region as usize) else {
            return 0;
        };
        draw.draw(
            &self.pipelines.caster,
            Shape::Buffers {
                vertices: &[self.inputs.patches],
                indices: Some(self.inputs.grid),
                elements: 0..self.patches.index_count(),
                instances: range,
            },
        );
        1
    }

    pub(crate) fn draw_probe(&self, _: &ProbeView, draw: &mut Draw<'_, Self, engine_places::probes::ProbeViews>) -> u32 {
        if self.frame.terrains.is_empty() {
            return 0;
        }
        let mut draws = 0;
        for holes in [false, true] {
            for terrain in self.frame.terrains.iter().filter(|terrain| terrain.holes == holes) {
                let Some(offset) = self.surface.draw_offset(terrain.terrain) else {
                    continue;
                };
                draw.draw(
                    &self.pipelines.probe[usize::from(holes)],
                    Shape::Indirect {
                        vertices: &[],
                        indices: Some(self.inputs.probe_meshlet),
                        args: self.inputs.probe_args,
                        offset,
                    },
                );
                draws += 1;
            }
        }
        draws
    }

    pub(crate) fn owns(shown: &DebugShown) -> bool {
        shown.owner == Some(TypeId::of::<TerrainShadingBrick>())
    }
}

fn pixels_per_meter(culling: Mat4, height: f32) -> f32 {
    culling.row(1).truncate().length() * height * 0.5
}

fn jittered(view_projection: Mat4, jitter: [f32; 2]) -> Mat4 {
    Mat4::from_cols(
        glam::Vec4::X,
        glam::Vec4::Y,
        glam::Vec4::Z,
        glam::Vec4::new(jitter[0], jitter[1], 0.0, 1.0),
    ) * view_projection
}
