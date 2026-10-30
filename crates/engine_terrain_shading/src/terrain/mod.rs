use engine_core::collections::HashMap;
use std::ops::Range;

use bytemuck::Zeroable;
use engine_core::glam::{DVec3, Mat4, Vec3, Vec4};
use engine_terrain::gpu::SurfaceData;
use engine_terrain::{Pyramid, RenderTerrain, RenderTerrains, TileKey};
use std::sync::Arc;

use engine_faces::heightfield::{
    CoverEntry, CoverSources, CoverTerms, GroundChange, HEIGHT_FORMAT, HeightfieldFace, LiveGround,
    PALETTE_BYTES, PaletteEntry, TERRAIN, TerrainInfo, TerrainUniform, TraceSources, ViewUniform,
    WaterDraws,
};
use engine_faces::shadows::KeptRefresh;
use engine_render::bindings::{Contribution, Sample, Sampling, Stages, View, sampler, texture};
use engine_render::cached::{Bound, Cached};
use engine_render::gpu_calls::Queue;
use engine_render::graph::{Lifetime, PassDecl, ResourceDecl, Stage, Targets};
use engine_render::mipmaps::MipGenerator;
use engine_render::parts::{Hook, OpaqueSlot, PartDecl, PartEntry};
use engine_render::pieces::PieceDecl;
use engine_render::targets::ScreenTarget;

use crate::bindings::{TERRAIN_CONTROL, TERRAIN_NORMALS};
use crate::gpu_layouts::constants::NOT_RESIDENT;
use crate::gpu_layouts::constants::PALETTE_ENTRIES;
use crate::gpu_layouts::constants::TERRAIN_GRID as GRID;
use crate::gpu_layouts::constants::TERRAIN_GRID_VERTICES as GRID_VERTICES;
use crate::gpu_layouts::constants::{CONTROL_GROUP, NORMALS_GROUP};
use crate::terrain_bounce::{Bounce, SceneSources, TerrainSources};
use crate::terrain_horizons::Horizons;
use crate::terrain_macro::MacroMaps;
use crate::terrain_maps::TerrainMaps;
use crate::terrain_sky::SkyMaps;

mod atlas;
#[cfg(debug_assertions)]
mod invariants;
mod part;
mod patches;
mod pipelines;
mod surface;
mod tables;
mod water_blocks;

use atlas::TileAtlas;
use part::{Slots, TerrainIds};
pub(crate) use patches::{PatchDraws, planes};
use pipelines::{GroupSources, TerrainPipelines};
use surface::SurfaceTrees;
use tables::TerrainTables;

use crate::gpu_layouts::terrain::HorizonsPiece;
pub(crate) use crate::gpu_layouts::terrain::PatchRecord;

pub(crate) static FRAME_HORIZONS: Contribution = Contribution {
    group: "frame",
    set: "horizons",
    bindings: &[
        texture(
            "terrain_horizons",
            Sample::Float,
            View::D2Array,
            Stages::FRAGMENT,
        ),
        sampler(
            "terrain_horizon_sampler",
            Sampling::Filtering,
            Stages::FRAGMENT,
        ),
    ],
};

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "terrain",
        brick: "terrain_shading",
        passes: &[
            PassDecl {
                name: "subdivision",
                stage: Stage::Prepare,
            },
            PassDecl {
                name: "bounce",
                stage: Stage::Prepare,
            },
            PassDecl {
                name: "terrain_ids",
                stage: Stage::Deferred,
            },
            PassDecl {
                name: "terrain_resolve",
                stage: Stage::Deferred,
            },
        ],
        resources: &[
            ResourceDecl {
                name: "leaves",
                lifetime: Lifetime::Kept,
            },
            ResourceDecl {
                name: "bounce",
                lifetime: Lifetime::Kept,
            },
            ResourceDecl {
                name: "terrain_ids",
                lifetime: Lifetime::Frame,
            },
        ],
        hooks: &[
            Hook::Opaque(OpaqueSlot::Ground),
            Hook::ShadowCasters,
            Hook::View,
        ],
        contributes: std::slice::from_ref(&FRAME_HORIZONS),
        effects: &[],
        labels: &[],
        lazy: false,
        faces: &[engine_render::offer_face!(TerrainPass => dyn HeightfieldFace)],
        needs: &[],
        piece: Some(PieceDecl {
            name: "horizons",
            module: "terrain::piece",
            ty: "HorizonsPiece",
            size: size_of::<HorizonsPiece>() as u64,
        }),
    },
    create: part::create,
};

const LAYER_SIZE: u32 = engine_terrain::TILE_STRIDE;
const SLOT: u64 = 256;
const VIEWS: u64 = 7;
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const CONTROL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const RIVER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Uint;

const ARRIVAL_SECONDS: f32 = 0.4;

const NO_HOLES: u32 = u32::MAX;
const HOLE_WORDS: usize = ((LAYER_SIZE - 2) * (LAYER_SIZE - 2)).div_ceil(32) as usize;

fn widen(range: Range<usize>, index: usize) -> Range<usize> {
    if range.is_empty() {
        index..index + 1
    } else {
        range.start.min(index)..range.end.max(index + 1)
    }
}

fn changed_run<T: bytemuck::Pod>(was: &[T], now: &[T]) -> Range<usize> {
    let same = |a: &T, b: &T| bytemuck::bytes_of(a) == bytemuck::bytes_of(b);
    let shared = now.len().min(was.len());
    let Some(first) = (0..now.len()).find(|&at| at >= shared || !same(&now[at], &was[at])) else {
        return 0..0;
    };
    let end = if now.len() > was.len() {
        now.len()
    } else {
        (first..now.len())
            .rev()
            .find(|&at| !same(&now[at], &was[at]))
            .map_or(first + 1, |at| at + 1)
    };
    first..end
}

pub(crate) struct FrameTerrain {
    pub terrain: usize,
    pub uniform: u32,
    pub holes: bool,
    pub maps: Vec<(usize, Range<u32>)>,
    pub screen: Option<[f32; 4]>,
}

impl FrameTerrain {
    pub(crate) fn scissor(&self, size: (u32, u32)) -> Option<[u32; 4]> {
        let (width, height) = (size.0 as f32, size.1 as f32);
        let Some([left, bottom, right, top]) = self.screen else {
            return (size.0 > 0 && size.1 > 0).then_some([0, 0, size.0, size.1]);
        };
        let pixel = |value: f32, extent: f32| engine_core::math::clamp(value, 0.0, extent) as u32;
        let x0 = pixel(((left * 0.5 + 0.5) * width).floor() - 1.0, width);
        let x1 = pixel(((right * 0.5 + 0.5) * width).ceil() + 1.0, width);
        let y0 = pixel(((0.5 - top * 0.5) * height).floor() - 1.0, height);
        let y1 = pixel(((0.5 - bottom * 0.5) * height).ceil() + 1.0, height);
        (x1 > x0 && y1 > y0).then_some([x0, y0, x1 - x0, y1 - y0])
    }
}

#[derive(Default)]
pub(crate) struct TerrainFrame {
    pub terrains: Vec<FrameTerrain>,
    pub shadow_patches: u32,
}

pub(crate) struct TerrainViews<'a> {
    pub camera: Mat4,
    pub culling: Mat4,
    pub eye: Vec3,
    pub regions: &'a [(usize, KeptRefresh)],
    pub settings: engine_terrain::TerrainSettings,
    pub water: engine_terrain::Water,
    pub fades: bool,
    pub now: f32,
    pub cover: Option<(engine_terrain::GroundCover, f32)>,
}

pub(crate) struct TerrainPass {
    pub(crate) atlas: TileAtlas,
    pub(crate) tables: TerrainTables,
    pub(crate) surface: SurfaceTrees,
    pub(crate) patches: PatchDraws,
    pub(crate) pipelines: TerrainPipelines,
    pub(crate) horizons: Horizons,
    pub(crate) maps: TerrainMaps,
    pub(crate) sky: SkyMaps,
    pub(crate) bounce: Bounce,
    pub(crate) cooking: bool,
    pub(crate) macro_maps: MacroMaps,
    pub(crate) water: water_blocks::WaterBlocks,
    ids: ScreenTarget<TerrainIds>,
    slots: Slots,
    drawn: bool,
    deferred: bool,
}

impl TerrainPass {
    pub(crate) fn new(device: &wgpu::Device, queue: &Queue) -> Self {
        let _owner = engine_render::memory::owned(PART.decl.brick);
        Self {
            atlas: TileAtlas::new(device),
            tables: TerrainTables::new(device),
            surface: SurfaceTrees::new(device),
            patches: PatchDraws::new(device),
            pipelines: TerrainPipelines::new(device),
            horizons: Horizons::new(device),
            maps: TerrainMaps::new(device, queue),
            sky: SkyMaps::new(device, queue),
            bounce: Bounce::new(device, queue),
            cooking: false,
            macro_maps: MacroMaps::new(device, queue),
            water: water_blocks::WaterBlocks::new(device),
            ids: ScreenTarget::default(),
            slots: Slots::default(),
            drawn: false,
            deferred: false,
        }
    }

    pub(crate) fn eye(&self) -> Vec3 {
        self.patches.eye()
    }

    pub(crate) fn resident(&self) -> u32 {
        self.atlas.slots.len() as u32
    }

    pub(crate) fn drawn(&self) -> bool {
        self.drawn
    }

    pub(crate) fn water_draws(&self, terrains: &RenderTerrains) -> Option<WaterDraws<'_>> {
        self.water.draws(self, terrains)
    }

    pub(crate) fn ground(&self, id: u64) -> Option<LiveGround<'_>> {
        Some(LiveGround {
            group: self.pipelines.surface_group()?,
            uniform: self.tables.uniform_of(id)?,
            terrain: id,
        })
    }

    pub(crate) fn holds(&self, id: u64, key: TileKey, tile: *const ()) -> bool {
        self.atlas
            .held
            .get(&(id, key))
            .is_some_and(|held| std::ptr::eq(Arc::as_ptr(held).cast::<()>(), tile))
    }

    fn place_horizons(
        &mut self,
        (device, queue): (&wgpu::Device, &Queue),
        terrains: &RenderTerrains,
        eye: Vec3,
        (settings, shadows): (&engine_terrain::TerrainSettings, bool),
    ) {
        let room = crate::terrain_horizons::room(device, terrains.terrains.len());
        self.bounce.place(terrains, (eye, settings), room);
        let windows = self.bounce.windows();
        (self.horizons).prepare(device, queue, terrains, (settings, shadows), windows);
    }

    fn prepare_terrains(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        terrains: &RenderTerrains,
        views: &TerrainViews<'_>,
        (max_layers, mips): (u32, &MipGenerator),
    ) -> bool {
        self.atlas.begin(views.now);
        self.patches.begin(views.eye);
        self.atlas.follow_materials(device, terrains);
        if let Some(edited) = self.atlas.sync(device, queue, terrains, max_layers) {
            self.tables
                .write_holes(device, queue, (&self.atlas, edited), max_layers);
        }
        self.water.sync(device, queue, &self.atlas, max_layers);
        self.patches.follow_ground(terrains, &self.atlas);
        self.atlas.fade(views.fades, self.tables.fade_words);
        let fades = (&self.atlas.fades[..], self.atlas.fades_dirty.clone());
        self.tables.write_fades(queue, fades);
        self.tables
            .write_directory(device, queue, terrains, &self.atlas);
        let tables = &self.tables;
        self.water.write_directory(
            queue,
            &self.atlas,
            (
                &tables.directory,
                tables.water_base,
                tables.directory_generation,
            ),
        );
        self.maps.prepare(device, queue, mips, terrains);
        self.tables.write_palette(queue, terrains, &self.maps);
        self.sky.prepare(device, queue, terrains, &views.settings);
        self.macro_maps.prepare(device, queue, terrains);
        let lighting = (
            &self.horizons,
            &self.sky,
            &self.macro_maps,
            &self.tables.palette_bases[..],
        );
        self.bounce
            .prepare(device, queue, terrains, lighting, self.cooking);
        if !terrains.terrains.iter().any(|terrain| terrain.drawn) {
            self.rest_surface(device);
            return false;
        }
        self.tables.size_uniforms(device, terrains);
        let water = &self.water;
        self.surface
            .prepare(device, queue, terrains, views, |id| water.holds(id));
        let parts = patches::Layout {
            atlas: &self.atlas,
            tables: &self.tables,
            surface: &self.surface,
            sky: &self.sky,
            macro_maps: &self.macro_maps,
        };
        self.patches.lay_out(device, terrains, views, &parts);
        let uniforms = (&self.tables.uniforms, self.tables.uniform_generation);
        self.patches.write(queue, views, uniforms);
        self.pipelines.ensure_group(
            device,
            &GroupSources {
                atlas: &self.atlas,
                tables: &self.tables,
                patches: &self.patches,
                surface: &self.surface,
                maps: &self.maps,
                sky: &self.sky,
                macro_maps: &self.macro_maps,
                water: &self.water,
            },
        );
        #[cfg(debug_assertions)]
        self.check(terrains);
        true
    }

    pub(crate) fn write_probe_view(
        &self,
        queue: &Queue,
        face: usize,
        relative: Mat4,
        eye: Vec3,
        center: Vec3,
    ) -> u32 {
        self.patches
            .write_probe_view(queue, face, relative, eye, center)
    }

    pub(crate) fn warm(&mut self, device: &wgpu::Device) {
        self.atlas.warm(device);
    }

    pub(crate) fn cover_sources(&self) -> Option<CoverSources<'_>> {
        let (heights, normals, control) = self.atlas.views()?;
        let tables = &self.tables;
        Some(CoverSources {
            heights,
            control,
            normals,
            directory: &tables.directory,
            palette: &tables.palette,
            covers: &tables.covers,
            macro_maps: self.macro_maps.view(),
            generation: (
                self.atlas.generation,
                tables.directory_generation,
                self.macro_maps.generation(),
            ),
            revision: tables.directory_revision,
        })
    }

    pub(crate) fn trace_sources(&self) -> TraceSources<'_> {
        let generation = self.atlas.generation;
        let arrays = self.atlas.views().map(|(heights, _, control)| {
            [heights, control]
                .map(|view| Bound::counted(generation, engine_render::pipelines::view(view)))
        });
        TraceSources {
            arrays,
            directory: self.tables.directory.bound(),
            palette: Bound::fixed(self.tables.palette.as_entire_binding()),
            macro_maps: Bound::counted(
                self.macro_maps.generation(),
                engine_render::pipelines::view(self.macro_maps.view()),
            ),
            sky: Bound::counted(
                self.sky.generation(),
                engine_render::pipelines::view(self.sky.view()),
            ),
        }
    }

    pub(crate) fn layer_of(&self, id: u64, key: TileKey) -> Option<u32> {
        self.atlas.slots.get(&(id, key)).copied()
    }

    pub(crate) fn highest(&self, id: u64) -> Option<f32> {
        self.atlas
            .tops
            .iter()
            .filter(|((terrain, _), _)| *terrain == id)
            .map(|(_, &top)| top)
            .reduce(engine_core::math::max)
    }

    pub(crate) fn layers(&self) -> (u32, u64) {
        (self.atlas.layers(), self.tables.directory_revision)
    }

    pub(crate) fn cover_tallest(&self) -> f32 {
        self.tables.cover_tallest
    }

    pub(crate) fn cover_terrain(&self, index: usize) -> (u32, u32, u32) {
        let tables = &self.tables;
        (
            tables.directory_bases.get(index).copied().unwrap_or(0),
            tables.palette_bases.get(index).copied().unwrap_or(0),
            self.macro_maps.layer(index),
        )
    }

    pub(crate) fn cover_terms(&self, index: usize) -> CoverTerms {
        self.tables
            .cover_terms
            .get(index)
            .copied()
            .unwrap_or_default()
    }

    pub(crate) fn terrain_info(
        &self,
        index: usize,
        origin: Vec3,
        pyramid: &Pyramid,
    ) -> TerrainInfo {
        let (directory, palette, colors) = self.cover_terrain(index);
        TerrainInfo {
            origin: origin.extend(pyramid.spacing(0)).to_array(),
            levels: [directory, pyramid.levels(), palette, colors],
        }
    }

    pub(crate) fn draw_surface(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        terrain: &FrameTerrain,
        view: u32,
    ) -> u32 {
        let (Some(group), Some(leaves)) =
            (self.pipelines.group(), self.surface.draw(terrain.terrain))
        else {
            return 0;
        };
        pass.set_bind_group(1, group, &[view, terrain.uniform]);
        leaves.record(pass);
        1
    }

    pub(crate) fn surface_leaves(&self) -> Option<(&wgpu::Buffer, u64)> {
        let leaves = self.surface.leaves();
        Some((leaves, leaves.generation()))
    }

    pub(crate) fn rest(&mut self, device: &wgpu::Device) {
        self.bounce.rest(device);
        self.rest_surface(device);
    }

    fn rest_surface(&mut self, device: &wgpu::Device) {
        self.surface.rest(device);
        self.pipelines.release_group();
    }

    pub(crate) fn subdivides(&self) -> bool {
        self.surface.any()
    }

    pub(crate) fn draw_shadow(&self, pass: &mut wgpu::RenderPass<'_>, map: usize) -> u32 {
        let (Some(pipeline), Some(group)) =
            (self.pipelines.shadow_pipeline(), self.pipelines.group())
        else {
            return 0;
        };
        pass.set_pipeline(pipeline);
        let mut draws = 0;
        for terrain in &self.patches.frame.terrains {
            if let Some((_, range)) = terrain.maps.iter().find(|(drawn, _)| *drawn == map) {
                draws += self.patches.draw(pass, group, range, (0, terrain.uniform));
            }
        }
        draws
    }

    fn encode_subdivision(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        camera: &wgpu::Buffer,
        profiler: &engine_render::profiling::GpuProfiler,
    ) {
        let Some(group) = self.pipelines.surface_group() else {
            return;
        };
        self.surface
            .encode(device, encoder, camera, group, profiler);
    }

    fn encode_bounce(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &engine_render::profiling::GpuProfiler,
        scene: SceneSources<'_>,
    ) {
        let terrain = TerrainSources {
            horizons: &self.horizons,
            sky: &self.sky,
            colors: &self.macro_maps,
            palette: &self.tables.palette,
        };
        self.bounce
            .encode(device, encoder, profiler, (terrain, scene));
    }

    pub(crate) fn bind(&self, pass: &mut wgpu::RenderPass<'_>, view: u32, uniform: u32) -> bool {
        let Some(group) = self.pipelines.group() else {
            return false;
        };
        pass.set_bind_group(1, group, &[view, uniform]);
        true
    }

    pub(crate) fn has_holes(&self) -> bool {
        self.patches
            .frame
            .terrains
            .iter()
            .any(|terrain| terrain.holes)
    }

    pub(crate) fn ground_changes(&self) -> &[GroundChange] {
        self.patches.ground_changes()
    }
}

impl HeightfieldFace for TerrainPass {
    fn drawn(&self) -> bool {
        self.drawn()
    }

    fn sky_layer(&self, index: usize) -> u32 {
        self.sky.layer(index)
    }

    fn ground_changes(&self) -> &[GroundChange] {
        self.ground_changes()
    }

    fn cover_sources(&self) -> Option<CoverSources<'_>> {
        self.cover_sources()
    }

    fn cover_terrain(&self, index: usize) -> (u32, u32, u32) {
        self.cover_terrain(index)
    }

    fn cover_terms(&self, index: usize) -> CoverTerms {
        self.cover_terms(index)
    }

    fn terrain_info(&self, index: usize, origin: Vec3, pyramid: &Pyramid) -> TerrainInfo {
        self.terrain_info(index, origin, pyramid)
    }

    fn cover_tallest(&self) -> f32 {
        self.cover_tallest()
    }

    fn trace_sources(&self) -> TraceSources<'_> {
        self.trace_sources()
    }

    fn ground(&self, terrain: u64) -> Option<LiveGround<'_>> {
        self.ground(terrain)
    }

    fn holds(&self, terrain: u64, key: TileKey, tile: *const ()) -> bool {
        self.holds(terrain, key, tile)
    }

    fn water_draws(&self, snapshot: &engine_core::RenderSnapshot) -> Option<WaterDraws<'_>> {
        self.water_draws(RenderTerrains::of(snapshot))
    }

    fn layer_of(&self, terrain: u64, key: TileKey) -> Option<u32> {
        self.layer_of(terrain, key)
    }

    fn layers(&self) -> (u32, u64) {
        self.layers()
    }

    fn highest(&self, terrain: u64) -> Option<f32> {
        self.highest(terrain)
    }
}
