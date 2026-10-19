//! Terrain: every resident tile's heights in one texture
//! array, the camera's patches as instances of one shared grid with skirts,
//! and their pipelines in the opaque passes, the shadow maps drawn every
//! frame and reflection probe faces.
//!
//! Tiles upload as frames list them and release when they leave the list;
//! the array grows by doubling up to the device's layers, beyond which
//! tiles drop (counted) and their patches do not draw. Each view culls the
//! camera's patches by their bounds: the camera's frustum, each shadow
//! map's sides. Every view draws the same vertices, at the positions the
//! camera sees, so terrain shadows itself without acne. Where terrains cast
//! long shadows through their horizon maps, only the near
//! cascades draw terrain, near their receivers.
//!
//! Each tile's normals are computed once on upload into a parallel array;
//! fragments read them through a directory of every level's resident tiles,
//! from the finest level loaded at their distance whatever level the
//! geometry draws at, so light and materials hold still as levels of
//! detail hand over.
//!
//! The pass is its parts: the tile atlas (`atlas`), the tables shaders
//! look up (`tables`), the frame's patches (`patches`) and the pipelines
//! and terrain group (`pipelines`), beside tiles' water (`water_blocks`),
//! the material maps, sky maps, horizon maps, runtime virtual texture and
//! deformation map. Debug builds
//! check what holds between them after every preparation (`invariants`).

use engine_core::collections::HashMap;
use std::ops::Range;

use bytemuck::Zeroable;
use engine_core::glam::{DVec3, Mat4, Vec3, Vec4};
use engine_core::terrain_data::gpu::SurfaceData;
use engine_core::{RenderTerrain, RenderTerrains, TileKey};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::bindings::{SHADOW_VIEW, TERRAIN, TERRAIN_CONTROL, TERRAIN_NORMALS};
use crate::cached::Cached;
use crate::gpu_calls::Queue;
use crate::gpu_layouts::constants::NOT_RESIDENT;
use crate::gpu_layouts::constants::TERRAIN_GRID as GRID;
use crate::gpu_layouts::constants::TERRAIN_GRID_VERTICES as GRID_VERTICES;
use crate::gpu_layouts::constants::{CONTROL_GROUP, NORMALS_GROUP};
use crate::gpu_layouts::terrain::{
    CoverEntry, PALETTE_BYTES, PALETTE_ENTRIES, PaletteEntry, TerrainUniform, ViewUniform,
};
use crate::mipmaps::MipGenerator;
use crate::render_graph::Targets;
use crate::terrain_horizons::Horizons;
use crate::terrain_maps::TerrainMaps;
use crate::terrain_sky::SkyMaps;

mod atlas;
#[cfg(debug_assertions)]
mod invariants;
mod patches;
mod pipelines;
mod tables;
mod water_blocks;

use atlas::TileAtlas;
pub(crate) use patches::{PatchDraws, planes, visible};
use pipelines::{GroupSources, TerrainPipelines};
use tables::TerrainTables;
pub(crate) use water_blocks::{GRID_INDICES as WATER_GRID_INDICES, water_patch_layout};

pub(crate) use crate::gpu_layouts::terrain::PatchRecord;

/// Triangles a patch draws: its grid's and its skirts'.
pub(crate) const PATCH_TRIANGLES: u32 = 2 * (GRID - 1) * (GRID - 1) + 8 * (GRID - 1);
/// Texels along each side of a layer: a tile's samples with their border.
const LAYER_SIZE: u32 = engine_core::terrain_data::TILE_STRIDE;
/// Uniform offsets of the views and terrains, at the WebGPU alignment.
const SLOT: u64 = 256;
/// The camera's view, then a reflection probe's six faces.
const VIEWS: u64 = 7;
pub(crate) const HEIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Each tile's normals, filtered.
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Snorm;
/// Each tile's control values: its samples' two strongest
/// materials, their blend and biome, read texel by texel.
const CONTROL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Seconds over which a tile that arrived grows out of its parent level,
/// its patches drawn morphed into the parent's at first, and
/// its normals and surface materials take over from the coarser level's:
/// the same time however fast frames come.
const ARRIVAL_SECONDS: f32 = 0.4;

/// `PatchRecord::holes` of tiles without holes.
const NO_HOLES: u32 = u32::MAX;
/// 32-bit words of a tile's hole bits.
const HOLE_WORDS: usize = ((LAYER_SIZE - 2) * (LAYER_SIZE - 2)).div_ceil(32) as usize;

/// Bytes of a terrain's uniform.
pub(crate) const TERRAIN_UNIFORM_BYTES: usize = size_of::<TerrainUniform>();

/// What water's pass draws tiles' water with: the height array, the views'
/// and terrains' uniforms, the water blocks' arrays, generations advancing
/// when any of them is another texture or buffer, this frame's patches
/// holding water and each terrain's uniform offset and range of them, the
/// grid's indices, and the blocks resident and dropped this frame for want
/// of layers.
pub(crate) struct TerrainWaterDraws<'a> {
    pub(crate) heights: &'a wgpu::TextureView,
    pub(crate) views: &'a wgpu::Buffer,
    pub(crate) uniforms: &'a wgpu::Buffer,
    pub(crate) surfaces: &'a wgpu::TextureView,
    pub(crate) info: &'a wgpu::TextureView,
    pub(crate) generation: (u64, u64, u64),
    pub(crate) instances: &'a wgpu::Buffer,
    pub(crate) indices: &'a wgpu::Buffer,
    pub(crate) terrains: &'a [(u32, Range<u32>)],
    pub(crate) blocks: [u32; 2],
}

/// What ground cover's growth reads of the terrain pass.
pub(crate) struct CoverSources<'a> {
    pub heights: &'a wgpu::TextureView,
    pub control: &'a wgpu::TextureView,
    pub directory: &'a wgpu::Buffer,
    pub palette: &'a wgpu::Buffer,
    pub covers: &'a wgpu::Buffer,
    /// Runtime deformation's map and its shading inputs.
    pub deformation: &'a wgpu::Buffer,
    pub shading: crate::deformation::DeformationShading,
    /// The terrains' sky maps, whose large-scale variety tints blades as
    /// it does the ground.
    pub sky: &'a wgpu::TextureView,
    pub generation: (u64, u64, u64, u64),
    /// Advances whenever the directory's resident tiles change.
    pub revision: u64,
}

/// One terrain's draws this frame: its uniform slot, the camera's patches,
/// every patch (for probe faces) and each shadow map's. The camera's and
/// every patch's ranges end with the patches with holes, which draw with
/// their own pipelines: as many as `*_holes`. And the rectangle of the
/// screen the camera's patches cover, in normalized device coordinates
/// (least x and y, then greatest; `None` where it is unbounded), which its
/// resolve shades within.
pub(crate) struct FrameTerrain {
    pub uniform: u32,
    pub camera: Range<u32>,
    pub camera_holes: u32,
    pub all: Range<u32>,
    pub all_holes: u32,
    pub maps: Vec<(usize, Range<u32>)>,
    pub screen: Option<[f32; 4]>,
}

impl FrameTerrain {
    /// `range` without and with holes.
    pub(crate) fn split(range: &Range<u32>, holes: u32) -> [Range<u32>; 2] {
        let middle = range.end - holes;
        [range.start..middle, middle..range.end]
    }

    /// The pixels of a `size` target its resolve shades: its screen
    /// rectangle, widened a pixel for rasterization's rounding (x, y, width
    /// and height); `None` where it covers none.
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

/// What a frame draws of every terrain.
#[derive(Default)]
pub(crate) struct TerrainFrame {
    pub terrains: Vec<FrameTerrain>,
    /// Patches the camera draws and their triangles, and patches drawn into
    /// shadow maps.
    pub patches: u32,
    pub triangles: u32,
    pub shadow_patches: u32,
    /// Terrain shows shaded, not a development view: shading may defer.
    pub shaded: bool,
}

/// What `TerrainPass::prepare` needs of the frame's views.
pub(crate) struct TerrainViews<'a> {
    /// The camera's projection of eye-relative positions, jittered on
    /// temporally antialiased frames.
    pub camera: Mat4,
    /// The camera's unjittered projection of eye-relative positions, for
    /// culling.
    pub culling: Mat4,
    pub eye: Vec3,
    /// Shadow maps drawn every frame and their world-to-clip matrices.
    pub maps: &'a [(usize, Mat4)],
    /// Whether reflection probes capture faces this frame, which draw every
    /// patch.
    pub probes: bool,
    /// Terrain settings, whether shadows run, and the direction toward the
    /// sun, for horizon maps.
    pub settings: engine_core::TerrainSettings,
    pub shadows: bool,
    pub sun: Vec3,
    /// Whether tiles that arrive grow out of their parents over a few frames:
    /// ordinary frames; exact ones draw every tile as it
    /// is. And the frame's presentation time in seconds, which their fades
    /// follow.
    pub fades: bool,
    pub now: f32,
    /// Ground cover's settings, sanitized, and the pixels a meter spans a
    /// meter from the eye, while cover grows: where blades thin out, the
    /// ground shows as its canopy.
    pub cover: Option<(engine_core::GroundCover, f32)>,
}

pub(crate) struct TerrainPass {
    /// Resident tiles' arrays and layers.
    pub(crate) atlas: TileAtlas,
    /// What shaders look up: fades and holes, the directory, palette and
    /// covers, and each terrain's uniform.
    pub(crate) tables: TerrainTables,
    /// This frame's patches and views.
    pub(crate) patches: PatchDraws,
    /// Pipelines and the terrain group.
    pub(crate) pipelines: TerrainPipelines,
    /// Long sun shadows.
    pub(crate) horizons: Horizons,
    /// Surface materials' maps.
    pub(crate) maps: TerrainMaps,
    /// Sky visibility.
    pub(crate) sky: SkyMaps,
    /// Runtime deformation's map.
    pub(crate) deformation: crate::deformation::DeformationPass,
    /// The runtime virtual texture.
    pub(crate) vt: crate::terrain_vt::VirtualTexture,
    /// Tiles' water.
    pub(crate) water: water_blocks::WaterBlocks,
}

impl TerrainPass {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            atlas: TileAtlas::new(device),
            tables: TerrainTables::new(device),
            patches: PatchDraws::new(device),
            pipelines: TerrainPipelines::new(device),
            horizons: Horizons::new(device),
            maps: TerrainMaps::new(device, queue),
            sky: SkyMaps::new(device, queue),
            deformation: crate::deformation::DeformationPass::new(device),
            vt: crate::terrain_vt::VirtualTexture::new(device),
            water: water_blocks::WaterBlocks::new(device),
        }
    }

    /// The eye this frame's terrain positions are relative to.
    pub(crate) fn eye(&self) -> Vec3 {
        self.patches.eye()
    }

    /// Layers in use.
    pub(crate) fn resident(&self) -> u32 {
        self.atlas.slots.len() as u32
    }

    /// Bytes of the height, normal and control arrays, the horizon and sky
    /// maps and the surface materials' maps.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.atlas.capacity_bytes()
            + self.horizons.capacity_bytes()
            + self.sky.capacity_bytes()
            + self.maps.capacity_bytes()
            + self.deformation.capacity_bytes()
            + self.vt.capacity_bytes()
            + self.water.capacity_bytes()
    }

    /// What water's pass draws tiles' water with; `None` before tiles and
    /// their water arrive.
    pub(crate) fn water_draws(&self) -> Option<TerrainWaterDraws<'_>> {
        let heights = self.atlas.heights()?;
        let (surfaces, info) = self.water.views()?;
        Some(TerrainWaterDraws {
            heights,
            views: &self.patches.views,
            uniforms: &self.tables.uniforms,
            surfaces,
            info,
            generation: (
                self.atlas.generation,
                self.tables.uniform_generation,
                self.water.generation,
            ),
            instances: &self.water.instances,
            indices: self.patches.indices(),
            terrains: &self.water.terrains,
            blocks: [self.water.resident(), self.water.dropped],
        })
    }

    /// Records this frame's horizon map builds.
    pub(crate) fn encode_horizons(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        self.horizons
            .encode(device, encoder, self.atlas.heights(), profiler);
    }

    /// Uploads tiles the frame's terrains list and the GPU lacks, releases
    /// those they no longer list, and lays out this frame's patches for the
    /// camera, reflection probe faces and the shadow maps of `views`.
    /// `max_layers` is the device's texture array layers; `mips` renders
    /// surface materials' mip levels. Returns whether any terrain draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        terrains: &RenderTerrains,
        views: &TerrainViews<'_>,
        (max_layers, mips): (u32, &MipGenerator),
    ) -> bool {
        let atlas = &mut self.atlas;
        (atlas.uploads, atlas.controls, atlas.now) = (0, 0, views.now);
        self.patches.begin(terrains, views.eye);
        atlas.changed.clear();
        if let Some(synced) = atlas.sync(device, queue, terrains, max_layers) {
            self.tables.write_holes(device, queue, &synced, max_layers);
        }
        self.water.sync(device, queue, &atlas.held, max_layers);
        let fades = atlas.fade_words(views.fades, self.tables.fade_words);
        self.tables.write_fades(queue, fades);
        self.tables
            .write_directory(device, queue, terrains, &self.atlas);
        let vt_pages = self.vt.wanted(&views.settings);
        self.maps.prepare(device, queue, mips, terrains, vt_pages);
        self.tables.write_palette(queue, terrains, &self.maps);
        let slots = &self.atlas.slots;
        let horizon_settings = (&views.settings, views.shadows);
        (self.horizons).prepare(device, queue, terrains, horizon_settings, views.sun, slots);
        self.sky
            .prepare(device, queue, terrains, &views.settings, slots);
        self.sky.build(device, queue, self.atlas.heights());
        // The runtime virtual texture.
        let tables = &self.tables;
        let textures = self.vt.prepare(
            device,
            queue,
            terrains,
            (views.eye, &views.settings),
            &self.maps,
            &self.sky,
            &self.atlas.changed,
            (
                &tables.directory,
                tables.vt_base,
                tables.directory_generation,
            ),
        );
        if terrains
            .terrains
            .iter()
            .all(|terrain| terrain.patches.is_empty())
        {
            self.water.terrains.clear();
            return false;
        }
        self.tables.size_uniforms(device, terrains);
        let parts = patches::Layout {
            atlas: &self.atlas,
            tables: &self.tables,
            sky: &self.sky,
            horizons: &self.horizons,
            shading: self.deformation.shading(),
            textures: &textures,
        };
        let uniforms = self.patches.lay_out(device, terrains, views, &parts);
        let eye = views.eye.as_dvec3();
        let seen = (views.culling, eye, views.fades);
        self.water.lay_out(device, queue, terrains, seen, &parts);
        queue.write_buffer(&self.tables.uniforms, 0, &uniforms);
        self.patches.write(queue, views);
        self.pipelines.ensure_group(
            device,
            &GroupSources {
                atlas: &self.atlas,
                tables: &self.tables,
                patches: &self.patches,
                maps: &self.maps,
                sky: &self.sky,
                deformation: &self.deformation,
            },
        );
        #[cfg(debug_assertions)]
        self.check(terrains);
        true
    }

    /// Writes reflection probe face `face`'s view, seen from `center` by the
    /// eye-relative projection `relative`, for the terrain whose eye is
    /// `eye`; returns its uniform offset.
    pub(crate) fn write_probe_view(
        &self,
        queue: &wgpu::Queue,
        face: usize,
        relative: Mat4,
        eye: Vec3,
        center: Vec3,
    ) -> u32 {
        self.patches
            .write_probe_view(queue, face, relative, eye, center)
    }

    /// Creates the compute pipelines tiles' normal maps, control values
    /// and sky maps build with unless they exist.
    pub(crate) fn warm(&mut self, device: &wgpu::Device) {
        self.atlas.warm(device);
        self.sky.warm(device);
    }

    /// What ground cover's growth reads: the height and
    /// control arrays, the directory, palette and covers, and generations
    /// advancing when any is another buffer or texture; `None` before
    /// tiles arrive.
    pub(crate) fn cover_sources(&self) -> Option<CoverSources<'_>> {
        let (heights, _, control) = self.atlas.views()?;
        let tables = &self.tables;
        Some(CoverSources {
            heights,
            control,
            directory: &tables.directory,
            palette: &tables.palette,
            covers: &tables.covers,
            deformation: self.deformation.map().0,
            shading: self.deformation.shading(),
            sky: self.sky.view(),
            generation: (
                self.atlas.generation,
                tables.directory_generation,
                self.deformation.map().1,
                self.sky.generation(),
            ),
            revision: tables.directory_revision,
        })
    }

    /// Terrain `index`'s place in the directory and palette, its densest
    /// cover (blades per square meter) and that cover's mean blade height
    /// (meters), as the last preparation wrote them, and its sky map's
    /// layer plus one (0 until built).
    pub(crate) fn cover_terrain(&self, index: usize) -> (u32, u32, [f32; 2], u32) {
        let tables = &self.tables;
        (
            tables.directory_bases.get(index).copied().unwrap_or(0),
            tables.palette_bases.get(index).copied().unwrap_or(0),
            tables
                .cover_densities
                .get(index)
                .copied()
                .unwrap_or([0.0; 2]),
            self.sky.layer(index),
        )
    }

    /// Draws `range` of this frame's patches of the terrain whose uniform
    /// sits at `uniform`, seen by the view at `view`, with the pipeline the
    /// caller set; returns the draws.
    pub(crate) fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        range: &Range<u32>,
        view: u32,
        uniform: u32,
    ) -> u32 {
        let Some(group) = self.pipelines.group() else {
            return 0;
        };
        self.patches.draw(pass, group, range, (view, uniform))
    }

    /// Records the runtime virtual texture's bakes through
    /// the camera's terrain group `camera`.
    pub(crate) fn encode_vt_bakes(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shader: &wgpu::ShaderModule,
        camera: (&crate::pipelines::GroupLayout, &wgpu::BindGroup),
    ) {
        if !self.vt.baking() {
            return;
        }
        let (Some(group), Some(heights)) = (self.pipelines.bake_group(), self.atlas.heights())
        else {
            return;
        };
        let tables = &self.tables;
        self.vt.encode_bakes(
            device,
            encoder,
            shader,
            camera,
            (&self.pipelines.layout, group),
            heights,
            &self.maps,
            |id| tables.uniform_of(id),
        );
    }

    /// Records the runtime virtual texture's feedback over
    /// terrain's visibility target `ids`, `size` pixels, with the camera's
    /// uniform buffer `camera`.
    pub(crate) fn encode_vt_feedback(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shader: &wgpu::ShaderModule,
        camera: &wgpu::Buffer,
        (ids, size): (&wgpu::TextureView, (u32, u32)),
    ) {
        let tables = &self.tables;
        self.vt.encode_feedback(
            device,
            encoder,
            shader,
            camera,
            (&self.patches.views, &tables.uniforms, &tables.directory),
            (ids, size),
            |id| tables.uniform_of(id),
        );
    }

    /// Binds the terrain group for the view at `view` and the terrain whose
    /// uniform sits at `uniform`; whether it exists.
    pub(crate) fn bind(&self, pass: &mut wgpu::RenderPass<'_>, view: u32, uniform: u32) -> bool {
        let Some(group) = self.pipelines.group() else {
            return false;
        };
        pass.set_bind_group(1, group, &[view, uniform]);
        true
    }

    /// Whether any patch drawn this frame has holes.
    pub(crate) fn has_holes(&self) -> bool {
        self.patches.has_holes()
    }
}
