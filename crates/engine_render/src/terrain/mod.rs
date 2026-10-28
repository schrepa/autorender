//! Terrain: every resident tile's heights in one texture
//! array, each terrain's ground drawn by the leaves of its subdivision on
//! the GPU (`surface`), and their pipelines in the opaque passes, the
//! regions of the shadow clipmap's tiles a frame redraws and reflection
//! probe faces.
//!
//! Tiles upload as frames list them and release when they leave the list;
//! the array grows by doubling up to the device's layers, beyond which
//! tiles drop (counted) and the ground draws the coarser levels resident
//! there. A terrain's subdivision splits its square by longest edge
//! bisection wherever a leaf's meshlet may stray more than the settings'
//! pixels from the ground's true surface, seen from the detail eye, as the
//! tiles' error pyramids bound it (`atlas`), and its
//! leaves draw that surface, its materials' mesh band moving it, with one indexed
//! indirect draw per terrain and view. The shadow clipmap's tiles, which
//! must not depend on where the eye stood, hold the ground at a level of
//! detail each level's texels choose, unmorphed, over the regions a frame
//! redraws (`patches`), and redraw over ground that arrives, leaves or
//! changes; past the last level the horizon maps shadow alone.
//!
//! Each tile's normal map, the gradient of the smooth surface through its
//! heights at each sample, lies in a parallel array: computed once on
//! upload for level-0 tiles and surroundings' tiles, and uploaded with its
//! control values from its shading for a terrain's coarser tiles, level 0's
//! averaged, so a coarser level looks like the finer one seen from afar.
//! Fragments blend them through a directory of every level's resident
//! tiles, at the levels a pixel's footprint asks for
//! (`terrain_shading_level`) whatever leaves the geometry draws, so light
//! and materials hold still as the subdivision changes.
//!
//! The pass is its parts: the tile atlas (`atlas`), the tables shaders
//! look up (`tables`), the subdivisions (`surface`), the frame's uniforms
//! and kept patches (`patches`) and the pipelines and terrain group
//! (`pipelines`), beside tiles' water
//! (`water_blocks`), the material maps, the sky and horizon maps its
//! lighting uploads, its bounce light, its macro colors and deformation
//! map. Debug builds check what holds between them after every preparation
//! (`invariants`).
//!
//! A preparation does what changed: the atlas looks at the tiles each
//! terrain's list gained, lost or replaced since it was last synced, and
//! works fades out again for the tiles fading in or changed; the
//! directory rewrites the entries of the tiles that took or left layers,
//! and water the blocks of the tiles held anew; the frame's records and
//! uniforms are written where they differ from the last frame's, from
//! vectors kept from frame to frame.

use engine_core::collections::HashMap;
use std::ops::Range;

use bytemuck::Zeroable;
use engine_core::glam::{DVec3, Mat4, Vec3, Vec4};
use engine_core::terrain_data::gpu::SurfaceData;
use engine_core::{RenderTerrain, RenderTerrains, TileKey};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::bindings::{SHADOW_VIEW, TERRAIN, TERRAIN_CONTROL, TERRAIN_NORMALS};
use crate::cached::{Bound, Cached};
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
use crate::terrain_bounce::{Bounce, SceneSources, TerrainSources};
use crate::terrain_horizons::Horizons;
use crate::terrain_macro::MacroMaps;
use crate::terrain_maps::TerrainMaps;
use crate::terrain_sky::SkyMaps;

mod atlas;
#[cfg(debug_assertions)]
mod invariants;
mod patches;
mod pipelines;
mod surface;
mod tables;
mod water_blocks;

use atlas::TileAtlas;
pub(crate) use patches::{PatchDraws, planes, visible};
use pipelines::{GroupSources, TerrainPipelines};
pub(crate) use surface::{SurfaceDraw, SurfaceTrees};
use tables::TerrainTables;
pub(crate) use water_blocks::TerrainWaterDraws;

pub(crate) use crate::gpu_layouts::terrain::PatchRecord;

/// Texels along each side of a layer: a tile's samples with their border.
const LAYER_SIZE: u32 = engine_core::terrain_data::TILE_STRIDE;
/// Uniform offsets of the views and terrains, at the WebGPU alignment.
const SLOT: u64 = 256;
/// The camera's view, then a reflection probe's six faces.
const VIEWS: u64 = 7;
pub(crate) const HEIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Each tile's normals: the smooth surface's rise along x and z at each
/// sample, the two halves of a float, gathered and blended in shaders (a
/// terrain's coarser tiles' from their shading, in the same words).
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
/// Each tile's control values: its samples' two strongest
/// materials, their blend and how wet their ground is, read texel by texel.
const CONTROL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
/// Each tile's rivers as the tile holds them, which the GPU's control
/// values read: its samples' meters to rivers' edges, then the grain, power
/// and flood of the bed of the river holding each
/// (`engine_terrain::Tile::river_texels`).
const RIVER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Uint;

/// Seconds over which a tile that arrived grows out of its parent level,
/// its normals and surface materials taking over from the coarser level's:
/// the same time however fast frames come.
const ARRIVAL_SECONDS: f32 = 0.4;

/// `PatchRecord::holes` of tiles without holes.
const NO_HOLES: u32 = u32::MAX;
/// 32-bit words of a tile's hole bits.
const HOLE_WORDS: usize = ((LAYER_SIZE - 2) * (LAYER_SIZE - 2)).div_ceil(32) as usize;

/// `range` widened to hold `index`.
fn widen(range: Range<usize>, index: usize) -> Range<usize> {
    if range.is_empty() {
        index..index + 1
    } else {
        range.start.min(index)..range.end.max(index + 1)
    }
}

/// The run of `now`'s elements from the first that differs from `was`'s to
/// the last, those past `was`'s end differing: what writing `now` over a
/// buffer holding `was` writes; empty where none differs.
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

/// What ground cover's growth reads of the terrain pass.
pub(crate) struct CoverSources<'a> {
    pub heights: &'a wgpu::TextureView,
    pub control: &'a wgpu::TextureView,
    pub normals: &'a wgpu::TextureView,
    pub directory: &'a wgpu::Buffer,
    pub palette: &'a wgpu::Buffer,
    pub covers: &'a wgpu::Buffer,
    /// Runtime deformation's map and its shading inputs.
    pub deformation: &'a wgpu::Buffer,
    pub shading: crate::deformation::DeformationShading,
    /// The terrains' macro colors, which tint blades as they do the
    /// ground.
    pub macro_maps: &'a wgpu::TextureView,
    pub generation: (u64, u64, u64, u64),
    /// Advances whenever the directory's resident tiles change.
    pub revision: u64,
}

/// What reflections' world trace reads of the terrain pass
/// (`reflections.wgsl`), each as a group binds it, counted by what it came
/// from: the height array, its mips holding the heights' peaks, and the
/// control array once tiles arrived; the directory, the palette, and the
/// macro colors and sky maps.
pub(crate) struct TraceSources<'a> {
    pub arrays: Option<[Bound<'a>; 2]>,
    pub directory: Bound<'a>,
    pub palette: Bound<'a>,
    pub macro_maps: Bound<'a>,
    pub sky: Bound<'a>,
}

/// One terrain drawing this frame: its index among the frame's terrains,
/// its uniform slot, whether any of its resident tiles has holes (its
/// leaves then draw with the pipelines cutting them), the patches each
/// region of shadow tiles redrawn this frame takes of its ground, by the
/// map its casters list as, and the rectangle of the screen its ground
/// covers, in normalized device coordinates (least x and y, then greatest;
/// `None` where it is unbounded), which its resolve shades within.
pub(crate) struct FrameTerrain {
    pub terrain: usize,
    pub uniform: u32,
    pub holes: bool,
    pub maps: Vec<(usize, Range<u32>)>,
    pub screen: Option<[f32; 4]>,
}

impl FrameTerrain {
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

/// What a frame draws of every terrain drawing, and the patches the
/// regions of shadow tiles redrawn draw.
#[derive(Default)]
pub(crate) struct TerrainFrame {
    pub terrains: Vec<FrameTerrain>,
    pub shadow_patches: u32,
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
    /// The detail eye
    /// ([`RenderSnapshot::detail_eye`](engine_core::RenderSnapshot::detail_eye)),
    /// which bounce light's levels lie around: the camera's eye unless
    /// frozen.
    pub detail_eye: Vec3,
    /// The regions of shadow tiles this frame redraws, which the ground
    /// casts into at a level of detail each one's texels choose: the map
    /// whose lists each reads, and its planes and texels.
    pub regions: &'a [(usize, crate::shadows::KeptRefresh)],
    /// Terrain settings, and whether shadows run, which horizon maps
    /// serve.
    pub settings: engine_core::TerrainSettings,
    pub shadows: bool,
    /// Water settings, sanitized: whether terrains' water subdivides, and
    /// how finely.
    pub water: engine_core::Water,
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
    /// Each terrain's subdivisions, whose leaves the camera's ground and the
    /// water its tiles hold draw.
    pub(crate) surface: SurfaceTrees,
    /// This frame's uniforms, views and the kept level's patches.
    pub(crate) patches: PatchDraws,
    /// Pipelines and the terrain group.
    pub(crate) pipelines: TerrainPipelines,
    /// Long sun shadows.
    pub(crate) horizons: Horizons,
    /// Surface materials' maps.
    pub(crate) maps: TerrainMaps,
    /// Sky visibility.
    pub(crate) sky: SkyMaps,
    /// Bounce light, relit every frame in levels around the detail eye.
    pub(crate) bounce: Bounce,
    /// Whether the session cooks, which makes bounce light's pipelines up
    /// front for the scene's pipeline list.
    pub(crate) cooking: bool,
    /// Macro colors.
    pub(crate) macro_maps: MacroMaps,
    /// Runtime deformation's map.
    pub(crate) deformation: crate::deformation::DeformationPass,
    /// Tiles' water.
    pub(crate) water: water_blocks::WaterBlocks,
}

impl TerrainPass {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
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
            deformation: crate::deformation::DeformationPass::new(device),
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

    /// Bytes of the height, normal, control and river arrays, the
    /// subdivisions, the horizon, sky and macro color maps, bounce light's
    /// and the surface materials' maps.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.atlas.capacity_bytes()
            + self.surface.capacity_bytes()
            + self.horizons.capacity_bytes()
            + self.sky.capacity_bytes()
            + self.bounce.capacity_bytes()
            + self.macro_maps.capacity_bytes()
            + self.maps.capacity_bytes()
            + self.deformation.capacity_bytes()
            + self.water.capacity_bytes()
    }

    /// What water's pass draws tiles' water with this frame's `terrains`;
    /// `None` before tiles and their water arrive.
    pub(crate) fn water_draws(&self, terrains: &RenderTerrains) -> Option<TerrainWaterDraws<'_>> {
        self.water.draws(self, terrains)
    }

    /// The terrain group as the subdivisions' passes bind it, with its
    /// layout, at the offset of the uniform of the terrain whose id is `id`
    /// this frame, with the tiles the GPU holds: what settling water raises
    /// its bed through, reading its materials as the live water's setup
    /// reads them; `None` before the group is made or where no uniform is
    /// laid out for it.
    pub(crate) fn ground(&self, id: u64) -> Option<crate::water_sim::LiveGround<'_>> {
        Some(crate::water_sim::LiveGround {
            layout: &self.pipelines.layout,
            group: self.pipelines.surface_group()?,
            uniform: self.tables.uniform_of(id)?,
            terrain: id,
            held: &self.atlas.held,
        })
    }

    /// Uploads tiles the frame's terrains list and the GPU lacks, releases
    /// those they no longer list, noting the ground that arrived, left or
    /// changed ([`Self::ground_changes`]), and lays out this frame's
    /// subdivisions, uniforms and the patches of the regions of shadow
    /// tiles `views` redraws.
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
        self.atlas.begin(views.now);
        self.patches.begin(views.eye);
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
        // Water's directory, in the directory buffer after the directory's
        // own words.
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
        // Bounce light's levels around the detail eye, which the horizon
        // records place, in the slots the horizon maps' layers hold.
        let room = crate::terrain_horizons::room(device, terrains.terrains.len());
        self.bounce
            .place(terrains, (views.detail_eye, &views.settings), room);
        let horizon_settings = (&views.settings, views.shadows);
        let windows = self.bounce.windows();
        (self.horizons).prepare(device, queue, terrains, horizon_settings, windows);
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
            self.surface.rest();
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
            shading: self.deformation.shading(),
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
                deformation: &self.deformation,
                water: &self.water,
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

    /// Creates the compute pipelines tiles' normal maps and control values
    /// build with unless they exist.
    pub(crate) fn warm(&mut self, device: &wgpu::Device) {
        self.atlas.warm(device);
    }

    /// What ground cover's growth reads: the height, control
    /// and normal arrays, the directory, palette and covers, and generations
    /// advancing when any is another buffer or texture; `None` before
    /// tiles arrive.
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
            deformation: self.deformation.map().0,
            shading: self.deformation.shading(),
            macro_maps: self.macro_maps.view(),
            generation: (
                self.atlas.generation,
                tables.directory_generation,
                self.deformation.map().1,
                self.macro_maps.generation(),
            ),
            revision: tables.directory_revision,
        })
    }

    /// What reflections' world trace reads ([`TraceSources`]).
    pub(crate) fn trace_sources(&self) -> TraceSources<'_> {
        let generation = self.atlas.generation;
        let arrays = self.atlas.views().map(|(heights, _, control)| {
            [heights, control].map(|view| Bound::counted(generation, crate::pipelines::view(view)))
        });
        TraceSources {
            arrays,
            directory: self.tables.directory.bound(),
            palette: Bound::fixed(self.tables.palette.as_entire_binding()),
            macro_maps: Bound::counted(
                self.macro_maps.generation(),
                crate::pipelines::view(self.macro_maps.view()),
            ),
            sky: Bound::counted(
                self.sky.generation(),
                crate::pipelines::view(self.sky.view()),
            ),
        }
    }

    /// The layer resident tile `key` of terrain `id` holds, where it holds
    /// one.
    pub(crate) fn layer_of(&self, id: u64, key: TileKey) -> Option<u32> {
        self.atlas.slots.get(&(id, key)).copied()
    }

    /// The highest of terrain `id`'s resident tiles' heights
    /// (terrain-local); none before one arrives.
    pub(crate) fn highest(&self, id: u64) -> Option<f32> {
        self.atlas
            .tops
            .iter()
            .filter(|((terrain, _), _)| *terrain == id)
            .map(|(_, &top)| top)
            .reduce(engine_core::math::max)
    }

    /// Layers the arrays hold, and a revision advancing whenever the
    /// directory's words change, as tiles take or leave layers.
    pub(crate) fn layers(&self) -> (u32, u64) {
        (self.atlas.layers(), self.tables.directory_revision)
    }

    /// The tallest blade any terrain material's cover names, as the last
    /// preparation wrote the covers (meters).
    pub(crate) fn cover_tallest(&self) -> f32 {
        self.tables.cover_tallest
    }

    /// Terrain `index`'s place in the directory and palette, its covers as
    /// ground cover's law takes them, as the last preparation wrote them,
    /// and its macro colors' layer plus one (0 until they arrive).
    pub(crate) fn cover_terrain(
        &self,
        index: usize,
    ) -> (u32, u32, crate::ground_cover::CoverTerms, u32) {
        let tables = &self.tables;
        (
            tables.directory_bases.get(index).copied().unwrap_or(0),
            tables.palette_bases.get(index).copied().unwrap_or(0),
            tables.cover_terms.get(index).copied().unwrap_or_default(),
            self.macro_maps.layer(index),
        )
    }

    /// Draws the leaves of this frame's subdivision of `terrain`, seen by the
    /// view at `view`, with the pipeline the caller set (`SurfaceDraw`, valid
    /// once the frame's subdivision pass has listed them); returns the draws.
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

    /// The leaves of this frame's subdivisions, which tiles' water draws on
    /// (`meshes/terrain_surface.wgsl`), and a generation advancing when they
    /// are another buffer.
    pub(crate) fn surface_leaves(&self) -> Option<(&wgpu::Buffer, u64)> {
        let leaves = self.surface.leaves();
        Some((leaves, leaves.generation()))
    }

    /// Draws nothing this frame, as no terrain is: bounce light rests and the
    /// subdivisions' stats read none drawn.
    pub(crate) fn rest(&mut self) {
        self.bounce.rest();
        self.surface.rest();
    }

    /// Whether any terrain subdivides this frame, which its pass then
    /// records.
    pub(crate) fn subdivides(&self) -> bool {
        self.surface.any()
    }

    /// Draws into the region of shadow tiles whose casters list as `map`
    /// the ground each terrain holds there. Returns the draws.
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

    /// Records this frame's subdivisions through the camera's uniform
    /// `camera` (`shader` the mesh module), timed by `profiler`.
    pub(crate) fn encode_subdivision(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        shader: &wgpu::ShaderModule,
        camera: &wgpu::Buffer,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let Some(group) = self.pipelines.surface_group() else {
            return;
        };
        let terrain = (&self.pipelines.layout, group);
        self.surface
            .encode(device, encoder, shader, camera, terrain, profiler);
    }

    /// Records bounce light's relighting, reading `scene`'s lighting, shadow
    /// lookups and clouds' shadows, timed by `profiler`.
    pub(crate) fn encode_bounce(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
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

    /// Binds the terrain group for the view at `view` and the terrain whose
    /// uniform sits at `uniform`; whether it exists.
    pub(crate) fn bind(&self, pass: &mut wgpu::RenderPass<'_>, view: u32, uniform: u32) -> bool {
        let Some(group) = self.pipelines.group() else {
            return false;
        };
        pass.set_bind_group(1, group, &[view, uniform]);
        true
    }

    /// Whether any terrain drawing this frame has holes.
    pub(crate) fn has_holes(&self) -> bool {
        self.patches
            .frame
            .terrains
            .iter()
            .any(|terrain| terrain.holes)
    }

    /// The ground that arrived, left or changed this frame, over which the
    /// shadow clipmap's tiles holding it redraw.
    pub(crate) fn ground_changes(&self) -> &[patches::GroundChange] {
        self.patches.ground_changes()
    }
}
