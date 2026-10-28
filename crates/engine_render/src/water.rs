//! Water: terrains' water as their tiles hold it, and the sea. Each terrain
//! whose tiles hold water subdivides a second tree over its square, its
//! leaves measured by how far they stray from the water's surface and those
//! over no water collapsed (`crate::terrain::SurfaceTrees`), and water
//! draws its leaves (`meshes/water.wgsl`): their vertices stand at the
//! water's surface, lifted by the pixels the drawn ground and the water's
//! mesh may stray from their surfaces, and each pixel reads the tiles'
//! water blocks where it lies through water's directory
//! (`crate::terrain::TerrainPass::water_draws`), at the level its size
//! picks, so the depth each pixel shades by is the tiles' water over that
//! level's ground, the shore falls where the ground rises through the
//! water, and none of it moves as the eye does: each shows the water's
//! state there and nothing else. Around the detail eye the same pixels read
//! the water running live (`crate::water_sim`) as a level of its own, its
//! cells' looks, the vertices standing on its surface where it stands
//! higher, and hand it back to the tiles' water across its rim and as far
//! as its cells stay a few pixels across: one water, drawn and shaded
//! through one path. Water draws after the opaque scene and its
//! decals and blends over the ground by how little light comes back through
//! it, writing the depth and surface targets, so later passes and
//! screen-space reflections see its surface. Before it draws, the opaque
//! depth copies into a texture of its own: what stands in the water behind
//! a pixel shortens the light's path to it. In frames drawing the effects'
//! targets the opaque color copies too, and water covers it whole with what
//! shows through it, bent by its surface and dimmed channel by channel over
//! the light's path; other frames blend water over the ground.
//!
//! Where terrains' rivers leave their beds, their jets draw last, as
//! sheets through the air (`crate::water_falls`), each terrain's through
//! its own uniform, by which they read the water under them: they show only
//! where they stand clear of it, and write their depth where they show.
//!
//! What turbulence makes on terrains' water, its white water, foam and the
//! disturbance of its surface, runs live on a field of its own around the
//! detail eye (`crate::water_foam_field`), stepped before water draws over
//! the water of the terrain the detail eye stands over, read as the water's
//! shading reads it; its textures bind beside what lies behind the water,
//! which its passes do not bind.
//!
//! Water is a feature of the renderer (`crate::render_features`).
//!
//! The sea draws first in the same pass, from a grid around
//! the eye: rings of [`OCEAN_SEGMENTS`] cells, a cell apart near the eye
//! and then each wider by a cell's share of its circumference, so cells
//! stay square, out to the sea's reach. Its sectors cull on their own;
//! vertices move by the waves longer than a few of their cells, and pixels
//! shade every wave longer than a few pixels. It draws through the
//! terrain's projection, jittered alike, so its shores meet the ground.

use engine_core::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::glam::{Mat4, Vec3};
use engine_core::math::max;
use engine_core::terrain_data::Fall;
use engine_core::{
    Environment, MAX_OCEAN_WAVES, Ocean, OceanStats, RenderSnapshot, RenderTerrain, Water,
    WaterStats, Wind,
};

use crate::bindings::{WATER, WATER_BEHIND};
use crate::gpu_layouts::terrain::{SubdivisionUniform, TerrainUniform, ViewUniform};
use crate::profiling::Span;
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{Access, FrameShape, Node, Pass, RenderGraph, Resource, Targets};
use crate::scene_shading::SceneShading;
use crate::targets::{FrameSize, ScreenTarget};
use crate::terrain::{SurfaceDraw, TerrainWaterDraws};
use crate::water_falls::{fall_layout, sheets};
use crate::water_foam_field::{FoamField, FoamView, foam_terrain};
use crate::water_sim::{Live, WaterSim};

/// Cells around the sea's grid, the sectors it splits into for culling,
/// and the rings a cell apart at its middle: about the cells around over
/// 2 pi, where a cell's share of the circumference reaches a cell.
const OCEAN_SEGMENTS: u32 = 192;
const OCEAN_SECTORS: u32 = 16;
const OCEAN_FIRST_RINGS: u32 = 31;
/// The sea grid's finest cell (meters), and how much wider it grows per
/// meter the eye stands over the sea.
const OCEAN_CELL: f32 = 0.3;
const OCEAN_CELL_PER_HEIGHT: f32 = 0.004;

use crate::gpu_layouts::water::{FallVertex, WaterUniform};

/// Bytes of the water group's uniform.
pub(crate) const WATER_UNIFORM_BYTES: usize = size_of::<WaterUniform>();

/// One-texel stand-ins for what the water group binds that the sea reads
/// none of: the terrain's height array, uniforms and directory, its
/// subdivisions' leaves, tiles' water blocks, water's directory, the live
/// water's looks (which tiles' water too reads none of while none runs)
/// and the tree of a terrain's water; and for the scene's color, which
/// frames blending water read none of.
struct Placeholders {
    surfaces: wgpu::Texture,
    info: wgpu::Texture,
    views: wgpu::Buffer,
    terrains: wgpu::Buffer,
    scene: wgpu::TextureView,
    live: wgpu::Buffer,
    storage: wgpu::Buffer,
    tree: wgpu::Buffer,
}

/// The sea on the GPU: its uniform and group, and this frame's rings and
/// the runs of sectors the camera sees.
struct OceanDraw {
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    rings: u32,
    runs: Vec<Range<u32>>,
}

/// Tiles' water this frame: each terrain drawing it, its id, its uniform's
/// offset, its water's tree's and that tree's draw of its leaves.
struct TileDraws {
    terrains: Vec<(u64, u32, u32, SurfaceDraw)>,
}

/// Terrains' falls' jets: their vertices and triangles' indices, and the
/// falls they were built from (each terrain's id, origin and falls, and the
/// indices of its jets).
struct FallDraws {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    built: Vec<(u64, Vec3, Arc<[Fall]>, Range<u32>)>,
}

/// The copies of the opaque scene water reads, the depth and, in frames
/// drawing the effects' targets, the color; and their group, which binds
/// the foam field's texture as of the generation `foam`.
struct BehindCopies {
    depth: wgpu::Texture,
    scene: Option<wgpu::Texture>,
    group: wgpu::BindGroup,
    foam: u64,
}

/// What water draws this frame: tiles' water blocks resident and dropped,
/// the leaves the water's subdivisions drew and the splits their room
/// refused as last read back, and the sea's sectors and their triangles.
pub(crate) struct WaterFrame {
    blocks: [u32; 2],
    subdivided: [u32; 2],
    sectors: u32,
    triangles: u32,
}

pub(crate) struct WaterPass {
    pub(crate) layout: crate::pipelines::GroupLayout,
    behind_layout: crate::pipelines::GroupLayout,
    behind: ScreenTarget<BehindCopies>,
    placeholders: Placeholders,
    // Water running live around the detail eye.
    sim: WaterSim,
    // The foam terrains' water carries, simulated on a field of its own
    // around the detail eye.
    foam: FoamField,
    // Tiles' water's uniform, and the group binding it with the terrain
    // pass's arrays, uniforms, directory, leaves and trees' uniforms and the
    // live water's looks, as of their generations.
    uniform: wgpu::Buffer,
    group: Option<(GroupGenerations, wgpu::BindGroup)>,
    tiles: Option<TileDraws>,
    pipeline_layout: Option<wgpu::PipelineLayout>,
    pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    ocean_pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    ocean: Option<OceanDraw>,
    // Terrains' falls' sheets, as of the falls they were built from, and
    // what builds them.
    falls: Option<FallDraws>,
    fall_pipelines: HashMap<Targets, wgpu::RenderPipeline>,
    fall_vertices: Vec<FallVertex>,
    fall_indices: Vec<u32>,
    // The scene features the pipelines are specialized for, and whether
    // those drawing into the debug targets draw for the water's views.
    shading: SceneShading,
    water_views: bool,
}

impl WaterPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            layout: crate::pipelines::layout(device, &WATER),
            behind_layout: crate::pipelines::layout(device, &WATER_BEHIND),
            behind: ScreenTarget::default(),
            placeholders: placeholders(device),
            sim: WaterSim::new(device),
            foam: FoamField::new(device),
            uniform: crate::buffers::uniform(device, "engine/water", WATER_UNIFORM_BYTES as u64),
            group: None,
            tiles: None,
            pipeline_layout: None,
            pipelines: HashMap::default(),
            ocean_pipelines: HashMap::default(),
            ocean: None,
            falls: None,
            fall_pipelines: HashMap::default(),
            fall_vertices: Vec::new(),
            fall_indices: Vec::new(),
            shading: SceneShading::ALL,
            water_views: false,
        }
    }

    /// Specializes the pipelines for `shading`'s scene features, and the one
    /// drawing terrains' water into the debug targets for whether frames
    /// show the water's views (`water_views`,
    /// `crate::debug_view::shows_water`): those made for others are
    /// dropped, to be created anew.
    pub(crate) fn specialize(&mut self, shading: SceneShading, water_views: bool) {
        if self.shading != shading {
            self.shading = shading;
            self.pipelines.clear();
            self.ocean_pipelines.clear();
            self.fall_pipelines.clear();
        }
        if self.water_views != water_views {
            self.water_views = water_views;
            self.pipelines.remove(&Targets::Debug);
        }
    }

    /// Builds the jets of `terrains`' falls unless they were built from the
    /// same falls; drops them where none fall.
    fn prepare_falls(&mut self, device: &wgpu::Device, terrains: &[RenderTerrain]) {
        let falling = || terrains.iter().filter(|terrain| !terrain.falls.is_empty());
        let same = self.falls.as_ref().is_some_and(|built| {
            built.built.len() == falling().count()
                && built
                    .built
                    .iter()
                    .zip(falling())
                    .all(|(a, b)| a.0 == b.id && a.1 == b.origin && Arc::ptr_eq(&a.2, &b.falls))
        });
        if same || (self.falls.is_none() && falling().next().is_none()) {
            return;
        }
        let runs = sheets(
            falling().map(|terrain| (terrain.origin, &*terrain.falls)),
            &mut self.fall_vertices,
            &mut self.fall_indices,
        );
        if self.fall_indices.is_empty() {
            self.falls = None;
            return;
        }
        let buffer = |label, contents: &[u8], usage| {
            wgpu::util::DeviceExt::create_buffer_init(
                device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                },
            )
        };
        self.falls = Some(FallDraws {
            vertices: buffer(
                "engine/water-falls",
                bytemuck::cast_slice(&self.fall_vertices),
                wgpu::BufferUsages::VERTEX,
            ),
            indices: buffer(
                "engine/water-fall-indices",
                bytemuck::cast_slice(&self.fall_indices),
                wgpu::BufferUsages::INDEX,
            ),
            built: falling()
                .zip(runs)
                .map(|(terrain, run)| (terrain.id, terrain.origin, terrain.falls.clone(), run))
                .collect(),
        });
    }

    /// The sea's sectors drawn this frame and their triangles.
    fn ocean_drawn(&self) -> (u32, u32) {
        self.ocean.as_ref().map_or((0, 0), |ocean| {
            let sectors: u32 = ocean.runs.iter().map(|run| run.len() as u32).sum();
            let per_sector = OCEAN_SEGMENTS / OCEAN_SECTORS;
            (sectors, sectors * ocean.rings * per_sector * 2)
        })
    }

    /// Copies the opaque `depth` and, when water refracts it, the
    /// `scene`'s color into the copies water reads, before it draws.
    fn copy_behind(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::Texture,
        scene: Option<&wgpu::Texture>,
    ) {
        let Some(copies) = self.behind.get() else {
            return;
        };
        let pairs = [
            Some((depth, &copies.depth)),
            scene.zip(copies.scene.as_ref()),
        ];
        for (from, to) in pairs.into_iter().flatten() {
            if from.size() == to.size() {
                encoder.copy_texture_to_texture(
                    from.as_image_copy(),
                    to.as_image_copy(),
                    from.size(),
                );
            }
        }
    }

    /// Prepares this frame's water: tiles' water on the leaves of its
    /// subdivisions as the terrain pass holds them (`tiles`) and the falls
    /// of `terrains`, shaded as `settings` say, and the sea `ocean`, culled
    /// through `cull` and drawn through `view` (both eye-relative) over the
    /// opaque `depth` and, when `refracts`, the color it copies; seen from
    /// `eye`, and by the detail eye and the display as `seen` says, which
    /// size and place the foam field, the wind `wind` dragging the water
    /// running live and the foam riding it, its waves and the live water's
    /// steps at the simulation time `seconds` the frame presents; `None` when
    /// none of it draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (tiles, terrains): (Option<TerrainWaterDraws<'_>>, &[RenderTerrain]),
        (settings, ocean): (Water, Ocean),
        (eye, seen, wind, seconds): (Vec3, FoamView, Wind, f64),
        (cull, view, depth, refracts): (Mat4, Mat4, &wgpu::Texture, bool),
    ) -> Option<WaterFrame> {
        self.tiles = None;
        let blocks = tiles.as_ref().map_or([0; 2], |tiles| tiles.blocks);
        let subdivided = tiles.as_ref().map_or([0; 2], |tiles| tiles.subdivided);
        let tiles = tiles.filter(|tiles| settings.enabled && !tiles.terrains.is_empty());
        self.prepare_falls(device, if settings.enabled { terrains } else { &[] });
        // Falls draw with tiles' water's group, each terrain's through the
        // uniform its tiles' water draws with: alongside tiles' water alone.
        if tiles.is_none() && !ocean.enabled {
            self.ocean = None;
            return None;
        }
        // Rivers running live where a terrain's window stands, pushed by the
        // solids standing in it, the wind dragging their surface.
        let live = terrains
            .iter()
            .find_map(|terrain| {
                Some(Live {
                    window: terrain.water_window.as_ref()?,
                    ahead: terrain.water_ahead.as_ref(),
                    touches: &terrain.water_touches[..],
                    beyond: terrain.water_touches_beyond,
                })
            })
            .filter(|_| settings.enabled);
        self.sim.prepare(device, queue, live, (wind, seconds));
        let (running, sim_generation) = self.sim.drawn();
        // The square of the window that runs: its first corner, the meters
        // across a cell and its cells along a side; its datum, the slots its
        // blocks turn by along x and z and its pool's slots along each side;
        // and the pixels a cell spans where its water hands back.
        let running = running.map(|drawn| {
            let window = drawn.window;
            let square = [
                window.origin[0],
                window.origin[1],
                window.cell,
                window.cells as f32,
            ];
            let [wrap_x, wrap_z] = drawn.wrap;
            (
                square,
                [
                    window.datum,
                    wrap_x as f32,
                    wrap_z as f32,
                    drawn.pool as f32,
                ],
                [window.pixels, 0.0, 0.0, 0.0],
            )
        });
        // The foam tiles' water carries, over the water of the terrain the
        // detail eye stands over, after the live steps it reads.
        let foamed = tiles
            .as_ref()
            .and_then(|tiles| foam_terrain(tiles, terrains, seen.detail));
        self.foam.prepare(
            (device, queue),
            foamed,
            seen,
            (&settings, wind.velocity, seconds),
        );
        if let Some(tiles) = tiles {
            self.prepare_tiles(device, &tiles, sim_generation);
            // Terrains' water gives way to the sea where it lies under it.
            let sea = if ocean.enabled {
                [ocean.level - eye.y, 1.0, 0.0, 0.0]
            } else {
                [0.0; 4]
            };
            let [red, green, blue] = settings.color;
            let sediment = settings.sediment_color;
            let uniform = WaterUniform {
                color: [red, green, blue, settings.clarity],
                absorption: absorption(settings.color, settings.clarity),
                ocean: sea,
                foam: self.foam.shading(),
                sim: running.map_or([0.0; 4], |(square, _, _)| square),
                sim_datum: running.map_or([0.0; 4], |(_, datum, _)| datum),
                sim_pixels: running.map_or([0.0; 4], |(_, _, pixels)| pixels),
                sediment: [sediment[0], sediment[1], sediment[2], settings.sediment],
                // Its vertices stand over its surface by the pixels the
                // drawn ground and its mesh may stray from theirs.
                lift: [tiles.targets[0], tiles.targets[1], 0.0, 0.0],
                ..WaterUniform::zeroed()
            };
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        }
        let sea = self.prepare_ocean(device, queue, (cull, view), ocean, (eye, seconds));
        if self.tiles.is_none() && !sea {
            return None;
        }
        // The copies, at the depth buffer's size: the frame's.
        let size = FrameSize {
            width: depth.width(),
            height: depth.height(),
        };
        // Copies of another frame's kind, or whose group binds the foam
        // field's texture before it changed, are made again.
        let foam = self.foam.generation();
        let stale = self
            .behind
            .get()
            .is_some_and(|copies| copies.scene.is_some() != refracts || copies.foam != foam);
        if stale {
            self.behind.release();
        }
        if self.behind.resizes(size) {
            let copies = self.behind_copies(device, size, depth.format(), refracts);
            self.behind.set(size, copies);
        }
        let (sectors, triangles) = self.ocean_drawn();
        Some(WaterFrame {
            blocks,
            subdivided,
            sectors,
            triangles,
        })
    }

    /// Prepares tiles' water over `tiles`: the group of the water layout
    /// its draws bind, made again as what it binds changes (the live
    /// water's looks as of `sim_generation` among it), and the draws of
    /// each terrain's water's tree.
    fn prepare_tiles(
        &mut self,
        device: &wgpu::Device,
        tiles: &TerrainWaterDraws<'_>,
        sim_generation: u64,
    ) {
        let generations = (tiles.generation, sim_generation);
        if self
            .group
            .as_ref()
            .is_none_or(|(bound, _)| *bound != generations)
        {
            let group = self.tile_group(device, tiles);
            self.group = Some((generations, group));
        }
        self.tiles = Some(TileDraws {
            terrains: tiles.terrains.clone(),
        });
    }

    /// The copies of the opaque scene at `size`, the depth's of `format`
    /// and, when water `refracts`, the color's; and their group, with the
    /// foam field's textures (the one-texel color in their place while none
    /// runs) and their sampler.
    fn behind_copies(
        &self,
        device: &wgpu::Device,
        size: FrameSize,
        format: wgpu::TextureFormat,
        refracts: bool,
    ) -> BehindCopies {
        let copy = |label, format| {
            crate::targets::texture_2d(
                device,
                label,
                size.pixels(),
                1,
                1,
                format,
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            )
        };
        let depth = copy("engine/water-depth", format);
        let scene = refracts.then(|| copy("engine/water-scene", crate::post::SCENE_FORMAT));
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let scene_view = scene
            .as_ref()
            .map(|scene| scene.create_view(&wgpu::TextureViewDescriptor::default()));
        let group = crate::pipelines::group(
            device,
            "engine/water-behind",
            &WATER_BEHIND,
            &self.behind_layout,
            &[
                ("water_depth", crate::pipelines::view(&depth_view)),
                (
                    "water_scene",
                    crate::pipelines::view(scene_view.as_ref().unwrap_or(&self.placeholders.scene)),
                ),
                (
                    "water_foam_field",
                    crate::pipelines::view(self.foam.view().unwrap_or(&self.placeholders.scene)),
                ),
                (
                    "water_foam_sampler",
                    crate::pipelines::sampler(self.foam.sampler()),
                ),
                (
                    "water_foam_spread",
                    crate::pipelines::view(
                        self.foam.spread_view().unwrap_or(&self.placeholders.scene),
                    ),
                ),
            ],
        );
        BehindCopies {
            depth,
            scene,
            group,
            foam: self.foam.generation(),
        }
    }

    /// The group of the water layout tiles' water binds: the terrain pass's
    /// height array, views' and terrains' uniforms, directory (with water's
    /// in it), leaves of its subdivisions and their trees' uniforms, and the
    /// water blocks (`tiles`); water's uniform; and the live water's looks.
    fn tile_group(&self, device: &wgpu::Device, tiles: &TerrainWaterDraws<'_>) -> wgpu::BindGroup {
        let looks = self
            .sim
            .drawn()
            .0
            .map_or(&self.placeholders.live, |drawn| drawn.looks);
        crate::pipelines::group(
            device,
            "engine/water-group",
            &WATER,
            &self.layout,
            &[
                ("terrain_heights", crate::pipelines::view(tiles.heights)),
                ("terrain_view", sized(tiles.views, size_of::<ViewUniform>())),
                (
                    "terrain",
                    sized(tiles.uniforms, size_of::<TerrainUniform>()),
                ),
                crate::buffers::entry("terrain_directory", tiles.directory),
                crate::buffers::entry("terrain_surface_leaves", tiles.leaves),
                ("water_surface", crate::pipelines::view(tiles.surfaces)),
                ("water_info", crate::pipelines::view(tiles.info)),
                ("water", self.uniform.as_entire_binding()),
                crate::buffers::entry("water_live", looks),
                (
                    "water_tree",
                    sized(tiles.trees, size_of::<SubdivisionUniform>()),
                ),
            ],
        )
    }

    /// Prepares the sea culled through `cull` and drawn through `view`
    /// (both eye-relative) from `eye`, its waves at simulation time
    /// `seconds`; returns whether any of it draws.
    fn prepare_ocean(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (cull, view): (Mat4, Mat4),
        ocean: Ocean,
        (eye, seconds): (Vec3, f64),
    ) -> bool {
        if !ocean.enabled {
            self.ocean = None;
            return false;
        }
        let draw = self
            .ocean
            .get_or_insert_with(|| ocean_draw(device, &self.layout, &self.placeholders));
        let waves = ocean.spectrum();
        let amplitude = waves.amplitude();
        // Cells widen as the eye rises over the sea.
        let above = max(eye.y - ocean.level, 0.0);
        let cell = OCEAN_CELL + above * OCEAN_CELL_PER_HEIGHT;
        let growth = 1.0 + std::f32::consts::TAU / OCEAN_SEGMENTS as f32;
        let first = OCEAN_FIRST_RINGS as f32 * cell;
        let widening = max((ocean.reach / first).ln() / growth.ln(), 0.0).ceil() as u32;
        draw.rings = OCEAN_FIRST_RINGS + widening;
        // Each sector a wedge from the eye out to the reach, as high as the
        // waves rise and fall, culled eye-relative.
        let planes = crate::terrain::planes(cull, false);
        let (low, high) = (
            ocean.level - eye.y - amplitude,
            ocean.level - eye.y + amplitude,
        );
        draw.runs.clear();
        for sector in 0..OCEAN_SECTORS {
            let angle =
                |step: f32| (sector as f32 + step) / OCEAN_SECTORS as f32 * std::f32::consts::TAU;
            let (mut min, mut max) = (Vec3::new(0.0, low, 0.0), Vec3::new(0.0, high, 0.0));
            for step in [0.0, 0.5, 1.0] {
                // A little past the reach, so the arc between stays inside.
                let (sin, cos) = engine_core::math::sin_cos(angle(step));
                let point = Vec3::new(cos, 0.0, sin) * (ocean.reach * 1.02);
                min = min.min(point.with_y(low));
                max = max.max(point.with_y(high));
            }
            if !crate::terrain::visible(&planes, min, max) {
                continue;
            }
            match draw.runs.last_mut() {
                Some(run) if run.end == sector => run.end = sector + 1,
                _ => draw.runs.push(sector..sector + 1),
            }
        }
        let phases = waves.phases(f64::from(eye.x), f64::from(eye.z), seconds);
        let mut ocean_waves = [[0.0; 4]; MAX_OCEAN_WAVES];
        let mut ocean_reach = [[0.0; 4]; MAX_OCEAN_WAVES / 4];
        for (index, wave) in waves.waves().iter().enumerate() {
            ocean_waves[index] = [
                wave.vector[0],
                wave.vector[1],
                wave.amplitude,
                phases[index],
            ];
            ocean_reach[index / 4][index % 4] = wave.reach;
        }
        let [red, green, blue] = ocean.color;
        // Whitecaps cover a share of the sea growing with the wind's speed
        // to the power 3.41 (Monahan's fit; about 2 % at 12 m/s): crests
        // whiten where the waves compress the surface more than that share
        // of it, which the spectrum's shape fixes.
        let speed = ocean.wind[0].hypot(ocean.wind[1]);
        let share = 3.84e-6 * speed.powf(3.41);
        let crests = 0.67 + 0.069 * (max(share, 1.0e-4) / 0.01).ln();
        // Its shores' foam reads its waves as one wave holding all their
        // variance, running up them.
        let swell = waves
            .waves()
            .iter()
            .map(|wave| wave.amplitude * wave.amplitude)
            .sum::<f32>()
            .sqrt();
        let heading = [ocean.wind[0] / speed, ocean.wind[1] / speed];
        let uniform = WaterUniform {
            relative_view_projection: view.to_cols_array_2d(),
            origin: [heading[0], heading[1], speed, 0.0],
            color: [red, green, blue, ocean.clarity],
            // Its crests' whitecaps, the wave its shores' foam reads, and how
            // high its waves rise together.
            foam: [crests, swell, 0.0, amplitude],
            absorption: absorption(ocean.color, ocean.clarity),
            ocean: [
                ocean.level - eye.y,
                cell,
                OCEAN_SEGMENTS as f32,
                OCEAN_FIRST_RINGS as f32,
            ],
            ocean_grid: [
                growth,
                ocean.reach,
                (OCEAN_SEGMENTS / OCEAN_SECTORS) as f32,
                waves.count as f32,
            ],
            ocean_waves,
            ocean_reach,
            sim: [0.0; 4],
            sim_datum: [0.0; 4],
            // The sea carries no sediment, and stands on no subdivision.
            sediment: [0.0; 4],
            lift: [0.0; 4],
            sim_pixels: [0.0; 4],
        };
        queue.write_buffer(&draw.uniform, 0, bytemuck::bytes_of(&uniform));
        !draw.runs.is_empty()
    }

    /// Creates the pipelines into `targets` this frame's water and sea
    /// draw with unless they exist, and tiles' water's when `coming`,
    /// before a bake gives any, and those of the foam field's passes once
    /// it runs; `camera` is water's first group (`bindings::CAMERA_WATER`).
    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &crate::pipelines::GroupLayout,
        (targets, coming): (Targets, bool),
        formats: (wgpu::TextureFormat, wgpu::TextureFormat),
    ) {
        if (coming || self.tiles.is_some()) && !self.pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Tiles);
            self.pipelines.insert(targets, pipeline);
        }
        if self.ocean.is_some() && !self.ocean_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Sea);
            self.ocean_pipelines.insert(targets, pipeline);
        }
        if self.falls.is_some() && !self.fall_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Falls);
            self.fall_pipelines.insert(targets, pipeline);
        }
        self.foam.ensure_pipelines(device, shader, &self.layout);
    }

    /// The layout water's pipelines share, its first group `camera`'s, made
    /// once.
    fn pipeline_layout(
        &mut self,
        device: &wgpu::Device,
        camera: &crate::pipelines::GroupLayout,
    ) -> wgpu::PipelineLayout {
        self.pipeline_layout
            .get_or_insert_with(|| {
                crate::pipelines::pipeline_layout(
                    device,
                    &crate::pipeline_layouts::WATER_PIPELINES,
                    &[camera, &self.layout, &self.behind_layout],
                )
            })
            .clone()
    }

    /// The pipeline of `surface` into `targets`.
    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        camera: &crate::pipelines::GroupLayout,
        targets: Targets,
        (canvas, depth): (wgpu::TextureFormat, wgpu::TextureFormat),
        surface: Surface,
    ) -> wgpu::RenderPipeline {
        // Water and the sea share one layout.
        let layout = self.pipeline_layout(device, camera);
        let (vertex, fragment) = match surface {
            Surface::Tiles => crate::entries::water(targets),
            Surface::Sea => crate::entries::ocean(targets),
            Surface::Falls => crate::entries::falls(targets),
        };
        // The scene color blends premultiplied over the ground, keeping the
        // ground's ambient share in its alpha (water refracting the scene
        // covers it whole but where its shores fade); the other targets
        // take the water's.
        let blended = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let mut color_targets = targets.color_targets(canvas);
        // Debug views write what drew each pixel over what is behind.
        if let Some(Some(scene)) = color_targets.first_mut()
            && targets != Targets::Debug
        {
            scene.blend = Some(blended);
        }
        let sheets = [Some(fall_layout())];
        // Terrains' water draws into the debug targets as the view shown
        // asks.
        let constants = match (targets, surface) {
            (Targets::Debug, Surface::Tiles) => self
                .shading
                .with(&crate::debug_view::water_constants(self.water_views)),
            _ => self.shading.constants().to_vec(),
        };
        crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: match surface {
                    Surface::Tiles => "engine/water",
                    Surface::Sea => "engine/ocean",
                    Surface::Falls => "engine/water-falls",
                },
                layout: &layout,
                module: shader,
                vertex_constants: &[],
                vertex,
                // Tiles' water's vertices come from its trees' leaves, the
                // sea's from their indices alone.
                buffers: match surface {
                    Surface::Tiles | Surface::Sea => &[],
                    Surface::Falls => &sheets,
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // Water is seen from above and below alike, falls from
                    // either side.
                    cull_mode: None,
                    ..Default::default()
                },
                depth: Some(wgpu::DepthStencilState {
                    format: depth,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                fragment: Some(crate::pipelines::Fragment {
                    entry: fragment,
                    constants: &constants,
                    targets: &color_targets,
                }),
            },
        )
    }

    /// Records this frame's water into `pass` with water's first group
    /// `camera` (`bindings::CAMERA_WATER`); returns its draws.
    fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera: &wgpu::BindGroup,
        targets: Targets,
    ) -> u32 {
        let Some(behind) = self.behind.get() else {
            return 0;
        };
        let mut draws = 0;
        // The sea first: terrains' water below its surface stays hidden.
        if let (Some(ocean), Some(pipeline)) = (&self.ocean, self.ocean_pipelines.get(&targets)) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, &ocean.group, &[0, 0, 0]);
            pass.set_bind_group(2, &behind.group, &[]);
            let vertices = ocean.rings * (OCEAN_SEGMENTS / OCEAN_SECTORS) * 6;
            for run in &ocean.runs {
                pass.draw(0..vertices, run.clone());
                draws += 1;
            }
        }
        let Some((_, group)) = &self.group else {
            return draws;
        };
        if let (Some(tiles), Some(pipeline)) = (&self.tiles, self.pipelines.get(&targets)) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(2, &behind.group, &[]);
            // Each terrain's water's leaves, through its uniform and its
            // water's tree's, seen by the camera's view, the first.
            for (_, uniform, tree, leaves) in &tiles.terrains {
                pass.set_bind_group(1, group, &[0, *uniform, *tree]);
                leaves.record(pass);
                draws += 1;
            }
        }
        // Falls last, over the water they land in: each terrain's jets
        // through its uniform, by which they read the water under them, of
        // terrains drawing tiles' water this frame.
        if let (Some(falls), Some(tiles), Some(pipeline)) =
            (&self.falls, &self.tiles, self.fall_pipelines.get(&targets))
        {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(2, &behind.group, &[]);
            pass.set_vertex_buffer(0, falls.vertices.slice(..));
            pass.set_index_buffer(falls.indices.slice(..), wgpu::IndexFormat::Uint32);
            for (id, _, _, run) in &falls.built {
                let Some(&(_, uniform, _, _)) =
                    tiles.terrains.iter().find(|terrain| terrain.0 == *id)
                else {
                    continue;
                };
                if run.is_empty() {
                    continue;
                }
                pass.set_bind_group(1, group, &[0, uniform, 0]);
                pass.draw_indexed(run.clone(), 0, 0..1);
                draws += 1;
            }
        }
        draws
    }
}

impl RenderFeature for WaterPass {
    type Frame = WaterFrame;
    type Stats = (WaterStats, OceanStats);

    /// Water blends over the opaque scene, with its shadows and probes, on
    /// the leaves of its subdivisions, and writes its depth, which
    /// transparent surfaces and screen effects then see.
    fn declare(_frame: &WaterFrame, shape: &FrameShape, graph: &mut RenderGraph) {
        let mut water = graph
            .add(Pass::Feature(FeaturePass::Water))
            .read_if(shape.shadows, Resource::ShadowMaps)
            .read_if(shape.subdivision, Resource::Leaves)
            .read(Resource::Probes);
        for &target in shape.main_targets() {
            water = water.color(target, Access::Load);
        }
        water.depth(Resource::Depth, Access::Load);
    }

    fn encode(
        &self,
        _frame: &WaterFrame,
        node: &Node,
        encode: &Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        // The live water's steps and looks, the foam field's over the water
        // they leave, and what lies behind the water, before it draws.
        self.sim.encode(encoder, encode.profiler);
        if let (Some(_), Some((_, group))) = (&self.tiles, &self.group) {
            self.foam.encode(encoder, group, encode.profiler);
        }
        self.copy_behind(encoder, encode.depth, encode.scene);
        let mut pass = encode.render_pass(
            encoder,
            node,
            "engine/water",
            encode.profiler.span("water").map(Span::render),
        );
        self.draw(&mut pass, encode.water_camera, encode.targets)
    }

    /// Bytes of the copies, the falls' sheets, the live water and the foam
    /// field: tiles' water blocks and its subdivisions count with the
    /// terrain's.
    fn capacity_bytes(&self) -> u64 {
        let copies = self.behind.size().map_or(0, |size| {
            let texels = u64::from(size.width) * u64::from(size.height);
            let scene = self
                .behind
                .get()
                .is_some_and(|copies| copies.scene.is_some());
            texels * if scene { 8 } else { 4 }
        });
        let falls = self
            .falls
            .as_ref()
            .map_or(0, |falls| falls.vertices.size() + falls.indices.size());
        copies + falls + self.sim.capacity_bytes() + self.foam.capacity_bytes()
    }

    /// The leaves the water's subdivisions drew and the splits their room
    /// refused, as last read back; tiles' water blocks resident and
    /// dropped; the cells along a side of the live square, those the
    /// budget of its cell steps and the device's bindings cut from it, the
    /// steps it ran and those it fell behind the clock by, and the solids
    /// pushing its water with those standing in it past the most that do;
    /// the cells along a side of the foam field; and the sea's sectors drawn
    /// and their triangles.
    fn stats(
        this: Option<&Self>,
        frame: Option<&WaterFrame>,
        _snapshot: &RenderSnapshot,
    ) -> (WaterStats, OceanStats) {
        let [live_cells, live_budget_cut, live_cut, live_steps, live_behind] =
            frame.and(this).map_or([0; 5], |this| this.sim.cells());
        let [touches, dropped_touches] = frame.and(this).map_or([0; 2], |this| this.sim.touches());
        let [leaves, refused] = frame.map_or([0; 2], |frame| frame.subdivided);
        (
            WaterStats {
                leaves,
                refused,
                blocks: frame.map_or(0, |frame| frame.blocks[0]),
                dropped_blocks: frame.map_or(0, |frame| frame.blocks[1]),
                live_cells,
                live_budget_cut,
                live_cut,
                live_steps,
                live_behind,
                touches,
                dropped_touches,
                foam_cells: frame.and(this).map_or(0, |this| this.foam.cells()),
            },
            OceanStats {
                sectors: frame.map_or(0, |frame| frame.sectors),
                triangles: frame.map_or(0, |frame| frame.triangles),
            },
        )
    }
}

/// The generations of what tiles' water's group binds: the terrain pass's
/// arrays, uniforms, directory (with water's in it), leaves and trees'
/// uniforms and the water blocks; and the live water's looks.
type GroupGenerations = ([u64; 6], u64);

/// What a water pipeline draws: terrains' water, live or settled, the sea
/// or falls' sheets.
#[derive(Clone, Copy)]
enum Surface {
    Tiles,
    Sea,
    Falls,
}

/// The first `bytes` of `buffer`, bound at a dynamic offset.
fn sized(buffer: &wgpu::Buffer, bytes: usize) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(bytes as u64),
    })
}

/// The light of the sea around an eye at `eye` under its surface,
/// its waves at simulation time `seconds`, as the lighting uniform's
/// `underwater`: what its water scatters of the sun and sky, dimmed by the
/// water above the eye (rgb), and the meters through which half the light
/// passes (w); zero above the sea.
pub(crate) fn underwater(environment: &Environment, eye: Vec3, seconds: f64) -> [f32; 4] {
    let ocean = environment.ocean;
    if !ocean.enabled {
        return [0.0; 4];
    }
    let waves = ocean.spectrum();
    if eye.y > ocean.level + waves.amplitude() {
        return [0.0; 4];
    }
    let surface = waves.surface(f64::from(eye.x), f64::from(eye.z), seconds);
    let depth = surface.height - eye.y;
    if depth <= 0.0 {
        return [0.0; 4];
    }
    let atmosphere = &environment.atmosphere;
    let light = atmosphere.sun_color * max(atmosphere.sun_direction.normalize_or(Vec3::Y).y, 0.0)
        + atmosphere.zenith * atmosphere.ambient;
    // Water scatters back many times what its color says it returns of a
    // single bounce.
    let color = Vec3::from_array(ocean.color) * light * 3.0 * (-depth / ocean.clarity).exp2();
    [color.x, color.y, color.z, ocean.clarity]
}

/// Per meter, how fast water whose light halves over `clarity` meters and
/// which scatters back `color` takes each channel (rgb): a channel as fast
/// as the color's luminance exceeds the channel's share of it (at most
/// eight times), weighted so the light's luminance halves over the
/// clarity. The channels water's color holds least go first, so the
/// ground behind it tints toward that color, and water as gray as its
/// color takes all alike.
fn absorption(color: [f32; 3], clarity: f32) -> [f32; 4] {
    const LUMINANCE: [f32; 3] = [0.2126, 0.7152, 0.0722];
    let luminance = max(
        color[0] * LUMINANCE[0] + color[1] * LUMINANCE[1] + color[2] * LUMINANCE[2],
        1.0e-4,
    );
    let weights = color.map(|channel| luminance / max(channel, luminance * 0.125));
    let mean = weights[0] * LUMINANCE[0] + weights[1] * LUMINANCE[1] + weights[2] * LUMINANCE[2];
    let rate = std::f32::consts::LN_2 / max(clarity, 0.01) / mean;
    [weights[0] * rate, weights[1] * rate, weights[2] * rate, 0.0]
}

/// The water group's stand-ins for what the sea reads none of, and the
/// color's for frames blending water.
fn placeholders(device: &wgpu::Device) -> Placeholders {
    let array = |label, format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let scene = array(
        "engine/water-placeholder-scene",
        wgpu::TextureFormat::Rgba8Unorm,
    );
    Placeholders {
        scene: scene.create_view(&wgpu::TextureViewDescriptor::default()),
        surfaces: array("engine/water-placeholder", wgpu::TextureFormat::R32Float),
        info: array(
            "engine/water-placeholder-info",
            wgpu::TextureFormat::Rgba32Uint,
        ),
        views: crate::buffers::uniform(
            device,
            "engine/water-placeholder-view",
            size_of::<ViewUniform>() as u64,
        ),
        terrains: crate::buffers::uniform(
            device,
            "engine/water-placeholder-terrain",
            size_of::<TerrainUniform>() as u64,
        ),
        live: crate::buffers::storage(
            device,
            "engine/water-placeholder-live",
            16,
            wgpu::BufferUsages::empty(),
        ),
        storage: crate::buffers::storage(
            device,
            "engine/water-placeholder-storage",
            256,
            wgpu::BufferUsages::empty(),
        ),
        tree: crate::buffers::uniform(
            device,
            "engine/water-placeholder-tree",
            size_of::<SubdivisionUniform>() as u64,
        ),
    }
}

/// The sea's GPU state: its uniform, and its group binding `placeholders`
/// for what only tiles' water reads.
fn ocean_draw(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
    placeholders: &Placeholders,
) -> OceanDraw {
    let array = |texture: &wgpu::Texture| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        })
    };
    let (surfaces, info) = (array(&placeholders.surfaces), array(&placeholders.info));
    let uniform = crate::buffers::uniform(device, "engine/ocean", WATER_UNIFORM_BYTES as u64);
    let group = crate::pipelines::group(
        device,
        "engine/ocean-group",
        &WATER,
        layout,
        &[
            ("terrain_heights", crate::pipelines::view(&surfaces)),
            (
                "terrain_view",
                sized(&placeholders.views, size_of::<ViewUniform>()),
            ),
            (
                "terrain",
                sized(&placeholders.terrains, size_of::<TerrainUniform>()),
            ),
            crate::buffers::entry("terrain_directory", &placeholders.storage),
            crate::buffers::entry("terrain_surface_leaves", &placeholders.storage),
            ("water_surface", crate::pipelines::view(&surfaces)),
            ("water_info", crate::pipelines::view(&info)),
            ("water", uniform.as_entire_binding()),
            crate::buffers::entry("water_live", &placeholders.live),
            (
                "water_tree",
                sized(&placeholders.tree, size_of::<SubdivisionUniform>()),
            ),
        ],
    );
    OceanDraw {
        uniform,
        group,
        rings: 0,
        runs: Vec::new(),
    }
}
