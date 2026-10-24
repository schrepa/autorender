//! Water: terrains' water as their tiles hold it, and the sea. Each frame
//! the leaves of each terrain's subdivision over which its tiles' water may
//! stand are listed on the GPU, in their order (`meshes/water.wgsl`'s
//! `cs_water_leaves` and `cs_water_list`), and water draws on them: its
//! vertices stand where the ground's do, at the surfaces the tiles' water
//! blocks hold there, read by position through water's directory
//! (`crate::terrain::TerrainPass::water_draws`), so the depth each pixel
//! shades by is the tiles' water over the true ground, the shore falls where
//! the ground rises through the water, and none of it moves as the eye
//! does. Water draws after the opaque scene and its decals and
//! blends over the ground by how little light comes back through it,
//! writing the depth and surface targets, so later passes and screen-space
//! reflections see its surface. Before it draws, the opaque depth copies
//! into a texture of its own: what stands in the water behind a pixel
//! shortens the light's path to it. In frames drawing the effects' targets
//! the opaque color copies too, and water covers it whole with what shows
//! through it, bent by its waves and dimmed channel by channel over the
//! light's path; other frames blend water over the ground.
//!
//! The leaves' listing has room for as many leaves of each terrain's
//! subdivision as the most any held when their counts last read back, a
//! few frames late, within what one storage binding holds: past it, leaves
//! go untested and their water does not draw, counted, until the room
//! grows to hold them.
//!
//! Water's waves (`crate::water_waves`) evolve before it draws, the same
//! tiles of them for terrains' water and the sea.
//!
//! Where terrains' rivers leave their beds, their jets draw last, as
//! sheets (`crate::water_falls`) over the water they land in, writing their
//! depth where they cover the scene.
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
use std::cell::Cell;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;

use bytemuck::Zeroable;
use engine_core::glam::{Mat4, Vec3};
use engine_core::math::max;
use engine_core::terrain_data::Fall;
use engine_core::{
    Environment, MAX_OCEAN_WAVES, Ocean, OceanStats, RenderSnapshot, RenderTables, RenderTerrain,
    Water, WaterStats, Wind,
};

use crate::bindings::{WATER, WATER_BEHIND};
use crate::buffers::{Growable, ReadbackRing};
use crate::gpu_layouts::terrain::{TerrainUniform, ViewUniform};
use crate::profiling::{GpuProfiler, Span};
use crate::render_features::{Encode, FeaturePass, RenderFeature};
use crate::render_graph::{Access, FrameShape, Node, Pass, RenderGraph, Resource, Targets};
use crate::scene_shading::SceneShading;
use crate::shaders::entry::meshes;
use crate::targets::{FrameSize, ScreenTarget};
use crate::terrain::TerrainWaterDraws;
use crate::water_fall_map::FallMap;
use crate::water_falls::{fall_layout, sheets};
use crate::water_sim::WaterSim;
use crate::water_waves::WaterWaves;

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

/// Leaves of each terrain's subdivision the listing has room for until
/// their counts first read back.
const FIRST_LEAVES: u32 = 4096;
/// Workgroups the test of a terrain's leaves dispatches, each invocation
/// testing the leaves as many apart as they run.
const LEAF_WORKGROUPS: u32 = 256;
/// Bytes from one terrain's leaf record to the next: the dynamic offsets'
/// alignment every device allows (WebGPU's
/// `minUniformBufferOffsetAlignment`).
const RECORD_SLOT: u64 = 256;

use crate::gpu_layouts::water::{FallVertex, WaterDraw, WaterLeafRecord, WaterUniform};

/// Bytes of the water group's uniform.
pub(crate) const WATER_UNIFORM_BYTES: usize = size_of::<WaterUniform>();

/// One-texel stand-ins for what the water group binds that the sea reads
/// none of: the terrain's height array, uniforms and directory, its
/// subdivision's leaves, tiles' water blocks, water's directory, its
/// leaves listed and a terrain's record of them; and for the scene's color,
/// which frames blending water read none of. The leaves' listing binds the
/// storage stand-in for the leaves listed, which it writes.
struct Placeholders {
    surfaces: wgpu::Texture,
    info: wgpu::Texture,
    views: wgpu::Buffer,
    terrains: wgpu::Buffer,
    scene: wgpu::TextureView,
    sim: wgpu::Buffer,
    storage: wgpu::Buffer,
    record: wgpu::Buffer,
}

/// The sea on the GPU: its uniform and group, and this frame's rings and
/// the runs of sectors the camera sees.
struct OceanDraw {
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    rings: u32,
    runs: Vec<Range<u32>>,
}

/// Tiles' water this frame: each terrain drawing it, its uniform's offset
/// and its subdivision's draw (the arguments' buffer and byte offset), in
/// the order of its records, bits, leaves listed and draws.
struct TileDraws {
    terrains: Vec<(u32, wgpu::Buffer, u64)>,
}

/// Tiles' water's groups of the water layout: the one its draws bind, and
/// the one the listing of its leaves binds, a stand-in for the leaves
/// listed, which it writes through its own group.
struct TileGroups {
    draws: wgpu::BindGroup,
    listing: wgpu::BindGroup,
}

/// What lists the leaves of terrains' subdivisions over which water may
/// stand: its pipelines, the test's and the listing's; each terrain's
/// record of where its leaves lie, the bits of its leaves holding water,
/// the leaves listed and its draw; the leaves each terrain has room for and
/// the terrains there is room for, `generation` advancing as any buffer is
/// another; the group binding what the listing writes; and its draws'
/// counts read back a few frames late (`counts`: the most leaves a
/// terrain's subdivision held, the leaves drawn, and the leaves past the
/// room, left untested).
struct Leaves {
    pipelines: Option<[wgpu::ComputePipeline; 2]>,
    records: wgpu::Buffer,
    bits: Growable,
    list: Growable,
    draws: wgpu::Buffer,
    room: u32,
    terrains: u32,
    generation: u64,
    lists: Option<wgpu::BindGroup>,
    readbacks: ReadbackRing<(u32, u32)>,
    counts: Rc<Cell<[u32; 3]>>,
}

impl Leaves {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            pipelines: None,
            records: crate::buffers::uniform(device, "engine/water-leaf-records", RECORD_SLOT),
            bits: Growable::new(
                device,
                "engine/water-leaf-bits",
                16,
                wgpu::BufferUsages::empty(),
            ),
            list: Growable::new(
                device,
                "engine/water-leaf-list",
                16,
                wgpu::BufferUsages::empty(),
            ),
            draws: draws_buffer(device, 1),
            room: 0,
            terrains: 0,
            generation: 0,
            lists: None,
            readbacks: ReadbackRing::new("engine/water-leaf-readback"),
            counts: Rc::new(Cell::new([0; 3])),
        }
    }

    /// Room for `terrains` terrains' leaves, as many each as the most a
    /// terrain's subdivision held when their counts last read back, and at
    /// least [`FIRST_LEAVES`], within what one storage binding holds;
    /// their records written as the room changes. Returns whether any
    /// buffer is another.
    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, terrains: u32) -> bool {
        let terrains = terrains.max(1);
        let wanted = (self.counts.get()[0].max(FIRST_LEAVES))
            .checked_next_power_of_two()
            .unwrap_or(u32::MAX);
        // Whole words of bits within what the list's binding holds.
        let fits = self.list.room() / 4 / u64::from(terrains);
        let within = fits.min(u64::from(u32::MAX)) as u32 / 32 * 32;
        let room = wanted.max(self.room).min(within);
        let mut replaced = self
            .list
            .grow(device, u64::from(terrains) * u64::from(room) * 4);
        replaced |= self
            .bits
            .grow(device, u64::from(terrains) * u64::from(room / 8));
        if u64::from(terrains) * size_of::<WaterDraw>() as u64 > self.draws.size() {
            self.draws = draws_buffer(device, terrains.next_power_of_two());
            replaced = true;
        }
        let record_bytes = u64::from(terrains) * RECORD_SLOT;
        if record_bytes > self.records.size() {
            self.records = crate::buffers::uniform(
                device,
                "engine/water-leaf-records",
                u64::from(terrains.next_power_of_two()) * RECORD_SLOT,
            );
            replaced = true;
            self.terrains = 0;
        }
        if replaced {
            self.generation += 1;
        }
        if (room, terrains) != (self.room, self.terrains) {
            let mut bytes = vec![0_u8; record_bytes as usize];
            for index in 0..terrains {
                let record = WaterLeafRecord {
                    places: [index * room, index * room / 32, room, index],
                };
                let at = (u64::from(index) * RECORD_SLOT) as usize;
                bytes[at..at + size_of::<WaterLeafRecord>()]
                    .copy_from_slice(bytemuck::bytes_of(&record));
            }
            queue.write_buffer(&self.records, 0, &bytes);
            if terrains != self.terrains {
                // The draws read back are as many as the terrains.
                self.readbacks.clear();
            }
            (self.room, self.terrains) = (room, terrains);
        }
        replaced
    }

    /// Bytes of its buffers.
    fn capacity_bytes(&self) -> u64 {
        self.records.size() + self.bits.size() + self.list.size() + self.draws.size()
    }
}

/// Terrains' falls' sheets: their vertices and triangles' indices, and the
/// falls they were built from (each terrain's origin and falls).
struct FallDraws {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    count: u32,
    built: Vec<(Vec3, Arc<[Fall]>)>,
}

/// The copies of the opaque scene water reads, the depth and, in frames
/// drawing the effects' targets, the color; and their group.
struct BehindCopies {
    depth: wgpu::Texture,
    scene: Option<wgpu::Texture>,
    group: wgpu::BindGroup,
}

/// What water draws this frame: tiles' water blocks resident and dropped,
/// and the sea's sectors and their triangles.
pub(crate) struct WaterFrame {
    blocks: [u32; 2],
    sectors: u32,
    triangles: u32,
}

pub(crate) struct WaterPass {
    pub(crate) layout: crate::pipelines::GroupLayout,
    behind_layout: crate::pipelines::GroupLayout,
    behind: ScreenTarget<BehindCopies>,
    placeholders: Placeholders,
    // What groups of the water layout fill where a pass leaves bindings
    // out: the listing's outputs in draws' groups, what lies behind the
    // water in the listing's.
    fillers: crate::pipelines::Fillers,
    // Created with the first frame drawing water, the fall map once the
    // scene's tables hold it.
    waves: Option<WaterWaves>,
    fall_map: Option<FallMap>,
    // Water simulated around the camera, where water asks for it.
    sim: WaterSim,
    // Tiles' water's uniform, and the groups binding it with the terrain
    // pass's arrays, uniforms, directory and leaves, the simulation's water
    // and the leaves' listing's, as of their generations.
    uniform: wgpu::Buffer,
    group: Option<(GroupGenerations, TileGroups)>,
    tiles: Option<TileDraws>,
    leaves: Leaves,
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
    sim_pipelines: HashMap<Targets, wgpu::RenderPipeline>,
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
            fillers: crate::pipelines::Fillers::new(device),
            waves: None,
            fall_map: None,
            sim: WaterSim::new(device),
            uniform: crate::buffers::uniform(device, "engine/water", WATER_UNIFORM_BYTES as u64),
            group: None,
            tiles: None,
            leaves: Leaves::new(device),
            pipeline_layout: None,
            pipelines: HashMap::default(),
            ocean_pipelines: HashMap::default(),
            ocean: None,
            falls: None,
            fall_pipelines: HashMap::default(),
            fall_vertices: Vec::new(),
            fall_indices: Vec::new(),
            sim_pipelines: HashMap::default(),
            shading: SceneShading::ALL,
            water_views: false,
        }
    }

    /// Specializes the pipelines for `shading`'s scene features, and those
    /// drawing tiles' and simulated water into the debug targets for
    /// whether frames show the water's views (`water_views`,
    /// `crate::debug_view::shows_water`): those made for others are
    /// dropped, to be created anew.
    pub(crate) fn specialize(&mut self, shading: SceneShading, water_views: bool) {
        if self.shading != shading {
            self.shading = shading;
            self.pipelines.clear();
            self.ocean_pipelines.clear();
            self.fall_pipelines.clear();
            self.sim_pipelines.clear();
        }
        if self.water_views != water_views {
            self.water_views = water_views;
            self.pipelines.remove(&Targets::Debug);
            self.sim_pipelines.remove(&Targets::Debug);
        }
    }

    /// Builds the sheets of `terrains`' falls unless they were built from
    /// the same falls; drops them where none fall.
    fn prepare_falls(&mut self, device: &wgpu::Device, terrains: &[RenderTerrain]) {
        let falling = || terrains.iter().filter(|terrain| !terrain.falls.is_empty());
        let same = self.falls.as_ref().is_some_and(|built| {
            built.built.len() == falling().count()
                && built
                    .built
                    .iter()
                    .zip(falling())
                    .all(|(a, b)| a.0 == b.origin && Arc::ptr_eq(&a.1, &b.falls))
        });
        if same || (self.falls.is_none() && falling().next().is_none()) {
            return;
        }
        sheets(
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
            count: self.fall_indices.len() as u32,
            built: falling()
                .map(|terrain| (terrain.origin, terrain.falls.clone()))
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

    /// Prepares this frame's water: tiles' water over the leaves of the
    /// terrains' subdivisions as the terrain pass holds them (`tiles`) and
    /// the falls of `terrains`, shaded as `settings` say with the waves'
    /// spectra and the fall map the scene's `tables` hold, and
    /// the sea `ocean`, culled through `cull` and drawn through `view`
    /// (both eye-relative) over the opaque `depth` and, when `refracts`,
    /// the color it copies; seen from `eye` in the wind `wind`, its waves
    /// at the simulation time `seconds` the frame presents; `None` when
    /// none of it draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (tiles, terrains): (Option<TerrainWaterDraws<'_>>, &[RenderTerrain]),
        (settings, ocean, tables): (Water, Ocean, &RenderTables),
        (eye, wind, seconds): (Vec3, Wind, f64),
        (cull, view, depth, refracts): (Mat4, Mat4, &wgpu::Texture, bool),
    ) -> Option<WaterFrame> {
        self.tiles = None;
        let blocks = tiles.as_ref().map_or([0; 2], |tiles| tiles.blocks);
        let tiles = tiles.filter(|tiles| settings.enabled && !tiles.terrains.is_empty());
        self.prepare_falls(device, if settings.enabled { terrains } else { &[] });
        // Falls draw with tiles' water's group, once any tiles' water has.
        let falling = self.falls.is_some() && (tiles.is_some() || self.group.is_some());
        if tiles.is_none() && !ocean.enabled && !falling {
            self.ocean = None;
            return None;
        }
        // Terrains' water and the sea ripple with the same waves.
        let waves = self.waves.get_or_insert_with(|| WaterWaves::new(device));
        waves.prepare(device, queue, tables.waves.as_ref(), seconds);
        let repeats = waves.repeats();
        if let (None, Some(map)) = (&self.fall_map, &tables.fall_map) {
            self.fall_map = Some(FallMap::new(device, queue, &map.table));
        }
        // Rivers simulated where a terrain's window stands.
        let window = terrains
            .iter()
            .find_map(|terrain| terrain.water_window.as_ref())
            .filter(|_| settings.enabled);
        self.sim.prepare(device, queue, window, seconds);
        let (simulated, sim_generation) = self.sim.drawn();
        // The square of the window that runs: its first corner, the meters
        // across a cell and its cells along a side, and its datum.
        let simulated = simulated.map(|drawn| {
            let window = drawn.window;
            let square = [
                window.origin[0],
                window.origin[1],
                window.cell,
                window.cells as f32,
            ];
            (square, [window.datum, 0.0, 0.0, 0.0])
        });
        if let Some(tiles) = tiles {
            self.prepare_tiles(device, queue, &tiles, sim_generation);
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
                ripples: ripples(settings, wind),
                foam: [settings.foam, settings.foam_width, 0.0, 0.0],
                absorption: absorption(settings.color, settings.clarity),
                waves: repeats,
                ocean: sea,
                sim: simulated.map_or([0.0; 4], |(square, _)| square),
                sim_datum: simulated.map_or([0.0; 4], |(_, datum)| datum),
                sediment: [sediment[0], sediment[1], sediment[2], settings.sediment],
                ..WaterUniform::zeroed()
            };
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        }
        let sea = self.prepare_ocean(
            device,
            queue,
            (cull, view),
            (settings, ocean),
            (eye, wind, seconds),
            repeats,
        );
        if self.tiles.is_none() && !sea && !falling {
            return None;
        }
        // The copies, at the depth buffer's size: the frame's.
        let size = FrameSize {
            width: depth.width(),
            height: depth.height(),
        };
        let refracted = self
            .behind
            .get()
            .is_some_and(|copies| copies.scene.is_some());
        if refracted != refracts {
            self.behind.release();
        }
        if self.behind.resizes(size) {
            let copies = self.behind_copies(device, size, depth.format(), refracts);
            self.behind.set(size, copies);
        }
        let (sectors, triangles) = self.ocean_drawn();
        Some(WaterFrame {
            blocks,
            sectors,
            triangles,
        })
    }

    /// Prepares tiles' water over `tiles`: room for each terrain's leaves
    /// and the group their listing writes them through, and the groups of
    /// the water layout tiles' water and its listing bind, made again as
    /// what they bind changes (the simulation's water as of
    /// `sim_generation` among it).
    fn prepare_tiles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tiles: &TerrainWaterDraws<'_>,
        sim_generation: u64,
    ) {
        let replaced = self
            .leaves
            .prepare(device, queue, tiles.terrains.len() as u32);
        if replaced || self.leaves.lists.is_none() {
            self.leaves.lists = Some(crate::pipelines::filled_group(
                device,
                "engine/water-leaf-lists",
                &WATER_BEHIND,
                &self.behind_layout,
                &[
                    crate::buffers::entry("water_leaf_bits", &self.leaves.bits),
                    crate::buffers::entry("water_leaf_list", &self.leaves.list),
                    crate::buffers::entry("water_leaf_draws", &self.leaves.draws),
                ],
                &self.fillers,
            ));
        }
        let generations = (tiles.generation, sim_generation, self.leaves.generation);
        if let (Some(waves), Some(map)) = (&self.waves, &self.fall_map)
            && self
                .group
                .as_ref()
                .is_none_or(|(bound, _)| *bound != generations)
        {
            let groups = TileGroups {
                draws: self.tile_group(
                    device,
                    ("engine/water-group", tiles),
                    (waves, map),
                    &self.leaves.list,
                ),
                listing: self.tile_group(
                    device,
                    ("engine/water-listing-group", tiles),
                    (waves, map),
                    &self.placeholders.storage,
                ),
            };
            self.group = Some((generations, groups));
        }
        self.tiles = Some(TileDraws {
            terrains: (tiles.terrains.iter())
                .map(|&(uniform, draw, offset)| (uniform, draw.clone(), offset))
                .collect(),
        });
    }

    /// The copies of the opaque scene at `size`, the depth's of `format`
    /// and, when water `refracts`, the color's; and their group, which
    /// fills what the leaves' listing writes.
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
        let group = crate::pipelines::filled_group(
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
            ],
            &self.fillers,
        );
        BehindCopies {
            depth,
            scene,
            group,
        }
    }

    /// The group `label` of the water layout tiles' water binds: the
    /// terrain pass's height array, views' and terrains' uniforms,
    /// directory and leaves of its subdivision, and the water blocks and
    /// water's directory (`tiles`); water's uniform, `waves` and falling
    /// water's `map`; the simulated water; and `listed` as the leaves
    /// listed, with the terrains' records of them.
    fn tile_group(
        &self,
        device: &wgpu::Device,
        (label, tiles): (&str, &TerrainWaterDraws<'_>),
        (waves, map): (&WaterWaves, &FallMap),
        listed: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let (moments, sampler) = waves.sampled();
        let (sim_cells, sim_ground, sim_beds) = self.sim.drawn().0.map_or(
            (
                &self.placeholders.sim,
                &self.placeholders.sim,
                &self.placeholders.sim,
            ),
            |drawn| (drawn.cells, drawn.ground, drawn.beds),
        );
        crate::pipelines::group(
            device,
            label,
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
                ("water_waves", crate::pipelines::view(moments)),
                ("water_wave_sampler", crate::pipelines::sampler(sampler)),
                ("water_fall_map", crate::pipelines::view(map.view())),
                crate::buffers::entry("water_sim_cells", sim_cells),
                crate::buffers::entry("water_sim_ground", sim_ground),
                crate::buffers::entry("water_sim_beds", sim_beds),
                crate::buffers::entry("water_directory", tiles.water_directory),
                crate::buffers::entry("water_leaves", listed),
                (
                    "water_leaf_record",
                    sized(&self.leaves.records, size_of::<WaterLeafRecord>()),
                ),
            ],
        )
    }

    /// Prepares the sea culled through `cull` and drawn through `view`
    /// (both eye-relative) from `eye`, its ripples as `water`'s in `wind`, the
    /// waves' tiles repeating `repeats` times a meter, its own waves at
    /// simulation time `seconds`; returns whether any of it draws.
    fn prepare_ocean(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (cull, view): (Mat4, Mat4),
        (water, ocean): (Water, Ocean),
        (eye, wind, seconds): (Vec3, Wind, f64),
        repeats: [f32; 4],
    ) -> bool {
        let (true, Some(waves), Some(map)) = (ocean.enabled, &self.waves, &self.fall_map) else {
            self.ocean = None;
            return false;
        };
        let draw = self.ocean.get_or_insert_with(|| {
            ocean_draw(device, &self.layout, &self.placeholders, (waves, map))
        });
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
        // to the power 3.41 (Monahan's fit; about 2 % at 12 m/s), twice it
        // at full foam: crests foam where the waves compress the surface
        // more than that share of it, which the spectrum's shape fixes.
        let speed = ocean.wind[0].hypot(ocean.wind[1]);
        let share = 3.84e-6 * speed.powf(3.41) * 2.0 * ocean.foam;
        let crests = 0.67 + 0.069 * (max(share, 1.0e-4) / 0.01).ln();
        let heading = [ocean.wind[0] / speed, ocean.wind[1] / speed];
        let uniform = WaterUniform {
            relative_view_projection: view.to_cols_array_2d(),
            origin: [heading[0], heading[1], speed, 0.0],
            color: [red, green, blue, ocean.clarity],
            ripples: ripples(water, wind),
            // Shores foam as wide as the waves rise.
            foam: [ocean.foam, 0.4 + amplitude, crests, amplitude],
            absorption: absorption(ocean.color, ocean.clarity),
            waves: repeats,
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
            // The sea carries no sediment.
            sediment: [0.0; 4],
        };
        queue.write_buffer(&draw.uniform, 0, bytemuck::bytes_of(&uniform));
        !draw.runs.is_empty()
    }

    /// Creates the pipelines into `targets` this frame's water and sea
    /// draw with unless they exist, and tiles' water's and its leaves'
    /// listing's when `coming`, before a bake gives any; `camera` is the
    /// main passes' first group.
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
        if (coming || self.tiles.is_some()) && self.leaves.pipelines.is_none() {
            let layout = self.pipeline_layout(device, camera);
            self.leaves.pipelines = Some([
                crate::pipelines::compute_entry(device, &layout, shader, meshes::cs_water_leaves),
                crate::pipelines::compute_entry(device, &layout, shader, meshes::cs_water_list),
            ]);
        }
        if self.ocean.is_some() && !self.ocean_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Sea);
            self.ocean_pipelines.insert(targets, pipeline);
        }
        if self.falls.is_some() && !self.fall_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Falls);
            self.fall_pipelines.insert(targets, pipeline);
        }
        if self.sim.drawn().0.is_some() && !self.sim_pipelines.contains_key(&targets) {
            let pipeline = self.pipeline(device, shader, camera, targets, formats, Surface::Sim);
            self.sim_pipelines.insert(targets, pipeline);
        }
    }

    /// The layout water's pipelines and its leaves' listing share, its
    /// first group `camera`'s, made once.
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
            Surface::Sim => crate::entries::sim_water(targets),
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
        // Tiles' and simulated water draw into the debug targets as the
        // view shown asks.
        let constants = match (targets, surface) {
            (Targets::Debug, Surface::Tiles | Surface::Sim) => self
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
                    Surface::Sim => "engine/water-sim-draw",
                },
                layout: &layout,
                module: shader,
                vertex_constants: &[],
                vertex,
                // Tiles' water's vertices come from the leaves listed, the
                // sea's from their indices alone.
                buffers: match surface {
                    Surface::Tiles | Surface::Sea | Surface::Sim => &[],
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

    /// Records the listing of this frame's terrains' water's leaves, seen
    /// from the main passes' first group `camera` and timed by `profiler`:
    /// each terrain's leaves as its subdivision's draw counts them, copied
    /// into its own draw; their bits cleared and set where water may stand
    /// over a leaf, and those leaves listed in their order; and a copy of
    /// the draws to read back.
    fn list_leaves(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        (camera, profiler): (&wgpu::BindGroup, &GpuProfiler),
    ) {
        let leaves = &self.leaves;
        let (Some(tiles), Some((_, groups)), Some([test, list]), Some(lists)) =
            (&self.tiles, &self.group, &leaves.pipelines, &leaves.lists)
        else {
            return;
        };
        let stride = size_of::<WaterDraw>() as u64;
        let counts = std::mem::offset_of!(WaterDraw, counts) as u64;
        // Its subdivision's draw's instances, a leaf each.
        let instances = std::mem::offset_of!(wgpu::util::DrawIndirectArgs, instance_count) as u64;
        let word = size_of::<u32>() as u64;
        for (index, (_, draw, offset)) in tiles.terrains.iter().enumerate() {
            let at = index as u64 * stride + counts;
            encoder.copy_buffer_to_buffer(draw, offset + instances, &leaves.draws, at, word);
        }
        encoder.clear_buffer(&leaves.bits, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/water-leaves"),
                timestamp_writes: profiler.span("water_leaves").map(|span| span.compute()),
            });
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(2, lists, &[]);
            // Every terrain's leaves tested, then each terrain's listed.
            for (pipeline, workgroups) in [(test, LEAF_WORKGROUPS), (list, 1)] {
                pass.set_pipeline(pipeline);
                for (index, (uniform, _, _)) in tiles.terrains.iter().enumerate() {
                    pass.set_bind_group(1, &groups.listing, &[0, *uniform, record_offset(index)]);
                    pass.dispatch_workgroups(workgroups, 1, 1);
                }
            }
        }
        let bytes = tiles.terrains.len() as u64 * stride;
        let tag = (leaves.room, tiles.terrains.len() as u32);
        leaves
            .readbacks
            .record(device, encoder, &leaves.draws, bytes, tag);
    }

    /// Whether tiles' water draws this frame: its leaves' listing ready.
    fn lists(&self) -> bool {
        self.tiles.is_some() && self.leaves.pipelines.is_some() && self.leaves.lists.is_some()
    }

    /// Records this frame's water into `pass` with the main passes' first
    /// group `camera`; returns its draws.
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
        let Some((_, groups)) = &self.group else {
            return draws;
        };
        if let (Some(tiles), Some(pipeline), true) =
            (&self.tiles, self.pipelines.get(&targets), self.lists())
        {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(2, &behind.group, &[]);
            // Each terrain's leaves listed, through its uniform and record,
            // seen by the camera's view, the first.
            for (index, (uniform, _, _)) in tiles.terrains.iter().enumerate() {
                pass.set_bind_group(1, &groups.draws, &[0, *uniform, record_offset(index)]);
                let offset = index as u64 * size_of::<WaterDraw>() as u64;
                pass.draw_indirect(&self.leaves.draws, offset);
                draws += 1;
            }
        }
        // The simulated water, its cells' middles drawn from its buffers
        // through the camera's view.
        if let (Some(drawn), Some(pipeline)) =
            (self.sim.drawn().0, self.sim_pipelines.get(&targets))
        {
            let quads = drawn.window.cells.saturating_sub(1);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, &groups.draws, &[0, 0, 0]);
            pass.set_bind_group(2, &behind.group, &[]);
            pass.set_index_buffer(drawn.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..quads * quads * 6, 0, 0..1);
            draws += 1;
        }
        // Falls last, over the water they land in: of its group they read
        // only water's uniform and waves.
        if let (Some(falls), Some(pipeline)) = (&self.falls, self.fall_pipelines.get(&targets)) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, camera, &[]);
            pass.set_bind_group(1, &groups.draws, &[0, 0, 0]);
            pass.set_bind_group(2, &behind.group, &[]);
            pass.set_vertex_buffer(0, falls.vertices.slice(..));
            pass.set_index_buffer(falls.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..falls.count, 0, 0..1);
            draws += 1;
        }
        draws
    }
}

impl RenderFeature for WaterPass {
    type Frame = WaterFrame;
    type Stats = (WaterStats, OceanStats);

    /// Water blends over the opaque scene, with its shadows and probes, and
    /// writes its depth, which transparent surfaces and screen effects then
    /// see.
    fn declare(_frame: &WaterFrame, shape: &FrameShape, graph: &mut RenderGraph) {
        let mut water = graph
            .add(Pass::Feature(FeaturePass::Water))
            .read_if(shape.shadows, Resource::ShadowMaps)
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
        // The waves, the simulated water's steps, the leaves tiles' water
        // draws on and what lies behind the water, before it draws.
        if let Some(waves) = &self.waves {
            waves.encode(encoder, encode.profiler);
        }
        self.sim.encode(encoder, encode.profiler);
        if self.lists() {
            self.list_leaves(encode.device, encoder, (encode.camera, encode.profiler));
        }
        self.copy_behind(encoder, encode.depth, encode.scene);
        let mut pass = encode.render_pass(
            encoder,
            node,
            "engine/water",
            encode.profiler.span("water").map(Span::render),
        );
        self.draw(&mut pass, encode.camera, encode.targets)
    }

    /// Requests the map of this frame's leaves' draws once it is submitted.
    fn after_submit(&mut self, _frame: Option<&WaterFrame>) {
        let counts = Rc::clone(&self.leaves.counts);
        self.leaves
            .readbacks
            .after_submit(move |words, (room, terrains)| {
                let stride = size_of::<WaterDraw>() / 4;
                let instances = std::mem::offset_of!(WaterDraw, args) / 4 + 1;
                let leaves = std::mem::offset_of!(WaterDraw, counts) / 4;
                let mut read = [0_u32; 3];
                for index in 0..terrains as usize {
                    let held = words.get(index * stride + leaves);
                    read[0] = read[0].max(held);
                    read[1] = read[1].saturating_add(words.get(index * stride + instances));
                    read[2] = read[2].saturating_add(held.saturating_sub(room));
                }
                counts.set(read);
            });
    }

    /// Bytes of the copies, the waves and the leaves' listing: tiles'
    /// water blocks count with the terrain's arrays.
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
        copies
            + falls
            + self.leaves.capacity_bytes()
            + self.sim.capacity_bytes()
            + self.waves.as_ref().map_or(0, WaterWaves::capacity_bytes)
            + self
                .fall_map
                .as_ref()
                .map_or(0, |_| FallMap::capacity_bytes())
    }

    /// The leaves tiles' water drew and those its listing left untested, as
    /// the last of their counts read back; tiles' water blocks resident and
    /// dropped; the cells along a side of the simulated square; and the
    /// sea's sectors drawn and their triangles.
    fn stats(
        this: Option<&Self>,
        frame: Option<&WaterFrame>,
        _snapshot: &RenderSnapshot,
    ) -> (WaterStats, OceanStats) {
        let simulated = frame
            .and(this)
            .and_then(|this| this.sim.drawn().0)
            .map_or(0, |drawn| drawn.window.cells);
        let [_, leaves, dropped_leaves] = frame
            .and(this)
            .map_or([0; 3], |this| this.leaves.counts.get());
        (
            WaterStats {
                leaves,
                dropped_leaves,
                blocks: frame.map_or(0, |frame| frame.blocks[0]),
                dropped_blocks: frame.map_or(0, |frame| frame.blocks[1]),
                simulated_cells: simulated,
            },
            OceanStats {
                sectors: frame.map_or(0, |frame| frame.sectors),
                triangles: frame.map_or(0, |frame| frame.triangles),
            },
        )
    }
}

/// The generations of what water's groups bind: the terrain pass's arrays,
/// uniforms, directory and leaves, the water blocks and water's directory;
/// the simulation's water; and the leaves' listing's buffers.
type GroupGenerations = ([u64; 6], u64, u64);

/// What a water pipeline draws: tiles' water, the sea, falls' sheets or
/// simulated water.
#[derive(Clone, Copy)]
enum Surface {
    Tiles,
    Sea,
    Falls,
    Sim,
}

/// The first `bytes` of `buffer`, bound at a dynamic offset.
fn sized(buffer: &wgpu::Buffer, bytes: usize) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset: 0,
        size: wgpu::BufferSize::new(bytes as u64),
    })
}

/// The dynamic offset of the leaf record of the `index`th terrain drawing
/// tiles' water.
fn record_offset(index: usize) -> u32 {
    (index as u64 * RECORD_SLOT) as u32
}

/// The draws of `terrains` terrains' water, which the listing of their
/// leaves writes, the pass draws from and the read-backs copy: each draws
/// nothing until its terrain's leaves are listed.
fn draws_buffer(device: &wgpu::Device, terrains: u32) -> wgpu::Buffer {
    let none = WaterDraw {
        args: [0; 4],
        counts: [0; 4],
    };
    wgpu::util::DeviceExt::create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("engine/water-leaf-draws"),
            contents: bytemuck::cast_slice(&vec![none; terrains as usize]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        },
    )
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

/// The water uniform's ripples: their strength in `water`, and how strongly
/// `wind` gusts and its velocity.
fn ripples(water: Water, wind: Wind) -> [f32; 4] {
    [
        water.ripples,
        wind.gusts,
        wind.velocity[0],
        wind.velocity[1],
    ]
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
        sim: crate::buffers::storage(
            device,
            "engine/water-placeholder-sim",
            16,
            wgpu::BufferUsages::empty(),
        ),
        storage: crate::buffers::storage(
            device,
            "engine/water-placeholder-storage",
            256,
            wgpu::BufferUsages::empty(),
        ),
        record: crate::buffers::uniform(
            device,
            "engine/water-placeholder-record",
            size_of::<WaterLeafRecord>() as u64,
        ),
    }
}

/// The sea's GPU state: its uniform, and its group binding `waves`,
/// falling water's `map` and `placeholders` for what only tiles' water
/// reads.
fn ocean_draw(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
    placeholders: &Placeholders,
    (waves, map): (&WaterWaves, &FallMap),
) -> OceanDraw {
    let (moments, sampler) = waves.sampled();
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
            ("water_waves", crate::pipelines::view(moments)),
            ("water_wave_sampler", crate::pipelines::sampler(sampler)),
            ("water_fall_map", crate::pipelines::view(map.view())),
            crate::buffers::entry("water_sim_cells", &placeholders.sim),
            crate::buffers::entry("water_sim_ground", &placeholders.sim),
            crate::buffers::entry("water_sim_beds", &placeholders.sim),
            crate::buffers::entry("water_directory", &placeholders.storage),
            crate::buffers::entry("water_leaves", &placeholders.storage),
            (
                "water_leaf_record",
                sized(&placeholders.record, size_of::<WaterLeafRecord>()),
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
