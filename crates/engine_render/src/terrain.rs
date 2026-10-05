//! Terrain (decision 051): every resident tile's heights in one texture
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
//! long shadows through their horizon maps (decision 052), only the near
//! cascades draw terrain, near their receivers.
//!
//! Each tile's normals are computed once on upload into a parallel array;
//! fragments read them through a directory of every level's resident tiles,
//! from the finest level loaded at their distance whatever level the
//! geometry draws at, so light and materials hold still as levels of
//! detail hand over.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use engine_core::glam::{DVec3, Mat4, Vec3, Vec4};
use engine_core::{RenderTerrain, RenderTerrains, TileKey};
use wgpu::util::DeviceExt;

use crate::bindings::{SHADOW_VIEW, TERRAIN, TERRAIN_NORMALS};
use crate::cached::Cached;
use crate::mipmaps::MipGenerator;
use crate::render_graph::Targets;
use crate::terrain_horizons::Horizons;
use crate::terrain_maps::TerrainMaps;

/// Vertices along a patch side, and the grid's before the skirts'.
const GRID: u32 = engine_core::terrain_data::PATCH_QUADS + 1;
const GRID_VERTICES: u32 = GRID * GRID;
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
/// Each tile's control values (decision 054): its samples' two strongest
/// materials, their blend and biome, read texel by texel.
const CONTROL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
/// Surface materials every terrain together can show: as many as fit the
/// 16 KiB uniform binding every device allows.
const PALETTE_ENTRIES: usize = 16_384 / size_of::<PaletteEntry>();
/// Bytes of the palette.
pub(crate) const PALETTE_BYTES: usize = PALETTE_ENTRIES * size_of::<PaletteEntry>();
/// A directory entry of a tile not resident.
const NOT_RESIDENT: u32 = u32::MAX;

/// A patch as `PatchInput` reads it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct PatchRecord {
    first: [i32; 2],
    shape: [f32; 4],
    atlas: [u32; 4],
    holes: u32,
}

/// `PatchRecord::holes` of tiles without holes.
const NO_HOLES: u32 = u32::MAX;
/// 32-bit words of a tile's hole bits.
const HOLE_WORDS: usize = ((LAYER_SIZE - 2) * (LAYER_SIZE - 2)).div_ceil(32) as usize;

/// WGSL `TerrainView`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct ViewUniform {
    relative_view_projection: [[f32; 4]; 4],
    eye_offset: [f32; 4],
}

/// WGSL `TerrainUniform`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
struct TerrainUniform {
    anchor: [f32; 4],
    eye: [f32; 4],
    origin: [f32; 4],
    directory: [u32; 4],
}

/// WGSL `TerrainPaletteEntry`: a surface material's linear color (its
/// base color factor with maps) and roughness; its tiling, height
/// contrast, projection and maps' layer plus one (0 without maps); its
/// normal scale, occlusion strength, metalness and mean occlusion; and its
/// maps' mean color and roughness, its factors applied.
type PaletteEntry = [f32; 16];

/// One terrain's draws this frame: its uniform slot, the camera's patches,
/// every patch (for probe faces) and each shadow map's. The camera's and
/// every patch's ranges end with the patches with holes, which draw with
/// their own pipelines: as many as `*_holes`.
pub(crate) struct FrameTerrain {
    pub uniform: u32,
    pub camera: Range<u32>,
    pub camera_holes: u32,
    pub all: Range<u32>,
    pub all_holes: u32,
    pub maps: Vec<(usize, Range<u32>)>,
}

impl FrameTerrain {
    /// `range` without and with holes.
    pub(crate) fn split(range: &Range<u32>, holes: u32) -> [Range<u32>; 2] {
        let middle = range.end - holes;
        [range.start..middle, middle..range.end]
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
    /// sun, for horizon maps (decision 052).
    pub settings: engine_core::TerrainSettings,
    pub shadows: bool,
    pub sun: Vec3,
}

struct Atlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    control: wgpu::Texture,
    control_view: wgpu::TextureView,
    normals: wgpu::Texture,
    normals_view: wgpu::TextureView,
    layers: u32,
}

pub(crate) struct TerrainPass {
    layout: wgpu::BindGroupLayout,
    atlas: Option<Atlas>,
    atlas_generation: u64,
    slots: HashMap<(u64, TileKey), u32>,
    free: Vec<u32>,
    next_layer: u32,
    /// Tiles revision each terrain's layers were last synced to.
    synced: HashMap<u64, u64>,
    indices: wgpu::Buffer,
    index_count: u32,
    instances: wgpu::Buffer,
    views: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    uniform_generation: u64,
    // Hole bits of the tiles with holes, and where each tile's start.
    holes: wgpu::Buffer,
    hole_offsets: HashMap<(u64, TileKey), u32>,
    holes_generation: u64,
    group: Cached<(u64, u64, u64, u64, u64), wgpu::BindGroup>,
    pipelines: HashMap<(Targets, bool, bool), wgpu::RenderPipeline>,
    probe_pipelines: HashMap<(bool, bool), wgpu::RenderPipeline>,
    shadow: Option<wgpu::RenderPipeline>,
    records: Vec<PatchRecord>,
    written_views: Option<ViewUniform>,
    /// The eye this frame's terrain positions are relative to.
    eye: Vec3,
    pub(crate) frame: TerrainFrame,
    /// Tiles uploaded this frame, resident on the GPU, and dropped for want
    /// of layers.
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    /// Long sun shadows (decision 052).
    pub(crate) horizons: Horizons,
    // Normal maps of tiles uploaded this frame (layer, spacing), the pass
    // computing them, and their sampler.
    normal_jobs: Vec<(u32, f32)>,
    normal_pass: Option<(wgpu::BindGroupLayout, wgpu::ComputePipeline)>,
    normal_sampler: wgpu::Sampler,
    // Every terrain's levels and their resident tiles' layers, and where
    // each terrain's start.
    directory: wgpu::Buffer,
    directory_written: Vec<u32>,
    directory_generation: u64,
    directory_bases: Vec<u32>,
    // Every terrain's surface materials (decision 054), where each
    // terrain's start, and what the buffer holds.
    palette: wgpu::Buffer,
    palette_bases: Vec<u32>,
    palette_written: Vec<PaletteEntry>,
    /// Surface materials past the palette's capacity, which show the
    /// palette's first instead.
    pub(crate) dropped_materials: u32,
    /// Surface materials' maps (decision 054).
    pub(crate) maps: TerrainMaps,
}

impl TerrainPass {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let indices = grid_indices();
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("engine/terrain-indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            layout: crate::pipelines::layout(device, &TERRAIN),
            atlas: None,
            atlas_generation: 0,
            slots: HashMap::new(),
            free: Vec::new(),
            next_layer: 0,
            synced: HashMap::new(),
            indices: index_buffer,
            index_count: indices.len() as u32,
            instances: instance_buffer(device, 256),
            views: crate::buffers::uniform(device, "engine/terrain-views", VIEWS * SLOT),
            uniforms: crate::buffers::uniform(device, "engine/terrain-uniforms", SLOT),
            uniform_generation: 0,
            holes: crate::buffers::storage(
                device,
                "engine/terrain-holes",
                16,
                wgpu::BufferUsages::empty(),
            ),
            hole_offsets: HashMap::new(),
            holes_generation: 0,
            group: Cached::default(),
            pipelines: HashMap::new(),
            probe_pipelines: HashMap::new(),
            shadow: None,
            records: Vec::new(),
            written_views: None,
            eye: Vec3::ZERO,
            frame: TerrainFrame::default(),
            uploads: 0,
            dropped: 0,
            horizons: Horizons::new(device),
            normal_jobs: Vec::new(),
            normal_pass: None,
            normal_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/terrain-normal-sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            directory: crate::buffers::storage(
                device,
                "engine/terrain-directory",
                16,
                wgpu::BufferUsages::empty(),
            ),
            directory_written: Vec::new(),
            directory_generation: 0,
            directory_bases: Vec::new(),
            palette: crate::buffers::uniform(
                device,
                "engine/terrain-palette",
                PALETTE_BYTES as u64,
            ),
            palette_bases: Vec::new(),
            palette_written: Vec::new(),
            dropped_materials: 0,
            maps: TerrainMaps::new(device, queue),
        }
    }

    /// The eye this frame's terrain positions are relative to.
    pub(crate) fn eye(&self) -> Vec3 {
        self.eye
    }

    /// Layers in use.
    pub(crate) fn resident(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Bytes of the height, normal and control arrays, the horizon maps
    /// and the surface materials' maps.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.atlas.as_ref().map_or(0, |atlas| {
            u64::from(atlas.layers) * u64::from(LAYER_SIZE * LAYER_SIZE) * 12
        }) + self.horizons.capacity_bytes()
            + self.maps.capacity_bytes()
    }

    /// Records this frame's horizon map builds (decision 052).
    pub(crate) fn encode_horizons(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &crate::profiling::GpuProfiler,
    ) {
        let heights = self.atlas.as_ref().map(|atlas| &atlas.view);
        self.horizons.encode(device, encoder, heights, profiler);
    }

    /// Uploads tiles the frame's terrains list and the GPU lacks, releases
    /// those they no longer list, and lays out this frame's patches for the
    /// camera, reflection probe faces and the shadow maps of `views`.
    /// `max_layers` is the device's texture array layers; `mips` renders
    /// surface materials' mip levels. Returns whether any terrain draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        views: &TerrainViews<'_>,
        (max_layers, mips): (u32, &MipGenerator),
    ) -> bool {
        self.uploads = 0;
        self.frame = TerrainFrame::default();
        self.eye = views.eye;
        self.sync_tiles(device, queue, terrains, max_layers);
        self.write_directory(device, queue, terrains);
        self.maps.prepare(device, queue, mips, terrains);
        self.write_palette(queue, terrains);
        self.horizons.prepare(
            device,
            queue,
            terrains,
            (&views.settings, views.shadows),
            views.sun,
            &self.slots,
        );
        if terrains
            .terrains
            .iter()
            .all(|terrain| terrain.patches.is_empty())
        {
            return false;
        }
        // Uniforms: one slot per terrain.
        let uniform_bytes = terrains.terrains.len() as u64 * SLOT;
        if uniform_bytes > self.uniforms.size() {
            self.uniforms = crate::buffers::uniform(
                device,
                "engine/terrain-uniforms",
                uniform_bytes.next_power_of_two(),
            );
            self.uniform_generation += 1;
        }
        let eye = views.eye.as_dvec3();
        self.records.clear();
        let mut uniforms = vec![0_u8; uniform_bytes as usize];
        for (index, terrain) in terrains.terrains.iter().enumerate() {
            // What the terrain shows: 0 shaded, then its development views.
            let view = match terrains.view {
                engine_core::TerrainView::Shaded => 0.0,
                engine_core::TerrainView::Levels => 1.0,
                engine_core::TerrainView::Normals => 2.0,
                engine_core::TerrainView::Tiles => 3.0,
                engine_core::TerrainView::Materials => 4.0,
            };
            let directory = [
                self.directory_bases.get(index).copied().unwrap_or(0),
                terrain.pyramid.levels(),
                self.palette_bases.get(index).copied().unwrap_or(0),
                0,
            ];
            let (uniform, anchors) = terrain_uniform(terrain, eye, view, directory);
            let at = index * SLOT as usize;
            uniforms[at..at + size_of::<TerrainUniform>()]
                .copy_from_slice(bytemuck::bytes_of(&uniform));
            let origin = terrain.origin;
            let relative_origin = (terrain.origin.as_dvec3() - eye).as_vec3();
            let camera_planes = planes(views.culling, false);
            let start = self.records.len() as u32;
            let mut all = Vec::new();
            for patch in &terrain.patches {
                let Some(record) = self.record(terrain, patch, &anchors) else {
                    continue;
                };
                all.push((record, patch.min, patch.max));
            }
            // Patches with holes last, each group nearest first as selected.
            all.sort_by_key(|(record, ..)| record.holes != NO_HOLES);
            let all_holes = all
                .iter()
                .filter(|(record, ..)| record.holes != NO_HOLES)
                .count() as u32;
            let mut camera_holes = 0;
            for &(record, min, max) in &all {
                if visible(&camera_planes, relative_origin + min, relative_origin + max) {
                    self.records.push(record);
                    camera_holes += u32::from(record.holes != NO_HOLES);
                }
            }
            let camera = start..self.records.len() as u32;
            let all_start = self.records.len() as u32;
            if views.probes {
                self.records.extend(all.iter().map(|&(record, ..)| record));
            }
            let all_range = all_start..self.records.len() as u32;
            let all_holes = if views.probes { all_holes } else { 0 };
            let mut maps = Vec::with_capacity(views.maps.len());
            // Terrains casting long shadows draw into the near cascades
            // only, within the reach their horizon maps leave them.
            let reach = self.horizons.cascade_reach(terrain.id);
            for &(map, clip) in views.maps {
                if reach.is_some() && map == crate::shadows::MOVING_MAP {
                    continue;
                }
                let sides = caster_planes(clip, reach);
                let first = self.records.len() as u32;
                for &(record, min, max) in &all {
                    if visible(&sides, origin + min, origin + max) {
                        self.records.push(record);
                    }
                }
                let range = first..self.records.len() as u32;
                self.frame.shadow_patches += range.len() as u32;
                maps.push((map, range));
            }
            self.frame.patches += camera.len() as u32;
            self.frame.terrains.push(FrameTerrain {
                uniform: (index as u64 * SLOT) as u32,
                camera,
                camera_holes,
                all: all_range,
                all_holes,
                maps,
            });
        }
        self.frame.triangles = self.frame.patches * PATCH_TRIANGLES;
        queue.write_buffer(&self.uniforms, 0, &uniforms);
        let record_bytes = (self.records.len() * size_of::<PatchRecord>()) as u64;
        if record_bytes > self.instances.size() {
            self.instances =
                instance_buffer(device, (self.records.len() as u64).next_power_of_two());
        }
        if !self.records.is_empty() {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.records));
        }
        let camera = ViewUniform {
            relative_view_projection: views.camera.to_cols_array_2d(),
            eye_offset: [0.0; 4],
        };
        if self.written_views != Some(camera) {
            queue.write_buffer(&self.views, 0, bytemuck::bytes_of(&camera));
            self.written_views = Some(camera);
        }
        self.ensure_group(device);
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
        let view = ViewUniform {
            relative_view_projection: relative.to_cols_array_2d(),
            eye_offset: (eye.as_dvec3() - center.as_dvec3())
                .as_vec3()
                .extend(0.0)
                .to_array(),
        };
        let offset = (1 + face as u64) * SLOT;
        queue.write_buffer(&self.views, offset, bytemuck::bytes_of(&view));
        offset as u32
    }

    /// The instance record of `patch`, or `None` when its tile or its
    /// parent has no layer.
    fn record(
        &self,
        terrain: &RenderTerrain,
        patch: &engine_core::Patch,
        anchors: &[i64],
    ) -> Option<PatchRecord> {
        let key = patch.tile;
        let level = u32::from(key.level);
        let layer = *self.slots.get(&(terrain.id, key))?;
        let spacing = terrain.pyramid.spacing(level);
        let first = [
            i64::from(key.x) * 256 + i64::from(patch.x) * 16 - anchors[level as usize * 2],
            i64::from(key.z) * 256 + i64::from(patch.z) * 16 - anchors[level as usize * 2 + 1],
        ];
        let (parent_layer, parent_texel) = match terrain.pyramid.parent(key) {
            Some(parent) => {
                let parent_layer = *self.slots.get(&(terrain.id, parent))?;
                let x = u32::from(key.x % 2) * 128 + u32::from(patch.x) * 8;
                let z = u32::from(key.z % 2) * 128 + u32::from(patch.z) * 8;
                (parent_layer, x | (z << 16))
            }
            None => (layer, 0),
        };
        let texel = (u32::from(patch.x) * 16) | ((u32::from(patch.z) * 16) << 16);
        let holes = if patch.holes {
            self.hole_offsets
                .get(&(terrain.id, key))
                .copied()
                .unwrap_or(NO_HOLES)
        } else {
            NO_HOLES
        };
        Some(PatchRecord {
            first: [first[0] as i32, first[1] as i32],
            shape: [spacing, patch.morph[0], patch.morph[1], 4.0 * spacing],
            atlas: [layer, parent_layer, texel, parent_texel],
            holes,
        })
    }

    /// Uploads tiles the GPU lacks and releases those no terrain lists.
    fn sync_tiles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        max_layers: u32,
    ) {
        let ids: Vec<u64> = terrains.terrains.iter().map(|terrain| terrain.id).collect();
        let current = terrains
            .terrains
            .iter()
            .all(|terrain| self.synced.get(&terrain.id) == Some(&terrain.tiles_revision))
            && self.synced.keys().all(|id| ids.contains(id));
        if current {
            return;
        }
        self.dropped = 0;
        // Release what no terrain lists any more.
        let mut listed: HashMap<(u64, TileKey), &std::sync::Arc<engine_core::Tile>> =
            HashMap::new();
        for terrain in &terrains.terrains {
            for tile in terrain.tiles.iter() {
                listed.insert((terrain.id, tile.key()), tile);
            }
        }
        let free = &mut self.free;
        self.slots.retain(|key, layer| {
            let keep = listed.contains_key(key);
            if !keep {
                free.push(*layer);
            }
            keep
        });
        // Upload the rest, coarse levels first.
        let mut missing: Vec<(u64, TileKey)> = listed
            .keys()
            .filter(|key| !self.slots.contains_key(key))
            .copied()
            .collect();
        missing.sort_unstable_by_key(|(id, key)| (std::cmp::Reverse(key.level), *id, key.x, key.z));
        let needed = self.slots.len() as u32 + missing.len() as u32;
        self.ensure_atlas(device, queue, needed.min(max_layers), max_layers);
        let capacity = self.atlas.as_ref().map_or(0, |atlas| atlas.layers);
        for key in missing {
            let layer = match self.free.pop() {
                Some(layer) => layer,
                None if self.next_layer < capacity => {
                    self.next_layer += 1;
                    self.next_layer - 1
                }
                None => {
                    self.dropped += 1;
                    continue;
                }
            };
            let atlas = self.atlas.as_ref().expect("allocated above");
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(listed[&key].heights()),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(LAYER_SIZE * 4),
                    rows_per_image: Some(LAYER_SIZE),
                },
                wgpu::Extent3d {
                    width: LAYER_SIZE,
                    height: LAYER_SIZE,
                    depth_or_array_layers: 1,
                },
            );
            // Its control values; without a surface, the first material
            // everywhere.
            let blank;
            let control = match listed[&key].control() {
                Some(control) => control,
                None => {
                    blank = vec![0_u32; (LAYER_SIZE * LAYER_SIZE) as usize];
                    &blank
                }
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas.control,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(control),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(LAYER_SIZE * 4),
                    rows_per_image: Some(LAYER_SIZE),
                },
                wgpu::Extent3d {
                    width: LAYER_SIZE,
                    height: LAYER_SIZE,
                    depth_or_array_layers: 1,
                },
            );
            self.slots.insert(key, layer);
            self.normal_jobs.push((layer, listed[&key].spacing()));
            self.uploads += 1;
        }
        self.compute_normals(device, queue);
        self.synced = terrains
            .terrains
            .iter()
            .map(|terrain| (terrain.id, terrain.tiles_revision))
            .collect();
        // Hole bits, rewritten whole when the tiles with holes change.
        let mut holed: Vec<(u64, TileKey)> = listed
            .iter()
            .filter(|(key, tile)| tile.hole_bits().is_some() && self.slots.contains_key(key))
            .map(|(key, _)| *key)
            .collect();
        holed.sort_unstable();
        let unchanged = holed.len() == self.hole_offsets.len()
            && holed.iter().all(|key| self.hole_offsets.contains_key(key));
        if !unchanged {
            self.hole_offsets.clear();
            let mut words: Vec<u32> = Vec::with_capacity(holed.len() * HOLE_WORDS);
            for key in holed {
                let bits = listed[&key].hole_bits().expect("filtered above");
                self.hole_offsets.insert(key, words.len() as u32);
                let mut tile: Vec<u32> = bits
                    .iter()
                    .flat_map(|word| [*word as u32, (word >> 32) as u32])
                    .collect();
                tile.truncate(HOLE_WORDS);
                words.extend_from_slice(&tile);
            }
            let bytes = (words.len() * 4) as u64;
            if bytes > self.holes.size() {
                self.holes = crate::buffers::storage(
                    device,
                    "engine/terrain-holes",
                    bytes.next_power_of_two(),
                    wgpu::BufferUsages::empty(),
                );
                self.holes_generation += 1;
            }
            if !words.is_empty() {
                queue.write_buffer(&self.holes, 0, bytemuck::cast_slice(&words));
            }
        }
    }

    /// Computes the normal maps of the tiles uploaded this frame, after
    /// their heights, in a submission of its own.
    fn compute_normals(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(atlas) = &self.atlas else {
            self.normal_jobs.clear();
            return;
        };
        if self.normal_jobs.is_empty() {
            return;
        }
        let (layout, pipeline) = self.normal_pass.get_or_insert_with(|| {
            let layout = crate::pipelines::layout(device, &TERRAIN_NORMALS);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("engine/terrain-normals.wgsl"),
                source: wgpu::ShaderSource::Wgsl(crate::gpu_data::TERRAIN_NORMALS_SHADER.into()),
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/terrain-normals-layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline =
                crate::pipelines::compute_entry(device, &pipeline_layout, &module, "tile_normals");
            (layout, pipeline)
        });
        // One uniform slot per job.
        let mut words = vec![0_u32; self.normal_jobs.len() * SLOT as usize / 4];
        for (index, &(layer, spacing)) in self.normal_jobs.iter().enumerate() {
            let at = index * SLOT as usize / 4;
            words[at] = layer;
            words[at + 1] = spacing.to_bits();
        }
        let jobs = crate::buffers::uniform(
            device,
            "engine/terrain-normal-jobs",
            (words.len() * 4) as u64,
        );
        queue.write_buffer(&jobs, 0, bytemuck::cast_slice(&words));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("engine/terrain-normals"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/terrain-normals"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            for (index, &(layer, _)) in self.normal_jobs.iter().enumerate() {
                let target = atlas.normals.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let group = crate::pipelines::group(
                    device,
                    "engine/terrain-normals-group",
                    &TERRAIN_NORMALS,
                    layout,
                    &[
                        ("heights", crate::pipelines::view(&atlas.view)),
                        (
                            "job",
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &jobs,
                                offset: index as u64 * SLOT,
                                size: wgpu::BufferSize::new(16),
                            }),
                        ),
                        ("normals", crate::pipelines::view(&target)),
                    ],
                );
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(LAYER_SIZE.div_ceil(16), LAYER_SIZE.div_ceil(16), 1);
            }
        }
        queue.submit([encoder.finish()]);
        self.normal_jobs.clear();
    }

    /// Writes every terrain's surface materials into the palette when they
    /// change, each terrain's after the one before, with the layers their
    /// maps lie in; materials past its capacity, and those whose maps the
    /// arrays cannot hold, are dropped and counted.
    fn write_palette(&mut self, queue: &wgpu::Queue, terrains: &RenderTerrains) {
        let mut entries: Vec<PaletteEntry> = Vec::new();
        self.palette_bases.clear();
        self.dropped_materials = self.maps.dropped;
        for terrain in &terrains.terrains {
            let room = PALETTE_ENTRIES - entries.len();
            let kept = terrain.surface.len().min(room);
            self.dropped_materials += (terrain.surface.len() - kept) as u32;
            // Out of room, a terrain shows the palette's first material.
            self.palette_bases.push(if kept == terrain.surface.len() {
                entries.len() as u32
            } else {
                0
            });
            let maps = terrain.maps.iter().chain(std::iter::repeat(&None));
            entries.extend(
                terrain.surface[..kept]
                    .iter()
                    .zip(maps)
                    .map(|(look, maps)| {
                        let projection = match look.projection {
                            engine_core::SurfaceProjection::Top => 0.0,
                            engine_core::SurfaceProjection::Biplanar => 1.0,
                        };
                        // An engine material's factors, its maps in their layer.
                        let (color, roughness, layer, scale, occlusion, metalness) = match maps {
                            Some(maps) => {
                                let material = &maps.material;
                                let color = material.color();
                                let surface = material.surface();
                                // Occlusion reads the packed map's red.
                                let packed = material.occlusion_texture().map(|map| map.texture)
                                    == material.metallic_roughness_texture().map(|map| map.texture);
                                (
                                    [color[0], color[1], color[2]],
                                    surface.roughness,
                                    self.maps.layer(maps),
                                    material.normal_scale(),
                                    if packed {
                                        material.occlusion_strength()
                                    } else {
                                        0.0
                                    },
                                    surface.metalness,
                                )
                            }
                            None => (look.color, look.roughness, None, 1.0, 0.0, 0.0),
                        };
                        // Where its maps repeat too small to show, their means.
                        let means = layer.map(|layer| self.maps.means(layer));
                        let far =
                            means.map_or([color[0], color[1], color[2], roughness], |means| {
                                [
                                    color[0] * means.color[0],
                                    color[1] * means.color[1],
                                    color[2] * means.color[2],
                                    roughness * means.roughness,
                                ]
                            });
                        let far_occlusion =
                            means.map_or(1.0, |means| 1.0 + occlusion * (means.occlusion - 1.0));
                        [
                            color[0],
                            color[1],
                            color[2],
                            roughness,
                            look.tiling,
                            look.height_contrast,
                            projection,
                            layer.map_or(0.0, |layer| (layer + 1) as f32),
                            scale,
                            occlusion,
                            metalness,
                            far_occlusion,
                            far[0],
                            far[1],
                            far[2],
                            far[3],
                        ]
                    }),
            );
        }
        if entries.is_empty() || entries == self.palette_written {
            return;
        }
        queue.write_buffer(&self.palette, 0, bytemuck::cast_slice(&entries));
        self.palette_written = entries;
    }

    /// Writes the directory of every terrain's levels and resident tiles
    /// when it changes: each level's entries start, tiles and reach, then
    /// its entries.
    fn write_directory(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
    ) {
        let mut words: Vec<u32> = Vec::new();
        self.directory_bases.clear();
        for terrain in &terrains.terrains {
            let pyramid = &terrain.pyramid;
            let base = words.len();
            self.directory_bases.push(base as u32);
            let levels = pyramid.levels() as usize;
            words.resize(base + 4 * levels, 0);
            for level in 0..pyramid.levels() {
                let [tiles_x, tiles_z] = pyramid.tiles(level);
                let header = base + 4 * level as usize;
                // Without a reach, a level shades nowhere but the coarsest.
                let reach = match terrain.reach.get(level as usize) {
                    Some(&reach) => reach,
                    None if level + 1 == pyramid.levels() => f32::MAX,
                    None => 0.0,
                };
                words[header] = words.len() as u32;
                words[header + 1] = tiles_x;
                words[header + 2] = tiles_z;
                words[header + 3] = reach.to_bits();
                for z in 0..tiles_z {
                    for x in 0..tiles_x {
                        let key = TileKey::new(level as u8, x as u16, z as u16);
                        words.push(
                            self.slots
                                .get(&(terrain.id, key))
                                .copied()
                                .unwrap_or(NOT_RESIDENT),
                        );
                    }
                }
            }
        }
        if words.is_empty() || words == self.directory_written {
            return;
        }
        let bytes = (words.len() * 4) as u64;
        if bytes > self.directory.size() {
            self.directory = crate::buffers::storage(
                device,
                "engine/terrain-directory",
                bytes.next_power_of_two(),
                wgpu::BufferUsages::empty(),
            );
            self.directory_generation += 1;
        }
        queue.write_buffer(&self.directory, 0, bytemuck::cast_slice(&words));
        self.directory_written = words;
    }

    /// Grows the height array to hold `needed` layers, doubling, up to
    /// `max_layers`; existing layers are copied over.
    fn ensure_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        needed: u32,
        max_layers: u32,
    ) {
        let current = self.atlas.as_ref().map_or(0, |atlas| atlas.layers);
        if needed <= current {
            return;
        }
        let layers = needed.next_power_of_two().max(16).min(max_layers.max(1));
        let array = |label: &'static str, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: LAYER_SIZE,
                    height: LAYER_SIZE,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC
                    | usage,
                view_formats: &[],
            })
        };
        let texture = array(
            "engine/terrain-heights",
            HEIGHT_FORMAT,
            wgpu::TextureUsages::empty(),
        );
        let normals = array(
            "engine/terrain-normals",
            NORMAL_FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING,
        );
        let control = array(
            "engine/terrain-control",
            CONTROL_FORMAT,
            wgpu::TextureUsages::empty(),
        );
        if let Some(old) = &self.atlas
            && self.next_layer > 0
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/terrain-grow"),
            });
            let extent = wgpu::Extent3d {
                width: LAYER_SIZE,
                height: LAYER_SIZE,
                depth_or_array_layers: self.next_layer,
            };
            encoder.copy_texture_to_texture(
                old.texture.as_image_copy(),
                texture.as_image_copy(),
                extent,
            );
            encoder.copy_texture_to_texture(
                old.normals.as_image_copy(),
                normals.as_image_copy(),
                extent,
            );
            encoder.copy_texture_to_texture(
                old.control.as_image_copy(),
                control.as_image_copy(),
                extent,
            );
            // Before this frame's tile writes, which the queue runs in order.
            queue.submit([encoder.finish()]);
        }
        let array_view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        self.atlas = Some(Atlas {
            view: array_view(&texture),
            normals_view: array_view(&normals),
            control_view: array_view(&control),
            texture,
            control,
            normals,
            layers,
        });
        self.atlas_generation += 1;
    }

    fn ensure_group(&mut self, device: &wgpu::Device) {
        let Some(atlas) = &self.atlas else {
            return;
        };
        let key = (
            self.atlas_generation,
            self.uniform_generation,
            self.holes_generation,
            self.directory_generation,
            self.maps.generation,
        );
        if self.group.stale(&key) {
            let group = crate::pipelines::group(
                device,
                "engine/terrain-group",
                &TERRAIN,
                &self.layout,
                &[
                    ("terrain_heights", crate::pipelines::view(&atlas.view)),
                    (
                        "terrain_view",
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.views,
                            offset: 0,
                            size: wgpu::BufferSize::new(size_of::<ViewUniform>() as u64),
                        }),
                    ),
                    (
                        "terrain",
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniforms,
                            offset: 0,
                            size: wgpu::BufferSize::new(size_of::<TerrainUniform>() as u64),
                        }),
                    ),
                    crate::buffers::entry("terrain_holes", &self.holes),
                    (
                        "terrain_normals",
                        crate::pipelines::view(&atlas.normals_view),
                    ),
                    (
                        "terrain_normal_sampler",
                        crate::pipelines::sampler(&self.normal_sampler),
                    ),
                    crate::buffers::entry("terrain_directory", &self.directory),
                    (
                        "terrain_control",
                        crate::pipelines::view(&atlas.control_view),
                    ),
                    (
                        "terrain_palette",
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.palette,
                            offset: 0,
                            size: None,
                        }),
                    ),
                    (
                        "terrain_albedo",
                        crate::pipelines::view(&self.maps.views[0]),
                    ),
                    (
                        "terrain_material_normals",
                        crate::pipelines::view(&self.maps.views[1]),
                    ),
                    ("terrain_orm", crate::pipelines::view(&self.maps.views[2])),
                    (
                        "terrain_material_sampler",
                        crate::pipelines::sampler(&self.maps.sampler),
                    ),
                ],
            );
            self.group.set(key, group);
        }
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
        let Some(group) = self.group.get() else {
            return 0;
        };
        if range.is_empty() {
            return 0;
        }
        pass.set_bind_group(1, group, &[view, uniform]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, range.clone());
        1
    }

    /// The opaque pipeline into `targets`, shading local lights when `lit`,
    /// of patches with `holes` or without.
    pub(crate) fn pipeline(
        &self,
        targets: Targets,
        lit: bool,
        holes: bool,
    ) -> Option<&wgpu::RenderPipeline> {
        self.pipelines.get(&(targets, lit, holes))
    }

    pub(crate) fn probe_pipeline(&self, lit: bool, holes: bool) -> Option<&wgpu::RenderPipeline> {
        self.probe_pipelines.get(&(lit, holes))
    }

    /// Whether any patch drawn this frame has holes.
    pub(crate) fn has_holes(&self) -> bool {
        self.frame
            .terrains
            .iter()
            .any(|terrain| terrain.camera_holes > 0 || terrain.all_holes > 0)
    }

    pub(crate) fn shadow_pipeline(&self) -> Option<&wgpu::RenderPipeline> {
        self.shadow.as_ref()
    }

    /// Creates the opaque pipeline into `targets` (`lit` or not) unless it
    /// exists: `camera` and `lights` are the main passes' first and lights
    /// groups, `empty` the group standing in for the poses.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ensure_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        (camera, empty, lights): (
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            Option<&wgpu::BindGroupLayout>,
        ),
        (targets, lit, holes): (Targets, bool, bool),
        canvas: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.pipelines.contains_key(&(targets, lit, holes)) {
            return;
        }
        let Some(layout) = pipeline_layout(device, camera, &self.layout, empty, lights, lit) else {
            return;
        };
        let (vertex, fragment) = crate::entries::terrain(Some(targets), lit);
        let color_targets = targets.color_targets(canvas);
        let pipeline = geometry(
            device,
            shader,
            &layout,
            (vertex, fragment, holes),
            &color_targets,
            depth,
            wgpu::FrontFace::Ccw,
        );
        self.pipelines.insert((targets, lit, holes), pipeline);
    }

    /// Creates the reflection probe faces' pipeline (`lit` or not) unless it
    /// exists; faces mirror the world, so front faces turn clockwise.
    pub(crate) fn ensure_probe_pipeline(
        &mut self,
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        (camera, empty, lights): (
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            Option<&wgpu::BindGroupLayout>,
        ),
        (lit, holes): (bool, bool),
        format: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) {
        if self.probe_pipelines.contains_key(&(lit, holes)) {
            return;
        }
        let Some(layout) = pipeline_layout(device, camera, &self.layout, empty, lights, lit) else {
            return;
        };
        let (vertex, fragment) = crate::entries::terrain(None, lit);
        let pipeline = geometry(
            device,
            shader,
            &layout,
            (vertex, fragment, holes),
            &[Some(wgpu::ColorTargetState::from(format))],
            depth,
            wgpu::FrontFace::Cw,
        );
        self.probe_pipelines.insert((lit, holes), pipeline);
    }

    /// Creates the shadow maps' depth pipeline unless it exists.
    pub(crate) fn ensure_shadow_pipeline(&mut self, device: &wgpu::Device) {
        if self.shadow.is_some() {
            return;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/terrain-shadow"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu_data::TERRAIN_SHADOW_SHADER.into()),
        });
        let view_layout = crate::pipelines::layout(device, &SHADOW_VIEW);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/terrain-shadow-layout"),
            bind_group_layouts: &[Some(&view_layout), Some(&self.layout)],
            immediate_size: 0,
        });
        let pipeline = crate::pipelines::geometry(
            device,
            &crate::pipelines::Geometry {
                label: "engine/terrain-shadow",
                layout: &layout,
                module: &module,
                vertex: crate::entries::TERRAIN_SHADOW_VERTEX,
                buffers: &[Some(patch_layout())],
                // Both faces cast, as meshes' casters do.
                primitive: wgpu::PrimitiveState::default(),
                depth: Some(wgpu::DepthStencilState {
                    format: crate::shadows::SHADOW_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    // Terrain is seen at grazing angles over texels meters
                    // wide: its depth also slides back by its slope, so the
                    // facets of a texel never shadow the surface they make.
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.5,
                        clamp: 0.0,
                    },
                }),
                fragment: None,
            },
        );
        self.shadow = Some(pipeline);
    }
}

/// The pipeline layout of terrain's mesh passes: the camera, terrain, and
/// for `lit` pipelines an empty group and the lights.
fn pipeline_layout(
    device: &wgpu::Device,
    camera: &wgpu::BindGroupLayout,
    terrain: &wgpu::BindGroupLayout,
    empty: &wgpu::BindGroupLayout,
    lights: Option<&wgpu::BindGroupLayout>,
    lit: bool,
) -> Option<wgpu::PipelineLayout> {
    let mut layouts = vec![Some(camera), Some(terrain)];
    if lit {
        layouts.push(Some(empty));
        layouts.push(Some(lights?));
    }
    Some(
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/terrain-pipeline-layout"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        }),
    )
}

fn geometry(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    (vertex, fragment, holes): (&str, &str, bool),
    targets: &[Option<wgpu::ColorTargetState>],
    depth: wgpu::TextureFormat,
    front_face: wgpu::FrontFace,
) -> wgpu::RenderPipeline {
    let constants: &[(&str, f64)] = if holes {
        &[("TERRAIN_HOLES", 1.0)]
    } else {
        &[]
    };
    crate::pipelines::geometry(
        device,
        &crate::pipelines::Geometry {
            label: "engine/terrain",
            layout,
            module: shader,
            vertex,
            buffers: &[Some(patch_layout())],
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face,
                cull_mode: Some(wgpu::Face::Back),
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
                constants,
                targets,
            }),
        },
    )
}

/// `PatchInput`'s locations 0 to 3.
const PATCH_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Sint32x2, 1 => Float32x4, 2 => Uint32x4, 3 => Uint32];

/// Patch instances.
fn patch_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<PatchRecord>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &PATCH_ATTRIBUTES,
    }
}

fn instance_buffer(device: &wgpu::Device, records: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/terrain-patches"),
        size: records * size_of::<PatchRecord>() as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The shared grid's triangles, each cell split from its (x, z+1) corner to
/// its (x+1, z) corner as collision splits it, facing up; then the skirts,
/// facing out of the patch.
fn grid_indices() -> Vec<u16> {
    let at = |x: u32, z: u32| (z * GRID + x) as u16;
    let skirt = |edge: u32, along: u32| (GRID_VERTICES + edge * GRID + along) as u16;
    let mut indices = Vec::new();
    for z in 0..GRID - 1 {
        for x in 0..GRID - 1 {
            let (a, b, c, d) = (at(x, z), at(x + 1, z), at(x, z + 1), at(x + 1, z + 1));
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let last = GRID - 1;
    for along in 0..last {
        // Edge z = 0 faces -z; z = last faces +z; x = 0 faces -x; x = last
        // faces +x.
        let (t0, t1, b0, b1) = (
            at(along, 0),
            at(along + 1, 0),
            skirt(0, along),
            skirt(0, along + 1),
        );
        indices.extend_from_slice(&[t0, t1, b0, t1, b1, b0]);
        let (t0, t1, b0, b1) = (
            at(along, last),
            at(along + 1, last),
            skirt(1, along),
            skirt(1, along + 1),
        );
        indices.extend_from_slice(&[t0, b0, t1, t1, b0, b1]);
        let (t0, t1, b0, b1) = (
            at(0, along),
            at(0, along + 1),
            skirt(2, along),
            skirt(2, along + 1),
        );
        indices.extend_from_slice(&[t0, b0, t1, t1, b0, b1]);
        let (t0, t1, b0, b1) = (
            at(last, along),
            at(last, along + 1),
            skirt(3, along),
            skirt(3, along + 1),
        );
        indices.extend_from_slice(&[t0, t1, b0, t1, b1, b0]);
    }
    indices
}

/// A terrain's uniform for a view from `eye`, and each level's anchor
/// sample (x, z): a point on every level's grid near the eye, from which
/// patches count their samples.
fn terrain_uniform(
    terrain: &RenderTerrain,
    eye: DVec3,
    view: f32,
    directory: [u32; 4],
) -> (TerrainUniform, Vec<i64>) {
    let pyramid = &terrain.pyramid;
    let levels = pyramid.levels();
    let top = f64::from(pyramid.spacing(levels - 1));
    let origin = terrain.origin.as_dvec3();
    let cells = ((eye - origin) / top).round();
    let anchor = origin + DVec3::new(cells.x * top, 0.0, cells.z * top);
    let anchors = (0..levels)
        .flat_map(|level| {
            let scale = 1_i64 << (levels - 1 - level);
            [cells.x as i64 * scale, cells.z as i64 * scale]
        })
        .collect();
    let relative = (anchor - eye).as_vec3();
    let uniform = TerrainUniform {
        anchor: [relative.x, relative.y, relative.z, pyramid.spacing(0)],
        eye: eye.as_vec3().extend(0.0).to_array(),
        origin: terrain.origin.extend(view).to_array(),
        directory,
    };
    (uniform, anchors)
}

/// The planes of `clip`'s frustum as `ax + by + cz + d >= 0` inside: its
/// sides, and unless `sides_only` its near and far planes (depth 0 to 1).
fn planes(clip: Mat4, sides_only: bool) -> Vec<Vec4> {
    let rows = [clip.row(0), clip.row(1), clip.row(2), clip.row(3)];
    let mut planes = vec![
        rows[3] + rows[0],
        rows[3] - rows[0],
        rows[3] + rows[1],
        rows[3] - rows[1],
    ];
    if !sides_only {
        planes.push(rows[2]);
        planes.push(rows[3] - rows[2]);
    }
    planes
}

/// The planes of map `clip` where terrain casts into it: its sides and,
/// with a `reach`, its receivers' depths widened that far toward the sun.
fn caster_planes(clip: Mat4, reach: Option<f32>) -> Vec<Vec4> {
    let mut planes = planes(clip, true);
    if let Some(reach) = reach {
        let (depth, w) = (clip.row(2), clip.row(3));
        // Depth changes by the third row's length per world unit.
        let per_unit = depth.truncate().length();
        planes.push(w - depth);
        planes.push(depth + w * (reach * per_unit));
    }
    planes
}

/// Whether the box from `min` to `max` reaches inside every plane.
fn visible(planes: &[Vec4], min: Vec3, max: Vec3) -> bool {
    planes.iter().all(|plane| {
        let corner = Vec3::new(
            if plane.x >= 0.0 { max.x } else { min.x },
            if plane.y >= 0.0 { max.y } else { min.y },
            if plane.z >= 0.0 { max.z } else { min.z },
        );
        plane.truncate().dot(corner) + plane.w >= 0.0
    })
}
