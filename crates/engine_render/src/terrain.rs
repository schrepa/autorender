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
//! camera sees, so terrain shadows itself without acne.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use engine_core::glam::{DVec3, Mat4, Vec3, Vec4};
use engine_core::{RenderTerrain, RenderTerrains, TileKey};
use wgpu::util::DeviceExt;

use crate::bindings::{SHADOW_VIEW, TERRAIN};
use crate::cached::Cached;
use crate::render_graph::Targets;

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
    grass: [f32; 4],
    dirt: [f32; 4],
    rock: [f32; 4],
    snow: [f32; 4],
}

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
}

struct Atlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
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
    group: Cached<(u64, u64, u64), wgpu::BindGroup>,
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
}

impl TerrainPass {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
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

    /// Bytes of the height array.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.atlas.as_ref().map_or(0, |atlas| {
            u64::from(atlas.layers) * u64::from(LAYER_SIZE * LAYER_SIZE) * 4
        })
    }

    /// Uploads tiles the frame's terrains list and the GPU lacks, releases
    /// those they no longer list, and lays out this frame's patches for the
    /// camera, reflection probe faces and the shadow maps of `views`.
    /// `max_layers` is the device's texture array layers. Returns whether
    /// any terrain draws.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        views: &TerrainViews<'_>,
        max_layers: u32,
    ) -> bool {
        self.uploads = 0;
        self.frame = TerrainFrame::default();
        self.eye = views.eye;
        self.sync_tiles(device, queue, terrains, max_layers);
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
            };
            let (uniform, anchors) = terrain_uniform(terrain, eye, view);
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
            for &(map, clip) in views.maps {
                let sides = planes(clip, true);
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
            self.slots.insert(key, layer);
            self.uploads += 1;
        }
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
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/terrain-heights"),
            size: wgpu::Extent3d {
                width: LAYER_SIZE,
                height: LAYER_SIZE,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HEIGHT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        if let Some(old) = &self.atlas
            && self.next_layer > 0
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/terrain-grow"),
            });
            encoder.copy_texture_to_texture(
                old.texture.as_image_copy(),
                texture.as_image_copy(),
                wgpu::Extent3d {
                    width: LAYER_SIZE,
                    height: LAYER_SIZE,
                    depth_or_array_layers: self.next_layer,
                },
            );
            // Before this frame's tile writes, which the queue runs in order.
            queue.submit([encoder.finish()]);
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("engine/terrain-heights"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.atlas = Some(Atlas {
            texture,
            view,
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
fn terrain_uniform(terrain: &RenderTerrain, eye: DVec3, view: f32) -> (TerrainUniform, Vec<i64>) {
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
    let material = &terrain.material;
    let color = |rgb: [f32; 3], w: f32| [rgb[0], rgb[1], rgb[2], w];
    let uniform = TerrainUniform {
        anchor: [relative.x, relative.y, relative.z, pyramid.spacing(0)],
        eye: eye.as_vec3().extend(0.0).to_array(),
        origin: terrain.origin.extend(view).to_array(),
        grass: color(material.grass, material.roughness),
        dirt: color(material.dirt, 0.0),
        rock: color(
            material.rock,
            engine_core::math::cos(material.rock_slope.to_radians()),
        ),
        snow: color(material.snow, material.snow_height),
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
