//! The terrain tables shaders look up: each layer's fades and the hole bits
//! of tiles with holes; the directory of every terrain's levels and
//! resident tiles' layers, with the runtime virtual texture's windows and
//! tables after it; every terrain's surface materials and their covers;
//! and each terrain's uniform.

use super::atlas::{Synced, TileAtlas};
use super::*;

pub(crate) struct TerrainTables {
    // One uniform slot per terrain, and each frame terrain's id, whose
    // uniform sits at its index's slot.
    pub(super) uniforms: wgpu::Buffer,
    pub(super) uniform_generation: u64,
    pub(super) uniform_ids: Vec<u64>,
    // Each layer's fades, then the hole bits of the tiles
    // with holes, and where each tile's start; the fades last written and
    // the words they take.
    pub(super) holes: wgpu::Buffer,
    pub(super) hole_offsets: HashMap<(u64, TileKey), u32>,
    pub(super) holes_generation: u64,
    fades_written: Vec<u32>,
    pub(super) fade_words: u32,
    // Every terrain's levels and their resident tiles' layers, and where
    // each terrain's start.
    pub(super) directory: wgpu::Buffer,
    pub(super) directory_written: Vec<u32>,
    pub(super) directory_generation: u64,
    // Advances whenever the directory's words change.
    pub(super) directory_revision: u64,
    pub(super) directory_bases: Vec<u32>,
    // Where the virtual texture's windows and tables start in the directory
    // buffer.
    pub(super) vt_base: u32,
    // Every terrain's surface materials, where each
    // terrain's start, and what the buffer holds.
    pub(super) palette: wgpu::Buffer,
    pub(super) palette_bases: Vec<u32>,
    palette_written: Vec<PaletteEntry>,
    /// Surface materials past the palette's capacity, which show the
    /// palette's first instead.
    pub(crate) dropped_materials: u32,
    // Every material's cover beside its palette entry,
    // what the buffer holds, and each terrain's densest cover.
    pub(super) covers: wgpu::Buffer,
    covers_written: Vec<CoverEntry>,
    // Each terrain's densest cover and its mean blade height.
    pub(super) cover_densities: Vec<[f32; 2]>,
}

impl TerrainTables {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            uniforms: crate::buffers::uniform(device, "engine/terrain-uniforms", SLOT),
            uniform_generation: 0,
            uniform_ids: Vec::new(),
            holes: crate::buffers::storage(
                device,
                "engine/terrain-holes",
                16,
                wgpu::BufferUsages::empty(),
            ),
            hole_offsets: HashMap::default(),
            holes_generation: 0,
            fades_written: Vec::new(),
            fade_words: 0,
            directory: crate::buffers::storage(
                device,
                "engine/terrain-directory",
                16,
                wgpu::BufferUsages::empty(),
            ),
            directory_written: Vec::new(),
            directory_generation: 0,
            directory_revision: 0,
            directory_bases: Vec::new(),
            vt_base: 0,
            palette: crate::buffers::uniform(
                device,
                "engine/terrain-palette",
                PALETTE_BYTES as u64,
            ),
            palette_bases: Vec::new(),
            palette_written: Vec::new(),
            dropped_materials: 0,
            covers: crate::buffers::storage(
                device,
                "engine/terrain-covers",
                (PALETTE_ENTRIES * size_of::<CoverEntry>()) as u64,
                wgpu::BufferUsages::empty(),
            ),
            covers_written: Vec::new(),
            cover_densities: Vec::new(),
        }
    }

    /// Rewrites the hole bits whole when the tiles with holes `synced`
    /// found changed, after each layer's fades for `max_layers` layers.
    pub(super) fn write_holes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        synced: &Synced<'_>,
        max_layers: u32,
    ) {
        // Each layer's fades come first: two layers a word.
        let fade_words = max_layers.div_ceil(2).max(1);
        let unchanged = !synced.edited
            && fade_words == self.fade_words
            && synced.holed.len() == self.hole_offsets.len()
            && synced
                .holed
                .iter()
                .all(|(key, _)| self.hole_offsets.contains_key(key));
        if unchanged {
            return;
        }
        self.fade_words = fade_words;
        self.hole_offsets.clear();
        let mut words: Vec<u32> = Vec::with_capacity(synced.holed.len() * HOLE_WORDS);
        for &(key, tile) in &synced.holed {
            let bits = tile.hole_bits().expect("filtered above");
            self.hole_offsets
                .insert(key, fade_words + words.len() as u32);
            let mut tile: Vec<u32> = bits
                .iter()
                .flat_map(|word| [*word as u32, (word >> 32) as u32])
                .collect();
            tile.truncate(HOLE_WORDS);
            words.extend_from_slice(&tile);
        }
        let start = u64::from(fade_words) * 4;
        let bytes = start + (words.len() * 4) as u64;
        if bytes > self.holes.size() {
            self.holes = crate::buffers::storage(
                device,
                "engine/terrain-holes",
                bytes.next_power_of_two(),
                wgpu::BufferUsages::empty(),
            );
            self.holes_generation += 1;
            // A new buffer holds no fades yet.
            self.fades_written.clear();
        }
        if !words.is_empty() {
            queue.write_buffer(&self.holes, start, bytemuck::cast_slice(&words));
        }
    }

    /// Writes each layer's fades, `words`, before the hole bits when they
    /// changed.
    pub(super) fn write_fades(&mut self, queue: &wgpu::Queue, words: Vec<u32>) {
        if words != self.fades_written {
            queue.write_buffer(&self.holes, 0, bytemuck::cast_slice(&words));
            self.fades_written = words;
        }
    }

    /// Writes every terrain's surface materials into the palette when they
    /// change, each terrain's after the one before, with the layers their
    /// `maps` lie in; materials past its capacity, and those whose maps the
    /// arrays cannot hold, are dropped and counted.
    pub(super) fn write_palette(
        &mut self,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        maps: &TerrainMaps,
    ) {
        self.write_covers(queue, terrains);
        let mut entries: Vec<PaletteEntry> = Vec::new();
        self.palette_bases.clear();
        self.dropped_materials = maps.dropped;
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
            let terrain_maps = terrain.maps.iter().chain(std::iter::repeat(&None));
            entries.extend(
                terrain.surface[..kept]
                    .iter()
                    .zip(terrain_maps)
                    .map(|(look, terrain_maps)| palette_entry(look, terrain_maps, maps)),
            );
        }
        if entries.is_empty() || entries == self.palette_written {
            return;
        }
        queue.write_buffer(&self.palette, 0, bytemuck::cast_slice(&entries));
        self.palette_written = entries;
    }

    /// Writes every terrain's materials' covers at their
    /// palette indices when they change, and notes each terrain's densest.
    fn write_covers(&mut self, queue: &wgpu::Queue, terrains: &RenderTerrains) {
        let mut entries: Vec<CoverEntry> = Vec::new();
        self.cover_densities.clear();
        for terrain in &terrains.terrains {
            let room = PALETTE_ENTRIES - entries.len();
            let kept = &terrain.surface[..terrain.surface.len().min(room)];
            self.cover_densities
                .push(if kept.len() == terrain.surface.len() {
                    kept.iter()
                        .filter_map(|look| look.cover)
                        .map(|cover| [cover.density, 0.5 * (cover.height[0] + cover.height[1])])
                        .fold([0.0; 2], |densest, cover| {
                            if cover[0] > densest[0] {
                                cover
                            } else {
                                densest
                            }
                        })
                } else {
                    [0.0; 2]
                });
            entries.extend(kept.iter().map(|look| {
                look.cover.map_or(CoverEntry::zeroed(), |cover| CoverEntry {
                    shape: [cover.density, cover.height[0], cover.height[1], cover.width],
                    color: [cover.color[0], cover.color[1], cover.color[2], cover.sway],
                })
            }));
        }
        if entries.is_empty() || entries == self.covers_written {
            return;
        }
        queue.write_buffer(&self.covers, 0, bytemuck::cast_slice(&entries));
        self.covers_written = entries;
    }

    /// Writes the directory of every terrain's levels and resident tiles
    /// (`atlas`'s layers) when it changes: each level's entries start,
    /// tiles and reach, then its entries.
    pub(super) fn write_directory(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        atlas: &TileAtlas,
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
                            atlas
                                .slots
                                .get(&(terrain.id, key))
                                .copied()
                                .unwrap_or(NOT_RESIDENT),
                        );
                    }
                }
            }
        }
        // The runtime virtual texture's windows and tables follow,
        // at a multiple of 64 words.
        self.vt_base = words.len().next_multiple_of(64) as u32;
        let texture_words = crate::terrain_vt::VirtualTexture::words(terrains.terrains.len());
        let bytes = ((self.vt_base as usize + texture_words) * 4) as u64;
        let grown = bytes > self.directory.size();
        if grown {
            self.directory = crate::buffers::storage(
                device,
                "engine/terrain-directory",
                bytes.next_power_of_two(),
                wgpu::BufferUsages::empty(),
            );
            self.directory_generation += 1;
        }
        if words.is_empty() || (words == self.directory_written && !grown) {
            return;
        }
        queue.write_buffer(&self.directory, 0, bytemuck::cast_slice(&words));
        self.directory_written = words;
        self.directory_revision += 1;
    }

    /// Grows the uniforms to a slot for each of `terrains`' terrains, whose
    /// ids it notes in order.
    pub(super) fn size_uniforms(&mut self, device: &wgpu::Device, terrains: &RenderTerrains) {
        let uniform_bytes = terrains.terrains.len() as u64 * SLOT;
        if uniform_bytes > self.uniforms.size() {
            self.uniforms = crate::buffers::uniform(
                device,
                "engine/terrain-uniforms",
                uniform_bytes.next_power_of_two(),
            );
            self.uniform_generation += 1;
        }
        self.uniform_ids = terrains.terrains.iter().map(|terrain| terrain.id).collect();
    }

    /// The uniform offset of the frame terrain `id`.
    pub(super) fn uniform_of(&self, id: u64) -> Option<u32> {
        self.uniform_ids
            .iter()
            .position(|&known| known == id)
            .map(|index| (index as u64 * SLOT) as u32)
    }
}

/// The palette entry of surface material `look` with `terrain_maps`, its
/// maps in `maps`' arrays.
fn palette_entry(
    look: &engine_core::SurfaceMaterial,
    terrain_maps: &Option<engine_core::RenderSurfaceMaps>,
    maps: &TerrainMaps,
) -> PaletteEntry {
    let projection = match look.projection {
        engine_core::SurfaceProjection::Top => 0.0,
        engine_core::SurfaceProjection::Biplanar => 1.0,
    };
    // An engine material's factors, its maps in their layer.
    let (color, roughness, layer, scale, occlusion, metalness) = match terrain_maps {
        Some(terrain_maps) => {
            let material = &terrain_maps.material;
            let color = material.color();
            let surface = material.surface();
            // Occlusion reads the packed map's red.
            let packed = material.occlusion_texture().map(|map| map.texture)
                == material.metallic_roughness_texture().map(|map| map.texture);
            (
                [color[0], color[1], color[2]],
                surface.roughness,
                maps.layer(terrain_maps),
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
    let means = layer.map(|layer| maps.means(layer));
    let far = means.map_or([color[0], color[1], color[2], roughness], |means| {
        [
            color[0] * means.color[0],
            color[1] * means.color[1],
            color[2] * means.color[2],
            roughness * means.roughness,
        ]
    });
    let far_occlusion = means.map_or(1.0, |means| 1.0 + occlusion * (means.occlusion - 1.0));
    // The slopes its normal maps lose to distance roughen
    // it.
    let slopes = means.map_or(0.0, |means| means.slopes * scale * scale);
    // Its cover's canopy: the blades' color
    // from root (the ground's mean) most of the way to tip,
    // as their upper parts show from afar, and their
    // leaves' area: blades a square meter, each about as
    // wide as its root and tapering up its mean height.
    let canopy = look.cover.map_or([0.0; 4], |cover| {
        let tip = 0.75;
        let height = 0.5 * (cover.height[0] + cover.height[1]);
        [
            far[0] + (cover.color[0] - far[0]) * tip,
            far[1] + (cover.color[1] - far[1]) * tip,
            far[2] + (cover.color[2] - far[2]) * tip,
            cover.density * height * cover.width * 0.6,
        ]
    });
    PaletteEntry {
        color: [color[0], color[1], color[2], roughness],
        params: [
            look.tiling,
            look.height_contrast,
            projection,
            layer.map_or(0.0, |layer| (layer + 1) as f32),
        ],
        maps: [scale, occlusion, metalness, far_occlusion],
        far,
        relief: [look.relief, look.deform, slopes, 0.0],
        canopy,
    }
}
