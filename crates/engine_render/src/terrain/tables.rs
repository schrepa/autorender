//! The terrain tables shaders look up: each layer's fades and the hole bits
//! of tiles with holes; the directory of every terrain's levels and
//! resident tiles' layers, with the runtime virtual texture's windows and
//! tables after it and water's directory after those (the ground's
//! fragments bind as many storage buffers as they may, so neither takes one
//! of its own); every terrain's surface materials, their covers and
//! what their placement layers' objects leave the ground where they
//! retire; and each terrain's uniform. Each is written where it changed: the
//! directory is laid out again only as the terrains or their pyramids
//! change, and otherwise rewrites the entries of the tiles that took or
//! left layers.

use super::atlas::TileAtlas;
use super::*;
use crate::gpu_layouts::constants::WATER_TILE_WORDS;
use engine_core::terrain_data::Pyramid;

pub(crate) struct TerrainTables {
    // One uniform slot per terrain, and each frame terrain's id, whose
    // uniform sits at its index's slot.
    pub(super) uniforms: wgpu::Buffer,
    pub(super) uniform_generation: u64,
    pub(super) uniform_ids: Vec<u64>,
    // Each layer's fades, then each layer's word naming where its tile's
    // hole bits start (all ones without), then the hole bits of the tiles
    // with holes, and where each tile's start; whether the buffer holds
    // no fades yet, and the words they take.
    pub(super) holes: crate::buffers::Growable,
    pub(super) hole_offsets: HashMap<(u64, TileKey), u32>,
    hole_layers: HashMap<(u64, TileKey), u32>,
    pub(super) holes_generation: u64,
    fades_stale: bool,
    pub(super) fade_words: u32,
    // Every terrain's levels and their resident tiles' layers, the
    // terrains (ids and pyramids) they are laid out for, and the words
    // they replaced, kept for the next layout.
    pub(super) directory: crate::buffers::Growable,
    pub(super) directory_written: Vec<u32>,
    directory_layout: Vec<(u64, Pyramid)>,
    directory_spare: Vec<u32>,
    pub(super) directory_generation: u64,
    // Advances whenever the directory's words change.
    pub(super) directory_revision: u64,
    // Where each terrain's start.
    pub(super) directory_bases: Vec<u32>,
    // Where the virtual texture's windows and tables start in the directory
    // buffer, and where water's directory starts after them.
    pub(super) vt_base: u32,
    pub(super) water_base: u32,
    // Every terrain's surface materials, where each
    // terrain's start, what the buffer holds and the entries it held
    // before, kept for the next frame's.
    pub(super) palette: wgpu::Buffer,
    pub(super) palette_bases: Vec<u32>,
    palette_written: Vec<PaletteEntry>,
    palette_spare: Vec<PaletteEntry>,
    // The terrain being written's placement layers as the ground takes
    // their objects on, kept from terrain to terrain.
    placed_layers: Vec<PlacedSums>,
    /// Surface materials past the palette's capacity, which show the
    /// palette's first instead.
    pub(crate) dropped_materials: u32,
    // Every material's cover beside its palette entry,
    // what the buffer holds and the entries it held before.
    pub(super) covers: wgpu::Buffer,
    covers_written: Vec<CoverEntry>,
    covers_spare: Vec<CoverEntry>,
    // Each terrain's densest cover and its blades' root width.
    pub(super) cover_densities: Vec<[f32; 2]>,
    // The tallest blade any material's cover grows.
    pub(super) cover_tallest: f32,
}

impl TerrainTables {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            uniforms: crate::buffers::uniform(device, "engine/terrain-uniforms", SLOT),
            uniform_generation: 0,
            uniform_ids: Vec::new(),
            holes: crate::buffers::Growable::new(
                device,
                "engine/terrain-holes",
                16,
                wgpu::BufferUsages::empty(),
            ),
            hole_offsets: HashMap::default(),
            hole_layers: HashMap::default(),
            holes_generation: 0,
            fades_stale: true,
            fade_words: 0,
            directory: crate::buffers::Growable::new(
                device,
                "engine/terrain-directory",
                16,
                wgpu::BufferUsages::empty(),
            ),
            directory_written: Vec::new(),
            directory_layout: Vec::new(),
            directory_spare: Vec::new(),
            directory_generation: 0,
            directory_revision: 0,
            directory_bases: Vec::new(),
            vt_base: 0,
            water_base: 0,
            palette: crate::buffers::uniform(
                device,
                "engine/terrain-palette",
                PALETTE_BYTES as u64,
            ),
            palette_bases: Vec::new(),
            palette_written: Vec::new(),
            palette_spare: Vec::new(),
            placed_layers: Vec::new(),
            dropped_materials: 0,
            covers: crate::buffers::storage(
                device,
                "engine/terrain-covers",
                (PALETTE_ENTRIES * size_of::<CoverEntry>()) as u64,
                wgpu::BufferUsages::empty(),
            ),
            covers_written: Vec::new(),
            covers_spare: Vec::new(),
            cover_densities: Vec::new(),
            cover_tallest: 0.0,
        }
    }

    /// Rewrites the hole bits whole when `atlas`' resident tiles with holes
    /// changed, or a tile replaced under its key changed its bits
    /// (`edited`), after each layer's fades for `max_layers` layers and each
    /// layer's word naming where its tile's bits start, which fragments look
    /// the bits up by wherever they stand (`terrain_holes_test`).
    pub(super) fn write_holes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        (atlas, edited): (&TileAtlas, bool),
        max_layers: u32,
    ) {
        // Each layer's fades come first: two layers a word.
        let fade_words = max_layers.div_ceil(2).max(1);
        let unchanged = !edited
            && fade_words == self.fade_words
            && atlas.holed.len() == self.hole_offsets.len()
            && atlas.holed.iter().all(|key| {
                self.hole_offsets.contains_key(key)
                    && atlas.slots.get(key) == self.hole_layers.get(key)
            });
        if unchanged {
            return;
        }
        self.fade_words = fade_words;
        self.hole_offsets.clear();
        self.hole_layers.clear();
        let mut words: Vec<u32> = vec![NO_HOLES; max_layers as usize];
        for &key in &atlas.holed {
            let Some(bits) = atlas.held.get(&key).and_then(|tile| tile.hole_bits()) else {
                continue;
            };
            let offset = fade_words + words.len() as u32;
            self.hole_offsets.insert(key, offset);
            if let Some(&layer) = atlas.slots.get(&key) {
                self.hole_layers.insert(key, layer);
                if let Some(word) = words.get_mut(layer as usize) {
                    *word = offset;
                }
            }
            words.extend(
                bits.iter()
                    .flat_map(|word| [*word as u32, (word >> 32) as u32])
                    .take(HOLE_WORDS),
            );
        }
        let start = u64::from(fade_words) * 4;
        let bytes = start + (words.len() * 4) as u64;
        if self.holes.grow(device, bytes) {
            self.holes_generation += 1;
            // A new buffer holds no fades yet.
            self.fades_stale = true;
        }
        if !words.is_empty() {
            queue.write_buffer(&self.holes, start, bytemuck::cast_slice(&words));
        }
    }

    /// Writes each layer's fades, `words`, before the hole bits where
    /// `dirty` changed them, or whole into a buffer holding none yet.
    pub(super) fn write_fades(
        &mut self,
        queue: &wgpu::Queue,
        (words, dirty): (&[u32], Range<usize>),
    ) {
        if words.is_empty() {
            return;
        }
        let run = if self.fades_stale {
            0..words.len()
        } else {
            dirty
        };
        self.fades_stale = false;
        if let Some(changed) = words.get(run.clone()).filter(|changed| !changed.is_empty()) {
            let offset = (run.start * 4) as u64;
            queue.write_buffer(&self.holes, offset, bytemuck::cast_slice(changed));
        }
    }

    /// Writes every terrain's surface materials into the palette when they
    /// change, each terrain's after the one before, with the layers their
    /// `maps` lie in and what its placement layers' objects leave the ground
    /// where they retire, their `pinned` coarsest levels staying resident
    /// everywhere; materials past its capacity, and those whose maps the
    /// arrays cannot hold, are dropped and counted.
    pub(super) fn write_palette(
        &mut self,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        (maps, pinned): (&TerrainMaps, u32),
    ) {
        self.write_covers(queue, terrains);
        let mut entries = std::mem::take(&mut self.palette_spare);
        entries.clear();
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
            self.placed_layers.clear();
            self.placed_layers.extend(
                terrain
                    .layers
                    .iter()
                    .map(|layer| placed_sums(terrain, layer)),
            );
            let layers = &self.placed_layers;
            let terrain_maps = terrain.maps.iter().chain(std::iter::repeat(&None));
            entries.extend(
                terrain.surface[..kept]
                    .iter()
                    .zip(terrain_maps)
                    .enumerate()
                    .map(|(material, (look, terrain_maps))| {
                        let placed = placed_terms(terrain, layers, (material, pinned));
                        palette_entry(look, terrain_maps, maps, placed)
                    }),
            );
        }
        if entries.is_empty() || entries == self.palette_written {
            self.palette_spare = entries;
            return;
        }
        queue.write_buffer(&self.palette, 0, bytemuck::cast_slice(&entries));
        self.palette_spare = std::mem::replace(&mut self.palette_written, entries);
    }

    /// Writes every terrain's materials' covers at their
    /// palette indices when they change, and notes each terrain's densest.
    fn write_covers(&mut self, queue: &wgpu::Queue, terrains: &RenderTerrains) {
        let mut entries = std::mem::take(&mut self.covers_spare);
        entries.clear();
        self.cover_densities.clear();
        for terrain in &terrains.terrains {
            let room = PALETTE_ENTRIES - entries.len();
            let kept = &terrain.surface[..terrain.surface.len().min(room)];
            self.cover_densities
                .push(if kept.len() == terrain.surface.len() {
                    kept.iter()
                        .filter_map(|look| look.cover)
                        .map(|cover| [cover.density, cover.width])
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
        self.cover_tallest = entries
            .iter()
            .flat_map(|entry| [entry.shape[1], entry.shape[2]])
            .fold(0.0, engine_core::math::max);
        if entries.is_empty() || entries == self.covers_written {
            self.covers_spare = entries;
            return;
        }
        queue.write_buffer(&self.covers, 0, bytemuck::cast_slice(&entries));
        self.covers_spare = std::mem::replace(&mut self.covers_written, entries);
    }

    /// Writes the directory of every terrain's levels and resident tiles
    /// (`atlas`'s layers) where it changed: each level's entries start,
    /// tiles and reach, then its entries. The same terrains over the same
    /// pyramids keep their layout: their levels' reach and the entries of
    /// the tiles that took or left layers this frame are worked out again,
    /// and only the words that changed are written.
    pub(super) fn write_directory(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        atlas: &TileAtlas,
    ) {
        let same =
            self.directory_layout.len() == terrains.terrains.len()
                && self.directory_layout.iter().zip(&terrains.terrains).all(
                    |(&(id, pyramid), terrain)| id == terrain.id && pyramid == terrain.pyramid,
                );
        if !same {
            self.lay_out_directory(device, queue, terrains, atlas);
            return;
        }
        let mut dirty = 0..0;
        for (terrain, &base) in terrains.terrains.iter().zip(&self.directory_bases) {
            for level in 0..terrain.pyramid.levels() {
                let at = base as usize + 4 * level as usize + 3;
                let bits = reach(terrain, level).to_bits();
                set_word(&mut self.directory_written, at, bits, &mut dirty);
            }
        }
        for &(id, key) in &atlas.changed {
            let Some(index) = terrains
                .terrains
                .iter()
                .position(|terrain| terrain.id == id)
            else {
                continue;
            };
            let (pyramid, level) = (&terrains.terrains[index].pyramid, u32::from(key.level));
            if level >= pyramid.levels() {
                continue;
            }
            let [tiles_x, tiles_z] = pyramid.tiles(level);
            if u32::from(key.x) >= tiles_x || u32::from(key.z) >= tiles_z {
                continue;
            }
            // The level's header holds where its entries start.
            let header =
                (self.directory_bases.get(index)).map(|&base| base as usize + 4 * level as usize);
            let Some(&first) = header.and_then(|header| self.directory_written.get(header)) else {
                continue;
            };
            let at = first as usize + (u32::from(key.z) * tiles_x + u32::from(key.x)) as usize;
            let layer = atlas.slots.get(&(id, key)).copied().unwrap_or(NOT_RESIDENT);
            set_word(&mut self.directory_written, at, layer, &mut dirty);
        }
        let bytes = self.directory_bytes(terrains.terrains.len(), atlas);
        let grown = self.directory.grow(device, bytes);
        if grown {
            self.directory_generation += 1;
        }
        let run = if grown {
            0..self.directory_written.len()
        } else {
            dirty
        };
        if self.directory_written.is_empty() || run.is_empty() {
            return;
        }
        let offset = (run.start * 4) as u64;
        let words = &self.directory_written[run];
        queue.write_buffer(&self.directory, offset, bytemuck::cast_slice(words));
        self.directory_revision += 1;
    }

    /// Lays the directory out whole for `terrains` over `atlas`'s layers,
    /// and writes it when it changed.
    fn lay_out_directory(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &RenderTerrains,
        atlas: &TileAtlas,
    ) {
        let mut words = std::mem::take(&mut self.directory_spare);
        words.clear();
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
                words[header] = words.len() as u32;
                words[header + 1] = tiles_x;
                words[header + 2] = tiles_z;
                words[header + 3] = reach(terrain, level).to_bits();
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
        self.directory_layout.clear();
        self.directory_layout.extend(
            terrains
                .terrains
                .iter()
                .map(|terrain| (terrain.id, terrain.pyramid)),
        );
        // The runtime virtual texture's windows and tables follow,
        // at a multiple of 64 words.
        self.vt_base = words.len().next_multiple_of(64) as u32;
        let bytes = self.directory_bytes(terrains.terrains.len(), atlas);
        let grown = self.directory.grow(device, bytes);
        if grown {
            self.directory_generation += 1;
        }
        if words.is_empty() || (words == self.directory_written && !grown) {
            self.directory_spare = words;
            return;
        }
        queue.write_buffer(&self.directory, 0, bytemuck::cast_slice(&words));
        self.directory_spare = std::mem::replace(&mut self.directory_written, words);
        self.directory_revision += 1;
    }

    /// The bytes the directory buffer holds for `terrains` terrains over
    /// `atlas`' layers: the directory's words, the virtual texture's windows
    /// and tables from `vt_base`, and water's directory after them, from
    /// `water_base`, which it places: a word for each patch of each layer's
    /// tile (`water_blocks`).
    fn directory_bytes(&mut self, terrains: usize, atlas: &TileAtlas) -> u64 {
        let texture_words = crate::terrain_vt::VirtualTexture::words(terrains);
        self.water_base = self.vt_base + texture_words as u32;
        let water_words = atlas.layers() as usize * WATER_TILE_WORDS as usize;
        ((self.water_base as usize + water_words) * 4) as u64
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
        self.uniform_ids.clear();
        self.uniform_ids
            .extend(terrains.terrains.iter().map(|terrain| terrain.id));
    }

    /// The palette entries of the frame's terrain `index`, which has
    /// `materials` surface materials, as the palette holds them; none where
    /// they did not fit.
    pub(super) fn palette_of(&self, index: usize, materials: usize) -> &[PaletteEntry] {
        let base = self.palette_bases.get(index).copied().unwrap_or(0) as usize;
        self.palette_written
            .get(base..base + materials)
            .unwrap_or_default()
    }

    /// The uniform offset of the frame terrain `id`.
    pub(super) fn uniform_of(&self, id: u64) -> Option<u32> {
        self.uniform_ids
            .iter()
            .position(|&known| known == id)
            .map(|index| (index as u64 * SLOT) as u32)
    }
}

/// The meters from the eye within which `terrain`'s level `level` shades:
/// its reach; without one, nowhere but at the coarsest level.
fn reach(terrain: &RenderTerrain, level: u32) -> f32 {
    match terrain.reach.get(level as usize) {
        Some(&reach) => reach,
        None if level + 1 == terrain.pyramid.levels() => f32::MAX,
        None => 0.0,
    }
}

/// Sets `words[at]` to `value`, widening `dirty` over it where it changed.
fn set_word(words: &mut [u32], at: usize, value: u32, dirty: &mut Range<usize>) {
    if let Some(word) = words.get_mut(at)
        && *word != value
    {
        *word = value;
        *dirty = widen(dirty.clone(), at);
    }
}

/// The palette entry of surface material `look` with `terrain_maps`, its
/// maps in `maps`' arrays, and its placement layers' objects' terms
/// `placed` ([`placed_terms`]).
fn palette_entry(
    look: &engine_core::SurfaceMaterial,
    terrain_maps: &Option<engine_core::RenderSurfaceMaps>,
    maps: &TerrainMaps,
    (placed, placed_size, placed_thinning): ([f32; 4], [f32; 4], [f32; 4]),
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
    // Where a whole repeat of its maps spans less than a pixel, their means.
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
    // The slopes its normal maps' mean averages away roughen it where their
    // means stand in, as their filtering's do nearer, scaled as their
    // deflections are.
    let slopes = means.map_or(0.0, |means| means.slopes * scale * scale);
    // Where its maps' means stand in, its heights' mean raises it as height
    // blending does.
    let heights = means.map_or([0.0; 4], |means| [means.height, 0.0, 0.0, 0.0]);
    // Its cover's canopy: the blades' tip color, which shaders mix with the
    // ground's at their roots as a view meets them (`terrain_variety.wgsl`'s
    // `cover_seen`), and their leaves' area: blades a square meter, each as
    // wide as its root and tapering up the height they stand on average.
    let canopy = look.cover.map_or([0.0; 4], |cover| {
        let height = COVER_RISEN * 0.5 * (cover.height[0] + cover.height[1]);
        [
            cover.color[0],
            cover.color[1],
            cover.color[2],
            cover.density * height * cover.width * COVER_FILLED,
        ]
    });
    // Materials displacing the ground show no relief, which parallax
    // fakes; the rest's as deep as parallax holds.
    let relief = if look.displacement > 0.0 {
        0.0
    } else {
        engine_core::math::min(look.relief, RELIEF_MOST)
    };
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
        relief: [relief, look.deform, slopes, look.displacement],
        canopy,
        placed,
        placed_size,
        placed_thinning,
        heights,
    }
}

/// The most meters a material's relief sinks the ground, a few
/// centimeters: deeper heights break parallax at grazing views and on
/// silhouettes, where materials displace the ground instead.
const RELIEF_MOST: f32 = 0.06;

/// The share of their full height ground cover's blades stand on average:
/// candidates ranked within the last quarter under their threshold grow
/// short, rising with it (`cover_grow.wgsl`), seven eighths.
const COVER_RISEN: f32 = 0.875;

/// The share of the rectangle a blade's root width and height span that it
/// fills, tapering as 1 - along^1.5 (`meshes/cover.wgsl`): the taper's
/// integral.
const COVER_FILLED: f32 = 0.6;

/// One placement layer's objects as the ground takes them on where they
/// retire, summed over the assets it chooses: the share of ground they
/// cover where everything allows them, and over it the sums of their linear
/// colors, of the base-2 logarithms of the bounding spheres' diameters of
/// those ranked first and last under its density (its largest and
/// smallest: their scale falls across its range as their rank rises,
/// `engine_terrain::placement`), and of the share of them standing far
/// away.
#[derive(Clone, Copy, Default)]
struct PlacedSums {
    covered: f32,
    color: [f32; 3],
    first: f32,
    last: f32,
    far: f32,
}

impl PlacedSums {
    fn add(&mut self, other: &Self) {
        self.covered += other.covered;
        for (sum, part) in self.color.iter_mut().zip(other.color) {
            *sum += part;
        }
        self.first += other.first;
        self.last += other.last;
        self.far += other.far;
    }
}

/// The base-2 logarithm of `meters`, the same on every host, as palette
/// entries far pages are keyed by must be.
fn octaves(meters: f32) -> f32 {
    engine_core::math::ln(engine_core::math::max(meters, 1.0e-6)) * std::f32::consts::LOG2_E
}

/// `layer`'s objects on `terrain` as the ground takes them on where they
/// retire ([`PlacedSums`]): each asset it chooses, by its share, at the
/// layer's density, its mask whole, its scales spread evenly across its
/// range; those ranked first and last as large and as small as the range
/// lets each asset stand, by how often it is chosen. Assets not here yet
/// bring none.
fn placed_sums(terrain: &RenderTerrain, layer: &engine_core::RenderPlacementLayer) -> PlacedSums {
    let [smallest, largest] = layer.scale;
    // The mean square of scales spread evenly from the smallest to the
    // largest.
    let squared = (smallest * smallest + smallest * largest + largest * largest) / 3.0;
    let mut sums = PlacedSums::default();
    let (mut chosen, mut first, mut last) = (0.0, 0.0, 0.0);
    for &(asset, share) in &layer.assets {
        let Some(Some(asset)) = terrain.assets.get(usize::from(asset)) else {
            continue;
        };
        let (area, diameter, color) = placed_asset(asset);
        let covered = layer.density * share * area * squared;
        sums.covered += covered;
        for (sum, channel) in sums.color.iter_mut().zip(color) {
            *sum += covered * channel;
        }
        chosen += share;
        first += share * octaves(diameter * largest);
        last += share * octaves(diameter * smallest);
    }
    if chosen > 0.0 {
        sums.first = sums.covered * first / chosen;
        sums.last = sums.covered * last / chosen;
    }
    sums.far = sums.covered * layer.far.1;
    sums
}

/// What a placed asset brings the ground at scale 1: the square meters an
/// object covers, the ellipse within its mesh's bounds across the ground;
/// its bounding sphere's diameter, around its mesh, its parts' finest and
/// its shadow proxy as culling bounds it (`renderer::scatter_draws`); and
/// its material's linear color, its base color texture's mean applied
/// where that is summarized.
fn placed_asset(asset: &engine_core::RenderPlacementAsset) -> (f32, f32, [f32; 3]) {
    let bounds = asset.mesh.mesh.bounds();
    let extent = bounds.max() - bounds.min();
    let area = std::f32::consts::FRAC_PI_4 * extent.x * extent.z;
    let (low, high) = asset
        .parts
        .iter()
        .filter_map(|part| part.meshes.first())
        .chain(&asset.shadow)
        .map(|mesh| mesh.mesh.bounds())
        .fold((bounds.min(), bounds.max()), |(low, high), held| {
            (low.min(held.min()), high.max(held.max()))
        });
    let material = &asset.material.material;
    let factor = material.color();
    let mean = material
        .base_color_texture()
        .and_then(|map| {
            asset
                .textures
                .iter()
                .find(|texture| texture.handle == map.texture)
        })
        .and_then(|texture| texture.texture.summary())
        .map_or([1.0; 3], |summary| summary.color);
    (
        area,
        (high - low).length(),
        [
            factor[0] * mean[0],
            factor[1] * mean[1],
            factor[2] * mean[2],
        ],
    )
}

/// What surface material `material` of `terrain` shows where its placement
/// layers' objects retire (`meshes/terrain.wgsl`'s `terrain_placed_share`),
/// from the layers' sums `layers`, its `pinned` coarsest levels staying
/// resident everywhere: the objects' mean linear color and the share of
/// the ground they cover where the material lies alone, at most all of it;
/// the base-2 logarithms of the bounding spheres' diameters of those ranked
/// first and last, over that ground; and how they thin with distance, as
/// culling thins them: the share standing far away, over that ground, and
/// the levels the layer covering most of it stands on near and far, each
/// plus one where it thins there (`scatter::thins_on`), else 0. Zeroes
/// where none stands on it.
fn placed_terms(
    terrain: &RenderTerrain,
    layers: &[PlacedSums],
    (material, pinned): (usize, u32),
) -> ([f32; 4], [f32; 4], [f32; 4]) {
    let mut sums = PlacedSums::default();
    let mut widest: Option<(&engine_core::RenderPlacementLayer, f32)> = None;
    for (layer, held) in terrain.layers.iter().zip(layers) {
        if layer.stands_on(material) {
            sums.add(held);
            if widest.is_none_or(|(_, covered)| held.covered > covered) {
                widest = Some((layer, held.covered));
            }
        }
    }
    let Some((widest, _)) = widest.filter(|_| sums.covered > 0.0) else {
        return ([0.0; 4], [0.0; 4], [0.0; 4]);
    };
    let covered = sums.covered;
    let levels = terrain.pyramid.levels();
    let code = |level: u8| {
        if crate::scatter::thins_on(level, levels, pinned) {
            f32::from(level) + 1.0
        } else {
            0.0
        }
    };
    (
        [
            sums.color[0] / covered,
            sums.color[1] / covered,
            sums.color[2] / covered,
            engine_core::math::min(covered, 1.0),
        ],
        [sums.first / covered, sums.last / covered, 0.0, 0.0],
        [
            sums.far / covered,
            code(widest.level),
            code(widest.far.0),
            0.0,
        ],
    )
}
