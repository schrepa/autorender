//! Tiles' water on the GPU: each resident tile's water where it stands over
//! the ground, a patch at a time, in two texture arrays of blocks as many
//! samples wide as a patch's grid: their surfaces, and their words, their
//! packed flow, whitewater and kind, their state and their nearest obstacle
//! (`engine_core::terrain_data::TileWater::info`), and a fourth, 0. A tile's
//! blocks upload as it arrives and free as it leaves or changes, following
//! the tiles the atlas held anew. Water's directory lists, for each layer of
//! the height array, the blocks of its tile's patches and a bit for each
//! patch over which water stands there or in a finer resident tile within
//! it, laid out again as tiles or their water change: shaders read a tile's
//! water at any point through it, and tell the leaves of the terrain's
//! subdivision water may stand over from those it cannot (`crate::water`).

#![expect(
    clippy::expect_used,
    reason = "the arrays are allocated before blocks upload"
)]

use std::collections::BTreeSet;

use super::atlas::TileAtlas;
use super::*;
use crate::gpu_layouts::constants::{
    NO_WATER_BLOCK, NO_WATER_SURFACE, WATER_BLOCK as BLOCK, WATER_BLOCKS_SIDE as SIDE,
    WATER_TILE_PATCHES as TILE_PATCHES, WATER_TILE_WORDS as TILE_WORDS,
};
use engine_core::terrain_data::{PATCHES_PER_TILE, Tile};

/// Blocks a layer holds, and its texels along a side.
const PER_LAYER: u32 = SIDE * SIDE;
const LAYER_TEXELS: u32 = BLOCK * SIDE;
/// The surface and info arrays' formats, and the bytes of each one's
/// texels: shaders unpack the info's words themselves.
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const INFO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
const SURFACE_BYTES: u32 = 4;
const INFO_BYTES: u32 = 16;
/// A tile's entry in water's directory: the blocks of its patches, then the
/// words of their bits.
const TILE_BLOCKS: usize = (TILE_PATCHES * TILE_PATCHES) as usize;
const BIT_WORDS: usize = TILE_WORDS as usize - TILE_BLOCKS;

/// A tile's bits: one for each of its patches, patch (x, z)'s at bit
/// z * `TILE_PATCHES` + x, set where water stands over it there or in a
/// finer resident tile within it.
type PatchBits = [u32; BIT_WORDS];

/// The arrays and their layers.
struct Arrays {
    surface: wgpu::Texture,
    surface_view: wgpu::TextureView,
    info: wgpu::Texture,
    info_view: wgpu::TextureView,
    layers: u32,
}

/// A tile's blocks: the tile whose water they hold, and each patch's block
/// (`NO_WATER_BLOCK` where its water stands nowhere over its ground).
struct TileBlocks {
    tile: Arc<Tile>,
    blocks: Box<[u32]>,
}

/// What water's pass draws tiles' water with: the height array, the views'
/// and terrains' uniforms, the terrain's directory and this frame's leaves
/// of its subdivision, the water blocks' arrays and water's directory, with
/// generations advancing when any of them is another texture or buffer;
/// each frame terrain holding water whose subdivision draws, its uniform's
/// offset and its subdivision's draw (its arguments' buffer and byte
/// offset); and the blocks resident and dropped this frame for want of
/// layers.
pub(crate) struct TerrainWaterDraws<'a> {
    pub(crate) heights: &'a wgpu::TextureView,
    pub(crate) views: &'a wgpu::Buffer,
    pub(crate) uniforms: &'a wgpu::Buffer,
    pub(crate) directory: &'a wgpu::Buffer,
    pub(crate) leaves: &'a wgpu::Buffer,
    pub(crate) surfaces: &'a wgpu::TextureView,
    pub(crate) info: &'a wgpu::TextureView,
    pub(crate) water_directory: &'a wgpu::Buffer,
    pub(crate) generation: [u64; 6],
    pub(crate) terrains: Vec<(u32, &'a wgpu::Buffer, u64)>,
    pub(crate) blocks: [u32; 2],
}

pub(crate) struct WaterBlocks {
    arrays: Option<Arrays>,
    /// Advances whenever the arrays are replaced.
    pub(crate) generation: u64,
    tiles: HashMap<(u64, TileKey), TileBlocks>,
    free: Vec<u32>,
    next: u32,
    /// Blocks uploaded this frame, and dropped for want of layers.
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    // Water's directory, the words it holds and those they replaced, kept
    // for the next layout, and the height array's layers it was laid out
    // for; the bits of every tile water stands over, coarser tiles' holding
    // their finer ones', as last worked out; and the terrains holding water.
    directory: crate::buffers::Growable,
    directory_written: Vec<u32>,
    directory_spare: Vec<u32>,
    directory_layers: Option<u32>,
    bits: HashMap<(u64, TileKey), PatchBits>,
    holding: BTreeSet<u64>,
}

impl WaterBlocks {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            arrays: None,
            generation: 0,
            tiles: HashMap::default(),
            free: Vec::new(),
            next: 0,
            uploads: 0,
            dropped: 0,
            directory: crate::buffers::Growable::new(
                device,
                "engine/water-directory",
                16,
                wgpu::BufferUsages::empty(),
            ),
            directory_written: Vec::new(),
            directory_spare: Vec::new(),
            directory_layers: None,
            bits: HashMap::default(),
            holding: BTreeSet::new(),
        }
    }

    /// The surface and info arrays' views, once a block uploaded.
    pub(crate) fn views(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        self.arrays
            .as_ref()
            .map(|arrays| (&arrays.surface_view, &arrays.info_view))
    }

    /// Blocks resident.
    pub(crate) fn resident(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    /// Bytes of the arrays and the directory.
    pub(super) fn capacity_bytes(&self) -> u64 {
        let arrays = self.arrays.as_ref().map_or(0, |arrays| {
            u64::from(arrays.layers)
                * u64::from(LAYER_TEXELS * LAYER_TEXELS)
                * u64::from(SURFACE_BYTES + INFO_BYTES)
        });
        arrays + self.directory.size()
    }

    /// What water's pass draws tiles' water with this frame, of `pass`'s
    /// parts and `terrains`; `None` before tiles and their water arrive, or
    /// the leaves of the terrain's subdivision.
    pub(super) fn draws<'a>(
        &'a self,
        pass: &'a TerrainPass,
        terrains: &RenderTerrains,
    ) -> Option<TerrainWaterDraws<'a>> {
        let heights = pass.atlas.heights()?;
        let (surfaces, info) = self.views()?;
        let (leaves, leaves_generation) = pass.surface_leaves()?;
        let terrains = (terrains.terrains.iter().enumerate())
            .filter(|(_, terrain)| self.holding.contains(&terrain.id))
            .filter_map(|(index, terrain)| {
                let uniform = pass.tables.uniform_of(terrain.id)?;
                let (draw, offset) = pass.surface_draw(index)?;
                Some((uniform, draw, offset))
            })
            .collect();
        Some(TerrainWaterDraws {
            heights,
            views: &pass.patches.views,
            uniforms: &pass.tables.uniforms,
            directory: &pass.tables.directory,
            leaves,
            surfaces,
            info,
            water_directory: &self.directory,
            generation: [
                pass.atlas.generation,
                pass.tables.uniform_generation,
                self.generation,
                pass.tables.directory_generation,
                self.directory.generation(),
                leaves_generation,
            ],
            terrains,
            blocks: [self.resident(), self.dropped],
        })
    }

    /// Uploads the blocks of the tiles `atlas` holds (by terrain and key)
    /// that arrived or changed, and frees those of tiles gone, looking only
    /// at the keys whose held tile changed since the last sync; a tile
    /// replaced under its key keeps the blocks of patches whose water
    /// stayed the same (its water made again where its rivers run
    /// otherwise). Layers grow up to `max_layers`. Where any tile changed,
    /// or the height array's layers did, water's directory is laid out
    /// again over `terrains`' pyramids.
    pub(super) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        (atlas, terrains): (&TileAtlas, &RenderTerrains),
        max_layers: u32,
    ) {
        let (held, changes) = (&atlas.held, &atlas.held_changes[..]);
        self.uploads = 0;
        self.dropped = 0;
        let mut replaced = Vec::new();
        let mut arrived = Vec::new();
        for &slot in changes {
            let had = self.tiles.get(&slot).map(|blocks| Arc::clone(&blocks.tile));
            match (had, held.get(&slot)) {
                (Some(had), Some(tile)) if Arc::ptr_eq(&had, tile) => {}
                (Some(_), Some(tile)) => {
                    if let Some(blocks) = self.tiles.remove(&slot) {
                        replaced.push((slot, blocks, Arc::clone(tile)));
                    }
                }
                (Some(_), None) => {
                    if let Some(blocks) = self.tiles.remove(&slot) {
                        let used = blocks.blocks.iter().copied();
                        self.free
                            .extend(used.filter(|&block| block != NO_WATER_BLOCK));
                    }
                }
                (None, Some(tile)) => arrived.push((slot, Arc::clone(tile))),
                (None, None) => {}
            }
        }
        // Lower blocks first, so the arrays stay as small as they can.
        self.free.sort_unstable_by(|a, b| b.cmp(a));
        for (slot, before, tile) in replaced {
            let blocks = self.upload(device, queue, tile, Some(before), max_layers);
            self.tiles.insert(slot, blocks);
        }
        for (slot, tile) in arrived {
            let blocks = self.upload(device, queue, tile, None, max_layers);
            self.tiles.insert(slot, blocks);
        }
        if !changes.is_empty() || self.directory_layers != Some(atlas.layers()) {
            self.write_directory(device, queue, (atlas, terrains));
        }
    }

    /// Whether it holds blocks for exactly the tiles `held` holds, each of
    /// that tile, which debug builds check.
    #[cfg(debug_assertions)]
    pub(super) fn follows(&self, held: &HashMap<(u64, TileKey), Arc<Tile>>) -> bool {
        self.tiles.len() == held.len()
            && self.tiles.iter().all(|(slot, blocks)| {
                held.get(slot)
                    .is_some_and(|tile| Arc::ptr_eq(tile, &blocks.tile))
            })
    }

    /// `tile`'s blocks, its patches holding water uploaded: in place of
    /// `before`'s, the tile it replaces under its key, whose blocks it keeps
    /// where its patches' water stayed the same and frees where they hold
    /// none.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        tile: Arc<Tile>,
        before: Option<TileBlocks>,
        max_layers: u32,
    ) -> TileBlocks {
        let patches = PATCHES_PER_TILE as usize;
        let (mut blocks, held) = match before {
            Some(before) => (before.blocks, Some(before.tile)),
            None => (
                vec![NO_WATER_BLOCK; patches * patches].into_boxed_slice(),
                None,
            ),
        };
        let stride = LAYER_SIZE as usize;
        let side = BLOCK as usize;
        // A patch's samples in the tile's layout.
        let samples = |px: usize, pz: usize| {
            (0..side).flat_map(move |z| {
                (0..side).map(move |x| (pz * (side - 1) + z + 1) * stride + px * (side - 1) + x + 1)
            })
        };
        let mut surface_texels = vec![0.0_f32; side * side];
        let mut info_texels = vec![[0_u32; 4]; side * side];
        for pz in 0..patches {
            for px in 0..patches {
                let index = pz * patches + px;
                let water = tile
                    .water()
                    .filter(|water| water.patch_holds_water(px as u32, pz as u32));
                let Some(water) = water else {
                    if blocks[index] != NO_WATER_BLOCK {
                        self.free.push(blocks[index]);
                        blocks[index] = NO_WATER_BLOCK;
                    }
                    continue;
                };
                let kept = blocks[index] != NO_WATER_BLOCK
                    && held
                        .as_ref()
                        .and_then(|held| held.water())
                        .is_some_and(|before| {
                            samples(px, pz).all(|at| {
                                before.surfaces()[at].to_bits() == water.surfaces()[at].to_bits()
                                    && before.info()[at] == water.info()[at]
                            })
                        });
                if kept {
                    continue;
                }
                let block = match blocks[index] {
                    NO_WATER_BLOCK => match self.allocate(device, queue, max_layers) {
                        Some(block) => block,
                        None => {
                            self.dropped += 1;
                            continue;
                        }
                    },
                    block => block,
                };
                for (texel, at) in samples(px, pz).enumerate() {
                    let surface = water.surfaces()[at];
                    surface_texels[texel] = if surface.is_finite() {
                        surface
                    } else {
                        NO_WATER_SURFACE
                    };
                    let [packed, state, obstacle] = water.info()[at];
                    info_texels[texel] = [packed, state, obstacle, 0];
                }
                self.write(queue, block, &surface_texels, &info_texels);
                blocks[index] = block;
                self.uploads += 1;
            }
        }
        TileBlocks { tile, blocks }
    }

    /// A free block, the arrays grown to hold it; `None` beyond
    /// `max_layers`.
    fn allocate(&mut self, device: &wgpu::Device, queue: &Queue, max_layers: u32) -> Option<u32> {
        if let Some(block) = self.free.pop() {
            return Some(block);
        }
        let block = self.next;
        let needed = block / PER_LAYER + 1;
        if needed > max_layers {
            return None;
        }
        self.ensure_arrays(device, queue, needed, max_layers);
        self.next += 1;
        Some(block)
    }

    /// Grows the arrays to hold `needed` layers, doubling, keeping the
    /// blocks uploaded.
    fn ensure_arrays(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        needed: u32,
        max_layers: u32,
    ) {
        let current = self.arrays.as_ref().map_or(0, |arrays| arrays.layers);
        if needed <= current {
            return;
        }
        let layers = needed.next_power_of_two().min(max_layers.max(1));
        let array = |label: &'static str, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: LAYER_TEXELS,
                    height: LAYER_TEXELS,
                    depth_or_array_layers: layers,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let surface = array("engine/water-surfaces", SURFACE_FORMAT);
        let info = array("engine/water-info", INFO_FORMAT);
        if let Some(old) = &self.arrays {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/water-grow"),
            });
            let extent = wgpu::Extent3d {
                width: LAYER_TEXELS,
                height: LAYER_TEXELS,
                depth_or_array_layers: old.layers,
            };
            encoder.copy_texture_to_texture(
                old.surface.as_image_copy(),
                surface.as_image_copy(),
                extent,
            );
            encoder.copy_texture_to_texture(old.info.as_image_copy(), info.as_image_copy(), extent);
            queue.submit_now([encoder.finish()]);
        }
        let view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        self.arrays = Some(Arrays {
            surface_view: view(&surface),
            info_view: view(&info),
            surface,
            info,
            layers,
        });
        self.generation += 1;
    }

    /// Writes block `block`'s texels.
    fn write(&self, queue: &Queue, block: u32, surfaces: &[f32], info: &[[u32; 4]]) {
        let arrays = self.arrays.as_ref().expect("a block allocates its arrays");
        let index = block % PER_LAYER;
        let origin = wgpu::Origin3d {
            x: index % SIDE * BLOCK,
            y: index / SIDE * BLOCK,
            z: block / PER_LAYER,
        };
        let extent = wgpu::Extent3d {
            width: BLOCK,
            height: BLOCK,
            depth_or_array_layers: 1,
        };
        for (texture, bytes, texel) in [
            (
                &arrays.surface,
                bytemuck::cast_slice(surfaces),
                SURFACE_BYTES,
            ),
            (&arrays.info, bytemuck::cast_slice(info), INFO_BYTES),
        ] {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(BLOCK * texel),
                    rows_per_image: Some(BLOCK),
                },
                extent,
            );
        }
    }

    /// Lays water's directory out for the layers `atlas` holds and writes
    /// the words that changed: each resident tile's patches' blocks and
    /// their bits, which say where water stands over each patch, there or
    /// in a finer resident tile within it, worked out level by level up
    /// each of `terrains`' pyramids, through tiles not resident too.
    #[expect(
        clippy::iter_over_hash_type,
        reason = "bits gather and words land by tile and layer, in any order"
    )]
    fn write_directory(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        (atlas, terrains): (&TileAtlas, &RenderTerrains),
    ) {
        self.bits.clear();
        self.holding.clear();
        for (&(id, key), tile) in &self.tiles {
            let mut bits = [0_u32; BIT_WORDS];
            for (patch, &block) in tile.blocks.iter().enumerate() {
                if block != NO_WATER_BLOCK {
                    bits[patch / 32] |= 1 << (patch % 32);
                }
            }
            if bits != [0; BIT_WORDS] {
                self.holding.insert(id);
                self.bits.insert((id, key), bits);
            }
        }
        // Each level's bits into its parents', finest first.
        let levels = (terrains.terrains.iter())
            .map(|terrain| terrain.pyramid.levels())
            .max()
            .unwrap_or(0);
        let mut below = Vec::new();
        for level in 0..levels {
            below.clear();
            below.extend(
                (self.bits.iter())
                    .filter(|((_, key), _)| u32::from(key.level) == level)
                    .map(|(&slot, &bits)| (slot, bits)),
            );
            for &((id, key), bits) in &below {
                let parent = (terrains.terrains.iter())
                    .find(|terrain| terrain.id == id)
                    .and_then(|terrain| terrain.pyramid.parent(key));
                let Some(parent) = parent else {
                    continue;
                };
                let held = self.bits.entry((id, parent)).or_insert([0; BIT_WORDS]);
                for (word, set) in held.iter_mut().zip(covered(&bits, key)) {
                    *word |= set;
                }
            }
        }
        let layers = atlas.layers();
        let mut words = std::mem::take(&mut self.directory_spare);
        words.clear();
        for _ in 0..layers {
            words.resize(words.len() + TILE_BLOCKS, NO_WATER_BLOCK);
            words.resize(words.len() + BIT_WORDS, 0);
        }
        for (slot, &layer) in &atlas.slots {
            let at = layer as usize * TILE_WORDS as usize;
            let Some(entry) = words.get_mut(at..at + TILE_WORDS as usize) else {
                continue;
            };
            if let Some(tile) = self.tiles.get(slot) {
                entry[..TILE_BLOCKS].copy_from_slice(&tile.blocks);
            }
            if let Some(bits) = self.bits.get(slot) {
                entry[TILE_BLOCKS..].copy_from_slice(bits);
            }
        }
        self.directory_layers = Some(layers);
        let grown = self.directory.grow(device, (words.len() * 4) as u64);
        let run = if grown {
            0..words.len()
        } else {
            changed_run(&self.directory_written, &words)
        };
        if !run.is_empty() {
            let offset = (run.start * 4) as u64;
            queue.write_buffer(&self.directory, offset, bytemuck::cast_slice(&words[run]));
        }
        self.directory_spare = std::mem::replace(&mut self.directory_written, words);
    }
}

/// The bits tile `key`'s `bits` set in its parent tile's: those of the
/// quarter of the parent's patches it lies under, each over two of its
/// patches along each side.
fn covered(bits: &PatchBits, key: TileKey) -> PatchBits {
    let side = TILE_PATCHES as usize;
    let half = side / 2;
    let (first_x, first_z) = (usize::from(key.x % 2) * half, usize::from(key.z % 2) * half);
    let mut parent = [0; BIT_WORDS];
    for patch in 0..side * side {
        if bits[patch / 32] & (1 << (patch % 32)) != 0 {
            let at = (first_z + patch / side / 2) * side + first_x + patch % side / 2;
            parent[at / 32] |= 1 << (at % 32);
        }
    }
    parent
}
