//! Tiles' water on the GPU: each resident tile's water where it stands over
//! the ground or continues under its banks, a patch at a time, in two
//! texture arrays of blocks as many samples wide as a patch's grid: their
//! surfaces beside their tile's ground, and their words, their packed flow,
//! whitewater and kind, their state and their nearest obstacle
//! (`engine_core::terrain_data::TileWater::info`), and a fourth, how fast
//! their surface and their depth fall along x and z, worked out as the block
//! uploads from the tile's samples on either side, so blocks meet across
//! patches. A tile's blocks upload as it arrives and free as it leaves or
//! changes, following the tiles the atlas held anew. Water's directory
//! lists, for each layer of the height array, the blocks of its tile's
//! patches, laid out again as tiles or their water change: shaders read a
//! tile's water at any point through it (`crate::water`).

#![expect(
    clippy::expect_used,
    reason = "the arrays are allocated before blocks upload"
)]

use std::collections::BTreeSet;

use super::atlas::TileAtlas;
use super::*;
use crate::gpu_layouts::constants::{
    NO_WATER_BLOCK, NO_WATER_SURFACE, WATER_BLOCK as BLOCK, WATER_BLOCKS_SIDE as SIDE,
    WATER_GRADIENT_RANGE, WATER_TILE_WORDS as TILE_WORDS,
};
use engine_core::terrain_data::{PATCHES_PER_TILE, Tile, TileWater};

/// Blocks a layer holds, and its texels along a side.
const PER_LAYER: u32 = SIDE * SIDE;
const LAYER_TEXELS: u32 = BLOCK * SIDE;
/// The surface and info arrays' formats, and the bytes of each one's
/// texels: shaders unpack the info's words themselves.
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
const INFO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
const SURFACE_BYTES: u32 = 8;
const INFO_BYTES: u32 = 16;
/// Texels of a block.
const BLOCK_TEXELS: usize = (BLOCK * BLOCK) as usize;

/// The arrays and their layers.
struct Arrays {
    surface: wgpu::Texture,
    surface_view: wgpu::TextureView,
    info: wgpu::Texture,
    info_view: wgpu::TextureView,
    layers: u32,
}

/// A tile's blocks: the tile whose water they hold, and each patch's block
/// (`NO_WATER_BLOCK` where none of its samples holds a surface).
struct TileBlocks {
    tile: Arc<Tile>,
    blocks: Box<[u32]>,
}

/// A block's texels: each sample's surface beside its ground, and its words.
struct Texels {
    surfaces: Vec<[f32; 2]>,
    info: Vec<[u32; 4]>,
}

impl Texels {
    fn new() -> Self {
        Self {
            surfaces: vec![[0.0; 2]; BLOCK_TEXELS],
            info: vec![[0; 4]; BLOCK_TEXELS],
        }
    }

    /// Whether it holds the same texels as `other`, bit for bit.
    fn same(&self, other: &Self) -> bool {
        bytemuck::cast_slice::<[f32; 2], u32>(&self.surfaces)
            == bytemuck::cast_slice::<[f32; 2], u32>(&other.surfaces)
            && self.info == other.info
    }
}

/// What water's pass draws tiles' water with: the height array, the views'
/// and terrains' uniforms, the terrain's directory, this frame's leaves of
/// the subdivisions and their trees' uniforms, the water blocks' arrays and
/// water's directory, with generations advancing when any of them is
/// another texture or buffer; each frame terrain whose water subdivides,
/// its uniform's offset, its water's tree's and that tree's draw of its
/// leaves; the pixels the drawn ground (x) and water's mesh (y) may stray
/// from their surfaces seen from the detail eye;
/// the blocks resident and dropped this frame for want of layers; and the
/// leaves the water's subdivisions drew and the splits their room refused,
/// as last read back.
pub(crate) struct TerrainWaterDraws<'a> {
    pub(crate) heights: &'a wgpu::TextureView,
    pub(crate) views: &'a wgpu::Buffer,
    pub(crate) uniforms: &'a wgpu::Buffer,
    pub(crate) directory: &'a wgpu::Buffer,
    pub(crate) leaves: &'a wgpu::Buffer,
    pub(crate) trees: &'a wgpu::Buffer,
    pub(crate) surfaces: &'a wgpu::TextureView,
    pub(crate) info: &'a wgpu::TextureView,
    pub(crate) water_directory: &'a wgpu::Buffer,
    pub(crate) generation: [u64; 7],
    pub(crate) terrains: Vec<(u32, u32, SurfaceDraw)>,
    pub(crate) targets: [f32; 2],
    pub(crate) blocks: [u32; 2],
    pub(crate) subdivided: [u32; 2],
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
    // for; and the terrains holding water.
    directory: crate::buffers::Growable,
    directory_written: Vec<u32>,
    directory_spare: Vec<u32>,
    directory_layers: Option<u32>,
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

    /// Whether terrain `id`'s resident tiles hold water, as the directory
    /// was last laid out.
    pub(super) fn holds(&self, id: u64) -> bool {
        self.holding.contains(&id)
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
    /// the leaves of the subdivisions.
    pub(super) fn draws<'a>(
        &'a self,
        pass: &'a TerrainPass,
        terrains: &RenderTerrains,
    ) -> Option<TerrainWaterDraws<'a>> {
        let heights = pass.atlas.heights()?;
        let (surfaces, info) = self.views()?;
        let (leaves, leaves_generation) = pass.surface_leaves()?;
        let trees = pass.surface.uniforms();
        let terrains = (terrains.terrains.iter().enumerate())
            .filter_map(|(index, terrain)| {
                let uniform = pass.tables.uniform_of(terrain.id)?;
                let (tree, leaves) = pass.surface.water(index)?;
                Some((uniform, tree, leaves))
            })
            .collect();
        Some(TerrainWaterDraws {
            heights,
            views: &pass.patches.views,
            uniforms: &pass.tables.uniforms,
            directory: &pass.tables.directory,
            leaves,
            trees,
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
                trees.generation(),
            ],
            terrains,
            targets: pass.surface.targets(),
            blocks: [self.resident(), self.dropped],
            subdivided: pass.surface.water_stats(),
        })
    }

    /// Uploads the blocks of the tiles `atlas` holds (by terrain and key)
    /// that arrived or changed, and frees those of tiles gone, looking only
    /// at the keys whose held tile changed since the last sync; a tile
    /// replaced under its key keeps the blocks whose texels stay the same
    /// (its water made again where its rivers run otherwise). Layers grow up
    /// to `max_layers`. Where any tile changed, or the height array's layers
    /// did, water's directory is laid out again.
    pub(super) fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &Queue,
        atlas: &TileAtlas,
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
            self.write_directory(device, queue, atlas);
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

    /// `tile`'s blocks, its patches holding a surface uploaded: in place of
    /// `before`'s, the tile it replaces under its key, whose blocks it keeps
    /// where their texels stay the same and frees where its patches hold
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
        let mut texels = Texels::new();
        let mut kept = Texels::new();
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
                block_texels(&tile, water, (px, pz), &mut texels);
                let same = blocks[index] != NO_WATER_BLOCK
                    && held.as_ref().is_some_and(|held| {
                        held.water().is_some_and(|before| {
                            block_texels(held, before, (px, pz), &mut kept);
                            kept.same(&texels)
                        })
                    });
                if same {
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
                self.write(queue, block, &texels);
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
    fn write(&self, queue: &Queue, block: u32, texels: &Texels) {
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
                bytemuck::cast_slice(&texels.surfaces),
                SURFACE_BYTES,
            ),
            (
                &arrays.info,
                bytemuck::cast_slice(&texels.info),
                INFO_BYTES,
            ),
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
    /// the words that changed: each resident tile's patches' blocks; and
    /// notes the terrains whose tiles hold any.
    #[expect(
        clippy::iter_over_hash_type,
        reason = "terrains gather and words land by tile and layer, in any order"
    )]
    fn write_directory(&mut self, device: &wgpu::Device, queue: &Queue, atlas: &TileAtlas) {
        self.holding.clear();
        for (&(id, _), tile) in &self.tiles {
            if tile.blocks.iter().any(|&block| block != NO_WATER_BLOCK) {
                self.holding.insert(id);
            }
        }
        let layers = atlas.layers();
        let entry = TILE_WORDS as usize;
        let mut words = std::mem::take(&mut self.directory_spare);
        words.clear();
        words.resize(layers as usize * entry, NO_WATER_BLOCK);
        for (slot, &layer) in &atlas.slots {
            let at = layer as usize * entry;
            if let (Some(held), Some(tile)) =
                (words.get_mut(at..at + entry), self.tiles.get(slot))
            {
                held.copy_from_slice(&tile.blocks);
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

/// Writes into `texels` the block of patch `(px, pz)` of `tile`, whose
/// water is `water`: each of its samples' surface (`NO_WATER_SURFACE` where
/// none reaches) beside the tile's ground, and its words, the fourth how
/// fast its surface and its depth fall along x and z (`pack_gradients`),
/// from the tile's samples either side.
fn block_texels(tile: &Tile, water: &TileWater, (px, pz): (usize, usize), texels: &mut Texels) {
    let stride = LAYER_SIZE as usize;
    let side = BLOCK as usize;
    let (heights, levels, words) = (tile.heights(), water.surfaces(), water.info());
    let spacing = tile.spacing();
    let surface = |at: usize| levels[at];
    let depth = |at: usize| levels[at] - heights[at];
    for z in 0..side {
        for x in 0..side {
            // The sample in the tile's layout, which holds a border of one.
            let at = (pz * (side - 1) + z + 1) * stride + px * (side - 1) + x + 1;
            let texel = z * side + x;
            let level = levels[at];
            texels.surfaces[texel] = [
                if level.is_finite() {
                    level
                } else {
                    NO_WATER_SURFACE
                },
                heights[at],
            ];
            let [packed, state, obstacle] = words[at];
            let gradients = pack_gradients([
                falling(&surface, at, 1, spacing),
                falling(&surface, at, stride, spacing),
                falling(&depth, at, 1, spacing),
                falling(&depth, at, stride, spacing),
            ]);
            texels.info[texel] = [packed, state, obstacle, gradients];
        }
    }
}

/// How fast `value` (of a tile's samples by their place in its layout, NaN
/// where a sample holds no surface) falls a meter along `step` places at
/// sample `at`, its samples `spacing` meters apart: between the samples
/// either side holding one, or between `at` and the one that does; none
/// where neither does or `at` holds none.
fn falling(value: &dyn Fn(usize) -> f32, at: usize, step: usize, spacing: f32) -> f32 {
    let (before, here, after) = (value(at - step), value(at), value(at + step));
    if !here.is_finite() {
        return 0.0;
    }
    match (before.is_finite(), after.is_finite()) {
        (true, true) => (before - after) / (2.0 * spacing),
        (true, false) => (before - here) / spacing,
        (false, true) => (here - after) / spacing,
        (false, false) => 0.0,
    }
}

/// A sample's `gradients` (how fast its surface falls along x and z, then
/// its depth, meters a meter) as the four signed bytes of a word, low
/// first, which shaders read as snorm shares: each the square root of its
/// share of `WATER_GRADIENT_RANGE`, signed, so gentle slopes keep fine
/// steps.
fn pack_gradients(gradients: [f32; 4]) -> u32 {
    gradients
        .iter()
        .enumerate()
        .fold(0, |word, (index, &gradient)| {
            let share = engine_core::math::clamp(gradient.abs() / WATER_GRADIENT_RANGE, 0.0, 1.0);
            let steps = (share.sqrt() * 127.0).round();
            let signed = if gradient < 0.0 { -steps } else { steps };
            word | u32::from(signed as i8 as u8) << (8 * index)
        })
}
