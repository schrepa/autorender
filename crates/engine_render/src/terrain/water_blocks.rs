//! Tiles' water on the GPU: each resident tile's water where it stands over
//! the ground or continues under its banks at its level, a patch at a time,
//! in two texture arrays of blocks as many samples wide as a patch's grid
//! and a sample more past each of its edges: their surfaces as water's pass
//! draws them beside their tile's ground, and their words, their packed
//! flow, whitewater and kind and their state
//! (`engine_core::terrain_data::TileWater::info`), the third the level the
//! water's mesh stands at over the sample
//! (`engine_core::terrain_data::WaterMesh`) and the fourth none, WebGPU
//! having no texel of three words; each sample past the patch's edges the
//! tile's beside them (its border past its own), so the smooth
//! surface water's pass reads through them (`meshes/water.wgsl`'s
//! `water_level`) meets itself across tiles and patches. A tile's blocks
//! upload as it arrives and free as it leaves or changes, following the
//! tiles the atlas held anew; a patch whose samples hold no surface
//! (`TileWater::patch_holds_water`) takes none, the water's mesh standing on
//! its ground there. Water's directory
//! lists, for each layer of the height array, the blocks of its tile's
//! patches, laid out again as tiles or their water change, in the terrain's
//! directory buffer after the directory's own words: shaders
//! read a tile's water at any point through it, water's pass
//! (`crate::water`) and the ground's fragments, which wet the ground at its
//! waterline and bind no more storage buffers than they did.

#![expect(
    clippy::expect_used,
    reason = "the arrays are allocated before blocks upload"
)]

use std::collections::BTreeSet;

use super::atlas::TileAtlas;
use super::*;
use crate::gpu_layouts::constants::{
    NO_WATER_BLOCK, NO_WATER_SURFACE, WATER_BLOCK as BLOCK, WATER_BLOCKS_SIDE as SIDE,
    WATER_TILE_WORDS as TILE_WORDS,
};
use engine_core::terrain_data::{PATCH_QUADS, PATCHES_PER_TILE, Tile, TileWater, WaterMesh};

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
/// (`NO_WATER_BLOCK` where none of its samples holds a surface, or no layer
/// held it).
struct TileBlocks {
    tile: Arc<Tile>,
    blocks: Box<[u32]>,
}

/// A block's texels: each sample's surface beside its ground, and its words
/// with the level the water's mesh stands at over it, and none.
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
/// and terrains' uniforms, the terrain's directory with water's in it, this
/// frame's leaves of the subdivisions and their trees' uniforms and the
/// water blocks' arrays, with generations advancing when any of them is
/// another texture or buffer; each frame terrain whose water subdivides,
/// its id, its uniform's offset, its water's tree's and that tree's draw of
/// its leaves; the pixels the drawn ground (x) and water's mesh (y) may stray
/// from their surfaces seen from the detail eye;
/// the blocks resident and dropped this frame for want of layers; the
/// leaves the water's subdivisions drew and the splits their room refused,
/// as last read back; and the terrain group as the subdivisions' passes
/// bind it, with its layout, once made, through which the live water's bed
/// reads the ground's materials (`meshes/water_bed.wgsl`).
pub(crate) struct TerrainWaterDraws<'a> {
    pub(crate) heights: &'a wgpu::TextureView,
    pub(crate) views: &'a wgpu::Buffer,
    pub(crate) uniforms: &'a wgpu::Buffer,
    pub(crate) directory: &'a wgpu::Buffer,
    pub(crate) leaves: &'a wgpu::Buffer,
    pub(crate) trees: &'a wgpu::Buffer,
    pub(crate) surfaces: &'a wgpu::TextureView,
    pub(crate) info: &'a wgpu::TextureView,
    pub(crate) generation: [u64; 6],
    pub(crate) terrains: Vec<(u64, u32, u32, SurfaceDraw)>,
    pub(crate) targets: [f32; 2],
    pub(crate) blocks: [u32; 2],
    pub(crate) subdivided: [u32; 2],
    pub(crate) ground: Option<(&'a crate::pipelines::GroupLayout, &'a wgpu::BindGroup)>,
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
    // Water's directory: the words the terrain's directory buffer holds of
    // it and those they replaced, kept for the next layout; the height
    // array's layers it was laid out for, and where it was written (the
    // word it starts at and the buffer's generation); whether tiles changed
    // since; and the terrains holding water.
    directory_written: Vec<u32>,
    directory_spare: Vec<u32>,
    directory_layers: Option<u32>,
    directory_at: Option<(u32, u64)>,
    directory_stale: bool,
    holding: BTreeSet<u64>,
    // The surfaces' array the terrain group binds before a block uploads,
    // while water's directory names none: a texel.
    stand_in: wgpu::TextureView,
}

impl WaterBlocks {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let stand_in = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/water-surfaces-stand-in"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SURFACE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self {
            arrays: None,
            generation: 0,
            tiles: HashMap::default(),
            free: Vec::new(),
            next: 0,
            uploads: 0,
            dropped: 0,
            directory_written: Vec::new(),
            directory_spare: Vec::new(),
            directory_layers: None,
            directory_at: None,
            directory_stale: true,
            holding: BTreeSet::new(),
            stand_in: stand_in.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }),
        }
    }

    /// The surface and info arrays' views, once a block uploaded.
    pub(crate) fn views(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        self.arrays
            .as_ref()
            .map(|arrays| (&arrays.surface_view, &arrays.info_view))
    }

    /// The surface array's view the terrain group binds, whose surfaces
    /// beside their ground the ground's fragments wet it by at its
    /// waterline: a stand-in before a block uploads.
    pub(super) fn surface_view(&self) -> &wgpu::TextureView {
        self.arrays
            .as_ref()
            .map_or(&self.stand_in, |arrays| &arrays.surface_view)
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

    /// Bytes of the arrays.
    pub(super) fn capacity_bytes(&self) -> u64 {
        self.arrays.as_ref().map_or(0, |arrays| {
            u64::from(arrays.layers)
                * u64::from(LAYER_TEXELS * LAYER_TEXELS)
                * u64::from(SURFACE_BYTES + INFO_BYTES)
        })
    }

    /// What water's pass draws tiles' water with this frame, of `pass`'s
    /// parts and `terrains`; `None` before tiles and their water arrive, or
    /// the leaves of the subdivisions.
    pub(super) fn draws<'a>(
        &'a self,
        pass: &'a TerrainPass,
        terrains: &RenderTerrains,
    ) -> Option<TerrainWaterDraws<'a>> {
        let (heights, _, _) = pass.atlas.views()?;
        let (surfaces, info) = self.views()?;
        let (leaves, leaves_generation) = pass.surface_leaves()?;
        let trees = pass.surface.uniforms();
        let terrains = (terrains.terrains.iter().enumerate())
            .filter_map(|(index, terrain)| {
                let uniform = pass.tables.uniform_of(terrain.id)?;
                let (tree, leaves) = pass.surface.water(index)?;
                Some((terrain.id, uniform, tree, leaves))
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
            generation: [
                pass.atlas.generation,
                pass.tables.uniform_generation,
                self.generation,
                pass.tables.directory_generation,
                leaves_generation,
                trees.generation(),
            ],
            terrains,
            targets: pass.surface.targets(),
            blocks: [self.resident(), self.dropped],
            subdivided: pass.surface.water_stats(),
            ground: pass.pipelines.surface_group().map(|group| (&pass.pipelines.layout, group)),
        })
    }

    /// Uploads the blocks of the tiles `atlas` holds (by terrain and key)
    /// that arrived or changed, and frees those of tiles gone, looking only
    /// at the keys whose held tile changed since the last sync; a tile
    /// replaced under its key keeps the blocks whose texels stay the same
    /// (its water made again where its rivers run otherwise). Layers grow up
    /// to `max_layers`. Where any tile changed, water's directory is laid
    /// out again as it is next written ([`Self::write_directory`]).
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
        if !changes.is_empty() {
            self.directory_stale = true;
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
                let holds = tile
                    .water()
                    .zip(tile.water_mesh())
                    .is_some_and(|(water, mesh)| {
                        let holds = water.patch_holds_water(px as u32, pz as u32);
                        if holds {
                            block_texels(&tile, water, mesh, (px, pz), &mut texels);
                        }
                        holds
                    });
                if !holds {
                    if blocks[index] != NO_WATER_BLOCK {
                        self.free.push(blocks[index]);
                        blocks[index] = NO_WATER_BLOCK;
                    }
                    continue;
                }
                let same = blocks[index] != NO_WATER_BLOCK
                    && held.as_ref().is_some_and(|held| {
                        held.water()
                            .zip(held.water_mesh())
                            .is_some_and(|(before, mesh)| {
                                block_texels(held, before, mesh, (px, pz), &mut kept);
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
            (&arrays.info, bytemuck::cast_slice(&texels.info), INFO_BYTES),
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

    /// Lays water's directory out for the layers `atlas` holds where tiles
    /// changed since, or the layers did, and writes the words that changed
    /// into the terrain's directory buffer `directory` from its word `base`,
    /// all of them into a new buffer (another `generation`) or at another
    /// place: each resident tile's patches' blocks; and notes the terrains
    /// whose tiles hold any. Words past what the buffer holds, which it
    /// could not grow to, are left out.
    #[expect(
        clippy::iter_over_hash_type,
        reason = "terrains gather and words land by tile and layer, in any order"
    )]
    pub(super) fn write_directory(
        &mut self,
        queue: &Queue,
        atlas: &TileAtlas,
        (directory, base, generation): (&wgpu::Buffer, u32, u64),
    ) {
        let layers = atlas.layers();
        let whole = self.directory_at != Some((base, generation));
        if !whole && !self.directory_stale && self.directory_layers == Some(layers) {
            return;
        }
        self.directory_stale = false;
        self.directory_at = Some((base, generation));
        self.holding.clear();
        for (&(id, _), tile) in &self.tiles {
            if tile.blocks.iter().any(|&block| block != NO_WATER_BLOCK) {
                self.holding.insert(id);
            }
        }
        let entry = TILE_WORDS as usize;
        let mut words = std::mem::take(&mut self.directory_spare);
        words.clear();
        words.resize(layers as usize * entry, NO_WATER_BLOCK);
        for (slot, &layer) in &atlas.slots {
            let at = layer as usize * entry;
            if let (Some(held), Some(tile)) = (words.get_mut(at..at + entry), self.tiles.get(slot))
            {
                held.copy_from_slice(&tile.blocks);
            }
        }
        self.directory_layers = Some(layers);
        let run = if whole {
            0..words.len()
        } else {
            changed_run(&self.directory_written, &words)
        };
        let room = (directory.size() / 4) as usize;
        let run = run.start..run.end.min(room.saturating_sub(base as usize));
        if !run.is_empty() {
            let offset = ((base as usize + run.start) * 4) as u64;
            queue.write_buffer(directory, offset, bytemuck::cast_slice(&words[run]));
        }
        self.directory_spare = std::mem::replace(&mut self.directory_written, words);
    }
}

/// Writes into `texels` the block of patch `(px, pz)` of `tile`, whose
/// water is `water` and its mesh `mesh`: of each of its samples and those a
/// sample past its edges, the tile's beside it (its border past its own),
/// the surface water's pass draws (`NO_WATER_SURFACE` where none is) beside
/// the tile's ground, and its words, the third the bits of the level the
/// water's mesh stands at over it and the fourth none.
fn block_texels(
    tile: &Tile,
    water: &TileWater,
    mesh: &WaterMesh,
    (px, pz): (usize, usize),
    texels: &mut Texels,
) {
    let stride = LAYER_SIZE as usize;
    let side = BLOCK as usize;
    let quads = PATCH_QUADS as usize;
    let (heights, words) = (tile.heights(), water.info());
    for z in 0..side {
        for x in 0..side {
            // The sample in the tile's layout, which holds a border of one,
            // the block's first a sample before the patch's.
            let at = (pz * quads + z) * stride + px * quads + x;
            let texel = z * side + x;
            let level = mesh.level(at);
            texels.surfaces[texel] = [
                if mesh.holds(at) {
                    level
                } else {
                    NO_WATER_SURFACE
                },
                heights[at],
            ];
            let [packed, state] = words[at];
            texels.info[texel] = [packed, state, level.to_bits(), 0];
        }
    }
}
