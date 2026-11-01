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
use engine_core::glam::Vec2;
use engine_terrain::shallow_water::roughness;
use engine_terrain::{PATCH_QUADS, PATCHES_PER_TILE, Tile, TileWater, WaterMesh};

const PER_LAYER: u32 = SIDE * SIDE;
const LAYER_TEXELS: u32 = BLOCK * SIDE;
const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
const INFO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
const SURFACE_BYTES: u32 = 8;
const INFO_BYTES: u32 = 16;
const BLOCK_TEXELS: usize = (BLOCK * BLOCK) as usize;

struct Arrays {
    surface: wgpu::Texture,
    surface_view: wgpu::TextureView,
    info: wgpu::Texture,
    info_view: wgpu::TextureView,
    layers: u32,
}

struct TileBlocks {
    tile: Arc<Tile>,
    blocks: Box<[u32]>,
}

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

    fn same(&self, other: &Self) -> bool {
        bytemuck::cast_slice::<[f32; 2], u32>(&self.surfaces)
            == bytemuck::cast_slice::<[f32; 2], u32>(&other.surfaces)
            && self.info == other.info
    }
}

pub(crate) struct WaterBlocks {
    arrays: Option<Arrays>,
    pub(crate) generation: u64,
    tiles: HashMap<(u64, TileKey), TileBlocks>,
    free: Vec<u32>,
    next: u32,
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    directory_written: Vec<u32>,
    directory_spare: Vec<u32>,
    directory_layers: Option<u32>,
    directory_at: Option<(u32, u64)>,
    directory_stale: bool,
    holding: BTreeSet<u64>,
    stand_in: [wgpu::TextureView; 2],
}

impl WaterBlocks {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let stand_in = |label: &'static str, format| {
            let texture = engine_render::memory::texture(
                device,
                &wgpu::TextureDescriptor {
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
                },
            );
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
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
            stand_in: [
                stand_in("engine/water-surfaces-stand-in", SURFACE_FORMAT),
                stand_in("engine/water-info-stand-in", INFO_FORMAT),
            ],
        }
    }

    pub(crate) fn views(&self) -> Option<(&wgpu::TextureView, &wgpu::TextureView)> {
        self.arrays
            .as_ref()
            .map(|arrays| (&arrays.surface_view, &arrays.info_view))
    }

    pub(super) fn surface_view(&self) -> &wgpu::TextureView {
        self.arrays
            .as_ref()
            .map_or(&self.stand_in[0], |arrays| &arrays.surface_view)
    }

    pub(super) fn info_view(&self) -> &wgpu::TextureView {
        self.arrays
            .as_ref()
            .map_or(&self.stand_in[1], |arrays| &arrays.info_view)
    }

    pub(crate) fn resident(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    pub(super) fn holds(&self, id: u64) -> bool {
        self.holding.contains(&id)
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        self.arrays.as_ref().map_or(0, |arrays| {
            u64::from(arrays.layers)
                * u64::from(LAYER_TEXELS * LAYER_TEXELS)
                * u64::from(SURFACE_BYTES + INFO_BYTES)
        })
    }

    pub(super) fn draws<'a>(
        &'a self,
        pass: &'a TerrainPass,
        terrains: &RenderTerrains,
    ) -> Option<WaterDraws<'a>> {
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
        Some(WaterDraws {
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
        })
    }

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

    #[cfg(debug_assertions)]
    pub(super) fn follows(&self, held: &HashMap<(u64, TileKey), Arc<Tile>>) -> bool {
        self.tiles.len() == held.len()
            && self.tiles.iter().all(|(slot, blocks)| {
                held.get(slot)
                    .is_some_and(|tile| Arc::ptr_eq(tile, &blocks.tile))
            })
    }

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
            engine_render::memory::texture(
                device,
                &wgpu::TextureDescriptor {
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
                },
            )
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

    fn write(&self, queue: &Queue, block: u32, texels: &Texels) {
        let arrays = self.arrays.as_ref().expect("block_arrays_allocated");
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
    let (heights, words, spacing) = (tile.heights(), water.info(), tile.spacing());
    for z in 0..side {
        for x in 0..side {
            let (column, row) = (px * quads + x, pz * quads + z);
            let at = row * stride + column;
            let local = Vec2::new(column as f32 - 1.0, row as f32 - 1.0) * spacing;
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
            let rough = roughness(tile.bed(local).as_ref());
            texels.info[texel] = [packed, state, level.to_bits(), rough.to_bits()];
        }
    }
}
