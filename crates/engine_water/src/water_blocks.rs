use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::DeviceLimits;
use engine_heightfield::TileKey;
use engine_heightfield::tile::{PATCH_QUADS, PATCHES_PER_TILE, TILE_STRIDE, bed_at};
use engine_places::ground::{GroundTiles, TileFace};
use engine_render::{Buffer, Prepare, Texture, wgpu};
use glam::Vec2;

use crate::gpu::{
    NO_WATER_BLOCK, NO_WATER_SURFACE, WATER_BLOCK as BLOCK, WATER_BLOCKS_SIDE as SIDE,
    WATER_TILE_WORDS as TILE_WORDS,
};
use crate::shallow_water::roughness;
use crate::state::WaterState;
use crate::water::WaterGpu;
use crate::{TileWater, WaterMesh};

const PER_LAYER: u32 = SIDE * SIDE;
const LAYER_TEXELS: u32 = BLOCK * SIDE;
pub(crate) const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
pub(crate) const INFO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;
const SURFACE_BYTES: u32 = 8;
const INFO_BYTES: u32 = 16;
const BLOCK_TEXELS: usize = (BLOCK * BLOCK) as usize;

#[derive(Clone)]
pub(crate) struct Texels {
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

#[derive(Clone)]
struct Held {
    face: TileFace,
    water: Option<Arc<TileWater>>,
    mesh: Option<Arc<WaterMesh>>,
}

impl Held {
    fn same(&self, other: &Self) -> bool {
        let alike = |a: Option<&Arc<TileWater>>, b: Option<&Arc<TileWater>>| match (a, b) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        self.face.stamp == other.face.stamp && alike(self.water.as_ref(), other.water.as_ref())
    }
}

struct TileBlocks {
    held: Held,
    blocks: Box<[u32]>,
}

#[derive(Default)]
pub(crate) struct Blocks {
    tiles: BTreeMap<(Entity, TileKey), TileBlocks>,
    free: Vec<u32>,
    next: u32,
    pub(crate) layers: u32,
    pub(crate) generation: u64,
    pub(crate) writes: Vec<(u32, Texels)>,
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    pub(crate) directory: Vec<u32>,
    pub(crate) written: Range<usize>,
    pub(crate) directory_generation: u64,
    directory_layers: Option<u32>,
    stale: bool,
}

impl Blocks {
    pub(crate) fn resident(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    pub(crate) fn current(&self, terrain: Entity, key: TileKey, water: Option<&Arc<TileWater>>) -> bool {
        self.tiles.get(&(terrain, key)).is_some_and(|tile| {
            match (tile.held.water.as_ref(), water) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
        })
    }

    fn sync(&mut self, held: Vec<((Entity, TileKey), Held)>, max_layers: u32) {
        self.uploads = 0;
        self.dropped = 0;
        self.writes.clear();
        let mut replaced = Vec::new();
        let mut arrived = Vec::new();
        let mut changed = false;
        let mut kept = BTreeMap::new();
        for (slot, tile) in held {
            match self.tiles.remove(&slot) {
                Some(had) if had.held.same(&tile) => {
                    kept.insert(slot, had);
                }
                Some(had) => {
                    changed = true;
                    replaced.push((slot, had, tile));
                }
                None => {
                    changed = true;
                    arrived.push((slot, tile));
                }
            }
        }
        for (_, gone) in std::mem::replace(&mut self.tiles, kept) {
            changed = true;
            self.free
                .extend(gone.blocks.iter().copied().filter(|&block| block != NO_WATER_BLOCK));
        }
        self.free.sort_unstable_by(|a, b| b.cmp(a));
        for (slot, before, tile) in replaced {
            let blocks = self.upload(tile, Some(before), max_layers);
            self.tiles.insert(slot, blocks);
        }
        for (slot, tile) in arrived {
            let blocks = self.upload(tile, None, max_layers);
            self.tiles.insert(slot, blocks);
        }
        if changed {
            self.stale = true;
        }
    }

    fn upload(&mut self, tile: Held, before: Option<TileBlocks>, max_layers: u32) -> TileBlocks {
        let patches = PATCHES_PER_TILE as usize;
        let (mut blocks, held) = match before {
            Some(before) => (before.blocks, Some(before.held)),
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
                    .water
                    .as_deref()
                    .zip(tile.mesh.as_deref())
                    .is_some_and(|(water, mesh)| {
                        let holds = water.patch_holds_water(px as u32, pz as u32);
                        if holds {
                            block_texels(&tile.face, water, mesh, (px, pz), &mut texels);
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
                        held.water
                            .as_deref()
                            .zip(held.mesh.as_deref())
                            .is_some_and(|(before, mesh)| {
                                block_texels(&held.face, before, mesh, (px, pz), &mut kept);
                                kept.same(&texels)
                            })
                    });
                if same {
                    continue;
                }
                let block = match blocks[index] {
                    NO_WATER_BLOCK => match self.allocate(max_layers) {
                        Some(block) => block,
                        None => {
                            self.dropped += 1;
                            continue;
                        }
                    },
                    block => block,
                };
                self.writes.push((block, texels.clone()));
                blocks[index] = block;
                self.uploads += 1;
            }
        }
        TileBlocks { held: tile, blocks }
    }

    fn allocate(&mut self, max_layers: u32) -> Option<u32> {
        if let Some(block) = self.free.pop() {
            return Some(block);
        }
        let block = self.next;
        let needed = block / PER_LAYER + 1;
        if needed > max_layers {
            return None;
        }
        if needed > self.layers {
            self.layers = needed.next_power_of_two().min(max_layers.max(1));
            self.generation += 1;
        }
        self.next += 1;
        Some(block)
    }

    fn rewrite(&mut self) {
        let mut texels = Texels::new();
        let patches = PATCHES_PER_TILE as usize;
        let written: Vec<u32> = self.writes.iter().map(|(block, _)| *block).collect();
        for tile in self.tiles.values() {
            let (Some(water), Some(mesh)) = (tile.held.water.as_deref(), tile.held.mesh.as_deref())
            else {
                continue;
            };
            for (index, &block) in tile.blocks.iter().enumerate() {
                if block == NO_WATER_BLOCK || written.contains(&block) {
                    continue;
                }
                block_texels(&tile.held.face, water, mesh, (index % patches, index / patches), &mut texels);
                self.writes.push((block, texels.clone()));
            }
        }
    }

    fn write_directory(&mut self, ground: &GroundTiles) {
        let layers = ground.layers;
        if !self.stale && self.directory_layers == Some(layers) {
            self.written = 0..0;
            return;
        }
        self.stale = false;
        let entry = TILE_WORDS as usize;
        let mut words = vec![NO_WATER_BLOCK; layers as usize * entry];
        for held in &ground.held {
            let at = held.layer as usize * entry;
            if let (Some(slot), Some(tile)) = (
                words.get_mut(at..at + entry),
                self.tiles.get(&(held.terrain, held.key)),
            ) {
                slot.copy_from_slice(&tile.blocks);
            }
        }
        let whole = self.directory_layers != Some(layers);
        self.directory_layers = Some(layers);
        self.written = if whole {
            self.directory_generation += 1;
            0..words.len()
        } else {
            changed_run(&self.directory, &words)
        };
        self.directory = words;
    }

    pub(crate) fn holding(&self) -> Vec<Entity> {
        let mut holding: Vec<Entity> = self
            .tiles
            .iter()
            .filter(|(_, tile)| tile.blocks.iter().any(|&block| block != NO_WATER_BLOCK))
            .map(|(&(terrain, _), _)| terrain)
            .collect();
        holding.dedup();
        holding
    }
}

fn changed_run(was: &[u32], now: &[u32]) -> Range<usize> {
    let shared = now.len().min(was.len());
    let Some(first) = (0..now.len()).find(|&at| at >= shared || now[at] != was[at]) else {
        return 0..0;
    };
    let end = if now.len() > was.len() {
        now.len()
    } else {
        (first..now.len())
            .rev()
            .find(|&at| now[at] != was[at])
            .map_or(first + 1, |at| at + 1)
    };
    first..end
}

fn block_texels(
    face: &TileFace,
    water: &TileWater,
    mesh: &WaterMesh,
    (px, pz): (usize, usize),
    texels: &mut Texels,
) {
    let stride = TILE_STRIDE as usize;
    let side = BLOCK as usize;
    let quads = PATCH_QUADS as usize;
    let (heights, words, spacing) = (&face.heights[..], water.info(), face.spacing);
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
            let bed = face
                .river_beds
                .as_deref()
                .and_then(|beds| bed_at(beds, heights, spacing, local));
            let rough = roughness(bed.as_ref());
            texels.info[texel] = [packed, state, level.to_bits(), rough.to_bits()];
        }
    }
}

pub(crate) fn place_blocks(
    ground: Res<GroundTiles>,
    limits: Res<DeviceLimits>,
    mut state: ResMut<WaterState>,
) {
    let state = &mut *state;
    let mut held = Vec::with_capacity(ground.held.len());
    for tile in &ground.held {
        let Some(terrain) = state.terrains.iter().find(|terrain| terrain.entity == tile.terrain)
        else {
            continue;
        };
        let Some(water) = terrain
            .held
            .get(&tile.key)
            .filter(|water| water.resident() && water.face.stamp == tile.stamp)
        else {
            continue;
        };
        held.push((
            (tile.terrain, tile.key),
            Held {
                face: water.face.clone(),
                water: water.water.clone(),
                mesh: water.mesh.clone(),
            },
        ));
    }
    held.sort_unstable_by_key(|(slot, _)| *slot);
    let generation = state.blocks.generation;
    state.blocks.sync(held, limits.texture_layers);
    if state.blocks.generation != generation {
        state.blocks.rewrite();
    }
    state.blocks.write_directory(&ground);
    state.watered = state.blocks.holding();
}

pub(crate) struct BlocksGpu {
    pub(crate) surface: Texture<WaterGpu>,
    pub(crate) info: Texture<WaterGpu>,
    pub(crate) directory: Buffer<WaterGpu>,
    directory_generation: Option<u64>,
}

impl BlocksGpu {
    pub(crate) fn new(
        surface: Texture<WaterGpu>,
        info: Texture<WaterGpu>,
        directory: Buffer<WaterGpu>,
    ) -> Self {
        Self {
            surface,
            info,
            directory,
            directory_generation: None,
        }
    }

    pub(crate) fn prepare(&mut self, blocks: &Blocks, cx: &mut Prepare<'_, WaterGpu>) {
        let (size, layers) = if blocks.layers == 0 {
            (1, 1)
        } else {
            (LAYER_TEXELS, blocks.layers)
        };
        let extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        };
        cx.allocate(self.surface, extent, 1);
        cx.allocate(self.info, extent, 1);
        let block = wgpu::Extent3d {
            width: BLOCK,
            height: BLOCK,
            depth_or_array_layers: 1,
        };
        for (at, texels) in &blocks.writes {
            let index = at % PER_LAYER;
            let origin = wgpu::Origin3d {
                x: index % SIDE * BLOCK,
                y: index / SIDE * BLOCK,
                z: at / PER_LAYER,
            };
            let rows = |texel: u32| wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(BLOCK * texel),
                rows_per_image: Some(BLOCK),
            };
            cx.write_texture(
                self.surface,
                origin,
                bytemuck::cast_slice(&texels.surfaces),
                rows(SURFACE_BYTES),
                block,
            );
            cx.write_texture(
                self.info,
                origin,
                bytemuck::cast_slice(&texels.info),
                rows(INFO_BYTES),
                block,
            );
        }
        let whole = self.directory_generation != Some(blocks.directory_generation);
        cx.allocate_buffer(self.directory, (blocks.directory.len() * 4) as u64);
        let run = if whole {
            0..blocks.directory.len()
        } else {
            blocks.written.clone()
        };
        self.directory_generation = Some(blocks.directory_generation);
        if !run.is_empty() {
            cx.write(self.directory, (run.start * 4) as u64, &blocks.directory[run]);
        }
    }
}
