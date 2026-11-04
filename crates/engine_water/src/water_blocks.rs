use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::{DeviceLimits, WorkClock};
use engine_heightfield::TileKey;
use engine_heightfield::tile::PATCHES_PER_TILE;
use engine_places::ground::{GroundTiles, GroundWork, HeldTile};
use engine_render::{Buffer, Prepare, Texture, wgpu};

use crate::TileWater;
use crate::drawn::{DrawnWater, INFO_BYTES, PATCH_BYTES, SURFACE_BYTES};
use crate::gpu::{
    NO_WATER_BLOCK, WATER_BLOCK as BLOCK, WATER_BLOCKS_SIDE as SIDE,
    WATER_TILE_WORDS as TILE_WORDS, WATER_WAITING,
};
use crate::state::{HeldWater, WaterState, WaterTerrain};
use crate::water::WaterGpu;
use crate::work::Work;

const PER_LAYER: u32 = SIDE * SIDE;
const LAYER_TEXELS: u32 = BLOCK * SIDE;
pub(crate) const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
pub(crate) const INFO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Uint;

type Slot = (Entity, TileKey);

struct TileBlocks {
    stamp: engine_core::Stamp,
    water: Option<Arc<TileWater>>,
    blocks: Box<[u32]>,
}

fn alike(a: Option<&Arc<TileWater>>, b: Option<&Arc<TileWater>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    }
}

fn knows(terrains: &[WaterTerrain], terrain: Entity) -> bool {
    terrains.iter().any(|held| held.entity == terrain)
}

fn ground_tile(ground: &GroundTiles, slot: Slot) -> Option<&HeldTile> {
    ground
        .held
        .binary_search_by_key(&slot, |tile| (tile.terrain, tile.key))
        .ok()
        .map(|at| &ground.held[at])
}

fn tile_water<'a>(terrains: &'a [WaterTerrain], tile: &HeldTile) -> Option<&'a HeldWater> {
    terrains
        .iter()
        .find(|terrain| terrain.entity == tile.terrain)
        .and_then(|terrain| terrain.held.get(&tile.key))
        .filter(|water| water.resident() && water.face.stamp == tile.stamp)
}

#[derive(Default)]
pub(crate) struct Blocks {
    tiles: BTreeMap<Slot, TileBlocks>,
    free: Vec<u32>,
    next: u32,
    pub(crate) layers: u32,
    pub(crate) writes: Vec<(u32, Arc<DrawnWater>, usize)>,
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    occupants: Vec<Option<Slot>>,
    layer_writes: Vec<(u32, Slot)>,
    whole_directory: bool,
    waiting: BTreeSet<Slot>,
    frames: u64,
    waited: BTreeMap<Slot, u64>,
}

impl Blocks {
    pub(crate) fn resident(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    pub(crate) fn current(
        &self,
        terrain: Entity,
        key: TileKey,
        water: Option<&Arc<TileWater>>,
    ) -> bool {
        self.tiles
            .get(&(terrain, key))
            .is_some_and(|tile| alike(tile.water.as_ref(), water))
    }

    pub(crate) fn caught_up(
        &self,
        terrain: Entity,
        key: TileKey,
        water: Option<&Arc<TileWater>>,
    ) -> bool {
        self.tiles
            .get(&(terrain, key))
            .is_none_or(|tile| alike(tile.water.as_ref(), water))
    }

    fn begin(&mut self) {
        self.frames += 1;
        self.uploads = 0;
        self.dropped = 0;
        self.writes.clear();
        self.layer_writes.clear();
        self.whole_directory = false;
    }

    fn walk(&mut self, ground: &GroundTiles, terrains: &[WaterTerrain]) {
        self.waiting.extend(self.tiles.keys().copied());
        let mut occupants: Vec<Option<Slot>> = vec![None; ground.layers as usize];
        let mut waited = BTreeMap::new();
        for tile in &ground.held {
            let slot = (tile.terrain, tile.key);
            if let Some(occupant) = occupants.get_mut(tile.layer as usize) {
                *occupant = Some(slot);
                if self.occupants.get(tile.layer as usize) != Some(&Some(slot)) {
                    self.layer_writes.push((tile.layer, slot));
                }
            }
            if !knows(terrains, tile.terrain) {
                continue;
            }
            self.waiting.insert(slot);
            if !self.tiles.contains_key(&slot) {
                let since = self.waited.get(&slot).copied().unwrap_or(self.frames);
                waited.insert(slot, since);
            }
        }
        self.occupants = occupants;
        self.waited = waited;
    }

    fn place(
        &mut self,
        ground: &GroundTiles,
        terrains: &[WaterTerrain],
        max_layers: u32,
        work: &Work,
    ) -> bool {
        let mut changed = false;
        let mut replacing = Vec::new();
        let mut arriving = Vec::new();
        for slot in std::mem::take(&mut self.waiting) {
            let tile = ground_tile(ground, slot);
            let Some(water) = tile.and_then(|tile| tile_water(terrains, tile)) else {
                let Some(gone) = self.tiles.remove(&slot) else {
                    continue;
                };
                changed = true;
                self.free.extend(
                    gone.blocks
                        .iter()
                        .copied()
                        .filter(|&block| block != NO_WATER_BLOCK),
                );
                if let Some(tile) = tile {
                    self.layer_writes.push((tile.layer, slot));
                    if knows(terrains, slot.0) {
                        self.waited.insert(slot, self.frames);
                    }
                }
                continue;
            };
            match self.tiles.get(&slot) {
                Some(held)
                    if held.stamp == water.face.stamp
                        && alike(held.water.as_ref(), water.water.as_ref()) => {}
                Some(_) => replacing.push((slot, water)),
                None => arriving.push((slot, water)),
            }
        }
        self.free.sort_unstable_by(|a, b| b.cmp(a));
        let mut uploaded = false;
        for (slot, water) in replacing {
            if uploaded && work.expired() {
                self.waiting.insert(slot);
                continue;
            }
            uploaded = true;
            changed = true;
            let before = self.tiles.remove(&slot);
            let blocks = self.upload(water, before, max_layers);
            self.tiles.insert(slot, blocks);
            if let Some(tile) = ground_tile(ground, slot) {
                self.layer_writes.push((tile.layer, slot));
            }
        }
        for (slot, water) in arriving {
            if uploaded && work.expired() {
                self.waiting.insert(slot);
                continue;
            }
            uploaded = true;
            changed = true;
            let blocks = self.upload(water, None, max_layers);
            self.count_wait(slot, &blocks.blocks);
            self.tiles.insert(slot, blocks);
            if let Some(tile) = ground_tile(ground, slot) {
                self.layer_writes.push((tile.layer, slot));
            }
        }
        changed
    }

    fn count_wait(&mut self, (terrain, key): Slot, blocks: &[u32]) {
        let Some(since) = self.waited.remove(&(terrain, key)) else {
            return;
        };
        if blocks.iter().all(|&block| block == NO_WATER_BLOCK) {
            return;
        }
        engine_core::profile::tally("water.wet_tile_waits", 1);
        let patches = PATCHES_PER_TILE as usize;
        let uncovered = blocks.iter().enumerate().any(|(at, &block)| {
            block != NO_WATER_BLOCK && !self.shown_above(terrain, key, (at % patches, at / patches))
        });
        if uncovered {
            engine_core::profile::tally("water.uncovered_tiles", 1);
            engine_core::profile::tally("water.uncovered_frames", self.frames - since);
        }
    }

    fn shown_above(&self, terrain: Entity, key: TileKey, (px, pz): (usize, usize)) -> bool {
        let half = PATCHES_PER_TILE as usize / 2;
        let (mut level, mut x, mut z, mut px, mut pz) = (key.level, key.x, key.z, px, pz);
        while level < u8::MAX {
            (px, pz) = (
                usize::from(x & 1) * half + px / 2,
                usize::from(z & 1) * half + pz / 2,
            );
            (level, x, z) = (level + 1, x / 2, z / 2);
            if let Some(tile) = self.tiles.get(&(terrain, TileKey::new(level, x, z))) {
                return tile.blocks[pz * 2 * half + px] != NO_WATER_BLOCK;
            }
            if x == 0 && z == 0 {
                break;
            }
        }
        false
    }

    fn upload(
        &mut self,
        tile: &HeldWater,
        before: Option<TileBlocks>,
        max_layers: u32,
    ) -> TileBlocks {
        let patches = PATCHES_PER_TILE as usize;
        let stamp = tile.face.stamp;
        let kept = before.as_ref().is_some_and(|before| before.stamp == stamp);
        let mut blocks = before.map_or_else(
            || vec![NO_WATER_BLOCK; patches * patches].into_boxed_slice(),
            |before| before.blocks,
        );
        let mut offset = 0;
        for pz in 0..patches {
            for px in 0..patches {
                let index = pz * patches + px;
                let drawn = tile
                    .water
                    .as_deref()
                    .is_some_and(|water| water.patch_holds_water(px as u32, pz as u32))
                    .then_some(tile.drawn.as_ref())
                    .flatten();
                let Some(drawn) = drawn else {
                    if blocks[index] != NO_WATER_BLOCK {
                        self.free.push(blocks[index]);
                        blocks[index] = NO_WATER_BLOCK;
                    }
                    continue;
                };
                let at = offset;
                offset += PATCH_BYTES;
                if kept && blocks[index] != NO_WATER_BLOCK && !drawn.changed(index) {
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
                self.writes.push((block, Arc::clone(drawn), at));
                blocks[index] = block;
                self.uploads += 1;
            }
        }
        TileBlocks {
            stamp,
            water: tile.water.clone(),
            blocks,
        }
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
        }
        self.next += 1;
        Some(block)
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

pub(crate) fn place_blocks(
    ground: Res<GroundTiles>,
    (limits, clock, work): (Res<DeviceLimits>, Res<WorkClock>, Res<GroundWork>),
    mut state: ResMut<WaterState>,
) {
    let state = &mut *state;
    let blocks = &mut state.blocks;
    blocks.begin();
    let whole = limits.is_changed()
        || state.dropped
        || state.terrains.iter().any(|terrain| terrain.touched_all);
    if whole || ground.is_changed() {
        blocks.whole_directory = whole;
        blocks.walk(&ground, &state.terrains);
    } else {
        for terrain in &state.terrains {
            let slots = terrain.touched.iter().map(|&key| (terrain.entity, key));
            blocks.waiting.extend(slots);
        }
    }
    if blocks.waiting.is_empty() {
        return;
    }
    let work = Work::until(&clock, &work, state.work_end);
    if blocks.place(&ground, &state.terrains, limits.texture_layers, &work) {
        state.watered = blocks.holding();
    }
}

pub(crate) struct BlocksGpu {
    pub(crate) surface: Texture<WaterGpu>,
    pub(crate) info: Texture<WaterGpu>,
    pub(crate) directory: Buffer<WaterGpu>,
    directory_layers: Option<u32>,
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
            directory_layers: None,
        }
    }

    pub(crate) fn forget(&mut self) {
        self.directory_layers = None;
    }

    pub(crate) fn prepare(
        &mut self,
        blocks: &Blocks,
        (ground, terrains): (&GroundTiles, &[WaterTerrain]),
        cx: &mut Prepare<'_, WaterGpu>,
    ) {
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
        cx.grow(self.surface, extent, 1);
        cx.grow(self.info, extent, 1);
        let block = wgpu::Extent3d {
            width: BLOCK,
            height: BLOCK,
            depth_or_array_layers: 1,
        };
        for (at, drawn, offset) in &blocks.writes {
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
                drawn.surfaces(*offset),
                rows(SURFACE_BYTES),
                block,
            );
            cx.write_texture(
                self.info,
                origin,
                drawn.info(*offset),
                rows(INFO_BYTES),
                block,
            );
        }
        let entry = TILE_WORDS as usize;
        let words = ground.layers as usize * entry;
        cx.allocate_buffer(self.directory, (words * 4) as u64);
        let whole = blocks.whole_directory || self.directory_layers != Some(ground.layers);
        self.directory_layers = Some(ground.layers);
        if whole {
            let mut directory = vec![NO_WATER_BLOCK; words];
            for tile in &ground.held {
                let at = tile.layer as usize * entry;
                let Some(slot) = directory.get_mut(at..at + entry) else {
                    continue;
                };
                match blocks.tiles.get(&(tile.terrain, tile.key)) {
                    Some(held) => slot.copy_from_slice(&held.blocks),
                    None if knows(terrains, tile.terrain) => slot.fill(WATER_WAITING),
                    None => {}
                }
            }
            if !directory.is_empty() {
                cx.write(self.directory, 0, &directory);
            }
            return;
        }
        for &(layer, slot) in &blocks.layer_writes {
            if layer >= ground.layers {
                continue;
            }
            let at = (layer as usize * entry * 4) as u64;
            if let Some(held) = blocks.tiles.get(&slot) {
                cx.write(self.directory, at, &held.blocks);
            } else {
                let fill = if knows(terrains, slot.0) {
                    WATER_WAITING
                } else {
                    NO_WATER_BLOCK
                };
                cx.write(self.directory, at, &vec![fill; entry]);
            }
        }
    }
}
