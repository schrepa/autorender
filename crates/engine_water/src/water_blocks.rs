use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::{max, min};
use engine_core::{Cursor, DeviceLimits, FrameBudget, RecordKey, Since};
use engine_heightfield::TileKey;
use engine_heightfield::tile::{PATCH_QUADS, PATCHES_PER_TILE};
use engine_open_world_kit::ground::{GroundTiles, HeldTile};
use engine_render::{Allocation, Buffer, Prepare, Texture, wgpu};
use glam::{Vec2, Vec3};

use crate::TileWater;
use crate::gpu::{
    NO_WATER_BLOCK, WATER_BLOCKS_SIDE as SIDE, WATER_TILE_WORDS as TILE_WORDS, WATER_WAITING,
    WATER_WET_AREAS as WET_AREAS,
};
use crate::jobs::drawn::{DrawnWater, INFO_BYTES, PATCH_BYTES, SURFACE_BYTES};
use crate::jobs::gpu::{NO_WATER_SURFACE, WATER_BLOCK as BLOCK};
use crate::state::{HeldWater, WaterState, WaterTerrain};
use crate::water::WaterGpu;

const PER_LAYER: u32 = SIDE * SIDE;
const LAYER_TEXELS: u32 = BLOCK * SIDE;
pub(crate) const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
pub(crate) const INFO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

type Slot = (Entity, TileKey);

struct TileBlocks {
    stamp: engine_core::Stamp,
    water: Option<Arc<TileWater>>,
    drawn: Option<Arc<DrawnWater>>,
    blocks: Box<[u32]>,
}

impl TileBlocks {
    fn shown(&self) -> Option<RecordKey> {
        self.drawn.as_ref().map(|drawn| drawn.record)
    }

    fn writes(&self, into: &mut Vec<(u32, Arc<DrawnWater>, usize)>) {
        let (Some(water), Some(drawn)) = (self.water.as_deref(), self.drawn.as_ref()) else {
            return;
        };
        let patches = PATCHES_PER_TILE as usize;
        let mut offset = 0;
        for pz in 0..patches {
            for px in 0..patches {
                if !water.patch_holds_water(px as u32, pz as u32) {
                    continue;
                }
                let at = offset;
                offset += PATCH_BYTES;
                let block = self.blocks[pz * patches + px];
                if block != NO_WATER_BLOCK {
                    into.push((block, Arc::clone(drawn), at));
                }
            }
        }
    }
}

enum TileState {
    Waiting { since: u64 },
    Held(TileBlocks),
    Behind(TileBlocks),
}

impl TileState {
    fn held(&self) -> Option<&TileBlocks> {
        match self {
            Self::Held(tile) | Self::Behind(tile) => Some(tile),
            Self::Waiting { .. } => None,
        }
    }

    fn into_held(self) -> Option<TileBlocks> {
        match self {
            Self::Held(tile) | Self::Behind(tile) => Some(tile),
            Self::Waiting { .. } => None,
        }
    }

    fn since(&self) -> Option<u64> {
        match self {
            Self::Waiting { since } => Some(*since),
            Self::Held(_) | Self::Behind(_) => None,
        }
    }
}

fn knows(terrains: &[WaterTerrain], terrain: Entity) -> bool {
    terrains.iter().any(|held| held.entity == terrain)
}

#[derive(Clone, Copy)]
struct TileArea {
    base: Vec3,
    corner: Vec2,
    size: f32,
}

fn tile_area(terrains: &[WaterTerrain], (terrain, key): Slot) -> Option<TileArea> {
    let held = terrains.iter().find(|held| held.entity == terrain)?;
    Some(TileArea {
        base: held.origin,
        corner: held.pyramid.tile_origin(key),
        size: held.pyramid.tile_size(u32::from(key.level)),
    })
}

#[derive(Clone, Copy)]
struct Placed {
    heights: [f32; 2],
    area: [f32; 4],
    wet: u32,
}

const UNPLACED: Placed = Placed {
    heights: [f32::NEG_INFINITY, f32::INFINITY],
    area: [
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::INFINITY,
    ],
    wet: u32::MAX,
};

fn placed(area: Option<TileArea>, (px, pz): (usize, usize), surfaces: &[u8]) -> Placed {
    let Some(area) = area else {
        return UNPLACED;
    };
    let step = area.size / PATCHES_PER_TILE as f32;
    let low = area.corner + Vec2::new(px as f32, pz as f32) * step;
    let high = low + Vec2::splat(step);
    Placed {
        heights: texel_heights(surfaces, area.base.y),
        area: [
            area.base.x + low.x,
            area.base.z + low.y,
            area.base.x + high.x,
            area.base.z + high.y,
        ],
        wet: wet_areas(surfaces),
    }
}

fn wet_areas(surfaces: &[u8]) -> u32 {
    let side = BLOCK as usize;
    let areas = WET_AREAS as usize;
    let cells = (PATCH_QUADS / WET_AREAS) as usize;
    let reach = (BLOCK - PATCH_QUADS) as usize;
    let mut wet = 0;
    let texels = surfaces.as_chunks::<{ SURFACE_BYTES as usize }>().0;
    for (at, texel) in texels.iter().enumerate() {
        let [water, ground] = bytemuck::cast::<_, [f32; 2]>(*texel);
        if !(water > NO_WATER_SURFACE * 0.5 && water > ground) {
            continue;
        }
        let read = |texel: usize| {
            let first = texel.saturating_sub(reach + cells - 1).div_ceil(cells);
            (first, (texel / cells).min(areas - 1))
        };
        let ((first_x, last_x), (first_z, last_z)) = (read(at % side), read(at / side));
        for z in first_z..=last_z {
            for x in first_x..=last_x {
                wet |= 1 << (z * areas + x);
            }
        }
    }
    wet
}

const NO_HEIGHTS: [f32; 2] = [f32::INFINITY, f32::NEG_INFINITY];

fn widen([low, high]: [f32; 2], [from, to]: [f32; 2]) -> [f32; 2] {
    [min(low, from), max(high, to)]
}

fn texel_heights(surfaces: &[u8], base: f32) -> [f32; 2] {
    surfaces
        .as_chunks::<{ SURFACE_BYTES as usize }>()
        .0
        .iter()
        .map(|texel| {
            let [water, ground] = bytemuck::cast::<_, [f32; 2]>(*texel);
            let height = if water > NO_WATER_SURFACE * 0.5 {
                water
            } else {
                ground
            };
            base + height
        })
        .fold(NO_HEIGHTS, |range, height| widen(range, [height, height]))
}

pub(crate) fn surface_reach(state: &WaterState, ground: &GroundTiles, swell: f32) -> [f32; 4] {
    let Some(heights) = ground_heights(ground, &state.terrains, &state.watered) else {
        return [0.0; 4];
    };
    let [low, high] = widen(heights, state.blocks.heights());
    if !(low <= high && low.is_finite() && high.is_finite() && swell.is_finite()) {
        return [0.0; 4];
    }
    [low - swell, high + swell, 1.0, 0.0]
}

fn ground_heights(
    ground: &GroundTiles,
    terrains: &[WaterTerrain],
    watered: &[Entity],
) -> Option<[f32; 2]> {
    let mut range = NO_HEIGHTS;
    for terrain in terrains
        .iter()
        .filter(|terrain| watered.contains(&terrain.entity))
    {
        let mut faces = terrain.held.iter().peekable();
        for tile in ground
            .held
            .iter()
            .filter(|tile| tile.terrain == terrain.entity)
        {
            while faces.next_if(|(key, _)| **key < tile.key).is_some() {}
            let (_, held) = faces
                .peek()
                .filter(|(key, held)| **key == tile.key && held.face.stamp == tile.stamp)?;
            let (low, high) = held.face.bounds;
            range = widen(range, [terrain.origin.y + low.y, terrain.origin.y + high.y]);
        }
    }
    Some(range)
}

fn ground_tile(ground: &GroundTiles, slot: Slot) -> Option<&HeldTile> {
    ground
        .held
        .binary_search_by_key(&slot, |tile| (tile.terrain, tile.key))
        .ok()
        .map(|at| &ground.held[at])
}

enum Shown<'a> {
    Ready(&'a HeldWater),
    Coming,
    Gone,
}

fn tile_water<'a>(terrains: &'a [WaterTerrain], tile: &HeldTile) -> Shown<'a> {
    let Some(water) = terrains
        .iter()
        .find(|terrain| terrain.entity == tile.terrain)
        .and_then(|terrain| terrain.held.get(&tile.key))
        .filter(|water| water.on_face())
    else {
        return Shown::Gone;
    };
    if water.face.stamp != tile.stamp {
        return Shown::Coming;
    }
    match water.made {
        Some(Ok(())) => Shown::Ready(water),
        None => Shown::Coming,
        Some(Err(_)) => Shown::Gone,
    }
}

fn patch_writes(tile: &HeldWater, before: Option<&TileBlocks>) -> u64 {
    let (Some(water), Some(drawn)) = (tile.water.as_deref(), tile.drawn.as_ref()) else {
        return 0;
    };
    let patches = PATCHES_PER_TILE as usize;
    let kept = before.filter(|before| before.stamp == tile.face.stamp);
    let mut writes = 0;
    for pz in 0..patches {
        for px in 0..patches {
            let index = pz * patches + px;
            if !water.patch_holds_water(px as u32, pz as u32) {
                continue;
            }
            let held = kept.is_some_and(|before| before.blocks[index] != NO_WATER_BLOCK);
            if !held || drawn.changed(index) {
                writes += 1;
            }
        }
    }
    writes
}

#[derive(Default)]
pub(crate) struct Blocks {
    tiles: BTreeMap<Slot, TileState>,
    free: Vec<u32>,
    next: u32,
    pub(crate) layers: u32,
    pub(crate) writes: Vec<(u32, Arc<DrawnWater>, usize)>,
    pub(crate) uploads: u32,
    pub(crate) dropped: u32,
    occupants: Vec<Option<Slot>>,
    layer_writes: Vec<(u32, Slot)>,
    whole_directory: bool,
    dirty: BTreeSet<Slot>,
    frames: u64,
    cursors: BTreeMap<Entity, Cursor>,
    placed: Vec<Option<Placed>>,
}

impl Blocks {
    pub(crate) fn resident(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    pub(crate) fn heights(&self) -> [f32; 2] {
        self.placed
            .iter()
            .flatten()
            .map(|placed| placed.heights)
            .fold(NO_HEIGHTS, widen)
    }

    pub(crate) fn wet_areas(&self) -> impl Iterator<Item = [f32; 4]> + '_ {
        self.placed
            .iter()
            .flatten()
            .filter(|placed| placed.wet != 0)
            .map(|placed| placed.area)
    }

    fn wet_words(&self) -> Vec<u32> {
        let mut words: Vec<u32> = self
            .placed
            .iter()
            .map(|placed| placed.map_or(0, |placed| placed.wet))
            .collect();
        words.resize((self.next as usize).max(words.len()).max(1), 0);
        words
    }

    fn release(&mut self, block: u32) {
        self.free.push(block);
        if let Some(placed) = self.placed.get_mut(block as usize) {
            *placed = None;
        }
    }

    fn held(&self, slot: Slot) -> Option<&TileBlocks> {
        self.tiles.get(&slot).and_then(TileState::held)
    }

    fn fall_behind(&mut self, slot: Slot, stamp: engine_core::Stamp) {
        let behind =
            matches!(self.tiles.get(&slot), Some(TileState::Held(held)) if held.stamp != stamp);
        if behind && let Some(TileState::Held(held)) = self.tiles.remove(&slot) {
            self.tiles.insert(slot, TileState::Behind(held));
        }
    }

    pub(crate) fn caught_up(
        &self,
        terrain: Entity,
        key: TileKey,
        shown: Option<RecordKey>,
    ) -> bool {
        self.held((terrain, key))
            .is_none_or(|tile| tile.shown() == shown)
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
        self.dirty.extend(self.tiles.keys().copied());
        let mut occupants: Vec<Option<Slot>> = vec![None; ground.layers as usize];
        let mut wanted = BTreeSet::new();
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
            self.dirty.insert(slot);
            wanted.insert(slot);
            let since = self.frames;
            self.tiles
                .entry(slot)
                .or_insert(TileState::Waiting { since });
        }
        self.tiles
            .retain(|slot, state| state.held().is_some() || wanted.contains(slot));
        self.occupants = occupants;
    }

    fn place(
        &mut self,
        ground: &GroundTiles,
        terrains: &[WaterTerrain],
        max_layers: u32,
        budget: &mut FrameBudget,
    ) -> bool {
        let mut changed = false;
        let mut replacing = Vec::new();
        let mut arriving = Vec::new();
        let mut caught = Vec::new();
        for slot in std::mem::take(&mut self.dirty) {
            let tile = ground_tile(ground, slot);
            let water = match tile.map_or(Shown::Gone, |tile| tile_water(terrains, tile)) {
                Shown::Ready(water) => water,
                Shown::Coming => {
                    if let Some(tile) = tile {
                        self.fall_behind(slot, tile.stamp);
                    }
                    continue;
                }
                Shown::Gone => {
                    changed |= self.drop_tile(slot, tile, terrains);
                    continue;
                }
            };
            let same =
                |held: &TileBlocks| held.stamp == water.face.stamp && held.shown() == water.shown();
            match self.tiles.get(&slot) {
                Some(TileState::Held(held)) if same(held) => {}
                Some(TileState::Behind(held)) if same(held) => caught.push(slot),
                Some(TileState::Held(_) | TileState::Behind(_)) => replacing.push((slot, water)),
                Some(TileState::Waiting { .. }) | None => arriving.push((slot, water)),
            }
        }
        for slot in caught {
            if let Some(TileState::Behind(held)) = self.tiles.remove(&slot) {
                self.tiles.insert(slot, TileState::Held(held));
            }
        }
        self.free.sort_unstable_by(|a, b| b.cmp(a));
        let mut uploaded = false;
        for (slot, water) in replacing {
            let bytes = patch_writes(water, self.held(slot)) * PATCH_BYTES as u64;
            if (uploaded && budget.expired()) || !budget.spend("water", bytes) {
                self.dirty.insert(slot);
                continue;
            }
            uploaded = true;
            changed = true;
            let before = self.tiles.remove(&slot).and_then(TileState::into_held);
            let area = tile_area(terrains, slot);
            let blocks = self.upload(water, before, (max_layers, area));
            self.tiles.insert(slot, TileState::Held(blocks));
            if let Some(tile) = ground_tile(ground, slot) {
                self.layer_writes.push((tile.layer, slot));
            }
        }
        for (slot, water) in arriving {
            let bytes = patch_writes(water, None) * PATCH_BYTES as u64;
            if (uploaded && budget.expired()) || !budget.spend("water", bytes) {
                self.dirty.insert(slot);
                continue;
            }
            uploaded = true;
            changed = true;
            let since = self.tiles.remove(&slot).and_then(|state| state.since());
            let blocks = self.upload(water, None, (max_layers, tile_area(terrains, slot)));
            self.count_wait(slot, since, &blocks.blocks);
            self.tiles.insert(slot, TileState::Held(blocks));
            if let Some(tile) = ground_tile(ground, slot) {
                self.layer_writes.push((tile.layer, slot));
            }
        }
        changed
    }

    fn drop_tile(
        &mut self,
        slot: Slot,
        tile: Option<&HeldTile>,
        terrains: &[WaterTerrain],
    ) -> bool {
        if self.held(slot).is_none() {
            return false;
        }
        let Some(gone) = self.tiles.remove(&slot).and_then(TileState::into_held) else {
            return false;
        };
        for &block in gone.blocks.iter().filter(|&&block| block != NO_WATER_BLOCK) {
            self.release(block);
        }
        if let Some(tile) = tile {
            self.layer_writes.push((tile.layer, slot));
            if knows(terrains, slot.0) {
                let since = self.frames;
                self.tiles.insert(slot, TileState::Waiting { since });
            }
        }
        true
    }

    fn count_wait(&mut self, (terrain, key): Slot, since: Option<u64>, blocks: &[u32]) {
        let Some(since) = since else {
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
            if let Some(tile) = self.held((terrain, TileKey::new(level, x, z))) {
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
        (max_layers, area): (u32, Option<TileArea>),
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
                        self.release(blocks[index]);
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
                let slot = block as usize;
                if self.placed.len() <= slot {
                    self.placed.resize(slot + 1, None);
                }
                self.placed[slot] = Some(placed(area, (px, pz), drawn.surfaces(at)));
                blocks[index] = block;
                self.uploads += 1;
            }
        }
        TileBlocks {
            stamp,
            water: tile.water.clone(),
            drawn: tile.drawn.clone(),
            blocks,
        }
    }

    fn all_writes(&self) -> Vec<(u32, Arc<DrawnWater>, usize)> {
        let mut writes = Vec::new();
        for tile in self.tiles.values().filter_map(TileState::held) {
            tile.writes(&mut writes);
        }
        writes
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
            .filter(|(_, state)| {
                state
                    .held()
                    .is_some_and(|tile| tile.blocks.iter().any(|&block| block != NO_WATER_BLOCK))
            })
            .map(|(&(terrain, _), _)| terrain)
            .collect();
        holding.dedup();
        holding
    }
}

pub(crate) fn place_blocks(
    ground: Res<GroundTiles>,
    limits: Res<DeviceLimits>,
    mut budget: ResMut<FrameBudget>,
    mut state: ResMut<WaterState>,
) {
    let state = &mut *state;
    let blocks = &mut state.blocks;
    blocks.begin();
    let mut whole = limits.is_changed() || state.dropped;
    let mut touched = Vec::new();
    let cursors = std::mem::take(&mut blocks.cursors);
    for terrain in &state.terrains {
        let mut cursor = cursors.get(&terrain.entity).copied().unwrap_or_default();
        match terrain.touched.since(&mut cursor) {
            Since::Same => {}
            Since::Listed(keys) => touched.extend(keys.iter().map(|&key| (terrain.entity, key))),
            Since::All => whole = true,
        }
        blocks.cursors.insert(terrain.entity, cursor);
    }
    if whole || ground.is_changed() {
        blocks.whole_directory = whole;
        blocks.walk(&ground, &state.terrains);
    } else {
        blocks.dirty.extend(touched);
    }
    if blocks.dirty.is_empty() {
        return;
    }
    if blocks.place(&ground, &state.terrains, limits.texture_layers, &mut budget) {
        state.watered = blocks.holding();
    }
}

pub(crate) struct BlocksGpu {
    pub(crate) surface: Texture<WaterGpu>,
    pub(crate) info: Texture<WaterGpu>,
    pub(crate) directory: Buffer<WaterGpu>,
    pub(crate) wet: Buffer<WaterGpu>,
}

impl BlocksGpu {
    pub(crate) fn new(
        (surface, info): (Texture<WaterGpu>, Texture<WaterGpu>),
        directory: Buffer<WaterGpu>,
        wet: Buffer<WaterGpu>,
    ) -> Self {
        Self {
            surface,
            info,
            directory,
            wet,
        }
    }

    fn prepare_wet(&self, blocks: &Blocks, cx: &mut Prepare<'_, WaterGpu>) {
        let words = blocks.wet_words();
        let made = cx.allocate_buffer(self.wet, (words.len() * 4) as u64);
        if made == Allocation::Fresh || !blocks.writes.is_empty() {
            cx.write(self.wet, 0, &words);
        }
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
        let surface = cx.grow(self.surface, extent, 1);
        let info = cx.grow(self.info, extent, 1);
        let refilled = (surface == Allocation::Fresh || info == Allocation::Fresh)
            .then(|| blocks.all_writes());
        self.prepare_wet(blocks, cx);
        let block = wgpu::Extent3d {
            width: BLOCK,
            height: BLOCK,
            depth_or_array_layers: 1,
        };
        for (at, drawn, offset) in refilled.as_ref().unwrap_or(&blocks.writes) {
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
        let made = cx.allocate_buffer(self.directory, (words * 4) as u64);
        if blocks.whole_directory || made == Allocation::Fresh {
            let mut directory = vec![NO_WATER_BLOCK; words];
            for tile in &ground.held {
                let at = tile.layer as usize * entry;
                let Some(slot) = directory.get_mut(at..at + entry) else {
                    continue;
                };
                match blocks.held((tile.terrain, tile.key)) {
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
            if let Some(held) = blocks.held(slot) {
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
