use std::collections::BTreeMap;
use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_places::ground::{
    Cover, GroundTiles, PALETTE_ENTRIES, PlacedTiles, PooledTile, TerrainEntry, TerrainTiles,
    TileKey,
};
use engine_render::{Buffer, Prepare};

use crate::gpu::{NO_PLACED, TraceTerrain};
use crate::part::TraceGpu;

const CANOPY_WORDS: u32 = 4 * PALETTE_ENTRIES;

type TileId = (Entity, TileKey);

pub(crate) struct TraceFrame<'a> {
    pub tiles: &'a TerrainTiles,
    pub ground: &'a GroundTiles,
    pub placed: &'a PlacedTiles,
    pub cover: &'a Cover,
    pub room: u64,
    pub ground_changed: bool,
    pub placed_changed: bool,
}

struct Grid {
    words: Range<u32>,
    slots: Range<u32>,
    identity: u64,
    level: u8,
    apron: f32,
    top: f32,
}

struct Reach {
    slot: u32,
    middle: [f32; 2],
    radius: f32,
    top: f32,
}

pub(crate) struct WorldTrace {
    terrains: u64,
    written: Vec<TraceTerrain>,
    placed: u64,
    words: Vec<u32>,
    dirty: Vec<Range<usize>>,
    grids: BTreeMap<TileId, Grid>,
    free: Vec<Range<u32>>,
    layers: u32,
    boxes: Vec<([f32; 4], [f32; 3])>,
    held: BTreeMap<TileId, u32>,
    tops: BTreeMap<Entity, f32>,
    known: bool,
    pooled: bool,
    tabled: bool,
}

impl Default for WorldTrace {
    fn default() -> Self {
        Self {
            terrains: size_of::<TraceTerrain>() as u64,
            written: Vec::new(),
            placed: 16,
            words: Vec::new(),
            dirty: Vec::new(),
            grids: BTreeMap::new(),
            free: Vec::new(),
            layers: 0,
            boxes: Vec::new(),
            held: BTreeMap::new(),
            tops: BTreeMap::new(),
            known: false,
            pooled: false,
            tabled: false,
        }
    }
}

pub(crate) fn traced<'a>(
    tiles: &'a TerrainTiles,
    ground: &'a GroundTiles,
) -> impl Iterator<Item = (&'a TerrainEntry, u32)> + 'a {
    tiles.terrains.iter().filter_map(|terrain| {
        let (_, table) = ground
            .tables
            .iter()
            .find(|(entity, _)| *entity == terrain.terrain)?;
        Some((terrain, *table))
    })
}

impl WorldTrace {
    pub(crate) fn forget(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TraceGpu>,
        (terrains, placed): (Buffer<TraceGpu>, Buffer<TraceGpu>),
        frame: &TraceFrame<'_>,
    ) {
        if !self.known || frame.ground_changed {
            self.hold(frame.ground);
            self.known = true;
        }
        let groups = &frame.placed.groups;
        let rebuilt = *groups != self.boxes;
        if !self.pooled || frame.ground.layers != self.layers || rebuilt {
            self.lay_out(frame.ground.layers, rebuilt);
            if rebuilt {
                self.boxes.clone_from(groups);
            }
        }
        if !self.pooled || frame.placed_changed {
            self.sync_grids(frame);
        }
        if !self.tabled || frame.ground_changed {
            self.write_layer_words();
        }
        self.pooled = true;
        self.tabled = true;
        let canopies = traced(frame.tiles, frame.ground).count() as u32 * CANOPY_WORDS;
        self.upload(cx, placed, (frame.room, canopies));
        self.write_terrains(cx, terrains, frame);
    }

    pub(crate) fn terrains(&self) -> u32 {
        self.written.len() as u32
    }

    fn hold(&mut self, ground: &GroundTiles) {
        self.held.clear();
        self.tops.clear();
        for tile in &ground.held {
            self.held.insert((tile.terrain, tile.key), tile.layer);
            self.tops
                .entry(tile.terrain)
                .and_modify(|top| *top = engine_core::math::max(*top, tile.top))
                .or_insert(tile.top);
        }
    }

    fn lay_out(&mut self, layers: u32, rebuilt: bool) {
        let mut words = vec![NO_PLACED; layers as usize];
        if rebuilt {
            self.grids.clear();
        }
        for grid in self.grids.values_mut() {
            let moved = words.len() as u32;
            let held = grid.words.start as usize..grid.words.end as usize;
            words.extend_from_slice(&self.words[held]);
            grid.words = moved..words.len() as u32;
        }
        self.words = words;
        self.free.clear();
        self.layers = layers;
        self.dirty = std::iter::once(0..self.words.len()).collect();
        self.tabled = false;
        if rebuilt {
            self.pooled = false;
        }
    }

    fn sync_grids(&mut self, frame: &TraceFrame<'_>) {
        let current: BTreeMap<TileId, &PooledTile> = frame
            .placed
            .tiles
            .iter()
            .map(|tile| ((tile.terrain, tile.key), tile))
            .collect();
        let gone: Vec<TileId> = self
            .grids
            .iter()
            .filter(|&(key, grid)| {
                current.get(key).is_none_or(|tile| {
                    tile.slots != grid.slots || tile.identity != grid.identity
                })
            })
            .map(|(key, _)| *key)
            .collect();
        for key in gone {
            if let Some(grid) = self.grids.remove(&key) {
                self.free_run(grid.words);
            }
        }
        for (key, tile) in current {
            if self.grids.contains_key(&key) {
                continue;
            }
            let terrains = &frame.tiles.terrains;
            let Some(terrain) = terrains.iter().find(|terrain| terrain.terrain == key.0) else {
                continue;
            };
            let reaches = reaches(terrain, key.1, tile);
            let Some((built, apron)) = grid(&reaches, terrain, key.1) else {
                continue;
            };
            let top = reaches.iter().fold(-f32::MAX, |top, reach| {
                engine_core::math::max(top, reach.top)
            });
            let words = self.place(&built);
            self.grids.insert(
                key,
                Grid {
                    words,
                    slots: tile.slots.clone(),
                    identity: tile.identity,
                    level: key.1.level,
                    apron,
                    top,
                },
            );
        }
        let freed: u32 = self.free.iter().map(|run| run.end - run.start).sum();
        let used: u32 = self
            .grids
            .values()
            .map(|grid| grid.words.end - grid.words.start)
            .sum();
        if freed > used {
            self.lay_out(self.layers, false);
        }
        self.tabled = false;
    }

    fn place(&mut self, built: &[u32]) -> Range<u32> {
        let count = built.len() as u32;
        let start = match self
            .free
            .iter()
            .position(|run| run.end - run.start >= count)
        {
            Some(index) => {
                let start = self.free[index].start;
                self.free[index].start += count;
                if self.free[index].is_empty() {
                    self.free.remove(index);
                }
                start
            }
            None => {
                let start = self.words.len() as u32;
                self.words.resize(self.words.len() + built.len(), 0);
                start
            }
        };
        let range = start as usize..(start + count) as usize;
        self.words[range.clone()].copy_from_slice(built);
        self.mark(range);
        start..start + count
    }

    fn free_run(&mut self, words: Range<u32>) {
        let at = self.free.partition_point(|run| run.start < words.start);
        self.free.insert(at, words);
        if at + 1 < self.free.len() && self.free[at].end == self.free[at + 1].start {
            self.free[at].end = self.free[at + 1].end;
            self.free.remove(at + 1);
        }
        if at > 0 && self.free[at - 1].end == self.free[at].start {
            self.free[at - 1].end = self.free[at].end;
            self.free.remove(at);
        }
    }

    fn write_layer_words(&mut self) {
        let count = self.layers;
        let layers = 0..count as usize;
        self.words[layers.clone()].fill(NO_PLACED);
        for (key, grid) in &self.grids {
            if let Some(layer) = self.held.get(key).copied().filter(|&layer| layer < count) {
                self.words[layer as usize] = grid.words.start;
            }
        }
        self.mark(layers);
    }

    fn mark(&mut self, words: Range<usize>) {
        self.dirty.push(words);
    }

    fn upload(
        &mut self,
        cx: &mut Prepare<'_, TraceGpu>,
        buffer: Buffer<TraceGpu>,
        (room, canopies): (u64, u32),
    ) {
        let bytes = (self.words.len() as u64 + u64::from(canopies)) * 4;
        let mut dirty = std::mem::take(&mut self.dirty);
        if let Some(size) = grown(self.placed, bytes, room) {
            self.placed = size;
            dirty = std::iter::once(0..self.words.len()).collect();
        }
        cx.allocate_buffer(buffer, self.placed);
        let held = ((self.placed / 4) as usize).min(self.words.len());
        for run in dirty {
            let run = run.start.min(held)..run.end.min(held);
            if !run.is_empty() {
                cx.write(buffer, (run.start * 4) as u64, &self.words[run]);
            }
        }
    }

    fn write_terrains(
        &mut self,
        cx: &mut Prepare<'_, TraceGpu>,
        buffer: Buffer<TraceGpu>,
        frame: &TraceFrame<'_>,
    ) {
        let mut records = Vec::with_capacity(self.written.len());
        let (first, last) = (
            TileKey::new(0, 0, 0),
            TileKey::new(u8::MAX, u16::MAX, u16::MAX),
        );
        let cover = if frame.cover.enabled {
            frame.cover.density
        } else {
            0.0
        };
        let words = self.words.len() as u32;
        for (terrain, table) in traced(frame.tiles, frame.ground) {
            let ground = self
                .tops
                .get(&terrain.terrain)
                .map_or(-f32::MAX, |top| terrain.origin.y + top);
            let (levels, apron, highest) = self
                .grids
                .range((terrain.terrain, first)..=(terrain.terrain, last))
                .fold(
                    (0_u32, 0.0_f32, ground),
                    |(levels, apron, highest), (_, grid)| {
                        (
                            levels | 1 << grid.level,
                            engine_core::math::max(apron, grid.apron),
                            engine_core::math::max(highest, grid.top),
                        )
                    },
                );
            let canopy = words + records.len() as u32 * CANOPY_WORDS;
            records.push(TraceTerrain {
                cutout: terrain
                    .cutout
                    .map_or([0.0; 4], |(low, high)| [low.x, low.y, high.x, high.y]),
                maps: [table, levels, canopy, 0],
                reach: [apron, cover, highest, 0.0],
            });
        }
        let bytes = (records.len() * size_of::<TraceTerrain>()) as u64;
        if let Some(size) = grown(self.terrains, bytes, frame.room) {
            self.terrains = size;
        }
        cx.allocate_buffer(buffer, self.terrains);
        if records == self.written {
            return;
        }
        if !records.is_empty() {
            cx.write(buffer, 0, &records);
        }
        self.written = records;
    }
}

fn grown(held: u64, bytes: u64, room: u64) -> Option<u64> {
    let size = bytes.next_power_of_two().min(room);
    (bytes > held && size > held).then_some(size)
}

fn reaches(terrain: &TerrainEntry, key: TileKey, tile: &PooledTile) -> Vec<Reach> {
    let raised = terrain
        .level_errors
        .iter()
        .take(usize::from(key.level))
        .fold(0.0, |sum, error| sum + error);
    let mut reaches = Vec::with_capacity((tile.slots.end - tile.slots.start) as usize);
    for (slot, &(middle, half)) in tile.slots.clone().zip(&tile.impostors) {
        let radius = (half.x * half.x + half.z * half.z).sqrt();
        if radius <= 0.0 || radius.is_nan() {
            continue;
        }
        reaches.push(Reach {
            slot,
            middle: [middle.x - terrain.origin.x, middle.z - terrain.origin.z],
            radius,
            top: middle.y + half.y + raised,
        });
    }
    reaches
}

fn grid(reaches: &[Reach], terrain: &TerrainEntry, key: TileKey) -> Option<(Vec<u32>, f32)> {
    if reaches.is_empty() {
        return None;
    }
    let size = terrain.pyramid.tile_size(u32::from(key.level));
    let apron = reaches.iter().fold(0.0, |widest, reach| {
        engine_core::math::max(widest, reach.radius)
    });
    let wide = size + 2.0 * apron;
    let spacing = wide / (reaches.len() as f32).sqrt();
    let narrowest = engine_core::math::max(2.0 * apron, spacing);
    let across = engine_core::math::max((wide / narrowest).floor(), 1.0) as u32;
    let cells = 1_u32 << across.ilog2();
    let width = wide / cells as f32;
    let corner = [
        f32::from(key.x) * size - apron,
        f32::from(key.z) * size - apron,
    ];
    let last = cells - 1;
    let cells_of = |reach: &Reach| {
        let cell = |value: f32, from: f32| {
            engine_core::math::clamp(((value - from) / width).floor(), 0.0, last as f32) as u32
        };
        [
            cell(reach.middle[0] - reach.radius, corner[0]),
            cell(reach.middle[0] + reach.radius, corner[0]),
            cell(reach.middle[1] - reach.radius, corner[1]),
            cell(reach.middle[1] + reach.radius, corner[1]),
        ]
    };
    let count = (cells * cells) as usize;
    let mut counts = vec![0_u32; count];
    for reach in reaches {
        let [x0, x1, z0, z1] = cells_of(reach);
        for z in z0..=z1 {
            for x in x0..=x1 {
                counts[(z * cells + x) as usize] += 1;
            }
        }
    }
    let mut ends = Vec::with_capacity(count);
    let mut total = 0_u32;
    for &listed in &counts {
        total += listed;
        ends.push(total);
    }
    let mut cursors: Vec<u32> = ends
        .iter()
        .zip(&counts)
        .map(|(end, listed)| end - listed)
        .collect();
    let mut entries = vec![0_u32; total as usize];
    let mut tops = vec![-f32::MAX; count];
    for reach in reaches {
        let [x0, x1, z0, z1] = cells_of(reach);
        for z in z0..=z1 {
            for x in x0..=x1 {
                let at = (z * cells + x) as usize;
                entries[cursors[at] as usize] = reach.slot;
                cursors[at] += 1;
                tops[at] = engine_core::math::max(tops[at], reach.top);
            }
        }
    }
    let pyramid = top_pyramid(tops, cells);
    let head = 4 + pyramid.len() + ends.len();
    let mut words = Vec::with_capacity(head + entries.len());
    words.extend([cells, apron.to_bits(), head as u32, 0]);
    words.extend(pyramid.iter().map(|top| top.to_bits()));
    words.extend(ends);
    words.extend(entries);
    Some((words, apron))
}

fn top_pyramid(tops: Vec<f32>, cells: u32) -> Vec<f32> {
    let mut side = cells as usize;
    let mut pyramid = tops.clone();
    let mut level = tops;
    while side > 1 {
        let half = side / 2;
        let top = |x: usize, z: usize| level[z * side + x];
        let coarser: Vec<f32> = (0..half * half)
            .map(|at| {
                let (x, z) = (2 * (at % half), 2 * (at / half));
                engine_core::math::max(
                    engine_core::math::max(top(x, z), top(x + 1, z)),
                    engine_core::math::max(top(x, z + 1), top(x + 1, z + 1)),
                )
            })
            .collect();
        pyramid.extend_from_slice(&coarser);
        level = coarser;
        side = half;
    }
    pyramid
}
