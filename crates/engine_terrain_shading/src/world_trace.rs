use std::collections::BTreeMap;
use std::ops::Range;

use bevy_ecs::prelude::*;
use engine_core::Cut;
use engine_heightfield::{TILE_QUADS, TILE_STRIDE};
use engine_open_world_kit::ground::{
    GroundTiles, PlacedTiles, PooledTile, TerrainEntry, TerrainTiles, TileFace,
};
use engine_render::{Allocation, Buffer, Prepare};
use engine_tiles::TileKey;

use crate::gpu::{NO_PLACED, PEAKS_SIDE, TRACE_LEVEL_WORDS, TraceTerrain};
use crate::state::lighting_level;
use crate::trace::TraceGpu;

type TileId = (Entity, TileKey);

pub(crate) struct TraceFrame<'a> {
    pub tiles: &'a TerrainTiles,
    pub ground: &'a GroundTiles,
    pub placed: &'a PlacedTiles,
    pub room: u64,
    pub placed_changed: bool,
}

struct Grid {
    words: Range<u32>,
    slots: Range<u32>,
    identity: u64,
    level: u8,
    apron: f32,
    stands: Stands,
}

struct Reach {
    slot: u32,
    middle: [f32; 2],
    radius: f32,
    stands: Stands,
}

#[derive(Clone, Copy)]
struct Stands {
    top: f32,
    widest: f32,
    tallest: f32,
    spread: f32,
}

impl Stands {
    const NONE: Self = Self {
        top: -f32::MAX,
        widest: 0.0,
        tallest: -f32::MAX,
        spread: 0.0,
    };

    fn with(self, other: Self) -> Self {
        Self {
            top: engine_core::math::max(self.top, other.top),
            widest: engine_core::math::max(self.widest, other.widest),
            tallest: engine_core::math::max(self.tallest, other.tallest),
            spread: engine_core::math::max(self.spread, other.spread),
        }
    }
}

#[derive(Clone, Copy)]
struct LevelReach {
    low: [f32; 2],
    high: [f32; 2],
    stands: Stands,
}

impl LevelReach {
    const NONE: Self = Self {
        low: [f32::MAX; 2],
        high: [-f32::MAX; 2],
        stands: Stands::NONE,
    };

    fn words(self, error: f32) -> [u32; TRACE_LEVEL_WORDS as usize] {
        [
            self.low[0],
            self.low[1],
            self.high[0],
            self.high[1],
            self.stands.top,
            self.stands.widest,
            self.stands.tallest + error,
            self.stands.spread,
        ]
        .map(f32::to_bits)
    }
}

pub(crate) struct WorldTrace {
    terrains: u64,
    listed: u32,
    placed: u64,
    words: Vec<u32>,
    dirty: Vec<Range<usize>>,
    grids: BTreeMap<TileId, Grid>,
    free: Vec<Range<u32>>,
    layers: u32,
    boxes: Vec<([f32; 4], [f32; 3])>,
    pooled: bool,
    table: Range<u32>,
    levels: u32,
    turns: u32,
}

impl Default for WorldTrace {
    fn default() -> Self {
        Self {
            terrains: size_of::<TraceTerrain>() as u64,
            listed: 0,
            placed: 16,
            words: Vec::new(),
            dirty: Vec::new(),
            grids: BTreeMap::new(),
            free: Vec::new(),
            layers: 0,
            boxes: Vec::new(),
            pooled: false,
            table: 0..0,
            levels: 0,
            turns: 0,
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
    pub(crate) fn prepare(
        &mut self,
        cx: &mut Prepare<'_, TraceGpu>,
        (terrains, placed): (Buffer<TraceGpu>, Buffer<TraceGpu>),
        frame: &TraceFrame<'_>,
    ) -> Cut {
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
        self.pooled = true;
        self.write_layer_words(frame.ground);
        self.write_levels(frame);
        let pool = self.upload(cx, placed, frame.room);
        self.write_terrains(cx, terrains, frame);
        pool
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
        self.table = 0..0;
        self.layers = layers;
        self.dirty = std::iter::once(0..self.words.len()).collect();
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
                current
                    .get(key)
                    .is_none_or(|tile| tile.slots != grid.slots || tile.identity != grid.identity)
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
            let stands = reaches
                .iter()
                .fold(Stands::NONE, |stands, reach| stands.with(reach.stands));
            let words = self.place(&built);
            self.grids.insert(
                key,
                Grid {
                    words,
                    slots: tile.slots.clone(),
                    identity: tile.identity,
                    level: key.1.level,
                    apron,
                    stands,
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

    fn write_layer_words(&mut self, ground: &GroundTiles) {
        let count = self.layers;
        let layers = 0..count as usize;
        self.words[layers.clone()].fill(NO_PLACED);
        for tile in &ground.held {
            if tile.layer < count
                && let Some(grid) = self.grids.get(&(tile.terrain, tile.key))
            {
                self.words[tile.layer as usize] = grid.words.start;
            }
        }
        self.mark(layers);
    }

    fn write_levels(&mut self, frame: &TraceFrame<'_>) {
        let (first, last) = (
            TileKey::new(0, 0, 0),
            TileKey::new(u8::MAX, u16::MAX, u16::MAX),
        );
        let levels = self
            .grids
            .values()
            .map(|grid| usize::from(grid.level) + 1)
            .max()
            .unwrap_or(0);
        let mut built = Vec::new();
        let mut turns = 0;
        for (terrain, _) in traced(frame.tiles, frame.ground) {
            let mut rows = vec![LevelReach::NONE; levels];
            for ((_, key), grid) in self
                .grids
                .range((terrain.terrain, first)..=(terrain.terrain, last))
            {
                let size = terrain.pyramid.tile_size(u32::from(key.level));
                let corner = [f32::from(key.x) * size, f32::from(key.z) * size];
                let row = &mut rows[usize::from(key.level)];
                for axis in 0..2 {
                    row.low[axis] = engine_core::math::min(row.low[axis], corner[axis] - grid.apron);
                    row.high[axis] =
                        engine_core::math::max(row.high[axis], corner[axis] + size + grid.apron);
                }
                row.stands = row.stands.with(grid.stands);
            }
            let baked = lighting_level(&terrain.pyramid, terrain.pinned_levels) as usize;
            for (level, row) in rows.iter().enumerate() {
                if row.low[0] <= row.high[0] {
                    let across = [row.high[0] - row.low[0], row.high[1] - row.low[1]];
                    let far = (across[0] * across[0] + across[1] * across[1]).sqrt();
                    let tiles = (far / terrain.pyramid.tile_size(level as u32)).ceil() as u32 + 1;
                    turns = turns.max(3 * tiles);
                }
                built.extend(row.words(errors(terrain, level.min(baked)..level.max(baked))));
            }
        }
        self.levels = levels as u32;
        self.turns = turns;
        let held = self.table.start as usize..self.table.end as usize;
        if held.len() != built.len() {
            if !held.is_empty() {
                self.free_run(self.table.clone());
            }
            self.table = if built.is_empty() {
                0..0
            } else {
                self.place(&built)
            };
        } else if self.words[held.clone()] != built[..] {
            self.words[held.clone()].copy_from_slice(&built);
            self.mark(held);
        }
    }

    pub(crate) fn turns(&self) -> [u32; 2] {
        [self.turns, 2 * PEAKS_SIDE]
    }

    fn mark(&mut self, words: Range<usize>) {
        self.dirty.push(words);
    }

    fn upload(
        &mut self,
        cx: &mut Prepare<'_, TraceGpu>,
        buffer: Buffer<TraceGpu>,
        room: u64,
    ) -> Cut {
        let bytes = self.words.len() as u64 * 4;
        let mut dirty = std::mem::take(&mut self.dirty);
        if let Some(size) = grown(self.placed, bytes, room) {
            self.placed = size;
        }
        if cx.allocate_buffer(buffer, self.placed) == Allocation::Fresh {
            dirty = std::iter::once(0..self.words.len()).collect();
        }
        let held = ((self.placed / 4) as usize).min(self.words.len());
        for run in dirty {
            let run = run.start.min(held)..run.end.min(held);
            if !run.is_empty() {
                cx.write(buffer, (run.start * 4) as u64, &self.words[run]);
            }
        }
        Cut::of(self.words.len() as u32, (self.placed / 4) as u32)
    }

    fn write_terrains(
        &mut self,
        cx: &mut Prepare<'_, TraceGpu>,
        buffer: Buffer<TraceGpu>,
        frame: &TraceFrame<'_>,
    ) {
        let mut records = Vec::with_capacity(self.listed as usize);
        let (first, last) = (
            TileKey::new(0, 0, 0),
            TileKey::new(u8::MAX, u16::MAX, u16::MAX),
        );
        let rows = self.levels * TRACE_LEVEL_WORDS;
        for (index, (terrain, table)) in traced(frame.tiles, frame.ground).enumerate() {
            let ground = frame
                .ground
                .held
                .iter()
                .filter(|tile| tile.terrain == terrain.terrain)
                .fold(-f32::MAX, |top, tile| {
                    engine_core::math::max(top, terrain.origin.y + tile.top)
                });
            let (levels, apron, highest) = self
                .grids
                .range((terrain.terrain, first)..=(terrain.terrain, last))
                .fold(
                    (0_u32, 0.0_f32, ground),
                    |(levels, apron, highest), (_, grid)| {
                        (
                            levels | 1 << grid.level,
                            engine_core::math::max(apron, grid.apron),
                            engine_core::math::max(highest, grid.stands.top),
                        )
                    },
                );
            records.push(TraceTerrain {
                cutout: terrain
                    .cutout
                    .map_or([0.0; 4], |(low, high)| [low.x, low.y, high.x, high.y]),
                maps: [
                    table,
                    levels,
                    self.table.start + index as u32 * rows,
                    self.levels,
                ],
                reach: [
                    apron,
                    errors(
                        terrain,
                        0..lighting_level(&terrain.pyramid, terrain.pinned_levels) as usize,
                    ),
                    highest,
                    frame
                        .tiles
                        .terrains
                        .iter()
                        .position(|held| held.terrain == terrain.terrain)
                        .map_or(-1.0, |at| at as f32),
                ],
            });
        }
        let bytes = (records.len() * size_of::<TraceTerrain>()) as u64;
        if let Some(size) = grown(self.terrains, bytes, frame.room) {
            self.terrains = size;
        }
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate_buffer(buffer, self.terrains);
        if !records.is_empty() {
            cx.write(buffer, 0, &records);
        }
        self.listed = records.len() as u32;
    }
}

fn grown(held: u64, bytes: u64, room: u64) -> Option<u64> {
    let size = bytes.next_power_of_two().min(room);
    (bytes > held && size > held).then_some(size)
}

fn errors(terrain: &TerrainEntry, levels: Range<usize>) -> f32 {
    terrain
        .level_errors
        .iter()
        .take(levels.end)
        .skip(levels.start)
        .sum()
}

fn ground_under(face: &TileFace, local: [f32; 2]) -> (f32, f32) {
    let corner = [face.bounds.0.x, face.bounds.0.z];
    let at = [0, 1].map(|axis| (local[axis] - corner[axis]) / face.spacing);
    let held = at.map(|at| engine_core::math::clamp(at, 0.0, TILE_QUADS as f32));
    let moved = [0, 1].map(|axis| (held[axis] - at[axis]) * face.spacing);
    let first =
        held.map(|held| engine_core::math::min(held.floor(), (TILE_QUADS - 1) as f32) as usize);
    let stride = TILE_STRIDE as usize;
    let mut ground = f32::MAX;
    for z in first[1]..=first[1] + 1 {
        for x in first[0]..=first[0] + 1 {
            ground = engine_core::math::min(ground, face.heights[(z + 1) * stride + x + 1]);
        }
    }
    (ground, (moved[0] * moved[0] + moved[1] * moved[1]).sqrt())
}

fn reaches(terrain: &TerrainEntry, key: TileKey, tile: &PooledTile) -> Vec<Reach> {
    let face = terrain.tiles.iter().find(|face| face.key == key);
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
        let local = [middle.x - terrain.origin.x, middle.z - terrain.origin.z];
        let top = middle.y + half.y + raised;
        let (tallest, moved) = face.map_or((f32::MAX, 0.0), |face| {
            let (ground, moved) = ground_under(face, local);
            (top - terrain.origin.y - ground, moved)
        });
        reaches.push(Reach {
            slot,
            middle: local,
            radius,
            stands: Stands {
                top,
                widest: 2.0 * engine_core::math::max(radius, half.y),
                tallest,
                spread: radius + moved,
            },
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
                tops[at] = engine_core::math::max(tops[at], reach.stands.top);
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
