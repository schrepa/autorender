use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::math::{max, min, sin_cos};
use engine_core::{Camera, DeviceLimits, Projection, View};
use engine_heightfield::TileKey;
use engine_heightfield::tile::{TILE_STRIDE, on_triangle, tile_cell};
use engine_places::ground::TerrainTiles;
use engine_places::water::WaterObstacle;
use glam::{Vec2, Vec3};

use crate::gpu::{SETUP_SOLID_ENTRIES, SIM_TOUCHES};
use crate::shallow_water::{live_split, step_seconds};
use crate::state::{WaterState, WaterTerrain};
use crate::water_kernels::{Capacity, live_largest_bytes};
use crate::{BLOCK_SIDE, Footprint, SteadyWater, TileWater, Water};

pub const LIVE_PIXELS: f32 = 4.0;
pub const LONGEST_FRAME: f64 = 1.0 / 30.0;
const SNAP: f32 = 32.0;
const FOLLOW: f32 = 0.75;
const PROBED: f32 = 32.0;
const DATUM_DRIFT: f32 = 256.0;

#[derive(Debug)]
pub struct WaterWindow {
    pub origin: [f32; 2],
    pub first: [i64; 2],
    pub cells: u32,
    pub asked: u32,
    pub pixels: f32,
    pub cell: f32,
    pub split: u32,
    pub snap: u32,
    pub datum: f32,
    pub sources: Vec<WaterSourceCell>,
    pub solids: Vec<Footprint>,
    pub reads: Vec<WaterRead>,
    pub placed: ([i64; 2], u32),
    pub placed_pixels: f32,
    pub runs: bool,
}

#[derive(Clone, Debug)]
pub struct WaterRead {
    pub key: TileKey,
    pub square: [Vec2; 2],
    pub tile: Option<(TileKey, engine_core::Stamp)>,
    pub water: Option<Arc<TileWater>>,
    pub settled: Option<Arc<SteadyWater>>,
    pub ready: bool,
}

impl WaterRead {
    pub fn same_tile(&self, other: &Self) -> bool {
        self.key == other.key
            && self.tile == other.tile
            && alike(self.water.as_ref(), other.water.as_ref())
    }

    fn same(&self, other: &Self) -> bool {
        self.same_tile(other)
            && alike(self.settled.as_ref(), other.settled.as_ref())
            && self.ready == other.ready
    }
}

fn alike<T>(a: Option<&Arc<T>>, b: Option<&Arc<T>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
    }
}

impl WaterWindow {
    pub fn square(&self) -> (Vec2, Vec2) {
        let (first, cells) = self.placed;
        let low = Vec2::from(self.origin)
            + (Vec2::new(first[0] as f32, first[1] as f32)
                - Vec2::new(self.first[0] as f32, self.first[1] as f32))
                * self.cell;
        (low, low + Vec2::splat(cells as f32 * self.cell))
    }

    pub fn middle(&self) -> Vec2 {
        self.square().0 + Vec2::splat(0.5 * self.placed.1 as f32 * self.cell)
    }

    pub fn blocks(&self) -> u32 {
        self.cells / BLOCK_SIDE
    }

    fn within(&mut self, cells: u32) {
        let side = self.blocks();
        let kept = (cells / BLOCK_SIDE).clamp(1, side);
        let (shift, n) = ((side - kept) / 2 * BLOCK_SIDE, kept * BLOCK_SIDE);
        let follows = FOLLOW * self.snap as f32 * BLOCK_SIDE as f32 * self.cell;
        let cell = self.cell;
        let holds = |cells: u32| max(0.5 * cells as f32 * cell - follows, cell);
        let was = self.cells;
        self.sources.retain_mut(|source| {
            let (i, j) = (source.cell % was, source.cell / was);
            let inside = i >= shift && j >= shift && i < shift + n && j < shift + n;
            if inside {
                source.cell = (j - shift) * n + i - shift;
            }
            inside
        });
        self.origin = (Vec2::from(self.origin) + Vec2::splat(shift as f32 * cell)).to_array();
        self.first = [
            self.first[0] + i64::from(shift),
            self.first[1] + i64::from(shift),
        ];
        self.cells = n;
        self.pixels = self.pixels * holds(was) / holds(n);
    }

    fn fit(&mut self, binding: u64) {
        let held = (self.sources.len(), self.solids.len());
        let fits = |blocks: u32| largest_bytes(blocks, held) <= binding;
        let side = self.blocks();
        if fits(side) {
            return;
        }
        let (mut low, mut high) = (0, side);
        while low + 1 < high {
            let middle = low + (high - low) / 2;
            if fits(middle) {
                low = middle;
            } else {
                high = middle;
            }
        }
        if low >= 1 && fits(low) {
            self.within(low * BLOCK_SIDE);
        } else {
            self.runs = false;
        }
    }

    pub fn touching(
        &self,
        obstacles: &[WaterObstacle],
        eye: Vec3,
        touching: &mut Vec<WaterObstacle>,
    ) -> u32 {
        let (low, high) = self.square();
        touching.clear();
        touching.extend(obstacles.iter().filter(|obstacle| {
            let reach = obstacle.reach();
            let (at, half) = (
                Vec2::new(obstacle.center.x, obstacle.center.z),
                Vec2::new(reach.x, reach.z),
            );
            (at + half).cmpge(low).all() && (at - half).cmple(high).all()
        }));
        touching.sort_by(|a, b| {
            a.center
                .distance_squared(eye)
                .total_cmp(&b.center.distance_squared(eye))
        });
        let beyond = touching.len().saturating_sub(SIM_TOUCHES as usize);
        touching.truncate(SIM_TOUCHES as usize);
        beyond as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSourceCell {
    pub cell: u32,
    pub rate: f32,
    pub way: [f32; 2],
}

fn live_reach(cell: f32, height: u32, fov: f32) -> f32 {
    let (sin, cos) = sin_cos(0.5 * fov);
    cell * height.max(1) as f32 * cos / max(2.0 * sin * LIVE_PIXELS, 1.0e-6)
}

fn live_cell(settled: f32, finest: f32) -> f32 {
    settled / live_split(settled, finest) as f32
}

pub fn live_steps(cell: f32) -> u32 {
    (LONGEST_FRAME / f64::from(step_seconds(cell))).ceil() as u32
}

fn largest_bytes(blocks: u32, (sources, solids): (usize, usize)) -> u64 {
    let cells = blocks * BLOCK_SIDE;
    let capacity = Capacity {
        blocks: blocks * blocks,
        mapped: blocks * blocks,
        edges: 2 * cells,
        sources: sources.max(1) as u32,
    };
    let inputs = (u64::from(capacity.sources) + 8 * u64::from(cells)) * 16;
    let list = u64::from(SETUP_SOLID_ENTRIES) * solids as u64 + u64::from(capacity.blocks);
    live_largest_bytes(capacity)
        .max(inputs)
        .max((list * 16).next_power_of_two())
}

pub(crate) fn place_water_windows(
    mut state: ResMut<WaterState>,
    camera: Res<Camera>,
    view: Res<View>,
    water: Res<Water>,
    (tiles, limits): (Res<TerrainTiles>, Res<DeviceLimits>),
) {
    let state = state.bypass_change_detection();
    let budget = water.live_budget;
    let wanted = tiles
        .terrains
        .iter()
        .find(|entry| !entry.surroundings && entry.baked.is_some())
        .map(|entry| entry.terrain);
    let height = view.display_size().y;
    let fov = match camera.sanitized().projection {
        Projection::Perspective {
            vertical_fov_radians,
        } => vertical_fov_radians,
        Projection::Orthographic { .. } => 0.0,
    };
    let eye = view.detail_eye();
    for terrain in &mut state.terrains {
        let settled = terrain.settled.cell();
        if Some(terrain.entity) != wanted || !(settled.is_finite() && settled > 0.0) {
            terrain.window = None;
            continue;
        }
        let cell = live_cell(settled, terrain.pyramid.spacing(0));
        let reach = live_reach(cell, height, fov);
        let device = (limits.storage_room, limits.is_changed());
        follow(terrain, (reach, cell, budget), eye, device);
    }
}

fn follow(
    state: &mut WaterTerrain,
    (reach, cell, budget): (f32, f32, u32),
    eye: Vec3,
    (binding, refit): (u64, bool),
) {
    let side = BLOCK_SIDE as f32;
    let snap_blocks = max((SNAP / (cell * side)).round(), 1.0);
    let snap = snap_blocks * side * cell;
    let block = cell * side;
    let asked = max((2.0 * (reach + FOLLOW * snap) / block).ceil(), 1.0);
    let afforded = (f64::from(budget) / f64::from(live_steps(cell))).sqrt() as f32 / side;
    let least = (2.0 * FOLLOW * snap / block).ceil() + 2.0;
    let cells = min(asked, max(afforded.floor(), least)) as u32 * BLOCK_SIDE;
    let asked = asked as u32 * BLOCK_SIDE;
    let size = cells as f32 * cell;
    let pixels = LIVE_PIXELS * max(reach / max(0.5 * size - FOLLOW * snap, cell), 1.0);
    let split = (state.settled.cell() / cell).round() as u32;
    let eye = Vec2::new(eye.x, eye.z);
    let flat = Vec2::new(state.origin.x, state.origin.z);
    let before = state
        .window
        .as_ref()
        .filter(|window| window.cell == cell && window.split == split);
    let staying = before.filter(|window| {
        window.placed.1 == cells && (eye - window.middle()).abs().max_element() <= FOLLOW * snap
    });
    let (first, datum) = if let Some(window) = staying {
        (window.placed.0, window.datum)
    } else {
        let at = ((eye - flat - Vec2::splat(0.5 * size)) / snap).round();
        let step = snap_blocks as i64 * i64::from(BLOCK_SIDE);
        let first = [at.x as i64 * step, at.y as i64 * step];
        let Some(lowest) = lowest(state, first, (cells, cell)) else {
            return;
        };
        let lowest = lowest.floor();
        let datum = before.map_or(lowest, |window| {
            if (lowest - window.datum).abs() <= DATUM_DRIFT {
                window.datum
            } else {
                lowest
            }
        });
        (first, datum)
    };
    if let Some(window) = staying
        && !refit
        && (window.asked, window.placed_pixels) == (asked, pixels)
        && !reads_moved(state, window)
    {
        return;
    }
    let reads = reads(state, first, (cells, cell));
    let changed: Vec<[Vec2; 2]> = state
        .window
        .as_ref()
        .map_or_else(Vec::new, |window| changed_squares(&window.reads, &reads));
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let mut window = WaterWindow {
        origin: (flat + low).to_array(),
        first,
        cells,
        asked,
        pixels,
        cell,
        split,
        snap: snap_blocks as u32,
        datum,
        sources: sources(state, (first, cells, cell)),
        solids: solids(state, (low, low + Vec2::splat(size))),
        reads,
        placed: (first, cells),
        placed_pixels: pixels,
        runs: true,
    };
    window.fit(binding);
    state.window = Some(window);
    state.window_moved = true;
    state.window_changed.extend(changed);
}

fn reads_moved(state: &WaterTerrain, window: &WaterWindow) -> bool {
    if !state.touched_all && state.touched.is_empty() {
        return false;
    }
    let pyramid = &state.pyramid;
    window.reads.iter().any(|read| {
        let middle = 0.5 * (read.square[0] + read.square[1]);
        let covered = state.touched_all
            || state
                .touched
                .iter()
                .any(|&key| pyramid.tile_at(u32::from(key.level), middle) == Some(key));
        covered && !read.same(&read_at(state, read.key))
    })
}

fn changed_squares(before: &[WaterRead], reads: &[WaterRead]) -> Vec<[Vec2; 2]> {
    reads
        .iter()
        .filter(|read| {
            before
                .iter()
                .any(|held| held.key == read.key && !held.same_tile(read))
        })
        .map(|read| read.square)
        .collect()
}

fn sample_height(state: &WaterTerrain, local: Vec2) -> Option<f32> {
    let pyramid = &state.pyramid;
    (0..pyramid.levels()).find_map(|level| {
        let key = pyramid.tile_at(level, local)?;
        let face = &state.held.get(&key)?.face;
        let (i, j, fu, fv) = tile_cell(face.spacing, local - pyramid.tile_origin(key))?;
        let stride = TILE_STRIDE as i32;
        let height = |x: i32, z: i32| face.heights[((z + 1) * stride + x + 1) as usize];
        let corners = [
            height(i, j),
            height(i + 1, j),
            height(i, j + 1),
            height(i + 1, j + 1),
        ];
        Some(on_triangle(corners, fu, fv))
    })
}

fn lowest(state: &WaterTerrain, first: [i64; 2], (cells, cell): (u32, f32)) -> Option<f32> {
    let size = cells as f32 * cell;
    let steps = max((size / PROBED).ceil(), 1.0) as u32;
    let apart = size / steps as f32;
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let mut lowest = f32::INFINITY;
    for j in 0..=steps {
        for i in 0..=steps {
            let local = low + Vec2::new(i as f32, j as f32) * apart;
            let height = sample_height(state, local)?;
            lowest = min(lowest, height);
        }
    }
    Some(state.origin.y + lowest)
}

fn reads(state: &WaterTerrain, first: [i64; 2], (cells, cell): (u32, f32)) -> Vec<WaterRead> {
    let pyramid = &state.pyramid;
    let size = pyramid.tile_size(0);
    let [columns, rows] = pyramid.tiles(0);
    let range = |low: f32, high: f32, count: u32| {
        let first = max((low / size).floor(), 0.0) as u32;
        let last = min((high / size).floor(), count as f32 - 1.0) as u32;
        first..=last
    };
    let margin = Vec2::splat(state.settled.cell());
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell - margin;
    let high = Vec2::new(
        (first[0] + i64::from(cells)) as f32,
        (first[1] + i64::from(cells)) as f32,
    ) * cell
        + margin;
    let mut reads = Vec::new();
    for z in range(low.y, high.y, rows) {
        for x in range(low.x, high.x, columns) {
            reads.push(read_at(state, TileKey::new(0, x as u16, z as u16)));
        }
    }
    reads
}

fn read_at(state: &WaterTerrain, key: TileKey) -> WaterRead {
    let pyramid = &state.pyramid;
    let size = pyramid.tile_size(0);
    let corner = pyramid.tile_origin(key);
    let middle = corner + Vec2::splat(0.5 * size);
    let tile = (0..pyramid.levels()).find_map(|level| {
        let held = pyramid.tile_at(level, middle)?;
        state.held.get(&held)
    });
    let ready = tile.is_some_and(|tile| {
        let held_back = state
            .resident(&key)
            .is_some_and(|held| held.watered.is_some());
        tile.face.key == key && state.settled.ready(key, tile.given.is_some() || held_back)
    });
    WaterRead {
        key,
        square: [corner, corner + Vec2::splat(size)],
        tile: tile.map(|tile| (tile.face.key, tile.face.stamp)),
        water: tile
            .filter(|tile| tile.resident())
            .and_then(|tile| tile.water.clone()),
        settled: state
            .settled
            .held
            .get(&key)
            .map(|(water, _)| Arc::clone(water)),
        ready,
    }
}

fn solids(state: &WaterTerrain, (low, high): (Vec2, Vec2)) -> Vec<Footprint> {
    state
        .settled
        .held
        .iter()
        .flat_map(|(_, (water, _))| water.solids.iter().copied())
        .filter(|solid| {
            let (at, half) = (solid.middle(), solid.reach());
            (at + half).cmpge(low).all() && (at - half).cmple(high).all()
        })
        .collect()
}

fn sources(
    state: &WaterTerrain,
    (first, cells, cell): ([i64; 2], u32, f32),
) -> Vec<WaterSourceCell> {
    let settled = state.settled.cell();
    let pyramid = &state.pyramid;
    let n = i64::from(cells);
    let area = settled * settled;
    let spans = |axis: usize, low: f32| {
        let high = low + settled;
        let from = ((low / cell).floor() as i64).max(first[axis]);
        let to = ((high / cell).ceil() as i64).min(first[axis] + n);
        (from..to).filter_map(move |at| {
            let overlap = min((at + 1) as f32 * cell, high) - max(at as f32 * cell, low);
            (overlap > 0.0).then_some((at - first[axis], overlap))
        })
    };
    let size = pyramid.tile_size(0);
    let [columns, rows] = pyramid.tiles(0);
    let range = |low: f32, high: f32, count: u32| {
        let first = max((low / size).floor(), 0.0) as u32;
        let last = min((high / size).floor(), count as f32 - 1.0) as u32;
        first..=last
    };
    let margin = Vec2::splat(settled);
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell - margin;
    let high = Vec2::new((first[0] + n) as f32, (first[1] + n) as f32) * cell + margin;
    let mut keys: Vec<TileKey> = range(low.y, high.y, rows)
        .flat_map(|z| {
            range(low.x, high.x, columns).map(move |x| TileKey::new(0, x as u16, z as u16))
        })
        .collect();
    keys.sort_unstable();
    let mut brought: Vec<(u32, f32, [f32; 2])> = Vec::new();
    for key in keys {
        let Some((water, _)) = state.settled.held.get(&key) else {
            continue;
        };
        let columns = i64::from(water.cells[0]);
        for source in &water.sources {
            let low = Vec2::new(
                (water.first[0] + i64::from(source.cell) % columns) as f32,
                (water.first[1] + i64::from(source.cell) / columns) as f32,
            ) * settled;
            for (z, along_z) in spans(1, low.y) {
                for (x, along_x) in spans(0, low.x) {
                    let rate = source.rate * along_x * along_z / area;
                    let way = [source.way[0] * rate, source.way[1] * rate];
                    brought.push(((z * n + x) as u32, rate, way));
                }
            }
        }
    }
    brought.sort_by_key(|&(at, _, _)| at);
    let mut merged: Vec<(WaterSourceCell, [f32; 2])> = Vec::with_capacity(brought.len());
    for (at, rate, way) in brought {
        match merged.last_mut() {
            Some((last, weighted)) if last.cell == at => {
                last.rate += rate;
                *weighted = [weighted[0] + way[0], weighted[1] + way[1]];
            }
            _ => merged.push((
                WaterSourceCell {
                    cell: at,
                    rate,
                    way: [0.0; 2],
                },
                way,
            )),
        }
    }
    merged
        .into_iter()
        .map(|(source, weighted)| {
            let share = if source.rate > 0.0 {
                1.0 / source.rate
            } else {
                0.0
            };
            WaterSourceCell {
                way: [weighted[0] * share, weighted[1] * share],
                ..source
            }
        })
        .collect()
}
