use std::sync::Arc;

use bevy_ecs::prelude::{DetectChangesMut, Res, ResMut};
use glam::{Vec2, Vec3};

use super::residency::TerrainState;
use super::{TerrainDebug, Terrains, Viewport, Water};
use crate::math::{max, min, sin_cos};
use crate::terrain_data::shallow_water::{live_split, step_seconds};
use crate::terrain_data::{BLOCK_SIDE, Footprint, SteadyWater, Tile, TileKey};
use crate::{Camera, WaterObstacle};

pub const LIVE_PIXELS: f32 = 4.0;
pub const LONGEST_FRAME: f64 = 1.0 / 30.0;
const SNAP: f32 = 32.0;
const FOLLOW: f32 = 0.75;
const PROBED: f32 = 32.0;
const DATUM_DRIFT: f32 = 256.0;
pub const SIM_TOUCHES: usize = 32;

#[derive(Clone, Debug)]
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
    pub sources: Arc<[WaterSourceCell]>,
    pub solids: Arc<[Footprint]>,
    pub reads: Arc<[WaterRead]>,
}

#[derive(Clone, Debug)]
pub struct WaterRead {
    pub key: TileKey,
    pub square: [Vec2; 2],
    pub tile: Option<Arc<Tile>>,
    pub settled: Option<Arc<SteadyWater>>,
    pub ready: bool,
}

impl WaterRead {
    pub fn same_tile(&self, other: &Self) -> bool {
        self.key == other.key && alike(self.tile.as_ref(), other.tile.as_ref())
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
        _ => false,
    }
}

impl WaterWindow {
    pub fn middle(&self) -> Vec2 {
        Vec2::from(self.origin) + Vec2::splat(0.5 * self.cells as f32 * self.cell)
    }

    pub fn square(&self) -> (Vec2, Vec2) {
        let low = Vec2::from(self.origin);
        (low, low + Vec2::splat(self.cells as f32 * self.cell))
    }

    pub fn blocks(&self) -> u32 {
        self.cells / BLOCK_SIDE
    }

    #[must_use]
    pub fn within(&self, cells: u32) -> Self {
        let side = self.blocks();
        let kept = (cells / BLOCK_SIDE).clamp(1, side);
        let (shift, n) = ((side - kept) / 2 * BLOCK_SIDE, kept * BLOCK_SIDE);
        let follows = FOLLOW * self.snap as f32 * BLOCK_SIDE as f32 * self.cell;
        let holds = |cells: u32| max(0.5 * cells as f32 * self.cell - follows, self.cell);
        let sources = self
            .sources
            .iter()
            .filter_map(|source| {
                let (i, j) = (source.cell % self.cells, source.cell / self.cells);
                let inside = i >= shift && j >= shift && i < shift + n && j < shift + n;
                inside.then(|| WaterSourceCell {
                    cell: (j - shift) * n + i - shift,
                    ..*source
                })
            })
            .collect();
        Self {
            origin: (Vec2::from(self.origin) + Vec2::splat(shift as f32 * self.cell)).to_array(),
            first: [
                self.first[0] + i64::from(shift),
                self.first[1] + i64::from(shift),
            ],
            cells: n,
            pixels: self.pixels * holds(self.cells) / holds(n),
            sources,
            ..self.clone()
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
        let beyond = touching.len().saturating_sub(SIM_TOUCHES);
        touching.truncate(SIM_TOUCHES);
        beyond as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterSourceCell {
    pub cell: u32,
    pub rate: f32,
    pub way: [f32; 2],
}

impl Terrains {
    pub fn water_windows(&self) -> impl Iterator<Item = (Vec2, Vec2)> + '_ {
        self.entries
            .iter()
            .filter_map(|state| state.hydrology.window.as_deref())
            .map(WaterWindow::square)
    }
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

pub(crate) fn place_water_windows(
    mut terrains: ResMut<Terrains>,
    camera: Res<Camera>,
    viewport: Option<Res<Viewport>>,
    debug: Option<Res<TerrainDebug>>,
    water: Option<Res<Water>>,
) {
    let terrains = terrains.bypass_change_detection();
    let water = water.map(|water| *water);
    let drawn = water.is_some_and(|water| water.enabled);
    let budget = water.unwrap_or_default().live_budget;
    let wanted = drawn
        .then(|| {
            terrains
                .entries
                .iter()
                .position(|state| state.around.is_none() && state.baked.is_some())
        })
        .flatten();
    let camera = camera.sanitized();
    let height = viewport.map_or_else(|| Viewport::default().height, |viewport| viewport.height);
    let eye = debug
        .and_then(|debug| debug.freeze)
        .unwrap_or(camera.position);
    for (index, state) in terrains.entries.iter_mut().enumerate() {
        let settled = state.settled.cell();
        if Some(index) != wanted || !(settled.is_finite() && settled > 0.0) {
            state.hydrology.window = None;
            continue;
        }
        let cell = live_cell(settled, state.terrain.pyramid.spacing(0));
        let reach = live_reach(cell, height, camera.vertical_fov_radians);
        follow(state, (reach, cell, budget), eye);
    }
}

fn follow(state: &mut TerrainState, (reach, cell, budget): (f32, f32, u32), eye: Vec3) {
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
    let flat = Vec2::new(state.terrain.origin.x, state.terrain.origin.z);
    let before = state
        .hydrology
        .window
        .clone()
        .filter(|window| window.cell == cell && window.split == split);
    let staying = before.as_ref().filter(|window| {
        window.cells == cells && (eye - window.middle()).abs().max_element() <= FOLLOW * snap
    });
    let (first, datum) = if let Some(window) = staying {
        (window.first, window.datum)
    } else {
        let at = ((eye - flat - Vec2::splat(0.5 * size)) / snap).round();
        let step = snap_blocks as i64 * i64::from(BLOCK_SIDE);
        let first = [at.x as i64 * step, at.y as i64 * step];
        let Some(lowest) = lowest(state, first, (cells, cell)) else {
            return;
        };
        let lowest = lowest.floor();
        let datum = before.as_ref().map_or(lowest, |window| {
            if (lowest - window.datum).abs() <= DATUM_DRIFT {
                window.datum
            } else {
                lowest
            }
        });
        (first, datum)
    };
    let reads = reads(state, first, (cells, cell));
    if let Some(window) = staying {
        let read = window.reads.len() == reads.len()
            && window
                .reads
                .iter()
                .zip(&reads)
                .all(|(held, read)| held.same(read));
        if read && (window.asked, window.pixels) == (asked, pixels) {
            return;
        }
    }
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    state.hydrology.window = Some(Arc::new(WaterWindow {
        origin: (flat + low).to_array(),
        first,
        cells,
        asked,
        pixels,
        cell,
        split,
        snap: snap_blocks as u32,
        datum,
        sources: sources(state, (first, cells, cell)).into(),
        solids: solids(state, (low, low + Vec2::splat(size))).into(),
        reads: reads.into(),
    }));
}

fn lowest(state: &TerrainState, first: [i64; 2], (cells, cell): (u32, f32)) -> Option<f32> {
    let size = cells as f32 * cell;
    let steps = max((size / PROBED).ceil(), 1.0) as u32;
    let apart = size / steps as f32;
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let mut lowest = f32::INFINITY;
    for j in 0..=steps {
        for i in 0..=steps {
            let (_, sample) = state.sample(low + Vec2::new(i as f32, j as f32) * apart)?;
            lowest = min(lowest, sample.height);
        }
    }
    Some(state.terrain.origin.y + lowest)
}

fn reads(state: &TerrainState, first: [i64; 2], (cells, cell): (u32, f32)) -> Vec<WaterRead> {
    let pyramid = &state.terrain.pyramid;
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
            let key = TileKey::new(0, x as u16, z as u16);
            let corner = pyramid.tile_origin(key);
            let middle = corner + Vec2::splat(0.5 * size);
            let tile = (0..pyramid.levels()).find_map(|level| {
                let held = pyramid.tile_at(level, middle)?;
                state.store.get(&held).cloned()
            });
            let held_back = state.store.watered(&key).is_some_and(Option::is_some);
            let ready = tile
                .as_ref()
                .is_some_and(|tile| tile.key() == key && state.settled.ready(key, tile, held_back));
            reads.push(WaterRead {
                key,
                square: [corner, corner + Vec2::splat(size)],
                tile,
                settled: state
                    .settled
                    .held
                    .get(&key)
                    .map(|(water, _)| Arc::clone(water)),
                ready,
            });
        }
    }
    reads
}

fn solids(state: &TerrainState, (low, high): (Vec2, Vec2)) -> Vec<Footprint> {
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
    state: &TerrainState,
    (first, cells, cell): ([i64; 2], u32, f32),
) -> Vec<WaterSourceCell> {
    let settled = state.settled.cell();
    let pyramid = &state.terrain.pyramid;
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
