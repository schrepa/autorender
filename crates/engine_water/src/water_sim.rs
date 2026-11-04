use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bytemuck::Zeroable;
use engine_core::{Camera, Cut, Error, FrameTime};
use engine_heightfield::TileKey;
use engine_places::ground::{GroundBaking, GroundTiles, TerrainTiles};
use engine_places::water::{WaterObstacle, WaterObstacles};
use engine_render::{ComputePipeline, Count, Encode, Install, Pass, Prepare};
use glam::{Mat3, Vec2, Vec3};

use crate::gpu::{
    LIVE_CELL, LIVE_HEAD, LIVE_NONE, SETUP_RISEN, SETUP_SOLID_ENTRIES, SETUP_STANDS, SETUP_WHOLE,
    SIM_BURSTING, SIM_LIVE, SIM_STANDING, SIM_STEPPING, SIM_TOUCHES, SimSetup, SimSlot, SimTouch,
    SimTouches, WATER_SETUP_RING_GROUP,
};
use crate::shallow_water::step_seconds;
use crate::state::WaterState;
use crate::water::WaterGpu;
use crate::water_kernels::{Capacity, Kernels, Layout, SetupHandles, SlotHandles, grid, squares};
use crate::windows::WaterRead;
use crate::{
    BLOCK_CELLS, BLOCK_SIDE, Footprint, LiveWater, NO_BLOCK, SteadyWater, StoredBlocks, Water,
    WaterWindow, live_steps,
};

const RISEN: usize = 256;
const SETUP_SOLID: usize = SETUP_SOLID_ENTRIES as usize;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Held {
    place: [i64; 2],
    runs: bool,
}

pub(crate) struct Listed {
    pub(crate) entries: Vec<[u32; 4]>,
    pub(crate) offset: u64,
    pub(crate) frame: SimSetup,
    pub(crate) blocks: u32,
    pub(crate) ghosts: u32,
}

pub(crate) struct Square {
    pub(crate) chunks: Vec<(u32, Arc<SteadyWater>)>,
    pub(crate) sources: Vec<[f32; 4]>,
    pub(crate) solids: Vec<[f32; 4]>,
}

pub(crate) struct SimDrawn {
    pub(crate) origin: [f32; 2],
    pub(crate) cell: f32,
    pub(crate) cells: u32,
    pub(crate) datum: f32,
    pub(crate) pixels: f32,
    pub(crate) wrap: [u32; 2],
    pub(crate) pool: u32,
}

const CELL_WORDS: u32 = BLOCK_CELLS as u32 * LIVE_CELL;

fn chunk_words(live: &LiveWater) -> u32 {
    LIVE_HEAD + live.blocks[0] * live.blocks[1] + live.wet.len() as u32 * CELL_WORDS
}

fn chunk_head(live: &LiveWater, at: u32) -> Vec<u32> {
    let [columns, rows] = live.blocks;
    let table = LIVE_HEAD as usize;
    let mut words = vec![LIVE_NONE; table + (columns * rows) as usize];
    words[..table].copy_from_slice(&[
        live.first[0] as i32 as u32,
        live.first[1] as i32 as u32,
        columns,
        rows,
    ]);
    let cells = at + words.len() as u32;
    for (index, &block) in live.wet.iter().enumerate() {
        words[table + block as usize] = cells + index as u32 * CELL_WORDS;
    }
    words
}

#[derive(Default)]
struct LivePool {
    table_room: u32,
    free: Vec<Range<u32>>,
    tiles: BTreeMap<TileKey, (Range<u32>, Arc<SteadyWater>)>,
    table: Vec<u32>,
}

impl LivePool {
    fn reset(&mut self, table_room: u32, room: u32) {
        self.table_room = table_room;
        self.free = vec![table_room..room];
        self.tiles.clear();
    }

    fn take(&mut self, words: u32) -> Option<u32> {
        let at = self
            .free
            .iter()
            .position(|free| free.end - free.start >= words)?;
        let start = self.free[at].start;
        self.free[at].start += words;
        if self.free[at].is_empty() {
            self.free.remove(at);
        }
        Some(start)
    }

    fn give(&mut self, range: Range<u32>) {
        let at = self.free.partition_point(|free| free.start < range.start);
        self.free.insert(at, range);
        if at + 1 < self.free.len() && self.free[at].end == self.free[at + 1].start {
            self.free[at].end = self.free[at + 1].end;
            self.free.remove(at + 1);
        }
        if at > 0 && self.free[at - 1].end == self.free[at].start {
            self.free[at - 1].end = self.free[at].end;
            self.free.remove(at);
        }
    }

    fn place(
        &mut self,
        reads: &[WaterRead],
        tiles: [u32; 4],
    ) -> Option<Vec<(u32, Arc<SteadyWater>)>> {
        let [least_x, least_z, columns, rows] = tiles;
        if columns * rows > self.table_room {
            return None;
        }
        let leaving: Vec<TileKey> = self
            .tiles
            .iter()
            .filter(|(key, (_, water))| {
                !reads.iter().any(|read| {
                    read.key == **key
                        && read
                            .settled
                            .as_ref()
                            .is_some_and(|settled| Arc::ptr_eq(settled, water))
                })
            })
            .map(|(&key, _)| key)
            .collect();
        for key in leaving {
            if let Some((range, _)) = self.tiles.remove(&key) {
                self.give(range);
            }
        }
        let mut chunks = Vec::new();
        for read in reads {
            let Some(water) = read.settled.as_ref() else {
                continue;
            };
            if self.tiles.contains_key(&read.key) {
                continue;
            }
            let words = chunk_words(&water.live);
            let at = self.take(words)?;
            self.tiles.insert(read.key, (at..at + words, Arc::clone(water)));
            chunks.push((at, Arc::clone(water)));
        }
        self.table.clear();
        self.table.resize((columns * rows) as usize, LIVE_NONE);
        for read in reads {
            if let Some((range, _)) = self.tiles.get(&read.key) {
                let column = u32::from(read.key.x) - least_x;
                let row = u32::from(read.key.z) - least_z;
                self.table[(row * columns + column) as usize] = range.start;
            }
        }
        Some(chunks)
    }
}

#[derive(Default)]
pub(crate) struct SimWork {
    pub(crate) layout: Option<(Layout, u64)>,
    pub(crate) setup: Option<((usize, usize), u64)>,
    pub(crate) square: Option<Square>,
    pub(crate) table: Option<StoredBlocks>,
    pub(crate) listed: Option<Listed>,
    pub(crate) slot: SimSlot,
    pub(crate) touches: Option<SimTouches>,
    pub(crate) steps: u32,
    pub(crate) sourced: u32,
    pub(crate) blocks: u32,
    pub(crate) terrain: u32,
    pub(crate) ready: bool,
    pub(crate) drawn: Option<SimDrawn>,
}

struct SimWindow {
    cells: u32,
    cell: f32,
    datum: f32,
    split: u32,
    asked: u32,
    placed: u32,
    pool: u32,
    stored: StoredBlocks,
    wrap: [u32; 2],
    layout: Layout,
    layout_generation: u64,
    room: usize,
    live_room: usize,
    setup_generation: u64,
    sourced: u32,
    solids: u32,
    tiles: [u32; 4],
    live: LivePool,
    held: Vec<Option<Held>>,
    sweep: usize,
    sweep_left: usize,
    changed: Vec<[Vec2; 2]>,
    restand: bool,
    table: bool,
    ready: bool,
    waiting: u32,
    whole: u32,
}

#[derive(Default)]
pub(crate) struct LiveSim {
    window: Option<SimWindow>,
    steps: u32,
    behind: u32,
    tick: Option<i64>,
    generation: u64,
    touches: [u32; 2],
    grounded: bool,
    made: bool,
    pub(crate) work: SimWork,
}

pub(crate) struct Live<'a> {
    window: &'a WaterWindow,
    moved: bool,
    changed: Vec<[Vec2; 2]>,
    sweep: bool,
    touches: &'a [WaterObstacle],
    beyond: u32,
    ground: Option<u32>,
}

impl LiveSim {
    fn prepare(
        &mut self,
        live: Option<Live<'_>>,
        resident: &dyn Fn(&WaterRead) -> bool,
        seconds: f64,
    ) {
        self.work = SimWork::default();
        let Some(live) = live else {
            if self.window.take().is_some() {
                self.generation += 1;
            }
            (self.steps, self.behind, self.tick, self.touches) = (0, 0, None, [0; 2]);
            return;
        };
        if live.ground.is_some() {
            self.grounded = true;
        }
        let sets_up = self.made && live.ground.is_some();
        let window = live.window;
        if self.window.is_none() || live.moved {
            let same_size = self.window.as_ref().is_some_and(|current| {
                current.cells == window.cells
                    && current.cell == window.cell
                    && current.pool == pool_of(window)
            });
            let square = if same_size {
                self.window
                    .as_mut()
                    .map(|current| current.follow(window, live.changed))
            } else {
                self.generation += 1;
                let (held, square) = SimWindow::new(window, self.generation);
                self.window = Some(held);
                Some(square)
            };
            self.work.square = square;
        }
        let Some(current) = self.window.as_mut() else {
            return;
        };
        if live.sweep {
            current.sweep_left = current.held.len();
        }
        let ready = current.ready;
        let (table, listed) = current.set_up(window, sets_up, resident);
        self.work.table = table;
        self.work.listed = listed.map(|mut listed| {
            listed.frame.terrain = [live.ground.unwrap_or(0), 0, 0, 0];
            listed
        });
        if current.ready != ready {
            self.generation += 1;
        }
        let (cells, cell) = (window.cells, window.cell);
        let step = step_seconds(cell);
        let tick = (seconds / f64::from(step)).floor() as i64;
        let most = i64::from(live_steps(cell));
        let passed = self.tick.map_or(0, |last| (tick - last).max(0));
        let starting = self.tick.map_or(tick, |last| last + 1);
        self.tick = Some(tick);
        (self.steps, self.behind) = if current.ready {
            (passed.min(most) as u32, (passed - most).max(0) as u32)
        } else {
            (0, 0)
        };
        let pushing = touching(window, live.touches);
        self.touches = [
            pushing.as_ref().map_or(0, |pushing| pushing.count[0]),
            live.beyond,
        ];
        let [columns, rows] = current.stored.blocks;
        let pooled = current.stored.stored() as u32;
        let [x, z] = window.first;
        self.work.layout = Some((current.layout, current.layout_generation));
        self.work.setup = Some(((current.room, current.live_room), current.setup_generation));
        self.work.slot = SimSlot {
            shape: [
                cells,
                cells,
                SIM_STEPPING | SIM_LIVE | SIM_BURSTING,
                current.sourced,
            ],
            step: [cell, step, 0.0, 0.0],
            blocks: [columns, rows, pooled, 0],
            place: [x as u32, z as u32, starting as u32, 0],
        };
        self.work.touches = pushing;
        self.work.steps = self.steps;
        self.work.sourced = current.sourced;
        self.work.blocks = pooled;
        self.work.terrain = live.ground.unwrap_or(0);
        self.work.ready = current.ready;
        self.work.drawn = current.ready.then(|| SimDrawn {
            origin: window.origin,
            cell,
            cells,
            datum: window.datum,
            pixels: window.pixels,
            wrap: current.wrap,
            pool: current.pool,
        });
    }

    fn rest(&mut self, enabled: bool) {
        self.prepare(None, &|_| false, 0.0);
        if !enabled {
            self.made = false;
        }
    }

    pub(crate) fn cells(&self) -> (Cut, [u32; 5]) {
        self.window.as_ref().filter(|window| window.ready).map_or(
            (Cut::default(), [0; 5]),
            |window| {
                (
                    Cut::of(window.asked, window.cells),
                    [
                        window.placed - window.cells,
                        self.steps,
                        self.behind,
                        window.whole,
                        window.waiting,
                    ],
                )
            },
        )
    }

    pub(crate) fn touches(&self) -> [u32; 2] {
        self.touches
    }
}

impl SimWindow {
    fn new(window: &WaterWindow, generation: u64) -> (Self, Square) {
        let pool = pool_of(window);
        let (stored, wrap) = wrapped(window, pool);
        let slots = stored.stored();
        let layout = Layout {
            capacity: Capacity::of(&stored, window.sources.len().max(1)),
            slots: 1,
            live: true,
        };
        let entries = entries(window, slots);
        let mut held = Self {
            cells: window.cells,
            cell: window.cell,
            datum: window.datum,
            split: window.split,
            asked: window.asked,
            placed: window.placed.1,
            pool,
            stored,
            wrap,
            layout,
            layout_generation: generation,
            room: entries.max(1).next_power_of_two(),
            live_room: 4,
            setup_generation: 0,
            sourced: 0,
            solids: 0,
            tiles: [0; 4],
            live: LivePool::default(),
            held: vec![None; slots],
            sweep: 0,
            sweep_left: slots,
            changed: Vec::new(),
            restand: true,
            table: true,
            ready: false,
            waiting: 0,
            whole: 0,
        };
        let square = held.write_square(window);
        (held, square)
    }

    fn follow(&mut self, window: &WaterWindow, changed: Vec<[Vec2; 2]>) -> Square {
        self.changed = changed;
        if (window.datum, window.split) != (self.datum, self.split) {
            self.held.fill(None);
        }
        (self.stored, self.wrap) = wrapped(window, self.pool);
        self.datum = window.datum;
        self.split = window.split;
        self.asked = window.asked;
        self.placed = window.placed.1;
        self.write_square(window)
    }

    fn write_square(&mut self, window: &WaterWindow) -> Square {
        let sources = window.sources.len() as u32;
        let grown = sources > self.layout.capacity.sources;
        if grown {
            self.layout.capacity.sources = sources.next_power_of_two();
        }
        let entries = entries(window, self.held.len());
        let tiles = tile_span(&window.reads);
        let room_grown = grown || entries > self.room;
        if room_grown {
            self.room = entries.max(self.room).max(1).next_power_of_two();
        }
        let placed = self.live.place(&window.reads, tiles).filter(|_| !room_grown);
        let chunks = placed.unwrap_or_else(|| {
            let table_room = (tiles[2] * tiles[3]).max(1).next_power_of_two();
            let words = window
                .reads
                .iter()
                .filter_map(|read| read.settled.as_ref())
                .map(|water| chunk_words(&water.live) as usize)
                .sum::<usize>()
                + table_room as usize;
            self.live_room = words.max(self.live_room).max(4).next_power_of_two();
            self.setup_generation += 1;
            self.live.reset(table_room, self.live_room as u32);
            self.live.place(&window.reads, tiles);
            Vec::new()
        });
        self.tiles = tiles;
        let sources = sources_of(&self.layout, window, &self.stored);
        self.sourced = sources.len() as u32;
        let solids = solids_of(&window.solids);
        self.solids = (solids.len() / SETUP_SOLID) as u32;
        (self.restand, self.table) = (true, true);
        self.sweep_left = self.held.len();
        Square {
            chunks,
            sources,
            solids,
        }
    }

    fn set_up(
        &mut self,
        window: &WaterWindow,
        sets_up: bool,
        resident: &dyn Fn(&WaterRead) -> bool,
    ) -> (Option<StoredBlocks>, Option<Listed>) {
        let side = window.blocks();
        let corner = window
            .first
            .map(|first| first.div_euclid(i64::from(BLOCK_SIDE)));
        let cell = window.cell;
        let block = cell * BLOCK_SIDE as f32;
        let changed = std::mem::take(&mut self.changed);
        let missing: Vec<[Vec2; 2]> = window
            .reads
            .iter()
            .filter(|read| !resident(read))
            .map(|read| read.square)
            .collect();
        let reaches = |place: [i64; 2], squares: &[[Vec2; 2]]| {
            let low = Vec2::new(place[0] as f32, place[1] as f32) * block - Vec2::splat(cell);
            let high = low + Vec2::splat(block + 2.0 * cell);
            squares
                .iter()
                .any(|[least, most]| high.cmpge(*least).all() && low.cmple(*most).all())
        };
        let restand = sets_up && std::mem::take(&mut self.restand);
        let mut listed: Vec<[u32; 4]> = Vec::new();
        let mut slots: Vec<u32> = Vec::new();
        let (mut whole, mut waiting, mut deferred) = (0, 0, false);
        for z in 0..side {
            for x in 0..side {
                let slot = self.stored.table[(z * side + x) as usize];
                let place = [corner[0] + i64::from(x), corner[1] + i64::from(z)];
                let touched = reaches(place, &changed);
                let held = &mut self.held[slot as usize];
                if !sets_up {
                    if held.is_none_or(|held| held.place != place) || touched {
                        *held = None;
                        deferred = true;
                    }
                    waiting += u32::from(held.is_none_or(|held| !held.runs));
                    continue;
                }
                let runs = !reaches(place, &missing);
                waiting += u32::from(!runs);
                let set = Some(Held { place, runs });
                if *held == set && (runs || !restand) && !touched {
                    continue;
                }
                self.table |= held.is_none_or(|held| held.runs != runs);
                *held = set;
                slots.push(slot);
                if runs {
                    whole += 1;
                    listed.push(entry(slot, place, SETUP_WHOLE));
                } else {
                    listed.push(entry(slot, place, SETUP_STANDS));
                }
            }
        }
        self.ready = !deferred;
        self.waiting = waiting;
        if !sets_up {
            return (None, None);
        }
        let table = std::mem::take(&mut self.table).then(|| self.table_words(window));
        let set = slots.len();
        slots.sort_unstable();
        let pooled = self.held.len();
        let mut risen = 0;
        while self.sweep_left > 0 && risen < RISEN && set != pooled {
            self.sweep_left -= 1;
            let slot = self.sweep;
            self.sweep = (slot + 1) % pooled;
            let Some(Held { place, runs: true }) = self.held[slot] else {
                continue;
            };
            if slots.binary_search(&(slot as u32)).is_ok() {
                continue;
            }
            listed.push(entry(slot as u32, place, SETUP_RISEN));
            risen += 1;
        }
        let ghosts = 4 * window.cells;
        self.whole = whole;
        let [first_x, first_z] = window.first;
        let frame = SimSetup {
            counts: [listed.len() as u32, self.room as u32, ghosts, self.solids],
            square: [first_x as u32, first_z as u32, window.cells, window.split],
            meters: [cell, window.datum, 0.0, 0.0],
            ring: [self.layout.ring_at(0) as u32, 0, 0, 0],
            tiles: self.tiles,
            terrain: [0; 4],
        };
        let blocks = listed.len() as u32;
        (
            table,
            Some(Listed {
                entries: listed,
                offset: u64::from(self.solids) * SETUP_SOLID as u64 * 16,
                frame,
                blocks,
                ghosts,
            }),
        )
    }

    fn table_words(&self, window: &WaterWindow) -> StoredBlocks {
        let side = window.blocks();
        let corner = window
            .first
            .map(|first| first.div_euclid(i64::from(BLOCK_SIDE)));
        let mut stored = self.stored.clone();
        for z in 0..side {
            for x in 0..side {
                let at = (z * side + x) as usize;
                let slot = stored.table[at];
                let place = [corner[0] + i64::from(x), corner[1] + i64::from(z)];
                let runs = self.held[slot as usize] == Some(Held { place, runs: true });
                if !runs {
                    stored.table[at] = slot | SIM_STANDING;
                    stored.places[slot as usize] = NO_BLOCK;
                }
            }
        }
        stored
    }
}

fn entry(slot: u32, place: [i64; 2], what: u32) -> [u32; 4] {
    let side = i64::from(BLOCK_SIDE);
    [
        slot,
        (place[0] * side) as u32,
        (place[1] * side) as u32,
        what,
    ]
}

fn tile_span(reads: &[WaterRead]) -> [u32; 4] {
    let Some(read) = reads.first() else {
        return [0; 4];
    };
    let (mut least, mut most) = ([read.key.x, read.key.z], [read.key.x, read.key.z]);
    for read in reads {
        least = [least[0].min(read.key.x), least[1].min(read.key.z)];
        most = [most[0].max(read.key.x), most[1].max(read.key.z)];
    }
    let [columns, rows] = [0, 1].map(|axis| u32::from(most[axis] - least[axis]) + 1);
    [u32::from(least[0]), u32::from(least[1]), columns, rows]
}

fn entries(window: &WaterWindow, slots: usize) -> usize {
    SETUP_SOLID * window.solids.len() + slots
}

fn pool_of(window: &WaterWindow) -> u32 {
    window.blocks()
}

fn wrap_of(window: &WaterWindow, pool: u32) -> [u32; 2] {
    window.first.map(|first| {
        first
            .div_euclid(i64::from(BLOCK_SIDE))
            .rem_euclid(i64::from(pool)) as u32
    })
}

fn wrapped(window: &WaterWindow, pool: u32) -> (StoredBlocks, [u32; 2]) {
    let side = window.blocks();
    let wrap = wrap_of(window, pool);
    let mut table = vec![0; (side * side) as usize];
    let mut places = vec![0; (pool * pool) as usize];
    for z in 0..side {
        for x in 0..side {
            let slot = (z + wrap[1]) % pool * pool + (x + wrap[0]) % pool;
            table[(z * side + x) as usize] = slot;
            places[slot as usize] = x | (z << 16);
        }
    }
    let stored = StoredBlocks {
        cells: [window.cells; 2],
        blocks: [side; 2],
        table,
        places,
    };
    (stored, wrap)
}

fn sources_of(layout: &Layout, window: &WaterWindow, stored: &StoredBlocks) -> Vec<[f32; 4]> {
    let n = window.cells;
    window
        .sources
        .iter()
        .filter_map(|source| {
            let at = stored.at(source.cell % n, source.cell / n)?;
            Some([
                f32::from_bits(at as u32),
                source.rate,
                source.way[0],
                source.way[1],
            ])
        })
        .take(layout.capacity.sources as usize)
        .collect()
}

fn solids_of(solids: &[Footprint]) -> Vec<[f32; 4]> {
    let mut entries: Vec<[f32; 4]> = Vec::with_capacity(SETUP_SOLID * solids.len());
    for solid in solids {
        let axes = Mat3::from_cols_array(&solid.axes);
        if axes.determinant() == 0.0 {
            continue;
        }
        let inverse = axes.inverse();
        let reach = solid.reach();
        let [x, y, z] = solid.center;
        entries.extend([
            [x, y, z, if solid.round { 1.0 } else { 0.0 }],
            inverse.x_axis.extend(reach.x).to_array(),
            inverse.y_axis.extend(reach.y).to_array(),
            inverse.z_axis.extend(0.0).to_array(),
        ]);
    }
    entries
}

fn touching(window: &WaterWindow, touches: &[WaterObstacle]) -> Option<SimTouches> {
    let corner = Vec3::new(window.origin[0], window.datum, window.origin[1]);
    let size = Vec2::splat(window.cells as f32 * window.cell);
    let mut pushing = SimTouches::zeroed();
    let (mut least, mut most) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    let mut count = 0;
    for touch in touches.iter().take(SIM_TOUCHES as usize) {
        let reach = touch.reach();
        let center = touch.center - corner;
        let (middle, half) = (Vec2::new(center.x, center.z), Vec2::new(reach.x, reach.z));
        let (from, to) = (middle - half, middle + half);
        let over = to.cmpge(Vec2::ZERO).all() && from.cmple(size).all();
        let fills = touch.axes.determinant().abs() > f32::MIN_POSITIVE;
        if !(over && fills) {
            continue;
        }
        let inverse = touch.axes.inverse();
        pushing.touch[count] = SimTouch {
            center: center
                .extend(if touch.round { 1.0 } else { 0.0 })
                .to_array(),
            inverse: [
                inverse.x_axis.extend(0.0).to_array(),
                inverse.y_axis.extend(0.0).to_array(),
                inverse.z_axis.extend(0.0).to_array(),
            ],
            velocity: touch.velocity.extend(reach.y).to_array(),
            reach: [from.x, from.y, to.x, to.y],
        };
        (least, most) = (least.min(from), most.max(to));
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let block = window.cell * BLOCK_SIDE as f32;
    let last = Vec2::splat((window.blocks() - 1) as f32);
    let first = (least / block).floor().clamp(Vec2::ZERO, last);
    let end = (most / block).floor().clamp(Vec2::ZERO, last);
    pushing.count = [count as u32, 0, 0, 0];
    pushing.blocks = [
        first.x as u32,
        first.y as u32,
        (end.x - first.x) as u32 + 1,
        (end.y - first.y) as u32 + 1,
    ];
    Some(pushing)
}

pub(crate) fn plan_sim(
    water: Res<Water>,
    obstacles: Res<WaterObstacles>,
    (ground, time, camera): (Res<GroundTiles>, Res<FrameTime>, Res<Camera>),
    (tiles, baking): (Res<TerrainTiles>, Res<GroundBaking>),
    mut state: ResMut<WaterState>,
) {
    let state = &mut *state;
    let seconds = time.presented_seconds;
    let mut moved = false;
    let mut changed = Vec::new();
    for terrain in &mut state.terrains {
        moved |= std::mem::take(&mut terrain.window_moved);
        changed.append(&mut terrain.window_changed);
    }
    if !water.enabled || state.watered.is_empty() {
        state.live.rest(water.enabled);
        return;
    }
    let eye = camera.position;
    let index_of = |entity: Entity| {
        ground
            .tables
            .iter()
            .find(|(terrain, _)| *terrain == entity)
            .map(|&(_, index)| index)
    };
    let mut touching = Vec::new();
    let found = state.terrains.iter().find_map(|terrain| {
        let window = terrain.window.as_ref().filter(|window| window.runs)?;
        let beyond = window.touching(&obstacles.obstacles, eye, &mut touching);
        let ground = index_of(terrain.entity).filter(|_| state.watered.contains(&terrain.entity));
        Some((terrain.entity, window, beyond, ground))
    });
    let blocks = &state.blocks;
    let sweep =
        ground.is_changed() || tiles.is_changed() || baking.is_changed() || blocks.uploads > 0;
    let resident = |read: &WaterRead| {
        let Some((entity, _, _, _)) = found else {
            return false;
        };
        read.ready
            && read.tile.is_some_and(|(_, stamp)| {
                ground
                    .held
                    .binary_search_by_key(&(entity, read.key), |held| (held.terrain, held.key))
                    .is_ok_and(|at| ground.held[at].stamp == stamp)
                    && blocks.current(entity, read.key, read.water.as_ref())
            })
    };
    let live = found.map(|(_, window, beyond, ground)| Live {
        window,
        moved,
        changed,
        sweep,
        touches: &touching,
        beyond,
        ground,
    });
    let sim = &mut state.live;
    sim.prepare(live, &resident, seconds);
    if sim.grounded {
        sim.made = true;
    }
}

pub(crate) struct SimGpu {
    slots: SlotHandles,
    setup: SetupHandles,
    kernels: Kernels,
    set_blocks: ComputePipeline<WaterGpu>,
    set_ring: ComputePipeline<WaterGpu>,
    pass: Pass<WaterGpu>,
    layout: Option<(Layout, u64)>,
    setup_held: Option<((usize, usize), u64)>,
    listed: Option<(u32, u32)>,
    run: Option<(u32, u32, u32, Option<[u32; 2]>)>,
}

impl SimGpu {
    pub(crate) fn install(cx: &mut Install<'_, WaterGpu>) -> Result<Self, Error> {
        use crate::shaders::water::setup;
        let slots = SlotHandles::install(
            cx,
            [
                "water_sim_ground",
                "water_sim_cells",
                "water_sim_depths",
                "water_sim_fluxes",
                "water_sim_inputs",
                "water_sim_blocks",
                "water_sim_steps",
                "water_sim_averages",
                "water_sim_slots",
                "water_sim_first",
                "water_sim_second",
                "water_sim_touches",
            ],
        );
        let setup_handles = SetupHandles::install(cx, ["water_setup", "water_setup_list"]);
        let pass = cx
            .pass("water_sim")
            .label("engine/water-sim")
            .span("water_sim")
            .id();
        let kernels = Kernels::install(cx, pass, &slots)?;
        let bound = crate::water::reached(cx, &[setup::SETUP_BLOCKS], setup_handles.bound(&slots));
        let set_blocks = cx.compute_pipeline(pass, setup::SETUP_BLOCKS, &bound)?;
        let bound = crate::water::reached(cx, &[setup::SETUP_RING], setup_handles.bound(&slots));
        let set_ring = cx.compute_pipeline(pass, setup::SETUP_RING, &bound)?;
        Ok(Self {
            slots,
            setup: setup_handles,
            kernels,
            set_blocks,
            set_ring,
            pass,
            layout: None,
            setup_held: None,
            listed: None,
            run: None,
        })
    }

    pub(crate) fn after(&self) -> Pass<WaterGpu> {
        self.pass
    }

    pub(crate) fn looks(&self) -> engine_render::Buffer<WaterGpu> {
        self.slots.averages
    }

    fn write_chunk(
        &self,
        room: usize,
        (at, water): (u32, &SteadyWater),
        cx: &mut Prepare<'_, WaterGpu>,
    ) {
        let word = |word: u32| SetupHandles::live_at(room) + u64::from(word) * 4;
        let head = chunk_head(&water.live, at);
        cx.write(self.setup.list, word(at), &head);
        if !water.live.cells.is_empty() {
            cx.write(
                self.setup.list,
                word(at + head.len() as u32),
                &water.live.cells,
            );
        }
    }

    pub(crate) fn prepare(&mut self, live: &LiveSim, cx: &mut Prepare<'_, WaterGpu>) {
        let work = &live.work;
        self.listed = None;
        self.run = None;
        let (Some((layout, generation)), Some((rooms, setup_generation))) =
            (work.layout, work.setup)
        else {
            self.layout = None;
            self.setup_held = None;
            self.slots.release(cx);
            self.setup.release(cx);
            return;
        };
        let fresh = self.layout.is_none_or(|(_, held)| held != generation);
        self.slots.hold(layout, fresh, cx);
        self.layout = Some((layout, generation));
        let setup_fresh = fresh
            || self
                .setup_held
                .is_none_or(|held| held != (rooms, setup_generation));
        self.setup.hold(rooms, setup_fresh, cx);
        self.setup_held = Some((rooms, setup_generation));
        let pool = live.window.as_ref().map(|window| &window.live);
        if setup_fresh && let Some(pool) = pool {
            for (range, water) in pool.tiles.values() {
                self.write_chunk(rooms.0, (range.start, water), cx);
            }
        }
        if (setup_fresh || work.square.is_some())
            && let Some(pool) = pool
            && !pool.table.is_empty()
        {
            cx.write(self.setup.list, SetupHandles::live_at(rooms.0), &pool.table);
        }
        if let Some(square) = &work.square {
            if !setup_fresh {
                for (at, water) in &square.chunks {
                    self.write_chunk(rooms.0, (*at, water), cx);
                }
            }
            if !square.sources.is_empty() {
                cx.write(self.slots.inputs, 0, &square.sources);
            }
            if !square.solids.is_empty() {
                cx.write(self.setup.list, 0, &square.solids);
            }
        }
        if let Some(table) = &work.table {
            self.slots.write_table(&layout, 0, table, cx);
        }
        if let Some(listed) = &work.listed {
            cx.write(self.setup.list, listed.offset, &listed.entries);
            cx.write(self.setup.frame, 0, &[listed.frame]);
            self.listed = Some((listed.blocks, listed.ghosts));
        }
        self.slots.write_frames(&layout, cx);
        cx.write(self.slots.squares, 0, &[squares(&[work.slot])]);
        let none = SimTouches::zeroed();
        cx.write(
            self.slots.touches,
            0,
            &[*work.touches.as_ref().unwrap_or(&none)],
        );
        if !work.ready {
            return;
        }
        let touched = work
            .touches
            .as_ref()
            .map(|touches| [touches.blocks[2], touches.blocks[3]]);
        self.run = Some((work.steps, work.sourced, work.blocks, touched));
        cx.run(self.pass);
    }

    pub(crate) fn forget(&mut self) {
        self.layout = None;
        self.setup_held = None;
    }

    pub(crate) fn encode(&mut self, pass: Pass<WaterGpu>, cx: &mut Encode<'_, WaterGpu>) -> bool {
        if pass != self.pass {
            return false;
        }
        let Some((steps, sourced, blocks, touched)) = self.run else {
            return true;
        };
        if let Some((listed, ghosts)) = self.listed {
            if listed > 0 {
                let [groups_x, groups_z] = grid(listed);
                cx.dispatch(&self.set_blocks, Count::Groups([groups_x, groups_z, 1]));
            }
            if ghosts > 0 {
                let groups = [ghosts.div_ceil(WATER_SETUP_RING_GROUP), 1, 1];
                cx.dispatch(&self.set_ring, Count::Groups(groups));
            }
        }
        if steps > 0 {
            self.kernels
                .encode_steps(cx, (steps, sourced, false, true), (blocks, 1, touched));
        }
        self.kernels.encode_drawn(cx, blocks);
        true
    }
}
