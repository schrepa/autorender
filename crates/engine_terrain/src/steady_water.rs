//! Water as it settles: each level-0 tile's rivers run to their steady
//! state by the shallow-water equations ([`crate::shallow_water`]), the
//! scheme the renderer runs around the camera, over a square of ground
//! reaching [`MARGIN`] meters past the tile, from the water the hydrology
//! bake stands there. Lakes hold their level; rivers bring their water
//! where they start (at heads and lakes' outlets), along their course as
//! the ground they drain grows, and where they cross into the square, so
//! what each carries is what its bake says it does; water leaves over the
//! square's edges. It runs until the water it holds and the way it runs
//! stop changing. What a tile keeps of it, a [`SteadyWater`], is a record
//! hosts keep and the cook ships: level-0 tiles' water stands, runs and
//! foams as it says ([`Tile::steadied`]), coarser tiles' runs and foams as
//! its summary does ([`Tile::flowing`]), and the window simulated around
//! the camera starts from it, takes its water at its edges and its rivers'
//! water where they bring it.

use std::sync::Arc;

use glam::{IVec2, Vec2};

use engine_math::{max, min};

use crate::codec::{self, Reader, Writer, record};
use crate::shallow_water::{self, Rates, ShallowWater, Source, Steps};
use crate::tile_water::{WATER_WIDTH_RANGE, pack, unpack};
use crate::{Generator, Layers, Pyramid, TILE_STRIDE, Tile, TileKey};

/// Meters past its tile the square a tile's water settles over reaches on
/// each side, so what the square's edges do to the water stays outside the
/// tile.
pub const MARGIN: f32 = 64.0;
/// Cells inside the square's edge within which rivers count as crossing
/// into it (closer, their water leaves again at once).
const ENTRY: f32 = 2.0;
/// The least water a source brings (cubic meters a second): what a river
/// gains along its course gathers until it brings this much.
const SOURCE_LEAST: f32 = 0.01;
/// Settling: steps between looks at the water and the least and most steps
/// it runs. It has settled once, over [`STILL_LOOKS`] looks in a row, the
/// water it holds changed by no more than [`SETTLED_VOLUME`] of it a look,
/// and its flowing cells' velocities in the tile changed by
/// [`SETTLED_SPEED`] meters a second or less on average, or by no less than
/// [`STILL_SHARE`] of their change [`STILL_LOOKS`] looks before: water
/// running unsteadily (down rough steep rock) varies about its steady flow
/// for good.
const LOOK_EVERY: u32 = 256;
const STEPS: [u32; 2] = [2048, 16_384];
const STILL_LOOKS: usize = 4;
const SETTLED_VOLUME: f64 = 1.0e-3;
const SETTLED_SPEED: f64 = 0.01;
const STILL_SHARE: f64 = 0.75;

/// What settling a tile's water came to: the cubic meters a second its
/// square's rivers bring and its water leaves over its edges and into
/// lakes, as its last step found them; the cubic meters it holds; its
/// cells holding flowing water and those stepping; the steps it ran; and
/// whether its water settled.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SteadyStats {
    pub sources: f32,
    pub outflow: f32,
    pub absorbed: f32,
    pub volume: f32,
    pub flowing: u32,
    pub stepping: u32,
    pub steps: u32,
    pub settled: bool,
}

record!(SteadyStats {
    sources,
    outflow,
    absorbed,
    volume,
    flowing,
    stepping,
    steps,
    settled,
});

impl SteadyStats {
    /// The share of what the rivers bring that does not leave: how far the
    /// water is from balancing.
    pub fn imbalance(&self) -> f32 {
        if self.sources <= 0.0 {
            return 0.0;
        }
        (self.sources - self.outflow - self.absorbed).abs() / self.sources
    }
}

/// Where a tile's rivers bring water to one of its cells (its index within
/// the tile, row by row): the cubic meters a second, and the way the water
/// runs as it comes, as a share of [`shallow_water::SOURCE_SPEED`] along x
/// and z.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SteadySource {
    pub cell: u32,
    pub rate: f32,
    pub way: [f32; 2],
}

record!(SteadySource { cell, rate, way });

/// A level-0 tile's settled water: its key, the meters across its cells
/// (whose middles lie `cell` apart from half a cell past the terrain's first
/// corner), its first cell along x and z and its cells along each, and per
/// cell, row by row, the surface its water stands at (terrain-local; NaN
/// where dry), its velocity along x and z (meters a second; NaN where a
/// lake holds it or it is dry) and its foam (0 none,
/// [`shallow_water::FOAM_MOST`] white all over; NaN where dry); where its
/// rivers bring water; its water seen from afar; and how settling came out.
#[derive(Clone, Debug, PartialEq)]
pub struct SteadyWater {
    pub key: TileKey,
    pub cell: f32,
    pub first: [i64; 2],
    pub cells: [u32; 2],
    surface: Box<[f32]>,
    velocity: [Box<[f32]>; 2],
    foam: Box<[f32]>,
    pub sources: Vec<SteadySource>,
    pub summary: SteadySummary,
    pub stats: SteadyStats,
}

/// A cell of settled water ([`SteadyWater::at`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SteadyCell {
    Dry,
    /// A lake holds it at `level`, with `foam` where rivers run into it.
    Held {
        level: f32,
        foam: f32,
    },
    /// Its water stands at `surface`, runs at `velocity` and carries
    /// `foam`.
    Flowing {
        surface: f32,
        velocity: [f32; 2],
        foam: f32,
    },
}

impl SteadyWater {
    /// Cell `(i, j)` of the terrain's lattice of cells (counted from its
    /// first corner), if this tile holds it.
    pub fn at(&self, i: i64, j: i64) -> Option<SteadyCell> {
        let (x, z) = (i - self.first[0], j - self.first[1]);
        if x < 0 || z < 0 || x >= i64::from(self.cells[0]) || z >= i64::from(self.cells[1]) {
            return None;
        }
        let at = (z * i64::from(self.cells[0]) + x) as usize;
        let surface = self.surface[at];
        if surface.is_nan() {
            return Some(SteadyCell::Dry);
        }
        let velocity = [self.velocity[0][at], self.velocity[1][at]];
        let foam = self.foam[at];
        if velocity[0].is_nan() {
            return Some(SteadyCell::Held {
                level: surface,
                foam,
            });
        }
        Some(SteadyCell::Flowing {
            surface,
            velocity,
            foam,
        })
    }
}

impl codec::Codec for SteadyWater {
    fn write(&self, out: &mut Writer) {
        let stride = self.cells[0] as usize;
        out.put(&self.key)
            .put(&self.cell)
            .put(&self.first)
            .put(&self.cells)
            .put(&self.sources)
            .put(&self.summary)
            .put(&self.stats);
        for values in [
            &self.surface,
            &self.velocity[0],
            &self.velocity[1],
            &self.foam,
        ] {
            codec::put_grid(out, values, stride);
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, codec::CodecError> {
        let key = input.get()?;
        let cell: f32 = input.get()?;
        let first = input.get()?;
        let cells: [u32; 2] = input.get()?;
        let sources: Vec<SteadySource> = input.get()?;
        let summary = input.get()?;
        let stats = input.get()?;
        let stride = cells[0] as usize;
        let count = stride * cells[1] as usize;
        let surface = codec::get_grid(input, stride)?;
        let x = codec::get_grid(input, stride)?;
        let z = codec::get_grid(input, stride)?;
        let foam = codec::get_grid(input, stride)?;
        if [surface.len(), x.len(), z.len(), foam.len()] != [count; 4]
            || !(cell.is_finite() && cell > 0.0)
            || sources.iter().any(|source| source.cell as usize >= count)
        {
            return Err(codec::CodecError("invalid settled water"));
        }
        Ok(Self {
            key,
            cell,
            first,
            cells,
            surface,
            velocity: [x, z],
            foam,
            sources,
            summary,
            stats,
        })
    }
}

/// The cells of the terrain's lattice of cells `cell` meters across whose
/// middles lie in `[low, high)` along an axis (terrain-local meters): the
/// first and how many.
fn lattice(low: f32, high: f32, cell: f32) -> (i64, u32) {
    let first = (low / cell - 0.5).ceil() as i64;
    let end = (high / cell - 0.5).ceil() as i64;
    (first, (end - first).max(0) as u32)
}

/// The square tile `key`'s water settles over, of the lattice of cells
/// `cell` meters across: its first cell along x and z and its cells along a
/// side, the tile's cells and [`MARGIN`] around them, within the terrain.
fn square(pyramid: &Pyramid, key: TileKey, cell: f32) -> ([i64; 2], usize) {
    let origin = pyramid.tile_origin(key);
    let size = pyramid.tile_size(0);
    let extent = pyramid.extent();
    let (x, count) = lattice(origin.x, origin.x + size, cell);
    let (z, _) = lattice(origin.y, origin.y + size, cell);
    let (_, along_x) = lattice(0.0, extent.x, cell);
    let (_, along_z) = lattice(0.0, extent.y, cell);
    let margin = (MARGIN / cell).ceil() as i64;
    let side = (i64::from(count) + 2 * margin)
        .min(i64::from(along_x))
        .min(i64::from(along_z));
    let place = |first: i64, along: u32| (first - margin).clamp(0, i64::from(along) - side);
    ([place(x, along_x), place(z, along_z)], side as usize)
}

/// The terrain-local rectangle the square tile `key`'s water settles over
/// in cells `cell` meters across covers, its corners included: what edits
/// reaching its ground reach.
pub fn steady_water_bounds(pyramid: &Pyramid, key: TileKey, cell: f32) -> (Vec2, Vec2) {
    let (first, side) = square(pyramid, key, cell);
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    (low, low + Vec2::splat(side as f32 * cell))
}

/// The level-0 tiles of `pyramid` the square from cell `first` `side`
/// cells `cell` meters across reaches, its corners included.
fn tiles_reached(pyramid: &Pyramid, first: [i64; 2], side: usize, cell: f32) -> Vec<TileKey> {
    let low = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let high = low + Vec2::splat(side as f32 * cell);
    let size = pyramid.tile_size(0);
    let [columns, rows] = pyramid.tiles(0);
    let range = |low: f32, high: f32, count: u32| {
        let first = max((low / size).floor(), 0.0) as u32;
        let last = min((high / size).floor(), count as f32 - 1.0) as u32;
        first..=last
    };
    let mut keys = Vec::new();
    for z in range(low.y, high.y, rows) {
        for x in range(low.x, high.x, columns) {
            keys.push(TileKey::new(0, x as u16, z as u16));
        }
    }
    keys
}

/// Level-0 tiles generated from a stack, by key.
struct Ground {
    pyramid: Pyramid,
    tiles: Vec<(TileKey, Tile)>,
}

impl Ground {
    /// The tile holding terrain-local `point`.
    fn tile(&self, point: Vec2) -> Option<(&Tile, Vec2)> {
        let key = self.pyramid.tile_at(0, point)?;
        let (_, tile) = self.tiles.iter().find(|(held, _)| *held == key)?;
        Some((tile, point - self.pyramid.tile_origin(key)))
    }

    fn height(&self, point: Vec2) -> f32 {
        self.tile(point)
            .and_then(|(tile, local)| tile.sample(local))
            .map_or(0.0, |sample| sample.height)
    }
}

/// The water a square starts settling from: its ground at its cells'
/// corners, each cell's water as the bake stands and runs it (its surface,
/// or its ground where dry, and its flow along x and z) and the level a
/// lake holds it at (NaN where none does), and where its rivers bring
/// water.
struct Start {
    corners: Vec<f32>,
    water: Vec<[f32; 3]>,
    held: Vec<f32>,
    sources: Vec<Source>,
}

/// The water of the square from cell `first`, `side` cells `cell` meters
/// across, over `ground`, as `layers`' bake stands it there.
fn start(layers: &Layers, ground: &Ground, (first, side): ([i64; 2], usize), cell: f32) -> Start {
    let n = side;
    let origin = Vec2::new(first[0] as f32, first[1] as f32) * cell;
    let mut corners = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            corners.push(ground.height(origin + Vec2::new(i as f32, j as f32) * cell));
        }
    }
    let bed = |at: usize| {
        let (i, j) = (at % n, at / n);
        0.25 * (corners[j * (n + 1) + i]
            + corners[j * (n + 1) + i + 1]
            + corners[(j + 1) * (n + 1) + i]
            + corners[(j + 1) * (n + 1) + i + 1])
    };
    let mut water = Vec::with_capacity(n * n);
    let mut held = Vec::with_capacity(n * n);
    for at in 0..n * n {
        let (i, j) = (at % n, at / n);
        let floor = bed(at);
        let middle = origin + (Vec2::new(i as f32, j as f32) + 0.5) * cell;
        let found = ground
            .tile(middle)
            .and_then(|(tile, local)| {
                let sample = tile.sample(local)?;
                (!sample.hole).then(|| tile.water_at(local)).flatten()
            })
            .filter(|point| point.surface > floor);
        let (cell_water, lake) = match found {
            Some(point) if point.lake => ([point.surface, 0.0, 0.0], point.surface),
            Some(point) => {
                let depth = point.surface - floor;
                (
                    [point.surface, depth * point.flow.x, depth * point.flow.y],
                    f32::NAN,
                )
            }
            None => ([floor, 0.0, 0.0], f32::NAN),
        };
        water.push(cell_water);
        held.push(lake);
    }
    let wet: Vec<bool> = (0..n * n)
        .map(|at| held[at].is_nan() && water[at][0] > bed(at))
        .collect();
    let sources = sources(layers, (origin, n, cell), (&held, &wet));
    Start {
        corners,
        water,
        held,
        sources,
    }
}

/// Where the baked rivers bring water to the square from `origin`
/// (terrain-local) over `n` cells `cell` meters across: at each point of a
/// river's course within it, what the river carries there less what it
/// carried at the point before within it (all it carries where it starts
/// at a head or a lake's outlet, or crosses into the square; nothing more
/// where it starts at a confluence, whose rivers bring it), gathered along
/// the course until [`SOURCE_LEAST`] or more; spread over the cells within
/// half its width no lake holds (`held`) that its baked water covers
/// (`wet`), and where it covers none there, further down the course: its
/// water comes where its bake stands water. Each cell's water is summed,
/// cells in order.
fn sources(
    layers: &Layers,
    (origin, n, cell): (Vec2, usize, f32),
    (held, wet): (&[f32], &[bool]),
) -> Vec<Source> {
    let Some(hydrology) = layers.hydrology() else {
        return Vec::new();
    };
    let size = n as f32 * cell;
    let margin = ENTRY * cell;
    let inside = |point: [f32; 2]| {
        let point = Vec2::from(point);
        point.x > origin.x + margin
            && point.y > origin.y + margin
            && point.x < origin.x + size - margin
            && point.y < origin.y + size - margin
    };
    let courses: Vec<_> = hydrology.rivers().courses().collect();
    let key = |point: [f32; 2]| [point[0].to_bits(), point[1].to_bits()];
    let mut ends: Vec<[u32; 2]> = courses
        .iter()
        .filter_map(|course| course.points().last().copied())
        .map(key)
        .collect();
    ends.sort_unstable();
    let last = n as i32 - 1;
    // Each cell's water and the way it brings it, weighted by its water.
    let mut brought: Vec<(u32, f32, [f32; 2])> = Vec::new();
    for course in &courses {
        let points = course.points();
        let mut gathered = 0.0_f32;
        for k in 0..points.len() {
            if !inside(points[k]) {
                gathered = 0.0;
                continue;
            }
            let before = if k == 0 {
                if ends.binary_search(&key(points[0])).is_ok() {
                    course.discharge(0)
                } else {
                    0.0
                }
            } else if inside(points[k - 1]) {
                course.discharge(k - 1)
            } else {
                0.0
            };
            gathered += max(course.discharge(k) - before, 0.0);
            if gathered < SOURCE_LEAST {
                continue;
            }
            let at = Vec2::from(points[k]);
            let reach = max(course.half(k), cell);
            let low = ((at - origin) / cell - reach / cell)
                .floor()
                .as_ivec2()
                .max(IVec2::ZERO);
            let high = ((at - origin) / cell + reach / cell)
                .ceil()
                .as_ivec2()
                .min(IVec2::splat(last));
            let mut covered = Vec::new();
            for j in low.y..=high.y {
                for i in low.x..=high.x {
                    let middle = origin + (Vec2::new(i as f32, j as f32) + 0.5) * cell;
                    let index = (j * n as i32 + i) as usize;
                    if middle.distance(at) <= reach && held[index].is_nan() && wet[index] {
                        covered.push(index as u32);
                    }
                }
            }
            // Where its baked water covers no cell there, what it gathered
            // goes on down the course.
            if covered.is_empty() {
                continue;
            }
            let run = if k + 1 < points.len() {
                Vec2::from(points[k + 1]) - at
            } else if k > 0 {
                at - Vec2::from(points[k - 1])
            } else {
                Vec2::X
            };
            let way = run.normalize_or(Vec2::X);
            let share = gathered / covered.len() as f32;
            brought.extend(
                covered
                    .into_iter()
                    .map(|index| (index, share, (way * share).to_array())),
            );
            gathered = 0.0;
        }
    }
    brought.sort_by_key(|&(cell, _, _)| cell);
    let mut sources: Vec<Source> = Vec::new();
    let mut weighted = [0.0_f32; 2];
    for (cell, rate, way) in brought {
        match sources.last_mut() {
            Some(last) if last.cell == cell => last.rate += rate,
            _ => {
                if let Some(last) = sources.last_mut() {
                    last.way = [weighted[0] / last.rate, weighted[1] / last.rate];
                }
                weighted = [0.0; 2];
                sources.push(Source {
                    cell,
                    rate,
                    way: [0.0; 2],
                });
            }
        }
        weighted = [weighted[0] + way[0], weighted[1] + way[1]];
    }
    if let Some(last) = sources.last_mut() {
        last.way = [weighted[0] / last.rate, weighted[1] / last.rate];
    }
    sources
}

/// Settling a square's water: steps, and at the last look its water and
/// the velocities of the cells its tile keeps, with what the looks before
/// found: whether its water held still, and how much its velocities
/// changed.
struct Settling {
    water: ShallowWater,
    steps: u32,
    volume: f64,
    kept: Vec<usize>,
    speeds: Vec<[f32; 2]>,
    looks: Vec<(bool, f64)>,
    last: Option<SteadyStats>,
}

impl Settling {
    fn new(water: ShallowWater, kept: Vec<usize>) -> Self {
        let mut settling = Self {
            volume: water.volume(),
            water,
            steps: 0,
            kept,
            speeds: Vec::new(),
            looks: Vec::new(),
            last: None,
        };
        settling.speeds = settling.velocities();
        settling
    }

    /// The velocities of the cells kept, zero where a lake holds them or
    /// they are dry.
    fn velocities(&self) -> Vec<[f32; 2]> {
        self.kept
            .iter()
            .map(|&at| self.water.flowing(at).map_or([0.0; 2], |[_, x, z]| [x, z]))
            .collect()
    }

    /// Runs up to `budget` cell steps; whether the water settled or ran its
    /// most steps.
    fn run(&mut self, budget: &mut u64) -> bool {
        loop {
            if *budget == 0 {
                return false;
            }
            *budget = budget.saturating_sub(self.water.listed().len().max(1) as u64);
            self.water.step(Steps::Local);
            self.steps += 1;
            if !self.steps.is_multiple_of(LOOK_EVERY) {
                continue;
            }
            let volume = self.water.volume();
            let speeds = self.velocities();
            // How much the flowing cells' velocities changed, on average.
            let (mut change, mut moving) = (0.0_f64, 0_u32);
            for (now, before) in speeds.iter().zip(&self.speeds) {
                if *now != [0.0; 2] || *before != [0.0; 2] {
                    let (x, z) = (now[0] - before[0], now[1] - before[1]);
                    change += f64::from((x * x + z * z).sqrt());
                    moving += 1;
                }
            }
            let change = change / f64::from(moving.max(1));
            let still = (volume - self.volume).abs() <= SETTLED_VOLUME * larger(volume, 1.0);
            self.looks.push((still, change));
            self.volume = volume;
            self.speeds = speeds;
            let settled = self.steps >= STEPS[0] && self.settled();
            self.last = Some(self.stats(settled));
            if settled || self.steps >= STEPS[1] {
                return true;
            }
        }
    }

    /// Whether the looks so far find the water settled.
    fn settled(&self) -> bool {
        let looks = &self.looks;
        if looks.len() <= STILL_LOOKS {
            return false;
        }
        let recent = &looks[looks.len() - STILL_LOOKS..];
        let (_, change) = looks[looks.len() - 1];
        let (_, before) = looks[looks.len() - 1 - STILL_LOOKS];
        recent.iter().all(|&(still, _)| still)
            && (change <= SETTLED_SPEED || change >= STILL_SHARE * before)
    }

    fn stats(&self, settled: bool) -> SteadyStats {
        let Rates {
            sources,
            outflow,
            absorbed,
        } = self.water.rates();
        let flowing = self
            .water
            .listed()
            .iter()
            .filter(|&&at| self.water.flowing(at as usize).is_some())
            .count();
        SteadyStats {
            sources: sources as f32,
            outflow: outflow as f32,
            absorbed: absorbed as f32,
            volume: self.volume as f32,
            flowing: flowing as u32,
            stepping: self.water.listed().len() as u32,
            steps: self.steps,
            settled,
        }
    }
}

fn larger(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// Settling tile `key`'s water: generating the level-0 tiles its square
/// reaches, then the water running, in steps a job's budget sizes.
pub struct SteadySolve {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    key: TileKey,
    cell: f32,
    // The square it settles over: its first cell and side.
    first: [i64; 2],
    side: usize,
    pending: Vec<TileKey>,
    generating: Option<Generator>,
    tiles: Vec<(TileKey, Tile)>,
    settling: Option<Settling>,
}

impl SteadySolve {
    /// Tile `key`'s water over `layers` (with their bakes) and `pyramid`,
    /// in cells `cell` meters across.
    pub fn new(layers: Arc<Layers>, pyramid: Pyramid, key: TileKey, cell: f32) -> Self {
        let (first, side) = square(&pyramid, key, cell);
        let pending = tiles_reached(&pyramid, first, side, cell);
        Self {
            layers,
            pyramid,
            key,
            cell,
            first,
            side,
            pending,
            generating: None,
            tiles: Vec::new(),
            settling: None,
        }
    }

    /// The cells of its square the tile keeps: the tile's first cell along
    /// x and z, its cells along each, and each kept cell's index in the
    /// square, row by row.
    fn kept_cells(&self) -> ([i64; 2], [u32; 2], Vec<usize>) {
        let origin = self.pyramid.tile_origin(self.key);
        let size = self.pyramid.tile_size(0);
        let (x, columns) = lattice(origin.x, origin.x + size, self.cell);
        let (z, rows) = lattice(origin.y, origin.y + size, self.cell);
        let n = self.side;
        let mut kept = Vec::with_capacity(columns as usize * rows as usize);
        for row in 0..rows as usize {
            for column in 0..columns as usize {
                let (i, j) = (
                    (x - self.first[0]) as usize + column,
                    (z - self.first[1]) as usize + row,
                );
                kept.push(j * n + i);
            }
        }
        ([x, z], [columns, rows], kept)
    }

    /// Works for about `budget` units (a cell's step, or a sample generated,
    /// each): the tile's settled water once done; an error where its ground
    /// does not generate.
    pub fn advance(&mut self, budget: &mut u64) -> Result<Option<SteadyWater>, String> {
        while self.settling.is_none() {
            if *budget == 0 {
                return Ok(None);
            }
            if let Some(generator) = &mut self.generating {
                let Some(tile) = generator.advance(budget) else {
                    if generator.failed() {
                        return Err("the ground's heights are invalid".into());
                    }
                    return Ok(None);
                };
                self.tiles.push((tile.key(), tile));
                self.generating = None;
                continue;
            }
            if let Some(key) = self.pending.pop() {
                self.generating = Some(Generator::new(self.layers.clone(), &self.pyramid, key));
                continue;
            }
            let ground = Ground {
                pyramid: self.pyramid,
                tiles: std::mem::take(&mut self.tiles),
            };
            let start = start(&self.layers, &ground, (self.first, self.side), self.cell);
            let foam = vec![0.0; self.side * self.side];
            let water = ShallowWater::new(
                (self.side, self.cell),
                &start.corners,
                (&start.water, &start.held, &foam),
                start.sources,
            )
            .ok_or("the square's water does not fit it")?;
            let (_, _, kept) = self.kept_cells();
            self.settling = Some(Settling::new(water, kept));
        }
        let Some(settling) = self.settling.as_mut() else {
            return Ok(None);
        };
        if !settling.run(budget) {
            return Ok(None);
        }
        let Some(settling) = self.settling.as_ref() else {
            return Ok(None);
        };
        Ok(Some(self.kept(settling)))
    }

    /// What the tile keeps of its square's `settling`, settled: its own
    /// cells, the water its rivers bring to them, and its summary.
    fn kept(&self, settling: &Settling) -> SteadyWater {
        let water = &settling.water;
        let (first, cells, kept) = self.kept_cells();
        let count = kept.len();
        let mut surface = vec![f32::NAN; count];
        let mut velocity = [vec![f32::NAN; count], vec![f32::NAN; count]];
        let mut foam = vec![f32::NAN; count];
        for (out, &at) in kept.iter().enumerate() {
            let ([level, _, _], held, cell_foam) = water.water(at);
            if let Some(held) = held {
                surface[out] = held;
                foam[out] = cell_foam;
            } else if let Some([_, x, z]) = water.flowing(at) {
                surface[out] = level;
                velocity[0][out] = x;
                velocity[1][out] = z;
                foam[out] = cell_foam;
            }
        }
        let mut sources: Vec<SteadySource> = Vec::new();
        for source in water.sources() {
            if let Ok(out) = kept.binary_search(&(source.cell as usize)) {
                sources.push(SteadySource {
                    cell: out as u32,
                    rate: source.rate,
                    way: source.way,
                });
            }
        }
        let summary = SteadySummary::of(cells, (&velocity, &foam));
        let [vx, vz] = velocity;
        SteadyWater {
            key: self.key,
            cell: self.cell,
            first,
            cells,
            surface: surface.into(),
            velocity: [vx.into(), vz.into()],
            foam: foam.into(),
            sources,
            summary,
            stats: settling.last.unwrap_or_default(),
        }
    }
}

/// Squares along each side of a tile its settled water's summary holds:
/// what coarser levels' water runs and foams by.
pub const SUMMARY_SIDE: u32 = 32;

/// A level-0 tile's settled water seen from afar ([`SteadyWater::summary`]):
/// over [`SUMMARY_SIDE`] squares along each side of the tile, row by row,
/// each square's flowing cells' mean velocity along x and z (meters a
/// second; NaN where none flows) and its wet cells' mean foam (NaN where
/// none is wet).
#[derive(Clone, Debug, PartialEq)]
pub struct SteadySummary {
    velocity: [Box<[f32]>; 2],
    foam: Box<[f32]>,
}

/// A square of a [`SteadySummary`]: its mean velocity where water flows in
/// it, and its mean foam where water is wet in it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SteadySquare {
    pub velocity: Option<[f32; 2]>,
    pub foam: Option<f32>,
}

impl SteadySummary {
    /// The summary of a tile's `cells` (along x and z) holding `velocity`
    /// and `foam` ([`SteadyWater`]'s, row by row).
    fn of(cells: [u32; 2], (velocity, foam): (&[Vec<f32>; 2], &[f32])) -> Self {
        let side = SUMMARY_SIDE as usize;
        let mut sums = vec![[0.0_f32; 5]; side * side];
        let (columns, rows) = (cells[0] as usize, cells[1] as usize);
        for row in 0..rows {
            for column in 0..columns {
                let at = row * columns + column;
                let square = (row * side / rows) * side + column * side / columns;
                let sum = &mut sums[square];
                if !foam[at].is_nan() {
                    sum[3] += foam[at];
                    sum[4] += 1.0;
                }
                if !velocity[0][at].is_nan() {
                    sum[0] += velocity[0][at];
                    sum[1] += velocity[1][at];
                    sum[2] += 1.0;
                }
            }
        }
        let mean = |total: f32, count: f32| {
            if count > 0.0 { total / count } else { f32::NAN }
        };
        Self {
            velocity: [
                sums.iter().map(|sum| mean(sum[0], sum[2])).collect(),
                sums.iter().map(|sum| mean(sum[1], sum[2])).collect(),
            ],
            foam: sums.iter().map(|sum| mean(sum[3], sum[4])).collect(),
        }
    }

    /// Square `(x, z)` of the summary, if inside.
    pub fn square(&self, x: u32, z: u32) -> Option<SteadySquare> {
        if x >= SUMMARY_SIDE || z >= SUMMARY_SIDE {
            return None;
        }
        let at = (z * SUMMARY_SIDE + x) as usize;
        let velocity = [self.velocity[0][at], self.velocity[1][at]];
        Some(SteadySquare {
            velocity: (!velocity[0].is_nan()).then_some(velocity),
            foam: (!self.foam[at].is_nan()).then_some(self.foam[at]),
        })
    }
}

impl codec::Codec for SteadySummary {
    fn write(&self, out: &mut Writer) {
        let stride = SUMMARY_SIDE as usize;
        for values in [&self.velocity[0], &self.velocity[1], &self.foam] {
            codec::put_grid(out, values, stride);
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, codec::CodecError> {
        let stride = SUMMARY_SIDE as usize;
        let x = codec::get_grid(input, stride)?;
        let z = codec::get_grid(input, stride)?;
        let foam = codec::get_grid(input, stride)?;
        if [x.len(), z.len(), foam.len()] != [stride * stride; 3] {
            return Err(codec::CodecError("invalid settled water summary"));
        }
        Ok(Self {
            velocity: [x, z],
            foam,
        })
    }
}

/// Meters of half width the water of a sample its own water left dry packs
/// as: a river's, about the width foam reaches out from its shores by.
const SETTLED_HALF_WIDTH: f32 = 8.0;

/// The values a cell or square blends into a sample by: its weight, and
/// what it holds.
struct Blend {
    known: bool,
    weight: f32,
    held: f32,
    level: f32,
    foam: f32,
    moving: f32,
    flow: Vec2,
}

impl Blend {
    const NONE: Self = Self {
        known: false,
        weight: 0.0,
        held: 0.0,
        level: 0.0,
        foam: 0.0,
        moving: 0.0,
        flow: Vec2::ZERO,
    };

    /// The flow it blends to, still where lakes hold most of it.
    fn flow(&self) -> Vec2 {
        if self.held * 2.0 > self.weight || self.moving <= 0.0 {
            Vec2::ZERO
        } else {
            self.flow / self.moving
        }
    }
}

/// The four points of a lattice `spacing` apart whose first lies half a
/// step past the terrain's first corner around terrain-local `point`: each
/// one's indices and the share it blends in by.
fn around(point: Vec2, spacing: f32) -> [(i64, i64, f32); 4] {
    let (u, v) = (point.x / spacing - 0.5, point.y / spacing - 0.5);
    let (i0, j0) = (u.floor() as i64, v.floor() as i64);
    let (fu, fv) = (u - i0 as f32, v - j0 as f32);
    [
        (i0, j0, (1.0 - fu) * (1.0 - fv)),
        (i0 + 1, j0, fu * (1.0 - fv)),
        (i0, j0 + 1, (1.0 - fu) * fv),
        (i0 + 1, j0 + 1, fu * fv),
    ]
}

impl Tile {
    /// This level-0 tile of `pyramid`, its water as settled water stands it
    /// at its samples: from the settled cells `cell` meters across `cells`
    /// gives (by their place on the terrain's lattice of cells; `None` where
    /// no record holds them), each sample takes the surface, velocity and
    /// foam of the cells around it holding water, blended by their
    /// nearness, held still where most are lakes'; dry where none holds
    /// any, and as its own water stood where no record holds the cells
    /// around it. The width its water packs as and whether a fall's sheet
    /// runs over it stay its own.
    #[must_use]
    pub fn steadied(
        &self,
        pyramid: &Pyramid,
        cell: f32,
        cells: &dyn Fn(i64, i64) -> Option<SteadyCell>,
    ) -> Self {
        let stride = TILE_STRIDE as usize;
        let spacing = self.spacing();
        let origin = pyramid.tile_origin(self.key());
        let own = self.water();
        let mut surface = vec![f32::NAN; stride * stride];
        let mut info = vec![0_u32; stride * stride];
        for z in 0..stride {
            for x in 0..stride {
                let at = z * stride + x;
                let (own_surface, own_info) = own.map_or((f32::NAN, 0), |water| {
                    (water.surfaces()[at], water.info()[at])
                });
                let point = origin + Vec2::new(x as f32 - 1.0, z as f32 - 1.0) * spacing;
                let mut blend = Blend::NONE;
                for (i, j, share) in around(point, cell) {
                    let Some(found) = cells(i, j) else {
                        continue;
                    };
                    blend.known = true;
                    match found {
                        SteadyCell::Dry => {}
                        SteadyCell::Held { level, foam } => {
                            blend.weight += share;
                            blend.held += share;
                            blend.level += level * share;
                            blend.foam += foam * share;
                        }
                        SteadyCell::Flowing {
                            surface: water,
                            velocity,
                            foam,
                        } => {
                            blend.weight += share;
                            blend.level += water * share;
                            blend.foam += foam * share;
                            blend.moving += share;
                            blend.flow += Vec2::from(velocity) * share;
                        }
                    }
                }
                if !blend.known {
                    surface[at] = own_surface;
                    info[at] = own_info;
                    continue;
                }
                if blend.weight <= 0.0 {
                    continue;
                }
                let (_, _, _, own_width) = unpack(own_info);
                let sheeted = own_surface.is_finite() && own_info & (1 << 23) != 0;
                let width = if own_surface.is_finite() {
                    own_width
                } else {
                    SETTLED_HALF_WIDTH
                };
                surface[at] = blend.level / blend.weight;
                info[at] = pack(
                    blend.flow(),
                    blend.foam / blend.weight / shallow_water::FOAM_MOST,
                    sheeted,
                    blend.held * 2.0 > blend.weight,
                    min(width, WATER_WIDTH_RANGE),
                );
            }
        }
        self.clone()
            .with_water(surface.into_boxed_slice(), info.into_boxed_slice())
    }

    /// This tile of `pyramid`, of a coarser level than 0, its water running
    /// and foaming as settled water's summaries say ([`SteadySummary`]):
    /// each sample its own water reaches takes the mean velocity and foam of
    /// the summaries' squares around it, `squares` giving each by its place
    /// on the terrain's lattice of squares (`None` where no summary holds
    /// it), blended by their nearness; still and clear where none of them
    /// holds water, and as its own water ran where no summary holds the
    /// squares around it. Its water's surface, lakes, width and falls' sheets
    /// stay its own.
    #[must_use]
    pub fn flowing(
        &self,
        pyramid: &Pyramid,
        squares: &dyn Fn(i64, i64) -> Option<SteadySquare>,
    ) -> Self {
        let Some(own) = self.water() else {
            return self.clone();
        };
        let stride = TILE_STRIDE as usize;
        let spacing = self.spacing();
        let origin = pyramid.tile_origin(self.key());
        let square = pyramid.tile_size(0) / SUMMARY_SIDE as f32;
        let surface: Box<[f32]> = own.surfaces().into();
        let mut info: Box<[u32]> = own.info().into();
        for z in 0..stride {
            for x in 0..stride {
                let at = z * stride + x;
                if !surface[at].is_finite() {
                    continue;
                }
                let point = origin + Vec2::new(x as f32 - 1.0, z as f32 - 1.0) * spacing;
                let mut blend = Blend::NONE;
                for (i, j, share) in around(point, square) {
                    let Some(found) = squares(i, j) else {
                        continue;
                    };
                    blend.known = true;
                    if let Some(foam) = found.foam {
                        blend.weight += share;
                        blend.foam += foam * share;
                    }
                    if let Some(velocity) = found.velocity {
                        blend.moving += share;
                        blend.flow += Vec2::from(velocity) * share;
                    }
                }
                if !blend.known {
                    continue;
                }
                let (_, _, lake, width) = unpack(info[at]);
                let sheeted = info[at] & (1 << 23) != 0;
                let foam = if blend.weight > 0.0 {
                    blend.foam / blend.weight
                } else {
                    0.0
                };
                let flow = if lake || blend.moving <= 0.0 {
                    Vec2::ZERO
                } else {
                    blend.flow / blend.moving
                };
                info[at] = pack(flow, foam / shallow_water::FOAM_MOST, sheeted, lake, width);
            }
        }
        self.clone().with_water(surface, info)
    }
}

/// The level-0 tiles of `pyramid` holding water `layers`' bake stands
/// within a sample of them, lakes or rivers, in key order.
pub fn watered_tiles(layers: &Layers, pyramid: &Pyramid) -> Vec<TileKey> {
    let Some(hydrology) = layers.hydrology() else {
        return Vec::new();
    };
    let size = pyramid.tile_size(0);
    let mut keys: Vec<TileKey> = pyramid
        .level_tiles(0)
        .filter(|&key| {
            let low = pyramid.tile_origin(key);
            hydrology.water_reaches(low, low + Vec2::splat(size))
        })
        .collect();
    keys.sort_unstable();
    keys
}
