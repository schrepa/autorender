//! Water as it settles: each level-0 tile's rivers run to their steady
//! state by the shallow-water equations ([`crate::shallow_water`]), which
//! the renderer runs on the GPU, around the camera and to settle tiles,
//! over a square of ground reaching [`MARGIN`] meters past the tile, from
//! the water the hydrology bake stands there. Lakes hold their level;
//! rivers bring their water where they start (at heads and lakes'
//! outlets), along their course as the ground they drain grows, and where
//! they cross into the square, so what each carries is what its bake says
//! it does; water leaves over the square's edges and none comes in over
//! them. A [`WaterStart`] job generates the square's ground and prepares
//! what settling starts from ([`SettleStart`]): the square at its cells'
//! size and at coarser ones, down to [`COARSEST`] meters across, each
//! level settling from the one before it ([`SettleStart::prolong`]), so
//! the water spreads and fills at the coarse levels' cost and the finest
//! settles the detail; a tile whose earlier record its host holds settles
//! at its cells' size from that water. Settled, the finest level's water,
//! averaged over its last steps, becomes what the tile keeps of it
//! ([`SettleStart::finish`]), a [`SteadyWater`], a record hosts keep and
//! the cook ships: level-0 tiles' water stands, runs and foams as it says
//! ([`Tile::steadied`]), coarser tiles' runs and foams as its summary does
//! ([`Tile::flowing`]), and the window simulated around the camera starts
//! from it, takes its water at its edges and its rivers' water where they
//! bring it.

use std::sync::Arc;

use glam::{IVec2, Vec2};

use engine_math::{max, min};

use crate::codec::{self, Reader, Writer, record};
use crate::shallow_water::{self, DRY, velocity};
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
/// Meters across the coarsest cells a square settles at first, and the
/// fewest cells along a side a level holds: coarser levels halve the cells'
/// size until either would be passed.
pub const COARSEST: f32 = 8.0;
const FEWEST: u32 = 16;

/// What settling a tile's water came to: the cubic meters a second its
/// square's rivers bring; the cubic meters its square holds, averaged over
/// its last steps; its cells holding flowing water; the steps its levels
/// ran, all told, and its levels; and whether its water settled before its
/// steps ran out (water running unsteadily, down rough steep rock, settles
/// once its averages stop changing).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SteadyStats {
    pub sources: f32,
    pub volume: f32,
    pub flowing: u32,
    pub steps: u32,
    pub levels: u32,
    pub settled: bool,
}

record!(SteadyStats {
    sources,
    volume,
    flowing,
    steps,
    levels,
    settled,
});

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
    sources: Vec<SteadySource>,
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
) -> Vec<SteadySource> {
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
    merged(brought)
}

/// Sources from the water brought to cells, `(cell, rate, way × rate)` in
/// cell order: each cell's rates summed, the way it brings them their
/// rate-weighted mean.
fn merged(brought: Vec<(u32, f32, [f32; 2])>) -> Vec<SteadySource> {
    let mut sources: Vec<SteadySource> = Vec::new();
    let mut weighted = [0.0_f32; 2];
    for (cell, rate, way) in brought {
        match sources.last_mut() {
            Some(last) if last.cell == cell => last.rate += rate,
            _ => {
                if let Some(last) = sources.last_mut() {
                    last.way = [weighted[0] / last.rate, weighted[1] / last.rate];
                }
                weighted = [0.0; 2];
                sources.push(SteadySource {
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

/// A level of a square settling ([`SettleStart`]): the meters across its
/// cells and its cells along a side; the ground at its cells' corners
/// (`side + 1` a row, rows along z); each cell's water to start from (its
/// surface, or its ground where dry, and its flow along x and z in square
/// meters a second) and the level a lake holds it at (NaN where none
/// does), row by row; and where its rivers bring water. Its heights stand
/// above its square's datum.
#[derive(Clone, Debug, PartialEq)]
pub struct SettleLevel {
    pub cell: f32,
    pub side: u32,
    pub corners: Vec<f32>,
    pub water: Vec<[f32; 3]>,
    pub held: Vec<f32>,
    pub sources: Vec<SteadySource>,
}

record!(SettleLevel {
    cell,
    side,
    corners,
    water,
    held,
    sources,
});

impl SettleLevel {
    /// Cell `at`'s ground: the mean of its corners'.
    pub fn bed(&self, at: usize) -> f32 {
        let n = self.side as usize;
        let (i, j) = (at % n, at / n);
        let corner = |x: usize, z: usize| self.corners[z * (n + 1) + x];
        0.25 * (corner(i, j) + corner(i + 1, j) + corner(i, j + 1) + corner(i + 1, j + 1))
    }

    /// Whether its arrays hold its cells: its corners, its water and its
    /// lakes' levels, and sources at cells within it.
    fn valid(&self) -> bool {
        let n = self.side as usize;
        self.side > 0
            && self.cell.is_finite()
            && self.cell > 0.0
            && self.corners.len() == (n + 1) * (n + 1)
            && self.water.len() == n * n
            && self.held.len() == n * n
            && self.sources.iter().all(|source| (source.cell as usize) < n * n)
    }

    /// This level at cells twice the size: its ground at every other
    /// corner; each cell held by a lake where most of the cells it covers
    /// are (at their levels' mean), else holding their mean depth over its
    /// own ground and their mean flow; its rivers' water brought to the
    /// cells covering those they brought it to.
    fn coarser(&self) -> Self {
        let n = self.side as usize;
        let side = n.div_ceil(2);
        let mut corners = Vec::with_capacity((side + 1) * (side + 1));
        for j in 0..=side {
            for i in 0..=side {
                corners.push(self.corners[(2 * j).min(n) * (n + 1) + (2 * i).min(n)]);
            }
        }
        let mut level = Self {
            cell: 2.0 * self.cell,
            side: side as u32,
            corners,
            water: Vec::with_capacity(side * side),
            held: Vec::with_capacity(side * side),
            sources: Vec::new(),
        };
        for at in 0..side * side {
            let (i, j) = (at % side, at / side);
            let (mut cells, mut lakes) = (0_u32, 0_u32);
            let (mut lake_level, mut depth) = (0.0_f32, 0.0_f32);
            let mut flow = [0.0_f32; 2];
            for (x, z) in [
                (2 * i, 2 * j),
                (2 * i + 1, 2 * j),
                (2 * i, 2 * j + 1),
                (2 * i + 1, 2 * j + 1),
            ] {
                if x >= n || z >= n {
                    continue;
                }
                let fine = z * n + x;
                cells += 1;
                if !self.held[fine].is_nan() {
                    lakes += 1;
                    lake_level += self.held[fine];
                    continue;
                }
                let [surface, along_x, along_z] = self.water[fine];
                depth += max(surface - self.bed(fine), 0.0);
                flow = [flow[0] + along_x, flow[1] + along_z];
            }
            if lakes > 0 && 2 * lakes >= cells {
                let held = lake_level / lakes as f32;
                level.water.push([held, 0.0, 0.0]);
                level.held.push(held);
                continue;
            }
            let share = 1.0 / cells.max(1) as f32;
            let bed = level.bed(at);
            let mean = depth * share;
            let surface = if mean > DRY { bed + mean } else { bed };
            level
                .water
                .push([surface, flow[0] * share, flow[1] * share]);
            level.held.push(f32::NAN);
        }
        let mut brought: Vec<(u32, f32, [f32; 2])> = self
            .sources
            .iter()
            .map(|source| {
                let (i, j) = (source.cell as usize % n, source.cell as usize / n);
                let cell = ((j / 2) * side + i / 2) as u32;
                let way = [source.way[0] * source.rate, source.way[1] * source.rate];
                (cell, source.rate, way)
            })
            .collect();
        brought.sort_by_key(|&(cell, _, _)| cell);
        level.sources = merged(brought);
        level
    }
}

/// What settling level-0 tile `key`'s water starts from, made by a
/// [`WaterStart`] job: the meters across the cells it settles at; the first
/// cell along x and z of the square it settles over, on the terrain's
/// lattice of cells that size; the height its heights stand above (the
/// square's lowest ground, whole meters down, which keeps them small for
/// the GPU's single precision); the tile's own cells within the square (its
/// first along x and z, and its cells along each); and the levels it
/// settles at, coarsest first, each starting from the one before it
/// ([`Self::prolong`]), the last at its cells' size: one alone where it
/// settles from the tile's earlier water.
#[derive(Clone, Debug, PartialEq)]
pub struct SettleStart {
    pub key: TileKey,
    pub cell: f32,
    pub first: [i64; 2],
    pub datum: f32,
    pub kept: [u32; 4],
    pub levels: Vec<SettleLevel>,
}

record!(SettleStart {
    key,
    cell,
    first,
    datum,
    kept,
    levels,
});

impl SettleStart {
    /// Whether it holds levels whose arrays hold their cells, each half the
    /// cells along a side of the next (rounding up), the last at its cells'
    /// size and holding the tile's cells.
    pub fn valid(&self) -> bool {
        let Some(finest) = self.levels.last() else {
            return false;
        };
        let [x, z, columns, rows] = self.kept;
        self.levels.iter().all(SettleLevel::valid)
            && self
                .levels
                .windows(2)
                .all(|pair| pair[0].side == pair[1].side.div_ceil(2))
            && finest.cell == self.cell
            && self.datum.is_finite()
            && x + columns <= finest.side
            && z + rows <= finest.side
    }

    /// The level at its cells' size.
    pub fn finest(&self) -> &SettleLevel {
        &self.levels[self.levels.len() - 1]
    }

    /// Level `level + 1`'s water to start from once level `level` settled
    /// to `state` (each cell's surface, flow along x and z and foam, row by
    /// row): each cell takes the surface of the coarser cell covering it
    /// over its own ground and that cell's velocity, dry where the coarser
    /// cell is or its ground stands above that surface, and its lake's
    /// level where one holds it; and the coarser cell's foam.
    pub fn prolong(&self, level: usize, state: &[[f32; 4]]) -> Vec<[f32; 4]> {
        let coarse = &self.levels[level];
        let fine = &self.levels[level + 1];
        let (n, m) = (fine.side as usize, coarse.side as usize);
        (0..n * n)
            .map(|at| {
                let (i, j) = (at % n, at / n);
                let parent = (j / 2).min(m - 1) * m + (i / 2).min(m - 1);
                let [surface, along_x, along_z, foam] =
                    state.get(parent).copied().unwrap_or_default();
                if !fine.held[at].is_nan() {
                    return [fine.held[at], 0.0, 0.0, foam];
                }
                let bed = fine.bed(at);
                let depth = surface - coarse.bed(parent);
                let own = max(surface, bed) - bed;
                if depth <= DRY || own <= DRY {
                    return [bed, 0.0, 0.0, 0.0];
                }
                [
                    bed + own,
                    velocity(depth, along_x) * own,
                    velocity(depth, along_z) * own,
                    foam,
                ]
            })
            .collect()
    }

    /// What the tile keeps of its square's water settled at its cells' size
    /// to `state` (each cell's surface, flow along x and z and foam, row by
    /// row, averaged over its last steps), after `steps` steps all told and
    /// `settled` or not: its own cells, standing at its lakes' levels where
    /// they hold them and its water's where it flows, dry elsewhere; the
    /// water its rivers bring to them; and its summary.
    pub fn finish(&self, state: &[[f32; 4]], steps: u32, settled: bool) -> SteadyWater {
        let finest = self.finest();
        let n = finest.side as usize;
        let [x0, z0, columns, rows] = self.kept.map(|value| value as usize);
        let count = columns * rows;
        let mut surface = vec![f32::NAN; count];
        let mut velocities = [vec![f32::NAN; count], vec![f32::NAN; count]];
        let mut foam = vec![f32::NAN; count];
        let mut flowing = 0;
        let mut volume = 0.0_f64;
        for (at, cell) in state.iter().enumerate().take(n * n) {
            if finest.held[at].is_nan() {
                volume += f64::from(max(cell[0] - finest.bed(at), 0.0));
            }
        }
        for row in 0..rows {
            for column in 0..columns {
                let out = row * columns + column;
                let at = (z0 + row) * n + x0 + column;
                let Some(&[level, along_x, along_z, cell_foam]) = state.get(at) else {
                    continue;
                };
                if !finest.held[at].is_nan() {
                    surface[out] = finest.held[at] + self.datum;
                    foam[out] = cell_foam;
                    continue;
                }
                let depth = level - finest.bed(at);
                if depth > DRY {
                    surface[out] = level + self.datum;
                    velocities[0][out] = velocity(depth, along_x);
                    velocities[1][out] = velocity(depth, along_z);
                    foam[out] = cell_foam;
                    flowing += 1;
                }
            }
        }
        let sources = finest
            .sources
            .iter()
            .filter_map(|source| {
                let (i, j) = (source.cell as usize % n, source.cell as usize / n);
                let inside = i >= x0 && j >= z0 && i < x0 + columns && j < z0 + rows;
                inside.then(|| SteadySource {
                    cell: ((j - z0) * columns + i - x0) as u32,
                    rate: source.rate,
                    way: source.way,
                })
            })
            .collect();
        let cells = [columns as u32, rows as u32];
        let summary = SteadySummary::of(cells, (&velocities, &foam));
        let [along_x, along_z] = velocities;
        let area = f64::from(finest.cell * finest.cell);
        SteadyWater {
            key: self.key,
            cell: self.cell,
            first: [self.first[0] + x0 as i64, self.first[1] + z0 as i64],
            cells,
            surface: surface.into(),
            velocity: [along_x.into(), along_z.into()],
            foam: foam.into(),
            sources,
            summary,
            stats: SteadyStats {
                sources: finest.sources.iter().map(|source| source.rate).sum(),
                volume: (volume * area) as f32,
                flowing,
                steps,
                levels: self.levels.len() as u32,
                settled,
            },
        }
    }
}

/// Preparing what settling tile `key`'s water starts from
/// ([`SettleStart`]): generating the level-0 tiles its square reaches, in
/// steps a job's budget sizes, then its levels; from `warm`, the tile's
/// earlier settled water at its cells' size, where its host holds it, the
/// finest alone.
pub struct WaterStart {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    key: TileKey,
    cell: f32,
    // The square it settles over: its first cell and side.
    first: [i64; 2],
    side: usize,
    warm: Option<SteadyWater>,
    pending: Vec<TileKey>,
    generating: Option<Generator>,
    tiles: Vec<(TileKey, Tile)>,
}

impl WaterStart {
    /// Tile `key`'s water over `layers` (with their bakes) and `pyramid`,
    /// in cells `cell` meters across, from `warm` where held.
    pub fn new(
        layers: Arc<Layers>,
        pyramid: Pyramid,
        key: TileKey,
        cell: f32,
        warm: Option<SteadyWater>,
    ) -> Self {
        let (first, side) = square(&pyramid, key, cell);
        let pending = tiles_reached(&pyramid, first, side, cell);
        Self {
            layers,
            pyramid,
            key,
            cell,
            first,
            side,
            warm,
            pending,
            generating: None,
            tiles: Vec::new(),
        }
    }

    /// Works for about `budget` units (a sample generated each): what
    /// settling starts from once done; an error where its ground does not
    /// generate.
    pub fn advance(&mut self, budget: &mut u64) -> Result<Option<SettleStart>, String> {
        loop {
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
            let Some(key) = self.pending.pop() else {
                break;
            };
            self.generating = Some(Generator::new(self.layers.clone(), &self.pyramid, key));
        }
        let ground = Ground {
            pyramid: self.pyramid,
            tiles: std::mem::take(&mut self.tiles),
        };
        let start = start(&self.layers, &ground, (self.first, self.side), self.cell);
        Ok(Some(self.levels(start)))
    }

    /// What settling starts from over `start`, the square's ground and
    /// water: its heights above the square's datum, the tile's cells within
    /// it, and its levels.
    fn levels(&self, start: Start) -> SettleStart {
        let lowest = start
            .corners
            .iter()
            .copied()
            .fold(f32::INFINITY, |low, height| min(low, height));
        let datum = if lowest.is_finite() {
            lowest.floor()
        } else {
            0.0
        };
        let lift = |height: f32| height - datum;
        let mut finest = SettleLevel {
            cell: self.cell,
            side: self.side as u32,
            corners: start.corners.iter().map(|&height| lift(height)).collect(),
            water: start
                .water
                .iter()
                .map(|&[surface, x, z]| [lift(surface), x, z])
                .collect(),
            held: start
                .held
                .iter()
                .map(|&level| if level.is_nan() { level } else { lift(level) })
                .collect(),
            sources: start.sources,
        };
        let origin = self.pyramid.tile_origin(self.key);
        let size = self.pyramid.tile_size(0);
        let (x, columns) = lattice(origin.x, origin.x + size, self.cell);
        let (z, rows) = lattice(origin.y, origin.y + size, self.cell);
        let kept = [
            (x - self.first[0]) as u32,
            (z - self.first[1]) as u32,
            columns,
            rows,
        ];
        let levels = match &self.warm {
            Some(warm) if warm.cell == self.cell && warm.key == self.key => {
                warmed(&mut finest, warm, self.first, datum);
                vec![finest]
            }
            _ => {
                let mut levels = vec![finest];
                loop {
                    let finer = &levels[levels.len() - 1];
                    if 2.0 * finer.cell > COARSEST || finer.side.div_ceil(2) < FEWEST {
                        break;
                    }
                    let coarser = finer.coarser();
                    levels.push(coarser);
                }
                levels.reverse();
                levels
            }
        };
        SettleStart {
            key: self.key,
            cell: self.cell,
            first: self.first,
            datum,
            kept,
            levels,
        }
    }
}

/// `level`'s water (the finest, over the square from cell `first`, its
/// heights above `datum`) where `warm`, the tile's earlier settled water,
/// holds the cells and no lake holds them now: its depth over the ground as
/// it stands now and its velocity, dry where it was dry or a lake held it.
fn warmed(level: &mut SettleLevel, warm: &SteadyWater, first: [i64; 2], datum: f32) {
    let n = level.side as usize;
    for at in 0..n * n {
        if !level.held[at].is_nan() {
            continue;
        }
        let (i, j) = ((at % n) as i64 + first[0], (at / n) as i64 + first[1]);
        let Some(cell) = warm.at(i, j) else {
            continue;
        };
        let bed = level.bed(at);
        level.water[at] = match cell {
            SteadyCell::Dry | SteadyCell::Held { .. } => [bed, 0.0, 0.0],
            SteadyCell::Flowing {
                surface, velocity, ..
            } => {
                let depth = max(surface - datum - bed, 0.0);
                if depth <= DRY {
                    [bed, 0.0, 0.0]
                } else {
                    [bed + depth, velocity[0] * depth, velocity[1] * depth]
                }
            }
        };
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
