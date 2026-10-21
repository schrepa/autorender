//! Water as it settles: rivers run to their steady state by the
//! shallow-water equations ([`crate::shallow_water`]), which the renderer
//! runs on the GPU, around the camera and to settle tiles' water for their
//! records, over regions of ground, from the water the hydrology bake
//! stands there. While a session cooks, a terrain's water settles whole, as
//! one region, where its cells number [`REGION_CELLS`] or fewer, else over
//! the fewest regions that do ([`settle_regions`]), each reaching
//! [`MARGIN`] meters past its tiles; a tile whose record no host holds, or
//! one under new edits, settles over a region of its own, reaching
//! [`MARGIN`] meters past it ([`SettleRegion::alone`]). Lakes hold their
//! level; rivers bring their water where they start (at heads and lakes'
//! outlets), along their course as the ground they drain grows, and where
//! they cross into the region, so what each carries is what its bake says
//! it does; water leaves over the region's edges and none comes in over
//! them. Each level-0 tile a region reaches gives its part
//! ([`WaterPart`]), which a job generating its ground makes: the ground and
//! the bake's water on the terrain's lattice of cells within it. The parts
//! make what settling starts from ([`SettleStart::assemble`]): the region's
//! cells, its water starting as the bake stands and runs it, already in
//! its channels, or as the records its host holds of its tiles under other
//! edits do. (Coarser cells than the bake's channels flood: their ground
//! holds the channels' bends and narrows as pits and dams.) Settled, its
//! water, averaged over its last steps, becomes what each tile the region
//! keeps keeps of it ([`SettleStart::finish`]), a [`SteadyWater`], a record
//! hosts keep and
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
use crate::{Generator, Layers, Pyramid, Stamp, TILE_STRIDE, Tile, TileKey};

/// Meters past its tiles a region's square reaches on each side, within
/// the terrain, so what its edges do to the water stays outside them.
pub const MARGIN: f32 = 64.0;
/// The most cells a region's square holds: the GPU takes about 124 bytes a
/// cell to settle them, two gibibytes, and binds 32 of them (each cell's
/// water's two states) in one buffer, half a gibibyte.
pub const REGION_CELLS: u64 = 1 << 24;
/// Cells inside the region's edge within which rivers count as crossing
/// into it (closer, their water leaves again at once).
const ENTRY: f32 = 2.0;
/// The least water a source brings (cubic meters a second): what a river
/// gains along its course gathers until it brings this much.
const SOURCE_LEAST: f32 = 0.01;

/// What settling a tile's water came to: the cubic meters a second its
/// rivers bring within it; the cubic meters it holds, averaged over its
/// region's last steps; its cells holding flowing water; the steps its
/// region ran; and whether its region's water settled before its steps ran
/// out (water running unsteadily, down rough steep rock, settles once its
/// averages stop changing).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SteadyStats {
    pub sources: f32,
    pub volume: f32,
    pub flowing: u32,
    pub steps: u32,
    pub settled: bool,
}

record!(SteadyStats {
    sources,
    volume,
    flowing,
    steps,
    settled,
});

/// Where rivers bring water to a cell (its index within the tile or the
/// region, row by row): the cubic meters a second, and the way the water
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

/// What level-0 tile `key`'s settled water's record is kept by, with the
/// code settling it: the stack its ground generates from (with its bakes,
/// shared as `stack`), `pyramid`, the tile, `stamps` (the terrain's edits
/// reaching the square it settles over alone, [`SettleRegion::alone`], in
/// order), the meters across its cells, and the rectangle of level-0 tiles
/// whose region its water settled over (first column and row, columns and
/// rows): a cook's region, or the tile alone. Edits farther upstream leave
/// its key as it was.
#[derive(Clone, Debug, PartialEq)]
pub struct SettledKey {
    pub stack: String,
    pub pyramid: Pyramid,
    pub key: TileKey,
    pub stamps: Vec<Stamp>,
    pub cell: f32,
    pub tiles: [u32; 4],
}

record!(SettledKey {
    stack,
    pyramid,
    key,
    stamps,
    cell,
    tiles,
});

impl SettledKey {
    /// What the record is kept by, with the settling code's.
    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
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

/// `count` tiles along an axis split into `parts` ranges as even as can
/// be: each one's first tile and tiles.
fn split(count: u32, parts: u32) -> impl Iterator<Item = (u32, u32)> {
    let at = move |part: u32| (u64::from(count) * u64::from(part) / u64::from(parts)) as u32;
    (0..parts).map(move |part| (at(part), at(part + 1) - at(part)))
}

/// The cells along axis `axis` (0 along x, 1 along z) of `pyramid`'s
/// lattice of cells `cell` meters across of the square over level-0 tiles
/// `tiles` (the first and how many), reaching [`MARGIN`] meters past them
/// within the terrain: its first cell and its cells.
fn reach(pyramid: &Pyramid, axis: usize, (first, tiles): (u32, u32), cell: f32) -> (i64, u32) {
    let size = pyramid.tile_size(0);
    let extent = pyramid.extent()[axis];
    let low = max(first as f32 * size - MARGIN, 0.0);
    let high = min((first + tiles) as f32 * size + MARGIN, extent);
    lattice(low, high, cell)
}

/// The rectangles of `pyramid`'s level-0 tiles (first column and row,
/// columns and rows) whose water settles together in cells `cell` meters
/// across while a session cooks, row by row: the terrain whole where its
/// cells number [`REGION_CELLS`] or fewer, else the fewest rectangles
/// splitting it evenly whose squares do (reaching [`MARGIN`] past them),
/// the most alike along x and z of those.
pub fn settle_regions(pyramid: &Pyramid, cell: f32) -> Vec<[u32; 4]> {
    let tiles = pyramid.tiles(0);
    // The most cells along an axis a region holds, the axis split into one
    // part, two and so on.
    let widest = |axis: usize| -> Vec<u64> {
        (1..=tiles[axis])
            .map(|parts| {
                split(tiles[axis], parts)
                    .map(|range| u64::from(reach(pyramid, axis, range, cell).1))
                    .max()
                    .unwrap_or(0)
            })
            .collect()
    };
    let (columns, rows) = (widest(0), widest(1));
    let mut best = (tiles[0], tiles[1]);
    let mut fewest = (u64::MAX, u32::MAX);
    for (x, width) in (1_u32..).zip(&columns) {
        for (z, depth) in (1_u32..).zip(&rows) {
            let rank = (u64::from(x) * u64::from(z), x.abs_diff(z));
            if width * depth <= REGION_CELLS && rank < fewest {
                (fewest, best) = (rank, (x, z));
            }
        }
    }
    let (along_x, along_z) = best;
    let mut regions = Vec::new();
    for (z, rows) in split(tiles[1], along_z) {
        for (x, columns) in split(tiles[0], along_x) {
            regions.push([x, z, columns, rows]);
        }
    }
    regions
}

/// A region whose water settles together: the meters across its cells; the
/// first cell along x and z of its square on the terrain's lattice of cells
/// that size, and its cells along each; and the level-0 tiles within it
/// whose water it keeps, each a record, in key order.
#[derive(Clone, Debug, PartialEq)]
pub struct SettleRegion {
    pub cell: f32,
    pub first: [i64; 2],
    pub cells: [u32; 2],
    pub kept: Vec<TileKey>,
}

impl SettleRegion {
    /// The region over `pyramid`'s level-0 tiles `tiles` (first column and
    /// row, columns and rows) in cells `cell` meters across, its square
    /// reaching [`MARGIN`] meters past them within the terrain, keeping the
    /// water of `kept`, tiles among them.
    pub fn over(pyramid: &Pyramid, tiles: [u32; 4], mut kept: Vec<TileKey>, cell: f32) -> Self {
        let [x, z, columns, rows] = tiles;
        let (first_x, along_x) = reach(pyramid, 0, (x, columns), cell);
        let (first_z, along_z) = reach(pyramid, 1, (z, rows), cell);
        kept.sort_unstable();
        kept.dedup();
        Self {
            cell,
            first: [first_x, first_z],
            cells: [along_x, along_z],
            kept,
        }
    }

    /// The region level-0 tile `key` of `pyramid` settles over alone in
    /// cells `cell` meters across, keeping its water.
    pub fn alone(pyramid: &Pyramid, key: TileKey, cell: f32) -> Self {
        let tiles = [u32::from(key.x), u32::from(key.z), 1, 1];
        Self::over(pyramid, tiles, vec![key], cell)
    }

    /// The terrain-local rectangle its square covers, its corners included.
    pub fn bounds(&self) -> (Vec2, Vec2) {
        let low = Vec2::new(self.first[0] as f32, self.first[1] as f32) * self.cell;
        let cells = Vec2::new(self.cells[0] as f32, self.cells[1] as f32);
        (low, low + cells * self.cell)
    }

    /// The level-0 tiles of `pyramid` holding its square's corners, whose
    /// parts it settles from ([`WaterPart`]), in key order.
    pub fn reached(&self, pyramid: &Pyramid) -> Vec<TileKey> {
        let (low, high) = self.bounds();
        let size = pyramid.tile_size(0);
        let [columns, rows] = pyramid.tiles(0);
        let range = |low: f32, high: f32, count: u32| {
            let tile = |at: f32| (max((at / size).floor(), 0.0) as u32).min(count - 1);
            tile(low)..=tile(high)
        };
        let mut keys = Vec::new();
        for z in range(low.y, high.y, rows) {
            for x in range(low.x, high.x, columns) {
                keys.push(TileKey::new(0, x as u16, z as u16));
            }
        }
        keys.sort_unstable();
        keys
    }
}

/// A level-0 tile's part of the water settling over the regions reaching
/// it ([`SettleStart::assemble`]), on the terrain's lattice of cells `cell`
/// meters across: the ground at the lattice's corners within the tile (the
/// terrain's far edges' with its last tiles), its first corner along x and
/// z and its corners along each, row by row; and at each cell whose middle
/// lies within it (its first along x and z and its cells along each, row
/// by row) the bake's water there: its surface (NaN where it holds none),
/// its velocity along x and z (meters a second), and 1 where a lake holds
/// it, else 0.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterPart {
    pub key: TileKey,
    pub cell: f32,
    pub corner: [i64; 2],
    pub corners: [u32; 2],
    pub ground: Vec<f32>,
    pub first: [i64; 2],
    pub cells: [u32; 2],
    pub water: Vec<[f32; 4]>,
}

record!(WaterPart {
    key,
    cell,
    corner,
    corners,
    ground,
    first,
    cells,
    water,
});

impl WaterPart {
    /// `tile`'s part, a level-0 tile of `pyramid`, on the lattice of cells
    /// `cell` meters across.
    fn of(pyramid: &Pyramid, tile: &Tile, cell: f32) -> Self {
        let key = tile.key();
        let origin = pyramid.tile_origin(key);
        let size = pyramid.tile_size(0);
        let extent = pyramid.extent();
        let tiles = pyramid.tiles(0);
        // The corners within the tile along an axis: from its first edge to
        // its last, the terrain's last corners with its last tile.
        let along = |axis: usize, at: u16| {
            let first = (origin[axis] / cell).ceil() as i64;
            let end = if u32::from(at) + 1 == tiles[axis] {
                i64::from(lattice(0.0, extent[axis], cell).1) + 1
            } else {
                ((origin[axis] + size) / cell).ceil() as i64
            };
            (first, (end - first).max(0) as u32)
        };
        let (corner_x, corners_x) = along(0, key.x);
        let (corner_z, corners_z) = along(1, key.z);
        let mut ground = Vec::with_capacity(corners_x as usize * corners_z as usize);
        for j in 0..i64::from(corners_z) {
            for i in 0..i64::from(corners_x) {
                let point = Vec2::new((corner_x + i) as f32, (corner_z + j) as f32) * cell;
                let local = (point - origin).clamp(Vec2::ZERO, Vec2::splat(size));
                ground.push(tile.sample(local).map_or(0.0, |sample| sample.height));
            }
        }
        let (first_x, columns) = lattice(origin.x, origin.x + size, cell);
        let (first_z, rows) = lattice(origin.y, origin.y + size, cell);
        let mut water = Vec::with_capacity(columns as usize * rows as usize);
        for j in 0..i64::from(rows) {
            for i in 0..i64::from(columns) {
                let middle = (Vec2::new((first_x + i) as f32, (first_z + j) as f32) + 0.5) * cell;
                let local = middle - origin;
                let found = tile
                    .sample(local)
                    .and_then(|sample| (!sample.hole).then(|| tile.water_at(local)).flatten());
                water.push(found.map_or([f32::NAN, 0.0, 0.0, 0.0], |point| {
                    let lake = if point.lake { 1.0 } else { 0.0 };
                    [point.surface, point.flow.x, point.flow.y, lake]
                }));
            }
        }
        Self {
            key,
            cell,
            corner: [corner_x, corner_z],
            corners: [corners_x, corners_z],
            ground,
            first: [first_x, first_z],
            cells: [columns, rows],
            water,
        }
    }

    /// Whether its arrays hold its corners and cells, on a lattice of cells
    /// of some size.
    pub(crate) fn valid(&self) -> bool {
        let count = |along: [u32; 2]| along[0] as usize * along[1] as usize;
        self.key.level == 0
            && self.cell.is_finite()
            && self.cell > 0.0
            && self.ground.len() == count(self.corners)
            && self.water.len() == count(self.cells)
    }
}

/// Making level-0 tile `key`'s part of the water settling over the regions
/// reaching it ([`WaterPart`]): generating its ground, in steps a job's
/// budget sizes, then reading the ground and its bake's water off it.
pub(crate) struct PartMaking {
    pyramid: Pyramid,
    cell: f32,
    generator: Generator,
}

impl PartMaking {
    /// Tile `key`'s part over `layers` (with their bakes) and `pyramid`, of
    /// the lattice of cells `cell` meters across.
    pub(crate) fn new(layers: Arc<Layers>, pyramid: Pyramid, key: TileKey, cell: f32) -> Self {
        Self {
            pyramid,
            cell,
            generator: Generator::new(layers, &pyramid, key),
        }
    }

    /// Works for about `budget` units (a sample generated each): the part
    /// once done; an error where its ground does not generate.
    pub(crate) fn advance(&mut self, budget: &mut u64) -> Result<Option<WaterPart>, String> {
        let Some(tile) = self.generator.advance(budget) else {
            if self.generator.failed() {
                return Err("the ground's heights are invalid".into());
            }
            return Ok(None);
        };
        Ok(Some(WaterPart::of(&self.pyramid, &tile, self.cell)))
    }
}

/// Copies the rows of `from`, a grid of `size` along x and z whose first
/// entry stands at `at` on a lattice, that lie within `into`, a grid of
/// `columns` along x and `rows` along z whose first stands at `first`, into
/// it.
fn copy<T: Copy>(
    into: &mut [T],
    (first, columns, rows): ([i64; 2], usize, usize),
    from: &[T],
    (at, size): ([i64; 2], [u32; 2]),
) {
    let width = i64::from(size[0]);
    let low = [at[0].max(first[0]), at[1].max(first[1])];
    let high = [
        (at[0] + width).min(first[0] + columns as i64),
        (at[1] + i64::from(size[1])).min(first[1] + rows as i64),
    ];
    if low[0] >= high[0] {
        return;
    }
    let count = (high[0] - low[0]) as usize;
    for z in low[1]..high[1] {
        let source = ((z - at[1]) * width + low[0] - at[0]) as usize;
        let target = ((z - first[1]) * columns as i64 + low[0] - first[0]) as usize;
        into[target..target + count].copy_from_slice(&from[source..source + count]);
    }
}

/// Where the baked rivers bring water to the rectangle from `origin`
/// (terrain-local) of `cells` along x and z `cell` meters across: at each
/// point of a river's course within it, what the river carries there less
/// what it carried at the point before within it (all it carries where it
/// starts at a head or a lake's outlet, or crosses into the rectangle;
/// nothing more where it starts at a confluence, whose rivers bring it),
/// gathered along the course until [`SOURCE_LEAST`] or more; spread over
/// the cells within half its width no lake holds (`held`) that its baked
/// water covers (`wet`), and where it covers none there, further down the
/// course: its water comes where its bake stands water. Each cell's water
/// is summed, cells in order.
fn sources(
    layers: &Layers,
    (origin, cells, cell): (Vec2, [u32; 2], f32),
    (held, wet): (&[f32], &[bool]),
) -> Vec<SteadySource> {
    let Some(hydrology) = layers.hydrology() else {
        return Vec::new();
    };
    let size = Vec2::new(cells[0] as f32, cells[1] as f32) * cell;
    let margin = ENTRY * cell;
    let inside = |point: [f32; 2]| {
        let point = Vec2::from(point);
        point.x > origin.x + margin
            && point.y > origin.y + margin
            && point.x < origin.x + size.x - margin
            && point.y < origin.y + size.y - margin
    };
    let courses: Vec<_> = hydrology.rivers().courses().collect();
    let stride = cells[0] as i32;
    let last = IVec2::new(cells[0] as i32 - 1, cells[1] as i32 - 1);
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
                if course.head() {
                    0.0
                } else {
                    course.discharge(0)
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
                .min(last);
            let mut covered = Vec::new();
            for j in low.y..=high.y {
                for i in low.x..=high.x {
                    let middle = origin + (Vec2::new(i as f32, j as f32) + 0.5) * cell;
                    let index = (j * stride + i) as usize;
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

/// The cells of a region settling ([`SettleStart`]): the meters across
/// them and how many along x and z; the ground at their corners (`cells +
/// 1` a row, rows along z); each cell's water to start from (its surface,
/// or its ground where dry, and its flow along x and z in square meters a
/// second) and the level a lake holds it at (NaN where none does), row by
/// row; and where its rivers bring water. Its heights stand above its
/// region's datum.
#[derive(Clone, Debug, PartialEq)]
pub struct SettleLevel {
    pub cell: f32,
    pub cells: [u32; 2],
    pub corners: Vec<f32>,
    pub water: Vec<[f32; 3]>,
    pub held: Vec<f32>,
    pub sources: Vec<SteadySource>,
}

impl SettleLevel {
    /// Cell `at`'s ground: the mean of its corners'.
    pub fn bed(&self, at: usize) -> f32 {
        let n = self.cells[0] as usize;
        let (i, j) = (at % n, at / n);
        let corner = |x: usize, z: usize| self.corners[z * (n + 1) + x];
        0.25 * (corner(i, j) + corner(i + 1, j) + corner(i, j + 1) + corner(i + 1, j + 1))
    }
}

/// What settling a region's water starts from ([`Self::assemble`]): the
/// meters across the cells it settles at; the first cell along x and z of
/// its square, on the terrain's lattice of cells that size; the height its
/// heights stand above (its lowest ground, whole meters down, which keeps
/// them small for the GPU's single precision); the tiles whose water it
/// keeps, each with its cells within the square (its first along x and z,
/// and its cells along each), in key order; and its cells.
#[derive(Clone, Debug, PartialEq)]
pub struct SettleStart {
    pub cell: f32,
    pub first: [i64; 2],
    pub datum: f32,
    pub kept: Vec<(TileKey, [u32; 4])>,
    pub level: SettleLevel,
}

impl SettleStart {
    /// What settling `region`'s water starts from over `layers` (with their
    /// bakes) and `pyramid`, from `parts`, those of the tiles its square
    /// reaches ([`SettleRegion::reached`]), in any order: its ground and
    /// each cell's water as the bake stands and runs it, or as `warm` holds
    /// it where its host holds earlier records of its water (the settled
    /// water of the terrain's lattice cell `(i, j)`, counted from its first
    /// corner, where one holds it), heights above its lowest ground, and the
    /// cells of the tiles it keeps. An error where a part it reaches is
    /// missing or on another lattice.
    pub fn assemble(
        layers: &Layers,
        pyramid: &Pyramid,
        region: &SettleRegion,
        parts: &[&WaterPart],
        warm: Option<&dyn Fn(i64, i64) -> Option<SteadyCell>>,
    ) -> Result<Self, String> {
        let cell = region.cell;
        let first = region.first;
        let [nx, nz] = region.cells.map(|count| count as usize);
        if nx == 0 || nz == 0 {
            return Err("settling water over no cells".into());
        }
        // The ground at its corners and the bake's water at its cells, from
        // the parts holding them; a cell no part holds keeps a negative
        // lake.
        let mut corners = vec![f32::NAN; (nx + 1) * (nz + 1)];
        let mut found = vec![[f32::NAN, 0.0, 0.0, -1.0]; nx * nz];
        for part in parts {
            if part.cell != cell {
                return Err("a part of settling water on another lattice".into());
            }
            copy(
                &mut corners,
                (first, nx + 1, nz + 1),
                &part.ground,
                (part.corner, part.corners),
            );
            copy(
                &mut found,
                (first, nx, nz),
                &part.water,
                (part.first, part.cells),
            );
        }
        if corners.iter().any(|height| height.is_nan()) || found.iter().any(|entry| entry[3] < 0.0)
        {
            return Err("settling water misses a part of the ground it reaches".into());
        }
        let bed = |at: usize| {
            let (i, j) = (at % nx, at / nx);
            let corner = |x: usize, z: usize| corners[z * (nx + 1) + x];
            0.25 * (corner(i, j) + corner(i + 1, j) + corner(i, j + 1) + corner(i + 1, j + 1))
        };
        let mut water = Vec::with_capacity(nx * nz);
        let mut held = Vec::with_capacity(nx * nz);
        for (at, &[surface, x, z, lake]) in found.iter().enumerate() {
            let floor = bed(at);
            if surface.is_nan() || surface <= floor {
                water.push([floor, 0.0, 0.0]);
                held.push(f32::NAN);
            } else if lake > 0.5 {
                water.push([surface, 0.0, 0.0]);
                held.push(surface);
            } else {
                let depth = surface - floor;
                water.push([surface, depth * x, depth * z]);
                held.push(f32::NAN);
            }
        }
        drop(found);
        let wet: Vec<bool> = (0..nx * nz)
            .map(|at| held[at].is_nan() && water[at][0] > bed(at))
            .collect();
        let origin = Vec2::new(first[0] as f32, first[1] as f32) * cell;
        let sources = sources(layers, (origin, region.cells, cell), (&held, &wet));
        drop(wet);
        let lowest = corners
            .iter()
            .copied()
            .fold(f32::INFINITY, |low, height| min(low, height));
        let datum = if lowest.is_finite() {
            lowest.floor()
        } else {
            0.0
        };
        for height in &mut corners {
            *height -= datum;
        }
        for entry in &mut water {
            entry[0] -= datum;
        }
        for level in &mut held {
            *level -= datum;
        }
        let mut level = SettleLevel {
            cell,
            cells: region.cells,
            corners,
            water,
            held,
            sources,
        };
        let size = pyramid.tile_size(0);
        let mut kept = Vec::with_capacity(region.kept.len());
        for &key in &region.kept {
            let origin = pyramid.tile_origin(key);
            let (x, columns) = lattice(origin.x, origin.x + size, cell);
            let (z, rows) = lattice(origin.y, origin.y + size, cell);
            let (x, z) = (x - first[0], z - first[1]);
            let outside = x < 0
                || z < 0
                || x + i64::from(columns) > nx as i64
                || z + i64::from(rows) > nz as i64;
            if outside {
                return Err("a tile settling water keeps lies outside its region".into());
            }
            kept.push((key, [x as u32, z as u32, columns, rows]));
        }
        if let Some(warm) = warm {
            warmed(&mut level, warm, first, datum);
        }
        Ok(Self {
            cell,
            first,
            datum,
            kept,
            level,
        })
    }

    /// What each tile it keeps keeps of its water settled to `state` (each
    /// cell's surface, flow along x and z and foam, row by row, averaged
    /// over its last steps), after `steps` steps and `settled` or not, in
    /// the order it keeps them: the tile's own cells, standing at its lakes'
    /// levels where they hold them and its water's where it flows, dry
    /// elsewhere; the water its rivers bring to them; and its summary.
    pub fn finish(&self, state: &[[f32; 4]], steps: u32, settled: bool) -> Vec<SteadyWater> {
        let finest = &self.level;
        let n = finest.cells[0] as usize;
        let area = f64::from(finest.cell * finest.cell);
        let mut made = Vec::with_capacity(self.kept.len());
        for &(key, kept) in &self.kept {
            let [x0, z0, columns, rows] = kept.map(|value| value as usize);
            let count = columns * rows;
            let mut surface = vec![f32::NAN; count];
            let mut velocities = [vec![f32::NAN; count], vec![f32::NAN; count]];
            let mut foam = vec![f32::NAN; count];
            let mut flowing = 0;
            let mut volume = 0.0_f64;
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
                        volume += f64::from(depth);
                    }
                }
            }
            let sources: Vec<SteadySource> = finest
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
            made.push(SteadyWater {
                key,
                cell: self.cell,
                first: [self.first[0] + x0 as i64, self.first[1] + z0 as i64],
                cells,
                surface: surface.into(),
                velocity: [along_x.into(), along_z.into()],
                foam: foam.into(),
                stats: SteadyStats {
                    sources: sources.iter().map(|source| source.rate).sum(),
                    volume: (volume * area) as f32,
                    flowing,
                    steps,
                    settled,
                },
                sources,
                summary,
            });
        }
        made
    }
}

/// `level`'s water (over the region from cell `first`, its heights above
/// `datum`) where `warm` (the settled water of the terrain's
/// lattice cells) holds the cells and no lake holds them now: its depth
/// over the ground as it stands now and its velocity, dry where it was dry
/// or a lake held it.
fn warmed(
    level: &mut SettleLevel,
    warm: &dyn Fn(i64, i64) -> Option<SteadyCell>,
    first: [i64; 2],
    datum: f32,
) {
    let n = level.cells[0] as usize;
    for at in 0..level.water.len() {
        if !level.held[at].is_nan() {
            continue;
        }
        let (i, j) = ((at % n) as i64 + first[0], (at / n) as i64 + first[1]);
        let Some(cell) = warm(i, j) else {
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
