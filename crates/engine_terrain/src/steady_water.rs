//! Water as it settles: rivers run to their steady state by the
//! shallow-water equations ([`crate::shallow_water`]), which the renderer
//! runs on the GPU to settle tiles' water for their records over regions of
//! ground, from the water the hydrology bake stands there, in the model the
//! water running live around the camera runs in: on its cells
//! ([`shallow_water::live_split`] of them along each side of a settled
//! cell), over its bed (level 0's ground on its triangles at their corners,
//! raised on the GPU by the relief of its materials at the cells' spacing as
//! the live water's setup raises it, `meshes/water_setup.wgsl`), friction
//! counting only the roughness that bed does not hold as its shape, and its
//! last looks stepping, and stirred by turbulence's bursts, as the live
//! water steps and is stirred. While a session cooks, a terrain's water
//! settles whole, as one region, where the cells it stores number
//! [`REGION_CELLS`] or fewer, else over the fewest regions whose cells do
//! ([`settle_regions`]), each reaching [`MARGIN`] meters past its tiles; a
//! tile whose record no host holds, or one under new edits, settles over a
//! region of its own, reaching [`MARGIN`] meters past it
//! ([`SettleRegion::alone`]). A region stores its cells in blocks of
//! [`BLOCK_SIDE`] along each side, and only the blocks its water can reach
//! ([`StoredBlocks`]): those holding water, a lake or cells its rivers bring
//! water to as it starts, and a few blocks around them. The cells of the
//! others are dry and high: water reaching them stands against them, and the
//! renderer counts it. Lakes hold their level; rivers bring their water
//! where they start (at heads and lakes' outlets), along their course as the
//! ground they drain grows, and where they cross into the region, so what
//! each carries is what its bake says it does; water leaves over the
//! region's edges and none comes in over them. Each level-0 tile a region
//! reaches gives its part ([`WaterPart`]), which a job generating its ground
//! and placing its objects makes: at its samples, their ground, materials,
//! smooth surface and beds; the bake's water on the terrain's lattice of
//! settled cells within it; and the solids its objects stand as. Solids
//! standing in water are ground the water flows around: each cell blocks the
//! flow through its faces by the share of it they cover, as deep as they
//! stand in its water ([`cover_cells`]). The parts make what settling starts
//! from ([`SettleStart::assemble`]): the region's stored cells with their
//! bare ground and the level-0 samples their corners lie among, by which the
//! GPU raises it, their water starting as the bake stands and runs it,
//! already in its channels, or as the records its host holds of its tiles
//! under other edits do, and its solids holding it back. Settled, its water,
//! averaged over its last steps with how much it varied over them, becomes
//! what each tile the region keeps keeps of it ([`SettleStart::finish`]), a
//! [`SteadyWater`] on the settled cells, each the mean of the cells over it:
//! a record hosts keep and the cook ships with the solids standing in it.
//! Level-0 tiles' water stands and runs as theirs and their neighbors' say
//! ([`Tile::steadied`]), coarser tiles' runs as the summaries under them do,
//! each sample averaging their squares over its own spacing
//! ([`Tile::flowing`]), so every level holds the finer water's average; each
//! tile's water is a job's record of its own, made on a worker from the
//! records it reads ([`Steadying`]), with the solids standing in it; and the
//! window running live around the camera starts from it, each settled
//! cell's live cells matched to its mean, and takes its water at its edges,
//! its rivers' water where they bring it and its solids.

use std::sync::Arc;

use glam::{IVec2, Vec2};

use engine_math::collections::HashMap;
use engine_math::{clamp, max, min};

use crate::codec::{self, Reader, Writer, record};
use crate::hydraulics::GRAVITY;
use crate::shading::gradient_word;
use crate::shallow_water::{self, CELL_MEANS, DRY, UNHELD, WET_MOST, resistance, velocity};
use crate::surface::smooth_gradient;
use crate::tile::on_triangle;
use crate::tile_water::{
    NONE, TileWater, WATER_SPAN_LEAST, WaterState, breaking, nearest, pack, pack_state, shears,
    turbulence, unpack,
};
use crate::{
    Bed, BiomeRules, Footprint, Generator, Layers, Pyramid, Reach, Solid, Stamp, TILE_QUADS,
    TILE_SAMPLES, TILE_STRIDE, Tile, TileKey,
};

/// Meters past its tiles a region's square reaches on each side, within
/// the terrain, so what its edges do to the water stays outside them.
pub const MARGIN: f32 = 64.0;
/// The most cells a region stores (live cells, [`shallow_water::live_split`]
/// of them along each side of a settled cell): the GPU takes about 214
/// bytes a cell to settle them, 0.8 gibibytes, and binds about 113 of them
/// (each cell's sums and means, and its block's looks) in one buffer, under
/// half a gibibyte.
pub const REGION_CELLS: u64 = 1 << 22;
/// Cells along each side of a block: a region stores its cells, and the
/// GPU steps them, a block at a time.
pub const BLOCK_SIDE: u32 = 8;
/// A block's cells, the corners of its cells, and the entries of its ground
/// ([`StoredBlocks`]): the ground at its cells' corners, then the level a
/// lake holds each of its cells at, each of its cells' bed's roughness, the
/// share of each of its cells covered by solids and the height their tops
/// stand at over it ([`cover_cells`]).
pub const BLOCK_CELLS: usize = (BLOCK_SIDE * BLOCK_SIDE) as usize;
const BLOCK_CORNERS: usize = ((BLOCK_SIDE + 1) * (BLOCK_SIDE + 1)) as usize;
pub const BLOCK_GROUND: usize = BLOCK_CORNERS + 4 * BLOCK_CELLS;
/// What a table of stored blocks holds for a block it stores nowhere.
pub const NO_BLOCK: u32 = u32::MAX;
/// Blocks a region stores around those holding water, a lake or cells its
/// rivers bring water to as it starts, on every side: room its water
/// spreads into as it settles.
const SPREAD: u32 = 2;
/// Samples along each side of a level-0 tile's part ([`WaterPart`]): its
/// own, 0 to 256, the last the next tile's first.
const SAMPLES: usize = TILE_SAMPLES as usize;
/// What a region's parts lacking some of its ground fail with.
const MISSING: &str = "settling water misses a part of the ground it reaches";
/// Cells inside the region's edge within which rivers count as crossing
/// into it (closer, their water leaves again at once).
const ENTRY: f32 = 2.0;
/// The least water a source brings (cubic meters a second): what a river
/// gains along its course gathers until it brings this much.
const SOURCE_LEAST: f32 = 0.01;
/// Points along each side of a cell its cover is measured at
/// ([`cover_cells`]), a quarter of its spacing apart: a solid covers a cell
/// by the share of them it stands on, so one thinner than a quarter of a
/// cell covers none and turns no flow, standing in the water alone
/// ([`TileWater::solids`]).
pub const COVER_POINTS: u32 = 4;

/// The kinds of reach channels run in ([`Reach`]): pools and riffles, plane
/// beds, steps, cascades and falls.
pub(crate) const REACHES: usize = Reach::Falls as usize + 1;

/// What settling a tile's water came to: the cubic meters a second its
/// rivers bring within it; the cubic meters it holds, averaged over its
/// region's last steps; its region's cells within it holding flowing water;
/// the steps its region ran; whether its region's water settled before its
/// steps ran out, as it stepped by its cells' own seconds and again as it
/// stirred (its means stationary: the water its region holds, and its
/// flowing cells' mean speed, energy lost, heave and swirl, each changing by
/// under a hundredth of itself from one look to the next, several looks in
/// a row); its flowing cells whose water ran at the fastest water runs over
/// its last steps, which the scheme's safety net held back; and of its
/// flowing cells in channels, by the reach they run in ([`Reach`]'s order),
/// their mean effective Manning's n (that of friction as the scheme applies
/// it over its bed, relief and all), the share of them whose water runs
/// faster than its waves (Froude above 1), and how many there are (0 each
/// where none is).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SteadyStats {
    pub sources: f32,
    pub volume: f32,
    pub flowing: u32,
    pub steps: u32,
    pub settled: bool,
    pub clamped: u32,
    pub friction: [f32; REACHES],
    pub fast: [f32; REACHES],
    pub reaches: [u32; REACHES],
}

record!(SteadyStats {
    sources,
    volume,
    flowing,
    steps,
    settled,
    clamped,
    friction,
    fast,
    reaches,
});

/// Where rivers bring water to a cell (its index within the tile, row by
/// row, or among the cells a region stores): the cubic meters a second,
/// and the way the water runs as it comes, as a share of
/// [`shallow_water::SOURCE_SPEED`] along x and z.
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
/// cell, row by row, the mean of the water of the live cells over it: the
/// surface its water stands at (terrain-local; NaN where dry), its velocity
/// along x and z (meters a second; NaN where a lake holds it or it is dry),
/// the energy it loses, how far its surface heaves, how its water swirls,
/// how far its velocity varies and the share of the time it left its bed
/// (as [`SteadyCell::Flowing`] holds them; NaN where a lake holds it or it
/// is dry) and the share of the time it was wet (0 to 1, dry cells too);
/// where its rivers bring water; its water seen from afar; how settling
/// came out; and the solids standing in its water whose middles its cells
/// hold (terrain-local, in their order).
#[derive(Clone, Debug, PartialEq)]
pub struct SteadyWater {
    pub key: TileKey,
    pub cell: f32,
    pub first: [i64; 2],
    pub cells: [u32; 2],
    surface: Box<[f32]>,
    velocity: [Box<[f32]>; 2],
    energy: Box<[f32]>,
    heave: Box<[f32]>,
    swirl: Box<[f32]>,
    rush: Box<[f32]>,
    wet: Box<[f32]>,
    lip: Box<[f32]>,
    pub sources: Vec<SteadySource>,
    pub summary: SteadySummary,
    pub stats: SteadyStats,
    pub solids: Vec<Footprint>,
}

/// A cell of settled water ([`SteadyWater::at`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SteadyCell {
    Dry,
    /// A lake holds it at `level`.
    Held { level: f32 },
    /// Its water stands at `surface` and runs at `velocity`, as its means
    /// over the time it was wet in its region's last steps, which was `wet`
    /// of them, [`shallow_water::WET_MOST`] or more; over that time, it
    /// loses `energy` watts a square meter of bed (to its bed's friction and
    /// across its faces); about their means, its surface heaves by `heave`
    /// meters and its velocity varies by `rush` meters a second, and its
    /// vorticity's root mean square is `swirl` a second; and its water left
    /// its bed (curving away under it faster than gravity holds it down)
    /// `lip` of it.
    Flowing {
        surface: f32,
        velocity: [f32; 2],
        energy: f32,
        heave: f32,
        swirl: f32,
        rush: f32,
        wet: f32,
        lip: f32,
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
        if velocity[0].is_nan() {
            return Some(SteadyCell::Held { level: surface });
        }
        Some(SteadyCell::Flowing {
            surface,
            velocity,
            energy: self.energy[at],
            heave: self.heave[at],
            swirl: self.swirl[at],
            rush: self.rush[at],
            wet: self.wet[at],
            lip: self.lip[at],
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
            .put(&self.stats)
            .put(&self.solids);
        for values in [
            &self.surface,
            &self.velocity[0],
            &self.velocity[1],
            &self.energy,
            &self.heave,
            &self.swirl,
            &self.rush,
            &self.wet,
            &self.lip,
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
        let solids: Vec<Footprint> = input.get()?;
        let stride = cells[0] as usize;
        let count = stride * cells[1] as usize;
        let surface = codec::get_grid(input, stride)?;
        let x = codec::get_grid(input, stride)?;
        let z = codec::get_grid(input, stride)?;
        let energy = codec::get_grid(input, stride)?;
        let heave = codec::get_grid(input, stride)?;
        let swirl = codec::get_grid(input, stride)?;
        let rush = codec::get_grid(input, stride)?;
        let wet = codec::get_grid(input, stride)?;
        let lip = codec::get_grid(input, stride)?;
        let grids = [&surface, &x, &z, &energy, &heave, &swirl, &rush, &wet, &lip];
        if grids.iter().any(|grid| grid.len() != count)
            || !(cell.is_finite() && cell > 0.0)
            || sources.iter().any(|source| source.cell as usize >= count)
            || !solids.iter().all(Footprint::valid)
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
            energy,
            heave,
            swirl,
            rush,
            wet,
            lip,
            sources,
            summary,
            stats,
            solids,
        })
    }
}

/// What a record of settled water holds before its cells: its tile, the
/// meters across its cells and its summary, which coarser tiles' water runs
/// by.
#[derive(Clone, Debug, PartialEq)]
pub struct SettledHead {
    pub key: TileKey,
    pub cell: f32,
    pub summary: SteadySummary,
}

impl SteadyWater {
    /// The head of the record of settled water `bytes`, read without its
    /// cells (its tile, its cells' size and its summary), and the bytes it
    /// takes from the record's first: what workers share for coarser tiles'
    /// water to read.
    pub fn head(bytes: &[u8]) -> Result<(SettledHead, usize), codec::CodecError> {
        let mut input = Reader::new(bytes);
        let key = input.get()?;
        let cell: f32 = input.get()?;
        let _: [i64; 2] = input.get()?;
        let _: [u32; 2] = input.get()?;
        let _: Vec<SteadySource> = input.get()?;
        let summary = input.get()?;
        if !(cell.is_finite() && cell > 0.0) {
            return Err(codec::CodecError("invalid settled water"));
        }
        Ok((
            SettledHead { key, cell, summary },
            bytes.len() - input.left(),
        ))
    }
}

/// What level-0 tile `key`'s settled water's record is kept by, with the
/// code settling it (the shaders raising its bed among it): the recipe its
/// ground generates and its objects stand from (its stack with its bakes,
/// its pyramid and its surface, shared as `recipe`), the tile, `stamps`
/// (the terrain's edits reaching the square it settles over alone,
/// [`SettleRegion::alone`], in order), the solids its surface's assets'
/// objects stand as (by asset; none for an asset whose objects stand as
/// none), what raises its bed, hashed (its surface materials' relief and
/// the height maps that relief reads, as its host finds them), the meters
/// across its cells, and the rectangle of level-0 tiles whose region its
/// water settled over (first column and row, columns and rows): a cook's
/// region, or the tile alone. Edits farther upstream leave its key as it
/// was.
#[derive(Clone, Debug, PartialEq)]
pub struct SettledKey {
    pub recipe: String,
    pub key: TileKey,
    pub stamps: Vec<Stamp>,
    pub solids: Vec<Option<Solid>>,
    pub relief: [u8; 16],
    pub cell: f32,
    pub tiles: [u32; 4],
}

record!(SettledKey {
    recipe,
    key,
    stamps,
    solids,
    relief,
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

/// The live cells along an axis over the settled cells from `first`, `cells`
/// of them, each split into `split`: their first and how many.
fn finer((first, cells): (i64, u32), split: u32) -> (i64, u32) {
    (first * i64::from(split), cells * split)
}

/// The rectangles of `pyramid`'s level-0 tiles (first column and row,
/// columns and rows) whose water settles together over settled cells `cell`
/// meters across while a session cooks, row by row: the terrain whole where
/// the live cells it would store number [`REGION_CELLS`] or fewer, else the
/// fewest rectangles splitting it evenly whose squares' do (reaching
/// [`MARGIN`] past them), the most alike along x and z of those. What a
/// square would store is found from `layers`' bake: the blocks its rivers'
/// channels and its water reach, and those around them a region stores
/// too.
pub fn settle_regions(layers: &Layers, pyramid: &Pyramid, cell: f32) -> Vec<[u32; 4]> {
    let tiles = pyramid.tiles(0);
    let split = shallow_water::live_split(cell, pyramid.spacing(0));
    let reached = Reached::of(layers, pyramid, cell / split as f32);
    let mut splits: Vec<(u32, u32)> = (1..=tiles[0])
        .flat_map(|x| (1..=tiles[1]).map(move |z| (x, z)))
        .collect();
    splits.sort_by_key(|&(x, z)| (u64::from(x) * u64::from(z), x.abs_diff(z)));
    let fits = |&(along_x, along_z): &(u32, u32)| {
        split(tiles[1], along_z).all(|rows| {
            let rows = finer(reach(pyramid, 1, rows, cell), split);
            split(tiles[0], along_x).all(|columns| {
                let columns = finer(reach(pyramid, 0, columns, cell), split);
                reached.cells(columns, rows) <= REGION_CELLS
            })
        })
    };
    let (along_x, along_z) = splits
        .into_iter()
        .find(fits)
        .unwrap_or((tiles[0], tiles[1]));
    let mut regions = Vec::new();
    for (z, rows) in split(tiles[1], along_z) {
        for (x, columns) in split(tiles[0], along_x) {
            regions.push([x, z, columns, rows]);
        }
    }
    regions
}

/// The blocks of a terrain's lattice of live cells (each [`BLOCK_SIDE`]
/// cells along each side, from its first corner) a region would store, as
/// its bake finds them: those within a cell of its rivers' channels or
/// whose middles its water covers, and the blocks around them; held as how
/// many of them lie from the first block to each corner of the blocks' grid
/// (`columns + 1` a row).
struct Reached {
    columns: usize,
    counts: Vec<u32>,
}

impl Reached {
    /// The blocks `layers`' bake reaches of `pyramid`'s lattice of live
    /// cells `cell` meters across.
    fn of(layers: &Layers, pyramid: &Pyramid, cell: f32) -> Self {
        let extent = pyramid.extent();
        let blocks = [0, 1].map(|axis| lattice(0.0, extent[axis], cell).1.div_ceil(BLOCK_SIDE));
        let [columns, rows] = blocks.map(|count| count as usize);
        let size = cell * BLOCK_SIDE as f32;
        let block = |at: f32, count: usize| {
            (max((at / size).floor(), 0.0) as usize).min(count.saturating_sub(1))
        };
        let mut marked = vec![false; columns * rows];
        if let Some(hydrology) = layers.hydrology().filter(|_| !marked.is_empty()) {
            for course in hydrology.rivers().courses() {
                for (index, &[x, z]) in course.points().iter().enumerate() {
                    let reach = course.half(index) + cell;
                    for row in block(z - reach, rows)..=block(z + reach, rows) {
                        for column in block(x - reach, columns)..=block(x + reach, columns) {
                            marked[row * columns + column] = true;
                        }
                    }
                }
            }
            for row in 0..rows {
                for column in 0..columns {
                    let middle = (Vec2::new(column as f32, row as f32) + 0.5) * size;
                    if hydrology.water_at(middle.x, middle.y).is_some() {
                        marked[row * columns + column] = true;
                    }
                }
            }
        }
        let marked = spread(&marked, blocks);
        let stride = columns + 1;
        let mut counts = vec![0_u32; stride * (rows + 1)];
        for row in 0..rows {
            for column in 0..columns {
                counts[(row + 1) * stride + column + 1] = u32::from(marked[row * columns + column])
                    + counts[row * stride + column + 1]
                    + counts[(row + 1) * stride + column]
                    - counts[row * stride + column];
            }
        }
        Self { columns, counts }
    }

    /// The cells a region would store of the square over the cells along x
    /// and z from `(first, cells)` of each: those of the blocks any of them
    /// lies in.
    fn cells(&self, (first_x, cells_x): (i64, u32), (first_z, cells_z): (i64, u32)) -> u64 {
        if cells_x == 0 || cells_z == 0 {
            return 0;
        }
        let stride = self.columns + 1;
        let rows = self.counts.len() / stride - 1;
        let side = i64::from(BLOCK_SIDE);
        let range = |first: i64, cells: u32, count: usize| {
            let low = (first.max(0) / side) as usize;
            let high = ((first + i64::from(cells) - 1).max(0) / side) as usize + 1;
            (low.min(count), high.min(count))
        };
        let (x0, x1) = range(first_x, cells_x, self.columns);
        let (z0, z1) = range(first_z, cells_z, rows);
        let at = |x: usize, z: usize| u64::from(self.counts[z * stride + x]);
        (at(x1, z1) + at(x0, z0) - at(x0, z1) - at(x1, z0)) * BLOCK_CELLS as u64
    }
}

/// The blocks within [`SPREAD`] blocks along x and z of one `marked` marks,
/// of `blocks` along each, row by row.
fn spread(marked: &[bool], blocks: [u32; 2]) -> Vec<bool> {
    let [columns, rows] = blocks.map(|count| count as usize);
    let reach = SPREAD as usize;
    let mut along = vec![false; marked.len()];
    for (at, _) in marked.iter().enumerate().filter(|&(_, &mark)| mark) {
        let (x, row) = (at % columns, at - at % columns);
        along[row + x.saturating_sub(reach)..=row + (x + reach).min(columns - 1)].fill(true);
    }
    let mut spread = vec![false; marked.len()];
    for (at, _) in along.iter().enumerate().filter(|&(_, &mark)| mark) {
        let (x, z) = (at % columns, at / columns);
        for row in z.saturating_sub(reach)..=(z + reach).min(rows - 1) {
            spread[row * columns + x] = true;
        }
    }
    spread
}

/// A region whose water settles together: the meters across its settled
/// cells, and the live cells along each side of one, on which it settles
/// ([`shallow_water::live_split`]); the first settled cell along x and z of
/// its square on the terrain's lattice of settled cells, and its settled
/// cells along each; and the level-0 tiles within it whose water it keeps,
/// each a record, in key order.
#[derive(Clone, Debug, PartialEq)]
pub struct SettleRegion {
    pub cell: f32,
    pub split: u32,
    pub first: [i64; 2],
    pub cells: [u32; 2],
    pub kept: Vec<TileKey>,
}

impl SettleRegion {
    /// The region over `pyramid`'s level-0 tiles `tiles` (first column and
    /// row, columns and rows) in settled cells `cell` meters across, its
    /// square reaching [`MARGIN`] meters past them within the terrain,
    /// keeping the water of `kept`, tiles among them.
    pub fn over(pyramid: &Pyramid, tiles: [u32; 4], mut kept: Vec<TileKey>, cell: f32) -> Self {
        let [x, z, columns, rows] = tiles;
        let (first_x, along_x) = reach(pyramid, 0, (x, columns), cell);
        let (first_z, along_z) = reach(pyramid, 1, (z, rows), cell);
        kept.sort_unstable();
        kept.dedup();
        Self {
            cell,
            split: shallow_water::live_split(cell, pyramid.spacing(0)),
            first: [first_x, first_z],
            cells: [along_x, along_z],
            kept,
        }
    }

    /// The region level-0 tile `key` of `pyramid` settles over alone in
    /// settled cells `cell` meters across, keeping its water.
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

/// A level-0 tile's part of the water settling over the regions reaching it
/// ([`SettleStart::assemble`]): at each of its samples (0 to 256 along x and
/// z, row by row, the last the next tile's first), its ground as its record
/// holds it; its control value, and its smooth surface's gradient as a
/// normal map's texel holds it (the GPU's own words, `tile_shape`'s, with
/// which it raises the ground of the cells settling over it by its
/// materials' relief, as it raises the live water's from the tile's
/// layers); and the bed its water runs over, its roughness's word
/// ([`shallow_water::roughness`], as tiles' water's words hold it) and its
/// reach (0 outside channels, else one more than its [`Reach`]'s). At each
/// cell of the terrain's lattice of settled cells `cell` meters across whose
/// middle lies within it (its first along x and z and its cells along each,
/// row by row), the bake's water there: its surface (NaN where it holds
/// none), its velocity along x and z (meters a second), and 1 where a lake
/// holds it, else 0. And the solids the objects placed on it stand as, in
/// their order ([`crate::Placed::footprint`]).
#[derive(Clone, Debug, PartialEq)]
pub struct WaterPart {
    pub key: TileKey,
    pub cell: f32,
    pub heights: Vec<f32>,
    pub control: Vec<u32>,
    pub gradients: Vec<u32>,
    pub rough: Vec<f32>,
    pub reach: Vec<u8>,
    pub first: [i64; 2],
    pub cells: [u32; 2],
    pub water: Vec<[f32; 4]>,
    pub solids: Vec<Footprint>,
}

record!(WaterPart {
    key,
    cell,
    heights,
    control,
    gradients,
    rough,
    reach,
    first,
    cells,
    water,
    solids,
});

/// The code the reach of a cell over `bed` (`None` outside channels)
/// travels as: 0 outside channels, else one more than its [`Reach`]'s.
fn reach_code(bed: Option<&Bed>) -> u8 {
    bed.map_or(0, |bed| bed.reach as u8 + 1)
}

impl WaterPart {
    /// `tile`'s part, a level-0 tile of `pyramid` with its objects placed and
    /// its control values given, as its record holds it, on the lattice of
    /// settled cells `cell` meters across, its objects standing as their
    /// assets' `solids` (by asset; none for an asset whose objects stand as
    /// none).
    fn of(pyramid: &Pyramid, tile: &Tile, (cell, solids): (f32, &[Option<Solid>])) -> Self {
        let key = tile.key();
        let origin = pyramid.tile_origin(key);
        let size = pyramid.tile_size(0);
        let spacing = tile.spacing();
        let stride = TILE_STRIDE as usize;
        let stored = tile.heights();
        let count = SAMPLES * SAMPLES;
        let mut heights = Vec::with_capacity(count);
        let mut control = Vec::with_capacity(count);
        let mut gradients = Vec::with_capacity(count);
        let mut rough = Vec::with_capacity(count);
        let mut reach = Vec::with_capacity(count);
        for z in 0..SAMPLES {
            for x in 0..SAMPLES {
                // Stored rows and columns hold samples -1 to 257.
                let at = (z + 1) * stride + x + 1;
                heights.push(stored[at]);
                control.push(tile.control().map_or(0, |values| values[at]));
                let around = [z, z + 1, z + 2]
                    .map(|row| [x, x + 1, x + 2].map(|column| stored[row * stride + column]));
                gradients.push(gradient_word(smooth_gradient(around, spacing)));
                let bed = tile.bed(Vec2::new(x as f32, z as f32) * spacing);
                rough.push(shallow_water::roughness(bed.as_ref()));
                reach.push(reach_code(bed.as_ref()));
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
        let solids = tile
            .placed()
            .iter()
            .filter(|placed| placed.near_level.is_none())
            .filter_map(|placed| {
                let solid = solids.get(usize::from(placed.asset))?.as_ref()?;
                Some(placed.footprint(solid))
            })
            .filter(Footprint::valid)
            .collect();
        Self {
            key,
            cell,
            heights,
            control,
            gradients,
            rough,
            reach,
            first: [first_x, first_z],
            cells: [columns, rows],
            water,
            solids,
        }
    }

    /// Whether its arrays hold its samples and cells, on a lattice of cells
    /// of some size, its ground and beds' roughness finite, their reaches
    /// known and its solids finite and invertible.
    pub(crate) fn valid(&self) -> bool {
        let samples = SAMPLES * SAMPLES;
        let cells = self.cells[0] as usize * self.cells[1] as usize;
        self.key.level == 0
            && self.cell.is_finite()
            && self.cell > 0.0
            && self.heights.len() == samples
            && self.control.len() == samples
            && self.gradients.len() == samples
            && self.rough.len() == samples
            && self.reach.len() == samples
            && self.water.len() == cells
            && self.heights.iter().all(|height| height.is_finite())
            && self.rough.iter().all(|rough| rough.is_finite())
            && self
                .reach
                .iter()
                .all(|&reach| usize::from(reach) <= REACHES)
            && self.solids.iter().all(Footprint::valid)
    }
}

/// Making level-0 tile `key`'s part of the water settling over the regions
/// reaching it ([`WaterPart`]): generating its ground, with its control
/// values, and placing its objects, in steps a job's budget sizes, then
/// reading its samples, its bake's water and its objects' solids off it as
/// its record holds it.
pub(crate) struct PartMaking {
    pyramid: Pyramid,
    cell: f32,
    solids: Vec<Option<Solid>>,
    generator: Generator,
}

impl PartMaking {
    /// Tile `key`'s part over `layers` (with their bakes) and `pyramid`, its
    /// samples given control values and its objects placed by `rules`,
    /// standing as their assets' `solids`, of the lattice of settled cells
    /// `cell` meters across.
    pub(crate) fn new(
        (layers, rules): (Arc<Layers>, Arc<BiomeRules>),
        pyramid: Pyramid,
        key: TileKey,
        (cell, solids): (f32, Vec<Option<Solid>>),
    ) -> Self {
        Self {
            pyramid,
            cell,
            solids,
            generator: Generator::new(layers, &pyramid, key).with_surface(rules),
        }
    }

    /// Works for about `budget` units (a sample generated each): the part
    /// once done; an error where its ground does not generate or read back.
    pub(crate) fn advance(&mut self, budget: &mut u64) -> Result<Option<WaterPart>, String> {
        let Some(made) = self.generator.advance(budget) else {
            if self.generator.failed() {
                return Err("the ground's heights are invalid".into());
            }
            return Ok(None);
        };
        // As its record holds it: heights at their steps of the grid, which
        // the GPU's layers hold.
        let tile: Tile = codec::decode(&codec::encode(&made)).map_err(|error| error.to_string())?;
        Ok(Some(WaterPart::of(
            &self.pyramid,
            &tile,
            (self.cell, &self.solids),
        )))
    }
}

/// The parts a region settles from ([`WaterPart`]), by their tiles: the
/// terrain's level-0 tiles along x and z and the meters between their
/// samples; and the parts, row by row of the tiles holding them, from the
/// tile `first` along x and z, `columns` of them a row.
struct Reading<'a> {
    tiles: [u32; 2],
    spacing: f32,
    first: [u32; 2],
    columns: usize,
    parts: Vec<&'a WaterPart>,
}

impl<'a> Reading<'a> {
    /// `region`'s parts over `pyramid`, from `parts`, those of the tiles its
    /// square reaches in any order; an error where one it reaches is
    /// missing or on another lattice.
    fn of(
        pyramid: &Pyramid,
        region: &SettleRegion,
        parts: &[&'a WaterPart],
    ) -> Result<Self, String> {
        if parts.iter().any(|part| part.cell != region.cell) {
            return Err("a part of settling water on another lattice".into());
        }
        let reached = region.reached(pyramid);
        // The rectangle of tiles it reaches along an axis: its first and how
        // many.
        let span = |line: fn(&TileKey) -> u16| {
            let lines = reached.iter().map(|key| u32::from(line(key)));
            let (first, last) = (lines.clone().min(), lines.max());
            first.zip(last).map(|(first, last)| (first, (last + 1 - first) as usize))
        };
        let (Some((x0, columns)), Some((z0, rows))) = (span(|key| key.x), span(|key| key.z))
        else {
            return Err(MISSING.into());
        };
        let mut grid = vec![None; columns * rows];
        for &part in parts {
            let (x, z) = (u32::from(part.key.x), u32::from(part.key.z));
            if x < x0 || z < z0 {
                continue;
            }
            let (x, z) = ((x - x0) as usize, (z - z0) as usize);
            if x < columns && z < rows {
                grid[z * columns + x] = Some(part);
            }
        }
        let parts: Vec<&WaterPart> = grid.into_iter().collect::<Option<_>>().ok_or(MISSING)?;
        Ok(Self {
            tiles: pyramid.tiles(0),
            spacing: pyramid.spacing(0),
            first: [x0, z0],
            columns,
            parts,
        })
    }

    /// The part of level-0 tile `(x, z)`, where it reads one.
    fn part(&self, x: u32, z: u32) -> Option<&'a WaterPart> {
        let (x, z) = (x.checked_sub(self.first[0])?, z.checked_sub(self.first[1])?);
        if x as usize >= self.columns {
            return None;
        }
        self.parts
            .get(z as usize * self.columns + x as usize)
            .copied()
    }

    /// The terrain's last level-0 sample along x and z, counted from its
    /// first corner.
    fn last(&self) -> [i64; 2] {
        self.tiles.map(|count| i64::from(count) * i64::from(TILE_QUADS))
    }

    /// The part holding the terrain's level-0 sample `at` (counted from its
    /// first corner), the sample nearest it within the terrain's, and its
    /// index among the part's samples: the tile it lies in, the far edges'
    /// their last tiles'.
    fn sample(&self, at: [i64; 2]) -> Option<(&'a WaterPart, usize)> {
        let quads = i64::from(TILE_QUADS);
        let last = self.last();
        let [x, z] = [0, 1].map(|axis| at[axis].clamp(0, last[axis]));
        let [tx, tz] = [(x, 0), (z, 1)]
            .map(|(at, axis)| (at / quads).min(i64::from(self.tiles[axis]) - 1));
        let part = self.part(tx as u32, tz as u32)?;
        Some((part, ((z - tz * quads) * (SAMPLES as i64) + x - tx * quads) as usize))
    }

    /// The words the GPU raises the ground by at the terrain's level-0
    /// sample `at` (`meshes/water_setup.wgsl`'s `setup_raise`): its control
    /// value, its gradient's halves and the bits of its bed's roughness's
    /// word, then nothing; none where no part holds it.
    fn words(&self, at: [i64; 2]) -> [u32; 4] {
        self.sample(at).map_or([0; 4], |(part, index)| {
            [
                part.control[index],
                part.gradients[index],
                part.rough[index].to_bits(),
                0,
            ]
        })
    }

    /// The reach of the bed at the terrain's level-0 sample `at`, as its
    /// part holds it.
    fn reach(&self, at: [i64; 2]) -> u8 {
        self.sample(at).map_or(0, |(part, index)| part.reach[index])
    }

    /// The ground at terrain-local `local` on level 0's triangles, as the
    /// GPU reads it there from the tiles' layers (`terrain_level_height`'s
    /// arithmetic at level 0): the tile holding it by its samples, the cell
    /// of samples holding it within that tile and where in the cell it lies,
    /// on the cell's triangle; the far edges' in their last tiles' last
    /// cells. None where no part holds it.
    fn ground(&self, local: Vec2) -> f32 {
        let samples = (local / self.spacing).max(Vec2::ZERO);
        let along = |at: f32, tiles: u32| {
            let tile = ((at / TILE_QUADS as f32) as u32).min(tiles - 1);
            let within = clamp(at - (tile * TILE_QUADS) as f32, 0.0, TILE_QUADS as f32);
            let corner = (within as u32).min(TILE_QUADS - 1);
            (tile, corner as usize, within - corner as f32)
        };
        let (tx, cx, fx) = along(samples.x, self.tiles[0]);
        let (tz, cz, fz) = along(samples.y, self.tiles[1]);
        let Some(part) = self.part(tx, tz) else {
            return 0.0;
        };
        let height = |dx: usize, dz: usize| part.heights[(cz + dz) * SAMPLES + cx + dx];
        on_triangle([height(0, 0), height(1, 0), height(0, 1), height(1, 1)], fx, fz)
    }

    /// The bake's water at the terrain's settled cell `at` (counted from its
    /// first corner), as the part of the tile holding its middle holds it;
    /// none (NaN) where no part holds it.
    fn water(&self, at: [i64; 2]) -> [f32; 4] {
        let none = [f32::NAN, 0.0, 0.0, 0.0];
        let Some(part) = self.parts.first() else {
            return none;
        };
        let cell = part.cell;
        let size = self.spacing * TILE_QUADS as f32;
        let tile = |axis: usize| {
            let middle = (at[axis] as f32 + 0.5) * cell;
            (max((middle / size).floor(), 0.0) as u32).min(self.tiles[axis] - 1)
        };
        let Some(part) = self.part(tile(0), tile(1)) else {
            return none;
        };
        let (x, z) = (at[0] - part.first[0], at[1] - part.first[1]);
        if x < 0 || z < 0 || x >= i64::from(part.cells[0]) || z >= i64::from(part.cells[1]) {
            return none;
        }
        part.water[(z * i64::from(part.cells[0]) + x) as usize]
    }

    /// The height the region's heights stand above: the lowest of its
    /// parts' ground, whole meters down.
    fn datum(&self) -> f32 {
        let lowest = self
            .parts
            .iter()
            .flat_map(|part| part.heights.iter().copied())
            .fold(f32::INFINITY, |low, height| min(low, height));
        if lowest.is_finite() {
            lowest.floor()
        } else {
            0.0
        }
    }
}

/// A cell as the bake stands its water, from the ground at its corners
/// ([`Reading::corners`]) and the bake's water there ([`WaterPart`]'s): its
/// water (its surface, or its ground where dry, and its flow along x and z
/// in square meters a second), the level a lake holds it at (NaN where none
/// does), and whether its water covers it, no lake holding it.
fn baked(corners: [f32; 4], [surface, x, z, lake]: [f32; 4]) -> ([f32; 3], f32, bool) {
    let floor = 0.25 * (corners[0] + corners[1] + corners[2] + corners[3]);
    if surface.is_nan() || surface <= floor {
        ([floor, 0.0, 0.0], f32::NAN, false)
    } else if lake > 0.5 {
        ([surface, 0.0, 0.0], surface, false)
    } else {
        let depth = surface - floor;
        ([surface, depth * x, depth * z], f32::NAN, true)
    }
}

/// A cell as its region starts, its heights above `datum`, from the ground
/// at its corners and the bake's water there ([`baked`]): the ground at its
/// middle; its water to start from, as the bake stands it, or as `warm`
/// holds it where it holds the cell and no lake does (its depth over the
/// ground as it stands now and its velocity, dry where it was dry or a lake
/// held it); and the level a lake holds it at (NaN where none does).
fn started(
    corners: [f32; 4],
    found: [f32; 4],
    datum: f32,
    warm: impl FnOnce() -> Option<SteadyCell>,
) -> (f32, [f32; 3], f32) {
    let (water, held, _) = baked(corners, found);
    let [a, b, c, d] = corners.map(|height| height - datum);
    let bed = 0.25 * (a + b + c + d);
    let held = held - datum;
    let mut water = [water[0] - datum, water[1], water[2]];
    if held.is_nan()
        && let Some(cell) = warm()
    {
        water = match cell {
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
    (bed, water, held)
}

/// Where the baked rivers bring water to the rectangle from `origin`
/// (terrain-local) of `cells` along x and z `cell` meters across: at each
/// point of a river's course within it, what the river carries there less
/// what it carried at the point before within it (all it carries where it
/// starts at a head or a lake's outlet, or crosses into the rectangle;
/// nothing more where it starts at a confluence, whose rivers bring it),
/// gathered along the course until [`SOURCE_LEAST`] or more; spread over
/// the cells within half its width that its baked water covers, no lake
/// holding them (`covered`, by index, row by row), and where it covers none
/// there, further down the course: its water comes where its bake stands
/// water. Each cell's water is summed, cells in order.
fn sources(
    layers: &Layers,
    (origin, cells, cell): (Vec2, [u32; 2], f32),
    covered: &dyn Fn(usize) -> bool,
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
            let mut wet = Vec::new();
            for j in low.y..=high.y {
                for i in low.x..=high.x {
                    let middle = origin + (Vec2::new(i as f32, j as f32) + 0.5) * cell;
                    let index = (j * stride + i) as usize;
                    if middle.distance(at) <= reach && covered(index) {
                        wet.push(index as u32);
                    }
                }
            }
            // Where its baked water covers no cell there, what it gathered
            // goes on down the course.
            if wet.is_empty() {
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
            let share = gathered / wet.len() as f32;
            brought.extend(
                wet.into_iter()
                    .map(|index| (index, share, (way * share).to_array())),
            );
            gathered = 0.0;
        }
    }
    brought.sort_by_key(|&(cell, _, _)| cell);
    merged(brought)
}

/// What the solids `footprints` (terrain-local) cover of the cells `cell`
/// meters across of the terrain's lattice of them from its cell `first`,
/// `cells` along x and z: each cell they stand on (along x and z from
/// `first`), in order, with the share of it they cover and the mean height
/// their tops stand at over that share (terrain-local). A cell's share is
/// that of its [`COVER_POINTS`] along each side, a quarter of its spacing
/// apart, a solid stands on, each standing at the highest top over it;
/// solids standing in a cell's water block the flow through its faces by
/// that share, as deep as they stand in it, so the settled flow turns
/// around them. Every host finds the same: footprints by cell in a fixed
/// order, the points in theirs.
pub fn cover_cells(
    footprints: &[Footprint],
    (first, cells): ([i64; 2], [u32; 2]),
    cell: f32,
) -> Vec<([u32; 2], [f32; 2])> {
    // Each footprint by the cells its shadow's bounds reach: row, column,
    // footprint.
    let mut reached: Vec<(u32, u32, u32)> = Vec::new();
    for (index, footprint) in footprints.iter().enumerate() {
        let (middle, reach) = (footprint.middle(), footprint.reach());
        let along = |low: f32, high: f32, axis: usize| {
            let from = (low / cell).floor() as i64 - first[axis];
            let to = (high / cell).floor() as i64 - first[axis];
            from.max(0)..=to.min(i64::from(cells[axis]) - 1)
        };
        let (low, high) = (middle - reach, middle + reach);
        for j in along(low.y, high.y, 1) {
            for i in along(low.x, high.x, 0) {
                reached.push((j as u32, i as u32, index as u32));
            }
        }
    }
    reached.sort_unstable();
    let points = COVER_POINTS as f32;
    let mut covered = Vec::new();
    for run in reached.chunk_by(|a, b| (a.0, a.1) == (b.0, b.1)) {
        let (j, i) = (run[0].0, run[0].1);
        let corner = Vec2::new(
            (first[0] + i64::from(i)) as f32,
            (first[1] + i64::from(j)) as f32,
        ) * cell;
        let (mut count, mut tops) = (0_u32, 0.0_f32);
        for b in 0..COVER_POINTS {
            for a in 0..COVER_POINTS {
                let point = corner + (Vec2::new(a as f32, b as f32) + 0.5) * (cell / points);
                let top = run
                    .iter()
                    .filter_map(|&(_, _, index)| footprints[index as usize].span(point))
                    .fold(f32::NEG_INFINITY, |top, [_, high]| max(top, high));
                if top > f32::NEG_INFINITY {
                    count += 1;
                    tops += top;
                }
            }
        }
        if count > 0 {
            let share = count as f32 / (points * points);
            covered.push(([i, j], [share, tops / count as f32]));
        }
    }
    covered
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

/// Which blocks of a rectangle of cells settling stores, and where: the
/// rectangle's cells along x and z, and its blocks along each
/// ([`BLOCK_SIDE`] cells along each side from its first cell, the last
/// along x or z reaching past its last cell where its cells do not fill
/// them); for each block, row by row, the slot of its pool storing it
/// ([`NO_BLOCK`] where none does); and for each slot, the block it stores
/// (its column, and its row above 16 bits), in the blocks' order. A slot
/// holds [`BLOCK_CELLS`] cells, its block's row by row, and
/// [`BLOCK_GROUND`] entries of ground: the ground at its cells' corners
/// (`BLOCK_SIDE + 1` a row, rows along z), then the level a lake holds each
/// of its cells at ([`shallow_water::UNHELD`] where none does), each of its
/// cells' bed's roughness ([`shallow_water::roughness`]), the share of each
/// of its cells covered by solids and the height their tops stand at over
/// it ([`cover_cells`]; 0 each where none covers it), its cells row by row.
/// Cells and corners past the rectangle's last hold its last's.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredBlocks {
    pub cells: [u32; 2],
    pub blocks: [u32; 2],
    pub table: Vec<u32>,
    pub places: Vec<u32>,
}

impl StoredBlocks {
    /// The rectangle of `cells` along x and z, storing every block.
    pub fn whole(cells: [u32; 2]) -> Self {
        let blocks = cells.map(|count| count.div_ceil(BLOCK_SIDE));
        Self::marked(cells, &vec![true; blocks[0] as usize * blocks[1] as usize])
    }

    /// The rectangle of `cells` along x and z, storing the blocks `kept`
    /// marks (row by row).
    fn marked(cells: [u32; 2], kept: &[bool]) -> Self {
        let blocks = cells.map(|count| count.div_ceil(BLOCK_SIDE));
        let mut table = Vec::with_capacity(kept.len());
        let mut places = Vec::new();
        for (block, &stored) in (0_u32..).zip(kept) {
            if stored {
                table.push(places.len() as u32);
                places.push((block % blocks[0]) | ((block / blocks[0]) << 16));
            } else {
                table.push(NO_BLOCK);
            }
        }
        Self {
            cells,
            blocks,
            table,
            places,
        }
    }

    /// The slots it fills: the blocks it stores.
    pub fn stored(&self) -> usize {
        self.places.len()
    }

    /// The slot storing block `(x, z)` (its column and row), if the
    /// rectangle has it and one does.
    pub fn slot(&self, x: i64, z: i64) -> Option<usize> {
        let [columns, rows] = self.blocks.map(i64::from);
        if x < 0 || z < 0 || x >= columns || z >= rows {
            return None;
        }
        let slot = self.table[(z * columns + x) as usize];
        (slot != NO_BLOCK).then_some(slot as usize)
    }

    /// Where its slots hold the rectangle's cell `(i, j)`: its index among
    /// their cells, if its block is stored.
    pub fn at(&self, i: u32, j: u32) -> Option<usize> {
        if i >= self.cells[0] || j >= self.cells[1] {
            return None;
        }
        let slot = self.slot(i64::from(i / BLOCK_SIDE), i64::from(j / BLOCK_SIDE))?;
        Some(slot * BLOCK_CELLS + ((j % BLOCK_SIDE) * BLOCK_SIDE + i % BLOCK_SIDE) as usize)
    }

    /// The rectangle's cell its slots hold at index `at` (along x and z),
    /// its last along x or z for one past it.
    pub fn cell(&self, at: usize) -> [u32; 2] {
        let place = self.places[at / BLOCK_CELLS];
        let local = (at % BLOCK_CELLS) as u32;
        [
            ((place & 0xffff) * BLOCK_SIDE + local % BLOCK_SIDE).min(self.cells[0] - 1),
            ((place >> 16) * BLOCK_SIDE + local / BLOCK_SIDE).min(self.cells[1] - 1),
        ]
    }

    /// Its slots' cells, in order, each from `cell` at its place in the
    /// rectangle ([`Self::cell`]).
    pub fn gather<T>(&self, cell: impl Fn(u32, u32) -> T) -> Vec<T> {
        (0..self.stored() * BLOCK_CELLS)
            .map(|at| {
                let [i, j] = self.cell(at);
                cell(i, j)
            })
            .collect()
    }

    /// Its slots' ground, in order: `corner` at each of their cells'
    /// corners, by its place among the rectangle's corners (its last along
    /// x or z for one past it); then `held`, `rough`, and the shares and
    /// then the tops `cover` holds, an entry each for each cell its slots
    /// hold, in order.
    pub fn ground(
        &self,
        corner: impl Fn(u32, u32) -> f32,
        held: &[f32],
        (rough, cover): (&[f32], &[[f32; 2]]),
    ) -> Vec<f32> {
        let side = BLOCK_SIDE;
        let mut ground = Vec::with_capacity(self.stored() * BLOCK_GROUND);
        for (slot, &place) in self.places.iter().enumerate() {
            let (x, z) = ((place & 0xffff) * side, (place >> 16) * side);
            for k in 0..=side {
                for l in 0..=side {
                    ground.push(corner(
                        (x + l).min(self.cells[0]),
                        (z + k).min(self.cells[1]),
                    ));
                }
            }
            let cells = slot * BLOCK_CELLS..(slot + 1) * BLOCK_CELLS;
            ground.extend_from_slice(&held[cells.clone()]);
            ground.extend_from_slice(&rough[cells.clone()]);
            ground.extend(cover[cells.clone()].iter().map(|[share, _]| share));
            ground.extend(cover[cells].iter().map(|[_, top]| top));
        }
        ground
    }
}

/// What settling a region's water starts from ([`Self::assemble`]): the
/// meters across the cells it settles on and those along each side of a
/// settled cell, whose records it makes; the first cell along x and z of its
/// square, on the terrain's lattice of cells that size; the height its
/// heights stand above (its lowest ground, whole meters down, which keeps
/// them small for the GPU's single precision); the tiles whose water it
/// keeps, each with its settled cells within the square (its first along x
/// and z, counted from the square's first settled cell, and its cells along
/// each), in key order; the blocks of the square's cells it stores
/// ([`StoredBlocks`]), with their ground, bare, which the GPU raises by its
/// materials' relief before it steps them (`meshes/water_setup.wgsl`'s
/// `setup_raise`), each of their cells' water to start from (its surface, or
/// its ground where dry, and its flow along x and z in square meters a
/// second, then nothing) and the reach its bed lies in (0 outside channels,
/// else one more than its [`Reach`]'s), in their slots' order; the level-0
/// samples each stored block's corners lie among, `window` along each side
/// from its first (counted from the terrain's first corner, `windows`), the
/// words the GPU raises its ground by at each, row by row, block by block,
/// and the terrain's last sample along x and z; where its rivers bring
/// water, by the index of their cells among those stored; and the solids
/// standing over the blocks it stores, its parts', in their order.
#[derive(Clone, Debug, PartialEq)]
pub struct SettleStart {
    pub cell: f32,
    pub split: u32,
    pub first: [i64; 2],
    pub datum: f32,
    pub kept: Vec<(TileKey, [u32; 4])>,
    pub stored: StoredBlocks,
    pub ground: Vec<f32>,
    pub water: Vec<[f32; 4]>,
    pub reaches: Vec<u8>,
    pub window: u32,
    pub windows: Vec<[i64; 2]>,
    pub samples: Vec<[u32; 4]>,
    pub last: [i64; 2],
    pub sources: Vec<SteadySource>,
    pub solids: Vec<Footprint>,
}

impl SettleStart {
    /// What settling `region`'s water starts from over `layers` (with their
    /// bakes) and `pyramid`, from `parts`, those of the tiles its square
    /// reaches ([`SettleRegion::reached`]), in any order: the blocks of its
    /// live cells its water can reach; their ground on level 0's triangles
    /// at their corners as the GPU reads it, heights above its lowest
    /// ground, with the samples the GPU raises it by; the roughness's word
    /// and the reach of the bed at the sample nearest each cell's middle, as
    /// the live water's setup reads them; each of their cells' water as the
    /// bake stands and runs the settled cell holding it over the cell's own
    /// ground, or as `warm` holds it where its host holds earlier records of
    /// its water (the settled water of the terrain's lattice of settled
    /// cells' cell `(i, j)`, counted from its first corner, where one holds
    /// it); the settled cells of the tiles it keeps; and the solids of its
    /// parts standing over the blocks it stores, with what they cover of its
    /// cells ([`cover_cells`]), a cell they cover whole starting dry where
    /// they stand over the water it would start with. An error where a part
    /// it reaches is missing or on another lattice.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub fn assemble(
        layers: &Layers,
        pyramid: &Pyramid,
        region: &SettleRegion,
        parts: &[&WaterPart],
        warm: Option<&dyn Fn(i64, i64) -> Option<SteadyCell>>,
    ) -> Result<Self, String> {
        let split = region.split;
        let cell = region.cell / split as f32;
        let (first_x, cells_x) = finer((region.first[0], region.cells[0]), split);
        let (first_z, cells_z) = finer((region.first[1], region.cells[1]), split);
        let (first, cells) = ([first_x, first_z], [cells_x, cells_z]);
        let [nx, nz] = cells;
        if nx == 0 || nz == 0 {
            return Err("settling water over no cells".into());
        }
        let kept = kept_cells(pyramid, region)?;
        let reading = Reading::of(pyramid, region, parts)?;
        let datum = reading.datum();
        let spacing = reading.spacing;
        // The ground at the square's corner `(i, j)`, and at the corners of
        // its cell `(i, j)`: its own, then along x, along z and along both.
        let corner = |i: u32, j: u32| {
            let at = Vec2::new(
                (first[0] + i64::from(i)) as f32,
                (first[1] + i64::from(j)) as f32,
            );
            reading.ground(at * cell)
        };
        let corners = |i: u32, j: u32| {
            [
                corner(i, j),
                corner(i + 1, j),
                corner(i, j + 1),
                corner(i + 1, j + 1),
            ]
        };
        // The settled cell holding the square's cell `(i, j)`, on the
        // terrain's lattice of settled cells, and the records' water there.
        let settled = |i: u32, j: u32| {
            [
                region.first[0] + i64::from(i / split),
                region.first[1] + i64::from(j / split),
            ]
        };
        let warmed = |at: [i64; 2]| warm.and_then(|warm| warm(at[0], at[1]));
        // The blocks to store: those holding water, a lake or cells rivers
        // bring water to, and those around them.
        let blocks = cells.map(|count| count.div_ceil(BLOCK_SIDE));
        let mut holding = holding(&reading, region, datum, &warmed);
        let origin = Vec2::new(first[0] as f32, first[1] as f32) * cell;
        let covered = |at: usize| {
            let (i, j) = (at as u32 % nx, at as u32 / nx);
            baked(corners(i, j), reading.water(settled(i, j))).2
        };
        let sources = sources(layers, (origin, cells, cell), &covered);
        for source in &sources {
            let (i, j) = (source.cell % nx, source.cell / nx);
            holding[((j / BLOCK_SIDE) * blocks[0] + i / BLOCK_SIDE) as usize] = true;
        }
        let stored = StoredBlocks::marked(cells, &spread(&holding, blocks));
        let count = stored.stored() * BLOCK_CELLS;
        // The solids standing over the blocks it stores, and what they cover
        // of their cells, their tops over the datum.
        let block = |at: f32, axis: usize| {
            ((at / cell).floor() as i64 - first[axis]).div_euclid(i64::from(BLOCK_SIDE))
        };
        let solids: Vec<Footprint> = reading
            .parts
            .iter()
            .flat_map(|part| part.solids.iter().copied())
            .filter(|solid| {
                let (low, high) = (
                    solid.middle() - solid.reach(),
                    solid.middle() + solid.reach(),
                );
                (block(low.y, 1)..=block(high.y, 1)).any(|z| {
                    (block(low.x, 0)..=block(high.x, 0)).any(|x| stored.slot(x, z).is_some())
                })
            })
            .collect();
        let mut cover = vec![[0.0_f32; 2]; count];
        for ([i, j], [share, top]) in cover_cells(&solids, (first, cells), cell) {
            if let Some(at) = stored.at(i, j) {
                cover[at] = [share, top - datum];
            }
        }
        let mut water = Vec::with_capacity(count);
        let mut held = Vec::with_capacity(count);
        let mut rough = Vec::with_capacity(count);
        let mut reaches = Vec::with_capacity(count);
        for at in 0..count {
            let [i, j] = stored.cell(at);
            let holder = settled(i, j);
            let found = reading.water(holder);
            let (bed, start, lake) = started(corners(i, j), found, datum, || warmed(holder));
            let [share, top] = cover[at];
            let start = if share >= 1.0 && top >= start[0] {
                [bed, 0.0, 0.0]
            } else {
                start
            };
            water.push([start[0], start[1], start[2], 0.0]);
            held.push(if lake.is_nan() { UNHELD } else { lake });
            // The bed at the sample nearest the cell's middle.
            let middle = (Vec2::new(
                (first[0] + i64::from(i)) as f32,
                (first[1] + i64::from(j)) as f32,
            ) + 0.5)
                * cell
                / spacing;
            let nearest = [middle.x, middle.y].map(|at| (at + 0.5).floor() as i64);
            rough.push(f32::from_bits(reading.words(nearest)[2]));
            reaches.push(reading.reach(nearest));
        }
        let ground = stored.ground(|i, j| corner(i, j) - datum, &held, (&rough, &cover));
        // The samples each block's corners lie among: from the cell of
        // samples holding its first corner, as the GPU finds it, as many
        // along each side as its corners span and one past each end.
        let last = reading.last();
        let window = (BLOCK_SIDE as f32 * cell / spacing).ceil() as u32 + 2;
        let mut windows = Vec::with_capacity(stored.stored());
        let mut samples = Vec::with_capacity(stored.stored() * (window * window) as usize);
        for &place in &stored.places {
            let block = [place & 0xffff, place >> 16];
            let start = [0, 1].map(|axis| {
                let lattice = first[axis] + i64::from(block[axis] * BLOCK_SIDE);
                let at = max(lattice as f32 * cell / spacing, 0.0);
                (at.floor() as i64).min(last[axis] - 1)
            });
            windows.push(start);
            for z in 0..i64::from(window) {
                for x in 0..i64::from(window) {
                    samples.push(reading.words([start[0] + x, start[1] + z]));
                }
            }
        }
        let sources = sources
            .into_iter()
            .filter_map(|source| {
                let at = stored.at(source.cell % nx, source.cell / nx)?;
                Some(SteadySource {
                    cell: at as u32,
                    ..source
                })
            })
            .collect();
        Ok(Self {
            cell,
            split,
            first,
            datum,
            kept,
            stored,
            ground,
            water,
            reaches,
            window,
            windows,
            samples,
            last,
            sources,
            solids,
        })
    }

    /// Its stored ground as it starts, bare.
    fn bare(&self) -> SlotGround<'_> {
        SlotGround(&self.ground)
    }

    /// Its stored cell `at`'s ground as it starts: the mean of its corners'.
    pub fn bed(&self, at: usize) -> f32 {
        self.bare().bed(at)
    }

    /// The level a lake holds its stored cell `at` at as it starts, if one
    /// does.
    pub fn held(&self, at: usize) -> Option<f32> {
        self.bare().held(at)
    }

    /// Whether the water settled to `means` ([`Self::finish`]'s) over
    /// `ground` stands in the cell `(i, j)` of its square: a lake holds it,
    /// or water stands in it most of the time; none where its block is not
    /// stored.
    fn wet(&self, (means, ground): (&[[f32; 4]], SlotGround<'_>), i: u32, j: u32) -> bool {
        let Some(at) = self.stored.at(i, j) else {
            return false;
        };
        let Some(&[water, _, shares]) = means.get(at * CELL_MEANS..(at + 1) * CELL_MEANS) else {
            return false;
        };
        ground.held(at).is_some() || (water[0] - ground.bed(at) > DRY && shares[0] >= WET_MOST)
    }

    /// Of its solids, those standing in the water settled to `means` over
    /// `ground`: any cell of its square within a cell of their shadow from
    /// above holds water.
    fn standing(&self, settled: (&[[f32; 4]], SlotGround<'_>)) -> Vec<bool> {
        let [nx, nz] = self.stored.cells;
        let cells = |low: f32, high: f32, axis: usize, count: u32| {
            let from = (low / self.cell).floor() as i64 - self.first[axis] - 1;
            let to = (high / self.cell).floor() as i64 - self.first[axis] + 1;
            from.max(0)..=to.min(i64::from(count) - 1)
        };
        self.solids
            .iter()
            .map(|solid| {
                let (low, high) = (
                    solid.middle() - solid.reach(),
                    solid.middle() + solid.reach(),
                );
                cells(low.y, high.y, 1, nz).any(|j| {
                    cells(low.x, high.x, 0, nx).any(|i| self.wet(settled, i as u32, j as u32))
                })
            })
            .collect()
    }

    /// What each tile it keeps keeps of its water settled to `means` (each
    /// stored cell's [`CELL_MEANS`] entries of its means over its region's
    /// last steps, in its slots' order) over `ground` (its slots' ground as
    /// the GPU raised it, as [`StoredBlocks::ground`] lays it out), after
    /// `steps` steps and `settled` or not, in the order it keeps them: each
    /// of the tile's own settled cells the mean of the water of its cells,
    /// standing at its lake's level where a lake holds most of them, or
    /// holds some and water stands in none of the others; else, where water
    /// stands in any of them most of the time, as it stands there while wet,
    /// their mean surface, their velocity their mean flow over their mean
    /// depth, and their means of the energy they lose and of their own
    /// spreads about their means; dry elsewhere and where its blocks are not
    /// stored. With them the water its rivers bring to its settled cells,
    /// cells in order; its summary, each flowing cell breaking white as its
    /// water does at its depth; how its water settled, its stats counting
    /// its region's cells within it; and the solids standing in its water
    /// whose middles its settled cells hold.
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub fn finish(
        &self,
        means: &[[f32; 4]],
        ground: &[f32],
        steps: u32,
        settled: bool,
    ) -> Vec<SteadyWater> {
        let raised = SlotGround(ground);
        let split = self.split;
        let area = f64::from(self.cell * self.cell);
        let record = self.cell * split as f32;
        let record_first = self.first.map(|first| first.div_euclid(i64::from(split)));
        let standing = self.standing((means, raised));
        let mut made = Vec::with_capacity(self.kept.len());
        for &(key, kept) in &self.kept {
            let [x0, z0, columns, rows] = kept;
            let count = columns as usize * rows as usize;
            let unknown = || vec![f32::NAN; count];
            let mut surface = unknown();
            let mut velocities = [unknown(), unknown()];
            let mut energy = unknown();
            let mut heave = unknown();
            let mut swirl = unknown();
            let mut rush = unknown();
            let mut wet = vec![0.0; count];
            let mut lip = unknown();
            let mut white = unknown();
            let (mut flowing, mut clamped) = (0, 0);
            let mut volume = 0.0_f64;
            // Each reach's channel cells flowing: their Manning's n summed,
            // those running faster than their waves, and their count.
            let mut friction = [0.0_f64; REACHES];
            let mut fast = [0_u32; REACHES];
            let mut counted = [0_u32; REACHES];
            for row in 0..rows {
                for column in 0..columns {
                    let out = (row * columns + column) as usize;
                    let mut sum = Sum::default();
                    for b in 0..split {
                        for a in 0..split {
                            let (i, j) = ((x0 + column) * split + a, (z0 + row) * split + b);
                            let Some(at) = self.stored.at(i, j) else {
                                continue;
                            };
                            let Some(&[water, motion, shares]) =
                                means.get(at * CELL_MEANS..(at + 1) * CELL_MEANS)
                            else {
                                continue;
                            };
                            sum.share += shares[0];
                            if let Some(level) = raised.held(at) {
                                sum.held += 1;
                                sum.level = level;
                                continue;
                            }
                            // Its water stands where it stands most of the
                            // time, as it stands while wet.
                            let bed = raised.bed(at);
                            let depth = water[0] - bed;
                            if depth <= DRY || shares[0] < WET_MOST {
                                continue;
                            }
                            sum.add((bed, depth), [water, motion, shares]);
                            flowing += 1;
                            clamped += u32::from(shares[2] > 0.0);
                            volume += f64::from(depth);
                            let Some(reach) = usize::from(self.reaches[at]).checked_sub(1) else {
                                continue;
                            };
                            // Manning's n holding the water back as hard as
                            // the scheme's friction does: k = g·n²/h^(7/3).
                            let moving = [velocity(depth, water[1]), velocity(depth, water[2])];
                            let held_back =
                                resistance(depth, raised.rough(at)) * raised.lean(at, self.cell);
                            let n = (held_back * depth * depth * depth.cbrt() / GRAVITY).sqrt();
                            let speed = moving[0] * moving[0] + moving[1] * moving[1];
                            friction[reach] += f64::from(n);
                            fast[reach] += u32::from(speed > GRAVITY * depth);
                            counted[reach] += 1;
                        }
                    }
                    wet[out] = sum.share / (split * split) as f32;
                    if 2 * sum.held > split * split || (sum.held > 0 && sum.wet == 0) {
                        surface[out] = sum.level + self.datum;
                        continue;
                    }
                    if sum.wet == 0 {
                        continue;
                    }
                    let n = sum.wet as f32;
                    let depth = sum.depth / n;
                    surface[out] = sum.bed / n + depth + self.datum;
                    velocities[0][out] = velocity(depth, sum.flow[0] / n);
                    velocities[1][out] = velocity(depth, sum.flow[1] / n);
                    energy[out] = sum.energy / n;
                    heave[out] = (sum.heave / n).sqrt();
                    swirl[out] = (sum.swirl / n).sqrt();
                    rush[out] = (sum.rush / n).sqrt();
                    wet[out] = sum.wet_share / n;
                    lip[out] = sum.lip / n;
                    white[out] = breaking(energy[out], depth, lip[out]);
                }
            }
            // What its rivers bring to its settled cells, each one's
            // cells' summed, the way they bring it their rate-weighted
            // mean.
            let mut brought: Vec<(u32, f32, [f32; 2])> = self
                .sources
                .iter()
                .filter_map(|source| {
                    let [i, j] = self.stored.cell(source.cell as usize);
                    let (i, j) = (i / split, j / split);
                    let inside = i >= x0 && j >= z0 && i < x0 + columns && j < z0 + rows;
                    let way = [source.way[0] * source.rate, source.way[1] * source.rate];
                    inside.then(|| ((j - z0) * columns + i - x0, source.rate, way))
                })
                .collect();
            brought.sort_by_key(|&(cell, _, _)| cell);
            let sources = merged(brought);
            let solids = self
                .solids
                .iter()
                .zip(&standing)
                .filter(|&(solid, &stands)| {
                    let middle = solid.middle() / record;
                    let i = middle.x.floor() as i64 - record_first[0] - i64::from(x0);
                    let j = middle.y.floor() as i64 - record_first[1] - i64::from(z0);
                    stands
                        && (0..i64::from(columns)).contains(&i)
                        && (0..i64::from(rows)).contains(&j)
                })
                .map(|(&solid, _)| solid)
                .collect();
            let cells = [columns, rows];
            let summary = SteadySummary::of(cells, &velocities, (&white, &swirl));
            let mean = |total: f64, count: u32| {
                if count > 0 {
                    (total / f64::from(count)) as f32
                } else {
                    0.0
                }
            };
            let [along_x, along_z] = velocities;
            made.push(SteadyWater {
                key,
                cell: record,
                first: [
                    record_first[0] + i64::from(x0),
                    record_first[1] + i64::from(z0),
                ],
                cells,
                surface: surface.into(),
                velocity: [along_x.into(), along_z.into()],
                energy: energy.into(),
                heave: heave.into(),
                swirl: swirl.into(),
                rush: rush.into(),
                wet: wet.into(),
                lip: lip.into(),
                stats: SteadyStats {
                    sources: sources.iter().map(|source| source.rate).sum(),
                    volume: (volume * area) as f32,
                    flowing,
                    steps,
                    settled,
                    clamped,
                    friction: std::array::from_fn(|reach| mean(friction[reach], counted[reach])),
                    fast: std::array::from_fn(|reach| mean(f64::from(fast[reach]), counted[reach])),
                    reaches: counted,
                },
                sources,
                summary,
                solids,
            });
        }
        made
    }
}

/// A settling region's ground as its slots hold it ([`StoredBlocks::ground`]'s
/// layout): bare as it starts, or raised as the GPU raised it.
#[derive(Clone, Copy)]
struct SlotGround<'a>(&'a [f32]);

impl SlotGround<'_> {
    /// The ground at its cell `at`'s corners: its own, then along x, along
    /// z and along both.
    fn corners(self, at: usize) -> [f32; 4] {
        let side = BLOCK_SIDE as usize;
        let ground = &self.0[(at / BLOCK_CELLS) * BLOCK_GROUND..];
        let (i, j) = (at % side, at % BLOCK_CELLS / side);
        let corner = |x: usize, z: usize| ground[z * (side + 1) + x];
        [
            corner(i, j),
            corner(i + 1, j),
            corner(i, j + 1),
            corner(i + 1, j + 1),
        ]
    }

    /// Its cell `at`'s ground: the mean of its corners'.
    fn bed(self, at: usize) -> f32 {
        let [a, b, c, d] = self.corners(at);
        0.25 * (a + b + c + d)
    }

    /// How its cell `at`'s ground leans, its cells `cell` meters across:
    /// φ² = 1 + |∇b|², its slope along x and z between its faces' middles.
    fn lean(self, at: usize, cell: f32) -> f32 {
        let [a, b, c, d] = self.corners(at);
        let across = 0.5 * ((b + d) - (a + c)) / cell;
        let along = 0.5 * ((c + d) - (a + b)) / cell;
        1.0 + across * across + along * along
    }

    /// The level a lake holds its cell `at` at, if one does.
    fn held(self, at: usize) -> Option<f32> {
        let level = self.0[(at / BLOCK_CELLS) * BLOCK_GROUND + BLOCK_CORNERS + at % BLOCK_CELLS];
        (level > 0.5 * UNHELD).then_some(level)
    }

    /// Its cell `at`'s bed's roughness, as the scheme reads it.
    fn rough(self, at: usize) -> f32 {
        let roughness = BLOCK_CORNERS + BLOCK_CELLS;
        self.0[(at / BLOCK_CELLS) * BLOCK_GROUND + roughness + at % BLOCK_CELLS]
    }
}

/// The water of the cells over a settled cell ([`SettleStart::finish`]):
/// the shares of the time they were wet, summed; those a lake holds, and its
/// level; and of those water stands in most of the time, how many, their
/// ground, depth and flow along x and z, the energy they lose, the variances
/// of their depth and velocity and the mean squares of their vorticity, and
/// the shares of the time they were wet and their water left their bed,
/// each summed.
#[derive(Default)]
struct Sum {
    share: f32,
    held: u32,
    level: f32,
    wet: u32,
    bed: f32,
    depth: f32,
    flow: [f32; 2],
    energy: f32,
    heave: f32,
    swirl: f32,
    rush: f32,
    wet_share: f32,
    lip: f32,
}

impl Sum {
    /// Adds a cell water stands in most of the time, over ground `bed`,
    /// `depth` deep, its means `water`, `motion` and `shares`
    /// ([`CELL_MEANS`]' entries).
    fn add(&mut self, (bed, depth): (f32, f32), [water, motion, shares]: [[f32; 4]; 3]) {
        self.wet += 1;
        self.bed += bed;
        self.depth += depth;
        self.flow = [self.flow[0] + water[1], self.flow[1] + water[2]];
        self.energy += motion[0];
        self.heave += max(motion[1], 0.0);
        self.swirl += max(motion[2], 0.0);
        self.rush += max(motion[3], 0.0);
        self.wet_share += shares[0];
        self.lip += shares[1];
    }
}

/// The settled cells within `region`'s square of each tile it keeps (its
/// first along x and z, counted from the square's first, and its cells along
/// each), in its order, over `pyramid`; an error where one lies outside the
/// square.
fn kept_cells(
    pyramid: &Pyramid,
    region: &SettleRegion,
) -> Result<Vec<(TileKey, [u32; 4])>, String> {
    let size = pyramid.tile_size(0);
    let [nx, nz] = region.cells.map(i64::from);
    region
        .kept
        .iter()
        .map(|&key| {
            let origin = pyramid.tile_origin(key);
            let (x, columns) = lattice(origin.x, origin.x + size, region.cell);
            let (z, rows) = lattice(origin.y, origin.y + size, region.cell);
            let (x, z) = (x - region.first[0], z - region.first[1]);
            if x < 0 || z < 0 || x + i64::from(columns) > nx || z + i64::from(rows) > nz {
                return Err("a tile settling water keeps lies outside its region".to_owned());
            }
            Ok((key, [x as u32, z as u32, columns, rows]))
        })
        .collect()
}

/// The blocks of `region`'s cells (its live cells), row by row, holding
/// water or a lake as it starts ([`started`], its heights above `datum`,
/// over the ground `reading` reads and from the records `warmed` gives of
/// its settled cells): those of the cells of each settled cell holding water
/// or a lake over its own ground.
fn holding(
    reading: &Reading<'_>,
    region: &SettleRegion,
    datum: f32,
    warmed: &dyn Fn([i64; 2]) -> Option<SteadyCell>,
) -> Vec<bool> {
    let [nx, nz] = region.cells;
    let split = region.split;
    let columns = (nx * split).div_ceil(BLOCK_SIDE);
    let rows = (nz * split).div_ceil(BLOCK_SIDE);
    let mut holding = vec![false; columns as usize * rows as usize];
    // The ground at the settled corners along row `j` of the square into
    // `row`.
    let corner_row = |j: u32, row: &mut Vec<f32>| {
        row.clear();
        row.extend((0..=nx).map(|i| {
            let at = Vec2::new(
                (region.first[0] + i64::from(i)) as f32,
                (region.first[1] + i64::from(j)) as f32,
            );
            reading.ground(at * region.cell)
        }));
    };
    let (mut below, mut above) = (Vec::new(), Vec::new());
    corner_row(0, &mut below);
    for j in 0..nz {
        corner_row(j + 1, &mut above);
        for i in 0..nx {
            let x = i as usize;
            let corners = [below[x], below[x + 1], above[x], above[x + 1]];
            let at = [region.first[0] + i64::from(i), region.first[1] + i64::from(j)];
            let (bed, water, held) = started(corners, reading.water(at), datum, || warmed(at));
            if !held.is_nan() || water[0] - bed > DRY {
                let (left, right) = (i * split / BLOCK_SIDE, ((i + 1) * split - 1) / BLOCK_SIDE);
                let (low, high) = (j * split / BLOCK_SIDE, ((j + 1) * split - 1) / BLOCK_SIDE);
                for z in low..=high {
                    for x in left..=right {
                        holding[(z * columns + x) as usize] = true;
                    }
                }
            }
        }
        std::mem::swap(&mut below, &mut above);
    }
    holding
}

/// Squares along each side of a level-0 tile its settled water's summary
/// holds, each four of the tile's quads across, as far apart as level 2's
/// samples: what coarser levels' water runs by, each sample
/// averaging those under its footprint ([`Tile::flowing`]).
pub const SUMMARY_SIDE: u32 = 64;

/// A level-0 tile's settled water seen from afar ([`SteadyWater::summary`]):
/// over [`SUMMARY_SIDE`] squares along each side of the tile, row by row,
/// of the cells over each square, its flowing cells' mean velocity along x
/// and z (meters a second), the mean of how strongly they break white (0 to
/// 1, each as its water does at its own depth) and mean swirl (a second),
/// NaN where none flows. Each is held, in memory as in its record, as the
/// bits of the half float nearest it, which keep NaN.
#[derive(Clone, Debug, PartialEq)]
pub struct SteadySummary {
    velocity: [Box<[u16]>; 2],
    white: Box<[u16]>,
    swirl: Box<[u16]>,
}

/// A square of a [`SteadySummary`]: where water flows in it, its mean
/// velocity, white water and swirl.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SteadySquare {
    pub velocity: Option<[f32; 2]>,
    pub white: Option<f32>,
    pub swirl: Option<f32>,
}

impl SteadySummary {
    /// The summary of a tile's `cells` (along x and z) holding `velocity`
    /// and `swirl` ([`SteadyWater`]'s, row by row) and `white`, how strongly
    /// each of its flowing cells breaks white: each cell counts in every
    /// square it overlaps, so a square finer than the cells takes the cells
    /// over it.
    fn of(cells: [u32; 2], velocity: &[Vec<f32>; 2], (white, swirl): (&[f32], &[f32])) -> Self {
        let side = SUMMARY_SIDE as usize;
        let mut sums = vec![[0.0_f32; 5]; side * side];
        let (columns, rows) = (cells[0] as usize, cells[1] as usize);
        // The squares along an axis that cell `at` of `count` overlaps.
        let over = |at: usize, count: usize| at * side / count..((at + 1) * side).div_ceil(count);
        for row in 0..rows {
            for column in 0..columns {
                let at = row * columns + column;
                for z in over(row, rows) {
                    for x in over(column, columns) {
                        let sum = &mut sums[z * side + x];
                        if !velocity[0][at].is_nan() {
                            sum[0] += velocity[0][at];
                            sum[1] += velocity[1][at];
                            sum[2] += 1.0;
                            sum[3] += white[at];
                            sum[4] += swirl[at];
                        }
                    }
                }
            }
        }
        let mean =
            |total: f32, count: f32| f16_bits(if count > 0.0 { total / count } else { f32::NAN });
        Self {
            velocity: [
                sums.iter().map(|sum| mean(sum[0], sum[2])).collect(),
                sums.iter().map(|sum| mean(sum[1], sum[2])).collect(),
            ],
            white: sums.iter().map(|sum| mean(sum[3], sum[2])).collect(),
            swirl: sums.iter().map(|sum| mean(sum[4], sum[2])).collect(),
        }
    }

    /// Square `(x, z)` of the summary, if inside.
    pub fn square(&self, x: u32, z: u32) -> Option<SteadySquare> {
        if x >= SUMMARY_SIDE || z >= SUMMARY_SIDE {
            return None;
        }
        let at = (z * SUMMARY_SIDE + x) as usize;
        let velocity = [
            f16_value(self.velocity[0][at]),
            f16_value(self.velocity[1][at]),
        ];
        let flowing = !velocity[0].is_nan();
        Some(SteadySquare {
            velocity: flowing.then_some(velocity),
            white: flowing.then(|| f16_value(self.white[at])),
            swirl: flowing.then(|| f16_value(self.swirl[at])),
        })
    }
}

impl codec::Codec for SteadySummary {
    fn write(&self, out: &mut Writer) {
        let stride = SUMMARY_SIDE as usize;
        for values in [
            &self.velocity[0],
            &self.velocity[1],
            &self.white,
            &self.swirl,
        ] {
            codec::put_wide_planes(out, values, stride);
        }
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, codec::CodecError> {
        let stride = SUMMARY_SIDE as usize;
        let x = codec::get_wide_planes(input, stride)?;
        let z = codec::get_wide_planes(input, stride)?;
        let white = codec::get_wide_planes(input, stride)?;
        let swirl = codec::get_wide_planes(input, stride)?;
        if [x.len(), z.len(), white.len(), swirl.len()] != [stride * stride; 4] {
            return Err(codec::CodecError("invalid settled water summary"));
        }
        Ok(Self {
            velocity: [x, z],
            white,
            swirl,
        })
    }
}

/// The bits of the half float nearest `value` (ties to even), as a
/// summary holds its values, and an error pyramid its bounds before they
/// round up ([`crate::ErrorPyramid`]): NaN stays NaN, and finite values
/// beyond the largest half float hold to it.
pub(crate) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let biased = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7f_ffff;
    if biased == 0xff {
        // Infinite, or a quiet NaN.
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    let exponent = biased as i32 - 127 + 15;
    // The half's bits before rounding, the mantissa they come from and the
    // bits of it rounding takes away: normal halves keep its top ten bits
    // under their exponent; subnormal ones shift it, its implicit bit with
    // it, into units of 2^-24.
    let (kept, whole, shift) = if exponent > 0 {
        (((exponent as u32) << 10) | (mantissa >> 13), mantissa, 13)
    } else if exponent >= -10 {
        let whole = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        (whole >> shift, whole, shift)
    } else {
        return sign;
    };
    let rest = whole & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    // A carry out of the mantissa raises the exponent, as it should.
    let rounded = kept + u32::from(rest > halfway || (rest == halfway && (kept & 1) == 1));
    sign | rounded.min(0x7bff) as u16
}

/// The value of the half float whose bits are `bits` ([`f16_bits`]'s),
/// exactly: NaN for NaN.
pub(crate) fn f16_value(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let magnitude = match exponent {
        // Subnormal: its mantissa counts 2^-24s.
        0 => (f32::from(bits & 0x3ff) / 16_777_216.0).to_bits(),
        0x1f => 0x7f80_0000 | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// The span each sample of a tile packs the settled water standing on it
/// with, in the layout of its samples, its own water being `own`: what its
/// own water's words hold where they hold a surface, wet or continued under
/// the banks, and where the settled water spread past them, what those of
/// the sample nearest it holding one hold ([`nearest`]), the water body it
/// spread from; the least span the words tell apart ([`WATER_SPAN_LEAST`])
/// in a tile whose own water holds none.
fn spans(own: Option<&TileWater>) -> Vec<f32> {
    let count = (TILE_STRIDE * TILE_STRIDE) as usize;
    let Some(own) = own else {
        return vec![WATER_SPAN_LEAST; count];
    };
    let held: Vec<bool> = own
        .surfaces()
        .iter()
        .map(|level| level.is_finite())
        .collect();
    nearest(&held)
        .iter()
        .map(|&seed| {
            if seed == NONE {
                WATER_SPAN_LEAST
            } else {
                unpack(own.info()[seed as usize][0]).2
            }
        })
        .collect()
}

/// The values cells or squares blend into a sample by: whether a record
/// holds any of them; the weights of those holding water, of those a lake
/// holds and of those flowing; and what they hold, each by its weight.
struct Blend {
    known: bool,
    weight: f32,
    held: f32,
    level: f32,
    moving: f32,
    flow: Vec2,
    turbulence: f32,
    white: f32,
}

impl Blend {
    const NONE: Self = Self {
        known: false,
        weight: 0.0,
        held: 0.0,
        level: 0.0,
        moving: 0.0,
        flow: Vec2::ZERO,
        turbulence: 0.0,
        white: 0.0,
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

/// The squares of a lattice `square` meters apart whose first lies half a
/// step past the terrain's first corner that a sample at terrain-local
/// `point`, `spacing` meters from its neighbors, averages over: those its
/// tent covers, nearer it than its spacing along x and along z (than a
/// square, where its spacing is finer), each one's indices and the share it
/// blends in by, how near it lies along x times along z, rows first. A tent
/// reaching a square either way covers the four squares around the point,
/// blending them as [`around`] does.
fn tent(point: Vec2, spacing: f32, square: f32) -> impl Iterator<Item = (i64, i64, f32)> {
    // Squares the tent reaches either way.
    let reach = max(spacing, square) / square;
    let covered = move |at: f32| {
        let first = (at - reach).floor() as i64 + 1;
        let last = (at + reach).ceil() as i64 - 1;
        (first..=last).filter_map(move |index| {
            let near = 1.0 - (index as f32 - at).abs() / reach;
            (near > 0.0).then_some((index, near))
        })
    };
    let (u, v) = (point.x / square - 0.5, point.y / square - 0.5);
    covered(v)
        .flat_map(move |(j, along_z)| covered(u).map(move |(i, along_x)| (i, j, along_x * along_z)))
}

/// The generalized minmod of `a`, `b` and `c`, as the shallow-water scheme
/// limits its slopes by: the least of them where all are positive, the
/// greatest where all are negative, else 0.
fn minmod(a: f32, b: f32, c: f32) -> f32 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        min(a, min(b, c))
    } else if a < 0.0 && b < 0.0 && c < 0.0 {
        max(a, max(b, c))
    } else {
        0.0
    }
}

/// A settled cell as the samples around it blend it ([`Planes`]): whether a
/// record holds it and water stands in it, and whether a lake holds that
/// water; the plane of its surface, its level and its own slope along x and
/// z (meters a meter: flat where a lake holds it, else as the shallow-water
/// scheme reconstructs it along each, the central difference of the levels
/// of its neighbors holding water limited by the generalized minmod so no
/// plane rises past them where the water falls away, one-sided where one
/// of them holds water, and flat where neither does); and its water's
/// velocity, turbulence and white water ([`turbulence`], and
/// [`breaking`] at its depth over its bed, the mean of the ground at its
/// corners as its region's cells stand on it, and as often as it leaves the
/// bed).
#[derive(Clone, Copy, Default)]
struct Plane {
    known: bool,
    wet: bool,
    held: bool,
    level: f32,
    slope: Vec2,
    velocity: Vec2,
    turbulence: f32,
    white: f32,
}

/// The planes of the settled cells a tile's samples blend ([`Plane`]), and
/// of a cell past them on every side, whose levels their slopes read: the
/// first cell along x and z, on the terrain's lattice of cells, and the
/// planes row by row, `columns` a row.
struct Planes {
    first: [i64; 2],
    columns: usize,
    planes: Vec<Plane>,
}

impl Planes {
    /// The planes of the cells `cell` meters across around the samples
    /// from `low` to `high` (terrain-local), each from the settled cell
    /// `cells` gives (by its place on the terrain's lattice of cells; `None`
    /// where no record holds it), over the ground `ground` gives at each
    /// terrain-local point.
    fn of(
        low: Vec2,
        high: Vec2,
        cell: f32,
        cells: &dyn Fn(i64, i64) -> Option<SteadyCell>,
        ground: &dyn Fn(Vec2) -> f32,
    ) -> Self {
        let [(i0, j0, _), ..] = around(low, cell);
        let [.., (i1, j1, _)] = around(high, cell);
        let first = [i0 - 1, j0 - 1];
        let (columns, rows) = ((i1 - i0 + 3) as usize, (j1 - j0 + 3) as usize);
        let found: Vec<Option<SteadyCell>> = (0..rows)
            .flat_map(|row| {
                (0..columns)
                    .map(move |column| cells(first[0] + column as i64, first[1] + row as i64))
            })
            .collect();
        // The level of the water standing in the cell at `(column, row)`,
        // if any does there.
        let standing = |column: i64, row: i64| -> Option<f32> {
            if column < 0 || row < 0 || column >= columns as i64 || row >= rows as i64 {
                return None;
            }
            match found[row as usize * columns + column as usize] {
                Some(SteadyCell::Held { level, .. }) => Some(level),
                Some(SteadyCell::Flowing { surface, .. }) => Some(surface),
                Some(SteadyCell::Dry) | None => None,
            }
        };
        // Along an axis, the slope of a cell standing at `own` between its
        // neighbors either side.
        let along = |before: Option<f32>, own: f32, after: Option<f32>| match (before, after) {
            (Some(before), Some(after)) => {
                let theta = shallow_water::THETA;
                minmod(
                    theta * (own - before),
                    0.5 * (after - before),
                    theta * (after - own),
                ) / cell
            }
            (Some(before), None) => (own - before) / cell,
            (None, Some(after)) => (after - own) / cell,
            (None, None) => 0.0,
        };
        let mut planes = Vec::with_capacity(found.len());
        for row in 0..rows {
            for column in 0..columns {
                let (x, z) = (column as i64, row as i64);
                let plane = match found[row * columns + column] {
                    None => Plane::default(),
                    Some(SteadyCell::Dry) => Plane {
                        known: true,
                        ..Plane::default()
                    },
                    Some(SteadyCell::Held { level }) => Plane {
                        known: true,
                        wet: true,
                        held: true,
                        level,
                        ..Plane::default()
                    },
                    Some(SteadyCell::Flowing {
                        surface,
                        velocity,
                        energy,
                        swirl,
                        rush,
                        lip,
                        ..
                    }) => {
                        let velocity = Vec2::from(velocity);
                        // Its bed, as its region's cells stand on the
                        // ground: the mean of the ground at its corners.
                        let (i, j) = (first[0] + x, first[1] + z);
                        let corner = |di: i64, dj: i64| {
                            ground(Vec2::new((i + di) as f32, (j + dj) as f32) * cell)
                        };
                        let bed =
                            0.25 * (corner(0, 0) + corner(1, 0) + corner(0, 1) + corner(1, 1));
                        Plane {
                            known: true,
                            wet: true,
                            held: false,
                            level: surface,
                            slope: Vec2::new(
                                along(standing(x - 1, z), surface, standing(x + 1, z)),
                                along(standing(x, z - 1), surface, standing(x, z + 1)),
                            ),
                            velocity,
                            turbulence: turbulence(rush, velocity.length(), swirl),
                            white: breaking(energy, surface - bed, lip),
                        }
                    }
                };
                planes.push(plane);
            }
        }
        Self {
            first,
            columns,
            planes,
        }
    }

    /// The plane of cell `(i, j)` of the terrain's lattice of cells; that of
    /// a cell no record holds past the cells it holds.
    fn at(&self, i: i64, j: i64) -> Plane {
        let (column, row) = (i - self.first[0], j - self.first[1]);
        if column < 0 || row < 0 || column >= self.columns as i64 {
            return Plane::default();
        }
        self.planes
            .get(row as usize * self.columns + column as usize)
            .copied()
            .unwrap_or_default()
    }
}

impl Tile {
    /// The water settled water stands at this level-0 tile of `pyramid`'s
    /// samples: from the settled cells `cell` meters across `cells` gives
    /// (by their place on the terrain's lattice of cells; `None` where no
    /// record holds them), each sample's surface blends the planes of the
    /// cells around it holding water, each its level and its own surface's
    /// slope times the way from its middle to the sample, by their
    /// nearness. Dry cells add nothing, and a sample whose blended surface
    /// stands under its ground is dry. Its velocity, turbulence and white
    /// water (each cell's breaking as deep as it stands over the tile's
    /// ground at its corners) blend alike, held still where most are
    /// lakes', and it takes its shear and the reach of its bed
    /// ([`Tile::bed`]); dry where none holds any, and as its own water stood
    /// where no record holds the cells around it. Whether a fall's sheet
    /// runs over it stays its own, and so does its span, the span of the
    /// water body its own water held nearest it where it held none there;
    /// and `solids` (terrain-local) stand in it. Surfaces stand at their
    /// steps of the height grid, as records hold them; `None` where no
    /// water stands on it.
    #[must_use]
    pub fn steadied(
        &self,
        pyramid: &Pyramid,
        (cell, cells): (f32, &dyn Fn(i64, i64) -> Option<SteadyCell>),
        solids: &[Footprint],
    ) -> Option<TileWater> {
        let stride = TILE_STRIDE as usize;
        let spacing = self.spacing();
        let origin = pyramid.tile_origin(self.key());
        let own = self.water();
        let place =
            |x: usize, z: usize| origin + Vec2::new(x as f32 - 1.0, z as f32 - 1.0) * spacing;
        let ground = |point: Vec2| self.ground_at(point - origin);
        let planes = Planes::of(
            place(0, 0),
            place(stride - 1, stride - 1),
            cell,
            cells,
            &ground,
        );
        let count = stride * stride;
        let mut surface = vec![f32::NAN; count];
        let mut info = vec![[0_u32; 2]; count];
        let mut flows = vec![Vec2::ZERO; count];
        let mut stirred = vec![None; count];
        let spans = spans(own);
        for z in 0..stride {
            for x in 0..stride {
                let at = z * stride + x;
                let (own_surface, own_info) = own.map_or((f32::NAN, [0; 2]), |water| {
                    (water.surfaces()[at], water.info()[at])
                });
                let point = place(x, z);
                let mut blend = Blend::NONE;
                for (i, j, share) in around(point, cell) {
                    let plane = planes.at(i, j);
                    if !plane.known {
                        continue;
                    }
                    blend.known = true;
                    if !plane.wet {
                        continue;
                    }
                    let middle = (Vec2::new(i as f32, j as f32) + 0.5) * cell;
                    blend.weight += share;
                    blend.level += (plane.level + plane.slope.dot(point - middle)) * share;
                    if plane.held {
                        blend.held += share;
                    } else {
                        blend.moving += share;
                        blend.flow += plane.velocity * share;
                        blend.turbulence += plane.turbulence * share;
                        blend.white += plane.white * share;
                    }
                }
                if !blend.known {
                    surface[at] = own_surface;
                    info[at] = own_info;
                    flows[at] = unpack(own_info[0]).0;
                    continue;
                }
                if blend.weight <= 0.0 {
                    continue;
                }
                let sheeted = own_surface.is_finite() && own_info[0] & (1 << 23) != 0;
                let flow = blend.flow();
                surface[at] = codec::on_grid(blend.level / blend.weight);
                info[at][0] = pack(flow, sheeted, blend.held * 2.0 > blend.weight, spans[at]);
                flows[at] = flow;
                stirred[at] = Some([blend.turbulence / blend.weight, blend.white / blend.weight]);
            }
        }
        self.stir(&surface, &mut info, &flows, &stirred);
        TileWater::new(
            surface.into_boxed_slice(),
            info.into_boxed_slice(),
            (self.heights(), spacing, self.origin()),
            solids,
        )
    }

    /// The water of this tile of `pyramid`, of a coarser level than 0,
    /// running, churning and breaking as settled water's summaries say
    /// ([`SteadySummary`]): each sample its own water stands over takes the
    /// mean velocity, swirl and white water of the summaries' flowing
    /// squares under its footprint, `squares` giving each by its place on the
    /// terrain's lattice of squares (`None` where no summary holds it),
    /// weighted by a tent reaching its spacing either way (a square, the
    /// four squares around it blended bilinearly, where its spacing is
    /// finer), so each level holds the finer water averaged over its own
    /// spacing; and its shear and the reach of its bed ([`Tile::bed`]);
    /// still and clear where none of them holds water, and as its own water
    /// ran where no summary holds any square under it. Its
    /// water's surface, lakes, spans, falls' sheets and solids stay its own,
    /// and its dry samples take theirs from it anew
    /// ([`TileWater::surfaces`]); `None` where it holds none.
    #[must_use]
    pub fn flowing(
        &self,
        pyramid: &Pyramid,
        squares: &dyn Fn(i64, i64) -> Option<SteadySquare>,
    ) -> Option<TileWater> {
        let own = self.water()?;
        let stride = TILE_STRIDE as usize;
        let spacing = self.spacing();
        let origin = pyramid.tile_origin(self.key());
        let square = pyramid.tile_size(0) / SUMMARY_SIDE as f32;
        let surface: Box<[f32]> = own.surfaces().into();
        let heights = self.heights();
        let mut info: Box<[[u32; 2]]> = own.info().into();
        let mut flows: Vec<Vec2> = info.iter().map(|words| unpack(words[0]).0).collect();
        let mut stirred = vec![None; info.len()];
        for z in 0..stride {
            for x in 0..stride {
                let at = z * stride + x;
                let wet = surface[at] > heights[at];
                if !wet {
                    continue;
                }
                let point = origin + Vec2::new(x as f32 - 1.0, z as f32 - 1.0) * spacing;
                let mut blend = Blend::NONE;
                for (i, j, share) in tent(point, spacing, square) {
                    let Some(found) = squares(i, j) else {
                        continue;
                    };
                    blend.known = true;
                    // Summaries hold no swings of speed, nor lips: their
                    // white water holds what their cells' lips throw.
                    if let Some(velocity) = found.velocity {
                        let velocity = Vec2::from(velocity);
                        let swirl = found.swirl.unwrap_or(0.0);
                        blend.moving += share;
                        blend.flow += velocity * share;
                        blend.turbulence += turbulence(0.0, velocity.length(), swirl) * share;
                        blend.white += found.white.unwrap_or(0.0) * share;
                    }
                }
                if !blend.known {
                    continue;
                }
                let (_, lake, span) = unpack(info[at][0]);
                let sheeted = info[at][0] & (1 << 23) != 0;
                let flowing = !lake && blend.moving > 0.0;
                let flow = if flowing {
                    blend.flow / blend.moving
                } else {
                    Vec2::ZERO
                };
                info[at][0] = pack(flow, sheeted, lake, span);
                flows[at] = flow;
                stirred[at] = Some(if flowing {
                    [blend.turbulence / blend.moving, blend.white / blend.moving]
                } else {
                    [0.0; 2]
                });
            }
        }
        self.stir(&surface, &mut info, &flows, &stirred);
        let tile = (self.heights(), spacing, self.origin());
        TileWater::new(surface, info, tile, own.solids())
    }

    /// Writes the state of the samples of this tile's water `stirred` holds
    /// (each one's turbulence and white water) into their words `info`
    /// ([`pack_state`]): with their shear, from the velocities `flows` holds
    /// at the samples around them, standing at `surface` ([`shears`]), and
    /// the reach of the stream the tile's bed holds there ([`Tile::bed`]),
    /// never blended.
    fn stir(
        &self,
        surface: &[f32],
        info: &mut [[u32; 2]],
        flows: &[Vec2],
        stirred: &[Option<[f32; 2]>],
    ) {
        let stride = TILE_STRIDE as usize;
        let spacing = self.spacing();
        let heights = self.heights();
        let shears = shears(flows, &|at| surface[at] > heights[at], spacing);
        for (at, stirred) in stirred.iter().enumerate() {
            let Some([churning, white]) = *stirred else {
                continue;
            };
            let local = Vec2::new((at % stride) as f32 - 1.0, (at / stride) as f32 - 1.0) * spacing;
            info[at][1] = pack_state(WaterState {
                turbulence: churning,
                breaking: white,
                shear: shears[at],
                reach: self.bed(local).map(|bed| bed.reach),
            });
        }
    }

    /// The ground at `local`, meters from the tile's first corner, on the
    /// triangles every consumer splits cells into, over its samples and
    /// their border, and at the border's nearest point past it: what the
    /// corners of the settled cells its samples blend stand on.
    fn ground_at(&self, local: Vec2) -> f32 {
        let last = (TILE_STRIDE - 2) as f32;
        let u = clamp(local.x / self.spacing(), -1.0, last);
        let v = clamp(local.y / self.spacing(), -1.0, last);
        let (i, j) = (min(u.floor(), last - 1.0), min(v.floor(), last - 1.0));
        let (x, z) = (i as i32, j as i32);
        let corners = [
            self.height(x, z),
            self.height(x + 1, z),
            self.height(x, z + 1),
            self.height(x + 1, z + 1),
        ];
        on_triangle(corners, u - i, v - j)
    }
}

/// What a tile's water reads of settled water ([`Steadying`]): records of
/// level-0 tiles whole, or their summaries, by tile.
pub(crate) enum SettledReads {
    Records(Vec<(TileKey, SteadyWater)>),
    Summaries(HashMap<TileKey, SteadySummary>),
}

/// Making a tile's water as settled water stands it at its samples (the
/// job [`crate::jobs::SteadiedJob`]): generating its ground, in steps a
/// job's budget sizes, reading it back as its record holds it, then
/// standing on it the water its reads give: a level-0 tile's from records
/// of cells `cell` meters across, with the solids standing in their water
/// ([`Tile::steadied`]), a coarser tile's from summaries, their squares on
/// the terrain's lattice of them, each level-0 tile holding
/// [`SUMMARY_SIDE`] along each side ([`Tile::flowing`]).
pub(crate) struct Steadying {
    pyramid: Pyramid,
    cell: f32,
    generator: Generator,
    reads: SettledReads,
}

impl Steadying {
    /// Tile `key`'s water over `layers` (with their bakes, under the edits
    /// reaching its ground) and `pyramid`, from `reads` of cells `cell`
    /// meters across.
    pub(crate) fn new(
        layers: Arc<Layers>,
        pyramid: Pyramid,
        key: TileKey,
        (cell, reads): (f32, SettledReads),
    ) -> Self {
        Self {
            pyramid,
            cell,
            generator: Generator::new(layers, &pyramid, key),
            reads,
        }
    }

    /// Works for about `budget` units (a sample generated each): once done,
    /// the tile's water, `None` within where none stands on it; an error
    /// where its ground does not generate or read back.
    pub(crate) fn advance(
        &mut self,
        budget: &mut u64,
    ) -> Result<Option<Option<TileWater>>, String> {
        let Some(made) = self.generator.advance(budget) else {
            if self.generator.failed() {
                return Err("the ground's heights are invalid".into());
            }
            return Ok(None);
        };
        // The tile as its record holds it: its water stands on those heights.
        let tile: Tile = codec::decode(&codec::encode(&made)).map_err(|error| error.to_string())?;
        let pyramid = &self.pyramid;
        let water = match &self.reads {
            SettledReads::Records(records) => {
                let cell = self.cell;
                let cells = |i: i64, j: i64| {
                    let middle = Vec2::new((i as f32 + 0.5) * cell, (j as f32 + 0.5) * cell);
                    let key = pyramid.tile_at(0, middle)?;
                    let (_, water) = records.iter().find(|(held, _)| *held == key)?;
                    water.at(i, j)
                };
                let solids: Vec<Footprint> = records
                    .iter()
                    .flat_map(|(_, water)| water.solids.iter().copied())
                    .collect();
                tile.steadied(pyramid, (cell, &cells), &solids)
            }
            SettledReads::Summaries(summaries) => tile.flowing(pyramid, &|i, j| {
                let side = i64::from(SUMMARY_SIDE);
                if i < 0 || j < 0 {
                    return None;
                }
                let x = u16::try_from(i / side).ok()?;
                let z = u16::try_from(j / side).ok()?;
                summaries
                    .get(&TileKey::new(0, x, z))?
                    .square((i % side) as u32, (j % side) as u32)
            }),
        };
        Ok(Some(water))
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
