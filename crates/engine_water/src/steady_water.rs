use std::sync::Arc;

use glam::{IVec2, Vec2};

use engine_core::math::GRAVITY;
use engine_math::collections::HashMap;
use engine_math::{clamp, max, min};

use crate::shallow_water::{self, CELL_MEANS, DRY, UNHELD, WET_MOST, resistance, velocity};
use crate::tile_water::{
    NONE, PointState, TileWater, WATER_SPAN_LEAST, breaking, generated, nearest, pack, pack_state,
    shears, tile_origin, tile_point, turbulence, unpack,
};
use engine_codec::{self as codec, Error, Reader, Writer, record};
use engine_heightfield::tile::{
    BiomeRules, Footprint, Generator, Solid, TILE_STRIDE, Tile, gradient_word, on_triangle,
    smooth_gradient,
};
use engine_heightfield::{Bed, Layers, Pyramid, Reach, Stamp, TILE_QUADS, TILE_SAMPLES, TileKey};

pub use crate::gpu::{BLOCK_SIDE, COVER_POINTS, NO_BLOCK};

pub const MARGIN: f32 = 64.0;
pub const REGION_CELLS: u64 = 1 << 22;
pub const BLOCK_CELLS: usize = (BLOCK_SIDE * BLOCK_SIDE) as usize;
const BLOCK_CORNERS: usize = ((BLOCK_SIDE + 1) * (BLOCK_SIDE + 1)) as usize;
pub const BLOCK_GROUND: usize = crate::gpu::BLOCK_GROUND as usize;
const SPREAD: u32 = 2;
const SAMPLES: usize = TILE_SAMPLES as usize;
const ENTRY: f32 = 2.0;
const SOURCE_LEAST: f32 = 0.01;

const _: () = assert!(BLOCK_GROUND == BLOCK_CORNERS + 4 * BLOCK_CELLS);

pub(crate) const REACHES: usize = Reach::Falls as usize + 1;

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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SteadySource {
    pub cell: u32,
    pub rate: f32,
    pub way: [f32; 2],
}

record!(SteadySource { cell, rate, way });

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
    pub live: LiveWater,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LiveWater {
    pub first: [i64; 2],
    pub blocks: [u32; 2],
    pub wet: Vec<u32>,
    pub cells: Vec<[f32; 3]>,
}

record!(LiveWater {
    first,
    blocks,
    wet,
    cells,
});

impl LiveWater {
    pub fn block(&self, x: i64, z: i64) -> Option<&[[f32; 3]]> {
        let (column, row) = (x - self.first[0], z - self.first[1]);
        let [columns, rows] = self.blocks.map(i64::from);
        if column < 0 || row < 0 || column >= columns || row >= rows {
            return None;
        }
        let at = self
            .wet
            .binary_search(&((row * columns + column) as u32))
            .ok()?;
        self.cells.get(at * BLOCK_CELLS..(at + 1) * BLOCK_CELLS)
    }

    fn valid(&self) -> bool {
        let spanned = u64::from(self.blocks[0]) * u64::from(self.blocks[1]);
        self.cells.len() == self.wet.len() * BLOCK_CELLS
            && self.wet.windows(2).all(|pair| pair[0] < pair[1])
            && self
                .wet
                .last()
                .is_none_or(|&last| u64::from(last) < spanned)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SteadyCell {
    Dry,
    Held {
        level: f32,
    },
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
        out.put(&self.live);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, codec::Error> {
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
        let live: LiveWater = input.get()?;
        let grids = [&surface, &x, &z, &energy, &heave, &swirl, &rush, &wet, &lip];
        if grids.iter().any(|grid| grid.len() != count)
            || !(cell.is_finite() && cell > 0.0)
            || sources.iter().any(|source| source.cell as usize >= count)
            || !solids.iter().all(Footprint::valid)
            || !live.valid()
        {
            return Err(codec::invalid("$.settled"));
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
            live,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SettledHead {
    pub key: TileKey,
    pub cell: f32,
    pub summary: SteadySummary,
}

impl SteadyWater {
    pub fn head(bytes: &[u8]) -> Result<(SettledHead, usize), codec::Error> {
        let mut input = Reader::new(bytes);
        let key = input.get()?;
        let cell: f32 = input.get()?;
        let _: [i64; 2] = input.get()?;
        let _: [u32; 2] = input.get()?;
        let _: Vec<SteadySource> = input.get()?;
        let summary = input.get()?;
        if !(cell.is_finite() && cell > 0.0) {
            return Err(codec::invalid("$.cell").got(cell));
        }
        Ok((
            SettledHead { key, cell, summary },
            bytes.len() - input.left(),
        ))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SettledKey {
    pub recipe: [u8; 32],
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
    pub fn identity(&self) -> Vec<u8> {
        codec::encode(self)
    }
}

fn lattice(low: f32, high: f32, cell: f32) -> (i64, u32) {
    let first = (low / cell - 0.5).ceil() as i64;
    let end = (high / cell - 0.5).ceil() as i64;
    (first, (end - first).max(0) as u32)
}

fn split(count: u32, parts: u32) -> impl Iterator<Item = (u32, u32)> {
    let at = move |part: u32| (u64::from(count) * u64::from(part) / u64::from(parts)) as u32;
    (0..parts).map(move |part| (at(part), at(part + 1) - at(part)))
}

fn reach(pyramid: &Pyramid, axis: usize, (first, tiles): (u32, u32), cell: f32) -> (i64, u32) {
    let size = pyramid.tile_size(0);
    let extent = pyramid.extent()[axis];
    let low = max(first as f32 * size - MARGIN, 0.0);
    let high = min((first + tiles) as f32 * size + MARGIN, extent);
    lattice(low, high, cell)
}

fn finer((first, cells): (i64, u32), split: u32) -> (i64, u32) {
    (first * i64::from(split), cells * split)
}

pub fn settle_regions(layers: &Layers, pyramid: &Pyramid, cell: f32) -> Vec<[u32; 4]> {
    let tiles = pyramid.tiles(0);
    let parts = shallow_water::live_split(cell, pyramid.spacing(0));
    let reached = Reached::of(layers, pyramid, cell / parts as f32);
    let mut splits: Vec<(u32, u32)> = (1..=tiles[0])
        .flat_map(|x| (1..=tiles[1]).map(move |z| (x, z)))
        .collect();
    splits.sort_by_key(|&(x, z)| (u64::from(x) * u64::from(z), x.abs_diff(z)));
    let fits = |&(along_x, along_z): &(u32, u32)| {
        split(tiles[1], along_z).all(|rows| {
            let rows = finer(reach(pyramid, 1, rows, cell), parts);
            split(tiles[0], along_x).all(|columns| {
                let columns = finer(reach(pyramid, 0, columns, cell), parts);
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

struct Reached {
    columns: usize,
    counts: Vec<u32>,
}

impl Reached {
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

#[derive(Clone, Debug, PartialEq)]
pub struct SettleRegion {
    pub cell: f32,
    pub split: u32,
    pub first: [i64; 2],
    pub cells: [u32; 2],
    pub kept: Vec<TileKey>,
}

impl SettleRegion {
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

    pub fn alone(pyramid: &Pyramid, key: TileKey, cell: f32) -> Self {
        let tiles = [u32::from(key.x), u32::from(key.z), 1, 1];
        Self::over(pyramid, tiles, vec![key], cell)
    }

    pub fn bounds(&self) -> (Vec2, Vec2) {
        let low = Vec2::new(self.first[0] as f32, self.first[1] as f32) * self.cell;
        let cells = Vec2::new(self.cells[0] as f32, self.cells[1] as f32);
        (low, low + cells * self.cell)
    }

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

fn reach_code(bed: Option<&Bed>) -> u8 {
    bed.map_or(0, |bed| bed.reach as u8 + 1)
}

impl WaterPart {
    fn of(
        pyramid: &Pyramid,
        (tile, own): (&Tile, Option<&TileWater>),
        (cell, solids): (f32, &[Option<Solid>]),
    ) -> Self {
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
                let found = tile_point(tile, own, local);
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

pub(crate) struct PartMaking {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    cell: f32,
    solids: Vec<Option<Solid>>,
    generator: Generator,
}

impl PartMaking {
    pub(crate) fn new(
        (layers, rules): (Arc<Layers>, Arc<BiomeRules>),
        pyramid: Pyramid,
        key: TileKey,
        (cell, solids): (f32, Vec<Option<Solid>>),
    ) -> Self {
        Self {
            generator: Generator::new(Arc::clone(&layers), &pyramid, key).with_surface(rules),
            layers,
            pyramid,
            cell,
            solids,
        }
    }

    pub(crate) fn advance(&mut self, budget: &mut u64) -> Result<Option<WaterPart>, Error> {
        let Some((tile, own)) =
            generated(&mut self.generator, budget, (&self.layers, &self.pyramid))?
        else {
            return Ok(None);
        };
        Ok(Some(WaterPart::of(
            &self.pyramid,
            (&tile, own.as_ref()),
            (self.cell, &self.solids),
        )))
    }
}

struct Reading<'a> {
    tiles: [u32; 2],
    spacing: f32,
    first: [u32; 2],
    columns: usize,
    parts: Vec<&'a WaterPart>,
}

impl<'a> Reading<'a> {
    fn of(
        pyramid: &Pyramid,
        region: &SettleRegion,
        parts: &[&'a WaterPart],
    ) -> Result<Self, Error> {
        if let Some(part) = parts.iter().find(|part| part.cell != region.cell) {
            return Err(codec::invalid("$.cell").got(part.cell));
        }
        let reached = region.reached(pyramid);
        let span = |line: fn(&TileKey) -> u16| {
            let lines = reached.iter().map(|key| u32::from(line(key)));
            let (first, last) = (lines.clone().min(), lines.max());
            first
                .zip(last)
                .map(|(first, last)| (first, (last + 1 - first) as usize))
        };
        let (Some((x0, columns)), Some((z0, rows))) = (span(|key| key.x), span(|key| key.z)) else {
            return Err(codec::invalid("$.parts"));
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
        let Some(parts) = grid.into_iter().collect::<Option<Vec<&WaterPart>>>() else {
            return Err(codec::invalid("$.parts"));
        };
        Ok(Self {
            tiles: pyramid.tiles(0),
            spacing: pyramid.spacing(0),
            first: [x0, z0],
            columns,
            parts,
        })
    }

    fn part(&self, x: u32, z: u32) -> Option<&'a WaterPart> {
        let (x, z) = (x.checked_sub(self.first[0])?, z.checked_sub(self.first[1])?);
        if x as usize >= self.columns {
            return None;
        }
        self.parts
            .get(z as usize * self.columns + x as usize)
            .copied()
    }

    fn last(&self) -> [i64; 2] {
        self.tiles
            .map(|count| i64::from(count) * i64::from(TILE_QUADS))
    }

    fn sample(&self, at: [i64; 2]) -> Option<(&'a WaterPart, usize)> {
        let quads = i64::from(TILE_QUADS);
        let last = self.last();
        let [x, z] = [0, 1].map(|axis| at[axis].clamp(0, last[axis]));
        let [tx, tz] =
            [(x, 0), (z, 1)].map(|(at, axis)| (at / quads).min(i64::from(self.tiles[axis]) - 1));
        let part = self.part(tx as u32, tz as u32)?;
        Some((
            part,
            ((z - tz * quads) * (SAMPLES as i64) + x - tx * quads) as usize,
        ))
    }

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

    fn reach(&self, at: [i64; 2]) -> u8 {
        self.sample(at).map_or(0, |(part, index)| part.reach[index])
    }

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
        on_triangle(
            [height(0, 0), height(1, 0), height(0, 1), height(1, 1)],
            fx,
            fz,
        )
    }

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

    fn datum(&self) -> f32 {
        let lowest = self
            .parts
            .iter()
            .flat_map(|part| part.heights.iter().copied())
            .fold(f32::INFINITY, min);
        if lowest.is_finite() {
            lowest.floor()
        } else {
            0.0
        }
    }
}

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

pub fn cover_cells(
    footprints: &[Footprint],
    (first, cells): ([i64; 2], [u32; 2]),
    cell: f32,
) -> Vec<([u32; 2], [f32; 2])> {
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

#[derive(Clone, Debug, PartialEq)]
pub struct StoredBlocks {
    pub cells: [u32; 2],
    pub blocks: [u32; 2],
    pub table: Vec<u32>,
    pub places: Vec<u32>,
}

impl StoredBlocks {
    pub fn whole(cells: [u32; 2]) -> Self {
        let blocks = cells.map(|count| count.div_ceil(BLOCK_SIDE));
        Self::marked(cells, &vec![true; blocks[0] as usize * blocks[1] as usize])
    }

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

    pub fn stored(&self) -> usize {
        self.places.len()
    }

    pub fn slot(&self, x: i64, z: i64) -> Option<usize> {
        let [columns, rows] = self.blocks.map(i64::from);
        if x < 0 || z < 0 || x >= columns || z >= rows {
            return None;
        }
        let slot = self.table[(z * columns + x) as usize];
        (slot != NO_BLOCK).then_some(slot as usize)
    }

    pub fn at(&self, i: u32, j: u32) -> Option<usize> {
        if i >= self.cells[0] || j >= self.cells[1] {
            return None;
        }
        let slot = self.slot(i64::from(i / BLOCK_SIDE), i64::from(j / BLOCK_SIDE))?;
        Some(slot * BLOCK_CELLS + ((j % BLOCK_SIDE) * BLOCK_SIDE + i % BLOCK_SIDE) as usize)
    }

    pub fn cell(&self, at: usize) -> [u32; 2] {
        let place = self.places[at / BLOCK_CELLS];
        let local = (at % BLOCK_CELLS) as u32;
        [
            ((place & 0xffff) * BLOCK_SIDE + local % BLOCK_SIDE).min(self.cells[0] - 1),
            ((place >> 16) * BLOCK_SIDE + local / BLOCK_SIDE).min(self.cells[1] - 1),
        ]
    }

    pub fn gather<T>(&self, cell: impl Fn(u32, u32) -> T) -> Vec<T> {
        (0..self.stored() * BLOCK_CELLS)
            .map(|at| {
                let [i, j] = self.cell(at);
                cell(i, j)
            })
            .collect()
    }

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
    ) -> Result<Self, Error> {
        let split = region.split;
        let cell = region.cell / split as f32;
        let (first_x, cells_x) = finer((region.first[0], region.cells[0]), split);
        let (first_z, cells_z) = finer((region.first[1], region.cells[1]), split);
        let (first, cells) = ([first_x, first_z], [cells_x, cells_z]);
        let [nx, nz] = cells;
        if nx == 0 || nz == 0 {
            return Err(codec::invalid("$.cells").got(cells));
        }
        let kept = kept_cells(pyramid, region)?;
        let reading = Reading::of(pyramid, region, parts)?;
        let datum = reading.datum();
        let spacing = reading.spacing;
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
        let settled = |i: u32, j: u32| {
            [
                region.first[0] + i64::from(i / split),
                region.first[1] + i64::from(j / split),
            ]
        };
        let warmed = |at: [i64; 2]| warm.and_then(|warm| warm(at[0], at[1]));
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
        for (at, &[share, top]) in cover.iter().enumerate() {
            let [i, j] = stored.cell(at);
            let holder = settled(i, j);
            let found = reading.water(holder);
            let (bed, start, lake) = started(corners(i, j), found, datum, || warmed(holder));
            let start = if share >= 1.0 && top >= start[0] {
                [bed, 0.0, 0.0]
            } else {
                start
            };
            water.push([start[0], start[1], start[2], 0.0]);
            held.push(if lake.is_nan() { UNHELD } else { lake });
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

    fn bare(&self) -> SlotGround<'_> {
        SlotGround(&self.ground)
    }

    pub fn bed(&self, at: usize) -> f32 {
        self.bare().bed(at)
    }

    pub fn held(&self, at: usize) -> Option<f32> {
        self.bare().held(at)
    }

    fn wet(&self, (means, ground): (&[[f32; 4]], SlotGround<'_>), i: u32, j: u32) -> bool {
        let Some(at) = self.stored.at(i, j) else {
            return false;
        };
        let Some(&[water, _, shares]) = means.get(at * CELL_MEANS..(at + 1) * CELL_MEANS) else {
            return false;
        };
        ground.held(at).is_some() || (water[0] - ground.bed(at) > DRY && shares[0] >= WET_MOST)
    }

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
                live: self.live((means, raised), kept),
            });
        }
        made
    }

    fn live(
        &self,
        (means, ground): (&[[f32; 4]], SlotGround<'_>),
        [x0, z0, columns, rows]: [u32; 4],
    ) -> LiveWater {
        let split = self.split;
        let side = i64::from(BLOCK_SIDE);
        let low = [x0 * split, z0 * split].map(i64::from);
        let high = [(x0 + columns) * split, (z0 + rows) * split].map(i64::from);
        let first = [0, 1].map(|axis| (self.first[axis] + low[axis]).div_euclid(side));
        let last = [0, 1].map(|axis| (self.first[axis] + high[axis] - 1).div_euclid(side));
        let blocks = [0, 1].map(|axis| (last[axis] - first[axis] + 1).max(0) as u32);
        let mut live = LiveWater {
            first,
            blocks,
            wet: Vec::new(),
            cells: Vec::new(),
        };
        let mut cells = [[f32::NAN; 3]; BLOCK_CELLS];
        for row in 0..blocks[1] {
            for column in 0..blocks[0] {
                let corner = [
                    (first[0] + i64::from(column)) * side - self.first[0],
                    (first[1] + i64::from(row)) * side - self.first[1],
                ];
                let [from, to] =
                    [0, side - 1].map(|past| corner.map(|at| (at + past).div_euclid(side)));
                let stored = (from[1]..=to[1])
                    .any(|z| (from[0]..=to[0]).any(|x| self.stored.slot(x, z).is_some()));
                if !stored {
                    continue;
                }
                let mut wet = false;
                for (k, held) in cells.iter_mut().enumerate() {
                    let i = corner[0] + (k % BLOCK_SIDE as usize) as i64;
                    let j = corner[1] + (k / BLOCK_SIDE as usize) as i64;
                    let inside = (low[0]..high[0]).contains(&i) && (low[1]..high[1]).contains(&j);
                    *held = [f32::NAN; 3];
                    if !inside {
                        continue;
                    }
                    if let Some(water) = self.live_cell((means, ground), i as u32, j as u32) {
                        *held = water;
                        wet = true;
                    }
                }
                if wet {
                    live.wet.push(row * blocks[0] + column);
                    live.cells.extend_from_slice(&cells);
                }
            }
        }
        live
    }

    fn live_cell(
        &self,
        (means, ground): (&[[f32; 4]], SlotGround<'_>),
        i: u32,
        j: u32,
    ) -> Option<[f32; 3]> {
        let at = self.stored.at(i, j)?;
        if let Some(level) = ground.held(at) {
            return Some([level + self.datum, f32::NAN, f32::NAN]);
        }
        let Some(&[water, _, shares]) = means.get(at * CELL_MEANS..(at + 1) * CELL_MEANS) else {
            return None;
        };
        let wet = water[0] - ground.bed(at) > DRY && shares[0] >= WET_MOST;
        wet.then_some([water[0] + self.datum, water[1], water[2]])
    }
}

#[derive(Clone, Copy)]
struct SlotGround<'a>(&'a [f32]);

impl SlotGround<'_> {
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

    fn bed(self, at: usize) -> f32 {
        let [a, b, c, d] = self.corners(at);
        0.25 * (a + b + c + d)
    }

    fn lean(self, at: usize, cell: f32) -> f32 {
        let [a, b, c, d] = self.corners(at);
        let across = 0.5 * ((b + d) - (a + c)) / cell;
        let along = 0.5 * ((c + d) - (a + b)) / cell;
        1.0 + across * across + along * along
    }

    fn held(self, at: usize) -> Option<f32> {
        let level = self.0[(at / BLOCK_CELLS) * BLOCK_GROUND + BLOCK_CORNERS + at % BLOCK_CELLS];
        (level > 0.5 * UNHELD).then_some(level)
    }

    fn rough(self, at: usize) -> f32 {
        let roughness = BLOCK_CORNERS + BLOCK_CELLS;
        self.0[(at / BLOCK_CELLS) * BLOCK_GROUND + roughness + at % BLOCK_CELLS]
    }
}

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

fn kept_cells(pyramid: &Pyramid, region: &SettleRegion) -> Result<Vec<(TileKey, [u32; 4])>, Error> {
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
                return Err(crate::jobs::outside(key));
            }
            Ok((key, [x as u32, z as u32, columns, rows]))
        })
        .collect()
}

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
            let at = [
                region.first[0] + i64::from(i),
                region.first[1] + i64::from(j),
            ];
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

pub const SUMMARY_SIDE: u32 = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct SteadySummary {
    velocity: [Box<[u16]>; 2],
    white: Box<[u16]>,
    swirl: Box<[u16]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SteadySquare {
    pub velocity: Option<[f32; 2]>,
    pub white: Option<f32>,
    pub swirl: Option<f32>,
}

impl SteadySummary {
    fn of(cells: [u32; 2], velocity: &[Vec<f32>; 2], (white, swirl): (&[f32], &[f32])) -> Self {
        let side = SUMMARY_SIDE as usize;
        let mut sums = vec![[0.0_f32; 5]; side * side];
        let (columns, rows) = (cells[0] as usize, cells[1] as usize);
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

    fn read(input: &mut Reader<'_>) -> Result<Self, codec::Error> {
        let stride = SUMMARY_SIDE as usize;
        let x = codec::get_wide_planes(input, stride)?;
        let z = codec::get_wide_planes(input, stride)?;
        let white = codec::get_wide_planes(input, stride)?;
        let swirl = codec::get_wide_planes(input, stride)?;
        if [x.len(), z.len(), white.len(), swirl.len()] != [stride * stride; 4] {
            return Err(codec::invalid("$.summary"));
        }
        Ok(Self {
            velocity: [x, z],
            white,
            swirl,
        })
    }
}

pub(crate) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let biased = (bits >> 23) & 0xff;
    let mantissa = bits & 0x7f_ffff;
    if biased == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x200 };
    }
    let exponent = biased as i32 - 127 + 15;
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
    let rounded = kept + u32::from(rest > halfway || (rest == halfway && (kept & 1) == 1));
    sign | rounded.min(0x7bff) as u16
}

pub(crate) fn f16_value(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    let magnitude = match exponent {
        0 => (f32::from(bits & 0x3ff) / 16_777_216.0).to_bits(),
        0x1f => 0x7f80_0000 | (mantissa << 13),
        _ => ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

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

    fn flow(&self) -> Vec2 {
        if self.held * 2.0 > self.weight || self.moving <= 0.0 {
            Vec2::ZERO
        } else {
            self.flow / self.moving
        }
    }
}

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

fn tent(point: Vec2, spacing: f32, square: f32) -> impl Iterator<Item = (i64, i64, f32)> {
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

fn minmod(a: f32, b: f32, c: f32) -> f32 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        min(a, min(b, c))
    } else if a < 0.0 && b < 0.0 && c < 0.0 {
        max(a, max(b, c))
    } else {
        0.0
    }
}

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

struct Planes {
    first: [i64; 2],
    columns: usize,
    planes: Vec<Plane>,
}

impl Planes {
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

#[must_use]
pub(crate) fn steadied(
    (tile, own): (&Tile, Option<&TileWater>),
    pyramid: &Pyramid,
    (cell, cells): (f32, &dyn Fn(i64, i64) -> Option<SteadyCell>),
    solids: &[Footprint],
) -> Option<TileWater> {
    let stride = TILE_STRIDE as usize;
    let spacing = tile.spacing();
    let origin = pyramid.tile_origin(tile.key());
    let place = |x: usize, z: usize| origin + Vec2::new(x as f32 - 1.0, z as f32 - 1.0) * spacing;
    let ground = |point: Vec2| ground_at(tile, point - origin);
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
    stir(tile, &surface, &mut info, &flows, &stirred);
    TileWater::new(
        surface.into_boxed_slice(),
        info.into_boxed_slice(),
        (
            tile.heights(),
            spacing,
            tile_origin(tile.key(), tile.spacing()),
        ),
        solids,
    )
}

#[must_use]
pub(crate) fn flowing(
    (tile, own): (&Tile, Option<&TileWater>),
    pyramid: &Pyramid,
    squares: &dyn Fn(i64, i64) -> Option<SteadySquare>,
) -> Option<TileWater> {
    let own = own?;
    let stride = TILE_STRIDE as usize;
    let spacing = tile.spacing();
    let origin = pyramid.tile_origin(tile.key());
    let square = pyramid.tile_size(0) / SUMMARY_SIDE as f32;
    let surface: Box<[f32]> = own.surfaces().into();
    let heights = tile.heights();
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
    stir(tile, &surface, &mut info, &flows, &stirred);
    let tile = (
        tile.heights(),
        spacing,
        tile_origin(tile.key(), tile.spacing()),
    );
    TileWater::new(surface, info, tile, own.solids())
}

fn stir(
    tile: &Tile,
    surface: &[f32],
    info: &mut [[u32; 2]],
    flows: &[Vec2],
    stirred: &[Option<[f32; 2]>],
) {
    let stride = TILE_STRIDE as usize;
    let spacing = tile.spacing();
    let heights = tile.heights();
    let shears = shears(flows, &|at| surface[at] > heights[at], spacing);
    for (at, stirred) in stirred.iter().enumerate() {
        let Some([churning, white]) = *stirred else {
            continue;
        };
        let local = Vec2::new((at % stride) as f32 - 1.0, (at / stride) as f32 - 1.0) * spacing;
        info[at][1] = pack_state(PointState {
            turbulence: churning,
            breaking: white,
            shear: shears[at],
            reach: tile.bed(local).map(|bed| bed.reach),
        });
    }
}

fn ground_at(tile: &Tile, local: Vec2) -> f32 {
    let last = (TILE_STRIDE - 2) as f32;
    let u = clamp(local.x / tile.spacing(), -1.0, last);
    let v = clamp(local.y / tile.spacing(), -1.0, last);
    let (i, j) = (min(u.floor(), last - 1.0), min(v.floor(), last - 1.0));
    let (x, z) = (i as i32, j as i32);
    let corners = [
        tile.height(x, z),
        tile.height(x + 1, z),
        tile.height(x, z + 1),
        tile.height(x + 1, z + 1),
    ];
    on_triangle(corners, u - i, v - j)
}

pub(crate) enum SettledReads {
    Records(Vec<(TileKey, SteadyWater)>),
    Summaries(HashMap<TileKey, SteadySummary>),
}

pub(crate) struct Steadying {
    layers: Arc<Layers>,
    pyramid: Pyramid,
    cell: f32,
    generator: Generator,
    reads: SettledReads,
}

impl Steadying {
    pub(crate) fn new(
        layers: Arc<Layers>,
        pyramid: Pyramid,
        key: TileKey,
        (cell, reads): (f32, SettledReads),
    ) -> Self {
        Self {
            generator: Generator::new(Arc::clone(&layers), &pyramid, key),
            layers,
            pyramid,
            cell,
            reads,
        }
    }

    pub(crate) fn advance(&mut self, budget: &mut u64) -> Result<Option<Option<TileWater>>, Error> {
        let Some((tile, own)) =
            generated(&mut self.generator, budget, (&self.layers, &self.pyramid))?
        else {
            return Ok(None);
        };
        let held = (&tile, own.as_ref());
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
                steadied(held, pyramid, (cell, &cells), &solids)
            }
            SettledReads::Summaries(summaries) => flowing(held, pyramid, &|i, j| {
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
