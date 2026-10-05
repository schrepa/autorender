//! Tiles: their samples, bounds and holes, sampling within them, and their
//! generation from a layer stack, a step at a time.

use std::ops::Range;
use std::sync::Arc;

use glam::{Vec2, Vec3};

use engine_math::{max, min};

use crate::layers::{Layers, state_len};
use crate::{
    PATCH_QUADS, PATCHES_PER_TILE, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey,
};

/// Cells with a hole bit: those starting at samples 0..=256, so tiles agree
/// on the cells along their shared edges.
const HOLE_CELLS: usize = TILE_SAMPLES as usize;
const HOLE_WORDS: usize = (HOLE_CELLS * HOLE_CELLS).div_ceil(64);

/// A tile's samples: heights in meters above the terrain's origin of
/// samples -1 to 257 along x and z (its samples and a border of one), row
/// by row along z, with their bounds and, where a stamp cut them, holes.
#[derive(Clone, Debug, PartialEq)]
pub struct Tile {
    key: TileKey,
    spacing: f32,
    heights: Box<[f32]>,
    holes: Option<Box<[u64]>>,
    bounds: [f32; 2],
    patch_bounds: Box<[[f32; 2]]>,
    patch_holes: [u64; 4],
}

/// The surface at a point: its height, the normal of its triangle and
/// whether its cell is a hole.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileSample {
    pub height: f32,
    pub normal: Vec3,
    pub hole: bool,
}

impl Tile {
    /// A tile of `key` at sample spacing `spacing` from its
    /// [`TILE_STRIDE`]-squared `heights` and optional hole bits (one per
    /// cell starting at samples 0..=256, row by row). `None` when the sizes
    /// do not match or a height is not finite.
    pub fn new(
        key: TileKey,
        spacing: f32,
        heights: Box<[f32]>,
        holes: Option<Box<[u64]>>,
    ) -> Option<Self> {
        let stride = TILE_STRIDE as usize;
        if heights.len() != stride * stride
            || holes.as_ref().is_some_and(|bits| bits.len() != HOLE_WORDS)
            || !heights.iter().all(|height| height.is_finite())
        {
            return None;
        }
        let holes = holes.filter(|bits| bits.iter().any(|&word| word != 0));
        let mut tile = Self {
            key,
            spacing,
            heights,
            holes,
            bounds: [0.0; 2],
            patch_bounds: vec![[0.0; 2]; (PATCHES_PER_TILE * PATCHES_PER_TILE) as usize]
                .into_boxed_slice(),
            patch_holes: [0; 4],
        };
        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        let patch = PATCH_QUADS as i32;
        for pz in 0..PATCHES_PER_TILE as i32 {
            for px in 0..PATCHES_PER_TILE as i32 {
                let mut patch_low = f32::INFINITY;
                let mut patch_high = f32::NEG_INFINITY;
                for z in pz * patch..=(pz + 1) * patch {
                    for x in px * patch..=(px + 1) * patch {
                        let height = tile.height(x, z);
                        patch_low = min(patch_low, height);
                        patch_high = max(patch_high, height);
                    }
                }
                let index = (pz * PATCHES_PER_TILE as i32 + px) as usize;
                tile.patch_bounds[index] = [patch_low, patch_high];
                low = min(low, patch_low);
                high = max(high, patch_high);
                let holed = (pz * patch..(pz + 1) * patch).any(|z| {
                    (px * patch..(px + 1) * patch).any(|x| tile.is_hole(x as u32, z as u32))
                });
                if holed {
                    tile.patch_holes[index / 64] |= 1 << (index % 64);
                }
            }
        }
        tile.bounds = [low, high];
        Some(tile)
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    /// Meters between samples.
    pub fn spacing(&self) -> f32 {
        self.spacing
    }

    /// Height of sample `(x, z)`, -1 to 257 along each axis.
    #[inline]
    pub fn height(&self, x: i32, z: i32) -> f32 {
        let stride = TILE_STRIDE as i32;
        self.heights[((z + 1) * stride + x + 1) as usize]
    }

    /// The largest difference, in meters, between this tile's heights and
    /// those of `parent`, the tile of the next coarser level holding it,
    /// between its samples: what this level's patches move by as they
    /// morph into the parent level's. Every third sample along each axis
    /// is compared, alternately on and between the parent's.
    pub fn morph_error(&self, parent: &Tile) -> f32 {
        let offset = [
            i32::from(self.key.x % 2) * 128,
            i32::from(self.key.z % 2) * 128,
        ];
        let mut largest = 0.0_f32;
        for z in (0..=TILE_QUADS as i32).step_by(3) {
            for x in (0..=TILE_QUADS as i32).step_by(3) {
                // The parent's samples around this one, half as far apart.
                let (px, pz) = (offset[0] * 2 + x, offset[1] * 2 + z);
                let (x0, z0) = (px.div_euclid(2), pz.div_euclid(2));
                let (fx, fz) = ((px % 2) as f32 * 0.5, (pz % 2) as f32 * 0.5);
                let a = parent.height(x0, z0);
                let b = parent.height(x0 + 1, z0);
                let c = parent.height(x0, z0 + 1);
                let d = parent.height(x0 + 1, z0 + 1);
                let coarse = (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fz;
                largest = max(largest, (self.height(x, z) - coarse).abs());
            }
        }
        largest
    }

    /// Every sample, the border included, row by row: the layout GPU tiles
    /// upload.
    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    /// Lowest and highest height of samples 0 to 256.
    pub fn bounds(&self) -> [f32; 2] {
        self.bounds
    }

    /// Lowest and highest height of patch `(x, z)`'s samples.
    pub fn patch_bounds(&self, x: u32, z: u32) -> [f32; 2] {
        self.patch_bounds[(z * PATCHES_PER_TILE + x) as usize]
    }

    /// Whether any cell of patch `(x, z)` is a hole.
    pub fn patch_has_holes(&self, x: u32, z: u32) -> bool {
        let index = (z * PATCHES_PER_TILE + x) as usize;
        self.patch_holes[index / 64] & (1 << (index % 64)) != 0
    }

    /// Hole bits of cells starting at samples 0..=256, row by row; `None`
    /// without holes.
    pub fn hole_bits(&self) -> Option<&[u64]> {
        self.holes.as_deref()
    }

    /// Whether the cell starting at sample `(x, z)` is a hole.
    pub fn is_hole(&self, x: u32, z: u32) -> bool {
        let Some(bits) = &self.holes else {
            return false;
        };
        if x as usize >= HOLE_CELLS || z as usize >= HOLE_CELLS {
            return false;
        }
        let index = z as usize * HOLE_CELLS + x as usize;
        bits[index / 64] & (1 << (index % 64)) != 0
    }

    /// The surface at `local`, meters from the tile's first corner, on the
    /// triangles every consumer splits cells into; `None` outside the tile.
    pub fn sample(&self, local: Vec2) -> Option<TileSample> {
        let size = TILE_QUADS as f32;
        let u = local.x / self.spacing;
        let v = local.y / self.spacing;
        if !(u.is_finite() && v.is_finite()) || u < 0.0 || v < 0.0 || u > size || v > size {
            return None;
        }
        let i = (u.floor() as i32).min(TILE_QUADS as i32 - 1);
        let j = (v.floor() as i32).min(TILE_QUADS as i32 - 1);
        let fu = u - i as f32;
        let fv = v - j as f32;
        let h00 = self.height(i, j);
        let h10 = self.height(i + 1, j);
        let h01 = self.height(i, j + 1);
        let h11 = self.height(i + 1, j + 1);
        // The diagonal runs from the cell's (x, z+1) corner to its (x+1, z)
        // corner.
        let (height, slope) = if fu + fv <= 1.0 {
            (
                h00 + fu * (h10 - h00) + fv * (h01 - h00),
                Vec2::new(h10 - h00, h01 - h00),
            )
        } else {
            (
                h11 + (1.0 - fu) * (h01 - h11) + (1.0 - fv) * (h10 - h11),
                Vec2::new(h11 - h01, h11 - h10),
            )
        };
        let normal = Vec3::new(-slope.x / self.spacing, 1.0, -slope.y / self.spacing).normalize();
        Some(TileSample {
            height,
            normal,
            hole: self.is_hole(i as u32, j as u32),
        })
    }
}

/// Coarse grid points along each axis: every fourth sample from sample -8
/// to 264, enough for Catmull-Rom interpolation at samples -1 to 257.
const GRID: usize = 69;
const GRID_FIRST: i64 = -2;
const GRID_STEP: i64 = 4;
/// Work units of a coarse grid point and of a noise layer's sample,
/// against one per sample of other layers: about their time.
const COARSE_COST: u64 = 24;
const NOISE_SAMPLE_COST: u64 = 4;

/// Four coarse states `taps` combined with Catmull-Rom `weights` into
/// `out`, value by value, in one order wherever states are interpolated.
#[inline(always)]
fn interpolate(out: &mut [f32], weights: [f32; 4], taps: [&[f32]; 4]) {
    let [a, b, c, d] = taps;
    for ((((value, a), b), c), d) in out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
        *value = weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d;
    }
}

/// Catmull-Rom weights at quarter steps, exact in `f32`.
const QUARTER_WEIGHTS: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [-0.070_312_5, 0.867_187_5, 0.226_562_5, -0.023_437_5],
    [-0.0625, 0.5625, 0.5625, -0.0625],
    [-0.023_437_5, 0.226_562_5, 0.867_187_5, -0.070_312_5],
];

#[derive(Clone, Copy, Debug)]
enum Stage {
    Coarse { layer: usize, row: usize },
    Samples { row: i32 },
    Holes { row: u32 },
    Done,
}

/// Generates one tile from a layer stack in steps, so streaming can spread
/// the work over frames. Every sample is a pure function of its position
/// and level: noise octaves at least 32 samples long are evaluated on a
/// grid of every fourth sample and interpolated, shorter ones down to four
/// samples at every sample, so neighboring tiles agree on shared samples.
pub struct Generator {
    layers: Arc<Layers>,
    key: TileKey,
    spacing: f32,
    first: [i64; 2],
    bands: Vec<Option<(usize, Range<usize>, usize)>>,
    // Each noise layer's coarse states, grid point by grid point.
    grids: Vec<Option<Box<[f32]>>>,
    stage: Stage,
    heights: Box<[f32]>,
    holes: Option<Box<[u64]>>,
    xs: Box<[f32]>,
    row: Box<[f32]>,
    values: Box<[f32]>,
    // A row's coarse states interpolated along z, grid column by column,
    // and one sample's.
    column: Box<[f32]>,
    state: Box<[f32]>,
    min: Vec2,
    max: Vec2,
}

impl Generator {
    pub fn new(layers: Arc<Layers>, pyramid: &Pyramid, key: TileKey) -> Self {
        let level = u32::from(key.level);
        let spacing = pyramid.spacing(level);
        let first = pyramid.first_sample(key);
        let stride = TILE_STRIDE as usize;
        let bands: Vec<_> = (0..layers.layers().len())
            .map(|index| layers.noise(index).map(|tables| tables.bands(spacing)))
            .collect();
        let lens: Vec<_> = bands
            .iter()
            .map(|bands| bands.as_ref().map(|(coarse, _, _)| state_len(*coarse)))
            .collect();
        let grids = lens
            .iter()
            .map(|len| len.map(|len| vec![0.0; GRID * GRID * len].into_boxed_slice()))
            .collect();
        let widest = lens.iter().flatten().copied().max().unwrap_or(0);
        let xs = (-1..=TILE_SAMPLES as i64)
            .map(|i| (first[0] + i) as f32 * spacing)
            .collect();
        let min = Vec2::new(
            (first[0] - 1) as f32 * spacing,
            (first[1] - 1) as f32 * spacing,
        );
        let max = Vec2::new(
            (first[0] + i64::from(TILE_SAMPLES)) as f32 * spacing,
            (first[1] + i64::from(TILE_SAMPLES)) as f32 * spacing,
        );
        let holes = layers
            .holes_reach(min, max)
            .then(|| vec![0; HOLE_WORDS].into_boxed_slice());
        Self {
            layers,
            key,
            spacing,
            first,
            bands,
            grids,
            stage: Stage::Coarse { layer: 0, row: 0 },
            heights: vec![0.0; stride * stride].into_boxed_slice(),
            holes,
            xs,
            row: vec![0.0; stride].into_boxed_slice(),
            values: vec![0.0; stride].into_boxed_slice(),
            column: vec![0.0; GRID * widest].into_boxed_slice(),
            state: vec![0.0; widest].into_boxed_slice(),
            min,
            max,
        }
    }

    pub fn key(&self) -> TileKey {
        self.key
    }

    /// Generates until the tile is done or `budget` work units run out
    /// (about one per sample), returning the tile once done. A step that
    /// starts always finishes, so a budget can go slightly negative in
    /// effect; callers pass what is left.
    pub fn advance(&mut self, budget: &mut u64) -> Option<Tile> {
        loop {
            if *budget == 0 {
                return None;
            }
            match self.stage {
                Stage::Coarse { layer, row } => {
                    if layer >= self.grids.len() {
                        self.stage = Stage::Samples { row: -1 };
                        continue;
                    }
                    let Some(tables) = self.layers.noise(layer) else {
                        self.stage = Stage::Coarse {
                            layer: layer + 1,
                            row: 0,
                        };
                        continue;
                    };
                    let (coarse, _, gullies) = self.bands[layer].clone().unwrap_or_default();
                    let z = (self.first[1] + (GRID_FIRST + row as i64) * GRID_STEP) as f32
                        * self.spacing;
                    let grid = self.grids[layer].as_mut().expect("noise layers have grids");
                    let len = state_len(coarse);
                    for column in 0..GRID {
                        let x = (self.first[0] + (GRID_FIRST + column as i64) * GRID_STEP) as f32
                            * self.spacing;
                        let at = (row * GRID + column) * len;
                        tables.coarse(x, z, coarse, gullies, &mut grid[at..at + len]);
                    }
                    *budget = budget.saturating_sub(GRID as u64 * COARSE_COST);
                    self.stage = if row + 1 < GRID {
                        Stage::Coarse {
                            layer,
                            row: row + 1,
                        }
                    } else {
                        Stage::Coarse {
                            layer: layer + 1,
                            row: 0,
                        }
                    };
                }
                Stage::Samples { row } => {
                    self.sample_row(row);
                    let layers = self
                        .grids
                        .iter()
                        .map(|grid| if grid.is_some() { NOISE_SAMPLE_COST } else { 1 })
                        .sum::<u64>();
                    *budget = budget.saturating_sub(u64::from(TILE_STRIDE) * layers.max(1));
                    self.stage = if row < TILE_SAMPLES as i32 {
                        Stage::Samples { row: row + 1 }
                    } else if self.holes.is_some() {
                        Stage::Holes { row: 0 }
                    } else {
                        Stage::Done
                    };
                }
                Stage::Holes { row } => {
                    let z = ((self.first[1] + i64::from(row)) as f32 + 0.5) * self.spacing;
                    let bits = self.holes.as_mut().expect("holes stage has bits");
                    for column in 0..HOLE_CELLS as u32 {
                        let x = ((self.first[0] + i64::from(column)) as f32 + 0.5) * self.spacing;
                        if self.layers.hole_at(x, z) {
                            let index = row as usize * HOLE_CELLS + column as usize;
                            bits[index / 64] |= 1 << (index % 64);
                        }
                    }
                    *budget = budget.saturating_sub(HOLE_CELLS as u64);
                    self.stage = if (row as usize) + 1 < HOLE_CELLS {
                        Stage::Holes { row: row + 1 }
                    } else {
                        Stage::Done
                    };
                }
                Stage::Done => {
                    let heights = std::mem::take(&mut self.heights);
                    return Tile::new(self.key, self.spacing, heights, self.holes.take());
                }
            }
        }
    }

    /// Heights of sample row `row` (-1 to 257), every layer applied in
    /// order.
    fn sample_row(&mut self, row: i32) {
        let z = (self.first[1] + i64::from(row)) as f32 * self.spacing;
        self.row.fill(0.0);
        let coarse_row = i64::from(row).div_euclid(GRID_STEP);
        let vertical = QUARTER_WEIGHTS[i64::from(row).rem_euclid(GRID_STEP) as usize];
        for layer in 0..self.layers.layers().len() {
            let Some(tables) = self.layers.noise(layer) else {
                self.layers
                    .apply_local(layer, &mut self.row, &self.xs, z, self.min, self.max);
                continue;
            };
            let (octaves, fine, _) = self.bands[layer].clone().unwrap_or_default();
            let len = state_len(octaves);
            let grid = self.grids[layer].as_ref().expect("noise layers have grids");
            // The coarse state along z first, for every grid column.
            let base = (coarse_row - 1 - GRID_FIRST) as usize;
            let tap = |row: usize, column: usize| {
                let at = ((base + row) * GRID + column) * len;
                &grid[at..at + len]
            };
            for (column, sum) in self.column.chunks_exact_mut(len).take(GRID).enumerate() {
                let taps = [0, 1, 2, 3].map(|row| tap(row, column));
                interpolate(sum, vertical, taps);
            }
            // Then along x for every sample, which sums every octave.
            let sum = &mut self.state[..len];
            for (index, value) in self.values.iter_mut().enumerate() {
                let sample = index as i64 - 1;
                let coarse = sample.div_euclid(GRID_STEP);
                let weights = QUARTER_WEIGHTS[sample.rem_euclid(GRID_STEP) as usize];
                let first = (coarse - 1 - GRID_FIRST) as usize;
                let taps = [0, 1, 2, 3].map(|tap| {
                    let at = (first + tap) * len;
                    &self.column[at..at + len]
                });
                interpolate(sum, weights, taps);
                *value = tables.finish(self.xs[index], z, sum, octaves, fine.clone());
            }
            self.layers.apply_noise(layer, &mut self.row, &self.values);
        }
        let stride = TILE_STRIDE as usize;
        let start = (row + 1) as usize * stride;
        self.heights[start..start + stride].copy_from_slice(&self.row);
    }
}

impl Layers {
    /// The height of global sample `sample` (x and z, counted from the
    /// terrain's first corner) at `level`, exactly as a generated tile holds
    /// it: the same coarse grid points, interpolation and layer order.
    pub fn sample_height(&self, pyramid: &Pyramid, level: u32, sample: [i64; 2]) -> f32 {
        let spacing = pyramid.spacing(level);
        let x = sample[0] as f32 * spacing;
        let z = sample[1] as f32 * spacing;
        let [cx, cz] = sample.map(|index| index.div_euclid(GRID_STEP));
        let horizontal = QUARTER_WEIGHTS[sample[0].rem_euclid(GRID_STEP) as usize];
        let vertical = QUARTER_WEIGHTS[sample[1].rem_euclid(GRID_STEP) as usize];
        let point = Vec2::new(x, z);
        let mut height = [0.0_f32];
        for layer in 0..self.layers().len() {
            let Some(tables) = self.noise(layer) else {
                self.apply_local(layer, &mut height, &[x], z, point, point);
                continue;
            };
            let (octaves, fine, gullies) = tables.bands(spacing);
            // The coarse state along z first for each column, then along x,
            // as tiles interpolate it.
            let len = state_len(octaves);
            // The sixteen grid points' states, column by column.
            let mut points = vec![0.0_f32; 16 * len];
            for (index, point) in points.chunks_exact_mut(len).enumerate() {
                let column = (cx - 1 + (index / 4) as i64) * GRID_STEP;
                let row = (cz - 1 + (index % 4) as i64) * GRID_STEP;
                tables.coarse(
                    column as f32 * spacing,
                    row as f32 * spacing,
                    octaves,
                    gullies,
                    point,
                );
            }
            let mut columns = vec![0.0_f32; 4 * len];
            for (column, value) in columns.chunks_exact_mut(len).enumerate() {
                let taps = [0, 1, 2, 3].map(|row| {
                    let at = (column * 4 + row) * len;
                    &points[at..at + len]
                });
                interpolate(value, vertical, taps);
            }
            let mut sum = vec![0.0_f32; len];
            let taps = [0, 1, 2, 3].map(|column| &columns[column * len..(column + 1) * len]);
            interpolate(&mut sum, horizontal, taps);
            self.apply_noise(
                layer,
                &mut height,
                &[tables.finish(x, z, &sum, octaves, fine)],
            );
        }
        height[0]
    }

    /// The height at terrain-local `point` on level 0's triangles, as
    /// generated tiles, renderers and collision hold it: four samples of
    /// [`Self::sample_height`].
    pub fn height_at(&self, pyramid: &Pyramid, point: Vec2) -> f32 {
        let spacing = pyramid.spacing(0);
        let u = point.x / spacing;
        let v = point.y / spacing;
        let (i, j) = (u.floor(), v.floor());
        let (fu, fv) = (u - i, v - j);
        let (i, j) = (i as i64, j as i64);
        let h = |di: i64, dj: i64| self.sample_height(pyramid, 0, [i + di, j + dj]);
        if fu + fv <= 1.0 {
            let h00 = h(0, 0);
            h00 + fu * (h(1, 0) - h00) + fv * (h(0, 1) - h00)
        } else {
            let h11 = h(1, 1);
            h11 + (1.0 - fu) * (h(0, 1) - h11) + (1.0 - fv) * (h(1, 0) - h11)
        }
    }
}
