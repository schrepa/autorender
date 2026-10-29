use std::ops::Range;
use std::sync::Arc;

use engine_core::TILE_SAMPLES;
use engine_terrain::Blend;
use engine_terrain::layer::{Below, LayerTile, Row, TileFrame};
use glam::Vec4;

use crate::tables::{NoiseTables, state_len};

pub(crate) const GRID: usize = 69;
pub(crate) const GRID_FIRST: i64 = -2;
pub(crate) const GRID_STEP: i64 = 4;
pub(crate) const COARSE_COST: u64 = 24;
pub(crate) const NOISE_SAMPLE_COST: u64 = 4;

#[inline(always)]
pub(crate) fn interpolate(out: &mut [f32], weights: [f32; 4], taps: [&[f32]; 4]) {
    let [a, b, c, d] = taps;
    for ((((value, a), b), c), d) in out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
        *value = weights[0] * a + weights[1] * b + weights[2] * c + weights[3] * d;
    }
}

pub(crate) const QUARTER_WEIGHTS: [[f32; 4]; 4] = [
    [0.0, 1.0, 0.0, 0.0],
    [-0.070_312_5, 0.867_187_5, 0.226_562_5, -0.023_437_5],
    [-0.0625, 0.5625, 0.5625, -0.0625],
    [-0.023_437_5, 0.226_562_5, 0.867_187_5, -0.070_312_5],
];

const LANE_WEIGHTS: [Vec4; 4] = {
    let w = QUARTER_WEIGHTS;
    [
        Vec4::new(w[0][0], w[1][0], w[2][0], w[3][0]),
        Vec4::new(w[0][1], w[1][1], w[2][1], w[3][1]),
        Vec4::new(w[0][2], w[1][2], w[2][2], w[3][2]),
        Vec4::new(w[0][3], w[1][3], w[2][3], w[3][3]),
    ]
};

#[inline(always)]
fn interpolate4(out: &mut [Vec4], taps: [&[f32]; 4]) {
    let [a, b, c, d] = taps;
    let [wa, wb, wc, wd] = LANE_WEIGHTS;
    for ((((value, &a), &b), &c), &d) in out.iter_mut().zip(a).zip(b).zip(c).zip(d) {
        *value = wa * a + wb * b + wc * c + wd * d;
    }
}

const HEIGHT_ROWS: (i32, i32) = (-1, TILE_SAMPLES as i32 + 1);

fn grid_rows() -> (usize, usize) {
    let (first, end) = HEIGHT_ROWS;
    let low = i64::from(first).div_euclid(GRID_STEP) - GRID_FIRST - 1;
    let high = i64::from(end - 1).div_euclid(GRID_STEP) - GRID_FIRST + 3;
    (low as usize, std::cmp::min(high as usize, GRID))
}

pub(crate) struct NoiseTile {
    tables: Arc<NoiseTables>,
    blend: Blend,
    first: [i64; 2],
    spacing: f32,
    octaves: usize,
    fine: Range<usize>,
    gullies: usize,
    next: usize,
    grid: Box<[f32]>,
    column: Box<[f32]>,
    state: Box<[f32]>,
    lanes: Box<[Vec4]>,
    values: Box<[f32]>,
}

impl NoiseTile {
    pub(crate) fn new(tables: Arc<NoiseTables>, frame: &TileFrame<'_>) -> Self {
        let (octaves, fine, gullies) = tables.bands(frame.spacing);
        let len = state_len(octaves);
        Self {
            blend: tables.params.blend,
            tables,
            first: frame.first,
            spacing: frame.spacing,
            octaves,
            fine,
            gullies,
            next: grid_rows().0,
            grid: vec![0.0; GRID * GRID * len].into_boxed_slice(),
            column: vec![0.0; GRID * len].into_boxed_slice(),
            state: vec![0.0; len].into_boxed_slice(),
            lanes: vec![Vec4::ZERO; len].into_boxed_slice(),
            values: vec![0.0; frame.xs.len()].into_boxed_slice(),
        }
    }

    fn coarse_row(&mut self, row: usize) {
        let (coarse, gullies) = (self.octaves, self.gullies);
        let z = (self.first[1] + (GRID_FIRST + row as i64) * GRID_STEP) as f32 * self.spacing;
        let (tables, grid) = (&self.tables, &mut self.grid);
        let len = state_len(coarse);
        let (first, spacing) = (self.first[0], self.spacing);
        let x = |column: usize| (first + (GRID_FIRST + column as i64) * GRID_STEP) as f32 * spacing;
        let mut column = 0;
        while column < GRID {
            let at = (row * GRID + column) * len;
            if column + 4 <= GRID {
                let xs = Vec4::from_array([0, 1, 2, 3].map(|lane| x(column + lane)));
                tables.coarse4(xs, z, coarse, gullies, &mut grid[at..at + 4 * len]);
                column += 4;
            } else {
                tables.coarse(x(column), z, coarse, gullies, &mut grid[at..at + len]);
                column += 1;
            }
        }
    }
}

impl LayerTile for NoiseTile {
    fn prepare(&mut self, budget: &mut u64) -> bool {
        let end = grid_rows().1;
        while self.next < end {
            if *budget == 0 {
                return false;
            }
            self.coarse_row(self.next);
            *budget = budget.saturating_sub(GRID as u64 * COARSE_COST);
            self.next += 1;
        }
        true
    }

    fn apply_row(&mut self, _below: Below<'_>, row: i32, line: &Row<'_>, heights: &mut [f32]) {
        let z = line.z;
        let coarse_row = i64::from(row).div_euclid(GRID_STEP);
        let vertical = QUARTER_WEIGHTS[i64::from(row).rem_euclid(GRID_STEP) as usize];
        let (octaves, fine) = (self.octaves, self.fine.clone());
        let len = state_len(octaves);
        let grid = &self.grid;
        let base = (coarse_row - 1 - GRID_FIRST) as usize;
        let tap = |row: usize, column: usize| {
            let at = ((base + row) * GRID + column) * len;
            &grid[at..at + len]
        };
        for (column, sum) in self.column.chunks_exact_mut(len).take(GRID).enumerate() {
            let taps = [0, 1, 2, 3].map(|row| tap(row, column));
            interpolate(sum, vertical, taps);
        }
        let sum = &mut self.state[..len];
        let lanes = &mut self.lanes[..len];
        let samples = self.values.len();
        let mut index = 0;
        while index < samples {
            let sample = index as i64 - 1;
            let coarse = sample.div_euclid(GRID_STEP);
            let quarter = sample.rem_euclid(GRID_STEP) as usize;
            let first = (coarse - 1 - GRID_FIRST) as usize;
            let taps = [0, 1, 2, 3].map(|tap| {
                let at = (first + tap) * len;
                &self.column[at..at + len]
            });
            if quarter == 0 && index + 4 <= samples {
                interpolate4(lanes, taps);
                let xs = Vec4::from_slice(&line.xs[index..index + 4]);
                self.tables
                    .finish4(xs, z, lanes, octaves, fine.clone())
                    .write_to_slice(&mut self.values[index..index + 4]);
                index += 4;
            } else {
                interpolate(sum, QUARTER_WEIGHTS[quarter], taps);
                self.values[index] =
                    self.tables
                        .finish(line.xs[index], z, sum, octaves, fine.clone());
                index += 1;
            }
        }
        let blend = self.blend;
        for (height, &value) in heights.iter_mut().zip(&self.values) {
            *height = blend.apply(*height, value, 1.0);
        }
    }
}
