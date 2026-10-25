//! Error pyramids: how far the triangles a subdivision draws a tile's
//! surfaces with may stray from them, bounded once per tile, so a renderer
//! splitting its ground or its water's surface to a pixel of error reads the
//! bound under a triangle instead of testing points of it. Each of a tile's
//! two pyramids, its ground's over its heights and its water's over its
//! water's surface ([`crate::TileWater::surfaces`]), holds for every cell of 2^k of
//! the tile's own samples along each side (k from 1 to [`ERROR_MIPS`], on the
//! grid of its samples 0 to 256) the most any triangle of longest edge
//! bisection lying in that cell, its corners standing on the surface, strays
//! from the surface its samples stand at, on the triangles every consumer
//! splits cells into: meters, rounded up to their half float so a bound
//! stays one.
//!
//! The bound comes from each cell's bilinear interpolation of its corners. A
//! cell of one quad strays from it by its twist t, a quarter of |h00 + h11 -
//! h10 - h01|, where its two triangles and the interpolation part most, at
//! its middle. A wider cell strays from its own by E, the most over its four
//! quarters of their own E and of how far its interpolation strays from
//! theirs, which, both bilinear, peaks at the quarters' corners, all
//! samples. A triangle filling half the cell then strays by at most E + t,
//! and one filling a quarter of it cut by both diagonals, its third corner
//! the cell's middle sample, by at most 2E + t: what the pyramid holds. A
//! water's pyramid marks the cells it cannot bound: [`DRY_CELL`] where no
//! sample is wet, so no water draws there, and [`WIDE_SHORE`] where a wet
//! sample shares a cell with one holding no surface. Every host computes the
//! same bits: fixed orders of plain sums and products.

use std::sync::OnceLock;

use engine_math::max;

use crate::steady_water::{f16_bits, f16_value};
use crate::{TILE_QUADS, TILE_STRIDE};

/// Levels of an error pyramid, from cells of 2 samples a side to one cell of
/// a tile's [`TILE_QUADS`]: cells of one quad are never held, as no triangle
/// narrower than the samples' spacing draws them better.
pub const ERROR_MIPS: u32 = TILE_QUADS.trailing_zeros();
/// Cells along a side of an error pyramid's finest level, each coarser
/// level holding half as many.
pub const ERROR_CELLS: u32 = TILE_QUADS / 2;
/// Cells of an error pyramid, all its levels.
pub const ERROR_TEXELS: usize = (4_usize.pow(ERROR_MIPS) - 1) / 3;
/// What a water's pyramid holds over a cell no wet sample lies in: no water
/// draws over it.
pub const DRY_CELL: f32 = -1.0;
/// What a water's pyramid holds over a cell holding a wet sample and one
/// holding no surface, a shore wider than the surface continues under its
/// banks ([`crate::WATER_REACH`]) or a levee it stops at: the largest half
/// float, so a subdivision splits until its pieces lie where the surface
/// reaches.
pub const WIDE_SHORE: f32 = 65504.0;

/// One of a tile's error pyramids: its cells' bounds as the bits of their
/// half floats, its levels one after another from the finest, each row by
/// row along z.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorPyramid {
    halves: Box<[u16]>,
}

impl ErrorPyramid {
    /// The pyramid of the ground standing at `heights`, a tile's samples in
    /// their stored layout ([`TILE_STRIDE`] a row, its border included).
    pub(crate) fn ground(heights: &[f32]) -> Self {
        Self::over(heights, None)
    }

    /// The pyramid of water whose surface stands at `surfaces` (NaN where no
    /// sample holds one) over `heights`, both a tile's samples in their
    /// stored layout.
    pub(crate) fn water(surfaces: &[f32], heights: &[f32]) -> Self {
        let wet: Vec<bool> = surfaces.iter().zip(heights).map(|(s, h)| s > h).collect();
        Self::over(surfaces, Some(&wet[..]))
    }

    /// The water's pyramid of a tile no water stands on: [`DRY_CELL`] over
    /// every cell.
    pub fn dry() -> &'static Self {
        static DRY: OnceLock<ErrorPyramid> = OnceLock::new();
        DRY.get_or_init(|| ErrorPyramid {
            halves: vec![half_up(DRY_CELL); ERROR_TEXELS].into_boxed_slice(),
        })
    }

    /// The bits of the half floats of level `mip`, whose cells span
    /// 2^(mip + 1) samples a side: [`ERROR_CELLS`] cells a side at the
    /// finest (0), each coarser level half as many, row by row along z.
    pub fn mip(&self, mip: u32) -> &[u16] {
        let side = |level: u32| (ERROR_CELLS >> level) as usize;
        let start: usize = (0..mip).map(|finer| side(finer) * side(finer)).sum();
        &self.halves[start..start + side(mip) * side(mip)]
    }

    /// The pyramid over the samples standing at `values` (a tile's, in their
    /// stored layout), a water's where `wet` marks its wet samples: each
    /// cell's bound, rounded up, and over a water's surface [`DRY_CELL`]
    /// where no sample of the cell is wet and [`WIDE_SHORE`] where one is and
    /// another holds no surface (NaN).
    fn over(values: &[f32], wet: Option<&[bool]>) -> Self {
        let (mut bounds, mut wets) = quads(values, wet);
        let mut held = Vec::with_capacity(ERROR_TEXELS);
        let mut cells = TILE_QUADS as usize;
        for mip in 0..ERROR_MIPS {
            let finer = (&bounds[..], &wets[..]);
            let coarse = coarser(values, finer, (cells, 2 << mip), wet.is_some(), &mut held);
            (bounds, wets) = coarse;
            cells /= 2;
        }
        Self {
            halves: held.into_boxed_slice(),
        }
    }
}

/// Where sample `(x, z)` of a tile's own (0 to 256 along each axis) lies in
/// its stored layout.
fn stored(x: usize, z: usize) -> usize {
    (z + 1) * TILE_STRIDE as usize + x + 1
}

/// The twist of a cell whose corners stand at `[h00, h10, h01, h11]`: the
/// most its corners' bilinear interpolation and either pair of triangles
/// over it part, at its middle.
fn twist([h00, h10, h01, h11]: [f32; 4]) -> f32 {
    0.25 * ((h00 + h11) - (h10 + h01)).abs()
}

/// Each quad of a tile's own samples standing at `values` (its stored
/// layout), row by row along z: how far its triangles stray from its
/// corners' bilinear interpolation (its twist; infinite where a corner holds
/// none), and whether `wet` marks a corner of it.
fn quads(values: &[f32], wet: Option<&[bool]>) -> (Vec<f32>, Vec<bool>) {
    let side = TILE_QUADS as usize;
    let mut bounds = Vec::with_capacity(side * side);
    let mut wets = Vec::with_capacity(side * side);
    for z in 0..side {
        for x in 0..side {
            let at = [
                stored(x, z),
                stored(x + 1, z),
                stored(x, z + 1),
                stored(x + 1, z + 1),
            ];
            let corners = at.map(|sample| values[sample]);
            let bound = if corners.iter().all(|corner| corner.is_finite()) {
                twist(corners)
            } else {
                f32::INFINITY
            };
            bounds.push(bound);
            wets.push(wet.is_some_and(|wet| at.iter().any(|&sample| wet[sample])));
        }
    }
    (bounds, wets)
}

/// The cells of the next coarser level, `span` samples a side, from the
/// bounds and wet marks of their quarters (`finer`, `cells` a side), over a
/// tile's samples standing at `values`: each one's bound, the most over its
/// quarters of their own and of how far its corners' bilinear interpolation
/// strays from theirs, at their corners (infinite where a quarter's is), and
/// whether a quarter is wet; and, pushed to `held`, what the pyramid holds of
/// it: twice its bound and its twist, rounded up, but the marks of a water's
/// pyramid (`marks`).
fn coarser(
    values: &[f32],
    (bounds, wets): (&[f32], &[bool]),
    (cells, span): (usize, usize),
    marks: bool,
    held: &mut Vec<u16>,
) -> (Vec<f32>, Vec<bool>) {
    let (next, half) = (cells / 2, span / 2);
    let mut coarse_bounds = Vec::with_capacity(next * next);
    let mut coarse_wets = Vec::with_capacity(next * next);
    for j in 0..next {
        for i in 0..next {
            let quarter = |di: usize, dj: usize| (2 * j + dj) * cells + 2 * i + di;
            let quarters = [quarter(0, 0), quarter(1, 0), quarter(0, 1), quarter(1, 1)];
            let wet = quarters.iter().any(|&index| wets[index]);
            let (x, z) = (i * span, j * span);
            let at = |dx: usize, dz: usize| values[stored(x + dx, z + dz)];
            let corners = [at(0, 0), at(span, 0), at(0, span), at(span, span)];
            let bound = if quarters.iter().any(|&index| bounds[index].is_infinite()) {
                f32::INFINITY
            } else {
                let [h00, h10, h01, h11] = corners;
                // How far the samples at its sides' middles and its middle
                // stand from its corners' interpolation.
                let bottom = (at(half, 0) - 0.5 * (h00 + h10)).abs();
                let top = (at(half, span) - 0.5 * (h01 + h11)).abs();
                let left = (at(0, half) - 0.5 * (h00 + h01)).abs();
                let right = (at(span, half) - 0.5 * (h10 + h11)).abs();
                let middle = (at(half, half) - 0.25 * ((h00 + h10) + (h01 + h11))).abs();
                // The most each quarter's corners do, in the quarters' order.
                let strays = [
                    max(max(bottom, left), middle),
                    max(max(bottom, right), middle),
                    max(max(left, top), middle),
                    max(max(right, top), middle),
                ];
                let mut most = 0.0;
                for (&index, stray) in quarters.iter().zip(strays) {
                    most = max(most, bounds[index] + stray);
                }
                most
            };
            let value = if marks && !wet {
                DRY_CELL
            } else if bound.is_infinite() {
                WIDE_SHORE
            } else {
                2.0 * bound + twist(corners)
            };
            held.push(half_up(value));
            coarse_bounds.push(bound);
            coarse_wets.push(wet);
        }
    }
    (coarse_bounds, coarse_wets)
}

/// The bits of the least half float at or above `value`, within the largest
/// one: a bound rounded up stays a bound.
fn half_up(value: f32) -> u16 {
    let bits = f16_bits(value);
    if value > 0.0 && f16_value(bits) < value {
        (bits + 1).min(f16_bits(WIDE_SHORE))
    } else {
        bits
    }
}
