//! Error pyramids: how far the triangles a subdivision draws a tile's
//! surfaces with may stray from them, bounded once per tile, so a renderer
//! splitting its ground or its water's mesh to a pixel of error reads the
//! bound under a triangle instead of testing points of it. Each of a tile's
//! two pyramids, its ground's over its heights and its water's over the
//! levels its water's mesh stands at ([`crate::WaterMesh`]), holds for every
//! cell of 2^k of the tile's own samples along each side (k from 1 to
//! [`ERROR_MIPS`], on the grid of its samples 0 to 256) the most any triangle
//! of longest edge bisection lying in that cell, its corners standing on the
//! surface, strays from the surface its samples stand at, on the triangles
//! every consumer splits cells into: meters, rounded up to their half float
//! so a bound stays one.
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
//! water's pyramid holds [`DRY_CELL`] over the cells no sample holding the
//! surface its water's pass draws lies in, where no water shows. Every host
//! computes the same bits: fixed orders of plain sums and products.

use std::sync::OnceLock;

use engine_math::max;

use crate::steady_water::{f16_bits, f16_value};
use crate::tile_water::WaterMesh;
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
/// What a water's pyramid holds over a cell no sample holding the surface
/// its water's pass draws lies in ([`crate::WaterMesh::holds`]): no water
/// shows over it.
pub const DRY_CELL: f32 = -1.0;
/// The largest half float, within which a bound rounded up stays.
const LARGEST_HALF: f32 = 65504.0;

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

    /// The pyramid of a tile's water's mesh `mesh`, over the levels it
    /// stands at, marking the samples holding the surface its water's pass
    /// draws.
    pub(crate) fn water(mesh: &WaterMesh) -> Self {
        let holding: Vec<bool> = (0..mesh.levels().len()).map(|at| mesh.holds(at)).collect();
        Self::over(mesh.levels(), Some(&holding[..]))
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
    /// stored layout), a water's mesh's where `holding` marks its samples
    /// holding a surface: each cell's bound, rounded up, and over a water's
    /// mesh [`DRY_CELL`] where no sample of the cell holds one.
    fn over(values: &[f32], holding: Option<&[bool]>) -> Self {
        let (mut bounds, mut holds) = quads(values, holding);
        let mut held = Vec::with_capacity(ERROR_TEXELS);
        let mut cells = TILE_QUADS as usize;
        let marks = holding.is_some();
        for mip in 0..ERROR_MIPS {
            let finer = (&bounds[..], &holds[..]);
            let coarse = coarser(values, finer, (cells, 2 << mip), marks, &mut held);
            (bounds, holds) = coarse;
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
/// corners' bilinear interpolation (its twist), and whether `holding` marks
/// a corner of it.
fn quads(values: &[f32], holding: Option<&[bool]>) -> (Vec<f32>, Vec<bool>) {
    let side = TILE_QUADS as usize;
    let mut bounds = Vec::with_capacity(side * side);
    let mut holds = Vec::with_capacity(side * side);
    for z in 0..side {
        for x in 0..side {
            let at = [
                stored(x, z),
                stored(x + 1, z),
                stored(x, z + 1),
                stored(x + 1, z + 1),
            ];
            bounds.push(twist(at.map(|sample| values[sample])));
            holds.push(holding.is_some_and(|holding| at.iter().any(|&sample| holding[sample])));
        }
    }
    (bounds, holds)
}

/// The cells of the next coarser level, `span` samples a side, from the
/// bounds and marks of their quarters (`finer`, `cells` a side), over a
/// tile's samples standing at `values`: each one's bound, the most over its
/// quarters of their own and of how far its corners' bilinear interpolation
/// strays from theirs, at their corners, and whether a quarter holds a
/// sample a water's mesh marks; and, pushed to `held`, what the pyramid
/// holds of it: twice its bound and its twist, rounded up, but the marks of
/// a water's pyramid (`marks`).
fn coarser(
    values: &[f32],
    (bounds, holds): (&[f32], &[bool]),
    (cells, span): (usize, usize),
    marks: bool,
    held: &mut Vec<u16>,
) -> (Vec<f32>, Vec<bool>) {
    let (next, half) = (cells / 2, span / 2);
    let mut coarse_bounds = Vec::with_capacity(next * next);
    let mut coarse_holds = Vec::with_capacity(next * next);
    for j in 0..next {
        for i in 0..next {
            let quarter = |di: usize, dj: usize| (2 * j + dj) * cells + 2 * i + di;
            let quarters = [quarter(0, 0), quarter(1, 0), quarter(0, 1), quarter(1, 1)];
            let holding = quarters.iter().any(|&index| holds[index]);
            let (x, z) = (i * span, j * span);
            let at = |dx: usize, dz: usize| values[stored(x + dx, z + dz)];
            let corners = [at(0, 0), at(span, 0), at(0, span), at(span, span)];
            let [h00, h10, h01, h11] = corners;
            // How far the samples at its sides' middles and its middle stand
            // from its corners' interpolation.
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
            let mut bound = 0.0;
            for (&index, stray) in quarters.iter().zip(strays) {
                bound = max(bound, bounds[index] + stray);
            }
            let value = if marks && !holding {
                DRY_CELL
            } else {
                2.0 * bound + twist(corners)
            };
            held.push(half_up(value));
            coarse_bounds.push(bound);
            coarse_holds.push(holding);
        }
    }
    (coarse_bounds, coarse_holds)
}

/// The bits of the least half float at or above `value`, within the largest
/// one: a bound rounded up stays a bound.
fn half_up(value: f32) -> u16 {
    let bits = f16_bits(value);
    if value > 0.0 && f16_value(bits) < value {
        (bits + 1).min(f16_bits(LARGEST_HALF))
    } else {
        bits
    }
}
