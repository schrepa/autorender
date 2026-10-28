use std::sync::OnceLock;

use engine_math::max;

use crate::steady_water::{f16_bits, f16_value};
use crate::tile_water::WaterMesh;
use crate::{TILE_QUADS, TILE_STRIDE};

pub const ERROR_MIPS: u32 = TILE_QUADS.trailing_zeros();
pub const ERROR_CELLS: u32 = TILE_QUADS / 2;
pub const ERROR_TEXELS: usize = (4_usize.pow(ERROR_MIPS) - 1) / 3;
pub const DRY_CELL: f32 = -1.0;
const LARGEST_HALF: f32 = 65504.0;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorPyramid {
    halves: Box<[u16]>,
}

impl ErrorPyramid {
    pub(crate) fn ground(heights: &[f32]) -> Self {
        Self::over(heights, None)
    }

    pub(crate) fn water(mesh: &WaterMesh) -> Self {
        let holding: Vec<bool> = (0..mesh.levels().len()).map(|at| mesh.holds(at)).collect();
        Self::over(mesh.levels(), Some(&holding[..]))
    }

    pub fn dry() -> &'static Self {
        static DRY: OnceLock<ErrorPyramid> = OnceLock::new();
        DRY.get_or_init(|| ErrorPyramid {
            halves: vec![half_up(DRY_CELL); ERROR_TEXELS].into_boxed_slice(),
        })
    }

    pub fn mip(&self, mip: u32) -> &[u16] {
        let side = |level: u32| (ERROR_CELLS >> level) as usize;
        let start: usize = (0..mip).map(|finer| side(finer) * side(finer)).sum();
        &self.halves[start..start + side(mip) * side(mip)]
    }

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

fn stored(x: usize, z: usize) -> usize {
    (z + 1) * TILE_STRIDE as usize + x + 1
}

fn twist([h00, h10, h01, h11]: [f32; 4]) -> f32 {
    0.25 * ((h00 + h11) - (h10 + h01)).abs()
}

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
            let bottom = (at(half, 0) - 0.5 * (h00 + h10)).abs();
            let top = (at(half, span) - 0.5 * (h01 + h11)).abs();
            let left = (at(0, half) - 0.5 * (h00 + h01)).abs();
            let right = (at(span, half) - 0.5 * (h10 + h11)).abs();
            let middle = (at(half, half) - 0.25 * ((h00 + h10) + (h01 + h11))).abs();
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

fn half_up(value: f32) -> u16 {
    let bits = f16_bits(value);
    if value > 0.0 && f16_value(bits) < value {
        (bits + 1).min(f16_bits(LARGEST_HALF))
    } else {
        bits
    }
}
