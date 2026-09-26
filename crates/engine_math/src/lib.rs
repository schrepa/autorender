//! Branch-free trigonometry for the engine and games.
//!
//! Browsers compile a data-dependent branch or a float `select` in Wasm as a
//! conditional jump. The standard library's `sinf`/`cosf` (musl, through
//! `compiler_builtins`) branches on each angle's magnitude and quadrant, so
//! angles that differ per object mispredict on most calls. These functions
//! choose quadrants with integer bit operations instead, and give the same
//! bits on native targets and in every browser (pure IEEE `f32` arithmetic,
//! no fused multiply-add). See `docs/math.md` for measurements.
//!
//! Accuracy against the exact values, rounded to `f32`:
//! - `|x| <= pi/4`: within 1 ulp; `|x| <= pi`: within 2 ulp.
//! - `|x| <=` [`REDUCTION_LIMIT`]: absolute error below `1.2e-7`.
//! - Larger, infinite or NaN `x`: musl's `sincosf` (within 1 ulp; NaN for
//!   infinities and NaN).
//!
//! The glam copy in `vendor/glam` routes its `f32` sine and cosine here, so
//! `Quat::from_rotation_y`, `from_axis_angle`, `from_euler`, `slerp` and the
//! other rotation constructors use the same routine.
#![no_std]

/// Largest `|x|` reduced by the fast path: quadrant indices below 2^13 keep
/// the reduction's first two products exact.
pub const REDUCTION_LIMIT: f32 = 12_866.0;

/// Shared constants of the reduction and the polynomials, for the SIMD
/// variant in `engine_core::math`. Every lane must use exactly these
/// operations to stay bit-identical to [`sin_cos`].
#[doc(hidden)]
pub mod lanes {
    /// 1.5 * 2^23: adding it rounds to the nearest integer and leaves the
    /// quadrant in the low mantissa bits.
    pub const SHIFTER: f32 = 12_582_912.0;
    pub const FRAC_2_PI: f32 = core::f32::consts::FRAC_2_PI;
    /// pi/2 in three parts (Cephes' DP1..DP3, doubled). The first has 8 and
    /// the second 11 significant bits, so their products with a quadrant
    /// index below 2^13 are exact.
    pub const PIO2_A: f32 = 1.570_312_5;
    pub const PIO2_B: f32 = 4.837_513e-4;
    pub const PIO2_C: f32 = 7.549_79e-8;
    /// Cephes `sinf`/`cosf` minimax coefficients on [-pi/4, pi/4].
    pub const S1: f32 = -1.666_665_5e-1;
    pub const S2: f32 = 8.332_161e-3;
    pub const S3: f32 = -1.951_529_6e-4;
    pub const C1: f32 = 4.166_664_6e-2;
    pub const C2: f32 = -1.388_731_6e-3;
    pub const C3: f32 = 2.443_315_7e-5;
}

/// Sine and cosine of `x` radians. See the crate documentation for accuracy.
#[inline]
pub fn sin_cos(x: f32) -> (f32, f32) {
    use lanes::*;
    // False for NaN, which musl handles with the other large arguments.
    let reducible = x.abs() <= REDUCTION_LIMIT;
    if !reducible {
        return large(x);
    }
    let shifted = x * FRAC_2_PI + SHIFTER;
    let j = shifted - SHIFTER;
    let quadrant = shifted.to_bits();
    let y = ((x - j * PIO2_A) - j * PIO2_B) - j * PIO2_C;
    let z = y * y;
    // On [-pi/4, pi/4] the sine has the sign of `y`; copying it keeps
    // sin(-0) = -0, which `y + y*z*p` would round to +0.
    let s = ((y + y * z * (S1 + z * (S2 + z * S3))).to_bits() & 0x7fff_ffff)
        | (y.to_bits() & 0x8000_0000);
    let c = (1.0 - 0.5 * z + z * z * (C1 + z * (C2 + z * C3))).to_bits();
    // Odd quadrants swap sine and cosine; quadrants 2-3 negate the sine and
    // 1-2 the cosine. Bit operations, not selects: see the crate docs.
    let swap = (quadrant & 1).wrapping_neg();
    let sin = s ^ ((s ^ c) & swap);
    let cos = c ^ ((s ^ c) & swap);
    (
        f32::from_bits(sin ^ ((quadrant & 2) << 30)),
        f32::from_bits(cos ^ ((quadrant.wrapping_add(1) & 2) << 30)),
    )
}

/// Sine of `x` radians, as [`sin_cos`].
#[inline]
pub fn sin(x: f32) -> f32 {
    sin_cos(x).0
}

/// Cosine of `x` radians, as [`sin_cos`].
#[inline]
pub fn cos(x: f32) -> f32 {
    sin_cos(x).1
}

/// Arguments beyond the fast path's exact reduction: rare, so a
/// well-predicted branch to musl (identical on every platform).
#[cold]
#[inline(never)]
fn large(x: f32) -> (f32, f32) {
    libm::sincosf(x)
}

#[cfg(test)]
mod tests;
