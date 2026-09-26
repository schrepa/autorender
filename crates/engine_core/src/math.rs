//! Trigonometry for gameplay and engine code (`docs/math.md`).
//!
//! Use these instead of `f32::sin`, `f32::cos` and `f32::sin_cos`: browsers
//! run the standard functions 2-5 times slower on angles that differ per
//! object, and their results differ between native builds and Wasm. glam's
//! rotation constructors and `Quat::slerp` use the same routine through the
//! vendored glam (`vendor/glam/VENDORED.md`).

use glam::Vec4;

pub use engine_math::{REDUCTION_LIMIT, cos, sin, sin_cos};

/// [`sin_cos`] of four angles in SIMD lanes, lane for lane the same bits as
/// the scalar function; about twice its throughput for bulk work.
#[inline]
pub fn sin_cos4(x: Vec4) -> (Vec4, Vec4) {
    use engine_math::lanes::*;
    // NaN compares false, so infinities, NaN and huge angles take the scalar
    // fallback like `sin_cos`.
    if !x.abs().cmple(Vec4::splat(REDUCTION_LIMIT)).all() {
        let lanes = x.to_array().map(sin_cos);
        return (
            Vec4::from_array(lanes.map(|(s, _)| s)),
            Vec4::from_array(lanes.map(|(_, c)| c)),
        );
    }
    // The scalar function's operations, lane for lane.
    let shifted = x * FRAC_2_PI + SHIFTER;
    let j = shifted - SHIFTER;
    let y = ((x - j * PIO2_A) - j * PIO2_B) - j * PIO2_C;
    let z = y * y;
    let s = (y + y * z * (S1 + z * (S2 + z * S3))).copysign(y);
    let c = 1.0 - 0.5 * z + z * z * (C1 + z * (C2 + z * C3));
    // Quadrant j mod 4 (exact: |j| < 2^13), applied with lane blends.
    let quadrant = j - (j * 0.25).floor() * 4.0;
    let first = quadrant.cmpeq(Vec4::ONE);
    let swap = first | quadrant.cmpeq(Vec4::splat(3.0));
    let (sin, cos) = (Vec4::select(swap, c, s), Vec4::select(swap, s, c));
    let negate_sin = quadrant.cmpge(Vec4::splat(2.0));
    let negate_cos = first | quadrant.cmpeq(Vec4::splat(2.0));
    (
        Vec4::select(negate_sin, -sin, sin),
        Vec4::select(negate_cos, -cos, cos),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn angles() -> impl Iterator<Item = f32> {
        let mut state = 0x2545_f491_u32;
        let special = [
            0.0,
            -0.0,
            f32::from_bits(1),
            core::f32::consts::FRAC_PI_4,
            -core::f32::consts::PI,
            REDUCTION_LIMIT,
            -REDUCTION_LIMIT,
            f32::from_bits(REDUCTION_LIMIT.to_bits() + 1),
            1.0e9,
            f32::INFINITY,
            f32::NAN,
        ];
        special.into_iter().chain((0..400_000).map(move |i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = state as f32 / u32::MAX as f32;
            // Mostly the fast range, some beyond it.
            let limit = if i % 97 == 0 {
                40_000.0
            } else {
                REDUCTION_LIMIT
            };
            (unit * 2.0 - 1.0) * limit
        }))
    }

    #[test]
    fn four_lanes_match_the_scalar_function_bit_for_bit() {
        let values: Vec<f32> = angles().collect();
        for chunk in values.as_chunks::<4>().0 {
            let (s, c) = sin_cos4(Vec4::from_array(*chunk));
            for (lane, &x) in chunk.iter().enumerate() {
                let (es, ec) = sin_cos(x);
                let same =
                    |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
                assert!(same(s[lane], es) && same(c[lane], ec), "{x}");
            }
        }
    }

    #[test]
    fn glam_rotations_use_the_engine_routine() {
        // Fails if the vendored glam patch is not in effect.
        for x in angles().filter(|x| x.is_finite()).take(10_000) {
            let (s, c) = sin_cos(x * 0.5);
            let expected = [0.0, s, 0.0, c].map(f32::to_bits);
            assert_eq!(
                Quat::from_rotation_y(x).to_array().map(f32::to_bits),
                expected,
                "{x}"
            );
        }
    }
}
