//! Trigonometry, minimum, maximum and clamping for gameplay and engine code
//! (`docs/math.md`).
//!
//! Use these instead of `f32::sin`, `f32::cos` and `f32::sin_cos`: browsers
//! run the standard functions 2-5 times slower on angles that differ per
//! object, and their results differ between native builds and Wasm. glam's
//! rotation constructors and `Quat::slerp` use the same routine through the
//! vendored glam (`vendor/glam/VENDORED.md`). Use [`min`], [`max`] and
//! [`clamp`] instead of the `f32` methods, which browsers compile to
//! data-dependent branches; glam's `Vec3A` and `Vec4` methods are already
//! single SIMD instructions.

use glam::Vec4;

pub use engine_math::{REDUCTION_LIMIT, clamp, cos, max, min, sin, sin_cos};

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

/// Up to four items handled together as SIMD lanes by a [`for_each4`]
/// kernel. Present lanes come first, in item order.
pub struct Lanes4<T> {
    items: [Option<T>; 4],
}

impl<T> Lanes4<T> {
    /// Present lanes: four, or one to three for the last group.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.iter().filter(|item| item.is_some()).count()
    }

    /// Always false: a kernel only sees groups with items.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items[0].is_none()
    }

    /// One `f32` of every lane, read by `field`; absent lanes read zero.
    #[inline]
    pub fn load(&self, field: impl Fn(&T) -> f32) -> Vec4 {
        Vec4::from_array(
            self.items
                .each_ref()
                .map(|item| item.as_ref().map_or(0.0, &field)),
        )
    }

    /// Visits every present lane with its index, e.g. to write results back.
    #[inline]
    pub fn for_each(&mut self, mut visit: impl FnMut(usize, &mut T)) {
        for (lane, item) in self.items.iter_mut().enumerate() {
            if let Some(item) = item {
                visit(lane, item);
            }
        }
    }
}

/// Runs `kernel` on `items` four at a time, then on the remaining one to
/// three, for bulk math in SIMD lanes: load fields into [`Vec4`]s with
/// [`Lanes4::load`], compute (e.g. [`sin_cos4`], which gives every lane the
/// scalar function's bits) and write each item back with
/// [`Lanes4::for_each`].
///
/// ```ignore
/// math::for_each4(query.iter_mut(), |lanes| {
///     let angle = lanes.load(|(spin, _)| spin.speed) * t;
///     let (sin, cos) = math::sin_cos4(angle * 0.5);
///     lanes.for_each(|lane, (_, transform)| {
///         transform.rotation = Quat::from_xyzw(0.0, sin[lane], 0.0, cos[lane]);
///     });
/// });
/// ```
#[inline]
pub fn for_each4<I: IntoIterator>(items: I, mut kernel: impl FnMut(&mut Lanes4<I::Item>)) {
    let mut lanes = Lanes4 {
        items: [None, None, None, None],
    };
    let mut len = 0;
    for item in items {
        lanes.items[len] = Some(item);
        len += 1;
        if len == 4 {
            kernel(&mut lanes);
            lanes.items = [None, None, None, None];
            len = 0;
        }
    }
    if len != 0 {
        kernel(&mut lanes);
    }
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
    fn four_lane_groups_visit_every_item_once_in_order() {
        for count in 0..11 {
            let mut values: Vec<f32> = (0..count).map(|index| index as f32).collect();
            let mut groups = Vec::new();
            for_each4(values.iter_mut(), |lanes| {
                let loaded = lanes.load(|value| **value);
                groups.push(lanes.len());
                lanes.for_each(|lane, value| {
                    assert_eq!(**value, loaded[lane]);
                    **value = loaded[lane] * 2.0 + 1.0;
                });
                // Absent lanes read zero.
                for lane in lanes.len()..4 {
                    assert_eq!(loaded[lane], 0.0);
                }
            });
            let expected: Vec<f32> = (0..count).map(|index| index as f32 * 2.0 + 1.0).collect();
            assert_eq!(values, expected);
            assert_eq!(groups.iter().sum::<usize>(), count);
            assert!(groups.iter().rev().skip(1).all(|&len| len == 4));
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
