extern crate std;

use super::lanes::*;
use super::*;
use std::vec::Vec;

/// Distance in units in the last place between two finite `f32` values.
fn ulps(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let bits = v.to_bits() as i64;
        if bits < 0x8000_0000 {
            bits
        } else {
            0x8000_0000 - bits
        }
    };
    key(a).abs_diff(key(b)) as u32
}

/// Reference sine and cosine in `f64`.
fn reference(x: f32) -> (f64, f64) {
    let x = x as f64;
    (x.sin(), x.cos())
}

/// Deterministic uniform samples in `[-limit, limit]`.
fn samples(limit: f32, count: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            ((unit * 2.0 - 1.0) * limit as f64) as f32
        })
        .collect()
}

/// Largest ulp distance and absolute error of [`sin_cos`] over `xs`.
fn errors(xs: &[f32]) -> (u32, f64) {
    xs.iter().fold((0, 0.0), |(max_ulps, max_abs), &x| {
        let (s, c) = sin_cos(x);
        let (rs, rc) = reference(x);
        let u = ulps(s, rs as f32).max(ulps(c, rc as f32));
        let a = (s as f64 - rs).abs().max((c as f64 - rc).abs());
        (max_ulps.max(u), max_abs.max(a))
    })
}

#[test]
fn reduction_constants_keep_their_exact_bits() {
    // The first two parts need trailing zero bits for exact products.
    assert_eq!(PIO2_A.to_bits(), 0x3FC9_0000);
    assert_eq!(PIO2_B.to_bits(), 0x39FD_A000);
    assert_eq!(PIO2_C.to_bits(), 0x33A2_2169);
    assert_eq!(SHIFTER.to_bits(), 0x4B40_0000);
    let coefficients = [S1, S2, S3, C1, C2, C3].map(f32::to_bits);
    assert_eq!(
        coefficients,
        [
            0xBE2A_AAA3,
            0x3C08_839E,
            0xB94C_A1F9,
            0x3D2A_AAA5,
            0xBAB6_061A,
            0x37CC_F5CE
        ]
    );
    let sum = PIO2_A as f64 + PIO2_B as f64 + PIO2_C as f64;
    // Within half an ulp of the third part.
    assert!((sum - core::f64::consts::FRAC_PI_2).abs() < 4e-15);
}

#[test]
fn quadrant_products_below_the_limit_are_exact() {
    let largest = (REDUCTION_LIMIT * FRAC_2_PI).ceil() as i32;
    assert!(largest < 1 << 13);
    for j in -largest..=largest {
        let j = j as f32;
        assert_eq!((j * PIO2_A) as f64, j as f64 * PIO2_A as f64);
        assert_eq!((j * PIO2_B) as f64, j as f64 * PIO2_B as f64);
    }
}

#[test]
fn accuracy_bounds_hold_on_sampled_ranges() {
    let quarter = errors(&samples(core::f32::consts::FRAC_PI_4, 2_000_000));
    assert!(quarter.0 <= 1, "{quarter:?}");
    let half_turn = errors(&samples(core::f32::consts::PI, 2_000_000));
    assert!(half_turn.0 <= 2, "{half_turn:?}");
    let wide = errors(&samples(REDUCTION_LIMIT, 4_000_000));
    assert!(wide.1 < 1.2e-7, "{wide:?}");
}

#[test]
fn sine_is_odd_and_cosine_even_bit_for_bit() {
    for x in samples(REDUCTION_LIMIT, 1_000_000) {
        let (s, c) = sin_cos(x);
        let (ns, nc) = sin_cos(-x);
        assert_eq!(ns.to_bits(), (-s).to_bits(), "{x}");
        assert_eq!(nc.to_bits(), c.to_bits(), "{x}");
    }
}

#[test]
fn special_values_and_the_fallback_match_musl() {
    assert_eq!(sin_cos(0.0), (0.0, 1.0));
    let (s, c) = sin_cos(-0.0);
    assert_eq!((s.to_bits(), c), ((-0.0f32).to_bits(), 1.0));
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let (s, c) = sin_cos(x);
        assert!(s.is_nan() && c.is_nan(), "{x}");
    }
    for x in [
        f32::from_bits(REDUCTION_LIMIT.to_bits() + 1),
        -20_000.0,
        1.0e6,
        3.0e30,
        f32::MAX,
    ] {
        assert_eq!(sin_cos(x), libm::sincosf(x), "{x}");
    }
    let tiny = f32::from_bits(1);
    assert_eq!(sin_cos(tiny), (tiny, 1.0));
    assert_eq!((sin(0.5), cos(0.5)), sin_cos(0.5));
}

/// Every `f32` in the fast range. Run with
/// `cargo test -p engine_math --release -- --ignored`.
#[test]
#[ignore = "exhaustive: every f32 below the reduction limit"]
fn exhaustive_accuracy_below_the_limit() {
    let limit = REDUCTION_LIMIT.to_bits();
    let pi = core::f32::consts::PI.to_bits();
    let (mut max_abs, mut max_ulps_pi) = (0.0f64, 0);
    for bits in (0..=limit).chain(0x8000_0000..=0x8000_0000 | limit) {
        let x = f32::from_bits(bits);
        let (s, c) = sin_cos(x);
        let (rs, rc) = reference(x);
        max_abs = max_abs
            .max((s as f64 - rs).abs())
            .max((c as f64 - rc).abs());
        if bits & 0x7fff_ffff <= pi {
            max_ulps_pi = max_ulps_pi.max(ulps(s, rs as f32)).max(ulps(c, rc as f32));
        }
    }
    assert!(max_abs < 1.2e-7, "{max_abs}");
    assert!(max_ulps_pi <= 2, "{max_ulps_pi}");
}

#[test]
// Compares with the standard functions the engine disallows elsewhere.
#[allow(clippy::disallowed_methods)]
fn min_max_and_clamp_follow_the_wasm_lane_rules() {
    let bits = |value: f32| value.to_bits();
    // `b < a ? b : a` and `a < b ? b : a`, including NaN and signed zeros.
    for (a, b) in [
        (1.0_f32, 2.0_f32),
        (2.0, 1.0),
        (-3.5, 3.5),
        (0.0, -0.0),
        (-0.0, 0.0),
        (f32::NAN, 1.0),
        (1.0, f32::NAN),
        (f32::INFINITY, f32::NEG_INFINITY),
        (f32::from_bits(1), 0.0),
    ] {
        let expected_min = if b < a { b } else { a };
        let expected_max = if a < b { b } else { a };
        assert_eq!(bits(min(a, b)), bits(expected_min), "min({a}, {b})");
        assert_eq!(bits(max(a, b)), bits(expected_max), "max({a}, {b})");
    }
    assert_eq!(bits(min(0.0, -0.0)), bits(0.0));
    assert_eq!(bits(min(-0.0, 0.0)), bits(-0.0));
    assert!(min(f32::NAN, 1.0).is_nan());
    assert_eq!(min(1.0, f32::NAN), 1.0);
    assert_eq!(clamp(5.0, 0.0, 1.0), 1.0);
    assert_eq!(clamp(-5.0, 0.0, 1.0), 0.0);
    assert_eq!(clamp(0.25, 0.0, 1.0), 0.25);
    assert!(clamp(f32::NAN, 0.0, 1.0).is_nan());
    // No panic on an inverted range: the upper bound wins.
    assert_eq!(clamp(0.5, 1.0, 0.0), 0.0);
    // Agrees with the standard functions on ordinary values.
    for x in samples(1_000.0, 10_000) {
        assert_eq!(min(x, 1.0), x.min(1.0));
        assert_eq!(max(x, -1.0), x.max(-1.0));
        assert_eq!(clamp(x, -2.0, 3.0), x.clamp(-2.0, 3.0));
    }
}
