#![no_std]
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
pub mod collections;

pub const REDUCTION_LIMIT: f32 = 12_866.0;

const SIGN: u32 = 0x8000_0000;
const ONE: u32 = 0x3f80_0000;
const INFINITY: u32 = 0x7f80_0000;
const TAN_PI_8: f32 = 0.414_213_57;
const A1: f32 = -3.333_295e-1;
const A2: f32 = 1.997_771_1e-1;
const A3: f32 = -1.387_768_5e-1;
const A4: f32 = 8.053_744_6e-2;

#[doc(hidden)]
pub mod lanes {
    pub const SHIFTER: f32 = 12_582_912.0;
    pub const FRAC_2_PI: f32 = core::f32::consts::FRAC_2_PI;
    pub const PIO2_A: f32 = 1.570_312_5;
    pub const PIO2_B: f32 = 4.837_513e-4;
    pub const PIO2_C: f32 = 7.549_79e-8;
    pub const S1: f32 = -1.666_665_5e-1;
    pub const S2: f32 = 8.332_161e-3;
    pub const S3: f32 = -1.951_529_6e-4;
    pub const C1: f32 = 4.166_664_6e-2;
    pub const C2: f32 = -1.388_731_6e-3;
    pub const C3: f32 = 2.443_315_7e-5;
}

#[inline]
pub fn sin_cos(x: f32) -> (f32, f32) {
    use lanes::*;
    let reducible = x.abs() <= REDUCTION_LIMIT;
    if !reducible {
        return large(x);
    }
    let shifted = x * FRAC_2_PI + SHIFTER;
    let j = shifted - SHIFTER;
    let quadrant = shifted.to_bits();
    let y = ((x - j * PIO2_A) - j * PIO2_B) - j * PIO2_C;
    let z = y * y;
    let s = ((y + y * z * (S1 + z * (S2 + z * S3))).to_bits() & 0x7fff_ffff)
        | (y.to_bits() & 0x8000_0000);
    let c = (1.0 - 0.5 * z + z * z * (C1 + z * (C2 + z * C3))).to_bits();
    let swap = (quadrant & 1).wrapping_neg();
    let sin = s ^ ((s ^ c) & swap);
    let cos = c ^ ((s ^ c) & swap);
    (
        f32::from_bits(sin ^ ((quadrant & 2) << 30)),
        f32::from_bits(cos ^ ((quadrant.wrapping_add(1) & 2) << 30)),
    )
}

#[inline]
pub fn sin(x: f32) -> f32 {
    sin_cos(x).0
}

#[inline]
pub fn cos(x: f32) -> f32 {
    sin_cos(x).1
}

#[inline]
fn all_if(on: bool) -> u32 {
    u32::from(on).wrapping_neg()
}

#[inline]
fn pick(mask: u32, on: u32, off: u32) -> u32 {
    off ^ ((off ^ on) & mask)
}

#[inline]
pub fn atan2(y: f32, x: f32) -> f32 {
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let (y_bits, x_bits) = (y.to_bits(), x.to_bits());
    let (along, across) = (y_bits & !SIGN, x_bits & !SIGN);
    let steep = all_if(along > across);
    let wide = pick(steep, along, across);
    let thin = pick(steep, across, along);
    let endless = all_if(thin == INFINITY) & all_if(wide == INFINITY);
    let empty = all_if(wide == 0);
    let ratio =
        f32::from_bits(pick(endless, ONE, thin)) / f32::from_bits(pick(endless | empty, ONE, wide));
    let far = all_if(ratio > TAN_PI_8);
    let t = f32::from_bits(pick(
        far,
        ((ratio - 1.0) / (ratio + 1.0)).to_bits(),
        ratio.to_bits(),
    ));
    let z = t * t;
    let near = f32::from_bits(FRAC_PI_4.to_bits() & far)
        + (t + t * z * (A1 + z * (A2 + z * (A3 + z * A4))));
    let right = f32::from_bits(FRAC_PI_2.to_bits() & steep)
        + f32::from_bits(near.to_bits() ^ (steep & SIGN));
    let back = (x_bits >> 31).wrapping_neg();
    let half =
        f32::from_bits(PI.to_bits() & back) + f32::from_bits(right.to_bits() ^ (back & SIGN));
    f32::from_bits(half.to_bits() ^ (y_bits & SIGN))
}

#[inline]
pub fn min(a: f32, b: f32) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::{f32x4_extract_lane, f32x4_pmin, f32x4_splat};
        f32x4_extract_lane::<0>(f32x4_pmin(f32x4_splat(a), f32x4_splat(b)))
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        if b < a { b } else { a }
    }
}

#[inline]
pub fn max(a: f32, b: f32) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::{f32x4_extract_lane, f32x4_pmax, f32x4_splat};
        f32x4_extract_lane::<0>(f32x4_pmax(f32x4_splat(a), f32x4_splat(b)))
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        if a < b { b } else { a }
    }
}

#[inline]
pub fn clamp(x: f32, low: f32, high: f32) -> f32 {
    min(max(x, low), high)
}

#[inline]
pub fn exp(x: f32) -> f32 {
    libm::expf(x)
}

#[inline]
pub fn ln(x: f32) -> f32 {
    libm::logf(x)
}

#[inline]
pub fn pow(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

#[inline]
pub fn atan(x: f32) -> f32 {
    libm::atanf(x)
}

#[cold]
#[inline(never)]
fn large(x: f32) -> (f32, f32) {
    libm::sincosf(x)
}
