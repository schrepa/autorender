//! Seeded gradient noise with derivatives and the erosion filter's gullies:
//! integer hashing and plain `f32` arithmetic with the engine's branch-free
//! trigonometry, so every target computes the same bits.
//!
//! The functions ending in 4 evaluate four points at once:
//! Wasm SIMD lanes where the target has them, each lane the scalar
//! function's IEEE operations in its order (no fused multiply-add, no
//! relaxed SIMD), so they give the same bits as it at each point.

use engine_math::{clamp, max, sin_cos};
use glam::Vec4;

use lanes::{Bits4, floor4};
pub(crate) use lanes::{max4, min4, sqrt4};

/// Unit gradients 22.5 degrees apart, turned half a step off the axes.
const GRADIENTS: [[f32; 2]; 16] = [
    [0.980_785_3, 0.195_090_32],
    [0.831_469_6, 0.555_570_2],
    [0.555_570_2, 0.831_469_6],
    [0.195_090_32, 0.980_785_3],
    [-0.195_090_32, 0.980_785_3],
    [-0.555_570_2, 0.831_469_6],
    [-0.831_469_6, 0.555_570_2],
    [-0.980_785_3, 0.195_090_32],
    [-0.980_785_3, -0.195_090_32],
    [-0.831_469_6, -0.555_570_2],
    [-0.555_570_2, -0.831_469_6],
    [-0.195_090_32, -0.980_785_3],
    [0.195_090_32, -0.980_785_3],
    [0.555_570_2, -0.831_469_6],
    [0.831_469_6, -0.555_570_2],
    [0.980_785_3, -0.195_090_32],
];

/// Two-dimensional gradient noise peaks near +-1/sqrt(2); this brings it to
/// about +-1.
const NOISE_SCALE: f32 = std::f32::consts::SQRT_2;

/// The magnitudes of the gradients' components: each component is one of
/// them, its index and sign bit operations on the gradient's index, which
/// lanes compute instead of loading from [`GRADIENTS`].
const MAGNITUDES: [f32; 4] = [
    GRADIENTS[0][0],
    GRADIENTS[1][0],
    GRADIENTS[2][0],
    GRADIENTS[3][0],
];

/// Gradient `i`'s x component: its magnitude's index and its sign.
const fn magnitude_x(i: u32) -> u32 {
    (i & 3) ^ (((i >> 2) & 1) * 3)
}
const fn sign_x(i: u32) -> u32 {
    ((i >> 2) ^ (i >> 3)) & 1
}

/// Gradient `i`'s z component: its magnitude's index and its sign.
const fn magnitude_z(i: u32) -> u32 {
    (i & 3) ^ 3 ^ (((i >> 2) & 1) * 3)
}
const fn sign_z(i: u32) -> u32 {
    (i >> 3) & 1
}

// The rules give every gradient bit for bit.
const _: () = {
    let mut i = 0;
    while i < 16 {
        let x = MAGNITUDES[magnitude_x(i) as usize].to_bits() ^ (sign_x(i) << 31);
        let z = MAGNITUDES[magnitude_z(i) as usize].to_bits() ^ (sign_z(i) << 31);
        assert!(GRADIENTS[i as usize][0].to_bits() == x);
        assert!(GRADIENTS[i as usize][1].to_bits() == z);
        i += 1;
    }
};

/// Chris Wellons' `lowbias32` finalizer: every output bit depends on every
/// input bit.
#[inline(always)]
fn mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

/// A hash of lattice point (x, z) under `seed`.
#[inline(always)]
pub(crate) fn hash(x: i32, z: i32, seed: u32) -> u32 {
    mix(mix(seed ^ x as u32).wrapping_add(z as u32))
}

#[inline(always)]
fn gradient_at(x: i32, z: i32, seed: u32) -> [f32; 2] {
    GRADIENTS[(hash(x, z, seed) & 15) as usize]
}

/// A lattice cell: its first corner and the point's offset in it.
#[inline(always)]
fn cell(x: f32, z: f32) -> (i32, i32, f32, f32) {
    let x0 = x.floor();
    let z0 = z.floor();
    (x0 as i32, z0 as i32, x - x0, z - z0)
}

/// Quintic fade (Perlin's improved noise) and its derivative.
#[inline(always)]
fn fade(t: f32) -> (f32, f32) {
    (
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
        30.0 * t * t * (t * (t - 2.0) + 1.0),
    )
}

/// Gradient noise at lattice coordinates (x, z), about -1 to 1, and its
/// derivatives along x and z.
#[inline]
pub(crate) fn gradient_noise(x: f32, z: f32, seed: u32) -> [f32; 3] {
    let (i, j, fx, fz) = cell(x, z);
    let (ux, dux) = fade(fx);
    let (uz, duz) = fade(fz);
    let ga = gradient_at(i, j, seed);
    let gb = gradient_at(i.wrapping_add(1), j, seed);
    let gc = gradient_at(i, j.wrapping_add(1), seed);
    let gd = gradient_at(i.wrapping_add(1), j.wrapping_add(1), seed);
    let va = ga[0] * fx + ga[1] * fz;
    let vb = gb[0] * (fx - 1.0) + gb[1] * fz;
    let vc = gc[0] * fx + gc[1] * (fz - 1.0);
    let vd = gd[0] * (fx - 1.0) + gd[1] * (fz - 1.0);
    let k1 = vb - va;
    let k2 = vc - va;
    let k3 = va - vb - vc + vd;
    let value = va + ux * k1 + uz * k2 + ux * uz * k3;
    let dx = ga[0]
        + ux * (gb[0] - ga[0])
        + uz * (gc[0] - ga[0])
        + ux * uz * (ga[0] - gb[0] - gc[0] + gd[0])
        + dux * (k1 + uz * k3);
    let dz = ga[1]
        + ux * (gb[1] - ga[1])
        + uz * (gc[1] - ga[1])
        + ux * uz * (ga[1] - gb[1] - gc[1] + gd[1])
        + duz * (k2 + ux * k3);
    [value * NOISE_SCALE, dx * NOISE_SCALE, dz * NOISE_SCALE]
}

/// [`gradient_noise`]'s value alone.
#[inline]
pub(crate) fn gradient_value(x: f32, z: f32, seed: u32) -> f32 {
    let (i, j, fx, fz) = cell(x, z);
    value_in(corners(i, j, seed), fx, fz)
}

/// The gradients at lattice cell `(i, j)`'s four corners under `seed`.
#[inline(always)]
fn corners(i: i32, j: i32, seed: u32) -> [[f32; 2]; 4] {
    [
        gradient_at(i, j, seed),
        gradient_at(i.wrapping_add(1), j, seed),
        gradient_at(i, j.wrapping_add(1), seed),
        gradient_at(i.wrapping_add(1), j.wrapping_add(1), seed),
    ]
}

/// [`gradient_value`] at offset `(fx, fz)` in a cell with gradients `corners`.
#[inline(always)]
fn value_in([ga, gb, gc, gd]: [[f32; 2]; 4], fx: f32, fz: f32) -> f32 {
    let (ux, _) = fade(fx);
    let (uz, _) = fade(fz);
    let va = ga[0] * fx + ga[1] * fz;
    let vb = gb[0] * (fx - 1.0) + gb[1] * fz;
    let vc = gc[0] * fx + gc[1] * (fz - 1.0);
    let vd = gd[0] * (fx - 1.0) + gd[1] * (fz - 1.0);
    let top = va + ux * (vb - va);
    let bottom = vc + ux * (vd - vc);
    (top + uz * (bottom - top)) * NOISE_SCALE
}

/// [`gradient_value`] at each point of `xs` and `zs`, into
/// `out`: points in the cell of the one before reuse its corners'
/// gradients, the same values hashed again.
pub(crate) fn gradient_values(xs: &[f32], zs: &[f32], seed: u32, out: &mut [f32]) {
    let mut held: Option<(i32, i32, [[f32; 2]; 4])> = None;
    for ((value, &x), &z) in out.iter_mut().zip(xs).zip(zs) {
        let (i, j, fx, fz) = cell(x, z);
        let gradients = match held {
            Some((hi, hj, gradients)) if hi == i && hj == j => gradients,
            _ => {
                let gradients = corners(i, j, seed);
                held = Some((i, j, gradients));
                gradients
            }
        };
        *value = value_in(gradients, fx, fz);
    }
}

/// [`gradient_noise`] at four points, lane by lane its
/// operations in its order: the same bits.
#[inline]
pub(crate) fn gradient_noise4(x: Vec4, z: Vec4, seed: u32) -> [Vec4; 3] {
    let x0 = floor4(x);
    let z0 = floor4(z);
    let (i, j) = (Bits4::truncate(x0), Bits4::truncate(z0));
    let (fx, fz) = (x - x0, z - z0);
    let (ux, dux) = fade4(fx);
    let (uz, duz) = fade4(fz);
    let one = Bits4::splat(1);
    let (i1, j1) = (i.wrapping_add(one), j.wrapping_add(one));
    let ga = gradient4(i, j, seed);
    let gb = gradient4(i1, j, seed);
    let gc = gradient4(i, j1, seed);
    let gd = gradient4(i1, j1, seed);
    let va = ga[0] * fx + ga[1] * fz;
    let vb = gb[0] * (fx - 1.0) + gb[1] * fz;
    let vc = gc[0] * fx + gc[1] * (fz - 1.0);
    let vd = gd[0] * (fx - 1.0) + gd[1] * (fz - 1.0);
    let k1 = vb - va;
    let k2 = vc - va;
    let k3 = va - vb - vc + vd;
    let value = va + ux * k1 + uz * k2 + ux * uz * k3;
    let dx = ga[0]
        + ux * (gb[0] - ga[0])
        + uz * (gc[0] - ga[0])
        + ux * uz * (ga[0] - gb[0] - gc[0] + gd[0])
        + dux * (k1 + uz * k3);
    let dz = ga[1]
        + ux * (gb[1] - ga[1])
        + uz * (gc[1] - ga[1])
        + ux * uz * (ga[1] - gb[1] - gc[1] + gd[1])
        + duz * (k2 + ux * k3);
    [value * NOISE_SCALE, dx * NOISE_SCALE, dz * NOISE_SCALE]
}

/// [`gradient_value`] at four points.
#[inline]
pub(crate) fn gradient_value4(x: Vec4, z: Vec4, seed: u32) -> Vec4 {
    let x0 = floor4(x);
    let z0 = floor4(z);
    let (i, j) = (Bits4::truncate(x0), Bits4::truncate(z0));
    let (fx, fz) = (x - x0, z - z0);
    let one = Bits4::splat(1);
    let (i1, j1) = (i.wrapping_add(one), j.wrapping_add(one));
    let ga = gradient4(i, j, seed);
    let gb = gradient4(i1, j, seed);
    let gc = gradient4(i, j1, seed);
    let gd = gradient4(i1, j1, seed);
    let (ux, _) = fade4(fx);
    let (uz, _) = fade4(fz);
    let va = ga[0] * fx + ga[1] * fz;
    let vb = gb[0] * (fx - 1.0) + gb[1] * fz;
    let vc = gc[0] * fx + gc[1] * (fz - 1.0);
    let vd = gd[0] * (fx - 1.0) + gd[1] * (fz - 1.0);
    let top = va + ux * (vb - va);
    let bottom = vc + ux * (vd - vc);
    (top + uz * (bottom - top)) * NOISE_SCALE
}

/// [`gullies`]' profile (its first value) at four points.
#[inline]
pub(crate) fn gullies4(x: Vec4, z: Vec4, across: [Vec4; 2], seed: u32) -> Vec4 {
    let x0 = floor4(x);
    let z0 = floor4(z);
    let (i, j) = (Bits4::truncate(x0), Bits4::truncate(z0));
    let (fx, fz) = (x - x0, z - z0);
    let wx = fx * fx * (3.0 - 2.0 * fx);
    let wz = fz * fz * (3.0 - 2.0 * fz);
    let mut c = Vec4::ZERO;
    let mut s = Vec4::ZERO;
    for (di, dj, weight) in [
        (0, 0, (1.0 - wx) * (1.0 - wz)),
        (1, 0, wx * (1.0 - wz)),
        (0, 1, (1.0 - wx) * wz),
        (1, 1, wx * wz),
    ] {
        let h = mix4(
            mix4(Bits4::splat(seed).xor(i.wrapping_add(Bits4::splat(di))))
                .wrapping_add(j.wrapping_add(Bits4::splat(dj))),
        );
        let cx = di as f32 + 0.1 + 0.8 * h.and(0xffff).to_f32() * (1.0 / 65_535.0);
        let cz = dj as f32 + 0.1 + 0.8 * h.shr(16).to_f32() * (1.0 / 65_535.0);
        let phase = ((fx - cx) * across[0] + (fz - cz) * across[1]) * std::f32::consts::TAU;
        let (sin, cos) = sin_cos4(phase);
        c += weight * cos;
        s += weight * sin;
    }
    let length = sqrt4(c * c + s * s);
    let scale = smooth4(length * 2.0) / max4(length, Vec4::splat(1.0e-6));
    c * scale
}

/// [`sin_cos`] of four angles: its operations lane by lane, angles beyond
/// its fast reduction (none in practice) one by one through it.
#[inline(always)]
fn sin_cos4(x: Vec4) -> (Vec4, Vec4) {
    use engine_math::lanes::{C1, C2, C3, FRAC_2_PI, PIO2_A, PIO2_B, PIO2_C, S1, S2, S3, SHIFTER};
    if !x
        .abs()
        .cmple(Vec4::splat(engine_math::REDUCTION_LIMIT))
        .all()
    {
        let lanes = x.to_array().map(sin_cos);
        return (
            Vec4::from_array(lanes.map(|(sin, _)| sin)),
            Vec4::from_array(lanes.map(|(_, cos)| cos)),
        );
    }
    let shifted = x * FRAC_2_PI + SHIFTER;
    let j = shifted - SHIFTER;
    let quadrant = Bits4::bits(shifted);
    let y = ((x - j * PIO2_A) - j * PIO2_B) - j * PIO2_C;
    let z = y * y;
    let s = Bits4::bits(y + y * z * (S1 + z * (S2 + z * S3)))
        .and(0x7fff_ffff)
        .or(Bits4::bits(y).and(0x8000_0000));
    let c = Bits4::bits(1.0 - 0.5 * z + z * z * (C1 + z * (C2 + z * C3)));
    let swap = quadrant.and(1).wrapping_mul(u32::MAX);
    let swapped = s.xor(c).mask(swap);
    (
        s.xor(swapped).xor(quadrant.and(2).shl(30)).float(),
        c.xor(swapped)
            .xor(quadrant.wrapping_add(Bits4::splat(1)).and(2).shl(30))
            .float(),
    )
}

/// [`smooth`] of four values.
#[inline(always)]
pub(crate) fn smooth4(t: Vec4) -> Vec4 {
    let t = min4(max4(t, Vec4::ZERO), Vec4::ONE);
    t * t * (3.0 - 2.0 * t)
}

/// [`fade`] of four values.
#[inline(always)]
fn fade4(t: Vec4) -> (Vec4, Vec4) {
    (
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0),
        30.0 * t * t * (t * (t - 2.0) + 1.0),
    )
}

/// [`mix`] of four values.
#[inline(always)]
fn mix4(h: Bits4) -> Bits4 {
    let h = h.xor(h.shr(16));
    let h = h.wrapping_mul(0x7feb_352d);
    let h = h.xor(h.shr(15));
    let h = h.wrapping_mul(0x846c_a68b);
    h.xor(h.shr(16))
}

/// [`gradient_at`] of four lattice points, their x and z components.
#[inline(always)]
fn gradient4(x: Bits4, z: Bits4, seed: u32) -> [Vec4; 2] {
    let i = mix4(mix4(Bits4::splat(seed).xor(x)).wrapping_add(z)).and(15);
    let low = i.and(3);
    let flip = i.shr(2).and(1).wrapping_mul(3);
    let high = i.shr(3);
    let sign_x = i.shr(2).xor(high).and(1).shl(31);
    [
        Bits4::bits(low.xor(flip).pick(MAGNITUDES))
            .xor(sign_x)
            .float(),
        Bits4::bits(low.xor(flip).xor(Bits4::splat(3)).pick(MAGNITUDES))
            .xor(high.shl(31))
            .float(),
    ]
}

/// Four lanes: Wasm SIMD where the target has it, arrays elsewhere; each
/// operation is the scalar one's, lane by lane.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod lanes {
    use core::arch::wasm32::{
        f32x4_convert_u32x4, f32x4_floor, f32x4_pmax, f32x4_pmin, f32x4_sqrt, i8x16_swizzle,
        i32x4_add, i32x4_mul, i32x4_shl, i32x4_trunc_sat_f32x4, u32x4_shr, u32x4_splat, v128,
        v128_and, v128_or, v128_xor,
    };
    use glam::Vec4;

    /// Four `u32` lanes.
    #[derive(Clone, Copy)]
    pub(super) struct Bits4(v128);

    impl Bits4 {
        #[inline(always)]
        pub(super) fn splat(value: u32) -> Self {
            Self(u32x4_splat(value))
        }

        /// Each lane `as i32` converts, its bits.
        #[inline(always)]
        pub(super) fn truncate(value: Vec4) -> Self {
            Self(i32x4_trunc_sat_f32x4(value.into()))
        }

        #[inline(always)]
        pub(super) fn bits(value: Vec4) -> Self {
            Self(value.into())
        }

        #[inline(always)]
        pub(super) fn float(self) -> Vec4 {
            self.0.into()
        }

        #[inline(always)]
        pub(super) fn xor(self, other: Self) -> Self {
            Self(v128_xor(self.0, other.0))
        }

        #[inline(always)]
        pub(super) fn and(self, mask: u32) -> Self {
            Self(v128_and(self.0, u32x4_splat(mask)))
        }

        /// Lane by lane `&`.
        #[inline(always)]
        pub(super) fn mask(self, other: Self) -> Self {
            Self(v128_and(self.0, other.0))
        }

        #[inline(always)]
        pub(super) fn or(self, other: Self) -> Self {
            Self(v128_or(self.0, other.0))
        }

        /// Each lane `as f32` converts.
        #[inline(always)]
        pub(super) fn to_f32(self) -> Vec4 {
            f32x4_convert_u32x4(self.0).into()
        }

        #[inline(always)]
        pub(super) fn shr(self, by: u32) -> Self {
            Self(u32x4_shr(self.0, by))
        }

        #[inline(always)]
        pub(super) fn shl(self, by: u32) -> Self {
            Self(i32x4_shl(self.0, by))
        }

        #[inline(always)]
        pub(super) fn wrapping_add(self, other: Self) -> Self {
            Self(i32x4_add(self.0, other.0))
        }

        #[inline(always)]
        pub(super) fn wrapping_mul(self, by: u32) -> Self {
            Self(i32x4_mul(self.0, u32x4_splat(by)))
        }

        /// Each lane's entry of `table`, the lanes being 0 to 3: one
        /// byte shuffle.
        #[inline(always)]
        pub(super) fn pick(self, table: [f32; 4]) -> Vec4 {
            let bytes = i32x4_add(
                i32x4_mul(self.0, u32x4_splat(0x0404_0404)),
                u32x4_splat(0x0302_0100),
            );
            Vec4::from(i8x16_swizzle(Vec4::from_array(table).into(), bytes))
        }
    }

    #[inline(always)]
    pub(super) fn floor4(value: Vec4) -> Vec4 {
        f32x4_floor(value.into()).into()
    }

    /// Square roots, correctly rounded as `f32::sqrt`'s.
    #[inline(always)]
    pub(crate) fn sqrt4(value: Vec4) -> Vec4 {
        f32x4_sqrt(value.into()).into()
    }

    /// [`engine_math::min`] lane by lane.
    #[inline(always)]
    pub(crate) fn min4(a: Vec4, b: Vec4) -> Vec4 {
        f32x4_pmin(a.into(), b.into()).into()
    }

    /// [`engine_math::max`] lane by lane.
    #[inline(always)]
    pub(crate) fn max4(a: Vec4, b: Vec4) -> Vec4 {
        f32x4_pmax(a.into(), b.into()).into()
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod lanes {
    use glam::Vec4;

    /// Four `u32` lanes.
    #[derive(Clone, Copy)]
    pub(super) struct Bits4([u32; 4]);

    impl Bits4 {
        #[inline(always)]
        pub(super) fn splat(value: u32) -> Self {
            Self([value; 4])
        }

        /// Each lane `as i32` converts, its bits.
        #[inline(always)]
        pub(super) fn truncate(value: Vec4) -> Self {
            Self(value.to_array().map(|value| value as i32 as u32))
        }

        #[inline(always)]
        pub(super) fn bits(value: Vec4) -> Self {
            Self(value.to_array().map(f32::to_bits))
        }

        #[inline(always)]
        pub(super) fn float(self) -> Vec4 {
            Vec4::from_array(self.0.map(f32::from_bits))
        }

        #[inline(always)]
        pub(super) fn xor(self, other: Self) -> Self {
            Self(std::array::from_fn(|lane| self.0[lane] ^ other.0[lane]))
        }

        #[inline(always)]
        pub(super) fn and(self, mask: u32) -> Self {
            Self(self.0.map(|value| value & mask))
        }

        /// Lane by lane `&`.
        #[inline(always)]
        pub(super) fn mask(self, other: Self) -> Self {
            Self(std::array::from_fn(|lane| self.0[lane] & other.0[lane]))
        }

        #[inline(always)]
        pub(super) fn or(self, other: Self) -> Self {
            Self(std::array::from_fn(|lane| self.0[lane] | other.0[lane]))
        }

        /// Each lane `as f32` converts.
        #[inline(always)]
        pub(super) fn to_f32(self) -> Vec4 {
            Vec4::from_array(self.0.map(|value| value as f32))
        }

        #[inline(always)]
        pub(super) fn shr(self, by: u32) -> Self {
            Self(self.0.map(|value| value >> by))
        }

        #[inline(always)]
        pub(super) fn shl(self, by: u32) -> Self {
            Self(self.0.map(|value| value << by))
        }

        #[inline(always)]
        pub(super) fn wrapping_add(self, other: Self) -> Self {
            Self(std::array::from_fn(|lane| {
                self.0[lane].wrapping_add(other.0[lane])
            }))
        }

        #[inline(always)]
        pub(super) fn wrapping_mul(self, by: u32) -> Self {
            Self(self.0.map(|value| value.wrapping_mul(by)))
        }

        /// Each lane's entry of `table`, the lanes being 0 to 3.
        #[inline(always)]
        pub(super) fn pick(self, table: [f32; 4]) -> Vec4 {
            Vec4::from_array(self.0.map(|index| table[index as usize & 3]))
        }
    }

    #[inline(always)]
    pub(super) fn floor4(value: Vec4) -> Vec4 {
        Vec4::from_array(value.to_array().map(f32::floor))
    }

    /// Square roots, correctly rounded as `f32::sqrt`'s.
    #[inline(always)]
    pub(crate) fn sqrt4(value: Vec4) -> Vec4 {
        Vec4::from_array(value.to_array().map(f32::sqrt))
    }

    /// [`engine_math::min`] lane by lane.
    #[inline(always)]
    pub(crate) fn min4(a: Vec4, b: Vec4) -> Vec4 {
        Vec4::from_array(std::array::from_fn(|lane| {
            engine_math::min(a[lane], b[lane])
        }))
    }

    /// [`engine_math::max`] lane by lane.
    #[inline(always)]
    pub(crate) fn max4(a: Vec4, b: Vec4) -> Vec4 {
        Vec4::from_array(std::array::from_fn(|lane| {
            engine_math::max(a[lane], b[lane])
        }))
    }
}

/// Hermite smoothstep of `t` clamped to [0, 1].
#[inline(always)]
pub(crate) fn smooth(t: f32) -> f32 {
    let t = clamp(t, 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One octave of the erosion filter's gullies at (x, z) in cell units:
/// stripes across `across` (a unit vector, the slope's sideways direction),
/// one per cell, whose phase drifts from cell to cell. The four nearest
/// cells' stripes, each about a jittered center, blend as unit phasors
/// weighted by distance and renormalized, so the stripes keep their depth
/// while they bend and branch (after Rune Skovbo Johansen's erosion
/// filter). Returns the blend's cosine and sine, faded out where the
/// phasors cancel: the gully profile and its slope across.
#[inline]
pub(crate) fn gullies(x: f32, z: f32, across: [f32; 2], seed: u32) -> [f32; 2] {
    let (i, j, fx, fz) = cell(x, z);
    let wx = fx * fx * (3.0 - 2.0 * fx);
    let wz = fz * fz * (3.0 - 2.0 * fz);
    let mut c = 0.0;
    let mut s = 0.0;
    for (di, dj, weight) in [
        (0, 0, (1.0 - wx) * (1.0 - wz)),
        (1, 0, wx * (1.0 - wz)),
        (0, 1, (1.0 - wx) * wz),
        (1, 1, wx * wz),
    ] {
        let h = hash(i.wrapping_add(di), j.wrapping_add(dj), seed);
        // A center within the cell's middle four fifths.
        let cx = di as f32 + 0.1 + 0.8 * (h & 0xffff) as f32 * (1.0 / 65_535.0);
        let cz = dj as f32 + 0.1 + 0.8 * (h >> 16) as f32 * (1.0 / 65_535.0);
        let phase = ((fx - cx) * across[0] + (fz - cz) * across[1]) * std::f32::consts::TAU;
        let (sin, cos) = sin_cos(phase);
        c += weight * cos;
        s += weight * sin;
    }
    let length = (c * c + s * s).sqrt();
    let scale = smooth(length * 2.0) / max(length, 1.0e-6);
    [c * scale, s * scale]
}
