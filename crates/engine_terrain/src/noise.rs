//! Seeded gradient noise with derivatives and the erosion filter's gullies:
//! integer hashing and plain `f32` arithmetic with the engine's branch-free
//! trigonometry, so every target computes the same bits.

use engine_math::{clamp, max, sin_cos};

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

/// [`gradient_value`] at each point of `xs` and `zs` (decision 113), into
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
