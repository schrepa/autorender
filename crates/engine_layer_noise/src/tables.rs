use engine_math::{clamp, max};
use engine_terrain::noise::{
    gradient_noise, gradient_noise4, gradient_value, gradient_value4, gullies, gullies4, max4,
    min4, smooth, smooth4, sqrt4,
};
use glam::Vec4;

use crate::layer::NoiseLayer;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Octave {
    m: [f32; 4],
    t: [f32; 2],
    amplitude: f32,
    wavelength: f32,
    seed: u32,
}

impl Octave {
    #[inline(always)]
    fn lattice(&self, x: f32, z: f32) -> (f32, f32) {
        (
            self.m[0] * x + self.m[1] * z + self.t[0],
            self.m[2] * x + self.m[3] * z + self.t[1],
        )
    }

    #[inline(always)]
    fn lattice4(&self, x: Vec4, z: Vec4) -> (Vec4, Vec4) {
        (
            self.m[0] * x + self.m[1] * z + self.t[0],
            self.m[2] * x + self.m[3] * z + self.t[1],
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GullyOctave {
    cell: f32,
    depth: f32,
    seed: u32,
}

const OCTAVES: usize = 28;
const GULLY_OCTAVES: usize = 5;

pub(crate) const STATE_HEAD: usize = 6;
pub(crate) const STATE_OCTAVE: usize = 1;

pub(crate) const fn state_len(octaves: usize) -> usize {
    STATE_HEAD + STATE_OCTAVE * octaves
}

#[derive(Clone, Copy, Debug)]
struct Sums {
    rolling: f32,
    ridged: f32,
    weight: f32,
    lattice_slope: [f32; 2],
}

impl Sums {
    const START: Self = Self {
        rolling: 0.0,
        ridged: 0.0,
        weight: 1.0,
        lattice_slope: [0.0; 2],
    };

    #[inline(always)]
    fn add(&mut self, octave: &Octave, smoothing: f32, [n, nx, nz]: [f32; 3]) -> (f32, f32) {
        self.lattice_slope[0] += nx;
        self.lattice_slope[1] += nz;
        let damping = 1.0
            / (1.0
                + smoothing
                    * (self.lattice_slope[0] * self.lattice_slope[0]
                        + self.lattice_slope[1] * self.lattice_slope[1]));
        let a = octave.amplitude * damping;
        self.rolling += a * n;
        (a, self.add_ridged(octave, n))
    }

    #[inline(always)]
    fn add_ridged(&mut self, octave: &Octave, n: f32) -> f32 {
        let crest = 1.0 - n.abs();
        let weighted = crest * self.weight;
        let signal = crest * weighted;
        self.ridged += octave.amplitude * signal;
        self.weight = clamp(signal * 2.0, 0.0, 1.0);
        weighted
    }
}

struct Sums4 {
    rolling: Vec4,
    ridged: Vec4,
    weight: Vec4,
    lattice_slope: [Vec4; 2],
}

impl Sums4 {
    const START: Self = Self {
        rolling: Vec4::ZERO,
        ridged: Vec4::ZERO,
        weight: Vec4::ONE,
        lattice_slope: [Vec4::ZERO; 2],
    };

    #[inline(always)]
    fn add(&mut self, octave: &Octave, smoothing: f32, [n, nx, nz]: [Vec4; 3]) -> (Vec4, Vec4) {
        self.lattice_slope[0] += nx;
        self.lattice_slope[1] += nz;
        let damping = 1.0
            / (1.0
                + smoothing
                    * (self.lattice_slope[0] * self.lattice_slope[0]
                        + self.lattice_slope[1] * self.lattice_slope[1]));
        let a = octave.amplitude * damping;
        self.rolling += a * n;
        (a, self.add_ridged(octave, n))
    }

    #[inline(always)]
    fn add_ridged(&mut self, octave: &Octave, n: Vec4) -> Vec4 {
        let crest = 1.0 - n.abs();
        let weighted = crest * self.weight;
        let signal = crest * weighted;
        self.ridged += octave.amplitude * signal;
        self.weight = min4(max4(signal * 2.0, Vec4::ZERO), Vec4::ONE);
        weighted
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NoiseTables {
    pub(crate) params: NoiseLayer,
    octaves: [Octave; OCTAVES],
    gullies: [GullyOctave; GULLY_OCTAVES],
}

pub(crate) const COARSE_SAMPLES: f32 = 32.0;
pub(crate) const FINEST_SAMPLES: f32 = 4.0;
pub(crate) const GULLY_SAMPLES: f32 = 16.0;
const CREASE: f32 = 1.0;

impl NoiseTables {
    pub(crate) fn new(params: NoiseLayer) -> Self {
        let (cos, sin) = (0.8_f64, 0.6_f64);
        let gain = 0.35 + 0.3 * f64::from(params.roughness);
        let mut amplitude = 1.0 - gain;
        let mut m = [
            1.0 / f64::from(params.wavelength),
            0.0,
            0.0,
            1.0 / f64::from(params.wavelength),
        ];
        let octaves = std::array::from_fn(|o| {
            let octave = Octave {
                m: m.map(|value| value as f32),
                t: [
                    (o as f64 * 17.137 + 0.31) as f32,
                    (o as f64 * 31.713 + 0.73) as f32,
                ],
                amplitude: amplitude as f32,
                wavelength: (f64::from(params.wavelength) / f64::from(1u32 << o.min(31))) as f32,
                seed: params
                    .seed
                    .wrapping_add((o as u32).wrapping_mul(0x9e37_79b9)),
            };
            amplitude *= gain;
            m = [
                2.0 * (cos * m[0] - sin * m[2]),
                2.0 * (cos * m[1] - sin * m[3]),
                2.0 * (sin * m[0] + cos * m[2]),
                2.0 * (sin * m[1] + cos * m[3]),
            ];
            octave
        });
        let gullies = std::array::from_fn(|k| {
            let cell = params.wavelength / (8u32 << k) as f32;
            GullyOctave {
                cell,
                depth: 0.05 * cell,
                seed: params.seed ^ 0x6a09_e667_u32.wrapping_add(k as u32),
            }
        });
        Self {
            params,
            octaves,
            gullies,
        }
    }

    pub(crate) fn bands(&self, spacing: f32) -> (usize, std::ops::Range<usize>, usize) {
        let coarse = self
            .octaves
            .iter()
            .take_while(|octave| octave.wavelength >= COARSE_SAMPLES * spacing)
            .count();
        let fine = coarse
            + self.octaves[coarse..]
                .iter()
                .take_while(|octave| octave.wavelength >= FINEST_SAMPLES * spacing)
                .count();
        let gullies = self
            .gullies
            .iter()
            .take_while(|octave| octave.cell >= GULLY_SAMPLES * spacing)
            .count();
        (coarse, coarse..fine, gullies)
    }

    #[inline]
    fn warp4(&self, x: Vec4, z: Vec4) -> [Vec4; 2] {
        let warp = self.params.warp;
        if warp == 0.0 {
            return [Vec4::ZERO; 2];
        }
        let w = self.params.wavelength;
        let (u, v) = (x / w, z / w);
        let seed = self.params.seed ^ 0xbb67_ae85;
        let a = gradient_value4(u * 0.75 + 11.3, v * 0.75 + 7.1, seed)
            + 0.5 * gradient_value4(u * 1.5 - 3.7, v * 1.5 + 19.9, seed ^ 1);
        let b = gradient_value4(u * 0.75 - 23.9, v * 0.75 + 41.7, seed ^ 2)
            + 0.5 * gradient_value4(u * 1.5 + 29.3, v * 1.5 - 13.1, seed ^ 3);
        let reach = warp * w * 0.35;
        [reach * a, reach * b]
    }

    #[inline]
    fn warp(&self, x: f32, z: f32) -> [f32; 2] {
        let warp = self.params.warp;
        if warp == 0.0 {
            return [0.0; 2];
        }
        let w = self.params.wavelength;
        let (u, v) = (x / w, z / w);
        let seed = self.params.seed ^ 0xbb67_ae85;
        let a = gradient_value(u * 0.75 + 11.3, v * 0.75 + 7.1, seed)
            + 0.5 * gradient_value(u * 1.5 - 3.7, v * 1.5 + 19.9, seed ^ 1);
        let b = gradient_value(u * 0.75 - 23.9, v * 0.75 + 41.7, seed ^ 2)
            + 0.5 * gradient_value(u * 1.5 + 29.3, v * 1.5 - 13.1, seed ^ 3);
        let reach = warp * w * 0.35;
        [reach * a, reach * b]
    }

    pub(crate) fn coarse(
        &self,
        x: f32,
        z: f32,
        octaves: usize,
        gully_octaves: usize,
        state: &mut [f32],
    ) {
        let params = &self.params;
        let warp = self.warp(x, z);
        let (px, pz) = (x + warp[0], z + warp[1]);
        let mut sums = Sums::START;
        let mut terms = [[0.0_f32; 6]; GULLY_OCTAVES + 1];
        let mut blended = [0.0_f32; GULLY_OCTAVES];
        let values = state[STATE_HEAD..].iter_mut();
        for ((index, octave), value) in self.octaves[..octaves].iter().enumerate().zip(values) {
            let (qx, qz) = octave.lattice(px, pz);
            let noise = gradient_noise(qx, qz, octave.seed);
            let [n, nx, nz] = noise;
            *value = n;
            let gx = octave.m[0] * nx + octave.m[2] * nz;
            let gz = octave.m[1] * nx + octave.m[3] * nz;
            let (a, weighted) = sums.add(octave, params.smoothing, noise);
            if let Some(term) = terms.get_mut(index) {
                let ridge = -2.0 * weighted * octave.amplitude;
                *term = [a * gx, a * gz, ridge * gx, ridge * gz, n, gx * gx + gz * gz];
            }
            if let Some(slot) = index.checked_sub(1).and_then(|k| blended.get_mut(k)) {
                *slot = blend(sums.rolling, sums.ridged, params.ridges);
            }
        }
        let mut gully_heights = 0.0;
        for (k, (octave, mixed)) in self.gullies[..gully_octaves]
            .iter()
            .zip(blended)
            .enumerate()
        {
            let width = CREASE * octave.cell;
            let (mut rolling, mut ridged) = ([0.0_f32; 2], [0.0_f32; 2]);
            for [rx, rz, gx, gz, n, slope] in &terms[..k + 2] {
                rolling[0] += rx;
                rolling[1] += rz;
                let turn = n / max((n * n + width * width * slope).sqrt(), 1.0e-12);
                ridged[0] += turn * gx;
                ridged[1] += turn * gz;
            }
            let mx = blend_gradient(rolling[0], ridged[0], params.ridges);
            let mz = blend_gradient(rolling[1], ridged[1], params.ridges);
            let h = clamp(mixed, 0.0, 1.0);
            let shape_slope = params.height * (1.0 + params.valleys * (2.0 * h - 1.0));
            let (gx, gz) = (shape_slope * mx, shape_slope * mz);
            let slope = (gx * gx + gz * gz).sqrt();
            let inverse = 1.0 / max(slope, 1.0e-6);
            let across = [-gz * inverse, gx * inverse];
            let [c, _] = gullies(px / octave.cell, pz / octave.cell, across, octave.seed);
            let depth = octave.depth * params.erosion * smooth((slope - 0.05) * (1.0 / 0.45));
            gully_heights += depth * c;
        }
        state[..STATE_HEAD].copy_from_slice(&[
            warp[0],
            warp[1],
            gully_heights,
            sums.rolling,
            sums.lattice_slope[0],
            sums.lattice_slope[1],
        ]);
    }

    pub(crate) fn coarse4(
        &self,
        x: Vec4,
        z: f32,
        octaves: usize,
        gully_octaves: usize,
        out: &mut [f32],
    ) {
        let params = &self.params;
        let warp = self.warp4(x, Vec4::splat(z));
        let (px, pz) = (x + warp[0], z + warp[1]);
        let mut sums = Sums4::START;
        let mut terms = [[Vec4::ZERO; 6]; GULLY_OCTAVES + 1];
        let mut blended = [Vec4::ZERO; GULLY_OCTAVES];
        let mut values = [Vec4::ZERO; OCTAVES];
        for ((index, octave), value) in self.octaves[..octaves].iter().enumerate().zip(&mut values)
        {
            let (qx, qz) = octave.lattice4(px, pz);
            let noise = gradient_noise4(qx, qz, octave.seed);
            let [n, nx, nz] = noise;
            *value = n;
            let gx = octave.m[0] * nx + octave.m[2] * nz;
            let gz = octave.m[1] * nx + octave.m[3] * nz;
            let (a, weighted) = sums.add(octave, params.smoothing, noise);
            if let Some(term) = terms.get_mut(index) {
                let ridge = -2.0 * weighted * octave.amplitude;
                *term = [a * gx, a * gz, ridge * gx, ridge * gz, n, gx * gx + gz * gz];
            }
            if let Some(slot) = index.checked_sub(1).and_then(|k| blended.get_mut(k)) {
                *slot = blend4(sums.rolling, sums.ridged, params.ridges);
            }
        }
        let mut gully_heights = Vec4::ZERO;
        for (k, (octave, mixed)) in self.gullies[..gully_octaves]
            .iter()
            .zip(blended)
            .enumerate()
        {
            let width = CREASE * octave.cell;
            let (mut rolling, mut ridged) = ([Vec4::ZERO; 2], [Vec4::ZERO; 2]);
            for &[rx, rz, gx, gz, n, slope] in &terms[..k + 2] {
                rolling[0] += rx;
                rolling[1] += rz;
                let turn = n / max4(sqrt4(n * n + width * width * slope), Vec4::splat(1.0e-12));
                ridged[0] += turn * gx;
                ridged[1] += turn * gz;
            }
            let mx = blend_gradient4(rolling[0], ridged[0], params.ridges);
            let mz = blend_gradient4(rolling[1], ridged[1], params.ridges);
            let h = min4(max4(mixed, Vec4::ZERO), Vec4::ONE);
            let shape_slope = params.height * (1.0 + params.valleys * (2.0 * h - 1.0));
            let (gx, gz) = (shape_slope * mx, shape_slope * mz);
            let slope = sqrt4(gx * gx + gz * gz);
            let inverse = 1.0 / max4(slope, Vec4::splat(1.0e-6));
            let across = [-gz * inverse, gx * inverse];
            let c = gullies4(px / octave.cell, pz / octave.cell, across, octave.seed);
            let depth = octave.depth * params.erosion * smooth4((slope - 0.05) * (1.0 / 0.45));
            gully_heights += depth * c;
        }
        let head = [
            warp[0],
            warp[1],
            gully_heights,
            sums.rolling,
            sums.lattice_slope[0],
            sums.lattice_slope[1],
        ];
        for (lane, state) in out.chunks_exact_mut(state_len(octaves)).take(4).enumerate() {
            let (first, rest) = state.split_at_mut(STATE_HEAD);
            for (value, lanes) in first.iter_mut().zip(&head) {
                *value = lanes[lane];
            }
            for (value, lanes) in rest.iter_mut().zip(&values) {
                *value = lanes[lane];
            }
        }
    }

    #[inline]
    pub(crate) fn finish(
        &self,
        x: f32,
        z: f32,
        state: &[f32],
        octaves: usize,
        fine: std::ops::Range<usize>,
    ) -> f32 {
        let params = &self.params;
        let (px, pz) = (x + state[0], z + state[1]);
        let mut sums = Sums {
            rolling: state[3],
            lattice_slope: [state[4], state[5]],
            ..Sums::START
        };
        for (octave, &n) in self.octaves[..octaves].iter().zip(&state[STATE_HEAD..]) {
            sums.add_ridged(octave, n);
        }
        for octave in &self.octaves[fine] {
            let (qx, qz) = octave.lattice(px, pz);
            sums.add(
                octave,
                params.smoothing,
                gradient_noise(qx, qz, octave.seed),
            );
        }
        let h = clamp(blend(sums.rolling, sums.ridged, params.ridges), 0.0, 1.0);
        let shaped = h + params.valleys * (h * h - h);
        params.offset + params.height * shaped + state[2]
    }

    #[inline]
    pub(crate) fn finish4(
        &self,
        x: Vec4,
        z: f32,
        state: &[Vec4],
        octaves: usize,
        fine: std::ops::Range<usize>,
    ) -> Vec4 {
        let params = &self.params;
        let (px, pz) = (x + state[0], z + state[1]);
        let mut sums = Sums4 {
            rolling: state[3],
            ridged: Vec4::ZERO,
            weight: Vec4::ONE,
            lattice_slope: [state[4], state[5]],
        };
        for (octave, &n) in self.octaves[..octaves].iter().zip(&state[STATE_HEAD..]) {
            sums.add_ridged(octave, n);
        }
        for octave in &self.octaves[fine] {
            let (qx, qz) = octave.lattice4(px, pz);
            sums.add(
                octave,
                params.smoothing,
                gradient_noise4(qx, qz, octave.seed),
            );
        }
        let h = min4(
            max4(blend4(sums.rolling, sums.ridged, params.ridges), Vec4::ZERO),
            Vec4::ONE,
        );
        let shaped = h + params.valleys * (h * h - h);
        params.offset + params.height * shaped + state[2]
    }
}

#[inline(always)]
fn blend(rolling: f32, ridged: f32, ridges: f32) -> f32 {
    0.5 + 0.8 * rolling + ridges * (ridged - 0.5 - 0.8 * rolling)
}

#[inline(always)]
fn blend_gradient(rolling: f32, ridged: f32, ridges: f32) -> f32 {
    0.8 * rolling + ridges * (ridged - 0.8 * rolling)
}

#[inline(always)]
fn blend4(rolling: Vec4, ridged: Vec4, ridges: f32) -> Vec4 {
    0.5 + 0.8 * rolling + ridges * (ridged - 0.5 - 0.8 * rolling)
}

#[inline(always)]
fn blend_gradient4(rolling: Vec4, ridged: Vec4, ridges: f32) -> Vec4 {
    0.8 * rolling + ridges * (ridged - 0.8 * rolling)
}
