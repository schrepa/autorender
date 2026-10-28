use std::sync::Arc;

use engine_math::{clamp, max, min};
use engine_settings::{InvalidSetting, Rule, Setting, settings};
use glam::{Vec2, Vec4};
use serde::{Deserialize, Serialize};

use crate::bake::{ErosionBake, HydrologyBake, PathBake, Stage};
use crate::given::{GivenRiver, RiversLayer};
use crate::hydrology::Hydrology;
use crate::noise::{
    gradient_noise, gradient_noise4, gradient_value, gradient_value4, gullies, gullies4, max4,
    min4, smooth, smooth4, sqrt4,
};
use crate::path::PathSample;
use crate::relief::{HummocksLayer, StepsLayer};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Blend {
    #[default]
    Add,
    Max,
    Min,
    Replace,
}

impl Setting for Blend {}

impl Blend {
    #[inline(always)]
    fn apply(self, below: f32, value: f32, weight: f32) -> f32 {
        let target = match self {
            Self::Add => below + value,
            Self::Max => max(below, value),
            Self::Min => min(below, value),
            Self::Replace => value,
        };
        below + (target - below) * weight
    }
}

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct NoiseLayer {
        pub seed: u32 = 0 => any,
        pub height: f32 = 900.0 => range(0.0, MAX_HEIGHT),
        pub offset: f32 = 0.0 => range(-MAX_HEIGHT, MAX_HEIGHT),
        pub wavelength: f32 = 3000.0 => range(16.0, 1.0e6),
        pub ridges: f32 = 0.3 => range(0.0, 1.0),
        pub warp: f32 = 0.35 => range(0.0, 1.0),
        pub erosion: f32 = 0.7 => range(0.0, 1.0),
        pub smoothing: f32 = 1.0 => range(0.0, 1.0),
        pub valleys: f32 = 0.7 => range(0.0, 1.0),
        pub roughness: f32 = 0.5 => range(0.0, 1.0),
        pub blend: Blend = Blend::Add => any,
    }
}

pub const MAX_HEIGHT: f32 = 20_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum StampOp {
    Raise(#[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = MAX_HEIGHT)))] f32),
    Lower(#[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = MAX_HEIGHT)))] f32),
    Flatten(
        #[cfg_attr(feature = "schema", schemars(range(min = -MAX_HEIGHT, max = MAX_HEIGHT)))] f32,
    ),
    Hole,
}

impl Setting for StampOp {
    fn check(&self, field: &'static str, _: Rule) -> Result<(), InvalidSetting> {
        let (variant, height, rule) = match *self {
            Self::Raise(height) => ("raise", height, Rule::Range(0.0, MAX_HEIGHT as f64)),
            Self::Lower(height) => ("lower", height, Rule::Range(0.0, MAX_HEIGHT as f64)),
            Self::Flatten(height) => (
                "flatten",
                height,
                Rule::Range(-MAX_HEIGHT as f64, MAX_HEIGHT as f64),
            ),
            Self::Hole => return Ok(()),
        };
        height
            .check(variant, rule)
            .map_err(|invalid| invalid.within(field))
    }
}

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct Stamp {
        pub center: [f32; 2] = [0.0; 2] => range(-Layers::MAX_COORDINATE, Layers::MAX_COORDINATE),
        pub radius: f32 = 1.0 => range(0.0, Layers::MAX_COORDINATE),
        #[serde(default)]
        pub falloff: f32 = 0.0 => range(0.0, Layers::MAX_COORDINATE),
        pub op: StampOp = StampOp::Hole => any,
    }
}

impl Stamp {
    #[inline]
    fn weight(&self, x: f32, z: f32) -> f32 {
        let dx = x - self.center[0];
        let dz = z - self.center[1];
        let distance = (dx * dx + dz * dz).sqrt();
        1.0 - smooth((distance - self.radius) / max(self.falloff, 1.0e-6))
    }

    fn reach(&self) -> f32 {
        self.radius + self.falloff
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeightImage {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: Box<[u16]>,
    pub(crate) levels: Vec<ImageLevel>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ImageLevel {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: Box<[f32]>,
}

impl ImageLevel {
    fn below(width: u32, height: u32, samples: &[f32]) -> Self {
        let half = |size: u32| (size - 1).div_ceil(2) + 1;
        let (next_width, next_height) = (half(width), half(height));
        let at = |x: i64, z: i64| {
            let x = x.clamp(0, i64::from(width) - 1) as usize;
            let z = z.clamp(0, i64::from(height) - 1) as usize;
            samples[z * width as usize + x]
        };
        let mut rows = vec![0.0_f32; next_width as usize * height as usize];
        for z in 0..i64::from(height) {
            for x in 0..i64::from(next_width) {
                let center = 2 * x;
                rows[z as usize * next_width as usize + x as usize] =
                    0.25 * at(center - 1, z) + 0.5 * at(center, z) + 0.25 * at(center + 1, z);
            }
        }
        let row = |x: i64, z: i64| {
            let z = z.clamp(0, i64::from(height) - 1) as usize;
            rows[z * next_width as usize + x as usize]
        };
        let mut next = vec![0.0_f32; next_width as usize * next_height as usize];
        for z in 0..i64::from(next_height) {
            for x in 0..i64::from(next_width) {
                let center = 2 * z;
                next[z as usize * next_width as usize + x as usize] =
                    0.25 * row(x, center - 1) + 0.5 * row(x, center) + 0.25 * row(x, center + 1);
            }
        }
        Self {
            width: next_width,
            height: next_height,
            samples: next.into_boxed_slice(),
        }
    }
}

impl HeightImage {
    pub fn new(width: u32, height: u32, samples: Vec<u16>) -> Option<Self> {
        if !(width >= 2 && height >= 2 && samples.len() == width as usize * height as usize) {
            return None;
        }
        let mut levels: Vec<ImageLevel> = Vec::new();
        let mut current: (u32, u32, Vec<f32>) = (
            width,
            height,
            samples.iter().map(|&sample| f32::from(sample)).collect(),
        );
        while current.0 > 2 || current.1 > 2 {
            let level = ImageLevel::below(current.0, current.1, &current.2);
            current = (level.width, level.height, level.samples.to_vec());
            levels.push(level);
        }
        Some(Self {
            width,
            height,
            samples: samples.into_boxed_slice(),
            levels,
        })
    }

    pub fn from_le_bytes(width: u32, height: u32, bytes: &[u8]) -> Option<Self> {
        if bytes.len() != width as usize * height as usize * 2 {
            return None;
        }
        let samples = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        Self::new(width, height, samples)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    #[inline]
    pub(crate) fn sample_smooth(&self, u: f32, v: f32, level: usize) -> f32 {
        let mut sample = 0.0;
        self.sample_smooth_row([u], v, level, |_, value| sample = value);
        sample
    }

    #[inline]
    pub(crate) fn sample_smooth_row(
        &self,
        us: impl IntoIterator<Item = f32>,
        v: f32,
        level: usize,
        each: impl FnMut(usize, f32),
    ) {
        match level
            .checked_sub(1)
            .and_then(|index| self.levels.get(index))
        {
            None => smooth_row(us, v, 1.0, each, |x, z| {
                let x = x.clamp(0, self.width as i32 - 1) as usize;
                let z = z.clamp(0, self.height as i32 - 1) as usize;
                f32::from(self.samples[z * self.width as usize + x])
            }),
            Some(coarse) => {
                let scale = 1.0 / (1_u32 << level) as f32;
                smooth_row(us, v, scale, each, |x, z| {
                    let x = x.clamp(0, coarse.width as i32 - 1) as usize;
                    let z = z.clamp(0, coarse.height as i32 - 1) as usize;
                    coarse.samples[z * coarse.width as usize + x]
                });
            }
        }
    }

    #[inline]
    pub(crate) fn sample(&self, u: f32, v: f32, level: usize) -> f32 {
        match level
            .checked_sub(1)
            .and_then(|index| self.levels.get(index))
        {
            None => catmull_rom_2d(u, v, |x, z| {
                let x = x.clamp(0, self.width as i32 - 1) as usize;
                let z = z.clamp(0, self.height as i32 - 1) as usize;
                f32::from(self.samples[z * self.width as usize + x])
            }),
            Some(coarse) => {
                let scale = 1.0 / (1_u32 << level) as f32;
                catmull_rom_2d(u * scale, v * scale, |x, z| {
                    let x = x.clamp(0, coarse.width as i32 - 1) as usize;
                    let z = z.clamp(0, coarse.height as i32 - 1) as usize;
                    coarse.samples[z * coarse.width as usize + x]
                })
            }
        }
    }
}

#[inline(always)]
fn catmull_rom_2d(u: f32, v: f32, texel: impl Fn(i32, i32) -> f32) -> f32 {
    let u0 = u.floor();
    let v0 = v.floor();
    let (x, z) = (u0 as i32, v0 as i32);
    let wx = catmull_rom(u - u0);
    let wz = catmull_rom(v - v0);
    let mut sum = 0.0;
    for (row, weight_z) in wz.iter().enumerate() {
        let zi = z - 1 + row as i32;
        let line = wx[0] * texel(x - 1, zi)
            + wx[1] * texel(x, zi)
            + wx[2] * texel(x + 1, zi)
            + wx[3] * texel(x + 2, zi);
        sum += weight_z * line;
    }
    sum
}

#[inline(always)]
fn smooth_row(
    us: impl IntoIterator<Item = f32>,
    v: f32,
    scale: f32,
    mut each: impl FnMut(usize, f32),
    texel: impl Fn(i32, i32) -> f32,
) {
    let v = v * scale;
    let v0 = v.floor();
    let z = v0 as i32;
    let wz = b_spline(v - v0);
    let column = |x: i32| {
        wz[0] * texel(x, z - 1)
            + wz[1] * texel(x, z)
            + wz[2] * texel(x, z + 1)
            + wz[3] * texel(x, z + 2)
    };
    let mut cell: Option<(i32, [f32; 4])> = None;
    for (index, u) in us.into_iter().enumerate() {
        let u = u * scale;
        let u0 = u.floor();
        let x = u0 as i32;
        let columns = match cell {
            Some((at, columns)) if at == x => columns,
            Some((at, [_, b, c, d])) if x.checked_sub(at) == Some(1) => [b, c, d, column(x + 2)],
            _ => [column(x - 1), column(x), column(x + 1), column(x + 2)],
        };
        cell = Some((x, columns));
        let wx = b_spline(u - u0);
        each(
            index,
            wx[0] * columns[0] + wx[1] * columns[1] + wx[2] * columns[2] + wx[3] * columns[3],
        );
    }
}

#[inline(always)]
fn b_spline(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    let t2 = t * t;
    let t3 = t2 * t;
    [
        s * s * s * (1.0 / 6.0),
        (3.0 * t3 - 6.0 * t2 + 4.0) * (1.0 / 6.0),
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) * (1.0 / 6.0),
        t3 * (1.0 / 6.0),
    ]
}

#[inline(always)]
pub(crate) fn catmull_rom(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ]
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeightmapLayer {
    pub image: Arc<HeightImage>,
    pub min: Vec2,
    pub size: Vec2,
    pub low: f32,
    pub high: f32,
    pub falloff: f32,
    pub blend: Blend,
    pub uri: Option<String>,
    pub source: Option<String>,
    pub attribution: Option<String>,
}

impl HeightmapLayer {
    pub(crate) fn texel(&self) -> f32 {
        let image = &self.image;
        max(
            self.size.x / (image.width - 1) as f32,
            self.size.y / (image.height - 1) as f32,
        )
    }

    fn level_for(&self, spacing: f32) -> usize {
        let texel = self.texel();
        let mut level = 0;
        while level < self.image.levels() && texel * (2_u32 << level) as f32 <= spacing * 1.0001 {
            level += 1;
        }
        level
    }

    #[inline]
    fn at(&self, x: f32, z: f32, level: usize) -> (f32, f32) {
        let local = Vec2::new(x, z) - self.min;
        let inside = min(
            min(local.x, self.size.x - local.x),
            min(local.y, self.size.y - local.y),
        );
        if inside < 0.0 {
            return (0.0, 0.0);
        }
        let weight = if self.falloff > 0.0 {
            smooth(inside / self.falloff)
        } else {
            1.0
        };
        let u = local.x / self.size.x * (self.image.width - 1) as f32;
        let v = local.y / self.size.y * (self.image.height - 1) as f32;
        let value = self.image.sample(u, v, level) * (1.0 / 65_535.0);
        (self.low + (self.high - self.low) * value, weight)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Layer {
    Noise(NoiseLayer),
    Heightmap(HeightmapLayer),
    Stamp(Stamp),
    Rivers(Arc<RiversLayer>),
    Hydrology(HydrologyStage),
    Erosion(ErosionStage),
    Path(PathStage),
    Steps(StepsLayer),
    Hummocks(HummocksLayer),
}

pub type PathStage = Stage<PathBake>;

pub type ErosionStage = Stage<ErosionBake>;

pub type HydrologyStage = Stage<HydrologyBake>;

#[derive(Clone, Debug, PartialEq)]
pub struct LayerError {
    pub layer: usize,
    pub field: std::borrow::Cow<'static, str>,
    pub rule: std::borrow::Cow<'static, str>,
}

impl LayerError {
    fn new(layer: usize, kind: &'static str, invalid: InvalidSetting) -> Self {
        let invalid = invalid.within(kind);
        Self {
            layer,
            field: invalid.field,
            rule: invalid.rule,
        }
    }
}

impl std::fmt::Display for LayerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "layer {} {} {}",
            self.layer, self.field, self.rule
        )
    }
}

impl std::error::Error for LayerError {}

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
    params: NoiseLayer,
    octaves: [Octave; OCTAVES],
    gullies: [GullyOctave; GULLY_OCTAVES],
}

pub(crate) const COARSE_SAMPLES: f32 = 32.0;
pub(crate) const FINEST_SAMPLES: f32 = 4.0;
pub(crate) const GULLY_SAMPLES: f32 = 16.0;
const CREASE: f32 = 1.0;

impl NoiseTables {
    fn new(params: NoiseLayer) -> Self {
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

#[derive(Clone, Debug, PartialEq)]
pub struct Layers {
    layers: Vec<Layer>,
    tables: Vec<Option<NoiseTables>>,
    pub(crate) offset: Vec2,
    pub(crate) fade: Option<Fade>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Fade {
    pub(crate) low: Vec2,
    pub(crate) high: Vec2,
    pub(crate) blend: f32,
}

fn check_stamp(index: usize, stamp: &Stamp) -> Result<(), LayerError> {
    stamp
        .validate()
        .map_err(|invalid| LayerError::new(index, "stamp", invalid))
}

const HEIGHTMAP_MIN: Rule = Rule::Range(
    -Layers::MAX_COORDINATE as f64,
    Layers::MAX_COORDINATE as f64,
);
const HEIGHTMAP_SIZE: Rule = Rule::Range(1.0e-3, Layers::MAX_COORDINATE as f64);
const HEIGHTMAP_HEIGHT: Rule = Rule::Range(-MAX_HEIGHT as f64, MAX_HEIGHT as f64);
const HEIGHTMAP_FALLOFF: Rule = Rule::Range(0.0, Layers::MAX_COORDINATE as f64);

impl HeightmapLayer {
    pub fn rule_of(field: &str) -> Option<Rule> {
        match field {
            "min" => Some(HEIGHTMAP_MIN),
            "size" => Some(HEIGHTMAP_SIZE),
            "low" | "high" => Some(HEIGHTMAP_HEIGHT),
            "falloff" => Some(HEIGHTMAP_FALLOFF),
            _ => None,
        }
    }

    fn validate(&self) -> Result<(), InvalidSetting> {
        self.min.to_array().check("min", HEIGHTMAP_MIN)?;
        self.size.to_array().check("size", HEIGHTMAP_SIZE)?;
        self.low.check("low", HEIGHTMAP_HEIGHT)?;
        self.high.check("high", HEIGHTMAP_HEIGHT)?;
        self.falloff.check("falloff", HEIGHTMAP_FALLOFF)
    }
}

impl Layers {
    pub const MAX_COORDINATE: f32 = 1.0e7;

    pub fn new(layers: Vec<Layer>) -> Result<Self, LayerError> {
        let last_hydrology = layers
            .iter()
            .rposition(|layer| matches!(layer, Layer::Hydrology(_)));
        for (index, layer) in layers.iter().enumerate() {
            let (kind, checked) = match layer {
                Layer::Noise(noise) => ("noise", noise.validate()),
                Layer::Heightmap(map) => ("heightmap", map.validate()),
                Layer::Stamp(stamp) => ("stamp", stamp.validate()),
                Layer::Rivers(rivers) => ("rivers", rivers.validate()),
                Layer::Erosion(stage) => ("erosion", stage.layer.validate()),
                Layer::Path(stage) => ("path", stage.layer.validate()),
                Layer::Hydrology(stage) => ("hydrology", stage.layer.validate()),
                Layer::Steps(steps) => ("steps", steps.validate()),
                Layer::Hummocks(hummocks) => ("hummocks", hummocks.validate()),
            };
            checked.map_err(|invalid| LayerError::new(index, kind, invalid))?;
            if matches!(layer, Layer::Steps(_) | Layer::Hummocks(_)) && last_hydrology.is_none() {
                return Err(LayerError::new(
                    index,
                    kind,
                    InvalidSetting::new(
                        "wetness",
                        "needs a hydrology layer in the stack, whose maps place it",
                    ),
                ));
            }
        }
        if let Some(index) = layers.iter().enumerate().position(|(index, layer)| {
            matches!(layer, Layer::Rivers(_)) && last_hydrology.is_none_or(|last| last < index)
        }) {
            return Err(LayerError::new(
                index,
                "rivers",
                InvalidSetting::new(
                    "lines",
                    "need a hydrology layer after them, which runs them",
                ),
            ));
        }
        let tables = layers
            .iter()
            .map(|layer| match layer {
                Layer::Noise(noise) => Some(NoiseTables::new(*noise)),
                Layer::Heightmap(_)
                | Layer::Stamp(_)
                | Layer::Rivers(_)
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_)
                | Layer::Steps(_)
                | Layer::Hummocks(_) => None,
            })
            .collect();
        Ok(Self {
            layers,
            tables,
            offset: Vec2::ZERO,
            fade: None,
        })
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn may_hole(&self) -> bool {
        self.layers.iter().any(|layer| match layer {
            Layer::Stamp(stamp) => matches!(stamp.op, StampOp::Hole),
            Layer::Path(stage) => stage.layer.tunnel_cut > 0.0,
            Layer::Noise(_)
            | Layer::Heightmap(_)
            | Layer::Rivers(_)
            | Layer::Hydrology(_)
            | Layer::Erosion(_)
            | Layer::Steps(_)
            | Layer::Hummocks(_) => false,
        })
    }

    pub fn has_hydrology(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(layer, Layer::Hydrology(_)))
    }

    #[must_use]
    pub fn surroundings(&self, offset: Vec2, (low, high): (Vec2, Vec2), blend: f32) -> Self {
        Self {
            offset,
            fade: Some(Fade {
                low,
                high,
                blend: max(blend, 0.0),
            }),
            ..self.clone()
        }
    }

    pub(crate) fn shown(&self, x: f32, z: f32) -> f32 {
        let Some(fade) = self.fade else {
            return 1.0;
        };
        let point = Vec2::new(x, z);
        let outside = (fade.low - point)
            .max(point - fade.high)
            .max(Vec2::ZERO)
            .length();
        if outside <= 0.0 {
            return 1.0;
        }
        if fade.blend <= 0.0 {
            return 0.0;
        }
        let t = min(outside / fade.blend, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }

    pub(crate) fn offset_samples(&self, spacing: f32) -> [i64; 2] {
        [
            (self.offset.x / spacing).round() as i64,
            (self.offset.y / spacing).round() as i64,
        ]
    }

    pub fn with_stamps(&self, stamps: &[Stamp]) -> Result<Self, LayerError> {
        let mut layers = self.layers.clone();
        let mut tables = self.tables.clone();
        for (offset, stamp) in stamps.iter().enumerate() {
            check_stamp(self.layers.len() + offset, stamp)?;
            layers.push(Layer::Stamp(*stamp));
            tables.push(None);
        }
        Ok(Self {
            layers,
            tables,
            offset: self.offset,
            fade: self.fade,
        })
    }

    #[must_use]
    pub fn with_drawn(&self, drawn: &[GivenRiver], baked: Option<&Self>) -> Self {
        let Some(first) = self
            .layers
            .iter()
            .position(|layer| matches!(layer, Layer::Hydrology(_)))
        else {
            return self.clone();
        };
        let mut layers: Vec<Layer> = self.layers.iter().map(unbaked).collect();
        let mut tables = self.tables.clone();
        if let Some(baked) = baked {
            for (layer, done) in layers[..first].iter_mut().zip(&baked.layers) {
                if unbaked(done) == *layer {
                    *layer = done.clone();
                }
            }
        }
        if !drawn.is_empty() {
            layers.insert(
                first,
                Layer::Rivers(Arc::new(RiversLayer {
                    lines: drawn.to_vec(),
                    ..RiversLayer::default()
                })),
            );
            tables.insert(first, None);
        }
        Self {
            layers,
            tables,
            offset: self.offset,
            fade: self.fade,
        }
    }

    pub fn changes_from(&self, before: &Self) -> Option<Vec<(Vec2, Vec2)>> {
        let shown = |stack: &Self| -> Vec<Layer> {
            stack
                .layers
                .iter()
                .filter(|layer| !matches!(layer, Layer::Rivers(_)))
                .cloned()
                .collect()
        };
        let (now, then) = (shown(self), shown(before));
        if now.len() != then.len() || self.offset != before.offset || self.fade != before.fade {
            return None;
        }
        let mut regions = Vec::new();
        for (now, then) in now.iter().zip(&then) {
            if now == then {
                continue;
            }
            match (now, then) {
                (Layer::Hydrology(now), Layer::Hydrology(then)) if now.layer == then.layer => {
                    let (Some(now), Some(then)) = (&now.baked, &then.baked) else {
                        return None;
                    };
                    if now.flow != then.flow
                        || now.wetness != then.wetness
                        || now.lakes != then.lakes
                        || now.water != then.water
                    {
                        return None;
                    }
                    regions.extend(now.rivers.changes_from(&then.rivers));
                }
                (Layer::Path(now), Layer::Path(then)) if now.layer == then.layer => {
                    let (Some(now), Some(then)) = (&now.baked, &then.baked) else {
                        return None;
                    };
                    let ((low, high), (from, to)) = (now.bounds(), then.bounds());
                    regions.push((low.min(from), high.max(to)));
                }
                _ => return None,
            }
        }
        Some(regions)
    }

    pub(crate) fn layer_mut(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    #[must_use]
    pub fn below(&self, index: usize) -> Self {
        Self {
            layers: self.layers[..index].to_vec(),
            tables: self.tables[..index].to_vec(),
            offset: self.offset,
            fade: self.fade,
        }
    }

    pub fn hydrology(&self) -> Option<&Hydrology> {
        self.hydrology_layer().map(|(_, hydrology)| hydrology)
    }

    pub(crate) fn hydrology_layer(&self) -> Option<(usize, &Hydrology)> {
        self.layers
            .iter()
            .enumerate()
            .find_map(|(index, layer)| match layer {
                Layer::Hydrology(HydrologyStage {
                    baked: Some(baked), ..
                }) => Some((index, baked.as_ref())),
                Layer::Noise(_)
                | Layer::Heightmap(_)
                | Layer::Stamp(_)
                | Layer::Rivers(_)
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_)
                | Layer::Steps(_)
                | Layer::Hummocks(_) => None,
            })
    }

    pub(crate) fn rivers_of(&self, index: usize) -> Option<&crate::Rivers> {
        match self.layers.get(index) {
            Some(Layer::Hydrology(HydrologyStage {
                baked: Some(baked), ..
            })) if !baked.rivers.is_empty() => Some(&baked.rivers),
            _ => None,
        }
    }

    pub(crate) fn path_profiles(&self) -> impl Iterator<Item = &crate::path::PathProfile> {
        self.layers.iter().filter_map(|layer| match layer {
            Layer::Path(PathStage {
                baked: Some(profile),
                ..
            }) => Some(profile.as_ref()),
            Layer::Noise(_)
            | Layer::Heightmap(_)
            | Layer::Stamp(_)
            | Layer::Rivers(_)
            | Layer::Hydrology(_)
            | Layer::Erosion(_)
            | Layer::Path(_)
            | Layer::Steps(_)
            | Layer::Hummocks(_) => None,
        })
    }

    pub fn path_at(&self, x: f32, z: f32) -> PathSample {
        let point = Vec2::new(x, z);
        self.layers
            .iter()
            .fold(PathSample::FAR, |nearest, layer| match layer {
                Layer::Path(PathStage {
                    baked: Some(profile),
                    ..
                }) => {
                    let sample = profile.sample(point);
                    if sample.edge() < nearest.edge() {
                        sample
                    } else {
                        nearest
                    }
                }
                Layer::Noise(_)
                | Layer::Heightmap(_)
                | Layer::Stamp(_)
                | Layer::Rivers(_)
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_)
                | Layer::Steps(_)
                | Layer::Hummocks(_) => nearest,
            })
    }

    pub(crate) fn noise(&self, index: usize) -> Option<&NoiseTables> {
        self.tables.get(index).and_then(Option::as_ref)
    }

    pub(crate) fn holes_reach(&self, min: Vec2, max: Vec2) -> bool {
        self.layers.iter().any(|layer| match layer {
            Layer::Stamp(
                stamp @ Stamp {
                    op: StampOp::Hole, ..
                },
            ) => reaches(stamp.center, stamp.radius, min, max),
            Layer::Path(PathStage {
                baked: Some(profile),
                ..
            }) => profile.portal_bounds().is_some_and(|(low, high)| {
                low.x <= max.x && low.y <= max.y && high.x >= min.x && high.y >= min.y
            }),
            Layer::Noise(_)
            | Layer::Heightmap(_)
            | Layer::Stamp(_)
            | Layer::Rivers(_)
            | Layer::Hydrology(_)
            | Layer::Erosion(_)
            | Layer::Path(_)
            | Layer::Steps(_)
            | Layer::Hummocks(_) => false,
        })
    }

    pub(crate) fn hole_at(&self, x: f32, z: f32) -> bool {
        self.layers.iter().any(|layer| match layer {
            Layer::Stamp(
                stamp @ Stamp {
                    op: StampOp::Hole, ..
                },
            ) => {
                let dx = x - stamp.center[0];
                let dz = z - stamp.center[1];
                dx * dx + dz * dz < stamp.radius * stamp.radius
            }
            Layer::Path(PathStage {
                baked: Some(profile),
                ..
            }) => profile.portal_hole(Vec2::new(x, z)),
            Layer::Noise(_)
            | Layer::Heightmap(_)
            | Layer::Stamp(_)
            | Layer::Rivers(_)
            | Layer::Hydrology(_)
            | Layer::Erosion(_)
            | Layer::Path(_)
            | Layer::Steps(_)
            | Layer::Hummocks(_) => false,
        })
    }

    pub(crate) fn apply_local(
        &self,
        index: usize,
        row: &mut [f32],
        xs: &[f32],
        z: f32,
        (min, max): (Vec2, Vec2),
        spacing: f32,
    ) {
        match &self.layers[index] {
            Layer::Noise(_) | Layer::Rivers(_) => {}
            Layer::Path(stage) => {
                let Some(profile) = &stage.baked else {
                    return;
                };
                let (low, high) = profile.bounds();
                if low.x > max.x || low.y > max.y || high.x < min.x || high.y < min.y {
                    return;
                }
                for (height, &x) in row.iter_mut().zip(xs) {
                    if let Some((distance, road, false)) = profile.nearest(Vec2::new(x, z)) {
                        *height += (road - *height) * profile.weight(distance) * self.shown(x, z);
                    }
                }
            }
            Layer::Erosion(stage) => {
                let Some(baked) = &stage.baked else {
                    return;
                };
                let below = row.to_vec();
                baked.wear_row(xs, z, spacing, &below, |index, worn| {
                    row[index] += worn * self.shown(xs[index], z);
                });
            }
            Layer::Hydrology(stage) => {
                let Some(baked) = &stage.baked else {
                    return;
                };
                if baked.rivers.is_empty() {
                    return;
                }
                for (height, &x) in row.iter_mut().zip(xs) {
                    let cut = baked.rivers.cut(x, z, *height, spacing);
                    *height = self.faded(*height, cut, x, z);
                }
            }
            Layer::Stamp(stamp) => {
                if !reaches(stamp.center, stamp.reach(), min, max) {
                    return;
                }
                for (height, &x) in row.iter_mut().zip(xs) {
                    let weight = stamp.weight(x, z) * self.shown(x, z);
                    *height = match stamp.op {
                        StampOp::Raise(amount) => *height + amount * weight,
                        StampOp::Lower(amount) => *height - amount * weight,
                        StampOp::Flatten(level) => *height + (level - *height) * weight,
                        StampOp::Hole => *height,
                    };
                }
            }
            Layer::Heightmap(map) => {
                let map_max = map.min + map.size;
                if map.min.x > max.x || map.min.y > max.y || map_max.x < min.x || map_max.y < min.y
                {
                    return;
                }
                let level = map.level_for(spacing);
                for (height, &x) in row.iter_mut().zip(xs) {
                    let (value, weight) = map.at(x, z, level);
                    *height = map.blend.apply(*height, value, weight * self.shown(x, z));
                }
            }
            Layer::Steps(steps) => steps.apply_row(self, (row, xs, z), spacing),
            Layer::Hummocks(hummocks) => hummocks.apply_row(self, (row, xs, z), spacing),
        }
    }

    pub(crate) fn cut_rivers(
        &self,
        raster: &crate::rivers::RiverRaster,
        at: usize,
        row: &mut [f32],
        xs: &[f32],
        z: f32,
    ) {
        for (column, (height, &x)) in row.iter_mut().zip(xs).enumerate() {
            let cut = raster.cut(at, column, *height);
            *height = self.faded(*height, cut, x, z);
        }
    }

    fn faded(&self, height: f32, changed: f32, x: f32, z: f32) -> f32 {
        let shown = self.shown(x, z);
        if shown >= 1.0 {
            changed
        } else {
            height + (changed - height) * shown
        }
    }

    pub(crate) fn apply_noise(&self, index: usize, row: &mut [f32], values: &[f32]) {
        if let Layer::Noise(noise) = &self.layers[index] {
            let blend = noise.blend;
            for (height, &value) in row.iter_mut().zip(values) {
                *height = blend.apply(*height, value, 1.0);
            }
        }
    }
}

fn unbaked(layer: &Layer) -> Layer {
    match layer {
        Layer::Hydrology(stage) => Layer::Hydrology(HydrologyStage {
            layer: stage.layer,
            baked: None,
        }),
        Layer::Erosion(stage) => Layer::Erosion(ErosionStage {
            layer: stage.layer,
            baked: None,
        }),
        Layer::Path(stage) => Layer::Path(PathStage {
            layer: stage.layer.clone(),
            baked: None,
        }),
        Layer::Noise(_)
        | Layer::Heightmap(_)
        | Layer::Stamp(_)
        | Layer::Rivers(_)
        | Layer::Steps(_)
        | Layer::Hummocks(_) => layer.clone(),
    }
}

fn reaches(center: [f32; 2], radius: f32, min: Vec2, max: Vec2) -> bool {
    let nearest = Vec2::from(center).clamp(min, max);
    nearest.distance_squared(Vec2::from(center)) <= radius * radius
}

use crate::codec::{Codec, CodecError, Reader, Writer, record, tagged};

tagged!(Blend, "invalid blend" {
    0 => Add,
    1 => Max,
    2 => Min,
    3 => Replace,
});

record!(NoiseLayer {
    seed,
    height,
    offset,
    wavelength,
    ridges,
    warp,
    erosion,
    smoothing,
    valleys,
    roughness,
    blend,
});

tagged!(StampOp, "invalid stamp" {
    0 => Raise(meters),
    1 => Lower(meters),
    2 => Flatten(height),
    3 => Hole,
});

record!(Stamp {
    center,
    radius,
    falloff,
    op,
});
record!(HeightImage {
    width,
    height,
    samples,
    levels,
});
record!(ImageLevel {
    width,
    height,
    samples,
});
record!(HeightmapLayer {
    image,
    min,
    size,
    low,
    high,
    falloff,
    blend,
    uri,
    source,
    attribution,
});
record!(PathStage { layer, baked });
record!(ErosionStage { layer, baked });
record!(HydrologyStage { layer, baked });
record!(Fade { low, high, blend });

tagged!(Layer, "invalid layer" {
    0 => Noise(layer),
    1 => Heightmap(layer),
    2 => Stamp(layer),
    3 => Hydrology(stage),
    4 => Erosion(stage),
    5 => Path(stage),
    6 => Rivers(layer),
    7 => Steps(layer),
    8 => Hummocks(layer),
});

impl Codec for Layers {
    fn write(&self, out: &mut Writer) {
        out.put(&self.layers).put(&self.offset).put(&self.fade);
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        let layers = input.get()?;
        let offset = input.get()?;
        let fade = input.get()?;
        let mut stack = Self::new(layers).map_err(|_| CodecError("invalid layers"))?;
        stack.offset = offset;
        stack.fade = fade;
        Ok(stack)
    }
}
