//! The layer stack: procedural noise with erosion, imported heightmaps,
//! stamps, what an erosion bake wears away (decision 068), the channels a
//! hydrology bake carves (decision 066) and roads (decision 069), applied
//! in order to every sample; later layers win.

use std::sync::Arc;

use engine_math::{clamp, max, min};
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::erosion::{Erosion, ErosionLayer};
use crate::hydrology::{Hydrology, HydrologyLayer};
use crate::noise::{gradient_noise, gradient_value, gullies, smooth};
use crate::path::{MAX_PATH_POINTS, PathLayer, PathProfile, PathSample};

/// How a layer's heights combine with the heights of the layers before it,
/// weighted by the layer's coverage: added, the higher or lower of the two,
/// or replacing them.
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

/// Procedural ground: fractal gradient noise whose fine octaves fade on
/// slopes (Quilez's derivative damping), optionally folded into ridges and
/// warped sideways, shaped toward wide valley floors, then cut by the
/// erosion filter's gullies running down its slopes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct NoiseLayer {
    /// Picks the pattern: the same seed gives the same heights everywhere.
    pub seed: u32,
    /// Meters from the lowest to the highest ground before gullies, 0 to
    /// 20000.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 20000.0)))]
    pub height: f32,
    /// Meters added to every height of the layer, -20000 to 20000.
    #[cfg_attr(feature = "schema", schemars(range(min = -20000.0, max = 20000.0)))]
    pub offset: f32,
    /// Meters across the largest features, 16 to 1000000.
    #[cfg_attr(feature = "schema", schemars(range(min = 16.0, max = 1000000.0)))]
    pub wavelength: f32,
    /// 0 rolls; 1 folds every octave into sharp ridges.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub ridges: f32,
    /// How far the ground warps sideways, as a share of the wavelength, 0
    /// to 1.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub warp: f32,
    /// How deep gullies cut down slopes, 0 (none) to 1.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub erosion: f32,
    /// How much fine detail fades on slopes, 0 to 1.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub smoothing: f32,
    /// 0 keeps the heights' spread; 1 widens valley floors and sharpens
    /// peaks.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub valleys: f32,
    /// How much each finer octave keeps of the coarser one's height, from
    /// 0 (smooth slopes up close) through 0.5 (every scale alike) to 1
    /// (rugged).
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub roughness: f32,
    pub blend: Blend,
}

impl Default for NoiseLayer {
    fn default() -> Self {
        Self {
            seed: 0,
            height: 900.0,
            offset: 0.0,
            wavelength: 3000.0,
            ridges: 0.3,
            warp: 0.35,
            erosion: 0.7,
            smoothing: 1.0,
            valleys: 0.7,
            roughness: 0.5,
            blend: Blend::Add,
        }
    }
}

/// What a stamp does where it reaches: raise or lower heights by meters,
/// level them toward a height, or cut holes in the terrain.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum StampOp {
    Raise(f32),
    Lower(f32),
    Flatten(f32),
    /// Cells whose centers lie within the radius are holes: no surface
    /// draws or collides there.
    Hole,
}

/// A local edit: full strength within `radius` meters of `center`, fading
/// out over `falloff` meters beyond.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Stamp {
    /// Terrain-local meters along x and z.
    pub center: [f32; 2],
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub radius: f32,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub falloff: f32,
    pub op: StampOp,
}

impl Stamp {
    /// The stamp's strength at terrain-local `(x, z)`, 0 to 1.
    #[inline]
    fn weight(&self, x: f32, z: f32) -> f32 {
        let dx = x - self.center[0];
        let dz = z - self.center[1];
        let distance = (dx * dx + dz * dz).sqrt();
        1.0 - smooth((distance - self.radius) / max(self.falloff, 1.0e-6))
    }

    /// Meters from the center beyond which the stamp has no effect.
    fn reach(&self) -> f32 {
        self.radius + self.falloff
    }
}

/// 16-bit height samples of an imported heightmap, row by row along z,
/// with coarser levels for coarser terrain.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightImage {
    width: u32,
    height: u32,
    samples: Box<[u16]>,
    levels: Vec<ImageLevel>,
}

/// A coarser level of a [`HeightImage`]: every other sample of the level
/// before, filtered `[1, 2, 1] / 4` along each axis, so terrain levels whose
/// samples lie further apart than the image's read it without aliasing.
#[derive(Clone, Debug, PartialEq)]
struct ImageLevel {
    width: u32,
    height: u32,
    samples: Box<[f32]>,
}

impl ImageLevel {
    /// The level after one `width` by `height` holding `samples`.
    fn below(width: u32, height: u32, samples: &[f32]) -> Self {
        let half = |size: u32| (size - 1).div_ceil(2) + 1;
        let (next_width, next_height) = (half(width), half(height));
        let at = |x: i64, z: i64| {
            let x = x.clamp(0, i64::from(width) - 1) as usize;
            let z = z.clamp(0, i64::from(height) - 1) as usize;
            samples[z * width as usize + x]
        };
        // Along x first, then along z, each `[1, 2, 1] / 4`.
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
    /// `samples` holds `width * height` values, row by row; at least two
    /// along each axis.
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

    /// Little-endian 16-bit samples (`.r16` files), row by row.
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

    /// How many coarser levels follow the image's own samples.
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Bilinear interpolation at image coordinates `(u, v)` in the image's
    /// own samples, 0 to 65535, edges clamped, read from `level` (0 the
    /// image's samples, each one after twice as far apart): it never
    /// overshoots its samples, as cubic interpolation does beside sharp
    /// steps.
    #[inline]
    pub(crate) fn sample_linear(&self, u: f32, v: f32, level: usize) -> f32 {
        let (scale, width, height) = match level
            .checked_sub(1)
            .and_then(|index| self.levels.get(index))
        {
            // Exact: a power of two.
            Some(coarse) => (1.0 / (1_u32 << level) as f32, coarse.width, coarse.height),
            None => (1.0, self.width, self.height),
        };
        let texel = |x: i32, z: i32| {
            let x = x.clamp(0, width as i32 - 1) as usize;
            let z = z.clamp(0, height as i32 - 1) as usize;
            match level
                .checked_sub(1)
                .and_then(|index| self.levels.get(index))
            {
                Some(coarse) => coarse.samples[z * width as usize + x],
                None => f32::from(self.samples[z * width as usize + x]),
            }
        };
        let (u, v) = (u * scale, v * scale);
        let (u0, v0) = (u.floor(), v.floor());
        let (x, z) = (u0 as i32, v0 as i32);
        let (fx, fz) = (u - u0, v - v0);
        let low = texel(x, z) + (texel(x + 1, z) - texel(x, z)) * fx;
        let high = texel(x, z + 1) + (texel(x + 1, z + 1) - texel(x, z + 1)) * fx;
        low + (high - low) * fz
    }

    /// Catmull-Rom interpolation at image coordinates `(u, v)` in the
    /// image's own samples, 0 to 65535, edges clamped, read from `level`
    /// (0 the image's samples, each one after twice as far apart).
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
                // Exact: a power of two.
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

/// Catmull-Rom interpolation at `(u, v)` of the samples `texel` reads.
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

/// Catmull-Rom weights of the four samples around fraction `t`.
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

/// An imported heightmap placed over the terrain: sample values 0 to
/// 65535 map to heights `low` to `high`, its samples span `size` meters from
/// `min`, and it fades in over `falloff` meters inside its edges.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightmapLayer {
    pub image: Arc<HeightImage>,
    pub min: Vec2,
    pub size: Vec2,
    pub low: f32,
    pub high: f32,
    pub falloff: f32,
    pub blend: Blend,
    /// The file the image was read from, for authoring tools.
    pub uri: Option<String>,
    /// Where its heights came from and the credit their data requires, as
    /// the import tool records them (decision 070).
    pub source: Option<String>,
    pub attribution: Option<String>,
}

impl HeightmapLayer {
    /// The image level terrain samples `spacing` meters apart read: the
    /// coarsest whose samples lie no further apart than the terrain's, so
    /// neighboring terrain levels differ only by the detail the coarser one
    /// leaves out.
    fn level_for(&self, spacing: f32) -> usize {
        let image = &self.image;
        let texel = max(
            self.size.x / (image.width - 1) as f32,
            self.size.y / (image.height - 1) as f32,
        );
        let mut level = 0;
        while level < image.levels() && texel * (2_u32 << level) as f32 <= spacing * 1.0001 {
            level += 1;
        }
        level
    }

    /// The layer's height and coverage at terrain-local `(x, z)`, from
    /// image level `level`.
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

/// One layer of a [`Layers`] stack.
#[derive(Clone, Debug, PartialEq)]
pub enum Layer {
    Noise(NoiseLayer),
    Heightmap(HeightmapLayer),
    Stamp(Stamp),
    /// Where water runs over the layers before it and the channels it
    /// carves (decision 066): nothing until baked.
    Hydrology(HydrologyStage),
    /// What rivers and creep wear away from the layers before it (decision
    /// 068): nothing until baked.
    Erosion(ErosionStage),
    /// A road leveling the layers before it to its profile (decision 069):
    /// nothing until baked.
    Path(PathStage),
}

/// A path layer, and its profile once baked.
#[derive(Clone, Debug, PartialEq)]
pub struct PathStage {
    pub layer: Arc<PathLayer>,
    pub baked: Option<Arc<PathProfile>>,
}

/// An erosion layer, and what its bake did once baked.
#[derive(Clone, Debug, PartialEq)]
pub struct ErosionStage {
    pub layer: ErosionLayer,
    pub baked: Option<Arc<Erosion>>,
}

/// A hydrology layer, and what its bake found once baked.
#[derive(Clone, Debug, PartialEq)]
pub struct HydrologyStage {
    pub layer: HydrologyLayer,
    pub baked: Option<Arc<Hydrology>>,
}

/// A layer's field outside its valid values: the layer's index, the field
/// and the range it must be finite within.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerError {
    pub layer: usize,
    pub field: &'static str,
    pub low: f32,
    pub high: f32,
}

impl std::fmt::Display for LayerError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "layer {} {} must be in [{}, {}]",
            self.layer, self.field, self.low, self.high
        )
    }
}

impl std::error::Error for LayerError {}

/// An octave's lattice: lattice coordinates `m * p + t` of terrain-local
/// point `p`, its amplitude, wavelength in meters and seed.
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
}

/// A gully octave: its cell size and depth in meters, and seed.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GullyOctave {
    cell: f32,
    depth: f32,
    seed: u32,
}

/// The most octaves a noise layer sums; finer ones fall below any level's
/// samples for wavelengths up to the largest allowed.
const OCTAVES: usize = 28;
const GULLY_OCTAVES: usize = 5;

/// What a point of the coarse grid holds of a noise layer, which samples
/// between grid points interpolate: the warp's offsets, the gullies'
/// heights, the fractal sum and the damping's lattice slope after the
/// coarse octaves, all smooth over the grid, then each coarse octave's
/// noise, from which every sample sums the ridged octaves itself: their
/// creases and weights would not interpolate. Levels so differ only by the
/// octaves they leave out.
pub(crate) const STATE_HEAD: usize = 6;
pub(crate) const STATE_OCTAVE: usize = 1;

/// The values a coarse state of `octaves` coarse octaves holds.
pub(crate) const fn state_len(octaves: usize) -> usize {
    STATE_HEAD + STATE_OCTAVE * octaves
}

/// A noise layer's sums as octaves add to them: fractal noise whose
/// octaves fade where the ground already slopes (Quilez) and Musgrave's
/// ridged multifractal, whose octaves are weighted by the ridge below them.
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

    /// Adds `octave`'s noise `n` with lattice gradient `(nx, nz)`;
    /// returns the fractal octave's damped amplitude and the ridged
    /// octave's crest times its weight, which their gradients scale by.
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

    /// Adds `octave`'s ridged term alone, of noise `n`; returns its crest
    /// times its weight.
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

/// A noise layer with its octaves laid out.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NoiseTables {
    params: NoiseLayer,
    octaves: [Octave; OCTAVES],
    gullies: [GullyOctave; GULLY_OCTAVES],
}

/// Octaves at least this many samples long are interpolated from the
/// coarse grid; shorter ones, down to [`FINEST_SAMPLES`], are evaluated at
/// every sample. Interpolated octaves span eight grid points and more, so
/// their noise and the fractal sum, whose damping adds up to three times
/// their frequency, interpolate closely, and neighboring levels, whose
/// grids lie twice as far apart, agree on the samples they share.
pub(crate) const COARSE_SAMPLES: f32 = 32.0;
pub(crate) const FINEST_SAMPLES: f32 = 4.0;
/// Gully octaves whose cells span at least this many samples are cut, from
/// the coarse grid: gully `k` follows octaves up to `k + 1`, four of its
/// cells long, which the grid then holds.
pub(crate) const GULLY_SAMPLES: f32 = 16.0;
/// How far from a ridged octave's crease, in a gully octave's cells, the
/// slope that gully runs across turns over where the ridged slope reverses:
/// gullies bend around crests within it instead of tearing the ground
/// there, and keep their course everywhere else.
const CREASE: f32 = 1.0;

impl NoiseTables {
    fn new(params: NoiseLayer) -> Self {
        // Each octave doubles the frequency and turns the lattice by
        // atan(3/4), so octaves never line up.
        let (cos, sin) = (0.8_f64, 0.6_f64);
        // Octave amplitudes fall by `gain` and sum to one; plain products,
        // the same bits on every target.
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

    /// Octaves interpolated from the coarse grid and octaves evaluated per
    /// sample at sample spacing `spacing`.
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

    /// How far point `(x, z)` warps sideways.
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

    /// The coarse state at `(x, z)` into `state` ([`state_len`] of
    /// `octaves` values): the first `octaves` octaves at the warped point,
    /// and `gully_octaves` gully octaves, gully `k` running across the
    /// slope of octaves 0 to `k + 1` (four of its cells long and more,
    /// their ridged creases rounded by [`CREASE`]), whatever the level.
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
        // The octaves the gullies follow, per meter: the fractal octave's
        // gradient, the ridged octave's before its sign, the noise and its
        // squared gradient.
        let mut terms = [[0.0_f32; 6]; GULLY_OCTAVES + 1];
        // The blended sum after octaves 1 to GULLY_OCTAVES.
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
            // The slope of octaves 0 to k + 1, each ridged octave's sign
            // turning over CREASE cells about its crease (its noise over
            // its gradient is the distance there).
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
            // The slope of the shaped octaves it follows.
            let h = clamp(mixed, 0.0, 1.0);
            let shape_slope = params.height * (1.0 + params.valleys * (2.0 * h - 1.0));
            let (gx, gz) = (shape_slope * mx, shape_slope * mz);
            let slope = (gx * gx + gz * gz).sqrt();
            let inverse = 1.0 / max(slope, 1.0e-6);
            let across = [-gz * inverse, gx * inverse];
            let [c, _] = gullies(px / octave.cell, pz / octave.cell, across, octave.seed);
            // Gullies cut slopes, not flats.
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

    /// The height at `(x, z)` from the coarse `state` there, interpolated:
    /// its `octaves` coarse octaves' ridged terms, then the fine octaves
    /// `fine`, sum exactly as at grid points, then valley floors widen and
    /// gullies cut in.
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
        // Widen valley floors.
        let h = clamp(blend(sums.rolling, sums.ridged, params.ridges), 0.0, 1.0);
        let shaped = h + params.valleys * (h * h - h);
        params.offset + params.height * shaped + state[2]
    }
}

/// The fractal and ridged sums blended by `ridges`, both in about 0..1.
#[inline(always)]
fn blend(rolling: f32, ridged: f32, ridges: f32) -> f32 {
    0.5 + 0.8 * rolling + ridges * (ridged - 0.5 - 0.8 * rolling)
}

/// The gradients of the sums [`blend`] blends, blended alike.
#[inline(always)]
fn blend_gradient(rolling: f32, ridged: f32, ridges: f32) -> f32 {
    0.8 * rolling + ridges * (ridged - 0.8 * rolling)
}

/// A validated stack of layers, applied in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Layers {
    layers: Vec<Layer>,
    tables: Vec<Option<NoiseTables>>,
}

/// Checks stamp layer `index`'s center, reach and operation.
fn check_stamp(index: usize, stamp: &Stamp) -> Result<(), LayerError> {
    let limit = Layers::MAX_COORDINATE;
    check(index, "center", stamp.center[0], -limit, limit)?;
    check(index, "center", stamp.center[1], -limit, limit)?;
    check(index, "radius", stamp.radius, 0.0, limit)?;
    check(index, "falloff", stamp.falloff, 0.0, limit)?;
    match stamp.op {
        StampOp::Raise(h) | StampOp::Lower(h) => check(index, "op", h, 0.0, 20_000.0),
        StampOp::Flatten(h) => check(index, "op", h, -20_000.0, 20_000.0),
        StampOp::Hole => Ok(()),
    }
}

fn check(
    layer: usize,
    field: &'static str,
    value: f32,
    low: f32,
    high: f32,
) -> Result<(), LayerError> {
    if value.is_finite() && (low..=high).contains(&value) {
        Ok(())
    } else {
        Err(LayerError {
            layer,
            field,
            low,
            high,
        })
    }
}

impl Layers {
    /// The largest coordinate magnitude, in meters, of anything a layer
    /// places.
    pub const MAX_COORDINATE: f32 = 1.0e7;

    pub fn new(layers: Vec<Layer>) -> Result<Self, LayerError> {
        let limit = Self::MAX_COORDINATE;
        for (index, layer) in layers.iter().enumerate() {
            match layer {
                Layer::Noise(noise) => {
                    check(index, "height", noise.height, 0.0, 20_000.0)?;
                    check(index, "offset", noise.offset, -20_000.0, 20_000.0)?;
                    check(index, "wavelength", noise.wavelength, 16.0, 1.0e6)?;
                    check(index, "ridges", noise.ridges, 0.0, 1.0)?;
                    check(index, "warp", noise.warp, 0.0, 1.0)?;
                    check(index, "erosion", noise.erosion, 0.0, 1.0)?;
                    check(index, "smoothing", noise.smoothing, 0.0, 1.0)?;
                    check(index, "valleys", noise.valleys, 0.0, 1.0)?;
                    check(index, "roughness", noise.roughness, 0.0, 1.0)?;
                }
                Layer::Heightmap(map) => {
                    check(index, "min", map.min.x, -limit, limit)?;
                    check(index, "min", map.min.y, -limit, limit)?;
                    check(index, "size", map.size.x, 1.0e-3, limit)?;
                    check(index, "size", map.size.y, 1.0e-3, limit)?;
                    check(index, "low", map.low, -20_000.0, 20_000.0)?;
                    check(index, "high", map.high, -20_000.0, 20_000.0)?;
                    check(index, "falloff", map.falloff, 0.0, limit)?;
                }
                Layer::Stamp(stamp) => check_stamp(index, stamp)?,
                Layer::Erosion(stage) => {
                    let erosion = &stage.layer;
                    check(index, "level", erosion.level as f32, 0.0, 15.0)?;
                    check(index, "rounds", erosion.rounds as f32, 1.0, 400.0)?;
                    check(index, "incision", erosion.incision, 0.0, 1.0)?;
                    check(index, "creep", erosion.creep, 0.0, 1000.0)?;
                }
                Layer::Path(stage) => {
                    let path = &stage.layer;
                    check(
                        index,
                        "points",
                        path.points.len() as f32,
                        2.0,
                        MAX_PATH_POINTS as f32,
                    )?;
                    for point in &path.points {
                        check(index, "points", point[0], -limit, limit)?;
                        check(index, "points", point[1], -limit, limit)?;
                    }
                    check(index, "width", path.width, 0.5, 200.0)?;
                    check(index, "shoulder", path.shoulder, 0.0, 500.0)?;
                    check(index, "smoothing", path.smoothing, 0.0, 5000.0)?;
                    check(index, "bridge_fill", path.bridge_fill, 0.0, 1000.0)?;
                    check(index, "tunnel_cut", path.tunnel_cut, 0.0, 2000.0)?;
                }
                Layer::Hydrology(stage) => {
                    let water = &stage.layer;
                    check(index, "level", water.level as f32, 0.0, 15.0)?;
                    check(index, "channel_area", water.channel_area, 100.0, 1.0e10)?;
                    check(index, "depth", water.depth, 0.0, 100.0)?;
                    check(index, "width", water.width, 0.1, 1000.0)?;
                    check(index, "max_depth", water.max_depth, 0.0, 100.0)?;
                    check(index, "max_width", water.max_width, 0.1, 1000.0)?;
                }
            }
        }
        let tables = layers
            .iter()
            .map(|layer| match layer {
                Layer::Noise(noise) => Some(NoiseTables::new(*noise)),
                _ => None,
            })
            .collect();
        Ok(Self { layers, tables })
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// These layers with `stamps` over them (decision 080): the same layers,
    /// bakes and noise tables, and the stamps validated as [`Self::new`]
    /// validates them.
    pub fn with_stamps(&self, stamps: &[Stamp]) -> Result<Self, LayerError> {
        let mut layers = self.layers.clone();
        let mut tables = self.tables.clone();
        for (offset, stamp) in stamps.iter().enumerate() {
            check_stamp(self.layers.len() + offset, stamp)?;
            layers.push(Layer::Stamp(*stamp));
            tables.push(None);
        }
        Ok(Self { layers, tables })
    }

    /// Layer `index`, to set what its bake found.
    pub(crate) fn layer_mut(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    /// The stack of the layers before `index`, which a bake there reads.
    pub fn below(&self, index: usize) -> Self {
        Self {
            layers: self.layers[..index].to_vec(),
            tables: self.tables[..index].to_vec(),
        }
    }

    /// The first baked hydrology, which masks read.
    pub fn hydrology(&self) -> Option<&Hydrology> {
        self.layers.iter().find_map(|layer| match layer {
            Layer::Hydrology(HydrologyStage {
                baked: Some(baked), ..
            }) => Some(baked.as_ref()),
            _ => None,
        })
    }

    /// The baked road whose edge lies nearest terrain-local `(x, z)`: meters
    /// from its line and half its width; far beyond every path's masks'
    /// reach.
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
                _ => nearest,
            })
    }

    pub(crate) fn noise(&self, index: usize) -> Option<&NoiseTables> {
        self.tables.get(index).and_then(Option::as_ref)
    }

    /// Whether a hole stamp or a tunnel's portal (decision 073) reaches the
    /// square from `min` to `max`.
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
            _ => false,
        })
    }

    /// Whether cell center `(x, z)` is a hole.
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
            _ => false,
        })
    }

    /// Applies layer `index` other than noise to `row`, heights at
    /// terrain-local x positions `xs` and z position `z`; layers that reach
    /// no point between `min` and `max` are skipped.
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
            Layer::Noise(_) => {}
            Layer::Path(stage) => {
                let Some(profile) = &stage.baked else {
                    return;
                };
                let (low, high) = profile.bounds();
                if low.x > max.x || low.y > max.y || high.x < min.x || high.y < min.y {
                    return;
                }
                for (height, &x) in row.iter_mut().zip(xs) {
                    // Bridges (decision 072) leave the ground under them.
                    if let Some((distance, road, false)) = profile.nearest(Vec2::new(x, z)) {
                        *height += (road - *height) * profile.weight(distance);
                    }
                }
            }
            Layer::Erosion(stage) => {
                let Some(baked) = &stage.baked else {
                    return;
                };
                for (height, &x) in row.iter_mut().zip(xs) {
                    *height += baked.change_at(x, z, spacing);
                }
            }
            Layer::Hydrology(stage) => {
                let Some(baked) = &stage.baked else {
                    return;
                };
                let level = baked.level_for(spacing);
                for (height, &x) in row.iter_mut().zip(xs) {
                    *height -= baked.carve_at(x, z, level);
                }
            }
            Layer::Stamp(stamp) => {
                if !reaches(stamp.center, stamp.reach(), min, max) {
                    return;
                }
                for (height, &x) in row.iter_mut().zip(xs) {
                    let weight = stamp.weight(x, z);
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
                    *height = map.blend.apply(*height, value, weight);
                }
            }
        }
    }

    /// Combines noise layer `index`'s heights `values` into `row`.
    pub(crate) fn apply_noise(&self, index: usize, row: &mut [f32], values: &[f32]) {
        if let Layer::Noise(noise) = &self.layers[index] {
            let blend = noise.blend;
            for (height, &value) in row.iter_mut().zip(values) {
                *height = blend.apply(*height, value, 1.0);
            }
        }
    }
}

/// Whether the disk of `radius` about `center` reaches the square from
/// `min` to `max`.
fn reaches(center: [f32; 2], radius: f32, min: Vec2, max: Vec2) -> bool {
    let nearest = Vec2::from(center).clamp(min, max);
    nearest.distance_squared(Vec2::from(center)) <= radius * radius
}
