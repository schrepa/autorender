//! The layer stack: procedural noise with erosion, imported heightmaps,
//! stamps, what an erosion bake wears away, the channels a
//! hydrology bake carves and roads, applied
//! in order to every sample; later layers win.

use std::sync::Arc;

use engine_math::{clamp, max, min};
use engine_settings::{InvalidSetting, Rule, Setting, settings};
use glam::{Vec2, Vec4};
use serde::{Deserialize, Serialize};

use crate::bake::{ErosionBake, HydrologyBake, PathBake, Stage};
use crate::hydrology::Hydrology;
use crate::noise::{
    gradient_noise, gradient_noise4, gradient_value, gradient_value4, gullies, gullies4, max4,
    min4, smooth, smooth4, sqrt4,
};
use crate::path::PathSample;

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
    /// Procedural ground: fractal gradient noise whose fine octaves fade on
    /// slopes (Quilez's derivative damping), optionally folded into ridges and
    /// warped sideways, shaped toward wide valley floors, then cut by the
    /// erosion filter's gullies running down its slopes.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct NoiseLayer {
        /// Picks the pattern: the same seed gives the same heights everywhere.
        pub seed: u32 = 0 => any,
        /// Meters from the lowest to the highest ground before gullies.
        pub height: f32 = 900.0 => range(0.0, MAX_HEIGHT),
        /// Meters added to every height of the layer.
        pub offset: f32 = 0.0 => range(-MAX_HEIGHT, MAX_HEIGHT),
        /// Meters across the largest features.
        pub wavelength: f32 = 3000.0 => range(16.0, 1.0e6),
        /// 0 rolls; 1 folds every octave into sharp ridges.
        pub ridges: f32 = 0.3 => range(0.0, 1.0),
        /// How far the ground warps sideways, as a share of the wavelength.
        pub warp: f32 = 0.35 => range(0.0, 1.0),
        /// How deep gullies cut down slopes: 0 none.
        pub erosion: f32 = 0.7 => range(0.0, 1.0),
        /// How much fine detail fades on slopes.
        pub smoothing: f32 = 1.0 => range(0.0, 1.0),
        /// 0 keeps the heights' spread; 1 widens valley floors and sharpens
        /// peaks.
        pub valleys: f32 = 0.7 => range(0.0, 1.0),
        /// How much each finer octave keeps of the coarser one's height, from
        /// 0 (smooth slopes up close) through 0.5 (every scale alike) to 1
        /// (rugged).
        pub roughness: f32 = 0.5 => range(0.0, 1.0),
        /// How the layer combines with the layers below it.
        pub blend: Blend = Blend::Add => any,
    }
}

/// The most meters a layer raises, lowers or offsets the ground.
pub const MAX_HEIGHT: f32 = 20_000.0;

/// What a stamp does where it reaches: raise or lower heights by meters,
/// level them toward a height, or cut holes in the terrain.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum StampOp {
    Raise(#[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = MAX_HEIGHT)))] f32),
    Lower(#[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = MAX_HEIGHT)))] f32),
    Flatten(
        #[cfg_attr(feature = "schema", schemars(range(min = -MAX_HEIGHT, max = MAX_HEIGHT)))] f32,
    ),
    /// Cells whose centers lie within the radius are holes: no surface
    /// draws or collides there.
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
    /// A local edit: full strength within `radius` meters of `center`, fading
    /// out over `falloff` meters beyond.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct Stamp {
        /// Terrain-local meters along x and z.
        pub center: [f32; 2] = [0.0; 2] => range(-Layers::MAX_COORDINATE, Layers::MAX_COORDINATE),
        /// Meters around the center at full strength.
        pub radius: f32 = 1.0 => range(0.0, Layers::MAX_COORDINATE),
        /// Meters beyond the radius over which it fades out.
        #[serde(default)]
        pub falloff: f32 = 0.0 => range(0.0, Layers::MAX_COORDINATE),
        /// What it does where it reaches.
        pub op: StampOp = StampOp::Hole => any,
    }
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
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: Box<[u16]>,
    pub(crate) levels: Vec<ImageLevel>,
}

/// A coarser level of a [`HeightImage`]: every other sample of the level
/// before, filtered `[1, 2, 1] / 4` along each axis, so terrain levels whose
/// samples lie further apart than the image's read it without aliasing.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ImageLevel {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: Box<[f32]>,
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
    /// the import tool records them.
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
    /// carves: nothing until baked.
    Hydrology(HydrologyStage),
    /// What rivers and creep wear away from the layers before it:
    /// nothing until baked.
    Erosion(ErosionStage),
    /// A road leveling the layers before it to its profile:
    /// nothing until baked.
    Path(PathStage),
}

/// A path layer, and its profile once baked.
pub type PathStage = Stage<PathBake>;

/// An erosion layer, and what its bake did once baked.
pub type ErosionStage = Stage<ErosionBake>;

/// A hydrology layer, and what its bake found once baked.
pub type HydrologyStage = Stage<HydrologyBake>;

/// A layer's field breaking its rule: the layer's index, the field below
/// the layer (its kind first, as authored: `noise.height`) and the rule.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerError {
    pub layer: usize,
    pub field: std::borrow::Cow<'static, str>,
    pub rule: std::borrow::Cow<'static, str>,
}

impl LayerError {
    /// The error of layer `layer`, of kind `kind`, breaking `invalid`.
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

    /// [`Self::lattice`] of four points.
    #[inline(always)]
    fn lattice4(&self, x: Vec4, z: Vec4) -> (Vec4, Vec4) {
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

/// [`Sums`] of four samples, lane by lane its operations.
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

    /// [`Self::warp`] of four points.
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

    /// [`Self::coarse`] at four grid points of row `z`, lane
    /// by lane its operations: the same bits, into `out`, the four states
    /// one after another.
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

    /// [`Self::finish`] at four samples of row `z`, from
    /// their states value by value in lanes: the same bits.
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

/// [`blend`] of four samples.
#[inline(always)]
fn blend4(rolling: Vec4, ridged: Vec4, ridges: f32) -> Vec4 {
    0.5 + 0.8 * rolling + ridges * (ridged - 0.5 - 0.8 * rolling)
}

/// [`blend_gradient`] of four samples.
#[inline(always)]
fn blend_gradient4(rolling: Vec4, ridged: Vec4, ridges: f32) -> Vec4 {
    0.8 * rolling + ridges * (ridged - 0.8 * rolling)
}

/// A validated stack of layers, applied in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Layers {
    layers: Vec<Layer>,
    tables: Vec<Option<NoiseTables>>,
    // As a terrain's surroundings draw them: a point reads
    // what the stack holds `offset` meters further along x and z, and
    // every layer but noise fades out beyond `fade`'s region.
    pub(crate) offset: Vec2,
    pub(crate) fade: Option<Fade>,
}

/// Where a stack's layers other than noise show: all of
/// them from `low` to `high` (the stack's own ground), fading out over
/// `blend` meters beyond.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Fade {
    pub(crate) low: Vec2,
    pub(crate) high: Vec2,
    pub(crate) blend: f32,
}

/// Checks stamp layer `index`.
fn check_stamp(index: usize, stamp: &Stamp) -> Result<(), LayerError> {
    stamp
        .validate()
        .map_err(|invalid| LayerError::new(index, "stamp", invalid))
}

/// The rules of an imported heightmap's placement: its corner,
const HEIGHTMAP_MIN: Rule = Rule::Range(
    -Layers::MAX_COORDINATE as f64,
    Layers::MAX_COORDINATE as f64,
);
/// its size,
const HEIGHTMAP_SIZE: Rule = Rule::Range(1.0e-3, Layers::MAX_COORDINATE as f64);
/// the heights its samples span,
const HEIGHTMAP_HEIGHT: Rule = Rule::Range(-MAX_HEIGHT as f64, MAX_HEIGHT as f64);
/// and its falloff.
const HEIGHTMAP_FALLOFF: Rule = Rule::Range(0.0, Layers::MAX_COORDINATE as f64);

impl HeightmapLayer {
    /// The rule of each placement field, as authored heightmaps name it.
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
    /// The largest coordinate magnitude, in meters, of anything a layer
    /// places.
    pub const MAX_COORDINATE: f32 = 1.0e7;

    pub fn new(layers: Vec<Layer>) -> Result<Self, LayerError> {
        for (index, layer) in layers.iter().enumerate() {
            let (kind, checked) = match layer {
                Layer::Noise(noise) => ("noise", noise.validate()),
                Layer::Heightmap(map) => ("heightmap", map.validate()),
                Layer::Stamp(stamp) => ("stamp", stamp.validate()),
                Layer::Erosion(stage) => ("erosion", stage.layer.validate()),
                Layer::Path(stage) => ("path", stage.layer.validate()),
                Layer::Hydrology(stage) => ("hydrology", stage.layer.validate()),
            };
            checked.map_err(|invalid| LayerError::new(index, kind, invalid))?;
        }
        let tables = layers
            .iter()
            .map(|layer| match layer {
                Layer::Noise(noise) => Some(NoiseTables::new(*noise)),
                Layer::Heightmap(_)
                | Layer::Stamp(_)
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_) => None,
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

    /// Whether its tiles may hold holes: it stamps one, or a road may
    /// tunnel. Renderers prepare to draw them before any
    /// tile does.
    pub fn may_hole(&self) -> bool {
        self.layers.iter().any(|layer| match layer {
            Layer::Stamp(stamp) => matches!(stamp.op, StampOp::Hole),
            Layer::Path(stage) => stage.layer.tunnel_cut > 0.0,
            Layer::Noise(_) | Layer::Heightmap(_) | Layer::Hydrology(_) | Layer::Erosion(_) => {
                false
            }
        })
    }

    /// Whether a hydrology layer gives it water once baked.
    pub fn has_hydrology(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(layer, Layer::Hydrology(_)))
    }

    /// These layers as a terrain's surroundings draw them:
    /// a point reads what this stack holds `offset` meters further along x
    /// and z, and every layer but noise (stamps, heightmaps, bakes and
    /// roads, and the masks they feed) shows from `low` to `high` of this
    /// stack's ground, fading out over `blend` meters beyond, so the ground
    /// meets this stack's at that region's edges and turns into its noise
    /// away from them. Offsets keep the noise only as whole multiples of
    /// four samples of every level drawn.
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

    /// How much of the layers other than noise shows at `(x, z)` of this
    /// stack's ground: all of them without a fade.
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

    /// The samples `spacing` meters apart the stack's offset spans along x
    /// and z.
    pub(crate) fn offset_samples(&self, spacing: f32) -> [i64; 2] {
        [
            (self.offset.x / spacing).round() as i64,
            (self.offset.y / spacing).round() as i64,
        ]
    }

    /// These layers with `stamps` over them: the same layers,
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
        Ok(Self {
            layers,
            tables,
            offset: self.offset,
            fade: self.fade,
        })
    }

    /// Layer `index`, to set what its bake found.
    pub(crate) fn layer_mut(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    /// The stack of the layers before `index`, which a bake there reads.
    #[must_use]
    pub fn below(&self, index: usize) -> Self {
        Self {
            layers: self.layers[..index].to_vec(),
            tables: self.tables[..index].to_vec(),
            offset: self.offset,
            fade: self.fade,
        }
    }

    /// The first baked hydrology, which masks read.
    pub fn hydrology(&self) -> Option<&Hydrology> {
        self.hydrology_layer().map(|(_, hydrology)| hydrology)
    }

    /// [`Self::hydrology`] with its layer's index.
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
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_) => None,
            })
    }

    /// The rivers of hydrology layer `index`, once baked.
    pub(crate) fn rivers_of(&self, index: usize) -> Option<&crate::Rivers> {
        match self.layers.get(index) {
            Some(Layer::Hydrology(HydrologyStage {
                baked: Some(baked), ..
            })) if !baked.rivers.is_empty() => Some(&baked.rivers),
            _ => None,
        }
    }

    /// The baked road whose edge lies nearest terrain-local `(x, z)`: meters
    /// from its line and half its width; far beyond every path's masks'
    /// reach.
    /// Its roads' baked profiles, in layer order.
    pub(crate) fn path_profiles(&self) -> impl Iterator<Item = &crate::path::PathProfile> {
        self.layers.iter().filter_map(|layer| match layer {
            Layer::Path(PathStage {
                baked: Some(profile),
                ..
            }) => Some(profile.as_ref()),
            Layer::Noise(_)
            | Layer::Heightmap(_)
            | Layer::Stamp(_)
            | Layer::Hydrology(_)
            | Layer::Erosion(_)
            | Layer::Path(_) => None,
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
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_) => nearest,
            })
    }

    pub(crate) fn noise(&self, index: usize) -> Option<&NoiseTables> {
        self.tables.get(index).and_then(Option::as_ref)
    }

    /// Whether a hole stamp or a tunnel's portal reaches the
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
            Layer::Noise(_)
            | Layer::Heightmap(_)
            | Layer::Stamp(_)
            | Layer::Hydrology(_)
            | Layer::Erosion(_)
            | Layer::Path(_) => false,
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
            Layer::Noise(_)
            | Layer::Heightmap(_)
            | Layer::Stamp(_)
            | Layer::Hydrology(_)
            | Layer::Erosion(_)
            | Layer::Path(_) => false,
        })
    }

    /// Applies layer `index` other than noise to `row`, heights at
    /// terrain-local x positions `xs` and z position `z`; layers that reach
    /// no point between `min` and `max` are skipped. Hydrology layers' rivers
    /// read the row through `span`'s row, where a tile has one,
    /// as they read sample by sample.
    #[expect(clippy::too_many_arguments, reason = "the generator's row inputs")]
    pub(crate) fn apply_local(
        &self,
        index: usize,
        row: &mut [f32],
        xs: &[f32],
        z: f32,
        (min, max): (Vec2, Vec2),
        spacing: f32,
        span: Option<(&crate::rivers::RiverSpan, usize)>,
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
                    // Bridges leave the ground under them.
                    if let Some((distance, road, false)) = profile.nearest(Vec2::new(x, z)) {
                        *height += (road - *height) * profile.weight(distance) * self.shown(x, z);
                    }
                }
            }
            Layer::Erosion(stage) => {
                let Some(baked) = &stage.baked else {
                    return;
                };
                for (height, &x) in row.iter_mut().zip(xs) {
                    *height += baked.change_at(x, z, spacing) * self.shown(x, z);
                }
            }
            Layer::Hydrology(stage) => {
                let Some(baked) = &stage.baked else {
                    return;
                };
                let level = baked.level_for(spacing);
                for (height, &x) in row.iter_mut().zip(xs) {
                    *height -= baked.carve_at(x, z, level) * self.shown(x, z);
                }
                // Rivers cut their beds below their water and
                // ease their banks to the ground at every level.
                let (low, high) = baked.rivers.bounds();
                if baked.rivers.is_empty()
                    || low.x > max.x
                    || low.y > max.y
                    || high.x < min.x
                    || high.y < min.y
                {
                    return;
                }
                let mut capped = row.to_vec();
                match span {
                    Some((span, at)) => {
                        baked
                            .rivers
                            .cap_row(span, at, (xs, z), &mut capped, spacing);
                    }
                    None => {
                        for (height, &x) in capped.iter_mut().zip(xs) {
                            *height = baked.rivers.cap(x, z, *height, spacing);
                        }
                    }
                }
                for ((height, &capped), &x) in row.iter_mut().zip(&capped).zip(xs) {
                    let shown = self.shown(x, z);
                    *height = if shown >= 1.0 {
                        capped
                    } else {
                        *height + (capped - *height) * shown
                    };
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

// Binary encoding.
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
});

/// The stack's layers with their bakes, its offset and its fade; reading
/// validates the layers as [`Layers::new`] does and rebuilds the noise
/// tables they derive.
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
