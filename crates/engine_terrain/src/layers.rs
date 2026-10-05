//! The layer stack: procedural noise with erosion, imported heightmaps and
//! stamps, applied in order to every sample; later layers win.

use std::sync::Arc;

use engine_math::{clamp, max, min};
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::noise::{gradient_noise, gradient_value, gullies, smooth};

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

/// 16-bit height samples of an imported heightmap, row by row along z.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightImage {
    width: u32,
    height: u32,
    samples: Box<[u16]>,
}

impl HeightImage {
    /// `samples` holds `width * height` values, row by row; at least two
    /// along each axis.
    pub fn new(width: u32, height: u32, samples: Vec<u16>) -> Option<Self> {
        (width >= 2 && height >= 2 && samples.len() == width as usize * height as usize).then(
            || Self {
                width,
                height,
                samples: samples.into_boxed_slice(),
            },
        )
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

    #[inline]
    fn texel(&self, x: i32, z: i32) -> f32 {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let z = z.clamp(0, self.height as i32 - 1) as usize;
        f32::from(self.samples[z * self.width as usize + x])
    }

    /// Catmull-Rom interpolation at image coordinates `(u, v)` in samples,
    /// 0 to 65535, edges clamped.
    #[inline]
    fn sample(&self, u: f32, v: f32) -> f32 {
        let u0 = u.floor();
        let v0 = v.floor();
        let (x, z) = (u0 as i32, v0 as i32);
        let wx = catmull_rom(u - u0);
        let wz = catmull_rom(v - v0);
        let mut sum = 0.0;
        for (row, weight_z) in wz.iter().enumerate() {
            let zi = z - 1 + row as i32;
            let line = wx[0] * self.texel(x - 1, zi)
                + wx[1] * self.texel(x, zi)
                + wx[2] * self.texel(x + 1, zi)
                + wx[3] * self.texel(x + 2, zi);
            sum += weight_z * line;
        }
        sum
    }
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
}

impl HeightmapLayer {
    /// The layer's height and coverage at terrain-local `(x, z)`.
    #[inline]
    fn at(&self, x: f32, z: f32) -> (f32, f32) {
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
        let value = self.image.sample(u, v) * (1.0 / 65_535.0);
        (self.low + (self.high - self.low) * value, weight)
    }
}

/// One layer of a [`Layers`] stack.
#[derive(Clone, Debug, PartialEq)]
pub enum Layer {
    Noise(NoiseLayer),
    Heightmap(HeightmapLayer),
    Stamp(Stamp),
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

/// What the coarse octaves leave at a point of the coarse grid, which
/// samples between grid points interpolate and the fine octaves continue,
/// so every octave sums the same way at every level and levels differ only
/// by the octaves they leave out: the warp's offsets, the fractal and
/// ridged sums, the ridged weight and the damping's lattice slope after the
/// coarse octaves, and the gullies' heights.
pub(crate) const STATE: usize = 8;
pub(crate) type CoarseState = [f32; STATE];

/// A noise layer with its octaves laid out.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NoiseTables {
    params: NoiseLayer,
    octaves: [Octave; OCTAVES],
    gullies: [GullyOctave; GULLY_OCTAVES],
}

/// Octaves at least this many samples long are interpolated from the
/// coarse grid; shorter ones, down to [`FINEST_SAMPLES`], are evaluated at
/// every sample.
pub(crate) const COARSE_SAMPLES: f32 = 16.0;
pub(crate) const FINEST_SAMPLES: f32 = 4.0;

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
            .take_while(|octave| octave.cell >= COARSE_SAMPLES * spacing)
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

    /// The coarse state at `(x, z)`: the first `octaves` octaves at the
    /// warped point, and `gully_octaves` gully octaves, gully `k` running
    /// across the slope of octaves 0 to `k + 1` (four of its cells long and
    /// more), whatever the level.
    pub(crate) fn coarse(
        &self,
        x: f32,
        z: f32,
        octaves: usize,
        gully_octaves: usize,
    ) -> CoarseState {
        let params = &self.params;
        let warp = self.warp(x, z);
        let (px, pz) = (x + warp[0], z + warp[1]);
        // Two sums of the same octaves: fractal noise whose octaves fade
        // where the ground already slopes (Quilez), and Musgrave's ridged
        // multifractal, whose octaves are weighted by the ridge below them,
        // with their gradients per meter.
        let mut rolling = 0.0;
        let mut rolling_gradient = [0.0_f32; 2];
        let mut lattice_slope = [0.0_f32; 2];
        let mut ridged = 0.0;
        let mut ridged_gradient = [0.0_f32; 2];
        let mut weight = 1.0;
        // The blended sum and its gradient after octaves 1 to GULLY_OCTAVES,
        // which the gullies run across.
        let mut slopes = [[0.0_f32; 3]; GULLY_OCTAVES];
        for (index, octave) in self.octaves[..octaves].iter().enumerate() {
            let (qx, qz) = octave.lattice(px, pz);
            let [n, nx, nz] = gradient_noise(qx, qz, octave.seed);
            let gx = octave.m[0] * nx + octave.m[2] * nz;
            let gz = octave.m[1] * nx + octave.m[3] * nz;
            lattice_slope[0] += nx;
            lattice_slope[1] += nz;
            let damping = 1.0
                / (1.0
                    + params.smoothing
                        * (lattice_slope[0] * lattice_slope[0]
                            + lattice_slope[1] * lattice_slope[1]));
            let a = octave.amplitude * damping;
            rolling += a * n;
            rolling_gradient[0] += a * gx;
            rolling_gradient[1] += a * gz;
            let crest = 1.0 - n.abs();
            let signal = crest * crest * weight;
            let change = -2.0 * crest * weight * octave.amplitude * 1.0_f32.copysign(n);
            ridged += octave.amplitude * signal;
            ridged_gradient[0] += change * gx;
            ridged_gradient[1] += change * gz;
            weight = clamp(signal * 2.0, 0.0, 1.0);
            if let Some(slot) = index.checked_sub(1).and_then(|k| slopes.get_mut(k)) {
                *slot = [
                    blend(rolling, ridged, params.ridges),
                    blend_gradient(rolling_gradient[0], ridged_gradient[0], params.ridges),
                    blend_gradient(rolling_gradient[1], ridged_gradient[1], params.ridges),
                ];
            }
        }
        let mut gully_heights = 0.0;
        for (octave, [mixed, mx, mz]) in self.gullies[..gully_octaves].iter().zip(slopes) {
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
        [
            warp[0],
            warp[1],
            rolling,
            ridged,
            weight,
            lattice_slope[0],
            lattice_slope[1],
            gully_heights,
        ]
    }

    /// The height at `(x, z)` from the coarse `state` there: the fine
    /// octaves `range` continue its sums exactly as coarse octaves add to
    /// them, then valley floors widen and gullies cut in.
    #[inline]
    pub(crate) fn finish(
        &self,
        x: f32,
        z: f32,
        state: &CoarseState,
        range: std::ops::Range<usize>,
    ) -> f32 {
        let params = &self.params;
        let (px, pz) = (x + state[0], z + state[1]);
        let [mut rolling, mut ridged, mut weight] = [state[2], state[3], state[4]];
        let mut lattice_slope = [state[5], state[6]];
        for octave in &self.octaves[range] {
            let (qx, qz) = octave.lattice(px, pz);
            let [n, nx, nz] = gradient_noise(qx, qz, octave.seed);
            lattice_slope[0] += nx;
            lattice_slope[1] += nz;
            let damping = 1.0
                / (1.0
                    + params.smoothing
                        * (lattice_slope[0] * lattice_slope[0]
                            + lattice_slope[1] * lattice_slope[1]));
            let a = octave.amplitude * damping;
            rolling += a * n;
            let crest = 1.0 - n.abs();
            let signal = crest * crest * weight;
            ridged += octave.amplitude * signal;
            weight = clamp(signal * 2.0, 0.0, 1.0);
        }
        // Widen valley floors.
        let h = clamp(blend(rolling, ridged, params.ridges), 0.0, 1.0);
        let shaped = h + params.valleys * (h * h - h);
        params.offset + params.height * shaped + state[7]
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
                Layer::Stamp(stamp) => {
                    check(index, "center", stamp.center[0], -limit, limit)?;
                    check(index, "center", stamp.center[1], -limit, limit)?;
                    check(index, "radius", stamp.radius, 0.0, limit)?;
                    check(index, "falloff", stamp.falloff, 0.0, limit)?;
                    match stamp.op {
                        StampOp::Raise(h) | StampOp::Lower(h) => {
                            check(index, "op", h, 0.0, 20_000.0)?
                        }
                        StampOp::Flatten(h) => check(index, "op", h, -20_000.0, 20_000.0)?,
                        StampOp::Hole => {}
                    }
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

    pub(crate) fn noise(&self, index: usize) -> Option<&NoiseTables> {
        self.tables.get(index).and_then(Option::as_ref)
    }

    /// Whether a hole stamp reaches the square from `min` to `max`.
    pub(crate) fn holes_reach(&self, min: Vec2, max: Vec2) -> bool {
        self.layers.iter().any(|layer| match layer {
            Layer::Stamp(
                stamp @ Stamp {
                    op: StampOp::Hole, ..
                },
            ) => reaches(stamp.center, stamp.radius, min, max),
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
        min: Vec2,
        max: Vec2,
    ) {
        match &self.layers[index] {
            Layer::Noise(_) => {}
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
                for (height, &x) in row.iter_mut().zip(xs) {
                    let (value, weight) = map.at(x, z);
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
