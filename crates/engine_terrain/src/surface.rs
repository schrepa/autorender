//! Terrain surfaces: which materials the ground is made of,
//! where. A surface names its materials and lists biomes; each biome has a
//! fitness mask saying where it applies and layers of materials, each
//! painted over the layers before it by its own mask; and one material may
//! cover its steep ground over every biome. Every tile sample
//! gets its control value: the two materials weighing most there, their
//! blend and the biome weighing most, a pure function of the sample's
//! position, height and the shape of the smooth surface through the samples
//! around it at its level ([`SampleShape`]), so neighboring tiles agree on
//! the samples they share. Renderers decide the steep material again at
//! every point of the ground, from the smooth surface's slope there.

use engine_math::{clamp, max, min, sin_cos};
use engine_settings::{ExtraRules, InvalidSetting, Rule, Setting, settings};
use serde::{Deserialize, Serialize};

use crate::PathSample;
use crate::noise::{gradient_value, gradient_values, smooth};

/// Materials a surface can name: control values hold 8-bit indices.
pub const MAX_SURFACE_MATERIALS: usize = 255;
/// Biomes a surface can list: control values hold 8-bit indices.
pub const MAX_BIOMES: usize = 255;
/// Validity guards: how deep masks nest, how many one list combines, how
/// many octaves one noise sums, and how many layers a biome paints.
const MAX_MASK_DEPTH: usize = 8;
const MAX_MASK_LIST: usize = 16;
const MAX_NOISE_OCTAVES: u32 = 8;
const MAX_BIOME_LAYERS: usize = 64;

/// A value from 0 to 1 at every sample, from the sample's height, slope,
/// curvature and position: where a biome applies, or where a material
/// covers its biome's ground. Ramps run from 0 at their first value to 1 at
/// their second, smoothly, falling when the first is the larger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Mask {
    /// The same value everywhere.
    Constant(f32),
    /// A ramp over meters above the terrain's origin.
    Height([f32; 2]),
    /// A ramp over the ground's slope, in degrees from level: the smooth
    /// surface's through its samples ([`SampleShape`]).
    Slope(
        #[cfg_attr(feature = "schema", schemars(inner(range(min = 0.0, max = MAX_SLOPE_DEGREES))))]
        [f32; 2],
    ),
    /// A ramp over the ground's curvature: how fast its slope grows per
    /// meter, positive in hollows and valleys, negative on ridges (the
    /// smooth surface's Laplacian, [`SampleShape`]).
    Curvature([f32; 2]),
    /// A ramp over the base-10 logarithm of the area draining through the
    /// ground, in square meters, as the terrain's hydrology bake found it:
    /// channels begin near its channel area. 0 without one.
    Flow([f32; 2]),
    /// A ramp over the bake's wetness index, ln(a / tan b): about 5 on
    /// steep ground, 10 and more in hollows and along streams.
    Wetness([f32; 2]),
    /// A ramp over how deep water would stand in a filled pit, in meters.
    Lake([f32; 2]),
    /// A ramp over meters from the nearest river's edge,
    /// negative in its water: `[2, 0]` covers river beds with two meters of
    /// banks. Beyond every river's reach (sixteen meters past its edge),
    /// and without rivers, it reads 1e9.
    River([f32; 2]),
    /// A ramp over meters from the nearest road's edge,
    /// negative on roads: `[1, 0]` covers roads with a meter of soft edge.
    /// It averages the ramp across each sample's footprint, so roads
    /// narrower than coarse levels' samples fade there instead of breaking
    /// into dashes. Further than 32 meters beyond every road's shoulders,
    /// and without roads, it reads 1e9.
    Path([f32; 2]),
    /// Gradient noise from 0 to 1.
    Noise(NoiseMask),
    /// The product of masks.
    Multiply(Vec<Mask>),
    /// The sum of masks.
    Add(Vec<Mask>),
    /// The smallest of masks.
    Min(Vec<Mask>),
    /// The largest of masks.
    Max(Vec<Mask>),
    /// One minus a mask.
    Invert(Box<Mask>),
    /// A mask remapped by a ramp: 0 up to `from`, 1 from `to`.
    Remap(Remap),
}

/// The steepest slope a mask names, in degrees: a wall.
pub const MAX_SLOPE_DEGREES: f32 = 90.0;

/// The tangent of a slope of `degrees` from level, which masks compare
/// gradients with, 90 degrees standing in for a vertical wall.
fn tangent(degrees: f32) -> f32 {
    let (sin, cos) = sin_cos(min(degrees, 89.9).to_radians());
    sin / cos
}

settings! {
    /// Gradient noise from 0 to 1, summed over octaves each half as long and
    /// half as strong.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct NoiseMask {
        /// Meters across its largest features.
        pub wavelength: f32 = 100.0 => range(0.01, 1.0e6),
        /// Picks the pattern.
        #[serde(default)]
        pub seed: u32 = 0 => any,
        /// Octaves summed.
        #[serde(default = "one_octave")]
        pub octaves: u32 = one_octave() => range(1, MAX_NOISE_OCTAVES),
    }
}

fn one_octave() -> u32 {
    1
}

/// A mask remapped: 0 up to `from`, 1 from `to`, smooth between.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Remap {
    pub mask: Box<Mask>,
    pub from: f32,
    pub to: f32,
}

impl Mask {
    /// 1 everywhere.
    pub fn one() -> Self {
        Self::Constant(1.0)
    }
}

fn is_one(mask: &Mask) -> bool {
    *mask == Mask::one()
}

fn one() -> f32 {
    1.0
}

fn is_unit(value: &f32) -> bool {
    *value == 1.0
}

settings! {
    /// One layer of a biome's ground. A biome's first layer covers its whole
    /// ground.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct BiomeLayer {
        /// The material painted, an ID of the surface's materials.
        pub material: String = String::new() => any,
        /// Where it paints over the layers before it.
        #[serde(default = "Mask::one", skip_serializing_if = "is_one")]
        pub mask: Mask = Mask::one() => any,
        /// The most it covers.
        #[serde(default = "one", skip_serializing_if = "is_unit")]
        pub opacity: f32 = one() => range(0.0, 1.0),
    }
}

/// A biome: where it applies (`fitness`, everywhere when omitted) and the
/// layers of its ground. Later biomes paint over earlier ones where they
/// fit; the first covers wherever none of the others do.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Biome {
    pub id: String,
    #[serde(default = "Mask::one", skip_serializing_if = "is_one")]
    pub fitness: Mask,
    pub layers: Vec<BiomeLayer>,
}

settings! {
    extra_rules;
    /// The material covering a surface's steep ground, painted over every
    /// biome and layer: none of it where the ground's slope is the first of
    /// `slope`'s degrees from level, all of it from the second, smoothly
    /// between, on the smooth surface through the ground's samples
    /// ([`SampleShape`]). Samples hold it as they hold their other materials;
    /// renderers decide it again at every point of the ground from the slope
    /// there, so its edges follow the ground's form, not its samples.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SteepLayer {
        /// The material covering it, an ID of the surface's materials.
        pub material: String = String::new() => any,
        /// The slopes its share rises between, in degrees from level.
        pub slope: [f32; 2] = [28.0, 35.0] => range(0.0, MAX_SLOPE_DEGREES),
    }
}

impl ExtraRules for SteepLayer {
    /// Its share rises with the slope.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.slope[0] < self.slope[1] {
            Ok(())
        } else {
            Err(InvalidSetting::new("slope", "must rise: its first slope below its second"))
        }
    }
}

/// A surface's steep layer as renderers decide it: the index of its
/// material and the tangents of the slopes its share rises between
/// ([`SteepLayer`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steep {
    pub material: u8,
    pub gradient: [f32; 2],
}

/// A surface as authored: its materials' IDs, in the order control values
/// index them, its biomes, the material covering its steep ground, and the
/// IDs of the assets its placement layers stand on it, in the order
/// placements index them.
#[derive(Clone, Debug, PartialEq)]
pub struct Biomes {
    pub materials: Vec<String>,
    pub biomes: Vec<Biome>,
    pub steep: Option<SteepLayer>,
    pub assets: Vec<String>,
    pub placements: Vec<crate::PlacementLayer>,
    /// Objects authors stand on it.
    pub placed: Vec<crate::AuthoredPlacement>,
}

/// What a surface refused: the path of the offending value and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BiomeError {
    pub path: String,
    pub rule: String,
}

impl std::fmt::Display for BiomeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.path, self.rule)
    }
}

impl std::error::Error for BiomeError {}

/// Which of a sample's inputs masks read: its water (flow,
/// wetness, lakes and rivers) and its roads. Inputs no mask reads are not
/// sampled, which leaves every mask's value as it was.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reads {
    pub(crate) water: bool,
    pub(crate) path: bool,
}

impl Reads {
    fn and(self, other: Self) -> Self {
        Self {
            water: self.water || other.water,
            path: self.path || other.path,
        }
    }
}

/// A row of samples as masks read it: each sample's x and
/// z, height, gradient, curvature, water and road, at a level whose samples
/// lie `spacing` meters apart.
pub(crate) struct RowInputs<'a> {
    pub(crate) xs: &'a [f32],
    pub(crate) zs: &'a [f32],
    pub(crate) spacing: f32,
    pub(crate) height: &'a [f32],
    pub(crate) gradient: &'a [f32],
    pub(crate) curvature: &'a [f32],
    pub(crate) water: &'a [crate::HydrologySample],
    pub(crate) path: &'a [PathSample],
}

/// Rows a row's masks evaluate their parts into, reused.
#[derive(Default)]
pub(crate) struct RowScratch {
    rows: Vec<Vec<f32>>,
}

impl RowScratch {
    fn take(&mut self, len: usize) -> Vec<f32> {
        let mut row = self.rows.pop().unwrap_or_default();
        row.resize(len, 0.0);
        row
    }

    fn give(&mut self, row: Vec<f32>) {
        self.rows.push(row);
    }
}

/// A mask ready to evaluate: slopes as gradients, noise as frequencies.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Compiled {
    Constant(f32),
    Height([f32; 2]),
    Gradient([f32; 2]),
    Curvature([f32; 2]),
    Flow([f32; 2]),
    Wetness([f32; 2]),
    Lake([f32; 2]),
    River([f32; 2]),
    Path([f32; 2]),
    Noise {
        frequency: f32,
        seed: u32,
        octaves: u32,
    },
    Multiply(Vec<Compiled>),
    Add(Vec<Compiled>),
    Min(Vec<Compiled>),
    Max(Vec<Compiled>),
    Invert(Box<Compiled>),
    Remap(Box<Compiled>, [f32; 2]),
}

/// The ground's shape at a sample, as masks, normals and renderers read
/// it: the gradient (rise per meter along x and z) and the Laplacian (per
/// meter) at the sample of the smooth surface through the samples, the
/// uniform cubic B-spline whose control points are their heights. The
/// ground's normal anywhere blends these gradients across its cell
/// ([`crate::Tile::sample`]), as renderers shade it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleShape {
    pub gradient: [f32; 2],
    pub curvature: f32,
}

impl SampleShape {
    /// The shape at the middle of `around`, the heights of the 3 × 3
    /// samples centered on it, `spacing` meters apart: rows along z from
    /// the least, each along x from the least.
    pub fn of(around: [[f32; 3]; 3], spacing: f32) -> Self {
        let [[a, b, c], [d, e, f], [g, h, i]] = around;
        Self {
            gradient: smooth_gradient(around, spacing),
            curvature: (a + b + c + d + f + g + h + i - 8.0 * e) / (3.0 * spacing * spacing),
        }
    }

    /// The slope's tangent: the gradient's length.
    pub fn slope(&self) -> f32 {
        let [x, z] = self.gradient;
        (x * x + z * z).sqrt()
    }
}

/// [`SampleShape::of`]'s gradient alone: the spline's central differences,
/// weighted 1, 4 and 1 across them.
pub(crate) fn smooth_gradient(around: [[f32; 3]; 3], spacing: f32) -> [f32; 2] {
    let [[a, b, c], [d, _, f], [g, h, i]] = around;
    let across = 12.0 * spacing;
    [
        ((c - a) + 4.0 * (f - d) + (i - g)) / across,
        ((g - a) + 4.0 * (h - b) + (i - c)) / across,
    ]
}

/// What a sample's masks read: its terrain-local position in meters, its
/// height above the terrain's origin, the slope's tangent and the
/// curvature per meter of the smooth surface there ([`SampleShape`]), and
/// the meters between samples at its level, below twice which noise fades
/// to its mean, so coarse levels show the same patches as fine ones,
/// filtered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleInputs {
    pub x: f32,
    pub z: f32,
    pub height: f32,
    pub gradient: f32,
    pub curvature: f32,
    pub spacing: f32,
    /// What the terrain's hydrology bake found there; zeros
    /// without one.
    pub water: crate::HydrologySample,
    /// The nearest road, as [`crate::Layers::path_at`]
    /// reads it.
    pub path: PathSample,
}

/// 0 at `ends[0]`, 1 at `ends[1]`, smooth between; a step where they meet.
#[inline]
fn ramp(value: f32, ends: [f32; 2]) -> f32 {
    if ends[0] == ends[1] {
        return if value >= ends[0] { 1.0 } else { 0.0 };
    }
    smooth((value - ends[0]) / (ends[1] - ends[0]))
}

/// Each sample's `input` of its hydrology sample in `waters`, ramped
/// between `ends`, into `out`.
fn water(
    out: &mut [f32],
    waters: &[crate::HydrologySample],
    ends: [f32; 2],
    input: impl Fn(&crate::HydrologySample) -> f32,
) {
    for (value, water) in out.iter_mut().zip(waters) {
        *value = ramp(input(water), ends);
    }
}

/// `masks` folded over `row` into `out` with `op`, from `start`, in the
/// masks' order.
fn fold(
    masks: &[Compiled],
    row: &RowInputs<'_>,
    out: &mut [f32],
    scratch: &mut RowScratch,
    start: f32,
    op: impl Fn(f32, f32) -> f32,
) {
    out.fill(start);
    let mut part = scratch.take(out.len());
    for mask in masks {
        mask.evaluate_row(row, &mut part, scratch);
        for (value, &next) in out.iter_mut().zip(&part) {
            *value = op(*value, next);
        }
    }
    scratch.give(part);
}

impl Compiled {
    pub(crate) fn from(mask: &Mask, path: &str, depth: usize) -> Result<Self, BiomeError> {
        let fail = |rule: &str| BiomeError {
            path: path.to_owned(),
            rule: rule.to_owned(),
        };
        if depth > MAX_MASK_DEPTH {
            return Err(fail("masks nest at most 8 deep"));
        }
        let finite = |values: &[f32]| values.iter().all(|value| value.is_finite());
        let list = |masks: &[Mask], name: &str| -> Result<Vec<Compiled>, BiomeError> {
            if masks.is_empty() || masks.len() > MAX_MASK_LIST {
                return Err(fail("combines 1 to 16 masks"));
            }
            masks
                .iter()
                .enumerate()
                .map(|(index, mask)| {
                    Self::from(mask, &format!("{path}.{name}[{index}]"), depth + 1)
                })
                .collect()
        };
        Ok(match mask {
            Mask::Constant(value) => {
                if !value.is_finite() {
                    return Err(fail("must be finite"));
                }
                Self::Constant(*value)
            }
            Mask::Height(ends) => {
                if !finite(ends) {
                    return Err(fail("heights must be finite"));
                }
                Self::Height(*ends)
            }
            Mask::Slope(ends) => {
                if let Err(invalid) =
                    ends.check("slope", Rule::Range(0.0, MAX_SLOPE_DEGREES as f64))
                {
                    return Err(fail(&invalid.rule));
                }
                // As gradients: the tangent of each slope.
                Self::Gradient(ends.map(tangent))
            }
            Mask::Curvature(ends) => {
                if !finite(ends) {
                    return Err(fail("curvatures must be finite"));
                }
                Self::Curvature(*ends)
            }
            Mask::Flow(ends) => {
                if !finite(ends) {
                    return Err(fail("flows must be finite"));
                }
                Self::Flow(*ends)
            }
            Mask::Wetness(ends) => {
                if !finite(ends) {
                    return Err(fail("wetness must be finite"));
                }
                Self::Wetness(*ends)
            }
            Mask::Lake(ends) => {
                if !finite(ends) {
                    return Err(fail("lake depths must be finite"));
                }
                Self::Lake(*ends)
            }
            Mask::River(ends) => {
                if !finite(ends) {
                    return Err(fail("river distances must be finite"));
                }
                Self::River(*ends)
            }
            Mask::Path(ends) => {
                if !finite(ends) {
                    return Err(fail("path distances must be finite"));
                }
                Self::Path(*ends)
            }
            Mask::Noise(noise) => {
                if let Err(invalid) = noise.validate() {
                    return Err(BiomeError {
                        path: format!("{path}.noise.{}", invalid.field),
                        rule: invalid.rule.into_owned(),
                    });
                }
                Self::Noise {
                    frequency: 1.0 / noise.wavelength,
                    seed: noise.seed,
                    octaves: noise.octaves,
                }
            }
            Mask::Multiply(masks) => Self::Multiply(list(masks, "multiply")?),
            Mask::Add(masks) => Self::Add(list(masks, "add")?),
            Mask::Min(masks) => Self::Min(list(masks, "min")?),
            Mask::Max(masks) => Self::Max(list(masks, "max")?),
            Mask::Invert(mask) => Self::Invert(Box::new(Self::from(
                mask,
                &format!("{path}.invert"),
                depth + 1,
            )?)),
            Mask::Remap(remap) => {
                if !(remap.from.is_finite() && remap.to.is_finite()) {
                    return Err(fail("remap ends must be finite"));
                }
                Self::Remap(
                    Box::new(Self::from(
                        &remap.mask,
                        &format!("{path}.remap.mask"),
                        depth + 1,
                    )?),
                    [remap.from, remap.to],
                )
            }
        })
    }

    /// Which of a sample's inputs the mask reads.
    pub(crate) fn reads(&self) -> Reads {
        match self {
            Self::Flow(_) | Self::Wetness(_) | Self::Lake(_) | Self::River(_) => Reads {
                water: true,
                path: false,
            },
            Self::Path(_) => Reads {
                water: false,
                path: true,
            },
            Self::Constant(_)
            | Self::Height(_)
            | Self::Gradient(_)
            | Self::Curvature(_)
            | Self::Noise { .. } => Reads::default(),
            Self::Multiply(masks) | Self::Add(masks) | Self::Min(masks) | Self::Max(masks) => masks
                .iter()
                .fold(Reads::default(), |reads, mask| reads.and(mask.reads())),
            Self::Invert(mask) | Self::Remap(mask, _) => mask.reads(),
        }
    }

    /// Work units of one evaluation, against one per sample of a layer.
    pub(crate) fn cost(&self) -> u64 {
        match self {
            Self::Constant(_) => 0,
            Self::Height(_)
            | Self::Gradient(_)
            | Self::Curvature(_)
            | Self::Flow(_)
            | Self::Wetness(_)
            | Self::Lake(_)
            | Self::River(_)
            | Self::Path(_) => 1,
            Self::Noise { octaves, .. } => 2 * u64::from(*octaves),
            Self::Multiply(masks) | Self::Add(masks) | Self::Min(masks) | Self::Max(masks) => {
                masks.iter().map(Self::cost).sum::<u64>() + 1
            }
            Self::Invert(mask) | Self::Remap(mask, _) => mask.cost() + 1,
        }
    }

    /// [`Self::evaluate`] for each sample of `row`, into
    /// `out`: every node over the whole row at once, each sample's value
    /// computed as one evaluation computes it, so the same bits.
    pub(crate) fn evaluate_row(
        &self,
        row: &RowInputs<'_>,
        out: &mut [f32],
        scratch: &mut RowScratch,
    ) {
        let each = |out: &mut [f32], values: &[f32], ends: [f32; 2]| {
            for (value, &input) in out.iter_mut().zip(values) {
                *value = ramp(input, ends);
            }
        };
        match self {
            Self::Constant(value) => out.fill(*value),
            Self::Height(ends) => each(out, row.height, *ends),
            Self::Gradient(ends) => each(out, row.gradient, *ends),
            Self::Curvature(ends) => each(out, row.curvature, *ends),
            Self::Flow(ends) => water(out, row.water, *ends, |water| water.flow),
            Self::Wetness(ends) => water(out, row.water, *ends, |water| water.wetness),
            Self::Lake(ends) => water(out, row.water, *ends, |water| water.lake),
            Self::River(ends) => water(out, row.water, *ends, |water| water.river),
            Self::Path(ends) => {
                for (value, &PathSample { distance, half }) in out.iter_mut().zip(row.path) {
                    *value = [-0.375, -0.125, 0.125, 0.375]
                        .iter()
                        .map(|offset| ramp((distance + offset * row.spacing).abs() - half, *ends))
                        .sum::<f32>()
                        * 0.25;
                }
            }
            Self::Noise {
                frequency,
                seed,
                octaves,
            } => {
                // Each sample's sum over the octaves, in their order; the
                // octaves' shifts, fades and amplitudes are the row's.
                out.fill(0.0);
                let mut values = scratch.take(out.len());
                let mut at_x = scratch.take(out.len());
                let mut at_z = scratch.take(out.len());
                let (mut weight, mut amplitude, mut scale) = (0.0_f32, 1.0_f32, *frequency);
                for octave in 0..*octaves {
                    let shift = octave as f32 * 17.13;
                    let fade = clamp(0.5 / (scale * row.spacing) - 1.0, 0.0, 1.0);
                    if fade > 0.0 {
                        for ((at_x, at_z), (&x, &z)) in at_x
                            .iter_mut()
                            .zip(at_z.iter_mut())
                            .zip(row.xs.iter().zip(row.zs))
                        {
                            *at_x = x * scale + shift;
                            *at_z = z * scale - shift;
                        }
                        gradient_values(
                            &at_x,
                            &at_z,
                            seed.wrapping_add(octave.wrapping_mul(0x9e37_79b9)),
                            &mut values,
                        );
                        let share = amplitude * fade;
                        for (sum, &value) in out.iter_mut().zip(&values) {
                            *sum += share * value;
                        }
                    }
                    weight += amplitude;
                    amplitude *= 0.5;
                    scale *= 2.0;
                }
                for sum in out.iter_mut() {
                    *sum = clamp(0.5 + 0.5 * *sum / weight, 0.0, 1.0);
                }
                scratch.give(values);
                scratch.give(at_x);
                scratch.give(at_z);
            }
            // The folds `product`, `sum` and the evaluation's own take: from
            // 1, -0 (the standard library's), +infinity and -infinity, in the
            // masks' order.
            Self::Multiply(masks) => fold(masks, row, out, scratch, 1.0, |a, b| a * b),
            Self::Add(masks) => fold(masks, row, out, scratch, -0.0, |a, b| a + b),
            Self::Min(masks) => fold(masks, row, out, scratch, f32::INFINITY, min),
            Self::Max(masks) => fold(masks, row, out, scratch, f32::NEG_INFINITY, max),
            Self::Invert(mask) => {
                mask.evaluate_row(row, out, scratch);
                for value in out.iter_mut() {
                    *value = 1.0 - *value;
                }
            }
            Self::Remap(mask, ends) => {
                mask.evaluate_row(row, out, scratch);
                for value in out.iter_mut() {
                    *value = ramp(*value, *ends);
                }
            }
        }
    }

    /// Its code for the GPU interpreter, in postfix order,
    /// onto `out`: the most values the interpreter's stack then holds.
    pub(crate) fn emit(&self, out: &mut Vec<u32>) -> u32 {
        use crate::gpu::op;
        let ramp = |out: &mut Vec<u32>, code: u32, ends: &[f32; 2]| {
            out.extend([code, ends[0].to_bits(), ends[1].to_bits()]);
            1
        };
        let list = |out: &mut Vec<u32>, code: u32, masks: &[Compiled]| {
            let mut depth = 1;
            for (index, mask) in masks.iter().enumerate() {
                depth = depth.max(index as u32 + mask.emit(out));
            }
            out.extend([code, masks.len() as u32]);
            depth
        };
        match self {
            Self::Constant(value) => {
                out.extend([op::CONSTANT, value.to_bits()]);
                1
            }
            Self::Height(ends) => ramp(out, op::HEIGHT, ends),
            Self::Gradient(ends) => ramp(out, op::GRADIENT, ends),
            Self::Curvature(ends) => ramp(out, op::CURVATURE, ends),
            Self::Flow(ends) => ramp(out, op::FLOW, ends),
            Self::Wetness(ends) => ramp(out, op::WETNESS, ends),
            Self::Lake(ends) => ramp(out, op::LAKE, ends),
            Self::River(ends) => ramp(out, op::RIVER, ends),
            Self::Path(ends) => ramp(out, op::PATH, ends),
            Self::Noise {
                frequency,
                seed,
                octaves,
            } => {
                out.extend([op::NOISE, frequency.to_bits(), *seed, *octaves]);
                1
            }
            Self::Multiply(masks) => list(out, op::MULTIPLY, masks),
            Self::Add(masks) => list(out, op::ADD, masks),
            Self::Min(masks) => list(out, op::MIN, masks),
            Self::Max(masks) => list(out, op::MAX, masks),
            Self::Invert(mask) => {
                let depth = mask.emit(out);
                out.push(op::INVERT);
                depth
            }
            Self::Remap(mask, ends) => {
                let depth = mask.emit(out);
                out.extend([op::REMAP, ends[0].to_bits(), ends[1].to_bits()]);
                depth
            }
        }
    }

    pub(crate) fn evaluate(&self, inputs: &SampleInputs) -> f32 {
        match self {
            Self::Constant(value) => *value,
            Self::Height(ends) => ramp(inputs.height, *ends),
            Self::Gradient(ends) => ramp(inputs.gradient, *ends),
            Self::Curvature(ends) => ramp(inputs.curvature, *ends),
            Self::Flow(ends) => ramp(inputs.water.flow, *ends),
            Self::Wetness(ends) => ramp(inputs.water.wetness, *ends),
            Self::Lake(ends) => ramp(inputs.water.lake, *ends),
            Self::River(ends) => ramp(inputs.water.river, *ends),
            Self::Path(ends) => {
                // The ramp at four points across the sample's footprint,
                // across the road.
                let PathSample { distance, half } = inputs.path;
                [-0.375, -0.125, 0.125, 0.375]
                    .iter()
                    .map(|offset| ramp((distance + offset * inputs.spacing).abs() - half, *ends))
                    .sum::<f32>()
                    * 0.25
            }
            Self::Noise {
                frequency,
                seed,
                octaves,
            } => {
                let (mut sum, mut weight, mut amplitude, mut scale) = (0.0, 0.0, 1.0, *frequency);
                for octave in 0..*octaves {
                    // Each octave shifted, so their lattices never line up;
                    // octaves fewer than four samples across fade out by
                    // two, as a filtered image's would.
                    let shift = octave as f32 * 17.13;
                    let fade = clamp(0.5 / (scale * inputs.spacing) - 1.0, 0.0, 1.0);
                    // Octaves faded out add nothing: zero
                    // times the noise's finite value.
                    if fade > 0.0 {
                        sum += amplitude
                            * fade
                            * gradient_value(
                                inputs.x * scale + shift,
                                inputs.z * scale - shift,
                                seed.wrapping_add(octave.wrapping_mul(0x9e37_79b9)),
                            );
                    }
                    weight += amplitude;
                    amplitude *= 0.5;
                    scale *= 2.0;
                }
                clamp(0.5 + 0.5 * sum / weight, 0.0, 1.0)
            }
            Self::Multiply(masks) => masks.iter().map(|mask| mask.evaluate(inputs)).product(),
            Self::Add(masks) => masks.iter().map(|mask| mask.evaluate(inputs)).sum(),
            Self::Min(masks) => masks
                .iter()
                .map(|mask| mask.evaluate(inputs))
                .fold(f32::INFINITY, min),
            Self::Max(masks) => masks
                .iter()
                .map(|mask| mask.evaluate(inputs))
                .fold(f32::NEG_INFINITY, max),
            Self::Invert(mask) => 1.0 - mask.evaluate(inputs),
            Self::Remap(mask, ends) => ramp(mask.evaluate(inputs), *ends),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct CompiledLayer {
    material: u8,
    mask: Compiled,
    opacity: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct CompiledBiome {
    fitness: Compiled,
    layers: Vec<CompiledLayer>,
}

/// A validated surface, ready to evaluate per sample: its biomes, then its
/// steep layer as one more painted over them, fit as steep as the ground
/// is, of its one material.
#[derive(Clone, Debug, PartialEq)]
pub struct BiomeRules {
    authored: Biomes,
    biomes: Vec<CompiledBiome>,
    steep: Option<Steep>,
    cost: u64,
    reads: Reads,
    placements: Vec<crate::placement::CompiledPlacement>,
    placed: Vec<crate::placement::CompiledAuthored>,
}

/// A sample's control value: the material weighing most (`a`), the one
/// weighing second (`b`, `a` again when none), `b`'s share of the two from
/// 0 to 255, and the biome weighing most (the steep layer counting as one
/// after the last).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Control {
    pub a: u8,
    pub b: u8,
    pub blend: u8,
    pub biome: u8,
}

impl Control {
    /// Packed as `a | b << 8 | blend << 16 | biome << 24`, the layout
    /// renderers read.
    pub fn packed(self) -> u32 {
        u32::from(self.a)
            | u32::from(self.b) << 8
            | u32::from(self.blend) << 16
            | u32::from(self.biome) << 24
    }

    pub fn unpack(value: u32) -> Self {
        Self {
            a: value as u8,
            b: (value >> 8) as u8,
            blend: (value >> 16) as u8,
            biome: (value >> 24) as u8,
        }
    }
}

impl Biomes {
    /// Validates and prepares the surface: 1 to 255 materials with unique
    /// IDs, 1 to 255 biomes with unique IDs, each with 1 to 64 layers
    /// naming its materials, and a steep layer naming one.
    pub fn compile(&self) -> Result<BiomeRules, BiomeError> {
        let fail = |path: String, rule: &str| BiomeError {
            path,
            rule: rule.to_owned(),
        };
        if self.materials.is_empty() || self.materials.len() > MAX_SURFACE_MATERIALS {
            return Err(fail("materials".into(), "needs 1 to 255 materials"));
        }
        for (index, id) in self.materials.iter().enumerate() {
            if self.materials[..index].contains(id) {
                return Err(fail(format!("materials[{index}]"), "IDs must be unique"));
            }
        }
        if self.biomes.is_empty() || self.biomes.len() > MAX_BIOMES {
            return Err(fail("biomes".into(), "needs 1 to 255 biomes"));
        }
        let mut biomes = Vec::with_capacity(self.biomes.len());
        for (index, biome) in self.biomes.iter().enumerate() {
            let at = format!("biomes[{index}]");
            if self.biomes[..index]
                .iter()
                .any(|other| other.id == biome.id)
            {
                return Err(fail(format!("{at}.id"), "IDs must be unique"));
            }
            if biome.layers.is_empty() || biome.layers.len() > MAX_BIOME_LAYERS {
                return Err(fail(format!("{at}.layers"), "needs 1 to 64 layers"));
            }
            let fitness = Compiled::from(&biome.fitness, &format!("{at}.fitness"), 0)?;
            let mut layers = Vec::with_capacity(biome.layers.len());
            for (layer_index, layer) in biome.layers.iter().enumerate() {
                let path = format!("{at}.layers[{layer_index}]");
                let material = self
                    .materials
                    .iter()
                    .position(|id| *id == layer.material)
                    .ok_or_else(|| fail(format!("{path}.material"), "must name a material"))?;
                if let Err(invalid) = layer.validate() {
                    return Err(fail(format!("{path}.{}", invalid.field), &invalid.rule));
                }
                layers.push(CompiledLayer {
                    material: material as u8,
                    mask: Compiled::from(&layer.mask, &format!("{path}.mask"), 0)?,
                    opacity: layer.opacity,
                });
            }
            biomes.push(CompiledBiome { fitness, layers });
        }
        let steep = match &self.steep {
            Some(layer) => {
                if let Err(invalid) = layer.validate() {
                    return Err(fail(format!("steep.{}", invalid.field), &invalid.rule));
                }
                let material = self
                    .materials
                    .iter()
                    .position(|id| *id == layer.material)
                    .ok_or_else(|| fail("steep.material".into(), "must name a material"))?;
                let steep = Steep {
                    material: material as u8,
                    gradient: layer.slope.map(tangent),
                };
                // Painted over every biome, as steep as the ground is.
                biomes.push(CompiledBiome {
                    fitness: Compiled::Gradient(steep.gradient),
                    layers: vec![CompiledLayer {
                        material: steep.material,
                        mask: Compiled::Constant(1.0),
                        opacity: 1.0,
                    }],
                });
                Some(steep)
            }
            None => None,
        };
        let cost = 1 + biomes
            .iter()
            .map(|biome| {
                biome.fitness.cost() + biome.layers.iter().map(|l| l.mask.cost()).sum::<u64>()
            })
            .sum::<u64>();
        let placements = crate::placement::compile(&self.placements, &self.materials, &self.assets)
            .map_err(|(path, rule)| BiomeError { path, rule })?;
        let placed = crate::placement::compile_authored(&self.placed, &self.assets)
            .map_err(|(path, rule)| BiomeError { path, rule })?;
        let reads = biomes.iter().fold(Reads::default(), |reads, biome| {
            biome
                .layers
                .iter()
                .fold(reads.and(biome.fitness.reads()), |reads, layer| {
                    reads.and(layer.mask.reads())
                })
        });
        Ok(BiomeRules {
            authored: self.clone(),
            biomes,
            steep,
            cost,
            reads,
            placements,
            placed,
        })
    }
}

impl BiomeRules {
    /// The biomes as authored.
    pub fn authored(&self) -> &Biomes {
        &self.authored
    }

    /// The steep layer, which renderers decide at every point of the
    /// ground; `None` without one.
    pub fn steep(&self) -> Option<Steep> {
        self.steep
    }

    /// Which of a sample's inputs the biomes' masks read.
    pub(crate) fn reads(&self) -> Reads {
        self.reads
    }

    /// These rules choosing materials alone, placing no objects: what a
    /// terrain's surroundings follow.
    #[must_use]
    pub fn without_placements(&self) -> Self {
        let mut rules = self.clone();
        rules.authored.assets.clear();
        rules.authored.placements.clear();
        rules.authored.placed.clear();
        rules.placements.clear();
        rules.placed.clear();
        rules
    }

    /// Whether `placing`'s objects stand on tiles of `level`.
    pub fn places(&self, level: u8, placing: crate::Placing) -> bool {
        let near = self.placements.iter().any(|layer| layer.level == level)
            || self.placed.iter().any(|object| object.level == level);
        let far = self
            .placements
            .iter()
            .any(|layer| layer.far.is_some_and(|(far, _)| far == level));
        match placing {
            crate::Placing::All => near || far,
            crate::Placing::Near => near,
            crate::Placing::Far => far,
        }
    }

    /// The authored objects standing on `tile`, appended to
    /// `out`; returns the work it took.
    pub fn place_authored(&self, tile: &crate::Tile, out: &mut Vec<crate::Placed>) -> u64 {
        crate::placement::place_authored(&self.placed, tile, out)
    }

    /// The objects row `row` of placement layer `layer`'s candidate squares
    /// stand on `tile`, appended to `out`: layers of other
    /// levels than `tile`'s hold none, nor layers `placing` leaves out.
    /// Returns the work it took (samples' worth) and the
    /// next row, none after the last layer's last.
    pub(crate) fn place_row(
        &self,
        tile: &crate::Tile,
        (layer, row): (usize, usize),
        out: &mut Vec<crate::Placed>,
        (layers, placing, scratch): (
            &crate::Layers,
            crate::Placing,
            &mut crate::placement::PlaceScratch,
        ),
    ) -> (u64, Option<(usize, usize)>) {
        crate::placement::place_row(
            &self.placements,
            tile,
            0x5eed_0057,
            (layer, row),
            out,
            (layers, placing, scratch, self),
        )
    }

    /// Work units of one sample's control value, against one per sample
    /// of a layer.
    pub fn cost(&self) -> u64 {
        self.cost
    }

    /// Its program for the GPU interpreter, onto `words`
    /// (where it starts at their length), laid out as
    /// [`crate::gpu::header::PROGRAM`] describes: the most values the
    /// interpreter's stack holds, its biomes, and the materials its control
    /// values index.
    pub(crate) fn program(&self, words: &mut Vec<u32>) -> (u32, u32, u32) {
        use crate::gpu::op;
        let table = words.len();
        words.extend(std::iter::repeat_n(0, 2 * self.biomes.len()));
        let (mut stack, mut materials) = (1, 0);
        let mut mask = |words: &mut Vec<u32>, compiled: &Compiled| {
            let at = words.len() as u32;
            stack = stack.max(compiled.emit(words));
            words.push(op::END);
            at
        };
        for (index, biome) in self.biomes.iter().enumerate() {
            // The first biome and each biome's first layer take no mask.
            words[table + 2 * index] = if index == 0 {
                u32::MAX
            } else {
                mask(words, &biome.fitness)
            };
            let list = words.len();
            words[table + 2 * index + 1] = list as u32;
            words.push(biome.layers.len() as u32);
            words.extend(std::iter::repeat_n(0, 3 * biome.layers.len()));
            for (layer_index, layer) in biome.layers.iter().enumerate() {
                let entry = list + 1 + 3 * layer_index;
                words[entry] = u32::from(layer.material);
                words[entry + 1] = layer.opacity.to_bits();
                words[entry + 2] = if layer_index == 0 {
                    u32::MAX
                } else {
                    mask(words, &layer.mask)
                };
                materials = materials.max(u32::from(layer.material) + 1);
            }
        }
        (stack, self.biomes.len() as u32, materials)
    }

    /// The control value at a sample with `inputs`. `weights` is scratch
    /// space, reused across samples.
    pub fn control(&self, inputs: &SampleInputs, weights: &mut Vec<(u8, f32)>) -> Control {
        self.control_with(
            |biome| self.biomes[biome].fitness.evaluate(inputs),
            |biome, layer| self.biomes[biome].layers[layer].mask.evaluate(inputs),
            weights,
        )
    }

    /// [`Self::control`] packed for each sample of `row`,
    /// into `out`: every biome's fitness and layer's mask evaluated over the
    /// row at once, each sample's control value then taken as one sample's
    /// is, from the same values.
    pub(crate) fn control_row(
        &self,
        row: &RowInputs<'_>,
        out: &mut [u32],
        (scratch, weights): (&mut RowScratch, &mut Vec<(u8, f32)>),
    ) {
        let len = out.len();
        let mut values: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.biomes.len());
        for (index, biome) in self.biomes.iter().enumerate() {
            let mut rows = Vec::with_capacity(biome.layers.len() + 1);
            // The first biome and layers take no mask.
            let mut fitness = scratch.take(if index == 0 { 0 } else { len });
            if index > 0 {
                biome.fitness.evaluate_row(row, &mut fitness, scratch);
            }
            rows.push(fitness);
            for (layer_index, layer) in biome.layers.iter().enumerate() {
                let mut mask = scratch.take(if layer_index == 0 { 0 } else { len });
                if layer_index > 0 {
                    layer.mask.evaluate_row(row, &mut mask, scratch);
                }
                rows.push(mask);
            }
            values.push(rows);
        }
        for (sample, control) in out.iter_mut().enumerate() {
            *control = self
                .control_with(
                    |biome| values[biome][0][sample],
                    |biome, layer| values[biome][layer + 1][sample],
                    weights,
                )
                .packed();
        }
        for rows in values {
            for row in rows {
                scratch.give(row);
            }
        }
    }

    /// The control value of a sample whose biomes' fitnesses and layers'
    /// masks (by biome, and by biome and layer) `fitness` and `mask` give,
    /// asked for only where the walk reaches them.
    fn control_with(
        &self,
        fitness: impl Fn(usize) -> f32,
        mask: impl Fn(usize, usize) -> f32,
        weights: &mut Vec<(u8, f32)>,
    ) -> Control {
        weights.clear();
        // Later biomes and layers paint over earlier ones: walk them from the
        // last, each taking what the later ones left, until nothing is left.
        let mut remaining = 1.0_f32;
        let mut biome = (0_usize, -1.0_f32);
        for (index, compiled) in self.biomes.iter().enumerate().rev() {
            let fit = if index == 0 {
                1.0
            } else {
                clamp(fitness(index), 0.0, 1.0)
            };
            let share = fit * remaining;
            remaining *= 1.0 - fit;
            if share > 0.0 {
                if share > biome.1 {
                    biome = (index, share);
                }
                let mut left = share;
                for (layer_index, layer) in compiled.layers.iter().enumerate().rev() {
                    let cover = if layer_index == 0 {
                        1.0
                    } else {
                        clamp(mask(index, layer_index) * layer.opacity, 0.0, 1.0)
                    };
                    let weight = cover * left;
                    left *= 1.0 - cover;
                    if weight > 0.0 {
                        match weights
                            .iter_mut()
                            .find(|(material, _)| *material == layer.material)
                        {
                            Some((_, total)) => *total += weight,
                            None => weights.push((layer.material, weight)),
                        }
                    }
                    if left <= 0.0 {
                        break;
                    }
                }
            }
            if remaining <= 0.0 {
                break;
            }
        }
        let (mut a, mut b) = ((0_u8, -1.0_f32), (0_u8, -1.0_f32));
        for &(material, weight) in weights.iter() {
            if weight > a.1 {
                b = a;
                a = (material, weight);
            } else if weight > b.1 {
                b = (material, weight);
            }
        }
        if b.1 <= 0.0 {
            b = (a.0, 0.0);
        }
        let blend = b.1 / max(a.1 + b.1, 1.0e-12);
        Control {
            a: a.0,
            b: b.0,
            blend: (blend * 255.0 + 0.5) as u8,
            biome: biome.0 as u8,
        }
    }
}

// Binary encoding.
use crate::codec::{record, tagged};

tagged!(Mask, "invalid mask" {
    0 => Constant(value),
    1 => Height(ends),
    2 => Slope(ends),
    3 => Curvature(ends),
    4 => Flow(ends),
    5 => Wetness(ends),
    6 => Lake(ends),
    7 => River(ends),
    8 => Path(ends),
    9 => Noise(noise),
    10 => Multiply(masks),
    11 => Add(masks),
    12 => Min(masks),
    13 => Max(masks),
    14 => Invert(mask),
    15 => Remap(remap),
});

record!(NoiseMask {
    wavelength,
    seed,
    octaves,
});
record!(Remap { mask, from, to });
record!(BiomeLayer {
    material,
    mask,
    opacity,
});
record!(Biome {
    id,
    fitness,
    layers,
});
record!(SteepLayer { material, slope });
record!(Biomes {
    materials,
    biomes,
    steep,
    assets,
    placements,
    placed,
});
