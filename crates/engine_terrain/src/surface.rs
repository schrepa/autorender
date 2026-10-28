use engine_math::{clamp, max, min, sin_cos};
use engine_settings::{ExtraRules, InvalidSetting, Rule, Setting, settings};
use serde::{Deserialize, Serialize};

use crate::layers::{Layer, Layers};
use crate::noise::{gradient_value, gradient_values, smooth};
use crate::{Erosion, PathSample};

pub const MAX_SURFACE_MATERIALS: usize = 255;
pub const MAX_BIOMES: usize = 255;
const MAX_MASK_DEPTH: usize = 8;
const MAX_MASK_LIST: usize = 16;
const MAX_NOISE_OCTAVES: u32 = 8;
const MAX_BIOME_LAYERS: usize = 64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Mask {
    Constant(f32),
    Height([f32; 2]),
    Slope(
        #[cfg_attr(feature = "schema", schemars(inner(range(min = 0.0, max = MAX_SLOPE_DEGREES))))]
        [f32; 2],
    ),
    Curvature([f32; 2]),
    Facing(Facing),
    Worn([f32; 2]),
    Deposit([f32; 2]),
    Flow([f32; 2]),
    Wetness([f32; 2]),
    Lake([f32; 2]),
    River([f32; 2]),
    Grain([f32; 2]),
    Power([f32; 2]),
    Flood([f32; 2]),
    Path([f32; 2]),
    Noise(NoiseMask),
    Multiply(Vec<Mask>),
    Add(Vec<Mask>),
    Min(Vec<Mask>),
    Max(Vec<Mask>),
    Invert(Box<Mask>),
    Remap(Remap),
}

pub const MAX_SLOPE_DEGREES: f32 = 90.0;

fn tangent(degrees: f32) -> f32 {
    let (sin, cos) = sin_cos(min(degrees, 89.9).to_radians());
    sin / cos
}

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct Facing {
        pub heading: f32 = 180.0 => range(-360.0, 360.0),
        pub ramp: [f32; 2] = [-0.3, 0.3] => finite,
    }
}

settings! {
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct NoiseMask {
        pub wavelength: f32 = 100.0 => range(0.01, 1.0e6),
        #[serde(default)]
        pub seed: u32 = 0 => any,
        #[serde(default = "one_octave")]
        pub octaves: u32 = one_octave() => range(1, MAX_NOISE_OCTAVES),
    }
}

fn one_octave() -> u32 {
    1
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Remap {
    pub mask: Box<Mask>,
    pub from: f32,
    pub to: f32,
}

impl Mask {
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
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct BiomeLayer {
        pub material: String = String::new() => any,
        #[serde(default = "Mask::one", skip_serializing_if = "is_one")]
        pub mask: Mask = Mask::one() => any,
        #[serde(default = "one", skip_serializing_if = "is_unit")]
        pub opacity: f32 = one() => range(0.0, 1.0),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Biome {
    pub id: String,
    #[serde(default = "Mask::one", skip_serializing_if = "is_one")]
    pub fitness: Mask,
    pub layers: Vec<BiomeLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wetness: Option<Mask>,
}

settings! {
    extra_rules;
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SteepLayer {
        pub material: String = String::new() => any,
        pub slope: [f32; 2] = [28.0, 35.0] => range(0.0, MAX_SLOPE_DEGREES),
    }
}

impl ExtraRules for SteepLayer {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.slope[0] < self.slope[1] {
            Ok(())
        } else {
            Err(InvalidSetting::new(
                "slope",
                "must rise: its first slope below its second",
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Steep {
    pub material: u8,
    pub gradient: [f32; 2],
}

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct MacroCauses {
        pub wet: f32 = 0.3 => range(0.0, 1.0),
        pub deposit: f32 = 0.2 => range(0.0, 1.0),
        pub worn: f32 = 0.2 => range(0.0, 1.0),
        pub steep: f32 = 0.15 => range(0.0, 1.0),
        pub relief: f32 = 0.15 => range(0.0, 1.0),
        pub dust: f32 = 0.2 => range(0.0, 1.0),
        pub aspect: f32 = 0.2 => range(0.0, 1.0),
        pub heading: f32 = 180.0 => range(-360.0, 360.0),
        pub broad: f32 = 0.06 => range(0.0, 1.0),
        pub patchy: f32 = 0.03 => range(0.0, 1.0),
        pub warmth: f32 = 0.04 => range(0.0, 1.0),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Biomes {
    pub materials: Vec<String>,
    pub biomes: Vec<Biome>,
    pub steep: Option<SteepLayer>,
    pub assets: Vec<String>,
    pub placements: Vec<crate::PlacementLayer>,
    pub placed: Vec<crate::AuthoredPlacement>,
    pub macro_colors: MacroCauses,
}

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reads {
    pub(crate) water: bool,
    pub(crate) path: bool,
    pub(crate) erosion: bool,
}

impl Reads {
    fn and(self, other: Self) -> Self {
        Self {
            water: self.water || other.water,
            path: self.path || other.path,
            erosion: self.erosion || other.erosion,
        }
    }
}

pub(crate) struct RowInputs<'a> {
    pub(crate) xs: &'a [f32],
    pub(crate) zs: &'a [f32],
    pub(crate) spacing: f32,
    pub(crate) height: &'a [f32],
    pub(crate) gradient: &'a [[f32; 2]],
    pub(crate) curvature: &'a [f32],
    pub(crate) water: &'a [crate::HydrologySample],
    pub(crate) erosion: &'a [f32],
    pub(crate) path: &'a [PathSample],
}

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

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Compiled {
    Constant(f32),
    Height([f32; 2]),
    Gradient([f32; 2]),
    Curvature([f32; 2]),
    Facing {
        toward: [f32; 2],
        ends: [f32; 2],
    },
    Worn([f32; 2]),
    Deposit([f32; 2]),
    Flow([f32; 2]),
    Wetness([f32; 2]),
    Lake([f32; 2]),
    River([f32; 2]),
    Grain([f32; 2]),
    Power([f32; 2]),
    Flood([f32; 2]),
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleShape {
    pub gradient: [f32; 2],
    pub curvature: f32,
}

impl SampleShape {
    pub fn of(around: [[f32; 3]; 3], spacing: f32) -> Self {
        let [[a, b, c], [d, e, f], [g, h, i]] = around;
        Self {
            gradient: smooth_gradient(around, spacing),
            curvature: (a + b + c + d + f + g + h + i - 8.0 * e) / (3.0 * spacing * spacing),
        }
    }
}

pub(crate) fn smooth_gradient(around: [[f32; 3]; 3], spacing: f32) -> [f32; 2] {
    let [[a, b, c], [d, _, f], [g, h, i]] = around;
    let across = 12.0 * spacing;
    [
        ((c - a) + 4.0 * (f - d) + (i - g)) / across,
        ((g - a) + 4.0 * (h - b) + (i - c)) / across,
    ]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleInputs {
    pub x: f32,
    pub z: f32,
    pub height: f32,
    pub gradient: [f32; 2],
    pub curvature: f32,
    pub spacing: f32,
    pub water: crate::HydrologySample,
    pub erosion: f32,
    pub path: PathSample,
}

#[inline]
fn ramp(value: f32, ends: [f32; 2]) -> f32 {
    if ends[0] == ends[1] {
        return if value >= ends[0] { 1.0 } else { 0.0 };
    }
    smooth((value - ends[0]) / (ends[1] - ends[0]))
}

fn ramps(out: &mut [f32], values: impl IntoIterator<Item = f32>, ends: [f32; 2]) {
    for (value, input) in out.iter_mut().zip(values) {
        *value = ramp(input, ends);
    }
}

#[inline]
fn slope_of([x, z]: [f32; 2]) -> f32 {
    (x * x + z * z).sqrt()
}

#[inline]
fn facing(toward: [f32; 2], [x, z]: [f32; 2]) -> f32 {
    toward[0] * x + toward[1] * z
}

#[inline]
fn worn(change: f32) -> f32 {
    max(-change, 0.0)
}

#[inline]
fn deposit(change: f32) -> f32 {
    max(change, 0.0)
}

pub(crate) fn erosions(layers: &Layers) -> impl Iterator<Item = &Erosion> {
    layers.layers().iter().filter_map(|layer| match layer {
        Layer::Erosion(stage) => stage.baked.as_deref(),
        Layer::Noise(_)
        | Layer::Heightmap(_)
        | Layer::Stamp(_)
        | Layer::Rivers(_)
        | Layer::Hydrology(_)
        | Layer::Path(_)
        | Layer::Steps(_)
        | Layer::Hummocks(_) => None,
    })
}

pub(crate) fn erosion_row(layers: &Layers, xs: &[f32], z: f32, spacing: f32, out: &mut [f32]) {
    out.fill(0.0);
    for erosion in erosions(layers) {
        erosion.change_row(xs.iter().copied(), z, spacing, |index, change| {
            out[index] += change;
        });
    }
}

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
        let ramped = |ends: &[f32; 2], what: &str, input: fn([f32; 2]) -> Self| {
            if finite(ends) {
                Ok(input(*ends))
            } else {
                Err(fail(&format!("{what} must be finite")))
            }
        };
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
            Mask::Height(ends) => ramped(ends, "heights", Self::Height)?,
            Mask::Slope(ends) => {
                if let Err(invalid) =
                    ends.check("slope", Rule::Range(0.0, MAX_SLOPE_DEGREES as f64))
                {
                    return Err(fail(&invalid.rule));
                }
                Self::Gradient(ends.map(tangent))
            }
            Mask::Curvature(ends) => ramped(ends, "curvatures", Self::Curvature)?,
            Mask::Facing(facing) => {
                if let Err(invalid) = facing.validate() {
                    return Err(BiomeError {
                        path: format!("{path}.facing.{}", invalid.field),
                        rule: invalid.rule.into_owned(),
                    });
                }
                let (sin, cos) = sin_cos(facing.heading.to_radians());
                Self::Facing {
                    toward: [-sin, cos],
                    ends: facing.ramp,
                }
            }
            Mask::Worn(ends) => ramped(ends, "depths worn", Self::Worn)?,
            Mask::Deposit(ends) => ramped(ends, "depths laid down", Self::Deposit)?,
            Mask::Flow(ends) => ramped(ends, "flows", Self::Flow)?,
            Mask::Wetness(ends) => ramped(ends, "wetness", Self::Wetness)?,
            Mask::Lake(ends) => ramped(ends, "lake depths", Self::Lake)?,
            Mask::River(ends) => ramped(ends, "river distances", Self::River)?,
            Mask::Grain(ends) => ramped(ends, "grain sizes", Self::Grain)?,
            Mask::Power(ends) => ramped(ends, "stream powers", Self::Power)?,
            Mask::Flood(ends) => ramped(ends, "heights above floods", Self::Flood)?,
            Mask::Path(ends) => ramped(ends, "path distances", Self::Path)?,
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

    pub(crate) fn reads(&self) -> Reads {
        match self {
            Self::Flow(_)
            | Self::Wetness(_)
            | Self::Lake(_)
            | Self::River(_)
            | Self::Grain(_)
            | Self::Power(_)
            | Self::Flood(_) => Reads {
                water: true,
                ..Reads::default()
            },
            Self::Path(_) => Reads {
                path: true,
                ..Reads::default()
            },
            Self::Worn(_) | Self::Deposit(_) => Reads {
                erosion: true,
                ..Reads::default()
            },
            Self::Constant(_)
            | Self::Height(_)
            | Self::Gradient(_)
            | Self::Curvature(_)
            | Self::Facing { .. }
            | Self::Noise { .. } => Reads::default(),
            Self::Multiply(masks) | Self::Add(masks) | Self::Min(masks) | Self::Max(masks) => masks
                .iter()
                .fold(Reads::default(), |reads, mask| reads.and(mask.reads())),
            Self::Invert(mask) | Self::Remap(mask, _) => mask.reads(),
        }
    }

    pub(crate) fn cost(&self) -> u64 {
        match self {
            Self::Constant(_) => 0,
            Self::Height(_)
            | Self::Gradient(_)
            | Self::Curvature(_)
            | Self::Facing { .. }
            | Self::Worn(_)
            | Self::Deposit(_)
            | Self::Flow(_)
            | Self::Wetness(_)
            | Self::Lake(_)
            | Self::River(_)
            | Self::Grain(_)
            | Self::Power(_)
            | Self::Flood(_)
            | Self::Path(_) => 1,
            Self::Noise { octaves, .. } => 2 * u64::from(*octaves),
            Self::Multiply(masks) | Self::Add(masks) | Self::Min(masks) | Self::Max(masks) => {
                masks.iter().map(Self::cost).sum::<u64>() + 1
            }
            Self::Invert(mask) | Self::Remap(mask, _) => mask.cost() + 1,
        }
    }

    pub(crate) fn evaluate_row(
        &self,
        row: &RowInputs<'_>,
        out: &mut [f32],
        scratch: &mut RowScratch,
    ) {
        let (gradients, waters) = (row.gradient.iter().copied(), row.water.iter());
        match self {
            Self::Constant(value) => out.fill(*value),
            Self::Height(ends) => ramps(out, row.height.iter().copied(), *ends),
            Self::Gradient(ends) => ramps(out, gradients.map(slope_of), *ends),
            Self::Curvature(ends) => ramps(out, row.curvature.iter().copied(), *ends),
            Self::Facing { toward, ends } => {
                ramps(
                    out,
                    gradients.map(|gradient| facing(*toward, gradient)),
                    *ends,
                );
            }
            Self::Worn(ends) => ramps(out, row.erosion.iter().copied().map(worn), *ends),
            Self::Deposit(ends) => ramps(out, row.erosion.iter().copied().map(deposit), *ends),
            Self::Flow(ends) => ramps(out, waters.map(|water| water.flow), *ends),
            Self::Wetness(ends) => ramps(out, waters.map(|water| water.wetness), *ends),
            Self::Lake(ends) => ramps(out, waters.map(|water| water.lake), *ends),
            Self::River(ends) => ramps(out, waters.map(|water| water.river), *ends),
            Self::Grain(ends) => ramps(out, waters.map(|water| water.grain), *ends),
            Self::Power(ends) => ramps(out, waters.map(|water| water.power), *ends),
            Self::Flood(ends) => ramps(out, waters.map(|water| water.flood), *ends),
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
            Self::Facing { toward, ends } => {
                out.extend([op::FACING, toward[0].to_bits(), toward[1].to_bits()]);
                out.extend(ends.map(f32::to_bits));
                1
            }
            Self::Worn(ends) => ramp(out, op::WORN, ends),
            Self::Deposit(ends) => ramp(out, op::DEPOSIT, ends),
            Self::Flow(ends) => ramp(out, op::FLOW, ends),
            Self::Wetness(ends) => ramp(out, op::WETNESS, ends),
            Self::Lake(ends) => ramp(out, op::LAKE, ends),
            Self::River(ends) => ramp(out, op::RIVER, ends),
            Self::Grain(ends) => ramp(out, op::GRAIN, ends),
            Self::Power(ends) => ramp(out, op::POWER, ends),
            Self::Flood(ends) => ramp(out, op::FLOOD, ends),
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
            Self::Gradient(ends) => ramp(slope_of(inputs.gradient), *ends),
            Self::Curvature(ends) => ramp(inputs.curvature, *ends),
            Self::Facing { toward, ends } => ramp(facing(*toward, inputs.gradient), *ends),
            Self::Worn(ends) => ramp(worn(inputs.erosion), *ends),
            Self::Deposit(ends) => ramp(deposit(inputs.erosion), *ends),
            Self::Flow(ends) => ramp(inputs.water.flow, *ends),
            Self::Wetness(ends) => ramp(inputs.water.wetness, *ends),
            Self::Lake(ends) => ramp(inputs.water.lake, *ends),
            Self::River(ends) => ramp(inputs.water.river, *ends),
            Self::Grain(ends) => ramp(inputs.water.grain, *ends),
            Self::Power(ends) => ramp(inputs.water.power, *ends),
            Self::Flood(ends) => ramp(inputs.water.flood, *ends),
            Self::Path(ends) => {
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
                    let shift = octave as f32 * 17.13;
                    let fade = clamp(0.5 / (scale * inputs.spacing) - 1.0, 0.0, 1.0);
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
    wetness: Option<Compiled>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BiomeRules {
    authored: Biomes,
    biomes: Vec<CompiledBiome>,
    steep: Option<Steep>,
    wet_biomes: usize,
    cost: u64,
    reads: Reads,
    placements: Vec<crate::placement::CompiledPlacement>,
    placed: Vec<crate::placement::CompiledAuthored>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Control {
    pub a: u8,
    pub b: u8,
    pub blend: u8,
    pub wet: u8,
}

impl Control {
    pub fn packed(self) -> u32 {
        u32::from(self.a)
            | u32::from(self.b) << 8
            | u32::from(self.blend) << 16
            | u32::from(self.wet) << 24
    }

    pub fn unpack(value: u32) -> Self {
        Self {
            a: value as u8,
            b: (value >> 8) as u8,
            blend: (value >> 16) as u8,
            wet: (value >> 24) as u8,
        }
    }
}

impl Biomes {
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
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
            let wetness = match &biome.wetness {
                Some(rule) => Some(Compiled::from(rule, &format!("{at}.wetness"), 0)?),
                None => None,
            };
            biomes.push(CompiledBiome {
                fitness,
                layers,
                wetness,
            });
        }
        let wet_biomes = if biomes.iter().any(|biome| biome.wetness.is_some()) {
            biomes.len()
        } else {
            0
        };
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
                biomes.push(CompiledBiome {
                    fitness: Compiled::Gradient(steep.gradient),
                    layers: vec![CompiledLayer {
                        material: steep.material,
                        mask: Compiled::Constant(1.0),
                        opacity: 1.0,
                    }],
                    wetness: None,
                });
                Some(steep)
            }
            None => None,
        };
        if let Err(invalid) = self.macro_colors.validate() {
            return Err(fail(
                format!("macro_colors.{}", invalid.field),
                &invalid.rule,
            ));
        }
        let cost = 1 + biomes
            .iter()
            .map(|biome| {
                biome.fitness.cost()
                    + biome.layers.iter().map(|l| l.mask.cost()).sum::<u64>()
                    + biome.wetness.as_ref().map_or(0, Compiled::cost)
            })
            .sum::<u64>();
        let placements = crate::placement::compile(&self.placements, &self.materials, &self.assets)
            .map_err(|(path, rule)| BiomeError { path, rule })?;
        let placed = crate::placement::compile_authored(&self.placed, &self.assets)
            .map_err(|(path, rule)| BiomeError { path, rule })?;
        let reads = biomes.iter().fold(Reads::default(), |reads, biome| {
            let reads = (biome.wetness.as_ref()).map_or(reads, |rule| reads.and(rule.reads()));
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
            wet_biomes,
            cost,
            reads,
            placements,
            placed,
        })
    }
}

impl BiomeRules {
    pub fn authored(&self) -> &Biomes {
        &self.authored
    }

    pub fn steep(&self) -> Option<Steep> {
        self.steep
    }

    pub(crate) fn reads(&self) -> Reads {
        self.reads
    }

    pub(crate) fn wet_biomes(&self) -> usize {
        self.wet_biomes
    }

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

    pub fn places(&self, level: u8, placing: crate::Placing) -> bool {
        let near = self.placements.iter().any(|layer| layer.level == level)
            || self.placed.iter().any(|object| object.level == level);
        let far = self.placements.iter().any(|layer| layer.far == Some(level));
        match placing {
            crate::Placing::All => near || far,
            crate::Placing::Near => near,
            crate::Placing::Far => far,
        }
    }

    pub fn place_authored(&self, tile: &crate::Tile, out: &mut Vec<crate::Placed>) -> u64 {
        crate::placement::place_authored(&self.placed, tile, out)
    }

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

    pub fn cost(&self) -> u64 {
        self.cost
    }

    pub(crate) fn program(&self, words: &mut Vec<u32>) -> (u32, u32, u32) {
        use crate::gpu::{BIOME_WORDS, op};
        let (table, entry) = (words.len(), BIOME_WORDS as usize);
        words.extend(std::iter::repeat_n(0, entry * self.biomes.len()));
        let (mut stack, mut materials) = (1, 0);
        let mut mask = |words: &mut Vec<u32>, compiled: &Compiled| {
            let at = words.len() as u32;
            stack = stack.max(compiled.emit(words));
            words.push(op::END);
            at
        };
        for (index, biome) in self.biomes.iter().enumerate() {
            let at = table + entry * index;
            words[at] = if index == 0 {
                u32::MAX
            } else {
                mask(words, &biome.fitness)
            };
            words[at + 2] = match &biome.wetness {
                Some(rule) => mask(words, rule),
                None => u32::MAX,
            };
            let list = words.len();
            words[at + 1] = list as u32;
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

    pub fn control(&self, inputs: &SampleInputs, weights: &mut Vec<(u8, f32)>) -> Control {
        self.control_with(
            |biome| self.biomes[biome].fitness.evaluate(inputs),
            |biome, layer| self.biomes[biome].layers[layer].mask.evaluate(inputs),
            |biome| (self.biomes[biome].wetness.as_ref()).map_or(0.0, |rule| rule.evaluate(inputs)),
            weights,
        )
    }

    pub(crate) fn control_row(
        &self,
        row: &RowInputs<'_>,
        out: &mut [u32],
        (scratch, weights): (&mut RowScratch, &mut Vec<(u8, f32)>),
    ) {
        let len = out.len();
        let mut values: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.biomes.len());
        for (index, biome) in self.biomes.iter().enumerate() {
            let mut rows = Vec::with_capacity(biome.layers.len() + 2);
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
            let mut wetness = scratch.take(if biome.wetness.is_some() { len } else { 0 });
            if let Some(rule) = &biome.wetness {
                rule.evaluate_row(row, &mut wetness, scratch);
            }
            rows.push(wetness);
            values.push(rows);
        }
        for (sample, control) in out.iter_mut().enumerate() {
            *control = self
                .control_with(
                    |biome| values[biome][0][sample],
                    |biome, layer| values[biome][layer + 1][sample],
                    |biome| values[biome][self.biomes[biome].layers.len() + 1][sample],
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

    fn control_with(
        &self,
        fitness: impl Fn(usize) -> f32,
        mask: impl Fn(usize, usize) -> f32,
        wetness: impl Fn(usize) -> f32,
        weights: &mut Vec<(u8, f32)>,
    ) -> Control {
        weights.clear();
        let mut remaining = 1.0_f32;
        let mut soaking = if self.wet_biomes > 0 { 1.0_f32 } else { 0.0 };
        let mut wet = 0.0_f32;
        for (index, compiled) in self.biomes.iter().enumerate().rev() {
            let fit = if index == 0 {
                1.0
            } else {
                clamp(fitness(index), 0.0, 1.0)
            };
            let share = fit * remaining;
            remaining *= 1.0 - fit;
            if index < self.wet_biomes {
                let held = fit * soaking;
                soaking *= 1.0 - fit;
                if held > 0.0 && compiled.wetness.is_some() {
                    wet += held * clamp(wetness(index), 0.0, 1.0);
                }
            }
            if share > 0.0 {
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
            if remaining <= 0.0 && soaking <= 0.0 {
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
            wet: (clamp(wet, 0.0, 1.0) * 255.0 + 0.5) as u8,
        }
    }
}

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
    16 => Grain(ends),
    17 => Power(ends),
    18 => Flood(ends),
    19 => Facing(facing),
    20 => Worn(ends),
    21 => Deposit(ends),
});

record!(Facing { heading, ramp });
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
    wetness,
});
record!(SteepLayer { material, slope });
record!(MacroCauses {
    wet,
    deposit,
    worn,
    steep,
    relief,
    dust,
    aspect,
    heading,
    broad,
    patchy,
    warmth,
});
record!(Biomes {
    materials,
    biomes,
    steep,
    assets,
    placements,
    placed,
    macro_colors,
});
