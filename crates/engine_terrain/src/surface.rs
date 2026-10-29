use engine_core::{Error, ErrorCode, Expected};
use engine_math::{clamp, max, min, sin_cos};
use engine_settings::{ExtraRules, InvalidSetting, Rule, Setting, settings};
use serde::{Deserialize, Serialize};

use crate::PathSample;
use crate::expr::{Expr, MAX_EXPR_DEPTH, MAX_EXPR_LIST};
use crate::gpu::op;
use crate::layer::maps;
use crate::layers::Layers;
use crate::noise::{gradient_value, gradient_values, smooth};

pub const MAX_SURFACE_MATERIALS: usize = 255;
pub const MAX_BIOMES: usize = 255;
const MAX_NOISE_OCTAVES: u32 = 8;
const MAX_BIOME_LAYERS: usize = 64;

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

fn is_one(expr: &Expr) -> bool {
    *expr == Expr::one()
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
        #[serde(default = "Expr::one", skip_serializing_if = "is_one")]
        pub mask: Expr = Expr::one() => any,
        #[serde(default = "one", skip_serializing_if = "is_unit")]
        pub opacity: f32 = one() => range(0.0, 1.0),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Biome {
    pub id: String,
    #[serde(default = "Expr::one", skip_serializing_if = "is_one")]
    pub fitness: Expr,
    pub layers: Vec<BiomeLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wetness: Option<Expr>,
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
                engine_settings::Expected::Sorted,
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

pub(crate) fn terrain_error(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidTerrain, path)
}

pub(crate) fn setting_error(invalid: InvalidSetting, at: &str) -> Error {
    Error {
        code: ErrorCode::InvalidTerrain,
        ..Error::from(invalid).within(at)
    }
}

pub(crate) fn unknown(path: impl Into<String>, got: &str, names: &[String]) -> Error {
    Error::new(ErrorCode::UnknownReference, path)
        .got(got)
        .variants(names)
}

pub(crate) fn unknown_map(path: &str, got: &str, published: Vec<String>) -> Error {
    Error::new(ErrorCode::UnknownMap, path).got(got).variants(
        maps::GPU
            .iter()
            .map(|name| (*name).to_owned())
            .chain(published),
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Reads {
    pub(crate) water: bool,
    pub(crate) path: bool,
    pub(crate) erosion: bool,
}

impl Reads {
    pub(crate) fn and(self, other: Self) -> Self {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Input {
    Height,
    Slope,
    Curvature,
    Flow,
    Wetness,
    Lake,
    River,
    Grain,
    Power,
    Flood,
    Worn,
    Deposit,
    Path,
}

const INPUTS: [(&str, Input); 13] = [
    (maps::HEIGHT, Input::Height),
    (maps::SLOPE, Input::Slope),
    (maps::CURVATURE, Input::Curvature),
    (maps::FLOW, Input::Flow),
    (maps::WETNESS, Input::Wetness),
    (maps::LAKE, Input::Lake),
    (maps::RIVER, Input::River),
    (maps::GRAIN, Input::Grain),
    (maps::POWER, Input::Power),
    (maps::FLOOD, Input::Flood),
    (maps::WORN, Input::Worn),
    (maps::DEPOSIT, Input::Deposit),
    (maps::PATH, Input::Path),
];

impl Input {
    pub(crate) fn named(name: &str) -> Option<Self> {
        INPUTS
            .iter()
            .find(|(known, _)| *known == name)
            .map(|&(_, input)| input)
    }

    fn op(self) -> u32 {
        match self {
            Self::Height => op::HEIGHT,
            Self::Slope => op::GRADIENT,
            Self::Curvature => op::CURVATURE,
            Self::Flow => op::FLOW,
            Self::Wetness => op::WETNESS,
            Self::Lake => op::LAKE,
            Self::River => op::RIVER,
            Self::Grain => op::GRAIN,
            Self::Power => op::POWER,
            Self::Flood => op::FLOOD,
            Self::Worn => op::WORN,
            Self::Deposit => op::DEPOSIT,
            Self::Path => op::PATH,
        }
    }

    fn reads(self) -> Reads {
        match self {
            Self::Flow
            | Self::Wetness
            | Self::Lake
            | Self::River
            | Self::Grain
            | Self::Power
            | Self::Flood => Reads {
                water: true,
                ..Reads::default()
            },
            Self::Path => Reads {
                path: true,
                ..Reads::default()
            },
            Self::Worn | Self::Deposit => Reads {
                erosion: true,
                ..Reads::default()
            },
            Self::Height | Self::Slope | Self::Curvature => Reads::default(),
        }
    }

    fn shaped(self) -> bool {
        match self {
            Self::Slope | Self::Curvature => true,
            Self::Height
            | Self::Flow
            | Self::Wetness
            | Self::Lake
            | Self::River
            | Self::Grain
            | Self::Power
            | Self::Flood
            | Self::Worn
            | Self::Deposit
            | Self::Path => false,
        }
    }

    fn at(self, inputs: &SampleInputs) -> f32 {
        match self {
            Self::Height => inputs.height,
            Self::Slope => slope_of(inputs.gradient),
            Self::Curvature => inputs.curvature,
            Self::Flow => inputs.water.flow,
            Self::Wetness => inputs.water.wetness,
            Self::Lake => inputs.water.lake,
            Self::River => inputs.water.river,
            Self::Grain => inputs.water.grain,
            Self::Power => inputs.water.power,
            Self::Flood => inputs.water.flood,
            Self::Worn => worn(inputs.erosion),
            Self::Deposit => deposit(inputs.erosion),
            Self::Path => path_value(inputs.path),
        }
    }

    fn ramp_row(self, row: &RowInputs<'_>, out: &mut [f32], ends: [f32; 2]) {
        let (gradients, waters) = (row.gradient.iter().copied(), row.water.iter());
        match self {
            Self::Height => ramps(out, row.height.iter().copied(), ends),
            Self::Slope => ramps(out, gradients.map(slope_of), ends),
            Self::Curvature => ramps(out, row.curvature.iter().copied(), ends),
            Self::Worn => ramps(out, row.erosion.iter().copied().map(worn), ends),
            Self::Deposit => ramps(out, row.erosion.iter().copied().map(deposit), ends),
            Self::Flow => ramps(out, waters.map(|water| water.flow), ends),
            Self::Wetness => ramps(out, waters.map(|water| water.wetness), ends),
            Self::Lake => ramps(out, waters.map(|water| water.lake), ends),
            Self::River => ramps(out, waters.map(|water| water.river), ends),
            Self::Grain => ramps(out, waters.map(|water| water.grain), ends),
            Self::Power => ramps(out, waters.map(|water| water.power), ends),
            Self::Flood => ramps(out, waters.map(|water| water.flood), ends),
            Self::Path => {
                for (value, &sample) in out.iter_mut().zip(row.path) {
                    *value = path_ramp(sample, row.spacing, ends);
                }
            }
        }
    }

    fn value_row(self, row: &RowInputs<'_>, out: &mut [f32]) {
        let (gradients, waters) = (row.gradient.iter().copied(), row.water.iter());
        match self {
            Self::Height => fill_row(out, row.height.iter().copied()),
            Self::Slope => fill_row(out, gradients.map(slope_of)),
            Self::Curvature => fill_row(out, row.curvature.iter().copied()),
            Self::Worn => fill_row(out, row.erosion.iter().copied().map(worn)),
            Self::Deposit => fill_row(out, row.erosion.iter().copied().map(deposit)),
            Self::Flow => fill_row(out, waters.map(|water| water.flow)),
            Self::Wetness => fill_row(out, waters.map(|water| water.wetness)),
            Self::Lake => fill_row(out, waters.map(|water| water.lake)),
            Self::River => fill_row(out, waters.map(|water| water.river)),
            Self::Grain => fill_row(out, waters.map(|water| water.grain)),
            Self::Power => fill_row(out, waters.map(|water| water.power)),
            Self::Flood => fill_row(out, waters.map(|water| water.flood)),
            Self::Path => fill_row(out, row.path.iter().copied().map(path_value)),
        }
    }
}

pub(crate) struct Scope {
    gpu: bool,
    users: Vec<(String, String)>,
}

impl Scope {
    pub(crate) fn gpu() -> Self {
        Self {
            gpu: true,
            users: Vec::new(),
        }
    }

    pub(crate) fn cpu() -> Self {
        Self {
            gpu: false,
            users: Vec::new(),
        }
    }

    pub(crate) fn users(self) -> Vec<(String, String)> {
        self.users
    }

    fn user(&mut self, name: &str, path: String) -> Result<usize, Error> {
        if self.gpu {
            return Err(Error::new(ErrorCode::UnreadableMap, path)
                .got(name)
                .variants(maps::GPU));
        }
        if let Some(index) = self.users.iter().position(|(known, _)| known == name) {
            return Ok(index);
        }
        self.users.push((name.to_owned(), path));
        Ok(self.users.len() - 1)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Compiled {
    Constant(f32),
    Ramp(Input, [f32; 2]),
    Value(Input),
    User {
        map: usize,
        ends: Option<[f32; 2]>,
    },
    Facing {
        toward: [f32; 2],
        ends: [f32; 2],
    },
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

fn fill_row(out: &mut [f32], values: impl IntoIterator<Item = f32>) {
    for (value, input) in out.iter_mut().zip(values) {
        *value = input;
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

#[inline]
fn path_value(PathSample { distance, half }: PathSample) -> f32 {
    distance.abs() - half
}

#[inline]
fn path_ramp(PathSample { distance, half }: PathSample, spacing: f32, ends: [f32; 2]) -> f32 {
    [-0.375, -0.125, 0.125, 0.375]
        .iter()
        .map(|offset| ramp((distance + offset * spacing).abs() - half, ends))
        .sum::<f32>()
        * 0.25
}

pub(crate) fn erosion_row(layers: &Layers, xs: &[f32], z: f32, spacing: f32, out: &mut [f32]) {
    out.fill(0.0);
    for change in layers.changes() {
        change.change_row(xs.iter().copied(), z, spacing, |index, value| {
            out[index] += value;
        });
    }
}

fn fold(
    masks: &[Compiled],
    (row, maps): (&RowInputs<'_>, &[Vec<f32>]),
    out: &mut [f32],
    scratch: &mut RowScratch,
    start: f32,
    op: impl Fn(f32, f32) -> f32,
) {
    out.fill(start);
    let mut part = scratch.take(out.len());
    for mask in masks {
        mask.evaluate_row(row, maps, &mut part, scratch);
        for (value, &next) in out.iter_mut().zip(&part) {
            *value = op(*value, next);
        }
    }
    scratch.give(part);
}

impl Compiled {
    pub(crate) fn from(
        expr: &Expr,
        path: &str,
        depth: usize,
        scope: &mut Scope,
    ) -> Result<Self, Error> {
        if depth > MAX_EXPR_DEPTH {
            return Err(terrain_error(path)
                .at_most(MAX_EXPR_DEPTH as u64)
                .got(depth));
        }
        Ok(match expr {
            Expr::Constant(value) => {
                if !value.is_finite() {
                    return Err(terrain_error(format!("{path}.constant"))
                        .expected(Expected::Finite)
                        .got(value));
                }
                Self::Constant(*value)
            }
            Expr::Map(name) => match Input::named(name) {
                Some(input) => Self::Value(input),
                None => Self::User {
                    map: scope.user(name, format!("{path}.map"))?,
                    ends: None,
                },
            },
            Expr::Ramp(ramp) => {
                let at = format!("{path}.ramp");
                let ends = [ramp.from, ramp.to];
                let input = Input::named(&ramp.map);
                if input == Some(Input::Slope) {
                    for (value, field) in [(ramp.from, "from"), (ramp.to, "to")] {
                        if let Err(invalid) =
                            value.check(field, Rule::Range(0.0, MAX_SLOPE_DEGREES as f64))
                        {
                            return Err(setting_error(invalid, &at));
                        }
                    }
                    return Ok(Self::Ramp(Input::Slope, ends.map(tangent)));
                }
                if !(ramp.from.is_finite() && ramp.to.is_finite()) {
                    return Err(terrain_error(at).expected(Expected::Finite).got(ends));
                }
                match input {
                    Some(input) => Self::Ramp(input, ends),
                    None => Self::User {
                        map: scope.user(&ramp.map, format!("{at}.map"))?,
                        ends: Some(ends),
                    },
                }
            }
            Expr::Facing(facing) => {
                if let Err(invalid) = facing.validate() {
                    return Err(setting_error(invalid, &format!("{path}.facing")));
                }
                let (sin, cos) = sin_cos(facing.heading.to_radians());
                Self::Facing {
                    toward: [-sin, cos],
                    ends: facing.ramp,
                }
            }
            Expr::Noise(noise) => {
                if let Err(invalid) = noise.validate() {
                    return Err(setting_error(invalid, &format!("{path}.noise")));
                }
                Self::Noise {
                    frequency: 1.0 / noise.wavelength,
                    seed: noise.seed,
                    octaves: noise.octaves,
                }
            }
            Expr::Mul(list) => Self::Multiply(Self::list(list, (path, "mul"), depth, scope)?),
            Expr::Add(list) => Self::Add(Self::list(list, (path, "add"), depth, scope)?),
            Expr::Min(list) => Self::Min(Self::list(list, (path, "min"), depth, scope)?),
            Expr::Max(list) => Self::Max(Self::list(list, (path, "max"), depth, scope)?),
            Expr::Invert(inner) => Self::Invert(Box::new(Self::from(
                inner,
                &format!("{path}.invert"),
                depth + 1,
                scope,
            )?)),
            Expr::Remap(remap) => {
                if !(remap.from.is_finite() && remap.to.is_finite()) {
                    return Err(terrain_error(format!("{path}.remap"))
                        .expected(Expected::Finite)
                        .got([remap.from, remap.to]));
                }
                Self::Remap(
                    Box::new(Self::from(
                        &remap.of,
                        &format!("{path}.remap.of"),
                        depth + 1,
                        scope,
                    )?),
                    [remap.from, remap.to],
                )
            }
        })
    }

    fn gpu(expr: &Expr, path: &str) -> Result<Self, Error> {
        Self::from(expr, path, 0, &mut Scope::gpu())
    }

    fn list(
        exprs: &[Expr],
        (path, name): (&str, &str),
        depth: usize,
        scope: &mut Scope,
    ) -> Result<Vec<Self>, Error> {
        if exprs.is_empty() || exprs.len() > MAX_EXPR_LIST {
            return Err(terrain_error(format!("{path}.{name}"))
                .range(1.0, MAX_EXPR_LIST as f64)
                .got(exprs.len()));
        }
        exprs
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                Self::from(expr, &format!("{path}.{name}[{index}]"), depth + 1, scope)
            })
            .collect()
    }

    pub(crate) fn reads(&self) -> Reads {
        match self {
            Self::Ramp(input, _) | Self::Value(input) => input.reads(),
            Self::Constant(_) | Self::User { .. } | Self::Facing { .. } | Self::Noise { .. } => {
                Reads::default()
            }
            Self::Multiply(masks) | Self::Add(masks) | Self::Min(masks) | Self::Max(masks) => masks
                .iter()
                .fold(Reads::default(), |reads, mask| reads.and(mask.reads())),
            Self::Invert(mask) | Self::Remap(mask, _) => mask.reads(),
        }
    }

    pub(crate) fn shaped(&self) -> bool {
        match self {
            Self::Ramp(input, _) | Self::Value(input) => input.shaped(),
            Self::Facing { .. } => true,
            Self::Constant(_) | Self::User { .. } | Self::Noise { .. } => false,
            Self::Multiply(masks) | Self::Add(masks) | Self::Min(masks) | Self::Max(masks) => {
                masks.iter().any(Self::shaped)
            }
            Self::Invert(mask) | Self::Remap(mask, _) => mask.shaped(),
        }
    }

    pub(crate) fn cost(&self) -> u64 {
        match self {
            Self::Constant(_) => 0,
            Self::Ramp(..) | Self::Value(_) | Self::User { .. } | Self::Facing { .. } => 1,
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
        maps: &[Vec<f32>],
        out: &mut [f32],
        scratch: &mut RowScratch,
    ) {
        match self {
            Self::Constant(value) => out.fill(*value),
            Self::Ramp(input, ends) => input.ramp_row(row, out, *ends),
            Self::Value(input) => input.value_row(row, out),
            Self::User { map, ends } => {
                out.fill(0.0);
                if let Some(values) = maps.get(*map) {
                    fill_row(out, values.iter().copied());
                }
                if let Some(ends) = ends {
                    for value in out.iter_mut() {
                        *value = ramp(*value, *ends);
                    }
                }
            }
            Self::Facing { toward, ends } => {
                ramps(
                    out,
                    (row.gradient.iter().copied()).map(|gradient| facing(*toward, gradient)),
                    *ends,
                );
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
            Self::Multiply(masks) => fold(masks, (row, maps), out, scratch, 1.0, |a, b| a * b),
            Self::Add(masks) => fold(masks, (row, maps), out, scratch, -0.0, |a, b| a + b),
            Self::Min(masks) => fold(masks, (row, maps), out, scratch, f32::INFINITY, min),
            Self::Max(masks) => fold(masks, (row, maps), out, scratch, f32::NEG_INFINITY, max),
            Self::Invert(mask) => {
                mask.evaluate_row(row, maps, out, scratch);
                for value in out.iter_mut() {
                    *value = 1.0 - *value;
                }
            }
            Self::Remap(mask, ends) => {
                mask.evaluate_row(row, maps, out, scratch);
                for value in out.iter_mut() {
                    *value = ramp(*value, *ends);
                }
            }
        }
    }

    pub(crate) fn emit(&self, out: &mut Vec<u32>) -> u32 {
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
            Self::Ramp(input, ends) => ramp(out, input.op(), ends),
            Self::Value(input) => {
                out.extend([op::VALUE, input.op()]);
                1
            }
            Self::User { .. } => {
                out.extend([op::CONSTANT, 0.0_f32.to_bits()]);
                1
            }
            Self::Facing { toward, ends } => {
                out.extend([op::FACING, toward[0].to_bits(), toward[1].to_bits()]);
                out.extend(ends.map(f32::to_bits));
                1
            }
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
            Self::Ramp(Input::Path, ends) => path_ramp(inputs.path, inputs.spacing, *ends),
            Self::Ramp(input, ends) => ramp(input.at(inputs), *ends),
            Self::Value(input) => input.at(inputs),
            Self::User { ends, .. } => ends.map_or(0.0, |ends| ramp(0.0, ends)),
            Self::Facing { toward, ends } => ramp(facing(*toward, inputs.gradient), *ends),
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
    pub fn compile(&self) -> Result<BiomeRules, Error> {
        if self.materials.is_empty() || self.materials.len() > MAX_SURFACE_MATERIALS {
            return Err(terrain_error("$.materials")
                .range(1.0, MAX_SURFACE_MATERIALS as f64)
                .got(self.materials.len()));
        }
        for (index, id) in self.materials.iter().enumerate() {
            if self.materials[..index].contains(id) {
                return Err(terrain_error(format!("$.materials[{index}]"))
                    .expected(Expected::Unique)
                    .got(id));
            }
        }
        if self.biomes.is_empty() || self.biomes.len() > MAX_BIOMES {
            return Err(terrain_error("$.biomes")
                .range(1.0, MAX_BIOMES as f64)
                .got(self.biomes.len()));
        }
        let mut biomes = Vec::with_capacity(self.biomes.len());
        for (index, biome) in self.biomes.iter().enumerate() {
            let at = format!("$.biomes[{index}]");
            if self.biomes[..index]
                .iter()
                .any(|other| other.id == biome.id)
            {
                return Err(terrain_error(format!("{at}.id"))
                    .expected(Expected::Unique)
                    .got(&biome.id));
            }
            if biome.layers.is_empty() || biome.layers.len() > MAX_BIOME_LAYERS {
                return Err(terrain_error(format!("{at}.layers"))
                    .range(1.0, MAX_BIOME_LAYERS as f64)
                    .got(biome.layers.len()));
            }
            let fitness = Compiled::gpu(&biome.fitness, &format!("{at}.fitness"))?;
            let mut layers = Vec::with_capacity(biome.layers.len());
            for (layer_index, layer) in biome.layers.iter().enumerate() {
                let path = format!("{at}.layers[{layer_index}]");
                let material = self
                    .materials
                    .iter()
                    .position(|id| *id == layer.material)
                    .ok_or_else(|| {
                        unknown(format!("{path}.material"), &layer.material, &self.materials)
                    })?;
                if let Err(invalid) = layer.validate() {
                    return Err(setting_error(invalid, &path));
                }
                layers.push(CompiledLayer {
                    material: material as u8,
                    mask: Compiled::gpu(&layer.mask, &format!("{path}.mask"))?,
                    opacity: layer.opacity,
                });
            }
            let wetness = match &biome.wetness {
                Some(rule) => Some(Compiled::gpu(rule, &format!("{at}.wetness"))?),
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
                    return Err(setting_error(invalid, "$.steep"));
                }
                let material = self
                    .materials
                    .iter()
                    .position(|id| *id == layer.material)
                    .ok_or_else(|| unknown("$.steep.material", &layer.material, &self.materials))?;
                let steep = Steep {
                    material: material as u8,
                    gradient: layer.slope.map(tangent),
                };
                biomes.push(CompiledBiome {
                    fitness: Compiled::Ramp(Input::Slope, steep.gradient),
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
            return Err(setting_error(invalid, "$.macro_colors"));
        }
        let cost = 1 + biomes
            .iter()
            .map(|biome| {
                biome.fitness.cost()
                    + biome.layers.iter().map(|l| l.mask.cost()).sum::<u64>()
                    + biome.wetness.as_ref().map_or(0, Compiled::cost)
            })
            .sum::<u64>();
        let placements =
            crate::placement::compile(&self.placements, &self.materials, &self.assets)?;
        let placed = crate::placement::compile_authored(&self.placed, &self.assets)?;
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

    pub fn check_maps(&self, layers: &Layers) -> Result<(), Error> {
        crate::placement::check_maps(&self.placements, layers)
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
        use crate::gpu::BIOME_WORDS;
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
                biome.fitness.evaluate_row(row, &[], &mut fitness, scratch);
            }
            rows.push(fitness);
            for (layer_index, layer) in biome.layers.iter().enumerate() {
                let mut mask = scratch.take(if layer_index == 0 { 0 } else { len });
                if layer_index > 0 {
                    layer.mask.evaluate_row(row, &[], &mut mask, scratch);
                }
                rows.push(mask);
            }
            let mut wetness = scratch.take(if biome.wetness.is_some() { len } else { 0 });
            if let Some(rule) = &biome.wetness {
                rule.evaluate_row(row, &[], &mut wetness, scratch);
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

use crate::codec::record;

record!(Facing { heading, ramp });
record!(NoiseMask {
    wavelength,
    seed,
    octaves,
});
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
