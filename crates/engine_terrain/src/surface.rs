//! Terrain surfaces (decision 054): which materials the ground is made of,
//! where. A surface names its materials and lists biomes; each biome has a
//! fitness mask saying where it applies and layers of materials, each
//! painted over the layers before it by its own mask. Every tile sample
//! gets its control value: the two materials weighing most there, their
//! blend and the biome weighing most, a pure function of the sample's
//! position, height, slope and curvature at its level, so neighboring tiles
//! agree on the samples they share.

use engine_math::{clamp, max, min, sin_cos};
use serde::{Deserialize, Serialize};

use crate::PathSample;
use crate::noise::{gradient_value, smooth};

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
    /// A ramp over the ground's slope, in degrees from level, 0 to 90.
    Slope([f32; 2]),
    /// A ramp over the ground's curvature: how fast its slope grows per
    /// meter, positive in hollows and valleys, negative on ridges.
    Curvature([f32; 2]),
    /// A ramp over the base-10 logarithm of the area draining through the
    /// ground, in square meters, as the terrain's hydrology bake found it
    /// (decision 066): channels begin near its channel area. 0 without one.
    Flow([f32; 2]),
    /// A ramp over the bake's wetness index, ln(a / tan b): about 5 on
    /// steep ground, 10 and more in hollows and along streams.
    Wetness([f32; 2]),
    /// A ramp over how deep water would stand in a filled pit, in meters.
    Lake([f32; 2]),
    /// A ramp over meters from the nearest river's edge (decision 092),
    /// negative in its water: `[2, 0]` covers river beds with two meters of
    /// banks. Beyond every river's reach (sixteen meters past its edge),
    /// and without rivers, it reads 1e9.
    River([f32; 2]),
    /// A ramp over meters from the nearest road's edge (decision 069),
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

/// Gradient noise from 0 to 1 with features `wavelength` meters across,
/// summed over `octaves` (1 to 8), each half as long and half as strong.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct NoiseMask {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.01, max = 1.0e6)))]
    pub wavelength: f32,
    #[serde(default)]
    pub seed: u32,
    #[serde(default = "one_octave")]
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 8)))]
    pub octaves: u32,
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

/// One layer of a biome's ground: `material` (an ID of the surface's
/// materials) painted over the layers before it where `mask` allows, at
/// most `opacity`. A biome's first layer covers its whole ground.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct BiomeLayer {
    pub material: String,
    #[serde(default = "Mask::one", skip_serializing_if = "is_one")]
    pub mask: Mask,
    #[serde(default = "one", skip_serializing_if = "is_unit")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub opacity: f32,
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

/// A surface as authored: its materials' IDs, in the order control values
/// index them, its biomes, and the IDs of the assets its placement layers
/// stand on it (decision 057), in the order placements index them.
#[derive(Clone, Debug, PartialEq)]
pub struct Biomes {
    pub materials: Vec<String>,
    pub biomes: Vec<Biome>,
    pub assets: Vec<String>,
    pub placements: Vec<crate::PlacementLayer>,
    /// Objects authors stand on it (decision 085).
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

/// What a sample's masks read: its terrain-local position in meters, its
/// height above the terrain's origin, the length of its height gradient
/// (the slope's tangent), its curvature per meter, and the meters between
/// samples at its level, below twice which noise fades to its mean, so
/// coarse levels show the same patches as fine ones, filtered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleInputs {
    pub x: f32,
    pub z: f32,
    pub height: f32,
    pub gradient: f32,
    pub curvature: f32,
    pub spacing: f32,
    /// What the terrain's hydrology bake found there (decision 066); zeros
    /// without one.
    pub water: crate::HydrologySample,
    /// The nearest road (decision 069), as [`crate::Layers::path_at`]
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
                if !ends
                    .iter()
                    .all(|end| end.is_finite() && (0.0..=90.0).contains(end))
                {
                    return Err(fail("slopes must be in [0, 90] degrees"));
                }
                // As gradients: the tangent of each slope, 90 degrees
                // standing in for a vertical wall.
                Self::Gradient(ends.map(|degrees| {
                    let (sin, cos) = sin_cos(min(degrees, 89.9).to_radians());
                    sin / cos
                }))
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
                if !(noise.wavelength.is_finite() && (0.01..=1.0e6).contains(&noise.wavelength)) {
                    return Err(fail("noise wavelength must be in [0.01, 1e6] meters"));
                }
                if !(1..=MAX_NOISE_OCTAVES).contains(&noise.octaves) {
                    return Err(fail("noise octaves must be in [1, 8]"));
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
                    sum += amplitude
                        * fade
                        * gradient_value(
                            inputs.x * scale + shift,
                            inputs.z * scale - shift,
                            seed.wrapping_add(octave.wrapping_mul(0x9e37_79b9)),
                        );
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

/// A validated surface, ready to evaluate per sample.
#[derive(Clone, Debug, PartialEq)]
pub struct BiomeRules {
    authored: Biomes,
    biomes: Vec<CompiledBiome>,
    cost: u64,
    placements: Vec<crate::placement::CompiledPlacement>,
    placed: Vec<crate::placement::CompiledAuthored>,
}

/// A sample's control value: the material weighing most (`a`), the one
/// weighing second (`b`, `a` again when none), `b`'s share of the two from
/// 0 to 255, and the biome weighing most.
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
    /// naming its materials.
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
                if !(layer.opacity.is_finite() && (0.0..=1.0).contains(&layer.opacity)) {
                    return Err(fail(format!("{path}.opacity"), "must be in [0, 1]"));
                }
                layers.push(CompiledLayer {
                    material: material as u8,
                    mask: Compiled::from(&layer.mask, &format!("{path}.mask"), 0)?,
                    opacity: layer.opacity,
                });
            }
            biomes.push(CompiledBiome { fitness, layers });
        }
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
        Ok(BiomeRules {
            authored: self.clone(),
            biomes,
            cost,
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

    /// These rules choosing materials alone, placing no objects: what a
    /// terrain's surroundings (decision 094) follow.
    pub fn without_placements(&self) -> Self {
        let mut rules = self.clone();
        rules.authored.assets.clear();
        rules.authored.placements.clear();
        rules.authored.placed.clear();
        rules.placements.clear();
        rules.placed.clear();
        rules
    }

    /// Whether any placement layer or authored object (decision 085)
    /// stands on tiles of `level`.
    pub fn places_on(&self, level: u8) -> bool {
        self.placements
            .iter()
            .any(|layer| layer.level == level || layer.far.is_some_and(|(far, _)| far == level))
            || self.placed.iter().any(|object| object.level == level)
    }

    /// The authored objects (decision 085) standing on `tile`, appended to
    /// `out`; returns the work it took.
    pub fn place_authored(&self, tile: &crate::Tile, out: &mut Vec<crate::Placed>) -> u64 {
        crate::placement::place_authored(&self.placed, tile, out)
    }

    /// The objects row `row` of placement layer `layer`'s candidate squares
    /// stand on `tile` (decision 057), appended to `out`: layers of other
    /// levels than `tile`'s hold none. Returns the work it took (samples'
    /// worth) and the next row, none after the last layer's last.
    pub fn place_row(
        &self,
        tile: &crate::Tile,
        (layer, row): (usize, usize),
        out: &mut Vec<crate::Placed>,
        layers: &crate::Layers,
    ) -> (u64, Option<(usize, usize)>) {
        crate::placement::place_row(
            &self.placements,
            tile,
            0x5eed_0057,
            (layer, row),
            out,
            layers,
        )
    }

    /// Work units of one sample's control value, against one per sample
    /// of a layer.
    pub fn cost(&self) -> u64 {
        self.cost
    }

    /// The control value at a sample with `inputs`. `weights` is scratch
    /// space, reused across samples.
    pub fn control(&self, inputs: &SampleInputs, weights: &mut Vec<(u8, f32)>) -> Control {
        weights.clear();
        // Later biomes and layers paint over earlier ones: walk them from the
        // last, each taking what the later ones left, until nothing is left.
        let mut remaining = 1.0_f32;
        let mut biome = (0_usize, -1.0_f32);
        for (index, compiled) in self.biomes.iter().enumerate().rev() {
            let fit = if index == 0 {
                1.0
            } else {
                clamp(compiled.fitness.evaluate(inputs), 0.0, 1.0)
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
                        clamp(layer.mask.evaluate(inputs) * layer.opacity, 0.0, 1.0)
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

// Binary encoding (decision 110).
use crate::codec::{Codec, CodecError, Reader, Writer, record};

impl Codec for Mask {
    fn write(&self, out: &mut Writer) {
        match self {
            Self::Constant(value) => out.put(&0_u8).put(value),
            Self::Height(ends) => out.put(&1_u8).put(ends),
            Self::Slope(ends) => out.put(&2_u8).put(ends),
            Self::Curvature(ends) => out.put(&3_u8).put(ends),
            Self::Flow(ends) => out.put(&4_u8).put(ends),
            Self::Wetness(ends) => out.put(&5_u8).put(ends),
            Self::Lake(ends) => out.put(&6_u8).put(ends),
            Self::River(ends) => out.put(&7_u8).put(ends),
            Self::Path(ends) => out.put(&8_u8).put(ends),
            Self::Noise(noise) => out.put(&9_u8).put(noise),
            Self::Multiply(masks) => out.put(&10_u8).put(masks),
            Self::Add(masks) => out.put(&11_u8).put(masks),
            Self::Min(masks) => out.put(&12_u8).put(masks),
            Self::Max(masks) => out.put(&13_u8).put(masks),
            Self::Invert(mask) => out.put(&14_u8).put(mask),
            Self::Remap(remap) => out.put(&15_u8).put(remap),
        };
    }

    fn read(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        Ok(match input.get::<u8>()? {
            0 => Self::Constant(input.get()?),
            1 => Self::Height(input.get()?),
            2 => Self::Slope(input.get()?),
            3 => Self::Curvature(input.get()?),
            4 => Self::Flow(input.get()?),
            5 => Self::Wetness(input.get()?),
            6 => Self::Lake(input.get()?),
            7 => Self::River(input.get()?),
            8 => Self::Path(input.get()?),
            9 => Self::Noise(input.get()?),
            10 => Self::Multiply(input.get()?),
            11 => Self::Add(input.get()?),
            12 => Self::Min(input.get()?),
            13 => Self::Max(input.get()?),
            14 => Self::Invert(input.get()?),
            15 => Self::Remap(input.get()?),
            _ => return Err(CodecError("invalid mask")),
        })
    }
}

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
record!(Biomes {
    materials,
    biomes,
    assets,
    placements,
    placed,
});
