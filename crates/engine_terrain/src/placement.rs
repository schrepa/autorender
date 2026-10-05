//! Terrain placement (decision 057): objects standing on a terrain's
//! surface, layer by layer. A layer lays a grid of candidate squares a
//! footprint wide over the terrain; each square holds one candidate,
//! placed, ranked and varied by a hash of the layer and the square, so
//! candidates stay where they are however tiles stream and whoever asks.
//! A candidate stands where its rank falls under the layer's density at its
//! point: the share of a square the density fills, times the layer's mask
//! and the share of the surface materials it stands on there. Placement
//! is a pure function of a tile's heights and control values; each
//! candidate belongs to the tile of the layer's level holding its point.
//! Coarser levels stay resident farther from the camera, so their layers'
//! objects show farther away; renderers set objects on the finest heights
//! resident.

use engine_math::{clamp, max, sin_cos};
use serde::{Deserialize, Serialize};

use crate::surface::{Compiled, Control, Mask, SampleInputs};
use crate::{TILE_QUADS, Tile};

/// Validity guards: how many layers a surface places, how many assets a
/// layer chooses from, and the coarsest level a layer's tiles may be.
const MAX_LAYERS: usize = 64;
const MAX_CHOICES: usize = 64;
const MAX_LEVEL: u8 = 15;

/// An asset a placement layer chooses, by ID of the surface's assets, and
/// how often against the layer's others.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PlacementChoice {
    pub asset: String,
    #[serde(default = "one")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub weight: f32,
}

/// One layer of objects on a surface: the assets it chooses from; objects
/// per hectare where everything allows (at most one per footprint
/// square); the meters between candidates (0.25 to 256); the surface
/// materials it stands on (IDs; any when empty), weighted by their share;
/// where its mask allows; its objects' scale range; how far they lean with
/// the ground (0 upright, 1 along its normal); how deep they sink, in
/// meters at scale 1; and the pyramid level whose tiles hold its
/// candidates (0, the finest, by default).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PlacementLayer {
    pub id: String,
    pub assets: Vec<PlacementChoice>,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub density: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.25, max = 256.0)))]
    pub footprint: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<String>,
    #[serde(default = "Mask::one")]
    pub mask: Mask,
    #[serde(default = "unit_scale")]
    pub scale: [f32; 2],
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub align: f32,
    #[serde(default)]
    pub sink: f32,
    #[serde(default, skip_serializing_if = "is_zero")]
    #[cfg_attr(feature = "schema", schemars(range(max = 15)))]
    pub level: u8,
}

fn is_zero(level: &u8) -> bool {
    *level == 0
}

fn one() -> f32 {
    1.0
}

fn unit_scale() -> [f32; 2] {
    [1.0, 1.0]
}

/// One object a layer placed: its base's terrain-local position, its turn
/// about the vertical (radians), how its up leans (x and z of its up
/// vector), its scale, how far its base sinks below the ground (meters),
/// and its asset and layer indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub position: [f32; 3],
    pub yaw: f32,
    pub lean: [f32; 2],
    pub scale: f32,
    pub sink: f32,
    pub asset: u16,
    pub layer: u16,
}

/// A compiled layer: its tiles' level, its asset indices and cumulative
/// weights, the share of a square its density fills, its materials as a
/// 256-bit set (none: any), its mask and its variations.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompiledPlacement {
    pub(crate) level: u8,
    choices: Vec<(u16, f32)>,
    fill: f32,
    footprint: f32,
    materials: Option<[u64; 4]>,
    mask: Compiled,
    scale: [f32; 2],
    align: f32,
    sink: f32,
}

/// Validates `layers` against the surface's `materials` and `assets` IDs;
/// errors name the offending path and rule.
pub(crate) fn compile(
    layers: &[PlacementLayer],
    materials: &[String],
    assets: &[String],
) -> Result<Vec<CompiledPlacement>, (String, String)> {
    let fail = |path: String, rule: &str| (path, rule.to_owned());
    if layers.len() > MAX_LAYERS {
        return Err(fail("placements".into(), "lists at most 64 layers"));
    }
    for (index, id) in assets.iter().enumerate() {
        if assets[..index].contains(id) {
            return Err(fail(format!("assets[{index}]"), "IDs must be unique"));
        }
    }
    let mut compiled = Vec::with_capacity(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        let at = format!("placements[{index}]");
        if layers[..index].iter().any(|other| other.id == layer.id) {
            return Err(fail(format!("{at}.id"), "IDs must be unique"));
        }
        if layer.assets.is_empty() || layer.assets.len() > MAX_CHOICES {
            return Err(fail(format!("{at}.assets"), "chooses from 1 to 64 assets"));
        }
        let mut choices = Vec::with_capacity(layer.assets.len());
        let mut total = 0.0;
        for (choice_index, choice) in layer.assets.iter().enumerate() {
            let path = format!("{at}.assets[{choice_index}]");
            let asset = assets
                .iter()
                .position(|id| *id == choice.asset)
                .ok_or_else(|| fail(format!("{path}.asset"), "must name an asset"))?;
            if !(choice.weight.is_finite() && choice.weight >= 0.0) {
                return Err(fail(
                    format!("{path}.weight"),
                    "must be finite and not negative",
                ));
            }
            total += choice.weight;
            choices.push((asset as u16, total));
        }
        if total <= 0.0 {
            return Err(fail(format!("{at}.assets"), "needs a positive weight"));
        }
        for choice in &mut choices {
            choice.1 /= total;
        }
        if !(layer.footprint.is_finite() && (0.25..=256.0).contains(&layer.footprint)) {
            return Err(fail(
                format!("{at}.footprint"),
                "must be in [0.25, 256] meters",
            ));
        }
        if !(layer.density.is_finite() && layer.density >= 0.0) {
            return Err(fail(
                format!("{at}.density"),
                "must be finite and not negative",
            ));
        }
        let fill = layer.density * layer.footprint * layer.footprint / 10_000.0;
        if fill > 1.0 {
            return Err(fail(
                format!("{at}.density"),
                "places at most one object per footprint square",
            ));
        }
        let set = if layer.materials.is_empty() {
            None
        } else {
            let mut set = [0_u64; 4];
            for (material_index, id) in layer.materials.iter().enumerate() {
                let material = materials
                    .iter()
                    .position(|known| known == id)
                    .ok_or_else(|| {
                        fail(
                            format!("{at}.materials[{material_index}]"),
                            "must name a surface material",
                        )
                    })?;
                set[material / 64] |= 1 << (material % 64);
            }
            Some(set)
        };
        let [low, high] = layer.scale;
        if !(low.is_finite() && high.is_finite() && low > 0.0 && low <= high) {
            return Err(fail(
                format!("{at}.scale"),
                "must be positive, the smaller first",
            ));
        }
        if !(layer.align.is_finite() && (0.0..=1.0).contains(&layer.align)) {
            return Err(fail(format!("{at}.align"), "must be in [0, 1]"));
        }
        if !layer.sink.is_finite() {
            return Err(fail(format!("{at}.sink"), "must be finite"));
        }
        if layer.level > MAX_LEVEL {
            return Err(fail(format!("{at}.level"), "must be in [0, 15]"));
        }
        compiled.push(CompiledPlacement {
            level: layer.level,
            choices,
            fill,
            footprint: layer.footprint,
            materials: set,
            mask: Compiled::from(&layer.mask, &format!("{at}.mask"), 0)
                .map_err(|error| (error.path, error.rule))?,
            scale: layer.scale,
            align: layer.align,
            sink: layer.sink,
        });
    }
    Ok(compiled)
}

fn hash(value: u32) -> u32 {
    let mut h = value.wrapping_mul(0x9e37_79b1);
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

fn unit(value: u32) -> f32 {
    (value >> 8) as f32 / 16_777_216.0
}

/// The share of control value `control` held by the materials in `set`.
fn material_share(control: Control, set: &Option<[u64; 4]>) -> f32 {
    let Some(set) = set else {
        return 1.0;
    };
    let has = |material: u8| set[usize::from(material) / 64] & (1 << (material % 64)) != 0;
    let blend = f32::from(control.blend) / 255.0;
    let mut share = 0.0;
    if has(control.a) {
        share += 1.0 - blend;
    }
    if has(control.b) {
        share += blend;
    }
    share
}

/// The work of placing one candidate square, against one per sample of a
/// layer: hashing it, and for those its rank keeps, sampling the tile.
const SQUARE_COST: u64 = 1;
const SAMPLE_COST: u64 = 8;

/// Places row `row` (from the first) of layer `layer`'s candidate squares
/// over `tile`, appending to `out`; `seed` varies them per terrain.
/// Layers of other levels than `tile`'s hold no rows. Returns the work it
/// took (samples' worth) and the next row, none after the last layer's
/// last.
pub(crate) fn place_row(
    layers: &[CompiledPlacement],
    tile: &Tile,
    seed: u32,
    (layer_index, row): (usize, usize),
    out: &mut Vec<Placed>,
) -> (u64, Option<(usize, usize)>) {
    let spacing = tile.spacing();
    let size = TILE_QUADS as f32 * spacing;
    let key = tile.key();
    let origin = [f32::from(key.x) * size, f32::from(key.z) * size];
    let Some(layer) = layers.get(layer_index) else {
        return (0, None);
    };
    if layer.fill <= 0.0 || layer.level != key.level {
        return (0, Some((layer_index + 1, 0)));
    }
    let footprint = layer.footprint;
    let layer_seed = hash(seed ^ (layer_index as u32).wrapping_mul(0x632b_e5ab));
    let first = [
        (origin[0] / footprint).floor() as i64,
        (origin[1] / footprint).floor() as i64,
    ];
    let last = [
        ((origin[0] + size) / footprint).floor() as i64,
        ((origin[1] + size) / footprint).floor() as i64,
    ];
    let square_z = first[1] + row as i64;
    for square_x in first[0]..=last[0] {
        let square = hash(
            layer_seed
                ^ (square_x as u32).wrapping_mul(0x8da6_b343)
                ^ (square_z as u32).wrapping_mul(0xd816_3841),
        );
        let rank = unit(hash(square ^ 0x1656_67b1));
        if rank >= layer.fill {
            continue;
        }
        // The candidate's point, which belongs to the tile holding
        // it.
        let point = [
            (square_x as f32 + unit(square)) * footprint,
            (square_z as f32 + unit(hash(square ^ 0x5bd1_e995))) * footprint,
        ];
        let local = [point[0] - origin[0], point[1] - origin[1]];
        if local[0] < 0.0 || local[1] < 0.0 || local[0] >= size || local[1] >= size {
            continue;
        }
        let Some(sample) = tile.sample(glam::Vec2::from(local)) else {
            continue;
        };
        if sample.hole {
            continue;
        }
        let nearest = [
            (local[0] / spacing).round() as i32,
            (local[1] / spacing).round() as i32,
        ];
        let Some(control) = tile.control_at(nearest[0], nearest[1]) else {
            continue;
        };
        let share = material_share(control, &layer.materials);
        if share <= 0.0 {
            continue;
        }
        let normal = sample.normal;
        let gradient = (normal.x * normal.x + normal.z * normal.z).sqrt() / max(normal.y, 1.0e-3);
        let height = |x: i32, z: i32| tile.height(x.clamp(-1, 257), z.clamp(-1, 257));
        let (x, z) = (nearest[0], nearest[1]);
        let curvature = (height(x - 1, z) + height(x + 1, z) + height(x, z - 1) + height(x, z + 1)
            - 4.0 * height(x, z))
            / (spacing * spacing);
        let inputs = SampleInputs {
            x: point[0],
            z: point[1],
            height: sample.height,
            gradient,
            curvature,
            spacing,
        };
        let keep = layer.fill * share * clamp(layer.mask.evaluate(&inputs), 0.0, 1.0);
        if rank >= keep {
            continue;
        }
        let pick = unit(hash(square ^ 0x27d4_eb2f));
        let asset = layer
            .choices
            .iter()
            .find(|(_, cumulative)| pick < *cumulative)
            .map_or(layer.choices[layer.choices.len() - 1].0, |(asset, _)| {
                *asset
            });
        let scale =
            layer.scale[0] + (layer.scale[1] - layer.scale[0]) * unit(hash(square ^ 0xc2b2_ae35));
        let yaw = unit(hash(square ^ 0x85eb_ca6b)) * std::f32::consts::TAU;
        // Its up leans toward the ground's normal as far as the
        // layer aligns; a wobble keeps upright objects from
        // standing too straight.
        let (wobble_sin, wobble_cos) =
            sin_cos(unit(hash(square ^ 0x9e37_79b9)) * std::f32::consts::TAU);
        let lean = [
            normal.x * layer.align + 0.03 * wobble_cos,
            normal.z * layer.align + 0.03 * wobble_sin,
        ];
        out.push(Placed {
            position: [point[0], sample.height - layer.sink * scale, point[1]],
            yaw,
            lean,
            scale,
            sink: layer.sink * scale,
            asset,
            layer: layer_index as u16,
        });
    }
    let squares = (last[0] - first[0] + 1) as u64;
    let kept = (squares as f32 * layer.fill).ceil() as u64;
    let cost = squares * SQUARE_COST + kept * (SAMPLE_COST + layer.mask.cost());
    let next = if square_z < last[1] {
        (layer_index, row + 1)
    } else {
        (layer_index + 1, 0)
    };
    (cost, Some(next))
}
