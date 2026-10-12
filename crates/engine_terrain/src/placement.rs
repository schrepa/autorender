//! Terrain placement: objects standing on a terrain's
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
//! resident. A layer may also stand far away: a coarser
//! level's tiles, which stay resident farther still (pinned levels
//! everywhere), place the share of its candidates whose rank falls under
//! that share of its density, a subset of its own objects in the same
//! places, which renderers hide where its own level's tiles are resident.
//!
//! Authors stand objects on a surface too: each at its own
//! point, on the tiles of its level (the finest for small things, coarser
//! for large ones, which then show as far as those tiles stay resident),
//! drawn, shadowed and collided with as the layers' objects are.
//!
//! A tile's objects may be placed in two parts: those of
//! its own level's layers and its authored objects, then those of layers
//! standing far away at its level, which far tiles hold by the tens of
//! thousands; [`merge_placed`] puts the parts in the order one pass places
//! them, so the tile holds the same objects either way.

use engine_math::{clamp, max, min, sin_cos};
use engine_settings::{ExtraRules, InvalidSetting, settings};
use serde::{Deserialize, Serialize};

use crate::surface::{Compiled, Control, Mask, Reads, RowInputs, RowScratch};
use crate::{TILE_QUADS, TILE_STRIDE, Tile};

/// The placement layer authored objects report collision
/// hits in, beside bridges' `u16::MAX`.
pub const AUTHORED_LAYER: u16 = u16::MAX - 1;

/// Which of a tile's objects a generator places: all of
/// them; those of layers of its own level and its authored objects; or
/// those of layers standing far away at its level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Placing {
    #[default]
    All,
    Near,
    Far,
}

/// A tile's objects placed in two parts, `near` (its own
/// level's layers' objects, then its authored ones) and `far` (layers
/// standing far away at its level), in the order one pass places them:
/// layer by layer, authored objects last.
pub fn merge_placed(near: &[Placed], far: &[Placed]) -> Vec<Placed> {
    let layered = near
        .iter()
        .position(|object| object.layer == AUTHORED_LAYER)
        .unwrap_or(near.len());
    let (layered, authored) = near.split_at(layered);
    let mut merged = Vec::with_capacity(near.len() + far.len());
    let (mut a, mut b) = (layered.iter().peekable(), far.iter().peekable());
    loop {
        let next = match (a.peek(), b.peek()) {
            (Some(near), Some(far)) if far.layer < near.layer => b.next(),
            (Some(_), _) => a.next(),
            (None, Some(_)) => b.next(),
            (None, None) => break,
        };
        merged.extend(next.copied());
    }
    merged.extend_from_slice(authored);
    merged
}

/// Validity guards: how many layers a surface places, how many assets a
/// layer chooses from, the coarsest level a layer's tiles may be, and the
/// widest bed an object rests on (meters at scale 1).
const MAX_LAYERS: usize = 64;
const MAX_CHOICES: usize = 64;
const MAX_LEVEL: u8 = crate::MAX_LEVEL as u8;
const MAX_BED: f32 = 1000.0;

settings! {
    /// An asset a placement layer chooses, by ID of the surface's assets, and
    /// how often against the layer's others.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct PlacementChoice {
        /// The asset's ID.
        pub asset: String = String::new() => any,
        /// How often it is chosen against the layer's others.
        #[serde(default = "one")]
        pub weight: f32 = one() => at_least(0.0),
    }
}

settings! {
    extra_rules;
    /// One layer of objects on a surface.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct PlacementLayer {
        /// The layer's ID.
        pub id: String = String::new() => any,
        /// The assets it chooses from.
        pub assets: Vec<PlacementChoice> = Vec::new() => any,
        /// Objects per hectare where everything allows, at most one per
        /// footprint square.
        pub density: f32 = 0.0 => at_least(0.0),
        /// The meters between candidates.
        pub footprint: f32 = 1.0 => range(0.25, 256.0),
        /// The surface materials it stands on (IDs; any when empty),
        /// weighted by their share.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub materials: Vec<String> = Vec::new() => any,
        /// Where it stands.
        #[serde(default = "Mask::one")]
        pub mask: Mask = Mask::one() => any,
        /// Its objects' scale range, the smaller first.
        #[serde(default = "unit_scale")]
        pub scale: [f32; 2] = unit_scale() => range(MIN_SCALE, MAX_SCALE),
        /// How far its objects lean with the ground: 0 upright, 1 along its
        /// normal.
        #[serde(default)]
        pub align: f32 = 0.0 => range(0.0, 1.0),
        /// How far they turn to face down its slope, their x across it and z
        /// down it: 0 any way, 1 straight down it.
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub face: f32 = 0.0 => range(0.0, 1.0),
        /// How deep they sink, in meters at scale 1, below the lowest ground
        /// within their bed.
        #[serde(default)]
        pub sink: f32 = 0.0 => finite,
        /// How far around them they rest on, in meters at scale 1: 0, the
        /// ground at their point; blocks on steep slopes stand in it instead
        /// of hanging over it.
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub bed: f32 = 0.0 => range(0.0, MAX_BED),
        /// The pyramid level whose tiles hold its candidates: 0, the finest.
        #[serde(default, skip_serializing_if = "is_zero")]
        pub level: u8 = 0 => range(0, MAX_LEVEL),
        /// Where it also stands far away.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub far: Option<FarPlacement> = None => any,
    }
}

impl ExtraRules for PlacementLayer {
    /// Its objects fit their footprints, its scale range runs smaller
    /// first, its choices weigh something, and its far placement stands on
    /// a coarser level.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.density * self.footprint * self.footprint / 10_000.0 > 1.0 {
            return Err(InvalidSetting::new(
                "density",
                "places at most one object per footprint square",
            ));
        }
        if self.scale[0] > self.scale[1] {
            return Err(InvalidSetting::new("scale", "must list the smaller first"));
        }
        if self.assets.iter().map(|choice| choice.weight).sum::<f32>() <= 0.0 {
            return Err(InvalidSetting::new("assets", "needs a positive weight"));
        }
        if self.far.is_some_and(|far| far.level <= self.level) {
            return Err(InvalidSetting::new(
                "far.level",
                "must be coarser than the layer's level",
            ));
        }
        Ok(())
    }
}

settings! {
    /// Where a placement layer also stands far away: on the tiles of a
    /// coarser level, a share of its objects, the candidates ranked first,
    /// where its own level's tiles are not resident.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct FarPlacement {
        /// The coarser level whose tiles hold them.
        pub level: u8 = 1 => range(1, MAX_LEVEL),
        /// The share of the layer's objects standing there.
        pub share: f32 = 1.0 => above_to(0.0, 1.0),
    }
}

/// The smallest and largest scale placed objects take.
const MIN_SCALE: f32 = 0.001;
const MAX_SCALE: f32 = 1000.0;

impl engine_settings::Setting for Mask {}

fn is_zero(level: &u8) -> bool {
    *level == 0
}

fn is_zero_f32(value: &f32) -> bool {
    *value == 0.0
}

fn one() -> f32 {
    1.0
}

fn unit_scale() -> [f32; 2] {
    [1.0, 1.0]
}

settings! {
    /// An object an author stands on the surface.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct AuthoredPlacement {
        /// Its asset, by ID of the surface's assets.
        pub asset: String = String::new() => any,
        /// Its base's terrain-local point `[x, z]`, where it stands on the
        /// ground.
        pub position: [f32; 2] = [0.0; 2] => finite,
        /// Its turn about the vertical, degrees.
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub yaw: f32 = 0.0 => finite,
        /// Its scale.
        #[serde(default = "one", skip_serializing_if = "is_one")]
        pub scale: f32 = one() => range(MIN_SCALE, MAX_SCALE),
        /// How deep its base sinks, in meters at scale 1.
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub sink: f32 = 0.0 => finite,
        /// How far it leans with the ground: 0 upright, 1 along its normal.
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub align: f32 = 0.0 => range(0.0, 1.0),
        /// The pyramid level whose tiles hold it: 0, the finest; coarser for
        /// objects seen from farther away.
        #[serde(default, skip_serializing_if = "is_zero")]
        pub level: u8 = 0 => range(0, MAX_LEVEL),
    }
}

fn is_one(value: &f32) -> bool {
    *value == 1.0
}

/// An authored object ready to stand: its asset index, point, turn
/// (radians), scale, sink, lean and level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompiledAuthored {
    asset: u16,
    position: [f32; 2],
    yaw: f32,
    scale: f32,
    sink: f32,
    align: f32,
    pub(crate) level: u8,
}

/// Validates `placed` against the surface's `assets` IDs; errors name the
/// offending path and rule.
pub(crate) fn compile_authored(
    placed: &[AuthoredPlacement],
    assets: &[String],
) -> Result<Vec<CompiledAuthored>, (String, String)> {
    let fail = |path: String, rule: &str| (path, rule.to_owned());
    placed
        .iter()
        .enumerate()
        .map(|(index, object)| {
            let at = format!("placed[{index}]");
            let asset = assets
                .iter()
                .position(|id| *id == object.asset)
                .ok_or_else(|| fail(format!("{at}.asset"), "must name an asset"))?;
            object.validate().map_err(|invalid| {
                (format!("{at}.{}", invalid.field), invalid.rule.into_owned())
            })?;
            Ok(CompiledAuthored {
                asset: asset as u16,
                position: object.position,
                yaw: object.yaw.to_radians(),
                scale: object.scale,
                sink: object.sink,
                align: object.align,
                level: object.level,
            })
        })
        .collect()
}

/// The authored objects standing on `tile`, appended to
/// `out`: those of its level whose point it holds, on its ground; none on
/// holes. Returns the work it took (samples' worth).
pub(crate) fn place_authored(
    placed: &[CompiledAuthored],
    tile: &Tile,
    out: &mut Vec<Placed>,
) -> u64 {
    let size = TILE_QUADS as f32 * tile.spacing();
    let key = tile.key();
    let origin = [f32::from(key.x) * size, f32::from(key.z) * size];
    let mut cost = 0;
    for object in placed.iter().filter(|object| object.level == key.level) {
        cost += 1;
        let local = [
            object.position[0] - origin[0],
            object.position[1] - origin[1],
        ];
        if local[0] < 0.0 || local[1] < 0.0 || local[0] >= size || local[1] >= size {
            continue;
        }
        let Some(sample) = tile.sample(glam::Vec2::from(local)) else {
            continue;
        };
        if sample.hole {
            continue;
        }
        cost += SAMPLE_COST;
        let normal = sample.normal;
        out.push(Placed {
            position: [
                object.position[0],
                sample.height - object.sink * object.scale,
                object.position[1],
            ],
            yaw: object.yaw,
            lean: [normal.x * object.align, normal.z * object.align],
            scale: object.scale,
            sink: object.sink * object.scale,
            ground: [normal.x, normal.z],
            asset: object.asset,
            layer: AUTHORED_LAYER,
            near_level: None,
            rank: 0.0,
        });
    }
    cost
}

/// One object a layer placed: its base's terrain-local position, its turn
/// about the vertical (radians), how its up leans (x and z of its up
/// vector), its scale, how far its base sinks below the ground (meters),
/// the ground's normal under it (x and z), its asset and
/// layer indices, for objects standing far away the level
/// whose resident tiles hold it instead, and its rank under the density at
/// its point (0 to 1; 0 for authored objects): the share of the density
/// past which it stands, which renderers thin layers by with distance.
/// Far away, a layer stands the objects ranked first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub position: [f32; 3],
    pub yaw: f32,
    pub lean: [f32; 2],
    pub scale: f32,
    pub sink: f32,
    pub ground: [f32; 2],
    pub asset: u16,
    pub layer: u16,
    pub near_level: Option<u8>,
    pub rank: f32,
}

impl Placed {
    /// The world matrix of the object on a terrain whose origin is
    /// `origin`, with its base at height `base` (terrain-local) instead of
    /// where it was placed: its up leaning as placed, turned about it by its
    /// yaw, and scaled.
    pub fn model(&self, origin: glam::Vec3, base: f32) -> glam::Mat4 {
        use glam::Vec3;
        let [lean_x, lean_z] = self.lean;
        let up = Vec3::new(
            lean_x,
            max(1.0 - lean_x * lean_x - lean_z * lean_z, 0.01).sqrt(),
            lean_z,
        )
        .normalize();
        let (sin, cos) = sin_cos(self.yaw);
        let forward = Vec3::new(cos, 0.0, sin);
        let tangent = (forward - up * forward.dot(up)).normalize_or(Vec3::X);
        let across = tangent.cross(up);
        let [x, _, z] = self.position;
        glam::Mat4::from_cols(
            (tangent * self.scale).extend(0.0),
            (up * self.scale).extend(0.0),
            (across * self.scale).extend(0.0),
            (origin + Vec3::new(x, base, z)).extend(1.0),
        )
    }
}

/// A compiled layer: its tiles' level, its asset indices and cumulative
/// weights, the share of a square its density fills, its materials as a
/// 256-bit set (none: any), its mask and its variations, and the level and
/// share it stands far away at.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompiledPlacement {
    pub(crate) level: u8,
    pub(crate) far: Option<(u8, f32)>,
    choices: Vec<(u16, f32)>,
    fill: f32,
    footprint: f32,
    materials: Option<[u64; 4]>,
    mask: Compiled,
    reads: Reads,
    scale: [f32; 2],
    align: f32,
    face: f32,
    sink: f32,
    bed: f32,
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
        layer
            .validate()
            .map_err(|invalid| (format!("{at}.{}", invalid.field), invalid.rule.into_owned()))?;
        let mut choices = Vec::with_capacity(layer.assets.len());
        let mut total = 0.0;
        for (choice_index, choice) in layer.assets.iter().enumerate() {
            let path = format!("{at}.assets[{choice_index}]");
            let asset = assets
                .iter()
                .position(|id| *id == choice.asset)
                .ok_or_else(|| fail(format!("{path}.asset"), "must name an asset"))?;
            total += choice.weight;
            choices.push((asset as u16, total));
        }
        for choice in &mut choices {
            choice.1 /= total;
        }
        let fill = layer.density * layer.footprint * layer.footprint / 10_000.0;
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
        let mask = Compiled::from(&layer.mask, &format!("{at}.mask"), 0)
            .map_err(|error| (error.path, error.rule))?;
        compiled.push(CompiledPlacement {
            level: layer.level,
            far: layer.far.map(|far| (far.level, far.share)),
            choices,
            fill,
            footprint: layer.footprint,
            materials: set,
            reads: mask.reads(),
            mask,
            scale: layer.scale,
            align: layer.align,
            face: layer.face,
            sink: layer.sink,
            bed: layer.bed,
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

/// Samples of the ground around an object resting on a bed: eight around
/// its edge and four halfway.
const BED_SAMPLES: u32 = 12;

/// The lowest of `height` and `tile`'s ground at the [`BED_SAMPLES`] points
/// within `reach` meters of tile-local point `local`, held within the
/// tile.
fn lowest_ground(tile: &Tile, local: [f32; 2], reach: f32, height: f32) -> f32 {
    let size = TILE_QUADS as f32 * tile.spacing();
    let mut lowest = height;
    for index in 0..BED_SAMPLES {
        let (distance, angle) = if index < 8 {
            (reach, index as f32 * std::f32::consts::FRAC_PI_4)
        } else {
            (
                reach * 0.5,
                (index - 8) as f32 * std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_4,
            )
        };
        let (sin, cos) = sin_cos(angle);
        let point = glam::Vec2::new(
            clamp(local[0] + cos * distance, 0.0, size),
            clamp(local[1] + sin * distance, 0.0, size),
        );
        if let Some(sample) = tile.sample(point) {
            lowest = min(lowest, sample.height);
        }
    }
    lowest
}

/// Places row `row` (from the first) of layer `layer`'s candidate squares
/// over `tile`, appending to `out`; `seed` varies them per terrain.
/// Layers hold rows on tiles of their level, and of the level they stand
/// far away at, where only their far share stands; others hold none.
/// Returns the work it took (samples' worth) and the next row, none after
/// the last layer's last.
/// A row of candidates whose masks a row evaluates at once,
/// reused row after row: each candidate's square, rank, point, offset in
/// its tile, ground, material share and its mask's inputs.
#[derive(Default)]
pub(crate) struct PlaceScratch {
    candidates: Vec<Candidate>,
    xs: Vec<f32>,
    zs: Vec<f32>,
    height: Vec<f32>,
    gradient: Vec<f32>,
    curvature: Vec<f32>,
    water: Vec<crate::HydrologySample>,
    path: Vec<crate::PathSample>,
    values: Vec<f32>,
    rows: RowScratch,
    // Control values the rules gave samples of a tile holding none,
    // packed, `u32::MAX` where not asked yet: each sample's
    // once a tile, however many candidates share it.
    controls: Vec<u32>,
}

#[derive(Clone, Copy)]
struct Candidate {
    square: u32,
    rank: f32,
    point: [f32; 2],
    local: [f32; 2],
    sample: crate::TileSample,
    share_of_materials: f32,
}

/// The rows of candidate squares layer `layer_index` of `layers` places on
/// tile `key` at `spacing` with `placing`'s objects, as
/// [`place_row`] walks them: none where it places nothing there.
pub(crate) fn layer_rows(
    layers: &[CompiledPlacement],
    layer_index: usize,
    key: crate::TileKey,
    spacing: f32,
    placing: Placing,
) -> u32 {
    let Some(layer) = layers.get(layer_index) else {
        return 0;
    };
    let size = TILE_QUADS as f32 * spacing;
    let origin = [f32::from(key.x) * size, f32::from(key.z) * size];
    let (share, near_level) = match layer.far {
        _ if layer.level == key.level => (1.0, None),
        Some((level, share)) if level == key.level => (share, Some(layer.level)),
        _ => (0.0, None),
    };
    let skipped = match placing {
        Placing::All => false,
        Placing::Near => near_level.is_some(),
        Placing::Far => near_level.is_none(),
    };
    if layer.fill * share <= 0.0 || skipped {
        return 0;
    }
    let first = (origin[1] / layer.footprint).floor() as i64;
    let last = ((origin[1] + size) / layer.footprint).floor() as i64;
    (last - first + 1).max(0) as u32
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(crate) fn place_row(
    layers: &[CompiledPlacement],
    tile: &Tile,
    seed: u32,
    (layer_index, row): (usize, usize),
    out: &mut Vec<Placed>,
    (stack, placing, scratch, rules): (
        &crate::Layers,
        Placing,
        &mut PlaceScratch,
        &crate::BiomeRules,
    ),
) -> (u64, Option<(usize, usize)>) {
    let water = stack.hydrology();
    let spacing = tile.spacing();
    let size = TILE_QUADS as f32 * spacing;
    let key = tile.key();
    let origin = [f32::from(key.x) * size, f32::from(key.z) * size];
    let Some(layer) = layers.get(layer_index) else {
        return (0, None);
    };
    // Far away, the candidates ranked first: a subset of the layer's own.
    let (share, near_level) = match layer.far {
        _ if layer.level == key.level => (1.0, None),
        Some((level, share)) if level == key.level => (share, Some(layer.level)),
        _ => (0.0, None),
    };
    let skipped = match placing {
        Placing::All => false,
        Placing::Near => near_level.is_some(),
        Placing::Far => near_level.is_none(),
    };
    let fill = layer.fill * share;
    if fill <= 0.0 || skipped {
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
    // The row's candidates that may stand, with their masks' inputs; then
    // their masks over the row at once, each as one evaluation reads it;
    // then, in order, those that stand.
    let PlaceScratch {
        candidates,
        xs,
        zs,
        height: heights,
        gradient: gradients,
        curvature: curvatures,
        water: waters,
        path: paths,
        values,
        rows,
        controls,
    } = scratch;
    candidates.clear();
    for list in [
        &mut *xs,
        &mut *zs,
        &mut *heights,
        &mut *gradients,
        &mut *curvatures,
    ] {
        list.clear();
    }
    waters.clear();
    paths.clear();
    for square_x in first[0]..=last[0] {
        let square = hash(
            layer_seed
                ^ (square_x as u32).wrapping_mul(0x8da6_b343)
                ^ (square_z as u32).wrapping_mul(0xd816_3841),
        );
        let rank = unit(hash(square ^ 0x1656_67b1));
        if rank >= fill {
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
        // Tiles whose control values the GPU gives hold
        // none: the rules give the sample's, the same bits.
        let control = match tile.control_at(nearest[0], nearest[1]) {
            Some(control) => control,
            None => {
                let stride = TILE_STRIDE as usize;
                let stored = |sample: i32| (sample.clamp(-1, stride as i32 - 2) + 1) as usize;
                let (column, row) = (stored(nearest[0]), stored(nearest[1]));
                if controls.len() != stride * stride {
                    controls.clear();
                    controls.resize(stride * stride, u32::MAX);
                }
                let held = &mut controls[row * stride + column];
                if *held == u32::MAX {
                    *held =
                        crate::gpu::sample_control(tile, (stack, rules), (column, row)).packed();
                }
                Control::unpack(*held)
            }
        };
        let share_of_materials = material_share(control, &layer.materials);
        if share_of_materials <= 0.0 {
            continue;
        }
        // The density at most, with the mask at 1: where the rank passes
        // even that, nothing here can stand.
        if rank >= layer.fill * share_of_materials * share {
            continue;
        }
        let normal = sample.normal;
        let gradient = (normal.x * normal.x + normal.z * normal.z).sqrt() / max(normal.y, 1.0e-3);
        let height = |x: i32, z: i32| tile.height(x.clamp(-1, 257), z.clamp(-1, 257));
        let (x, z) = (nearest[0], nearest[1]);
        let curvature = (height(x - 1, z) + height(x + 1, z) + height(x, z - 1) + height(x, z + 1)
            - 4.0 * height(x, z))
            / (spacing * spacing);
        candidates.push(Candidate {
            square,
            rank,
            point,
            local,
            sample,
            share_of_materials,
        });
        xs.push(point[0]);
        zs.push(point[1]);
        heights.push(sample.height);
        gradients.push(gradient);
        curvatures.push(curvature);
        // What the mask reads alone.
        waters.push(match water {
            Some(water) if layer.reads.water => {
                let river = water
                    .rivers
                    .nearest(point[0], point[1])
                    .map_or(crate::hydrology::RIVER_FAR, |river| {
                        river.distance - river.half
                    });
                water.masks_at(point[0], point[1], spacing, river)
            }
            _ => crate::HydrologySample::default(),
        });
        paths.push(if layer.reads.path {
            stack.path_at(point[0], point[1])
        } else {
            crate::PathSample::FAR
        });
    }
    values.clear();
    values.resize(candidates.len(), 0.0);
    layer.mask.evaluate_row(
        &RowInputs {
            xs,
            zs,
            spacing,
            height: heights,
            gradient: gradients,
            curvature: curvatures,
            water: waters,
            path: paths,
        },
        values,
        rows,
    );
    for (candidate, &mask) in candidates.iter().zip(values.iter()) {
        let Candidate {
            square,
            rank,
            point,
            local,
            sample,
            share_of_materials,
        } = *candidate;
        let normal = sample.normal;
        // The density here, and of it the share standing on this tile.
        let full = layer.fill * share_of_materials * clamp(mask, 0.0, 1.0);
        if rank >= full * share {
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
        // Turned at random, or as far as the layer faces them, across the
        // slope with z down it, the rest of their turn at random.
        let turn = unit(hash(square ^ 0x85eb_ca6b));
        let downhill = (normal.x * normal.x + normal.z * normal.z).sqrt();
        let yaw = if layer.face > 0.0 && downhill > 0.05 {
            // libm's, as browsers compute it: the platform's differs.
            libm::atan2f(-normal.x, normal.z)
                + (turn - 0.5) * std::f32::consts::TAU * (1.0 - layer.face)
        } else {
            turn * std::f32::consts::TAU
        };
        // Its up leans toward the ground's normal as far as the
        // layer aligns; a wobble keeps upright objects from
        // standing too straight.
        let (wobble_sin, wobble_cos) =
            sin_cos(unit(hash(square ^ 0x9e37_79b9)) * std::f32::consts::TAU);
        let lean = [
            normal.x * layer.align + 0.03 * wobble_cos,
            normal.z * layer.align + 0.03 * wobble_sin,
        ];
        // Resting on its bed, it sinks below the lowest ground there.
        let below = if layer.bed > 0.0 {
            sample.height - lowest_ground(tile, local, layer.bed * scale, sample.height)
        } else {
            0.0
        };
        let sink = layer.sink * scale + below;
        out.push(Placed {
            position: [point[0], sample.height - sink, point[1]],
            yaw,
            lean,
            scale,
            sink,
            ground: [normal.x, normal.z],
            asset,
            layer: layer_index as u16,
            near_level,
            rank: rank / full,
        });
    }
    let squares = (last[0] - first[0] + 1) as u64;
    let kept = (squares as f32 * fill).ceil() as u64;
    let bed = if layer.bed > 0.0 {
        u64::from(BED_SAMPLES) * SAMPLE_COST
    } else {
        0
    };
    let cost = squares * SQUARE_COST + kept * (SAMPLE_COST + layer.mask.cost() + bed);
    let next = if square_z < last[1] {
        (layer_index, row + 1)
    } else {
        (layer_index + 1, 0)
    };
    (cost, Some(next))
}

crate::codec::record!(PlacementChoice { asset, weight });
crate::codec::record!(FarPlacement { level, share });
crate::codec::record!(PlacementLayer {
    id,
    assets,
    density,
    footprint,
    materials,
    mask,
    scale,
    align,
    face,
    sink,
    bed,
    level,
    far,
});
crate::codec::record!(AuthoredPlacement {
    asset,
    position,
    yaw,
    scale,
    sink,
    align,
    level,
});
crate::codec::record!(Placed {
    position,
    yaw,
    lean,
    scale,
    sink,
    ground,
    asset,
    layer,
    near_level,
    rank,
});
