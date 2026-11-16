use engine_codec::{Error, Expected, ExtraRules, InvalidSetting, Rule, fields};
use engine_math::{clamp, max, min, sin_cos};
use engine_place::GpuBits;
use serde::{Deserialize, Serialize};

use super::expr::{Expr, Ramp, producer, read_map};
use super::surface::{
    Compiled, Reads, RowInputs, RowScratch, Scope, erosion_row, rested, setting_error,
    terrain_error,
    unknown, unknown_map,
};
use super::{Control, Footprint, Solid, Tile};
use crate::{TILE_QUADS, TILE_STRIDE};

pub const AUTHORED_LAYER: u16 = u16::MAX - 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Placing {
    #[default]
    All,
    Near,
    Far,
}

const MAX_LAYERS: usize = 64;
const MAX_CHOICES: usize = 64;
const MAX_LEVEL: u8 = crate::MAX_LEVEL as u8;
const MAX_BED: f32 = 1000.0;
const MAX_REACH: i64 = 2;
const YIELD_DEPTH: u32 = 2;

fields! {
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct PlacementChoice {
        pub asset: String = String::new() => any,
        #[serde(default = "one")]
        pub weight: f32 = one() => at_least(0.0),
    }
}

fields! {
    extra_rules;
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct PlacementLayer {
        pub id: String = String::new() => any,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub group: String = String::new() => any,
        pub assets: Vec<PlacementChoice> = Vec::new() => any,
        pub density: f32 = 0.0 => at_least(0.0),
        pub footprint: f32 = 1.0 => range(0.25, 256.0),
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub room: f32 = 0.0 => at_least(0.0),
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub materials: Vec<String> = Vec::new() => any,
        #[serde(default = "Expr::one")]
        pub mask: Expr = Expr::one() => any,
        #[serde(default = "unit_scale")]
        pub scale: [f32; 2] = unit_scale() => range(MIN_SCALE, MAX_SCALE),
        #[serde(default)]
        pub align: f32 = 0.0 => range(0.0, 1.0),
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub downhill: f32 = 0.0 => at_least(0.0),
        #[serde(default)]
        pub sink: f32 = 0.0 => finite,
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub bed: f32 = 0.0 => range(0.0, MAX_BED),
        #[serde(default, skip_serializing_if = "is_zero")]
        pub level: u8 = 0 => range(0, MAX_LEVEL),
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub far: Option<FarPlacement> = None => any,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub sized: Option<SizedBy> = None => any,
    }
}

fields! {
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct SizedBy {
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub map: String = String::new() => any,
        pub from: f32 = 0.0 => finite,
        pub to: f32 = 1.0 => finite,
        #[serde(default = "one")]
        pub tendency: f32 = one() => range(0.0, 1.0),
    }
}

impl ExtraRules for PlacementLayer {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.density * self.footprint * self.footprint / 10_000.0 > 1.0 {
            return Err(InvalidSetting::new(
                "density",
                engine_codec::Expected::Number,
            ));
        }
        if self.scale[0] > self.scale[1] {
            return Err(InvalidSetting::new("scale", engine_codec::Expected::Sorted));
        }
        let widest = MAX_REACH as f32 * self.footprint / self.scale[1];
        if self.room > widest {
            return Err(
                InvalidSetting::broken("room", Rule::Range(0.0, f64::from(widest))).got(self.room),
            );
        }
        if self.assets.iter().map(|choice| choice.weight).sum::<f32>() <= 0.0 {
            return Err(InvalidSetting::new(
                "assets",
                engine_codec::Expected::Nonzero,
            ));
        }
        if self.far.is_some_and(|far| far.level <= self.level) {
            return Err(InvalidSetting::new(
                "far.level",
                engine_codec::Expected::Above,
            ));
        }
        Ok(())
    }
}

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct FarPlacement {
        pub level: u8 = 1 => range(1, MAX_LEVEL),
    }
}

const MIN_SCALE: f32 = 0.001;
const MAX_SCALE: f32 = 1000.0;

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

fields! {
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct AuthoredPlacement {
        pub asset: String = String::new() => any,
        pub position: [f32; 2] = [0.0; 2] => finite,
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub yaw: f32 = 0.0 => finite,
        #[serde(default = "one", skip_serializing_if = "is_one")]
        pub scale: f32 = one() => range(MIN_SCALE, MAX_SCALE),
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub sink: f32 = 0.0 => finite,
        #[serde(default, skip_serializing_if = "is_zero_f32")]
        pub align: f32 = 0.0 => range(0.0, 1.0),
        #[serde(default, skip_serializing_if = "is_zero")]
        pub level: u8 = 0 => range(0, MAX_LEVEL),
    }
}

fn is_one(value: &f32) -> bool {
    *value == 1.0
}

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

pub(crate) fn compile_authored(
    placed: &[AuthoredPlacement],
    assets: &[String],
) -> Result<Vec<CompiledAuthored>, Error> {
    placed
        .iter()
        .enumerate()
        .map(|(index, object)| {
            let at = format!("$.placed[{index}]");
            let asset = assets
                .iter()
                .position(|id| *id == object.asset)
                .ok_or_else(|| unknown(format!("{at}.asset"), &object.asset, assets))?;
            object
                .validate()
                .map_err(|invalid| setting_error(invalid, &at))?;
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

pub(crate) fn place_authored(
    placed: &[CompiledAuthored],
    tile: &Tile,
    from: Option<u8>,
    out: &mut Vec<Placed>,
) -> u64 {
    let size = TILE_QUADS as f32 * tile.spacing();
    let key = tile.key();
    let origin = [f32::from(key.x) * size, f32::from(key.z) * size];
    let mut cost = 0;
    let on = |level: u8| from.map_or(level, |from| level.max(from));
    for object in placed.iter().filter(|object| on(object.level) == key.level) {
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

    pub fn footprint(&self, solid: &Solid) -> Footprint {
        use glam::{Mat3, Mat4, Vec3};
        let model = self.model(Vec3::ZERO, self.position[1])
            * Mat4::from_translation(Vec3::from_array(solid.center));
        let axes = Mat3::from_mat4(model) * Mat3::from_diagonal(Vec3::from_array(solid.half));
        Footprint {
            center: model.w_axis.truncate().to_array(),
            axes: axes.to_cols_array(),
            round: solid.round,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obstacle {
    pub center: glam::Vec2,
    pub radius: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct Sizing {
    by: Compiled,
    own: bool,
    tendency: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct Member {
    layer: u16,
    choices: Vec<(u16, f32)>,
    fill: f32,
    materials: Option<[u64; 4]>,
    mask: Compiled,
    sized: Option<Sizing>,
    scale: [f32; 2],
    room: f32,
    align: f32,
    downhill: f32,
    sink: f32,
    bed: f32,
}

#[derive(Clone, Copy)]
struct Standing {
    member: usize,
    ranked: f32,
    scale: f32,
    room: f32,
}

type Site<'a> = (&'a [f32], &'a [f32]);

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompiledPlacement {
    pub(crate) level: u8,
    pub(crate) far: Option<u8>,
    footprint: f32,
    reach: i64,
    members: Vec<Member>,
    reads: Reads,
    users: Vec<(String, String)>,
}

impl Member {
    fn compile(
        (layer, index, at): (&PlacementLayer, usize, &str),
        (materials, assets): (&[String], &[String]),
        scope: &mut Scope,
    ) -> Result<Self, Error> {
        let mut choices = Vec::with_capacity(layer.assets.len());
        let mut total = 0.0;
        for (choice_index, choice) in layer.assets.iter().enumerate() {
            let path = format!("{at}.assets[{choice_index}]");
            let asset = assets
                .iter()
                .position(|id| *id == choice.asset)
                .ok_or_else(|| unknown(format!("{path}.asset"), &choice.asset, assets))?;
            total += choice.weight;
            choices.push((asset as u16, total));
        }
        for choice in &mut choices {
            choice.1 /= total;
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
                        unknown(format!("{at}.materials[{material_index}]"), id, materials)
                    })?;
                set[material / 64] |= 1 << (material % 64);
            }
            Some(set)
        };
        let mask = Compiled::from(&layer.mask, &format!("{at}.mask"), 0, scope)?;
        let sized = match &layer.sized {
            Some(sized) => {
                let path = format!("{at}.sized");
                (sized.validate()).map_err(|invalid| setting_error(invalid, &path))?;
                let own = sized.map.is_empty();
                let by = if own {
                    Compiled::User {
                        map: 0,
                        ends: Some([sized.from, sized.to]),
                    }
                } else {
                    let ramp = Expr::Ramp(Ramp {
                        map: sized.map.clone(),
                        from: sized.from,
                        to: sized.to,
                    });
                    Compiled::from(&ramp, &path, 0, scope)?
                };
                Some(Sizing {
                    by,
                    own,
                    tendency: sized.tendency,
                })
            }
            None => None,
        };
        Ok(Self {
            layer: index as u16,
            choices,
            fill: layer.density * layer.footprint * layer.footprint / 10_000.0,
            materials: set,
            mask,
            sized,
            scale: layer.scale,
            room: layer.room,
            align: layer.align,
            downhill: layer.downhill,
            sink: layer.sink,
            bed: layer.bed,
        })
    }

    fn scaled(&self, ranked: f32, size: f32) -> f32 {
        let [low, high] = self.scale;
        match &self.sized {
            Some(sizing) => {
                let unit = (1.0 - ranked) * (1.0 - sizing.tendency)
                    + clamp(size, 0.0, 1.0) * sizing.tendency;
                low + (high - low) * unit
            }
            None => high - (high - low) * ranked,
        }
    }

    fn placed(
        &self,
        candidate: &Candidate,
        standing: Standing,
        tile: &Tile,
        near_level: Option<u8>,
    ) -> Placed {
        let Candidate {
            spot,
            local,
            sample,
            ..
        } = *candidate;
        let normal = sample.normal;
        let pick = unit(hash(spot.key ^ 0x27d4_eb2f));
        let asset = self
            .choices
            .iter()
            .find(|(_, cumulative)| pick < *cumulative)
            .map_or(self.choices[self.choices.len() - 1].0, |(asset, _)| *asset);
        let scale = standing.scale;
        let (turn_sin, turn_cos) =
            sin_cos(unit(hash(spot.key ^ 0x85eb_ca6b)) * std::f32::consts::TAU);
        let downhill = self.downhill / max(normal.y, 1.0e-3);
        let yaw = libm::atan2f(
            turn_sin + normal.z * downhill,
            turn_cos + normal.x * downhill,
        );
        let (wobble_sin, wobble_cos) =
            sin_cos(unit(hash(spot.key ^ 0x9e37_79b9)) * std::f32::consts::TAU);
        let lean = [
            normal.x * self.align + 0.03 * wobble_cos,
            normal.z * self.align + 0.03 * wobble_sin,
        ];
        let below = if self.bed > 0.0 {
            sample.height - lowest_ground(tile, local, self.bed * scale, sample.height)
        } else {
            0.0
        };
        let sink = self.sink * scale + below;
        Placed {
            position: [spot.point[0], sample.height - sink, spot.point[1]],
            yaw,
            lean,
            scale,
            sink,
            ground: [normal.x, normal.z],
            asset,
            layer: self.layer,
            near_level,
            rank: standing.ranked,
        }
    }
}

impl CompiledPlacement {
    fn most(&self) -> f32 {
        min(self.members.iter().map(|member| member.fill).sum(), 1.0)
    }

    fn cost(&self) -> u64 {
        let masks: u64 = (self.members.iter())
            .map(|member| member.mask.cost())
            .sum();
        let bed = if self.members.iter().any(|member| member.bed > 0.0) {
            u64::from(BED_SAMPLES) * SAMPLE_COST
        } else {
            0
        };
        let around = (2 * self.reach + 1).pow(2) as u64 - 1;
        SAMPLE_COST + masks + bed + around * ASKED_COST
    }

    fn standing(&self, spot: &Spot, (weights, sizes): Site<'_>) -> Option<Standing> {
        let total: f32 = weights.iter().sum();
        let full = min(total, 1.0);
        if spot.rank >= full {
            return None;
        }
        let pick = spot.pick * total;
        let mut reached = 0.0;
        let member = (weights.iter())
            .position(|weight| {
                reached += weight;
                pick < reached
            })
            .or_else(|| weights.iter().rposition(|weight| *weight > 0.0))?;
        let ranked = spot.rank / full;
        let layer = &self.members[member];
        let scale = layer.scaled(ranked, sizes[member]);
        Some(Standing {
            member,
            ranked,
            scale,
            room: layer.room * scale,
        })
    }

    fn yields(
        &self,
        seed: u32,
        (spot, [x, z], room): (&Spot, [i64; 2], f32),
        site: Site<'_>,
        depth: u32,
    ) -> bool {
        for across in -self.reach..=self.reach {
            for along in -self.reach..=self.reach {
                if across == 0 && along == 0 {
                    continue;
                }
                let square = [x + along, z + across];
                let other = Spot::at(seed, square, self.footprint);
                let Some(standing) = self.standing(&other, site) else {
                    continue;
                };
                let first =
                    standing.room > room || (standing.room == room && other.rank < spot.rank);
                let need = 0.5 * (standing.room + room);
                let apart = [
                    other.point[0] - spot.point[0],
                    other.point[1] - spot.point[1],
                ];
                if !first || apart[0] * apart[0] + apart[1] * apart[1] >= need * need {
                    continue;
                }
                if depth > 1 && self.yields(seed, (&other, square, standing.room), site, depth - 1)
                {
                    continue;
                }
                return true;
            }
        }
        false
    }
}

fn matching(first: &PlacementLayer, layer: &PlacementLayer, at: &str) -> Result<(), Error> {
    let differs = |field: &str| terrain_error(format!("{at}.{field}")).expected(Expected::Matching);
    if layer.footprint != first.footprint {
        return Err(differs("footprint")
            .got(layer.footprint)
            .nearest(first.footprint));
    }
    if layer.level != first.level {
        return Err(differs("level").got(layer.level).nearest(first.level));
    }
    if layer.far != first.far {
        let level = |layer: &PlacementLayer| layer.far.map(|far| far.level);
        return Err(differs("far").got(level(layer)).nearest(level(first)));
    }
    Ok(())
}

pub(crate) fn compile(
    layers: &[PlacementLayer],
    materials: &[String],
    assets: &[String],
) -> Result<Vec<CompiledPlacement>, Error> {
    if layers.len() > MAX_LAYERS {
        return Err(terrain_error("$.placements")
            .at_most(MAX_LAYERS as u64)
            .got(layers.len()));
    }
    for (index, id) in assets.iter().enumerate() {
        if assets[..index].contains(id) {
            return Err(terrain_error(format!("$.assets[{index}]"))
                .expected(Expected::Unique)
                .got(id));
        }
    }
    let mut groups: Vec<(CompiledPlacement, Scope)> = Vec::new();
    let mut homes = Vec::with_capacity(layers.len());
    for (index, layer) in layers.iter().enumerate() {
        let at = format!("$.placements[{index}]");
        if layers[..index].iter().any(|other| other.id == layer.id) {
            return Err(terrain_error(format!("{at}.id"))
                .expected(Expected::Unique)
                .got(&layer.id));
        }
        if layer.assets.is_empty() || layer.assets.len() > MAX_CHOICES {
            return Err(terrain_error(format!("{at}.assets"))
                .range(1.0, MAX_CHOICES as f64)
                .got(layer.assets.len()));
        }
        layer
            .validate()
            .map_err(|invalid| setting_error(invalid, &at))?;
        let shared = (layers[..index].iter())
            .position(|other| !layer.group.is_empty() && other.group == layer.group);
        let home = match shared {
            Some(first) => {
                matching(&layers[first], layer, &at)?;
                homes[first]
            }
            None => {
                let group = CompiledPlacement {
                    level: layer.level,
                    far: layer.far.map(|far| far.level),
                    footprint: layer.footprint,
                    reach: 0,
                    members: Vec::new(),
                    reads: Reads::default(),
                    users: Vec::new(),
                };
                groups.push((group, Scope::cpu()));
                groups.len() - 1
            }
        };
        homes.push(home);
        let (group, scope) = &mut groups[home];
        let member = Member::compile((layer, index, &at), (materials, assets), scope)?;
        let reads = group.reads.and(member.mask.reads());
        group.reads = (member.sized.iter()).fold(reads, |reads, size| reads.and(size.by.reads()));
        let widest = member.room * member.scale[1] / group.footprint;
        group.reach = group.reach.max(widest.ceil() as i64).min(MAX_REACH);
        group.members.push(member);
    }
    Ok(groups
        .into_iter()
        .map(|(group, scope)| CompiledPlacement {
            users: scope.users(),
            ..group
        })
        .collect())
}

pub(crate) fn check_maps(layers: &[CompiledPlacement], stack: &crate::Layers) -> Result<(), Error> {
    let published: Vec<String> = (stack.layers().iter())
        .flat_map(|layer| layer.outputs())
        .collect();
    for layer in layers {
        let missing = (layer.users.iter()).find(|(name, _)| !published.contains(name));
        if let Some((name, path)) = missing {
            return Err(unknown_map(path, name, published));
        }
    }
    Ok(())
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

const SQUARE_COST: u64 = 1;
const SAMPLE_COST: u64 = 8;
const ASKED_COST: u64 = 2;

const BED_SAMPLES: u32 = 12;

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

#[derive(Default)]
pub(crate) struct PlaceScratch {
    candidates: Vec<Candidate>,
    inputs: MaskInputs,
    values: Vec<f32>,
    suited: Vec<f32>,
    weights: Vec<f32>,
    sizes: Vec<f32>,
    rows: RowScratch,
    controls: Vec<u32>,
}

#[derive(Default)]
struct MaskInputs {
    xs: Vec<f32>,
    zs: Vec<f32>,
    height: Vec<f32>,
    gradient: Vec<[f32; 2]>,
    curvature: Vec<f32>,
    water: Vec<crate::HydrologySample>,
    erosion: Vec<f32>,
    path: Vec<crate::PathSample>,
    maps: Vec<Vec<f32>>,
}

impl MaskInputs {
    fn clear(&mut self, users: usize) {
        for list in [
            &mut self.xs,
            &mut self.zs,
            &mut self.height,
            &mut self.curvature,
            &mut self.erosion,
        ] {
            list.clear();
        }
        self.gradient.clear();
        self.water.clear();
        self.path.clear();
        self.maps.resize_with(users, Vec::new);
        for map in &mut self.maps {
            map.clear();
        }
    }

    fn push(
        &mut self,
        (tile, stack, reads): (&Tile, &crate::Layers, Reads),
        (point, local, [x, z]): ([f32; 2], [f32; 2], [i32; 2]),
        sample: &super::TileSample,
    ) {
        let spacing = tile.spacing();
        let normal = sample.normal;
        let lift = max(normal.y, 1.0e-3);
        let height = |x: i32, z: i32| tile.height(x.clamp(-1, 257), z.clamp(-1, 257));
        self.xs.push(point[0]);
        self.zs.push(point[1]);
        self.height.push(sample.height);
        self.gradient.push([-normal.x / lift, -normal.z / lift]);
        self.curvature.push(
            (height(x - 1, z) + height(x + 1, z) + height(x, z - 1) + height(x, z + 1)
                - 4.0 * height(x, z))
                / (spacing * spacing),
        );
        self.water.push(match stack.hydrology() {
            Some(water) if reads.water => {
                let local = glam::Vec2::from(local);
                crate::HydrologySample {
                    lake: tile.lake_depth(local),
                    ..tile.river_at(local).with_maps(water, point, spacing)
                }
            }
            _ => crate::HydrologySample::default(),
        });
        let mut change = [0.0];
        if reads.erosion {
            erosion_row(stack, &[point[0]], point[1], spacing, &mut change);
        }
        let shape = (self.gradient.len() - 1, self.curvature.len() - 1);
        self.erosion.push(rested(
            change[0],
            stack.rests(crate::layer::maps::DEPOSIT),
            self.gradient[shape.0],
            self.curvature[shape.1],
        ));
        self.path.push(if reads.path {
            stack.path_at(point[0], point[1])
        } else {
            crate::PathSample::FAR
        });
    }

    fn push_maps(
        &mut self,
        (stack, users, producers): MapSources<'_>,
        ([x, z], spacing, height): ([f32; 2], f32, f32),
    ) {
        let at = glam::Vec2::new(x, z);
        let shape = (
            self.gradient.last().copied().unwrap_or_default(),
            self.curvature.last().copied().unwrap_or_default(),
        );
        let row = crate::layer::Row {
            xs: std::slice::from_ref(&x),
            z,
            spacing,
            low: at,
            high: at,
        };
        for ((values, (name, _)), &producer) in self.maps.iter_mut().zip(users).zip(producers) {
            let mut value = [0.0_f32];
            read_map(stack, producer, name, &row, &[height], &mut value);
            let rests = producer
                .and_then(|index| stack.layers().get(index)?.rests(name))
                .map_or(1.0, |rests| rests.share(shape.0, shape.1));
            values.push(value[0] * rests);
        }
    }

    fn row(&self, spacing: f32) -> RowInputs<'_> {
        RowInputs {
            xs: &self.xs,
            zs: &self.zs,
            spacing,
            height: &self.height,
            gradient: &self.gradient,
            curvature: &self.curvature,
            water: &self.water,
            erosion: &self.erosion,
            path: &self.path,
        }
    }
}

#[derive(Clone, Copy)]
struct Spot {
    key: u32,
    rank: f32,
    pick: f32,
    point: [f32; 2],
}

impl Spot {
    fn at(seed: u32, [x, z]: [i64; 2], footprint: f32) -> Self {
        let key =
            hash(seed ^ (x as u32).wrapping_mul(0x8da6_b343) ^ (z as u32).wrapping_mul(0xd816_3841));
        Self {
            key,
            rank: unit(hash(key ^ 0x1656_67b1)),
            pick: unit(hash(key ^ 0x7f4a_7c15)),
            point: [
                (x as f32 + unit(key)) * footprint,
                (z as f32 + unit(hash(key ^ 0x5bd1_e995))) * footprint,
            ],
        }
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    square: [i64; 2],
    spot: Spot,
    local: [f32; 2],
    sample: super::TileSample,
}

struct Ground<'a> {
    tile: &'a Tile,
    stack: &'a crate::Layers,
    rules: &'a super::BiomeRules,
    origin: [f32; 2],
    size: f32,
}

impl Ground<'_> {
    fn control(&self, nearest: [i32; 2], controls: &mut Vec<u32>) -> Control {
        if let Some(control) = self.tile.control_at(nearest[0], nearest[1]) {
            return control;
        }
        let stride = TILE_STRIDE as usize;
        let stored = |sample: i32| (sample.clamp(-1, stride as i32 - 2) + 1) as usize;
        let (column, row) = (stored(nearest[0]), stored(nearest[1]));
        if controls.len() != stride * stride {
            controls.clear();
            controls.resize(stride * stride, u32::MAX);
        }
        let held = &mut controls[row * stride + column];
        if *held == u32::MAX {
            let from = (self.stack, self.rules);
            *held = super::gpu::sample_control(self.tile, from, (column, row)).pack();
        }
        Control::unpack(*held).unwrap_or_default()
    }
}

fn gather(
    group: &CompiledPlacement,
    ground: &Ground<'_>,
    (seed, squares, square_z): (u32, std::ops::RangeInclusive<i64>, i64),
    scratch: &mut PlaceScratch,
) {
    let PlaceScratch {
        candidates,
        inputs,
        weights,
        controls,
        ..
    } = scratch;
    let (tile, stack) = (ground.tile, ground.stack);
    let spacing = tile.spacing();
    candidates.clear();
    weights.clear();
    inputs.clear(group.users.len());
    let producers: Vec<Option<usize>> = (group.users.iter())
        .map(|(name, _)| producer(stack.layers(), name))
        .collect();
    let most = group.most();
    for square_x in squares {
        let square = [square_x, square_z];
        let spot = Spot::at(seed, square, group.footprint);
        if spot.rank >= most {
            continue;
        }
        let local = [
            spot.point[0] - ground.origin[0],
            spot.point[1] - ground.origin[1],
        ];
        let size = ground.size;
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
        let control = ground.control(nearest, controls);
        let held = weights.len();
        let mut most_here = 0.0;
        for member in &group.members {
            let share = material_share(control, &member.materials);
            most_here += member.fill * share;
            weights.push(share);
        }
        if spot.rank >= min(most_here, 1.0) {
            weights.truncate(held);
            continue;
        }
        candidates.push(Candidate {
            square,
            spot,
            local,
            sample,
        });
        inputs.push((tile, stack, group.reads), (spot.point, local, nearest), &sample);
        inputs.push_maps(
            (stack, &group.users, &producers),
            (spot.point, spacing, sample.height),
        );
    }
}

fn weigh(group: &CompiledPlacement, spacing: f32, scratch: &mut PlaceScratch) {
    let PlaceScratch {
        candidates,
        inputs,
        values,
        suited,
        weights,
        sizes,
        rows,
        ..
    } = scratch;
    let (count, members) = (candidates.len(), group.members.len());
    let row = inputs.row(spacing);
    sizes.clear();
    sizes.resize(count * members, 0.0);
    for list in [&mut *values, &mut *suited] {
        list.clear();
        list.resize(count, 0.0);
    }
    for (index, member) in group.members.iter().enumerate() {
        member.mask.evaluate_row(&row, &inputs.maps, values, rows);
        for (at, &mask) in values.iter().enumerate() {
            let (share, mask) = (weights[at * members + index], clamp(mask, 0.0, 1.0));
            suited[at] = share * mask;
            weights[at * members + index] = member.fill * share * mask;
        }
        let Some(sizing) = &member.sized else {
            continue;
        };
        let maps = if sizing.own {
            std::slice::from_ref(&*suited)
        } else {
            inputs.maps.as_slice()
        };
        sizing.by.evaluate_row(&row, maps, values, rows);
        for (at, &size) in values.iter().enumerate() {
            sizes[at * members + index] = size;
        }
    }
}

pub(crate) fn place_row(
    groups: &[CompiledPlacement],
    tile: &Tile,
    seed: u32,
    (group_index, row): (usize, usize),
    out: &mut Vec<Placed>,
    (stack, (placing, from), scratch, rules): (
        &crate::Layers,
        (Placing, Option<u8>),
        &mut PlaceScratch,
        &super::BiomeRules,
    ),
) -> (u64, Option<(usize, usize)>) {
    let spacing = tile.spacing();
    let size = TILE_QUADS as f32 * spacing;
    let key = tile.key();
    let origin = [f32::from(key.x) * size, f32::from(key.z) * size];
    let Some(group) = groups.get(group_index) else {
        return (0, None);
    };
    let (level, far) = match from {
        Some(from) => (group.level.max(from), None),
        None => (group.level, group.far),
    };
    let near_level = match far {
        _ if level == key.level => None,
        Some(far) if far == key.level => Some(group.level),
        _ => return (0, Some((group_index + 1, 0))),
    };
    let skipped = match placing {
        Placing::All => false,
        Placing::Near => near_level.is_some(),
        Placing::Far => near_level.is_none(),
    };
    let most = group.most();
    if most <= 0.0 || skipped {
        return (0, Some((group_index + 1, 0)));
    }
    let footprint = group.footprint;
    let group_seed = hash(seed ^ u32::from(group.members[0].layer).wrapping_mul(0x632b_e5ab));
    let first = [
        (origin[0] / footprint).floor() as i64,
        (origin[1] / footprint).floor() as i64,
    ];
    let last = [
        ((origin[0] + size) / footprint).floor() as i64,
        ((origin[1] + size) / footprint).floor() as i64,
    ];
    let square_z = first[1] + row as i64;
    let ground = Ground {
        tile,
        stack,
        rules,
        origin,
        size,
    };
    gather(group, &ground, (group_seed, first[0]..=last[0], square_z), scratch);
    weigh(group, spacing, scratch);
    let members = group.members.len();
    for (index, candidate) in scratch.candidates.iter().enumerate() {
        let held = index * members..(index + 1) * members;
        let site = (&scratch.weights[held.clone()], &scratch.sizes[held]);
        let Some(standing) = group.standing(&candidate.spot, site) else {
            continue;
        };
        let me = (&candidate.spot, candidate.square, standing.room);
        if group.reach > 0 && group.yields(group_seed, me, site, YIELD_DEPTH) {
            continue;
        }
        out.push(group.members[standing.member].placed(candidate, standing, tile, near_level));
    }
    let squares = (last[0] - first[0] + 1) as u64;
    let kept = (squares as f32 * most).ceil() as u64;
    let cost = squares * SQUARE_COST + kept * group.cost();
    let next = if square_z < last[1] {
        (group_index, row + 1)
    } else {
        (group_index + 1, 0)
    };
    (cost, Some(next))
}

engine_codec::record!(PlacementChoice { asset, weight });
engine_codec::record!(FarPlacement { level });
engine_codec::record!(PlacementLayer {
    id,
    group,
    assets,
    density,
    footprint,
    room,
    materials,
    mask,
    scale,
    align,
    downhill,
    sink,
    bed,
    level,
    far,
    sized,
});
engine_codec::record!(SizedBy {
    map,
    from,
    to,
    tendency,
});
engine_codec::record!(AuthoredPlacement {
    asset,
    position,
    yaw,
    scale,
    sink,
    align,
    level,
});
engine_codec::record!(Placed {
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
engine_codec::tagged!(Placing {
    0 => All,
    1 => Near,
    2 => Far,
});

type MapSources<'a> = (
    &'a crate::Layers,
    &'a [(String, String)],
    &'a [Option<usize>],
);
