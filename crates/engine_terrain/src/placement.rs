use engine_core::{Error, Expected};
use engine_math::{clamp, max, min, sin_cos};
use engine_core::{ExtraRules, InvalidSetting, fields};
use serde::{Deserialize, Serialize};

use crate::expr::Expr;
use crate::expr_layer::{producer, read_map};
use crate::surface::{
    Compiled, Control, Reads, RowInputs, RowScratch, Scope, erosion_row, setting_error,
    terrain_error, unknown, unknown_map,
};
use crate::{Placed, TILE_QUADS, TILE_STRIDE, Tile};

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
        pub assets: Vec<PlacementChoice> = Vec::new() => any,
        pub density: f32 = 0.0 => at_least(0.0),
        pub footprint: f32 = 1.0 => range(0.25, 256.0),
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
    }
}

impl ExtraRules for PlacementLayer {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.density * self.footprint * self.footprint / 10_000.0 > 1.0 {
            return Err(InvalidSetting::new(
                "density",
                engine_core::Expected::Number,
            ));
        }
        if self.scale[0] > self.scale[1] {
            return Err(InvalidSetting::new(
                "scale",
                engine_core::Expected::Sorted,
            ));
        }
        if self.assets.iter().map(|choice| choice.weight).sum::<f32>() <= 0.0 {
            return Err(InvalidSetting::new(
                "assets",
                engine_core::Expected::Nonzero,
            ));
        }
        if self.far.is_some_and(|far| far.level <= self.level) {
            return Err(InvalidSetting::new(
                "far.level",
                engine_core::Expected::Above,
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

pub fn footprint(placed: &Placed, solid: &Solid) -> Footprint {
    use glam::{Mat3, Mat4, Vec3};
    let model = placed.model(Vec3::ZERO, placed.position[1])
        * Mat4::from_translation(Vec3::from_array(solid.center));
    let axes = Mat3::from_mat4(model) * Mat3::from_diagonal(Vec3::from_array(solid.half));
    Footprint {
        center: model.w_axis.truncate().to_array(),
        axes: axes.to_cols_array(),
        round: solid.round,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Solid {
    pub center: [f32; 3],
    pub half: [f32; 3],
    pub round: bool,
}

impl Solid {
    pub fn valid(&self) -> bool {
        self.center.iter().all(|value| value.is_finite())
            && self.half.iter().all(|&half| half.is_finite() && half > 0.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obstacle {
    pub center: glam::Vec2,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Footprint {
    pub center: [f32; 3],
    pub axes: [f32; 9],
    pub round: bool,
}

impl Footprint {
    fn axes(&self) -> glam::Mat3 {
        glam::Mat3::from_cols_array(&self.axes)
    }

    pub fn middle(&self) -> glam::Vec2 {
        glam::Vec2::new(self.center[0], self.center[2])
    }

    pub fn reach(&self) -> glam::Vec2 {
        let rows = self.axes().transpose();
        let along = |row: glam::Vec3| {
            if self.round {
                row.length()
            } else {
                row.abs().element_sum()
            }
        };
        glam::Vec2::new(along(rows.x_axis), along(rows.z_axis))
    }

    pub fn span(&self, point: glam::Vec2) -> Option<[f32; 2]> {
        let inverse = self.axes().inverse();
        let at = inverse
            * (glam::Vec3::new(point.x, 0.0, point.y) - glam::Vec3::from_array(self.center));
        let up = inverse.y_axis;
        if self.round {
            let rise = up.dot(up);
            let along = at.dot(up);
            let reach = along * along - rise * (at.dot(at) - 1.0);
            if !(rise > 0.0 && reach >= 0.0) {
                return None;
            }
            let root = reach.sqrt();
            return Some([(-along - root) / rise, (-along + root) / rise]);
        }
        let mut span = [f32::NEG_INFINITY, f32::INFINITY];
        for k in 0..3 {
            if up[k] != 0.0 {
                let (a, b) = ((-1.0 - at[k]) / up[k], (1.0 - at[k]) / up[k]);
                span = [max(span[0], min(a, b)), min(span[1], max(a, b))];
            } else if at[k].abs() > 1.0 {
                return None;
            }
        }
        (span[0] <= span[1]).then_some(span)
    }

    pub(crate) fn valid(&self) -> bool {
        self.center
            .iter()
            .chain(&self.axes)
            .all(|value| value.is_finite())
            && self.axes().determinant() != 0.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompiledPlacement {
    pub(crate) level: u8,
    pub(crate) far: Option<u8>,
    choices: Vec<(u16, f32)>,
    fill: f32,
    footprint: f32,
    materials: Option<[u64; 4]>,
    mask: Compiled,
    reads: Reads,
    users: Vec<(String, String)>,
    scale: [f32; 2],
    align: f32,
    downhill: f32,
    sink: f32,
    bed: f32,
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
    let mut compiled = Vec::with_capacity(layers.len());
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
                        unknown(format!("{at}.materials[{material_index}]"), id, materials)
                    })?;
                set[material / 64] |= 1 << (material % 64);
            }
            Some(set)
        };
        let mut scope = Scope::cpu();
        let mask = Compiled::from(&layer.mask, &format!("{at}.mask"), 0, &mut scope)?;
        compiled.push(CompiledPlacement {
            level: layer.level,
            far: layer.far.map(|far| far.level),
            choices,
            fill,
            footprint: layer.footprint,
            materials: set,
            reads: mask.reads(),
            users: scope.users(),
            mask,
            scale: layer.scale,
            align: layer.align,
            downhill: layer.downhill,
            sink: layer.sink,
            bed: layer.bed,
        });
    }
    Ok(compiled)
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
        sample: &crate::TileSample,
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
        self.erosion.push(change[0]);
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
            values.push(value[0]);
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
struct Candidate {
    square: u32,
    rank: f32,
    point: [f32; 2],
    local: [f32; 2],
    sample: crate::TileSample,
    share_of_materials: f32,
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
    let spacing = tile.spacing();
    let size = TILE_QUADS as f32 * spacing;
    let key = tile.key();
    let origin = [f32::from(key.x) * size, f32::from(key.z) * size];
    let Some(layer) = layers.get(layer_index) else {
        return (0, None);
    };
    let near_level = match layer.far {
        _ if layer.level == key.level => None,
        Some(level) if level == key.level => Some(layer.level),
        _ => return (0, Some((layer_index + 1, 0))),
    };
    let skipped = match placing {
        Placing::All => false,
        Placing::Near => near_level.is_some(),
        Placing::Far => near_level.is_none(),
    };
    let fill = layer.fill;
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
    let PlaceScratch {
        candidates,
        inputs,
        values,
        rows,
        controls,
    } = scratch;
    candidates.clear();
    inputs.clear(layer.users.len());
    let producers: Vec<Option<usize>> = (layer.users.iter())
        .map(|(name, _)| producer(stack.layers(), name))
        .collect();
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
        if rank >= fill * share_of_materials {
            continue;
        }
        candidates.push(Candidate {
            square,
            rank,
            point,
            local,
            sample,
            share_of_materials,
        });
        inputs.push((tile, stack, layer.reads), (point, local, nearest), &sample);
        inputs.push_maps(
            (stack, &layer.users, &producers),
            (point, spacing, sample.height),
        );
    }
    values.clear();
    values.resize(candidates.len(), 0.0);
    layer
        .mask
        .evaluate_row(&inputs.row(spacing), &inputs.maps, values, rows);
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
        let full = fill * share_of_materials * clamp(mask, 0.0, 1.0);
        if rank >= full {
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
        let ranked = rank / full;
        let scale = layer.scale[1] - (layer.scale[1] - layer.scale[0]) * ranked;
        let (turn_sin, turn_cos) =
            sin_cos(unit(hash(square ^ 0x85eb_ca6b)) * std::f32::consts::TAU);
        let downhill = layer.downhill / max(normal.y, 1.0e-3);
        let yaw = libm::atan2f(
            turn_sin + normal.z * downhill,
            turn_cos + normal.x * downhill,
        );
        let (wobble_sin, wobble_cos) =
            sin_cos(unit(hash(square ^ 0x9e37_79b9)) * std::f32::consts::TAU);
        let lean = [
            normal.x * layer.align + 0.03 * wobble_cos,
            normal.z * layer.align + 0.03 * wobble_sin,
        ];
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
            rank: ranked,
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
crate::codec::record!(FarPlacement { level });
crate::codec::record!(PlacementLayer {
    id,
    assets,
    density,
    footprint,
    materials,
    mask,
    scale,
    align,
    downhill,
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
crate::codec::record!(Solid {
    center,
    half,
    round
});
crate::codec::record!(Footprint {
    center,
    axes,
    round
});

type MapSources<'a> = (
    &'a crate::Layers,
    &'a [(String, String)],
    &'a [Option<usize>],
);
