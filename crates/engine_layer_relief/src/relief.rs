use std::any::Any;
use std::sync::Arc;

use engine_codec::{Reader, Writer};
use engine_core::{Error, InvalidSetting, fields, parse_value};
use engine_heightfield::hydrology::{FLOW_RANGE, WETNESS_RANGE, WaterGrid, decode};
use engine_heightfield::layer::{
    Below, LayerFiles, LayerKind, Made, Place, Row, TerrainLayer, no_files, no_points, same_as,
};
use engine_heightfield::noise::{gradient_value, hash, smooth};
use engine_heightfield::tile::{producer, read_map, unknown_map};
use engine_heightfield::{HydrologyBake, INVALID_TERRAIN, MISSING_LAYER};
use engine_math::{clamp, max, min, sin_cos};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const RESOLVED: f32 = 2.0;

const STRATA_WARP: u32 = 0x243f_6a88;
const STRATA_BEDS: u32 = 0xa54f_f53a;
const STRATA_JOINTS: u32 = 0x3c6e_f372;
const JOINT_BLOCKS: u32 = 0x5be0_cd19;
const SPAN_MARKS: u32 = 0x510e_527f;
const SPAN_PARTS: u32 = 0x6a09_e667;
const MAX_SPLITS: u32 = 8;
const HUMMOCK_PATCHES: u32 = 0x9b05_688c;
const HUMMOCK_MOUNDS: u32 = 0x1f83_d9ab;

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct JointSet {
        pub heading: f32 = 0.0 => range(-360.0, 360.0),
        pub spacing: f32 = 1.5 => range(0.1, 1000.0),
        pub uneven: f32 = 0.0 => range(0.0, 0.45),
        pub splits: u32 = 1 => range(1, MAX_SPLITS),
        pub offset: f32 = 0.0 => range(0.0, 1.0),
    }
}

fields! {
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct StrataLayer {
        pub seed: u32 = 0 => any,
        pub thickness: f32 = 6.0 => range(0.5, 500.0),
        pub uneven: f32 = 0.0 => range(0.0, 0.45),
        pub splits: u32 = 1 => range(1, MAX_SPLITS),
        pub ledges: f32 = 0.7 => range(0.0, 1.0),
        pub strength: f32 = 0.8 => range(0.0, 1.0),
        pub varied: f32 = 0.45 => range(0.0, 1.0),
        pub riser: f32 = 0.3 => range(0.05, 1.0),
        pub patches: f32 = 14.0 => range(0.5, 1.0e5),
        pub contrast: f32 = 0.0 => range(0.0, 100.0),
        pub joints: Vec<JointSet> = Vec::new() => any,
        pub dip: [f32; 2] = [0.0, 0.0] => range(-1.0, 1.0),
        pub warp: [f32; 2] = [0.0, 1000.0] => range(0.0, 1.0e6),
        pub mask: String = String::new() => any,
    }
}

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct HummocksLayer {
        pub seed: u32 = 0 => any,
        pub height: f32 = 0.5 => range(0.0, 10.0),
        pub across: f32 = 1.5 => range(0.1, 100.0),
        pub slope: [f32; 2] = [15.0, 6.0] => range(0.0, 89.0),
        pub wetness: [f32; 2] = [8.0, 11.0] => range(WETNESS_RANGE[0], WETNESS_RANGE[1]),
        pub patches: f32 = 80.0 => range(1.0, 1.0e5),
        pub cover: f32 = 0.7 => range(0.0, 1.0),
        pub edge: f32 = 0.15 => range(0.001, 1.0),
        pub clear: f32 = 16.0 => range(0.001, 1.0e5),
    }
}

pub static STRATA: LayerKind = LayerKind {
    name: "strata",
    source: crate::SOURCE.hash(),
    from_json: strata_from_json,
    read: read_strata,
    files: no_files,
    schema: engine_heightfield::schema::<StrataLayer>,
    points: no_points,
};

pub static HUMMOCKS: LayerKind = LayerKind {
    name: "hummocks",
    source: crate::SOURCE.hash(),
    from_json: hummocks_from_json,
    read: read_hummocks,
    files: no_files,
    schema: engine_heightfield::schema::<HummocksLayer>,
    points: no_points,
};

fn strata(strata: StrataLayer) -> Made {
    checked(strata.validate())?;
    for (index, set) in strata.joints.iter().enumerate() {
        set.validate()
            .map_err(|invalid| invalid.error(INVALID_TERRAIN, &format!("$.joints[{index}]")))?;
    }
    Ok(Arc::new(strata))
}

fn strata_from_json(settings: &Value, _files: &dyn LayerFiles) -> Made {
    strata(parse_value(settings.clone())?)
}

fn read_strata(input: &mut Reader<'_>) -> Made {
    strata(input.get()?)
}

fn hummocks_from_json(settings: &Value, _files: &dyn LayerFiles) -> Made {
    let hummocks: HummocksLayer = parse_value(settings.clone())?;
    checked(hummocks.validate())?;
    Ok(Arc::new(hummocks))
}

fn read_hummocks(input: &mut Reader<'_>) -> Made {
    let hummocks: HummocksLayer = input.get()?;
    checked(hummocks.validate())?;
    Ok(Arc::new(hummocks))
}

fn checked(valid: Result<(), InvalidSetting>) -> Result<(), Error> {
    valid.map_err(|invalid| invalid.error(INVALID_TERRAIN, "$"))
}

fn water_below(place: Place<'_>) -> Result<(), Error> {
    if place.below().iter().any(|layer| layer.makes_water()) {
        Ok(())
    } else {
        Err(Error::new(MISSING_LAYER, "$").variants(["hydrology"]))
    }
}

fn unit(value: u32) -> f32 {
    (value >> 8) as f32 * (1.0 / 16_777_216.0)
}

impl TerrainLayer for StrataLayer {
    fn kind(&self) -> &'static LayerKind {
        &STRATA
    }

    fn write(&self, out: &mut Writer) {
        out.put(self);
    }

    fn settings(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        same_as(self, other)
    }

    fn check(&self, place: Place<'_>) -> Result<(), Error> {
        if self.mask.is_empty() {
            return Ok(());
        }
        let published: Vec<String> = (place.below().iter())
            .flat_map(|layer| layer.outputs())
            .collect();
        if published.contains(&self.mask) {
            Ok(())
        } else {
            Err(unknown_map("$.mask", &self.mask, published))
        }
    }

    fn apply_row(&self, below: Below<'_>, at: &Row<'_>, row: &mut [f32]) {
        let Row { xs, z, spacing, .. } = *at;
        let thickest = self.thickness * (1.0 + 2.0 * self.uneven);
        if thickest <= RESOLVED * spacing || self.strength <= 0.0 || self.ledges <= 0.0 {
            return;
        }
        let count = row.len().min(xs.len());
        let mut exposed = vec![1.0_f32; count];
        if !self.mask.is_empty() {
            let masked = Row {
                xs: &xs[..count],
                ..*at
            };
            read_map(
                below.stack(),
                producer(below.layers(), &self.mask),
                &self.mask,
                &masked,
                &row[..count],
                &mut exposed,
            );
        }
        let joints: Vec<Joints> = (self.joints.iter().enumerate())
            .map(|(index, set)| Joints::new(set, self, index as u32))
            .collect();
        for ((height, &x), &exposed) in row.iter_mut().zip(xs).zip(&exposed) {
            let weight = clamp(exposed, 0.0, 1.0);
            if weight <= 0.0 {
                continue;
            }
            let shown = below.shown(x, z);
            if shown > 0.0 {
                *height += weight * shown * self.ledge(&joints, (x, z, spacing), *height);
            }
        }
    }

    fn row_cost(&self) -> u64 {
        2 + self.joints.len() as u64
    }
}

struct Span {
    id: i32,
    low: f32,
    high: f32,
}

fn span(at: f32, uneven: f32, splits: u32, roll: &impl Fn(i32, u32) -> f32) -> Span {
    let mark = |index: i32| index as f32 + uneven * (2.0 * roll(index, SPAN_MARKS) - 1.0);
    let nearest = (at + 0.5).floor() as i32;
    let between = mark(nearest);
    let (stretch, low, high) = if at >= between {
        (nearest, between, mark(nearest + 1))
    } else {
        (nearest - 1, mark(nearest - 1), between)
    };
    let parts = (1 + (roll(stretch, SPAN_PARTS) * splits as f32) as u32).min(splits);
    let width = (high - low) / parts as f32;
    let part = (((at - low) / width) as u32).min(parts - 1);
    Span {
        id: stretch
            .wrapping_mul(MAX_SPLITS as i32)
            .wrapping_add(part as i32),
        low: low + part as f32 * width,
        high: low + (part + 1) as f32 * width,
    }
}

struct Joints {
    normal: [f32; 2],
    spacing: f32,
    uneven: f32,
    splits: u32,
    offset: f32,
    band: f32,
    seed: u32,
}

impl Joints {
    fn new(set: &JointSet, layer: &StrataLayer, index: u32) -> Self {
        let (sin, cos) = sin_cos(set.heading.to_radians());
        let narrowest = (1.0 - 2.0 * set.uneven) / set.splits as f32;
        Self {
            normal: [-sin / set.spacing, cos / set.spacing],
            spacing: set.spacing,
            uneven: set.uneven,
            splits: set.splits,
            offset: set.offset,
            band: min(layer.riser / set.spacing, narrowest),
            seed: layer.seed ^ STRATA_JOINTS.wrapping_mul(index + 1),
        }
    }

    fn stands(&self, (x, z, spacing): (f32, f32, f32), bed: i32, thick: f32) -> f32 {
        let roll = |index: i32, salt: u32| unit(hash(index, bed, self.seed ^ salt));
        let grain = RESOLVED * spacing / (self.spacing * thick);
        let side = |block: &Span| {
            let resolved = smooth((block.high - block.low) / grain - 1.0);
            resolved * (2.0 * roll(block.id, JOINT_BLOCKS) - 1.0)
        };
        let across = (x * self.normal[0] + z * self.normal[1]) / thick + roll(0, STRATA_JOINTS);
        let half = 0.5 * self.band;
        let behind = span(across - half, self.uneven, self.splits, &roll);
        let ahead = span(across + half, self.uneven, self.splits, &roll);
        if behind.id == ahead.id {
            return self.offset * side(&behind);
        }
        let crossed = smooth((across - ahead.low) / self.band + 0.5);
        let (from, to) = (side(&behind), side(&ahead));
        self.offset * (from + (to - from) * crossed)
    }
}

impl StrataLayer {
    fn ledge(&self, joints: &[Joints], (x, z, spacing): (f32, f32, f32), height: f32) -> f32 {
        let [amplitude, wavelength] = self.warp;
        let across = 1.0 / max(wavelength, 1.0);
        let datum = self.dip[0] * x
            + self.dip[1] * z
            + amplitude * gradient_value(x * across, z * across, self.seed ^ STRATA_WARP);
        let level = (height + datum) / self.thickness;
        let seed = self.seed ^ STRATA_BEDS;
        let bed = span(level, self.uneven, self.splits, &|index, salt| {
            unit(hash(index, 3, seed ^ salt))
        });
        let number = bed.id;
        if unit(hash(number, 0, seed)) >= self.ledges {
            return 0.0;
        }
        let share = bed.high - bed.low;
        let thick = share * self.thickness;
        let resolved = smooth(thick / (RESOLVED * spacing) - 1.0);
        if resolved <= 0.0 {
            return 0.0;
        }
        let within = (level - bed.low) / share;
        let firm = 1.0 - self.varied * unit(hash(number, 1, seed));
        let stands: f32 = (joints.iter())
            .map(|set| set.stands((x, z, spacing), number, thick))
            .sum();
        let place = clamp(unit(hash(number, 2, seed)) + stands, 0.0, 1.0);
        let riser = 0.5 * self.riser + (1.0 - self.riser) * place;
        let stepped = smooth((within - riser) / self.riser + 0.5);
        resolved * thick * self.strength * firm * self.run(x, z, number) * (stepped - within)
    }

    fn run(&self, x: f32, z: f32, bed: i32) -> f32 {
        if self.contrast <= 0.0 {
            return 1.0;
        }
        let along = 1.0 / self.patches;
        let shift = (bed % 1024) as f32;
        let broken = gradient_value(
            x * along + 7.3 * shift,
            z * along - 3.1 * shift,
            self.seed ^ STRATA_BEDS,
        );
        smooth(0.5 + self.contrast * broken)
    }
}

impl TerrainLayer for HummocksLayer {
    fn kind(&self) -> &'static LayerKind {
        &HUMMOCKS
    }

    fn write(&self, out: &mut Writer) {
        out.put(self);
    }

    fn settings(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn same(&self, other: &dyn TerrainLayer) -> bool {
        same_as(self, other)
    }

    fn check(&self, place: Place<'_>) -> Result<(), Error> {
        water_below(place)
    }

    fn apply_row(&self, below: Below<'_>, at: &Row<'_>, row: &mut [f32]) {
        let Row { xs, z, spacing, .. } = *at;
        if self.height <= 0.0 || self.cover <= 0.0 || 2.0 * self.across < RESOLVED * spacing {
            return;
        }
        let placement = Placement::new(
            (self.slope, self.wetness),
            (self.patches, self.cover, self.edge),
            self.seed ^ HUMMOCK_PATCHES,
        );
        let half = 0.5 * self.height;
        relieve(below, (&placement, self.clear), (row, xs, z, spacing), |x, z| {
            half * blobs(x, z, self.across, self.seed ^ HUMMOCK_MOUNDS)
        });
    }
}

struct Placement {
    slope: [f32; 2],
    wetness: [f32; 2],
    patches: f32,
    cover: f32,
    edge: f32,
    seed: u32,
}

impl Placement {
    fn new(
        (slope, wetness): ([f32; 2], [f32; 2]),
        (patches, cover, edge): (f32, f32, f32),
        seed: u32,
    ) -> Self {
        Self {
            slope: slope.map(|degrees| libm::tanf(degrees.to_radians())),
            wetness,
            patches,
            cover,
            edge,
            seed,
        }
    }

    fn weight(&self, x: f32, z: f32, slope: f32, wetness: f32) -> f32 {
        let ramps = ramp(slope, self.slope) * ramp(wetness, self.wetness);
        if ramps <= 0.0 {
            return 0.0;
        }
        let patch = 0.5 + 0.5 * blobs(x, z, self.patches, self.seed);
        ramps * smooth((patch - (1.0 - self.cover)) / self.edge)
    }
}

fn relieve(
    below: Below<'_>,
    (placement, clear): (&Placement, f32),
    (row, xs, z, spacing): (&mut [f32], &[f32], f32, f32),
    mut shape: impl FnMut(f32, f32) -> f32,
) {
    let Some(hydrology) = below.water() else {
        return;
    };
    let count = row.len().min(xs.len());
    let level = hydrology.level_for(spacing);
    let scale = 1.0 / hydrology.spacing;
    let v = z * scale;
    let mut flows = vec![0.0_f32; count];
    hydrology.flow.sample_smooth_row(
        xs[..count].iter().map(|&x| x * scale),
        v,
        level,
        |index, sample| flows[index] = sample,
    );
    hydrology.wetness.sample_smooth_row(
        xs[..count].iter().map(|&x| x * scale),
        v,
        level,
        |index, sample| {
            let x = xs[index];
            let wetness = decode(sample, WETNESS_RANGE);
            let flow = decode(flows[index], FLOW_RANGE);
            let slope = map_slope(flow, wetness, hydrology.spacing);
            let placed = placement.weight(x, z, slope, wetness) * below.shown(x, z);
            if placed <= 0.0 {
                return;
            }
            let weight = placed * calm(hydrology, (x, z, spacing), clear);
            if weight > 0.0 {
                row[index] += weight * shape(x, z);
            }
        },
    );
}

fn map_slope(flow: f32, wetness: f32, spacing: f32) -> f32 {
    libm::expf(flow * std::f32::consts::LN_10 - wetness) / spacing
}

pub(crate) fn calm(
    hydrology: &HydrologyBake,
    (x, z, spacing): (f32, f32, f32),
    clear: f32,
) -> f32 {
    let half = 0.5 * clear;
    let banks = smooth((hydrology.rivers.edge(x, z, spacing) - half) / half);
    if banks <= 0.0 {
        return 0.0;
    }
    banks * (1.0 - lake_share(&hydrology.water, x, z))
}

fn lake_share(water: &WaterGrid, x: f32, z: f32) -> f32 {
    if water.levels.is_empty() {
        return 0.0;
    }
    let (u, v) = (x / water.spacing, z / water.spacing);
    if !(u.is_finite() && v.is_finite()) {
        return 0.0;
    }
    let (i, j) = (u.floor(), v.floor());
    let (fu, fv) = (u - i, v - j);
    let (last_x, last_z) = (water.width as i64 - 1, water.height as i64 - 1);
    let held = |dx: i64, dz: i64| {
        let column = (i as i64 + dx).clamp(0, last_x) as usize;
        let row = (j as i64 + dz).clamp(0, last_z) as usize;
        if water.lake[row * water.width + column] == 0 {
            0.0
        } else {
            1.0
        }
    };
    (1.0 - fu) * (1.0 - fv) * held(0, 0)
        + fu * (1.0 - fv) * held(1, 0)
        + (1.0 - fu) * fv * held(0, 1)
        + fu * fv * held(1, 1)
}

fn ramp(value: f32, [from, to]: [f32; 2]) -> f32 {
    if from == to {
        return if value >= to { 1.0 } else { 0.0 };
    }
    smooth((value - from) / (to - from))
}

pub(crate) fn blobs(x: f32, z: f32, across: f32, seed: u32) -> f32 {
    let lattice = 2.0 * across;
    let (u, v) = (x / lattice, z / lattice);
    gradient_value(0.8 * u - 0.6 * v + 0.37, 0.6 * u + 0.8 * v + 0.61, seed)
}

engine_codec::record!(JointSet {
    heading,
    spacing,
    uneven,
    splits,
    offset,
});
engine_codec::record!(StrataLayer {
    seed,
    thickness,
    uneven,
    splits,
    ledges,
    strength,
    varied,
    riser,
    patches,
    contrast,
    joints,
    dip,
    warp,
    mask,
});
engine_codec::record!(HummocksLayer {
    seed,
    height,
    across,
    slope,
    wetness,
    patches,
    cover,
    edge,
    clear,
});
