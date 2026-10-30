use std::any::Any;
use std::sync::Arc;

use engine_core::{Error, ErrorCode};
use engine_math::min;
use engine_settings::{InvalidSetting, parse_value, settings};
use engine_terrain::codec::{Reader, Writer};
use engine_terrain::hydrology::{FLOW_RANGE, Hydrology, WETNESS_RANGE, WaterGrid, decode};
use engine_terrain::layer::{
    Below, LayerFiles, LayerKind, Made, Place, Row, TerrainLayer, no_files, same_as,
};
use engine_terrain::noise::{gradient_value, smooth};
use engine_terrain::rivers::{BANK_REACH, BANK_SLOPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const STEP_HEIGHT: f32 = 3.0;
const JOINT_SPACING: f32 = 4.0;
const HUMMOCK_HEIGHT: f32 = 0.5;
const HUMMOCK_ACROSS: f32 = 1.5;

const RESOLVED: f32 = 2.0;
const PATCH_EDGE: f32 = 0.15;

const STEP_PATCHES: u32 = 0x3c6e_f372;
const STEP_BEDDING: u32 = 0xa54f_f53a;
const STEP_JOINTS: u32 = 0x510e_527f;
const HUMMOCK_PATCHES: u32 = 0x9b05_688c;
const HUMMOCK_MOUNDS: u32 = 0x1f83_d9ab;

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct StepsLayer {
        pub seed: u32 = 0 => any,
        pub height: f32 = STEP_HEIGHT => range(0.1, 100.0),
        pub joints: f32 = JOINT_SPACING => range(0.5, 1000.0),
        pub slope: [f32; 2] = [repose(), 40.0] => range(0.0, 89.0),
        pub wetness: [f32; 2] = [8.0, 5.0] => range(WETNESS_RANGE[0], WETNESS_RANGE[1]),
        pub patches: f32 = 60.0 => range(1.0, 1.0e5),
        pub cover: f32 = 0.6 => range(0.0, 1.0),
    }
}

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct HummocksLayer {
        pub seed: u32 = 0 => any,
        pub height: f32 = HUMMOCK_HEIGHT => range(0.0, 10.0),
        pub across: f32 = HUMMOCK_ACROSS => range(0.1, 100.0),
        pub slope: [f32; 2] = [15.0, 6.0] => range(0.0, 89.0),
        pub wetness: [f32; 2] = [8.0, 11.0] => range(WETNESS_RANGE[0], WETNESS_RANGE[1]),
        pub patches: f32 = 80.0 => range(1.0, 1.0e5),
        pub cover: f32 = 0.7 => range(0.0, 1.0),
    }
}

pub static STEPS: LayerKind = LayerKind {
    name: "steps",
    source: crate::SOURCE,
    from_json: steps_from_json,
    read: read_steps,
    files: no_files,
    schema: engine_terrain::layer_schema!(steps_schema),
    points: engine_terrain::layer::no_points,
};

pub static HUMMOCKS: LayerKind = LayerKind {
    name: "hummocks",
    source: crate::SOURCE,
    from_json: hummocks_from_json,
    read: read_hummocks,
    files: no_files,
    schema: engine_terrain::layer_schema!(hummocks_schema),
    points: engine_terrain::layer::no_points,
};

fn repose() -> f32 {
    libm::atanf(BANK_SLOPE).to_degrees()
}

fn steps_from_json(settings: &Value, _files: &dyn LayerFiles) -> Made {
    let steps: StepsLayer = parse_value(settings.clone())?;
    checked(steps.validate())?;
    Ok(Arc::new(steps))
}

fn read_steps(input: &mut Reader<'_>) -> Made {
    let steps: StepsLayer = input.get()?;
    checked(steps.validate())?;
    Ok(Arc::new(steps))
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

#[cfg(feature = "schema")]
fn steps_schema() -> schemars::Schema {
    schemars::schema_for!(StepsLayer)
}

#[cfg(feature = "schema")]
fn hummocks_schema() -> schemars::Schema {
    schemars::schema_for!(HummocksLayer)
}

fn checked(valid: Result<(), InvalidSetting>) -> Result<(), Error> {
    valid.map_err(|invalid| invalid.error(ErrorCode::InvalidTerrain, "$"))
}

fn water_below(place: Place<'_>) -> Result<(), Error> {
    if place.below().iter().any(|layer| layer.makes_water()) {
        Ok(())
    } else {
        Err(Error::new(ErrorCode::MissingLayer, "$").variants(["hydrology"]))
    }
}

impl TerrainLayer for StepsLayer {
    fn kind(&self) -> &'static LayerKind {
        &STEPS
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
        let placement = Placement::new(
            (self.slope, self.wetness),
            (self.patches, self.cover),
            self.seed ^ STEP_PATCHES,
        );
        let longest = self.height / placement.gentlest();
        if self.cover <= 0.0 || longest <= RESOLVED * spacing {
            return;
        }
        relieve(
            below,
            &placement,
            (row, xs, z, spacing),
            |x, z, height, slope| self.step(x, z, (height, slope), spacing),
        );
    }
}

impl StepsLayer {
    fn step(&self, x: f32, z: f32, (height, slope): (f32, f32), spacing: f32) -> f32 {
        if slope <= 0.0 {
            return 0.0;
        }
        let run = self.height / slope;
        let resolved = smooth(run / spacing - RESOLVED);
        if resolved <= 0.0 {
            return 0.0;
        }
        let riser = min(RESOLVED * spacing / run, 1.0);
        let bedding =
            height / self.height + 0.5 * blobs(x, z, self.patches, self.seed ^ STEP_BEDDING);
        let within = bedding - bedding.floor();
        let lifted = smooth((within - 0.5) / riser + 0.5);
        let ledge = smooth(1.0 + 2.0 * blobs(x, z, self.joints, self.seed ^ STEP_JOINTS));
        self.height * (lifted - within) * resolved * ledge
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
            (self.patches, self.cover),
            self.seed ^ HUMMOCK_PATCHES,
        );
        let half = 0.5 * self.height;
        relieve(below, &placement, (row, xs, z, spacing), |x, z, _, _| {
            half * blobs(x, z, self.across, self.seed ^ HUMMOCK_MOUNDS)
        });
    }
}

struct Placement {
    slope: [f32; 2],
    wetness: [f32; 2],
    patches: f32,
    cover: f32,
    seed: u32,
}

impl Placement {
    fn new(
        (slope, wetness): ([f32; 2], [f32; 2]),
        (patches, cover): (f32, f32),
        seed: u32,
    ) -> Self {
        Self {
            slope: slope.map(|degrees| libm::tanf(degrees.to_radians())),
            wetness,
            patches,
            cover,
            seed,
        }
    }

    fn gentlest(&self) -> f32 {
        if self.slope[0] <= self.slope[1] {
            self.slope[0]
        } else {
            0.0
        }
    }

    fn weight(&self, x: f32, z: f32, slope: f32, wetness: f32) -> f32 {
        let ramps = ramp(slope, self.slope) * ramp(wetness, self.wetness);
        if ramps <= 0.0 {
            return 0.0;
        }
        let patch = 0.5 + 0.5 * blobs(x, z, self.patches, self.seed);
        ramps * smooth((patch - (1.0 - self.cover)) / PATCH_EDGE)
    }
}

fn relieve(
    below: Below<'_>,
    placement: &Placement,
    (row, xs, z, spacing): (&mut [f32], &[f32], f32, f32),
    mut shape: impl FnMut(f32, f32, f32, f32) -> f32,
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
            let weight = placed * calm(hydrology, x, z, spacing);
            if weight > 0.0 {
                row[index] += weight * shape(x, z, row[index], slope);
            }
        },
    );
}

fn map_slope(flow: f32, wetness: f32, spacing: f32) -> f32 {
    libm::expf(flow * std::f32::consts::LN_10 - wetness) / spacing
}

fn calm(hydrology: &Hydrology, x: f32, z: f32, spacing: f32) -> f32 {
    let half = 0.5 * BANK_REACH;
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

fn blobs(x: f32, z: f32, across: f32, seed: u32) -> f32 {
    let lattice = 2.0 * across;
    let (u, v) = (x / lattice, z / lattice);
    gradient_value(0.8 * u - 0.6 * v + 0.37, 0.6 * u + 0.8 * v + 0.61, seed)
}

engine_terrain::codec::record!(StepsLayer {
    seed,
    height,
    joints,
    slope,
    wetness,
    patches,
    cover,
});
engine_terrain::codec::record!(HummocksLayer {
    seed,
    height,
    across,
    slope,
    wetness,
    patches,
    cover,
});
