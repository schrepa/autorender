use engine_math::min;
use engine_settings::settings;
use serde::{Deserialize, Serialize};

use crate::hydrology::{FLOW_RANGE, Hydrology, WETNESS_RANGE, WaterGrid, decode};
use crate::layers::Layers;
use crate::noise::{gradient_value, smooth};
use crate::rivers::{BANK_REACH, BANK_SLOPE};

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

fn repose() -> f32 {
    libm::atanf(BANK_SLOPE).to_degrees()
}

impl StepsLayer {
    pub(crate) fn apply_row(
        &self,
        layers: &Layers,
        (row, xs, z): (&mut [f32], &[f32], f32),
        spacing: f32,
    ) {
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
            layers,
            &placement,
            (row, xs, z, spacing),
            |x, z, height, slope| self.step(x, z, (height, slope), spacing),
        );
    }

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

impl HummocksLayer {
    pub(crate) fn apply_row(
        &self,
        layers: &Layers,
        (row, xs, z): (&mut [f32], &[f32], f32),
        spacing: f32,
    ) {
        if self.height <= 0.0 || self.cover <= 0.0 || 2.0 * self.across < RESOLVED * spacing {
            return;
        }
        let placement = Placement::new(
            (self.slope, self.wetness),
            (self.patches, self.cover),
            self.seed ^ HUMMOCK_PATCHES,
        );
        let half = 0.5 * self.height;
        relieve(layers, &placement, (row, xs, z, spacing), |x, z, _, _| {
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
    layers: &Layers,
    placement: &Placement,
    (row, xs, z, spacing): (&mut [f32], &[f32], f32, f32),
    mut shape: impl FnMut(f32, f32, f32, f32) -> f32,
) {
    let Some(hydrology) = layers.hydrology() else {
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
            let placed = placement.weight(x, z, slope, wetness) * layers.shown(x, z);
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

crate::codec::record!(StepsLayer {
    seed,
    height,
    joints,
    slope,
    wetness,
    patches,
    cover,
});
crate::codec::record!(HummocksLayer {
    seed,
    height,
    across,
    slope,
    wetness,
    patches,
    cover,
});
