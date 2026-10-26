//! Relief by cause: layers a game adds to its terrain's stack to break the
//! ground where the causes that break real ground act. Rock steps terrace
//! steep ground along the bedding and the joints of the rock it bares;
//! earth hummocks (thúfur) heave wet, gentle ground into mounds. Each lies
//! where its ramps of the ground's slope and wetness index allow, as the
//! stack's hydrology bake maps them, in patches of noise, and keeps off the
//! water that bake found and the banks beside it, so its rivers' channels,
//! its lakes and the tiles' water stay as it found them; roads after them
//! level them.
//!
//! They move nothing until the stack's hydrology is baked, wherever they
//! stand in it, so its bake never reads them. Their shapes average to
//! nothing: a level whose samples lie too far apart to resolve one holds the
//! ground it stands on, as levels leave out noise octaves too fine for their
//! samples. Each sample is a pure function of its position, the height below
//! it and its level, in plain `f32` arithmetic, `libm` and the noise's own:
//! the same bits on every target.

use engine_math::min;
use engine_settings::settings;
use serde::{Deserialize, Serialize};

use crate::hydrology::{FLOW_RANGE, Hydrology, WETNESS_RANGE, WaterGrid, decode};
use crate::layers::Layers;
use crate::noise::{gradient_value, smooth};
use crate::rivers::{BANK_REACH, BANK_SLOPE};

/// Meters from one bedrock step's tread up to the next: bedding planes and
/// joints lie 1 to 3 meters apart where steep slopes bare rock, ISRM's
/// (1978) wide to very wide spacing. Steps take the widest, which meter
/// samples resolve on slopes up to 45 degrees.
const STEP_HEIGHT: f32 = 3.0;
/// Meters between the joints breaking a ledge: ISRM's (1978) very wide
/// spacing, 2 to 6 meters.
const JOINT_SPACING: f32 = 4.0;
/// Meters from the troughs between earth hummocks (thúfur) to the tallest
/// ones' tops, and meters across one: they stand 0.2 to 0.5 meters high and
/// 1 to 2 meters across (Schunke and Zoltai 1988; Grab 2005).
const HUMMOCK_HEIGHT: f32 = 0.5;
const HUMMOCK_ACROSS: f32 = 1.5;

/// The fewest samples a wavelength spans on a level that shows it: two, the
/// shortest samples resolve (Nyquist).
const RESOLVED: f32 = 2.0;
/// The share of their noise's range over which patches' edges soften.
const PATCH_EDGE: f32 = 0.15;

/// Salts of each noise the relief layers read, apart from one another and
/// from the noise layers' under the same seed.
const STEP_PATCHES: u32 = 0x3c6e_f372;
const STEP_BEDDING: u32 = 0xa54f_f53a;
const STEP_JOINTS: u32 = 0x510e_527f;
const HUMMOCK_PATCHES: u32 = 0x9b05_688c;
const HUMMOCK_MOUNDS: u32 = 0x1f83_d9ab;

settings! {
    /// Rock steps: ground too steep to hold soil bares its rock, which
    /// breaks along its bedding planes, lying level, into treads and down
    /// its joints into risers, the ledges broken where joints cross them. A
    /// step runs down the slope its height over the slope's tangent: steps
    /// fade out as that run nears two samples, the shortest the samples
    /// resolve, and each riser spans two of them.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct StepsLayer {
        /// Picks the pattern of its patches, bedding and joints.
        pub seed: u32 = 0 => any,
        /// Meters from one tread up to the next.
        pub height: f32 = STEP_HEIGHT => range(0.1, 100.0),
        /// Meters between the joints breaking its ledges.
        pub joints: f32 = JOINT_SPACING => range(0.5, 1000.0),
        /// Slopes (degrees) over which it fades in, none at the first and
        /// all from the second: from the steepest soil rests at, its angle
        /// of repose, to where the ground bares its rock.
        pub slope: [f32; 2] = [repose(), 40.0] => range(0.0, 89.0),
        /// Wetness indexes over which it fades in, none at the first and all
        /// from the second: falling, so it keeps to spurs and planar slopes
        /// and leaves the hollows water gathers in to the soil and debris
        /// filling them.
        pub wetness: [f32; 2] = [8.0, 5.0] => range(WETNESS_RANGE[0], WETNESS_RANGE[1]),
        /// Meters across the patches it lies in, each with its bedding
        /// shifted from the next one's.
        pub patches: f32 = 60.0 => range(1.0, 1.0e5),
        /// The share of the ground its ramps allow that its patches cover:
        /// 0 none, 1 all.
        pub cover: f32 = 0.6 => range(0.0, 1.0),
    }
}

settings! {
    /// Earth hummocks (thúfur): frost heaves wet, fine soil under turf into
    /// mounds with troughs between them, on gentle ground. They are the
    /// blobs of gradient noise whose lattice spans a hummock and a trough;
    /// levels whose samples lie further apart than a hummock is across, the
    /// shortest wavelength they resolve being two samples, hold the ground
    /// they stand on.
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct HummocksLayer {
        /// Picks the pattern of its patches and hummocks.
        pub seed: u32 = 0 => any,
        /// Meters from the troughs to the tallest hummocks' tops.
        pub height: f32 = HUMMOCK_HEIGHT => range(0.0, 10.0),
        /// Meters across a hummock.
        pub across: f32 = HUMMOCK_ACROSS => range(0.1, 100.0),
        /// Slopes (degrees) over which it fades in, none at the first and
        /// all from the second: falling, so it keeps to gentle ground, where
        /// frost heaves the soil up rather than drawing it downhill into
        /// lobes.
        pub slope: [f32; 2] = [15.0, 6.0] => range(0.0, 89.0),
        /// Wetness indexes over which it fades in, none at the first and all
        /// from the second: where water gathers in the soil frost heaves.
        pub wetness: [f32; 2] = [8.0, 11.0] => range(WETNESS_RANGE[0], WETNESS_RANGE[1]),
        /// Meters across the patches it lies in.
        pub patches: f32 = 80.0 => range(1.0, 1.0e5),
        /// The share of the ground its ramps allow that its patches cover:
        /// 0 none, 1 all.
        pub cover: f32 = 0.7 => range(0.0, 1.0),
    }
}

/// The steepest slope soil rests at, in degrees: its angle of repose, at
/// which river banks rise ([`BANK_SLOPE`]).
fn repose() -> f32 {
    libm::atanf(BANK_SLOPE).to_degrees()
}

impl StepsLayer {
    /// Adds the steps to `row`, heights at terrain-local x positions `xs`
    /// and z position `z` of `layers`' ground, samples `spacing` meters
    /// apart, where [`relieve`] lets them stand; nothing on levels too
    /// coarse to resolve a step on the gentlest ground they lie on.
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

    /// Meters the steps move ground `height` meters high at terrain-local
    /// `(x, z)` where it slopes `slope` (a tangent), its samples `spacing`
    /// meters apart: each tread's rise above the slope it cuts and each
    /// riser's fall below it, which average to nothing over a step.
    fn step(&self, x: f32, z: f32, (height, slope): (f32, f32), spacing: f32) -> f32 {
        if slope <= 0.0 {
            return 0.0;
        }
        // A step's run down the slope, and how much of it the samples
        // resolve: none at two samples, all from three.
        let run = self.height / slope;
        let resolved = smooth(run / spacing - RESOLVED);
        if resolved <= 0.0 {
            return 0.0;
        }
        // Each riser spans two samples of the run, the steepest face the
        // samples show without their grid printing into its edge; its tread
        // the rest.
        let riser = min(RESOLVED * spacing / run, 1.0);
        // Where the ground stands within its step, its patch's bedding
        // shifted by up to half a step from the next patch's.
        let bedding =
            height / self.height + 0.5 * blobs(x, z, self.patches, self.seed ^ STEP_BEDDING);
        let within = bedding - bedding.floor();
        let lifted = smooth((within - 0.5) / riser + 0.5);
        // Joints break the ledges where noise of their spacing dips lowest:
        // the gaps narrower than the ledges between them.
        let ledge = smooth(1.0 + 2.0 * blobs(x, z, self.joints, self.seed ^ STEP_JOINTS));
        self.height * (lifted - within) * resolved * ledge
    }
}

impl HummocksLayer {
    /// Adds the hummocks to `row`, heights at terrain-local x positions `xs`
    /// and z position `z` of `layers`' ground, samples `spacing` meters
    /// apart, where [`relieve`] lets them stand; nothing on levels whose
    /// samples lie further apart than a hummock is across, half its noise's
    /// lattice.
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

/// Where a relief layer lies: its slope ramp's ends as tangents, its
/// wetness ramp, and its patches' meters across, cover and seed.
struct Placement {
    slope: [f32; 2],
    wetness: [f32; 2],
    patches: f32,
    cover: f32,
    seed: u32,
}

impl Placement {
    /// The placement of a layer with slope ramp `slope` (degrees) and
    /// wetness ramp `wetness`, in patches `patches` meters across covering
    /// `cover` of the ground, picked by `seed`.
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

    /// The tangent of the gentlest slope the layer lies on: 0 where its
    /// slope ramp falls.
    fn gentlest(&self) -> f32 {
        if self.slope[0] <= self.slope[1] {
            self.slope[0]
        } else {
            0.0
        }
    }

    /// How much of the layer lies at terrain-local `(x, z)`, 0 to 1, where
    /// the ground slopes `slope` (a tangent) and its wetness index reads
    /// `wetness`: both ramps' share, in its patches.
    fn weight(&self, x: f32, z: f32, slope: f32, wetness: f32) -> f32 {
        let ramps = ramp(slope, self.slope) * ramp(wetness, self.wetness);
        if ramps <= 0.0 {
            return 0.0;
        }
        let patch = 0.5 + 0.5 * blobs(x, z, self.patches, self.seed);
        ramps * smooth((patch - (1.0 - self.cover)) / PATCH_EDGE)
    }
}

/// Adds a relief layer to `row`, heights at terrain-local x positions `xs`
/// and z position `z` of `layers`' ground, samples `spacing` meters apart:
/// at each point the meters `shape` moves the ground, given the point, the
/// height below it and the ground's slope there (a tangent), times how much
/// of the layer `placement` lays there, the water lets stand ([`calm`]) and
/// the stack shows. Slopes and wetness indexes come from the stack's
/// hydrology bake's maps, read a row at a time through their uniform cubic
/// B-splines as its masks read them; without that bake, nothing.
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

/// The steepest slope down (a tangent) where the hydrology bake's maps,
/// their samples `spacing` meters apart, read the base-10 logarithm of
/// upstream area `flow` (square meters) and the wetness index `wetness`:
/// the index ln(a / tan b) undone, a the area draining through a meter of
/// contour, the upstream area over the maps' spacing.
fn map_slope(flow: f32, wetness: f32, spacing: f32) -> f32 {
    libm::expf(flow * std::f32::consts::LN_10 - wetness) / spacing
}

/// How freely relief moves the ground at terrain-local `(x, z)` beside the
/// water `hydrology` found, as samples `spacing` meters apart see it, 0 to
/// 1: not at all on a river's channel and the inner half of its banks'
/// reach, fading in over the outer half, and only as far as a lake's water
/// and its reach past the shore leave the grid's samples around the point.
fn calm(hydrology: &Hydrology, x: f32, z: f32, spacing: f32) -> f32 {
    let half = 0.5 * BANK_REACH;
    let banks = smooth((hydrology.rivers.edge(x, z, spacing) - half) / half);
    if banks <= 0.0 {
        return 0.0;
    }
    banks * (1.0 - lake_share(&hydrology.water, x, z))
}

/// The share of the four samples of `water`'s grid around terrain-local
/// `(x, z)` that a lake's water or its reach past the shore holds, each by
/// its nearness to the point.
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

/// 0 at `from`, 1 from `to`, smooth between: falling where `from` is the
/// larger, a step where they meet.
fn ramp(value: f32, [from, to]: [f32; 2]) -> f32 {
    if from == to {
        return if value >= to { 1.0 } else { 0.0 };
    }
    smooth((value - from) / (to - from))
}

/// Gradient noise, about -1 to 1, at terrain-local `(x, z)`, whose blobs
/// of one sign are about `across` meters across: over a lattice twice that,
/// which spans a blob and the one of the other sign beside it, turned by
/// atan(3/4) and shifted off the terrain's axes and origin so it never
/// lines up with the samples.
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
