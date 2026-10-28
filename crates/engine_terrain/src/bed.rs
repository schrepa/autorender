//! Stream beds: the form a reach's slope gives a river's bed, and the
//! grains and power of the flood that fills its channel. A reach's slope
//! sets its form (Montgomery and Buffington 1997): pools and riffles where
//! it runs gently, a plane bed steeper, steps and pools steeper still, a
//! cascade of boulders steeper again and falls steepest, each form's
//! quantities blending into the next's across a fifth of the slope either
//! side of where they meet. Its bankfull flood sizes its channel as
//! channels widen and deepen downstream (Leopold and Maddock 1953), no
//! wider for its depth than banks that hold keep channels (Rosgen 1994).
//! The flood's mean depth and slope set its grains, the median as large as
//! the flood moves no more than its form's share past the stress that
//! starts grains moving, a stress rising with slope (Lamb, Dietrich and
//! Venditti 2008), and its power. Steps of the clasts floods leave stand in
//! steps' and cascades' beds, each a drop at its crest and an adverse tread
//! up to the next, as steep as their slope keeps them (Canovaro and Solari
//! 2007) and spaced between regular and random (Erikson and Turowski 2026);
//! boulders raise rounded knobs on the bed, scattered over cascades and in
//! rows along steps' crests. Steps and knobs the terrain's finest samples
//! cannot hold are left to friction.
//!
//! Every step uses only IEEE arithmetic, `libm`'s functions and the
//! engine's own cosine, so the same bake gives the same bits natively and
//! in the browser.

use std::f32::consts::{PI, TAU};

use engine_math::{clamp, cos, max, min};
use glam::Vec2;
use serde::Serialize;

use crate::hydraulics::{GRAVITY, along_line};
use crate::rivers::RIVER_STEP;

/// The bankfull flood (cubic meters a second) at which channels begin:
/// what a 0.05 square kilometer hollow brings in flood at about 3 cubic
/// meters a second each square kilometer, where Montgomery and
/// Foufoula-Georgiou (1993) find channels beginning on hillsides of about
/// 11 degrees (drained area times the slope's tangent squared near 2000
/// square meters).
pub(crate) const HEAD_FLOOD: f32 = 0.16;
/// The fewest of the grid's samples a channel spans from bank to bank:
/// narrower channels are cut that wide, as deep as they carry their flood,
/// so the water settling over the grid stays in them.
pub const NARROWEST: f32 = 2.0;

/// The slopes (meters a meter) at which a reach's bed changes form
/// (Montgomery and Buffington 1997): gentler than the first, pools and
/// riffles; to the second, a plane bed; to the third, steps and pools; to
/// the fourth, a cascade; steeper, falls.
const FORMS: [f32; 4] = [0.015, 0.03, 0.08, 0.3];
/// The share of the slope at which two forms meet, either side of it, over
/// which the one's quantities blend into the other's.
const BLEND: f32 = 0.2;
/// How channels full to their banks widen and deepen downstream with the
/// discharge Q they carry (cubic meters a second; downstream hydraulic
/// geometry, Leopold and Maddock 1953): 2.0 Q^0.54 meters across at their
/// banks, and 0.26 Q^0.34 meters deep on the mean, their section's area
/// over its width, as hydraulic geometry measures depth.
const WIDTH_LAW: [f32; 2] = [2.0, 0.54];
const DEPTH_LAW: [f32; 2] = [0.26, 0.34];
/// The most times its mean depth a channel stands wide where its banks
/// hold: soil bound by roots in meadows, and boulders and rock down steep
/// mountain reaches, keep channels under 12 (Rosgen 1994's E and A
/// streams).
const WIDTH_DEPTH: f32 = 12.0;
/// A channel's mean depth against its depth at its middle: two thirds, as
/// its section is a parabola.
const MEAN_DEPTH: f32 = 2.0 / 3.0;
/// The critical Shields stress of grains on a bed falling a meter a meter,
/// gentler beds' the less as the fourth root of their slope (Lamb, Dietrich
/// and Venditti 2008), and grains' submerged specific gravity (quartz's).
const CRITICAL_SHIELDS: f32 = 0.15;
const SUBMERGED: f32 = 1.65;
/// How many times its critical Shields stress a bankfull flood's stress on
/// a bed's median grain stands, Pool to Falls: barely past it on gentle
/// beds, well past it on steep ones, whose steps and boulders take much of
/// the flood's stress; falls keep a cascade's.
const EXCESS: [f32; 5] = [1.5, 1.5, 2.0, 3.0, 3.0];
/// How many times its median grain the grain 84% of a bed's grains are
/// finer than is.
const D84_MEDIAN: f32 = 2.2;
/// The density of water (kilograms a cubic meter).
const WATER_DENSITY: f32 = 1000.0;
/// Channel widths from one pool to the next (pools and riffles keep five to
/// seven apart), and the share of the channel's depth pools stand deeper
/// and riffles shallower on a flat (half as deep again in a pool as over a
/// riffle), the less the steeper the bed, to none at a plane bed.
const POOLS_APART: f32 = 6.0;
const POOL_DEPTH: f32 = 0.2;
/// The size of the clasts steps form from against their bankfull flood's
/// mean depth, Pool to Falls: none on gentle beds, which form no steps,
/// half as large as the flood is deep on steps and pools and as large on
/// cascades; falls keep a cascade's.
const STEP_CLASTS: [f32; 5] = [0.0, 0.0, 0.5, 1.0, 1.0];
/// A step's height against the clasts it forms from.
const STEP_HEIGHT: f32 = 0.8;
/// How steep steps stand against their reach's slope (their height over
/// the length from one crest to the next, over the slope): 0.646 S^-0.419
/// (Canovaro and Solari 2007), from as steep as the slope to twice as
/// steep.
const STEEPNESS: [f32; 2] = [0.646, -0.419];
const STEEPEST: [f32; 2] = [1.0, 2.0];
/// The least and most mean spacing from one step's crest to the next, in
/// channel widths.
const STEPS_APART: [f32; 2] = [0.4, 4.0];
/// The share of their mean spacing steps stand apart at least, an
/// exponential draw adding the rest on the mean, so they stand between
/// regular and random, never periodic (Erikson and Turowski 2026).
const STEP_LEAST: f32 = 0.3;
/// How many of the terrain's finest sample spacings steps must stand apart
/// on the mean to be carved: closer steps are left to friction.
const STEP_SAMPLES: f32 = 3.0;
/// Knobs: their radius against the clasts steps form from, at least
/// KNOB_LEAST meters; their height against their radius; the share of a
/// cascade's bed they cover; and how many stand along a step's crest, the
/// fewest and the most.
const KNOB_RADII: [f32; 2] = [0.5, 1.0];
const KNOB_LEAST: f32 = 0.5;
const KNOB_HEIGHT: f32 = 0.8;
const KNOB_COVER: f32 = 0.3;
const CREST_KNOBS: [u32; 2] = [2, 4];
/// The widest a knob's radius grows (meters): half the meters between a
/// river's points, so a knob reaches no segment past those beside its own,
/// which samples read it from with their own.
const KNOB_REACH: f32 = 0.5 * RIVER_STEP;
/// Seeds of the hashes drawing a river's steps, the knobs along their
/// crests and the knobs scattered over its cascades; each river draws them
/// under a key of its own.
const STEP_SEED: u32 = 0x5b1d_c3a7;
const ROW_SEED: u32 = 0x3c6e_f372;
const SCATTER_SEED: u32 = 0xa54f_f53a;

/// The kind of reach a stream runs in, by its slope (Montgomery and Buffington 1997).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[repr(u8)]
pub enum Reach {
    Pool = 0,
    Plane = 1,
    Steps = 2,
    Cascade = 3,
    Falls = 4,
}

impl Reach {
    /// Every reach, by its number.
    pub const ALL: [Self; 5] = [
        Self::Pool,
        Self::Plane,
        Self::Steps,
        Self::Cascade,
        Self::Falls,
    ];

    /// The reach a stream falling `slope` (meters a meter) runs in: past
    /// each of [`FORMS`]' slopes, the next.
    pub(crate) fn of(slope: f32) -> Self {
        Self::ALL[FORMS.iter().filter(|&&edge| slope >= edge).count()]
    }
}

/// A stream's bed where it runs past a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bed {
    pub reach: Reach,
    /// Median grain size, meters.
    pub d50: f32,
    /// The size 84% of the bed's grains are finer than, meters.
    pub d84: f32,
    /// Stream power per bed area at the bankfull flood, watts per square meter.
    pub power: f32,
    /// The bankfull flood's water level, meters (as heights are).
    pub flood: f32,
    /// Whether steps or knobs are carved into the ground here (friction then counts grains alone).
    pub carved: bool,
}

/// A step's crest on a river's segment: the share of the way along it the
/// crest stands (0 to 1) and how far the bed drops there (meters).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Crest {
    pub share: f32,
    pub drop: f32,
}

/// A rounded knob a boulder raises on a river's bed: the segment of its
/// river it stands on, its middle (terrain-local x and z), its radius and
/// how high it rises at its middle (meters).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Knob {
    pub segment: u32,
    pub middle: [f32; 2],
    pub radius: f32,
    pub height: f32,
}

impl Knob {
    /// How high it raises the bed at terrain-local `point` (meters): a
    /// dome's height there.
    #[inline]
    pub(crate) fn rise(&self, point: Vec2) -> f32 {
        let offset = point - Vec2::from(self.middle);
        let reach = offset.dot(offset) / (self.radius * self.radius);
        if reach < 1.0 {
            self.height * (1.0 - reach).sqrt()
        } else {
            0.0
        }
    }
}

/// How much a reach is of each form, Pool to Falls: shares summing to one,
/// each form whole well within its slopes and blending into the next across
/// [`BLEND`] of the slope either side of where they meet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shares([f32; 5]);

impl Shares {
    /// The shares of a reach falling `slope` (meters a meter).
    pub(crate) fn of(slope: f32) -> Self {
        let past = FORMS.map(|edge| {
            clamp(
                (slope - (1.0 - BLEND) * edge) / (2.0 * BLEND * edge),
                0.0,
                1.0,
            )
        });
        Self([
            1.0 - past[0],
            past[0] - past[1],
            past[1] - past[2],
            past[2] - past[3],
            past[3],
        ])
    }

    /// The share of `reach`.
    pub(crate) fn share(&self, reach: Reach) -> f32 {
        self.0[reach as usize]
    }

    /// `values`, one for each form (Pool to Falls), as the shares blend
    /// them.
    pub(crate) fn blend(&self, values: [f32; 5]) -> f32 {
        let mut blended = 0.0;
        for (share, value) in self.0.iter().zip(values) {
            blended += share * value;
        }
        blended
    }
}

/// The width across its banks and the depth at its middle (meters) of a
/// channel carrying `flood` full to its banks (cubic meters a second): as
/// channels widen and deepen downstream ([`WIDTH_LAW`], [`DEPTH_LAW`]),
/// narrowed at the same section to [`WIDTH_DEPTH`] times its mean depth
/// where the laws stand it wider, its middle half as deep again as its
/// mean ([`MEAN_DEPTH`]).
pub(crate) fn bankfull(flood: f32) -> (f32, f32) {
    let width = WIDTH_LAW[0] * libm::powf(flood, WIDTH_LAW[1]);
    let depth = DEPTH_LAW[0] * libm::powf(flood, DEPTH_LAW[1]);
    let ratio = min(width / depth, WIDTH_DEPTH);
    let section = width * depth;
    (
        (section * ratio).sqrt(),
        (section / ratio).sqrt() / MEAN_DEPTH,
    )
}

/// The median grain and the grain 84% of grains are finer than (meters) of
/// a bed under a bankfull flood `depth` meters deep at its channel's middle
/// falling `slope`, in a reach of `shares`: the flood's Shields stress on
/// the median grain (its mean depth, [`MEAN_DEPTH`] of its middle's, times
/// its slope over the grain's submerged specific gravity and size) stands
/// its form's [`EXCESS`] over the critical stress, which rises with slope
/// ([`CRITICAL_SHIELDS`]).
pub(crate) fn grains(depth: f32, slope: f32, shares: &Shares) -> (f32, f32) {
    let critical = CRITICAL_SHIELDS * libm::powf(slope, 0.25);
    let median = MEAN_DEPTH * depth * slope / (SUBMERGED * critical * shares.blend(EXCESS));
    (median, D84_MEDIAN * median)
}

/// The power a bankfull flood carrying `flood` cubic meters a second down
/// `slope` spends on each square meter of a bed `width` meters across
/// (watts): ρ g Q S / W.
pub(crate) fn power(flood: f32, slope: f32, width: f32) -> f32 {
    WATER_DENSITY * GRAVITY * flood * slope / width
}

/// A river's channel along its line as its floods size it: at each point
/// the discharge of its flood (cubic meters a second), its reach's slope
/// (meters a meter) and shares of each form, its channel's width and depth
/// at its middle (meters), the level of its banks (terrain-local meters),
/// and whether it is cut into the ground there: not where the heights
/// already hold it.
pub(crate) struct Bankfull {
    pub(crate) floods: Vec<f32>,
    pub(crate) slopes: Vec<f32>,
    pub(crate) shares: Vec<Shares>,
    pub(crate) widths: Vec<f32>,
    pub(crate) depths: Vec<f32>,
    pub(crate) banks: Vec<f32>,
    pub(crate) cut: Vec<bool>,
}

/// The forms a river's bed takes along its line: at each point how much
/// deeper than its bankfull depth its bed lies in the pools and riffles of
/// a gentle reach (`pools`, meters, negative over riffles) and on the tread
/// behind a step's crest (`dips`, meters), the meters along its line to the
/// crest whose pool its water stands in (`held`, NaN where none holds it),
/// and whether steps are carved there; at each point the crest standing on
/// the segment to the next, if one does; and the knobs boulders raise on
/// the bed, segment by segment.
pub(crate) struct Forms {
    pub(crate) pools: Vec<f32>,
    pub(crate) dips: Vec<f32>,
    pub(crate) held: Vec<f32>,
    pub(crate) stepped: Vec<bool>,
    pub(crate) crests: Vec<Option<Crest>>,
    pub(crate) knobs: Vec<Knob>,
}

/// The forms the bed of a river at `points`, `along` meters along its line
/// from the first, takes in the channel `bankfull` sizes, where the
/// terrain's finest samples lie `finest` meters apart: pools and riffles on
/// gentle reaches, POOLS_APART widths apart, deepest and shallowest by
/// POOL_DEPTH of the channel's depth on a flat and by none from `FORMS[0]` on;
/// steps on steps' and cascades' reaches ([`steps`]); and knobs where
/// boulders stand ([`knobs`]). Forms stand by distance along the river from
/// phases its first point's place gives, so the same river forms alike in
/// every bake.
pub(crate) fn forms((points, along): (&[Vec2], &[f32]), bankfull: &Bankfull, finest: f32) -> Forms {
    let key = place_key(points[0]);
    let count = points.len();
    let mut turn = unit_of(key);
    let mut pools = Vec::with_capacity(count);
    for index in 0..count {
        if index > 0 {
            turn += (along[index] - along[index - 1]) / (POOLS_APART * bankfull.widths[index - 1]);
        }
        let strength = POOL_DEPTH * max(1.0 - bankfull.slopes[index] / FORMS[0], 0.0);
        pools.push(bankfull.depths[index] * strength * cos(TAU * (turn - turn.floor())));
    }
    let steps = steps(along, bankfull, (key, finest));
    let knobs = knobs((points, along), bankfull, &steps.rows, (key, finest));
    Forms {
        pools,
        dips: steps.dips,
        held: steps.held,
        stepped: steps.stepped,
        crests: steps.crests,
        knobs,
    }
}

/// A river's steps as [`steps`] lays them: [`Forms`]' `dips`, `held`,
/// `stepped` and `crests`, and the carved crests knobs stand along.
struct Steps {
    dips: Vec<f32>,
    held: Vec<f32>,
    stepped: Vec<bool>,
    crests: Vec<Option<Crest>>,
    rows: Vec<Row>,
}

/// A carved step's crest, which a row of knobs stands along: the segment
/// holding it, the share of the way along that segment it stands, and the
/// step's number along its river.
#[derive(Clone, Copy)]
struct Row {
    segment: usize,
    share: f32,
    number: u32,
}

/// The size of the clasts steps form from (meters) at a river's point
/// `index` in the channel `bankfull` sizes: [`STEP_CLASTS`] of its flood's
/// mean depth, as its reach's shares blend them.
fn clasts(bankfull: &Bankfull, index: usize) -> f32 {
    MEAN_DEPTH * bankfull.depths[index] * bankfull.shares[index].blend(STEP_CLASTS)
}

/// The steps of a river whose points stand `along` meters along its line
/// from the first, in the channel `bankfull` sizes, the river `key` names,
/// where the terrain's finest samples lie `finest` meters apart. Steps
/// stand where its reach forms them, on steps' and cascades' reaches: each
/// a crest, where the bed drops c S L (the steps' steepness c, the reach's
/// slope S and the length L to the next crest), then a tread rising
/// (c - 1) S L back up to the next crest, so the bed falls the reach's
/// slope on the mean and a pool stands behind each crest, H (1 - 1/c) deep
/// at its deepest for a step H high. Their mean spacing is the length such a
/// step takes at that steepness, H / (c S) for steps STEP_HEIGHT as high as
/// the clasts they form from, within STEPS_APART widths; each step spans
/// STEP_LEAST of that and an exponential draw of the rest, and at least
/// `STEPS_APART[0]` widths. Steps whose mean spacing spans STEP_SAMPLES of the
/// finest samples or more are carved, closer ones left to friction; where
/// the reach only partly forms steps, they drop as much less.
fn steps(along: &[f32], bankfull: &Bankfull, (key, finest): (u32, f32)) -> Steps {
    let (slopes, shares, widths) = (
        &bankfull.slopes[..],
        &bankfull.shares[..],
        &bankfull.widths[..],
    );
    let count = along.len();
    let end = along[count - 1];
    let forming: Vec<f32> = shares
        .iter()
        .map(|shares| shares.share(Reach::Steps) + shares.share(Reach::Cascade))
        .collect();
    let steepness: Vec<f32> = slopes
        .iter()
        .map(|&slope| {
            clamp(
                STEEPNESS[0] * libm::powf(slope, STEEPNESS[1]),
                STEEPEST[0],
                STEEPEST[1],
            )
        })
        .collect();
    let spacing: Vec<f32> = (0..count)
        .map(|index| {
            let height = STEP_HEIGHT * clasts(bankfull, index);
            clamp(
                height / (steepness[index] * slopes[index]),
                STEPS_APART[0] * widths[index],
                STEPS_APART[1] * widths[index],
            )
        })
        .collect();
    let at = |values: &[f32], place: f32| along_line(values, along, place);
    let seed = keyed(key, STEP_SEED);
    let mut steps = Steps {
        dips: vec![0.0; count],
        held: vec![f32::NAN; count],
        stepped: vec![false; count],
        crests: vec![None; count],
        rows: Vec::new(),
    };
    // The first crest stands a draw's share of a mean spacing above the
    // river's first point, so the river begins partway along a tread.
    let mut crest = -unit_of(keyed(seed, u32::MAX)) * spacing[0];
    let (mut point, mut number) = (0, 0_u32);
    loop {
        let here = max(crest, 0.0);
        let mean = at(&spacing, here);
        // An exponential draw of mean one.
        let draw = -libm::logf(1.0 - unit_of(keyed(seed, number)));
        let length = max(
            mean * (STEP_LEAST + (1.0 - STEP_LEAST) * draw),
            STEPS_APART[0] * at(widths, here),
        );
        let next = crest + length;
        let first = point;
        while point < count && along[point] < next {
            point += 1;
        }
        if mean >= STEP_SAMPLES * finest && at(&forming, here) > 0.0 {
            // How far the bed drops where the reach forms steps whole.
            let fall = at(&steepness, here) * at(slopes, here) * length;
            for index in first..point {
                steps.dips[index] = forming[index] * fall * (next - along[index]) / length;
                steps.held[index] = next;
                steps.stepped[index] = forming[index] > 0.0;
            }
            if (0.0..end).contains(&crest) {
                let segment = along
                    .partition_point(|&place| place <= crest)
                    .clamp(1, count - 1)
                    - 1;
                let run = along[segment + 1] - along[segment];
                let share = if run > 0.0 {
                    (crest - along[segment]) / run
                } else {
                    0.0
                };
                // Crests closer together than the river's points drop
                // together at the first.
                steps.crests[segment]
                    .get_or_insert(Crest { share, drop: 0.0 })
                    .drop += at(&forming, crest) * fall;
                steps.rows.push(Row {
                    segment,
                    share,
                    number,
                });
            }
        }
        if next >= end || !next.is_finite() {
            break;
        }
        crest = next;
        number += 1;
    }
    steps
}

/// The knobs boulders raise on the bed of a river at `points`, `along`
/// meters along its line from the first, in the channel `bankfull` sizes,
/// the river `key` names, where the terrain's finest samples lie `finest`
/// meters apart, segment by segment: a row of CREST_KNOBS along each carved
/// step's crest (`rows`) on steps' reaches, each knob somewhere in its own
/// share of the channel's width; and knobs scattered at random over
/// cascades' beds, as many as cover KNOB_COVER of the bed. Each is
/// KNOB_RADII of the clasts steps form from in radius, within KNOB_LEAST and
/// KNOB_REACH, and stands KNOB_HEIGHT of its radius high where the reach
/// forms its kind of knob whole, the less where it partly does; knobs
/// narrower than the finest samples lie apart are left to friction.
fn knobs(
    (points, along): (&[Vec2], &[f32]),
    bankfull: &Bankfull,
    rows: &[Row],
    (key, finest): (u32, f32),
) -> Vec<Knob> {
    let (shares, widths) = (&bankfull.shares[..], &bankfull.widths[..]);
    let count = points.len();
    let clasts: Vec<f32> = (0..count).map(|index| clasts(bankfull, index)).collect();
    let stepping: Vec<f32> = shares
        .iter()
        .map(|shares| shares.share(Reach::Steps))
        .collect();
    // The least and most radius of knobs among clasts that large.
    let radii = |clasts: f32| KNOB_RADII.map(|share| clamp(share * clasts, KNOB_LEAST, KNOB_REACH));
    let (row_seed, scatter_seed) = (keyed(key, ROW_SEED), keyed(key, SCATTER_SEED));
    let mut knobs = Vec::new();
    let mut rows = rows.iter().peekable();
    for segment in 0..count - 1 {
        let (a, run) = (points[segment], points[segment + 1] - points[segment]);
        let length = along[segment + 1] - along[segment];
        // To the river's right, looking downstream, as samples read it.
        let across = run.normalize_or(Vec2::X).perp();
        let lerp = |values: &[f32], share: f32| {
            values[segment] + (values[segment + 1] - values[segment]) * share
        };
        let mut raise = |share: f32, offset: f32, radius: f32, height: f32| {
            if 2.0 * radius >= finest {
                knobs.push(Knob {
                    segment: segment as u32,
                    middle: (a + run * share + across * offset).to_array(),
                    radius,
                    height,
                });
            }
        };
        while let Some(row) = rows.next_if(|row| row.segment == segment) {
            let steps = lerp(&stepping, row.share);
            if steps <= 0.0 {
                continue;
            }
            let width = lerp(widths, row.share);
            let [low, high] = radii(lerp(&clasts, row.share));
            let seed = keyed(row_seed, row.number);
            let spread = CREST_KNOBS[1] - CREST_KNOBS[0] + 1;
            let many = CREST_KNOBS[0] + (unit_of(keyed(seed, 0)) * spread as f32) as u32;
            for knob in 0..many {
                let radius = low + (high - low) * unit_of(keyed(seed, 2 * knob + 1));
                let slot = (knob as f32 + unit_of(keyed(seed, 2 * knob + 2))) / many as f32;
                raise(
                    row.share,
                    (slot - 0.5) * width,
                    radius,
                    KNOB_HEIGHT * radius * steps,
                );
            }
        }
        let cascade = 0.5
            * (shares[segment].share(Reach::Cascade) + shares[segment + 1].share(Reach::Cascade));
        if cascade <= 0.0 {
            continue;
        }
        let width = 0.5 * (widths[segment] + widths[segment + 1]);
        let [low, high] = radii(0.5 * (clasts[segment] + clasts[segment + 1]));
        // Knobs falling at random over a bed cover all of it but e to the
        // minus their areas over its own: as many as leave KNOB_COVER
        // covered on the mean.
        let area = PI * (low * low + low * high + high * high) / 3.0;
        let expected = -libm::logf(1.0 - KNOB_COVER) * length * width / area;
        let seed = keyed(scatter_seed, segment as u32);
        let many = (expected + unit_of(keyed(seed, 0))) as u32;
        for knob in 0..many {
            let radius = low + (high - low) * unit_of(keyed(seed, 3 * knob + 1));
            raise(
                unit_of(keyed(seed, 3 * knob + 2)),
                (unit_of(keyed(seed, 3 * knob + 3)) - 0.5) * width,
                radius,
                KNOB_HEIGHT * radius * cascade,
            );
        }
    }
    knobs
}

/// A number in [0, 1) hashed from `key`.
fn unit_of(key: u32) -> f32 {
    let mut h = key;
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    ((h ^ (h >> 16)) >> 8) as f32 / 16_777_216.0
}

/// `key` hashed with `value`: a key of its own for each value.
fn keyed(key: u32, value: u32) -> u32 {
    let mut h = key ^ value.wrapping_mul(0x9e37_79b9);
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

/// The key of a river whose first point is `point`: its place, so the same
/// river forms alike in every bake.
fn place_key(point: Vec2) -> u32 {
    point.x.to_bits().wrapping_mul(0x8da6_b343) ^ point.y.to_bits().wrapping_mul(0xd816_3841)
}

crate::codec::tagged!(Reach, "invalid reach" {
    0 => Pool,
    1 => Plane,
    2 => Steps,
    3 => Cascade,
    4 => Falls,
});
crate::codec::record!(Crest { share, drop });
crate::codec::record!(Knob {
    segment,
    middle,
    radius,
    height,
});
