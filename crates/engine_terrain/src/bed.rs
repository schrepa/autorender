use std::f32::consts::{PI, TAU};

use engine_math::{clamp, cos, max, min};
use glam::Vec2;
use serde::Serialize;

use crate::hydraulics::{GRAVITY, along_line};
use crate::rivers::RIVER_STEP;

pub(crate) const HEAD_FLOOD: f32 = 0.16;
pub const NARROWEST: f32 = 2.0;

const FORMS: [f32; 4] = [0.015, 0.03, 0.08, 0.3];
const BLEND: f32 = 0.2;
const WIDTH_LAW: [f32; 2] = [2.0, 0.54];
const DEPTH_LAW: [f32; 2] = [0.26, 0.34];
const WIDTH_DEPTH: f32 = 12.0;
const MEAN_DEPTH: f32 = 2.0 / 3.0;
const CRITICAL_SHIELDS: f32 = 0.15;
const SUBMERGED: f32 = 1.65;
const EXCESS: [f32; 5] = [1.5, 1.5, 2.0, 3.0, 3.0];
const D84_MEDIAN: f32 = 2.2;
const WATER_DENSITY: f32 = 1000.0;
const POOLS_APART: f32 = 6.0;
const POOL_DEPTH: f32 = 0.2;
const STEP_CLASTS: [f32; 5] = [0.0, 0.0, 0.5, 1.0, 1.0];
const STEP_HEIGHT: f32 = 0.8;
const STEEPNESS: [f32; 2] = [0.646, -0.419];
const STEEPEST: [f32; 2] = [1.0, 2.0];
const STEPS_APART: [f32; 2] = [0.4, 4.0];
const STEP_LEAST: f32 = 0.3;
const STEP_SAMPLES: f32 = 3.0;
const KNOB_RADII: [f32; 2] = [0.5, 1.0];
const KNOB_LEAST: f32 = 0.5;
const KNOB_HEIGHT: f32 = 0.8;
const KNOB_COVER: f32 = 0.3;
const CREST_KNOBS: [u32; 2] = [2, 4];
const KNOB_REACH: f32 = 0.5 * RIVER_STEP;
const STEP_SEED: u32 = 0x5b1d_c3a7;
const ROW_SEED: u32 = 0x3c6e_f372;
const SCATTER_SEED: u32 = 0xa54f_f53a;

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
    pub const ALL: [Self; 5] = [
        Self::Pool,
        Self::Plane,
        Self::Steps,
        Self::Cascade,
        Self::Falls,
    ];

    pub(crate) fn of(slope: f32) -> Self {
        Self::ALL[FORMS.iter().filter(|&&edge| slope >= edge).count()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bed {
    pub reach: Reach,
    pub d50: f32,
    pub d84: f32,
    pub power: f32,
    pub flood: f32,
    pub carved: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Crest {
    pub share: f32,
    pub drop: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Knob {
    pub segment: u32,
    pub middle: [f32; 2],
    pub radius: f32,
    pub height: f32,
}

impl Knob {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shares([f32; 5]);

impl Shares {
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

    pub(crate) fn share(&self, reach: Reach) -> f32 {
        self.0[reach as usize]
    }

    pub(crate) fn blend(&self, values: [f32; 5]) -> f32 {
        let mut blended = 0.0;
        for (share, value) in self.0.iter().zip(values) {
            blended += share * value;
        }
        blended
    }
}

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

pub(crate) fn grains(depth: f32, slope: f32, shares: &Shares) -> (f32, f32) {
    let critical = CRITICAL_SHIELDS * libm::powf(slope, 0.25);
    let median = MEAN_DEPTH * depth * slope / (SUBMERGED * critical * shares.blend(EXCESS));
    (median, D84_MEDIAN * median)
}

pub(crate) fn power(flood: f32, slope: f32, width: f32) -> f32 {
    WATER_DENSITY * GRAVITY * flood * slope / width
}

pub(crate) struct Bankfull {
    pub(crate) floods: Vec<f32>,
    pub(crate) slopes: Vec<f32>,
    pub(crate) shares: Vec<Shares>,
    pub(crate) widths: Vec<f32>,
    pub(crate) depths: Vec<f32>,
    pub(crate) banks: Vec<f32>,
    pub(crate) cut: Vec<bool>,
}

pub(crate) struct Forms {
    pub(crate) pools: Vec<f32>,
    pub(crate) dips: Vec<f32>,
    pub(crate) held: Vec<f32>,
    pub(crate) stepped: Vec<bool>,
    pub(crate) crests: Vec<Option<Crest>>,
    pub(crate) knobs: Vec<Knob>,
}

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

struct Steps {
    dips: Vec<f32>,
    held: Vec<f32>,
    stepped: Vec<bool>,
    crests: Vec<Option<Crest>>,
    rows: Vec<Row>,
}

#[derive(Clone, Copy)]
struct Row {
    segment: usize,
    share: f32,
    number: u32,
}

fn clasts(bankfull: &Bankfull, index: usize) -> f32 {
    MEAN_DEPTH * bankfull.depths[index] * bankfull.shares[index].blend(STEP_CLASTS)
}

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
    let mut crest = -unit_of(keyed(seed, u32::MAX)) * spacing[0];
    let (mut point, mut number) = (0, 0_u32);
    loop {
        let here = max(crest, 0.0);
        let mean = at(&spacing, here);
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
    let radii = |clasts: f32| KNOB_RADII.map(|share| clamp(share * clasts, KNOB_LEAST, KNOB_REACH));
    let (row_seed, scatter_seed) = (keyed(key, ROW_SEED), keyed(key, SCATTER_SEED));
    let mut knobs = Vec::new();
    let mut rows = rows.iter().peekable();
    for segment in 0..count - 1 {
        let (a, run) = (points[segment], points[segment + 1] - points[segment]);
        let length = along[segment + 1] - along[segment];
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

fn unit_of(key: u32) -> f32 {
    let mut h = key;
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    ((h ^ (h >> 16)) >> 8) as f32 / 16_777_216.0
}

fn keyed(key: u32, value: u32) -> u32 {
    let mut h = key ^ value.wrapping_mul(0x9e37_79b9);
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

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
