use std::f32::consts::{PI, TAU};

use engine_math::{GRAVITY, clamp, cos, max, min};
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::hydraulics::{along_line, resistance};
use crate::noise::smooth;
use crate::rivers::{Form, RIVER_STEP};

pub const NARROWEST: f32 = 2.0;

const FORMS: [f32; 4] = [0.015, 0.03, 0.08, 0.3];
const WIDTH_LAW: [f32; 2] = [2.0, 0.54];
const DEPTH_LAW: [f32; 2] = [0.26, 0.34];
const MEAN_DEPTH: f32 = 2.0 / 3.0;
const CRITICAL_SHIELDS: f32 = 0.15;
const SUBMERGED: f32 = 1.65;
const EXCESS: [f32; 5] = [1.5, 1.5, 2.0, 3.0, 3.0];
const D84_MEDIAN: f32 = 2.2;
const WATER_DENSITY: f32 = 1000.0;
const STEEPNESS: [f32; 2] = [0.646, -0.419];
const STEEPEST: [f32; 2] = [1.0, 2.0];
const STEP_SAMPLES: f32 = 3.0;
const RISER_SAMPLES: f32 = 0.5;
const KNOB_SAMPLES: f32 = 1.0;
const POOL_PEAK: f32 = 6.75;
const NOTCHES: usize = 3;
const KNOB_REACH: f32 = 0.5 * RIVER_STEP;
const STEP_SEED: u32 = 0x5b1d_c3a7;
const CREST_SEED: u32 = 0x1f83_d9ab;
const ROW_SEED: u32 = 0x3c6e_f372;
const SCATTER_SEED: u32 = 0xa54f_f53a;
const LEDGE_SEED: u32 = 0x7b3f_91c5;

engine_codec::fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Pools {
        pub apart: f32 = 6.0 => above_to(0.0, 100.0),
        pub depth: f32 = 0.3 => range(0.0, 1.0),
        pub bend: f32 = 0.6 => above_to(0.0, 1.0),
        pub ends: f32 = 2.0 => above_to(0.0, 100.0),
        pub crest: f32 = 0.5 => range(0.0, 1.0),
    }
}

engine_codec::fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Steps {
        pub clasts: [f32; 5] = [0.0, 0.0, 0.5, 1.0, 1.0] => range(0.0, 4.0),
        pub height: f32 = 0.8 => range(0.0, 4.0),
        pub apart: [f32; 2] = [1.0, 4.0] => above_to(0.0, 100.0),
        pub spread: [f32; 2] = [0.7, 1.4] => above_to(0.0, 10.0),
        pub riser_slope: f32 = 1.5 => above_to(0.0, 100.0),
        pub tread: f32 = 0.25 => range(0.0, 0.95),
        pub pool_aspect: f32 = 0.15 => range(0.0, 1.0),
        pub pool_deepest: f32 = 0.5 => range(0.0, 1.0),
        pub skew: f32 = 0.4 => range(0.0, 2.0),
        pub arch: [f32; 2] = [-0.15, 0.4] => range(-2.0, 2.0),
        pub notch_room: f32 = 5.0 => above_to(0.0, 100.0),
        pub notch_half: f32 = 1.2 => above_to(0.0, 100.0),
        pub notch_bite: [f32; 2] = [1.6, 3.2] => range(0.0, 100.0),
    }
}

engine_codec::fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Boulders {
        pub back: f32 = 0.5 => range(0.0, 4.0),
        pub span: f32 = 1.0 / 6.0 => range(0.0, 1.0),
        pub height: f32 = 0.8 => range(0.0, 4.0),
        pub rise: [f32; 2] = [0.5, 0.9] => range(0.0, 4.0),
        pub cover: f32 = 0.3 => range(0.0, 0.95),
    }
}

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

    pub fn of(slope: f32) -> Self {
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
pub struct Crest {
    pub segment: u32,
    pub place: f32,
    pub drop: f32,
    pub run: f32,
    pub tread: f32,
    pub pool: f32,
    pub skew: f32,
    pub arch: f32,
    pub bite: f32,
    pub set: f32,
    pub notch: f32,
    pub notches: [f32; NOTCHES],
}

impl Crest {
    #[inline]
    fn notched(&self, across: f32) -> f32 {
        self.notches.iter().fold(0.0, |deepest, &middle| {
            let off = (across - middle) / self.notch;
            let open = max(1.0 - off * off, 0.0);
            max(deepest, open * open)
        })
    }

    #[inline]
    pub fn lip(&self, across: f32) -> f32 {
        let bowed = self.skew * across + self.arch * across * across;
        self.place + self.set + bowed - self.bite * self.notched(across)
    }

    #[inline]
    pub fn fallen(&self, past: f32) -> f32 {
        let share = clamp(past / self.run, 0.0, 1.0);
        share * share * (3.0 - 2.0 * share)
    }

    #[inline]
    pub fn scour(&self, past: f32, lip: f32) -> f32 {
        let room = self.place + self.tread - lip;
        let share = (past - self.run) / room;
        if room > 0.0 && share > 0.0 && share < 1.0 {
            let deepest = self.pool * min(room / self.tread, 1.0);
            deepest * POOL_PEAK * share * (1.0 - share) * (1.0 - share)
        } else {
            0.0
        }
    }

    fn wander(&self) -> [f32; 2] {
        let skew = self.skew.abs();
        [
            skew + max(-self.arch, 0.0) + self.bite,
            skew + max(self.arch, 0.0) + self.bite,
        ]
    }

    pub fn reach(&self) -> [f32; 2] {
        let [above, below] = self.wander();
        [above, self.run + max(below, self.tread)]
    }

    fn fitted(mut self, end: f32) -> Self {
        let [above, below] = self.wander();
        let fit = |room: f32, wander: f32| {
            let room = max(room, 0.0);
            if wander > room { room / wander } else { 1.0 }
        };
        let fit = min(
            fit(self.place, above),
            fit(end - self.place - self.run, below),
        );
        self.skew *= fit;
        self.arch *= fit;
        self.bite *= fit;
        self.set = self.bite * self.notched(0.0);
        self
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Knob {
    pub segment: u32,
    pub middle: [f32; 2],
    pub radius: f32,
    pub height: f32,
}

impl Knob {
    #[inline]
    pub fn rise(&self, point: Vec2) -> f32 {
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
pub struct Shares([f32; 5]);

impl Shares {
    pub fn of(slope: f32, blend: f32) -> Self {
        let past = FORMS.map(|edge| {
            clamp(
                (slope - (1.0 - blend) * edge) / (2.0 * blend * edge),
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

    pub fn share(&self, reach: Reach) -> f32 {
        self.0[reach as usize]
    }

    pub fn blend(&self, values: [f32; 5]) -> f32 {
        let mut blended = 0.0;
        for (share, value) in self.0.iter().zip(values) {
            blended += share * value;
        }
        blended
    }
}

pub fn bankfull(flood: f32, aspect: f32) -> (f32, f32) {
    let width = WIDTH_LAW[0] * libm::powf(flood, WIDTH_LAW[1]);
    let depth = DEPTH_LAW[0] * libm::powf(flood, DEPTH_LAW[1]);
    let ratio = min(width / depth, aspect);
    let section = width * depth;
    (
        (section * ratio).sqrt(),
        (section / ratio).sqrt() / MEAN_DEPTH,
    )
}

pub fn grains(depth: f32, slope: f32, shares: &Shares) -> (f32, f32) {
    let critical = CRITICAL_SHIELDS * libm::powf(slope, 0.25);
    let median = MEAN_DEPTH * depth * slope / (SUBMERGED * critical * shares.blend(EXCESS));
    (median, D84_MEDIAN * median)
}

pub fn power(flood: f32, slope: f32, width: f32) -> f32 {
    WATER_DENSITY * GRAVITY * flood * slope / width
}

pub struct Bankfull {
    pub floods: Vec<f32>,
    pub slopes: Vec<f32>,
    pub shares: Vec<Shares>,
    pub widths: Vec<f32>,
    pub depths: Vec<f32>,
    pub banks: Vec<f32>,
    pub cut: Vec<bool>,
    pub bends: Vec<f32>,
    pub swings: Vec<f32>,
    pub ledges: Vec<Crest>,
    pub stairs: Vec<(usize, usize)>,
    pub rock: Vec<f32>,
}

pub struct Forms {
    pub pools: Vec<f32>,
    pub dips: Vec<f32>,
    pub held: Vec<f32>,
    pub stepped: Vec<bool>,
    pub crests: Vec<Crest>,
    pub knobs: Vec<Knob>,
}

pub fn forms(
    (points, along): (&[Vec2], &[f32]),
    bankfull: &Bankfull,
    form: &Form,
    finest: f32,
) -> Forms {
    let set = &form.pools;
    let key = place_key(points[0]);
    let count = points.len();
    let end = along[count - 1];
    let share = libm::powf(1.0 / form.shape.floods, 6.0 / 13.0);
    let mut turn = unit_of(key);
    let mut pools = Vec::with_capacity(count);
    for index in 0..count {
        let (width, depth) = (bankfull.widths[index], bankfull.depths[index]);
        if index > 0 {
            turn += (along[index] - along[index - 1]) / (set.apart * bankfull.widths[index - 1]);
        }
        let strength = set.depth * bankfull.shares[index].share(Reach::Pool);
        let ends = smooth(min(along[index], end - along[index]) / (set.ends * width));
        let rhythm = ends * cos(TAU * (turn - turn.floor()));
        let bent = smooth(bankfull.bends[index].abs() / set.bend);
        let swung = max(bankfull.swings[index], bent);
        let shape = rhythm + (2.0 * bent - 1.0 - rhythm) * swung;
        let slope = bankfull.slopes[index];
        let water = share * depth;
        let (_, d84) = grains(depth, slope, &bankfull.shares[index]);
        let froude = min((slope * resistance(MEAN_DEPTH * water / d84)).sqrt(), 1.0);
        let room = max(
            1.0 + 0.5 * froude * froude - 1.5 * libm::powf(froude, 2.0 / 3.0),
            0.0,
        );
        pools.push(max(depth * strength * shape, -set.crest * water * room));
    }
    let steps = steps(along, bankfull, form, (key, finest));
    let knobs = knobs(
        (points, along),
        bankfull,
        (&form.boulders, &steps.crests),
        (key, finest),
    );
    let forming = forming(bankfull);
    let mut crests = steps.crests;
    let mut stepped = steps.stepped;
    for ledge in &bankfull.ledges {
        let rock = 1.0 - along_line(&forming, along, ledge.place);
        if rock > 0.0 {
            let segment = ledge.segment as usize;
            crests.push(Crest {
                drop: rock * ledge.drop,
                pool: rock * ledge.pool,
                ..*ledge
            });
            stepped[segment] = true;
            stepped[segment + 1] = true;
        }
    }
    crests.sort_by(|a, b| a.place.total_cmp(&b.place));
    Forms {
        pools,
        dips: steps.dips,
        held: steps.held,
        stepped,
        crests,
        knobs,
    }
}

struct Stepped {
    dips: Vec<f32>,
    held: Vec<f32>,
    stepped: Vec<bool>,
    crests: Vec<Crest>,
}

fn clasts(bankfull: &Bankfull, index: usize, clasts: [f32; 5]) -> f32 {
    MEAN_DEPTH * bankfull.depths[index] * bankfull.shares[index].blend(clasts)
}

fn steepness(slope: f32) -> f32 {
    clamp(
        STEEPNESS[0] * libm::powf(slope, STEEPNESS[1]),
        STEEPEST[0],
        STEEPEST[1],
    )
}

fn forming(bankfull: &Bankfull) -> Vec<f32> {
    bankfull
        .shares
        .iter()
        .map(|shares| 1.0 - shares.share(Reach::Pool) - shares.share(Reach::Plane))
        .collect()
}

fn mix(from: f32, to: f32, share: f32) -> f32 {
    from + (to - from) * share
}

fn segment_at(along: &[f32], place: f32) -> usize {
    along
        .partition_point(|&at| at <= place)
        .clamp(1, along.len() - 1)
        - 1
}

pub fn ledges(
    form: &Form,
    (key, finest): (u32, f32),
    (along, falls): (&[f32], &[f32]),
    (widths, depths, floods): (&[f32], &[f32], &[f32]),
    banks: &mut [f32],
) -> (Vec<Crest>, Vec<(usize, usize)>) {
    let (set, profile) = (&form.steps, &form.profile);
    let end = along[along.len() - 1];
    let ledge_seed = keyed(key, LEDGE_SEED);
    let (mut crests, mut stairs) = (Vec::new(), Vec::new());
    for (index, &fall) in falls.iter().enumerate() {
        if fall <= 0.0 {
            continue;
        }
        let (width, depth) = (widths[index], depths[index]);
        let seed = keyed(ledge_seed, index as u32);
        let draw = |salt: usize| {
            mix(
                set.spread[0],
                set.spread[1],
                unit_of(keyed(seed, salt as u32)),
            )
        };
        let risers = (fall / max(profile.riser * depth, 1.0e-3)).ceil() as usize;
        let mut places = vec![along[index]];
        for step in 1..risers {
            let place = places[step - 1] - profile.bench * depth * draw(2 * step);
            if place <= 0.0 {
                break;
            }
            places.push(place);
        }
        if places.len() > 1 {
            stairs.push((segment_at(along, places[places.len() - 1]), index + 1));
        }
        let shares: Vec<f32> = (0..places.len()).map(|step| draw(2 * step + 1)).collect();
        let whole: f32 = shares.iter().sum();
        for (step, &place) in places.iter().enumerate() {
            let drop = fall * shares[step] / whole;
            let length = if step == 0 {
                min(set.apart[0] * width, end - place)
            } else {
                places[step - 1] - place
            };
            let mut crest = lipped(
                set,
                keyed(seed, u32::MAX - step as u32),
                (segment_at(along, place), place),
                (drop, length),
                (width, depth, profile.slope(floods[index])),
                (finest, end),
            );
            if step == 0 {
                let run = min(crest.run, along[index + 1] - place);
                crest.tread += crest.run - run;
                crest.run = run;
            } else {
                let first = along.partition_point(|&at| at <= place);
                for point in first..=index {
                    banks[point] -= drop * crest.fallen(along[point] - place);
                }
            }
            crests.push(crest);
        }
    }
    crests.sort_by(|a, b| a.place.total_cmp(&b.place));
    (crests, stairs)
}

pub fn fallen(crests: &[Crest], along: &[f32]) -> Vec<f32> {
    let mut fallen = vec![0.0_f32; along.len()];
    for crest in crests {
        let first = along.partition_point(|&at| at <= crest.place);
        for (total, &at) in fallen.iter_mut().zip(along).skip(first) {
            *total += crest.drop * crest.fallen(at - crest.place);
        }
    }
    fallen
}

fn lipped(
    set: &Steps,
    seed: u32,
    (segment, place): (usize, f32),
    (drop, length): (f32, f32),
    (width, depth, slope): (f32, f32, f32),
    (finest, end): (f32, f32),
) -> Crest {
    let draw = |salt: u32| unit_of(keyed(seed, salt));
    let half = 0.5 * width;
    let run = min(
        max(RISER_SAMPLES * finest, drop / set.riser_slope),
        (1.0 - set.tread) * length,
    );
    let tread = length - run;
    let pool = min(
        min((steepness(slope) - 1.0) * drop, set.pool_aspect * tread),
        set.pool_deepest * depth,
    );
    let many = ((width / max(set.notch_room * depth, 1.0e-3)) as usize).clamp(1, NOTCHES);
    let mut notches = [0.0; NOTCHES];
    for (slot, notch) in notches.iter_mut().enumerate() {
        let slot = slot.min(many - 1);
        *notch = -1.0 + 2.0 * (slot as f32 + 0.25 + 0.5 * draw(5 + slot as u32)) / many as f32;
    }
    let notch = max(set.notch_half * depth, 1.0e-3);
    Crest {
        segment: segment as u32,
        place,
        drop,
        run,
        tread,
        pool,
        skew: half * set.skew * (2.0 * draw(2) - 1.0),
        arch: half * mix(set.arch[0], set.arch[1], draw(3)),
        bite: depth * mix(set.notch_bite[0], set.notch_bite[1], draw(4)),
        set: 0.0,
        notch: notch / max(half, notch),
        notches,
    }
    .fitted(end)
}

fn steps(along: &[f32], bankfull: &Bankfull, form: &Form, (key, finest): (u32, f32)) -> Stepped {
    let set = &form.steps;
    let (slopes, widths, depths) = (
        &bankfull.slopes[..],
        &bankfull.widths[..],
        &bankfull.depths[..],
    );
    let count = along.len();
    let end = along[count - 1];
    let forming = forming(bankfull);
    let floor: Vec<f32> = bankfull
        .banks
        .iter()
        .zip(depths)
        .map(|(&banks, &depth)| banks - depth)
        .collect();
    let least = STEP_SAMPLES * finest;
    let spacing: Vec<f32> = (0..count)
        .map(|index| {
            let width = widths[index];
            let height = set.height * clasts(bankfull, index, set.clasts);
            max(
                clamp(
                    height / (steepness(slopes[index]) * slopes[index]),
                    set.apart[0] * width,
                    set.apart[1] * width,
                ),
                least,
            )
        })
        .collect();
    let at = |values: &[f32], place: f32| along_line(values, along, place);
    let seed = keyed(key, STEP_SEED);
    let mut places = Vec::new();
    let mut place = unit_of(keyed(seed, u32::MAX)) * spacing[0];
    while place < end {
        let draw = unit_of(keyed(seed, places.len() as u32));
        places.push(place);
        place += max(
            at(&spacing, place) * mix(set.spread[0], set.spread[1], draw),
            least,
        );
    }
    let mut steps = Stepped {
        dips: vec![0.0; count],
        held: vec![f32::NAN; count],
        stepped: vec![false; count],
        crests: Vec::with_capacity(places.len()),
    };
    let crest_seed = keyed(key, CREST_SEED);
    let mut point = 0;
    for (number, &place) in places.iter().enumerate() {
        let next = places.get(number + 1).copied().unwrap_or(end);
        while point < count && along[point] < place {
            point += 1;
        }
        let level = at(&floor, next);
        let drop = at(&forming, place) * (at(&floor, place) - level);
        if drop.is_nan() || drop <= 0.0 {
            continue;
        }
        let crest = lipped(
            set,
            keyed(crest_seed, number as u32),
            (segment_at(along, place), place),
            (drop, next - place),
            (at(widths, place), at(depths, place), at(slopes, place)),
            (finest, end),
        );
        while point < count && along[point] < next {
            let fall = forming[point] * max(floor[point] - level, 0.0);
            steps.dips[point] = fall - drop * (1.0 - crest.fallen(along[point] - place));
            steps.held[point] = next;
            steps.stepped[point] = forming[point] > 0.0;
            point += 1;
        }
        steps.crests.push(crest);
    }
    steps
}

fn knobs(
    (points, along): (&[Vec2], &[f32]),
    bankfull: &Bankfull,
    (set, crests): (&Boulders, &[Crest]),
    (key, finest): (u32, f32),
) -> Vec<Knob> {
    let (shares, widths, depths) = (
        &bankfull.shares[..],
        &bankfull.widths[..],
        &bankfull.depths[..],
    );
    let count = points.len();
    let at = |values: &[f32], place: f32| along_line(values, along, place);
    let forming = forming(bankfull);
    let (row_seed, scatter_seed) = (keyed(key, ROW_SEED), keyed(key, SCATTER_SEED));
    let mut knobs = Vec::new();
    let mut raise = |segment: usize, middle: Vec2, radius: f32, height: f32| {
        if radius >= KNOB_SAMPLES * finest && height > 0.0 {
            knobs.push(Knob {
                segment: segment as u32,
                middle: middle.to_array(),
                radius,
                height,
            });
        }
    };
    for (number, crest) in crests.iter().enumerate() {
        let (width, depth) = (at(widths, crest.place), at(depths, crest.place));
        let stepped = at(&forming, crest.place);
        if stepped <= 0.0 {
            continue;
        }
        let half = 0.5 * width;
        let seed = keyed(row_seed, number as u32);
        let mut gaps = crest.notches;
        gaps.sort_by(f32::total_cmp);
        let mut from = -1.0;
        for (slot, to) in gaps.into_iter().chain([1.0]).enumerate() {
            let (across, gap) = (0.5 * (from + to), (to - from) * half);
            from = to;
            let radius = min(0.5 * (gap - crest.notch * half), KNOB_REACH);
            let place = crest.lip(across) - set.back * radius;
            let segment = segment_at(along, place);
            let (a, run) = (points[segment], points[segment + 1] - points[segment]);
            let length = along[segment + 1] - along[segment];
            let share = if length > 0.0 {
                clamp((place - along[segment]) / length, 0.0, 1.0)
            } else {
                0.0
            };
            let rise = mix(set.rise[0], set.rise[1], unit_of(keyed(seed, slot as u32)));
            raise(
                segment,
                a + run * share + run.normalize_or(Vec2::X).perp() * (across * half),
                radius,
                min(set.height * radius, rise * depth) * stepped,
            );
        }
    }
    let cascading: Vec<f32> = shares
        .iter()
        .map(|shares| shares.share(Reach::Cascade))
        .collect();
    for segment in 0..count - 1 {
        let length = along[segment + 1] - along[segment];
        let width = 0.5 * (widths[segment] + widths[segment + 1]);
        let depth = 0.5 * (depths[segment] + depths[segment + 1]);
        let (low, high) = (KNOB_SAMPLES * finest, min(KNOB_REACH, set.span * width));
        let rough = 0.5 * (cascading[segment] + cascading[segment + 1]);
        if high < low || rough <= 0.0 {
            continue;
        }
        let (a, run) = (points[segment], points[segment + 1] - points[segment]);
        let out = run.normalize_or(Vec2::X).perp();
        let area = PI * (low * low + low * high + high * high) / 3.0;
        let expected = -libm::logf(1.0 - set.cover) * length * width / area;
        let seed = keyed(scatter_seed, segment as u32);
        let many = (expected + unit_of(keyed(seed, 0))) as u32;
        for knob in 0..many {
            let draw = |salt: u32| unit_of(keyed(seed, 4 * knob + salt));
            let (share, across) = (draw(2), 2.0 * draw(3) - 1.0);
            let radius = mix(low, high, draw(1));
            let rise = mix(set.rise[0], set.rise[1], draw(4));
            raise(
                segment,
                a + run * share + out * (across * 0.5 * width),
                radius,
                min(set.height * radius, rise * depth) * rough,
            );
        }
    }
    knobs.sort_by_key(|knob| knob.segment);
    knobs
}

pub fn unit_of(key: u32) -> f32 {
    let mut h = key;
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    ((h ^ (h >> 16)) >> 8) as f32 / 16_777_216.0
}

pub fn keyed(key: u32, value: u32) -> u32 {
    let mut h = key ^ value.wrapping_mul(0x9e37_79b9);
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

pub fn place_key(point: Vec2) -> u32 {
    point.x.to_bits().wrapping_mul(0x8da6_b343) ^ point.y.to_bits().wrapping_mul(0xd816_3841)
}

engine_codec::record!(Pools {
    apart,
    depth,
    bend,
    ends,
    crest,
});
engine_codec::record!(Steps {
    clasts,
    height,
    apart,
    spread,
    riser_slope,
    tread,
    pool_aspect,
    pool_deepest,
    skew,
    arch,
    notch_room,
    notch_half,
    notch_bite,
});
engine_codec::record!(Boulders {
    back,
    span,
    height,
    rise,
    cover,
});
engine_codec::tagged!(Reach {
    0 => Pool,
    1 => Plane,
    2 => Steps,
    3 => Cascade,
    4 => Falls,
});
engine_codec::record!(Crest {
    segment,
    place,
    drop,
    run,
    tread,
    pool,
    skew,
    arch,
    bite,
    set,
    notch,
    notches,
});
engine_codec::record!(Knob {
    segment,
    middle,
    radius,
    height,
});
