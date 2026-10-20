//! Steady open-channel flow along a bake's rivers, each in its channel. A
//! channel's top is the closest level to its banks that never rises
//! downstream (a monotone fit of the finest ground along it, smoothed), and
//! its bed lies the channel's depth below; the river cuts that channel and
//! no other (`Rivers::cap`). Each river carries the discharge its
//! drained area yields, by default what runs rivers a set share of the way
//! up their channels where the channels' slope is typical
//! (`filling_yield`), and stands as deep as uniform flow down the
//! channel's slope needs (Manning's formula for a wide channel), up to its
//! banks: pools stand deep and slow where the channel levels out, riffles
//! run shallow and fast where it steepens. A lake or the river it joins
//! holds its water back: from such an outlet, its level rises upstream
//! against friction until its own uniform flow stands higher (backwater),
//! so rivers meet their confluences and lakes at one level.
//! Where a river runs steep its bed falls in steps, as rock does: irregular
//! ledges with the bed between falling more gently and a pocket scoured
//! under each ledge's face; steep channels run narrower, and water skimming
//! down them meets the friction of the rough rock. Off faces steeper than
//! water clings to, and higher than its critical depth twice over, the river
//! leaves its bed as a jet on the parabola its speed and gravity give, and
//! lands where the jet meets the water below. A steep stretch falls as one
//! cascade: its water clinging to the rock and leaping off its ledges. Foam
//! forms
//! where water runs faster than its waves (Froude number above one), where
//! it slows back through a hydraulic jump and where it lands from a fall,
//! and rides the current downstream, thinning as it goes.
//!
//! Every step uses only IEEE arithmetic and `libm`'s functions, so the same
//! bake gives the same bits natively and in the browser.

use engine_math::{clamp, max, min};
use glam::{Vec2, Vec3};

use crate::hydrology::{FALL_HEIGHT, Fall};
use crate::rivers::{BANK_MARGIN, MIN_DEPTH, River, RiverLine, STREAM_MARGIN};

/// Gravity water falls and waves run under (meters a second squared).
pub const GRAVITY: f32 = 9.81;
/// The shallowest water a river's flow keeps (meters), and the gentlest
/// slope its surface reads, so still reaches stay finite.
pub(crate) const SHALLOWEST: f32 = 0.03;
const GENTLEST: f32 = 2.0e-4;
/// The least discharge a river carries (cubic meters a second).
pub(crate) const TRICKLE: f32 = 1.0e-3;
/// Meters either side of a point its surface is averaged over, and its
/// slope read across.
const SMOOTHING: f32 = 9.0;
const SLOPE_REACH: f32 = 12.0;
/// The steepest drop (rise over run) a river's water clings to its bed
/// down: off steeper faces at least LEAP_DEPTHS critical depths and
/// LEAP_LEAST meters high it leaps clear as a nappe (Chanson: nappe flow
/// holds below about 0.49 critical depths a step's height at 45 degrees);
/// down gentler ones it skims over the rock.
const CLING_SLOPE: f32 = 1.2;
const LEAP_DEPTHS: f32 = 2.0;
const LEAP_LEAST: f32 = 1.5;
/// Off a step's face (a drop at a crest, [`stepped`]) the water leaps clear
/// as a nappe where the face stands LEAP_DEPTHS critical depths high or
/// more, and at least NAPPE_LEAST meters; lower, it skims over it.
const NAPPE_LEAST: f32 = 0.4;
/// Where a river's bed falls STEPPED or more a meter (its smooth fit read
/// across SLOPE_REACH) it falls in steps, as rock does: crests a share of
/// the channel's width apart (STEP_SPACING), and STEP_RUN meters apart at
/// the least and most, each step's face taking a share of what the bed
/// falls from its crest to the next (STEP_SHARE), at most STEP_HIGHEST
/// meters, and the bed between falling the rest: ledges of rock the water
/// skims over. SLIDES of the crests start slides instead, the bed falling
/// evenly to a crest SLIDE_SPACING widths on (at most SLIDE_LONGEST meters):
/// smooth rock the water runs down whole; and LEDGES of them stand over a
/// face as high as STEP_SHARE says up to LEDGE_HIGHEST meters, the next
/// crest LEDGE_RUN meters on: the drops the water leaps off. Under each
/// face the water landing there scours a pocket POCKET_SHARE of the face
/// deep (at most POCKET_DEEPEST meters), as long as the face is high but
/// at least half the channel's width.
const STEPPED: f32 = 0.25;
const STEP_SPACING: [f32; 2] = [0.5, 2.0];
const STEP_RUN: [f32; 2] = [3.0, 12.0];
const STEP_SHARE: [f32; 2] = [0.15, 0.7];
const STEP_HIGHEST: f32 = 2.0;
const SLIDES: f32 = 0.3;
const SLIDE_SPACING: [f32; 2] = [1.5, 3.5];
const SLIDE_LONGEST: f32 = 30.0;
const LEDGES: f32 = 0.12;
const LEDGE_RUN: [f32; 2] = [10.0, 25.0];
const LEDGE_HIGHEST: f32 = 8.0;
const POCKET_SHARE: f32 = 0.35;
const POCKET_DEEPEST: f32 = 2.5;
/// How much wider a channel runs where its pocket is a meter or more deep:
/// the pools under steps open out, the crests between them close in.
const POCKET_WIDENING: f32 = 0.3;
/// Channels narrow as they steepen, their width as the slope to the -3/16
/// (Finnegan et al. 2005), from NARROWING_FROM a meter down to NARROWEST
/// of their width.
const NARROWING_FROM: f32 = 0.05;
const NARROWEST: f32 = 0.55;
/// Darcy's friction factor of water skimming down a stepped or rough steep
/// bed (Chanson: about 1, against a smooth chute's hundredths): over beds
/// falling through SKIMMING a meter (from the first to the second) the
/// water comes to stand as deep as that friction holds it, and runs no
/// faster.
const SKIMMING_FRICTION: f32 = 1.0;
const SKIMMING: [f32; 2] = [0.1, 0.3];
/// Meters the water may stand over its channel's top where a lake or the
/// river it joins holds it back before the banks cap it.
const BANK_FLOOD: f32 = 0.3;
/// The fastest water runs against its waves (Froude number): faster, it
/// breaks into aerated whitewater that bulks out and runs no faster.
const MAX_FROUDE: f32 = 4.0;
/// Seconds foam lasts as the current carries it.
const FOAM_SECONDS: f32 = 6.0;
/// The Froude number above which rapids foam, and the one at which they
/// whiten fully: water running a little faster than its waves only ruffles,
/// a torrent breaks white. Rapids and jumps whiten water at most
/// RAPIDS_WHITE: whiter than that, it has landed from a fall.
const RAPIDS: [f32; 2] = [1.2, 3.5];
const RAPIDS_WHITE: f32 = 0.7;
/// Water skimming down a steep bed takes in air once its turbulence reaches
/// its surface, INCEPTION_DEPTHS of its depths down the steep bed (at least
/// INCEPTION_LEAST meters), and comes to hold an air fraction of about 0.9
/// sin of the bed's angle (Chanson): it whitens as that fraction rises
/// through AIRED, to RAPIDS_WHITE. Beds falling STEEP_AIR a meter or more
/// are steep.
const INCEPTION_DEPTHS: f32 = 6.0;
const INCEPTION_LEAST: f32 = 1.0;
const AIRED: [f32; 2] = [0.15, 0.55];
const STEEP_AIR: f32 = 0.1;
/// A fall's jet sampled at least every this many meters of its run and of
/// its drop, so a plunge's curve bends smoothly.
const JET_STEP: f32 = 0.5;
/// Stretches whose bed falls CASCADE_SLOPE or more a meter fall as
/// cascades, white water clinging to the rock; gentler stretches no more
/// than BRIDGE widths long between them are the cascade's pools, which it
/// runs on through.
const CASCADE_SLOPE: f32 = 0.45;
const BRIDGE: f32 = 2.0;
/// The slowest water runs down a cascade (meters a second), so its seconds
/// there stay finite.
const SLOWEST: f32 = 0.5;
/// The least share of its channel's half width a river's water spans (its
/// channel shaped shallower where its flow would run narrower), the least
/// its channel's banks stand over its water (meters), and the meters either
/// side of a point its share is averaged over, so its water widens and
/// narrows gradually along its course.
const FILL_LEAST: f32 = 0.6;
const FREEBOARD: f32 = 0.1;
const FILL_SMOOTHING: f32 = 12.0;

/// How the bake's rivers flow: the discharge each square meter of drained
/// land yields (cubic meters a second) and Manning's roughness of their
/// beds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Flow {
    pub(crate) yield_rate: f32,
    pub(crate) roughness: f32,
}

/// A river's line with what holds it: the finest ground at each point (its
/// middle and either bank, terrain-local meters), the level it starts from
/// when it leaves a lake, and the level its outlet holds it to (a lake's, or
/// the river it joins), if any.
pub(crate) struct Reach<'a> {
    pub(crate) line: &'a RiverLine,
    pub(crate) ground: &'a [[f32; 3]],
    pub(crate) source: Option<f32>,
    pub(crate) outlet: Option<f32>,
}

/// The depth (meters) of uniform flow in a wide channel `width` meters across
/// carrying `discharge` down `slope`, Manning's `roughness`: q = h^5/3 S^1/2
/// / n per meter of width.
fn normal_depth(discharge: f32, width: f32, slope: f32, roughness: f32) -> f32 {
    let per_meter = discharge / width;
    libm::powf(per_meter * roughness / slope.sqrt(), 0.6)
}

/// The depth (meters) `discharge` stands over `width` meters skimming down
/// `slope` against SKIMMING_FRICTION: q = d (8 g d S / f)^1/2.
fn skimming_depth(discharge: f32, width: f32, slope: f32) -> f32 {
    let per_meter = discharge / width;
    libm::cbrtf(per_meter * per_meter * SKIMMING_FRICTION / (8.0 * GRAVITY * slope))
}

/// The share of its width a channel keeps falling `slope` a meter.
fn narrowing(slope: f32) -> f32 {
    let steeper = max(slope, NARROWING_FROM) / NARROWING_FROM;
    min(max(libm::powf(steeper, -3.0 / 16.0), NARROWEST), 1.0)
}

/// `low` to `high` by `share`.
fn between([low, high]: [f32; 2], share: f32) -> f32 {
    low + (high - low) * share
}

/// 0 at or below `low`, rising smoothly to 1 at `high`.
fn ramp(value: f32, [low, high]: [f32; 2]) -> f32 {
    let t = min(max((value - low) / (high - low), 0.0), 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A number in [0, 1) hashed from `key`.
fn unit_of(key: u32) -> f32 {
    let mut h = key;
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    ((h ^ (h >> 16)) >> 8) as f32 / 16_777_216.0
}

/// The key of a river's point at `point`: its place, so the same ground
/// steps alike in every bake.
fn place_key(point: Vec2) -> u32 {
    point.x.to_bits().wrapping_mul(0x8da6_b343) ^ point.y.to_bits().wrapping_mul(0xd816_3841)
}

/// The depth (meters) at which `discharge` crosses `width` meters as fast as
/// its waves: (q^2 / g)^1/3.
fn critical_depth(discharge: f32, width: f32) -> f32 {
    let per_meter = discharge / width;
    libm::cbrtf(per_meter * per_meter / GRAVITY)
}

/// The slope friction takes from flow `depth` deep carrying `discharge` over
/// `width` (Manning's formula solved for it).
pub(crate) fn friction_slope(discharge: f32, width: f32, depth: f32, roughness: f32) -> f32 {
    let per_meter = discharge / width;
    let r = per_meter * roughness / libm::powf(depth, 5.0 / 3.0);
    r * r
}

/// The share of a stretch's ground a river's channel stands above where it
/// cannot follow the ground without rising: low, so channels cut through
/// the rises along their course rather than stand on banks built up over
/// the hollows between.
const FIT_SHARE: f32 = 0.2;

/// The fit to `values` that never rises along them, `weights` each, in
/// place: pooled adjacent violators, each pool standing at the level
/// FIT_SHARE of its weight lies below (a weighted quantile, the least
/// absolute error weighted that way).
fn falling_fit(values: &mut [f32], weights: &[f32]) {
    // Pools of equal level: their values with their weights (by value),
    // their total weight and their level.
    struct Pool {
        members: Vec<(f32, f32)>,
        weight: f32,
        level: f32,
    }
    let level = |members: &[(f32, f32)], weight: f32| {
        let wanted = weight * FIT_SHARE;
        let mut below = 0.0;
        for &(value, member) in members {
            below += member;
            if below >= wanted {
                return value;
            }
        }
        members[members.len() - 1].0
    };
    let mut pools: Vec<Pool> = Vec::with_capacity(values.len());
    for (&value, &weight) in values.iter().zip(weights) {
        pools.push(Pool {
            members: vec![(value, weight)],
            weight,
            level: value,
        });
        while pools.len() >= 2 && pools[pools.len() - 2].level < pools[pools.len() - 1].level {
            let low = pools.pop().expect("two pools");
            let high = pools.last_mut().expect("two pools");
            let mut members = Vec::with_capacity(high.members.len() + low.members.len());
            let (mut a, mut b) = (
                high.members.iter().peekable(),
                low.members.iter().peekable(),
            );
            while let (Some(&&x), Some(&&y)) = (a.peek(), b.peek()) {
                if x.0.total_cmp(&y.0).is_le() {
                    members.push(x);
                    a.next();
                } else {
                    members.push(y);
                    b.next();
                }
            }
            members.extend(a);
            members.extend(b);
            high.weight += low.weight;
            high.level = level(&members, high.weight);
            high.members = members;
        }
    }
    let mut index = 0;
    for pool in pools {
        let length = pool.members.len();
        values[index..index + length].fill(pool.level);
        index += length;
    }
}

/// `values` averaged over [`SMOOTHING`] meters either side of each point
/// (`along` meters from the first) but where `kept` (a stepped bed's, whose
/// ledges smoothing would wear away), never rising along them.
fn smoothed(values: &[f32], along: &[f32], kept: &[bool]) -> Vec<f32> {
    let count = values.len();
    let mut out = Vec::with_capacity(count);
    let (mut first, mut last) = (0, 0);
    for index in 0..count {
        // The window around the point, as wide either side as both allow.
        let reach = min(
            SMOOTHING,
            min(along[index] - along[0], along[count - 1] - along[index]),
        );
        while along[index] - along[first] > reach {
            first += 1;
        }
        while last + 1 < count && along[last + 1] - along[index] <= reach {
            last += 1;
        }
        if kept[index] {
            out.push(values[index]);
            continue;
        }
        let sum: f32 = values[first..=last].iter().sum();
        out.push(sum / (last - first + 1) as f32);
    }
    for index in 1..count {
        out[index] = min(out[index], out[index - 1]);
    }
    out
}

/// The slope of `top` (meters a meter, downstream) at each point, `along`
/// meters from the first: across the points within SLOPE_REACH either side,
/// so the flats and steps its fit leaves read as the slope they make
/// together, never gentler than GENTLEST.
fn slopes(top: &[f32], along: &[f32]) -> Vec<f32> {
    let count = top.len();
    let (mut first, mut last) = (0, 0);
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        while along[index] - along[first] > SLOPE_REACH {
            first += 1;
        }
        while last + 1 < count && along[last + 1] - along[index] <= SLOPE_REACH {
            last += 1;
        }
        let (before, after) = (
            first.min(index.saturating_sub(1)),
            last.max((index + 1).min(count - 1)),
        );
        let run = max(along[after] - along[before], 1.0e-3);
        out.push(max((top[before] - top[after]) / run, GENTLEST));
    }
    out
}

/// The points of a river's flow: its surface, depth, speed and discharge.
struct Profile {
    surface: Vec<f32>,
    depth: Vec<f32>,
    speed: Vec<f32>,
    discharge: Vec<f32>,
}

/// Where a river's points stand along its line (meters from its first).
pub(crate) fn distances(points: &[Vec2]) -> Vec<f32> {
    let mut along = Vec::with_capacity(points.len());
    along.push(0.0);
    for pair in points.windows(2) {
        along.push(along[along.len() - 1] + pair[0].distance(pair[1]));
    }
    along
}

/// The stretches of `targets` (the highest each point's water may stand)
/// dropping steeper than CLING_SLOPE by at least `least` at their first
/// point: each its first and last point.
fn drops(targets: &[f32], along: &[f32], least: &[f32]) -> Vec<(usize, usize)> {
    let steep = |index: usize| {
        targets[index] - targets[index + 1] > CLING_SLOPE * (along[index + 1] - along[index])
    };
    let mut found = Vec::new();
    let mut index = 0;
    while index + 1 < targets.len() {
        if !steep(index) {
            index += 1;
            continue;
        }
        let lip = index;
        while index + 1 < targets.len() && steep(index) {
            index += 1;
        }
        if targets[lip] - targets[index] >= least[lip] {
            found.push((lip, index));
        }
    }
    found
}

/// The highest each of `reach`'s points' channel may stand: below the
/// lowest of its ground and banks, streams a little, rivers a little more.
fn tops(reach: &Reach<'_>) -> Vec<f32> {
    let line = reach.line;
    (0..line.points.len())
        .map(|index| {
            let [middle, left, right] = reach.ground[index];
            let margin = if line.depth[index] < MIN_DEPTH {
                STREAM_MARGIN
            } else {
                BANK_MARGIN
            };
            min(middle, min(left, right)) - margin
        })
        .collect()
}

/// A river's bed as a smooth fit gives it: its level at each point and how
/// fast it falls there (meters a meter, read across SLOPE_REACH).
struct Plan {
    top: Vec<f32>,
    slopes: Vec<f32>,
}

impl Plan {
    /// The smooth fit of `targets`, `along` meters from the first.
    fn of(targets: &[f32], along: &[f32]) -> Self {
        let count = targets.len();
        let mut level = targets.to_vec();
        falling_fit(&mut level, &vec![1.0; count]);
        let top = smoothed(&level, along, &vec![false; count]);
        let slopes = slopes(&top, along);
        Self { top, slopes }
    }
}

/// What steps make of a river's bed: the meters each point's bed is scoured
/// below its channel (pockets under the steps' faces), the points whose
/// stepped level its fit keeps as it stands, and the meters each crest's
/// face drops to the next point (none elsewhere).
struct Steps {
    pocket: Vec<f32>,
    kept: Vec<bool>,
    face: Vec<f32>,
}

/// Steps `targets` where `plan` falls STEPPED or more a meter, along a
/// river at `points`, `along` meters from its first and `widths` across,
/// where `river` (streams step none).
fn stepped(
    targets: &mut [f32],
    plan: &Plan,
    (points, along, widths): (&[Vec2], &[f32], &[f32]),
    river: impl Fn(usize) -> bool,
) -> Steps {
    let count = targets.len();
    let mut steps = Steps {
        pocket: vec![0.0; count],
        kept: vec![false; count],
        face: vec![0.0; count],
    };
    let steep = |index: usize| plan.slopes[index] >= STEPPED && river(index);
    let mut crest = 0;
    while crest + 1 < count {
        if !steep(crest) {
            crest += 1;
            continue;
        }
        // The next crest a share of the width on, while the bed stays
        // steep: the face drops a share of what the bed falls to it (a
        // slide's none), the bed below falling the rest.
        let key = place_key(points[crest]);
        let kind = unit_of(key ^ 0x2f6b_7a91);
        let (slide, ledge) = (kind < SLIDES, kind >= 1.0 - LEDGES);
        let spacing = if ledge {
            between(LEDGE_RUN, unit_of(key))
        } else if slide {
            min(
                widths[crest] * between(SLIDE_SPACING, unit_of(key)),
                SLIDE_LONGEST,
            )
        } else {
            clamp(
                widths[crest] * between(STEP_SPACING, unit_of(key)),
                STEP_RUN[0],
                STEP_RUN[1],
            )
        };
        let mut next = crest + 1;
        while next + 1 < count && steep(next) && along[next] - along[crest] < spacing {
            next += 1;
        }
        let fall = max(plan.top[crest] - plan.top[next], 0.0);
        let face = if slide {
            0.0
        } else {
            min(
                between(STEP_SHARE, unit_of(key ^ 0x68e3_1da4)) * fall,
                if ledge { LEDGE_HIGHEST } else { STEP_HIGHEST },
            )
        };
        let run = max(along[next] - along[crest + 1], 1.0e-3);
        steps.face[crest] = face;
        targets[crest] = min(targets[crest], plan.top[crest]);
        steps.kept[crest] = true;
        for index in crest + 1..=next {
            let share = (along[index] - along[crest + 1]) / run;
            targets[index] = min(
                targets[index],
                plan.top[crest] - face - (fall - face) * share,
            );
            steps.kept[index] = true;
        }
        // The pocket the water landing under the face scours, deepest at
        // the face.
        let deepest = min(POCKET_SHARE * face, POCKET_DEEPEST);
        let length = max(face, 0.5 * widths[crest]);
        for index in crest + 1..next {
            let from = (along[index] - along[crest + 1]) / length;
            if from >= 1.0 {
                break;
            }
            steps.pocket[index] = max(steps.pocket[index], deepest * (1.0 - from * from));
        }
        crest = next;
    }
    steps
}

/// The discharge each square meter drained yields (cubic meters a second)
/// for `reaches`' rivers to run `fill` of the way up their channels where
/// their channels' slope is typical: at every point of a river (streams
/// aside), what uniform flow that deep carries down its channel's slope
/// over the area it drains, the middle of those; `None` without rivers.
pub(crate) fn filling_yield(reaches: &[Reach<'_>], fill: f32, roughness: f32) -> Option<f32> {
    let mut yields = Vec::new();
    for reach in reaches {
        let line = reach.line;
        let count = line.points.len();
        if count < 2 {
            continue;
        }
        let along = distances(&line.points);
        let mut level = tops(reach);
        falling_fit(&mut level, &vec![1.0; count]);
        let top = smoothed(&level, &along, &vec![false; count]);
        for (index, &slope) in slopes(&top, &along).iter().enumerate() {
            if line.depth[index] < MIN_DEPTH {
                continue;
            }
            let width = max(2.0 * line.half[index], 0.2);
            let depth = fill * line.depth[index];
            let discharge = width * libm::powf(depth, 5.0 / 3.0) * slope.sqrt() / roughness;
            yields.push(discharge / max(line.area[index], 1.0));
        }
    }
    if yields.is_empty() {
        return None;
    }
    yields.sort_by(f32::total_cmp);
    Some(yields[yields.len() / 2])
}

/// The water of the stretch `range` of a river (its channel's tops,
/// distances, widths, depths and the discharges it carries): its channel's
/// fit top and its bed the channel's depth below, starting at `source` when
/// it leaves a lake and drawn down to `outlet` at its end, and the water
/// standing in it as deep as uniform flow runs, up to the channel's top,
/// held back to `outlet`, with its speed there.
fn stretch(
    range: std::ops::Range<usize>,
    (targets, along): (&[f32], &[f32]),
    (widths, channels, discharges): (&[f32], &[f32], &[f32]),
    (source, outlet): (Option<f32>, Option<f32>),
    (flow, steps): (&Flow, &Steps),
    profile: &mut Profile,
) {
    let start = range.start;
    let (kept, pocket) = (&steps.kept[range.clone()], &steps.pocket[range.clone()]);
    let along = &along[range.clone()];
    let mut level = targets[range.clone()].to_vec();
    let mut weights = vec![1.0_f32; level.len()];
    // A lake's outlet starts the river at its level; an outlet below the
    // ground it reaches draws the river down to meet it.
    if let Some(source) = source {
        level[0] = source;
        weights[0] = 1.0e4;
    }
    if let Some(outlet) = outlet
        && outlet < level[level.len() - 1]
    {
        let last = level.len() - 1;
        level[last] = outlet;
        weights[last] = 1.0e4;
    }
    falling_fit(&mut level, &weights);
    let top = smoothed(&level, along, kept);
    let count = top.len();
    let slopes = slopes(&top, along);
    // The water as deep as uniform flow down the top's slope runs (down
    // steep beds as deep as skimming over the rock holds it), up to the
    // channel's depth, standing that deep over a bed the channel's depth
    // below its top, but never rising downstream: where it would stand
    // higher than the water above it (a riffle opening into a pool), it
    // stands at that water's level, as deep over a bed scoured deeper.
    let mut bed = Vec::with_capacity(count);
    let mut surface: Vec<f32> = Vec::with_capacity(count);
    for index in 0..count {
        let (width, discharge) = (widths[start + index], discharges[start + index]);
        let channel = max(channels[start + index], SHALLOWEST);
        let uniform = max(
            normal_depth(discharge, width, slopes[index], flow.roughness),
            SHALLOWEST,
        );
        let skimming =
            skimming_depth(discharge, width, slopes[index]) * ramp(slopes[index], SKIMMING);
        let depth = min(max(uniform, skimming), channel);
        let own = top[index] - channel + depth;
        let level = surface.last().map_or(own, |&above| min(own, above));
        surface.push(level);
        bed.push(level - depth);
    }
    // A lake's outlet: the water leaves it at its level.
    if let Some(source) = source {
        surface[0] = max(surface[0], min(source, top[0]));
    }
    // Held back from the outlet: the level holding it rises upstream as
    // friction takes its share of the flow over each run, at the depth it
    // stands over the bed below, until it stands no higher than the river's
    // own flow. Water held above its lowest bank stands no higher than the
    // bank allows, while the level holding it runs on.
    if let Some(outlet) = outlet
        && outlet > surface[count - 1]
    {
        let mut level = outlet;
        surface[count - 1] = min(outlet, top[count - 1] + BANK_FLOOD);
        for index in (0..count - 1).rev() {
            let below = index + 1;
            let (width, discharge) = (widths[start + below], discharges[start + below]);
            let depth = max(level - bed[below], SHALLOWEST);
            level += friction_slope(discharge, width, depth, flow.roughness)
                * (along[below] - along[index]);
            if level <= surface[index] {
                break;
            }
            surface[index] = min(level, top[index] + BANK_FLOOD);
        }
    }
    // Under each step's face the bed is scoured deeper, and the water stands
    // in the pocket as high as it leaves it.
    let mut leaving = f32::NAN;
    for index in (0..count).rev() {
        if pocket[index] > 0.0 && leaving.is_finite() {
            bed[index] -= pocket[index];
            surface[index] = max(leaving, surface[index] - pocket[index]);
        } else {
            leaving = surface[index];
        }
    }
    for index in 0..count {
        let (width, discharge) = (widths[start + index], discharges[start + index]);
        let depth = max(surface[index] - bed[index], SHALLOWEST);
        profile.surface[start + index] = surface[index];
        profile.depth[start + index] = depth;
        // Water its channel cannot hold runs no faster than friction lets
        // it at the depth it stands: what is more spills past its banks.
        let friction = libm::powf(depth, 2.0 / 3.0) * slopes[index].sqrt() / flow.roughness;
        profile.speed[start + index] = min(
            min(discharge / (width * depth), friction),
            MAX_FROUDE * (GRAVITY * depth).sqrt(),
        );
    }
}

/// Where the jet leaving `lip` lands: the first later point whose water the
/// parabola from the lip's surface at `speed` meets, and the jet's path to
/// there (terrain-local). `None` when it never leaves the drop's ground.
fn jet(
    lip: usize,
    (points, along, targets): (&[Vec2], &[f32], &[f32]),
    surface: f32,
    speed: f32,
) -> Option<(usize, Vec<[f32; 3]>)> {
    let height = |run: f32| surface - GRAVITY * run * run / (2.0 * speed * speed);
    let landing = (lip + 1..targets.len()).find(|&index| {
        let run = along[index] - along[lip];
        height(run) <= targets[index]
    })?;
    if landing <= lip + 1 {
        return None;
    }
    // The flight to where the parabola meets the landing's water.
    let (from, to) = (along[landing - 1] - along[lip], along[landing] - along[lip]);
    let (above, below) = (
        height(from) - targets[landing - 1],
        height(to) - targets[landing],
    );
    let share = if above > below {
        above / (above - below)
    } else {
        1.0
    };
    let reach = from + (to - from) * min(max(share, 0.0), 1.0);
    let direction = (points[lip + 1] - points[lip]).normalize_or(Vec2::X);
    let drop = surface - height(reach);
    let steps = max(((reach + drop) / JET_STEP).ceil(), 1.0) as usize;
    let path = (0..=steps)
        .map(|step| {
            let run = reach * step as f32 / steps as f32;
            let at = points[lip] + direction * run;
            [at.x, height(run), at.y]
        })
        .collect();
    Some((landing, path))
}

/// A jet as [`river`] finds it: its lip and the first point after where it
/// lands, the speed it leaves at, its path and the speed it lands at.
struct Jet {
    lip: usize,
    landing: usize,
    speed: f32,
    path: Vec<[f32; 3]>,
    landing_speed: f32,
}

/// The falls of a river at `points`, its water as `profile` runs it and
/// leaping off its ledges as `jets` do: every stretch whose bed falls
/// CASCADE_SLOPE or more a meter (`slopes`, its smooth fit's) or that
/// leaps, falling FALL_HEIGHT or more or leaping at all, falls as one: its
/// water's path down it, clinging to the rock on its water and through the
/// air on each jet and clear of each step's face it leaps (`nappes`, from
/// its points), and the seconds it takes; and the points their sheets run
/// over. The river is `widths` across, carries `discharges` and drains
/// `areas`; where `river` is false streams run, which fall in none.
fn cascades(
    jets: &[Jet],
    (points, slopes, nappes_at): (&[Vec2], &[f32], &[bool]),
    profile: &Profile,
    (widths, discharges, areas): (&[f32], &[f32], &[f32]),
    river: impl Fn(usize) -> bool,
) -> (Vec<Fall>, Vec<bool>) {
    let count = points.len();
    let along = distances(points);
    let leaping = |at: usize| jets.iter().find(|jet| jet.lip == at);
    let steep = |at: usize| river(at) && slopes[at] >= CASCADE_SLOPE;
    let mut falls = Vec::new();
    let mut sheeted = vec![false; count];
    let mut first = 0;
    while first < count {
        if !(steep(first) || leaping(first).is_some()) {
            first += 1;
            continue;
        }
        // Down the stretch while it stays steep, through each jet and each
        // of its pools.
        let mut last = first;
        let mut at = first;
        while at < count {
            if let Some(jet) = leaping(at) {
                last = jet.landing.min(count - 1);
                at = jet.landing;
            } else if steep(at) {
                last = at;
                at += 1;
            } else if let Some(next) = (at + 1..count)
                .take_while(|&next| along[next] - along[at] <= BRIDGE * widths[at])
                .find(|&next| steep(next) || leaping(next).is_some())
            {
                last = next - 1;
                at = next;
            } else {
                break;
            }
        }
        let (mut path, mut times, mut aloft) = (Vec::new(), Vec::new(), Vec::new());
        let (mut across, mut nappes) = (Vec::new(), Vec::new());
        let (mut time, mut previous, mut landing_speed) = (0.0, None, None);
        let mut at = first;
        while at <= last {
            if let Some(jet) = leaping(at) {
                // Through the air, as long as its run at the speed it
                // left at.
                let lip = Vec3::from_array(jet.path[0]);
                if let Some(previous) = previous {
                    time += lip.distance(previous) / max(profile.speed[at], SLOWEST);
                }
                let leaving = time;
                for point in &jet.path {
                    let run = Vec2::new(point[0] - lip.x, point[2] - lip.z).length();
                    time = leaving + run / jet.speed;
                    path.push(*point);
                    times.push(time);
                    aloft.push(true);
                    nappes.push(false);
                    across.push(widths[at]);
                }
                if let Some(landed) = aloft.last_mut() {
                    *landed = false;
                }
                previous = path.last().copied().map(Vec3::from_array);
                landing_speed = Some(jet.landing_speed);
                at = jet.landing;
                continue;
            }
            let point = Vec3::new(points[at].x, profile.surface[at], points[at].y);
            if let Some(previous) = previous {
                time += point.distance(previous) / max(profile.speed[at], SLOWEST);
            }
            path.push(point.to_array());
            times.push(time);
            aloft.push(false);
            nappes.push(at < last && nappes_at[at]);
            across.push(widths[at]);
            previous = Some(point);
            landing_speed = None;
            at += 1;
        }
        let leaps = aloft.iter().any(|&flying| flying);
        let drop = path
            .first()
            .zip(path.last())
            .map_or(0.0, |(top, bottom)| top[1] - bottom[1]);
        if path.len() >= 2 && (leaps || drop >= FALL_HEIGHT) {
            falls.push(Fall {
                top: path[0],
                bottom: path[path.len() - 1],
                path,
                times,
                aloft,
                nappes,
                widths: across,
                sky: Vec::new(),
                width: widths[first],
                area: areas[first],
                speed: landing_speed.unwrap_or(profile.speed[last]),
                discharge: discharges[first],
            });
            sheeted[first..=last].fill(true);
        }
        first = last + 1;
    }
    (falls, sheeted)
}

/// The least each point's face must fall for a river's water to leap off
/// it, carrying `discharges` over `widths`: its critical depth twice over,
/// and at least LEAP_LEAST meters.
fn leaping_heights(discharges: &[f32], widths: &[f32]) -> Vec<f32> {
    discharges
        .iter()
        .zip(widths)
        .map(|(&discharge, &width)| max(LEAP_DEPTHS * critical_depth(discharge, width), LEAP_LEAST))
        .collect()
}

/// The levels a river's channel is cut to where its water stands at
/// `surface`, its points `along` meters from its first: that water, and
/// beneath each of `jets` the line straight from the lip's water to where
/// the jet lands.
fn cut_beneath(surface: &[f32], jets: &[Jet], along: &[f32]) -> Vec<f32> {
    let mut cut = surface.to_vec();
    for jet in jets {
        let (high, low) = (surface[jet.lip], surface[jet.landing]);
        let run = max(along[jet.landing] - along[jet.lip], 1.0e-3);
        for (index, level) in cut
            .iter_mut()
            .enumerate()
            .take(jet.landing)
            .skip(jet.lip + 1)
        {
            *level = high + (low - high) * (along[index] - along[jet.lip]) / run;
        }
    }
    cut
}

/// The meters across a river half `half` wide.
fn widths(half: &[f32]) -> Vec<f32> {
    half.iter().map(|&half| max(2.0 * half, 0.2)).collect()
}

/// The half widths of river `line`'s channel, narrower as its smooth fit
/// `plan` steepens and wider over the pockets its steps scour, and the
/// steps its highest levels `targets` fall in, `along` meters from its
/// first point (streams step none).
fn steps_of(
    line: &RiverLine,
    plan: &Plan,
    along: &[f32],
    targets: &mut [f32],
) -> (Vec<f32>, Steps) {
    let half: Vec<f32> = line
        .half
        .iter()
        .zip(&plan.slopes)
        .map(|(&half, &slope)| half * narrowing(slope))
        .collect();
    let steps = stepped(
        targets,
        plan,
        (&line.points, along, &widths(&half)),
        |index| line.depth[index] >= MIN_DEPTH,
    );
    let half = half
        .iter()
        .zip(&steps.pocket)
        .map(|(&half, &pocket)| half * (1.0 + POCKET_WIDENING * min(pocket, 1.0)))
        .collect();
    (half, steps)
}

/// `reach`'s river as `flow` runs it, and the falls where it leaves its bed.
pub(crate) fn river(reach: &Reach<'_>, flow: &Flow) -> (River, Vec<Fall>) {
    let line = reach.line;
    let count = line.points.len();
    let along = distances(&line.points);
    let stream = |index: usize| line.depth[index] < MIN_DEPTH;
    let mut targets = tops(reach);
    // Steep channels run narrower, and fall in steps.
    let plan = Plan::of(&targets, &along);
    let (half, steps) = steps_of(line, &plan, &along, &mut targets);
    let widths = widths(&half);
    let discharges: Vec<f32> = line
        .area
        .iter()
        .map(|&area| max(area * flow.yield_rate, TRICKLE))
        .collect();
    let least = leaping_heights(&discharges, &widths);
    let mut profile = Profile {
        surface: vec![f32::NAN; count],
        depth: vec![0.0; count],
        speed: vec![0.0; count],
        discharge: discharges.clone(),
    };
    let mut airborne = vec![false; count];
    let mut jets = Vec::new();
    // Stretch by stretch down the river, each fall's jet landing where the
    // next begins.
    let drops: Vec<(usize, usize)> = drops(&targets, &along, &least)
        .into_iter()
        .filter(|&(lip, _)| !stream(lip))
        .collect();
    let mut start = 0;
    let mut source = reach.source;
    for (lip, _) in drops {
        if lip < start + 1 {
            continue;
        }
        stretch(
            start..lip + 1,
            (&targets, &along),
            (&widths, &line.depth, &discharges),
            (source, None),
            (flow, &steps),
            &mut profile,
        );
        // The lip passes its flow at least as fast as its waves.
        let critical = critical_depth(discharges[lip], widths[lip]);
        let speed = max(profile.speed[lip], (GRAVITY * critical).sqrt());
        let Some((landing, path)) = jet(
            lip,
            (&line.points, &along, &targets),
            profile.surface[lip],
            speed,
        ) else {
            continue;
        };
        for flying in &mut airborne[lip..landing] {
            *flying = true;
        }
        let drop = profile.surface[lip] - path[path.len() - 1][1];
        jets.push(Jet {
            lip,
            landing,
            speed,
            path,
            landing_speed: (speed * speed + 2.0 * GRAVITY * max(drop, 0.0)).sqrt(),
        });
        start = landing;
        source = None;
    }
    if start + 1 < count {
        stretch(
            start..count,
            (&targets, &along),
            (&widths, &line.depth, &discharges),
            (source, reach.outlet),
            (flow, &steps),
            &mut profile,
        );
    } else if start < count {
        profile.surface[start] = targets[start];
        profile.depth[start] = SHALLOWEST;
    }
    // Off a step's face high enough, where no jet leaves, the water leaps
    // clear as a nappe.
    let nappes: Vec<bool> = (0..count)
        .map(|index| {
            !stream(index)
                && !airborne[index]
                && steps.face[index]
                    >= max(
                        LEAP_DEPTHS * critical_depth(discharges[index], widths[index]),
                        NAPPE_LEAST,
                    )
        })
        .collect();
    let (falls, sheeted) = cascades(
        &jets,
        (&line.points, &plan.slopes, &nappes),
        &profile,
        (&widths, &discharges, &line.area),
        |index| !stream(index),
    );
    let cut = cut_beneath(&profile.surface, &jets, &along);
    // Points in the air carry no surface; stretches leaving them are falls.
    for index in 0..count {
        if airborne[index] && index > 0 && airborne[index - 1] {
            profile.surface[index] = f32::NAN;
        }
    }
    let foam = foam(&profile, &along, &airborne, &cut);
    let fill = fills(&profile.depth, &line.depth, &along);
    let river = River {
        points: line.points.iter().map(|point| point.to_array()).collect(),
        surface: profile.surface,
        cut,
        half,
        channel: line.depth.clone(),
        fill,
        depth: profile.depth,
        speed: profile.speed,
        discharge: profile.discharge,
        foam,
        falls: airborne,
        sheeted,
        nappes,
        head: line.head && reach.source.is_none(),
    };
    (river, falls)
}

/// The share of its channel's half width a river's water spans at each
/// point, `depth` deep (at its middle) in a channel `channel` deep, `along`
/// meters from its first: as much as its flow fills of the channel, but at
/// least FILL_LEAST and no more than leaves its banks FREEBOARD over it
/// (the bound that wins where water too shallow for both runs), then
/// averaged over FILL_SMOOTHING meters either side within each point's
/// bounds, never so narrow that its banks stand higher over its water than
/// its channel's own, or FREEBOARD over water deeper than its channel (a
/// pool scoured under a step). Its channel is shaped as deep as that share
/// of its width needs (`Rivers::cap`).
fn fills(depth: &[f32], channel: &[f32], along: &[f32]) -> Vec<f32> {
    let count = depth.len();
    let bounds = |index: usize| {
        let depth = max(depth[index], SHALLOWEST);
        let most = (depth / (depth + FREEBOARD)).sqrt();
        let banked = (depth / max(max(channel[index], SHALLOWEST), depth + FREEBOARD)).sqrt();
        (min(max(FILL_LEAST, banked), most), most)
    };
    let own: Vec<f32> = (0..count)
        .map(|index| {
            let (least, most) = bounds(index);
            let share = (max(depth[index], SHALLOWEST) / max(channel[index], SHALLOWEST)).sqrt();
            min(max(share, least), most)
        })
        .collect();
    let (mut first, mut last) = (0, 0);
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        while along[index] - along[first] > FILL_SMOOTHING {
            first += 1;
        }
        while last + 1 < count && along[last + 1] - along[index] <= FILL_SMOOTHING {
            last += 1;
        }
        let sum: f32 = own[first..=last].iter().sum();
        let (least, most) = bounds(index);
        out.push(min(max(sum / (last - first + 1) as f32, least), most));
    }
    out
}

/// How much foam rides each point of `profile`, its bed cut to `levels`:
/// rapids whiten water running faster than its waves, a hydraulic jump
/// where it slows back, water skimming down steep rock where it takes in
/// air, and water landing from a fall, each carried on downstream at the
/// river's speed and thinning over [`FOAM_SECONDS`].
fn foam(profile: &Profile, along: &[f32], airborne: &[bool], levels: &[f32]) -> Vec<f32> {
    foam_along(&profile.depth, &profile.speed, (along, levels), airborne)
}

/// Foam `entering` `river` at its start from the rivers ending there,
/// carried down it as [`foam_along`] carries its own, thinning over
/// [`FOAM_SECONDS`].
pub(crate) fn foam_entering(river: &mut River, entering: f32) {
    let mut carried = entering;
    for index in 0..river.foam.len() {
        if index > 0 {
            let run = Vec2::from(river.points[index]).distance(Vec2::from(river.points[index - 1]));
            carried *= libm::expf(-run / max(river.speed[index - 1], 0.05) / FOAM_SECONDS);
        }
        river.foam[index] = max(river.foam[index], carried);
    }
}

/// [`foam`] of water `depth` deep running at `speed` at each point, its bed
/// cut to `levels` there.
pub(crate) fn foam_along(
    depth: &[f32],
    speed: &[f32],
    (along, levels): (&[f32], &[f32]),
    airborne: &[bool],
) -> Vec<f32> {
    let count = along.len();
    let froude = |index: usize| {
        let depth = max(depth[index], SHALLOWEST);
        speed[index] / (GRAVITY * depth).sqrt()
    };
    // How fast the bed falls at a point, across its neighbors.
    let slope = |index: usize| {
        let (before, after) = (index.saturating_sub(1), (index + 1).min(count - 1));
        max(levels[before] - levels[after], 0.0) / max(along[after] - along[before], 1.0e-3)
    };
    let mut foam = vec![0.0_f32; count];
    let mut carried = 0.0_f32;
    // Meters the bed has run steep down to the point.
    let mut steep = 0.0_f32;
    for index in 0..count {
        let falls = slope(index);
        steep = if falls >= STEEP_AIR && index > 0 {
            steep + along[index] - along[index - 1]
        } else {
            0.0
        };
        if airborne[index] && index > 0 && airborne[index - 1] {
            carried = 1.0;
            continue;
        }
        let number = froude(index);
        let rapids = min(
            max((number - RAPIDS[0]) / (RAPIDS[1] - RAPIDS[0]), 0.0),
            1.0,
        ) * RAPIDS_WHITE;
        let jump = if index > 0 && froude(index - 1) > 1.0 && number < 1.0 {
            min(froude(index - 1) - 1.0, 1.0) * RAPIDS_WHITE
        } else {
            0.0
        };
        // Past its inception point steep water holds about 0.9 sin of the
        // bed's angle of air.
        let air = 0.9 * falls / (1.0 + falls * falls).sqrt();
        let inception = max(
            INCEPTION_DEPTHS * max(depth[index], SHALLOWEST),
            INCEPTION_LEAST,
        );
        let whitened = min(max((air - AIRED[0]) / (AIRED[1] - AIRED[0]), 0.0), 1.0);
        let aired = whitened * min(steep / inception, 1.0) * RAPIDS_WHITE;
        // Foam rides the current, and lands with the jet that carried it.
        if index > 0 && !airborne[index - 1] {
            let run = along[index] - along[index - 1];
            let seconds = run / max(speed[index - 1], 0.05);
            carried *= libm::expf(-seconds / FOAM_SECONDS);
        }
        carried = max(carried, max(max(rapids, jump), aired));
        foam[index] = carried;
    }
    foam
}
