//! Rivers' channels and their water. Every point of a river carries its
//! bankfull flood, given or what its drained area yields, which fills its
//! channel to its banks: the flood sizes what was not given of its channel
//! and the slope of its reach shapes its bed (`bed.rs`). Its banks stand
//! along the river's long profile: the ground along its line, never rising
//! downstream, so it cuts through the rims of hollows shallower than a lake
//! rather than pooling behind them, and falling at least the gentlest slope
//! a channel is taken to fall, from the level of the lake it leaves or below
//! the banks of the rivers meeting where it begins; and a share of its
//! depth below the lower ground either side of it, so where it runs across
//! a hillside its flood stays in it. Where an imported heightmap resolves
//! its channel, the ground it is cut into is the finest heights' lowest
//! across it, and where those heights already hold the channel (its banks
//! stand over that ground by the depth its flood needs, and its water over
//! it) its flood stands its depth over it and nothing is cut, nor any step
//! or knob carved. Its bed lies its depth below, in the forms its reach
//! takes: pools and riffles where it runs gently, a plane bed steeper, steps
//! with pools behind their crests steeper still, and a cascade of steps and
//! boulders steepest. Its water runs at the flow it carries most of the
//! time, a `FLOODS`th of its floods', standing in its channel as deep as
//! Manning's law has that flow stand in it, and level behind a step's crest
//! where a step holds it back. Where its water drops steeper than water
//! clings to, by its critical depth twice over or more, the river falls:
//! off the drop's lip it leaves its bed as a jet on the parabola its speed
//! and gravity give where that clears the rock, landing where the jet meets
//! the water below, its bed running straight beneath; and the steep stretch
//! holding the drop falls as one cascade, its water clinging to the rock
//! and leaping off its ledges.
//!
//! Every step uses only IEEE arithmetic, `libm`'s functions and the
//! engine's own cosine, so the same bake gives the same bits natively and
//! in the browser.

use engine_math::{clamp, max, min};
use glam::{Vec2, Vec3};

use crate::bed::{self, Bankfull, Crest, Forms, Knob, Reach, Shares};
use crate::hydrology::Fall;
use crate::rivers::{Given, River, RiverLine};

/// Gravity water falls and waves run under (meters a second squared).
pub const GRAVITY: f32 = 9.81;
/// The shallowest water a river's flow keeps (meters).
pub(crate) const SHALLOWEST: f32 = 0.03;
/// The least discharge a river carries (cubic meters a second).
pub(crate) const TRICKLE: f32 = 1.0e-3;
/// The gentlest a channel is taken to fall (meters a meter), so its water
/// runs across flats and its bed's grains and steps stay finite there.
const FLATTEST: f32 = 2.0e-4;
/// Channel widths either side of a point over which its reach's slope is
/// read: ten in all, as a reach's form holds over many widths (Montgomery
/// and Buffington 1997).
const REACH_WIDTHS: f32 = 5.0;
/// How many times the flow a river carries most of the time its floods
/// carry, which fill its channel: a channel full to its banks carries about
/// three times a river's mean flow.
pub(crate) const FLOODS: f32 = 3.0;
/// The share of its depth a channel's banks stand below the lower ground
/// either side of it at least, so the flood filling it stays in it, and the
/// meters past its edge that ground is read at.
const HELD_DEPTH: f32 = 0.25;
pub(crate) const BESIDE: f32 = 1.0;
/// Ferguson's (2007) law of flow over a bed of grains, its speed U against
/// its shear velocity u* where its hydraulic radius stands R over the bed's
/// D84: U/u* = a1 a2 (R/D84) / (a1² + a2² (R/D84)^5/3)^1/2, a1 for flow
/// deep over the grains and a2 for flow shallow among them.
const FERGUSON: [f32; 2] = [6.5, 2.5];
/// The steepest drop (rise over run) a river's water clings to its bed
/// down: off steeper faces at least LEAP_DEPTHS critical depths and
/// LEAP_LEAST meters high it leaps clear as a jet (Chanson: nappe flow
/// holds below about 0.49 critical depths a step's height at 45 degrees).
const CLING_SLOPE: f32 = 1.2;
const LEAP_DEPTHS: f32 = 2.0;
const LEAP_LEAST: f32 = 1.5;
/// A fall's jet sampled at least every this many meters of its run and of
/// its drop, so a plunge's curve bends smoothly.
const JET_STEP: f32 = 0.5;
/// Stretches whose bed falls CASCADE_SLOPE or more a meter about a drop
/// fall with it as one cascade, white water clinging to the rock; gentler
/// stretches no more than BRIDGE widths long between them are the
/// cascade's pools, which it runs on through.
const CASCADE_SLOPE: f32 = 0.45;
const BRIDGE: f32 = 2.0;
/// The slowest water runs down a cascade (meters a second), so its seconds
/// there stay finite.
const SLOWEST: f32 = 0.5;

/// How a bake's rivers flow: the discharge each square meter drained brings
/// them in flood, full to their banks (cubic meters a second), and the
/// meters between the samples of the terrain's finest level, which a step's
/// spacing and a knob's width must span to be carved.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Flow {
    pub(crate) runoff: f32,
    pub(crate) finest: f32,
}

/// A river's line with what holds it: the ground it is cut into
/// (terrain-local meters at terrain-local x and z, read between the samples
/// of its bake's grid), what the finest heights give across its channel at
/// a point running a way (a unit vector) half a width (meters) either side,
/// where imported elevation resolves it, and the level it starts at where
/// it leaves a lake (the lake's) or a confluence (the lowest banks of the
/// rivers ending there).
pub(crate) struct Course<'a> {
    pub(crate) line: &'a RiverLine,
    pub(crate) ground: &'a dyn Fn(Vec2) -> f32,
    pub(crate) across: &'a dyn Fn(Vec2, Vec2, f32) -> Option<Across>,
    pub(crate) start: Option<f32>,
}

/// What the finest heights give across a river's channel at a point, where
/// an imported heightmap gives the ground there fine enough to resolve it:
/// how far it does (0 where the channel spans one of its samples, 1 from
/// [`bed::NARROWEST`] of them on, in proportion between), the lowest ground
/// across the channel, the lower of the ground at its two edges, and the
/// lower of the ground [`BESIDE`] meters past them (terrain-local meters).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Across {
    pub(crate) share: f32,
    pub(crate) lowest: f32,
    pub(crate) banks: f32,
    pub(crate) beside: f32,
}

/// The slope friction takes from water `depth` deep at the middle of a
/// parabolic channel `width` meters across at its surface carrying
/// `discharge` over grains whose D84 is `d84` meters, by Ferguson's law
/// ([`FERGUSON`]): its section, two thirds of its width by its depth, with
/// a hydraulic radius of about two thirds of its depth.
pub(crate) fn friction_slope(discharge: f32, width: f32, depth: f32, d84: f32) -> f32 {
    let radius = 2.0 / 3.0 * depth;
    let speed = discharge / (width * radius);
    let relative = radius / d84;
    let [deep, shallow] = FERGUSON.map(|coefficient| coefficient * coefficient);
    // The square of the speed over the shear velocity, (g R S)^1/2.
    let ratio =
        deep * shallow * relative * relative / (deep + shallow * libm::powf(relative, 5.0 / 3.0));
    speed * speed / (GRAVITY * radius * ratio)
}

/// The depth (meters) at which `discharge` crosses `width` meters as fast as
/// its waves: (q^2 / g)^1/3.
fn critical_depth(discharge: f32, width: f32) -> f32 {
    let per_meter = discharge / width;
    libm::cbrtf(per_meter * per_meter / GRAVITY)
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

/// `values` at a river's points (two or more), `along` meters along its
/// line from the first, at `place` meters along it: between the points
/// either side, the first's or the last's beyond its ends.
pub(crate) fn along_line(values: &[f32], along: &[f32], place: f32) -> f32 {
    let after = along
        .partition_point(|&at| at <= place)
        .clamp(1, along.len() - 1);
    let before = after - 1;
    let run = along[after] - along[before];
    let share = if run > 0.0 {
        clamp((place - along[before]) / run, 0.0, 1.0)
    } else {
        0.0
    };
    values[before] + (values[after] - values[before]) * share
}

/// The long profile of a river whose ground along its middle is `ground`
/// (read between the samples of its grid) at
/// points `along` meters from its first: from `start` where it starts at a
/// level of its own, falling downstream at least [`FLATTEST`], the
/// gentlest slope a channel is taken to fall, so its water runs across
/// flats as its channel carries it.
fn profile(ground: &[f32], along: &[f32], start: Option<f32>) -> Vec<f32> {
    let mut banks: Vec<f32> = Vec::with_capacity(ground.len());
    for (index, &height) in ground.iter().enumerate() {
        let level = match banks.last() {
            Some(&above) => min(height, above - FLATTEST * (along[index] - along[index - 1])),
            None => start.unwrap_or(height),
        };
        banks.push(level);
    }
    banks
}

/// The slope of `banks` (meters a meter, downstream) at each of its points,
/// `along` meters from the first, as channels `widths` across read it over
/// their reach: across REACH_WIDTHS widths either side and at least the
/// points beside it, never gentler than FLATTEST.
fn slopes(banks: &[f32], along: &[f32], widths: &[f32]) -> Vec<f32> {
    let count = banks.len();
    (0..count)
        .map(|index| {
            let reach = REACH_WIDTHS * widths[index];
            let before = along
                .partition_point(|&at| at < along[index] - reach)
                .min(index.saturating_sub(1));
            let after = along
                .partition_point(|&at| at <= along[index] + reach)
                .saturating_sub(1)
                .max((index + 1).min(count - 1));
            let run = max(along[after] - along[before], 1.0e-3);
            max((banks[before] - banks[after]) / run, FLATTEST)
        })
        .collect()
}

/// The lower of the ground either side of each of a river's `points`, read
/// half its channel's width (`widths`) and [`BESIDE`] meters out from its
/// middle, square to its line's tangent, which at each point runs from the
/// point before it to the point after it.
fn beside(points: &[Vec2], widths: &[f32], ground: &dyn Fn(Vec2) -> f32) -> Vec<f32> {
    let last = points.len() - 1;
    (0..=last)
        .map(|index| {
            let tangent = points[(index + 1).min(last)] - points[index.saturating_sub(1)];
            let out = tangent.normalize_or(Vec2::X).perp() * (0.5 * widths[index] + BESIDE);
            min(ground(points[index] + out), ground(points[index] - out))
        })
        .collect()
}

/// `course`'s channel as its floods size it, each square meter it drains
/// bringing `runoff` in flood where its flood is not given, its points
/// `along` meters along its line from the first: the channel carrying them
/// full to its banks, as wide and deep as given, or else as hydraulic
/// geometry stands it and at least [`bed::NARROWEST`] of the grid's samples
/// wide. The ground beneath it is the grid's, or where an imported heightmap
/// resolves its channel the finest heights' lowest across it, in proportion
/// as it does. Its reach's slope, read along the profile of that ground over
/// as many widths as the channel stands wide, sets its reach's form; its
/// banks then stand along the profile of that ground, held [`HELD_DEPTH`] of
/// the channel's depth below the lower ground either side of it
/// ([`beside`]), so its flood stays in it; but where the finest heights
/// already hold the channel (their ground at its edges stands over their
/// lowest across it by its depth, wholly resolved), its depth over that
/// lowest ground. There nothing is cut where its water, standing as its
/// flow does, stands over that ground.
fn channel(course: &Course<'_>, flow: &Flow, along: &[f32]) -> Bankfull {
    let line = course.line;
    let count = line.points.len();
    // What was given at each point: width, depth and flood, NaN where not.
    let given = |index: usize, value: usize| {
        line.given
            .as_ref()
            .map_or(f32::NAN, |given| given.values[index][value])
    };
    let floods: Vec<f32> = (0..count)
        .map(|index| {
            let flood = given(index, 2);
            if flood.is_finite() {
                flood
            } else {
                line.area[index] * flow.runoff
            }
        })
        .collect();
    let narrowest = bed::NARROWEST * flow.finest;
    let grid: Vec<f32> = line
        .points
        .iter()
        .map(|&point| (course.ground)(point))
        .collect();
    let (widths, depths): (Vec<f32>, Vec<f32>) = (0..count)
        .map(|index| {
            let (width, depth) = bed::bankfull(floods[index]);
            let (given_width, given_depth) = (given(index, 0), given(index, 1));
            (
                if given_width.is_finite() {
                    given_width
                } else {
                    max(width, narrowest)
                },
                if given_depth.is_finite() {
                    given_depth
                } else {
                    depth
                },
            )
        })
        .unzip();
    let across: Vec<Option<Across>> = (0..count)
        .map(|index| {
            let (before, after) = (index.saturating_sub(1), (index + 1).min(count - 1));
            let way = (line.points[after] - line.points[before]).normalize_or(Vec2::X);
            (course.across)(line.points[index], way, 0.5 * widths[index])
        })
        .collect();
    let finest = |values: &[f32], index: usize, of: fn(&Across) -> f32| match &across[index] {
        Some(found) => values[index] + (of(found) - values[index]) * found.share,
        None => values[index],
    };
    let ground: Vec<f32> = (0..count)
        .map(|index| finest(&grid, index, |found| found.lowest))
        .collect();
    let slopes = slopes(&profile(&ground, along, course.start), along, &widths);
    let shares: Vec<Shares> = slopes.iter().map(|&slope| Shares::of(slope)).collect();
    let sides = beside(&line.points, &widths, course.ground);
    let holds: Vec<bool> = (0..count)
        .map(|index| {
            across[index].is_some_and(|found| {
                found.share >= 1.0 && found.banks - found.lowest >= depths[index]
            })
        })
        .collect();
    let tops: Vec<f32> = (0..count)
        .map(|index| match across[index] {
            Some(found) if holds[index] => found.lowest + depths[index],
            _ => min(
                ground[index],
                finest(&sides, index, |found| found.beside) - HELD_DEPTH * depths[index],
            ),
        })
        .collect();
    let banks = profile(&tops, along, course.start);
    // Where its water, standing as the flow it carries most of the time
    // does (the share of its depth [`river`] stands it), stands over the
    // heights' lowest ground, they hold it.
    let share = libm::powf(1.0 / FLOODS, 6.0 / 13.0);
    let cut = (0..count)
        .map(|index| {
            !(holds[index]
                && across[index].is_some_and(|found| {
                    banks[index] - (1.0 - share) * depths[index] > found.lowest
                }))
        })
        .collect();
    Bankfull {
        floods,
        slopes,
        shares,
        widths,
        depths,
        banks,
        cut,
    }
}

/// The stretches of `levels` (the water's at each point, `along` meters
/// from the first) dropping steeper than CLING_SLOPE by at least `least` at
/// their first point: each its lip and its foot.
fn drops(levels: &[f32], along: &[f32], least: &[f32]) -> Vec<(usize, usize)> {
    let steep = |index: usize| {
        levels[index] - levels[index + 1] > CLING_SLOPE * (along[index + 1] - along[index])
    };
    let mut found = Vec::new();
    let mut index = 0;
    while index + 1 < levels.len() {
        if !steep(index) {
            index += 1;
            continue;
        }
        let lip = index;
        while index + 1 < levels.len() && steep(index) {
            index += 1;
        }
        if levels[lip] - levels[index] >= least[lip] {
            found.push((lip, index));
        }
    }
    found
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

/// A jet as [`jets`] finds it: its lip and the first point after where it
/// lands, the speed it leaves at, its path and the speed it lands at.
struct Jet {
    lip: usize,
    landing: usize,
    speed: f32,
    path: Vec<[f32; 3]>,
    landing_speed: f32,
}

/// The jets of a river at `points`, `along` meters from its first, its water
/// standing at `levels` and running at `speeds`, carrying `discharges` over
/// `widths`: off the lip of each of its `drops` beyond where the jet before
/// lands, the water leaving at least as fast as its waves, where it flies
/// clear of the drop's ground.
fn jets(
    drops: &[(usize, usize)],
    (points, along): (&[Vec2], &[f32]),
    (levels, speeds): (&[f32], &[f32]),
    (discharges, widths): (&[f32], &[f32]),
) -> Vec<Jet> {
    let mut found: Vec<Jet> = Vec::new();
    for &(lip, _) in drops {
        if found.last().is_some_and(|jet| lip <= jet.landing) {
            continue;
        }
        let critical = critical_depth(discharges[lip], widths[lip]);
        let speed = max(speeds[lip], (GRAVITY * critical).sqrt());
        let Some((landing, path)) = jet(lip, (points, along, levels), levels[lip], speed) else {
            continue;
        };
        let drop = levels[lip] - path[path.len() - 1][1];
        found.push(Jet {
            lip,
            landing,
            speed,
            path,
            landing_speed: (speed * speed + 2.0 * GRAVITY * max(drop, 0.0)).sqrt(),
        });
    }
    found
}

/// The falls of a river at `points`, `along` meters from its first, its
/// water standing at `surface` and running at `speeds`, dropping sharply
/// down its `drops` and leaping off their lips as `jets` do: each stretch
/// whose bed falls CASCADE_SLOPE or more a meter (`slopes`) or drops,
/// through its jets and the pools between, falls as one where it holds a
/// drop: its water's path down it, clinging to the rock on its water and
/// through the air on each jet, and the seconds it takes; and the points
/// their sheets run over. The river is `widths` across, carries
/// `discharges` and drains `areas`.
fn cascades(
    (drops, jets): (&[(usize, usize)], &[Jet]),
    (points, along, slopes): (&[Vec2], &[f32], &[f32]),
    (surface, speeds): (&[f32], &[f32]),
    (widths, discharges, areas): (&[f32], &[f32], &[f32]),
) -> (Vec<Fall>, Vec<bool>) {
    let count = points.len();
    let leaping = |at: usize| jets.iter().find(|jet| jet.lip == at);
    let steep = |at: usize| {
        slopes[at] >= CASCADE_SLOPE || drops.iter().any(|&(lip, foot)| at >= lip && at <= foot)
    };
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
        let (mut path, mut times, mut aloft, mut across) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut time, mut previous, mut landing_speed) = (0.0, None, None);
        let mut at = first;
        while at <= last {
            if let Some(jet) = leaping(at) {
                // Through the air, as long as its run at the speed it
                // left at.
                let lip = Vec3::from_array(jet.path[0]);
                if let Some(previous) = previous {
                    time += lip.distance(previous) / max(speeds[at], SLOWEST);
                }
                let leaving = time;
                for point in &jet.path {
                    let run = Vec2::new(point[0] - lip.x, point[2] - lip.z).length();
                    time = leaving + run / jet.speed;
                    path.push(*point);
                    times.push(time);
                    aloft.push(true);
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
            let point = Vec3::new(points[at].x, surface[at], points[at].y);
            if let Some(previous) = previous {
                time += point.distance(previous) / max(speeds[at], SLOWEST);
            }
            path.push(point.to_array());
            times.push(time);
            aloft.push(false);
            across.push(widths[at]);
            previous = Some(point);
            landing_speed = None;
            at += 1;
        }
        // Steep rock without a drop leaves the water running as the
        // river's own.
        let dropping = drops.iter().any(|&(lip, _)| lip >= first && lip <= last);
        if path.len() >= 2 && dropping {
            falls.push(Fall {
                top: path[0],
                bottom: path[path.len() - 1],
                path,
                times,
                aloft,
                widths: across,
                sky: Vec::new(),
                width: widths[first],
                area: areas[first],
                speed: landing_speed.unwrap_or(speeds[last]),
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

/// A river's `values` at its points `along` meters from its first (its
/// bed's levels, its channel's depths), beneath each of `jets` straight
/// from the lip's to the landing's, so no ground stands between the pool a
/// fall leaves and the one it lands in.
fn cut_beneath(values: &mut [f32], jets: &[Jet], along: &[f32]) {
    for jet in jets {
        let (high, low) = (values[jet.lip], values[jet.landing]);
        let run = max(along[jet.landing] - along[jet.lip], 1.0e-3);
        for index in jet.lip + 1..jet.landing {
            values[index] = high + (low - high) * (along[index] - along[jet.lip]) / run;
        }
    }
}

/// `forms` where a river's channel is `cut` into the ground at each point:
/// where the heights hold it instead, its bed is theirs, so no step or knob
/// is carved there and its water stands behind no step's crest.
fn uncarved(mut forms: Forms, cut: &[bool]) -> Forms {
    for (index, &carved) in cut.iter().enumerate() {
        if !carved {
            forms.dips[index] = 0.0;
            forms.held[index] = f32::NAN;
            forms.stepped[index] = false;
        }
    }
    for (crest, pair) in forms.crests.iter_mut().zip(cut.windows(2)) {
        if !(pair[0] && pair[1]) {
            *crest = None;
        }
    }
    forms
        .knobs
        .retain(|knob| cut[knob.segment as usize] && cut[knob.segment as usize + 1]);
    forms
}

/// What a river's bed is at each of its points: its reach, its median grain
/// and D84 (meters), the power of its bankfull flood (watts a square meter
/// of bed) and whether steps or knobs are carved there; at each point the
/// crest of a step on the segment to the next, if one stands there; and the
/// knobs on it, segment by segment.
struct Beds {
    reach: Vec<Reach>,
    d50: Vec<f32>,
    d84: Vec<f32>,
    power: Vec<f32>,
    carved: Vec<bool>,
    crest: Vec<Option<Crest>>,
    knobs: Vec<Knob>,
}

/// The bed of a river in the channel `bankfull` sizes, shaped into `forms`,
/// where `jets` leap over it (from each point to the next where `airborne`)
/// and falls' sheets run over its points `sheeted`: beneath a jet its bed
/// runs straight, with neither steps nor knobs, and wherever its water
/// falls its reach is falls.
fn beds(
    bankfull: &Bankfull,
    forms: Forms,
    (jets, airborne, sheeted): (&[Jet], &[bool], &[bool]),
) -> Beds {
    let Forms {
        stepped: mut carved,
        crests: mut crest,
        mut knobs,
        ..
    } = forms;
    for jet in jets {
        crest[jet.lip..jet.landing].fill(None);
        carved[jet.lip + 1..jet.landing].fill(false);
    }
    knobs.retain(|knob| !airborne[knob.segment as usize]);
    for knob in &knobs {
        let segment = knob.segment as usize;
        carved[segment] = true;
        carved[segment + 1] = true;
    }
    let count = carved.len();
    let (d50, d84) = (0..count)
        .map(|index| {
            bed::grains(
                bankfull.depths[index],
                bankfull.slopes[index],
                &bankfull.shares[index],
            )
        })
        .unzip();
    Beds {
        reach: (0..count)
            .map(|index| {
                if sheeted[index] || airborne[index] {
                    Reach::Falls
                } else {
                    Reach::of(bankfull.slopes[index])
                }
            })
            .collect(),
        d50,
        d84,
        power: (0..count)
            .map(|index| {
                bed::power(
                    bankfull.floods[index],
                    bankfull.slopes[index],
                    bankfull.widths[index],
                )
            })
            .collect(),
        carved,
        crest,
        knobs,
    }
}

/// `course`'s river as `flow` sizes and runs it, and the falls where it
/// leaves its bed.
pub(crate) fn river(course: &Course<'_>, flow: &Flow) -> (River, Vec<Fall>) {
    let line = course.line;
    let count = line.points.len();
    let along = distances(&line.points);
    let bankfull = channel(course, flow, &along);
    let Bankfull {
        floods,
        slopes,
        widths,
        depths,
        banks,
        ..
    } = &bankfull;
    let forms = uncarved(
        bed::forms((&line.points, &along), &bankfull, flow.finest),
        &bankfull.cut,
    );
    // Its bed lies its channel's depth below its banks, deeper in its
    // forms.
    let mut bed: Vec<f32> = (0..count)
        .map(|index| banks[index] - depths[index] - forms.pools[index] - forms.dips[index])
        .collect();
    let mut depth: Vec<f32> = banks
        .iter()
        .zip(&bed)
        .map(|(&banks, &bed)| banks - bed)
        .collect();
    // The water it carries most of the time. A parabolic channel carries
    // as the 13/6th power of how deep its water stands, so a FLOODSth of its
    // floods stands (1 / FLOODS)^(6/13) of its depth deep over its bed's
    // treads; behind a step's crest, level with the water over the crest.
    let discharges: Vec<f32> = floods.iter().map(|&flood| flood / FLOODS).collect();
    let share = libm::powf(1.0 / FLOODS, 6.0 / 13.0);
    let over = |place: f32| {
        along_line(banks, &along, place) - (1.0 - share) * along_line(depths, &along, place)
    };
    let mut surface: Vec<f32> = (0..count)
        .map(|index| {
            let own = banks[index] - (1.0 - share) * depths[index] - forms.dips[index];
            let crest = forms.held[index];
            if crest.is_finite() {
                max(own, over(crest))
            } else {
                own
            }
        })
        .collect();
    // As fast as its discharge crosses the part of its channel's parabola
    // its water fills.
    let speed_of = |index: usize, surface: f32, bed: f32, depth: f32| {
        let water = max(surface - bed, SHALLOWEST);
        let across = widths[index] * (water / max(depth, water)).sqrt();
        discharges[index] / (2.0 / 3.0 * across * water)
    };
    let mut speed: Vec<f32> = (0..count)
        .map(|index| speed_of(index, surface[index], bed[index], depth[index]))
        .collect();
    let drops = drops(&surface, &along, &leaping_heights(&discharges, widths));
    let jets = jets(
        &drops,
        (&line.points, &along),
        (&surface, &speed),
        (&discharges, widths),
    );
    // Beneath each jet its channel runs straight from the lip to where it
    // lands, and its water flies.
    cut_beneath(&mut bed, &jets, &along);
    cut_beneath(&mut depth, &jets, &along);
    let mut airborne = vec![false; count];
    for jet in &jets {
        airborne[jet.lip..jet.landing].fill(true);
        for index in jet.lip + 1..jet.landing {
            speed[index] = speed_of(index, bed[index] + depth[index], bed[index], depth[index]);
            surface[index] = f32::NAN;
        }
    }
    let (falls, sheeted) = cascades(
        (&drops, &jets),
        (&line.points, &along, slopes),
        (&surface, &speed),
        (widths, &discharges, &line.area),
    );
    let beds = beds(&bankfull, forms, (&jets, &airborne, &sheeted));
    let given = line.given.as_ref().map_or_else(Given::default, |given| {
        let first = given.values.first().copied().unwrap_or([f32::NAN; 3]);
        Given {
            source: Some(given.source),
            width: first[0].is_finite(),
            depth: first[1].is_finite(),
            flood: first[2].is_finite(),
        }
    });
    let river = River {
        points: line.points.iter().map(|point| point.to_array()).collect(),
        discharge: discharges,
        half: widths.iter().map(|&width| 0.5 * width).collect(),
        depth,
        bed,
        surface,
        speed,
        falls: airborne,
        sheeted,
        reach: beds.reach,
        d50: beds.d50,
        d84: beds.d84,
        power: beds.power,
        flood: banks.clone(),
        carved: beds.carved,
        cut: bankfull.cut.clone(),
        crest: beds.crest,
        knobs: beds.knobs,
        head: line.head && course.start.is_none(),
        given,
    };
    (river, falls)
}
