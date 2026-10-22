//! Rivers' channels and their water, from one flow law. Every point of a
//! river carries the discharge its drained area yields in flood, and its
//! channel takes its shape from that discharge alone, as floods cut
//! channels: as wide as a set number of meters times its square root, as
//! channels full to their banks are, and as deep at its middle as Manning's
//! formula needs for its parabolic section to carry it full to its banks
//! down the river's slope. Its banks stand along the river's long profile:
//! the ground along its line, never rising downstream, so it cuts through
//! the rims of hollows shallower than a lake rather than pooling behind
//! them, and falling at least the gentlest slope its channel is sized for,
//! from the level of the lake it leaves or below the banks of the rivers
//! meeting where it begins. Its bed lies its depth below, in the forms the
//! profile's slope sets (Montgomery and Buffington 1997): pools and riffles
//! where it runs gently, a plane bed steeper, steps with pools scoured
//! below them steeper still, and a cascade following the profile steepest.
//! Its water runs at the flow it carries most of the time, a [`FLOODS`]th
//! of its floods', standing in its channel as high as that flow needs by
//! the same law, and level behind a step's crest where a step holds it
//! back. Where its water drops steeper than water clings to, by its
//! critical depth twice over or more, the river falls: off the drop's lip
//! it leaves its bed as a jet on the parabola its speed and gravity give
//! where that clears the rock, landing where the jet meets the water below,
//! its bed running straight beneath; and the steep stretch holding the drop
//! falls as one cascade, its water clinging to the rock and leaping off its
//! ledges.
//!
//! Every step uses only IEEE arithmetic, `libm`'s functions and the
//! engine's own cosine, so the same bake gives the same bits natively and
//! in the browser.

use engine_math::{cos, max, min};
use glam::{Vec2, Vec3};

use crate::hydrology::Fall;
use crate::rivers::{River, RiverLine};

/// Gravity water falls and waves run under (meters a second squared).
pub const GRAVITY: f32 = 9.81;
/// The shallowest water a river's flow keeps (meters).
pub(crate) const SHALLOWEST: f32 = 0.03;
/// The least discharge a river carries (cubic meters a second).
pub(crate) const TRICKLE: f32 = 1.0e-3;
/// The gentlest a channel is taken to fall (meters a meter), so its depth
/// stays finite on flats.
const FLATTEST: f32 = 2.0e-4;
/// Channel widths either side of a point over which a river's slope is
/// read.
const PROFILE_WIDTHS: f32 = 2.0;
/// How many times the flow a river carries most of the time its floods
/// carry, which cut its channel: a channel full to its banks carries about
/// three times a river's mean flow.
pub(crate) const FLOODS: f32 = 3.0;
/// The slopes (meters a meter) at which a bed's forms change (Montgomery
/// and Buffington 1997): gentler than the first, pools and riffles; to the
/// second, a plane bed; to the third, steps and pools; steeper, a cascade.
const FORMS: [f32; 3] = [0.015, 0.03, 0.065];
/// Channel widths from one pool to the next, and the share of the
/// channel's depth pools stand deeper and riffles shallower on a flat (half
/// as deep again in a pool as over a riffle), the less the steeper the bed,
/// to none at a plane bed.
const POOLS_APART: f32 = 6.0;
const POOL_DEPTH: f32 = 0.2;
/// Channel widths from one step to the next where steps begin and where
/// cascades begin, and the share of a step's height the pool below it is
/// scoured.
const STEPS_APART: [f32; 2] = [3.0, 1.5];
const SCOUR: f32 = 0.25;
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
/// them in flood, full to their banks (cubic meters a second), the meters
/// wide a channel carrying a cubic meter a second stands, and Manning's
/// roughness of their beds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Flow {
    pub(crate) runoff: f32,
    pub(crate) width: f32,
    pub(crate) roughness: f32,
}

/// A river's line with what holds it: the ground at each of its points (at
/// its middle, terrain-local meters), and the
/// level it starts at where it leaves a lake (the lake's) or a confluence
/// (the lowest banks of the rivers ending there).
pub(crate) struct Reach<'a> {
    pub(crate) line: &'a RiverLine,
    pub(crate) ground: &'a [f32],
    pub(crate) start: Option<f32>,
}

/// The depth at its middle (meters) of a parabolic channel `width` meters
/// across that carries `discharge` full to its banks down `slope`,
/// Manning's `roughness`: its section, two thirds of its width by its
/// depth, with a hydraulic radius of about two thirds of its depth, carries
/// Q = A R^2/3 S^1/2 / n.
fn channel_depth(discharge: f32, width: f32, slope: f32, roughness: f32) -> f32 {
    1.5 * libm::powf(discharge * roughness / (width * slope.sqrt()), 0.6)
}

/// The slope friction takes from water `depth` deep at the middle of a
/// parabolic channel `width` meters across at its surface carrying
/// `discharge` ([`channel_depth`] solved for its slope).
pub(crate) fn friction_slope(discharge: f32, width: f32, depth: f32, roughness: f32) -> f32 {
    let r = discharge * roughness / (width * libm::powf(depth / 1.5, 5.0 / 3.0));
    r * r
}

/// `low` to `high` by `share`.
fn between([low, high]: [f32; 2], share: f32) -> f32 {
    low + (high - low) * share
}

/// A number in [0, 1) hashed from `key`.
fn unit_of(key: u32) -> f32 {
    let mut h = key;
    h = (h ^ (h >> 16)).wrapping_mul(0x7feb_352d);
    h = (h ^ (h >> 15)).wrapping_mul(0x846c_a68b);
    ((h ^ (h >> 16)) >> 8) as f32 / 16_777_216.0
}

/// The key of a river's point at `point`: its place, so the same river
/// forms alike in every bake.
fn place_key(point: Vec2) -> u32 {
    point.x.to_bits().wrapping_mul(0x8da6_b343) ^ point.y.to_bits().wrapping_mul(0xd816_3841)
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

/// The long profile of a river whose ground along its middle is `ground`
/// (read between the samples of its grid) at
/// points `along` meters from its first: from `start` where it starts at a
/// level of its own, falling downstream at least [`FLATTEST`], the
/// gentlest slope its channel is sized for, so its water runs across flats
/// as its channel carries it.
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
/// `along` meters from the first, its channel `widths` across: across
/// PROFILE_WIDTHS widths either side and at least the points beside it,
/// never gentler than FLATTEST.
fn slopes(banks: &[f32], along: &[f32], widths: &[f32]) -> Vec<f32> {
    let count = banks.len();
    (0..count)
        .map(|index| {
            let reach = PROFILE_WIDTHS * widths[index];
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

/// A river's banks and bed as the forms its slope sets shape them: along
/// its long profile `banks`, falling `slopes`, at points `along` meters from
/// its first (at `first`), its channel `widths` across and `depths` deep.
/// Where it falls gentler than FORMS[0] its bed deepens into pools
/// POOLS_APART widths apart and shallows over the riffles between them;
/// from FORMS[1] to FORMS[2] it falls in steps STEPS_APART widths apart,
/// each taking the profile's fall to the next, so its water stands level
/// from a step's crest back to the step above, over a pool scoured SCOUR of
/// the step's height deep under that step and shallowing to the crest;
/// elsewhere its bed follows the profile. Forms stand by distance along the
/// river from a phase its first point's place gives, so the same river
/// forms alike in every bake. The level of its banks and water at each
/// point, and its bed.
fn forms(
    banks: &[f32],
    slopes: &[f32],
    (along, first): (&[f32], Vec2),
    (widths, depths): (&[f32], &[f32]),
) -> (Vec<f32>, Vec<f32>) {
    let count = banks.len();
    let on_steps = |index: usize| slopes[index] >= FORMS[1] && slopes[index] < FORMS[2];
    // Meters from one step to the next, closer as the bed steepens.
    let spacing = |index: usize| {
        let steep = min(
            max((slopes[index] - FORMS[1]) / (FORMS[2] - FORMS[1]), 0.0),
            1.0,
        );
        widths[index] * between(STEPS_APART, steep)
    };
    // How many pools and steps lie upstream of each point.
    let phase = unit_of(place_key(first));
    let (mut pool_phase, mut step_phase) = (vec![phase; count], vec![phase; count]);
    for index in 1..count {
        let run = along[index] - along[index - 1];
        pool_phase[index] = pool_phase[index - 1] + run / (POOLS_APART * widths[index - 1]);
        step_phase[index] = step_phase[index - 1] + run / spacing(index - 1);
    }
    // On steps the water stands at the level of the next crest down,
    // which lies on the profile; where steps end, at least as high as the
    // water below, so it never rises downstream.
    let mut level = banks.to_vec();
    let mut crest = f32::NAN;
    for index in (0..count).rev() {
        if index + 1 < count {
            let (here, next) = (step_phase[index], step_phase[index + 1]);
            let ahead = here.ceil();
            if ahead <= next {
                let share = if next > here {
                    (ahead - here) / (next - here)
                } else {
                    0.0
                };
                crest = banks[index] + (banks[index + 1] - banks[index]) * share;
            }
        }
        if on_steps(index) && crest.is_finite() {
            level[index] = crest;
        }
    }
    for index in (0..count.saturating_sub(1)).rev() {
        level[index] = max(level[index], level[index + 1]);
    }
    let bed = (0..count)
        .map(|index| {
            let slope = slopes[index];
            let deep = if slope < FORMS[0] {
                let strength = POOL_DEPTH * (1.0 - slope / FORMS[0]);
                let turn = pool_phase[index] - pool_phase[index].floor();
                depths[index] * (1.0 + strength * cos(std::f32::consts::TAU * turn))
            } else if on_steps(index) {
                // Deepest just below the step above, shallowing to the
                // crest below.
                let since = step_phase[index] - step_phase[index].floor();
                depths[index] + SCOUR * slope * spacing(index) * (1.0 - since)
            } else {
                depths[index]
            };
            level[index] - deep
        })
        .collect();
    (level, bed)
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

/// `reach`'s river as `flow` sizes and runs it, and the falls where it
/// leaves its bed.
pub(crate) fn river(reach: &Reach<'_>, flow: &Flow) -> (River, Vec<Fall>) {
    let line = reach.line;
    let count = line.points.len();
    let along = distances(&line.points);
    // The discharge its drained area yields in flood, and the channel
    // carrying it full to its banks.
    let floods: Vec<f32> = line.area.iter().map(|&area| area * flow.runoff).collect();
    let widths: Vec<f32> = floods
        .iter()
        .map(|&discharge| flow.width * discharge.sqrt())
        .collect();
    let banks = profile(reach.ground, &along, reach.start);
    let slopes = slopes(&banks, &along, &widths);
    let depths: Vec<f32> = (0..count)
        .map(|index| channel_depth(floods[index], widths[index], slopes[index], flow.roughness))
        .collect();
    let (level, mut bed) = forms(
        &banks,
        &slopes,
        (&along, line.points[0]),
        (&widths, &depths),
    );
    let mut depth: Vec<f32> = level.iter().zip(&bed).map(|(&level, &bed)| level - bed).collect();
    // The water it carries most of the time. A parabolic channel carries
    // as the 13/6th power of how deep its water stands, so a FLOODSth of its
    // floods stands (1 / FLOODS)^(6/13) of its depth deep; behind a step's
    // crest, level with the crest.
    let discharges: Vec<f32> = floods.iter().map(|&flood| flood / FLOODS).collect();
    let share = libm::powf(1.0 / FLOODS, 6.0 / 13.0);
    let mut surface: Vec<f32> = (0..count)
        .map(|index| {
            if level[index] < banks[index] {
                level[index]
            } else {
                level[index] - (1.0 - share) * depths[index]
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
    let drops = drops(&surface, &along, &leaping_heights(&discharges, &widths));
    let jets = jets(
        &drops,
        (&line.points, &along),
        (&surface, &speed),
        (&discharges, &widths),
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
        (&line.points, &along, &slopes),
        (&surface, &speed),
        (&widths, &discharges, &line.area),
    );
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
        head: line.head && reach.start.is_none(),
    };
    (river, falls)
}
