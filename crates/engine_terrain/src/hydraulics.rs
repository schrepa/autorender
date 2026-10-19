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
//! Where the ground drops away faster than water leaving the lip can fall,
//! the river leaves its bed as a jet on the parabola its speed and gravity
//! give, and lands in a pool where the jet meets the water below; where it
//! lands in a steep chute another jet leaves soon after, the jets and the
//! chutes between fall as one cascade. Foam forms
//! where water runs faster than its waves (Froude number above one), where
//! it slows back through a hydraulic jump and where it lands from a fall,
//! and rides the current downstream, thinning as it goes.
//!
//! Every step uses only IEEE arithmetic and `libm`'s functions, so the same
//! bake gives the same bits natively and in the browser.

use engine_math::{max, min};
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
/// The steepest drop (rise over run) a river follows down its bed; falls
/// steeper than this over at least [`FALL_HEIGHT`] throw the water clear.
const JET_SLOPE: f32 = 1.0;
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
/// A fall's jet sampled every this many meters along its flight.
const JET_STEP: f32 = 1.0;
/// The most meters of chute between where one jet lands and the next
/// leaves for the two to fall as one cascade, and the least that chute's
/// water falls a meter.
const CASCADE_GAP: f32 = 80.0;
const CASCADE_SLOPE: f32 = 0.35;
/// The slowest water runs down a cascade's chutes (meters a second), so
/// its seconds there stay finite.
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
/// (`along` meters from the first), never rising along them.
fn smoothed(values: &[f32], along: &[f32]) -> Vec<f32> {
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
/// dropping steeper than [`JET_SLOPE`] by [`FALL_HEIGHT`] or more: each its
/// first and last point.
fn drops(targets: &[f32], along: &[f32]) -> Vec<(usize, usize)> {
    let steep = |index: usize| {
        targets[index] - targets[index + 1] > JET_SLOPE * (along[index + 1] - along[index])
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
        if targets[lip] - targets[index] >= FALL_HEIGHT {
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
        let top = smoothed(&level, &along);
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
    flow: &Flow,
    profile: &mut Profile,
) {
    let start = range.start;
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
    let top = smoothed(&level, along);
    let count = top.len();
    let slopes = slopes(&top, along);
    // The water as deep as uniform flow down the top's slope runs, up to
    // the channel's depth, standing that deep over a bed the channel's
    // depth below its top, but never rising downstream: where it would
    // stand higher than the water above it (a riffle opening into a pool),
    // it stands at that water's level, as deep over a bed scoured deeper.
    let mut bed = Vec::with_capacity(count);
    let mut surface: Vec<f32> = Vec::with_capacity(count);
    for index in 0..count {
        let (width, discharge) = (widths[start + index], discharges[start + index]);
        let channel = max(channels[start + index], SHALLOWEST);
        let depth = min(
            max(
                normal_depth(discharge, width, slopes[index], flow.roughness),
                SHALLOWEST,
            ),
            channel,
        );
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
    let steps = max((reach / JET_STEP).ceil(), 1.0) as usize;
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

/// The falls of `jets`, in order down a river along `points`, `along`
/// meters from its first, its water as `profile` runs it: each jet a fall,
/// but jets landing in a chute the next leaves within CASCADE_GAP meters,
/// its water falling CASCADE_SLOPE a meter or more, fall as one cascade,
/// their paths joined by the chutes' water. The river is `widths` across,
/// carries `discharges` and drains `areas`.
fn cascades(
    jets: &[Jet],
    (points, along): (&[Vec2], &[f32]),
    profile: &Profile,
    (widths, discharges, areas): (&[f32], &[f32], &[f32]),
) -> Vec<Fall> {
    let mut falls = Vec::new();
    let mut first = 0;
    while first < jets.len() {
        let mut last = first;
        while let Some(next) = jets.get(last + 1) {
            let landed = jets[last].landing;
            let run = along[next.lip] - along[landed];
            let fallen = profile.surface[landed] - profile.surface[next.lip];
            if !(run <= CASCADE_GAP && fallen >= CASCADE_SLOPE * run) {
                break;
            }
            last += 1;
        }
        let (mut path, mut times, mut aloft) = (Vec::new(), Vec::new(), Vec::new());
        let mut time = 0.0;
        for (index, jet) in jets[first..=last].iter().enumerate() {
            // Through the air, as long as its run at the speed it left at.
            let lip = Vec2::new(jet.path[0][0], jet.path[0][2]);
            let leaving = time;
            for point in &jet.path {
                let run = Vec2::new(point[0], point[2]).distance(lip);
                time = leaving + run / jet.speed;
                path.push(*point);
                times.push(time);
                aloft.push(true);
            }
            // It lands, and runs down the chute to the next lip.
            if let Some(landed) = aloft.last_mut() {
                *landed = false;
            }
            let Some(next) = jets.get(first + index + 1).filter(|_| first + index < last) else {
                continue;
            };
            let mut previous = Vec3::from_array(path[path.len() - 1]);
            for (at, place) in points
                .iter()
                .enumerate()
                .take(next.lip + 1)
                .skip(jet.landing)
            {
                let point = Vec3::new(place.x, profile.surface[at], place.y);
                time += previous.distance(point) / max(profile.speed[at], SLOWEST);
                previous = point;
                if at < next.lip {
                    path.push(point.to_array());
                    times.push(time);
                    aloft.push(false);
                }
            }
        }
        let (top, bottom) = (jets[first].lip, &jets[last]);
        falls.push(Fall {
            top: jets[first].path[0],
            bottom: bottom.path[bottom.path.len() - 1],
            path,
            times,
            aloft,
            width: widths[top],
            area: areas[top],
            speed: bottom.landing_speed,
            discharge: discharges[top],
        });
        first = last + 1;
    }
    falls
}

/// `reach`'s river as `flow` runs it, and the falls where it leaves its bed.
pub(crate) fn river(reach: &Reach<'_>, flow: &Flow) -> (River, Vec<Fall>) {
    let line = reach.line;
    let count = line.points.len();
    let along = distances(&line.points);
    let stream = |index: usize| line.depth[index] < MIN_DEPTH;
    let targets = tops(reach);
    let widths: Vec<f32> = line.half.iter().map(|&half| max(2.0 * half, 0.2)).collect();
    let discharges: Vec<f32> = line
        .area
        .iter()
        .map(|&area| max(area * flow.yield_rate, TRICKLE))
        .collect();
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
    let drops: Vec<(usize, usize)> = drops(&targets, &along)
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
            flow,
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
            flow,
            &mut profile,
        );
    } else if start < count {
        profile.surface[start] = targets[start];
        profile.depth[start] = SHALLOWEST;
    }
    let falls = cascades(
        &jets,
        (&line.points, &along),
        &profile,
        (&widths, &discharges, &line.area),
    );
    // Beneath each jet its channel is cut straight from the lip's water to
    // where the jet lands.
    let mut cut = profile.surface.clone();
    for jet in &jets {
        let (high, low) = (profile.surface[jet.lip], profile.surface[jet.landing]);
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
    // Points in the air carry no surface; stretches leaving them are falls.
    for index in 0..count {
        if airborne[index] && index > 0 && airborne[index - 1] {
            profile.surface[index] = f32::NAN;
        }
    }
    let foam = foam(&profile, &along, &airborne);
    let fill = fills(&profile.depth, &line.depth, &along);
    let river = River {
        points: line.points.iter().map(|point| point.to_array()).collect(),
        surface: profile.surface,
        cut,
        half: line.half.clone(),
        channel: line.depth.clone(),
        fill,
        depth: profile.depth,
        speed: profile.speed,
        discharge: profile.discharge,
        foam,
        falls: airborne,
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
/// bounds. Its channel is shaped as deep as that share of its width needs
/// (`Rivers::cap`).
fn fills(depth: &[f32], channel: &[f32], along: &[f32]) -> Vec<f32> {
    let count = depth.len();
    let bounds = |index: usize| {
        let depth = max(depth[index], SHALLOWEST);
        let most = (depth / (depth + FREEBOARD)).sqrt();
        (min(FILL_LEAST, most), most)
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

/// How much foam rides each point of `profile`: rapids whiten water running
/// faster than its waves, a hydraulic jump where it slows back, and water
/// landing from a fall, each carried on downstream at the river's speed and
/// thinning over [`FOAM_SECONDS`].
fn foam(profile: &Profile, along: &[f32], airborne: &[bool]) -> Vec<f32> {
    foam_along(&profile.depth, &profile.speed, along, airborne)
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

/// [`foam`] of water `depth` deep running at `speed` at each point.
pub(crate) fn foam_along(
    depth: &[f32],
    speed: &[f32],
    along: &[f32],
    airborne: &[bool],
) -> Vec<f32> {
    let count = along.len();
    let froude = |index: usize| {
        let depth = max(depth[index], SHALLOWEST);
        speed[index] / (GRAVITY * depth).sqrt()
    };
    let mut foam = vec![0.0_f32; count];
    let mut carried = 0.0_f32;
    for index in 0..count {
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
        // Foam rides the current, and lands with the jet that carried it.
        if index > 0 && !airborne[index - 1] {
            let run = along[index] - along[index - 1];
            let seconds = run / max(speed[index - 1], 0.05);
            carried *= libm::expf(-seconds / FOAM_SECONDS);
        }
        carried = max(carried, max(rapids, jump));
        foam[index] = carried;
    }
    foam
}
