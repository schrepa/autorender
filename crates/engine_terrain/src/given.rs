//! Rivers someone gave: lines a developer draws over a terrain or an
//! import tool writes from vector data (OpenStreetMap waterways, a national
//! hydrography dataset), each with what its data says of its channel, in a
//! rivers layer the hydrology bake after it runs as its courses before it
//! derives any of its own. The bake follows each given river along the
//! centripetal Catmull-Rom spline through its points (Yuksel, Schaefer and
//! Keyser 2011), resampled every [`RIVER_STEP`] meters, over the stretches
//! its grid holds outside lakes; moves its points across its line toward
//! the lowest ground no further than its data's precision, never over a
//! rise; joins rivers where one ends on another (a confluence), splitting
//! the other there; and marks the grid's samples within each one's
//! channel. Water reaching those samples runs on in the given river, which
//! gathers it, so the bake derives no course for that water, and where a
//! given river ends on open ground its water runs on over the ground from
//! there. Values given are used as given; what nobody gave is derived as
//! for the bake's own courses. Every step uses IEEE arithmetic in a fixed
//! order, so the same rivers bake to the same bits on every host.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use engine_math::collections::HashMap;
use engine_math::{clamp, max, min};
use engine_settings::{ExtraRules, InvalidSetting, settings};
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::bed::{NARROWEST, bankfull};
use crate::hydrology::{Grid, WaterGrid};
use crate::path::distance_to_segment;
use crate::rivers::{FOOTPRINT, GivenLine, MEASURED, RIVER_STEP, RiverLine, smooth_line};

/// Rivers a rivers layer lists at most, and points a given river runs
/// through at most.
pub const MAX_GIVEN_RIVERS: usize = 4096;
pub const MAX_RIVER_POINTS: usize = 16_384;
/// Characters a given river's name holds at most, and the source and the
/// attribution of a rivers layer or an imported heightmap.
pub const MAX_CREDIT: usize = 1024;
/// The widest, deepest and fullest a given channel stands: the widest
/// river, the Amazon, spreads some 40 kilometers across in flood and then
/// carries some 300,000 cubic meters a second; the deepest, the lower Congo,
/// runs over 200 meters deep.
pub const MAX_WIDTH: f32 = 50_000.0;
pub const MAX_DEPTH: f32 = 500.0;
pub const MAX_DISCHARGE: f32 = 1.0e6;
/// The most meters a rivers layer's lines may stand off their rivers' true
/// middles.
pub const MAX_PRECISION: f32 = 1000.0;

settings! {
    extra_rules;
    /// One river given: the line it runs along, from upstream, and what its
    /// data says of its channel. Its width, depth and discharge each hold
    /// one value for the whole river or one for each of its points; what it
    /// leaves out the bake derives, as for its own courses.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct GivenRiver {
        /// Its name, as its data or its author calls it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "schema", schemars(length(max = MAX_CREDIT)))]
        pub name: Option<String> = None => any,
        /// Terrain-local meters along x and z it runs through, from
        /// upstream: it runs along the smooth line through them.
        #[cfg_attr(feature = "schema", schemars(length(min = 2, max = MAX_RIVER_POINTS)))]
        pub points: Vec<[f32; 2]> = Vec::new()
            => range(-crate::Layers::MAX_COORDINATE, crate::Layers::MAX_COORDINATE),
        /// Meters across its channel from bank to bank; omitted, as wide as
        /// downstream hydraulic geometry stands a channel its flood fills.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub width: Vec<f32> = Vec::new() => above_to(0.0, MAX_WIDTH),
        /// Meters its channel's middle lies below its banks; omitted, as
        /// deep as downstream hydraulic geometry stands a channel its flood
        /// fills.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub depth: Vec<f32> = Vec::new() => above_to(0.0, MAX_DEPTH),
        /// Cubic meters a second its bankfull flood carries, which fills its
        /// channel to its banks; omitted, what the area draining into it
        /// brings, as its hydrology layer's discharge says.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub discharge: Vec<f32> = Vec::new() => above_to(0.0, MAX_DISCHARGE),
    }
}

impl ExtraRules for GivenRiver {
    /// It runs through 2 to [`MAX_RIVER_POINTS`] points; its width, depth
    /// and discharge each hold no value, one, or one for each point; its
    /// name holds at most [`MAX_CREDIT`] characters.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if !(2..=MAX_RIVER_POINTS).contains(&self.points.len()) {
            return Err(InvalidSetting::new(
                "points",
                format!("must hold 2 to {MAX_RIVER_POINTS} points"),
            ));
        }
        for (field, values) in [
            ("width", &self.width),
            ("depth", &self.depth),
            ("discharge", &self.discharge),
        ] {
            if !(values.len() <= 1 || values.len() == self.points.len()) {
                return Err(InvalidSetting::new(
                    field,
                    "must hold one value for the whole river or one for each of its points",
                ));
            }
        }
        if self
            .name
            .as_ref()
            .is_some_and(|name| name.chars().count() > MAX_CREDIT)
        {
            return Err(InvalidSetting::new(
                "name",
                format!("must hold at most {MAX_CREDIT} characters"),
            ));
        }
        Ok(())
    }
}

settings! {
    extra_rules;
    /// Rivers given over the terrain: lines its developer draws, or an
    /// import tool writes from vector data, which the hydrology bake after
    /// it runs as its courses, sized as they say, deriving only what they
    /// leave out and the courses of water no given river gathers. `source`
    /// and `attribution` (up to 1024 characters each) record where they
    /// came from and the credit their data requires, as the import tool
    /// writes them.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct RiversLayer {
        /// The rivers.
        #[cfg_attr(feature = "schema", schemars(length(min = 1, max = MAX_GIVEN_RIVERS)))]
        pub lines: Vec<GivenRiver> = Vec::new() => any,
        /// Meters its rivers' lines may stand off their true middles, as
        /// precise as their data is: the bake moves each point that far at
        /// most across its line toward the lowest ground, never over a rise;
        /// 0 keeps them where they are given.
        #[serde(default)]
        pub precision: f32 = 0.0 => range(0.0, MAX_PRECISION),
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "schema", schemars(length(max = MAX_CREDIT)))]
        pub source: Option<String> = None => any,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "schema", schemars(length(max = MAX_CREDIT)))]
        pub attribution: Option<String> = None => any,
    }
}

impl ExtraRules for RiversLayer {
    /// It lists 1 to [`MAX_GIVEN_RIVERS`] rivers; its source and its
    /// attribution hold at most [`MAX_CREDIT`] characters each.
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if !(1..=MAX_GIVEN_RIVERS).contains(&self.lines.len()) {
            return Err(InvalidSetting::new(
                "lines",
                format!("must hold 1 to {MAX_GIVEN_RIVERS} rivers"),
            ));
        }
        for (field, text) in [("source", &self.source), ("attribution", &self.attribution)] {
            if text
                .as_ref()
                .is_some_and(|text| text.chars().count() > MAX_CREDIT)
            {
                return Err(InvalidSetting::new(
                    field,
                    format!("must hold at most {MAX_CREDIT} characters"),
                ));
            }
        }
        Ok(())
    }
}

impl GivenRiver {
    /// What it gives `along` of the way through its points (their index,
    /// fractions between): its channel's width and depth and its bankfull
    /// flood, NaN where it gives none, between its points' where it gives
    /// one for each.
    pub(crate) fn given_at(&self, along: f32) -> [f32; 3] {
        let last = self.points.len() - 1;
        let index = clamp(along.floor(), 0.0, last as f32) as usize;
        let next = (index + 1).min(last);
        let share = along - index as f32;
        [&self.width, &self.depth, &self.discharge].map(|values| match values.len() {
            0 => f32::NAN,
            1 => values[0],
            _ => values[index] + (values[next] - values[index]) * share,
        })
    }
}

/// The rivers layers below a hydrology layer, each with its index in the
/// stack.
pub(crate) type GivenLayers<'a> = [(usize, &'a RiversLayer)];

/// What a bake's grid says of where water runs, as given rivers read it:
/// the samples in the order its flood reached them, each one's receiver
/// (`u32::MAX` off the grid), the area draining through each (square
/// meters) and whether a lake holds it.
pub(crate) struct Drained<'a> {
    pub(crate) order: &'a [u32],
    pub(crate) receivers: &'a [u32],
    pub(crate) area: &'a [f32],
    pub(crate) lakes: &'a [bool],
}

/// What a bake makes of the rivers given to it: their courses, in an order
/// where each comes after those ending where it starts, each knowing what
/// was given of it, with the area draining through each point; the area
/// draining through each sample of the grid, the water given rivers gather
/// held in the samples within their channels and not passed on, but where
/// their water runs on over the ground; which samples lie within their
/// channels; and where their water runs on: each such sample, ascending,
/// with the point the river it carries ends at.
pub(crate) struct GivenCourses {
    pub(crate) lines: Vec<RiverLine>,
    pub(crate) area: Vec<f32>,
    pub(crate) within: Vec<bool>,
    pub(crate) continued: Vec<(u32, Vec2)>,
}

/// A given river's stretch over a bake's grid: its points (terrain-local,
/// from upstream), where each lies along its river's points (their index,
/// fractions between), its river (by its place among the rivers given), its
/// layer's precision, the widest half its river gives its channel (0 where
/// it gives none), whether its first and last points are its river's own
/// (rather than where it leaves the grid or meets a lake), whether its
/// water runs on from its last point (where its river ends or a lake takes
/// it, not off the grid), and the points other rivers meet it at, which
/// split it.
struct Run {
    points: Vec<Vec2>,
    along: Vec<f32>,
    river: usize,
    precision: f32,
    half: f32,
    ends: [bool; 2],
    onward: bool,
    splits: Vec<usize>,
}

/// A piece of a given river between where it begins, ends, meets another
/// river, crosses a lake or leaves the grid, which the bake runs as one of
/// its courses: its points, where each lies along its river's points, its
/// river, and whether its water runs on from its last point unless a piece
/// starts there.
struct Piece {
    points: Vec<Vec2>,
    along: Vec<f32>,
    river: usize,
    onward: bool,
}

/// The centripetal Catmull-Rom spline through `p[1]` and `p[2]`, a share
/// `t` (from 0 to 1) of the way from one to the other: Barry and Goldman's
/// pyramid over knots apart by the square root of the distance between
/// points, which never cusps or loops within a span.
fn centripetal(p: [Vec2; 4], t: f32) -> Vec2 {
    let knot = |a: Vec2, b: Vec2| a.distance(b).sqrt();
    let t1 = knot(p[0], p[1]);
    let t2 = t1 + knot(p[1], p[2]);
    let t3 = t2 + knot(p[2], p[3]);
    let u = t1 + (t2 - t1) * t;
    let blend = |a: Vec2, b: Vec2, from: f32, to: f32| a + (b - a) * ((u - from) / (to - from));
    let a1 = blend(p[0], p[1], 0.0, t1);
    let a2 = blend(p[1], p[2], t1, t2);
    let a3 = blend(p[2], p[3], t2, t3);
    let b1 = blend(a1, a2, 0.0, t2);
    let b2 = blend(a2, a3, t1, t3);
    blend(b1, b2, t1, t2)
}

/// `river`'s line: points along the centripetal Catmull-Rom spline through
/// its own (each end mirrored about itself), from its first to its last,
/// every [`RIVER_STEP`] meters or so along it, and where each lies along
/// its points; none where its points all stand at one place.
fn resample(river: &GivenRiver) -> Option<(Vec<Vec2>, Vec<f32>)> {
    // Its distinct points, each with its index.
    let mut kept: Vec<(Vec2, usize)> = Vec::with_capacity(river.points.len());
    for (index, &point) in river.points.iter().enumerate() {
        let point = Vec2::from(point);
        if kept.last().is_none_or(|&(last, _)| last != point) {
            kept.push((point, index));
        }
    }
    if kept.len() < 2 {
        return None;
    }
    let last = kept.len() - 1;
    // The control points, the mirror before the first and after the last
    // around them.
    let control = |index: usize| -> Vec2 {
        if index == 0 {
            2.0 * kept[0].0 - kept[1].0
        } else if index > last + 1 {
            2.0 * kept[last].0 - kept[last - 1].0
        } else {
            kept[index - 1].0
        }
    };
    // Span `span`'s point `t` of the way along it, from point `span` to the
    // next.
    let at = |span: usize, t: f32| {
        centripetal(
            [
                control(span),
                control(span + 1),
                control(span + 2),
                control(span + 3),
            ],
            t,
        )
    };
    let place = |u: f32| {
        let span = (u.floor() as usize).min(last - 1);
        (span, u - span as f32)
    };
    // Its length measured from its start, at each of `measured`'s places.
    let mut measured: Vec<(f32, f32)> = Vec::with_capacity(last * MEASURED + 1);
    measured.push((0.0, 0.0));
    let (mut length, mut previous) = (0.0_f32, kept[0].0);
    for step in 1..=last * MEASURED {
        let u = step as f32 / MEASURED as f32;
        let (span, t) = place(u);
        let point = at(span, t);
        length += point.distance(previous);
        previous = point;
        measured.push((u, length));
    }
    let steps = max((length / RIVER_STEP).ceil(), 1.0) as usize;
    let mut points = Vec::with_capacity(steps + 1);
    let mut along = Vec::with_capacity(steps + 1);
    let mut interval = 0;
    for step in 0..=steps {
        let wanted = length * step as f32 / steps as f32;
        while interval + 2 < measured.len() && measured[interval + 1].1 < wanted {
            interval += 1;
        }
        let ((from, before), (to, after)) = (measured[interval], measured[interval + 1]);
        let share = if after > before {
            clamp((wanted - before) / (after - before), 0.0, 1.0)
        } else {
            0.0
        };
        let (span, t) = place(from + (to - from) * share);
        points.push(at(span, t));
        let (first, next) = (kept[span].1 as f32, kept[span + 1].1 as f32);
        along.push(first + (next - first) * t);
    }
    // Its ends exactly at its own.
    points[0] = kept[0].0;
    along[0] = kept[0].1 as f32;
    points[steps] = kept[last].0;
    along[steps] = kept[last].1 as f32;
    Some((points, along))
}

/// A stretch of a given river ([`stretches`]): its points, how far along
/// the river each lies, whether it starts at the river's first point and
/// ends at its last, and whether its water runs on from its end.
type Stretch = (Vec<Vec2>, Vec<f32>, [bool; 2], bool);

/// The stretches of a river's `points` (each `along` its own) a bake's
/// `grid` holds outside lakes, as a lake holds a course whose ground lies
/// below its level: two points or more each, with whether each starts at
/// the river's first point and ends at its last, and whether its water
/// runs on from its end: where the river ends or a lake takes it, not where
/// it leaves the grid.
fn stretches(
    (points, along): (&[Vec2], &[f32]),
    grid: &Grid<'_>,
    water: &WaterGrid,
) -> Vec<Stretch> {
    let extent = Vec2::new((grid.width - 1) as f32, (grid.rows() - 1) as f32) * grid.spacing;
    let held = |point: Vec2| point.cmpge(Vec2::ZERO).all() && point.cmple(extent).all();
    let kept = |point: Vec2| {
        held(point)
            && !water
                .lake_reach(point.x, point.y)
                .is_some_and(|(level, _)| grid.height(point.x, point.y) < level)
    };
    let last = points.len() - 1;
    let mut found = Vec::new();
    let mut push = |from: usize, to: usize| {
        if to > from {
            found.push((
                points[from..=to].to_vec(),
                along[from..=to].to_vec(),
                [from == 0, to == last],
                to == last || held(points[to + 1]),
            ));
        }
    };
    let mut start = None;
    for (index, &point) in points.iter().enumerate() {
        match (kept(point), start) {
            (true, None) => start = Some(index),
            (false, Some(from)) => {
                push(from, index - 1);
                start = None;
            }
            (true, Some(_)) | (false, None) => {}
        }
    }
    if let Some(from) = start {
        push(from, last);
    }
    found
}

/// Moves each of `points` but its first and last up to `precision` meters
/// across its line (square to the way from the point before it to the
/// point after) toward the lowest ground of `grid` there, read every half
/// of its samples outward on either side while the ground stands no higher
/// than at the point itself, so no point crosses a rise: the nearest of the
/// lowest, the point itself where nothing lies lower.
fn snap(points: &mut [Vec2], precision: f32, grid: &Grid<'_>) {
    if precision <= 0.0 || points.len() < 3 {
        return;
    }
    let step = 0.5 * grid.spacing;
    let steps = (precision / step).floor() as usize;
    let original = points.to_vec();
    for index in 1..original.len() - 1 {
        let point = original[index];
        let across = (original[index + 1] - original[index - 1])
            .normalize_or(Vec2::X)
            .perp();
        let start = grid.height(point.x, point.y);
        let mut lowest = (start, 0.0_f32);
        for side in [1.0_f32, -1.0] {
            for out in 1..=steps {
                let offset = side * out as f32 * step;
                let at = point + across * offset;
                let ground = grid.height(at.x, at.y);
                if ground > start {
                    break;
                }
                if ground < lowest.0 {
                    lowest = (ground, offset);
                }
            }
        }
        points[index] = point + across * lowest.1;
    }
}

/// The bounds (lowest and highest x and z) of `points`.
fn bounds(points: &[Vec2]) -> (Vec2, Vec2) {
    points.iter().fold(
        (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
        |(low, high), &point| (low.min(point), high.max(point)),
    )
}

/// Joins `runs` where one's own end lies within reach of another river's
/// run, a grid `spacing` meters apart: the end moves onto the other's
/// point nearest it, or onto its first or last point where that lies
/// within a step, and the other splits there; meetings within a step of
/// one another share one point. The reach is the end's precision, the
/// narrowest channel the grid holds, or the half the other river gives its
/// channel, whichever is farthest: confluences where one river ends on
/// another, and rivers given in pieces end to end.
fn join(runs: &mut [Run], spacing: f32) {
    let spans: Vec<(Vec2, Vec2)> = runs.iter().map(|run| bounds(&run.points)).collect();
    // Each own end's meeting: its run, whether it is the run's last point,
    // the other run, the segment of it and the share of the way along that.
    let mut meetings: Vec<(usize, bool, usize, usize, f32)> = Vec::new();
    for (index, run) in runs.iter().enumerate() {
        let last = run.points.len() - 1;
        for (end, at) in [(false, 0), (true, last)] {
            if !run.ends[usize::from(end)] {
                continue;
            }
            let point = run.points[at];
            let mut best: Option<(f32, usize, usize, f32)> = None;
            for (other, them) in runs.iter().enumerate() {
                if them.river == run.river {
                    continue;
                }
                let reach = max(max(run.precision, NARROWEST * spacing), them.half);
                let (low, high) = spans[other];
                if (point + reach).cmplt(low).any() || (point - reach).cmpgt(high).any() {
                    continue;
                }
                for segment in 0..them.points.len() - 1 {
                    let (distance, t) =
                        distance_to_segment(point, them.points[segment], them.points[segment + 1]);
                    if distance <= reach && best.is_none_or(|(nearest, ..)| distance < nearest) {
                        best = Some((distance, other, segment, t));
                    }
                }
            }
            if let Some((_, other, segment, t)) = best {
                meetings.push((index, end, other, segment, t));
            }
        }
    }
    // Where each meeting lands: an end of its other run, or the point of it
    // nearest the meeting, which splits it; each by its run and point.
    let mut lands: Vec<Option<(usize, usize)>> = vec![None; meetings.len()];
    let mut places: Vec<Vec<(usize, usize)>> = vec![Vec::new(); runs.len()];
    for (number, &(_, _, other, segment, t)) in meetings.iter().enumerate() {
        let them = &runs[other];
        let last = them.points.len() - 1;
        let vertex = if t < 0.5 { segment } else { segment + 1 };
        let point = them.points[vertex];
        if vertex == 0 || point.distance(them.points[0]) <= RIVER_STEP {
            lands[number] = Some((other, 0));
        } else if vertex == last || point.distance(them.points[last]) <= RIVER_STEP {
            lands[number] = Some((other, last));
        } else {
            places[other].push((vertex, number));
        }
    }
    for (index, run) in runs.iter_mut().enumerate() {
        let mut at = std::mem::take(&mut places[index]);
        at.sort_unstable();
        let mut cut: Vec<usize> = Vec::new();
        for (vertex, meeting) in at {
            // The split here, or the one before where that stands within a
            // step.
            let split = match cut.last() {
                Some(&before) if run.points[before].distance(run.points[vertex]) <= RIVER_STEP => {
                    before
                }
                _ => {
                    cut.push(vertex);
                    vertex
                }
            };
            lands[meeting] = Some((index, split));
        }
        run.splits = cut;
    }
    // Each meeting's end takes the point it lands on, an end of another run
    // following that end where it moves too: again while any end moves.
    for _ in 0..=meetings.len() {
        let mut moved = false;
        for (number, &(index, end, ..)) in meetings.iter().enumerate() {
            let Some((other, vertex)) = lands[number] else {
                continue;
            };
            let target = runs[other].points[vertex];
            let run = &mut runs[index];
            let at = if end { run.points.len() - 1 } else { 0 };
            if run.points[at] != target {
                run.points[at] = target;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
}

/// `runs` split where other rivers meet them, into the pieces the bake runs
/// as courses.
fn split_runs(runs: &[Run]) -> Vec<Piece> {
    let mut found = Vec::new();
    for run in runs {
        let last = run.points.len() - 1;
        let mut bounds = Vec::with_capacity(run.splits.len() + 2);
        bounds.push(0);
        bounds.extend_from_slice(&run.splits);
        bounds.push(last);
        bounds.dedup();
        for pair in bounds.windows(2) {
            let (from, to) = (pair[0], pair[1]);
            if to <= from {
                continue;
            }
            found.push(Piece {
                points: run.points[from..=to].to_vec(),
                along: run.along[from..=to].to_vec(),
                river: run.river,
                onward: run.onward && to == last,
            });
        }
    }
    found
}

/// The sample of `grid` nearest terrain-local `point`, within the grid.
fn nearest(grid: &Grid<'_>, point: Vec2) -> usize {
    let x = clamp(
        (point.x / grid.spacing).round(),
        0.0,
        (grid.width - 1) as f32,
    ) as usize;
    let z = clamp(
        (point.y / grid.spacing).round(),
        0.0,
        (grid.rows() - 1) as f32,
    ) as usize;
    z * grid.width + x
}

/// The most of `area` (square meters, by sample of `grid`) draining through
/// the sample nearest terrain-local `point` or one beside it: what reaches
/// a river there, whichever of the grid's courses it follows.
fn drained_near(area: &[f32], grid: &Grid<'_>, point: Vec2) -> f32 {
    let center = nearest(grid, point);
    let (x, z) = ((center % grid.width) as i64, (center / grid.width) as i64);
    let (columns, rows) = (grid.width as i64, grid.rows() as i64);
    let mut most = 0.0_f32;
    for dz in -1..=1 {
        for dx in -1..=1 {
            let (sx, sz) = (x + dx, z + dz);
            if sx >= 0 && sz >= 0 && sx < columns && sz < rows {
                most = max(most, area[(sz * columns + sx) as usize]);
            }
        }
    }
    most
}

/// The area draining through each sample of a grid whose flood reached its
/// samples in `order`, each draining to its receiver in `receivers`, every
/// sample `cell` square meters: summed in the reverse of the flood's order,
/// as the bake sums it, but for the samples `owner` gives a given river's
/// point, which hold the water reaching them and pass none on, with
/// `injected` more running on at the samples it names, in their order.
fn gather(
    (order, receivers): (&[u32], &[u32]),
    owner: &[u32],
    injected: &[(u32, f32)],
    cell: f32,
) -> Vec<f32> {
    let mut area = vec![cell; receivers.len()];
    for &(sample, water) in injected {
        area[sample as usize] += water;
    }
    for &sample in order.iter().rev() {
        let sample = sample as usize;
        let receiver = receivers[sample];
        if receiver != u32::MAX && owner[sample] == u32::MAX {
            area[receiver as usize] += area[sample];
        }
    }
    area
}

/// Each river given (its river, its rivers layer's index in the stack and
/// its own in that layer, and its layer's precision), in order.
type Listing<'a> = [(&'a GivenRiver, [u32; 2], f32)];

/// The meters across a given river's channel `along` of the way through its
/// points, at terrain-local `point`, as the bake first sees it: as given, or
/// as its flood stands it, the area draining near it bringing what is not
/// given.
type Channel<'a> = dyn Fn(&GivenRiver, f32, Vec2) -> f32 + 'a;

/// The stretches of `rivers` a bake's `grid` holds outside the lakes of
/// `water`, each moved toward the lowest ground within its precision.
fn given_runs(rivers: &Listing<'_>, grid: &Grid<'_>, water: &WaterGrid) -> Vec<Run> {
    let mut runs = Vec::new();
    for (number, &(river, _, precision)) in rivers.iter().enumerate() {
        let Some((points, along)) = resample(river) else {
            continue;
        };
        let half = river
            .width
            .iter()
            .fold(0.0_f32, |widest, &width| max(widest, 0.5 * width));
        for (points, along, ends, onward) in stretches((&points, &along), grid, water) {
            let mut run = Run {
                points,
                along,
                river: number,
                precision,
                half,
                ends,
                onward,
                splits: Vec::new(),
            };
            snap(&mut run.points, precision, grid);
            runs.push(run);
        }
    }
    runs
}

/// `pieces` of rivers whose data is imprecise run as smooth as their
/// channels (as `channel` gives them) are wide, as the bake's own courses
/// do: their points shape them no closer together than that, along the
/// uniform cubic B-spline through those ([`smooth_line`]), their ends kept.
fn reshape(pieces: &mut [Piece], rivers: &Listing<'_>, channel: &Channel<'_>) {
    for piece in pieces {
        let (river, _, precision) = rivers[piece.river];
        let last = piece.points.len() - 1;
        if precision <= 0.0 || last < 2 {
            continue;
        }
        let mut shaping = vec![0];
        for index in 1..last {
            let from = piece.points[shaping[shaping.len() - 1]];
            let width = channel(river, piece.along[index], piece.points[index]);
            if from.distance(piece.points[index]) >= width {
                shaping.push(index);
            }
        }
        shaping.push(last);
        let points: Vec<Vec2> = shaping.iter().map(|&index| piece.points[index]).collect();
        let along: Vec<f32> = shaping.iter().map(|&index| piece.along[index]).collect();
        (piece.points, piece.along) = smooth_line(&points, &along);
    }
}

/// Where each of `pieces`' points begins among all of theirs, one after
/// another (and their count, last), and the samples of `grid` within each
/// piece's channel (as `channel` gives it) and a sample's square past it,
/// outside lakes (`lakes`): by sample, the point of the nearest piece
/// nearest it, `u32::MAX` for none.
fn footprint(
    (pieces, rivers): (&[Piece], &Listing<'_>),
    channel: &Channel<'_>,
    grid: &Grid<'_>,
    lakes: &[bool],
) -> (Vec<usize>, Vec<u32>) {
    let count = lakes.len();
    let rows = count / grid.width;
    let mut offsets = Vec::with_capacity(pieces.len() + 1);
    offsets.push(0_usize);
    for piece in pieces {
        offsets.push(offsets[offsets.len() - 1] + piece.points.len());
    }
    let mut owner = vec![u32::MAX; count];
    let mut closest = vec![f32::INFINITY; count];
    for (number, piece) in pieces.iter().enumerate() {
        let (river, _, _) = rivers[piece.river];
        let halves: Vec<f32> = (0..piece.points.len())
            .map(|index| 0.5 * channel(river, piece.along[index], piece.points[index]))
            .collect();
        for segment in 0..piece.points.len() - 1 {
            let (a, b) = (piece.points[segment], piece.points[segment + 1]);
            let reach = max(halves[segment], halves[segment + 1]) + FOOTPRINT * grid.spacing;
            let low = ((a.min(b) - Vec2::splat(reach)) / grid.spacing).ceil();
            let high = ((a.max(b) + Vec2::splat(reach)) / grid.spacing).floor();
            let first = low.max(Vec2::ZERO);
            let end = Vec2::new(
                min(high.x, (grid.width - 1) as f32),
                min(high.y, (rows - 1) as f32),
            );
            if first.x > end.x || first.y > end.y {
                continue;
            }
            for z in first.y as usize..=end.y as usize {
                for x in first.x as usize..=end.x as usize {
                    let sample = z * grid.width + x;
                    if lakes[sample] {
                        continue;
                    }
                    let place = Vec2::new(x as f32, z as f32) * grid.spacing;
                    let (distance, t) = distance_to_segment(place, a, b);
                    if distance <= reach && distance < closest[sample] {
                        closest[sample] = distance;
                        let point = if t < 0.5 { segment } else { segment + 1 };
                        owner[sample] = (offsets[number] + point) as u32;
                    }
                }
            }
        }
    }
    (offsets, owner)
}

/// The bits of terrain-local `point`, which pieces meeting there share.
fn key(point: Vec2) -> [u32; 2] {
    [point.x.to_bits(), point.y.to_bits()]
}

/// The pieces starting at each point, by its bits ([`key`]).
type Starting = HashMap<[u32; 2], Vec<usize>>;

/// How `pieces` meet: those starting at each point, by its bits ([`key`]);
/// for each piece, those ending where it starts; and every piece after
/// those ending where it starts, the first ready first, pieces in a loop,
/// which no data should hold, last in order.
fn meetings(pieces: &[Piece]) -> (Starting, Vec<Vec<usize>>, Vec<usize>) {
    let mut starting: Starting = HashMap::default();
    let mut ending: HashMap<[u32; 2], Vec<usize>> = HashMap::default();
    for (number, piece) in pieces.iter().enumerate() {
        starting
            .entry(key(piece.points[0]))
            .or_default()
            .push(number);
        ending
            .entry(key(piece.points[piece.points.len() - 1]))
            .or_default()
            .push(number);
    }
    let fed: Vec<Vec<usize>> = pieces
        .iter()
        .map(|piece| {
            ending
                .get(&key(piece.points[0]))
                .cloned()
                .unwrap_or_default()
        })
        .collect();
    let mut waiting: Vec<usize> = fed.iter().map(Vec::len).collect();
    let mut ready: BinaryHeap<Reverse<usize>> = (0..pieces.len())
        .filter(|&number| waiting[number] == 0)
        .map(Reverse)
        .collect();
    let mut sequence = Vec::with_capacity(pieces.len());
    let mut placed = vec![false; pieces.len()];
    while let Some(Reverse(number)) = ready.pop() {
        sequence.push(number);
        placed[number] = true;
        let piece = &pieces[number];
        if let Some(next) = starting.get(&key(piece.points[piece.points.len() - 1])) {
            for &next in next {
                waiting[next] -= 1;
                if waiting[next] == 0 && !placed[next] {
                    ready.push(Reverse(next));
                }
            }
        }
    }
    sequence.extend((0..pieces.len()).filter(|&number| !placed[number]));
    (starting, fed, sequence)
}

/// A piece passing its water on, and the sample it passes it to.
type Passed = (usize, u32);

/// Where `pieces` whose water runs on from their end (where their rivers
/// end or a lake takes them, and no piece starts) pass it on: down the
/// courses of `drained` from the sample of `grid` nearest their end while
/// those stay within their own channel (`owner`, of points numbered from
/// `offsets`), nowhere where the courses leave the grid: each such piece
/// with the sample it passes its water to, and the samples a course carries
/// the water on from (where no other river's channel or a lake takes it),
/// ascending, each with the point the river bringing it ends at.
fn onward(
    (pieces, starting): (&[Piece], &Starting),
    (offsets, owner): (&[usize], &[u32]),
    grid: &Grid<'_>,
    drained: &Drained<'_>,
) -> (Vec<Passed>, Vec<(u32, Vec2)>) {
    let piece_of = |owned: u32| offsets.partition_point(|&offset| offset <= owned as usize) - 1;
    let mut passed = Vec::new();
    let mut continued: Vec<(u32, Vec2)> = Vec::new();
    for (number, piece) in pieces.iter().enumerate() {
        let end = piece.points[piece.points.len() - 1];
        if !piece.onward || starting.contains_key(&key(end)) {
            continue;
        }
        let mut sample = nearest(grid, end);
        let mut kept = true;
        while owner[sample] != u32::MAX && piece_of(owner[sample]) == number {
            let receiver = drained.receivers[sample];
            if receiver == u32::MAX {
                kept = false;
                break;
            }
            sample = receiver as usize;
        }
        if !kept {
            continue;
        }
        passed.push((number, sample as u32));
        if owner[sample] == u32::MAX && !drained.lakes[sample] {
            continued.push((sample as u32, end));
        }
    }
    continued.sort_by_key(|&(sample, _)| sample);
    continued.dedup_by_key(|&mut (sample, _)| sample);
    (passed, continued)
}

/// The area draining through each sample of `drained`'s grid, every sample
/// `cell` square meters, the samples within given rivers' channels
/// (`owner`, of points numbered from `offsets`) holding what reaches them,
/// and the area draining through each point of each piece: what reaches its
/// points' samples and what the pieces ending where it starts (`fed`)
/// carry, summed in `sequence`, each piece passing what it carries on
/// where it ends on open ground (`passed`); summed again while what pieces
/// pass on changes, as one running on into another river's channel feeds
/// it.
fn gathered(
    drained: &Drained<'_>,
    (offsets, owner): (&[usize], &[u32]),
    (fed, sequence, passed): (&[Vec<usize>], &[usize], &[Passed]),
    cell: f32,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let pieces = fed.len();
    let mut injected: Vec<(u32, f32)> = Vec::new();
    let mut rounds = 0;
    loop {
        let area = gather((drained.order, drained.receivers), owner, &injected, cell);
        let mut own = vec![0.0_f32; offsets[pieces]];
        for (sample, &owned) in owner.iter().enumerate() {
            if owned != u32::MAX {
                own[owned as usize] += area[sample];
            }
        }
        let mut totals = vec![0.0_f32; pieces];
        let mut areas: Vec<Vec<f32>> = vec![Vec::new(); pieces];
        for &number in sequence {
            let mut carried = fed[number]
                .iter()
                .fold(0.0_f32, |sum, &from| sum + totals[from]);
            let points = &own[offsets[number]..offsets[number + 1]];
            let mut along = Vec::with_capacity(points.len());
            for &reaching in points {
                carried += reaching;
                along.push(max(carried, cell));
            }
            totals[number] = carried;
            areas[number] = along;
        }
        let mut next: Vec<(u32, f32)> = passed
            .iter()
            .map(|&(number, sample)| (sample, totals[number]))
            .collect();
        next.sort_by_key(|&(sample, _)| sample);
        if next == injected || rounds >= pieces {
            return (area, areas);
        }
        injected = next;
        rounds += 1;
    }
}

/// What a hydrology bake makes of the rivers `given` it (the rivers layers
/// below it) over its `grid`, where `water` holds its lakes and `drained`
/// says where water runs, each square meter drained bringing `runoff` cubic
/// meters a second in flood, its channels at least `narrowest` meters
/// across: none where no given river runs over the grid outside lakes.
/// Each piece of a given river marks the samples within its channel
/// ([`footprint`]); the area draining through each sample is then summed
/// with the water reaching those samples held in their piece, each piece
/// gathering it and what the pieces ending where it starts carry, and
/// passing it on over the ground where it ends on open ground
/// ([`gathered`]).
pub(crate) fn courses(
    given: &GivenLayers<'_>,
    (grid, water): (&Grid<'_>, &WaterGrid),
    drained: &Drained<'_>,
    (runoff, narrowest): (f32, f32),
) -> Option<GivenCourses> {
    let rivers: Vec<(&GivenRiver, [u32; 2], f32)> = given
        .iter()
        .flat_map(|&(index, layer)| {
            layer
                .lines
                .iter()
                .enumerate()
                .map(move |(line, river)| (river, [index as u32, line as u32], layer.precision))
        })
        .collect();
    let mut runs = given_runs(&rivers, grid, water);
    if runs.is_empty() {
        return None;
    }
    join(&mut runs, grid.spacing);
    let mut pieces = split_runs(&runs);
    let channel = |river: &GivenRiver, along: f32, point: Vec2| -> f32 {
        let values = river.given_at(along);
        let flood = if values[2].is_finite() {
            values[2]
        } else {
            drained_near(drained.area, grid, point) * runoff
        };
        if values[0].is_finite() {
            values[0]
        } else {
            max(bankfull(flood).0, narrowest)
        }
    };
    reshape(&mut pieces, &rivers, &channel);
    let (offsets, owner) = footprint((&pieces, &rivers), &channel, grid, drained.lakes);
    let (starting, fed, sequence) = meetings(&pieces);
    let (passed, continued) = onward((&pieces, &starting), (&offsets, &owner), grid, drained);
    let (area, areas) = gathered(
        drained,
        (&offsets, &owner),
        (&fed, &sequence, &passed),
        grid.spacing * grid.spacing,
    );
    let lines = sequence
        .iter()
        .map(|&number| {
            let piece = &pieces[number];
            let (river, source, _) = rivers[piece.river];
            RiverLine {
                points: piece.points.clone(),
                area: areas[number].clone(),
                head: fed[number].is_empty(),
                given: Some(GivenLine {
                    source,
                    values: piece
                        .along
                        .iter()
                        .map(|&along| river.given_at(along))
                        .collect(),
                }),
            }
        })
        .collect();
    Some(GivenCourses {
        lines,
        area,
        within: owner.iter().map(|&owned| owned != u32::MAX).collect(),
        continued,
    })
}

crate::codec::record!(GivenRiver {
    name,
    points,
    width,
    depth,
    discharge,
});
crate::codec::record!(RiversLayer {
    lines,
    precision,
    source,
    attribution,
});
