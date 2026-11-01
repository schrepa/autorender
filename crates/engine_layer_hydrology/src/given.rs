use std::cmp::Reverse;
use std::collections::BinaryHeap;

use engine_codec::{Allowed, Error, Expected};
use engine_core::{InvalidSetting, Rule, Setting};
use engine_heightfield::bed::{NARROWEST, bankfull};
use engine_heightfield::given::{GivenRiver, MAX_CREDIT, MAX_GIVEN_RIVERS, MAX_PRECISION};
use engine_heightfield::hydrology::{Grid, WaterGrid};
use engine_heightfield::path::distance_to_segment;
use engine_heightfield::rivers::{FOOTPRINT, GivenLine, RIVER_STEP, RiverLine};
use engine_math::collections::HashMap;
use engine_math::{clamp, max, min};
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::rivers::{MEASURED, smooth_line};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RiversLayer {
    #[schemars(length(min = 1, max = MAX_GIVEN_RIVERS))]
    pub lines: Vec<GivenRiver>,
    #[serde(default)]
    pub precision: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = MAX_CREDIT))]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = MAX_CREDIT))]
    pub attribution: Option<String>,
}

impl RiversLayer {
    pub fn validate(&self) -> Result<(), Error> {
        for (index, line) in self.lines.iter().enumerate() {
            line.validate()
                .map_err(|error| error.within(&format!("$.lines[{index}]")))?;
        }
        self.precision
            .check("precision", Rule::Range(0.0, f64::from(MAX_PRECISION)))?;
        if !(1..=MAX_GIVEN_RIVERS).contains(&self.lines.len()) {
            return Err(InvalidSetting::new("lines", Expected::Array)
                .allowed(Allowed::Range {
                    min: 1.0,
                    max: MAX_GIVEN_RIVERS as f64,
                })
                .into());
        }
        for (field, text) in [("source", &self.source), ("attribution", &self.attribution)] {
            if text
                .as_ref()
                .is_some_and(|text| text.chars().count() > MAX_CREDIT)
            {
                return Err(InvalidSetting::new(field, Expected::String)
                    .allowed(Allowed::AtMost {
                        max: MAX_CREDIT as u64,
                    })
                    .into());
            }
        }
        Ok(())
    }
}

fn given_at(river: &GivenRiver, along: f32) -> [f32; 3] {
    let last = river.points.len() - 1;
    let index = clamp(along.floor(), 0.0, last as f32) as usize;
    let next = (index + 1).min(last);
    let share = along - index as f32;
    [&river.width, &river.depth, &river.discharge].map(|values| match values.len() {
        0 => f32::NAN,
        1 => values[0],
        _ => values[index] + (values[next] - values[index]) * share,
    })
}

pub(crate) type GivenLayers<'a> = [(usize, &'a [GivenRiver], f32)];

pub(crate) struct Drained<'a> {
    pub(crate) order: &'a [u32],
    pub(crate) receivers: &'a [u32],
    pub(crate) area: &'a [f32],
    pub(crate) lakes: &'a [bool],
}

pub(crate) struct GivenCourses {
    pub(crate) lines: Vec<RiverLine>,
    pub(crate) area: Vec<f32>,
    pub(crate) within: Vec<bool>,
    pub(crate) continued: Vec<(u32, Vec2)>,
}

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

struct Piece {
    points: Vec<Vec2>,
    along: Vec<f32>,
    river: usize,
    onward: bool,
}

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

fn resample(river: &GivenRiver) -> Option<(Vec<Vec2>, Vec<f32>)> {
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
    let control = |index: usize| -> Vec2 {
        if index == 0 {
            2.0 * kept[0].0 - kept[1].0
        } else if index > last + 1 {
            2.0 * kept[last].0 - kept[last - 1].0
        } else {
            kept[index - 1].0
        }
    };
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
    points[0] = kept[0].0;
    along[0] = kept[0].1 as f32;
    points[steps] = kept[last].0;
    along[steps] = kept[last].1 as f32;
    Some((points, along))
}

type Stretch = (Vec<Vec2>, Vec<f32>, [bool; 2], bool);

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

fn bounds(points: &[Vec2]) -> (Vec2, Vec2) {
    points.iter().fold(
        (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
        |(low, high), &point| (low.min(point), high.max(point)),
    )
}

fn join(runs: &mut [Run], spacing: f32) {
    let spans: Vec<(Vec2, Vec2)> = runs.iter().map(|run| bounds(&run.points)).collect();
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

type Listing<'a> = [(&'a GivenRiver, [u32; 2], f32)];

type Channel<'a> = dyn Fn(&GivenRiver, f32, Vec2) -> f32 + 'a;

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

fn key(point: Vec2) -> [u32; 2] {
    [point.x.to_bits(), point.y.to_bits()]
}

type Starting = HashMap<[u32; 2], Vec<usize>>;

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

type Passed = (usize, u32);

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

pub(crate) fn courses(
    given: &GivenLayers<'_>,
    (grid, water): (&Grid<'_>, &WaterGrid),
    drained: &Drained<'_>,
    (runoff, narrowest): (f32, f32),
) -> Option<GivenCourses> {
    let rivers: Vec<(&GivenRiver, [u32; 2], f32)> = given
        .iter()
        .flat_map(|&(index, lines, precision)| {
            lines
                .iter()
                .enumerate()
                .map(move |(line, river)| (river, [index as u32, line as u32], precision))
        })
        .collect();
    let mut runs = given_runs(&rivers, grid, water);
    if runs.is_empty() {
        return None;
    }
    join(&mut runs, grid.spacing);
    let mut pieces = split_runs(&runs);
    let channel = |river: &GivenRiver, along: f32, point: Vec2| -> f32 {
        let values = given_at(river, along);
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
                        .map(|&along| given_at(river, along))
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

engine_codec::record!(RiversLayer {
    lines,
    precision,
    source,
    attribution,
});
