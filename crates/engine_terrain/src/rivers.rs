//! Rivers (decision 092): the channels a hydrology bake (decision 066)
//! finds, as lines with water of their own. A bake traces each channel
//! from its head or a confluence down its receivers to the next
//! confluence, a lake or the grid's edge, rounds the grid's steps off its
//! line and resamples it every [`RIVER_STEP`] meters, and sizes it by the
//! area it drains. Its water then stands below the finest ground along it
//! and its banks and never rises downstream; a river leaving a confluence
//! starts no higher than the rivers meeting there. Terrain cuts the bed
//! below that water at every level of detail and eases its banks up to the
//! ground, so ground the bake never saw cannot rise through the water, and
//! renderers draw the water as ribbons flat across each river.

use engine_math::{max, min};
use glam::Vec2;
use serde::Serialize;

/// Meters between a river's points.
pub const RIVER_STEP: f32 = 3.0;
/// Meters below the lowest of a river's ground and banks its water stands.
const BANK_MARGIN: f32 = 0.25;
/// Meters beyond a river's edge its banks are read at.
pub const BANK_OFFSET: f32 = 1.0;
/// How steeply banks rise from the water at most (meters a meter), the
/// meters from the edge they are cut over (the cut easing back to the
/// ground over their outer half), and the meters over which a cut rounds
/// into the ground it meets.
const BANK_SLOPE: f32 = 0.6;
const BANK_REACH: f32 = 12.0;
const BANK_ROUNDING: f32 = 1.5;
/// The least water a river holds at its middle (meters): shallower points
/// are streams' (decision 097).
pub const MIN_DEPTH: f32 = 0.35;
/// Meters below the lowest of a stream's ground and banks its water stands
/// (decision 097), following the ground rather than never rising, and the
/// meters from its edge its banks are cut over: a stream cuts its course a
/// little into the ground instead of a trench through every rise.
const STREAM_MARGIN: f32 = 0.12;
const STREAM_BANK_REACH: f32 = 2.5;
/// Rounds of corner cutting that smooth a river's line off the grid.
const SMOOTHING: usize = 3;
/// Meters across the cells of the grid that finds the rivers near a point.
const CELL: f32 = 32.0;

/// One river or stream (decision 097) between confluences: its points
/// (terrain-local x and z, from upstream), and at each its water's surface
/// (terrain-local meters), half its wet width, the water's depth at its
/// middle and its speed downstream (meters a second). Stretches dropping
/// steeper than the bake's falls (decision 075) carry no surface, which
/// waterfalls show instead. A river beginning at a head, where no channel
/// flows in, rather than at a confluence, is `head`: its water starts there
/// (decision 097).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct River {
    pub points: Vec<[f32; 2]>,
    pub surface: Vec<f32>,
    pub half: Vec<f32>,
    pub depth: Vec<f32>,
    pub speed: Vec<f32>,
    pub falls: Vec<bool>,
    pub head: bool,
}

impl River {
    /// Whether the stretch from point `index` to the next falls.
    pub fn falls_at(&self, index: usize) -> bool {
        self.falls.get(index).copied().unwrap_or(false)
    }
}

/// A river as a point sees it: meters from its line, and there its water's
/// surface, half its width, its depth at its middle, its speed and the way
/// it runs (a unit vector along x and z).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverPoint {
    pub distance: f32,
    pub surface: f32,
    pub half: f32,
    pub depth: f32,
    pub speed: f32,
    pub direction: [f32; 2],
}

/// A terrain's rivers and a grid of the river segments within cutting
/// reach of each cell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rivers {
    rivers: Vec<River>,
    min: Vec2,
    columns: usize,
    rows: usize,
    starts: Vec<u32>,
    segments: Vec<[u32; 2]>,
}

impl Rivers {
    /// `rivers` with the grid that finds them.
    pub fn new(rivers: Vec<River>) -> Self {
        let mut low = Vec2::splat(f32::INFINITY);
        let mut high = Vec2::splat(f32::NEG_INFINITY);
        let mut widest = 0.0_f32;
        for river in &rivers {
            for (point, &half) in river.points.iter().zip(&river.half) {
                low = low.min(Vec2::from(*point));
                high = high.max(Vec2::from(*point));
                widest = max(widest, half);
            }
        }
        if rivers.iter().all(|river| river.points.len() < 2) {
            return Self {
                rivers,
                ..Self::default()
            };
        }
        // Coarse levels widen beds to a sample (`cap`): leave room for the
        // coarsest level that cuts them.
        let reach = widest + BANK_REACH + 64.0;
        let corner = low - Vec2::splat(reach);
        let columns = (((high.x + reach - corner.x) / CELL).ceil() as usize).max(1);
        let rows = (((high.y + reach - corner.y) / CELL).ceil() as usize).max(1);
        let mut lists: Vec<Vec<[u32; 2]>> = vec![Vec::new(); columns * rows];
        for (index, river) in rivers.iter().enumerate() {
            for segment in 0..river.points.len().saturating_sub(1) {
                let (a, b) = (
                    Vec2::from(river.points[segment]),
                    Vec2::from(river.points[segment + 1]),
                );
                let half = max(river.half[segment], river.half[segment + 1]);
                // Rivers widen to most of a coarse level's sample; streams
                // keep their width (decision 097), so they reach no further
                // than a river's banks and list in far fewer cells.
                let stream = max(river.depth[segment], river.depth[segment + 1]) < MIN_DEPTH;
                let margin = half + BANK_REACH + if stream { 0.0 } else { 64.0 };
                let first = ((a.min(b) - Vec2::splat(margin) - corner) / CELL).floor();
                let last = ((a.max(b) + Vec2::splat(margin) - corner) / CELL).floor();
                for row in max(first.y, 0.0) as usize..=min(last.y, rows as f32 - 1.0) as usize {
                    for column in
                        max(first.x, 0.0) as usize..=min(last.x, columns as f32 - 1.0) as usize
                    {
                        lists[row * columns + column].push([index as u32, segment as u32]);
                    }
                }
            }
        }
        let mut starts = Vec::with_capacity(lists.len() + 1);
        let mut segments = Vec::new();
        for list in lists {
            starts.push(segments.len() as u32);
            segments.extend(list);
        }
        starts.push(segments.len() as u32);
        Self {
            rivers,
            min: corner,
            columns,
            rows,
            starts,
            segments,
        }
    }

    pub fn rivers(&self) -> &[River] {
        &self.rivers
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The square holding every point a river's cut reaches.
    pub fn bounds(&self) -> (Vec2, Vec2) {
        (
            self.min,
            self.min + Vec2::new(self.columns as f32, self.rows as f32) * CELL,
        )
    }

    /// The segments listed for the cell holding terrain-local `point`.
    fn near(&self, point: Vec2) -> &[[u32; 2]] {
        if self.segments.is_empty() {
            return &[];
        }
        let cell = ((point - self.min) / CELL).floor();
        if !(cell.x >= 0.0 && cell.y >= 0.0) {
            return &[];
        }
        let (column, row) = (cell.x as usize, cell.y as usize);
        if column >= self.columns || row >= self.rows {
            return &[];
        }
        let at = row * self.columns + column;
        &self.segments[self.starts[at] as usize..self.starts[at + 1] as usize]
    }

    /// River `river`'s segment `segment` as `point` sees it.
    fn on_segment(&self, [river, segment]: [u32; 2], point: Vec2) -> RiverPoint {
        let river = &self.rivers[river as usize];
        let segment = segment as usize;
        let (a, b) = (
            Vec2::from(river.points[segment]),
            Vec2::from(river.points[segment + 1]),
        );
        let run = b - a;
        let length = run.length();
        let t = if length > 0.0 {
            min(max((point - a).dot(run) / (length * length), 0.0), 1.0)
        } else {
            0.0
        };
        let at = |values: &[f32]| values[segment] + (values[segment + 1] - values[segment]) * t;
        let direction = if length > 0.0 { run / length } else { Vec2::X };
        RiverPoint {
            distance: point.distance(a + run * t),
            surface: at(&river.surface),
            half: at(&river.half),
            depth: at(&river.depth),
            speed: at(&river.speed),
            direction: direction.to_array(),
        }
    }

    /// The river nearest terrain-local `(x, z)` within cutting reach.
    pub fn nearest(&self, x: f32, z: f32) -> Option<RiverPoint> {
        let point = Vec2::new(x, z);
        self.near(point)
            .iter()
            .map(|&segment| self.on_segment(segment, point))
            .filter(|river| river.distance < river.half + BANK_REACH)
            .min_by(|a, b| a.distance.total_cmp(&b.distance))
    }

    /// `ground` at terrain-local `(x, z)` as rivers cut it at a level whose
    /// samples lie `spacing` meters apart: below each river's water within
    /// its width (at least most of a sample wide, so coarse levels keep the
    /// cut; streams only their own, decision 097), its bed deepest at the
    /// middle, and its banks rising from the water at most [`BANK_SLOPE`] for
    /// [`BANK_REACH`] meters (streams' `STREAM_BANK_REACH`), the cut easing
    /// back to the ground over their outer half, each cut rounding into the
    /// ground over [`BANK_ROUNDING`] meters.
    pub fn cap(&self, x: f32, z: f32, ground: f32, spacing: f32) -> f32 {
        let point = Vec2::new(x, z);
        let mut height = ground;
        for &segment in self.near(point) {
            let river = self.on_segment(segment, point);
            if !river.surface.is_finite() {
                continue;
            }
            // Streams (decision 097) keep their own width at every level,
            // too narrow for coarse samples to carve (their water fades out
            // there), and cut their banks over a short reach.
            let stream = river.depth < MIN_DEPTH;
            let half = if stream {
                river.half
            } else {
                max(river.half, 0.75 * spacing)
            };
            let reach = if stream {
                STREAM_BANK_REACH
            } else {
                BANK_REACH
            };
            let cap = if river.distance <= half {
                let across = river.distance / half;
                river.surface - river.depth * (1.0 - across * across)
            } else if river.distance < half + reach {
                let out = (river.distance - half) / reach;
                let bank = river.surface + (river.distance - half) * BANK_SLOPE;
                // Easing back to the ground: no step where the cut ends.
                let ease = max(out * 2.0 - 1.0, 0.0);
                bank + (max(height, bank) - bank) * ease * ease * (3.0 - 2.0 * ease)
            } else {
                continue;
            };
            height = smooth_min(height, cap, BANK_ROUNDING);
        }
        height
    }

    /// The river water at terrain-local `(x, z)`: its surface and its depth
    /// over the bed there; `None` outside every river's width.
    pub fn water_at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        let point = Vec2::new(x, z);
        self.near(point)
            .iter()
            .map(|&segment| self.on_segment(segment, point))
            .filter(|river| river.surface.is_finite() && river.distance < river.half)
            .map(|river| {
                let across = river.distance / river.half;
                (river.surface, river.depth * (1.0 - across * across))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }
}

/// The lower of `a` and `b`, rounded where they come within `k` of each
/// other (a polynomial smooth minimum).
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - (a - b).abs(), 0.0) / k;
    min(a, b) - h * h * k * 0.25
}

/// A river's line from the bake's grid before its water is set: its
/// points, at each half its width and its depth at the middle, and whether
/// it begins at a head.
pub(crate) struct RiverLine {
    pub(crate) points: Vec<Vec2>,
    pub(crate) half: Vec<f32>,
    pub(crate) depth: Vec<f32>,
    pub(crate) head: bool,
}

/// What a bake's grid says of where water runs: upstream areas, receivers,
/// lake samples and the flood's rank of each sample (upstream samples rank
/// later).
pub(crate) struct Channels<'a> {
    pub(crate) area: &'a [f32],
    pub(crate) receivers: &'a [u32],
    pub(crate) lakes: &'a [bool],
    pub(crate) rank: &'a [u32],
}

/// The rivers of a grid `width` samples wide, `spacing` meters apart: each
/// channel (samples draining `channel_area` or more outside lakes) from its
/// head or a confluence down to the next confluence, a lake or the edge,
/// upstream rivers first; sized as `size` says for a sample's drained area
/// (half its width and its depth).
pub(crate) fn trace(
    channels: &Channels<'_>,
    (width, spacing): (usize, f32),
    channel_area: f32,
    size: impl Fn(f32) -> (f32, f32),
) -> Vec<RiverLine> {
    let Channels {
        area,
        receivers,
        lakes,
        rank,
    } = *channels;
    let count = area.len();
    let channel = |sample: usize| area[sample] >= channel_area && !lakes[sample];
    let mut inflows = vec![0_u8; count];
    for (sample, &receiver) in receivers.iter().enumerate() {
        if channel(sample) && receiver != u32::MAX {
            let entry = &mut inflows[receiver as usize];
            *entry = entry.saturating_add(1);
        }
    }
    // Heads (no channel flows in) and confluences (two or more do) start
    // rivers, upstream ones first.
    let mut starts: Vec<usize> = (0..count)
        .filter(|&sample| channel(sample) && inflows[sample] != 1)
        .collect();
    starts.sort_by_key(|&sample| std::cmp::Reverse(rank[sample]));
    let place = |sample: usize| {
        Vec2::new(
            (sample % width) as f32 * spacing,
            (sample / width) as f32 * spacing,
        )
    };
    let mut lines = Vec::with_capacity(starts.len());
    for start in starts {
        let mut course = vec![start];
        let mut current = start;
        loop {
            let receiver = receivers[current];
            if receiver == u32::MAX {
                break;
            }
            let next = receiver as usize;
            course.push(next);
            // Into a lake, off its channels, or into a confluence, which
            // starts its own river.
            if lakes[next] || !channel(next) || inflows[next] >= 2 {
                break;
            }
            current = next;
        }
        if course.len() < 2 {
            continue;
        }
        let points: Vec<Vec2> = course.iter().map(|&sample| place(sample)).collect();
        let sizes: Vec<(f32, f32)> = course.iter().map(|&sample| size(area[sample])).collect();
        let mut line = smooth_line(
            &points,
            &sizes.iter().map(|size| size.0).collect::<Vec<_>>(),
            &sizes.iter().map(|size| size.1).collect::<Vec<_>>(),
        );
        line.head = inflows[start] == 0;
        lines.push(line);
    }
    lines
}

/// The line through `points` with `half` widths and `depth`s rounded off
/// the grid (corner cutting, its ends kept) and resampled every
/// [`RIVER_STEP`] meters.
fn smooth_line(points: &[Vec2], half: &[f32], depth: &[f32]) -> RiverLine {
    let mut line: Vec<(Vec2, f32, f32)> = points
        .iter()
        .zip(half)
        .zip(depth)
        .map(|((&point, &half), &depth)| (point, half, depth))
        .collect();
    for _ in 0..SMOOTHING {
        if line.len() < 3 {
            break;
        }
        let mut cut = Vec::with_capacity(line.len() * 2);
        cut.push(line[0]);
        for pair in line.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let mix = |t: f32| {
                (
                    a.0.lerp(b.0, t),
                    a.1 + (b.1 - a.1) * t,
                    a.2 + (b.2 - a.2) * t,
                )
            };
            cut.push(mix(0.25));
            cut.push(mix(0.75));
        }
        cut.push(line[line.len() - 1]);
        // The first and last cuts fall within the ends' segments; the ends
        // stay where they were.
        cut.remove(1);
        cut.remove(cut.len() - 2);
        line = cut;
    }
    // Resampled every RIVER_STEP meters along its length.
    let mut along = vec![0.0_f32];
    for pair in line.windows(2) {
        along.push(along[along.len() - 1] + pair[0].0.distance(pair[1].0));
    }
    let total = along[along.len() - 1];
    let steps = max((total / RIVER_STEP).ceil(), 1.0) as usize;
    let mut out = RiverLine {
        points: Vec::with_capacity(steps + 1),
        half: Vec::with_capacity(steps + 1),
        depth: Vec::with_capacity(steps + 1),
        head: false,
    };
    let mut segment = 0;
    for step in 0..=steps {
        let at = total * step as f32 / steps as f32;
        while segment + 2 < along.len() && along[segment + 1] < at {
            segment += 1;
        }
        let run = along[segment + 1] - along[segment];
        let t = if run > 0.0 {
            min(max((at - along[segment]) / run, 0.0), 1.0)
        } else {
            0.0
        };
        let (a, b) = (line[segment], line[segment + 1]);
        out.points.push(a.0.lerp(b.0, t));
        out.half.push(a.1 + (b.1 - a.1) * t);
        out.depth.push(a.2 + (b.2 - a.2) * t);
    }
    out
}

/// The way across a river's line at point `index` (a unit vector, left of
/// the way it runs).
pub(crate) fn across(points: &[Vec2], index: usize) -> Vec2 {
    let last = points.len() - 1;
    let (a, b) = (
        points[index.saturating_sub(1)],
        points[(index + 1).min(last)],
    );
    let run = (b - a).normalize_or(Vec2::X);
    Vec2::new(-run.y, run.x)
}

/// River `line` with its water set from the ground read at its points
/// (`ground`: at its middle and each bank, terrain-local meters): below the
/// lowest of them by [`BANK_MARGIN`], never rising downstream, starting no
/// higher than `incoming` (the rivers meeting where it starts); along
/// streams (points shallower than [`MIN_DEPTH`], decision 097) just below
/// the lowest by `STREAM_MARGIN`, rising with the ground. Its speed comes
/// from the water's fall, and rivers' stretches falling steeper than
/// `fall_slope` are marked; streams tumble down theirs.
pub(crate) fn profile(
    line: &RiverLine,
    ground: &[[f32; 3]],
    incoming: Option<f32>,
    fall_slope: f32,
) -> River {
    let count = line.points.len();
    let mut surface = Vec::with_capacity(count);
    let mut level = incoming.unwrap_or(f32::INFINITY);
    let stream = |index: usize| line.depth[index] < MIN_DEPTH;
    for (index, read) in ground.iter().take(count).enumerate() {
        let lowest = min(read[0], min(read[1], read[2]));
        level = if stream(index) {
            lowest - STREAM_MARGIN
        } else {
            min(level, lowest - BANK_MARGIN)
        };
        surface.push(level);
    }
    let mut speed = Vec::with_capacity(count);
    let mut falls = Vec::with_capacity(count);
    for index in 0..count {
        let next = (index + 1).min(count - 1);
        let run = max(line.points[index].distance(line.points[next]), 1.0e-3);
        let drop = max(surface[index] - surface[next], 0.0);
        let slope = drop / run;
        // Steeper water runs faster, up to a torrent.
        speed.push(min(0.6 + 12.0 * slope.sqrt(), 4.0));
        falls.push(next != index && slope > fall_slope && !stream(index));
    }
    River {
        points: line.points.iter().map(|point| point.to_array()).collect(),
        surface,
        half: line.half.clone(),
        depth: line.depth.clone(),
        speed,
        falls,
        head: line.head,
    }
}

crate::codec::record!(River {
    points,
    surface,
    half,
    depth,
    speed,
    falls,
    head,
});
crate::codec::record!(Rivers {
    rivers,
    min,
    columns,
    rows,
    starts,
    segments,
});
