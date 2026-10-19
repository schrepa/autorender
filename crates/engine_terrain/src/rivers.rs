//! Rivers: the channels a hydrology bake
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
//!
//! Tiles read rivers a row of samples at a time: a
//! [`RiverSpan`] holds the segments listed in any cell holding one of the
//! tile's samples, in the grid's order, and each row visits a segment's
//! samples within its reach alone, in the order reading sample by sample
//! takes them; samples beyond a segment's reach, which reading sample by
//! sample skips after measuring them, are never measured. Rows read the
//! same bits as samples read one by one.

use std::ops::Range;

use engine_math::{max, min};
use glam::Vec2;
use serde::Serialize;

/// Meters between a river's points.
pub const RIVER_STEP: f32 = 3.0;
/// Segments of a river whose water [`Rivers::differs_from`] bounds at a
/// time.
const STRETCH: usize = 16;
/// Meters below the lowest of a river's ground and banks its water stands.
pub(crate) const BANK_MARGIN: f32 = 0.25;
/// Meters beyond a river's edge its banks are read at.
pub const BANK_OFFSET: f32 = 1.0;
/// How steeply banks rise from a channel's top at most (meters a meter),
/// the meters from the edge they are cut over (the cut easing back to the
/// ground over their outer half; ground below the top rises to it at the
/// edge, easing back over the whole reach), and the meters over which a
/// cut rounds into the ground it meets.
pub(crate) const BANK_SLOPE: f32 = 0.6;
pub(crate) const BANK_REACH: f32 = 12.0;
const BANK_ROUNDING: f32 = 1.5;
/// The least water a river holds at its middle (meters): shallower points
/// are streams'.
pub const MIN_DEPTH: f32 = 0.35;
/// Samples past a river's edge its water's surface reaches in a tile, under
/// the banks, so the triangle holding its shore has water at every corner.
pub(crate) const WATER_PAST: f32 = 1.5;
/// Meters below the lowest of a stream's ground and banks its water stands,
/// following the ground rather than never rising, and the
/// meters from its edge its banks are cut over: a stream cuts its course a
/// little into the ground instead of a trench through every rise.
pub(crate) const STREAM_MARGIN: f32 = 0.12;
const STREAM_BANK_REACH: f32 = 2.5;
/// Rounds of corner cutting that smooth a river's line off the grid.
const SMOOTHING: usize = 3;
/// Meters across the cells of the grid that finds the rivers near a point.
pub(crate) const CELL: f32 = 32.0;
/// How far past a reading's own reach rows still measure a segment:
/// a share and meters, so rounding never skips a sample
/// reading sample by sample would take.
const REACH_SLACK: [f32; 2] = [1.0e-4, 1.0e-3];

/// The reach `reach` with [`REACH_SLACK`], and its square.
fn slack(reach: f32) -> (f32, f32) {
    let reach = reach * (1.0 + REACH_SLACK[0]) + REACH_SLACK[1];
    (reach, reach * reach)
}

/// One river or stream between confluences: its points
/// (terrain-local x and z, from upstream), and at each its water's surface
/// (terrain-local meters), half its channel's width, the water's depth at
/// its middle, its channel's depth there below its banks as its drained
/// area sizes it, the share of the channel's half width its water spans
/// (its channel shaped to that, [`Rivers::cap`]), its mean speed
/// downstream (meters a second), the discharge it carries (cubic meters a
/// second) and how much foam rides it (0 to 1), as [`crate::hydraulics`]
/// solves its flow in its channel. Where it leaves its bed over a
/// drop and falls through the air its points carry no surface (`NaN`) and
/// the stretches between them are `falls`: a [`crate::Fall`] holds its jet.
/// Its channel is cut to its water's surface, and beneath its jets straight
/// from the lip's water to where the jet lands (`cut`), so no ground stands
/// between the pools a fall leaves and lands in.
/// A river beginning at a head, where no channel flows in, rather than at a
/// confluence or a lake's outlet, is `head`: its water starts there.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct River {
    pub points: Vec<[f32; 2]>,
    pub surface: Vec<f32>,
    pub cut: Vec<f32>,
    pub half: Vec<f32>,
    pub depth: Vec<f32>,
    pub channel: Vec<f32>,
    pub fill: Vec<f32>,
    pub speed: Vec<f32>,
    pub discharge: Vec<f32>,
    pub foam: Vec<f32>,
    pub falls: Vec<bool>,
    pub head: bool,
}

impl River {
    /// Whether the stretch from point `index` to the next falls.
    pub(crate) fn falls_at(&self, index: usize) -> bool {
        self.falls.get(index).copied().unwrap_or(false)
    }

    /// The way the river runs (a unit vector along x and z) `t` of the way
    /// along its segment from point `segment`: the ways it runs at the
    /// segment's ends blended, each the mean of the segments meeting there,
    /// so it turns smoothly through its bends and the segments on either
    /// side of a bend agree where they meet.
    pub(crate) fn heading(&self, segment: usize, t: f32) -> Vec2 {
        let run = |index: usize| {
            (Vec2::from(self.points[index + 1]) - Vec2::from(self.points[index]))
                .normalize_or_zero()
        };
        let last = self.points.len().saturating_sub(2);
        let at_point = |point: usize| {
            let before = run(point.saturating_sub(1).min(last));
            let after = run(point.min(last));
            (before + after).try_normalize().unwrap_or(after)
        };
        let (start, end) = (at_point(segment), at_point(segment + 1));
        (start + (end - start) * t)
            .try_normalize()
            .unwrap_or_else(|| run(segment.min(last)))
    }
}

/// A river as a point sees it: meters from its line, and there its water's
/// surface, the level its channel is cut to, half its width, its water's
/// depth at its middle and its
/// channel's, the share of its half width its water spans, its speed, the
/// way it runs (a unit vector along x and z), the discharge it carries and
/// the foam riding it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RiverPoint {
    pub distance: f32,
    pub surface: f32,
    pub cut: f32,
    pub half: f32,
    pub depth: f32,
    pub channel: f32,
    pub fill: f32,
    pub speed: f32,
    pub direction: [f32; 2],
    pub discharge: f32,
    pub foam: f32,
}

/// A river's point a place lies at: the river and point, meters across from
/// the river's middle line there (positive to its right, looking
/// downstream) and the way it runs (a unit vector along x and z).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Station {
    pub river: u32,
    pub point: u32,
    pub across: f32,
    pub direction: [f32; 2],
}

/// A terrain's rivers and a grid of the river segments within cutting
/// reach of each cell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rivers {
    pub(crate) rivers: Vec<River>,
    pub(crate) min: Vec2,
    pub(crate) columns: usize,
    pub(crate) rows: usize,
    pub(crate) starts: Vec<u32>,
    pub(crate) segments: Vec<[u32; 2]>,
}

impl Rivers {
    /// `rivers` with the grid that finds them.
    pub(crate) fn new(rivers: Vec<River>) -> Self {
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
                // keep their width, so they reach no further
                // than a river's banks and list in far fewer cells.
                let stream = max(river.channel[segment], river.channel[segment + 1]) < MIN_DEPTH;
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

    pub(crate) fn rivers(&self) -> &[River] {
        &self.rivers
    }

    /// How many rivers there are.
    pub fn len(&self) -> usize {
        self.rivers.len()
    }

    /// Where these rivers' water differs from `other`'s: the bounds (lowest
    /// and highest x and z) of each stretch of 16 segments whose
    /// water does, out as far as either's widest half there; everywhere
    /// either reaches where they are other rivers than the same ones along
    /// the same lines (as dams hold them back); none where no water
    /// differs.
    pub fn differs_from(&self, other: &Self) -> Vec<(Vec2, Vec2)> {
        let same_lines = self.rivers.len() == other.rivers.len()
            && self
                .rivers
                .iter()
                .zip(&other.rivers)
                .all(|(a, b)| a.points == b.points);
        if !same_lines {
            let ((a, b), (c, d)) = (self.bounds(), other.bounds());
            return vec![(a.min(c), b.max(d))];
        }
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
        let mut regions = Vec::new();
        for (a, b) in self.rivers.iter().zip(&other.rivers) {
            let count = a.points.len();
            let mut first = 0;
            while first + 1 < count {
                let points = first..(first + STRETCH + 1).min(count);
                first = points.end - 1;
                let differs = [
                    (&a.surface, &b.surface),
                    (&a.half, &b.half),
                    (&a.depth, &b.depth),
                    (&a.speed, &b.speed),
                    (&a.foam, &b.foam),
                ]
                .iter()
                .any(|(a, b)| !same(&a[points.clone()], &b[points.clone()]));
                if !differs {
                    continue;
                }
                let widest = a.half[points.clone()]
                    .iter()
                    .chain(&b.half[points.clone()])
                    .fold(0.0, |m, &h| max(m, h));
                let (mut low, mut high) =
                    (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
                for &point in &a.points[points] {
                    low = low.min(Vec2::from(point));
                    high = high.max(Vec2::from(point));
                }
                regions.push((low - Vec2::splat(widest), high + Vec2::splat(widest)));
            }
        }
        regions
    }

    /// These rivers' grid holding `rivers` instead: the same rivers along the
    /// same lines, whose water a pool or a changed flow raised or lowered.
    /// Their widths stay within the banks the grid reaches.
    pub(crate) fn with_rivers(&self, rivers: Vec<River>) -> Self {
        Self {
            rivers,
            min: self.min,
            columns: self.columns,
            rows: self.rows,
            starts: self.starts.clone(),
            segments: self.segments.clone(),
        }
    }

    /// The river point nearest terrain-local `(x, z)`, within a river's
    /// banks.
    pub(crate) fn station(&self, x: f32, z: f32) -> Option<Station> {
        let point = Vec2::new(x, z);
        let mut nearest: Option<(f32, Station)> = None;
        for &[river, segment] in self.near(point) {
            let line = &self.rivers[river as usize];
            let index = segment as usize;
            let (a, b) = (
                Vec2::from(line.points[index]),
                Vec2::from(line.points[index + 1]),
            );
            let run = b - a;
            let length = run.length();
            if length <= 0.0 {
                continue;
            }
            let t = min(max((point - a).dot(run) / (length * length), 0.0), 1.0);
            let gap = point - (a + run * t);
            let distance = gap.length();
            let half = line.half[index] + (line.half[index + 1] - line.half[index]) * t;
            if distance > half + BANK_REACH || nearest.is_some_and(|(best, _)| best <= distance) {
                continue;
            }
            let direction = line.heading(index, t);
            nearest = Some((
                distance,
                Station {
                    river,
                    point: if t < 0.5 { segment } else { segment + 1 },
                    across: gap.dot(Vec2::new(-direction.y, direction.x)),
                    direction: direction.to_array(),
                },
            ));
        }
        nearest.map(|(_, station)| station)
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
        let direction = if length > 0.0 {
            river.heading(segment, t)
        } else {
            Vec2::X
        };
        RiverPoint {
            distance: point.distance(a + run * t),
            surface: at(&river.surface),
            cut: at(&river.cut),
            half: at(&river.half),
            depth: at(&river.depth),
            channel: at(&river.channel),
            fill: at(&river.fill),
            speed: at(&river.speed),
            direction: direction.to_array(),
            discharge: at(&river.discharge),
            foam: at(&river.foam),
        }
    }

    /// Whether `point` lies beyond `reach` (with [`REACH_SLACK`]) of river
    /// `river`'s segment `segment`, measured as [`Self::on_segment`]
    /// measures it, without its square root.
    fn beyond(&self, [river, segment]: [u32; 2], point: Vec2, reach: impl Fn(f32) -> f32) -> bool {
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
        let gap = point - (a + run * t);
        let (_, squared) = slack(reach(max(river.half[segment], river.half[segment + 1])));
        gap.dot(gap) > squared
    }

    /// The river nearest terrain-local `(x, z)` within cutting reach.
    pub(crate) fn nearest(&self, x: f32, z: f32) -> Option<RiverPoint> {
        let point = Vec2::new(x, z);
        self.near(point)
            .iter()
            .filter(|&&segment| !self.beyond(segment, point, |half| half + BANK_REACH))
            .map(|&segment| self.on_segment(segment, point))
            .filter(|river| river.distance < river.half + BANK_REACH)
            .min_by(|a, b| a.distance.total_cmp(&b.distance))
    }

    /// The cell of terrain-local `point` the grid lists segments in, as
    /// [`Self::near`] finds it: its column and row, `None` off the grid.
    fn cell(&self, point: Vec2) -> Option<(usize, usize)> {
        let cell = ((point - self.min) / CELL).floor();
        if !(cell.x >= 0.0 && cell.y >= 0.0) {
            return None;
        }
        let (column, row) = (cell.x as usize, cell.y as usize);
        (column < self.columns && row < self.rows).then_some((column, row))
    }

    /// The segments near a tile's samples, at x positions
    /// `xs` and z positions `zs` (both ascending) `spacing` meters apart.
    pub(crate) fn span(&self, xs: &[f32], zs: &[f32], spacing: f32) -> RiverSpan {
        let (Some(&x0), Some(&x1), Some(&z0), Some(&z1)) =
            (xs.first(), xs.last(), zs.first(), zs.last())
        else {
            return RiverSpan::default();
        };
        if self.segments.is_empty() {
            return RiverSpan::default();
        }
        // Every cell holding a sample lies between the corners' cells.
        let clamp_cell = |point: Vec2| {
            let cell = ((point - self.min) / CELL).floor();
            (
                min(max(cell.x, 0.0), self.columns as f32 - 1.0) as usize,
                min(max(cell.y, 0.0), self.rows as f32 - 1.0) as usize,
            )
        };
        let (first_column, first_row) = clamp_cell(Vec2::new(x0, z0));
        let (last_column, last_row) = clamp_cell(Vec2::new(x1, z1));
        // The cells' lists, unless they hold more entries than there are
        // segments (far tiles): then every segment whose listing meets them.
        let entries: usize = (first_row..=last_row)
            .map(|row| {
                let at = row * self.columns;
                (self.starts[at + last_column + 1] - self.starts[at + first_column]) as usize
            })
            .sum();
        let count: usize = self
            .rivers
            .iter()
            .map(|river| river.points.len().saturating_sub(1))
            .sum();
        let mut listed: Vec<[u32; 2]> = Vec::new();
        if entries <= count {
            for row in first_row..=last_row {
                for column in first_column..=last_column {
                    let at = row * self.columns + column;
                    listed.extend_from_slice(
                        &self.segments[self.starts[at] as usize..self.starts[at + 1] as usize],
                    );
                }
            }
            listed.sort_unstable();
            listed.dedup();
        } else {
            for (index, river) in self.rivers.iter().enumerate() {
                for segment in 0..river.points.len().saturating_sub(1) {
                    let (columns, rows) = self.listing(river, segment);
                    if columns.0 <= last_column
                        && columns.1 >= first_column
                        && rows.0 <= last_row
                        && rows.1 >= first_row
                    {
                        listed.push([index as u32, segment as u32]);
                    }
                }
            }
        }
        let columns: Vec<Option<usize>> = xs
            .iter()
            .map(|&x| self.cell(Vec2::new(x, z0)).map(|(column, _)| column))
            .collect();
        let rows: Vec<Option<usize>> = zs
            .iter()
            .map(|&z| self.cell(Vec2::new(x0, z)).map(|(_, row)| row))
            .collect();
        let mut segments = Vec::with_capacity(listed.len());
        let mut by_row: Vec<Vec<u32>> = vec![Vec::new(); zs.len()];
        for [river_index, segment] in listed {
            let river = &self.rivers[river_index as usize];
            let index = segment as usize;
            let (a, b) = (
                Vec2::from(river.points[index]),
                Vec2::from(river.points[index + 1]),
            );
            let half = max(river.half[index], river.half[index + 1]);
            // Rows within the widest reach any reading takes, the cut's or
            // its water's.
            let (reach, _) =
                slack(max(half, 0.75 * spacing) + max(BANK_REACH, WATER_PAST * spacing));
            let (low, high) = (a.min(b), a.max(b));
            let from = zs.partition_point(|&z| z < low.y - reach);
            let to = zs.partition_point(|&z| z <= high.y + reach);
            if from >= to {
                continue;
            }
            let (cells_x, cells_z) = self.listing(river, index);
            let run = b - a;
            for row in &mut by_row[from..to] {
                row.push(segments.len() as u32);
            }
            segments.push(SpanSegment {
                river: river_index,
                segment,
                a,
                run,
                length: run.length(),
                low,
                high,
                half,
                cells: [cells_x.0, cells_x.1, cells_z.0, cells_z.1],
            });
        }
        RiverSpan {
            segments,
            columns,
            rows,
            by_row,
        }
    }

    /// The cells listing river `river`'s segment `index`, as [`Self::new`]
    /// lists them: first and last column, and first and last row.
    fn listing(&self, river: &River, index: usize) -> ((usize, usize), (usize, usize)) {
        let (a, b) = (
            Vec2::from(river.points[index]),
            Vec2::from(river.points[index + 1]),
        );
        let half = max(river.half[index], river.half[index + 1]);
        let stream = max(river.channel[index], river.channel[index + 1]) < MIN_DEPTH;
        let margin = half + BANK_REACH + if stream { 0.0 } else { 64.0 };
        let first = ((a.min(b) - Vec2::splat(margin) - self.min) / CELL).floor();
        let last = ((a.max(b) + Vec2::splat(margin) - self.min) / CELL).floor();
        (
            (
                max(first.x, 0.0) as usize,
                min(last.x, self.columns as f32 - 1.0) as usize,
            ),
            (
                max(first.y, 0.0) as usize,
                min(last.y, self.rows as f32 - 1.0) as usize,
            ),
        )
    }

    /// Where row `row` of `span`'s tile reads segment `near`:
    /// the range of its samples within `reach` of it, x positions `xs`;
    /// empty off the cells listing it.
    fn span_samples(
        span: &RiverSpan,
        near: &SpanSegment,
        row: usize,
        xs: &[f32],
        reach: f32,
    ) -> std::ops::Range<usize> {
        let listed =
            span.rows[row].is_some_and(|cell| cell >= near.cells[2] && cell <= near.cells[3]);
        if !listed {
            return 0..0;
        }
        let from = xs.partition_point(|&x| x < near.low.x - reach);
        let to = xs.partition_point(|&x| x <= near.high.x + reach);
        from..to.max(from)
    }

    /// [`Self::cap`] of row `row` of `span`'s tile: its
    /// samples' `heights`, at x positions `xs` and z position `z`, in place.
    pub(crate) fn cap_row(
        &self,
        span: &RiverSpan,
        row: usize,
        (xs, z): (&[f32], f32),
        heights: &mut [f32],
        spacing: f32,
    ) {
        let mut shaped = vec![Shaping::NONE; heights.len()];
        for &index in &span.by_row[row] {
            let near = &span.segments[index as usize];
            let (reach, squared) = slack(max(near.half, 0.75 * spacing) + BANK_REACH);
            let river = &self.rivers[near.river as usize];
            let segment = near.segment as usize;
            for sample in Self::span_samples(span, near, row, xs, reach) {
                let listed = span.columns[sample]
                    .is_some_and(|cell| cell >= near.cells[0] && cell <= near.cells[1]);
                if !listed {
                    continue;
                }
                let point = Vec2::new(xs[sample], z);
                let t = near.project(point);
                let gap = point - (near.a + near.run * t);
                let distance_squared = gap.dot(gap);
                if distance_squared > squared {
                    continue;
                }
                let at =
                    |values: &[f32]| values[segment] + (values[segment + 1] - values[segment]) * t;
                let point = RiverCut {
                    distance: distance_squared.sqrt(),
                    surface: at(&river.cut),
                    half: at(&river.half),
                    depth: at(&river.depth),
                    channel: at(&river.channel),
                    fill: at(&river.fill),
                };
                shaped[sample].add(&point, heights[sample], spacing);
            }
        }
        for (height, shaping) in heights.iter_mut().zip(&shaped) {
            *height = shaping.height(*height);
        }
    }

    /// The rivers' water over row `row` of `span`'s tile, at x positions
    /// `xs` and z position `z`, `spacing` meters apart: into `best`, for
    /// each sample of `columns` within a river's water or [`WATER_PAST`]
    /// samples past its edge, the river it lies deepest within, and its
    /// water's surface, flow and whitewater there. A river's water is as wide as its cut
    /// ([`Self::cap`]): streams their own width, rivers at least most of a
    /// sample; airborne stretches hold none.
    pub(crate) fn water_row(
        &self,
        span: &RiverSpan,
        row: usize,
        (xs, z, columns): (&[f32], f32, Range<usize>),
        spacing: f32,
        best: &mut [RiverWater],
    ) {
        for &index in &span.by_row[row] {
            let near = &span.segments[index as usize];
            let river = &self.rivers[near.river as usize];
            let segment = near.segment as usize;
            let (surfaces, halves) = (
                [river.surface[segment], river.surface[segment + 1]],
                [river.half[segment], river.half[segment + 1]],
            );
            if !(surfaces[0].is_finite() && surfaces[1].is_finite()) {
                continue;
            }
            let stream = max(river.channel[segment], river.channel[segment + 1]) < MIN_DEPTH;
            let widest = if stream {
                near.half
            } else {
                max(near.half, 0.75 * spacing)
            };
            let past = if stream {
                min(WATER_PAST * spacing, BANK_REACH)
            } else {
                WATER_PAST * spacing
            };
            let (reach, squared) = slack(widest + past);
            let samples = Self::span_samples(span, near, row, xs, reach);
            for sample in samples.start.max(columns.start)..samples.end.min(columns.end) {
                let listed = span.columns[sample]
                    .is_some_and(|cell| cell >= near.cells[0] && cell <= near.cells[1]);
                if !listed {
                    continue;
                }
                let point = Vec2::new(xs[sample], z);
                let t = near.project(point);
                let gap = point - (near.a + near.run * t);
                let distance_squared = gap.dot(gap);
                if distance_squared > squared {
                    continue;
                }
                let along = |values: [f32; 2]| values[0] + (values[1] - values[0]) * t;
                let half = along(halves);
                let half = if stream {
                    half
                } else {
                    max(half, 0.75 * spacing)
                };
                let inside = distance_squared.sqrt() - half;
                if inside >= best[sample].inside || inside > past {
                    continue;
                }
                let speed = along([river.speed[segment], river.speed[segment + 1]]);
                let surface = along(surfaces);
                let bed = surface - along([river.depth[segment], river.depth[segment + 1]]);
                best[sample] = RiverWater {
                    inside,
                    surface,
                    bed,
                    flow: river.heading(segment, t) * speed,
                    white: along([river.foam[segment], river.foam[segment + 1]]),
                    width: half * along([river.fill[segment], river.fill[segment + 1]]),
                };
            }
        }
    }

    /// For each sample of row `row` of `span`'s tile at x positions `xs`
    /// and z position `z`, the nearest river within cutting reach as
    /// [`Self::nearest`] finds it: meters from its edge
    /// (its distance less half its width), `far` where none is.
    pub(crate) fn nearest_row(
        &self,
        span: &RiverSpan,
        row: usize,
        (xs, z): (&[f32], f32),
        far: f32,
        out: &mut [f32],
    ) {
        out.fill(far);
        let mut nearest = vec![f32::INFINITY; out.len()];
        for &index in &span.by_row[row] {
            let near = &span.segments[index as usize];
            let (reach, squared) = slack(near.half + BANK_REACH);
            let river = &self.rivers[near.river as usize];
            let segment = near.segment as usize;
            for sample in Self::span_samples(span, near, row, xs, reach) {
                let listed = span.columns[sample]
                    .is_some_and(|cell| cell >= near.cells[0] && cell <= near.cells[1]);
                if !listed {
                    continue;
                }
                let point = Vec2::new(xs[sample], z);
                let t = near.project(point);
                let gap = point - (near.a + near.run * t);
                let distance_squared = gap.dot(gap);
                if distance_squared > squared {
                    continue;
                }
                let distance = distance_squared.sqrt();
                let half =
                    river.half[segment] + (river.half[segment + 1] - river.half[segment]) * t;
                // The first nearest in the grid's order, as `min_by` keeps.
                if distance < half + BANK_REACH && distance.total_cmp(&nearest[sample]).is_lt() {
                    nearest[sample] = distance;
                    out[sample] = distance - half;
                }
            }
        }
    }

    /// `ground` at terrain-local `(x, z)` as rivers shape it at a level
    /// whose samples lie `spacing` meters apart. Within a river's channel
    /// (at least most of a sample wide, so coarse levels keep it; streams
    /// only their own width) the ground is the channel's: its bed under
    /// the water at the middle, rising through the water's surface at the
    /// share of its half width the water spans to its top at its edges,
    /// cut into the ground or built up out of it alike, so its water always
    /// stands in it and meets its banks before its edges; where channels overlap, the one the point lies deepest within
    /// holds it, as it holds the water there. Beyond, banks rise from the
    /// top at most `BANK_SLOPE` for `BANK_REACH` meters (streams'
    /// `STREAM_BANK_REACH`), the cut easing back to the ground over their
    /// outer half and rounding into it over `BANK_ROUNDING` meters, and
    /// ground below the top rises to it at the edge, easing back over the
    /// reach.
    pub(crate) fn cap(&self, x: f32, z: f32, ground: f32, spacing: f32) -> f32 {
        let point = Vec2::new(x, z);
        let mut shaping = Shaping::NONE;
        for &segment in self.near(point) {
            // Segments beyond any cut's reach leave it alone.
            if self.beyond(segment, point, |half| {
                max(half, 0.75 * spacing) + BANK_REACH
            }) {
                continue;
            }
            let river = self.on_segment(segment, point);
            let cut = RiverCut {
                distance: river.distance,
                surface: river.cut,
                half: river.half,
                depth: river.depth,
                channel: river.channel,
                fill: river.fill,
            };
            shaping.add(&cut, ground, spacing);
        }
        shaping.height(ground)
    }

    /// The river water at terrain-local `(x, z)` as bakes after the
    /// hydrology's read it before there are tiles: its surface and its depth
    /// over the channel the river cuts there; `None` outside every river's
    /// width or where its channel holds none.
    pub(crate) fn water_at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        let point = Vec2::new(x, z);
        self.near(point)
            .iter()
            .filter(|&&segment| !self.beyond(segment, point, |half| half))
            .map(|&segment| self.on_segment(segment, point))
            .filter(|river| river.surface.is_finite() && river.distance < river.half)
            .map(|river| {
                let across = river.distance / (river.half * river.fill);
                (river.surface, river.depth * (1.0 - across * across))
            })
            .filter(|&(_, depth)| depth > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }
}

/// The segments a tile's samples read rivers through: those
/// the grid lists in any cell holding a sample, in the grid's order; each
/// x and z position's cell column and row (`None` off the grid); and for
/// each row of samples, the segments whose reach it crosses.
#[derive(Clone, Debug, Default)]
pub(crate) struct RiverSpan {
    segments: Vec<SpanSegment>,
    columns: Vec<Option<usize>>,
    rows: Vec<Option<usize>>,
    by_row: Vec<Vec<u32>>,
}

/// A segment as a tile's rows read it: its river and index,
/// its first end, run and length as [`Rivers::on_segment`] measures them,
/// the box its ends span, its widest half width and the cells listing it
/// (first and last column, then row).
#[derive(Clone, Copy, Debug)]
struct SpanSegment {
    river: u32,
    segment: u32,
    a: Vec2,
    run: Vec2,
    length: f32,
    low: Vec2,
    high: Vec2,
    half: f32,
    cells: [usize; 4],
}

impl SpanSegment {
    /// How far along it the point nearest `point` lies, 0 to 1, as
    /// [`Rivers::on_segment`] finds it.
    #[inline(always)]
    fn project(&self, point: Vec2) -> f32 {
        if self.length > 0.0 {
            min(
                max(
                    (point - self.a).dot(self.run) / (self.length * self.length),
                    0.0,
                ),
                1.0,
            )
        } else {
            0.0
        }
    }
}

/// A river's water at a sample as [`Rivers::water_row`] finds it: meters
/// from its edge (negative within its water), and there its surface and
/// its channel's bed (terrain-local meters), its flow along x and z (meters
/// a second), its whitewater (0 to 1) and the half width of its water
/// (meters).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RiverWater {
    pub(crate) inside: f32,
    pub(crate) surface: f32,
    pub(crate) bed: f32,
    pub(crate) flow: Vec2,
    pub(crate) white: f32,
    pub(crate) width: f32,
}

impl RiverWater {
    /// No river's water.
    pub(crate) const NONE: Self = Self {
        inside: f32::INFINITY,
        surface: f32::NAN,
        bed: f32::NAN,
        flow: Vec2::ZERO,
        white: 0.0,
        width: 0.0,
    };
}

/// What [`Rivers::cap`] reads of a segment at a point: its distance, and
/// there the level its channel is cut to (its water's surface, or beneath
/// a jet the line it is cut to), half the river's width, the water's depth,
/// the channel's and the share of its half width the water spans. Streams
/// keep their own width at every level, too narrow for coarse samples to
/// carve (their water fades out there), and shape their banks over a short
/// reach.
struct RiverCut {
    distance: f32,
    surface: f32,
    half: f32,
    depth: f32,
    channel: f32,
    fill: f32,
}

impl RiverCut {
    /// How deep the channel is shaped below its top: as deep as its water's
    /// parabola from the bed through the surface at the share of the half
    /// width the water spans reaches at its edges.
    #[inline(always)]
    fn shaped(&self) -> f32 {
        self.depth / (self.fill * self.fill)
    }

    /// Half the channel's width at a level whose samples lie `spacing`
    /// meters apart, and the meters beyond its edge its banks reach.
    fn widths(&self, spacing: f32) -> (f32, f32) {
        if self.channel < MIN_DEPTH {
            (self.half, STREAM_BANK_REACH)
        } else {
            (max(self.half, 0.75 * spacing), BANK_REACH)
        }
    }

    /// Where the channel holds the point at a level whose samples lie
    /// `spacing` meters apart: meters within its edge (0 or less) and the
    /// channel's ground there, its bed under the water at the middle rising
    /// to its top at the edges; `None` outside it or where its water is
    /// airborne.
    #[inline(always)]
    fn channel(&self, spacing: f32) -> Option<(f32, f32)> {
        let (half, _) = self.widths(spacing);
        if !self.surface.is_finite() || self.distance > half {
            return None;
        }
        let across = self.distance / half;
        let bed = self.surface - self.depth;
        Some((self.distance - half, bed + self.shaped() * across * across))
    }

    /// `height` beyond the channel's edge as its banks shape it at a level
    /// whose samples lie `spacing` meters apart: ground above the channel's
    /// top cut to a bank rising from it, ground below it built up to it at
    /// the edge, each easing back to the ground; `None` beyond their reach,
    /// within the channel or where its water is airborne.
    #[inline(always)]
    fn bank(&self, height: f32, spacing: f32) -> Option<f32> {
        let (half, reach) = self.widths(spacing);
        let beyond = self.distance - half;
        if !self.surface.is_finite() || beyond <= 0.0 || beyond >= reach {
            return None;
        }
        let top = self.surface - self.depth + self.shaped();
        let out = beyond / reach;
        if height < top {
            return Some(top + (height - top) * smoothstep(out));
        }
        // Rounding no more than the bank is wide, so the edge keeps the
        // channel's top.
        let cut = smooth_min(
            height,
            top + beyond * BANK_SLOPE,
            max(min(BANK_ROUNDING, beyond), 1.0e-3),
        );
        Some(cut + (height - cut) * smoothstep(max(out * 2.0 - 1.0, 0.0)))
    }
}

/// How rivers shape the ground at a point: the channel holding it (meters
/// within its edge and its ground there) and its banks' deepest cut and
/// highest build.
#[derive(Clone, Copy, Debug)]
struct Shaping {
    inside: f32,
    channel: f32,
    cut: f32,
    built: f32,
}

impl Shaping {
    /// No river near.
    const NONE: Self = Self {
        inside: f32::INFINITY,
        channel: f32::NAN,
        cut: f32::INFINITY,
        built: f32::NEG_INFINITY,
    };

    /// Adds segment `cut` at ground `height`, at a level whose samples lie
    /// `spacing` meters apart: the first channel the point lies deepest
    /// within holds it, as [`Rivers::water_row`] gives it that channel's
    /// water.
    #[inline(always)]
    fn add(&mut self, cut: &RiverCut, height: f32, spacing: f32) {
        if let Some((inside, channel)) = cut.channel(spacing) {
            if inside < self.inside {
                self.inside = inside;
                self.channel = channel;
            }
        } else if let Some(shaped) = cut.bank(height, spacing) {
            if shaped > height {
                self.built = max(self.built, shaped);
            } else {
                self.cut = min(self.cut, shaped);
            }
        }
    }

    /// Ground `height` as the rivers shape it: the channel's where one
    /// holds it, else built up where a bank builds, else cut.
    fn height(&self, height: f32) -> f32 {
        if self.inside.is_finite() {
            self.channel
        } else if self.built > height {
            self.built
        } else {
            min(height, self.cut)
        }
    }
}

/// The lower of `a` and `b`, rounded where they come within `k` of each
/// other (a polynomial smooth minimum).
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - (a - b).abs(), 0.0) / k;
    min(a, b) - h * h * k * 0.25
}

/// 0 at 0 rising smoothly to 1 at 1.
fn smoothstep(t: f32) -> f32 {
    let t = min(max(t, 0.0), 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A river's line from the bake's grid before its water is set: its
/// points, at each half its width, its channel's depth at the middle and
/// the area draining through it (square meters), and whether it begins at a
/// head.
pub(crate) struct RiverLine {
    pub(crate) points: Vec<Vec2>,
    pub(crate) half: Vec<f32>,
    pub(crate) depth: Vec<f32>,
    pub(crate) area: Vec<f32>,
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
        let drained: Vec<f32> = course.iter().map(|&sample| area[sample]).collect();
        let sizes: Vec<(f32, f32)> = drained.iter().map(|&drained| size(drained)).collect();
        let mut line = smooth_line(
            &points,
            [
                &sizes.iter().map(|size| size.0).collect::<Vec<_>>(),
                &sizes.iter().map(|size| size.1).collect::<Vec<_>>(),
                &drained,
            ],
        );
        line.head = inflows[start] == 0;
        lines.push(line);
    }
    lines
}

/// The line through `points` with their `values` (half widths, depths
/// and drained areas) rounded off the grid (corner cutting, its ends kept)
/// and resampled every [`RIVER_STEP`] meters.
fn smooth_line(points: &[Vec2], values: [&[f32]; 3]) -> RiverLine {
    let mut line: Vec<(Vec2, [f32; 3])> = points
        .iter()
        .enumerate()
        .map(|(index, &point)| (point, values.map(|value| value[index])))
        .collect();
    let mix = |a: (Vec2, [f32; 3]), b: (Vec2, [f32; 3]), t: f32| {
        (
            a.0.lerp(b.0, t),
            [0, 1, 2].map(|value| a.1[value] + (b.1[value] - a.1[value]) * t),
        )
    };
    for _ in 0..SMOOTHING {
        if line.len() < 3 {
            break;
        }
        let mut cut = Vec::with_capacity(line.len() * 2);
        cut.push(line[0]);
        for pair in line.windows(2) {
            cut.push(mix(pair[0], pair[1], 0.25));
            cut.push(mix(pair[0], pair[1], 0.75));
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
        area: Vec::with_capacity(steps + 1),
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
        let (point, [half, depth, area]) = mix(line[segment], line[segment + 1], t);
        out.points.push(point);
        out.half.push(half);
        out.depth.push(depth);
        out.area.push(area);
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

crate::codec::record!(River {
    points,
    surface,
    cut,
    half,
    depth,
    channel,
    fill,
    speed,
    discharge,
    foam,
    falls,
    head,
});
// Rivers travel without the grid that finds them, which readers build
// again from the same rivers, to the same bits: most of a shared stack's
// bytes otherwise.
impl crate::codec::Codec for Rivers {
    fn write(&self, out: &mut crate::codec::Writer) {
        self.rivers.write(out);
    }

    fn read(input: &mut crate::codec::Reader<'_>) -> Result<Self, crate::codec::CodecError> {
        Ok(Self::new(Vec::read(input)?))
    }
}
