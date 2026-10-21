//! Rivers: the channels a hydrology bake
//! finds, as lines with water of their own. A bake traces each channel
//! from its head or a confluence down its receivers to the next
//! confluence, a lake or the grid's edge, as a smooth line through the
//! grid's samples resampled every [`RIVER_STEP`] meters, and sizes its
//! channel by the discharge the area it drains yields. Its banks and water
//! then stand along the finest ground on its line, smoothed and never
//! rising downstream; a river leaving a confluence starts no higher than
//! the rivers meeting there.
//!
//! Terrain cuts every channel with one cross-section: a parabola from the
//! bed up to the banks' top at the channel's edges, its deepest point
//! leaning toward the outer bank of a bend, then banks rising at soil's
//! angle of repose into ground that stands higher; ground lower than the
//! section stays, and where channels meet the lowest section holds. A tile
//! rasterizes the rivers near it once ([`RiverRaster`]): each segment
//! stamps the samples within its banks' reach, a coarse level's sample
//! taking the section's lowest over its square, and the tile's heights,
//! water and surface masks read that raster. Heights asked for at a point,
//! and bakes running before there are tiles, read the same section a point
//! at a time, to the same bits.

use std::ops::RangeInclusive;

use engine_math::{max, min};
use glam::Vec2;

use crate::hydrology::RIVER_FAR;
use serde::Serialize;

/// Meters between a river's points.
pub const RIVER_STEP: f32 = 3.0;
/// Segments of a river whose water [`Rivers::differs_from`] bounds at a
/// time.
const STRETCH: usize = 16;
/// How steeply soil banks rise from a channel's top (meters a meter):
/// their angle of repose, 33 degrees.
pub(crate) const BANK_SLOPE: f32 = 0.65;
/// Meters past a river's edge its banks cut and masks read the meters to
/// its edge. A bank cuts no deeper anywhere than [`BANK_SLOPE`] falling
/// from the ground at its reach, so it meets ground steeper than itself
/// there instead of stepping down to it.
pub(crate) const BANK_REACH: f32 = 16.0;
/// Meters over which a river's banks round into the ground they meet.
const BANK_ROUNDING: f32 = 1.5;
/// The share of its half width a bend shifts a channel's deepest point
/// toward its outer bank at most.
const LEAN: f32 = 1.0 / 3.0;
/// How far a sample's square reaches across a line at most, in samples:
/// half its diagonal, rounded up.
const FOOTPRINT: f32 = 0.75;
/// Meters across the cells of the grid that finds the rivers near a point.
const CELL: f32 = 32.0;

/// One river or stream between confluences: its points (terrain-local x
/// and z, from upstream, about [`RIVER_STEP`] meters apart along a smooth
/// line), and at each the discharge it carries (cubic meters a second),
/// half its channel's width at its banks (meters), its channel's depth
/// at its middle below its banks (meters), its bed at its middle
/// (terrain-local meters: the channel's lowest ground, cut straight
/// beneath a fall's jet from the lip to where it lands), its water's
/// surface (terrain-local meters: the level of its banks, the bed plus the
/// depth, NaN where its water flies through the air), its water's mean
/// speed (meters a second); whether the stretch from each point to the
/// next is a fall's jet (`falls`, a [`crate::Fall`] holding it) and
/// whether a fall's sheet runs over each point (`sheeted`); and whether it
/// begins at a head, where no channel flows in, rather than at a
/// confluence or a lake's outlet. Across the river its channel is a
/// parabola from its bed at the middle up to its banks `half` meters either
/// side, and along a segment its values change evenly from one point's to
/// the next's.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct River {
    pub points: Vec<[f32; 2]>,
    pub discharge: Vec<f32>,
    pub half: Vec<f32>,
    pub depth: Vec<f32>,
    pub bed: Vec<f32>,
    pub surface: Vec<f32>,
    pub speed: Vec<f32>,
    pub falls: Vec<bool>,
    pub sheeted: Vec<bool>,
    pub head: bool,
}

impl River {
    /// Whether the stretch from point `index` to the next is a fall's jet.
    pub(crate) fn falls_at(&self, index: usize) -> bool {
        self.falls.get(index).copied().unwrap_or(false)
    }

    /// Whether a fall's sheet runs over the stretch from point `index` to
    /// the next.
    pub(crate) fn sheeted_at(&self, index: usize) -> bool {
        self.sheeted.get(index).copied().unwrap_or(false)
            && self.sheeted.get(index + 1).copied().unwrap_or(false)
    }

    /// The way the river runs (a unit vector along x and z) `t` (0 to 1) of
    /// the way along its segment from point `segment`: its line's tangent,
    /// which at each point runs from the point before it to the point after
    /// it (along the first and last segments at the ends) and turns evenly
    /// from one point's to the next's, so it turns smoothly from segment to
    /// segment as the line does.
    pub(crate) fn heading(&self, segment: usize, t: f32) -> Vec2 {
        let last = self.points.len().saturating_sub(1);
        let point = |index: usize| Vec2::from(self.points[index.min(last)]);
        let tangent = |index: usize| {
            (point(index + 1) - point(index.saturating_sub(1))).normalize_or_zero()
        };
        let (start, end) = (tangent(segment), tangent(segment + 1));
        (start + (end - start) * t)
            .try_normalize()
            .unwrap_or_else(|| (point(segment + 1) - point(segment)).normalize_or(Vec2::X))
    }
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

/// A terrain's rivers, where the deepest point of each one's channel lies
/// across it at each of its points ([`shifts`]), the widest half width of
/// any, and a grid of the river segments within their banks' reach of each
/// cell.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rivers {
    pub(crate) rivers: Vec<River>,
    shifts: Vec<Vec<f32>>,
    widest: f32,
    min: Vec2,
    columns: usize,
    rows: usize,
    starts: Vec<u32>,
    segments: Vec<[u32; 2]>,
}

impl Rivers {
    /// `rivers` with the grid that finds them.
    pub(crate) fn new(rivers: Vec<River>) -> Self {
        let shifts = rivers.iter().map(shifts).collect();
        let widest = widest(&rivers);
        let mut low = Vec2::splat(f32::INFINITY);
        let mut high = Vec2::splat(f32::NEG_INFINITY);
        for river in &rivers {
            for point in &river.points {
                low = low.min(Vec2::from(*point));
                high = high.max(Vec2::from(*point));
            }
        }
        if rivers.iter().all(|river| river.points.len() < 2) {
            return Self {
                rivers,
                shifts,
                widest,
                ..Self::default()
            };
        }
        let reach = widest + BANK_REACH;
        let corner = low - Vec2::splat(reach);
        let columns = (((high.x + reach - corner.x) / CELL).ceil() as usize).max(1);
        let rows = (((high.y + reach - corner.y) / CELL).ceil() as usize).max(1);
        let mut lists: Vec<Vec<[u32; 2]>> = vec![Vec::new(); columns * rows];
        for (index, river) in rivers.iter().enumerate() {
            for segment in 0..river.points.len().saturating_sub(1) {
                let (along, across) = listing((corner, columns, rows), river, segment);
                for row in across.0..=across.1 {
                    for column in along.0..=along.1 {
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
            shifts,
            widest,
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
            shifts: rivers.iter().map(shifts).collect(),
            widest: widest(&rivers),
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
        let Some((columns, rows)) = self.cells(point, point) else {
            return &[];
        };
        let at = *rows.start() * self.columns + *columns.start();
        &self.segments[self.starts[at] as usize..self.starts[at + 1] as usize]
    }

    /// The cells the box from terrain-local `low` to `high` meets, held
    /// within the grid: their columns and rows; `None` where the box misses
    /// the grid or it lists nothing.
    fn cells(
        &self,
        low: Vec2,
        high: Vec2,
    ) -> Option<(RangeInclusive<usize>, RangeInclusive<usize>)> {
        if self.segments.is_empty() {
            return None;
        }
        let first = ((low - self.min) / CELL).floor();
        let last = ((high - self.min) / CELL).floor();
        let (columns, rows) = (self.columns as f32, self.rows as f32);
        if !(last.x >= 0.0 && last.y >= 0.0 && first.x < columns && first.y < rows) {
            return None;
        }
        let held = |cell: f32, count: f32| min(max(cell, 0.0), count - 1.0) as usize;
        Some((
            held(first.x, columns)..=held(last.x, columns),
            held(first.y, rows)..=held(last.y, rows),
        ))
    }

    /// The segments listed in the cells the box from terrain-local `low` to
    /// `high` meets, each once, in the grid's order: where those cells hold
    /// more entries than there are segments (far tiles), every segment
    /// listed in any of them.
    fn listed(&self, low: Vec2, high: Vec2) -> Vec<[u32; 2]> {
        let Some((columns, rows)) = self.cells(low, high) else {
            return Vec::new();
        };
        let (first, last) = (*columns.start(), *columns.end());
        let span = |row: usize| {
            let at = row * self.columns;
            self.starts[at + first] as usize..self.starts[at + last + 1] as usize
        };
        let entries: usize = rows.clone().map(|row| span(row).len()).sum();
        let count: usize = self
            .rivers
            .iter()
            .map(|river| river.points.len().saturating_sub(1))
            .sum();
        let mut listed = Vec::new();
        if entries <= count {
            for row in rows {
                listed.extend_from_slice(&self.segments[span(row)]);
            }
            listed.sort_unstable();
            listed.dedup();
        } else {
            let grid = (self.min, self.columns, self.rows);
            for (index, river) in self.rivers.iter().enumerate() {
                for segment in 0..river.points.len().saturating_sub(1) {
                    let (along, across) = listing(grid, river, segment);
                    if along.0 <= last
                        && along.1 >= first
                        && across.0 <= *rows.end()
                        && across.1 >= *rows.start()
                    {
                        listed.push([index as u32, segment as u32]);
                    }
                }
            }
        }
        listed
    }

    /// River `river`'s segment `segment`, ready for samples to read.
    fn segment(&self, [river, segment]: [u32; 2]) -> Segment<'_> {
        let line = &self.rivers[river as usize];
        let index = segment as usize;
        let (a, b) = (
            Vec2::from(line.points[index]),
            Vec2::from(line.points[index + 1]),
        );
        let run = b - a;
        let length = run.dot(run);
        Segment {
            line,
            shifts: &self.shifts[river as usize],
            index,
            a,
            b,
            run,
            length,
            normal: if length > 0.0 {
                Vec2::new(-run.y, run.x) / length.sqrt()
            } else {
                Vec2::Y
            },
            next: line
                .points
                .get(index + 2)
                .map(|&next| Vec2::from(next) - b),
        }
    }

    /// What the rivers give terrain-local `point` as a sample standing for
    /// the square `spacing` meters a side about it sees them (0 for the
    /// point alone): every segment listed in a cell the square's reach
    /// meets, taken as a tile's raster takes them, so the same bits.
    fn take(&self, point: Vec2, spacing: f32) -> Taken {
        let mut taken = Taken::NONE;
        let reach = Vec2::splat(FOOTPRINT * spacing);
        let Some((columns, rows)) = self.cells(point - reach, point + reach) else {
            return taken;
        };
        for row in rows {
            let at = row * self.columns;
            let listed = &self.segments[self.starts[at + *columns.start()] as usize
                ..self.starts[at + *columns.end() + 1] as usize];
            for &id in listed {
                if let Some(reading) = self.segment(id).read(point, spacing) {
                    taken.add(id, &reading);
                }
            }
        }
        taken
    }

    /// The water of the river whose section a sample at terrain-local
    /// `point` takes (`taken`, standing for the square `spacing` meters a
    /// side about it), and that section: `None` where the square lies past
    /// that river's edge, its water flies through the air there or no river
    /// reaches it.
    fn water(&self, taken: &Taken, point: Vec2, spacing: f32) -> Option<(RiverWater, f32)> {
        if taken.holder[0] == u32::MAX || taken.edge > 0.0 {
            return None;
        }
        let segment = self.segment(taken.holder);
        let reading = segment.read(point, spacing)?;
        let (line, index) = (segment.line, segment.index);
        if reading.edge > 0.0 || line.falls_at(index) {
            return None;
        }
        let at =
            |values: &[f32]| values[index] + (values[index + 1] - values[index]) * reading.along;
        let surface = at(&line.surface);
        surface.is_finite().then(|| {
            (
                RiverWater {
                    surface,
                    flow: line.heading(index, reading.along) * at(&line.speed),
                    sheeted: line.sheeted_at(index),
                    half: at(&line.half),
                },
                reading.section,
            )
        })
    }

    /// Meters from the nearest river's edge to the square `spacing` meters
    /// a side about terrain-local `(x, z)` (0 for the point alone), as a
    /// tile's raster measures them at a sample there: negative within a
    /// river's channel, [`RIVER_FAR`] beyond every river's banks' reach.
    pub(crate) fn edge(&self, x: f32, z: f32, spacing: f32) -> f32 {
        self.take(Vec2::new(x, z), spacing).edge
    }

    /// `ground` at terrain-local `(x, z)` as the rivers cut it at a level
    /// whose samples lie `spacing` meters apart: as a tile's raster cuts its
    /// sample there.
    pub(crate) fn cut(&self, x: f32, z: f32, ground: f32, spacing: f32) -> f32 {
        self.take(Vec2::new(x, z), spacing).cut(ground)
    }

    /// The river water at terrain-local `(x, z)` as bakes after the
    /// hydrology's read it before there are tiles: its surface and its depth
    /// over the section its river cuts there; `None` outside every river's
    /// channel, where its water flies through the air or stands on the
    /// section.
    pub(crate) fn water_at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        let point = Vec2::new(x, z);
        let (water, section) = self.water(&self.take(point, 0.0), point, 0.0)?;
        let depth = water.surface - section;
        (depth > 0.0).then_some((water.surface, depth))
    }

    /// Whether a river's channel, out as far again as it is wide, reaches
    /// the terrain-local rectangle from `low` to `high`.
    pub(crate) fn reaches(&self, low: Vec2, high: Vec2) -> bool {
        let pad = Vec2::splat(self.widest);
        self.listed(low - pad, high + pad)
            .iter()
            .any(|&[river, segment]| {
                let line = &self.rivers[river as usize];
                let index = segment as usize;
                let (a, b) = (
                    Vec2::from(line.points[index]),
                    Vec2::from(line.points[index + 1]),
                );
                let reach = Vec2::splat(2.0 * max(line.half[index], line.half[index + 1]));
                (a.min(b) - reach).cmple(high).all() && (a.max(b) + reach).cmpge(low).all()
            })
    }
}

/// The widest half width of any of `rivers`' points.
fn widest(rivers: &[River]) -> f32 {
    rivers
        .iter()
        .flat_map(|river| &river.half)
        .fold(0.0, |widest, &half| max(widest, half))
}

/// Where the deepest point of `river`'s channel lies at each of its points:
/// meters across its line (positive to its right, looking downstream),
/// toward the outer bank of the bend there by a share of its half width
/// that grows with the bend's curvature times that half width, up to
/// [`LEAN`]; none at its ends.
fn shifts(river: &River) -> Vec<f32> {
    let points = &river.points;
    let mut shifts = vec![0.0; points.len()];
    for index in 1..points.len().saturating_sub(1) {
        let before = Vec2::from(points[index]) - Vec2::from(points[index - 1]);
        let after = Vec2::from(points[index + 1]) - Vec2::from(points[index]);
        let lengths = before.length() + after.length();
        if lengths <= 0.0 {
            continue;
        }
        // Per meter, positive where the line turns right.
        let curvature =
            2.0 * before.normalize_or_zero().perp_dot(after.normalize_or_zero()) / lengths;
        let bend = curvature * river.half[index];
        shifts[index] = -river.half[index] * bend / (1.0 + bend.abs() / LEAN);
    }
    shifts
}

/// The cells listing river `river`'s segment `index` in the grid whose
/// first corner, columns and rows `grid` gives: those within its banks'
/// reach of it, its first and last column, then its first and last row.
fn listing(
    (corner, columns, rows): (Vec2, usize, usize),
    river: &River,
    index: usize,
) -> ((usize, usize), (usize, usize)) {
    let (a, b) = (
        Vec2::from(river.points[index]),
        Vec2::from(river.points[index + 1]),
    );
    let margin = Vec2::splat(max(river.half[index], river.half[index + 1]) + BANK_REACH);
    let first = ((a.min(b) - margin - corner) / CELL).floor();
    let last = ((a.max(b) + margin - corner) / CELL).floor();
    (
        (
            max(first.x, 0.0) as usize,
            min(last.x, columns as f32 - 1.0) as usize,
        ),
        (
            max(first.y, 0.0) as usize,
            min(last.y, rows as f32 - 1.0) as usize,
        ),
    )
}

/// A segment as samples read it: its river, the shifts of its channel's
/// deepest point and its index, its ends, run and length squared, the unit
/// way across it (to the river's right, looking downstream), and the run
/// of the river's next segment, none past its last.
#[derive(Clone, Copy)]
struct Segment<'a> {
    line: &'a River,
    shifts: &'a [f32],
    index: usize,
    a: Vec2,
    b: Vec2,
    run: Vec2,
    length: f32,
    normal: Vec2,
    next: Option<Vec2>,
}

/// A segment as a sample reads it: how far along it the sample's nearest
/// point of it lies (0 to 1), meters from the sample's square to the
/// river's edge there (negative within its channel), and the section's
/// lowest over the square (terrain-local meters).
#[derive(Clone, Copy, Debug)]
struct Reading {
    along: f32,
    edge: f32,
    section: f32,
}

impl Segment<'_> {
    /// The segment as a sample at terrain-local `point` reads it, standing
    /// for the square `spacing` meters a side about it (0 for the point
    /// alone): `None` beyond its banks' reach, and past its ends but for the
    /// line's own ends and the outer side of a bend after it, where no
    /// other segment's part of the line lies nearer. Across the line the
    /// section is a parabola from the bed at the channel's deepest point
    /// ([`shifts`]) up to the banks' top (the bed plus the depth) at either
    /// edge, so a bend's outer bank stands steeper than its inner one, then
    /// banks rising at [`BANK_SLOPE`]; the square takes its lowest, at its
    /// point nearest the deepest. Values run linearly along the segment.
    #[inline]
    fn read(&self, point: Vec2, spacing: f32) -> Option<Reading> {
        let projected = if self.length > 0.0 {
            (point - self.a).dot(self.run) / self.length
        } else {
            0.0
        };
        let held = if projected < 0.0 {
            self.index == 0
        } else if projected > 1.0 {
            self.next.is_none_or(|next| (point - self.b).dot(next) < 0.0)
        } else {
            true
        };
        if !held {
            return None;
        }
        let along = min(max(projected, 0.0), 1.0);
        let at = |values: &[f32]| {
            values[self.index] + (values[self.index + 1] - values[self.index]) * along
        };
        let gap = point - (self.a + self.run * along);
        let distance = gap.length();
        // The square's reach across the line.
        let reach = 0.5 * spacing * (self.normal.x.abs() + self.normal.y.abs());
        let half = at(&self.line.half);
        let edge = max(distance - reach, 0.0) - half;
        if edge >= BANK_REACH {
            return None;
        }
        let across = if gap.dot(self.normal) < 0.0 {
            -distance
        } else {
            distance
        };
        let shift = at(self.shifts);
        let nearest = across - min(max(across - shift, -reach), reach);
        let (from, width) = if nearest < shift {
            (shift - nearest, half + shift)
        } else {
            (nearest - shift, half - shift)
        };
        let (bed, depth) = (at(&self.line.bed), at(&self.line.depth));
        let section = if from <= width {
            let share = from / max(width, 1.0e-3);
            bed + depth * share * share
        } else {
            bed + depth + BANK_SLOPE * (from - width)
        };
        Some(Reading {
            along,
            edge,
            section,
        })
    }
}

/// What a sample takes of the rivers reaching it: the lowest section of
/// any (terrain-local meters; infinite where none reaches), the river and
/// segment holding it, the first of them where sections tie, and meters
/// to the nearest river's edge ([`RIVER_FAR`] where none reaches): the same
/// whatever order segments come in.
#[derive(Clone, Copy, Debug)]
struct Taken {
    section: f32,
    holder: [u32; 2],
    edge: f32,
}

impl Taken {
    /// No river's.
    const NONE: Self = Self {
        section: f32::INFINITY,
        holder: [u32::MAX; 2],
        edge: RIVER_FAR,
    };

    /// Takes river and segment `id`'s `reading`.
    #[inline]
    fn add(&mut self, id: [u32; 2], reading: &Reading) {
        self.edge = min(self.edge, reading.edge);
        if reading.section < self.section
            || (reading.section == self.section && id < self.holder)
        {
            self.section = reading.section;
            self.holder = id;
        }
    }

    /// `ground` as the rivers cut it: down to the lowest section, rounding
    /// into the ground over [`BANK_ROUNDING`] but no more than the bank
    /// there is wide, so channels keep their section, and past every
    /// channel's edge no deeper than [`BANK_SLOPE`] falling from the ground
    /// at the banks' reach. Ground lower than the section stays: nothing is
    /// built up.
    fn cut(&self, ground: f32) -> f32 {
        if self.holder[0] == u32::MAX {
            return ground;
        }
        let rounding = max(min(self.edge, BANK_ROUNDING), 1.0e-3);
        let cut = smooth_min(ground, self.section, rounding);
        if self.edge > 0.0 {
            max(cut, ground - BANK_SLOPE * (BANK_REACH - self.edge))
        } else {
            cut
        }
    }
}

/// A river's water at a sample: its surface, flat across the river
/// (terrain-local meters), its flow along x and z (meters a second),
/// whether a fall's sheet runs over it, and half its channel's width there
/// (meters).
#[derive(Clone, Copy, Debug)]
pub(crate) struct RiverWater {
    pub(crate) surface: f32,
    pub(crate) flow: Vec2,
    pub(crate) sheeted: bool,
    pub(crate) half: f32,
}

/// The rivers near a tile rasterized once over samples at x positions
/// `xs` and z positions `zs` (both ascending), a level's samples `spacing`
/// meters apart, each standing for its square: what each sample takes of
/// them, row by row, as [`Rivers::cut`] takes it at one point; and the
/// segments near them, those first `stamped`.
pub(crate) struct RiverRaster {
    xs: Box<[f32]>,
    zs: Box<[f32]>,
    spacing: f32,
    taken: Box<[Taken]>,
    listed: Vec<[u32; 2]>,
    stamped: usize,
}

impl RiverRaster {
    /// `rivers` to rasterize over the samples at `xs` and `zs`
    /// ([`Self::stamp`]); `None` where no river's banks reach near them.
    pub(crate) fn new(rivers: &Rivers, xs: &[f32], zs: &[f32], spacing: f32) -> Option<Self> {
        let (&x0, &x1, &z0, &z1) = (xs.first()?, xs.last()?, zs.first()?, zs.last()?);
        let footprint = Vec2::splat(FOOTPRINT * spacing);
        let listed = rivers.listed(Vec2::new(x0, z0) - footprint, Vec2::new(x1, z1) + footprint);
        (!listed.is_empty()).then(|| Self {
            xs: xs.into(),
            zs: zs.into(),
            spacing,
            taken: vec![Taken::NONE; xs.len() * zs.len()].into_boxed_slice(),
            listed,
            stamped: 0,
        })
    }

    /// `rivers` rasterized over the samples at `xs` and `zs` at once;
    /// `None` where no river reaches any.
    pub(crate) fn whole(rivers: &Rivers, xs: &[f32], zs: &[f32], spacing: f32) -> Option<Self> {
        let mut raster = Self::new(rivers, xs, zs, spacing)?;
        let mut budget = u64::MAX;
        raster.stamp(rivers, &mut budget);
        raster.reached().then_some(raster)
    }

    /// Stamps the segments near its samples, `rivers`' own, each over the
    /// samples within its banks' reach, until `budget` (samples' worth) runs
    /// out: whether every one is stamped. A segment that starts always
    /// finishes.
    pub(crate) fn stamp(&mut self, rivers: &Rivers, budget: &mut u64) -> bool {
        let (xs, zs, spacing) = (&self.xs, &self.zs, self.spacing);
        let footprint = FOOTPRINT * spacing;
        while let Some(&id) = self.listed.get(self.stamped) {
            if *budget == 0 {
                return false;
            }
            self.stamped += 1;
            let segment = rivers.segment(id);
            let (half, index) = (&segment.line.half, segment.index);
            let reach = max(half[index], half[index + 1]) + BANK_REACH + footprint;
            let (low, high) = (segment.a.min(segment.b), segment.a.max(segment.b));
            let rows = zs.partition_point(|&z| z < low.y - reach)
                ..zs.partition_point(|&z| z <= high.y + reach);
            let mut work = 1_u64;
            for row in rows {
                let z = zs[row];
                // The row's samples within reach of the segment's box.
                let off = max(max(low.y - z, z - high.y), 0.0);
                let wide = max(reach * reach - off * off, 0.0).sqrt();
                let columns = xs.partition_point(|&x| x < low.x - wide)
                    ..xs.partition_point(|&x| x <= high.x + wide);
                work += columns.len() as u64;
                for column in columns {
                    if let Some(reading) = segment.read(Vec2::new(xs[column], z), spacing) {
                        self.taken[row * xs.len() + column].add(id, &reading);
                    }
                }
            }
            *budget = budget.saturating_sub(work);
        }
        true
    }

    /// Whether a river reaches any of its samples, as stamped so far.
    pub(crate) fn reached(&self) -> bool {
        self.taken.iter().any(|taken| taken.holder[0] != u32::MAX)
    }

    /// `ground` at the sample of row `row` and column `column` as the
    /// rivers cut it ([`Rivers::cut`]).
    #[inline]
    pub(crate) fn cut(&self, row: usize, column: usize, ground: f32) -> f32 {
        self.taken[row * self.xs.len() + column].cut(ground)
    }

    /// The river water at the sample of row `row` and column `column`:
    /// the water of the river whose section it takes, where the sample's
    /// square meets that river's channel and its water stands there.
    pub(crate) fn water(&self, rivers: &Rivers, row: usize, column: usize) -> Option<RiverWater> {
        let point = Vec2::new(self.xs[column], self.zs[row]);
        rivers
            .water(&self.taken[row * self.xs.len() + column], point, self.spacing)
            .map(|(water, _)| water)
    }

    /// Every sample's meters to the nearest river's edge as a tile holds
    /// them ([`crate::tile::river_step`]), row by row; `None` where every
    /// sample lies beyond every river's reach.
    pub(crate) fn edges(&self) -> Option<Box<[u16]>> {
        self.taken
            .iter()
            .any(|taken| taken.edge < RIVER_FAR)
            .then(|| {
                self.taken
                    .iter()
                    .map(|taken| crate::tile::river_step(taken.edge))
                    .collect()
            })
    }
}

/// The lower of `a` and `b`, rounded where they come within `k` of each
/// other (a polynomial smooth minimum).
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - (a - b).abs(), 0.0) / k;
    min(a, b) - h * h * k * 0.25
}

/// A river's line from the bake's grid before its channel and water are
/// set: its points (terrain-local x and z, from upstream, about
/// [`RIVER_STEP`] meters apart along a smooth line), the area draining
/// through each (square meters), and whether it begins at a head.
pub(crate) struct RiverLine {
    pub(crate) points: Vec<Vec2>,
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
/// upstream rivers first, as a smooth line ([`smooth_line`]) with the area
/// draining through it. Where it ends in a lake or a confluence, whose
/// sample other water reaches too, its last point carries its own water.
pub(crate) fn trace(
    channels: &Channels<'_>,
    (width, spacing): (usize, f32),
    channel_area: f32,
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
        let mut joins = false;
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
                joins = true;
                break;
            }
            current = next;
        }
        if course.len() < 2 {
            continue;
        }
        let points: Vec<Vec2> = course.iter().map(|&sample| place(sample)).collect();
        let mut drained: Vec<f32> = course.iter().map(|&sample| area[sample]).collect();
        if joins {
            let last = drained.len() - 1;
            drained[last] = drained[last - 1];
        }
        let mut line = smooth_line(&points, &drained);
        line.head = inflows[start] == 0;
        lines.push(line);
    }
    lines
}

/// Places along each segment of a line's spline its length is measured at,
/// so its resampled points lie evenly along it.
const MEASURED: usize = 8;

/// The weights of a uniform cubic B-spline's four control points `t` (0 to
/// 1) of the way along the segment they shape.
fn spline_weights(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    let (t2, t3) = (t * t, t * t * t);
    [
        s * s * s / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (3.0 * (t + t2 - t3) + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// The line through the grid samples `points` (a channel's course, two or
/// more) and the areas draining through them, `drained` (square meters):
/// the uniform cubic B-spline whose control points they are, each end
/// mirrored about itself so the line starts and ends exactly at its end
/// samples heading along its first and last steps, resampled every
/// [`RIVER_STEP`] meters along its length, the same spline carrying the
/// area along. Lines meeting at a sample start and end at the same point.
fn smooth_line(points: &[Vec2], drained: &[f32]) -> RiverLine {
    let last = points.len() - 1;
    let mut control: Vec<(Vec2, f32)> = Vec::with_capacity(last + 3);
    control.push((2.0 * points[0] - points[1], 2.0 * drained[0] - drained[1]));
    control.extend(points.iter().copied().zip(drained.iter().copied()));
    control.push((
        2.0 * points[last] - points[last - 1],
        2.0 * drained[last] - drained[last - 1],
    ));
    // The spline at `u` (0 at the first sample, `last` at the last): its
    // point and the area draining through it.
    let at = |u: f32| {
        let segment = (u.floor() as usize).min(last - 1);
        let (mut point, mut area) = (Vec2::ZERO, 0.0_f32);
        for (offset, weight) in spline_weights(u - segment as f32).into_iter().enumerate() {
            let (place, drains) = control[segment + offset];
            point += place * weight;
            area += drains * weight;
        }
        (point, area)
    };
    // Its length measured from its start, at each of `measured`'s places.
    let mut measured: Vec<(f32, f32)> = Vec::with_capacity(last * MEASURED + 1);
    measured.push((0.0, 0.0));
    let (mut length, mut previous) = (0.0_f32, points[0]);
    for place in 1..=last * MEASURED {
        let u = place as f32 / MEASURED as f32;
        let (point, _) = at(u);
        length += point.distance(previous);
        previous = point;
        measured.push((u, length));
    }
    let steps = max((length / RIVER_STEP).ceil(), 1.0) as usize;
    let mut line = RiverLine {
        points: Vec::with_capacity(steps + 1),
        area: Vec::with_capacity(steps + 1),
        head: false,
    };
    let mut interval = 0;
    for step in 0..=steps {
        let wanted = length * step as f32 / steps as f32;
        while interval + 2 < measured.len() && measured[interval + 1].1 < wanted {
            interval += 1;
        }
        let ((from, before), (to, after)) = (measured[interval], measured[interval + 1]);
        let share = if after > before {
            min(max((wanted - before) / (after - before), 0.0), 1.0)
        } else {
            0.0
        };
        let (point, area) = at(from + (to - from) * share);
        line.points.push(point);
        line.area.push(area);
    }
    // Its ends exactly at its end samples, where the rivers meeting there
    // start and end too.
    line.points[0] = points[0];
    line.area[0] = drained[0];
    line.points[steps] = points[last];
    line.area[steps] = drained[last];
    line
}

crate::codec::record!(River {
    points,
    discharge,
    half,
    depth,
    bed,
    surface,
    speed,
    falls,
    sheeted,
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
