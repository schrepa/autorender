use std::ops::RangeInclusive;

use engine_math::collections::HashMap;
use engine_math::{clamp, max, min};
use glam::Vec2;

use crate::bed::{Bed, Crest, Knob, NARROWEST, Reach};
use crate::hydrology::RIVER_FAR;
use serde::Serialize;

pub const RIVER_STEP: f32 = 3.0;
const RISER: f32 = 0.5;
const BED_REACH: f32 = 1.0;
pub const BANK_SLOPE: f32 = 0.65;
pub const BANK_REACH: f32 = 16.0;
const BANK_ROUNDING: f32 = 1.5;
const LEAN: f32 = 1.0 / 3.0;
pub const FOOTPRINT: f32 = 0.75;
const CELL: f32 = 32.0;

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct River {
    pub points: Vec<[f32; 2]>,
    pub discharge: Vec<f32>,
    pub half: Vec<f32>,
    pub depth: Vec<f32>,
    pub bed: Vec<f32>,
    pub surface: Vec<f32>,
    pub speed: Vec<f32>,
    pub falls: Vec<bool>,
    pub sheeted: Vec<bool>,
    pub reach: Vec<Reach>,
    pub d50: Vec<f32>,
    pub d84: Vec<f32>,
    pub power: Vec<f32>,
    pub flood: Vec<f32>,
    pub carved: Vec<bool>,
    pub cut: Vec<bool>,
    pub crest: Vec<Option<Crest>>,
    pub knobs: Vec<Knob>,
    pub head: bool,
    pub given: Given,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Given {
    pub source: Option<[u32; 2]>,
    pub width: bool,
    pub depth: bool,
    pub flood: bool,
}

impl River {
    pub fn falls_at(&self, index: usize) -> bool {
        self.falls.get(index).copied().unwrap_or(false)
    }

    pub fn sheeted_at(&self, index: usize) -> bool {
        self.sheeted.get(index).copied().unwrap_or(false)
            && self.sheeted.get(index + 1).copied().unwrap_or(false)
    }

    pub fn heading(&self, segment: usize, t: f32) -> Vec2 {
        let last = self.points.len().saturating_sub(1);
        let point = |index: usize| Vec2::from(self.points[index.min(last)]);
        let tangent =
            |index: usize| (point(index + 1) - point(index.saturating_sub(1))).normalize_or_zero();
        let (start, end) = (tangent(segment), tangent(segment + 1));
        (start + (end - start) * t)
            .try_normalize()
            .unwrap_or_else(|| (point(segment + 1) - point(segment)).normalize_or(Vec2::X))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Station {
    pub river: u32,
    pub point: u32,
    pub across: f32,
    pub direction: [f32; 2],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RiverNetwork {
    pub into: Vec<Option<usize>>,
    pub sources: Vec<Vec<usize>>,
    pub order: Vec<usize>,
}

impl RiverNetwork {
    pub fn new(rivers: &[River]) -> Self {
        let count = rivers.len();
        let mut into = vec![None; count];
        for (index, river) in rivers.iter().enumerate() {
            let (Some(&end), true) = (river.points.last(), river.points.len() >= 2) else {
                continue;
            };
            let end = Vec2::from(end);
            let mut nearest: Option<(f32, usize)> = None;
            for (other, next) in rivers.iter().enumerate() {
                if other == index || next.points.len() < 2 {
                    continue;
                }
                let distance = end.distance(Vec2::from(next.points[0]));
                if distance < 1.5 * RIVER_STEP && nearest.is_none_or(|(best, _)| distance < best) {
                    nearest = Some((distance, other));
                }
            }
            into[index] = nearest.map(|(_, other)| other);
        }
        let mut sources = vec![Vec::new(); count];
        let mut waiting = vec![0_usize; count];
        for (index, next) in into.iter().enumerate() {
            if let Some(next) = *next {
                sources[next].push(index);
                waiting[next] += 1;
            }
        }
        let mut order = Vec::with_capacity(count);
        let mut ready: Vec<usize> = (0..count)
            .rev()
            .filter(|&index| waiting[index] == 0)
            .collect();
        while let Some(index) = ready.pop() {
            order.push(index);
            if let Some(next) = into[index] {
                waiting[next] -= 1;
                if waiting[next] == 0 {
                    ready.push(next);
                }
            }
        }
        if order.len() < count {
            let mut placed = vec![false; count];
            for &index in &order {
                placed[index] = true;
            }
            order.extend((0..count).filter(|&index| !placed[index]));
        }
        Self {
            into,
            sources,
            order,
        }
    }
}

fn along_rivers(rivers: &[River]) -> Vec<Vec<f32>> {
    rivers
        .iter()
        .map(|river| {
            let points: Vec<Vec2> = river
                .points
                .iter()
                .map(|&point| Vec2::from(point))
                .collect();
            crate::hydraulics::distances(&points)
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rivers {
    pub rivers: Vec<River>,
    network: RiverNetwork,
    along: Vec<Vec<f32>>,
    shifts: Vec<Vec<f32>>,
    widest: f32,
    min: Vec2,
    columns: usize,
    rows: usize,
    starts: Vec<u32>,
    segments: Vec<[u32; 2]>,
}

impl Rivers {
    pub fn new(rivers: Vec<River>) -> Self {
        let shifts = rivers.iter().map(shifts).collect();
        let widest = widest(&rivers);
        let network = RiverNetwork::new(&rivers);
        let along = along_rivers(&rivers);
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
                network,
                along,
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
            network,
            along,
            shifts,
            widest,
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

    pub fn len(&self) -> usize {
        self.rivers.len()
    }

    pub fn network(&self) -> &RiverNetwork {
        &self.network
    }

    pub fn along(&self) -> &[Vec<f32>] {
        &self.along
    }

    pub fn changes_from(&self, before: &Self) -> Vec<(Vec2, Vec2)> {
        let key = |river: &River| {
            river
                .points
                .first()
                .map(|point| [point[0].to_bits(), point[1].to_bits()])
        };
        let listed = |rivers: &[River]| {
            let mut by: HashMap<[u32; 2], Vec<usize>> = HashMap::default();
            for (number, river) in rivers.iter().enumerate() {
                if let Some(key) = key(river) {
                    by.entry(key).or_default().push(number);
                }
            }
            let bytes: Vec<Vec<u8>> = rivers.iter().map(engine_codec::encode).collect();
            (by, bytes)
        };
        let (ours, theirs) = (listed(&self.rivers), listed(&before.rivers));
        let mut regions = Vec::new();
        for (rivers, (_, bytes), (by, others)) in [
            (&self.rivers, &ours, &theirs),
            (&before.rivers, &theirs, &ours),
        ] {
            for (number, river) in rivers.iter().enumerate() {
                let same = key(river)
                    .and_then(|key| by.get(&key))
                    .is_some_and(|candidates| {
                        candidates
                            .iter()
                            .any(|&other| others[other] == bytes[number])
                    });
                if same || river.points.is_empty() {
                    continue;
                }
                let (mut low, mut high) =
                    (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
                for &point in &river.points {
                    low = low.min(Vec2::from(point));
                    high = high.max(Vec2::from(point));
                }
                let pad = Vec2::splat(
                    river
                        .half
                        .iter()
                        .fold(0.0_f32, |widest, &half| max(widest, half))
                        + BANK_REACH,
                );
                regions.push((low - pad, high + pad));
            }
        }
        regions
    }

    #[must_use]
    pub fn with_rivers(&self, rivers: Vec<River>) -> Self {
        Self {
            shifts: rivers.iter().map(shifts).collect(),
            widest: widest(&rivers),
            rivers,
            network: self.network.clone(),
            along: self.along.clone(),
            min: self.min,
            columns: self.columns,
            rows: self.rows,
            starts: self.starts.clone(),
            segments: self.segments.clone(),
        }
    }

    pub fn station(&self, x: f32, z: f32) -> Option<Station> {
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

    pub fn bounds(&self) -> (Vec2, Vec2) {
        (
            self.min,
            self.min + Vec2::new(self.columns as f32, self.rows as f32) * CELL,
        )
    }

    fn near(&self, point: Vec2) -> &[[u32; 2]] {
        let Some((columns, rows)) = self.cells(point, point) else {
            return &[];
        };
        let at = *rows.start() * self.columns + *columns.start();
        &self.segments[self.starts[at] as usize..self.starts[at + 1] as usize]
    }

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

    fn segment(&self, [river, segment]: [u32; 2]) -> Segment<'_> {
        let line = &self.rivers[river as usize];
        let index = segment as usize;
        let (a, b) = (
            Vec2::from(line.points[index]),
            Vec2::from(line.points[index + 1]),
        );
        let run = b - a;
        let length = run.dot(run);
        let (crest, riser, drop) = line.crest[index].map_or((0.5, 1.0, 0.0), |crest| {
            let riser = min(RISER / length.sqrt(), 1.0);
            (
                clamp(crest.share, 0.5 * riser, 1.0 - 0.5 * riser),
                riser,
                crest.drop,
            )
        });
        let first = line
            .knobs
            .partition_point(|knob| knob.segment + 1 < segment);
        let end = line
            .knobs
            .partition_point(|knob| knob.segment <= segment + 1);
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
            next: line.points.get(index + 2).map(|&next| Vec2::from(next) - b),
            crest,
            riser,
            drop,
            knobs: &line.knobs[first..end],
        }
    }

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
        let at = |values: &[f32], along: f32| {
            values[index] + (values[index + 1] - values[index]) * along
        };
        let surface = at(&line.surface, reading.stepped);
        surface.is_finite().then(|| {
            (
                RiverWater {
                    surface,
                    flow: line.heading(index, reading.along) * at(&line.speed, reading.along),
                    sheeted: line.sheeted_at(index),
                    span: 2.0 * at(&line.half, reading.along),
                },
                reading.section,
            )
        })
    }

    fn bed(&self, taken: &Taken, point: Vec2, spacing: f32) -> Option<Bed> {
        if taken.holder[0] == u32::MAX || taken.edge > BED_REACH {
            return None;
        }
        let segment = self.segment(taken.holder);
        let reading = segment.read(point, spacing)?;
        if reading.edge > BED_REACH {
            return None;
        }
        let (line, index) = (segment.line, segment.index);
        let at =
            |values: &[f32]| values[index] + (values[index + 1] - values[index]) * reading.along;
        let nearer = if reading.along < 0.5 {
            index
        } else {
            index + 1
        };
        Some(Bed {
            reach: line.reach[nearer],
            d50: at(&line.d50),
            d84: at(&line.d84),
            power: at(&line.power),
            flood: at(&line.flood),
            carved: line.carved[nearer],
        })
    }

    pub fn bed_at(&self, x: f32, z: f32) -> Option<Bed> {
        let point = Vec2::new(x, z);
        self.bed(&self.take(point, 0.0), point, 0.0)
    }

    pub fn edge(&self, x: f32, z: f32, spacing: f32) -> f32 {
        self.take(Vec2::new(x, z), spacing).edge
    }

    pub fn cut(&self, x: f32, z: f32, ground: f32, spacing: f32) -> f32 {
        self.take(Vec2::new(x, z), spacing).cut(ground)
    }

    pub fn water_at(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        let point = Vec2::new(x, z);
        let (water, section) = self.water(&self.take(point, 0.0), point, 0.0)?;
        let depth = water.surface - section;
        (depth > 0.0).then_some((water.surface, depth))
    }

    pub fn reaches(&self, low: Vec2, high: Vec2) -> bool {
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

fn widest(rivers: &[River]) -> f32 {
    rivers
        .iter()
        .flat_map(|river| &river.half)
        .fold(0.0, |widest, &half| max(widest, half))
}

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
        let curvature = 2.0
            * before
                .normalize_or_zero()
                .perp_dot(after.normalize_or_zero())
            / lengths;
        let bend = curvature * river.half[index];
        shifts[index] = -river.half[index] * bend / (1.0 + bend.abs() / LEAN);
    }
    shifts
}

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
    crest: f32,
    riser: f32,
    drop: f32,
    knobs: &'a [Knob],
}

#[derive(Clone, Copy, Debug)]
struct Reading {
    along: f32,
    stepped: f32,
    edge: f32,
    section: f32,
    floor: f32,
}

impl Segment<'_> {
    #[inline]
    fn stepped(&self, along: f32) -> f32 {
        if self.drop == 0.0 {
            return along;
        }
        let x = clamp((along - self.crest) / self.riser + 0.5, 0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    }

    #[inline]
    fn knobs_at(&self, point: Vec2, spacing: f32) -> f32 {
        self.knobs
            .iter()
            .filter(|knob| 2.0 * knob.radius >= spacing)
            .fold(0.0, |highest, knob| max(highest, knob.rise(point)))
    }

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
            self.next
                .is_none_or(|next| (point - self.b).dot(next) < 0.0)
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
        let stepped = self.stepped(along);
        let lift = self.drop * (along - stepped);
        let (bed, depth) = (at(&self.line.bed) + lift, at(&self.line.depth) - lift);
        let shaped = if from <= width {
            let share = from / max(width, 1.0e-3);
            bed + depth * share * share
        } else {
            bed + depth + BANK_SLOPE * (from - width)
        };
        let section = shaped + self.knobs_at(point, spacing);
        let nearer = if along < 0.5 {
            self.index
        } else {
            self.index + 1
        };
        let floor = if self.line.cut[nearer] || 2.0 * half < NARROWEST * spacing {
            section
        } else {
            f32::INFINITY
        };
        Some(Reading {
            along,
            stepped,
            edge,
            section,
            floor,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Taken {
    section: f32,
    holder: [u32; 2],
    edge: f32,
    floor: f32,
}

impl Taken {
    const NONE: Self = Self {
        section: f32::INFINITY,
        holder: [u32::MAX; 2],
        edge: RIVER_FAR,
        floor: f32::INFINITY,
    };

    #[inline]
    fn add(&mut self, id: [u32; 2], reading: &Reading) {
        self.edge = min(self.edge, reading.edge);
        self.floor = min(self.floor, reading.floor);
        if reading.section < self.section || (reading.section == self.section && id < self.holder) {
            self.section = reading.section;
            self.holder = id;
        }
    }

    fn cut(&self, ground: f32) -> f32 {
        if self.holder[0] == u32::MAX {
            return ground;
        }
        let rounding = max(min(self.edge, BANK_ROUNDING), 1.0e-3);
        let cut = smooth_min(ground, self.floor, rounding);
        if self.edge > 0.0 {
            max(cut, ground - BANK_SLOPE * (BANK_REACH - self.edge))
        } else {
            cut
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RiverWater {
    pub surface: f32,
    pub flow: Vec2,
    pub sheeted: bool,
    pub span: f32,
}

pub struct RiverRaster {
    xs: Box<[f32]>,
    zs: Box<[f32]>,
    spacing: f32,
    taken: Box<[Taken]>,
    listed: Vec<[u32; 2]>,
    stamped: usize,
}

impl RiverRaster {
    pub fn new(rivers: &Rivers, xs: &[f32], zs: &[f32], spacing: f32) -> Option<Self> {
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

    pub fn whole(rivers: &Rivers, xs: &[f32], zs: &[f32], spacing: f32) -> Option<Self> {
        let mut raster = Self::new(rivers, xs, zs, spacing)?;
        let mut budget = u64::MAX;
        raster.stamp(rivers, &mut budget);
        raster.reached().then_some(raster)
    }

    pub fn stamp(&mut self, rivers: &Rivers, budget: &mut u64) -> bool {
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

    pub fn reached(&self) -> bool {
        self.taken.iter().any(|taken| taken.holder[0] != u32::MAX)
    }

    #[inline]
    pub fn cut(&self, row: usize, column: usize, ground: f32) -> f32 {
        self.taken[row * self.xs.len() + column].cut(ground)
    }

    pub fn water(&self, rivers: &Rivers, row: usize, column: usize) -> Option<RiverWater> {
        let point = Vec2::new(self.xs[column], self.zs[row]);
        rivers
            .water(
                &self.taken[row * self.xs.len() + column],
                point,
                self.spacing,
            )
            .map(|(water, _)| water)
    }

    pub fn bed(&self, rivers: &Rivers, row: usize, column: usize) -> Option<Bed> {
        let point = Vec2::new(self.xs[column], self.zs[row]);
        rivers.bed(
            &self.taken[row * self.xs.len() + column],
            point,
            self.spacing,
        )
    }

    pub fn edges(&self, step: impl Fn(f32) -> u16) -> Option<Box<[u16]>> {
        self.taken
            .iter()
            .any(|taken| taken.edge < RIVER_FAR)
            .then(|| self.taken.iter().map(|taken| step(taken.edge)).collect())
    }
}

fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - (a - b).abs(), 0.0) / k;
    min(a, b) - h * h * k * 0.25
}

pub struct RiverLine {
    pub points: Vec<Vec2>,
    pub area: Vec<f32>,
    pub head: bool,
    pub given: Option<GivenLine>,
}

pub struct GivenLine {
    pub source: [u32; 2],
    pub values: Vec<[f32; 3]>,
}

engine_codec::record!(River {
    points,
    discharge,
    half,
    depth,
    bed,
    surface,
    speed,
    falls,
    sheeted,
    reach,
    d50,
    d84,
    power,
    flood,
    carved,
    cut,
    crest,
    knobs,
    head,
    given,
});
engine_codec::record!(Given {
    source,
    width,
    depth,
    flood,
});
impl engine_codec::Codec for Rivers {
    fn write(&self, out: &mut engine_codec::Writer) {
        self.rivers.write(out);
    }

    fn read(input: &mut engine_codec::Reader<'_>) -> Result<Self, engine_codec::Error> {
        Ok(Self::new(Vec::read(input)?))
    }
}
