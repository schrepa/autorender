use std::ops::RangeInclusive;

use engine_math::collections::HashMap;
use engine_math::{clamp, max, min};
use glam::Vec2;

use crate::bed::{Bed, Boulders, Crest, Knob, NARROWEST, Pools, Reach, Steps};
use crate::hydraulics::{Falls, Profile, Shape, distances};
use crate::hydrology::RIVER_FAR;
use crate::noise::smooth;
use serde::{Deserialize, Serialize};

pub const RIVER_STEP: f32 = 3.0;
pub const BANK_REACH: f32 = 16.0;
pub const FOOTPRINT: f32 = 0.75;
const CELL: f32 = 32.0;
const BARE: f32 = 1.0e-7;

engine_codec::fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Section {
        pub bend: f32 = 0.4 => above_to(0.0, 10.0),
        pub lean: f32 = 0.5 => range(0.0, 0.9),
        pub shelf: f32 = 0.8 => range(0.0, 1.0),
        pub island: f32 = 0.5 => above_to(0.0, 1.0),
        pub rounding: f32 = 0.5 => range(0.0, 4.0),
    }
}

engine_codec::fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Valley {
        pub terraces: u32 = 1 => range(0, 8),
        pub rise: f32 = 0.9 => range(0.0, 10.0),
        pub plain_rise: f32 = 0.25 => range(0.0, 1.0),
        pub bank: [f32; 2] = [0.25, 0.7] => above_to(0.0, 100.0),
        pub wall: f32 = 2.0 => range(0.0, 100.0),
        pub walled: [f32; 2] = [0.5, 2.0] => range(0.0, 100.0),
        pub tread: f32 = 0.25 => range(0.0, 100.0),
        pub tread_rise: f32 = 0.15 => range(0.0, 10.0),
        pub reach: f32 = 0.5 => range(0.0, 100.0),
        pub fade: f32 = 0.65 => range(0.0, 100.0),
    }
}

engine_codec::fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(default, deny_unknown_fields)]
    pub struct Deposits {
        pub wet: [f32; 2] = [1.3, 0.6] => range(0.0, 10.0),
        pub fining: f32 = 0.35 => above_to(0.0, 10.0),
        pub sand: f32 = 5.0e-4 => range(0.0, 1.0),
        pub earth: f32 = 3.0e-5 => range(0.0, 1.0),
        pub earth_rise: [f32; 2] = [0.1, 0.3] => range(0.0, 10.0),
        pub earth_reach: f32 = 0.5 => above_to(0.0, 100.0),
        pub sand_spread: f32 = 0.35 => range(0.0, 1.0),
        pub silt_spread: [f32; 2] = [1.0 / 3.0, 1.0] => range(0.0, 1.0),
        pub sides: f32 = 2.0 => range(0.0, 100.0),
        pub flood: [f32; 2] = [0.25, 1.0] => range(0.0, 100.0),
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Form {
    pub profile: Profile,
    pub shape: Shape,
    pub falls: Falls,
    pub pools: Pools,
    pub steps: Steps,
    pub boulders: Boulders,
    pub section: Section,
    pub valley: Valley,
    pub deposits: Deposits,
}

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
    pub crest: Vec<Crest>,
    pub knobs: Vec<Knob>,
    pub bend: Vec<f32>,
    pub plain: Vec<[f32; 2]>,
    pub bar: Vec<f32>,
    pub rock: Vec<f32>,
    pub wall: Vec<f32>,
    pub margin: Vec<[f32; 2]>,
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
        let mut starts: HashMap<[i64; 2], Vec<usize>> = HashMap::default();
        for (other, next) in rivers.iter().enumerate() {
            if next.points.len() >= 2 {
                starts
                    .entry(joining_cell(Vec2::from(next.points[0])))
                    .or_default()
                    .push(other);
            }
        }
        for (index, river) in rivers.iter().enumerate() {
            let (Some(&end), true) = (river.points.last(), river.points.len() >= 2) else {
                continue;
            };
            let end = Vec2::from(end);
            let [x, z] = joining_cell(end);
            let mut nearest: Option<(f32, usize)> = None;
            let near = (-2..=2_i64).flat_map(|dz| {
                (-2..=2).map(move |dx| [x.saturating_add(dx), z.saturating_add(dz)])
            });
            for other in near.filter_map(|cell| starts.get(&cell)).flatten().copied() {
                if other == index {
                    continue;
                }
                let distance = end.distance(Vec2::from(rivers[other].points[0]));
                let closer = nearest.is_none_or(|(best, at)| (distance, other) < (best, at));
                if distance < JOINING && closer {
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

const JOINING: f32 = 1.5 * RIVER_STEP;

fn joining_cell(point: Vec2) -> [i64; 2] {
    [
        (point.x / JOINING).floor() as i64,
        (point.y / JOINING).floor() as i64,
    ]
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
            distances(&points)
        })
        .collect()
}

pub fn point_along(points: &[Vec2], along: &[f32], place: f32) -> Vec2 {
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
    points[before] + (points[after] - points[before]) * share
}

pub fn bends(points: &[Vec2], widths: &[f32], full: f32) -> Vec<f32> {
    let along = distances(points);
    let end = along[along.len() - 1];
    (0..points.len())
        .map(|index| {
            let place = along[index];
            let reach = min(max(widths[index], RIVER_STEP), min(place, end - place));
            if reach < 0.5 * RIVER_STEP {
                return 0.0;
            }
            let before = points[index] - point_along(points, &along, place - reach);
            let after = point_along(points, &along, place + reach) - points[index];
            let lengths = before.length() + after.length();
            if lengths <= 0.0 {
                return 0.0;
            }
            let curvature = 2.0
                * before
                    .normalize_or_zero()
                    .perp_dot(after.normalize_or_zero())
                / lengths;
            clamp(curvature * widths[index] / full, -1.0, 1.0)
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rivers {
    pub rivers: Vec<River>,
    form: Form,
    network: RiverNetwork,
    along: Vec<Vec<f32>>,
    shifts: Vec<Vec<f32>>,
    reaches: Vec<[f32; 2]>,
    spans: Vec<f32>,
    widest: f32,
    min: Vec2,
    columns: usize,
    rows: usize,
    starts: Vec<u32>,
    segments: Vec<[u32; 2]>,
}

impl Rivers {
    pub fn new(rivers: Vec<River>, form: Form) -> Self {
        let shifts = shifts(&rivers, form.section.lean);
        let reaches = rivers.iter().map(reaches).collect();
        let spans: Vec<f32> = rivers.iter().map(span).collect();
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
                form,
                network,
                along,
                shifts,
                reaches,
                spans,
                widest,
                ..Self::default()
            };
        }
        let reach = widest
            + spans
                .iter()
                .fold(0.0, |farthest, &span| max(farthest, span));
        let corner = low - Vec2::splat(reach);
        let columns = (((high.x + reach - corner.x) / CELL).ceil() as usize).max(1);
        let rows = (((high.y + reach - corner.y) / CELL).ceil() as usize).max(1);
        let (starts, segments) = cell_segments((corner, columns, rows), &rivers, &spans);
        Self {
            rivers,
            form,
            network,
            along,
            shifts,
            reaches,
            spans,
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

    pub fn form(&self) -> &Form {
        &self.form
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
                        + span(river),
                );
                regions.push((low - pad, high + pad));
            }
        }
        regions
    }

    #[must_use]
    pub fn with_rivers(&self, rivers: Vec<River>) -> Self {
        Self {
            shifts: shifts(&rivers, self.form.section.lean),
            reaches: rivers.iter().map(reaches).collect(),
            spans: rivers.iter().map(span).collect(),
            widest: widest(&rivers),
            rivers,
            form: self.form,
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
            let [low, high] = [0, 1].map(|side| {
                line.margin[index][side]
                    + (line.margin[index + 1][side] - line.margin[index][side]) * t
            });
            let reach = max(low, high);
            if distance > half + reach || nearest.is_some_and(|(best, _)| best <= distance) {
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
                    let (along, across) = listing(grid, (river, self.spans[index]), segment);
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
        let along = &self.along[river as usize];
        let (start, span) = (along[index], along[index + 1] - along[index]);
        let [above, below] = self.reaches[river as usize];
        let crests = line
            .crest
            .partition_point(|crest| crest.place < start - below)
            ..line
                .crest
                .partition_point(|crest| crest.place <= start + span + above);
        let first = line
            .knobs
            .partition_point(|knob| knob.segment + 1 < segment);
        let end = line
            .knobs
            .partition_point(|knob| knob.segment <= segment + 1);
        Segment {
            line,
            form: &self.form,
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
            start,
            span,
            crests: &line.crest[crests],
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
        let surface = at(&line.surface, reading.along) + reading.lift;
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
        if taken.holder[0] == u32::MAX {
            return None;
        }
        let segment = self.segment(taken.holder);
        let reading = segment.read(point, spacing)?;
        let (line, index) = (segment.line, segment.index);
        let at =
            |values: &[f32]| values[index] + (values[index + 1] - values[index]) * reading.along;
        let median = at(&line.d50);
        let grain = reading.grain(
            &self.form.deposits,
            median,
            (at(&line.surface), at(&line.flood)),
            at(&line.depth),
        );
        if grain < BARE {
            return None;
        }
        let nearer = if reading.along < 0.5 {
            index
        } else {
            index + 1
        };
        Some(Bed {
            reach: line.reach[nearer],
            d50: grain,
            d84: grain * at(&line.d84) / max(median, BARE),
            power: at(&line.power),
            flood: at(&line.flood),
            carved: line.carved[nearer],
        })
    }

    pub fn bed_at(&self, x: f32, z: f32) -> Option<Bed> {
        let point = Vec2::new(x, z);
        self.bed(&self.take(point, 0.0), point, 0.0)
    }

    pub fn bank_at(&self, x: f32, z: f32, spacing: f32) -> Option<(f32, f32)> {
        let point = Vec2::new(x, z);
        let taken = self.take(point, spacing);
        if taken.holder[0] == u32::MAX {
            return None;
        }
        let segment = self.segment(taken.holder);
        let reading = segment.read(point, spacing)?;
        let (line, index) = (segment.line, segment.index);
        let water = line.surface[index]
            + (line.surface[index + 1] - line.surface[index]) * reading.along
            + reading.lift;
        Some((reading.outer, water))
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
}

fn widest(rivers: &[River]) -> f32 {
    rivers
        .iter()
        .flat_map(|river| &river.half)
        .fold(0.0, |widest, &half| max(widest, half))
}

fn reaches(river: &River) -> [f32; 2] {
    river.crest.iter().fold([0.0; 2], |farthest, crest| {
        let [above, below] = crest.reach();
        [max(farthest[0], above), max(farthest[1], below)]
    })
}

fn shifts(rivers: &[River], lean: f32) -> Vec<Vec<f32>> {
    rivers
        .iter()
        .map(|river| {
            river
                .half
                .iter()
                .zip(&river.bend)
                .map(|(&half, &bend)| -lean * half * bend)
                .collect()
        })
        .collect()
}

fn span(river: &River) -> f32 {
    river
        .plain
        .iter()
        .zip(&river.margin)
        .fold(0.0, |widest, (plain, reach)| {
            max(widest, max(reach[0] + plain[0], reach[1] + plain[1]))
        })
}

fn cell_segments(
    grid: (Vec2, usize, usize),
    rivers: &[River],
    spans: &[f32],
) -> (Vec<u32>, Vec<[u32; 2]>) {
    let columns = grid.1;
    let visit = |cell: &mut dyn FnMut(usize, [u32; 2])| {
        for (index, river) in rivers.iter().enumerate() {
            for segment in 0..river.points.len().saturating_sub(1) {
                let (along, across) = listing(grid, (river, spans[index]), segment);
                for row in across.0..=across.1 {
                    for column in along.0..=along.1 {
                        cell(row * columns + column, [index as u32, segment as u32]);
                    }
                }
            }
        }
    };
    let mut starts = vec![0_u32; columns * grid.2 + 1];
    visit(&mut |at, _| starts[at + 1] += 1);
    for at in 1..starts.len() {
        starts[at] += starts[at - 1];
    }
    let mut cursors = starts.clone();
    let mut segments = vec![[0_u32; 2]; starts.last().copied().unwrap_or(0) as usize];
    visit(&mut |at, pair| {
        segments[cursors[at] as usize] = pair;
        cursors[at] += 1;
    });
    (starts, segments)
}

fn listing(
    (corner, columns, rows): (Vec2, usize, usize),
    (river, span): (&River, f32),
    index: usize,
) -> ((usize, usize), (usize, usize)) {
    let (a, b) = (
        Vec2::from(river.points[index]),
        Vec2::from(river.points[index + 1]),
    );
    let margin = Vec2::splat(max(river.half[index], river.half[index + 1]) + span);
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
    form: &'a Form,
    shifts: &'a [f32],
    index: usize,
    a: Vec2,
    b: Vec2,
    run: Vec2,
    length: f32,
    normal: Vec2,
    next: Option<Vec2>,
    start: f32,
    span: f32,
    crests: &'a [Crest],
    knobs: &'a [Knob],
}

#[derive(Clone, Copy, Debug)]
struct Reading {
    along: f32,
    lift: f32,
    edge: f32,
    section: f32,
    floor: f32,
    allow: f32,
    round: f32,
    outer: f32,
    share: f32,
    beyond: f32,
    plain: f32,
    hump: f32,
}

impl Reading {
    fn grain(&self, set: &Deposits, median: f32, (surface, flood): (f32, f32), depth: f32) -> f32 {
        let water = surface + self.lift;
        let above = if water.is_finite() {
            self.section - water
        } else {
            -1.0
        };
        let fades = |value: f32, [full, gone]: [f32; 2]| {
            clamp((gone - value) / max(gone - full, 1.0e-6), 0.0, 1.0)
        };
        let deep = max(depth, 1.0e-3);
        let bar = max(clamp(-set.sides * self.outer, 0.0, 1.0), self.hump);
        let earth = set.earth * fades(above / deep, set.earth_rise);
        if self.share <= 1.0 {
            if above <= 0.0 {
                return median * (set.wet[0] + (set.wet[1] - set.wet[0]) * self.share);
            }
            let rise = clamp(above / (set.fining * deep), 0.0, 1.0);
            let shelf = max(median * set.wet[1] * (1.0 - rise), set.sand);
            return earth + (shelf - earth) * bar;
        }
        let spread = max(set.sand_spread * self.plain, deep);
        let sand = set.sand * smooth(1.0 - self.beyond / spread);
        let silt = set.earth * fades(self.beyond / max(self.plain, deep), set.silt_spread);
        let rim = earth * clamp(1.0 - self.beyond / (set.earth_reach * deep), 0.0, 1.0);
        let reached = fades((self.section - flood) / deep, set.flood);
        (rim + (max(sand, silt) - rim) * bar) * reached
    }
}

fn section(shelf: f32, share: f32, outer: f32, lip: f32) -> f32 {
    let shelf = shelf * clamp(-outer, 0.0, 1.0);
    shelf * share * share * (3.0 - 2.0 * share) + (1.0 - shelf) * libm::powf(share, lip)
}

fn valley(
    set: &Valley,
    (beyond, plain): (f32, f32),
    (rock, wall, earth): (f32, f32, f32),
    (rise, tread): (f32, f32),
) -> f32 {
    let mut left = beyond;
    let mut height = 0.0;
    let mut scarp = rise;
    if rise > 0.0 && plain > 0.0 {
        if left < plain {
            return set.plain_rise * rise * left / plain;
        }
        left -= plain;
        height = set.plain_rise * rise;
        scarp = rise - height;
    }
    let face = rock / wall;
    if left < face {
        return height + wall * left;
    }
    left -= face;
    height += rock;
    if rise > 0.0 {
        let run = scarp / earth;
        if left < run {
            return height + earth * left;
        }
        left -= run;
        height += scarp;
        for _ in 0..set.terraces {
            if left < tread {
                return height + set.tread_rise * rise * left / tread;
            }
            left -= tread;
            height += set.tread_rise * rise;
            let run = rise / earth;
            if left < run {
                return height + earth * left;
            }
            left -= run;
            height += rise;
        }
    }
    height + earth * left
}

impl Segment<'_> {
    #[inline]
    fn stepped(&self, along: f32, across: f32) -> (f32, f32) {
        let place = self.start + self.span * along;
        let (mut lift, mut scour) = (0.0, 0.0);
        for crest in self.crests {
            let lip = crest.lip(across);
            let past = place - lip;
            let [first, last] =
                [self.start, self.start + self.span].map(|end| crest.fallen(end - crest.place));
            lift += crest.drop * (first + (last - first) * along - crest.fallen(past));
            scour += crest.scour(past, lip);
        }
        (lift, scour)
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
        let sides = |values: &[[f32; 2]]| {
            [0, 1].map(|side| {
                values[self.index][side]
                    + (values[self.index + 1][side] - values[self.index][side]) * along
            })
        };
        let [low, high] = sides(&self.line.plain);
        let [near, off] = sides(&self.line.margin);
        if edge >= max(near + low, off + high) {
            return None;
        }
        let across = if gap.dot(self.normal) < 0.0 {
            -distance
        } else {
            distance
        };
        let shift = at(self.shifts);
        let nearest = across - min(max(across - shift, -reach), reach);
        let (from, width, side, (plain, far)) = if nearest < shift {
            (shift - nearest, half + shift, -1.0, (low, near))
        } else {
            (nearest - shift, half - shift, 1.0, (high, off))
        };
        if edge >= far + plain {
            return None;
        }
        let outer = -side * at(&self.line.bend);
        let share = from / max(width, 1.0e-3);
        let unit = clamp(across / max(half, 1.0e-3), -1.0, 1.0);
        let (lift, scour) = if share <= 1.0 {
            self.stepped(along, unit)
        } else {
            (0.0, 0.0)
        };
        let lowered = lift - scour;
        let (bed, depth) = (at(&self.line.bed) + lowered, at(&self.line.depth) - lowered);
        let bar = at(&self.line.bar);
        let (set, vale) = (&self.form.section, &self.form.valley);
        let lens = max(1.0 - unit * unit / (set.island * set.island), 0.0);
        let hump = if share <= 1.0 && bar > 0.0 {
            lens * lens
        } else {
            0.0
        };
        let beyond = from - width;
        let rock = at(&self.line.rock);
        let [gentle, steep] = vale.bank;
        let earth = gentle + (steep - gentle) * clamp(0.5 + 0.5 * outer, 0.0, 1.0);
        let wall = max(at(&self.line.wall), earth);
        let shaped = if share <= 1.0 {
            let [shallow, deep] = vale.walled;
            let cut = (rock / max(depth, 1.0e-3) - shallow) / max(deep - shallow, 1.0e-6);
            let bank = earth + (wall - earth) * smooth(cut);
            let lip = max(bank * width / max(depth, 1.0e-3), 1.0);
            bed + depth * section(set.shelf, share, outer, lip) + bar * hump
        } else {
            let open = clamp((low + high) / max(2.0 * half, 1.0e-3), 0.0, 1.0);
            bed + depth
                + valley(
                    vale,
                    (beyond, plain),
                    (rock, wall, earth),
                    (vale.rise * depth * open, open * vale.tread * 2.0 * half),
                )
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
            lift,
            edge,
            section,
            floor,
            allow: vale.fade * (far + plain - edge),
            round: set.rounding * max(depth, 0.0),
            outer,
            share,
            beyond,
            plain,
            hump,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Taken {
    section: f32,
    holder: [u32; 2],
    edge: f32,
    floor: f32,
    allow: f32,
    round: f32,
}

impl Taken {
    const NONE: Self = Self {
        section: f32::INFINITY,
        holder: [u32::MAX; 2],
        edge: RIVER_FAR,
        floor: f32::INFINITY,
        allow: 0.0,
        round: 0.0,
    };

    #[inline]
    fn add(&mut self, id: [u32; 2], reading: &Reading) {
        self.edge = min(self.edge, reading.edge);
        self.allow = max(self.allow, reading.allow);
        if reading.floor < self.floor {
            self.floor = reading.floor;
            self.round = reading.round;
        }
        if reading.section < self.section || (reading.section == self.section && id < self.holder) {
            self.section = reading.section;
            self.holder = id;
        }
    }

    fn cut(&self, ground: f32) -> f32 {
        if self.holder[0] == u32::MAX {
            return ground;
        }
        let rounding = max(min(self.edge, self.round), 1.0e-3);
        let cut = smooth_min(ground, self.floor, rounding);
        if self.edge > 0.0 {
            max(cut, ground - self.allow)
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
            let span = rivers.spans[id[0] as usize];
            let reach = max(half[index], half[index + 1]) + span + footprint;
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
    pub level: Vec<f32>,
    pub bend: Vec<f32>,
    pub plain: Vec<[f32; 2]>,
    pub bar: Vec<f32>,
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
    bend,
    plain,
    bar,
    rock,
    wall,
    margin,
    head,
    given,
});
engine_codec::record!(Given {
    source,
    width,
    depth,
    flood,
});
engine_codec::record!(Section {
    bend,
    lean,
    shelf,
    island,
    rounding,
});
engine_codec::record!(Valley {
    terraces,
    rise,
    plain_rise,
    bank,
    wall,
    walled,
    tread,
    tread_rise,
    reach,
    fade,
});
engine_codec::record!(Deposits {
    wet,
    fining,
    sand,
    earth,
    earth_rise,
    earth_reach,
    sand_spread,
    silt_spread,
    sides,
    flood,
});
engine_codec::record!(Form {
    profile,
    shape,
    falls,
    pools,
    steps,
    boulders,
    section,
    valley,
    deposits,
});
impl engine_codec::Codec for Rivers {
    fn write(&self, out: &mut engine_codec::Writer) {
        self.form.write(out);
        self.rivers.write(out);
    }

    fn read(input: &mut engine_codec::Reader<'_>) -> Result<Self, engine_codec::Error> {
        let form = Form::read(input)?;
        Ok(Self::new(Vec::read(input)?, form))
    }
}
