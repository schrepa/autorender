use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::MAX_LEVEL;
use engine_math::{max, min};
use engine_settings::{ExtraRules, InvalidSetting, settings};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::layers::Layers;
use crate::noise::smooth;
use crate::{Generator, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey};

pub const PROFILE_STEP: f32 = 4.0;
pub const MAX_PATH_POINTS: usize = 4096;
pub const MASK_MARGIN: f32 = 32.0;
pub const PATH_FAR: f32 = 1.0e9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathSample {
    pub distance: f32,
    pub half: f32,
}

impl PathSample {
    pub const FAR: Self = Self {
        distance: PATH_FAR,
        half: 0.0,
    };

    #[inline]
    pub fn edge(&self) -> f32 {
        self.distance - self.half
    }
}

settings! {
    extra_rules;
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct PathLayer {
        #[cfg_attr(feature = "schema", schemars(length(min = 2, max = MAX_PATH_POINTS)))]
        pub points: Vec<[f32; 2]> = Vec::new()
            => range(-crate::Layers::MAX_COORDINATE, crate::Layers::MAX_COORDINATE),
        #[serde(default = "default_width")]
        pub width: f32 = default_width() => range(0.5, 200.0),
        #[serde(default = "default_shoulder")]
        pub shoulder: f32 = default_shoulder() => range(0.0, 500.0),
        #[serde(default = "default_smoothing")]
        pub smoothing: f32 = default_smoothing() => range(0.0, 5000.0),
        #[serde(default = "default_bridge_fill")]
        pub bridge_fill: f32 = default_bridge_fill() => range(0.0, 1000.0),
        #[serde(default = "default_tunnel_cut")]
        pub tunnel_cut: f32 = default_tunnel_cut() => range(0.0, 2000.0),
    }
}

impl ExtraRules for PathLayer {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if (2..=MAX_PATH_POINTS).contains(&self.points.len()) {
            Ok(())
        } else {
            Err(
                InvalidSetting::new("points", engine_settings::Expected::Array).allowed(
                    engine_settings::Allowed::Range {
                        min: 2.0,
                        max: MAX_PATH_POINTS as f64,
                    },
                ),
            )
        }
    }
}

fn default_width() -> f32 {
    6.0
}

fn default_shoulder() -> f32 {
    6.0
}

fn default_smoothing() -> f32 {
    80.0
}

fn default_bridge_fill() -> f32 {
    4.0
}

fn default_tunnel_cut() -> f32 {
    25.0
}

const PORTAL_CUT: f32 = 3.0;
pub const PORTAL_HEIGHT: f32 = 8.0;
const PORTAL_MARGIN: f32 = 1.5;

const ABUTMENT: f32 = 0.5;
const SPAN_GAP: f32 = 12.0;
pub(crate) const RIVER_DEPTH: f32 = 0.25;
const CLEARANCE: f32 = 2.0;
const DECK_GRADE: f32 = 0.08;
const PIER_SPACING: f32 = 24.0;
const PIER_DEPTH: f32 = 3.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Bridge {
    pub deck: Vec<Vec3>,
    pub width: f32,
    pub piers: Vec<(Vec3, f32)>,
    pub tunnel: bool,
}

impl PathLayer {
    pub(crate) fn profile_points(&self) -> (Vec<Vec2>, f32) {
        let line: Vec<Vec2> = self.points.iter().copied().map(Vec2::from).collect();
        let length: f32 = line.windows(2).map(|pair| pair[0].distance(pair[1])).sum();
        let count = max((length / PROFILE_STEP).ceil(), 1.0) as usize;
        let step = length / count as f32;
        let mut points = Vec::with_capacity(count + 1);
        let (mut segment, mut start) = (0, 0.0);
        for index in 0..=count {
            let along = index as f32 * step;
            while segment + 2 < line.len()
                && start + line[segment].distance(line[segment + 1]) < along
            {
                start += line[segment].distance(line[segment + 1]);
                segment += 1;
            }
            let (a, b) = (line[segment], line[segment + 1]);
            let run = a.distance(b);
            let t = if run > 0.0 {
                min(max((along - start) / run, 0.0), 1.0)
            } else {
                0.0
            };
            points.push(a + (b - a) * t);
        }
        (points, step)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct PathStats {
    pub length: f32,
    pub samples: usize,
    pub steepest: f32,
    pub deepest_cut: f32,
    pub highest_fill: f32,
    pub bridges: usize,
    pub bridged: f32,
    pub tunnels: usize,
    pub tunneled: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathProfile {
    pub(crate) line: Vec<Vec2>,
    pub(crate) along: Vec<f32>,
    step: f32,
    heights: Vec<f32>,
    ground: Vec<f32>,
    pub(crate) spans: Vec<[f32; 2]>,
    pub(crate) tunnels: Vec<[f32; 2]>,
    tunnel_box: Option<(Vec2, Vec2)>,
    pub(crate) half: f32,
    shoulder: f32,
    pub(crate) radius: f32,
    pub(crate) cell: f32,
    pub(crate) min: Vec2,
    pub(crate) columns: usize,
    pub(crate) rows: usize,
    pub(crate) starts: Vec<u32>,
    pub(crate) segments: Vec<u32>,
    pub stats: PathStats,
}

impl PathProfile {
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn new(layer: &PathLayer, step: f32, raw: &[f32], water: &[Option<f32>]) -> Self {
        let radius = if step > 0.0 {
            (layer.smoothing / (2.0 * step)).round() as usize
        } else {
            0
        };
        let average = |values: &[f32]| -> Vec<f32> {
            let last = values.len() - 1;
            (0..values.len())
                .map(|index| {
                    let (first, end) = (index.saturating_sub(radius), (index + radius).min(last));
                    values[first..=end].iter().sum::<f32>() / (end - first + 1) as f32
                })
                .collect()
        };
        let mut heights = if radius == 0 {
            raw.to_vec()
        } else {
            average(&average(raw))
        };
        let tunnels = tunnel(layer, step, raw, &mut heights);
        let spans = bridge(layer, step, raw, water, &mut heights, &tunnels);
        let mut stats = PathStats {
            samples: raw.len(),
            bridges: spans.len(),
            bridged: spans.iter().map(|[start, end]| end - start).sum(),
            tunnels: tunnels.len(),
            tunneled: tunnels.iter().map(|[start, end]| end - start).sum(),
            ..PathStats::default()
        };
        for (index, (&road, &ground)) in heights.iter().zip(raw).enumerate() {
            let along = index as f32 * step;
            if !spans
                .iter()
                .chain(&tunnels)
                .any(|&[start, end]| (start..=end).contains(&along))
            {
                stats.deepest_cut = max(stats.deepest_cut, ground - road);
                stats.highest_fill = max(stats.highest_fill, road - ground);
            }
            if index > 0 && step > 0.0 {
                stats.steepest = max(stats.steepest, (road - heights[index - 1]).abs() / step);
            }
        }
        let line: Vec<Vec2> = layer.points.iter().copied().map(Vec2::from).collect();
        let mut along = Vec::with_capacity(line.len());
        let mut length = 0.0;
        for (index, &point) in line.iter().enumerate() {
            if index > 0 {
                length += line[index - 1].distance(point);
            }
            along.push(length);
        }
        stats.length = length;
        let tunnel_box = tunnels
            .iter()
            .fold(None, |bounds: Option<(Vec2, Vec2)>, &[start, end]| {
                let count = max(((end - start) / PROFILE_STEP).ceil(), 1.0) as usize;
                (0..=count).fold(bounds, |bounds, index| {
                    let point = point_on(
                        &line,
                        &along,
                        start + (end - start) * index as f32 / count as f32,
                    );
                    Some(match bounds {
                        Some((low, high)) => (low.min(point), high.max(point)),
                        None => (point, point),
                    })
                })
            })
            .map(|(low, high)| {
                let margin = Vec2::splat(0.5 * layer.width + PORTAL_MARGIN + 1.0);
                (low - margin, high + margin)
            });
        let half = 0.5 * layer.width;
        let reach = half + layer.shoulder + MASK_MARGIN;
        let cell = max(reach, 16.0);
        let (mut low, mut high) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for &point in &line {
            low = low.min(point);
            high = high.max(point);
        }
        let corner = ((low - Vec2::splat(reach)) / cell).floor() * cell;
        let columns = ((high.x + reach - corner.x) / cell).floor() as usize + 1;
        let rows = ((high.y + reach - corner.y) / cell).floor() as usize + 1;
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); columns * rows];
        let within = reach + cell * std::f32::consts::FRAC_1_SQRT_2;
        for segment in 0..line.len() - 1 {
            let (a, b) = (line[segment], line[segment + 1]);
            let first = ((a.min(b) - Vec2::splat(reach) - corner) / cell).floor();
            let last = ((a.max(b) + Vec2::splat(reach) - corner) / cell).floor();
            let column_end = (last.x as usize).min(columns - 1);
            let row_end = (last.y as usize).min(rows - 1);
            for row in max(first.y, 0.0) as usize..=row_end {
                for column in max(first.x, 0.0) as usize..=column_end {
                    let center = corner + (Vec2::new(column as f32, row as f32) + 0.5) * cell;
                    if distance_to_segment(center, a, b).0 <= within {
                        lists[row * columns + column].push(segment as u32);
                    }
                }
            }
        }
        let mut starts = Vec::with_capacity(lists.len() + 1);
        let mut segments = Vec::new();
        for list in &lists {
            starts.push(segments.len() as u32);
            segments.extend_from_slice(list);
        }
        starts.push(segments.len() as u32);
        Self {
            line,
            along,
            step,
            heights,
            ground: raw.to_vec(),
            spans,
            tunnels,
            tunnel_box,
            half,
            shoulder: layer.shoulder,
            radius: reach,
            cell,
            min: corner,
            columns,
            rows,
            starts,
            segments,
            stats,
        }
    }

    pub(crate) fn bounds(&self) -> (Vec2, Vec2) {
        (
            self.min,
            self.min + Vec2::new(self.columns as f32, self.rows as f32) * self.cell,
        )
    }

    #[inline]
    fn height_along(&self, along: f32) -> f32 {
        if self.step <= 0.0 {
            return self.heights[0];
        }
        let at = max(along / self.step, 0.0);
        let index = (at as usize).min(self.heights.len() - 1);
        let next = (index + 1).min(self.heights.len() - 1);
        let t = min(at - index as f32, 1.0);
        self.heights[index] + (self.heights[next] - self.heights[index]) * t
    }

    #[inline]
    fn bridged(&self, along: f32) -> bool {
        self.spans
            .iter()
            .chain(&self.tunnels)
            .any(|&[start, end]| along > start && along < end)
    }

    fn point_along(&self, along: f32) -> Vec2 {
        point_on(&self.line, &self.along, along)
    }

    pub(crate) fn portal_hole(&self, point: Vec2) -> bool {
        let Some((low, high)) = self.tunnel_box else {
            return false;
        };
        if point.cmplt(low).any() || point.cmpgt(high).any() {
            return false;
        }
        let Some((distance, along)) = self.locate(point) else {
            return false;
        };
        distance < self.half + PORTAL_MARGIN
            && self
                .tunnels
                .iter()
                .any(|&[start, end]| along > start && along < end)
            && self.ground_along(along) - self.height_along(along) < PORTAL_HEIGHT
    }

    pub(crate) fn portal_bounds(&self) -> Option<(Vec2, Vec2)> {
        self.tunnel_box
    }

    fn ground_along(&self, along: f32) -> f32 {
        if self.step <= 0.0 {
            return self.ground[0];
        }
        let at = max(along / self.step, 0.0);
        let index = (at as usize).min(self.ground.len() - 1);
        let next = (index + 1).min(self.ground.len() - 1);
        let t = min(at - index as f32, 1.0);
        self.ground[index] + (self.ground[next] - self.ground[index]) * t
    }

    pub fn bridges(&self) -> Vec<Bridge> {
        let tunnels = self.tunnels.iter().map(|&span| (span, true));
        self.spans
            .iter()
            .map(|&span| (span, false))
            .chain(tunnels)
            .map(|([start, end], tunnel)| {
                let length = end - start;
                let count = max((length / PROFILE_STEP).ceil(), 1.0) as usize;
                let deck = (0..=count)
                    .map(|index| {
                        let along = start + length * index as f32 / count as f32;
                        let point = self.point_along(along);
                        Vec3::new(point.x, self.height_along(along), point.y)
                    })
                    .collect();
                let bays = if tunnel {
                    1
                } else {
                    max((length / PIER_SPACING).ceil(), 1.0) as usize
                };
                let piers = (1..bays)
                    .filter_map(|index| {
                        let along = start + length * index as f32 / bays as f32;
                        let (top, ground) = (self.height_along(along), self.ground_along(along));
                        (top - ground >= PIER_DEPTH).then(|| {
                            let point = self.point_along(along);
                            (Vec3::new(point.x, top, point.y), ground)
                        })
                    })
                    .collect();
                Bridge {
                    deck,
                    width: 2.0 * self.half,
                    piers,
                    tunnel,
                }
            })
            .collect()
    }

    pub(crate) fn nearest(&self, point: Vec2) -> Option<(f32, f32, bool)> {
        let (distance, along) = self.locate(point)?;
        Some((distance, self.height_along(along), self.bridged(along)))
    }

    fn locate(&self, point: Vec2) -> Option<(f32, f32)> {
        let cell = ((point - self.min) / self.cell).floor();
        if !(cell.x >= 0.0 && cell.y >= 0.0) {
            return None;
        }
        let (column, row) = (cell.x as usize, cell.y as usize);
        if column >= self.columns || row >= self.rows {
            return None;
        }
        let at = row * self.columns + column;
        let mut best: Option<(f32, usize, f32)> = None;
        for &segment in &self.segments[self.starts[at] as usize..self.starts[at + 1] as usize] {
            let segment = segment as usize;
            let (distance, t) =
                distance_to_segment(point, self.line[segment], self.line[segment + 1]);
            if best.is_none_or(|(nearest, _, _)| distance < nearest) {
                best = Some((distance, segment, t));
            }
        }
        let (distance, segment, t) = best.filter(|(distance, _, _)| *distance < self.radius)?;
        let along = self.along[segment] + (self.along[segment + 1] - self.along[segment]) * t;
        Some((distance, along))
    }

    #[inline]
    pub(crate) fn weight(&self, distance: f32) -> f32 {
        1.0 - smooth((distance - self.half) / max(self.shoulder, 1.0e-6))
    }

    #[inline]
    pub fn sample(&self, point: Vec2) -> PathSample {
        match self.nearest(point) {
            Some((distance, _, false)) => PathSample {
                distance,
                half: self.half,
            },
            _ => PathSample::FAR,
        }
    }
}

fn point_on(line: &[Vec2], along: &[f32], distance: f32) -> Vec2 {
    let segment = along
        .partition_point(|&value| value <= distance)
        .saturating_sub(1)
        .min(line.len() - 2);
    let run = along[segment + 1] - along[segment];
    let t = if run > 0.0 {
        min(max((distance - along[segment]) / run, 0.0), 1.0)
    } else {
        0.0
    };
    line[segment] + (line[segment + 1] - line[segment]) * t
}

fn tunnel(layer: &PathLayer, step: f32, raw: &[f32], heights: &mut [f32]) -> Vec<[f32; 2]> {
    if layer.tunnel_cut <= 0.0 || step <= 0.0 || heights.len() < 3 {
        return Vec::new();
    }
    let count = heights.len();
    let cut = |heights: &[f32], index: usize| raw[index] - heights[index];
    let mut runs: Vec<[usize; 2]> = Vec::new();
    let mut index = 0;
    while index < count {
        if cut(heights, index) <= layer.tunnel_cut {
            index += 1;
            continue;
        }
        let mut first = index;
        while first > 0 && cut(heights, first - 1) > PORTAL_CUT {
            first -= 1;
        }
        let mut last = index;
        while last + 1 < count && cut(heights, last + 1) > PORTAL_CUT {
            last += 1;
        }
        match runs.last_mut() {
            Some(previous) if (first as f32 - previous[1] as f32) * step < SPAN_GAP => {
                previous[1] = last;
            }
            _ => runs.push([first, last]),
        }
        index = last + 1;
    }
    let mut spans = Vec::with_capacity(runs.len());
    for [first, last] in runs {
        if last <= first {
            continue;
        }
        let (from, to) = (heights[first], heights[last]);
        for (index, height) in heights.iter_mut().enumerate().take(last + 1).skip(first) {
            *height = from + (to - from) * (index - first) as f32 / (last - first) as f32;
        }
        spans.push([first as f32 * step, last as f32 * step]);
    }
    spans
}

fn bridge(
    layer: &PathLayer,
    step: f32,
    raw: &[f32],
    water: &[Option<f32>],
    heights: &mut [f32],
    tunnels: &[[f32; 2]],
) -> Vec<[f32; 2]> {
    if layer.bridge_fill <= 0.0 || step <= 0.0 || heights.len() < 3 {
        return Vec::new();
    }
    let count = heights.len();
    let tunneled = |index: usize| {
        let along = index as f32 * step;
        tunnels
            .iter()
            .any(|&[start, end]| (start..=end).contains(&along))
    };
    let wet = |index: usize| {
        water
            .get(index)
            .copied()
            .flatten()
            .filter(|_| !tunneled(index))
    };
    let fill = |index: usize| {
        if tunneled(index) {
            0.0
        } else {
            heights[index] - raw[index]
        }
    };
    let mut runs: Vec<[usize; 2]> = Vec::new();
    let mut index = 0;
    while index < count {
        if !(fill(index) > layer.bridge_fill || wet(index).is_some()) {
            index += 1;
            continue;
        }
        let mut end = index;
        while end + 1 < count && (fill(end + 1) > layer.bridge_fill || wet(end + 1).is_some()) {
            end += 1;
        }
        let mut first = index;
        while first > 0 && (fill(first - 1) > ABUTMENT || wet(first - 1).is_some()) {
            first -= 1;
        }
        let mut last = end;
        while last + 1 < count && (fill(last + 1) > ABUTMENT || wet(last + 1).is_some()) {
            last += 1;
        }
        match runs.last_mut() {
            Some(previous) if (first as f32 - previous[1] as f32) * step < SPAN_GAP => {
                previous[1] = last;
            }
            _ => runs.push([first, last]),
        }
        index = last + 1;
    }
    let mut spans = Vec::with_capacity(runs.len());
    for [first, last] in runs {
        if last <= first {
            continue;
        }
        let (from, to) = (heights[first], heights[last]);
        let line =
            |index: usize| from + (to - from) * (index - first) as f32 / (last - first) as f32;
        let top = (first..=last)
            .filter_map(|index| wet(index).map(|surface| surface + CLEARANCE))
            .fold(f32::NEG_INFINITY, max);
        for (index, height) in heights.iter_mut().enumerate().take(last + 1).skip(first) {
            let ramp = min(
                from + DECK_GRADE * (index - first) as f32 * step,
                to + DECK_GRADE * (last - index) as f32 * step,
            );
            *height = max(line(index), min(top, ramp));
        }
        spans.push([first as f32 * step, last as f32 * step]);
    }
    spans
}

#[inline]
pub(crate) fn distance_to_segment(point: Vec2, a: Vec2, b: Vec2) -> (f32, f32) {
    let along = b - a;
    let length = along.length_squared();
    let t = if length > 0.0 {
        min(max((point - a).dot(along) / length, 0.0), 1.0)
    } else {
        0.0
    };
    (point.distance(a + along * t), t)
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeightGrid {
    pub first: [i64; 2],
    pub width: usize,
    pub height: usize,
    pub spacing: f32,
    pub heights: Vec<f32>,
}

impl Layers {
    pub fn height_grid(&self, pyramid: &Pyramid, level: u32, low: Vec2, high: Vec2) -> HeightGrid {
        let level = level.min(pyramid.levels() - 1);
        let spacing = pyramid.spacing(level);
        let [tiles_x, tiles_z] = pyramid.tiles(level);
        let quads = i64::from(TILE_QUADS);
        let last = [i64::from(tiles_x) * quads, i64::from(tiles_z) * quads];
        let first = [
            ((low.x / spacing).floor() as i64).clamp(0, last[0]),
            ((low.y / spacing).floor() as i64).clamp(0, last[1]),
        ];
        let end = [
            ((high.x / spacing).ceil() as i64).clamp(first[0], last[0]),
            ((high.y / spacing).ceil() as i64).clamp(first[1], last[1]),
        ];
        let (width, height) = (
            (end[0] - first[0] + 1) as usize,
            (end[1] - first[1] + 1) as usize,
        );
        let mut heights = vec![0.0; width * height];
        let layers = Arc::new(self.clone());
        let tile_of = |sample: i64, tiles: u32| (sample / quads).min(i64::from(tiles) - 1);
        let (stride, samples) = (i64::from(TILE_STRIDE), i64::from(TILE_SAMPLES));
        for tz in tile_of(first[1], tiles_z)..=tile_of(end[1], tiles_z) {
            for tx in tile_of(first[0], tiles_x)..=tile_of(end[0], tiles_x) {
                let key = TileKey {
                    level: level as u8,
                    x: tx as u16,
                    z: tz as u16,
                };
                let mut unlimited = u64::MAX;
                let tile = Generator::new(layers.clone(), pyramid, key)
                    .advance(&mut unlimited)
                    .expect("unlimited_budget_finishes");
                let rows = (first[1] - tz * quads).max(0)..=(end[1] - tz * quads).min(samples - 1);
                let columns =
                    (first[0] - tx * quads).max(0)..=(end[0] - tx * quads).min(samples - 1);
                for row in rows {
                    let to = (tz * quads + row - first[1]) as usize * width;
                    for column in columns.clone() {
                        heights[to + (tx * quads + column - first[0]) as usize] =
                            tile.heights()[((row + 1) * stride + column + 1) as usize];
                    }
                }
            }
        }
        HeightGrid {
            first,
            width,
            height,
            spacing,
            heights,
        }
    }
}

settings! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(default, deny_unknown_fields)]
    pub struct RouteOptions {
        pub level: u32 = 3 => range(0, MAX_LEVEL),
        pub max_grade: f32 = 0.12 => range(0.001, 10.0),
        pub grade_cost: f32 = 100.0 => range(0.0, 1.0e6),
        pub water_cost: f32 = 50.0 => range(0.0, 1.0e6),
        pub turn_cost: f32 = 40.0 => range(0.0, 1.0e6),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Route {
    pub points: Vec<[f32; 2]>,
    pub length: f32,
    pub steepest: f32,
    pub hairpin_steps: usize,
}

const STEPS: [(i64, i64); 32] = [
    (1, 0),
    (3, 1),
    (2, 1),
    (3, 2),
    (1, 1),
    (2, 3),
    (1, 2),
    (1, 3),
    (0, 1),
    (-1, 3),
    (-1, 2),
    (-2, 3),
    (-1, 1),
    (-3, 2),
    (-2, 1),
    (-3, 1),
    (-1, 0),
    (-3, -1),
    (-2, -1),
    (-3, -2),
    (-1, -1),
    (-2, -3),
    (-1, -2),
    (-1, -3),
    (0, -1),
    (1, -3),
    (1, -2),
    (2, -3),
    (1, -1),
    (3, -2),
    (2, -1),
    (3, -1),
];

const TURNS: i64 = 4;
const HAIRPIN_GRADE: f32 = 1.0;

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub fn route(
    layers: &Layers,
    pyramid: &Pyramid,
    from: Vec2,
    to: Vec2,
    options: &RouteOptions,
) -> Option<Route> {
    let margin = max(0.5 * from.distance(to), 256.0);
    let grid = layers.height_grid(
        pyramid,
        options.level,
        from.min(to) - Vec2::splat(margin),
        from.max(to) + Vec2::splat(margin),
    );
    let spacing = grid.spacing;
    let (width, height) = (grid.width as i64, grid.height as i64);
    let local = |sample: (i64, i64)| {
        Vec2::new(
            (grid.first[0] + sample.0) as f32 * spacing,
            (grid.first[1] + sample.1) as f32 * spacing,
        )
    };
    let node = |point: Vec2| {
        (
            ((point.x / spacing).round() as i64 - grid.first[0]).clamp(0, width - 1),
            ((point.y / spacing).round() as i64 - grid.first[1]).clamp(0, height - 1),
        )
    };
    let index = |sample: (i64, i64)| (sample.1 * width + sample.0) as usize;
    let sample_of = |index: usize| (index as i64 % width, index as i64 / width);
    let water = layers.hydrology();
    let wet: Vec<bool> = (0..grid.heights.len())
        .map(|at| {
            let point = local(sample_of(at));
            water.is_some_and(|water| {
                water
                    .water_at(point.x, point.y)
                    .is_some_and(|(surface, _)| surface > grid.heights[at])
            })
        })
        .collect();
    let (start, goal) = (node(from), node(to));
    let distance = |a: (i64, i64), b: (i64, i64)| {
        (((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1)) as f32).sqrt() * spacing
    };
    let ways = STEPS.map(|(dx, dz)| Vec2::new(dx as f32, dz as f32).normalize());
    let directions = STEPS.len();
    let mut cost = vec![f32::INFINITY; grid.heights.len() * directions];
    let mut came = vec![u32::MAX; grid.heights.len() * directions];
    let mut open = BinaryHeap::new();
    for way in 0..directions {
        let state = index(start) * directions + way;
        cost[state] = 0.0;
        open.push(Reverse((Ordered(distance(start, goal)), state as u32)));
    }
    let mut reached = None;
    while let Some(Reverse((Ordered(estimate), state))) = open.pop() {
        let state = state as usize;
        let (at, way) = (state / directions, state % directions);
        let current = sample_of(at);
        let here = cost[state];
        if estimate > here + distance(current, goal) {
            continue;
        }
        if current == goal {
            reached = Some(state);
            break;
        }
        let rise = grid.heights[at];
        for turn in -TURNS..=TURNS {
            let next_way = (way as i64 + turn).rem_euclid(directions as i64) as usize;
            let (dx, dz) = STEPS[next_way];
            let next = (current.0 + dx, current.1 + dz);
            if next.0 < 0 || next.1 < 0 || next.0 >= width || next.1 >= height {
                continue;
            }
            let run = distance(current, next);
            let grade = (grid.heights[index(next)] - rise).abs() / run;
            let limit = if turn.abs() >= 2 {
                max(options.max_grade, HAIRPIN_GRADE)
            } else {
                options.max_grade
            };
            if grade > limit {
                continue;
            }
            let water = if wet[index(next)] {
                options.water_cost
            } else {
                0.0
            };
            let bend = 1.0 - ways[way].dot(ways[next_way]);
            let total = here
                + run * (1.0 + options.grade_cost * grade * grade + water)
                + options.turn_cost * bend;
            let next_state = index(next) * directions + next_way;
            if total < cost[next_state] {
                cost[next_state] = total;
                came[next_state] = state as u32;
                open.push(Reverse((
                    Ordered(total + distance(next, goal)),
                    next_state as u32,
                )));
            }
        }
    }
    let mut state = reached?;
    let mut walk = vec![goal];
    while came[state] != u32::MAX {
        state = came[state] as usize;
        walk.push(sample_of(state / directions));
    }
    walk.reverse();
    let (mut length, mut steepest, mut hairpin_steps) = (0.0, 0.0_f32, 0);
    let mut previous: Option<usize> = None;
    for pair in walk.windows(2) {
        let step = (pair[1].0 - pair[0].0, pair[1].1 - pair[0].1);
        let way = STEPS
            .iter()
            .position(|&candidate| candidate == step)
            .expect("walk_step_known");
        let run = distance(pair[0], pair[1]);
        length += run;
        let grade = (grid.heights[index(pair[1])] - grid.heights[index(pair[0])]).abs() / run;
        let turning = previous.is_some_and(|previous| {
            let turn = (way as i64 - previous as i64).rem_euclid(directions as i64);
            turn.min(directions as i64 - turn) >= 2
        });
        if turning && grade > options.max_grade {
            hairpin_steps += 1;
        } else {
            steepest = max(steepest, grade);
        }
        previous = Some(way);
    }
    let points: Vec<Vec2> = walk.iter().map(|&sample| local(sample)).collect();
    let points = round_corners(&simplify(&points, 0.5 * spacing), 2.0 * spacing, 3);
    Some(Route {
        points: points.iter().map(Vec2::to_array).collect(),
        length,
        steepest,
        hairpin_steps,
    })
}

fn simplify(points: &[Vec2], tolerance: f32) -> Vec<Vec2> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut spans = vec![(0, points.len() - 1)];
    while let Some((first, last)) = spans.pop() {
        let (a, b) = (points[first], points[last]);
        let mut farthest = (0.0_f32, first);
        for (index, &point) in points.iter().enumerate().take(last).skip(first + 1) {
            let off = distance_to_segment(point, a, b).0;
            if off > farthest.0 {
                farthest = (off, index);
            }
        }
        if farthest.0 > tolerance {
            keep[farthest.1] = true;
            spans.push((first, farthest.1));
            spans.push((farthest.1, last));
        }
    }
    points
        .iter()
        .zip(keep)
        .filter_map(|(&point, kept)| kept.then_some(point))
        .collect()
}

fn round_corners(points: &[Vec2], cut: f32, rounds: usize) -> Vec<Vec2> {
    let mut points = points.to_vec();
    let mut cut = cut;
    for _ in 0..rounds {
        if points.len() < 3 {
            break;
        }
        let mut next = Vec::with_capacity(2 * points.len());
        next.push(points[0]);
        for corner in points.windows(3) {
            let (a, b, c) = (corner[0], corner[1], corner[2]);
            let (back, ahead) = ((a - b).normalize_or_zero(), (c - b).normalize_or_zero());
            if -back.dot(ahead) > 0.99 {
                next.push(b);
                continue;
            }
            next.push(b + back * min(cut, 0.25 * a.distance(b)));
            next.push(b + ahead * min(cut, 0.25 * b.distance(c)));
        }
        next.push(points[points.len() - 1]);
        points = next;
        cut *= 0.5;
    }
    points
}

#[derive(Clone, Copy, PartialEq)]
struct Ordered(f32);

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

crate::codec::record!(PathLayer {
    points,
    width,
    shoulder,
    smoothing,
    bridge_fill,
    tunnel_cut,
});
crate::codec::record!(PathStats {
    length,
    samples,
    steepest,
    deepest_cut,
    highest_fill,
    bridges,
    bridged,
    tunnels,
    tunneled,
});
crate::codec::record!(PathProfile {
    line,
    along,
    step,
    heights,
    ground,
    spans,
    tunnels,
    tunnel_box,
    half,
    shoulder,
    radius,
    cell,
    min,
    columns,
    rows,
    starts,
    segments,
    stats,
});
