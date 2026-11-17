use engine_codec::{Allowed, Expected};
use engine_core::{ExtraRules, InvalidSetting, fields};
use engine_heightfield::Layers;
use engine_heightfield::noise::smooth;
use engine_heightfield::path::{Bridge, PathSample, distance_to_segment};
use engine_math::{max, min};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

pub const PROFILE_STEP: f32 = 4.0;
pub const MAX_PATH_POINTS: usize = 4096;
pub const MASK_MARGIN: f32 = 32.0;

fields! {
    extra_rules;
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[serde(deny_unknown_fields)]
    pub struct PathLayer {
        #[schemars(length(min = 2, max = MAX_PATH_POINTS))]
        pub points: Vec<[f32; 2]> = Vec::new()
            => range(-Layers::MAX_COORDINATE, Layers::MAX_COORDINATE),
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
                InvalidSetting::new("points", Expected::Array).allowed(Allowed::Range {
                    min: 2.0,
                    max: MAX_PATH_POINTS as f64,
                }),
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

    pub(crate) fn gpu_words(&self, words: &mut Vec<u32>) {
        let at = words.len();
        push_f32(
            words,
            [self.half, self.radius, self.cell, self.min.x, self.min.y],
        );
        words.push(self.columns as u32);
        words.push(self.rows as u32);
        words.extend([0; 6]);
        words[at + 7] = words.len() as u32;
        words.extend(self.starts.iter().copied());
        words[at + 8] = words.len() as u32;
        words.extend(self.segments.iter().copied());
        words[at + 9] = words.len() as u32;
        push_f32(words, self.line.iter().flat_map(|point| [point.x, point.y]));
        words[at + 10] = words.len() as u32;
        push_f32(words, self.along.iter().copied());
        words[at + 11] = words.len() as u32;
        let spans: Vec<[f32; 2]> = self.spans.iter().chain(&self.tunnels).copied().collect();
        push_f32(words, spans.iter().flat_map(|span| *span));
        words[at + 12] = spans.len() as u32;
    }
}

fn push_f32(words: &mut Vec<u32>, values: impl IntoIterator<Item = f32>) {
    words.extend(values.into_iter().map(f32::to_bits));
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

engine_codec::record!(PathLayer {
    points,
    width,
    shoulder,
    smoothing,
    bridge_fill,
    tunnel_cut,
});
engine_codec::record!(PathStats {
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
engine_codec::record!(PathProfile {
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
