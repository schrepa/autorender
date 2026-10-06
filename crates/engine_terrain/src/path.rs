//! Paths (decision 069): roads and tracks along the ground. A path is a
//! line through terrain-local points with a width, shoulders and a
//! smoothing distance. Its bake reads the ground the layers below it give
//! along the line every [`PROFILE_STEP`] meters or so and averages it over
//! the smoothing distance into the road's own gentle grade; the layer then
//! levels the ground across the road to that profile, easing back into the
//! ground over the shoulders, and the `path` mask reads how far a sample
//! lies from the nearest road's edge. Where a road would fill too high
//! over the ground or cross standing water it bridges instead (decision
//! 072): its deck runs straight between the points where it meets the
//! ground, climbing to clear water, and the ground and water under it stay
//! as they are. Where it would cut too deep it tunnels (decision 073): it
//! runs straight between its portals, the ground above stays, and the
//! terrain opens at the portals. A route finder lays paths for agents and
//! games: the
//! cheapest way between two points over the ground, steep grades and water
//! costing more, the steepest grades refused.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use engine_math::{max, min};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::layers::Layers;
use crate::noise::smooth;
use crate::{Generator, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey};

/// Meters between the points of a path's profile, at most.
pub const PROFILE_STEP: f32 = 4.0;
/// Most points a path may name.
pub const MAX_PATH_POINTS: usize = 4096;
/// Meters beyond a path's shoulders within which `path` masks see it.
pub const MASK_MARGIN: f32 = 32.0;
/// What `path` masks read further from every path, and without paths.
pub const PATH_FAR: f32 = 1.0e9;

/// The road nearest a point (by its edge): meters from its line, and half
/// its width; [`PATH_FAR`] meters away beyond every path's masks' reach.
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

    /// Meters from the road's edge, negative on it.
    #[inline]
    pub fn edge(&self) -> f32 {
        self.distance - self.half
    }
}

/// A road or track along the ground (decision 069).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct PathLayer {
    /// Terrain-local meters along x and z the road runs through, 2 to
    /// 4096 of them.
    pub points: Vec<[f32; 2]>,
    /// Meters of level road across, 0.5 to 200.
    #[serde(default = "default_width")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.5, max = 200.0)))]
    pub width: f32,
    /// Meters on each side over which the road eases back into the
    /// ground, 0 to 500.
    #[serde(default = "default_shoulder")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 500.0)))]
    pub shoulder: f32,
    /// Meters along the road over which its height averages the ground's,
    /// 0 to 5000: longer is gentler, cutting deeper and filling higher.
    #[serde(default = "default_smoothing")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 5000.0)))]
    pub smoothing: f32,
    /// Meters the road would fill above the ground beyond which it bridges
    /// instead, as it does over rivers and lakes (decision 072), 0 to 1000;
    /// 0 never bridges.
    #[serde(default = "default_bridge_fill")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1000.0)))]
    pub bridge_fill: f32,
    /// Meters the road would cut into the ground beyond which it tunnels
    /// instead (decision 073), 0 to 2000; 0 never tunnels.
    #[serde(default = "default_tunnel_cut")]
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 2000.0)))]
    pub tunnel_cut: f32,
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

/// Meters a tunnel's span reaches past where its road cuts less than this
/// into the ground: its portals.
const PORTAL_CUT: f32 = 3.0;
/// Meters above a tunnel's road within which the ground opens at its
/// portals.
pub const PORTAL_HEIGHT: f32 = 8.0;
/// Meters past a tunnel road's edges its portals open.
const PORTAL_MARGIN: f32 = 1.5;

/// Meters a bridge's span reaches past where its road fills less than this
/// over the ground: where it meets the ground.
const ABUTMENT: f32 = 0.5;
/// Meters between bridged stretches below which they join into one span.
const SPAN_GAP: f32 = 12.0;
/// Meters of water the hydrology bake's grid holds under a road beyond
/// which it bridges.
pub(crate) const RIVER_DEPTH: f32 = 0.25;
/// Meters a deck clears the water under it by, and the steepest it climbs
/// from its ends toward that (rise per meter).
const CLEARANCE: f32 = 2.0;
const DECK_GRADE: f32 = 0.08;
/// Meters between a bridge's piers, at most, and the depth below its deck
/// the ground must fall to for one to stand there.
const PIER_SPACING: f32 = 24.0;
const PIER_DEPTH: f32 = 3.0;

/// A bridge (decision 072), or a tunnel (decision 073): its road's
/// centerline from one end to the other (terrain-local x and z, the road's
/// height as y) every few meters, its road's width, its piers (where each
/// stands on its centerline, terrain-local at the deck's height, and the
/// ground or water bed there; tunnels have none), and whether it tunnels
/// through the ground rather than spanning over it.
#[derive(Clone, Debug, PartialEq)]
pub struct Bridge {
    pub deck: Vec<Vec3>,
    pub width: f32,
    pub piers: Vec<(Vec3, f32)>,
    pub tunnel: bool,
}

impl PathLayer {
    /// The points the bake reads the ground at, evenly along the line no
    /// more than [`PROFILE_STEP`] apart, and the meters between them.
    pub(crate) fn profile_points(&self) -> (Vec<Vec2>, f32) {
        let line: Vec<Vec2> = self.points.iter().copied().map(Vec2::from).collect();
        let length: f32 = line.windows(2).map(|pair| pair[0].distance(pair[1])).sum();
        let count = max((length / PROFILE_STEP).ceil(), 1.0) as usize;
        let step = length / count as f32;
        let mut points = Vec::with_capacity(count + 1);
        let (mut segment, mut start) = (0, 0.0);
        for index in 0..=count {
            let along = index as f32 * step;
            // The segment holding `along`, the last for the end.
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

/// What a path's bake found: meters along it, the points its profile read
/// the ground at, its steepest grade (rise per meter), the most it cuts
/// into and fills over the ground along its line off its bridges and
/// tunnels (meters), its bridges and the meters they span, and its tunnels
/// and the meters they run.
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

/// A path's profile: its line, the meters along it at each point, the
/// road's height and the ground's every `step` meters along it, the
/// stretches it bridges and tunnels (meters along it) and the square its
/// tunnels lie in, and a grid of the segments near each cell.
#[derive(Clone, Debug, PartialEq)]
pub struct PathProfile {
    line: Vec<Vec2>,
    along: Vec<f32>,
    step: f32,
    heights: Vec<f32>,
    ground: Vec<f32>,
    spans: Vec<[f32; 2]>,
    tunnels: Vec<[f32; 2]>,
    tunnel_box: Option<(Vec2, Vec2)>,
    half: f32,
    shoulder: f32,
    // Square cells `cell` meters wide from `min`, `columns` across, each
    // listing the segments within `radius` of its points.
    radius: f32,
    cell: f32,
    min: Vec2,
    columns: usize,
    rows: usize,
    starts: Vec<u32>,
    segments: Vec<u32>,
    pub stats: PathStats,
}

impl PathProfile {
    /// The profile of `layer` from the ground `raw` read at its profile
    /// points, `step` meters apart, and the surface of any water standing
    /// there.
    pub(crate) fn new(layer: &PathLayer, step: f32, raw: &[f32], water: &[Option<f32>]) -> Self {
        // Averaged over the smoothing distance, twice: a tent along the road.
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
        // The square the tunnels lie in, past their roads' edges.
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
        // The segments within reach of each cell's points: within the reach
        // of its center plus half its diagonal.
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

    /// The square holding every point within its masks' reach.
    pub(crate) fn bounds(&self) -> (Vec2, Vec2) {
        (
            self.min,
            self.min + Vec2::new(self.columns as f32, self.rows as f32) * self.cell,
        )
    }

    /// The road's height `along` meters along it.
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

    /// Whether `along` meters along the road lies on a bridge or in a
    /// tunnel, which leave the ground as it is.
    #[inline]
    fn bridged(&self, along: f32) -> bool {
        self.spans
            .iter()
            .chain(&self.tunnels)
            .any(|&[start, end]| along > start && along < end)
    }

    /// The road's line `along` meters along it.
    fn point_along(&self, along: f32) -> Vec2 {
        point_on(&self.line, &self.along, along)
    }

    /// Whether its tunnels' portals open the ground at terrain-local
    /// `point` (decision 073): within the road's width and
    /// [`PORTAL_MARGIN`] of a tunnel's line, where the ground stands less
    /// than [`PORTAL_HEIGHT`] above the road.
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

    /// The square its tunnels' portals may open the ground in.
    pub(crate) fn portal_bounds(&self) -> Option<(Vec2, Vec2)> {
        self.tunnel_box
    }

    /// The ground's height `along` meters along the road, as the bake read
    /// it.
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

    /// Its bridges (decision 072) and tunnels (decision 073): each road's
    /// centerline every few meters, and bridges' piers no more than
    /// [`PIER_SPACING`] apart wherever the ground falls [`PIER_DEPTH`] below
    /// the deck.
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

    /// The distance from the line to `point`, the road's height at the
    /// line's nearest point and whether a bridge carries it there, within
    /// masks' reach.
    pub(crate) fn nearest(&self, point: Vec2) -> Option<(f32, f32, bool)> {
        let (distance, along) = self.locate(point)?;
        Some((distance, self.height_along(along), self.bridged(along)))
    }

    /// The distance from the line to `point` and the meters along the line
    /// at its nearest point, within masks' reach.
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

    /// How much the road levels the ground `distance` meters from its line:
    /// fully within half its width, easing out across its shoulder.
    #[inline]
    pub(crate) fn weight(&self, distance: f32) -> f32 {
        1.0 - smooth((distance - self.half) / max(self.shoulder, 1.0e-6))
    }

    /// The road as `point` sees it: meters from its line and half its
    /// width; far beyond masks' reach and under bridges, whose ground the
    /// road leaves alone.
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

/// The point of the line through `line` (meters along it at each, `along`)
/// `distance` meters along it.
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

/// Finds where `layer`'s road, of heights `heights` over the ground `raw`
/// (`step` meters apart), tunnels, and lays its tunnels' straight roads
/// into `heights`; returns the tunneled stretches, meters along it.
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
        // Out to its portals, where the road cuts little.
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
        // A straight road between its portals.
        let (from, to) = (heights[first], heights[last]);
        for (index, height) in heights.iter_mut().enumerate().take(last + 1).skip(first) {
            *height = from + (to - from) * (index - first) as f32 / (last - first) as f32;
        }
        spans.push([first as f32 * step, last as f32 * step]);
    }
    spans
}

/// Finds where `layer`'s road, of heights `heights` over the ground `raw`
/// (`step` meters apart, water standing at `water`), bridges outside its
/// `tunnels`, and lays its decks into `heights`; returns the bridged
/// stretches, meters along it.
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
    // Stretches filling too high or crossing water, reaching out to where
    // the road meets the ground.
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
        // Reaching the ends of the road, it starts or ends on the bridge.
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
        // A straight deck between its ends, climbing from each at no more
        // than its grade toward a level that clears the water under it.
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

/// The distance from `point` to the segment from `a` to `b`, and how far
/// along it (0 to 1) its nearest point lies.
#[inline]
fn distance_to_segment(point: Vec2, a: Vec2, b: Vec2) -> (f32, f32) {
    let along = b - a;
    let length = along.length_squared();
    let t = if length > 0.0 {
        min(max((point - a).dot(along) / length, 0.0), 1.0)
    } else {
        0.0
    };
    (point.distance(a + along * t), t)
}

/// Heights of a rectangle of one pyramid level's samples, as generated
/// tiles hold them: `width` by `height` samples from global sample
/// `first`, `spacing` meters apart.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightGrid {
    pub first: [i64; 2],
    pub width: usize,
    pub height: usize,
    pub spacing: f32,
    pub heights: Vec<f32>,
}

impl Layers {
    /// The heights of `level`'s samples from terrain-local `low` to `high`
    /// (clamped to the terrain), generated tile by tile at once: its work
    /// grows with the rectangle.
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
                    .expect("an unlimited budget finishes the tile");
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

/// How a route finder weighs the ground (decision 069): the pyramid level
/// whose samples it walks between, the steepest grade it climbs (rise per
/// meter), how much a meter's cost grows per squared grade, the meters a
/// meter through standing water costs on top, and the meters a turn costs
/// per unit of one less the cosine of its angle (a right angle costs it
/// once, turning back twice), so switchbacks run long legs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct RouteOptions {
    pub level: u32,
    pub max_grade: f32,
    pub grade_cost: f32,
    pub water_cost: f32,
    pub turn_cost: f32,
}

impl Default for RouteOptions {
    fn default() -> Self {
        Self {
            level: 3,
            max_grade: 0.12,
            grade_cost: 100.0,
            water_cost: 50.0,
            turn_cost: 40.0,
        }
    }
}

/// A route found: its terrain-local points, corners rounded, the meters
/// along its walk between samples, the steepest grade of its steps within
/// the limit, and how many of its turning steps cross steeper ground, as
/// hairpins do.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Route {
    pub points: Vec<[f32; 2]>,
    pub length: f32,
    pub steepest: f32,
    pub hairpin_steps: usize,
}

/// Steps from a sample to the samples up to three away along each axis in
/// every direction they lie in alone, in turning order: 32 of them, 7 to 18
/// degrees apart, so routes follow contours closely.
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

/// Steps a route may take after each step: those within four of it in
/// [`STEPS`], turning 45 degrees or so at most.
const TURNS: i64 = 4;
/// The steepest ground (rise per meter) a step turning two or more steps of
/// [`STEPS`] from the one before may cross, whatever the grade limit:
/// hairpins cross the fall line, and the path's profile and shoulders build
/// them as platforms.
const HAIRPIN_GRADE: f32 = 1.0;

/// The cheapest route from `from` to `to` (terrain-local) over the heights
/// `layers` give on `pyramid`, as `options` weigh them: walking between the
/// samples of their level within a rectangle around both ends, half their
/// distance (at least 256 meters) wider on each side, each step costing its
/// length times one plus the grade cost times its squared grade plus the
/// water cost where water stands, and its turn from the step before, which
/// it turns from by 45 degrees or so at most; steps steeper than the limit
/// are refused, but for turning steps, which hairpins may take across ground
/// as steep as [`HAIRPIN_GRADE`]. It searches samples and the way the step
/// reaching each ran together (A*), 32 states a sample. `None` when no route
/// keeps within the limits.
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
    // Where water stands, as the terrain's hydrology bake found it: lakes
    // and rivers (decision 092).
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
    // Each step's way, and what turning from one step to another costs.
    let ways = STEPS.map(|(dx, dz)| Vec2::new(dx as f32, dz as f32).normalize());
    let directions = STEPS.len();
    // States: a sample and the step reaching it, `sample * 32 + step`; the
    // start's states hold every step at no cost, so the first step turns
    // freely.
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
            // A cheaper way to this state was found after this entry.
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
            .expect("walks take steps");
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
    // Straight runs collapse to their ends, within half a sample of the
    // walk, and corners round within two samples.
    let points = round_corners(&simplify(&points, 0.5 * spacing), 2.0 * spacing, 3);
    Some(Route {
        points: points.iter().map(Vec2::to_array).collect(),
        length,
        steepest,
        hairpin_steps,
    })
}

/// `points` without those within `tolerance` of the line between the
/// points kept around them (Douglas and Peucker).
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

/// `points` with each corner turning more than about 8 degrees cut
/// `rounds` times (Chaikin), the first cut at most `cut` meters and a
/// quarter of each side from the corner, each later one half as far.
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

/// A cost ordered totally, lowest first in the route finder's queue.
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
