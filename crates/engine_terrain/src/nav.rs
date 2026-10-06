//! Navigation over a generated terrain (decision 081): the samples of a
//! pyramid level an agent can stand on (not holes, not under water deeper
//! than it wades, not within its radius of a placed obstacle) and the
//! shortest walk between two points across them, each step between
//! neighboring samples no steeper than it climbs. A* walks the eight
//! neighbors of each sample without cutting corners; the walk is then
//! pulled taut wherever a straight stretch stays walkable, and follows the
//! ground in steps of a few meters. Walks longer than a level's sample
//! budget walk coarser levels and say which.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use engine_math::{max, min};
use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::{Layers, Pyramid};

/// How an agent walks: the finest pyramid level it walks (0 by default);
/// the steepest step it climbs (rise per meter); its radius, by which
/// obstacles widen (meters); the deepest water it wades (meters); the
/// meters around the walk's ends it may stray; and the most samples a walk
/// searches, beyond which it walks a coarser level.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct NavOptions {
    #[cfg_attr(feature = "schema", schemars(range(max = 15)))]
    pub level: u32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.01, max = 10.0)))]
    pub max_slope: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100.0)))]
    pub radius: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100.0)))]
    pub max_water: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 10000.0)))]
    pub margin: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 1024)))]
    pub max_samples: u32,
}

impl Default for NavOptions {
    fn default() -> Self {
        Self {
            level: 0,
            max_slope: 0.8,
            radius: 0.5,
            max_water: 0.6,
            margin: 96.0,
            max_samples: 1 << 20,
        }
    }
}

impl NavOptions {
    /// The first field out of range, and its rule.
    pub fn validate(&self) -> Result<(), (&'static str, &'static str)> {
        let within =
            |value: f32, low: f32, high: f32| value.is_finite() && (low..=high).contains(&value);
        if self.level > 15 {
            return Err(("level", "must be at most 15"));
        }
        if !within(self.max_slope, 0.01, 10.0) {
            return Err(("max_slope", "must be in [0.01, 10]"));
        }
        if !within(self.radius, 0.0, 100.0) {
            return Err(("radius", "must be in [0, 100]"));
        }
        if !within(self.max_water, 0.0, 100.0) {
            return Err(("max_water", "must be in [0, 100]"));
        }
        if !within(self.margin, 0.0, 10_000.0) {
            return Err(("margin", "must be in [0, 10000]"));
        }
        if self.max_samples < 1024 {
            return Err(("max_samples", "must be at least 1024"));
        }
        Ok(())
    }
}

/// A placed obstacle's footprint: its terrain-local center and radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obstacle {
    pub center: Vec2,
    pub radius: f32,
}

/// A walk: its points (terrain-local, following the ground), its length
/// (meters), the level it walked and the samples it searched.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NavPath {
    pub points: Vec<[f32; 3]>,
    pub length: f32,
    pub level: u32,
    pub searched: usize,
}

/// Meters between the points a walk follows the ground by.
const FOLLOW: f32 = 4.0;
/// Samples around an end searched for ground to stand on.
const SNAP: i64 = 8;

/// The shortest walk across `layers`' terrain from terrain-local `from` to
/// `to` among `obstacles`, as `options` walk; `None` where none reaches.
pub fn walk(
    layers: &Layers,
    pyramid: &Pyramid,
    from: Vec2,
    to: Vec2,
    obstacles: &[Obstacle],
    options: &NavOptions,
) -> Option<NavPath> {
    let low = from.min(to) - Vec2::splat(options.margin);
    let high = from.max(to) + Vec2::splat(options.margin);
    // The finest level within the sample budget.
    let mut level = options.level.min(pyramid.levels() - 1);
    while level + 1 < pyramid.levels() {
        let spacing = pyramid.spacing(level);
        let size = (high - low) / spacing;
        if size.x * size.y <= options.max_samples as f32 {
            break;
        }
        level += 1;
    }
    let grid = layers.height_grid(pyramid, level, low, high);
    let (width, height, spacing) = (grid.width, grid.height, grid.spacing);
    if width < 2 || height < 2 {
        return None;
    }
    let first = Vec2::new(grid.first[0] as f32, grid.first[1] as f32) * spacing;
    let position = |sample: usize| {
        first + Vec2::new((sample % width) as f32, (sample / width) as f32) * spacing
    };
    let water = layers.hydrology().map(|hydrology| &hydrology.water);
    let mut walkable: Vec<bool> = (0..width * height)
        .map(|sample| {
            let point = position(sample);
            !layers.hole_at(point.x, point.y)
                && !water
                    .and_then(|water| water.at(point.x, point.y))
                    .is_some_and(|(_, depth)| depth > options.max_water)
        })
        .collect();
    // Obstacles, widened by the agent.
    for obstacle in obstacles {
        let reach = obstacle.radius + options.radius;
        let lowest = ((obstacle.center - Vec2::splat(reach) - first) / spacing).floor();
        let highest = ((obstacle.center + Vec2::splat(reach) - first) / spacing).ceil();
        if highest.x < 0.0
            || highest.y < 0.0
            || lowest.x >= width as f32
            || lowest.y >= height as f32
        {
            continue;
        }
        for row in max(lowest.y, 0.0) as usize..=min(highest.y, (height - 1) as f32) as usize {
            for column in max(lowest.x, 0.0) as usize..=min(highest.x, (width - 1) as f32) as usize
            {
                let sample = row * width + column;
                if position(sample).distance(obstacle.center) <= reach {
                    walkable[sample] = false;
                }
            }
        }
    }
    let at = |point: Vec2| {
        let cell = ((point - first) / spacing).round();
        (cell.x >= 0.0 && cell.y >= 0.0 && (cell.x as usize) < width && (cell.y as usize) < height)
            .then(|| cell.y as usize * width + cell.x as usize)
    };
    // The nearest ground to stand on around each end.
    let snap = |point: Vec2| {
        let center = at(point)?;
        let (column, row) = ((center % width) as i64, (center / width) as i64);
        let mut best: Option<(f32, usize)> = None;
        for dz in -SNAP..=SNAP {
            for dx in -SNAP..=SNAP {
                let (x, z) = (column + dx, row + dz);
                if x < 0 || z < 0 || x >= width as i64 || z >= height as i64 {
                    continue;
                }
                let sample = z as usize * width + x as usize;
                let distance = position(sample).distance(point);
                if walkable[sample] && best.is_none_or(|(nearest, _)| distance < nearest) {
                    best = Some((distance, sample));
                }
            }
        }
        best.map(|(_, sample)| sample)
    };
    let (start, goal) = (snap(from)?, snap(to)?);
    let heights = &grid.heights;
    let climbable =
        |a: usize, b: usize, run: f32| (heights[a] - heights[b]).abs() <= options.max_slope * run;
    // A* over the eight neighbors, never cutting a corner.
    let goal_point = position(goal);
    let mut cost = vec![f32::INFINITY; width * height];
    let mut parent = vec![u32::MAX; width * height];
    let mut open = BinaryHeap::new();
    cost[start] = 0.0;
    open.push(Reverse((
        Ordered(position(start).distance(goal_point)),
        start,
    )));
    let mut searched = 0;
    while let Some(Reverse((_, sample))) = open.pop() {
        if sample == goal {
            break;
        }
        searched += 1;
        let (column, row) = ((sample % width) as i64, (sample / width) as i64);
        for (dx, dz) in [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ] {
            let (x, z) = (column + dx, row + dz);
            if x < 0 || z < 0 || x >= width as i64 || z >= height as i64 {
                continue;
            }
            let next = z as usize * width + x as usize;
            if !walkable[next] {
                continue;
            }
            let diagonal = dx != 0 && dz != 0;
            if diagonal
                && !(walkable[row as usize * width + x as usize]
                    && walkable[z as usize * width + column as usize])
            {
                continue;
            }
            let run = if diagonal {
                spacing * std::f32::consts::SQRT_2
            } else {
                spacing
            };
            if !climbable(sample, next, run) {
                continue;
            }
            let reached = cost[sample] + run + (heights[next] - heights[sample]).abs();
            if reached < cost[next] {
                cost[next] = reached;
                parent[next] = sample as u32;
                open.push(Reverse((
                    Ordered(reached + position(next).distance(goal_point)),
                    next,
                )));
            }
        }
    }
    if !cost[goal].is_finite() {
        return None;
    }
    let mut cells = vec![goal];
    while let Some(&last) = cells.last() {
        let previous = parent[last];
        if previous == u32::MAX {
            break;
        }
        cells.push(previous as usize);
    }
    cells.reverse();
    // The ground's height at `point`, between its samples.
    let ground = |point: Vec2| {
        let cell = ((point - first) / spacing).clamp(
            Vec2::ZERO,
            Vec2::new((width - 1) as f32, (height - 1) as f32),
        );
        let (x0, z0) = (
            min(cell.x.floor(), (width - 2) as f32) as usize,
            min(cell.y.floor(), (height - 2) as f32) as usize,
        );
        let (fx, fz) = (cell.x - x0 as f32, cell.y - z0 as f32);
        let value = |x: usize, z: usize| heights[z * width + x];
        let low = value(x0, z0) + (value(x0 + 1, z0) - value(x0, z0)) * fx;
        let up = value(x0, z0 + 1) + (value(x0 + 1, z0 + 1) - value(x0, z0 + 1)) * fx;
        low + (up - low) * fz
    };
    // Whether walking straight from `a` to `b` stays on ground the agent
    // stands on and climbs.
    let straight = |a: Vec2, b: Vec2| {
        let steps = max((a.distance(b) / (0.5 * spacing)).ceil(), 1.0) as usize;
        let step = a.distance(b) / steps as f32;
        let mut last = ground(a);
        (1..=steps).all(|index| {
            let point = a.lerp(b, index as f32 / steps as f32);
            let height = ground(point);
            let ok = at(point).is_some_and(|sample| walkable[sample])
                && (height - last).abs() <= options.max_slope * step + 1.0e-3;
            last = height;
            ok
        })
    };
    // Pulled taut: from each anchor, straight to the farthest cell it sees.
    let mut anchors = vec![position(cells[0])];
    let mut index = 0;
    while index + 1 < cells.len() {
        let mut reach = index + 1;
        for candidate in (index + 2..cells.len()).rev() {
            if straight(position(cells[index]), position(cells[candidate])) {
                reach = candidate;
                break;
            }
        }
        anchors.push(position(cells[reach]));
        index = reach;
    }
    // Following the ground every few meters.
    let mut points = vec![anchors[0].extend(ground(anchors[0]))];
    for pair in anchors.windows(2) {
        let steps = max((pair[0].distance(pair[1]) / FOLLOW).ceil(), 1.0) as usize;
        for step in 1..=steps {
            let point = pair[0].lerp(pair[1], step as f32 / steps as f32);
            points.push(point.extend(ground(point)));
        }
    }
    let length = points
        .windows(2)
        .map(|pair| pair[0].distance(pair[1]))
        .sum();
    Some(NavPath {
        points: points
            .iter()
            .map(|point| [point.x, point.z, point.y])
            .collect(),
        length,
        level,
        searched,
    })
}

/// A float the heap orders.
#[derive(Clone, Copy, Debug, PartialEq)]
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
