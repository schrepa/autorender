use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use engine_core::fields;
use engine_heightfield::path::distance_to_segment;
use engine_math::{max, min};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::{
    Generator, Layers, MAX_LEVEL, Pyramid, TILE_QUADS, TILE_SAMPLES, TILE_STRIDE, TileKey,
};

#[derive(Clone, Debug, PartialEq)]
pub struct HeightGrid {
    pub first: [i64; 2],
    pub width: usize,
    pub height: usize,
    pub spacing: f32,
    pub heights: Vec<f32>,
}

pub fn height_grid(
    below: &Layers,
    pyramid: &Pyramid,
    level: u32,
    low: Vec2,
    high: Vec2,
) -> HeightGrid {
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
    let layers = Arc::new(below.clone());
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

fields! {
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
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
    let grid = height_grid(
        layers,
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
