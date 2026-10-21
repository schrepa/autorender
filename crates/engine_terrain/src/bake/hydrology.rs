//! The hydrology bake: the level's grid gathered, then where water runs
//! and pools over it at once, then the finest ground along each river's
//! middle read a point at a time, and its channel and water set from it,
//! upstream rivers first.

use engine_math::collections::HashMap;
use std::sync::Arc;

use engine_math::{max, min};
use glam::Vec2;

use super::{Bake, Gather};
use crate::Pyramid;
use crate::hydraulics::{Flow, Reach, river};
use crate::hydrology::{Hydrology, HydrologyLayer, bake};
use crate::layers::Layers;
use crate::rivers::{RiverLine, Rivers};

/// Where water runs over the layers before it and the channels it carves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HydrologyBake;

/// A hydrology bake in progress: its layer, its level's grid, and once
/// gathered and computed, its rivers being read.
pub struct HydrologyWork {
    layer: HydrologyLayer,
    gather: Gather,
    rivers: Option<RiverJob>,
}

impl Bake for HydrologyBake {
    type Layer = HydrologyLayer;
    type Found = Hydrology;
    type Work = HydrologyWork;

    fn start(layer: &HydrologyLayer, pyramid: &Pyramid) -> HydrologyWork {
        HydrologyWork {
            layer: *layer,
            gather: Gather::new(layer.level, pyramid),
            rivers: None,
        }
    }

    fn advance(
        work: &mut HydrologyWork,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Hydrology> {
        if let Some(rivers) = &mut work.rivers {
            return rivers.advance(below, pyramid, budget);
        }
        if !work.gather.advance(below, pyramid, budget) {
            return None;
        }
        // Its rivers' channels and water follow from the finest ground
        // along them, read a point at a time.
        let gather = &work.gather;
        let spacing = pyramid.spacing(gather.level);
        let (hydrology, lines) = bake(&gather.heights, gather.width, spacing, &work.layer);
        let ground = lines
            .iter()
            .map(|line| Vec::with_capacity(line.points.len()))
            .collect();
        work.rivers
            .insert(RiverJob {
                hydrology: Some(hydrology),
                lines,
                ground,
                layer: work.layer,
                grid: Grid {
                    heights: gather.heights.clone(),
                    width: gather.width,
                    spacing,
                },
            })
            .advance(below, pyramid, budget)
    }
}

/// A hydrology bake's rivers being read: what the bake found, the rivers'
/// lines, the finest ground read at their points' middles so far, the layer
/// saying how they flow, and the bake's grid, which says how much sky their
/// falls see.
struct RiverJob {
    hydrology: Option<Hydrology>,
    lines: Vec<RiverLine>,
    ground: Vec<Vec<f32>>,
    layer: HydrologyLayer,
    grid: Grid,
}

/// The ways around (unit vectors along x and z) the sky a fall sees is
/// read along, and the meters out its ground is read at: the nearest, then
/// twice as far each time, out to the farthest.
const SKY_WAYS: [[f32; 2]; 8] = [
    [1.0, 0.0],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ],
    [0.0, 1.0],
    [
        -std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ],
    [-1.0, 0.0],
    [
        -std::f32::consts::FRAC_1_SQRT_2,
        -std::f32::consts::FRAC_1_SQRT_2,
    ],
    [0.0, -1.0],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        -std::f32::consts::FRAC_1_SQRT_2,
    ],
];
const SKY_NEAREST: f32 = 16.0;
const SKY_FARTHEST: f32 = 256.0;

/// A bake's grid of heights: its samples, `width` a row, `spacing` meters
/// apart.
struct Grid {
    heights: Vec<f32>,
    width: usize,
    spacing: f32,
}

impl Grid {
    /// The ground's height at terrain-local `(x, z)`, between its four
    /// nearest samples, the grid's edge holding beyond it.
    fn height(&self, x: f32, z: f32) -> f32 {
        let rows = self.heights.len() / self.width;
        let x = min(max(x / self.spacing, 0.0), (self.width - 1) as f32);
        let z = min(max(z / self.spacing, 0.0), (rows - 1) as f32);
        let (column, row) = (x.floor() as usize, z.floor() as usize);
        let (right, below) = ((column + 1).min(self.width - 1), (row + 1).min(rows - 1));
        let (fx, fz) = (x - column as f32, z - row as f32);
        let at = |row: usize, column: usize| self.heights[row * self.width + column];
        let top = at(row, column) + (at(row, right) - at(row, column)) * fx;
        let bottom = at(below, column) + (at(below, right) - at(below, column)) * fx;
        top + (bottom - top) * fz
    }

    /// How much of the sky water at terrain-local `point` sees (0 to 1):
    /// each way around open but for the steepest rise of the ground out to
    /// SKY_FARTHEST meters (1 less the sine of its angle), averaged.
    fn sky(&self, point: [f32; 3]) -> f32 {
        let mut open = 0.0;
        for [x, z] in SKY_WAYS {
            let mut steepest = 0.0_f32;
            let mut reach = SKY_NEAREST;
            while reach <= SKY_FARTHEST {
                let rise = self.height(point[0] + x * reach, point[2] + z * reach) - point[1];
                steepest = max(steepest, rise / reach);
                reach *= 2.0;
            }
            open += 1.0 - steepest / (1.0 + steepest * steepest).sqrt();
        }
        open / SKY_WAYS.len() as f32
    }
}

impl RiverJob {
    /// Reads the finest ground of `below` along the rivers' middles within
    /// `budget`; then sets every river's channel and water as the layer's
    /// flow law gives them, upstream rivers first: a river leaving a lake
    /// starts at the lake's level, and one leaving a confluence at the
    /// lowest of the rivers ending there.
    fn advance(
        &mut self,
        below: &Layers,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Hydrology> {
        let spacing = pyramid.spacing(0);
        let hydrology = self.hydrology.as_ref()?;
        let cost = below.height_cost();
        for (line, ground) in self.lines.iter().zip(&mut self.ground) {
            while ground.len() < line.points.len() {
                if *budget == 0 {
                    return None;
                }
                let point = line.points[ground.len()];
                let sample = [
                    (point.x / spacing).round() as i64,
                    (point.y / spacing).round() as i64,
                ];
                ground.push(below.sample_height(pyramid, 0, sample));
                *budget = budget.saturating_sub(cost);
            }
        }
        let layer = &self.layer;
        let flow = Flow {
            // Cubic meters a second a square kilometer, a square meter at
            // a time.
            runoff: layer.discharge * 1.0e-6,
            width: layer.width,
            roughness: layer.roughness,
        };
        let water = &hydrology.water;
        // The lowest level of the rivers ending at each place, by its bits.
        // Lines run upstream first: the rivers ending where one starts
        // come before it.
        let key = |point: Vec2| [point.x.to_bits(), point.y.to_bits()];
        let mut ends: HashMap<[u32; 2], f32> = HashMap::default();
        let mut rivers = Vec::with_capacity(self.lines.len());
        let mut falls = Vec::new();
        for (line, ground) in self.lines.iter().zip(&self.ground) {
            let first = line.points[0];
            let start = if line.head {
                water.lake_beside(first.x, first.y)
            } else {
                ends.get(&key(first)).copied()
            };
            let (solved, found) = river(&Reach { line, ground, start }, &flow);
            let end = line.points[line.points.len() - 1];
            let level = solved.surface[solved.surface.len() - 1];
            if level.is_finite() {
                ends.entry(key(end))
                    .and_modify(|lowest| *lowest = min(*lowest, level))
                    .or_insert(level);
            }
            rivers.push(solved);
            falls.extend(found);
        }
        for fall in &mut falls {
            fall.sky = fall
                .path
                .iter()
                .map(|&point| self.grid.sky(point))
                .collect();
        }
        let mut hydrology = self.hydrology.take()?;
        hydrology.rivers = Arc::new(Rivers::new(rivers));
        hydrology.stats.falls = falls.len();
        hydrology.falls = falls.into();
        Some(hydrology)
    }
}
