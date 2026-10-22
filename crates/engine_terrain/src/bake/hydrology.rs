//! The hydrology bake: the level's grid gathered and a grid of the
//! landscape [`MAP_LEVELS`] levels coarser, then at once where water runs
//! and pools over each, the rivers traced along the level's own grid, and
//! each river's channel, bed and water set from the ground along its line
//! and beside it, upstream rivers first.

use engine_math::collections::HashMap;
use std::sync::Arc;

use engine_math::{max, min};
use glam::Vec2;

use super::{Bake, Gather};
use crate::Pyramid;
use crate::hydraulics::{Course, Flow, river};
use crate::hydrology::{Hydrology, HydrologyLayer, MAP_LEVELS, bake};
use crate::layers::Layers;
use crate::rivers::Rivers;

/// Where water runs over the layers before it and the channels it carves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HydrologyBake;

/// A hydrology bake in progress: its layer, its level's grid and the
/// landscape's.
pub struct HydrologyWork {
    layer: HydrologyLayer,
    gather: Gather,
    landscape: Gather,
}

impl Bake for HydrologyBake {
    type Layer = HydrologyLayer;
    type Found = Hydrology;
    type Work = HydrologyWork;

    fn start(layer: &HydrologyLayer, pyramid: &Pyramid) -> HydrologyWork {
        HydrologyWork {
            layer: *layer,
            gather: Gather::new(layer.level, pyramid),
            landscape: Gather::new(layer.level + MAP_LEVELS, pyramid),
        }
    }

    fn advance(
        work: &mut HydrologyWork,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Hydrology> {
        if !work.landscape.advance(below, pyramid, budget)
            || !work.gather.advance(below, pyramid, budget)
        {
            return None;
        }
        let HydrologyWork {
            layer,
            gather,
            landscape,
        } = work;
        let grid = Grid {
            heights: &gather.heights,
            width: gather.width,
            spacing: pyramid.spacing(gather.level),
        };
        let (mut hydrology, lines) = bake(
            (grid.heights, grid.width, grid.spacing),
            (
                &landscape.heights,
                landscape.width,
                pyramid.spacing(landscape.level),
            ),
            layer,
        );
        let flow = Flow {
            // Cubic meters a second a square kilometer, a square meter at
            // a time.
            runoff: layer.discharge * 1.0e-6,
            finest: pyramid.spacing(0),
        };
        let water = &hydrology.water;
        // The lowest banks of the rivers ending at each place, by its bits.
        // Lines run upstream first: the rivers ending where one starts
        // come before it.
        let key = |point: Vec2| [point.x.to_bits(), point.y.to_bits()];
        let mut ends: HashMap<[u32; 2], f32> = HashMap::default();
        let mut rivers = Vec::with_capacity(lines.len());
        let mut falls = Vec::new();
        // Its ground: the grid's, along its line the ground it is cut into,
        // through the rims of hollows shallower than a lake as rivers cut
        // through them, not over them, and beside it the ground its banks
        // must hold its flood below.
        let ground = |point: Vec2| grid.height(point.x, point.y);
        for line in &lines {
            let first = line.points[0];
            let start = if line.head {
                water.lake_beside(first.x, first.y)
            } else {
                ends.get(&key(first)).copied()
            };
            let (solved, found) = river(
                &Course {
                    line,
                    ground: &ground,
                    start,
                },
                &flow,
            );
            let end = line.points.len() - 1;
            let level = solved.flood[end];
            let end = line.points[end];
            if level.is_finite() {
                ends.entry(key(end))
                    .and_modify(|lowest| *lowest = min(*lowest, level))
                    .or_insert(level);
            }
            rivers.push(solved);
            falls.extend(found);
        }
        for fall in &mut falls {
            fall.sky = fall.path.iter().map(|&point| grid.sky(point)).collect();
        }
        let stats = &mut hydrology.stats;
        for river in &rivers {
            for &reach in &river.reach {
                stats.reaches[reach as usize] += 1;
            }
            stats.steps += river.crest.iter().flatten().count();
            stats.knobs += river.knobs.len();
        }
        stats.falls = falls.len();
        hydrology.rivers = Arc::new(Rivers::new(rivers));
        hydrology.falls = falls.into();
        Some(hydrology)
    }
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
#[derive(Clone, Copy)]
struct Grid<'a> {
    heights: &'a [f32],
    width: usize,
    spacing: f32,
}

impl Grid<'_> {
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
