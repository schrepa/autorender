//! The hydrology bake: the level's grid gathered, then where water runs
//! and pools over it at once, then each river's ground read from the finest
//! level, a point at a time, and its flow in its channel solved downstream
//! first.

use engine_math::collections::HashMap;
use std::sync::Arc;

use engine_math::{max, min};
use glam::Vec2;

use super::{Bake, Gather};
use crate::Pyramid;
use crate::hydraulics::{Flow, Reach, filling_yield, foam_entering, river};
use crate::hydrology::{Hydrology, HydrologyLayer, bake};
use crate::layers::Layers;
use crate::rivers::{BANK_OFFSET, River, RiverLine, Rivers, across};

/// Meters below the water it meets a river ends: held back to a lake's or the
/// river it joins' level, its own surface slides just under theirs instead of
/// standing at the same depth; rivers joining another at one place end a step
/// further below each, the one carrying most highest.
const MEETING_DROP: f32 = 0.03;

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
        // Its rivers' water follows from the finest ground along
        // them, read a point at a time.
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

/// Foam rides on into the river the water joins: each of `rivers` (along
/// `lines`) starts with the foam the rivers ending where it starts bring
/// (`meeting` lists them by that place's bits), by the discharge each
/// carries, upstream rivers first.
fn carry_foam(
    rivers: &mut [Option<River>],
    lines: &[RiverLine],
    meeting: &HashMap<[u32; 2], Vec<usize>>,
) {
    for index in 0..rivers.len() {
        let start = lines[index].points[0];
        let joining = meeting
            .get(&[start.x.to_bits(), start.y.to_bits()])
            .map_or(&[][..], Vec::as_slice);
        let (mut foam, mut discharge) = (0.0, 0.0);
        for river in joining.iter().filter_map(|&other| rivers[other].as_ref()) {
            let last = river.foam.len() - 1;
            foam += river.foam[last] * river.discharge[last];
            discharge += river.discharge[last];
        }
        if discharge > 0.0
            && let Some(river) = &mut rivers[index]
        {
            foam_entering(river, foam / discharge);
        }
    }
}

/// A hydrology bake's rivers being read: what the bake
/// found, the rivers' lines, the finest ground read at their points so far
/// (at each line's middle and its banks), the layer saying how they
/// flow, and the bake's grid, which says how much sky their falls see.
struct RiverJob {
    hydrology: Option<Hydrology>,
    lines: Vec<RiverLine>,
    ground: Vec<Vec<[f32; 3]>>,
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
    /// Reads the finest ground of `below` along the rivers within `budget`;
    /// then solves every river's flow in its channel, downstream rivers
    /// first, each held back by the lake it enters or the river it joins
    /// and starting at the level of the lake it leaves, with the yield the
    /// layer gives or, by default, the one that runs them `fill` of the way
    /// up their channels.
    fn advance(
        &mut self,
        below: &Layers,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Hydrology> {
        let spacing = pyramid.spacing(0);
        let hydrology = self.hydrology.as_ref()?;
        let read = |at: Vec2| {
            let sample = [
                (at.x / spacing).round() as i64,
                (at.y / spacing).round() as i64,
            ];
            below.sample_height(pyramid, 0, sample)
        };
        let cost = below.height_cost() * 3;
        for (line, ground) in self.lines.iter().zip(&mut self.ground) {
            while ground.len() < line.points.len() {
                if *budget == 0 {
                    return None;
                }
                let index = ground.len();
                let point = line.points[index];
                let side = across(&line.points, index) * (line.half[index] + BANK_OFFSET);
                ground.push([read(point), read(point + side), read(point - side)]);
                *budget = budget.saturating_sub(cost);
            }
        }
        // Each river's start, which rivers ending there join. Lines run
        // upstream first: downstream rivers come later.
        let key = |point: Vec2| [point.x.to_bits(), point.y.to_bits()];
        let starts: HashMap<[u32; 2], usize> = self
            .lines
            .iter()
            .enumerate()
            .map(|(index, line)| (key(line.points[0]), index))
            .collect();
        // Rivers ending at one place, the one draining most first.
        let mut meeting: HashMap<[u32; 2], Vec<usize>> = HashMap::default();
        for (index, line) in self.lines.iter().enumerate() {
            meeting
                .entry(key(line.points[line.points.len() - 1]))
                .or_default()
                .push(index);
        }
        let mut step = vec![1.0_f32; self.lines.len()];
        for joining in meeting.values_mut() {
            joining.sort_by(|&a, &b| {
                let area = |index: usize| self.lines[index].area[self.lines[index].area.len() - 1];
                area(b).total_cmp(&area(a)).then(a.cmp(&b))
            });
            for (rank, &index) in joining.iter().enumerate() {
                step[index] = (rank + 1) as f32;
            }
        }
        let layer = &self.layer;
        let yield_rate = if layer.discharge > 0.0 {
            // Cubic meters a second a square kilometer, a square meter at
            // a time.
            Some(layer.discharge * 1.0e-6)
        } else {
            let reaches: Vec<Reach<'_>> = self
                .lines
                .iter()
                .zip(&self.ground)
                .map(|(line, ground)| Reach {
                    line,
                    ground,
                    source: None,
                    outlet: None,
                })
                .collect();
            filling_yield(&reaches, layer.fill, layer.roughness)
        };
        let flow = Flow {
            yield_rate: yield_rate.unwrap_or(0.0),
            roughness: layer.roughness,
        };
        let water = &hydrology.water;
        let mut rivers: Vec<Option<crate::rivers::River>> = vec![None; self.lines.len()];
        let mut falls = Vec::new();
        for index in (0..self.lines.len()).rev() {
            let line = &self.lines[index];
            let end = line.points[line.points.len() - 1];
            let outlet = match starts.get(&key(end)) {
                Some(&joined) if joined != index => rivers[joined]
                    .as_ref()
                    .map(|river| river.surface[0])
                    .filter(|level| level.is_finite()),
                _ => water.lake_level(end.x, end.y),
            }
            .map(|level| level - MEETING_DROP * step[index]);
            let source = if line.head {
                water.lake_beside(line.points[0].x, line.points[0].y)
            } else {
                None
            };
            let reach = Reach {
                line,
                ground: &self.ground[index],
                source,
                outlet,
            };
            let (solved, jets) = river(&reach, &flow);
            rivers[index] = Some(solved);
            falls.extend(jets);
        }
        carry_foam(&mut rivers, &self.lines, &meeting);
        for fall in &mut falls {
            fall.sky = fall
                .path
                .iter()
                .map(|&point| self.grid.sky(point))
                .collect();
        }
        let mut hydrology = self.hydrology.take()?;
        hydrology.rivers = Arc::new(Rivers::new(rivers.into_iter().flatten().collect()));
        hydrology.stats.falls = falls.len();
        hydrology.falls = falls.into();
        Some(hydrology)
    }
}
