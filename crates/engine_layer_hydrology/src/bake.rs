use std::sync::Arc;

use engine_heightfield::bed::NARROWEST;
use engine_heightfield::given::GivenRiver;
use engine_heightfield::hydraulics::{Across, BESIDE, Course, Flow, river};
use engine_heightfield::hydrology::{Grid, MAP_LEVELS};
use engine_heightfield::layer::{FineRegion, TerrainLayer};
use engine_heightfield::rivers::Rivers;
use engine_heightfield::{Gather, Layers, Pyramid, TileHeights};
use engine_math::collections::HashMap;
use engine_math::{clamp, max, min};
use glam::Vec2;

use crate::hydrology::{HydrologyLayer, bake};
use crate::{HydrologyStage, RiversLayer};

pub(crate) fn hydrology_stage(
    layer: &HydrologyLayer,
    below: &Arc<Layers>,
    pyramid: &Pyramid,
    tiles: TileHeights,
) -> Arc<dyn TerrainLayer> {
    let landscape = Gather::new(layer.level + MAP_LEVELS, below, pyramid, tiles);
    let gather = Gather::new(layer.level, below, pyramid, tiles);
    let grid = Grid {
        heights: &gather.heights,
        width: gather.width,
        spacing: pyramid.spacing(gather.level),
    };
    let given: Vec<(usize, &[GivenRiver], f32)> = below
        .layers()
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            let lines = entry.given_rivers()?;
            let precision = entry
                .as_any()
                .downcast_ref::<RiversLayer>()
                .map_or(0.0, |rivers| rivers.precision);
            Some((index, lines, precision))
        })
        .collect();
    let (mut hydrology, lines) = bake(
        (grid.heights, grid.width, grid.spacing),
        (
            &landscape.heights,
            landscape.width,
            pyramid.spacing(landscape.level),
        ),
        (layer, &given, pyramid.spacing(0)),
    );
    let flow = Flow {
        runoff: layer.discharge * 1.0e-6,
        finest: pyramid.spacing(0),
    };
    let water = &hydrology.water;
    let key = |point: Vec2| [point.x.to_bits(), point.y.to_bits()];
    let mut ends: HashMap<[u32; 2], f32> = HashMap::default();
    let mut rivers = Vec::with_capacity(lines.len());
    let mut falls = Vec::new();
    let ground = |point: Vec2| grid.height(point.x, point.y);
    let imported = Imported::new(below, pyramid);
    let across = |point: Vec2, way: Vec2, half: f32| imported.across(point, way, half);
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
                across: &across,
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
        fall.sky = fall.path.iter().map(|&point| sky(&grid, point)).collect();
    }
    let stats = &mut hydrology.stats;
    for river in &rivers {
        for &reach in &river.reach {
            stats.reaches[reach as usize] += 1;
        }
        stats.steps += river.crest.iter().flatten().count();
        stats.knobs += river.knobs.len();
        stats.uncut += river.cut.iter().filter(|&&cut| !cut).count();
    }
    stats.falls = falls.len();
    hydrology.rivers = Arc::new(Rivers::new(rivers));
    hydrology.falls = falls.into();
    Arc::new(HydrologyStage {
        layer: *layer,
        baked: Some(Arc::new(hydrology)),
    })
}

const ACROSS: usize = 4;

struct Imported<'a> {
    below: &'a Layers,
    pyramid: &'a Pyramid,
    maps: Vec<FineRegion>,
}

impl<'a> Imported<'a> {
    fn new(below: &'a Layers, pyramid: &'a Pyramid) -> Self {
        let maps = below
            .layers()
            .iter()
            .filter_map(|entry| entry.fine_region())
            .collect();
        Self {
            below,
            pyramid,
            maps,
        }
    }

    fn across(&self, point: Vec2, way: Vec2, half: f32) -> Option<Across> {
        let texel = self
            .maps
            .iter()
            .filter(|region| point.cmpge(region.low).all() && point.cmple(region.high).all())
            .fold(f32::INFINITY, |finest, region| min(finest, region.texel));
        if !texel.is_finite() {
            return None;
        }
        let share = clamp((2.0 * half / texel - 1.0) / (NARROWEST - 1.0), 0.0, 1.0);
        if share <= 0.0 {
            return None;
        }
        let spacing = self.pyramid.spacing(0);
        let out = way.perp();
        let ground = |offset: f32| {
            let at = point + out * offset;
            self.below.sample_height(
                self.pyramid,
                0,
                [
                    (at.x / spacing).round() as i64,
                    (at.y / spacing).round() as i64,
                ],
            )
        };
        let banks = min(ground(half), ground(-half));
        let mut lowest = min(banks, ground(0.0));
        for quarter in 1..ACROSS {
            let offset = half * quarter as f32 / ACROSS as f32;
            lowest = min(lowest, min(ground(offset), ground(-offset)));
        }
        Some(Across {
            share,
            lowest,
            banks,
            beside: min(ground(half + BESIDE), ground(-(half + BESIDE))),
        })
    }
}

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

fn sky(grid: &Grid<'_>, point: [f32; 3]) -> f32 {
    let mut open = 0.0;
    for [x, z] in SKY_WAYS {
        let mut steepest = 0.0_f32;
        let mut reach = SKY_NEAREST;
        while reach <= SKY_FARTHEST {
            let rise = grid.height(point[0] + x * reach, point[2] + z * reach) - point[1];
            steepest = max(steepest, rise / reach);
            reach *= 2.0;
        }
        open += 1.0 - steepest / (1.0 + steepest * steepest).sqrt();
    }
    open / SKY_WAYS.len() as f32
}
