//! The hydrology bake: the level's grid gathered and a grid of the
//! landscape [`MAP_LEVELS`] levels coarser, then at once where water runs
//! and pools over each, the rivers the rivers layers below it give and
//! those traced along the level's own grid where the water left gathers,
//! and each river's channel, bed and water set from the ground along its
//! line and beside it, given rivers and upstream rivers first: the finest
//! heights across it where an imported heightmap resolves its channel.

use engine_math::collections::HashMap;
use std::sync::Arc;

use engine_math::{clamp, max, min};
use glam::Vec2;

use super::{Bake, Gather};
use crate::Pyramid;
use crate::bed::NARROWEST;
use crate::given::RiversLayer;
use crate::hydraulics::{Across, BESIDE, Course, Flow, river};
use crate::hydrology::{Grid, Hydrology, HydrologyLayer, MAP_LEVELS, bake};
use crate::layers::{Layer, Layers};
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
        // The rivers the rivers layers below it give, in their order.
        let given: Vec<(usize, &RiversLayer)> = below
            .layers()
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| match layer {
                Layer::Rivers(rivers) => Some((index, rivers.as_ref())),
                Layer::Noise(_)
                | Layer::Heightmap(_)
                | Layer::Stamp(_)
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_) => None,
            })
            .collect();
        let (mut hydrology, lines) = bake(
            (grid.heights, grid.width, grid.spacing),
            (
                &landscape.heights,
                landscape.width,
                pyramid.spacing(landscape.level),
            ),
            (&*layer, &given, pyramid.spacing(0)),
        );
        let flow = Flow {
            // Cubic meters a second a square kilometer, a square meter at
            // a time.
            runoff: layer.discharge * 1.0e-6,
            finest: pyramid.spacing(0),
        };
        let water = &hydrology.water;
        // The lowest banks of the rivers ending at each place, by its bits.
        // Lines run given rivers first, then upstream first: the rivers
        // ending where one starts come before it.
        let key = |point: Vec2| [point.x.to_bits(), point.y.to_bits()];
        let mut ends: HashMap<[u32; 2], f32> = HashMap::default();
        let mut rivers = Vec::with_capacity(lines.len());
        let mut falls = Vec::new();
        // Its ground: the grid's, along its line the ground it is cut into,
        // through the rims of hollows shallower than a lake as rivers cut
        // through them, not over them, and beside it the ground its banks
        // must hold its flood below; and the finest heights across its
        // channel, where an imported heightmap resolves it.
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
            fall.sky = fall.path.iter().map(|&point| grid.sky(point)).collect();
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
        Some(hydrology)
    }
}

/// Places either side of a channel's middle its finest heights are read at
/// across it: its quarters out to its edges.
const ACROSS: usize = 4;

/// Where imported heightmaps give the ground beneath a hydrology bake: the
/// layers below it, the pyramid they lie over, and each heightmap's square
/// at full strength (inside its falloff) with the meters between its
/// samples.
struct Imported<'a> {
    below: &'a Layers,
    pyramid: &'a Pyramid,
    maps: Vec<(Vec2, Vec2, f32)>,
}

impl<'a> Imported<'a> {
    fn new(below: &'a Layers, pyramid: &'a Pyramid) -> Self {
        let maps = below
            .layers()
            .iter()
            .filter_map(|layer| match layer {
                Layer::Heightmap(map) => {
                    let inside = Vec2::splat(map.falloff);
                    Some((map.min + inside, map.min + map.size - inside, map.texel()))
                }
                Layer::Noise(_)
                | Layer::Stamp(_)
                | Layer::Rivers(_)
                | Layer::Hydrology(_)
                | Layer::Erosion(_)
                | Layer::Path(_) => None,
            })
            .collect();
        Self {
            below,
            pyramid,
            maps,
        }
    }

    /// What the finest heights give across a river's channel at
    /// terrain-local `point`, running `way` (a unit vector), `half` meters
    /// either side of its middle, where a heightmap gives the ground there
    /// at full strength: as far as the finest of them resolves the channel
    /// (wholly where it spans [`NARROWEST`] of its samples), the lowest of
    /// the level-0 samples nearest its middle, its quarters ([`ACROSS`]) and
    /// its edges, the lower at its two edges, and the lower [`BESIDE`]
    /// meters past them; `None` where none resolves it.
    fn across(&self, point: Vec2, way: Vec2, half: f32) -> Option<Across> {
        let texel = self
            .maps
            .iter()
            .filter(|(low, high, _)| point.cmpge(*low).all() && point.cmple(*high).all())
            .fold(f32::INFINITY, |finest, &(_, _, texel)| min(finest, texel));
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

impl Grid<'_> {
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
