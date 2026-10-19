//! The hydrology bake: the level's grid gathered, then where water runs
//! and pools over it at once, then each river's ground read from the finest
//! level, a point at a time, and its flow in its channel solved downstream
//! first.

use engine_math::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;

use super::{Bake, Gather};
use crate::Pyramid;
use crate::hydraulics::{Flow, Reach, filling_yield, foam_entering, river};
use crate::hydrology::{Hydrology, HydrologyLayer, bake};
use crate::layers::Layers;
use crate::rivers::{BANK_OFFSET, RiverLine, Rivers, across};

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
            })
            .advance(below, pyramid, budget)
    }
}

/// A hydrology bake's rivers being read: what the bake
/// found, the rivers' lines, the finest ground read at their points so far
/// (at each line's middle and its banks), and the layer saying how they
/// flow.
struct RiverJob {
    hydrology: Option<Hydrology>,
    lines: Vec<RiverLine>,
    ground: Vec<Vec<[f32; 3]>>,
    layer: HydrologyLayer,
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
        // Foam rides on into the river the water joins: each river starts
        // with the foam the rivers ending there bring, by the discharge each
        // carries, upstream rivers first.
        for index in 0..rivers.len() {
            let joining = meeting
                .get(&key(self.lines[index].points[0]))
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
        let mut hydrology = self.hydrology.take()?;
        hydrology.rivers = Arc::new(Rivers::new(rivers.into_iter().flatten().collect()));
        hydrology.stats.falls = falls.len();
        hydrology.falls = falls.into();
        Some(hydrology)
    }
}
