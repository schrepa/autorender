//! The hydrology bake: the level's grid gathered, then where water runs
//! and pools over it at once, then each river's water from the finest
//! ground along it, a point at a time.

use engine_math::collections::HashMap;
use std::sync::Arc;

use glam::Vec2;

use super::{Bake, Gather};
use crate::Pyramid;
use crate::hydrology::{FALL_SLOPE, Hydrology, HydrologyLayer, bake};
use crate::layers::Layers;
use crate::rivers::{BANK_OFFSET, RiverLine, Rivers, across, profile};

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
            })
            .advance(below, pyramid, budget)
    }
}

/// A hydrology bake's rivers being read: what the bake
/// found, the rivers' lines, and the finest ground read at their points so
/// far (at each line's middle and its banks).
struct RiverJob {
    hydrology: Option<Hydrology>,
    lines: Vec<RiverLine>,
    ground: Vec<Vec<[f32; 3]>>,
}

impl RiverJob {
    /// Reads the finest ground of `below` along the rivers within `budget`,
    /// the hydrology's channels carved into it; then sets every river's
    /// water, upstream rivers first, a river leaving a confluence starting
    /// no higher than those meeting there.
    fn advance(
        &mut self,
        below: &Layers,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Hydrology> {
        let spacing = pyramid.spacing(0);
        let hydrology = self.hydrology.as_ref()?;
        let level = hydrology.level_for(spacing);
        let cost = below.height_cost() * 3;
        for (line, ground) in self.lines.iter().zip(&mut self.ground) {
            while ground.len() < line.points.len() {
                if *budget == 0 {
                    return None;
                }
                let index = ground.len();
                let point = line.points[index];
                let side = across(&line.points, index) * (line.half[index] + BANK_OFFSET);
                let read = |at: Vec2| {
                    let sample = [
                        (at.x / spacing).round() as i64,
                        (at.y / spacing).round() as i64,
                    ];
                    below.sample_height(pyramid, 0, sample) - hydrology.carve_at(at.x, at.y, level)
                };
                ground.push([read(point), read(point + side), read(point - side)]);
                *budget = budget.saturating_sub(cost);
            }
        }
        let key = |point: Vec2| [point.x.to_bits(), point.y.to_bits()];
        let mut ends: HashMap<[u32; 2], f32> = HashMap::default();
        let mut rivers = Vec::with_capacity(self.lines.len());
        for (line, ground) in self.lines.iter().zip(&self.ground) {
            let incoming = ends.get(&key(line.points[0])).copied();
            let river = profile(line, ground, incoming, FALL_SLOPE);
            let end = key(line.points[line.points.len() - 1]);
            let last = river.surface[river.surface.len() - 1];
            ends.entry(end)
                .and_modify(|level| *level = engine_math::min(*level, last))
                .or_insert(last);
            rivers.push(river);
        }
        let mut hydrology = self.hydrology.take()?;
        hydrology.rivers = Arc::new(Rivers::new(rivers));
        Some(hydrology)
    }
}
