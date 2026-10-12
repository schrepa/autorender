//! The erosion bake: the level's grid gathered, then what rivers and creep
//! wear away from it, a round at a time.

use std::sync::Arc;

use super::{Bake, Gather};
use crate::Pyramid;
use crate::erosion::{Erosion, ErosionLayer, erode_round, finish};
use crate::layers::Layers;

/// What rivers and creep wear away from the layers before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ErosionBake;

/// An erosion bake in progress: its layer, its level's grid, and once
/// gathered, the heights it started from and the rounds done.
pub struct ErosionWork {
    layer: ErosionLayer,
    gather: Gather,
    original: Option<Vec<f32>>,
    rounds: u32,
}

impl Bake for ErosionBake {
    type Layer = ErosionLayer;
    type Found = Erosion;
    type Work = ErosionWork;

    fn start(layer: &ErosionLayer, pyramid: &Pyramid) -> ErosionWork {
        ErosionWork {
            layer: *layer,
            gather: Gather::new(layer.level, pyramid),
            original: None,
            rounds: 0,
        }
    }

    /// Gathers, then erodes a round per grid's worth of `budget`.
    fn advance(
        work: &mut ErosionWork,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Erosion> {
        if !work.gather.advance(below, pyramid, budget) {
            return None;
        }
        let ErosionWork {
            layer,
            gather,
            original,
            rounds,
        } = work;
        let spacing = pyramid.spacing(gather.level);
        let original = original.get_or_insert_with(|| gather.heights.clone());
        let cost = gather.heights.len() as u64 * 8;
        while *rounds < layer.rounds {
            if *budget == 0 {
                return None;
            }
            erode_round(&mut gather.heights, gather.width, spacing, layer);
            *rounds += 1;
            *budget = budget.saturating_sub(cost);
        }
        Some(finish(
            original,
            &gather.heights,
            gather.width,
            spacing,
            *rounds,
        ))
    }
}
