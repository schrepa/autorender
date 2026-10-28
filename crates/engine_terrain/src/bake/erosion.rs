use std::sync::Arc;

use super::{Bake, Gather};
use crate::Pyramid;
use crate::erosion::{Erosion, ErosionLayer, Scratch, erode_round, finish};
use crate::layers::Layers;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ErosionBake;

pub struct ErosionWork {
    layer: ErosionLayer,
    gather: Gather,
    original: Option<Vec<f32>>,
    rounds: u32,
    scratch: Scratch,
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
            scratch: Scratch::default(),
        }
    }

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
            scratch,
        } = work;
        let spacing = pyramid.spacing(gather.level);
        let original = original.get_or_insert_with(|| gather.heights.clone());
        let cost = gather.heights.len() as u64 * 8;
        while *rounds < layer.rounds {
            if *budget == 0 {
                return None;
            }
            erode_round(
                &mut gather.heights,
                (gather.width, spacing),
                layer,
                *rounds,
                scratch,
            );
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
