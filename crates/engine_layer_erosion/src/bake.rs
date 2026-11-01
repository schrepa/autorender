use std::sync::Arc;

use engine_heightfield::layer::{LayerBake, TerrainLayer};
use engine_heightfield::{Gather, Layers, Pyramid, StartTile};

use crate::ErosionStage;
use crate::erosion::{ErosionLayer, Scratch, erode_round, finish};

pub struct ErosionWork {
    layer: ErosionLayer,
    gather: Gather,
    original: Option<Vec<f32>>,
    rounds: u32,
    scratch: Scratch,
}

impl ErosionWork {
    pub(crate) fn new(layer: &ErosionLayer, pyramid: &Pyramid, tiles: StartTile) -> Self {
        Self {
            layer: *layer,
            gather: Gather::new(layer.level, pyramid, tiles),
            original: None,
            rounds: 0,
            scratch: Scratch::default(),
        }
    }
}

impl LayerBake for ErosionWork {
    fn advance(
        &mut self,
        below: &Arc<Layers>,
        pyramid: &Pyramid,
        budget: &mut u64,
    ) -> Option<Arc<dyn TerrainLayer>> {
        if !self.gather.advance(below, pyramid, budget) {
            return None;
        }
        let Self {
            layer,
            gather,
            original,
            rounds,
            scratch,
        } = self;
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
        Some(Arc::new(ErosionStage {
            layer: *layer,
            baked: Some(Arc::new(finish(
                original,
                &gather.heights,
                gather.width,
                spacing,
                *rounds,
            ))),
        }))
    }
}
