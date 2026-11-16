use std::sync::Arc;

use engine_heightfield::layer::TerrainLayer;
use engine_heightfield::{Gather, Layers, Pyramid, TileHeights};

use crate::ErosionStage;
use crate::erosion::{ErosionLayer, Ground};

pub(crate) fn erosion_stage(
    layer: &ErosionLayer,
    below: &Arc<Layers>,
    pyramid: &Pyramid,
    tiles: TileHeights,
) -> Arc<dyn TerrainLayer> {
    let gather = Gather::new(layer.level, below, pyramid, tiles);
    let spacing = pyramid.spacing(gather.level);
    let original = gather.heights.clone();
    let mut ground = Ground::new(gather.heights, gather.width, spacing, &layer.beds);
    for round in 0..layer.rounds {
        ground.round(layer, round);
    }
    ground.settle(layer.rounds);
    Arc::new(ErosionStage {
        layer: *layer,
        baked: Some(Arc::new(ground.finish(&original, layer))),
    })
}
