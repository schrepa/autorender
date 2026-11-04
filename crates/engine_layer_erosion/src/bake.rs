use std::sync::Arc;

use engine_heightfield::layer::TerrainLayer;
use engine_heightfield::{Gather, Layers, Pyramid, TileHeights};

use crate::ErosionStage;
use crate::erosion::{ErosionLayer, Scratch, erode_round, finish};

pub(crate) fn erosion_stage(
    layer: &ErosionLayer,
    below: &Arc<Layers>,
    pyramid: &Pyramid,
    tiles: TileHeights,
) -> Arc<dyn TerrainLayer> {
    let mut gather = Gather::new(layer.level, below, pyramid, tiles);
    let spacing = pyramid.spacing(gather.level);
    let original = gather.heights.clone();
    let mut scratch = Scratch::default();
    for round in 0..layer.rounds {
        erode_round(
            &mut gather.heights,
            (gather.width, spacing),
            layer,
            round,
            &mut scratch,
        );
    }
    Arc::new(ErosionStage {
        layer: *layer,
        baked: Some(Arc::new(finish(
            &original,
            &gather.heights,
            gather.width,
            spacing,
            layer.rounds,
        ))),
    })
}
