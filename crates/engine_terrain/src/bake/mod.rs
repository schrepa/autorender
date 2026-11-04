#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use std::sync::Arc;

use crate::{Generator, Layers, Pyramid, TileKey};

fn tile_heights(below: Arc<Layers>, pyramid: &Pyramid, key: TileKey) -> Box<[f32]> {
    let mut whole = u64::MAX;
    let tile = Generator::new(below, pyramid, key)
        .advance(&mut whole)
        .expect("baked_below_is_finite");
    Box::from(tile.heights())
}

#[must_use]
pub fn bake_all(below: &Layers, pyramid: &Pyramid) -> Layers {
    let mut layers = below.clone();
    while let Some(index) = layers.unbaked() {
        let under = Arc::new(layers.below(index));
        let baked = layers.layers()[index]
            .bake(&under, pyramid, tile_heights)
            .expect("unbaked_layer_bakes");
        layers = layers.with_baked(index, baked);
    }
    layers
}
