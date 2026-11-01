#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod relief;

use engine_core::{Brick, Bricks, Source};
use engine_places::terrain::TerrainLayers;

pub use relief::{HUMMOCKS, HummocksLayer, STEPS, StepsLayer};

pub struct LayerReliefBrick;

impl Brick for LayerReliefBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<TerrainLayers>(&STEPS, None)
            .join::<TerrainLayers>(&HUMMOCKS, None);
    }
}
