#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod relief;
mod site;
mod talus;

use engine_core::{Brick, Bricks, Source};
use engine_heightfield::{INVALID_TERRAIN, MISSING_LAYER, UNKNOWN_MAP};
use engine_open_world_kit::terrain::TerrainLayers;

pub use relief::{HUMMOCKS, HummocksLayer, JointSet, STRATA, StrataLayer};
pub use site::{EXPOSURE, SHELTER, SITE, SNOW, SOIL, SUN, Site, SiteLayer, SiteStage};
pub use talus::{DEBRIS, FALL, Runnels, TALUS, Talus, TalusLayer, TalusStage};

pub struct LayerReliefBrick;

impl Brick for LayerReliefBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .errors(&[INVALID_TERRAIN, MISSING_LAYER, UNKNOWN_MAP])
            .join::<TerrainLayers>(&STRATA, None)
            .join::<TerrainLayers>(&HUMMOCKS, None)
            .join::<TerrainLayers>(&SITE, None)
            .join::<TerrainLayers>(&TALUS, None);
    }
}
