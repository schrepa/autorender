#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

use engine_core::{Brick, Bricks};
use engine_terrain::layer::{AddTerrainLayers, LayerKind};

pub struct LayerErosionBrick;

impl Brick for LayerErosionBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.terrain_layers(KINDS);
    }
}

pub static KINDS: &[&LayerKind] = &[];
