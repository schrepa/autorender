#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod relief;

use engine_core::{Brick, Bricks};
use engine_terrain::layer::{AddTerrainLayers, LayerKind};

pub use relief::{HUMMOCKS, HummocksLayer, STEPS, StepsLayer};

pub struct LayerReliefBrick;

impl Brick for LayerReliefBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.terrain_layers(KINDS);
    }
}

pub static KINDS: &[&LayerKind] = &[&STEPS, &HUMMOCKS];
