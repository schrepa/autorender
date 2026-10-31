#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod relief;

use engine_core::{Brick, Bricks, Join};
use engine_terrain::layer::{LayerKind, TerrainLayersList};

pub use relief::{HUMMOCKS, HummocksLayer, STEPS, StepsLayer};

pub struct LayerReliefBrick;

impl Brick for LayerReliefBrick {
    fn register(bricks: &mut Bricks<'_>) {
        for &kind in KINDS {
            bricks.join(Join::<TerrainLayersList>::new().cpu(kind));
        }
    }
}

pub static KINDS: &[&LayerKind] = &[&STEPS, &HUMMOCKS];
