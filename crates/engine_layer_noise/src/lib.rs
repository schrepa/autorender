#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/source.rs"));

mod layer;
mod tables;
mod tile;

pub use layer::{Noise, NoiseLayer};

use engine_core::{Brick, Bricks};
use engine_terrain::layer::{AddTerrainLayers, LayerKind, no_files};

pub struct LayerNoiseBrick;

impl Brick for LayerNoiseBrick {
    fn register(bricks: &mut Bricks<'_>) {
        bricks.terrain_layers(KINDS);
    }
}

pub static NOISE_KIND: LayerKind = LayerKind {
    name: "noise",
    source: SOURCE,
    from_json: layer::from_json,
    read: layer::read,
    files: no_files,
    schema: engine_terrain::layer_schema!(layer::schema),
};

pub static KINDS: &[&LayerKind] = &[&NOISE_KIND];
