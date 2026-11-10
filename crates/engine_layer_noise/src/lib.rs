#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod layer;
mod tables;
mod tile;

pub use layer::{Noise, NoiseLayer};

use engine_core::{Brick, Bricks, Source};
use engine_heightfield::INVALID_TERRAIN;
use engine_heightfield::layer::{LayerKind, no_files, no_points};
use engine_open_world_kit::terrain::TerrainLayers;

pub struct LayerNoiseBrick;

impl Brick for LayerNoiseBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .errors(&[INVALID_TERRAIN])
            .join::<TerrainLayers>(&NOISE_KIND, None);
    }
}

pub static NOISE_KIND: LayerKind = LayerKind {
    name: "noise",
    source: SOURCE.hash(),
    from_json: layer::from_json,
    read: layer::read,
    files: no_files,
    schema: engine_heightfield::schema::<NoiseLayer>,
    points: no_points,
};
