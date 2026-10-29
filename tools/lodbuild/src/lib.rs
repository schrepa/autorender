#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod casters;
mod chain;
mod distance;
pub mod simplify;

use engine_assets::{LodInput, MeshLods, MeshVertex};
use engine_settings::Error;

pub const KIND: &str = engine_assets::LOD_RECORD;

pub fn run(input: &[u8]) -> Result<Vec<u8>, String> {
    let input = LodInput::from_bytes(input).map_err(|error| error.to_string())?;
    let lods = chain::levels(&input).map_err(|error| error.to_string())?;
    Ok(lods.to_bytes())
}

pub fn levels(vertices: &[MeshVertex], indices: &[u32]) -> Result<MeshLods, Error> {
    chain::levels(&LodInput {
        vertices: vertices.to_vec(),
        indices: indices.to_vec(),
    })
}
