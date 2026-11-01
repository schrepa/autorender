#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod casters;
mod chain;
mod distance;
mod lod;
pub mod simplify;

use engine_codec::Error;

pub use lod::{LodInput, LodLevel, MeshLods, MeshVertex, input_bytes};

pub fn run(input: &[u8]) -> Result<Vec<u8>, Error> {
    let input = LodInput::from_bytes(input)?;
    let lods = chain::levels(&input)?;
    Ok(lods.to_bytes())
}

pub fn levels(vertices: &[MeshVertex], indices: &[u32]) -> Result<MeshLods, Error> {
    chain::levels(&LodInput {
        vertices: vertices.to_vec(),
        indices: indices.to_vec(),
    })
}
