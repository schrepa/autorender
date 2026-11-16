#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/digest.rs"));

mod chain;
mod distance;
mod lod;
pub mod simplify;

use engine_codec::Error;

pub use lod::{LodInput, LodLevel, MeshLods, MeshVertex, authored_bytes, input_bytes};

pub fn run(input: &[u8]) -> Result<Vec<u8>, Error> {
    let input = LodInput::from_bytes(input)?;
    let lods = chain::levels(&input)?;
    Ok(lods.to_bytes())
}

pub fn run_authored(input: &[u8]) -> Result<Vec<u8>, Error> {
    let meshes = lod::authored_inputs(input)?;
    let errors = match meshes.split_first() {
        Some((finest, levels)) => chain::authored_errors(finest, levels),
        None => Vec::new(),
    };
    Ok(errors.iter().flat_map(|error| error.to_le_bytes()).collect())
}

pub fn levels(vertices: &[MeshVertex], indices: &[u32]) -> Result<MeshLods, Error> {
    chain::levels(&LodInput {
        vertices: vertices.to_vec(),
        indices: indices.to_vec(),
    })
}
