//! Offline levels of detail: each mesh's coarser levels, simplified from
//! it with meshoptimizer's simplifier ([`simplify`], a bit-exact Rust
//! port) and the deviation each shows measured offline, and the triangles
//! shadow maps draw of each. The cook makes them for every mesh a scene
//! draws (the levels' job, [`engine_assets::LOD_RECORD`], which native
//! hosts' workers run) and the importer writes them into the mesh files it
//! makes. Native tools and the cook only: no runtime and no browser runs
//! this crate; runtimes load what it made ([`engine_assets::MeshLods`]).
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod casters;
mod chain;
mod distance;
pub mod simplify;

use engine_assets::{LodInput, MeshLods, MeshVertex};

/// The kind of the levels' job, which native hosts' workers run.
pub const KIND: &str = engine_assets::LOD_RECORD;

/// The levels' job: the levels of the mesh `input` describes
/// ([`engine_assets::lod_input`]), as their record's bytes.
pub fn run(input: &[u8]) -> Result<Vec<u8>, String> {
    let input = LodInput::from_bytes(input).map_err(|error| error.to_string())?;
    let lods = chain::levels(&input).map_err(|error| error.to_string())?;
    Ok(lods.to_bytes())
}

/// The levels of the mesh of `vertices` and `indices`, as the importer
/// writes them beside its mesh.
pub fn levels(vertices: &[MeshVertex], indices: &[u32]) -> Result<MeshLods, String> {
    chain::levels(&LodInput {
        vertices: vertices.to_vec(),
        indices: indices.to_vec(),
    })
    .map_err(|error| error.to_string())
}
