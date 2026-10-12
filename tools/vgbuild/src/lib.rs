//! Offline virtual geometry: cluster LOD hierarchies built with
//! the vendored meshoptimizer's `clusterlod`, packed into the engine's
//! GPU-ready virtual mesh format. Native tools only; the browser never runs
//! this crate.

pub mod dag;
mod ffi;
pub mod pack;
pub mod procedural;

pub use dag::{Bounds, Cluster, Dag, Group, Node, NodeKind, Source, View, build};
pub use pack::{PAGE_WORDS, PINNED_TRIANGLES, pack, pin_error};

/// meshoptimizer's own mesh simplification: the reference
/// `engine_assets::simplify` ports bit for bit, which `cargo xtask pure
/// simplify` holds it to.
pub mod meshoptimizer {
    pub use crate::ffi::{SimplifyInput, simplify};
}

/// The `EVG1` bytes of `source`'s cluster LOD hierarchy.
pub fn build_virtual_mesh(source: &Source<'_>) -> Result<Vec<u8>, String> {
    let dag = build(source)?;
    Ok(pack(&dag, source)
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect())
}
