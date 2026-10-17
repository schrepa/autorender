//! Small, engine-independent primitives for developer commands. Cargo remains
//! the authority on build freshness; this library manages output lifetimes.

pub mod analyzer;
mod analyzer_check;
pub mod bindings;
pub mod builds;
pub mod cache;
pub mod cargo_context;
pub mod execution;
pub mod files;
pub mod jobs;
pub mod native;
pub mod request_contract;
mod symbols;
pub mod tooling_check;
mod tooling_workflows;

use std::path::{Path, PathBuf};

/// Select a workspace at runtime, so a copied helper never uses its build tree.
pub fn workspace(from: &Path) -> Result<PathBuf, String> {
    let from = files::canonical(from)?;
    from.ancestors()
        .find(|path| path.join("tools/xtask/Cargo.toml").is_file())
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("no engine workspace above {}", from.display()))
}
