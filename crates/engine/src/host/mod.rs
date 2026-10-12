//! What hosts wire: job workers, terrain tile fetches and the terrain's
//! debug views; and what the page and the native host share instead of
//! each keeping its own: the spread of measured values.

mod statistics;

pub use crate::app_host::{JobMessage, now_ms};
pub use engine_core::{
    FileHeader, GAMEPLAY_JOBS, IDLE_JOBS, JobId, JobRequest, JobShare, Jobs, TerrainDebug,
    TerrainView, TileRequest, URGENT_JOBS, VIEW_JOBS, content_hash, job_class, job_priority,
};
pub use statistics::{Distribution, median};

/// The types the web host shares with the engine, in TypeScript: what
/// `cargo xtask schema` writes to `web/src/generated/host.ts`.
#[cfg(all(feature = "schema", not(target_arch = "wasm32")))]
pub fn typescript() -> String {
    let distribution =
        serde_json::to_value(schemars::schema_for!(Distribution)).expect("schemas serialize");
    crate::tooling::interfaces(&distribution, "Distribution", "engine::host::Distribution")
}
