//! Optional development inspection and source authoring, outside gameplay loops.
//!
//! Call [`DevTools::execute`] at an exclusive frame boundary. Inspectors are
//! explicitly registered read-only projections; no reflection or serialization
//! runs during ordinary frames. The browser host owns sessions and controls.

mod animation;
mod collision;
mod controls;
mod error;
mod inspect;
mod protocol;
mod terrain;
mod views;

#[cfg(not(target_arch = "wasm32"))]
pub mod authoring;

pub use animation::{Animate, animate};
pub use controls::{validated_camera, validated_presentation_alpha};
pub use engine_settings::ErrorCode;
pub use error::DevError;
pub use inspect::{
    Context, DevTools, InspectorDescriptor, InspectorKind, RenderedMesh, RendererFacts,
};
pub use protocol::bounds;
pub use protocol::bounds::{MAX_DESCRIPTION_BYTES, MAX_PAGE_SIZE};
pub use protocol::{
    BenchmarkMode, BrowserControl, CollisionFilterRequest, CollisionSensorFilter,
    CollisionShapeRequest, DevRequest, Handler, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES,
    PROTOCOL_VERSION, TerrainViewRequest,
};

/// The ops of queries and controls, as requests name them, from the
/// protocol's schema (build.rs).
mod ops {
    include!(concat!(env!("OUT_DIR"), "/ops.rs"));
}
pub use ops::{CONTROL_OPS, QUERY_OPS};

/// The protocol as hosts and tools need it: its version, request limit and
/// ops, which `cargo xtask schema` writes for the web host and scripts.
pub const PROTOCOL_SUMMARY: &str = include_str!(concat!(env!("OUT_DIR"), "/protocol.json"));

/// The JSON schema of [`engine_core::FrameStats`], what every reply's
/// `rendering` holds, which `cargo xtask schema` turns into the web host's
/// types.
pub const FRAME_STATS_SCHEMA: &str =
    include_str!(concat!(env!("OUT_DIR"), "/frame_stats.schema.json"));
pub use terrain::{Press, TerrainEditing, change_world, edit_terrain, nav_go, press_track};

pub const MAX_INSPECTOR_BYTES: usize = 16 * 1024;
/// Inspectors a game may register beside the engine's own.
pub const MAX_INSPECTORS: usize = 64;
pub const MAX_SCAN_ENTITIES: usize = 10_000;

/// `value` checked against request bound `rule`: an error of code `code`
/// at `path` stating the rule.
pub(crate) fn bounded<T: engine_settings::Setting>(
    value: T,
    rule: engine_settings::Rule,
    code: engine_settings::ErrorCode,
    path: &str,
) -> Result<T, DevError> {
    match value.check("value", rule) {
        Ok(()) => Ok(value),
        Err(invalid) => Err(DevError::new(code, path, invalid.rule)),
    }
}
