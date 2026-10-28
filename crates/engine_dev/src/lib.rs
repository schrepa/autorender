#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bricks;
mod bridge;
mod controls;
mod digest;
mod error;
mod inspect;
mod protocol;
mod reply;
pub mod results;
mod views;

#[cfg(not(target_arch = "wasm32"))]
pub mod authoring;
#[cfg(not(target_arch = "wasm32"))]
pub mod schemas;

pub use bricks::is_control;
pub use bridge::{Artifact, Effect, HostContext, Outcome, Step};
pub use controls::{OperationCall, validated_camera, validated_presentation_alpha};
pub use engine_settings::ErrorCode;
pub use error::DevError;
pub use inspect::{
    Context, DevTools, InspectorDescriptor, InspectorKind, RenderedMesh, RendererFacts,
    TOOLING_TYPES, reserved_names,
};
pub use protocol::bounds;
pub use protocol::bounds::{MAX_DESCRIPTION_BYTES, MAX_PAGE_SIZE};
pub use protocol::{
    BenchmarkMode, BrowserControl, Control, DevRequest, EnvironmentControl, HostControl,
    MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, PROTOCOL_VERSION, parse_request,
};
pub use reply::{Observation, Reply, SceneObservation};

mod ops {
    include!("generated/ops.rs");
}
pub use ops::{CONTROL_OPS, QUERY_OPS};

pub const FRAME_STATS_SCHEMA: &str = include_str!("generated/frame_stats.schema.json");

pub const MAX_INSPECTOR_BYTES: usize = 16 * 1024;
pub const MAX_INSPECTORS: usize = 64;
pub const MAX_SCAN_ENTITIES: usize = 10_000;

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
