#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bricks;
mod bridge;
mod controls;
mod digest;
mod explain;
mod inspect;
mod profile;
mod protocol;
mod reply;
pub mod results;
mod views;
mod watch;

#[cfg(not(target_arch = "wasm32"))]
pub mod authoring;
#[cfg(not(target_arch = "wasm32"))]
mod consumers;
#[cfg(not(target_arch = "wasm32"))]
pub mod schemas;

pub use bricks::operation_names;
pub use bridge::{Artifact, Effect, HostContext, Outcome, Step};
pub use controls::{OperationCall, validated_camera, validated_presentation_alpha};
pub use engine_settings::ErrorCode;
pub use explain::EntityFacts;
pub use inspect::{
    Context, DevTools, InspectorDescriptor, InspectorKind, RenderedMesh, RendererFacts,
    TOOLING_TYPES, reserved_names,
};
pub use profile::profiled;
pub use protocol::bounds;
pub use protocol::bounds::{MAX_PAGE_SIZE, MAX_SCAN_ENTITIES};
pub use protocol::{
    BenchmarkMode, BrowserControl, CaptureRequest, CheckLevel, CheckRequest, CompareRequest,
    Control, DeepProfile, DescribeRequest, Detail, Envelope, EnvironmentControl, ExplainQuestion,
    HostControl, InspectRequest, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, PROTOCOL_VERSION,
    ProfileRequest, Query, Vendor, Verb, WatchRequest, WatchedComponent, named, request,
};
pub use reply::{Observation, Reply, SceneObservation};
pub use watch::{Watcher, stat};

pub const FRAME_STATS_SCHEMA: &str = include_str!("generated/frame_stats.schema.json");

pub const MAX_INSPECTOR_BYTES: usize = 16 * 1024;
pub const MAX_INSPECTORS: usize = 64;

pub(crate) fn bounded<T: engine_settings::Setting>(
    value: T,
    rule: engine_settings::Rule,
    code: engine_settings::ErrorCode,
    path: &str,
) -> Result<T, engine_settings::Error> {
    match value.check("value", rule) {
        Ok(()) => Ok(value),
        Err(invalid) => Err(engine_settings::Error {
            path: path.to_owned(),
            ..invalid.error(code, "$")
        }),
    }
}
