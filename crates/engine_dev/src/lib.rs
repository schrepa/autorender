#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bricks;
mod bridge;
mod controls;
mod digest;
mod inspect;
mod profile;
mod protocol;
mod reply;
pub mod results;
mod views;

#[cfg(not(target_arch = "wasm32"))]
pub mod authoring;
pub mod schemas;

pub use bricks::operation_names;
pub use bridge::{Artifact, Effect, HostContext, Outcome, Step};
pub use controls::{OperationCall, validated_camera, validated_presentation_alpha};
pub use engine_settings::ErrorCode;
pub use inspect::{
    Context, DevTools, InspectorDescriptor, InspectorKind, RendererFacts, TOOLING_TYPES, ToolCall,
    reserved_names,
};
pub use profile::profiled;
pub use protocol::bounds;
pub use protocol::bounds::{MAX_PAGE_SIZE, MAX_SCAN_ENTITIES};
pub use protocol::{
    BrowserControl, CaptureRequest, CompareRequest, Control, DeepProfile, DescribeRequest, Detail,
    Envelope, EnvironmentControl, InspectRequest, MAX_RESPONSE_BYTES, PROTOCOL_VERSION,
    PointerControl, ProfileRequest, Query, TouchControl, Verb,
};
pub use reply::{Observation, Reply, SceneObservation};

pub const MAX_INSPECTOR_BYTES: usize = 16 * 1024;

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
