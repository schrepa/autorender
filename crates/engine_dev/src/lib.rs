//! Optional development inspection and source authoring, outside gameplay loops.
//!
//! Call [`DevTools::execute`] at an exclusive frame boundary. Inspectors are
//! explicitly registered read-only projections; no reflection or serialization
//! runs during ordinary frames. The browser host owns sessions and controls.

mod collision;
mod controls;
mod error;
mod inspect;
mod protocol;
mod terrain;
mod views;

#[cfg(not(target_arch = "wasm32"))]
pub mod authoring;

pub use controls::{validated_camera, validated_presentation_alpha};
pub use error::DevError;
pub use inspect::{DevTools, InspectorDescriptor, InspectorKind};
pub use protocol::{
    BenchmarkMode, BrowserControl, CollisionFilterRequest, CollisionSensorFilter,
    CollisionShapeRequest, DevRequest, TerrainViewRequest,
};
pub use terrain::{Press, TerrainEditing, change_world, edit_terrain, nav_go, press_track};

/// Initial development protocol version, independent of scene format versions.
pub const PROTOCOL_VERSION: u32 = 1;
pub const MAX_REQUEST_BYTES: usize = 16 * 1024;
pub const MAX_RESPONSE_BYTES: usize = 64 * 1024;
pub const MAX_INSPECTOR_BYTES: usize = 16 * 1024;
/// Inspectors a game may register beside the engine's own.
pub const MAX_INSPECTORS: usize = 64;
/// Bytes of an inspector's description.
pub const MAX_DESCRIPTION_BYTES: usize = 512;
pub const MAX_SCAN_ENTITIES: usize = 10_000;
pub const MAX_PAGE_SIZE: usize = 100;
