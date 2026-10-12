//! The envelope every development reply is: whether the request succeeded,
//! the engine's state it observed, and its result or error. The page's
//! types of replies are generated from it.

use serde::Serialize;
use serde_json::Value;

use crate::DevError;
use crate::views::PresentationStateView;

/// A development request's reply. Hosts add to a successful one's result
/// what they carry out: the observation again, whether they are paused,
/// the authored scene's source and the last frame's statistics
/// (`rendering`).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct Reply {
    /// The protocol's version; replies a host makes alone may leave it out.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    /// Whether the request succeeded.
    pub ok: bool,
    /// The engine's state the request observed; a reply from a host whose
    /// engine stopped has none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation: Option<Observation>,
    /// What a request that succeeded answers: its op's fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        not(target_arch = "wasm32"),
        schemars(with = "Option<serde_json::Map<String, Value>>")
    )]
    pub result: Option<Value>,
    /// Why the request failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DevError>,
}

/// The engine's state a reply observed. Counters and revisions are decimal
/// strings, which keep 64-bit values exact in JavaScript.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct Observation {
    /// The frame, as `FrameTime` counts it.
    pub frame: Option<String>,
    /// The last fixed tick, as `FixedTime` counts it.
    pub tick: Option<String>,
    /// The authored scene running, where one is.
    pub scene: Option<SceneObservation>,
    /// What the last frame presented.
    pub presentation: PresentationStateView,
}

/// The authored scene a reply observed.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SceneObservation {
    /// The scene's ID.
    pub id: String,
    /// The scene's revision, a decimal string.
    pub revision: String,
}
