use serde::Serialize;
use serde_json::Value;

use crate::DevError;
use crate::views::PresentationStateView;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct Reply {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<u32>,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation: Option<Observation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(
        not(target_arch = "wasm32"),
        schemars(with = "Option<serde_json::Map<String, Value>>")
    )]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DevError>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct Observation {
    pub frame: Option<String>,
    pub tick: Option<String>,
    pub scene: Option<SceneObservation>,
    pub presentation: PresentationStateView,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SceneObservation {
    pub id: String,
    pub revision: String,
}
