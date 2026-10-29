use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Reply(#[cfg_attr(not(target_arch = "wasm32"), schemars(with = "crate::Reply"))] Value),
    Effect(Effect),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Effect {
    Pause {
        paused: bool,
    },
    Wait {
        within_ms: f64,
    },
    Frame {
        timestamp: f64,
    },
    Show,
    Game {
        name: String,
        args: Map<String, Value>,
    },
    Source,
    Save {
        name: String,
        contents: Value,
    },
    Stream {
        records: Vec<Value>,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub now: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub interrupted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Artifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub missing: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: String,
    pub mime_type: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct HostContext {
    pub width: u32,
    pub height: u32,
    pub paused: bool,
    pub device_pixel_ratio: f64,
    pub user_agent: String,
    #[serde(default)]
    pub source: Option<String>,
    pub timing: String,
}
