use engine_settings::ErrorCode;
use serde::Serialize;

/// Stable machine-readable code and affected field; messages are for people.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DevError {
    /// What failed, as the catalog names it.
    pub code: ErrorCode,
    /// Where in the request: a JSON path such as `$.ticks`.
    pub path: String,
    /// What failed, for people.
    pub message: String,
}

impl DevError {
    pub fn new(code: ErrorCode, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl From<engine_scene::SceneError> for DevError {
    fn from(error: engine_scene::SceneError) -> Self {
        Self {
            code: error.code,
            path: error.path,
            message: error.message,
        }
    }
}

impl std::fmt::Display for DevError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {}: {}", self.code, self.path, self.message)
    }
}

impl std::error::Error for DevError {}
