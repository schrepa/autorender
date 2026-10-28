use std::{error::Error, fmt};

use engine_settings::ErrorCode;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SceneError {
    pub code: ErrorCode,
    pub path: String,
    pub message: String,
}

impl SceneError {
    pub(crate) fn new(
        code: ErrorCode,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for SceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at {}: {}",
            self.code, self.path, self.message
        )
    }
}

impl Error for SceneError {}

impl From<engine_core::Error> for SceneError {
    fn from(error: engine_core::Error) -> Self {
        Self {
            code: error.code,
            path: error.path,
            message: error.message,
        }
    }
}

pub(crate) fn broken_rule(
    code: ErrorCode,
    path: &str,
    invalid: engine_settings::InvalidSetting,
) -> SceneError {
    SceneError::new(code, format!("{path}.{}", invalid.field), invalid.rule)
}
