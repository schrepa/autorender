use std::{error::Error, fmt};

use engine_settings::ErrorCode;
use serde::Serialize;

/// Actionable load failure. `code` is stable; `message` is intended for people.
///
/// `path` uses JSON paths such as `$.entities[2].mesh`. JSON syntax failures
/// include their line and column in `message` and use `$` for the path.
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

/// The error of a value at authored path `path` breaking its field's rule:
/// code `code`, the field's own path, and the rule.
pub(crate) fn broken_rule(
    code: ErrorCode,
    path: &str,
    invalid: engine_settings::InvalidSetting,
) -> SceneError {
    SceneError::new(code, format!("{path}.{}", invalid.field), invalid.rule)
}
