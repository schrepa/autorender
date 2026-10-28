use serde::de::DeserializeOwned;

use crate::SceneError;

pub fn parse_json<T: DeserializeOwned>(source: &str) -> Result<T, SceneError> {
    engine_core::parse_json(source).map_err(SceneError::from)
}

pub fn parse_value<T: DeserializeOwned>(value: serde_json::Value) -> Result<T, SceneError> {
    engine_core::parse_value(value).map_err(SceneError::from)
}
