use std::fmt::Write;

use engine_settings::ErrorCode;
use serde::de::DeserializeOwned;
use serde_json::error::Category;
use serde_path_to_error::Segment;

use crate::SceneError;

pub fn parse_json<T: DeserializeOwned>(source: &str) -> Result<T, SceneError> {
    let mut deserializer = serde_json::Deserializer::from_str(source);
    let value =
        serde_path_to_error::deserialize(&mut deserializer).map_err(|error| failure_at(&error))?;
    deserializer
        .end()
        .map_err(|error| failure(error, "$".to_owned()))?;
    Ok(value)
}

pub fn parse_value<T: DeserializeOwned>(value: serde_json::Value) -> Result<T, SceneError> {
    serde_path_to_error::deserialize(value).map_err(|error| failure_at(&error))
}

fn failure_at(error: &serde_path_to_error::Error<serde_json::Error>) -> SceneError {
    let mut path = String::from("$");
    for segment in error.path() {
        match segment {
            Segment::Seq { index } => write!(path, "[{index}]"),
            Segment::Map { key } => write!(path, ".{key}"),
            Segment::Enum { variant } => write!(path, ".{variant}"),
            Segment::Unknown => write!(path, ".?"),
        }
        .expect("strings write");
    }
    failure_of(error.inner(), path)
}

fn failure(error: serde_json::Error, path: String) -> SceneError {
    failure_of(&error, path)
}

fn failure_of(error: &serde_json::Error, mut path: String) -> SceneError {
    let message = error.to_string();
    let name_field = |path: &mut String| {
        if let Some(field) = message.split('`').nth(1)
            && !path.ends_with(&format!(".{field}"))
        {
            path.push('.');
            path.push_str(field);
        }
    };
    let code = match error.classify() {
        Category::Syntax | Category::Eof | Category::Io => {
            path = "$".to_owned();
            ErrorCode::InvalidJson
        }
        Category::Data if message.starts_with("unknown field") => {
            name_field(&mut path);
            ErrorCode::UnknownField
        }
        Category::Data if message.starts_with("missing field") => {
            name_field(&mut path);
            ErrorCode::MissingField
        }
        Category::Data if message.starts_with("invalid type") => ErrorCode::InvalidType,
        Category::Data if message.starts_with("unknown variant") => ErrorCode::UnknownVariant,
        Category::Data => ErrorCode::InvalidValue,
    };
    SceneError {
        code,
        path,
        message,
    }
}
