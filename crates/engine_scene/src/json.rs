//! JSON the engine reads, scene sources and development requests, parsed
//! so a failure names the value at fault: its JSON path and what is wrong
//! with it.

use std::fmt::Write;

use engine_settings::ErrorCode;
use serde::de::DeserializeOwned;
use serde_json::error::Category;
use serde_path_to_error::Segment;

use crate::SceneError;

/// `source` as a `T`. A failure's code is `invalid_json` for malformed JSON
/// and otherwise says what the value at its path breaks: `unknown_field`,
/// `missing_field`, `invalid_type`, `unknown_variant` or `invalid_value`.
pub fn parse_json<T: DeserializeOwned>(source: &str) -> Result<T, SceneError> {
    let mut deserializer = serde_json::Deserializer::from_str(source);
    let value = serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
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
        failure(error.into_inner(), path)
    })?;
    deserializer
        .end()
        .map_err(|error| failure(error, "$".to_owned()))?;
    Ok(value)
}

/// `error` at `path` as a scene error of the code its kind names; a field
/// unknown or missing extends the path.
fn failure(error: serde_json::Error, mut path: String) -> SceneError {
    let message = error.to_string();
    // The field a message names, joined to the path unless it ends there.
    let name_field = |path: &mut String| {
        if let Some(field) = message.split('`').nth(1)
            && !path.ends_with(&format!(".{field}"))
        {
            path.push('.');
            path.push_str(field);
        }
    };
    let code = match error.classify() {
        // Malformed JSON has no value at fault.
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
