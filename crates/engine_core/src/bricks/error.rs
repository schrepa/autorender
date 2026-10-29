use std::fmt::{self, Write};

use engine_settings::ErrorCode;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use serde_json::error::Category;
use serde_path_to_error::Segment;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Error {
    pub code: ErrorCode,
    pub path: String,
    pub message: String,
}

impl Error {
    pub fn new(code: ErrorCode, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at {}: {}",
            self.code, self.path, self.message
        )
    }
}

impl std::error::Error for Error {}

pub fn parse_json<T: DeserializeOwned>(source: &str) -> Result<T, Error> {
    let mut deserializer = serde_json::Deserializer::from_str(source);
    let value =
        serde_path_to_error::deserialize(&mut deserializer).map_err(|error| failure_at(&error))?;
    deserializer
        .end()
        .map_err(|error| failure_of(&error, "$".to_owned()))?;
    Ok(value)
}

pub fn parse_value<T: DeserializeOwned>(value: Value) -> Result<T, Error> {
    serde_path_to_error::deserialize(value).map_err(|error| failure_at(&error))
}

pub fn parse_value_at<T: DeserializeOwned>(value: Value, path: &str) -> Result<T, Error> {
    parse_value(value).map_err(|mut error| {
        if let Some(rest) = error.path.strip_prefix('$') {
            error.path = format!("{path}{rest}");
        }
        error
    })
}

fn failure_at(error: &serde_path_to_error::Error<serde_json::Error>) -> Error {
    let mut path = String::from("$");
    for segment in error.path() {
        match segment {
            Segment::Seq { index } => write!(path, "[{index}]"),
            Segment::Map { key } => write!(path, ".{key}"),
            Segment::Enum { variant } => write!(path, ".{variant}"),
            Segment::Unknown => write!(path, ".?"),
        }
        .expect("string_writes");
    }
    failure_of(error.inner(), path)
}

fn failure_of(error: &serde_json::Error, mut path: String) -> Error {
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
    Error {
        code,
        path,
        message,
    }
}
