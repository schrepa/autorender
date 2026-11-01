use std::fmt::Write;

use serde::de::DeserializeOwned;
use serde_json::Value;
use serde_json::error::Category;
use serde_path_to_error::Segment;

use crate::{Error, ErrorCode, Expected};

pub const JS_ERROR_NAME: &str = "EngineError";

#[cfg(target_arch = "wasm32")]
pub fn js_error(error: Error) -> js_sys::wasm_bindgen::JsValue {
    use js_sys::wasm_bindgen::JsCast;
    let line = error.to_string();
    let thrown = js_sys::Error::new(&line);
    thrown.set_name(JS_ERROR_NAME);
    let fields = js_sys::JSON::parse(&line)
        .ok()
        .and_then(|fields| fields.dyn_into::<js_sys::Object>().ok());
    if let Some(fields) = fields {
        let _ = js_sys::Object::assign(thrown.as_ref(), &fields);
    }
    thrown.into()
}

pub fn parse_json<T: DeserializeOwned>(source: &str) -> Result<T, Error> {
    let mut deserializer = serde_json::Deserializer::from_str(source);
    let value = serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
        let found = serde_json::from_str::<Value>(source).ok();
        failure_at(&error, found.as_ref())
    })?;
    deserializer
        .end()
        .map_err(|error| failure_of(&error, "$".to_owned(), None))?;
    Ok(value)
}

pub fn parse_value<T: DeserializeOwned>(value: Value) -> Result<T, Error> {
    serde_path_to_error::deserialize(&value).map_err(|error| failure_at(&error, Some(&value)))
}

pub fn parse_value_at<T: DeserializeOwned>(value: Value, path: &str) -> Result<T, Error> {
    parse_value(value).map_err(|error| error.within(path))
}

fn failure_at(
    error: &serde_path_to_error::Error<serde_json::Error>,
    source: Option<&Value>,
) -> Error {
    let mut path = String::from("$");
    for segment in error.path() {
        match segment {
            Segment::Seq { index } => write!(path, "[{index}]"),
            Segment::Map { key } => write!(path, ".{key}"),
            Segment::Enum { variant } => write!(path, ".{variant}"),
            Segment::Unknown => write!(path, ".?"),
        }
        .expect("strings_write");
    }
    let found = source.and_then(|source| {
        error
            .path()
            .iter()
            .try_fold(source, |value, segment| match segment {
                Segment::Seq { index } => value.get(*index),
                Segment::Map { key } => value.get(key.as_str()),
                Segment::Enum { variant } => value.get(variant.as_str()),
                Segment::Unknown => None,
            })
    });
    failure_of(error.inner(), path, found)
}

fn failure_of(error: &serde_json::Error, mut path: String, found: Option<&Value>) -> Error {
    let message = error.to_string();
    let quoted: Vec<&str> = message.split('`').skip(1).step_by(2).collect();
    let name_field = |path: &mut String| {
        if let Some(field) = quoted.first()
            && !path.ends_with(&format!(".{field}"))
        {
            path.push('.');
            path.push_str(field);
        }
    };
    let failure = match error.classify() {
        Category::Syntax | Category::Eof | Category::Io => {
            return Error::new(ErrorCode::InvalidJson, "$")
                .expected(Expected::Json)
                .got(serde_json::json!({ "line": error.line(), "column": error.column() }));
        }
        Category::Data if message.starts_with("unknown field") => {
            name_field(&mut path);
            let failure = Error::new(ErrorCode::UnknownField, path).expected(Expected::Absent);
            let failure = match quoted.first() {
                Some(field) => failure.got(field),
                None => failure,
            };
            return failure.variants(quoted.iter().skip(1).copied());
        }
        Category::Data if message.starts_with("missing field") => {
            name_field(&mut path);
            return Error::new(ErrorCode::MissingField, path).expected(Expected::Present);
        }
        Category::Data if message.starts_with("unknown variant") => {
            let failure = Error::new(ErrorCode::UnknownVariant, path);
            let failure = match quoted.first() {
                Some(variant) => failure.got(variant),
                None => failure,
            };
            return failure.variants(quoted.iter().skip(1).copied());
        }
        Category::Data if message.starts_with("invalid type") => {
            Error::new(ErrorCode::InvalidType, path)
        }
        Category::Data => match carried(&message) {
            Some(refusal) => return refusal.within(&path),
            None => Error::new(ErrorCode::InvalidValue, path),
        },
    };
    let failure = match wanted(&message) {
        Some(expected) => failure.expected(expected),
        None => failure,
    };
    match found {
        Some(value) => failure.got(value),
        None => failure,
    }
}

fn carried(message: &str) -> Option<Error> {
    let end = message.rfind('}')?;
    let line = message.get(..=end)?;
    if !line.starts_with('{') {
        return None;
    }
    serde_json::from_str(line).ok()
}

fn wanted(message: &str) -> Option<Expected> {
    let (_, wanted) = message.rsplit_once(", expected ")?;
    let wanted = wanted.split(" at line ").next().unwrap_or(wanted);
    let has = |words: &[&str]| words.iter().any(|word| wanted.contains(word));
    if has(&[
        "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize",
        "integer",
    ]) {
        Some(Expected::Integer)
    } else if has(&["f32", "f64", "number", "float"]) {
        Some(Expected::Number)
    } else if has(&["bool"]) {
        Some(Expected::Boolean)
    } else if has(&["sequence", "array", "tuple", "list"]) {
        Some(Expected::Array)
    } else if has(&["map", "struct", "object"]) {
        Some(Expected::Object)
    } else if has(&["string", "str", "char", "identifier"]) {
        Some(Expected::String)
    } else {
        None
    }
}
