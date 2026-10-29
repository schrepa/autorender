use std::fmt::{self, Write};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::error::Category;
use serde_path_to_error::Segment;

use crate::ErrorCode;

pub const MAX_GOT_BYTES: usize = 256;

pub const JS_ERROR_NAME: &str = "EngineError";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Error {
    pub code: ErrorCode,
    pub path: String,
    #[serde(flatten)]
    pub detail: Box<Detail>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Detail {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<Expected>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed: Option<Allowed>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nearest: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub got: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum Expected {
    Number,
    Integer,
    String,
    Boolean,
    Object,
    Array,
    Id,
    Path,
    File,
    Entity,
    Unique,
    Present,
    Absent,
    Finite,
    Normalized,
    Sorted,
    Nonzero,
    Above,
    PowerOfTwo,
    Multiple,
    Json,
    Utf8,
    Uri,
    Base64,
    Image,
    Paused,
    Running,
    Gpu,
    Live,
    NonEmpty,
    Readable,
    Writable,
    Matching,
    Supported,
    Acyclic,
    Opaque,
    Triangles,
    Invertible,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Allowed {
    Range { min: f64, max: f64 },
    AtMost { max: u64 },
    AtLeast { min: u64 },
    Variants { values: Vec<String> },
}

impl Error {
    pub fn new(code: ErrorCode, path: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            detail: Box::default(),
        }
    }

    #[must_use]
    pub fn expected(mut self, expected: Expected) -> Self {
        self.detail.expected = Some(expected);
        self
    }

    #[must_use]
    pub fn allowed(mut self, allowed: Allowed) -> Self {
        self.detail.allowed = Some(allowed);
        self.detail.guess();
        self
    }

    #[must_use]
    pub fn range(self, min: f64, max: f64) -> Self {
        self.allowed(Allowed::Range { min, max })
    }

    #[must_use]
    pub fn at_most(self, max: u64) -> Self {
        self.allowed(Allowed::AtMost { max })
    }

    #[must_use]
    pub fn at_least(self, min: u64) -> Self {
        self.allowed(Allowed::AtLeast { min })
    }

    #[must_use]
    pub fn variants<S: Into<String>>(self, values: impl IntoIterator<Item = S>) -> Self {
        self.allowed(Allowed::Variants {
            values: values.into_iter().map(Into::into).collect(),
        })
    }

    #[must_use]
    pub fn got(mut self, value: impl Serialize) -> Self {
        self.detail.got = Some(given(value));
        self.detail.guess();
        self
    }

    #[must_use]
    pub fn nearest(mut self, value: impl Serialize) -> Self {
        self.detail.nearest = Some(given(value));
        self
    }

    #[must_use]
    pub fn at(mut self, path: impl Into<String>) -> Self {
        self.path = path.into();
        self
    }

    #[must_use]
    pub fn within(mut self, prefix: &str) -> Self {
        self.path = joined(prefix, &self.path);
        self
    }

    #[must_use]
    pub fn coded(mut self, code: ErrorCode) -> Self {
        self.code = code;
        self
    }
}

impl Detail {
    fn guess(&mut self) {
        if self.nearest.is_some() {
            return;
        }
        let (Some(Allowed::Variants { values }), Some(Value::String(got))) =
            (&self.allowed, &self.got)
        else {
            return;
        };
        self.nearest = closest(got, values).map(|value| Value::String(value.clone()));
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&serde_json::to_string(self).map_err(|_| fmt::Error)?)
    }
}

impl std::error::Error for Error {}

#[cfg(target_arch = "wasm32")]
impl From<Error> for wasm_bindgen::JsValue {
    fn from(error: Error) -> Self {
        use wasm_bindgen::JsCast;
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
}

pub fn closest<'a>(got: &str, allowed: &'a [String]) -> Option<&'a String> {
    allowed
        .iter()
        .map(|candidate| (distance(got, candidate), candidate))
        .filter(|&(distance, candidate)| distance <= candidate.len().max(got.len()) / 2)
        .min_by_key(|&(distance, _)| distance)
        .map(|(_, candidate)| candidate)
}

fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, left) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, &right) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(left != right));
            diagonal = above;
        }
    }
    row[b.len()]
}

fn given(value: impl Serialize) -> Value {
    cut(serde_json::to_value(value).unwrap_or(Value::Null))
}

fn cut(value: Value) -> Value {
    match value {
        Value::String(mut text) if text.len() > MAX_GOT_BYTES => {
            let mut end = MAX_GOT_BYTES;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
            Value::String(text)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(cut).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, cut(value)))
                .collect(),
        ),
        other @ (Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)) => other,
    }
}

fn joined(prefix: &str, path: &str) -> String {
    match path.strip_prefix('$') {
        Some(rest) => format!("{prefix}{rest}"),
        None if path.is_empty() => prefix.to_owned(),
        None => path.to_owned(),
    }
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
        Category::Data => Error::new(ErrorCode::InvalidValue, path),
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
