use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ErrorCode;

pub const MAX_GOT_BYTES: usize = 256;

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
    pub fn got_whole(mut self, value: impl Serialize) -> Self {
        self.detail.got = Some(serde_json::to_value(value).unwrap_or(Value::Null));
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
