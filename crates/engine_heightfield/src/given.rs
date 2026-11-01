use engine_codec::{Allowed, Error, ErrorCode, Expected};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{Layers, bound};

pub const MAX_GIVEN_RIVERS: usize = 4096;
pub const MAX_RIVER_POINTS: usize = 16_384;
pub const MAX_CREDIT: usize = 1024;
pub const MAX_WIDTH: f32 = 50_000.0;
pub const MAX_DEPTH: f32 = 500.0;
pub const MAX_DISCHARGE: f32 = 1.0e6;
pub const MAX_PRECISION: f32 = 1000.0;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[schemars(transform = state_rules)]
#[serde(deny_unknown_fields)]
pub struct GivenRiver {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(max = MAX_CREDIT))]
    pub name: Option<String>,
    #[schemars(length(min = 2, max = MAX_RIVER_POINTS))]
    pub points: Vec<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub width: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depth: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub discharge: Vec<f32>,
}

impl GivenRiver {
    pub fn validate(&self) -> Result<(), Error> {
        let [low, high] = reach();
        for (index, point) in self.points.iter().enumerate() {
            for (axis, &value) in point.iter().enumerate() {
                if !(low as f32..=high as f32).contains(&value) {
                    return Err(broken(format!("$.points[{index}][{axis}]"))
                        .range(low, high)
                        .got(bound(value)));
                }
            }
        }
        for (field, values, most) in [
            ("width", &self.width, MAX_WIDTH),
            ("depth", &self.depth, MAX_DEPTH),
            ("discharge", &self.discharge, MAX_DISCHARGE),
        ] {
            let most = bound(most);
            if let Some((index, &value)) = values
                .iter()
                .enumerate()
                .find(|&(_, &value)| !(value > 0.0 && value <= most as f32))
            {
                return Err(broken(format!("$.{field}[{index}]"))
                    .expected(Expected::Above)
                    .range(0.0, most)
                    .got(bound(value)));
            }
        }
        if !(2..=MAX_RIVER_POINTS).contains(&self.points.len()) {
            return Err(broken("$.points")
                .expected(Expected::Array)
                .allowed(Allowed::Range {
                    min: 2.0,
                    max: MAX_RIVER_POINTS as f64,
                }));
        }
        for (field, values) in [
            ("width", &self.width),
            ("depth", &self.depth),
            ("discharge", &self.discharge),
        ] {
            if !(values.len() <= 1 || values.len() == self.points.len()) {
                return Err(broken(format!("$.{field}")).expected(Expected::Matching));
            }
        }
        if self
            .name
            .as_ref()
            .is_some_and(|name| name.chars().count() > MAX_CREDIT)
        {
            return Err(broken("$.name")
                .expected(Expected::String)
                .allowed(Allowed::AtMost {
                    max: MAX_CREDIT as u64,
                }));
        }
        Ok(())
    }
}

fn broken(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::OutOfRange, path)
}

fn reach() -> [f64; 2] {
    [bound(-Layers::MAX_COORDINATE), bound(Layers::MAX_COORDINATE)]
}

fn state_rules(schema: &mut schemars::Schema) {
    let [low, high] = reach();
    stated(schema, "points", [("minimum", low), ("maximum", high)]);
    for (field, most) in [
        ("width", MAX_WIDTH),
        ("depth", MAX_DEPTH),
        ("discharge", MAX_DISCHARGE),
    ] {
        stated(schema, field, [("exclusiveMinimum", bound(0.0)), ("maximum", bound(most))]);
    }
}

fn stated(schema: &mut schemars::Schema, field: &str, keywords: [(&str, f64); 2]) {
    let Some(property) = schema
        .get_mut("properties")
        .and_then(|properties| properties.get_mut(field))
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    let target = numbers(property);
    for (keyword, value) in keywords {
        target.insert(keyword.to_owned(), Value::from(value));
    }
}

fn numbers(schema: &mut Map<String, Value>) -> &mut Map<String, Value> {
    let array = schema.get("type").is_some_and(|kind| {
        kind == "array"
            || kind
                .as_array()
                .is_some_and(|kinds| kinds.iter().any(|kind| kind == "array"))
    });
    if array {
        let items = schema
            .get_mut("items")
            .and_then(Value::as_object_mut)
            .expect("array_schema_has_items");
        return numbers(items);
    }
    schema
}

engine_codec::record!(GivenRiver {
    name,
    points,
    width,
    depth,
    discharge,
});
