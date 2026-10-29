use std::sync::Arc;

use engine_core::glam::Vec2;
use engine_core::{Error, Expected, Grounded, Pyramid};
use engine_settings::ErrorCode;
use engine_terrain::Layers;
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

pub const GROUND_HEIGHT: &str = "ground_height";

pub const PATH_LAYER: &str = "path";

pub const MAX_ALONG: usize = 10_000;

const MAX_DISTANCE: f32 = 1.0e7;

const MIN_EVERY: f32 = 0.01;

const MAX_OFFSET: f32 = 1.0e4;

const MAX_REACH: f32 = 1.0e6;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GroundHeight {
    through: String,
    at: [f32; 2],
}

pub fn resolve_values(
    value: &mut Value,
    named: &[(String, Arc<Layers>)],
    pyramid: &Pyramid,
    path: &str,
) -> Result<(), Error> {
    let rule = match value {
        Value::Object(fields) if fields.len() == 1 => fields.get(GROUND_HEIGHT).cloned(),
        Value::Null
        | Value::Bool(_)
        | Value::Number(_)
        | Value::String(_)
        | Value::Array(_)
        | Value::Object(_) => None,
    };
    if let Some(rule) = rule {
        *value = ground_height(rule, named, pyramid, &format!("{path}.{GROUND_HEIGHT}"))?;
        return Ok(());
    }
    match value {
        Value::Object(fields) => {
            for (key, field) in fields.iter_mut().filter(|(_, field)| nested(field)) {
                resolve_values(field, named, pyramid, &format!("{path}.{key}"))?;
            }
        }
        Value::Array(items) => {
            for (index, item) in items.iter_mut().enumerate().filter(|(_, item)| nested(item)) {
                resolve_values(item, named, pyramid, &format!("{path}[{index}]"))?;
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
    Ok(())
}

fn nested(value: &Value) -> bool {
    value.is_object() || value.is_array()
}

fn ground_height(
    rule: Value,
    named: &[(String, Arc<Layers>)],
    pyramid: &Pyramid,
    path: &str,
) -> Result<Value, Error> {
    let GroundHeight { through, at }: GroundHeight = engine_core::parse_value_at(rule, path)?;
    let Some((_, layers)) = named.iter().find(|(id, _)| *id == through) else {
        return Err(
            Error::new(ErrorCode::UnknownReference, format!("{path}.through"))
                .got(&through)
                .variants(named.iter().map(|(id, _)| id.clone())),
        );
    };
    for (index, coordinate) in at.into_iter().enumerate() {
        if !coordinate.is_finite() || coordinate.abs() > Layers::MAX_COORDINATE {
            let bound = f64::from(Layers::MAX_COORDINATE);
            return Err(
                Error::new(ErrorCode::InvalidNumber, format!("{path}.at[{index}]"))
                    .expected(Expected::Finite)
                    .range(-bound, bound)
                    .got(coordinate),
            );
        }
    }
    let height = layers.height_at(pyramid, Vec2::from(at));
    Number::from_f64(f64::from(height))
        .map(Value::Number)
        .ok_or_else(|| Error::new(ErrorCode::InvalidNumber, path).expected(Expected::Finite))
}

fn standing() -> Option<Grounded> {
    Some(Grounded::default())
}

engine_settings::settings! {
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SceneAlong {
        pub id: String = String::new() => id,
        pub prefab: String = String::new() => id,
        pub terrain: String = String::new() => id,
        pub layer: String = String::new() => id,
        #[serde(default)]
        pub first: f32 = 0.0 => range(0.0, MAX_DISTANCE),
        pub every: f32 = 1.0 => range(MIN_EVERY, MAX_DISTANCE),
        #[serde(default)]
        pub offset: f32 = 0.0 => range(-MAX_OFFSET, MAX_OFFSET),
        #[serde(default = "standing")]
        pub grounded: Option<Grounded> = standing() => any,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub streamed: Option<SceneAlongStreamed> = None => any,
    }
}

engine_settings::settings! {
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
    #[serde(deny_unknown_fields)]
    pub struct SceneAlongStreamed {
        pub reach: f32 = 0.0 => range(0.0, MAX_REACH),
    }
}

impl SceneAlong {
    pub fn spots(&self, points: &[[f32; 2]], path: &str) -> Result<Vec<(Vec2, f32)>, Error> {
        let mut spots = Vec::new();
        let mut next = self.first;
        let mut walked = 0.0;
        for pair in points.windows(2) {
            let (from, to) = (Vec2::from(pair[0]), Vec2::from(pair[1]));
            let length = from.distance(to);
            if length <= 0.0 {
                continue;
            }
            let along = (to - from) / length;
            while next <= walked + length {
                if spots.len() == MAX_ALONG {
                    return Err(Error::new(ErrorCode::OutOfRange, format!("{path}.every"))
                        .at_most(MAX_ALONG as u64)
                        .got(self.every));
                }
                let point = from + along * (next - walked) + along.perp() * self.offset;
                spots.push((point, -along.y.atan2(along.x)));
                next += self.every;
            }
            walked += length;
        }
        Ok(spots)
    }
}
