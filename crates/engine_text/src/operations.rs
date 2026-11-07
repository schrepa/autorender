use bevy_ecs::prelude::*;
use engine_core::{Error, Operation};
use engine_places_rendering::text::{DEFAULT_TEXT_SIZE, Text, TextAlign, TextRun};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Label;
use crate::typeset::ensure;

fn schema_of<T: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null)
}

fn default_size() -> f32 {
    DEFAULT_TEXT_SIZE
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct TextMeasure {
    text: String,
    #[serde(default = "default_size")]
    size: f32,
    #[serde(default)]
    max_width: Option<f32>,
    #[serde(default)]
    align: TextAlign,
}

impl Operation for TextMeasure {
    const NAME: &'static str = "text_measure";
    const CONTROL: bool = false;
    type Params = ResMut<'static, Text>;
    type Request = Self;

    fn run(mut text: SystemParamItem<'_, '_, Self::Params>, request: Self) -> Result<Value, Error> {
        ensure(&mut text);
        let size = text.0.measure(&TextRun {
            text: request.text,
            size: request.size,
            max_width: request.max_width,
            align: request.align,
            ..TextRun::default()
        });
        Ok(json!({ "width": size.width, "height": size.height }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct TextRuns {}

impl Operation for TextRuns {
    const NAME: &'static str = "text_runs";
    type Params = Query<'static, 'static, &'static Label>;
    type Request = Self;

    fn run(labels: SystemParamItem<'_, '_, Self::Params>, _: Self) -> Result<Value, Error> {
        let labels: Vec<&Label> = labels.iter().collect();
        Ok(json!({ "labels": labels }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }
}
