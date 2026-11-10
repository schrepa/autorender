use bevy_ecs::prelude::*;
use engine_core::{Error, Operation};
use engine_places_rendering::text::{DEFAULT_TEXT_SIZE, Text, TextAlign, TextRun};
use serde::{Deserialize, Serialize};

use crate::Label;
use crate::typeset::ensure;

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
    type Reply = TextSize;

    fn run(
        mut text: SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<TextSize, Error> {
        ensure(&mut text);
        let size = text.0.measure(&TextRun {
            text: request.text,
            size: request.size,
            max_width: request.max_width,
            align: request.align,
            ..TextRun::default()
        });
        Ok(TextSize {
            width: size.width,
            height: size.height,
        })
    }
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub(crate) struct TextSize {
    width: f32,
    height: f32,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct TextRuns {}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub(crate) struct TextLabels {
    labels: Vec<Label>,
}

impl Operation for TextRuns {
    const NAME: &'static str = "text_runs";
    type Params = Query<'static, 'static, &'static Label>;
    type Request = Self;
    type Reply = TextLabels;

    fn run(labels: SystemParamItem<'_, '_, Self::Params>, _: Self) -> Result<TextLabels, Error> {
        Ok(TextLabels {
            labels: labels.iter().cloned().collect(),
        })
    }
}
