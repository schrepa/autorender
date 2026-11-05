use std::any::Any;

use bevy_ecs::prelude::Resource;
use engine_place::{Error, Slot};
use serde::{Deserialize, Serialize};

pub const DEFAULT_TEXT_SIZE: f32 = 16.0;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FontId(pub u32);

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextRun {
    pub text: String,
    pub font: Option<FontId>,
    pub size: f32,
    pub color: [f32; 4],
    pub position: [f32; 2],
    pub max_width: Option<f32>,
    pub align: TextAlign,
    pub clip: Option<[f32; 4]>,
    pub step: Option<u32>,
}

impl Default for TextRun {
    fn default() -> Self {
        Self {
            text: String::new(),
            font: None,
            size: DEFAULT_TEXT_SIZE,
            color: [1.0; 4],
            position: [0.0; 2],
            max_width: None,
            align: TextAlign::Left,
            clip: None,
            step: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TextSize {
    pub width: f32,
    pub height: f32,
}

pub trait Typeset: Any + Send + Sync + 'static {
    fn register_font(&self, bytes: &[u8]) -> Result<FontId, Error>;

    fn measure(&self, run: &TextRun) -> TextSize;

    fn queue(&self, run: TextRun);
}

struct NoText;

impl Typeset for NoText {
    fn register_font(&self, _: &[u8]) -> Result<FontId, Error> {
        Ok(FontId::default())
    }

    fn measure(&self, _: &TextRun) -> TextSize {
        TextSize::default()
    }

    fn queue(&self, _: TextRun) {}
}

#[derive(Resource)]
pub struct Text(pub Box<dyn Typeset>);

impl Default for Text {
    fn default() -> Self {
        Self(Box::new(NoText))
    }
}

impl Slot for Text {
    const NAME: &'static str = "text";
}
