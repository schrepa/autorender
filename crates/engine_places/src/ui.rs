use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use bevy_ecs::prelude::{Component, Resource};
use engine_place::Slot;
use serde::{Deserialize, Serialize};

use crate::text::{DEFAULT_TEXT_SIZE, TextAlign};

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum UiKind {
    #[default]
    Panel,
    Label,
    Image,
    Bar,
    Button,
    List,
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum UiDirection {
    #[default]
    Column,
    Row,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UiLength {
    #[default]
    Auto,
    Px(f32),
    Percent(f32),
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum UiAlign {
    Start,
    Center,
    End,
    #[default]
    Stretch,
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum UiJustify {
    #[default]
    Start,
    Center,
    End,
    SpaceBetween,
    SpaceAround,
    SpaceEvenly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct UiInsets {
    pub left: Option<f32>,
    pub top: Option<f32>,
    pub right: Option<f32>,
    pub bottom: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UiImageRect {
    pub u: f32,
    pub v: f32,
    pub width: f32,
    pub height: f32,
}

impl UiImageRect {
    pub const WHOLE: Self = Self {
        u: 0.0,
        v: 0.0,
        width: 1.0,
        height: 1.0,
    };
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UiAction {
    Go(String),
    Push(String),
    Pop,
    Emit(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct UiNode {
    pub id: Option<String>,
    pub kind: UiKind,
    pub direction: UiDirection,
    pub gap: f32,
    pub padding: [f32; 4],
    pub margin: [f32; 4],
    pub width: UiLength,
    pub height: UiLength,
    pub grow: f32,
    pub align: UiAlign,
    pub justify: UiJustify,
    pub absolute: Option<UiInsets>,
    pub color: [f32; 4],
    pub radius: f32,
    pub border_color: [f32; 4],
    pub border_width: f32,
    pub text: String,
    pub text_size: f32,
    pub text_color: [f32; 4],
    pub text_align: TextAlign,
    pub bind: Option<String>,
    pub image: Option<String>,
    pub image_rect: UiImageRect,
    pub bar_value: f32,
    pub bar_color: [f32; 4],
    pub hover_color: Option<[f32; 4]>,
    pub press_color: Option<[f32; 4]>,
    pub focus_color: Option<[f32; 4]>,
    pub action: Option<UiAction>,
    pub children: Vec<UiNode>,
}

impl Default for UiNode {
    fn default() -> Self {
        Self {
            id: None,
            kind: UiKind::default(),
            direction: UiDirection::default(),
            gap: 0.0,
            padding: [0.0; 4],
            margin: [0.0; 4],
            width: UiLength::default(),
            height: UiLength::default(),
            grow: 0.0,
            align: UiAlign::default(),
            justify: UiJustify::default(),
            absolute: None,
            color: [0.0; 4],
            radius: 0.0,
            border_color: [0.0; 4],
            border_width: 0.0,
            text: String::new(),
            text_size: DEFAULT_TEXT_SIZE,
            text_color: [1.0; 4],
            text_align: TextAlign::default(),
            bind: None,
            image: None,
            image_rect: UiImageRect::WHOLE,
            bar_value: 0.0,
            bar_color: [1.0; 4],
            hover_color: None,
            press_color: None,
            focus_color: None,
            action: None,
            children: Vec::new(),
        }
    }
}

#[derive(
    Component, Clone, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(default, deny_unknown_fields)]
pub struct UiRoot {
    pub screen: Option<String>,
    pub font: Option<String>,
    pub root: UiNode,
    #[serde(skip)]
    #[schemars(skip)]
    pub bytes: Option<Arc<[u8]>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum UiValue {
    Text(String),
    Number(f32),
}

#[derive(Default)]
struct Values {
    held: BTreeMap<String, UiValue>,
    revision: u64,
}

#[derive(Resource, Default)]
pub struct UiValues {
    values: Mutex<Values>,
}

impl Slot for UiValues {
    const NAME: &'static str = "ui_values";
}

impl UiValues {
    pub fn set_text(&self, name: &str, text: &str) {
        self.set(name, UiValue::Text(text.to_owned()));
    }

    pub fn set_number(&self, name: &str, number: f32) {
        self.set(name, UiValue::Number(number));
    }

    pub fn get(&self, name: &str) -> Option<UiValue> {
        self.lock().held.get(name).cloned()
    }

    pub fn revision(&self) -> u64 {
        self.lock().revision
    }

    fn set(&self, name: &str, value: UiValue) {
        let mut values = self.lock();
        if values.held.get(name) != Some(&value) {
            values.held.insert(name.to_owned(), value);
            values.revision += 1;
        }
    }

    fn lock(&self) -> MutexGuard<'_, Values> {
        self.values.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[derive(Resource, Default)]
pub struct UiEvents {
    names: Vec<String>,
}

impl Slot for UiEvents {
    const NAME: &'static str = "ui_events";
}

impl UiEvents {
    pub fn pressed(&self, name: &str) -> bool {
        self.names.iter().any(|held| held == name)
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    #[doc(hidden)]
    pub fn emit(&mut self, name: &str) {
        self.names.push(name.to_owned());
    }

    #[doc(hidden)]
    pub fn clear(&mut self) {
        self.names.clear();
    }
}
