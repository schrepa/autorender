use bevy_ecs::prelude::Resource;
use engine_place::Slot;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LightPreset {
    Day,
    #[default]
    GoldenHour,
    Dusk,
    Night,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lighting {
    pub highlights: bool,
}

impl Slot for Lighting {
    const NAME: &'static str = "lighting";
    const WGSL: Option<&'static str> = Some(include_str!("places/lighting.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::LIGHTING];
}
