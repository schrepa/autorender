use bevy_ecs::prelude::Resource;
use engine_place::{List, Slot};
use serde::{Deserialize, Serialize};

use crate::shadows::{
    DEBUG_SHADOWED, DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL, DEBUG_SUN_NONE,
};

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

pub struct SunBlockers;

impl List for SunBlockers {
    const NAME: &'static str = "sun_blockers";
    type Member = ();
    const WGSL: Option<&'static str> = Some(contract!(
        "places/sun_blockers.wgsl",
        DEBUG_SUN_NONE,
        DEBUG_SUN_LEVEL,
        DEBUG_SUN_HORIZON,
        DEBUG_SUN_BEYOND,
        DEBUG_SHADOWED,
    ));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::SUN_BLOCKERS];
}

pub struct Ambient;

impl List for Ambient {
    const NAME: &'static str = "ambient";
    type Member = ();
    const WGSL: Option<&'static str> = Some(include_str!("places/ambient.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::AMBIENT];
}

pub struct Interiors;

impl List for Interiors {
    const NAME: &'static str = "interior";
    type Member = ();
    const WGSL: Option<&'static str> = Some(include_str!("places/interior.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::INTERIOR];
}
