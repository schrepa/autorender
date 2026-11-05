use bevy_ecs::prelude::Resource;
use engine_place::{List, Slot};

use crate::shadows::{
    DEBUG_SHADOWED, DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL, DEBUG_SUN_NONE,
};

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lighting {
    pub highlights: bool,
}

impl Slot for Lighting {
    const NAME: &'static str = "lighting";
    const WGSL: Option<&'static str> = Some(include_str!("places/lighting.wgsl"));
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
}

pub struct Ambient;

impl List for Ambient {
    const NAME: &'static str = "ambient";
    type Member = ();
    const WGSL: Option<&'static str> = Some(include_str!("places/ambient.wgsl"));
}

pub struct Interiors;

impl List for Interiors {
    const NAME: &'static str = "interior";
    type Member = ();
    const WGSL: Option<&'static str> = Some(include_str!("places/interior.wgsl"));
}
