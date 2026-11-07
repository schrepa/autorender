use bevy_ecs::prelude::Resource;
use engine_place::{List, Slot};

use crate::shadows::{
    DEBUG_SHADOWED, DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL, DEBUG_SUN_NONE,
};

pub const SPECULAR_VARIANCE: f32 = 0.15;
pub const SPECULAR_LIMIT: f32 = 0.2;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lighting {
    pub highlights: bool,
}

impl Slot for Lighting {
    const NAME: &'static str = "lighting";
    const WGSL: Option<&'static str> = Some(engine_places::contract!(
        "places/lighting.wgsl";
        SPECULAR_VARIANCE,
        SPECULAR_LIMIT,
    ));
}

pub struct SunBlockers;

impl List for SunBlockers {
    const NAME: &'static str = "sun_blockers";
    type Member = ();
    const WGSL: Option<&'static str> = Some(engine_places::contract!(
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
