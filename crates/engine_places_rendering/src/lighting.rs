use bevy_ecs::prelude::Resource;
use engine_place::{List, Slot};

use crate::shadows::{DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL, DEBUG_SUN_NONE};

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
    const WGSL: Option<&'static str> = Some(engine_places::contract!(
        "places/sun_blockers.wgsl",
        DEBUG_SUN_NONE,
        DEBUG_SUN_LEVEL,
        DEBUG_SUN_HORIZON,
        DEBUG_SUN_BEYOND,
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

pub mod inputs {
    use engine_place::PointInput;
    use engine_places::surface;

    pub const ROUGHNESS: PointInput = PointInput::new(surface::ROUGHNESS, "f32", "1.0");
    pub const METALNESS: PointInput = PointInput::new(surface::METALNESS, "f32", "0.0");
    pub const OCCLUSION: PointInput = PointInput::new(surface::OCCLUSION, "f32", "1.0");
    pub const TRANSMISSION: PointInput = PointInput::new(surface::TRANSMISSION, "f32", "0.0");
    pub const DIRECT_SOFTNESS: PointInput = PointInput::new("direct_softness", "f32", "0.0");
    pub const IGNORE_FACE: PointInput = PointInput::new("ignore_face", "bool", "false");
    pub const ADDED_LIGHT: PointInput =
        PointInput::new("added_light", "vec3<f32>", "vec3<f32>(0.0)");
    pub const ADDED_INDIRECT: PointInput =
        PointInput::new("added_indirect", "vec3<f32>", "vec3<f32>(0.0)");
    pub const AMBIENT_BEND: PointInput =
        PointInput::new("ambient_bend", "vec3<f32>", "vec3<f32>(0.0)");
    pub const SMOOTHER_ACROSS: PointInput = PointInput::new("smoother_across", "f32", "0.0");
    pub const RECEIVER_LIFT_FIRST: PointInput = PointInput::new("receiver_lift_first", "u32", "0u");
    pub const RECEIVER_LIFTS: PointInput =
        PointInput::new("receiver_lifts", "vec4<f32>", "vec4<f32>(0.0)");
    pub const THICKNESS: PointInput = PointInput::new("thickness", "f32", "0.0");
    pub const DIRECT_BLOCKED: PointInput = PointInput::new("direct_blocked", "f32", "0.0");
    pub const DIRECT_KNOWN: PointInput = PointInput::new("direct_known", "bool", "false");
}
