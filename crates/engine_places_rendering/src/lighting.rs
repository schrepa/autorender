use bevy_ecs::prelude::Resource;
use engine_assets::Material;
use engine_place::{List, Slot};

use crate::shadows::{DEBUG_SUN_BEYOND, DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL, DEBUG_SUN_NONE};

pub const ROUGHNESS: &str = "roughness";
pub const METALNESS: &str = "metalness";
pub const OCCLUSION: &str = "occlusion";
pub const TRANSMISSION: &str = "transmission";
pub const OCCLUSION_STRENGTH: &str = "occlusion_strength";
pub const METALLIC_ROUGHNESS_MAP: &str = "metallic_roughness";
pub const OCCLUSION_MAP: &str = "occlusion";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceValues {
    pub roughness: f32,
    pub metalness: f32,
    pub occlusion_strength: f32,
    pub transmission: f32,
}

impl SurfaceValues {
    pub const MATTE: Self = Self {
        roughness: 0.8,
        metalness: 0.0,
        occlusion_strength: 1.0,
        transmission: 0.0,
    };

    pub fn of(material: Option<&Material>) -> Self {
        let Some(material) = material else {
            return Self::MATTE;
        };
        Self {
            roughness: material
                .surface_value(ROUGHNESS)
                .unwrap_or(Self::MATTE.roughness),
            metalness: material
                .surface_value(METALNESS)
                .unwrap_or(Self::MATTE.metalness),
            occlusion_strength: material
                .value(OCCLUSION_STRENGTH)
                .unwrap_or(Self::MATTE.occlusion_strength),
            transmission: material
                .value(TRANSMISSION)
                .unwrap_or(Self::MATTE.transmission),
        }
    }

    pub const fn units(self) -> [f32; 2] {
        [self.roughness, self.metalness]
    }
}

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
