use bevy_ecs::prelude::{Component, Resource};
use engine_place::Slot;
use glam::Vec3;
use serde::{Deserialize, Serialize};

pub const MAX_SPOT_DEGREES: f32 = 89.0;
pub const MIN_SPOT_DEGREES: f32 = 0.1;
pub const MAX_LIGHT_RANGE: f32 = 10_000.0;
pub const LIGHT_INTERIOR: u32 = 1;
pub const LIGHT_EXTERIOR: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "AuthoredSpot", into = "AuthoredSpot")]
pub struct Spot {
    pub direction: Vec3,
    pub inner_angle: f32,
    pub outer_angle: f32,
    pub authored_degrees: Option<[f32; 2]>,
}

impl Default for Spot {
    fn default() -> Self {
        Self {
            direction: Vec3::NEG_Y,
            inner_angle: 25.0_f32.to_radians(),
            outer_angle: 40.0_f32.to_radians(),
            authored_degrees: None,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct AuthoredSpot {
    direction: Vec3,
    inner_angle_degrees: f32,
    outer_angle_degrees: f32,
}

impl Default for AuthoredSpot {
    fn default() -> Self {
        Self::from(Spot::default())
    }
}

impl From<AuthoredSpot> for Spot {
    fn from(authored: AuthoredSpot) -> Self {
        Self {
            direction: authored.direction,
            inner_angle: authored.inner_angle_degrees.to_radians(),
            outer_angle: authored.outer_angle_degrees.to_radians(),
            authored_degrees: Some([authored.inner_angle_degrees, authored.outer_angle_degrees]),
        }
    }
}

impl From<Spot> for AuthoredSpot {
    fn from(spot: Spot) -> Self {
        Self {
            direction: spot.direction,
            inner_angle_degrees: spot.inner_angle.to_degrees(),
            outer_angle_degrees: spot.outer_angle.to_degrees(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightRegion {
    #[default]
    Both,
    Interior,
    Exterior,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocalLight {
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub offset: Vec3,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spot: Option<Spot>,
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub radius: f32,
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub length: f32,
    #[serde(skip_serializing_if = "is_both")]
    pub region: LightRegion,
}

fn is_both(region: &LightRegion) -> bool {
    *region == LightRegion::Both
}

impl Default for LocalLight {
    fn default() -> Self {
        Self::point(Vec3::new(1.0, 0.85, 0.6), 20.0, 12.0)
    }
}

impl LocalLight {
    pub fn point(color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            color,
            intensity,
            range,
            offset: Vec3::ZERO,
            spot: None,
            radius: 0.0,
            length: 0.0,
            region: LightRegion::Both,
        }
    }

    pub fn spot(
        color: Vec3,
        intensity: f32,
        range: f32,
        direction: Vec3,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self {
            spot: Some(Spot {
                direction,
                inner_angle,
                outer_angle,
                authored_degrees: None,
            }),
            ..Self::point(color, intensity, range)
        }
    }

    #[must_use]
    pub fn with_offset(mut self, offset: Vec3) -> Self {
        self.offset = offset;
        self
    }

    #[must_use]
    pub fn with_shape(mut self, radius: f32, length: f32) -> Self {
        self.radius = radius;
        self.length = length;
        self
    }

    #[must_use]
    pub fn with_region(mut self, region: LightRegion) -> Self {
        self.region = region;
        self
    }
}

#[derive(Resource, Default)]
pub struct Lights;

impl Slot for Lights {
    const NAME: &'static str = "lights";
    const WGSL: Option<&'static str> = Some(contract!(
        "places/lights.wgsl",
        LIGHT_INTERIOR,
        LIGHT_EXTERIOR,
    ));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::LIGHTS];
}
