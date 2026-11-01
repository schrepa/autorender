use bevy_ecs::prelude::{Component, Resource};
use engine_place::Slot;
use glam::Vec3;

pub const MAX_SPOT_DEGREES: f32 = 89.0;
pub const MIN_SPOT_DEGREES: f32 = 0.1;
pub const MAX_LIGHT_RANGE: f32 = 10_000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spot {
    pub direction: Vec3,
    pub inner_angle: f32,
    pub outer_angle: f32,
}

impl Default for Spot {
    fn default() -> Self {
        Self {
            direction: Vec3::NEG_Y,
            inner_angle: 25.0_f32.to_radians(),
            outer_angle: 40.0_f32.to_radians(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LightRegion {
    #[default]
    Both,
    Interior,
    Exterior,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct LocalLight {
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub offset: Vec3,
    pub spot: Option<Spot>,
    pub radius: f32,
    pub length: f32,
    pub region: LightRegion,
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
    const WGSL: Option<&'static str> = Some(include_str!("places/lights.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::LIGHTS];
}
