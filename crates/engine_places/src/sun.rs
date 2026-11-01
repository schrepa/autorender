use bevy_ecs::prelude::Resource;
use engine_place::{Slot, gpu_struct};
use glam::{Quat, Vec3};

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SunPath {
    pub direction: Option<Vec3>,
    pub hours: Option<f32>,
    pub moon: Option<Vec3>,
    pub stars: Quat,
}

impl SunPath {
    pub const DEFAULT_STARS: Quat = Quat::IDENTITY;
}

impl Default for SunPath {
    fn default() -> Self {
        Self {
            direction: None,
            hours: None,
            moon: None,
            stars: Self::DEFAULT_STARS,
        }
    }
}

impl Slot for SunPath {
    const NAME: &'static str = "sun_path";
}

gpu_struct! {
    #[derive(Resource)]
    pub struct Sun {
        pub direction: [f32; 4],
        pub color: [f32; 4],
    }
}

impl Sun {
    pub const NONE: Self = Self {
        direction: [0.0, 1.0, 0.0, 0.0],
        color: [0.0, 0.0, 0.0, 0.0],
    };
}

impl Slot for Sun {
    const NAME: &'static str = "sun";
    const WGSL: Option<&'static str> = Some(include_str!("places/sun.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::SUN];
}
