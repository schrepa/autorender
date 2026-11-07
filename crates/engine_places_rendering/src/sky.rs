use bevy_ecs::prelude::Resource;
use engine_place::{Slot, StoreId};

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct SkyLight {
    pub on: bool,
    pub sun_intensity: f32,
    pub sun_size: f32,
    pub adaptation: f32,
    pub environment: Option<StoreId>,
}

impl Slot for SkyLight {
    const NAME: &'static str = "sky";
    const WGSL: Option<&'static str> = Some(include_str!("places/sky.wgsl"));
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct EyeMedium {
    pub surface: Option<f32>,
}

impl Slot for EyeMedium {
    const NAME: &'static str = "eye_medium";
}

#[derive(Resource, Default)]
pub struct OpenSky;

impl Slot for OpenSky {
    const NAME: &'static str = "open_sky";
    const WGSL: Option<&'static str> = Some(include_str!("places/open_sky.wgsl"));
}

#[derive(Resource, Default)]
pub struct SkyVisibility;

impl Slot for SkyVisibility {
    const NAME: &'static str = "sky_visibility";
    const WGSL: Option<&'static str> = Some(include_str!("places/sky_visibility.wgsl"));
}
