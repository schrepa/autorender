use bevy_ecs::prelude::Resource;
use engine_place::{DrawPoint, Slot};

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct SkyLight {
    pub on: bool,
    pub sun_intensity: f32,
    pub sun_size: f32,
    pub adaptation: f32,
    pub environment: u64,
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
pub struct EnvironmentLayers;

impl Slot for EnvironmentLayers {
    const NAME: &'static str = "environment_layers";
}

impl DrawPoint for EnvironmentLayers {
    type Layer = ();
    type Args = EnvironmentFaces;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvironmentFaces {
    pub level: u32,
    pub first: u32,
}
