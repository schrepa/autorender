use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Default)]
pub struct Exposure;

impl Slot for Exposure {
    const NAME: &'static str = "exposure";
    const WGSL: Option<&'static str> = Some(include_str!("places/exposure.wgsl"));
}
