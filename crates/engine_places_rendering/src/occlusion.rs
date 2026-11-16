use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Default)]
pub struct AmbientVisibility;

impl Slot for AmbientVisibility {
    const NAME: &'static str = "ambient_occlusion";
    const WGSL: Option<&'static str> = Some(include_str!("places/ambient_occlusion.wgsl"));
}
