use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Instances;

impl Slot for Instances {
    const NAME: &'static str = "instances";
    const WGSL: Option<&'static str> = Some(include_str!("places/instances.wgsl"));
}
