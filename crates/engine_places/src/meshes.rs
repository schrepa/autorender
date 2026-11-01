use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Default)]
pub struct Instances;

impl Slot for Instances {
    const NAME: &'static str = "instances";
    const WGSL: Option<&'static str> = Some(include_str!("places/instances.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::INSTANCES];
}
