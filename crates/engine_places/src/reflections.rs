use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Default)]
pub struct Reflected;

impl Slot for Reflected {
    const NAME: &'static str = "reflections";
    const WGSL: Option<&'static str> = Some(include_str!("places/reflections.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::REFLECTIONS];
}
