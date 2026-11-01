use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Default)]
pub struct Glow;

impl Slot for Glow {
    const NAME: &'static str = "bloom";
    const WGSL: Option<&'static str> = Some(include_str!("places/bloom.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::BLOOM];
}
