use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Skin;

impl Slot for Skin {
    const NAME: &'static str = "skin";
    const WGSL: Option<&'static str> = Some(include_str!("places/skin.wgsl"));
}
