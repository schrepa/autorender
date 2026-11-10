use bevy_ecs::prelude::Resource;
use engine_place::{AttributeFormat, Slot, StreamShape};

pub const JOINTS: StreamShape = StreamShape {
    name: "joints",
    format: AttributeFormat::Uint32x4,
};
pub const WEIGHTS: StreamShape = StreamShape {
    name: "weights",
    format: AttributeFormat::Float32x4,
};
pub const STREAMS: &[StreamShape] = &[JOINTS, WEIGHTS];

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Skin;

impl Slot for Skin {
    const NAME: &'static str = "skin";
    const WGSL: Option<&'static str> = Some(include_str!("places/skin.wgsl"));
}
