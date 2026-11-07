use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct OcclusionSource {
    pub texel_share: f32,
}

impl Slot for OcclusionSource {
    const NAME: &'static str = "occlusion_source";
}
