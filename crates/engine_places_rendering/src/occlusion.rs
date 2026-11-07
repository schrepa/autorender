use bevy_ecs::prelude::Resource;
use engine_place::Slot;

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct OcclusionSource {
    pub texel_share: f32,
}

impl Slot for OcclusionSource {
    const NAME: &'static str = "occlusion_source";
}

#[derive(Resource, Default)]
pub struct AmbientVisibility;

impl Slot for AmbientVisibility {
    const NAME: &'static str = "ambient_occlusion";
    const WGSL: Option<&'static str> = Some(include_str!("places/ambient_occlusion.wgsl"));
}
