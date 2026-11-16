use bevy_ecs::prelude::Resource;
use bytemuck::{Pod, Zeroable};
use engine_core::{GpuBits, GpuStruct, gpu_struct};

use crate::constants::SKY_LAYERS;

const _: () = assert!(SKY_LAYERS == 26, "sky_views_hold_every_layer");

pub(crate) const GENERATED: &[&str] = &[
    crate::constants::WGSL,
    <engine_places_rendering::shadows::SunShown as GpuBits>::WGSL,
    ShadowLevel::WGSL,
    SkyViews::WGSL,
];

gpu_struct! {
    #[derive(Resource)]
    pub struct ShadowPiece {
        pub state: [f32; 4],
        pub level_state: [f32; 4],
        pub soft: [f32; 4],
        pub sky: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct ShadowLevel {
        pub rows: [[f32; 4]; 3],
        pub window: [f32; 4],
        pub band: [f32; 4],
        pub moving: [f32; 4],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct ComposeData {
    pub place: [f32; 4],
    pub source: [i32; 4],
}

gpu_struct! {
    pub(crate) struct SkyViews {
        pub clips: [[[f32; 4]; 4]; 26],
        pub directions: [[f32; 4]; 26],
        pub counts: [f32; 4],
        pub occlusion: [f32; 4],
    }
}
