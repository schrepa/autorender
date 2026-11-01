use bevy_ecs::prelude::Resource;
use bytemuck::{Pod, Zeroable};
use engine_core::{GpuStruct, gpu_struct};
use engine_places::shadows::SHADOW_REGIONS;

use crate::constants::{MAX_MOVING_TILES, MAX_SHADOW_LEVELS, SKY_LAYERS};

const MOVING_TILES: usize = MAX_MOVING_TILES as usize;
const LAYERS: usize = SKY_LAYERS as usize;

const _: () = assert!(MAX_SHADOW_LEVELS == 20, "piece_and_cull_hold_every_level");
const _: () = assert!(SHADOW_REGIONS == 7, "cull_holds_every_region");

pub(crate) const GENERATED: &[&str] = &[crate::constants::WGSL, CasterCull::WGSL];

gpu_struct! {
    #[derive(Resource)]
    pub struct ShadowPiece {
        pub light: [[f32; 4]; 3],
        pub state: [f32; 4],
        pub level_state: [f32; 4],
        pub atlas: [f32; 4],
        pub windows: [[f32; 4]; 20],
        pub bands: [[f32; 4]; 20],
    }
}

gpu_struct! {
    pub(crate) struct CasterCull {
        pub light: [[f32; 4]; 3],
        pub regions: [[f32; 4]; 7],
        pub region_depths: [[f32; 4]; 7],
        pub windows: [[f32; 4]; 20],
        pub counts: [u32; 4],
        pub moving: [u32; 4],
        pub mode: [u32; 4],
        pub sky: [[f32; 4]; 3],
        pub sky_rect: [f32; 4],
        pub sky_depths: [f32; 4],
    }
}

#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct Routes {
    pub clips: [[[f32; 4]; 4]; MOVING_TILES],
    pub slots: [[f32; 4]; MOVING_TILES],
    pub atlas: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct ComposeData {
    pub place: [f32; 4],
    pub source: [i32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct SkyViews {
    pub clips: [[[f32; 4]; 4]; LAYERS],
    pub directions: [[f32; 4]; LAYERS],
    pub counts: [f32; 4],
    pub occlusion: [f32; 4],
}
