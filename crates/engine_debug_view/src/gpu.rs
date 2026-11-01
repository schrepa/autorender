use bevy_ecs::prelude::Resource;
use engine_core::{gpu_consts, gpu_struct};

gpu_consts! {
    pub const DEBUG_VIEW_KINDS: u32 = 1;
    pub const DEBUG_VIEW_LEVELS: u32 = 2;
    pub const DEBUG_VIEW_CARDS: u32 = 3;
    pub const DEBUG_VIEW_OVERDRAW: u32 = 6;
    pub const DEBUG_VIEW_NORMALS: u32 = 7;
}

gpu_struct! {
    #[derive(Resource)]
    pub(crate) struct DebugViewPiece {
        pub view: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct ProbedTargets {
        pub ids: [u32; 4],
        pub terrain: [u32; 2],
        pub count: f32,
        pub depth: f32,
    }
}
