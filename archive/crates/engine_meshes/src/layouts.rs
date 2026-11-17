use engine_core::{gpu_bits, gpu_consts, gpu_struct};

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CandidateState {
    pub visible: bool,
    pub moving: bool,
    pub phase_one: bool,
    pub phase_two: bool,
    pub subpixel: bool,
    pub view: bool,
    #[bits(2)]
    pub level: u32,
    #[bits(2)]
    pub view_level: u32,
}

gpu_consts! {
    pub const LOD_LEVELS: u32 = 3;
    pub const CULL_GROUP: u32 = 256;
    pub const GROUP: u32 = 64;
    pub const ARGS_GROUP: u32 = 64;
    pub const UNWATCHED: u32 = u32::MAX;
    pub const JOINTS_AT: u32 = 9;
    pub const WEIGHTS_AT: u32 = 10;
}

const _: () = assert!(
    JOINTS_AT == engine_core::FIRST_NAMED_LOCATION && WEIGHTS_AT == JOINTS_AT + 1,
    "skin_streams_follow_the_instance_inputs"
);

const _: () = assert!(
    LOD_LEVELS == engine_places::meshes::LAST_LEVEL,
    "the_last_level_follows_the_mesh_levels"
);

pub(crate) const DRAW_ARGS_BYTES: u64 = 20;

gpu_struct! {
    pub(crate) struct PresentationFrame {
        pub candidates: u32,
        pub budget: u32,
        pub groups: u32,
        pub batches: u32,
        pub mask: u32,
        pub occlusion: u32,
        pub plane: u32,
        pub lod_error: f32,
        pub watched: u32,
        pub last_level: u32,
        pub pixel_angle: f32,
        pub texel: f32,
        pub eye: [f32; 4],
        pub bases: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct CandidateData {
        pub slot: u32,
        pub batch: u32,
        pub position: u32,
        pub pad: u32,
    }
}

gpu_struct! {
    pub(crate) struct BatchData {
        pub start: u32,
        pub count: u32,
        pub index_count: u32,
        pub last_triangles: u32,
        pub bounds_min: [f32; 4],
        pub bounds_max: [f32; 4],
        pub first_index: u32,
        pub base_vertex: i32,
        pub first_instance: u32,
        pub first_plane: u32,
        pub lod_ranges: [u32; 4],
        pub lod_errors: [f32; 4],
    }
}
