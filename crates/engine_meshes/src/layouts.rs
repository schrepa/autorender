use engine_core::{gpu_consts, gpu_struct};

gpu_consts! {
    pub const VISIBLE: u32 = 1;
    pub const PHASE1: u32 = 8;
    pub const PHASE2: u32 = 16;
    pub const PROBE: u32 = 128;
    pub const LOD_LEVELS: u32 = 3;
    pub const IMPOSTOR_VERTICES: u32 = 6;
    pub const CULL_GROUP: u32 = 256;
    pub const GROUP: u32 = 64;
    pub const PARENT_GROUP: u32 = 64;
    pub const ARGS_GROUP: u32 = 64;
    pub const PYRAMID_GROUP: u32 = 8;
    pub const STILLNESS: u32 = 1;
    pub const MOVING_MAP: u32 = 7;
    pub const SKY_MAP: u32 = 8;
    pub const MOVING_PAIRS: u32 = 8;
}

pub(crate) const DRAW_ARGS_BYTES: u64 = 20;
pub(crate) const COMPOSED_BYTES: u64 = 112;

gpu_struct! {
    pub(crate) struct PresentationFrame {
        pub alpha: f32,
        pub candidates: u32,
        pub budget: u32,
        pub culling: u32,
        pub groups: u32,
        pub batches: u32,
        pub node_capacity: u32,
        pub blend: u32,
        pub planes: [[f32; 4]; 6],
        pub absolute_view_projection: [[f32; 4]; 4],
        pub motion: u32,
        pub mask: u32,
        pub occlusion: u32,
        pub subpixel: u32,
        pub view_projection: [[f32; 4]; 4],
        pub viewport: [f32; 4],
        pub plane: u32,
        pub lod_error: f32,
        pub pad: [u32; 2],
        pub eye: [f32; 4],
        pub fade: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct CandidateData {
        pub slot: u32,
        pub node: u32,
        pub batch: u32,
        pub position: u32,
    }
}

gpu_struct! {
    pub(crate) struct BatchData {
        pub start: u32,
        pub count: u32,
        pub index_count: u32,
        pub group: u32,
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

gpu_struct! {
    pub(crate) struct CasterFrame {
        pub candidates: u32,
        pub groups: u32,
        pub capacity: u32,
        pub mode: u32,
    }
}
