use engine_core::{GpuBits, gpu_bits, gpu_consts, gpu_struct};

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CandidateState {
    pub visible: bool,
    pub moving: bool,
    pub valid: bool,
    pub phase_one: bool,
    pub phase_two: bool,
    pub occluded: bool,
    pub subpixel: bool,
    pub view: bool,
    #[bits(2)]
    pub level: u32,
    #[bits(2)]
    pub view_level: u32,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LodHistory {
    Unset = 0,
    Held = 1,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LodFade {
    #[bits(2)]
    pub shown: u32,
    #[bits(2)]
    pub previous: u32,
    #[bits(16)]
    pub progress: u32,
    #[bits(1, at = 31)]
    pub history: LodHistory,
}

impl LodFade {
    pub(crate) fn drawn_level(word: u32) -> Option<u32> {
        Self::unpack(word)
            .filter(|fade| fade.history == LodHistory::Held)
            .map(|fade| fade.shown)
    }
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CardFrames {
    #[bits(8)]
    pub first: u32,
    #[bits(8)]
    pub second: u32,
    #[bits(8)]
    pub third: u32,
    #[bits(8)]
    pub origin: u32,
}

gpu_consts! {
    pub const LOD_LEVELS: u32 = 3;
    pub const LOD_FADED: u32 = 65535;
    pub const NO_NODE: u32 = u32::MAX;
    pub const CULL_GROUP: u32 = 256;
    pub const GROUP: u32 = 64;
    pub const PARENT_GROUP: u32 = 64;
    pub const ARGS_GROUP: u32 = 64;
    pub const BLEND_GROUP: u32 = 64;
    pub const UNWATCHED: u32 = u32::MAX;
}

const _: () = assert!(
    LOD_FADED == <LodFade as GpuBits>::FIELDS[2].mask(),
    "lod_faded_is_the_progress_mask"
);

gpu_struct! {
    pub(crate) struct BlendedView {
        pub view: u32,
        pub rows: u32,
        pub batches: u32,
        pub pad: u32,
    }
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
        pub pad_nodes: u32,
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
        pub watched: u32,
        pub pad: u32,
        pub eye: [f32; 4],
        pub fade: [f32; 4],
        pub bases: [u32; 4],
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
        pub pad: u32,
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
