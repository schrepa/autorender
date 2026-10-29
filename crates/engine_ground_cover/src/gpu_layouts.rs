use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct GrowUniform as ["Grow"] {
        pub origin: [f32; 4],
        pub eye: [f32; 4],
        pub grid: [i32; 4],
        pub counts: [u32; 4],
        pub scale: [f32; 4],
        pub tier: [f32; 4],
        pub planes: [[f32; 4]; 6],
        pub foliage: [f32; 4],
        pub view: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct FrameUniform as ["CoverFrame"] {
        pub wind: [f32; 4],
        pub ground: [f32; 4],
        pub blades: [u32; 4],
    }
}

engine_render::gpu_consts! {
    pub(crate) const BLADE_BYTES: u64 = 32 => ["size_of CoverBlade"];
    pub(crate) const COVER_LEAVES: u32 = 5 => ["COVER_LEAVES"];
    pub(crate) const COVER_LUSH_RATIO: f32 = 2.0 => ["COVER_LUSH_RATIO"];
    pub(crate) const STRAIGHT_JOINTS: u32 = 3 => ["STRAIGHT_JOINTS"];
    pub(crate) const STRAIGHT_FIRST_VERTEX: u32 = 1024 => ["STRAIGHT_FIRST_VERTEX"];
}
