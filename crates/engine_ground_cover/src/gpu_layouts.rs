use bytemuck::{Pod, Zeroable};

pub(crate) const PALETTE_BYTES: usize = PALETTE_ENTRIES * size_of::<PaletteEntry>();

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

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct PaletteEntry as ["PaletteEntry" in "cover::grow"] {
        pub color: [f32; 4],
        pub params: [f32; 4],
        pub maps: [f32; 4],
        pub far: [f32; 4],
        pub relief: [f32; 4],
        pub canopy: [f32; 4],
        pub placed: [f32; 4],
        pub placed_size: [f32; 4],
        pub heights: [f32; 4],
    }
}

engine_render::gpu_consts! {
    pub(crate) const BLADE_BYTES: u64 = 32 => ["size_of CoverBlade"];
    pub(crate) const COVER_LEAVES: u32 = 5 => ["COVER_LEAVES"];
    pub(crate) const COVER_LUSH_RATIO: f32 = 2.0 => ["COVER_LUSH_RATIO"];
    pub(crate) const STRAIGHT_JOINTS: u32 = 3 => ["STRAIGHT_JOINTS"];
    pub(crate) const STRAIGHT_FIRST_VERTEX: u32 = 1024 => ["STRAIGHT_FIRST_VERTEX"];

    pub(crate) const PALETTE_ENTRIES: usize = 16_384 / size_of::<PaletteEntry>() => ["PALETTE_ENTRIES" in "cover::grow"];
}
