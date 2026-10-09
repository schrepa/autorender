//! Upload structs of `crate::ground_cover`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `Grow` of `cover_grow.wgsl`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct GrowUniform as ["Grow"] {
        pub origin: [f32; 4],
        pub eye: [f32; 4],
        pub grid: [i32; 4],
        pub counts: [u32; 4],
        pub scale: [f32; 4],
        pub tier: [f32; 4],
        pub planes: [[f32; 4]; 6],
        pub deform_grid: [i32; 4],
        pub deform: [f32; 4],
        pub ranges: [[f32; 4]; 2],
    }

    /// WGSL `CoverFrame` of `meshes/cover.wgsl`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct FrameUniform as ["CoverFrame"] {
        pub wind: [f32; 4],
    }
}
