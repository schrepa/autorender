//! Upload structs of `crate::deformation`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `Stamp`.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct StampRecord as ["Stamp" in DEFORMATION_SHADER] {
        pub ends: [f32; 4],
        pub shape: [f32; 4],
        pub bounds: [i32; 4],
    }

    /// WGSL `Job`.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct JobUniform as ["Job" in DEFORMATION_SHADER] {
        pub first: [i32; 4],
        pub counts: [u32; 4],
        pub scale: [f32; 4],
    }
}
