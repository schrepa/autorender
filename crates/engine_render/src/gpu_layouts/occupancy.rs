//! Upload structs of `crate::occupancy`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `Splat` of `occupancy.wgsl`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SplatUniform as ["Splat"] {
        pub area: [f32; 4],
        pub counts: [u32; 4],
    }

    /// WGSL `Cover`: a draw group's footprint radius, the bottom and top of
    /// what it covers above its origin (mesh units), and its density.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub(crate) struct Cover as ["Cover" in OCCUPANCY_SHADER] {
        pub footprint: f32,
        pub bottom: f32,
        pub top: f32,
        pub density: f32,
    }
}
