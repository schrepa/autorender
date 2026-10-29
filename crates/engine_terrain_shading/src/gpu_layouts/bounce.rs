use bytemuck::{Pod, Zeroable};

pub(crate) const BINS: usize = engine_terrain::bounce::BINS;

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct BounceFrame as ["BounceFrame"] {
        pub bins: [[f32; 4]; BINS],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct BounceTerrain as ["BounceTerrain"] {
        pub place: [f32; 4],
        pub grid: [u32; 4],
        pub layers: [u32; 4],
        pub sources: [u32; 4],
        pub level: [u32; 4],
        pub kept: [u32; 4],
        pub finer: [u32; 4],
    }
}
