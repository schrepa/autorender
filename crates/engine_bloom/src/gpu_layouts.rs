use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct BloomPiece as ["BloomPiece" in "bloom::piece"] {
        pub threshold: [f32; 4],
    }
}
