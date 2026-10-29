use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct LightingPiece as ["LightingPiece" in "lighting::piece"] {
        pub model: [f32; 4],
        pub highlights: [f32; 4],
    }
}
