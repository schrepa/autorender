use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SunPiece as ["SunPiece" in "sun::piece"] {
        pub direction: [f32; 4],
        pub color: [f32; 4],
    }
}
