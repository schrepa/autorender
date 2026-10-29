use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct DitherPiece as ["DitherPiece" in "dither::piece"] {
        pub flags: [f32; 4],
    }
}
