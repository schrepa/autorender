use bytemuck::{Pod, Zeroable};

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub(crate) struct DisplayPiece as ["DisplayPiece" in "tonemap::piece"] {
        pub grade: [f32; 4],
        pub shadow_tint: [f32; 4],
        pub highlight_tint: [f32; 4],
        pub exposure: [f32; 4],
    }
}
