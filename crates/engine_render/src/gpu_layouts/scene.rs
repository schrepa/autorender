use bytemuck::{Pod, Zeroable};

gpu_structs! {
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct Vertex {
        pub position: [f32; 3],
        pub normal: [i16; 2],
        pub uv: [f32; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct ExtraVertex {
        pub color: [u16; 4],
        pub uv1: [f32; 2],
    }
}
