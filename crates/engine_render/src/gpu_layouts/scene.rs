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

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct MaterialUniform as ["MaterialMaps"] {
        pub transforms: [[f32; 4]; 10],
        pub emissive: [f32; 4],
        pub params: [f32; 4],
        pub variant: [f32; 4],
        pub top: [f32; 4],
        pub layer: [f32; 4],
    }
}
