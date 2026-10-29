use bytemuck::{Pod, Zeroable};

gpu_structs! {
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct ImpostorRecord as ["ImpostorRecord"] {
        pub sphere: [f32; 4],
        pub extent: [f32; 3],
        pub layer: u32,
        pub emissive: [f32; 3],
        pub flags: u32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub struct ImpostorBakeUniform as ["ImpostorBake"] {
        pub sphere: [f32; 4],
        pub flags: [u32; 4],
        pub tile: [u32; 4],
    }
}
