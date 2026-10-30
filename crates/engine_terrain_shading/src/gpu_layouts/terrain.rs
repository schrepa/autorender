use bytemuck::{Pod, Zeroable};

use super::constants::HORIZON_RECORDS;

engine_render::gpu_structs! {
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct PatchRecord {
        pub first: [i32; 2],
        pub shape: [f32; 2],
        pub atlas: [u32; 2],
        pub holes: u32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct HorizonsPiece as ["HorizonsPiece" in "terrain::piece"] {
        pub records: [[f32; 4]; HORIZON_RECORDS + 1],
    }
}
