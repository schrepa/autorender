use bytemuck::{Pod, Zeroable};
use engine_core::PoseRows;

gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct DecalData as ["DecalRecord"] {
        pub model: PoseRows,
        pub color: [f32; 4],
        pub params: [f32; 4],
        pub pose: [u32; 4],
    }
}
