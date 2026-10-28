use bytemuck::{Pod, Zeroable};

gpu_structs! {
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct FrameUniform as ["Frame" in SCATTER_CULL_SHADER] {
        pub eye: [f32; 4],
        pub counts: [u32; 4],
        pub strides: [u32; 4],
        pub fade: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ViewData as ["View" in SCATTER_CULL_SHADER] {
        pub planes: [[f32; 4]; 6],
        pub limits: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct Record as ["Record"] {
        pub center: [f32; 3],
        pub radius: f32,
        pub group: u32,
        pub scale: f32,
        pub terrain: u32,
        pub sink: f32,
        pub state: u32,
        pub pad: [u32; 3],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TerrainInfo as ["TerrainInfo"] {
        pub origin: [f32; 4],
        pub levels: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SnapUniform as ["Snap"] {
        pub counts: [u32; 4],
        pub eye: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct GroupData as ["Group" in SCATTER_CULL_SHADER] {
        pub ids: u32,
        pub count: u32,
        pub levels: u32,
        pub camera: u32,
        pub errors: [f32; 4],
        pub impostor: f32,
        pub record: u32,
        pub casters: u32,
        pub parts: u32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct DrawArgs as ["DrawArgs" in SCATTER_CULL_SHADER] {
        pub index_count: u32,
        pub instance_count: u32,
        pub first_index: u32,
        pub base_vertex: i32,
        pub first_instance: u32,
    }
}
