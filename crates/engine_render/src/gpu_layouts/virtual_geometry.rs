use bytemuck::{Pod, Zeroable};

use super::constants::{MAX_VIEWS, VIRTUAL_SHADOW_PLANES};

gpu_structs! {
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct MeshRecord as ["VirtualMeshRecord", "Mesh" in VIRTUAL_SHADER] {
        pub bases: [u32; 4],
        pub info: [u32; 4],
        pub position_min: [f32; 4],
        pub position_step: [f32; 4],
        pub uv: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct ViewRecord as ["View" in VIRTUAL_SHADER] {
        pub planes: [[f32; 4]; VIRTUAL_SHADOW_PLANES],
        pub eye: [f32; 4],
        pub lod: [f32; 4],
        pub cone: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct FrameUniform as ["Frame" in VIRTUAL_SHADER] {
        pub counts: [u32; 4],
        pub view_projection: [[f32; 4]; 4],
        pub previous_view_projection: [[f32; 4]; 4],
        pub viewport: [f32; 4],
        pub tests: [u32; 4],
        pub feedback: [u32; 4],
        pub views: [ViewRecord; MAX_VIEWS],
    }
}
