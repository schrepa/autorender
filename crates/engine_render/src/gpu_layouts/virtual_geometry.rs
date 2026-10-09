//! Upload structs of `crate::virtual_geometry`.

use bytemuck::{Pod, Zeroable};

use super::constants::{MAX_VIEWS, VIRTUAL_SHADOW_PLANES};

gpu_structs! {
    /// A virtual mesh in the arena: its node, group and cluster bases and
    /// where the pool's page slots start; its levels (one root node each)
    /// and hierarchy depth; its quantization's minimum and step; and its
    /// texture coordinates' minimum and step.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct MeshRecord as ["VirtualMeshRecord", "Mesh" in VIRTUAL_SHADER] {
        pub bases: [u32; 4],
        pub info: [u32; 4],
        pub position_min: [f32; 4],
        pub position_step: [f32; 4],
        pub uv: [f32; 4],
    }

    /// A view selection culls for: its inward planes (the camera's frustum,
    /// or where a shadow map's casters can darken what the camera sees;
    /// unused ones hold everything); the eye and the smallest distance
    /// errors project from; cot(fovy / 2) / 2, the largest projected error
    /// drawn, the normal cone test and, for shadow views, the error accepted
    /// outside the camera's frustum; and the direction it looks along.
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    pub(crate) struct ViewRecord as ["View" in VIRTUAL_SHADER] {
        pub planes: [[f32; 4]; VIRTUAL_SHADOW_PLANES],
        pub eye: [f32; 4],
        pub lod: [f32; 4],
        pub direction: [f32; 4],
    }

    /// Selection's uniform: the node queue and group list capacities, bins
    /// and views; the projections this frame and the last one rasterized;
    /// the viewport (width, height, depth pyramid levels); the camera's
    /// tests, its first phase-two bin and the retest list's capacity; the
    /// counter words where used and requested groups' bits start and the
    /// groups buffer's word where their residency bits start; and the views.
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
