//! Upload structs of `crate::scatter`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `Frame` of `scatter_cull.wgsl`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct FrameUniform as ["Frame" in SCATTER_CULL_SHADER] {
        pub eye: [f32; 4],
        pub counts: [u32; 4],
        pub strides: [u32; 4],
        pub fade: [f32; 4],
    }

    /// WGSL `View`: a view's frustum planes, the radius objects need in it,
    /// the error its texels hide and whether it is the kept shadow map's.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ViewData as ["View" in SCATTER_CULL_SHADER] {
        pub planes: [[f32; 4]; 6],
        pub limits: [f32; 4],
    }

    /// WGSL `Record`: an object's bounding sphere, draw group and scale, the
    /// terrain word (see [`TERRAIN_INDEX`]), how far its base sinks below the
    /// ground, and how the camera shows it ([`STATE_FRESH`]).
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

    /// WGSL `TerrainInfo` of `scatter_snap.wgsl`: a terrain's origin and
    /// level-0 spacing, and where its levels start in the directory and how
    /// many there are.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TerrainInfo as ["TerrainInfo"] {
        pub origin: [f32; 4],
        pub levels: [u32; 4],
    }

    /// WGSL `Group`: where its lists start, its objects, its levels, those
    /// the camera draws, the coarser ones' errors, and its impostor's error
    /// and record.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct GroupData as ["Group" in SCATTER_CULL_SHADER] {
        pub ids: u32,
        pub count: u32,
        pub levels: u32,
        pub camera: u32,
        pub errors: [f32; 4],
        pub impostor: f32,
        pub record: u32,
        pub pad: [u32; 2],
    }

    /// An indexed indirect draw's arguments.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct DrawArgs as ["DrawArgs" in SCATTER_CULL_SHADER] {
        pub index_count: u32,
        pub instance_count: u32,
        pub first_index: u32,
        pub base_vertex: i32,
        pub first_instance: u32,
    }
}
