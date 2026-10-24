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

    /// WGSL `View`: a view's frustum planes, the radius objects need in it
    /// and the error its texels hide.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ViewData as ["View" in SCATTER_CULL_SHADER] {
        pub planes: [[f32; 4]; 6],
        pub limits: [f32; 4],
    }

    /// WGSL `Record`: an object's bounding sphere, draw group and scale, the
    /// terrain word (see [`TERRAIN_INDEX`](super::constants::TERRAIN_INDEX)),
    /// how far its base sinks below the ground, and how the camera shows it
    /// ([`STATE_FRESH`](crate::scatter::STATE_FRESH)).
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
    /// level-0 spacing; where its levels start in the directory, how many
    /// there are, where its materials start in the palette and its macro
    /// colors' layer plus one (0 until they arrive); and the meters from
    /// the eye up to which each of its first eight levels draws.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct TerrainInfo as ["TerrainInfo"] {
        pub origin: [f32; 4],
        pub levels: [u32; 4],
        pub ranges: [[f32; 4]; 2],
    }

    /// WGSL `Snap` of `scatter_snap.wgsl`: the slots and terrains, and the
    /// eye with the meters a pixel spans a meter from it.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SnapUniform as ["Snap"] {
        pub counts: [u32; 4],
        pub eye: [f32; 4],
    }

    /// WGSL `Group`: where its lists start, its objects, its levels, those
    /// the camera draws, the coarser ones' errors, its impostor's error
    /// and record, the levels shadow maps draw its objects at (its shadow
    /// proxy alone counts one; 0 casting nothing), and the groups after it
    /// drawing its objects' parts.
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
