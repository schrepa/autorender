//! Upload structs of `crate::lights`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// The light clusters' uniform of `lights.wgsl` and `meshes/declarations.wgsl`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ClusterUniform as ["Clusters"] {
        pub view_projection: [[f32; 4]; 4],
        /// Depth along the view of point p: dot(xyz, p) + w (clip w).
        pub depth: [f32; 4],
        /// x and y projection scales, clustered depth range start and end.
        pub projection: [f32; 4],
        /// Tile width and height in pixels; slice = log2(depth) * z + w.
        pub grid: [f32; 4],
        /// Tiles across and down, depth slices, and the per-cluster budget
        /// ([`NO_BUDGET`](crate::lights::NO_BUDGET) without one).
        pub counts: [u32; 4],
        /// Lights, an unused word, the entries the cluster lists' storage holds,
        /// and 1 when only lights without a slot or on still slots are binned
        /// (reflection probe faces).
        pub lights: [u32; 4],
    }

    /// One extracted light, as `lights.wgsl` bins it.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct LightInput as ["Input" in LIGHT_SHADER] {
        pub position: [f32; 3],
        pub range: f32,
        pub color: [f32; 3],
        pub slot: u32,
        pub direction: [f32; 3],
        pub cos_inner: f32,
        pub cos_outer: f32,
        pub pad: [f32; 3],
    }
}
