//! Upload structs of `crate::impostors`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// An impostor as `meshes/impostors.wgsl` reads it.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub(crate) struct ImpostorRecord as ["ImpostorRecord"] {
        /// Mesh-space center and radius of the sphere every frame frames.
        pub sphere: [f32; 4],
        /// Mesh-space half extents of the mesh's bounds.
        pub extent: [f32; 3],
        pub layer: u32,
        /// The material's emissive color (glTF), linear.
        pub emissive: [f32; 3],
        /// Texels across a frame, `HEMISPHERE`, `MASKED`, `ROOMS`, the
        /// frames along each side of the grid from `IMPOSTOR_FRAMES_SHIFT`,
        /// and `FOLIAGE`.
        pub flags: u32,
    }

    /// The impostor being baked, as `meshes/impostors.wgsl` reads it.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ImpostorBakeUniform as ["ImpostorBake"] {
        pub sphere: [f32; 4],
        /// The record's flags; the rest unused.
        pub flags: [u32; 4],
        /// The frames a pass draws: the grid's column and row of the first,
        /// those along each of its rows, and the cells along each side of
        /// the pass's target.
        pub tile: [u32; 4],
    }
}
