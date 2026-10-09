//! Upload structs of `crate::decals`.

use bytemuck::{Pod, Zeroable};
use engine_core::PoseRows;

gpu_structs! {
    /// A decal as `meshes/decals.wgsl` reads it.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct DecalData as ["DecalRecord"] {
        /// The box's world matrix, unless `pose[0]` names its palette entry.
        pub model: PoseRows,
        pub color: [f32; 4],
        /// Fade cosine, roughness, metalness and emission.
        pub params: [f32; 4],
        pub pose: [u32; 4],
    }
}
