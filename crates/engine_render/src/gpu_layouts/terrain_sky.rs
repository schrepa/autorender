//! Upload structs of `crate::terrain_sky`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `Build` (`terrain_sky.wgsl`).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct BuildUniform as ["Build" in TERRAIN_SKY_SHADER] {
        pub shape: [f32; 4],
        pub counts: [u32; 4],
        pub rows: [u32; 4],
    }
}
