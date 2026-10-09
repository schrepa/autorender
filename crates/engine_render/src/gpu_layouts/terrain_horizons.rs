//! Upload structs of `crate::terrain_horizons`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `Build` (`terrain_horizon.wgsl`).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct BuildUniform as ["Build" in TERRAIN_HORIZON_SHADER] {
        pub place: [f32; 4],
        pub shape: [f32; 4],
        pub origin: [f32; 4],
        pub counts: [u32; 4],
    }
}
