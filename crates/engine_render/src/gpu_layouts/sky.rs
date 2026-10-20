//! Upload structs of `crate::sky`.

use bytemuck::{Pod, Zeroable};

gpu_structs! {
    /// WGSL `SkyUniform` (`sky_common.wgsl`).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SkyUniform as ["SkyUniform"] {
        pub rayleigh: [f32; 4],
        pub mie: [f32; 4],
        pub ozone: [f32; 4],
        pub ground: [f32; 4],
        pub radii: [f32; 4],
        pub sun: [f32; 4],
        pub frame: [f32; 4],
        /// The direction toward the moon, and its light: its brightness
        /// times the share of its face the sun lights (0 without one).
        pub moon: [f32; 4],
    }
}
