//! Upload structs of `crate::terrain_bounce`.

use bytemuck::{Pod, Zeroable};

/// Bins of a probe's view (`engine_core::terrain_data::bounce::BINS`).
pub(crate) const BINS: usize = engine_core::terrain_data::bounce::BINS;

gpu_structs! {
    /// WGSL `BounceFrame` (`terrain_bounce.wgsl`): each bin's moments, the
    /// integral over it of the direction (x, y and z) and its solid angle
    /// (w).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct BounceFrame as ["BounceFrame"] {
        pub bins: [[f32; 4]; BINS],
    }

    /// WGSL `BounceTerrain` (`terrain_bounce.wgsl`): one terrain's bounce
    /// light this frame. Its first corner (world x and z) and its extent
    /// (meters along x and z); its probe columns along a side, its ground's
    /// texels along a side and the first column (x and z) of the window
    /// relighting; the window's columns along a side (0: it relights none),
    /// where its blocks' words and its probes start among every terrain's,
    /// and its place among the horizon records; its ground's first layer,
    /// its light's layer, its probes' first layer among the bounce layers
    /// and its sky map's layer plus one (0 without one); and where its
    /// materials start in the palette.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct BounceTerrain as ["BounceTerrain"] {
        pub place: [f32; 4],
        pub grid: [u32; 4],
        pub window: [u32; 4],
        pub layers: [u32; 4],
        pub materials: [u32; 4],
    }
}
