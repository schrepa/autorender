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
    /// light at one level of its probes this frame. Its first corner (world
    /// x and z) and its extent (meters along x and z); its finest level's
    /// probe columns along a side, its ground's texels along a side (0: it
    /// relights none), where its materials start in the palette and its
    /// macro colors' layer plus one (0 without them); its ground's first
    /// layer, its light's layer, its probes' first layer among the bounce
    /// layers and its sky map's layer plus one (0 without one); where its
    /// blocks' words, its probes and this level's mean probes (from the
    /// third level on) start among every terrain's, and its place among the
    /// horizon records; the level, its window's columns along a side (0: it
    /// relights none at this level) and the window's first column (x and
    /// z); the window's first texel (x and y) in the bounce layers and where
    /// its probes start in the scratch; and the finer level's window, whose
    /// probes this one averages where it holds all four under one: its first
    /// column (x and z), its columns along a side (0: none relit) and where
    /// its probes start in the scratch.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct BounceTerrain as ["BounceTerrain"] {
        pub place: [f32; 4],
        pub grid: [u32; 4],
        pub layers: [u32; 4],
        pub sources: [u32; 4],
        pub level: [u32; 4],
        pub kept: [u32; 4],
        pub finer: [u32; 4],
    }
}
