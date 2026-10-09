//! Upload structs of `crate::water`.

use bytemuck::{Pod, Zeroable};
use engine_core::MAX_OCEAN_WAVES;

gpu_structs! {
    /// WGSL `WaterUniform` of `meshes/water.wgsl`; the sea's parts are zero
    /// for terrains' water, and the terrain parts for the sea's.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct WaterUniform as ["WaterUniform"] {
        pub relative_view_projection: [[f32; 4]; 4],
        pub origin: [f32; 4],
        pub grid: [u32; 4],
        pub color: [f32; 4],
        pub ripples: [f32; 4],
        pub foam: [f32; 4],
        pub ocean: [f32; 4],
        pub ocean_grid: [f32; 4],
        pub ocean_waves: [[f32; 4]; MAX_OCEAN_WAVES],
        pub ocean_reach: [[f32; 4]; MAX_OCEAN_WAVES / 4],
    }

    /// WGSL `WaterChunk`'s instance part: its first sample and the samples
    /// between its vertices.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ChunkRecord {
        pub first: [u32; 2],
        pub stride: u32,
    }

    /// A point of a river's ribbon: terrain-local, and the water's depth over
    /// the bed there, its flow along x and z and half the river's width
    /// (`meshes/water.wgsl`'s `RiverVertex`).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct RiverVertex {
        pub position: [f32; 3],
        pub water: [f32; 4],
    }
}
