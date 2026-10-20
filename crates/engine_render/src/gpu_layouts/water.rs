//! Upload structs of `crate::water`.

use bytemuck::{Pod, Zeroable};
use engine_core::MAX_OCEAN_WAVES;

use super::constants::WAVE_TILES;

gpu_structs! {
    /// WGSL `WaterUniform` of `meshes/water.wgsl`; the sea's parts are zero
    /// for terrains' water, whose projection is the terrain view's.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct WaterUniform as ["WaterUniform"] {
        pub relative_view_projection: [[f32; 4]; 4],
        pub origin: [f32; 4],
        pub color: [f32; 4],
        pub ripples: [f32; 4],
        pub foam: [f32; 4],
        pub absorption: [f32; 4],
        pub waves: [f32; 4],
        pub ocean: [f32; 4],
        pub ocean_grid: [f32; 4],
        pub ocean_waves: [[f32; 4]; MAX_OCEAN_WAVES],
        pub ocean_reach: [[f32; 4]; MAX_OCEAN_WAVES / 4],
    }

    /// WGSL `WaveFrame` of `water_waves.wgsl`.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct WaveFrame as ["WaveFrame" in WATER_WAVES_SHADER] {
        pub tiles: [[f32; 4]; WAVE_TILES],
        pub clock: [f32; 4],
    }

    /// WGSL `WaterPatchInput`'s instance part: a patch holding water as
    /// terrain's `PatchRecord` draws it (its first sample, shape, atlas
    /// layers and texels, fade), and its water's blocks: its own, its
    /// parent tile's over it (`NO_WATER_BLOCK` without) and where it starts
    /// in that block (x | z << 16).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct WaterPatchRecord {
        pub first: [i32; 2],
        pub shape: [f32; 4],
        pub atlas: [u32; 4],
        pub water: [u32; 4],
        pub fade: f32,
    }

    /// WGSL `FallInput` of `meshes/water_falls.wgsl`: a vertex of a waterfall's
    /// sheet (`crate::water_falls`): where it stands (world space), the
    /// sheet's face there, turned out from the rock it falls from, and the
    /// way its water runs and how fast (w, meters a second); where across
    /// the sheet it lies (-1 to 1), the seconds since the water passing it
    /// left the lip (negative up the river), the meters a second it left at
    /// and the sheet's half width at the lip (meters); and the seconds
    /// since it last took off where it flies, less those since it last
    /// landed down a chute, the sheet's thickness there (meters), its
    /// fall's number and 1 where its water flies; and how white its water
    /// has turned, how far the air has broken it up and how much of it shows
    /// (0 to 1 each), and 1 where it is a boil; and how much of the sky it
    /// sees (0 to 1).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct FallVertex {
        pub position: [f32; 3],
        pub normal: [f32; 3],
        pub tangent: [f32; 4],
        pub jet: [f32; 4],
        pub fall: [f32; 4],
        pub state: [f32; 4],
        pub sky: f32,
    }
}
