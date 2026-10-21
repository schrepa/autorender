//! Upload structs of `crate::water`.

use bytemuck::{Pod, Zeroable};
use engine_core::MAX_OCEAN_WAVES;

use super::constants::{SIM_SLOTS, WAVE_TILES};

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
        pub sim: [f32; 4],
    }

    /// WGSL `SimFrame` of `water_sim.wgsl`: the stage, the slots the
    /// buffers hold, the blocks a slot's pool holds and the blocks a
    /// slot's table maps; the cells the square around the camera moved by;
    /// and the most cells rivers bring water to a slot holds and the faces
    /// along its west and south edges a slot holds.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimFrame as ["SimFrame" in WATER_SIM_SHADER] {
        pub stage: [u32; 4],
        pub shift: [i32; 4],
        pub inputs: [u32; 4],
    }

    /// WGSL `SimSlot` of `water_sim.wgsl`: a slot's rectangle, its cells
    /// along x and z, flags and cells rivers bring water to; the meters
    /// across a cell, the seconds of a step alike for every cell, Manning's
    /// roughness and the steps its sums hold at a look; and its blocks
    /// along x and z and those its pool stores.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimSlot as ["SimSlot" in WATER_SIM_SHADER] {
        pub shape: [u32; 4],
        pub step: [f32; 4],
        pub blocks: [u32; 4],
    }

    /// WGSL `SimSlots` of `water_sim.wgsl`: every slot's rectangle.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimSlots as ["SimSlots" in WATER_SIM_SHADER] {
        pub slot: [SimSlot; SIM_SLOTS],
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
