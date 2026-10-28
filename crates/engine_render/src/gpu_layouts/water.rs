//! Upload structs of `crate::water`.

use bytemuck::{Pod, Zeroable};
use engine_core::{MAX_OCEAN_WAVES, SIM_TOUCHES};

use super::constants::SIM_SLOTS;

gpu_structs! {
    /// WGSL `WaterUniform` of `meshes/water.wgsl`; the sea's parts are zero
    /// for terrains' water, whose projection is the terrain view's and whose
    /// `foam` holds its particles' texels, textures, handover and the cells
    /// of the water they ride (`crate::water_particles`), and the sea carries
    /// no sediment, stands on no subdivision and runs nowhere live, its
    /// `lift` and live water's parts zero.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct WaterUniform as ["WaterUniform"] {
        pub relative_view_projection: [[f32; 4]; 4],
        pub origin: [f32; 4],
        pub color: [f32; 4],
        pub foam: [f32; 4],
        pub absorption: [f32; 4],
        pub ocean: [f32; 4],
        pub ocean_grid: [f32; 4],
        pub ocean_waves: [[f32; 4]; MAX_OCEAN_WAVES],
        pub ocean_reach: [[f32; 4]; MAX_OCEAN_WAVES / 4],
        pub sim: [f32; 4],
        pub sim_datum: [f32; 4],
        pub sediment: [f32; 4],
        pub lift: [f32; 4],
        pub sim_pixels: [f32; 4],
    }

    /// WGSL `SimFrame` of `water_sim.wgsl`: the stage, the slots the
    /// buffers hold, the blocks a slot's pool holds and the blocks a
    /// slot's table maps; and the most cells rivers bring water to a slot
    /// holds and the faces along its west and south edges a slot holds, then
    /// nothing.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimFrame as ["SimFrame" in WATER_SIM_SHADER] {
        pub stage: [u32; 4],
        pub inputs: [u32; 4],
    }

    /// WGSL `SimSlot` of `water_sim.wgsl`: a slot's rectangle, its cells
    /// along x and z, flags and cells rivers bring water to; the meters
    /// across a cell and the seconds of a step alike for every cell, then
    /// nothing; its blocks along x and z and those its pool stores, then
    /// nothing; and its first cell on the terrain's lattice of its cells
    /// along x and z (an `i32`'s bits each) and the index of its frame's
    /// first step, which turbulence's bursts are drawn by, then nothing.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimSlot as ["SimSlot" in WATER_SIM_SHADER] {
        pub shape: [u32; 4],
        pub step: [f32; 4],
        pub blocks: [u32; 4],
        pub place: [u32; 4],
    }

    /// WGSL `SimSlots` of `water_sim.wgsl`: every slot's rectangle.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimSlots as ["SimSlots" in WATER_SIM_SHADER] {
        pub slot: [SimSlot; SIM_SLOTS],
    }

    /// WGSL `SimTouch` of `water_sim.wgsl`: a solid standing in the live
    /// square's water, its center over the square's first corner and datum
    /// (meters) and 1 for a ball, else 0; the inverse of the axes mapping
    /// the unit cube (or ball) to it, a column each; its velocity (meters a
    /// second) and the meters it reaches up and down from its center; and
    /// its footprint's least and greatest corners over the square (meters,
    /// x and z).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimTouch as ["SimTouch" in WATER_SIM_SHADER] {
        pub center: [f32; 4],
        pub inverse: [[f32; 4]; 3],
        pub velocity: [f32; 4],
        pub reach: [f32; 4],
    }

    /// WGSL `SimTouches` of `water_sim.wgsl`: how many solids stand in the
    /// live square's water this frame, then nothing; the blocks of the
    /// square they reach, the first along x and z and how many along each;
    /// and each solid, nearest the eye first.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimTouches as ["SimTouches" in WATER_SIM_SHADER] {
        pub count: [u32; 4],
        pub blocks: [u32; 4],
        pub touch: [SimTouch; SIM_TOUCHES],
    }

    /// WGSL `SimSetup` of `meshes/water_setup.wgsl`. The live water's: the
    /// blocks of the live square a frame sets up (each with what it sets up
    /// of it), then nothing, the cells past its edges it sets up (none where
    /// it sets up none) and the solids standing in the settled water over
    /// it; its first cell on the terrain's lattice of its cells along x and
    /// z (an `i32`'s bits each), its cells along a side and the live cells
    /// along each side of a settled cell; the meters across a cell and the
    /// height its heights stand above (world meters), then nothing; where
    /// its ring starts among the simulation's inputs, then nothing; and the
    /// level-0 tiles whose records' live cells its setup holds, the first
    /// along x and z and how many along each. Settling water's: the blocks
    /// whose ground it raises, then nothing; the level-0 samples along each
    /// side of a block's samples and the terrain's last sample along x and
    /// z, then nothing; the meters across a cell and the height its heights
    /// stand above (terrain-local meters), then nothing; and nothing, twice.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimSetup as ["SimSetup"] {
        pub counts: [u32; 4],
        pub square: [u32; 4],
        pub meters: [f32; 4],
        pub ring: [u32; 4],
        pub tiles: [u32; 4],
    }

    /// WGSL `ParticleFrame` of `meshes/water_particles.wgsl`: the first
    /// texel of water's particles' window along x and z (world texels), then
    /// that of the window its pools' particles ran in and its textures were
    /// splatted over the frame before; its lattice's first point along x and
    /// z (world lattice points, `PARTICLE_LATTICE` texels apart),
    /// `PARTICLE_MARGIN` points before its square's, its points along a side
    /// and the window's texels along a side; the meters across a texel, the
    /// seconds the frame steps and the wind (meters a second along x and
    /// z); 1 where its pools hold none yet, the index of the simulation's
    /// step the frame begins in and of the one it ends in, and the least
    /// steps a slot of its events spans; the most foam and wave particles
    /// its pools hold, the half holding the frame before's and its lattice's
    /// points; and the seconds of a simulation step and the meters across
    /// the cells of the water the particles ride, then nothing.
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ParticleFrame as ["ParticleFrame"] {
        pub window: [i32; 4],
        pub lattice: [i32; 4],
        pub step: [f32; 4],
        pub flags: [u32; 4],
        pub pool: [u32; 4],
        pub eddies: [f32; 4],
    }

    /// WGSL `FallInput` of `meshes/water_falls.wgsl`: a vertex of a
    /// waterfall's jet (`crate::water_falls`): where it stands (world
    /// space) and the sheet's face there, turned out from the rock it falls
    /// from; how far across the sheet it stands from the jet's middle
    /// (meters, either way), half the width of the water there, the meters
    /// each of the water's edges has spread and the sheet's thickness
    /// (meters); how white its water has turned and how far the air has
    /// broken it up (0 to 1 each); and how much of the sky it sees (0 to 1).
    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct FallVertex {
        pub position: [f32; 3],
        pub normal: [f32; 3],
        pub sheet: [f32; 4],
        pub state: [f32; 2],
        pub sky: f32,
    }
}
