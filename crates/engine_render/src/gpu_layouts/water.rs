use bytemuck::{Pod, Zeroable};
use engine_core::{MAX_OCEAN_WAVES, SIM_TOUCHES};

use super::constants::SIM_SLOTS;

gpu_structs! {
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

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimFrame as ["SimFrame" in WATER_SIM_SHADER] {
        pub stage: [u32; 4],
        pub inputs: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimSlot as ["SimSlot" in WATER_SIM_SHADER] {
        pub shape: [u32; 4],
        pub step: [f32; 4],
        pub blocks: [u32; 4],
        pub place: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimSlots as ["SimSlots" in WATER_SIM_SHADER] {
        pub slot: [SimSlot; SIM_SLOTS],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimTouch as ["SimTouch" in WATER_SIM_SHADER] {
        pub center: [f32; 4],
        pub inverse: [[f32; 4]; 3],
        pub velocity: [f32; 4],
        pub reach: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimTouches as ["SimTouches" in WATER_SIM_SHADER] {
        pub count: [u32; 4],
        pub blocks: [u32; 4],
        pub touch: [SimTouch; SIM_TOUCHES],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct SimSetup as ["SimSetup"] {
        pub counts: [u32; 4],
        pub square: [u32; 4],
        pub meters: [f32; 4],
        pub ring: [u32; 4],
        pub tiles: [u32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct ParticleFrame as ["ParticleFrame"] {
        pub window: [i32; 4],
        pub lattice: [i32; 4],
        pub step: [f32; 4],
        pub flags: [u32; 4],
        pub pool: [u32; 4],
        pub eddies: [f32; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
    pub(crate) struct FallVertex {
        pub position: [f32; 3],
        pub normal: [f32; 3],
        pub sheet: [f32; 4],
        pub state: [f32; 2],
        pub sky: f32,
    }
}
