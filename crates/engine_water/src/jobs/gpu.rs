use engine_core::{gpu_bits, gpu_consts};
use engine_heightfield::tile::PATCH_QUADS;

gpu_consts! {
    pub const NO_WATER_SURFACE: f32 = -1.0e30;
    pub const WATER_FLOW_RANGE: f32 = 8.0;
    pub const WATER_SHEAR_RANGE: f32 = 8.0;
    pub const WATER_SPAN_LEAST: f32 = 0.25;
    pub const WATER_SPAN_MOST: f32 = 4096.0;
    pub const WATER_SPAN_STEPS: f32 = 126.0;
    pub const DRY: f32 = 1.0e-4;
    pub const DENSITY: f32 = 1000.0;
    pub const GRAVITY: f32 = 9.81;
    pub const COARSE_GRAVEL_D84: f32 = 0.05;
    pub const WATER_RUSH_RANGE: f32 = 0.5;
    pub const WATER_BREAK_RANGE: f32 = 2.0;
    pub const FROUDE_GUARD_DEPTH: f32 = 0.02;
    pub const WATER_BLOCK: u32 = 19;
}

const _: () = assert!(
    WATER_BLOCK == PATCH_QUADS + 3,
    "WATER_BLOCK is not PATCH_QUADS + 3"
);

const _: () = assert!(
    GRAVITY.to_bits() == engine_math::GRAVITY.to_bits(),
    "GRAVITY differs from engine_math::GRAVITY"
);

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WaterInfoWord {
    #[bits(8)]
    pub flow_x: u8,
    #[bits(8)]
    pub flow_z: u8,
    #[bits(8)]
    pub sheet: u8,
    #[bits(8)]
    pub kind: u8,
}

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WaterStateWord {
    #[bits(8)]
    pub turbulence: u8,
    #[bits(8)]
    pub breaking: u8,
    #[bits(8)]
    pub shear: u8,
    #[bits(8)]
    pub reach: u8,
}
