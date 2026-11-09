use engine_core::{gpu_bits, gpu_consts};
use engine_heightfield::tile::PATCH_QUADS;

gpu_consts! {
    pub const NO_WATER_SURFACE: f32 = -1.0e30;
    pub const WATER_FLOW_RANGE: f32 = 8.0;
    pub const WATER_SHEAR_RANGE: f32 = 8.0;
    pub const WATER_SPAN_LEAST: f32 = 0.25;
    pub const WATER_SPAN_MOST: f32 = 4096.0;
    pub const WATER_SPAN_STEPS: f32 = 126.0;
    pub const BLOCK_SIDE: u32 = 8;
    pub const BLOCK_GROUND: u32 = (BLOCK_SIDE + 1) * (BLOCK_SIDE + 1) + 4 * BLOCK_SIDE * BLOCK_SIDE;
    pub const NO_BLOCK: u32 = 0xffffffff;
    pub const COVER_POINTS: u32 = 4;
    pub const WATER_SIM_GROUP: u32 = BLOCK_SIDE;
    pub const WATER_LIVE_BLOCK: u32 = BLOCK_SIDE;
    pub const SIM_RIM: u32 = BLOCK_SIDE;
    pub const LIVE_HELD: f32 = 1.0e30;
    pub const THETA: f32 = 1.3;
    pub const THIN: f32 = 0.01;
    pub const DRY: f32 = 1.0e-4;
    pub const WET_MOST: f32 = 0.5;
    pub const UNHELD: f32 = -1.0e30;
    pub const DENSITY: f32 = 1000.0;
    pub const FERGUSON_A1: f32 = 6.5;
    pub const FERGUSON_A2: f32 = 2.5;
    pub const SHALLOWEST: f32 = 0.1;
    pub const DEEPEST: f32 = 30.0;
    pub const STRICKLER: f32 = 20.4;
    pub const GROUND_D84: f32 = 0.05;
    pub const COURANT: f32 = 0.24;
    pub const CELL_MEANS: u32 = 3;
    pub const TURBULENT_SPEED: f32 = 0.5;
    pub const TURBULENT_SWIRL: f32 = 4.0;
    pub const BREAK_ONSET: f32 = 0.6;
    pub const BREAK_FULL: f32 = 1.2;
    pub const BREAK_THINNEST: f32 = 0.02;
    pub const WATER_BLOCK: u32 = 19;
}

const _: () = assert!(
    WATER_BLOCK == PATCH_QUADS + 3,
    "WATER_BLOCK is not PATCH_QUADS + 3"
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

#[gpu_bits]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockPlace {
    #[bits(16)]
    pub x: u32,
    #[bits(16)]
    pub z: u32,
}
