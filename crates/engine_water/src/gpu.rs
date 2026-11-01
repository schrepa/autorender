use bytemuck::{Pod, Zeroable};
use engine_core::{gpu_consts, gpu_struct};
use engine_heightfield::tile::{PATCH_QUADS, PATCHES_PER_TILE};

gpu_consts! {
    pub const NO_WATER_SURFACE: f32 = -1.0e30;
    pub const NO_WATER_BLOCK: u32 = 0xffffffff;
    pub const WATER_FLOW_RANGE: f32 = 8.0;
    pub const WATER_SHEAR_RANGE: f32 = 8.0;
    pub const WATER_GRADIENT_RANGE: f32 = 4.0;
    pub const WATER_SPAN_LEAST: f32 = 0.25;
    pub const WATER_SPAN_MOST: f32 = 4096.0;
    pub const WATER_SPAN_STEPS: f32 = 126.0;
    pub const BLOCK_SIDE: u32 = 8;
    pub const BLOCK_GROUND: u32 = (BLOCK_SIDE + 1) * (BLOCK_SIDE + 1) + 4 * BLOCK_SIDE * BLOCK_SIDE;
    pub const NO_BLOCK: u32 = 0xffffffff;
    pub const COVER_POINTS: u32 = 4;
    pub const WATER_SIM_GROUP: u32 = BLOCK_SIDE;
    pub const WATER_LIVE_BLOCK: u32 = BLOCK_SIDE;
    pub const SIM_SLOTS: u32 = 64;
    pub const SIM_CELL_SUMS: u32 = 4;
    pub const SIM_BLOCK_LOOKS: u32 = 3;
    pub const SIM_LIVE_LOOKS: u32 = 2;
    pub const SIM_STEPPING: u32 = 1;
    pub const SIM_LOCAL: u32 = 2;
    pub const SIM_OPEN: u32 = 4;
    pub const SIM_SUMMING: u32 = 8;
    pub const SIM_LOOKING: u32 = 16;
    pub const SIM_SPARSE: u32 = 32;
    pub const SIM_LIVE: u32 = 64;
    pub const SIM_BURSTING: u32 = 128;
    pub const SIM_KEEPING: u32 = 256;
    pub const SIM_RIM: u32 = BLOCK_SIDE;
    pub const SIM_TOUCHES: u32 = 32;
    pub const WATER_SIM_SOURCE_GROUP: u32 = 64;
    pub const WATER_SETUP_RING_GROUP: u32 = 64;
    pub const SETUP_SOLID_ENTRIES: u32 = 4;
    pub const SETUP_WHOLE: u32 = 0;
    pub const SETUP_RISEN: u32 = 1;
    pub const SETUP_STANDS: u32 = 2;
    pub const SIM_STANDING: u32 = 0x80000000;
    pub const LIVE_NONE: u32 = 0xffffffff;
    pub const LIVE_HEAD: u32 = 4;
    pub const LIVE_CELL: u32 = 3;
    pub const LIVE_HELD: f32 = 1.0e30;
    pub const PARTICLE_GROUP: u32 = 256;
    pub const PARTICLE_SIDE: u32 = 8;
    pub const PARTICLE_LATTICE: u32 = 8;
    pub const PARTICLE_MARGIN: u32 = 2;
    pub const PARTICLE_FOAM_LIVE: u32 = 0;
    pub const PARTICLE_WAVES_LIVE: u32 = 1;
    pub const PARTICLE_FOAM_BORN: u32 = 2;
    pub const PARTICLE_WAVES_BORN: u32 = 3;
    pub const PARTICLE_FOAM_DROPPED: u32 = 4;
    pub const PARTICLE_WAVES_DROPPED: u32 = 5;
    pub const PARTICLE_FOAM_BEFORE: u32 = 6;
    pub const PARTICLE_WAVES_BEFORE: u32 = 7;
    pub const PARTICLE_ARGS: u32 = 8;
    pub const PARTICLE_DRAW: u32 = 14;
    pub const PARTICLE_ITEMS: u32 = 18;
    pub const SURFACE_BEND: f32 = 7.0;
    pub const SURFACE_BEND_MOST: f32 = 1.0;
    pub const SURFACE_WIND_DRIFT: f32 = 0.03;
    pub const THETA: f32 = 1.3;
    pub const THIN: f32 = 0.01;
    pub const DRY: f32 = 1.0e-4;
    pub const WET_MOST: f32 = 0.5;
    pub const FASTEST: f32 = 20.0;
    pub const UNHELD: f32 = -1.0e30;
    pub const DENSITY: f32 = 1000.0;
    pub const FERGUSON_A1: f32 = 6.5;
    pub const FERGUSON_A2: f32 = 2.5;
    pub const SHALLOWEST: f32 = 0.1;
    pub const DEEPEST: f32 = 30.0;
    pub const STRICKLER: f32 = 20.4;
    pub const GROUND_D84: f32 = 0.05;
    pub const COURANT: f32 = 0.24;
    pub const LOCAL_SHARE: f32 = 0.9;
    pub const SLOWEST: f32 = 1.0;
    pub const FOAM_LIFE: f32 = 20.0;
    pub const FOAM_WET: f32 = 0.005;
    pub const FOAM_LOSS: f32 = 0.02;
    pub const FOAM_POWER: f32 = 500.0;
    pub const SOURCE_SPEED: f32 = 1.5;
    pub const CELL_MEANS: u32 = 3;
    pub const TURBULENT_SPEED: f32 = 0.5;
    pub const TURBULENT_SWIRL: f32 = 4.0;
    pub const BREAK_ONSET: f32 = 0.6;
    pub const BREAK_FULL: f32 = 1.2;
    pub const BREAK_THINNEST: f32 = 0.02;
    pub const WATER_BLOCK: u32 = PATCH_QUADS + 3;
    pub const WATER_BLOCKS_SIDE: u32 = 16;
    pub const WATER_TILE_PATCHES: u32 = PATCHES_PER_TILE;
    pub const WATER_TILE_WORDS: u32 = WATER_TILE_PATCHES * WATER_TILE_PATCHES;
    pub const WATER_LEAF_FIRST: u32 = 5;
    pub const DEBUG_SPAN_STEPS: f32 = 254.0;
}

gpu_struct! {
    pub(crate) struct WaterUniform {
        pub color: [f32; 4],
        pub foam: [f32; 4],
        pub absorption: [f32; 4],
        pub sim: [f32; 4],
        pub sim_datum: [f32; 4],
        pub sediment: [f32; 4],
        pub lift: [f32; 4],
        pub sim_pixels: [f32; 4],
        pub terrains: [u32; 4],
    }
}

gpu_struct! {
    #[derive(bevy_ecs::resource::Resource)]
    pub(crate) struct WaterPiece {
        pub debug: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SimFrame {
        pub stage: [u32; 4],
        pub inputs: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SimSlot {
        pub shape: [u32; 4],
        pub step: [f32; 4],
        pub blocks: [u32; 4],
        pub place: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SimTouch {
        pub center: [f32; 4],
        pub inverse: [[f32; 4]; 3],
        pub velocity: [f32; 4],
        pub reach: [f32; 4],
    }
}

gpu_struct! {
    pub(crate) struct SimSetup {
        pub counts: [u32; 4],
        pub square: [u32; 4],
        pub meters: [f32; 4],
        pub ring: [u32; 4],
        pub tiles: [u32; 4],
        pub terrain: [u32; 4],
    }
}

gpu_struct! {
    pub(crate) struct ParticleFrame {
        pub window: [i32; 4],
        pub lattice: [i32; 4],
        pub step: [f32; 4],
        pub flags: [u32; 4],
        pub pool: [u32; 4],
        pub eddies: [f32; 4],
        pub terrain: [u32; 4],
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct SimSlots {
    pub slot: [SimSlot; SIM_SLOTS as usize],
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct SimTouches {
    pub count: [u32; 4],
    pub blocks: [u32; 4],
    pub touch: [SimTouch; SIM_TOUCHES as usize],
}

#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct FallVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub sheet: [f32; 4],
    pub state: [f32; 2],
    pub sky: f32,
    pub terrain: u32,
}

pub(crate) const GENERATED: &[&str] = &[
    WGSL,
    <WaterUniform as engine_core::GpuStruct>::WGSL,
    <SimFrame as engine_core::GpuStruct>::WGSL,
    <SimSlot as engine_core::GpuStruct>::WGSL,
    <SimTouch as engine_core::GpuStruct>::WGSL,
    <SimSetup as engine_core::GpuStruct>::WGSL,
    <ParticleFrame as engine_core::GpuStruct>::WGSL,
];
