engine_core::gpu_consts! {
    pub const NONE: u32 = u32::MAX;
    pub const SURFACE_PATH_FAR: f32 = 1.0e9;
    pub const SURFACE_RIVER_FAR: f32 = 1.0e9;
    pub const SURFACE_RIVER_EDGE_STEP: f32 = 0.0625;
    pub const SURFACE_RIVER_EDGE_ZERO: u32 = 32768;
    pub const SURFACE_RIVER_EDGE_FAR: u32 = 65535;
}

const _: () = assert!(SURFACE_PATH_FAR == engine_heightfield::path::PATH_FAR);
const _: () = assert!(SURFACE_RIVER_FAR == engine_heightfield::RIVER_FAR);
const _: () = assert!(SURFACE_RIVER_EDGE_STEP == engine_heightfield::tile::RIVER_EDGE_STEP);
const _: () = assert!(SURFACE_RIVER_EDGE_ZERO == engine_heightfield::tile::RIVER_EDGE_ZERO as u32);
const _: () = assert!(SURFACE_RIVER_EDGE_FAR == engine_heightfield::tile::RIVER_EDGE_FAR as u32);
