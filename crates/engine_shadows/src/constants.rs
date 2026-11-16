use engine_core::gpu_consts;

gpu_consts! {
    pub const LEVEL_SNAP: u32 = 64;
    pub const LEVEL_MARGIN: f32 = 1.0;
    pub const SOFT_GATHERS: u32 = 4;
    pub const SOFT_READS: u32 = SOFT_GATHERS * SOFT_GATHERS;
    pub const SKY_DIRECTIONS: u32 = 13;
    pub const SKY_LAYERS: u32 = 2 * SKY_DIRECTIONS;
    pub const DEBUG_SHOWN_SHADOWS: u32 = 1;
    pub const DEBUG_SHOWN_CASTERS: u32 = 2;
}
