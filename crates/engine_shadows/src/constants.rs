use engine_core::gpu_consts;

gpu_consts! {
    pub const LEVEL_WINDOW: u32 = 0;
    pub const LEVEL_BAND: u32 = 4;
    pub const LEVEL_BEFORE: u32 = 8;
    pub const LEVEL_WORDS: u32 = 12;
    pub const SHADOW_APRON: u32 = 1;
    pub const SKY_DIRECTIONS: u32 = 13;
    pub const SKY_LAYERS: u32 = 2 * SKY_DIRECTIONS;
    pub const MARK_GROUP: u32 = 8;
    pub const DEBUG_SHOWN_SHADOWS: u32 = 1;
    pub const DEBUG_SHOWN_CASTERS: u32 = 2;
}
