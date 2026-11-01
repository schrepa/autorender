use engine_core::gpu_consts;

gpu_consts! {
    pub const MAX_SHADOW_LEVELS: u32 = 20;
    pub const MAX_MOVING_TILES: u32 = 256;
    pub const SHADOW_REGIONS: u32 = 7;
    pub const SHADOW_APRON: u32 = 1;
    pub const SKY_DIRECTIONS: u32 = 13;
    pub const SKY_LAYERS: u32 = 2 * SKY_DIRECTIONS;
    pub const SKY_REFRESH: u32 = 2;
    pub const MARK_GROUP: u32 = 8;
    pub const KEPT_GROUND_TEXELS: f32 = 2.0;
    pub const DEBUG_SUN_NONE: u32 = 0;
    pub const DEBUG_SUN_LEVEL: u32 = 1;
    pub const DEBUG_SUN_HORIZON: u32 = DEBUG_SUN_LEVEL + MAX_SHADOW_LEVELS;
    pub const DEBUG_SUN_BEYOND: u32 = DEBUG_SUN_HORIZON + 1;
    pub const DEBUG_SHADOWED: u32 = 0x80;
}

const _: () = assert!(
    KEPT_GROUND_TEXELS == engine_places::shadows::GROUND_TEXELS,
    "kept_ground_texels_is_the_ground_place_value"
);
