use engine_core::gpu_consts;
use engine_places::shadows::{DEBUG_SUN_HORIZON, DEBUG_SUN_LEVEL, GROUND_TEXELS};

gpu_consts! {
    pub const MAX_SHADOW_LEVELS: u32 = 20;
    pub const MAX_MOVING_TILES: u32 = 256;
    pub const SHADOW_APRON: u32 = 1;
    pub const SKY_DIRECTIONS: u32 = 13;
    pub const SKY_LAYERS: u32 = 2 * SKY_DIRECTIONS;
    pub const SKY_REFRESH: u32 = 2;
    pub const MARK_GROUP: u32 = 8;
    pub const KEPT_GROUND_TEXELS: f32 = 2.0;
    pub const DEBUG_SHOWN_SHADOWS: u32 = 1;
    pub const DEBUG_SHOWN_CASTERS: u32 = 2;
}

const _: () = assert!(
    KEPT_GROUND_TEXELS == GROUND_TEXELS,
    "kept_ground_texels_is_the_ground_place_value"
);

const _: () = assert!(
    DEBUG_SUN_HORIZON == DEBUG_SUN_LEVEL + MAX_SHADOW_LEVELS,
    "the_sun_horizon_code_follows_every_level"
);
