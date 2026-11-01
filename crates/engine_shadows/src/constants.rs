use engine_render::gpu_layouts::scene::MAX_LEVELS;

gpu_consts! {
    pub const DEBUG_SUN_NONE: u32 = 0 => ["DEBUG_SUN_NONE"];
    pub const DEBUG_SUN_LEVEL: u32 = 1 => ["DEBUG_SUN_LEVEL"];
    pub const DEBUG_SUN_HORIZON: u32 = DEBUG_SUN_LEVEL + MAX_LEVELS as u32 => ["DEBUG_SUN_HORIZON"];
    pub const DEBUG_SUN_BEYOND: u32 = DEBUG_SUN_HORIZON + 1 => ["DEBUG_SUN_BEYOND"];
    pub const DEBUG_SHADOWED: u32 = 0x80 => ["DEBUG_SHADOWED"];
}
