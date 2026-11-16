use super::*;

settings! {
    #[section("terrain")]
    #[switch(on)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneTerrainSettings")]
    pub struct TerrainSettings {
        pub pixel_error: f32 = 6.0 => range(1.0, 64.0),
        pub pinned_levels: u32 = 2 => range(1, 8),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tiles: Option<u32> = None => budget,
        pub collision_radius: f32 = 192.0 => range(0.0, 4096.0),
        pub surroundings: bool = true => any,
        pub cache_bytes: u64 = 256 << 20 => any,
        pub gpu_materials: bool = true => any,
        pub tile_feed: u32 = 4096 => at_least(1),
    }
}
