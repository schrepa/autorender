use engine_core::{Cut, settings, stats};

use crate::constants::{MAX_MOVING_TILES, MAX_SHADOW_LEVELS};

settings! {
    #[section("shadows")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneShadows")]
    pub struct ShadowSettings {
        pub enabled: bool = true => any,
        pub levels: u32 = 16 => range(1, MAX_SHADOW_LEVELS),
        pub finest_texel: f32 = 0.04 => range(0.005, 1.0),
        pub tile_texels: u32 = 128 => power_of_two(64, 512),
        pub atlas_slots: u32 = 1024 => range(64, 4096),
        pub redraw_budget: u32 = 32 => range(1, 4096),
        pub moving_tiles: u32 = MAX_MOVING_TILES => range(0, MAX_MOVING_TILES),
    }
}

impl ShadowSettings {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    #[section("sky_occlusion")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSkyOcclusion")]
    pub struct SkyOcclusion {
        pub enabled: bool = true => any,
        pub strength: f32 = 1.0 => range(0.0, 1.0),
        pub floor: f32 = 0.0 => range(0.0, 1.0),
        pub near_distance: f32 = 48.0 => range(8.0, 256.0),
        pub far_distance: f32 = 400.0 => range(64.0, 4000.0),
        pub directions: u32 = 13 => range(5.0, 13.0),
    }
}

impl SkyOcclusion {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub fn darkens(&self) -> bool {
        self.enabled && self.strength > 0.0
    }

    pub fn direction_count(&self) -> u32 {
        match self.directions {
            ..9 => 5,
            9..13 => 9,
            _ => 13,
        }
    }
}

stats! {
    #[section("shadow_stats")]
    pub struct ShadowStats {
        pub active: bool,
        pub tiles_marked: u32,
        pub tiles_resident: u32,
        pub tiles_drawn: u32,
        pub tiles_reclaimed: u32,
        pub sky_maps: u32,
        pub sky_drawn: u32,
        pub tile_cut: Cut,
        pub moving_cut: Cut,
    }
}
