use engine_core::{Cut, settings, stats};

use crate::constants::SOFT_GATHERS;

settings! {
    #[section("shadows")]
    #[switch(on)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneShadows")]
    pub struct ShadowSettings {
        pub levels: u32 = 10 => at_least(1),
        pub finest_texel: f32 = 0.04 => range(0.005, 1.0),
        pub pixel_error: f32 = 2.0 => range(0.5, 8.0),
        pub moving_levels: u32 = 4 => any,
        pub sample_texels: f32 = 2.0 => at_least(0.0),
        pub soft_texels: f32 = 3.0 => range(2.0, 2 * SOFT_GATHERS - 1),
        pub changes: u32 = 1024 => at_least(1),
    }
}

settings! {
    #[section("sky_occlusion")]
    #[switch(on)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSkyOcclusion")]
    pub struct SkyOcclusion {
        pub strength: f32 = 1.0 => range(0.0, 1.0),
        pub floor: f32 = 0.0 => range(0.0, 1.0),
        pub near_distance: f32 = 48.0 => range(8.0, 256.0),
        pub far_distance: f32 = 400.0 => range(64.0, 4000.0),
        pub directions: u32 = 13 => range(5.0, 13.0),
        pub fade_seconds: f32 = 0.2 => at_least(0.0),
    }
}

impl SkyOcclusion {
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
        pub draws: u32,
        pub levels: u32,
        pub side: u32,
        pub bytes: u64,
        pub regions: Cut,
        pub redraws: u32,
        pub moving: u32,
        pub sky_maps: u32,
        pub sky_drawn: u32,
        pub level_cut: Cut,
    }
}
