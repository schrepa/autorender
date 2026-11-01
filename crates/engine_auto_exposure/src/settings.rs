use engine_core::{settings, stats};

settings! {
    #[section("auto_exposure")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneAutoExposure")]
    pub struct AutoExposure {
        pub enabled: bool = false => any,
        pub strength: f32 = 1.0 => range(0.0, 1.0),
        pub compensation: f32 = 0.0 => range(-8.0, 8.0),
        pub night: f32 = 1.0 => range(0.0, 16.0),
        pub brighten: f32 = 3.0 => range(0.0, 16.0),
        pub darken: f32 = 1.0 => range(0.0, 16.0),
        pub to_brighter: f32 = 0.5 => range(0.0, 60.0),
        pub to_darker: f32 = 2.0 => range(0.0, 60.0),
        pub skip_dark: f32 = 0.3 => range(0.0, 0.45),
        pub skip_bright: f32 = 0.05 => range(0.0, 0.45),
        pub center: f32 = 2.0 => range(1.0, 16.0),
        pub sky: f32 = 0.5 => range(0.0, 1.0),
    }
}

stats! {
    #[section("exposure_stats")]
    #[derive(Copy, Eq)]
    pub struct ExposureStats {
        pub active: bool,
        pub stops: i32,
        pub metered: i32,
        pub view: i32,
    }
}
