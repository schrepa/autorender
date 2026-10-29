use super::*;

pub const MAX_BOUNCE_LEVELS: u32 = 8;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTerrainSettings")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct TerrainSettings {
        pub enabled: bool = true => any,
        pub subdivision_error: f32 = 2.0 => range(0.25, 64.0),
        pub subdivision_depth: u32 = 27 => range(10, 29),
        pub subdivision_leaves: u32 = 131_072 => range(1024, 4_000_000),
        pub pixel_error: f32 = 6.0 => range(1.0, 64.0),
        pub pinned_levels: u32 = 2 => range(1, 8),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tiles: Option<u32> = None => budget,
        pub samples_per_frame: u32 = 24_576 => at_least(1),
        pub loading_work_ms: f32 = 50.0 => range(0.1, 1000.0),
        pub work_ms: f32 = 4.0 => range(0.1, 1000.0),
        pub collision_radius: f32 = 192.0 => range(0.0, 4096.0),
        pub sun_shadows: bool = true => any,
        pub shadow_resolution: u32 = 512 => range(64, 4096),
        pub shadow_softness: f32 = 0.01 => range(0.0, 0.2),
        pub sky_visibility: bool = true => any,
        pub sky_resolution: u32 = 1024 => range(64, 4096),
        pub bounce: bool = true => any,
        pub bounce_spacing: f32 = 16.0 => range(1.0, 1024.0),
        pub bounce_columns: u32 = 64 => multiple(2, 8, 1024),
        pub bounce_levels: u32 = MAX_BOUNCE_LEVELS => range(1, MAX_BOUNCE_LEVELS),
        pub macro_resolution: u32 = 2048 => range(64, 4096),
        pub relief_steps: u32 = 8 => range(2, 64),
        pub surroundings: bool = true => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub motion_detail: Option<MotionDetail> = None => any,
        pub cache_bytes: u64 = 256 << 20 => any,
        pub upload_bytes: u32 = 8 << 20 => at_least(1),
        pub gpu_materials: bool = true => any,
    }
}

settings! {
    extra_rules;
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTerrainMotionDetail")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct MotionDetail {
        pub speed: f32 = 30.0 => range(0.0, 100_000.0),
        pub fastest: f32 = 300.0 => range(0.0, 100_000.0),
        pub levels: u32 = 2 => range(0, 8),
        pub settle: f32 = 0.2 => range(0.0, 10.0),
    }
}

impl ExtraRules for MotionDetail {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        if self.fastest >= self.speed {
            Ok(())
        } else {
            Err(InvalidSetting::new("fastest", engine_settings::Expected::Above))
        }
    }

    fn sanitize_extra(self) -> Self {
        Self {
            fastest: engine_core::math::max(self.fastest, self.speed),
            ..self
        }
    }
}

impl MotionDetail {
    pub fn waiting(&self, speed: f32) -> u32 {
        if speed <= self.speed || speed.is_nan() || self.levels == 0 {
            return 0;
        }
        let share = if self.fastest > self.speed {
            engine_core::math::clamp((speed - self.speed) / (self.fastest - self.speed), 0.0, 1.0)
        } else {
            1.0
        };
        ((share * self.levels as f32).ceil() as u32).clamp(1, self.levels)
    }
}

impl TerrainSettings {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
