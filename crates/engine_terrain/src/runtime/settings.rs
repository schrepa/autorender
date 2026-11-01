use super::*;

settings! {
    #[section("terrain")]
    #[check(nested_rules)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneTerrainSettings")]
    pub struct TerrainSettings {
        pub enabled: bool = true => any,
        pub pixel_error: f32 = 6.0 => range(1.0, 64.0),
        pub pinned_levels: u32 = 2 => range(1, 8),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tiles: Option<u32> = None => budget,
        pub samples_per_frame: u32 = 24_576 => at_least(1),
        pub loading_work_ms: f32 = 50.0 => range(0.1, 1000.0),
        pub work_ms: f32 = 4.0 => range(0.1, 1000.0),
        pub collision_radius: f32 = 192.0 => range(0.0, 4096.0),
        pub surroundings: bool = true => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub motion_detail: Option<MotionDetail> = None => any,
        pub cache_bytes: u64 = 256 << 20 => any,
        pub upload_bytes: u32 = 8 << 20 => at_least(1),
        pub gpu_materials: bool = true => any,
    }
}

fn nested_rules(settings: &TerrainSettings) -> Result<(), InvalidSetting> {
    Setting::check(&settings.motion_detail, "motion_detail", Rule::Any)
}

fields! {
    extra_rules;
    #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
    #[schemars(rename = "SceneTerrainMotionDetail")]
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
            Err(InvalidSetting::new("fastest", Expected::Above))
        }
    }

    fn sanitize_extra(self) -> Self {
        Self {
            fastest: engine_math::max(self.fastest, self.speed),
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
            engine_math::clamp((speed - self.speed) / (self.fastest - self.speed), 0.0, 1.0)
        } else {
            1.0
        };
        ((share * self.levels as f32).ceil() as u32).clamp(1, self.levels)
    }
}
