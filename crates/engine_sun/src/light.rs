use bevy_ecs::prelude::Resource;
use engine_settings::{Expected, ExtraRules, InvalidSetting, settings};
use glam::Vec3;
use serde::{Deserialize, Serialize};

use engine_core::EnvironmentSection;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SunValues {
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
}

impl LightPreset {
    pub fn sun(self) -> SunValues {
        let (sun_direction, sun_color) = match self {
            Self::Day => (Vec3::new(0.55, 0.75, 0.35), Vec3::new(2.3, 2.1, 1.75)),
            Self::GoldenHour => (Vec3::new(0.82, 0.45, 0.1), Vec3::new(2.6, 1.6, 0.85)),
            Self::Dusk => (Vec3::new(0.9, 0.07, 0.4), Vec3::new(2.2, 0.85, 0.5)),
            Self::Night => (Vec3::new(-0.35, 0.6, -0.45), Vec3::new(0.35, 0.45, 0.85)),
        };
        SunValues {
            sun_direction,
            sun_color,
        }
    }
}

fn vec3_or(authored: Option<[f32; 3]>, preset: Vec3) -> Vec3 {
    authored.map_or(preset, Vec3::from_array)
}

settings! {
    extra_rules;
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneSun")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Sunlight {
        pub preset: LightPreset = LightPreset::GoldenHour => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub sun_direction: Option<[f32; 3]> = None => finite,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub sun_color: Option<[f32; 3]> = None => at_least(0.0),
    }
}

impl ExtraRules for Sunlight {
    fn check_extra(&self) -> Result<(), InvalidSetting> {
        match self.sun_direction {
            Some(direction) if Vec3::from_array(direction).length_squared() == 0.0 => {
                Err(InvalidSetting::new("sun_direction", Expected::Nonzero).got(direction))
            }
            Some(_) | None => Ok(()),
        }
    }

    fn sanitize_extra(self) -> Self {
        match self.check_extra() {
            Ok(()) => self,
            Err(_) => Self {
                sun_direction: None,
                ..self
            },
        }
    }
}

impl Sunlight {
    pub fn values(&self) -> SunValues {
        let preset = self.preset.sun();
        SunValues {
            sun_direction: vec3_or(self.sun_direction, preset.sun_direction),
            sun_color: vec3_or(self.sun_color, preset.sun_color),
        }
    }
}

light_sections! {
    Sunlight: "sun",
}
