use bevy_ecs::prelude::Resource;
use engine_settings::settings;
use glam::Vec3;
use serde::{Deserialize, Serialize};

use engine_core::EnvironmentSection;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradingValues {
    pub exposure: f32,
    pub saturation: f32,
    pub contrast: f32,
    pub vibrance: f32,
    pub shadow_tint: Vec3,
    pub highlight_tint: Vec3,
}

impl LightPreset {
    pub fn grading(self) -> GradingValues {
        match self {
            Self::Day => GradingValues {
                exposure: 0.85,
                saturation: 1.2,
                contrast: 1.12,
                vibrance: 0.35,
                shadow_tint: Vec3::new(0.85, 0.95, 1.2),
                highlight_tint: Vec3::new(1.08, 1.02, 0.92),
            },
            Self::GoldenHour => GradingValues {
                exposure: 0.75,
                saturation: 1.2,
                contrast: 1.15,
                vibrance: 0.4,
                shadow_tint: Vec3::new(0.75, 0.9, 1.35),
                highlight_tint: Vec3::new(1.15, 1.0, 0.85),
            },
            Self::Dusk => GradingValues {
                exposure: 0.95,
                saturation: 1.15,
                contrast: 1.18,
                vibrance: 0.35,
                shadow_tint: Vec3::new(0.75, 0.8, 1.45),
                highlight_tint: Vec3::new(1.25, 0.95, 0.75),
            },
            Self::Night => GradingValues {
                exposure: 1.35,
                saturation: 1.2,
                contrast: 1.2,
                vibrance: 0.5,
                shadow_tint: Vec3::new(0.8, 0.9, 1.35),
                highlight_tint: Vec3::new(1.15, 1.0, 0.85),
            },
        }
    }
}

fn vec3_or(authored: Option<[f32; 3]>, preset: Vec3) -> Vec3 {
    authored.map_or(preset, Vec3::from_array)
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneGrading")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Grading {
        pub preset: LightPreset = LightPreset::GoldenHour => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub exposure: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub saturation: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub contrast: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub vibrance: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub shadow_tint: Option<[f32; 3]> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub highlight_tint: Option<[f32; 3]> = None => at_least(0.0),
    }
}

impl Grading {
    pub fn values(&self) -> GradingValues {
        let preset = self.preset.grading();
        GradingValues {
            exposure: self.exposure.unwrap_or(preset.exposure),
            saturation: self.saturation.unwrap_or(preset.saturation),
            contrast: self.contrast.unwrap_or(preset.contrast),
            vibrance: self.vibrance.unwrap_or(preset.vibrance),
            shadow_tint: vec3_or(self.shadow_tint, preset.shadow_tint),
            highlight_tint: vec3_or(self.highlight_tint, preset.highlight_tint),
        }
    }
}

light_sections! {
    Grading: "grading",
}
