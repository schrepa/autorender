use bevy_ecs::prelude::Resource;
use engine_settings::settings;
use glam::Vec3;
use serde::{Deserialize, Serialize};

use engine_core::EnvironmentSection;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyLightingValues {
    pub zenith: Vec3,
    pub horizon: Vec3,
    pub ground: Vec3,
    pub ambient: f32,
    pub fog_start: f32,
    pub fog_density: f32,
    pub fog_falloff: f32,
    pub fog_max: f32,
    pub aerial_start: f32,
    pub aerial_density: f32,
    pub aerial_falloff: f32,
    pub aerial_color: Vec3,
    pub aerial_glow: f32,
}

impl LightPreset {
    pub fn sky_light(self) -> SkyLightingValues {
        match self {
            Self::Day => SkyLightingValues {
                zenith: Vec3::new(0.1, 0.3, 0.9),
                horizon: Vec3::new(0.55, 0.75, 1.0),
                ground: Vec3::new(0.24, 0.21, 0.18),
                ambient: 0.6,
                fog_start: 80.0,
                fog_density: 0.0011,
                fog_falloff: 0.02,
                fog_max: 0.4,
                aerial_start: 1000.0,
                aerial_density: 1.0e-4,
                aerial_falloff: 5.5e-4,
                aerial_color: Vec3::new(0.56, 0.72, 1.0),
                aerial_glow: 0.25,
            },
            Self::GoldenHour => SkyLightingValues {
                zenith: Vec3::new(0.05, 0.16, 0.62),
                horizon: Vec3::new(1.15, 0.56, 0.32),
                ground: Vec3::new(0.2, 0.13, 0.12),
                ambient: 0.62,
                fog_start: 80.0,
                fog_density: 0.0014,
                fog_falloff: 0.022,
                fog_max: 0.45,
                aerial_start: 1000.0,
                aerial_density: 1.1e-4,
                aerial_falloff: 6.0e-4,
                aerial_color: Vec3::new(0.36, 0.48, 0.92),
                aerial_glow: 0.7,
            },
            Self::Dusk => SkyLightingValues {
                zenith: Vec3::new(0.03, 0.04, 0.32),
                horizon: Vec3::new(1.0, 0.36, 0.42),
                ground: Vec3::new(0.08, 0.05, 0.09),
                ambient: 0.65,
                fog_start: 60.0,
                fog_density: 0.0016,
                fog_falloff: 0.02,
                fog_max: 0.45,
                aerial_start: 800.0,
                aerial_density: 1.4e-4,
                aerial_falloff: 6.0e-4,
                aerial_color: Vec3::new(0.3, 0.3, 0.66),
                aerial_glow: 0.9,
            },
            Self::Night => SkyLightingValues {
                zenith: Vec3::new(0.008, 0.012, 0.045),
                horizon: Vec3::new(0.07, 0.09, 0.22),
                ground: Vec3::new(0.02, 0.02, 0.04),
                ambient: 0.9,
                fog_start: 40.0,
                fog_density: 0.0022,
                fog_falloff: 0.02,
                fog_max: 0.7,
                aerial_start: 600.0,
                aerial_density: 1.6e-4,
                aerial_falloff: 6.0e-4,
                aerial_color: Vec3::new(0.05, 0.07, 0.16),
                aerial_glow: 0.15,
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
        schemars(rename = "SceneSkyLight")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct SkyLighting {
        pub preset: LightPreset = LightPreset::GoldenHour => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub zenith: Option<[f32; 3]> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub horizon: Option<[f32; 3]> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub ground: Option<[f32; 3]> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub ambient: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub fog_start: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub fog_density: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub fog_falloff: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub fog_max: Option<f32> = None => range(0.0, 1.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub aerial_start: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub aerial_density: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub aerial_falloff: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub aerial_color: Option<[f32; 3]> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub aerial_glow: Option<f32> = None => at_least(0.0),
    }
}

impl SkyLighting {
    pub fn values(&self) -> SkyLightingValues {
        let preset = self.preset.sky_light();
        SkyLightingValues {
            zenith: vec3_or(self.zenith, preset.zenith),
            horizon: vec3_or(self.horizon, preset.horizon),
            ground: vec3_or(self.ground, preset.ground),
            ambient: self.ambient.unwrap_or(preset.ambient),
            fog_start: self.fog_start.unwrap_or(preset.fog_start),
            fog_density: self.fog_density.unwrap_or(preset.fog_density),
            fog_falloff: self.fog_falloff.unwrap_or(preset.fog_falloff),
            fog_max: self.fog_max.unwrap_or(preset.fog_max),
            aerial_start: self.aerial_start.unwrap_or(preset.aerial_start),
            aerial_density: self.aerial_density.unwrap_or(preset.aerial_density),
            aerial_falloff: self.aerial_falloff.unwrap_or(preset.aerial_falloff),
            aerial_color: vec3_or(self.aerial_color, preset.aerial_color),
            aerial_glow: self.aerial_glow.unwrap_or(preset.aerial_glow),
        }
    }
}

light_sections! {
    SkyLighting: "sky_light",
}
