use bevy_ecs::prelude::Resource;
use engine_settings::{Setting, settings};
use serde::{Deserialize, Serialize};

use engine_core::EnvironmentSection;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum LightPreset {
    Day,
    #[default]
    GoldenHour,
    Dusk,
    Night,
}

impl Setting for LightPreset {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceLightValues {
    pub rim: f32,
    pub specular: f32,
    pub emission: f32,
}

impl LightPreset {
    pub fn surface_light(self) -> SurfaceLightValues {
        let (rim, emission) = match self {
            Self::Day => (0.15, 0.1),
            Self::GoldenHour => (0.45, 0.5),
            Self::Dusk => (0.6, 1.0),
            Self::Night => (0.8, 1.0),
        };
        SurfaceLightValues {
            rim,
            specular: 1.0,
            emission,
        }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneSurfaceLight")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct SurfaceLight {
        pub preset: LightPreset = LightPreset::GoldenHour => any,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub rim: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub specular: Option<f32> = None => at_least(0.0),
        #[serde(skip_serializing_if = "Option::is_none")]
        pub emission: Option<f32> = None => at_least(0.0),
    }
}

impl SurfaceLight {
    pub fn values(&self) -> SurfaceLightValues {
        let preset = self.preset.surface_light();
        SurfaceLightValues {
            rim: self.rim.unwrap_or(preset.rim),
            specular: self.specular.unwrap_or(preset.specular),
            emission: self.emission.unwrap_or(preset.emission),
        }
    }
}

light_sections! {
    SurfaceLight: "surface_light",
}
