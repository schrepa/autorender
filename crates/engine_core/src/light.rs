use bevy_ecs::prelude::Resource;
use engine_settings::{Expected, ExtraRules, InvalidSetting, Setting, settings};
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::EnvironmentSection;

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
pub struct SunValues {
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
}

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceLightValues {
    pub rim: f32,
    pub specular: f32,
    pub emission: f32,
}

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

macro_rules! light_sections {
    ($($section:ident: $key:literal),* $(,)?) => {
        $(
            impl EnvironmentSection for $section {
                const KEY: &'static str = $key;
                const NAME: &'static str = stringify!($section);
                const OPT_IN: bool = false;

                fn section_off() -> Self {
                    Self::default()
                }

                fn section_check(&self) -> Result<(), InvalidSetting> {
                    self.validate()
                }

                fn section_sanitized(self) -> Self {
                    self.sanitized()
                }

                #[cfg(feature = "schema")]
                fn schema() -> schemars::Schema {
                    schemars::schema_for!($section)
                }
            }
        )*
    };
}

light_sections! {
    Sunlight: "sun",
    SkyLighting: "sky_light",
    SurfaceLight: "surface_light",
    Grading: "grading",
}
