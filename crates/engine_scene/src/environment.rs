//! A scene's environment (decision 022): its atmosphere (decision 018) and
//! rendering effects (decisions 020 and 021) as authored data. Version 5
//! documents with an `environment` apply it when instantiated; development
//! controls apply the same sections temporarily, and inspection reports the
//! current resources in the same shape.

use engine_core::{
    AmbientOcclusion, Atmosphere, Bloom, Shadows, bevy_ecs::prelude::World, glam::Vec3,
};
use serde::{Deserialize, Serialize};

use crate::SceneError;

/// A starting point for an atmosphere; fields set beside it override it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum AtmospherePreset {
    Day,
    #[default]
    GoldenHour,
    Dusk,
    Night,
}

impl AtmospherePreset {
    pub fn atmosphere(self) -> Atmosphere {
        match self {
            Self::Day => Atmosphere::day(),
            Self::GoldenHour => Atmosphere::golden_hour(),
            Self::Dusk => Atmosphere::dusk(),
            Self::Night => Atmosphere::night(),
        }
    }
}

/// Sun, sky, fog and grade (decision 018): a preset and the fields that
/// differ from it. Colors are linear RGB and may exceed 1.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneAtmosphere {
    pub preset: AtmospherePreset,
    /// Direction toward the sun; any nonzero length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sun_direction: Option<[f32; 3]>,
    /// Sun color times intensity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sun_color: Option<[f32; 3]>,
    /// Sky straight up, at the horizon, and below it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zenith: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub horizon: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ground: Option<[f32; 3]>,
    /// Scale of the sky's ambient light.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ambient: Option<f32>,
    /// Fog begins this far from the camera (world units).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fog_start: Option<f32>,
    /// Fog density per world unit at height zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fog_density: Option<f32>,
    /// How fast fog thins with height, per world unit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fog_falloff: Option<f32>,
    /// The most fog can cover.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fog_max: Option<f32>,
    /// Silhouette light in the horizon color.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rim: Option<f32>,
    /// Scale of sun highlights and sky reflections; 1 is physically based.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specular: Option<f32>,
    /// Scale of material emission: dim by day, full at night.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emission: Option<f32>,
    /// Exposure before tone mapping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure: Option<f32>,
    /// Saturation after tone mapping; 1 keeps it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saturation: Option<f32>,
}

impl From<&Atmosphere> for SceneAtmosphere {
    /// Every field set: resolving it reproduces `atmosphere` exactly.
    fn from(atmosphere: &Atmosphere) -> Self {
        Self {
            preset: AtmospherePreset::GoldenHour,
            sun_direction: Some(atmosphere.sun_direction.to_array()),
            sun_color: Some(atmosphere.sun_color.to_array()),
            zenith: Some(atmosphere.zenith.to_array()),
            horizon: Some(atmosphere.horizon.to_array()),
            ground: Some(atmosphere.ground.to_array()),
            ambient: Some(atmosphere.ambient),
            fog_start: Some(atmosphere.fog_start),
            fog_density: Some(atmosphere.fog_density),
            fog_falloff: Some(atmosphere.fog_falloff),
            fog_max: Some(atmosphere.fog_max),
            rim: Some(atmosphere.rim),
            specular: Some(atmosphere.specular),
            emission: Some(atmosphere.emission),
            exposure: Some(atmosphere.exposure),
            saturation: Some(atmosphere.saturation),
        }
    }
}

impl SceneAtmosphere {
    /// The atmosphere at authored path `path`: its preset with overrides.
    /// Values must be finite and nonnegative, the fog maximum at most 1 and
    /// the sun direction nonzero.
    pub fn resolve(&self, path: &str) -> Result<Atmosphere, SceneError> {
        let mut atmosphere = self.preset.atmosphere();
        let invalid = |field: &str, rule: &str| {
            SceneError::new(
                "invalid_environment",
                format!("{path}.{field}"),
                rule.to_owned(),
            )
        };
        let colors = [
            ("sun_color", self.sun_color, &mut atmosphere.sun_color),
            ("zenith", self.zenith, &mut atmosphere.zenith),
            ("horizon", self.horizon, &mut atmosphere.horizon),
            ("ground", self.ground, &mut atmosphere.ground),
        ];
        for (field, value, target) in colors {
            if let Some(value) = value {
                let value = Vec3::from_array(value);
                if !value.is_finite() || value.min_element() < 0.0 {
                    return Err(invalid(
                        field,
                        "color channels must be finite and nonnegative",
                    ));
                }
                *target = value;
            }
        }
        if let Some(direction) = self.sun_direction {
            let direction = Vec3::from_array(direction);
            if !direction.is_finite() || direction.length_squared() == 0.0 {
                return Err(invalid("sun_direction", "must be finite and nonzero"));
            }
            atmosphere.sun_direction = direction;
        }
        let scalars = [
            ("ambient", self.ambient, &mut atmosphere.ambient),
            ("fog_start", self.fog_start, &mut atmosphere.fog_start),
            ("fog_density", self.fog_density, &mut atmosphere.fog_density),
            ("fog_falloff", self.fog_falloff, &mut atmosphere.fog_falloff),
            ("fog_max", self.fog_max, &mut atmosphere.fog_max),
            ("rim", self.rim, &mut atmosphere.rim),
            ("specular", self.specular, &mut atmosphere.specular),
            ("emission", self.emission, &mut atmosphere.emission),
            ("exposure", self.exposure, &mut atmosphere.exposure),
            ("saturation", self.saturation, &mut atmosphere.saturation),
        ];
        for (field, value, target) in scalars {
            if let Some(value) = value {
                if !value.is_finite() || value < 0.0 {
                    return Err(invalid(field, "must be finite and nonnegative"));
                }
                *target = value;
            }
        }
        if atmosphere.fog_max > 1.0 {
            return Err(invalid("fog_max", "must be at most 1"));
        }
        Ok(atmosphere)
    }
}

/// A world box (minimum and maximum corners).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SceneBounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// Sun shadows (decision 020); omitted fields take the engine defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneShadows {
    pub enabled: bool,
    /// Region to keep shadows for: the whole scene gives the sharpest, stable
    /// shadows. Without it the maps follow the camera.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounds: Option<SceneBounds>,
    /// Shadows fade out this far from the camera (world units).
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001)))]
    pub distance: f32,
    /// View distance of the sharper map that follows the camera.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001)))]
    pub near_distance: f32,
    /// Texels per side of the far maps: 1024, 2048 or 4096.
    #[cfg_attr(feature = "schema", schemars(range(min = 1024, max = 4096)))]
    pub resolution: u32,
}

impl Default for SceneShadows {
    fn default() -> Self {
        Self::from(&Shadows::default())
    }
}

impl From<&Shadows> for SceneShadows {
    fn from(shadows: &Shadows) -> Self {
        Self {
            enabled: shadows.enabled,
            bounds: shadows.bounds.map(|(min, max)| SceneBounds {
                min: min.to_array(),
                max: max.to_array(),
            }),
            distance: shadows.distance,
            near_distance: shadows.near_distance,
            resolution: shadows.resolution,
        }
    }
}

impl SceneShadows {
    pub fn resolve(&self, path: &str) -> Result<Shadows, SceneError> {
        let invalid = |field: &str, rule: &str| {
            SceneError::new(
                "invalid_environment",
                format!("{path}.{field}"),
                rule.to_owned(),
            )
        };
        if !self.distance.is_finite() || self.distance <= 0.0 {
            return Err(invalid("distance", "must be finite and positive"));
        }
        if !self.near_distance.is_finite()
            || self.near_distance <= 0.0
            || self.near_distance >= self.distance
        {
            return Err(invalid(
                "near_distance",
                "must be finite, positive and less than distance",
            ));
        }
        if ![1024, 2048, 4096].contains(&self.resolution) {
            return Err(invalid("resolution", "must be 1024, 2048 or 4096"));
        }
        let bounds = match self.bounds {
            Some(bounds) => {
                let (min, max) = (Vec3::from_array(bounds.min), Vec3::from_array(bounds.max));
                if !min.is_finite() || !max.is_finite() || !min.cmple(max).all() {
                    return Err(invalid(
                        "bounds",
                        "corners must be finite with min at most max on every axis",
                    ));
                }
                Some((min, max))
            }
            None => None,
        };
        Ok(Shadows {
            enabled: self.enabled,
            bounds,
            distance: self.distance,
            near_distance: self.near_distance,
            resolution: self.resolution,
        })
    }
}

/// Ambient occlusion (decision 021); an intensity of 0 turns it off.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneAmbientOcclusion {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub intensity: f32,
    /// World distance within which geometry occludes.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001)))]
    pub radius: f32,
}

impl Default for SceneAmbientOcclusion {
    fn default() -> Self {
        Self::from(&AmbientOcclusion::default())
    }
}

impl From<&AmbientOcclusion> for SceneAmbientOcclusion {
    fn from(occlusion: &AmbientOcclusion) -> Self {
        Self {
            intensity: occlusion.intensity,
            radius: occlusion.radius,
        }
    }
}

impl SceneAmbientOcclusion {
    pub fn resolve(&self, path: &str) -> Result<AmbientOcclusion, SceneError> {
        if !(0.0..=1.0).contains(&self.intensity) {
            return Err(SceneError::new(
                "invalid_environment",
                format!("{path}.intensity"),
                "must be in [0, 1]",
            ));
        }
        if !self.radius.is_finite() || self.radius <= 0.0 {
            return Err(SceneError::new(
                "invalid_environment",
                format!("{path}.radius"),
                "must be finite and positive",
            ));
        }
        Ok(AmbientOcclusion {
            intensity: self.intensity,
            radius: self.radius,
        })
    }
}

/// Bloom (decision 021); an intensity of 0 turns it off.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneBloom {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 4.0)))]
    pub intensity: f32,
    /// Exposed brightness above which light glows.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0)))]
    pub threshold: f32,
}

impl Default for SceneBloom {
    fn default() -> Self {
        Self::from(&Bloom::default())
    }
}

impl From<&Bloom> for SceneBloom {
    fn from(bloom: &Bloom) -> Self {
        Self {
            intensity: bloom.intensity,
            threshold: bloom.threshold,
        }
    }
}

impl SceneBloom {
    pub fn resolve(&self, path: &str) -> Result<Bloom, SceneError> {
        if !(0.0..=4.0).contains(&self.intensity) {
            return Err(SceneError::new(
                "invalid_environment",
                format!("{path}.intensity"),
                "must be in [0, 4]",
            ));
        }
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            return Err(SceneError::new(
                "invalid_environment",
                format!("{path}.threshold"),
                "must be finite and nonnegative",
            ));
        }
        Ok(Bloom {
            intensity: self.intensity,
            threshold: self.threshold,
        })
    }
}

/// A scene's look (decision 022). In a document, every omitted section takes
/// the engine default; in a development control, omitted sections stay as
/// they are.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneEnvironment {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atmosphere: Option<SceneAtmosphere>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadows: Option<SceneShadows>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ambient_occlusion: Option<SceneAmbientOcclusion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bloom: Option<SceneBloom>,
}

/// Resolved sections of an environment; `None` where a section is omitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentSettings {
    pub atmosphere: Option<Atmosphere>,
    pub shadows: Option<Shadows>,
    pub ambient_occlusion: Option<AmbientOcclusion>,
    pub bloom: Option<Bloom>,
}

impl SceneEnvironment {
    /// Every section, from `world`'s current resources: resolving it
    /// reproduces them. Absent effect resources read as off.
    pub fn current(world: &World) -> Self {
        let atmosphere = world
            .get_resource::<Atmosphere>()
            .copied()
            .unwrap_or_default();
        let shadows = world.get_resource::<Shadows>().copied().unwrap_or(Shadows {
            enabled: false,
            ..Shadows::default()
        });
        let occlusion =
            world
                .get_resource::<AmbientOcclusion>()
                .copied()
                .unwrap_or(AmbientOcclusion {
                    intensity: 0.0,
                    ..AmbientOcclusion::default()
                });
        let bloom = world.get_resource::<Bloom>().copied().unwrap_or(Bloom {
            intensity: 0.0,
            ..Bloom::default()
        });
        Self {
            atmosphere: Some(SceneAtmosphere::from(&atmosphere)),
            shadows: Some(SceneShadows::from(&shadows)),
            ambient_occlusion: Some(SceneAmbientOcclusion::from(&occlusion)),
            bloom: Some(SceneBloom::from(&bloom)),
        }
    }

    /// The sections at authored path `path`, validated; omitted ones are
    /// `None`.
    pub fn resolve(&self, path: &str) -> Result<EnvironmentSettings, SceneError> {
        Ok(EnvironmentSettings {
            atmosphere: self
                .atmosphere
                .map(|section| section.resolve(&format!("{path}.atmosphere")))
                .transpose()?,
            shadows: self
                .shadows
                .map(|section| section.resolve(&format!("{path}.shadows")))
                .transpose()?,
            ambient_occlusion: self
                .ambient_occlusion
                .map(|section| section.resolve(&format!("{path}.ambient_occlusion")))
                .transpose()?,
            bloom: self
                .bloom
                .map(|section| section.resolve(&format!("{path}.bloom")))
                .transpose()?,
        })
    }
}

impl EnvironmentSettings {
    /// Inserts every resolved section into `world`, leaving the others.
    pub fn apply_sections(&self, world: &mut World) {
        if let Some(atmosphere) = self.atmosphere {
            world.insert_resource(atmosphere);
        }
        if let Some(shadows) = self.shadows {
            world.insert_resource(shadows);
        }
        if let Some(occlusion) = self.ambient_occlusion {
            world.insert_resource(occlusion);
        }
        if let Some(bloom) = self.bloom {
            world.insert_resource(bloom);
        }
    }

    /// Inserts all four resources into `world`, defaults for omitted
    /// sections: a document's environment owns the whole look.
    pub fn apply_all(&self, world: &mut World) {
        world.insert_resource(self.atmosphere.unwrap_or_default());
        world.insert_resource(self.shadows.unwrap_or_default());
        world.insert_resource(self.ambient_occlusion.unwrap_or_default());
        world.insert_resource(self.bloom.unwrap_or_default());
    }
}
