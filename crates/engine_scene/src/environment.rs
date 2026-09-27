//! A scene's environment (decision 022): its atmosphere (decision 018) and
//! rendering effects (decisions 020, 021, 023 to 026) as authored data.
//! Version 5 documents with an `environment` apply it when instantiated
//! (local lights, particles, sun shafts and reflections need version 6);
//! development controls apply the same sections temporarily, and inspection
//! reports the current resources in the same shape.

use engine_core::{
    AmbientOcclusion, Atmosphere, Bloom, LocalLights, MAX_LOCAL_LIGHTS, MAX_PARTICLES,
    MAX_SHADOW_CASCADES, Particles, Reflections, Shadows, SunShafts, bevy_ecs::prelude::World,
    glam::Vec3,
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
    /// Contrast around mid grey after tone mapping: 1 keeps it, and the
    /// curve, which keeps black and white, is steepest at 1.5. Version 6
    /// (decision 027), like the grade fields below.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contrast: Option<f32>,
    /// Saturation added to muted colors more than to vivid ones; 0 adds none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vibrance: Option<f32>,
    /// Hue of dark tones after tone mapping (split toning), as a color gain
    /// scaled to keep luminance; white or black keeps them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_tint: Option<[f32; 3]>,
    /// Hue of bright tones, like `shadow_tint`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highlight_tint: Option<[f32; 3]>,
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
            contrast: Some(atmosphere.contrast),
            vibrance: Some(atmosphere.vibrance),
            shadow_tint: Some(atmosphere.shadow_tint.to_array()),
            highlight_tint: Some(atmosphere.highlight_tint.to_array()),
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
            ("shadow_tint", self.shadow_tint, &mut atmosphere.shadow_tint),
            (
                "highlight_tint",
                self.highlight_tint,
                &mut atmosphere.highlight_tint,
            ),
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
            ("contrast", self.contrast, &mut atmosphere.contrast),
            ("vibrance", self.vibrance, &mut atmosphere.vibrance),
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
    /// View distance of the sharper maps that follow the camera.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001)))]
    pub near_distance: f32,
    /// Sharper maps (cascades) splitting `near_distance`, each reaching a
    /// third as far as the next: 1 to 4, 3 when omitted. Version 6
    /// (decision 028).
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 4)))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cascades: Option<u32>,
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
            cascades: Some(shadows.cascades),
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
        let cascades = self.cascades.unwrap_or(Shadows::default().cascades);
        if !(1..=MAX_SHADOW_CASCADES).contains(&cascades) {
            return Err(invalid(
                "cascades",
                &format!("must be 1 to {MAX_SHADOW_CASCADES}"),
            ));
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
            cascades,
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

/// Local lighting settings (decision 023). Requires version 6.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneLocalLights {
    pub enabled: bool,
    /// Most lights a frame shades; beyond it the nearest to the camera win.
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 4096)))]
    pub max_lights: u32,
    /// Scales every light's intensity, e.g. to dim lamps by day.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100.0)))]
    pub intensity: f32,
    /// View distance beyond which local light fades out: distant lights
    /// cost no shading.
    #[cfg_attr(feature = "schema", schemars(range(min = 1.0, max = 100000.0)))]
    pub distance: f32,
}

impl Default for SceneLocalLights {
    fn default() -> Self {
        Self::from(&LocalLights::default())
    }
}

impl From<&LocalLights> for SceneLocalLights {
    fn from(lights: &LocalLights) -> Self {
        Self {
            enabled: lights.enabled,
            max_lights: lights.max_lights,
            intensity: lights.intensity,
            distance: lights.distance,
        }
    }
}

impl SceneLocalLights {
    pub fn resolve(&self, path: &str) -> Result<LocalLights, SceneError> {
        if !(1..=MAX_LOCAL_LIGHTS).contains(&self.max_lights) {
            return Err(section_error(path, "max_lights", "must be in [1, 4096]"));
        }
        if !(self.intensity.is_finite() && (0.0..=100.0).contains(&self.intensity)) {
            return Err(section_error(path, "intensity", "must be in [0, 100]"));
        }
        if !(self.distance.is_finite() && (1.0..=100_000.0).contains(&self.distance)) {
            return Err(section_error(path, "distance", "must be in [1, 100000]"));
        }
        Ok(LocalLights {
            enabled: self.enabled,
            max_lights: self.max_lights,
            intensity: self.intensity,
            distance: self.distance,
        })
    }
}

/// Particle settings (decision 026). Requires version 6.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneParticles {
    pub enabled: bool,
    /// Most particles a frame draws; emitters beyond it are skipped.
    #[cfg_attr(feature = "schema", schemars(range(min = 1, max = 262144)))]
    pub max_particles: u32,
}

impl Default for SceneParticles {
    fn default() -> Self {
        Self::from(&Particles::default())
    }
}

impl From<&Particles> for SceneParticles {
    fn from(particles: &Particles) -> Self {
        Self {
            enabled: particles.enabled,
            max_particles: particles.max_particles,
        }
    }
}

impl SceneParticles {
    pub fn resolve(&self, path: &str) -> Result<Particles, SceneError> {
        if !(1..=MAX_PARTICLES).contains(&self.max_particles) {
            return Err(section_error(
                path,
                "max_particles",
                "must be in [1, 262144]",
            ));
        }
        Ok(Particles {
            enabled: self.enabled,
            max_particles: self.max_particles,
        })
    }
}

/// Sun shafts (decision 024): sunlight scattered by the fog between shadow
/// casters. Requires version 6 and shadows; an intensity of 0 turns them off.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneSunShafts {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 4.0)))]
    pub intensity: f32,
    /// Forward scattering: higher concentrates light toward the sun.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 0.95)))]
    pub anisotropy: f32,
    /// World distance along each view ray light is gathered over.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001, max = 100000.0)))]
    pub distance: f32,
}

impl Default for SceneSunShafts {
    fn default() -> Self {
        Self::from(&SunShafts::default())
    }
}

impl From<&SunShafts> for SceneSunShafts {
    fn from(shafts: &SunShafts) -> Self {
        Self {
            intensity: shafts.intensity,
            anisotropy: shafts.anisotropy,
            distance: shafts.distance,
        }
    }
}

impl SceneSunShafts {
    pub fn resolve(&self, path: &str) -> Result<SunShafts, SceneError> {
        if !(0.0..=4.0).contains(&self.intensity) {
            return Err(section_error(path, "intensity", "must be in [0, 4]"));
        }
        if !(0.0..=0.95).contains(&self.anisotropy) {
            return Err(section_error(path, "anisotropy", "must be in [0, 0.95]"));
        }
        if !(self.distance.is_finite() && self.distance > 0.0 && self.distance <= 100_000.0) {
            return Err(section_error(
                path,
                "distance",
                "must be positive and at most 100000",
            ));
        }
        Ok(SunShafts {
            intensity: self.intensity,
            anisotropy: self.anisotropy,
            distance: self.distance,
        })
    }
}

/// Screen-space reflections (decision 025). Requires version 6; an intensity
/// of 0 turns them off.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneReflections {
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub intensity: f32,
    /// Surfaces rougher than this reflect only the sky.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.05, max = 1.0)))]
    pub max_roughness: f32,
    /// World distance reflection rays travel.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.001, max = 100000.0)))]
    pub distance: f32,
}

impl Default for SceneReflections {
    fn default() -> Self {
        Self::from(&Reflections::default())
    }
}

impl From<&Reflections> for SceneReflections {
    fn from(reflections: &Reflections) -> Self {
        Self {
            intensity: reflections.intensity,
            max_roughness: reflections.max_roughness,
            distance: reflections.distance,
        }
    }
}

impl SceneReflections {
    pub fn resolve(&self, path: &str) -> Result<Reflections, SceneError> {
        if !(0.0..=1.0).contains(&self.intensity) {
            return Err(section_error(path, "intensity", "must be in [0, 1]"));
        }
        if !(0.05..=1.0).contains(&self.max_roughness) {
            return Err(section_error(path, "max_roughness", "must be in [0.05, 1]"));
        }
        if !(self.distance.is_finite() && self.distance > 0.0 && self.distance <= 100_000.0) {
            return Err(section_error(
                path,
                "distance",
                "must be positive and at most 100000",
            ));
        }
        Ok(Reflections {
            intensity: self.intensity,
            max_roughness: self.max_roughness,
            distance: self.distance,
        })
    }
}

fn section_error(path: &str, field: &str, rule: &str) -> SceneError {
    SceneError::new(
        "invalid_environment",
        format!("{path}.{field}"),
        rule.to_owned(),
    )
}

/// A scene's look (decision 022). In a document, every omitted section takes
/// the engine's default state (sun shafts and reflections: off); in a
/// development control, omitted sections stay as they are.
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
    /// Version 6 sections (decisions 023 to 026).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_lights: Option<SceneLocalLights>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sun_shafts: Option<SceneSunShafts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reflections: Option<SceneReflections>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub particles: Option<SceneParticles>,
}

/// Resolved sections of an environment; `None` where a section is omitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentSettings {
    pub atmosphere: Option<Atmosphere>,
    pub shadows: Option<Shadows>,
    pub ambient_occlusion: Option<AmbientOcclusion>,
    pub bloom: Option<Bloom>,
    pub local_lights: Option<LocalLights>,
    pub sun_shafts: Option<SunShafts>,
    pub reflections: Option<Reflections>,
    pub particles: Option<Particles>,
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
        let local_lights = world
            .get_resource::<LocalLights>()
            .copied()
            .unwrap_or(LocalLights {
                enabled: false,
                ..LocalLights::default()
            });
        let particles = world
            .get_resource::<Particles>()
            .copied()
            .unwrap_or(Particles {
                enabled: false,
                ..Particles::default()
            });
        let shafts = world
            .get_resource::<SunShafts>()
            .copied()
            .unwrap_or(SunShafts {
                intensity: 0.0,
                ..SunShafts::default()
            });
        let reflections = world
            .get_resource::<Reflections>()
            .copied()
            .unwrap_or(Reflections {
                intensity: 0.0,
                ..Reflections::default()
            });
        Self {
            atmosphere: Some(SceneAtmosphere::from(&atmosphere)),
            shadows: Some(SceneShadows::from(&shadows)),
            ambient_occlusion: Some(SceneAmbientOcclusion::from(&occlusion)),
            bloom: Some(SceneBloom::from(&bloom)),
            local_lights: Some(SceneLocalLights::from(&local_lights)),
            sun_shafts: Some(SceneSunShafts::from(&shafts)),
            reflections: Some(SceneReflections::from(&reflections)),
            particles: Some(SceneParticles::from(&particles)),
        }
    }

    /// The first version 6 field present, as its path below the environment:
    /// the grade of decision 027, shadow cascades (decision 028) and the
    /// sections of decisions 023 to 026.
    pub(crate) fn version_six_field(&self) -> Option<&'static str> {
        let atmosphere = self.atmosphere.unwrap_or_default();
        let cascades = self
            .shadows
            .is_some_and(|shadows| shadows.cascades.is_some());
        [
            ("atmosphere.contrast", atmosphere.contrast.is_some()),
            ("atmosphere.vibrance", atmosphere.vibrance.is_some()),
            ("atmosphere.shadow_tint", atmosphere.shadow_tint.is_some()),
            (
                "atmosphere.highlight_tint",
                atmosphere.highlight_tint.is_some(),
            ),
            ("shadows.cascades", cascades),
            ("local_lights", self.local_lights.is_some()),
            ("sun_shafts", self.sun_shafts.is_some()),
            ("reflections", self.reflections.is_some()),
            ("particles", self.particles.is_some()),
        ]
        .into_iter()
        .find_map(|(name, present)| present.then_some(name))
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
            local_lights: self
                .local_lights
                .map(|section| section.resolve(&format!("{path}.local_lights")))
                .transpose()?,
            sun_shafts: self
                .sun_shafts
                .map(|section| section.resolve(&format!("{path}.sun_shafts")))
                .transpose()?,
            reflections: self
                .reflections
                .map(|section| section.resolve(&format!("{path}.reflections")))
                .transpose()?,
            particles: self
                .particles
                .map(|section| section.resolve(&format!("{path}.particles")))
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
        if let Some(lights) = self.local_lights {
            world.insert_resource(lights);
        }
        if let Some(shafts) = self.sun_shafts {
            world.insert_resource(shafts);
        }
        if let Some(reflections) = self.reflections {
            world.insert_resource(reflections);
        }
        if let Some(particles) = self.particles {
            world.insert_resource(particles);
        }
    }

    /// Sets every section in `world`, omitted ones to the engine's default
    /// state: a document's environment owns the whole look. Sun shafts and
    /// reflections are opt-in, so omitting them removes them.
    pub fn apply_all(&self, world: &mut World) {
        world.insert_resource(self.atmosphere.unwrap_or_default());
        world.insert_resource(self.shadows.unwrap_or_default());
        world.insert_resource(self.ambient_occlusion.unwrap_or_default());
        world.insert_resource(self.bloom.unwrap_or_default());
        world.insert_resource(self.local_lights.unwrap_or_default());
        world.insert_resource(self.particles.unwrap_or_default());
        match self.sun_shafts {
            Some(shafts) => world.insert_resource(shafts),
            None => {
                world.remove_resource::<SunShafts>();
            }
        }
        match self.reflections {
            Some(reflections) => world.insert_resource(reflections),
            None => {
                world.remove_resource::<Reflections>();
            }
        }
    }
}
