//! A scene's environment (decision 022): its atmosphere (decision 018) and
//! rendering effects (decisions 020, 021, 023 to 026, 038, 039, 043, 044 and
//! 047) as authored data.
//! Version 5 documents with an `environment` apply it when instantiated
//! (local lights, particles, sun shafts and reflections need version 6);
//! development controls apply the same sections temporarily, and inspection
//! reports the current resources in the same shape.

use engine_core::{
    AmbientOcclusion, Atmosphere, Bloom, Decals, LevelOfDetail, LocalLights, MAX_SHADOW_CASCADES,
    Particles, ReflectionProbes, Reflections, Refraction, Shadows, SunShafts, VirtualGeometry,
    bevy_ecs::prelude::World, glam::Vec3,
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
    /// An optional budget of lights a frame shades; beyond it the nearest to
    /// the camera win. Omitted, every light shades.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_lights: Option<u32>,
    /// An optional budget of lights one view cluster lists. Omitted, every
    /// light reaching a cluster shades there.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lights_per_cluster: Option<u32>,
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
            lights_per_cluster: lights.lights_per_cluster,
            intensity: lights.intensity,
            distance: lights.distance,
        }
    }
}

impl SceneLocalLights {
    pub fn resolve(&self, path: &str) -> Result<LocalLights, SceneError> {
        if self.max_lights == Some(0) {
            return Err(section_error(path, "max_lights", "must be at least 1"));
        }
        if self.lights_per_cluster == Some(0) {
            return Err(section_error(
                path,
                "lights_per_cluster",
                "must be at least 1",
            ));
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
            lights_per_cluster: self.lights_per_cluster,
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
    /// An optional budget of particles a frame draws; beyond it the emitters
    /// nearest the camera win. Omitted, every emitter draws.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_particles: Option<u32>,
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
        if self.max_particles == Some(0) {
            return Err(section_error(path, "max_particles", "must be at least 1"));
        }
        Ok(Particles {
            enabled: self.enabled,
            max_particles: self.max_particles,
        })
    }
}

/// Decal settings (decision 039). Requires version 8.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneDecals {
    pub enabled: bool,
    /// An optional budget of decals a frame draws; beyond it the decals
    /// nearest the camera win. Omitted, every decal draws.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_decals: Option<u32>,
}

impl Default for SceneDecals {
    fn default() -> Self {
        Self::from(&Decals::default())
    }
}

impl From<&Decals> for SceneDecals {
    fn from(decals: &Decals) -> Self {
        Self {
            enabled: decals.enabled,
            max_decals: decals.max_decals,
        }
    }
}

impl SceneDecals {
    pub fn resolve(&self, path: &str) -> Result<Decals, SceneError> {
        if self.max_decals == Some(0) {
            return Err(section_error(path, "max_decals", "must be at least 1"));
        }
        Ok(Decals {
            enabled: self.enabled,
            max_decals: self.max_decals,
        })
    }
}

/// Reflection probe settings (decision 043). Requires version 8.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneReflectionProbes {
    pub enabled: bool,
    /// Texels across each face of every environment map, the sky's
    /// included: a power of two from 16 to 1024.
    #[cfg_attr(feature = "schema", schemars(range(min = 16, max = 1024)))]
    pub resolution: u32,
    /// An optional budget of probes kept captured; beyond it the probes
    /// nearest the camera win. Omitted, as many as the GPU's texture arrays
    /// hold.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_probes: Option<u32>,
    /// Cube faces captured per frame while probes wait: more fill new places
    /// sooner, fewer keep those frames lighter.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    pub faces_per_frame: u32,
}

impl Default for SceneReflectionProbes {
    fn default() -> Self {
        Self::from(&ReflectionProbes::default())
    }
}

impl From<&ReflectionProbes> for SceneReflectionProbes {
    fn from(probes: &ReflectionProbes) -> Self {
        Self {
            enabled: probes.enabled,
            resolution: probes.resolution,
            max_probes: probes.max_probes,
            faces_per_frame: probes.faces_per_frame,
        }
    }
}

impl SceneReflectionProbes {
    pub fn resolve(&self, path: &str) -> Result<ReflectionProbes, SceneError> {
        if !(self.resolution.is_power_of_two() && (16..=1024).contains(&self.resolution)) {
            return Err(section_error(
                path,
                "resolution",
                "must be a power of two from 16 to 1024",
            ));
        }
        if self.max_probes == Some(0) {
            return Err(section_error(path, "max_probes", "must be at least 1"));
        }
        if self.faces_per_frame == 0 {
            return Err(section_error(path, "faces_per_frame", "must be at least 1"));
        }
        Ok(ReflectionProbes {
            enabled: self.enabled,
            resolution: self.resolution,
            max_probes: self.max_probes,
            faces_per_frame: self.faces_per_frame,
        })
    }
}

/// Virtual geometry settings (decision 044): whether meshes built as cluster
/// hierarchies draw the clusters each view needs, and how coarse those may
/// be. Requires version 8.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneVirtualGeometry {
    pub enabled: bool,
    /// The largest simplification error a drawn cluster may show, in pixels
    /// of the viewport height, from 0.25 to 64: smaller draws more
    /// triangles.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.25, max = 64.0)))]
    pub pixel_error: f32,
    /// An optional budget of streamed pages (128 KiB each) kept resident;
    /// beyond it the least recently used make room, and pinned pages always
    /// stay. Omitted, only the device's limits bound them.
    #[cfg_attr(feature = "schema", schemars(range(min = 1)))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_resident_pages: Option<u32>,
    /// Whether the camera's clusters shade through the visibility buffer
    /// (decision 045), each pixel once; false shades them as they draw.
    /// Omitted, true.
    #[serde(skip_serializing_if = "is_yes")]
    pub visibility_buffer: bool,
}

fn is_yes(value: &bool) -> bool {
    *value
}

impl Default for SceneVirtualGeometry {
    fn default() -> Self {
        Self::from(&VirtualGeometry::default())
    }
}

impl From<&VirtualGeometry> for SceneVirtualGeometry {
    fn from(settings: &VirtualGeometry) -> Self {
        Self {
            enabled: settings.enabled,
            pixel_error: settings.pixel_error,
            max_resident_pages: settings.max_resident_pages,
            visibility_buffer: settings.visibility_buffer,
        }
    }
}

impl SceneVirtualGeometry {
    pub fn resolve(&self, path: &str) -> Result<VirtualGeometry, SceneError> {
        if !(self.pixel_error.is_finite() && (0.25..=64.0).contains(&self.pixel_error)) {
            return Err(section_error(path, "pixel_error", "must be in [0.25, 64]"));
        }
        if self.max_resident_pages == Some(0) {
            return Err(section_error(
                path,
                "max_resident_pages",
                "must be at least 1",
            ));
        }
        Ok(VirtualGeometry {
            enabled: self.enabled,
            pixel_error: self.pixel_error,
            max_resident_pages: self.max_resident_pages,
            visibility_buffer: self.visibility_buffer,
        })
    }
}

/// Level of detail settings (decision 047): whether instances draw simplified
/// levels of their meshes where the simplification stays within a pixel
/// error. Requires version 8.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneLevelOfDetail {
    pub enabled: bool,
    /// The largest simplification error a drawn level may show, in pixels of
    /// the viewport height, from 0.25 to 64: smaller draws finer levels
    /// farther away.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.25, max = 64.0)))]
    pub pixel_error: f32,
}

impl Default for SceneLevelOfDetail {
    fn default() -> Self {
        Self::from(&LevelOfDetail::default())
    }
}

impl From<&LevelOfDetail> for SceneLevelOfDetail {
    fn from(settings: &LevelOfDetail) -> Self {
        Self {
            enabled: settings.enabled,
            pixel_error: settings.pixel_error,
        }
    }
}

impl SceneLevelOfDetail {
    pub fn resolve(&self, path: &str) -> Result<LevelOfDetail, SceneError> {
        if !(self.pixel_error.is_finite() && (0.25..=64.0).contains(&self.pixel_error)) {
            return Err(section_error(path, "pixel_error", "must be in [0.25, 64]"));
        }
        Ok(LevelOfDetail {
            enabled: self.enabled,
            pixel_error: self.pixel_error,
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

/// Refraction through glass (decision 038). Requires version 8; a strength
/// of 0 turns it off.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct SceneRefraction {
    /// Scales how far rays bend through transmissive materials.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 4.0)))]
    pub strength: f32,
    /// Blur radius in pixels that fully rough glass reaches.
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 64.0)))]
    pub blur: f32,
}

impl Default for SceneRefraction {
    fn default() -> Self {
        Self::from(&Refraction::default())
    }
}

impl From<&Refraction> for SceneRefraction {
    fn from(refraction: &Refraction) -> Self {
        Self {
            strength: refraction.strength,
            blur: refraction.blur,
        }
    }
}

impl SceneRefraction {
    pub fn resolve(&self, path: &str) -> Result<Refraction, SceneError> {
        if !(0.0..=4.0).contains(&self.strength) {
            return Err(section_error(path, "strength", "must be in [0, 4]"));
        }
        if !(0.0..=64.0).contains(&self.blur) {
            return Err(section_error(path, "blur", "must be in [0, 64]"));
        }
        Ok(Refraction {
            strength: self.strength,
            blur: self.blur,
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
/// the engine's default state (sun shafts, reflections and refraction: off);
/// in a development control, omitted sections stay as they are.
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
    /// Version 8 sections (decisions 038, 039, 043, 044 and 047).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refraction: Option<SceneRefraction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decals: Option<SceneDecals>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reflection_probes: Option<SceneReflectionProbes>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub virtual_geometry: Option<SceneVirtualGeometry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level_of_detail: Option<SceneLevelOfDetail>,
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
    pub refraction: Option<Refraction>,
    pub decals: Option<Decals>,
    pub reflection_probes: Option<ReflectionProbes>,
    pub virtual_geometry: Option<VirtualGeometry>,
    pub level_of_detail: Option<LevelOfDetail>,
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
        let refraction = world
            .get_resource::<Refraction>()
            .copied()
            .unwrap_or(Refraction {
                strength: 0.0,
                ..Refraction::default()
            });
        let decals = world.get_resource::<Decals>().copied().unwrap_or(Decals {
            enabled: false,
            ..Decals::default()
        });
        let probes =
            world
                .get_resource::<ReflectionProbes>()
                .copied()
                .unwrap_or(ReflectionProbes {
                    enabled: false,
                    ..ReflectionProbes::default()
                });
        let virtual_geometry =
            world
                .get_resource::<VirtualGeometry>()
                .copied()
                .unwrap_or(VirtualGeometry {
                    enabled: false,
                    ..VirtualGeometry::default()
                });
        let level_of_detail =
            world
                .get_resource::<LevelOfDetail>()
                .copied()
                .unwrap_or(LevelOfDetail {
                    enabled: false,
                    ..LevelOfDetail::default()
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
            refraction: Some(SceneRefraction::from(&refraction)),
            decals: Some(SceneDecals::from(&decals)),
            reflection_probes: Some(SceneReflectionProbes::from(&probes)),
            virtual_geometry: Some(SceneVirtualGeometry::from(&virtual_geometry)),
            level_of_detail: Some(SceneLevelOfDetail::from(&level_of_detail)),
        }
    }

    /// The first version 8 section present (decisions 038, 039, 043, 044
    /// and 047).
    pub(crate) fn version_eight_field(&self) -> Option<&'static str> {
        [
            ("refraction", self.refraction.is_some()),
            ("decals", self.decals.is_some()),
            ("reflection_probes", self.reflection_probes.is_some()),
            ("virtual_geometry", self.virtual_geometry.is_some()),
            ("level_of_detail", self.level_of_detail.is_some()),
        ]
        .into_iter()
        .find_map(|(name, present)| present.then_some(name))
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
            refraction: self
                .refraction
                .map(|section| section.resolve(&format!("{path}.refraction")))
                .transpose()?,
            decals: self
                .decals
                .map(|section| section.resolve(&format!("{path}.decals")))
                .transpose()?,
            reflection_probes: self
                .reflection_probes
                .map(|section| section.resolve(&format!("{path}.reflection_probes")))
                .transpose()?,
            virtual_geometry: self
                .virtual_geometry
                .map(|section| section.resolve(&format!("{path}.virtual_geometry")))
                .transpose()?,
            level_of_detail: self
                .level_of_detail
                .map(|section| section.resolve(&format!("{path}.level_of_detail")))
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
        if let Some(refraction) = self.refraction {
            world.insert_resource(refraction);
        }
        if let Some(decals) = self.decals {
            world.insert_resource(decals);
        }
        if let Some(probes) = self.reflection_probes {
            world.insert_resource(probes);
        }
        if let Some(settings) = self.virtual_geometry {
            world.insert_resource(settings);
        }
        if let Some(settings) = self.level_of_detail {
            world.insert_resource(settings);
        }
    }

    /// Sets every section in `world`, omitted ones to the engine's default
    /// state: a document's environment owns the whole look. Sun shafts,
    /// reflections and refraction are opt-in, so omitting them removes them.
    pub fn apply_all(&self, world: &mut World) {
        world.insert_resource(self.atmosphere.unwrap_or_default());
        world.insert_resource(self.shadows.unwrap_or_default());
        world.insert_resource(self.ambient_occlusion.unwrap_or_default());
        world.insert_resource(self.bloom.unwrap_or_default());
        world.insert_resource(self.local_lights.unwrap_or_default());
        world.insert_resource(self.particles.unwrap_or_default());
        world.insert_resource(self.decals.unwrap_or_default());
        world.insert_resource(self.reflection_probes.unwrap_or_default());
        world.insert_resource(self.virtual_geometry.unwrap_or_default());
        world.insert_resource(self.level_of_detail.unwrap_or_default());
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
        match self.refraction {
            Some(refraction) => world.insert_resource(refraction),
            None => {
                world.remove_resource::<Refraction>();
            }
        }
    }
}
