//! A scene's environment (decision 022): its atmosphere (decision 018) and
//! rendering effects (decisions 020, 021, 023 to 026, 038, 039, 043, 044 and
//! 047) as authored data.
//! Version 5 documents with an `environment` apply it when instantiated
//! (local lights, particles, sun shafts and reflections need version 6);
//! development controls apply the same sections temporarily, and inspection
//! reports the current resources in the same shape.
//!
//! Every effect section is the engine's settings type itself (`Shadows`,
//! `Bloom` and so on), which checks itself with `validate`; only the
//! atmosphere is authored differently, as a preset with overrides.

use engine_core::{
    AmbientOcclusion, Atmosphere, Bloom, Decals, Environment, InvalidSetting, LevelOfDetail,
    LocalLights, Particles, ReflectionProbes, Reflections, Refraction, Shadows, SunShafts,
    VirtualGeometry, bevy_ecs::prelude::World, glam::Vec3,
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
    /// The atmosphere at authored path `path`: its preset with overrides,
    /// checked by [`Atmosphere::validate`]: values must be finite and
    /// nonnegative, the fog maximum at most 1 and the sun direction nonzero.
    pub fn resolve(&self, path: &str) -> Result<Atmosphere, SceneError> {
        let mut atmosphere = self.preset.atmosphere();
        let colors = [
            (self.sun_color, &mut atmosphere.sun_color),
            (self.zenith, &mut atmosphere.zenith),
            (self.horizon, &mut atmosphere.horizon),
            (self.ground, &mut atmosphere.ground),
            (self.shadow_tint, &mut atmosphere.shadow_tint),
            (self.highlight_tint, &mut atmosphere.highlight_tint),
        ];
        for (value, target) in colors {
            if let Some(value) = value {
                *target = Vec3::from_array(value);
            }
        }
        if let Some(direction) = self.sun_direction {
            atmosphere.sun_direction = Vec3::from_array(direction);
        }
        let scalars = [
            (self.ambient, &mut atmosphere.ambient),
            (self.fog_start, &mut atmosphere.fog_start),
            (self.fog_density, &mut atmosphere.fog_density),
            (self.fog_falloff, &mut atmosphere.fog_falloff),
            (self.fog_max, &mut atmosphere.fog_max),
            (self.rim, &mut atmosphere.rim),
            (self.specular, &mut atmosphere.specular),
            (self.emission, &mut atmosphere.emission),
            (self.exposure, &mut atmosphere.exposure),
            (self.saturation, &mut atmosphere.saturation),
            (self.contrast, &mut atmosphere.contrast),
            (self.vibrance, &mut atmosphere.vibrance),
        ];
        for (value, target) in scalars {
            if let Some(value) = value {
                *target = value;
            }
        }
        // Presets are valid, so the first invalid field is an override.
        atmosphere
            .validate()
            .map_err(|invalid| section_error(path, invalid))?;
        Ok(atmosphere)
    }
}

/// The error of section `path`'s `invalid` field.
fn section_error(path: &str, invalid: InvalidSetting) -> SceneError {
    SceneError::new(
        "invalid_environment",
        format!("{path}.{}", invalid.field),
        invalid.rule,
    )
}

/// `section` when absent or valid by `validate`, at authored path
/// `{path}.{name}`.
fn checked<T>(
    section: Option<T>,
    path: &str,
    name: &str,
    validate: fn(&T) -> Result<(), InvalidSetting>,
) -> Result<Option<T>, SceneError> {
    if let Some(settings) = &section {
        validate(settings).map_err(|invalid| section_error(&format!("{path}.{name}"), invalid))?;
    }
    Ok(section)
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
    pub shadows: Option<Shadows>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ambient_occlusion: Option<AmbientOcclusion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bloom: Option<Bloom>,
    /// Version 6 sections (decisions 023 to 026).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_lights: Option<LocalLights>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sun_shafts: Option<SunShafts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reflections: Option<Reflections>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub particles: Option<Particles>,
    /// Version 8 sections (decisions 038, 039, 043, 044 and 047).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refraction: Option<Refraction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decals: Option<Decals>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reflection_probes: Option<ReflectionProbes>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub virtual_geometry: Option<VirtualGeometry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level_of_detail: Option<LevelOfDetail>,
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
    /// Every section, from `world`'s current resources
    /// ([`Environment::current`]): resolving it reproduces them. Absent
    /// effect resources read as off.
    pub fn current(world: &World) -> Self {
        let environment = Environment::current(world);
        Self {
            atmosphere: Some(SceneAtmosphere::from(&environment.atmosphere)),
            shadows: Some(environment.shadows),
            ambient_occlusion: Some(environment.ambient_occlusion),
            bloom: Some(environment.bloom),
            local_lights: Some(environment.local_lights),
            sun_shafts: Some(environment.sun_shafts),
            reflections: Some(environment.reflections),
            particles: Some(environment.particles),
            refraction: Some(environment.refraction),
            decals: Some(environment.decals),
            reflection_probes: Some(environment.reflection_probes),
            virtual_geometry: Some(environment.virtual_geometry),
            level_of_detail: Some(environment.level_of_detail),
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
    /// the grade of decision 027, a shadow cascade count other than the
    /// default (decision 028) and the sections of decisions 023 to 026.
    pub(crate) fn version_six_field(&self) -> Option<&'static str> {
        let atmosphere = self.atmosphere.unwrap_or_default();
        let cascades = self
            .shadows
            .is_some_and(|shadows| shadows.cascades != Shadows::default().cascades);
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
            shadows: checked(self.shadows, path, "shadows", Shadows::validate)?,
            ambient_occlusion: checked(
                self.ambient_occlusion,
                path,
                "ambient_occlusion",
                AmbientOcclusion::validate,
            )?,
            bloom: checked(self.bloom, path, "bloom", Bloom::validate)?,
            local_lights: checked(
                self.local_lights,
                path,
                "local_lights",
                LocalLights::validate,
            )?,
            sun_shafts: checked(self.sun_shafts, path, "sun_shafts", SunShafts::validate)?,
            reflections: checked(self.reflections, path, "reflections", Reflections::validate)?,
            particles: checked(self.particles, path, "particles", Particles::validate)?,
            refraction: checked(self.refraction, path, "refraction", Refraction::validate)?,
            decals: checked(self.decals, path, "decals", Decals::validate)?,
            reflection_probes: checked(
                self.reflection_probes,
                path,
                "reflection_probes",
                ReflectionProbes::validate,
            )?,
            virtual_geometry: checked(
                self.virtual_geometry,
                path,
                "virtual_geometry",
                VirtualGeometry::validate,
            )?,
            level_of_detail: checked(
                self.level_of_detail,
                path,
                "level_of_detail",
                LevelOfDetail::validate,
            )?,
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
