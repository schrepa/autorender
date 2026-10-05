//! A scene's environment (decision 022): its atmosphere (decision 018) and
//! rendering effects (decisions 020, 021, 023 to 026, 038, 039, 043, 044 and
//! 047) and terrain settings (decision 051) as authored data.
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
    TerrainSettings, VirtualGeometry,
    bevy_ecs::prelude::{Resource, World},
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
    /// Aerial perspective (decision 053), version 9: beyond this distance,
    /// air of `aerial_density` per unit at height zero (0: off), thinning by
    /// `aerial_falloff` per unit up, dims toward `aerial_color` and, toward
    /// the sun, an `aerial_glow` of its color.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aerial_start: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aerial_density: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aerial_falloff: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aerial_color: Option<[f32; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aerial_glow: Option<f32>,
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
            aerial_start: Some(atmosphere.aerial_start),
            aerial_density: Some(atmosphere.aerial_density),
            aerial_falloff: Some(atmosphere.aerial_falloff),
            aerial_color: Some(atmosphere.aerial_color.to_array()),
            aerial_glow: Some(atmosphere.aerial_glow),
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
        let atmosphere = self.atmosphere();
        // Presets are valid, so the first invalid field is an override.
        atmosphere
            .validate()
            .map_err(|invalid| section_error(path, invalid))?;
        Ok(atmosphere)
    }

    /// Its preset with overrides, unchecked: [`Self::resolve`] checks them.
    pub fn atmosphere(&self) -> Atmosphere {
        let mut atmosphere = self.preset.atmosphere();
        let colors = [
            (self.sun_color, &mut atmosphere.sun_color),
            (self.zenith, &mut atmosphere.zenith),
            (self.horizon, &mut atmosphere.horizon),
            (self.ground, &mut atmosphere.ground),
            (self.shadow_tint, &mut atmosphere.shadow_tint),
            (self.highlight_tint, &mut atmosphere.highlight_tint),
            (self.aerial_color, &mut atmosphere.aerial_color),
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
            (self.aerial_start, &mut atmosphere.aerial_start),
            (self.aerial_density, &mut atmosphere.aerial_density),
            (self.aerial_falloff, &mut atmosphere.aerial_falloff),
            (self.aerial_glow, &mut atmosphere.aerial_glow),
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
        atmosphere
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

/// An effect section of a scene environment (decision 022): the engine's
/// settings resource, which checks itself; the scene version that first
/// authors it; and whether it is opt-in (off unless named), which a
/// document's environment removes when it omits the section instead of
/// setting its default.
pub trait EnvironmentSection: Resource + Copy + Default + serde::Serialize {
    /// The settings type's name, which development tools inspect it by.
    const NAME: &'static str;
    const SINCE: u32;
    const OPT_IN: bool;
    /// The first invalid field and the rule it breaks, if any.
    fn check(&self) -> Result<(), InvalidSetting>;
    /// The section's JSON schema.
    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema;
}

macro_rules! section {
    ($settings:ident, since $since:literal, opt_in $opt_in:literal) => {
        impl EnvironmentSection for $settings {
            const NAME: &'static str = stringify!($settings);
            const SINCE: u32 = $since;
            const OPT_IN: bool = $opt_in;
            fn check(&self) -> Result<(), InvalidSetting> {
                self.validate()
            }
            #[cfg(feature = "schema")]
            fn schema() -> schemars::Schema {
                schemars::schema_for!($settings)
            }
        }
    };
}

section!(Shadows, since 5, opt_in false);
section!(AmbientOcclusion, since 5, opt_in false);
section!(Bloom, since 5, opt_in false);
section!(LocalLights, since 6, opt_in false);
section!(SunShafts, since 6, opt_in true);
section!(Reflections, since 6, opt_in true);
section!(Particles, since 6, opt_in false);
section!(Refraction, since 8, opt_in true);
section!(Decals, since 8, opt_in false);
section!(ReflectionProbes, since 8, opt_in false);
section!(VirtualGeometry, since 8, opt_in false);
section!(LevelOfDetail, since 8, opt_in false);
section!(TerrainSettings, since 9, opt_in false);

/// Receives an environment's effect sections ([`SceneEnvironment::visit`]),
/// each with its authored name.
pub trait SectionVisitor {
    fn section<T: EnvironmentSection>(&mut self, name: &'static str, section: Option<T>);
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
    /// Version 9: terrain streaming and level of detail (decision 051).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terrain: Option<TerrainSettings>,
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
            terrain: Some(environment.terrain),
        }
    }

    /// Hands every effect section (all but the atmosphere) to `visitor`, in
    /// authored order: the one list of them.
    pub fn visit(&self, visitor: &mut impl SectionVisitor) {
        visitor.section("shadows", self.shadows);
        visitor.section("ambient_occlusion", self.ambient_occlusion);
        visitor.section("bloom", self.bloom);
        visitor.section("local_lights", self.local_lights);
        visitor.section("sun_shafts", self.sun_shafts);
        visitor.section("reflections", self.reflections);
        visitor.section("particles", self.particles);
        visitor.section("refraction", self.refraction);
        visitor.section("decals", self.decals);
        visitor.section("reflection_probes", self.reflection_probes);
        visitor.section("virtual_geometry", self.virtual_geometry);
        visitor.section("level_of_detail", self.level_of_detail);
        visitor.section("terrain", self.terrain);
    }

    /// The names of the sections present, the atmosphere first.
    pub fn sections(&self) -> Vec<&'static str> {
        struct Names(Vec<&'static str>);
        impl SectionVisitor for Names {
            fn section<T: EnvironmentSection>(&mut self, name: &'static str, section: Option<T>) {
                if section.is_some() {
                    self.0.push(name);
                }
            }
        }
        let mut names = Names(Vec::new());
        if self.atmosphere.is_some() {
            names.0.push("atmosphere");
        }
        self.visit(&mut names);
        names.0
    }

    /// The first effect section present that scene version `version`
    /// introduced.
    pub(crate) fn section_since(&self, version: u32) -> Option<&'static str> {
        struct Since(u32, Option<&'static str>);
        impl SectionVisitor for Since {
            fn section<T: EnvironmentSection>(&mut self, name: &'static str, section: Option<T>) {
                if self.1.is_none() && section.is_some() && T::SINCE == self.0 {
                    self.1 = Some(name);
                }
            }
        }
        let mut since = Since(version, None);
        self.visit(&mut since);
        since.1
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
        ]
        .into_iter()
        .find_map(|(name, present)| present.then_some(name))
        .or_else(|| self.section_since(6))
    }

    /// The first version 9 field present, as its path below the environment:
    /// aerial perspective (decision 053) or terrain settings (decision 051).
    pub(crate) fn version_nine_field(&self) -> Option<&'static str> {
        let atmosphere = self.atmosphere.unwrap_or_default();
        [
            ("atmosphere.aerial_start", atmosphere.aerial_start.is_some()),
            (
                "atmosphere.aerial_density",
                atmosphere.aerial_density.is_some(),
            ),
            (
                "atmosphere.aerial_falloff",
                atmosphere.aerial_falloff.is_some(),
            ),
            ("atmosphere.aerial_color", atmosphere.aerial_color.is_some()),
            ("atmosphere.aerial_glow", atmosphere.aerial_glow.is_some()),
        ]
        .into_iter()
        .find_map(|(name, present)| present.then_some(name))
        .or_else(|| self.section_since(9))
    }

    /// Checks every section present at authored path `path`.
    pub fn validate(&self, path: &str) -> Result<(), SceneError> {
        struct FirstInvalid<'p>(&'p str, Option<SceneError>);
        impl SectionVisitor for FirstInvalid<'_> {
            fn section<T: EnvironmentSection>(&mut self, name: &'static str, section: Option<T>) {
                if self.1.is_none()
                    && let Some(Err(invalid)) = section.map(|section| section.check())
                {
                    self.1 = Some(section_error(&format!("{}.{name}", self.0), invalid));
                }
            }
        }
        if let Some(atmosphere) = &self.atmosphere {
            atmosphere.resolve(&format!("{path}.atmosphere"))?;
        }
        let mut first = FirstInvalid(path, None);
        self.visit(&mut first);
        first.1.map_or(Ok(()), Err)
    }

    /// Inserts every section present into `world`, leaving the others. Call
    /// on an environment [`Self::validate`] accepted.
    pub fn apply_sections(&self, world: &mut World) {
        struct Insert<'w>(&'w mut World);
        impl SectionVisitor for Insert<'_> {
            fn section<T: EnvironmentSection>(&mut self, _: &'static str, section: Option<T>) {
                if let Some(section) = section {
                    self.0.insert_resource(section);
                }
            }
        }
        if let Some(atmosphere) = &self.atmosphere {
            world.insert_resource(atmosphere.atmosphere());
        }
        self.visit(&mut Insert(world));
    }

    /// Sets every section in `world`, omitted ones to the engine's default
    /// state: a document's environment owns the whole look. Opt-in effects
    /// (sun shafts, reflections and refraction) are off by default, so
    /// omitting them removes them. Call on an environment
    /// [`Self::validate`] accepted.
    pub fn apply_all(&self, world: &mut World) {
        struct Own<'w>(&'w mut World);
        impl SectionVisitor for Own<'_> {
            fn section<T: EnvironmentSection>(&mut self, _: &'static str, section: Option<T>) {
                match section {
                    Some(section) => self.0.insert_resource(section),
                    None if T::OPT_IN => {
                        self.0.remove_resource::<T>();
                    }
                    None => self.0.insert_resource(T::default()),
                }
            }
        }
        world.insert_resource(
            self.atmosphere
                .map_or_else(Atmosphere::default, |atmosphere| atmosphere.atmosphere()),
        );
        self.visit(&mut Own(world));
    }
}
