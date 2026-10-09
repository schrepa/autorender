//! A scene's environment: its atmosphere and
//! rendering effects
//! and terrain settings as authored data.
//! Version 5 documents with an `environment` apply it when instantiated
//! (each section needs the scene version its settings type declares);
//! development controls apply the same sections temporarily, and inspection
//! reports the current resources in the same shape.
//!
//! Every effect section is the engine's settings type itself (`Shadows`,
//! `Bloom` and so on), which checks itself with `validate`; only the
//! atmosphere is authored differently, as a preset with overrides.

use engine_core::{
    Atmosphere, Environment, EnvironmentSection, InvalidSetting, Shadows, bevy_ecs::prelude::World,
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

/// Sun, sky, fog and grade: a preset and the fields that
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
    /// Aerial perspective, version 9: beyond this distance,
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
    /// curve, which keeps black and white, is steepest at 1.5. Version 6,
    /// like the grade fields below.
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

/// Receives an environment's effect sections ([`SceneEnvironment::visit`]),
/// each with its authored name.
pub trait SectionVisitor {
    fn section<T: EnvironmentSection>(&mut self, name: &'static str, section: Option<T>);
}

/// engine_scene's part of [`engine_core::environment_sections`]: the
/// authored [`SceneEnvironment`], read back from a world, and the visit of
/// its sections.
macro_rules! scene_sections {
    ($($key:ident: $settings:ident { since: $since:literal, opt_in: $opt_in:literal }),* $(,)?) => {
        /// A scene's look. In a document, every omitted section takes the
        /// engine's default state (opt-in sections: off); in a development
        /// control, omitted sections stay as they are. Each section is the
        /// engine's settings type, which documents its fields.
        #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
        #[serde(default, deny_unknown_fields)]
        pub struct SceneEnvironment {
            #[serde(skip_serializing_if = "Option::is_none")]
            pub atmosphere: Option<SceneAtmosphere>,
            $(
                #[serde(skip_serializing_if = "Option::is_none")]
                pub $key: Option<engine_core::$settings>,
            )*
        }

        impl SceneEnvironment {
            /// Every section, from `world`'s current resources
            /// ([`Environment::current`]): resolving it reproduces them.
            /// Absent section resources read as off.
            pub fn current(world: &World) -> Self {
                let environment = Environment::current(world);
                Self {
                    atmosphere: Some(SceneAtmosphere::from(&environment.atmosphere)),
                    $($key: Some(environment.$key),)*
                }
            }

            /// Hands every section but the atmosphere to `visitor`, in
            /// authored order.
            pub fn visit(&self, visitor: &mut impl SectionVisitor) {
                $(visitor.section(stringify!($key), self.$key);)*
            }
        }
    };
}

engine_core::environment_sections!(scene_sections);

impl SceneEnvironment {
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

    /// The first section present that a scene version after `version`
    /// introduced, and that version.
    pub(crate) fn section_after(&self, version: u32) -> Option<(&'static str, u32)> {
        struct After(u32, Option<(&'static str, u32)>);
        impl SectionVisitor for After {
            fn section<T: EnvironmentSection>(&mut self, name: &'static str, section: Option<T>) {
                if self.1.is_none() && section.is_some() && T::SINCE > self.0 {
                    self.1 = Some((name, T::SINCE));
                }
            }
        }
        let mut after = After(version, None);
        self.visit(&mut after);
        after.1
    }

    /// The keys of the sections that are off unless a scene names them.
    pub fn opt_in_sections() -> Vec<&'static str> {
        struct OptIn(Vec<&'static str>);
        impl SectionVisitor for OptIn {
            fn section<T: EnvironmentSection>(&mut self, name: &'static str, _: Option<T>) {
                if T::OPT_IN {
                    self.0.push(name);
                }
            }
        }
        let mut keys = OptIn(Vec::new());
        Self::default().visit(&mut keys);
        keys.0
    }

    /// Every section's key with the scene version that introduced it.
    pub(crate) fn section_versions() -> Vec<(u32, &'static str)> {
        struct Versions(Vec<(u32, &'static str)>);
        impl SectionVisitor for Versions {
            fn section<T: EnvironmentSection>(&mut self, name: &'static str, _: Option<T>) {
                self.0.push((T::SINCE, name));
            }
        }
        let mut versions = Versions(Vec::new());
        Self::default().visit(&mut versions);
        versions.0
    }

    /// The first version 6 field present, as its path below the environment:
    /// the grade's contrast, vibrance and tints, and a shadow cascade count
    /// other than the default.
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
    }

    /// The first version 9 field present, as its path below the environment:
    /// aerial perspective.
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
    }

    /// Checks every section present at authored path `path`.
    pub fn validate(&self, path: &str) -> Result<(), SceneError> {
        struct FirstInvalid<'p>(&'p str, Option<SceneError>);
        impl SectionVisitor for FirstInvalid<'_> {
            fn section<T: EnvironmentSection>(&mut self, name: &'static str, section: Option<T>) {
                if self.1.is_none()
                    && let Some(Err(invalid)) = section.map(|section| section.section_check())
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
    /// state: a document's environment owns the whole look. Opt-in sections
    /// are off by default, so omitting them removes them. Call on an environment
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
