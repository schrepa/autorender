//! A scene's environment: its atmosphere and
//! rendering effects
//! and terrain settings as authored data.
//! Documents with an `environment` apply it when instantiated;
//! development controls apply the same sections temporarily, and inspection
//! reports the current resources in the same shape.
//!
//! Every effect section is the engine's settings type itself (`Shadows`,
//! `Bloom` and so on), which checks itself with `validate`; only the
//! atmosphere is authored differently, as a preset with overrides.

use engine_core::{
    Atmosphere, Environment, EnvironmentSection, InvalidSetting, bevy_ecs::prelude::World,
    glam::Vec3,
};
use engine_settings::ErrorCode;
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

/// How an atmosphere value is authored: vectors as arrays.
pub trait AuthoredValue: Copy {
    type Authored: Copy;
    fn authored(self) -> Self::Authored;
    fn from_authored(value: Self::Authored) -> Self;
}

impl AuthoredValue for f32 {
    type Authored = f32;

    fn authored(self) -> f32 {
        self
    }

    fn from_authored(value: f32) -> Self {
        value
    }
}

impl AuthoredValue for Vec3 {
    type Authored = [f32; 3];

    fn authored(self) -> [f32; 3] {
        self.to_array()
    }

    fn from_authored(value: [f32; 3]) -> Self {
        Vec3::from_array(value)
    }
}

/// engine_scene's part of [`engine_core::atmosphere_fields`]: the authored
/// atmosphere, its schema's rules and its overrides of a preset.
macro_rules! scene_atmosphere {
    ($($(#[$doc:meta])* $field:ident: $ty:ident, $rule:ident $(($($arg:expr),*))?;)*) => {
        /// Sun, sky, fog and grade: a preset and the fields that
        /// differ from it. Colors are linear RGB and may exceed 1.
        #[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
        #[cfg_attr(
            feature = "schema",
            derive(schemars::JsonSchema),
            schemars(transform = engine_core::state_atmosphere_rules)
        )]
        #[serde(default, deny_unknown_fields)]
        pub struct SceneAtmosphere {
            /// The preset the other fields override.
            pub preset: AtmospherePreset,
            $(
                $(#[$doc])*
                #[serde(skip_serializing_if = "Option::is_none")]
                pub $field: Option<<$ty as AuthoredValue>::Authored>,
            )*
        }

        impl From<&Atmosphere> for SceneAtmosphere {
            /// Every field set: resolving it reproduces `atmosphere` exactly.
            fn from(atmosphere: &Atmosphere) -> Self {
                Self {
                    preset: AtmospherePreset::GoldenHour,
                    $($field: Some(atmosphere.$field.authored()),)*
                }
            }
        }

        impl SceneAtmosphere {
            /// Its preset with overrides, unchecked: [`Self::resolve`] checks
            /// them.
            pub fn atmosphere(&self) -> Atmosphere {
                let mut atmosphere = self.preset.atmosphere();
                $(if let Some(value) = self.$field {
                    atmosphere.$field = AuthoredValue::from_authored(value);
                })*
                atmosphere
            }
        }
    };
}

engine_core::atmosphere_fields!(scene_atmosphere);

impl SceneAtmosphere {
    /// The atmosphere at authored path `path`: its preset with overrides,
    /// checked by [`Atmosphere::validate`].
    pub fn resolve(&self, path: &str) -> Result<Atmosphere, SceneError> {
        let atmosphere = self.atmosphere();
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
        ErrorCode::InvalidEnvironment,
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
    ($($key:ident: $settings:ident { opt_in: $opt_in:literal, feature: $feature:ident }),* $(,)?) => {
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
                    // Sections of features the engine lacks stay absent.
                    None if T::FEATURE
                        .is_some_and(|feature| !engine_core::has_feature(self.0, feature)) => {}
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
