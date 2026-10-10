//! Scene lighting and atmosphere: a directional sun, a sky
//! gradient that also colors ambient light, reflections and fog, aerial
//! perspective, and the grade of the final image: exposure,
//! contrast, saturation, vibrance and split toning. Every
//! extracted frame carries a copy.

use bevy_ecs::prelude::Resource;
use glam::Vec3;

use crate::environment::InvalidSetting;
use engine_settings::Setting;

/// Every atmosphere field once: its doc, which authors read too, its type,
/// the scene version that introduced it and the [`engine_settings::Rule`] its
/// values keep.
/// engine_core builds [`Atmosphere`] and its validation from this list, and
/// engine_scene the authored atmosphere: its fields, schema, overrides and
/// version checks.
#[macro_export]
macro_rules! atmosphere_fields {
    ($then:ident) => {
        $then! {
            /// Direction toward the sun; any nonzero length, normalized when
            /// rendered.
            sun_direction: Vec3, since 5, any;
            /// Sun color times intensity.
            sun_color: Vec3, since 5, at_least(0.0);
            /// Sky color straight up: upward surfaces take their ambient light
            /// from it.
            zenith: Vec3, since 5, at_least(0.0);
            /// Sky color at the horizon: vertical surfaces take their ambient
            /// light from it.
            horizon: Vec3, since 5, at_least(0.0);
            /// Sky color below the horizon: downward surfaces take their
            /// ambient light from it.
            ground: Vec3, since 5, at_least(0.0);
            /// Scale of the sky's ambient light on surfaces.
            ambient: f32, since 5, at_least(0.0);
            /// Fog begins this far from the camera (world units).
            fog_start: f32, since 5, at_least(0.0);
            /// Fog density per world unit at height zero.
            fog_density: f32, since 5, at_least(0.0);
            /// How fast fog thins with height, per world unit.
            fog_falloff: f32, since 5, at_least(0.0);
            /// The most fog can cover.
            fog_max: f32, since 5, range(0.0, 1.0);
            /// Aerial perspective begins this far from the camera (world
            /// units): beyond it, air dims what lies behind it toward the light
            /// it scatters. Unlike fog it is unbounded and spans kilometers.
            aerial_start: f32, since 9, at_least(0.0);
            /// Aerial extinction per world unit at height zero; 0 turns it off.
            aerial_density: f32, since 9, at_least(0.0);
            /// How fast the air thins with height, per world unit.
            aerial_falloff: f32, since 9, at_least(0.0);
            /// The color distant ridges fade into away from the sun.
            aerial_color: Vec3, since 9, at_least(0.0);
            /// How strongly distant ridges fade into a glow of the sun's color
            /// toward the sun.
            aerial_glow: f32, since 9, at_least(0.0);
            /// Silhouette light in the horizon color, facing away from the
            /// camera.
            rim: f32, since 5, at_least(0.0);
            /// Scale of sun highlights and sky reflections; 1 is physically
            /// based.
            specular: f32, since 5, at_least(0.0);
            /// Scale of material emission: how strongly lamps, lit windows and
            /// other artificial lights show in this light, dim by day and full
            /// at night.
            emission: f32, since 5, at_least(0.0);
            /// Exposure before tone mapping.
            exposure: f32, since 5, at_least(0.0);
            /// Saturation after tone mapping; 1 keeps it.
            saturation: f32, since 5, at_least(0.0);
            /// Contrast around mid grey after tone mapping: the slope of an
            /// S-curve that keeps black and white; 0 flattens, 1 keeps it, and
            /// the curve is steepest at 1.5.
            contrast: f32, since 6, at_least(0.0);
            /// Saturation added to muted colors more than to vivid ones; 0 adds
            /// none.
            vibrance: f32, since 6, at_least(0.0);
            /// Hue of dark tones after tone mapping (split toning), as a color
            /// gain scaled to keep luminance; white, like black, keeps them.
            /// Cool shadows against warm highlights read as vivid.
            shadow_tint: Vec3, since 6, at_least(0.0);
            /// Hue of bright tones, like `shadow_tint`.
            highlight_tint: Vec3, since 6, at_least(0.0);
        }
    };
}

/// engine_core's part of [`atmosphere_fields`]: the atmosphere, its
/// validation, and the rules an authored atmosphere's schema states.
macro_rules! core_atmosphere {
    ($($(#[$doc:meta])* $field:ident: $ty:ident, since $since:literal, $rule:ident $(($($arg:expr),*))?;)*) => {
        /// Lighting of the rendered scene, read when frames are extracted.
        ///
        /// Colors are linear RGB. The sun and sky may exceed 1: the renderer
        /// tone maps the result. Vertical surfaces take their ambient light
        /// from the horizon color, upward ones from the zenith and downward
        /// ones from the ground, and fog takes the sky's color behind what it
        /// covers, so the three sky colors set the scene's palette.
        #[derive(Resource, Clone, Copy, Debug, PartialEq)]
        pub struct Atmosphere {
            $($(#[$doc])* pub $field: $ty,)*
        }

        impl Atmosphere {
            /// Ok when every field keeps its rule, as an authored
            /// atmosphere's schema states it, and the sun has a direction;
            /// otherwise the first field breaking one.
            pub fn validate(&self) -> Result<(), InvalidSetting> {
                $(Setting::check(&self.$field, stringify!($field), engine_settings::rule::$rule($($($arg),*)?))?;)*
                if !self.sun_direction.is_finite() || self.sun_direction.length_squared() == 0.0 {
                    return Err(InvalidSetting::new(
                        "sun_direction",
                        "must be finite and nonzero",
                    ));
                }
                Ok(())
            }
        }

        /// States each atmosphere field's rule in the schema of an authored
        /// atmosphere, whose fields share their names.
        #[cfg(feature = "schema")]
        pub fn state_atmosphere_rules(schema: &mut schemars::Schema) {
            $(engine_settings::state(schema, stringify!($field), engine_settings::rule::$rule($($($arg),*)?));)*
        }
    };
}

crate::atmosphere_fields!(core_atmosphere);

impl Default for Atmosphere {
    fn default() -> Self {
        Self::golden_hour()
    }
}

impl Atmosphere {
    /// A high sun under a deep blue sky, with clear air, sky-blue shadows and
    /// sunlit highlights.
    pub fn day() -> Self {
        Self {
            sun_direction: Vec3::new(0.55, 0.75, 0.35),
            sun_color: Vec3::new(2.3, 2.1, 1.75),
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
            rim: 0.15,
            specular: 1.0,
            emission: 0.1,
            exposure: 0.85,
            saturation: 1.2,
            contrast: 1.12,
            vibrance: 0.35,
            shadow_tint: Vec3::new(0.85, 0.95, 1.2),
            highlight_tint: Vec3::new(1.08, 1.02, 0.92),
        }
    }

    /// A low warm sun, an amber horizon under a deep blue zenith, and violet
    /// shadows against golden highlights: the default.
    pub fn golden_hour() -> Self {
        Self {
            sun_direction: Vec3::new(0.82, 0.45, 0.1),
            sun_color: Vec3::new(2.6, 1.6, 0.85),
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
            rim: 0.45,
            specular: 1.0,
            emission: 0.5,
            exposure: 0.75,
            saturation: 1.2,
            contrast: 1.15,
            vibrance: 0.4,
            shadow_tint: Vec3::new(0.75, 0.9, 1.35),
            highlight_tint: Vec3::new(1.15, 1.0, 0.85),
        }
    }

    /// The sun at the horizon under an indigo sky, with pink haze, blue-violet
    /// shadows and coral highlights.
    pub fn dusk() -> Self {
        Self {
            sun_direction: Vec3::new(0.9, 0.07, 0.4),
            sun_color: Vec3::new(2.2, 0.85, 0.5),
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
            rim: 0.6,
            specular: 1.0,
            emission: 1.0,
            exposure: 0.95,
            saturation: 1.15,
            contrast: 1.18,
            vibrance: 0.35,
            shadow_tint: Vec3::new(0.75, 0.8, 1.45),
            highlight_tint: Vec3::new(1.25, 0.95, 0.75),
        }
    }

    /// Cool moonlight under a dark sky, with a deep blue haze.
    pub fn night() -> Self {
        Self {
            sun_direction: Vec3::new(-0.35, 0.6, -0.45),
            sun_color: Vec3::new(0.35, 0.45, 0.85),
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
            rim: 0.8,
            specular: 1.0,
            emission: 1.0,
            exposure: 1.35,
            saturation: 1.2,
            contrast: 1.2,
            vibrance: 0.5,
            shadow_tint: Vec3::new(0.8, 0.9, 1.35),
            highlight_tint: Vec3::new(1.15, 1.0, 0.85),
        }
    }

    /// This atmosphere when [`Self::validate`] accepts it, otherwise the
    /// default.
    #[must_use]
    pub fn sanitized(self) -> Self {
        if self.validate().is_ok() {
            self
        } else {
            Self::default()
        }
    }
}
