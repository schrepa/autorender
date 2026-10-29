use bevy_ecs::prelude::Resource;
use glam::Vec3;

use engine_settings::{Expected, InvalidSetting, Setting};

#[macro_export]
macro_rules! atmosphere_fields {
    ($then:ident) => {
        $then! {
            sun_direction: Vec3, any;
            sun_color: Vec3, at_least(0.0);
            zenith: Vec3, at_least(0.0);
            horizon: Vec3, at_least(0.0);
            ground: Vec3, at_least(0.0);
            ambient: f32, at_least(0.0);
            fog_start: f32, at_least(0.0);
            fog_density: f32, at_least(0.0);
            fog_falloff: f32, at_least(0.0);
            fog_max: f32, range(0.0, 1.0);
            aerial_start: f32, at_least(0.0);
            aerial_density: f32, at_least(0.0);
            aerial_falloff: f32, at_least(0.0);
            aerial_color: Vec3, at_least(0.0);
            aerial_glow: f32, at_least(0.0);
            rim: f32, at_least(0.0);
            specular: f32, at_least(0.0);
            emission: f32, at_least(0.0);
            exposure: f32, at_least(0.0);
            saturation: f32, at_least(0.0);
            contrast: f32, at_least(0.0);
            vibrance: f32, at_least(0.0);
            shadow_tint: Vec3, at_least(0.0);
            highlight_tint: Vec3, at_least(0.0);
        }
    };
}

macro_rules! core_atmosphere {
    ($($(#[$doc:meta])* $field:ident: $ty:ident, $rule:ident $(($($arg:expr),*))?;)*) => {
        #[derive(Resource, Clone, Copy, Debug, PartialEq)]
        pub struct Atmosphere {
            $($(#[$doc])* pub $field: $ty,)*
        }

        impl Atmosphere {
            pub fn validate(&self) -> Result<(), InvalidSetting> {
                $(Setting::check(&self.$field, stringify!($field), engine_settings::rule::$rule($($($arg),*)?))?;)*
                if !self.sun_direction.is_finite() || self.sun_direction.length_squared() == 0.0 {
                    return Err(InvalidSetting::new("sun_direction", Expected::Nonzero)
                        .got(self.sun_direction.to_array()));
                }
                Ok(())
            }
        }

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

    #[must_use]
    pub fn sanitized(self) -> Self {
        if self.validate().is_ok() {
            self
        } else {
            Self::default()
        }
    }
}
