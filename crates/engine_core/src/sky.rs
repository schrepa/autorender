//! A physically based sky.
//!
//! [`Sky`] is opt-in. With it the atmosphere is air over a planet:
//! renderers integrate Rayleigh and Mie scattering and ozone absorption
//! (Hillaire 2020) into lookup tables, draw the sky and reflections from
//! them, light the scene with the sun's color after its path through the
//! air and the sky's ambient color, and haze what lies far away by the air
//! in front of it. The authored [`crate::Atmosphere`] keeps the sun's
//! direction, ambient strength and the grade; its sky colors, fog and aerial
//! perspective stand aside.

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// A physically based sky. Defaults are Earth's. Opt-in.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneSky")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Sky {
        /// Whether the sky draws.
        pub enabled: bool = true => any,
        /// The planet's radius, kilometers.
        pub planet_radius: f32 = 6360.0 => range(100.0, 100_000.0),
        /// The atmosphere's thickness, kilometers.
        pub atmosphere_height: f32 = 100.0 => range(1.0, 1000.0),
        /// Rayleigh scattering per kilometer at the ground: red, green, blue.
        pub rayleigh: [f32; 3] = [5.802e-3, 13.558e-3, 33.1e-3] => range(0.0, 1.0),
        /// The height over which Rayleigh scattering thins by e, kilometers.
        pub rayleigh_height: f32 = 8.0 => range(0.1, 100.0),
        /// Mie scattering per kilometer.
        pub mie: f32 = 3.996e-3 => range(0.0, 1.0),
        /// Mie absorption per kilometer.
        pub mie_absorption: f32 = 4.4e-3 => range(0.0, 1.0),
        /// The height over which Mie scattering thins by e, kilometers.
        pub mie_height: f32 = 1.2 => range(0.1, 100.0),
        /// The forward lobe of Mie scattering's phase.
        pub mie_g: f32 = 0.8 => range(0.0, 0.99),
        /// Ozone absorption per kilometer at its densest, 25 km up.
        pub ozone: [f32; 3] = [0.65e-3, 1.881e-3, 0.085e-3] => range(0.0, 1.0),
        /// The ground's albedo.
        pub ground_albedo: [f32; 3] = [0.3, 0.3, 0.3] => range(0.0, 1.0),
        /// The sun's intensity: its color's scale above the air.
        pub sun_intensity: f32 = 3.0 => range(0.0, 1000.0),
        /// The sun's angular radius, degrees.
        pub sun_size: f32 = 0.27 => range(0.01, 10.0),
        /// How far air hazes what lies away from the camera, as a multiple of
        /// real distance; 0 none.
        pub aerial: f32 = 1.0 => range(0.0, 100.0),
        /// The height of the world's origin above the ground of the planet,
        /// meters.
        pub altitude: f32 = 0.0 => range(0.0, 100_000.0),
        /// How far exposure adapts to the sky's light, so that dawn, dusk and
        /// night stay readable: 0 not at all, 1 fully.
        pub adaptation: f32 = 0.5 => range(0.0, 1.0),
    }
}

impl Sky {
    /// What a world without these settings reads as: the atmosphere's sky.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
