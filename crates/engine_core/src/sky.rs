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

settings! {
    /// Haze: distant surfaces fade into the sky behind them. Fog and the air
    /// take the sky's own light, blurred, in each view's direction (looking
    /// down, the sky's just above the horizon, so far ground melts into the
    /// lit haze there), and the air begins at the eye, at least as thick as
    /// `visibility` asks, thinning with height as the atmosphere's (or the
    /// physically based sky's) air does, as thick as `strength` makes it.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneHaze")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Haze {
        /// Whether fog and the air take the sky's light.
        pub enabled: bool = true => any,
        /// The distance (world units) at which the air leaves a fiftieth of
        /// what lies level with the eye; 0 takes twice the view's far
        /// distance, so the world fades into the sky before it ends. The air
        /// is never thinner than the atmosphere's own.
        pub visibility: f32 = 0.0 => at_least(0.0),
        /// How thick the air is, as a multiple of what `visibility` (or the
        /// atmosphere's own air) makes it: 0 clear, 1 as thick as that.
        pub strength: f32 = 0.7 => range(0.0, 4.0),
    }
}

impl Haze {
    /// What a world without these settings reads as: no haze.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    /// The night sky and the sky's high veil: stars turning with the time
    /// of day, where the sky behind them is dark; the moon, which the time
    /// of day places (without one there is none) and the sun lights, giving
    /// its phase, and whose light brightens the physically based sky's
    /// night; and thin cirrus high above at any hour. Opt-in.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneNightSky")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct NightSky {
        /// How bright the stars are: 0 hides them, 1 is a clear night's.
        pub stars: f32 = 1.0 => range(0.0, 8.0),
        /// How bright the moon is: 0 hides it, 1 is the full moon's.
        pub moon: f32 = 1.0 => range(0.0, 8.0),
        /// The moon's age as a share of its month: 0 new, 0.5 full, waning
        /// toward 1.
        pub moon_age: f32 = 0.5 => range(0.0, 1.0),
        /// How much of the sky cirrus covers: 0 none.
        pub cirrus: f32 = 0.35 => range(0.0, 1.0),
    }
}

impl NightSky {
    /// What a world without these settings reads as: no stars, moon or
    /// cirrus.
    pub fn off() -> Self {
        Self {
            stars: 0.0,
            moon: 0.0,
            cirrus: 0.0,
            ..Self::default()
        }
    }
}
