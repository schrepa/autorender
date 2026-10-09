//! A physically based sky and the time of day.
//!
//! [`Sky`] is opt-in. With it the atmosphere is air over a planet:
//! renderers integrate Rayleigh and Mie scattering and ozone absorption
//! (Hillaire 2020) into lookup tables, draw the sky and reflections from
//! them, light the scene with the sun's color after its path through the
//! air and the sky's ambient color, and haze what lies far away by the air
//! in front of it. The authored [`crate::Atmosphere`] keeps the sun's
//! direction, ambient strength and the grade; its sky colors, fog and aerial
//! perspective stand aside.
//!
//! [`TimeOfDay`] is opt-in too: it moves the sun along its path for a
//! latitude and day of the year, from the hour, in steps of whole minutes
//! so that what holds still for the sun (kept shadow maps) redraws once a
//! step rather than every frame.

use bevy_ecs::prelude::*;
use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::FrameTime;
use crate::math::{max, sin_cos};
use crate::rules::{ExtraRules, settings};

settings! {
    /// A physically based sky. Defaults are Earth's. Opt-in. Requires version 9.
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

impl ExtraRules for Sky {}

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
    /// The time of day. Opt-in. Requires version 9.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTimeOfDay")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct TimeOfDay {
        /// Whether it moves the sun.
        pub enabled: bool = true => any,
        /// The hour.
        pub hours: f32 = 15.0 => range(0.0, 24.0),
        /// How many hours pass each second; 0 holds it.
        pub speed: f32 = 0.0 => range(0.0, 24.0),
        /// The minutes of each step the sun moves by.
        pub step: f32 = 2.0 => range(0.1, 60.0),
        /// The latitude that, with the day, sets the sun's path.
        pub latitude: f32 = 45.0 => range(-90.0, 90.0),
        /// The day of the year.
        pub day: f32 = 172.0 => range(0.0, 365.0),
        /// The yaw of north in the world: degrees from -z toward +x.
        pub north: f32 = 0.0 => range(-360.0, 360.0),
    }
}

impl ExtraRules for TimeOfDay {}

impl TimeOfDay {
    /// What a world without these settings reads as: the atmosphere's sun.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// The hour the sun stands at: [`Self::hours`] down to a whole step.
    pub fn stepped_hours(&self) -> f32 {
        let step = self.step / 60.0;
        (self.hours / step).floor() * step
    }

    /// The direction toward the sun at [`Self::stepped_hours`]: up is +y,
    /// north is -z turned by [`Self::north`] toward +x, east is +x turned
    /// alike. Below the horizon at night.
    pub fn sun_direction(&self) -> Vec3 {
        let radians = |degrees: f32| degrees.to_radians();
        // The sun's declination over the year and its hour angle (west of
        // the meridian positive).
        let (_, year) = sin_cos(std::f32::consts::TAU / 365.0 * (self.day + 10.0));
        let declination = radians(-23.44 * year);
        let hour_angle = radians(15.0 * (self.stepped_hours() - 12.0));
        let (sin_latitude, cos_latitude) = sin_cos(radians(self.latitude));
        let (sin_declination, cos_declination) = sin_cos(declination);
        let (sin_hour, cos_hour) = sin_cos(hour_angle);
        let up = sin_declination * sin_latitude + cos_declination * cos_hour * cos_latitude;
        let north = sin_declination * cos_latitude - cos_declination * cos_hour * sin_latitude;
        let east = -cos_declination * sin_hour;
        let (sin_north, cos_north) = sin_cos(radians(self.north));
        let north_axis = Vec3::new(-sin_north, 0.0, -cos_north);
        let east_axis = Vec3::new(cos_north, 0.0, -sin_north);
        (Vec3::Y * up + north_axis * north + east_axis * east).normalize_or(Vec3::Y)
    }
}

/// Advances the time of day by the frame's time.
pub(crate) fn advance_time_of_day(time: Res<FrameTime>, day: Option<ResMut<TimeOfDay>>) {
    let Some(mut day) = day else {
        return;
    };
    if !day.enabled || day.speed <= 0.0 || time.delta_seconds <= 0.0 {
        return;
    }
    let hours = day.hours + day.speed * time.delta_seconds;
    day.hours = max(hours.rem_euclid(24.0), 0.0);
}
