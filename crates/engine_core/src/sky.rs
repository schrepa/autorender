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
use crate::environment::{InvalidSetting, check_range};
use crate::math::{clamp, max, sin_cos};

/// A physically based sky: whether it draws; the planet's
/// radius and the atmosphere's thickness (kilometers); Rayleigh scattering
/// per kilometer at the ground (red, green, blue) and the height over which
/// it thins by e (km); Mie scattering and absorption per kilometer, their
/// scale height and the forward lobe of their phase (0 to 0.99); ozone
/// absorption per kilometer at its densest, 25 km up; the ground's albedo;
/// the sun's intensity (its color's scale above the air) and angular radius
/// (degrees); how far air hazes what lies away from the camera, as a
/// multiple of real distance (0 none); the height of the world's origin
/// above the ground of the planet (meters); and how far exposure adapts to
/// the sky's light (0 not at all, 1 fully), so that dawn, dusk and night
/// stay readable. Defaults are Earth's. Opt-in. Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneSky")
)]
#[serde(default, deny_unknown_fields)]
pub struct Sky {
    pub enabled: bool,
    #[cfg_attr(feature = "schema", schemars(range(min = 100.0, max = 100000.0)))]
    pub planet_radius: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 1.0, max = 1000.0)))]
    pub atmosphere_height: f32,
    pub rayleigh: [f32; 3],
    #[cfg_attr(feature = "schema", schemars(range(min = 0.1, max = 100.0)))]
    pub rayleigh_height: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub mie: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub mie_absorption: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.1, max = 100.0)))]
    pub mie_height: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 0.99)))]
    pub mie_g: f32,
    pub ozone: [f32; 3],
    pub ground_albedo: [f32; 3],
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1000.0)))]
    pub sun_intensity: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.01, max = 10.0)))]
    pub sun_size: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100.0)))]
    pub aerial: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 100000.0)))]
    pub altitude: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 1.0)))]
    pub adaptation: f32,
}

impl Default for Sky {
    fn default() -> Self {
        Self {
            enabled: true,
            planet_radius: 6360.0,
            atmosphere_height: 100.0,
            rayleigh: [5.802e-3, 13.558e-3, 33.1e-3],
            rayleigh_height: 8.0,
            mie: 3.996e-3,
            mie_absorption: 4.4e-3,
            mie_height: 1.2,
            mie_g: 0.8,
            ozone: [0.65e-3, 1.881e-3, 0.085e-3],
            ground_albedo: [0.3, 0.3, 0.3],
            sun_intensity: 3.0,
            sun_size: 0.27,
            aerial: 1.0,
            altitude: 0.0,
            adaptation: 0.5,
        }
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

    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("planet_radius", self.planet_radius, 100.0..=100_000.0)?;
        check_range("atmosphere_height", self.atmosphere_height, 1.0..=1000.0)?;
        check_range("rayleigh_height", self.rayleigh_height, 0.1..=100.0)?;
        check_range("mie", self.mie, 0.0..=1.0)?;
        check_range("mie_absorption", self.mie_absorption, 0.0..=1.0)?;
        check_range("mie_height", self.mie_height, 0.1..=100.0)?;
        check_range("mie_g", self.mie_g, 0.0..=0.99)?;
        check_range("sun_intensity", self.sun_intensity, 0.0..=1000.0)?;
        check_range("sun_size", self.sun_size, 0.01..=10.0)?;
        check_range("aerial", self.aerial, 0.0..=100.0)?;
        check_range("altitude", self.altitude, 0.0..=100_000.0)?;
        check_range("adaptation", self.adaptation, 0.0..=1.0)?;
        for (field, values, high) in [
            ("rayleigh", self.rayleigh, 1.0),
            ("ozone", self.ozone, 1.0),
            ("ground_albedo", self.ground_albedo, 1.0),
        ] {
            if !values.iter().all(|value| (0.0..=high).contains(value)) {
                return Err(InvalidSetting::new(field, "values must be in [0, 1]"));
            }
        }
        Ok(())
    }

    /// Every value in range: out-of-range values clamp, unusable ones fall
    /// back to the defaults.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let within = |value: f32, low: f32, high: f32, fallback: f32| {
            if value.is_finite() {
                clamp(value, low, high)
            } else {
                fallback
            }
        };
        let triple = |values: [f32; 3], fallback: [f32; 3]| {
            let mut out = values;
            for (value, fallback) in out.iter_mut().zip(fallback) {
                *value = within(*value, 0.0, 1.0, fallback);
            }
            out
        };
        Self {
            enabled: self.enabled,
            planet_radius: within(self.planet_radius, 100.0, 100_000.0, defaults.planet_radius),
            atmosphere_height: within(
                self.atmosphere_height,
                1.0,
                1000.0,
                defaults.atmosphere_height,
            ),
            rayleigh: triple(self.rayleigh, defaults.rayleigh),
            rayleigh_height: within(self.rayleigh_height, 0.1, 100.0, defaults.rayleigh_height),
            mie: within(self.mie, 0.0, 1.0, defaults.mie),
            mie_absorption: within(self.mie_absorption, 0.0, 1.0, defaults.mie_absorption),
            mie_height: within(self.mie_height, 0.1, 100.0, defaults.mie_height),
            mie_g: within(self.mie_g, 0.0, 0.99, defaults.mie_g),
            ozone: triple(self.ozone, defaults.ozone),
            ground_albedo: triple(self.ground_albedo, defaults.ground_albedo),
            sun_intensity: within(self.sun_intensity, 0.0, 1000.0, defaults.sun_intensity),
            sun_size: within(self.sun_size, 0.01, 10.0, defaults.sun_size),
            aerial: within(self.aerial, 0.0, 100.0, defaults.aerial),
            altitude: within(self.altitude, 0.0, 100_000.0, defaults.altitude),
            adaptation: within(self.adaptation, 0.0, 1.0, defaults.adaptation),
        }
    }
}

/// The time of day: whether it moves the sun; the hour (0 to
/// 24); how many hours pass each second (0 holds it, up to 24); the
/// minutes of each step the sun moves by (0.1 to 60); the latitude and day
/// of the year (0 to 365) that set the sun's path; and the yaw of north in
/// the world (degrees from -z toward +x). Opt-in. Requires version 9.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(
    feature = "schema",
    derive(schemars::JsonSchema),
    schemars(rename = "SceneTimeOfDay")
)]
#[serde(default, deny_unknown_fields)]
pub struct TimeOfDay {
    pub enabled: bool,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 24.0)))]
    pub hours: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 24.0)))]
    pub speed: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.1, max = 60.0)))]
    pub step: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = -90.0, max = 90.0)))]
    pub latitude: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = 0.0, max = 365.0)))]
    pub day: f32,
    #[cfg_attr(feature = "schema", schemars(range(min = -360.0, max = 360.0)))]
    pub north: f32,
}

impl Default for TimeOfDay {
    fn default() -> Self {
        Self {
            enabled: true,
            hours: 15.0,
            speed: 0.0,
            step: 2.0,
            latitude: 45.0,
            day: 172.0,
            north: 0.0,
        }
    }
}

impl TimeOfDay {
    /// What a world without these settings reads as: the atmosphere's sun.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// The first field out of range.
    pub fn validate(&self) -> Result<(), InvalidSetting> {
        check_range("hours", self.hours, 0.0..=24.0)?;
        check_range("speed", self.speed, 0.0..=24.0)?;
        check_range("step", self.step, 0.1..=60.0)?;
        check_range("latitude", self.latitude, -90.0..=90.0)?;
        check_range("day", self.day, 0.0..=365.0)?;
        check_range("north", self.north, -360.0..=360.0)
    }

    /// Every value in range: out-of-range values clamp, unusable ones fall
    /// back to the defaults.
    pub fn sanitized(self) -> Self {
        let defaults = Self::default();
        let within = |value: f32, low: f32, high: f32, fallback: f32| {
            if value.is_finite() {
                clamp(value, low, high)
            } else {
                fallback
            }
        };
        Self {
            enabled: self.enabled,
            hours: within(self.hours, 0.0, 24.0, defaults.hours),
            speed: within(self.speed, 0.0, 24.0, defaults.speed),
            step: within(self.step, 0.1, 60.0, defaults.step),
            latitude: within(self.latitude, -90.0, 90.0, defaults.latitude),
            day: within(self.day, 0.0, 365.0, defaults.day),
            north: within(self.north, -360.0, 360.0, defaults.north),
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
