#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::*;
use glam::Vec3;

use engine_core::{Brick, Bricks, FrameTime, Schedule, Source, settings};
use engine_math::{max, sin_cos};
use engine_places::sun::{SunPath, north_axis, star_turn, stepped_hours};

settings! {
    #[section("time_of_day")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneTimeOfDay")]
    pub struct TimeOfDay {
        pub enabled: bool = false => any,
        pub hours: f32 = 15.0 => range(0.0, 24.0),
        pub speed: f32 = 0.0 => range(0.0, 24.0),
        pub step: f32 = 2.0 => range(0.1, 60.0),
        pub latitude: f32 = 45.0 => range(-90.0, 90.0),
        pub day: f32 = 172.0 => range(0.0, 365.0),
        pub north: f32 = 0.0 => range(-360.0, 360.0),
        pub moon_age: f32 = 0.5 => range(0.0, 1.0),
    }
}

impl TimeOfDay {
    fn sun_direction(&self) -> Vec3 {
        let (declination, hour_angle) = self.sun_place();
        self.sky_direction(declination, hour_angle)
    }

    fn moon_direction(&self) -> Vec3 {
        let (declination, hour_angle) = self.sun_place();
        let turn = std::f32::consts::TAU * self.moon_age;
        let (_, cos_turn) = sin_cos(turn);
        self.sky_direction(declination * cos_turn, hour_angle - turn)
    }

    fn sun_place(&self) -> (f32, f32) {
        let (_, year) = sin_cos(std::f32::consts::TAU / 365.0 * (self.day + 10.0));
        (
            (-23.44 * year).to_radians(),
            (15.0 * (stepped_hours(self.hours, self.step) - 12.0)).to_radians(),
        )
    }

    fn sky_direction(&self, declination: f32, hour_angle: f32) -> Vec3 {
        let (sin_latitude, cos_latitude) = sin_cos(self.latitude.to_radians());
        let (sin_declination, cos_declination) = sin_cos(declination);
        let (sin_hour, cos_hour) = sin_cos(hour_angle);
        let up = sin_declination * sin_latitude + cos_declination * cos_hour * cos_latitude;
        let north = sin_declination * cos_latitude - cos_declination * cos_hour * sin_latitude;
        let east = -cos_declination * sin_hour;
        let (sin_north, cos_north) = sin_cos(self.north.to_radians());
        let east_axis = Vec3::new(cos_north, 0.0, -sin_north);
        (Vec3::Y * up + north_axis(self.north) * north + east_axis * east).normalize_or(Vec3::Y)
    }
}

#[derive(Resource, Default)]
struct DayTime {
    hours: Option<f32>,
}

impl DayTime {
    fn clock(&self, settings: &TimeOfDay) -> TimeOfDay {
        TimeOfDay {
            hours: self.hours.unwrap_or(settings.hours),
            ..*settings
        }
    }
}

fn restart(settings: &Res<TimeOfDay>, day: &mut ResMut<DayTime>) {
    if day.hours.is_none() || settings.is_changed_after(day.last_changed()) {
        day.hours = Some(settings.hours);
    }
}

fn advance_time_of_day(time: Res<FrameTime>, settings: Res<TimeOfDay>, mut day: ResMut<DayTime>) {
    restart(&settings, &mut day);
    if settings.speed <= 0.0 || time.delta_seconds <= 0.0 {
        return;
    }
    let hours = day.hours.unwrap_or(settings.hours) + settings.speed * time.delta_seconds;
    day.hours = Some(max(hours.rem_euclid(24.0), 0.0));
}

fn publish_path(settings: Res<TimeOfDay>, mut day: ResMut<DayTime>, mut path: ResMut<SunPath>) {
    restart(&settings, &mut day);
    let clock = day.clock(&settings);
    path.set_if_neq(SunPath {
        direction: Some(clock.sun_direction()),
        hours: Some(clock.hours),
        moon: Some(clock.moon_direction()),
        stars: star_turn(
            stepped_hours(clock.hours, clock.step),
            clock.latitude,
            clock.day,
            clock.north,
        ),
    });
}

pub struct TimeOfDayBrick;

impl Brick for TimeOfDayBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|day: Res<TimeOfDay>| day.enabled)
            .settings::<TimeOfDay>()
            .resource::<DayTime>()
            .fill::<SunPath>(None)
            .systems(Schedule::Presentation, advance_time_of_day)
            .systems(Schedule::ViewSync, publish_path);
    }
}
