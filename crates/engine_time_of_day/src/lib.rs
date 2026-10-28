#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use bevy_ecs::prelude::*;
use glam::Vec3;
use serde::{Deserialize, Serialize};

use engine_core::math::{max, sin_cos};
use engine_core::{
    Atmosphere, Brick, Bricks, EnvironmentSection, FrameTime, Presentation, RenderSnapshot,
    ViewSync,
};
use engine_settings::{InvalidSetting, settings};

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneTimeOfDay")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct TimeOfDay {
        pub enabled: bool = true => any,
        pub hours: f32 = 15.0 => range(0.0, 24.0),
        pub speed: f32 = 0.0 => range(0.0, 24.0),
        pub step: f32 = 2.0 => range(0.1, 60.0),
        pub latitude: f32 = 45.0 => range(-90.0, 90.0),
        pub day: f32 = 172.0 => range(0.0, 365.0),
        pub north: f32 = 0.0 => range(-360.0, 360.0),
    }
}

impl EnvironmentSection for TimeOfDay {
    const KEY: &'static str = "time_of_day";
    const NAME: &'static str = "TimeOfDay";
    const OPT_IN: bool = true;

    fn section_off() -> Self {
        Self::off()
    }

    fn section_check(&self) -> Result<(), InvalidSetting> {
        self.validate()
    }

    fn section_sanitized(self) -> Self {
        self.sanitized()
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(TimeOfDay)
    }
}

impl TimeOfDay {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub fn stepped_hours(&self) -> f32 {
        let step = self.step / 60.0;
        (self.hours / step).floor() * step
    }

    pub fn sun_direction(&self) -> Vec3 {
        let (declination, hour_angle) = self.sun_place();
        self.sky_direction(declination, hour_angle)
    }

    pub fn moon_direction(&self, age: f32) -> Vec3 {
        let (declination, hour_angle) = self.sun_place();
        let turn = std::f32::consts::TAU * age;
        let (_, cos_turn) = sin_cos(turn);
        self.sky_direction(declination * cos_turn, hour_angle - turn)
    }

    pub fn star_turn(&self) -> glam::Quat {
        let (sin_latitude, cos_latitude) = sin_cos(self.latitude.to_radians());
        let (sin_north, cos_north) = sin_cos(self.north.to_radians());
        let north = self.north_axis();
        let east = Vec3::new(cos_north, 0.0, -sin_north);
        let meridian = Vec3::Y * cos_latitude - north * sin_latitude;
        let pole = Vec3::Y * sin_latitude + north * cos_latitude;
        let sidereal = (self.stepped_hours() - 12.0) / 24.0 + (self.day - 80.0) / 365.25;
        let (sin_turn, cos_turn) = sin_cos(std::f32::consts::TAU * sidereal);
        let equatorial = [
            meridian * cos_turn - east * sin_turn,
            meridian * sin_turn + east * cos_turn,
            pole,
        ];
        let galactic = galactic_axes()
            .map(|axis| equatorial[0] * axis.x + equatorial[1] * axis.y + equatorial[2] * axis.z);
        glam::Quat::from_mat3(
            &glam::Mat3::from_cols(galactic[0], galactic[1], galactic[2]).transpose(),
        )
    }

    fn sun_place(&self) -> (f32, f32) {
        let (_, year) = sin_cos(std::f32::consts::TAU / 365.0 * (self.day + 10.0));
        (
            (-23.44 * year).to_radians(),
            (15.0 * (self.stepped_hours() - 12.0)).to_radians(),
        )
    }

    fn north_axis(&self) -> Vec3 {
        let (sin_north, cos_north) = sin_cos(self.north.to_radians());
        Vec3::new(-sin_north, 0.0, -cos_north)
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
        (Vec3::Y * up + self.north_axis() * north + east_axis * east).normalize_or(Vec3::Y)
    }
}

fn galactic_axes() -> [Vec3; 3] {
    let toward = |right_ascension: f32, declination: f32| {
        let (sin_ascension, cos_ascension) = sin_cos(right_ascension.to_radians());
        let (sin_declination, cos_declination) = sin_cos(declination.to_radians());
        Vec3::new(
            cos_declination * cos_ascension,
            cos_declination * sin_ascension,
            sin_declination,
        )
    };
    let pole = toward(192.859_48, 27.128_25);
    let centre = toward(266.405, -28.936_17);
    let centre = (centre - pole * centre.dot(pole)).normalize_or(Vec3::X);
    [centre, pole.cross(centre), pole]
}

fn advance_time_of_day(time: Res<FrameTime>, day: Option<ResMut<TimeOfDay>>) {
    let Some(mut day) = day else {
        return;
    };
    if !day.enabled || day.speed <= 0.0 || time.delta_seconds <= 0.0 {
        return;
    }
    let hours = day.hours + day.speed * time.delta_seconds;
    day.hours = max(hours.rem_euclid(24.0), 0.0);
}

struct PlacedSun {
    authored: Vec3,
    placed: Vec3,
}

fn place_sun(
    day: Option<Res<TimeOfDay>>,
    atmosphere: Option<ResMut<Atmosphere>>,
    mut held: Local<Option<PlacedSun>>,
) {
    let Some(mut atmosphere) = atmosphere else {
        return;
    };
    let sun = atmosphere.sun_direction;
    let kept = held.take().filter(|held| held.placed == sun);
    match day.filter(|day| day.enabled) {
        Some(day) => {
            let placed = day.sanitized().sun_direction();
            if sun != placed {
                atmosphere.sun_direction = placed;
            }
            *held = Some(PlacedSun {
                authored: kept.map_or(sun, |kept| kept.authored),
                placed,
            });
        }
        None => {
            if let Some(kept) = kept {
                atmosphere.sun_direction = kept.authored;
            }
        }
    }
}

fn extract_time_of_day(world: &World, snapshot: &mut RenderSnapshot) {
    snapshot.extracted.insert(
        world
            .get_resource::<TimeOfDay>()
            .copied()
            .map_or_else(TimeOfDay::off, TimeOfDay::sanitized),
    );
}

pub struct TimeOfDayBrick;

impl Brick for TimeOfDayBrick {
    fn register(bricks: &mut Bricks) {
        bricks
            .settings::<TimeOfDay>()
            .systems(Presentation, advance_time_of_day)
            .systems(ViewSync, place_sun)
            .extract(extract_time_of_day);
    }
}
