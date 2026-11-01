use bevy_ecs::prelude::Resource;
use engine_math::sin_cos;
use engine_place::{Slot, gpu_struct};
use glam::{Mat3, Quat, Vec3};

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct SunPath {
    pub direction: Option<Vec3>,
    pub hours: Option<f32>,
    pub moon: Option<Vec3>,
    pub stars: Quat,
}

impl Default for SunPath {
    fn default() -> Self {
        Self {
            direction: None,
            hours: None,
            moon: None,
            stars: star_turn(stepped_hours(15.0, 2.0), 45.0, 172.0, 0.0),
        }
    }
}

pub fn stepped_hours(hours: f32, step: f32) -> f32 {
    let step = step / 60.0;
    (hours / step).floor() * step
}

pub fn north_axis(north: f32) -> Vec3 {
    let (sin_north, cos_north) = sin_cos(north.to_radians());
    Vec3::new(-sin_north, 0.0, -cos_north)
}

pub fn star_turn(hours: f32, latitude: f32, day: f32, north: f32) -> Quat {
    let (sin_latitude, cos_latitude) = sin_cos(latitude.to_radians());
    let (sin_north, cos_north) = sin_cos(north.to_radians());
    let toward = north_axis(north);
    let east = Vec3::new(cos_north, 0.0, -sin_north);
    let meridian = Vec3::Y * cos_latitude - toward * sin_latitude;
    let pole = Vec3::Y * sin_latitude + toward * cos_latitude;
    let sidereal = (hours - 12.0) / 24.0 + (day - 80.0) / 365.25;
    let (sin_turn, cos_turn) = sin_cos(std::f32::consts::TAU * sidereal);
    let equatorial = [
        meridian * cos_turn - east * sin_turn,
        meridian * sin_turn + east * cos_turn,
        pole,
    ];
    let galactic = galactic_axes()
        .map(|axis| equatorial[0] * axis.x + equatorial[1] * axis.y + equatorial[2] * axis.z);
    Quat::from_mat3(&Mat3::from_cols(galactic[0], galactic[1], galactic[2]).transpose())
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

impl Slot for SunPath {
    const NAME: &'static str = "sun_path";
}

gpu_struct! {
    #[derive(Resource)]
    pub struct Sun {
        pub direction: [f32; 4],
        pub color: [f32; 4],
    }
}

impl Sun {
    pub const NONE: Self = Self {
        direction: [0.0, 1.0, 0.0, 0.0],
        color: [0.0, 0.0, 0.0, 0.0],
    };
}

impl Slot for Sun {
    const NAME: &'static str = "sun";
    const WGSL: Option<&'static str> = Some(include_str!("places/sun.wgsl"));
    const SMOKES: &'static [engine_place::Smoke] = &[crate::smokes::SUN];
}
