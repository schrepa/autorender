use std::f32::consts::TAU;

use engine::math;
use engine::prelude::Resource;
use glam::{Quat, Vec3};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Drift {
    pub arc_degrees: f32,
    pub rise_m: f32,
    pub period_seconds: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LobbyCamera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub fov_degrees: f32,
    pub drift: Drift,
}

impl LobbyCamera {
    #[must_use]
    pub fn moved(self, by: Vec3) -> Self {
        Self {
            position: (Vec3::from_array(self.position) + by).to_array(),
            target: (Vec3::from_array(self.target) + by).to_array(),
            ..self
        }
    }

    pub fn at(&self, seconds: f64) -> (Vec3, Vec3) {
        let target = Vec3::from_array(self.target);
        let from = Vec3::from_array(self.position) - target;
        let period = f64::from(self.drift.period_seconds);
        let share = if period > 0.0 {
            (seconds / period).rem_euclid(1.0) as f32
        } else {
            0.0
        };
        let (sin, cos) = math::sin_cos(share * TAU);
        let turn = Quat::from_rotation_y((self.drift.arc_degrees * 0.5).to_radians() * sin);
        let rise = Vec3::Y * (self.drift.rise_m * cos);
        (target + turn * from + rise, target)
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct LobbyView {
    pub camera: Option<LobbyCamera>,
    pub since: f64,
}

impl LobbyView {
    pub fn shot(&self, now: f64) -> Option<(Vec3, Vec3, f32)> {
        self.camera.map(|camera| {
            let (position, target) = camera.at(now - self.since);
            (position, target, camera.fov_degrees)
        })
    }
}
