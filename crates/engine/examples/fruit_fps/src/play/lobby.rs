use std::f32::consts::{PI, TAU};

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

fn ease_in_out_cubic(t: f32) -> f32 {
    let t = math::clamp(t, 0.0, 1.0);
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        let k = -2.0 * t + 2.0;
        1.0 - k * k * k * 0.5
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct LobbyView {
    pub camera: Option<LobbyCamera>,
    pub since: f64,
    pub prev: Option<(Vec3, Vec3, f32)>,
    pub blend_since: f64,
    pub blend_duration: f64,
    pub nudge_since: f64,
    pub nudge_sign: f32,
}

impl LobbyView {
    pub fn transition_to(&mut self, camera: LobbyCamera, now: f64, duration: f64) {
        let from = if duration > 0.0 { self.shot(now) } else { None };
        self.prev = from;
        self.camera = Some(camera);
        self.since = now;
        self.blend_since = now;
        self.blend_duration = if from.is_some() { duration } else { 0.0 };
    }

    pub fn nudge(&mut self, now: f64, sign: f32) {
        self.nudge_since = now;
        self.nudge_sign = if sign >= 0.0 { 1.0 } else { -1.0 };
    }

    pub fn shot(&self, now: f64) -> Option<(Vec3, Vec3, f32)> {
        let camera = self.camera?;
        let (mut position, mut target) = camera.at(now - self.since);
        let mut fov = camera.fov_degrees;
        if let Some((from_pos, from_target, from_fov)) = self.prev
            && self.blend_duration > 0.0
        {
            let raw = ((now - self.blend_since) / self.blend_duration) as f32;
            if raw < 1.0 {
                let u = ease_in_out_cubic(raw);
                let arc = math::sin(math::clamp(raw, 0.0, 1.0) * PI);
                let travel = (position - from_pos).with_y(0.0);
                let side = Vec3::new(-travel.z, 0.0, travel.x)
                    .try_normalize()
                    .unwrap_or(Vec3::X);
                position = from_pos.lerp(position, u) + side * (0.24 * arc);
                target = from_target.lerp(target, u);
                fov = from_fov + (fov - from_fov) * u;
            }
        }
        if self.nudge_since > 0.0 {
            let nt = ((now - self.nudge_since) / 0.48) as f32;
            if nt > 0.0 && nt < 1.0 {
                let tail = 1.0 - nt;
                let env = math::sin(nt * PI) * tail * tail;
                let turn = Quat::from_rotation_y(2.2_f32.to_radians() * self.nudge_sign * env);
                position = target + turn * (position - target);
                fov -= 1.2 * env;
            }
        }
        Some((position, target, fov))
    }
}
